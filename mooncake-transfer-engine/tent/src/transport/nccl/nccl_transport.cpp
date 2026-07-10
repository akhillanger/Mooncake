// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/transport/nccl/nccl_transport.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "tent/common/status.h"
#include "tent/runtime/platform.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/slab.h"
#include "tent/runtime/topology.h"
#include "tent/transport/nccl/paged_gin.h"

extern "C" cudaError_t tentNcclGinLaunchPut(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t dst_window,
    size_t dst_offset, ncclWindow_t src_window, size_t src_offset,
    size_t total_bytes, unsigned long long signal_value, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchGet(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t remote_window,
    size_t remote_offset, ncclWindow_t local_window, size_t local_offset,
    size_t total_bytes, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchWaitSignal(
    ncclDevComm_t dev_comm, int lanes, int signal_base,
    unsigned long long signal_value, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchWaitAck(
    ncclDevComm_t dev_comm, int peer, int lanes,
    unsigned long long signal_value, cudaStream_t stream);

namespace mooncake {
namespace tent {
namespace {

Status ncclStatus(ncclResult_t result, const char* expr) {
    if (result == ncclSuccess) return Status::OK();
    return Status::InternalError(std::string(expr) + ": " +
                                 ncclGetErrorString(result) + LOC_MARK);
}

Status cudaStatus(cudaError_t result, const char* expr) {
    if (result == cudaSuccess) return Status::OK();
    return Status::InternalError(std::string(expr) + ": " +
                                 cudaGetErrorString(result) + LOC_MARK);
}

#define CHECK_NCCL(call)                       \
    do {                                       \
        Status _s = ncclStatus(call, #call);   \
        if (!_s.ok()) return _s;               \
    } while (0)

int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string serializeUniqueId(const ncclUniqueId& id) {
    static constexpr char kHex[] = "0123456789abcdef";
    const auto* bytes = reinterpret_cast<const unsigned char*>(&id);
    std::string out;
    out.resize(sizeof(id) * 2);
    for (size_t i = 0; i < sizeof(id); ++i) {
        out[2 * i] = kHex[bytes[i] >> 4];
        out[2 * i + 1] = kHex[bytes[i] & 0xf];
    }
    return out;
}

Status deserializeUniqueId(const std::string& raw, ncclUniqueId& id) {
    if (raw.size() != sizeof(id) * 2) {
        return Status::InvalidArgument(
            "Invalid NCCL unique id size" LOC_MARK);
    }
    auto* bytes = reinterpret_cast<unsigned char*>(&id);
    for (size_t i = 0; i < sizeof(id); ++i) {
        int hi = hexValue(raw[2 * i]);
        int lo = hexValue(raw[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return Status::InvalidArgument(
                "Invalid NCCL unique id encoding" LOC_MARK);
        }
        bytes[i] = static_cast<unsigned char>((hi << 4) | lo);
    }
    return Status::OK();
}

std::string makeSessionKey(const std::string& local_name,
                           const std::string& remote_name, int local_device,
                           int remote_device) {
    std::ostringstream ss;
    ss << local_name << "->" << remote_name << ":cuda" << local_device
       << "->cuda" << remote_device;
    return ss.str();
}

std::string makeWindowKey(const std::string& session_key,
                          const char* purpose, uint64_t base,
                          uint64_t length) {
    std::ostringstream ss;
    ss << session_key << ":window:" << purpose << ":" << std::hex << base
       << ":" << length;
    return ss.str();
}

std::string makePairedWindowKey(const std::string& session_key,
                                uint64_t source_base,
                                uint64_t target_base, uint64_t source_length,
                                uint64_t target_length) {
    std::ostringstream ss;
    ss << session_key << ":window:paired:" << std::hex << source_base
       << ":" << target_base << ":" << source_length << ":"
       << target_length;
    return ss.str();
}

std::string makeRemoteWindowKey(const std::string& window_key) {
    return std::string("remote:") + window_key;
}

bool containsRange(uint64_t outer_base, uint64_t outer_length,
                   uint64_t inner_base, uint64_t inner_length) {
    if (inner_base < outer_base) return false;
    if (inner_length > outer_length) return false;
    return inner_base - outer_base <= outer_length - inner_length;
}

Status setCudaDevice(int device, int& previous_device) {
    CHECK_CUDA(cudaGetDevice(&previous_device));
    CHECK_CUDA(cudaSetDevice(device));
    return Status::OK();
}

bool peerInLsaTeam(ncclComm_t comm, int peer_rank) {
    ncclTeam_t world = ncclTeamWorld(comm);
    ncclTeam_t lsa = ncclTeamLsa(comm);
    return ncclTeamRankIsMember(lsa, world, peer_rank);
}

using SteadyClock = std::chrono::steady_clock;

bool envFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return false;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

double elapsedMs(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               SteadyClock::now() - start)
        .count();
}

Status getLsaPeerPointer(ncclWindow_t window, size_t offset, int peer,
                         void** ptr) {
    *ptr = nullptr;
    CHECK_NCCL(ncclGetPeerDevicePointer(window, offset, peer, ptr));
    if (!*ptr) {
        return Status::InvalidArgument(
            "NCCL peer window is not LSA reachable" LOC_MARK);
    }
    return Status::OK();
}

}  // namespace

struct NcclTransport::CommState {
    std::mutex mu;
    std::mutex submission_mu;
    std::condition_variable cv;
    ncclComm_t comm = nullptr;
    ncclDevComm_t dev_comm{};
    bool dev_comm_created = false;
    cudaStream_t completion_stream = nullptr;
    std::atomic<uint64_t> signal_epoch{0};
    size_t lanes = 1;
    Status status;
    bool initializing = false;
    bool ready = false;
    bool peer_in_lsa = false;
    int device_index = -1;
    int local_rank = -1;
    int peer_rank = -1;
};

struct NcclTransport::WindowState {
    std::mutex mu;
    std::condition_variable cv;
    ncclWindow_t window = nullptr;
    void* local_buffer = nullptr;
    uint64_t base = 0;
    uint64_t length = 0;
    bool owns_local_buffer = false;
    bool initializing = false;
    bool ready = false;
    bool source_window = false;
    int device_index = -1;
    std::string session_key;
    Status status;
};

struct NcclTransport::TransferContext {
    SegmentID target_id = 0;
    std::string remote_segment_name;
    std::string remote_rpc_addr;
    uint64_t target_base = 0;
    uint64_t target_length = 0;
    uint64_t target_offset = 0;
    uint64_t target_request_base = 0;
    uint64_t target_request_length = 0;
    uint64_t target_buffer_base = 0;
    uint64_t target_buffer_length = 0;
    uint64_t source_base = 0;
    uint64_t source_length = 0;
    uint64_t source_offset = 0;
    uint64_t source_request_base = 0;
    uint64_t source_request_length = 0;
    uint64_t source_buffer_base = 0;
    uint64_t source_buffer_length = 0;
    int local_device = -1;
    int remote_device = -1;
    std::string session_key;
    std::string window_key;
    std::string source_window_key;
};

NcclTransport::NcclTransport() = default;

NcclTransport::~NcclTransport() { uninstall(); }

Status NcclTransport::install(std::string& local_segment_name,
                              std::shared_ptr<ControlService> metadata,
                              std::shared_ptr<Topology> local_topology,
                              std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "NCCL transport has been installed" LOC_MARK);
    }

    if (Platform::getLoader().type() != "cuda") {
        return Status::InvalidArgument(
            "NCCL transport requires CUDA platform" LOC_MARK);
    }

    platform_ = dynamic_cast<CudaPlatform*>(&Platform::getLoader());
    if (!platform_) {
        return Status::InvalidArgument(
            "NCCL transport could not load CUDA platform" LOC_MARK);
    }

    CHECK_NCCL(ncclGetVersion(&nccl_version_));
    if (nccl_version_ < 23000) {
        return Status::InvalidArgument(
            "NCCL device GIN GET requires NCCL 2.30 or newer" LOC_MARK);
    }

    metadata_ = std::move(metadata);
    local_segment_name_ = local_segment_name;
    local_topology_ = std::move(local_topology);
    conf_ = std::move(conf);
    allow_external_window_buffers_ =
        conf_ ? conf_->get("transports/nccl/allow_external_window_buffers",
                           false)
              : false;
    if (conf_) {
        params_.max_concurrent_tasks =
            conf_->get("transports/nccl/max_concurrent_tasks",
                       params_.max_concurrent_tasks);
        params_.gin_lanes =
            conf_->get("transports/nccl/gin_lanes", params_.gin_lanes);
        params_.wait_ack =
            conf_->get("transports/nccl/wait_ack", params_.wait_ack);
        params_.force_gin =
            conf_->get("transports/nccl/force_gin", params_.force_gin);
    }
    if (std::getenv("MC_NCCL_FORCE_GIN")) {
        params_.force_gin = envFlagEnabled("MC_NCCL_FORCE_GIN");
    }
    if (params_.max_concurrent_tasks == 0) params_.max_concurrent_tasks = 1;
    if (params_.gin_lanes == 0) params_.gin_lanes = 1;
    if (params_.gin_lanes > 16) params_.gin_lanes = 16;
    if (params_.gin_lanes > 1 && !std::getenv("NCCL_GIN_NCONNECTIONS")) {
        const std::string connections = std::to_string(params_.gin_lanes);
        if (setenv("NCCL_GIN_NCONNECTIONS", connections.c_str(), 0) != 0) {
            LOG(WARNING) << "Failed to set NCCL_GIN_NCONNECTIONS="
                         << connections;
        } else {
            LOG(INFO) << "Defaulting NCCL_GIN_NCONNECTIONS=" << connections
                      << " for NCCL GIN context striping";
        }
    }
    shutting_down_.store(false, std::memory_order_release);
    thread_pool_ = std::make_unique<ThreadPool>(params_.max_concurrent_tasks);

    metadata_->setBootstrapNcclCallback(
        [this](const NcclBootstrapDesc& request, NcclBootstrapDesc& response) {
            auto status = onBootstrapNccl(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclWindowCallback(
        [this](const NcclWindowDesc& request, NcclWindowDesc& response) {
            auto status = onRegisterNcclWindow(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclWindowDeregisterCallback(
        [this](const NcclWindowDesc& request, NcclWindowDesc& response) {
            auto status = onDeregisterNcclWindow(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclWindowBatchCallback(
        [this](const NcclWindowBatchDesc& request,
               NcclWindowBatchDesc& response) {
            auto status = onRegisterNcclWindowBatch(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclWindowBatchDeregisterCallback(
        [this](const NcclWindowBatchDesc& request,
               NcclWindowBatchDesc& response) {
            auto status = onDeregisterNcclWindowBatch(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });
    metadata_->setNcclSignalCallback(
        [this](const NcclSignalDesc& request, NcclSignalDesc& response) {
            auto status = onWaitNcclSignal(request, response);
            if (!status.ok()) response.reply_msg = status.ToString();
            return status.ok() ? 0 : -1;
        });

    // Cross-node transfers use device-side NCCL GIN. Same-node LSA peers use
    // NCCL peer device pointers and CUDA D2D copies over NVLink.
    caps.gpu_to_gpu = true;
    installed_ = true;

    LOG(INFO) << "NCCL transport installed: version=" << nccl_version_
              << " allow_external_window_buffers="
              << allow_external_window_buffers_
              << " max_concurrent_tasks=" << params_.max_concurrent_tasks
              << " gin_lanes=" << params_.gin_lanes
              << " wait_ack=" << params_.wait_ack
              << " force_gin=" << params_.force_gin;
    return Status::OK();
}

Status NcclTransport::uninstall() {
    if (!installed_) return Status::OK();

    if (metadata_) {
        metadata_->setBootstrapNcclCallback(nullptr);
        metadata_->setNcclWindowCallback(nullptr);
        metadata_->setNcclWindowDeregisterCallback(nullptr);
        metadata_->setNcclWindowBatchCallback(nullptr);
        metadata_->setNcclWindowBatchDeregisterCallback(nullptr);
        metadata_->setNcclSignalCallback(nullptr);
    }

    shutting_down_.store(true, std::memory_order_release);
    thread_pool_.reset();

    std::vector<std::thread> background_threads;
    {
        std::lock_guard<std::mutex> lock(background_mutex_);
        background_threads.swap(background_threads_);
    }
    for (auto& thread : background_threads) {
        if (thread.joinable()) thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        for (auto& [_, window] : windows_) {
            if (!window) continue;
            std::unique_lock<std::mutex> state_lock(window->mu);
            if (window->window) {
                std::shared_ptr<CommState> comm_state;
                ncclComm_t comm = nullptr;
                {
                    std::lock_guard<std::mutex> comm_lock(comm_mutex_);
                    auto comm_it = comms_.find(window->session_key);
                    if (comm_it != comms_.end()) comm_state = comm_it->second;
                }
                bool lsa_session = false;
                if (comm_state) {
                    std::lock_guard<std::mutex> comm_state_lock(
                        comm_state->mu);
                    if (comm_state->ready) {
                        comm = comm_state->comm;
                        lsa_session = comm_state->peer_in_lsa;
                    }
                }
                if (lsa_session) {
                    LOG(INFO) << "Deferring NCCL LSA window cleanup for session "
                              << window->session_key;
                    continue;
                }
                if (!comm) {
                    LOG(WARNING) << "Skipping NCCL window deregister without "
                                    "ready communicator for session "
                                 << window->session_key;
                    continue;
                }
                int previous_device = 0;
                auto status = setCudaDevice(window->device_index,
                                            previous_device);
                if (status.ok()) {
                    auto result =
                        ncclCommWindowDeregister(comm, window->window);
                    if (result != ncclSuccess)
                        LOG(WARNING) << "ncclCommWindowDeregister failed: "
                                     << ncclGetErrorString(result);
                    cudaSetDevice(previous_device);
                }
            }
            if (window->owns_local_buffer && window->local_buffer) {
                auto result = ncclMemFree(window->local_buffer);
                if (result != ncclSuccess)
                    LOG(WARNING) << "ncclMemFree window buffer failed: "
                                 << ncclGetErrorString(result);
            }
        }
        windows_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        for (auto& [_, comm] : comms_) {
            if (comm && comm->ready) {
                int previous_device = 0;
                auto status = setCudaDevice(comm->device_index,
                                            previous_device);
                if (status.ok() && comm->comm) {
                    if (comm->peer_in_lsa) {
                        LOG(INFO) << "Deferring NCCL LSA communicator cleanup "
                                     "for device "
                                  << comm->device_index;
                    } else {
                        if (comm->dev_comm_created) {
                            auto result = ncclDevCommDestroy(comm->comm,
                                                             &comm->dev_comm);
                            if (result != ncclSuccess)
                                LOG(WARNING)
                                    << "ncclDevCommDestroy failed: "
                                    << ncclGetErrorString(result);
                        }
                        auto result = ncclCommDestroy(comm->comm);
                        if (result != ncclSuccess)
                            LOG(WARNING) << "ncclCommDestroy failed: "
                                         << ncclGetErrorString(result);
                    }
                }
                if (comm->completion_stream) {
                    cudaStreamDestroy(comm->completion_stream);
                    comm->completion_stream = nullptr;
                }
                if (status.ok()) cudaSetDevice(previous_device);
            }
        }
        comms_.clear();
    }

    std::lock_guard<std::mutex> lock(allocation_mutex_);
    if (!nccl_allocations_.empty()) {
        LOG(WARNING) << "NCCL transport uninstalling with "
                     << nccl_allocations_.size()
                     << " tracked ncclMemAlloc buffers still live";
    }
    metadata_.reset();
    local_topology_.reset();
    conf_.reset();
    platform_ = nullptr;
    installed_ = false;
    return Status::OK();
}

Status NcclTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto nccl_batch = Slab<NcclSubBatch>::Get().allocate();
    if (!nccl_batch)
        return Status::InternalError("Unable to allocate NCCL sub-batch");
    batch = nccl_batch;
    nccl_batch->task_list.reserve(max_size);
    nccl_batch->max_size = max_size;
    CHECK_STATUS(platform_->getStreamFromPool(nccl_batch->stream));
    return Status::OK();
}

Status NcclTransport::freeSubBatch(SubBatchRef& batch) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    for (auto& task : nccl_batch->task_list) {
        auto event = task.completion_event.exchange(
            nullptr, std::memory_order_acq_rel);
        if (event) cudaEventDestroy(event);
    }
    Slab<NcclSubBatch>::Get().deallocate(nccl_batch);
    batch = nullptr;
    return Status::OK();
}

Status NcclTransport::markFailed(NcclTask& task, const std::string& reason) {
    LOG(WARNING) << "NCCL host RMA task failed: " << reason;
    task.transferred_bytes.store(0, std::memory_order_release);
    task.status_word.store(TransferStatusEnum::FAILED,
                           std::memory_order_release);
    return Status::OK();
}

Status NcclTransport::buildTransferContext(const Request& request,
                                           TransferContext& ctx) {
    return buildTransferContext(request, request.length, request.length, ctx,
                                false);
}

Status NcclTransport::buildTransferContext(const Request& request,
                                           size_t source_length,
                                           size_t target_length,
                                           TransferContext& ctx,
                                           bool use_full_buffer_extents) {
    if (request.target_id == LOCAL_SEGMENT_ID) {
        return Status::InvalidArgument(
            "NCCL host RMA expects a remote target segment" LOC_MARK);
    }
    if (Platform::getLoader().getMemoryType(request.source) != MTYPE_CUDA) {
        return Status::InvalidArgument(
            "NCCL host RMA source must be CUDA memory" LOC_MARK);
    }

    auto local_locations = Platform::getLoader().getLocation(request.source, 1);
    if (local_locations.empty()) {
        return Status::InvalidArgument(
            "Unable to resolve local CUDA source location" LOC_MARK);
    }
    LocationParser local_location(local_locations[0].location);
    if (local_location.type() != "cuda" || local_location.index() < 0) {
        return Status::InvalidArgument(
            "Unable to resolve local CUDA device" LOC_MARK);
    }

    BufferDesc target_buffer;
    Status status = metadata_->segmentManager().withCachedSegment(
        request.target_id, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::Memory) {
                return Status::NeedsRefreshCache(
                    "NCCL target segment is not memory" LOC_MARK);
            }
            auto* buffer = segment->findBuffer(request.target_offset,
                                              target_length);
            if (!buffer) {
                return Status::NeedsRefreshCache(
                    "Requested address is not in registered buffer" LOC_MARK);
            }
            target_buffer = *buffer;
            ctx.remote_segment_name = segment->name;
            ctx.remote_rpc_addr = segment->rpc_server_addr;
            return Status::OK();
        });
    if (!status.ok()) return status;

    LocationParser remote_location(target_buffer.location);
    if (remote_location.type() != "cuda" || remote_location.index() < 0) {
        return Status::InvalidArgument(
            "NCCL host RMA target must be CUDA memory" LOC_MARK);
    }

    ctx.target_id = request.target_id;
    ctx.target_request_base = request.target_offset;
    ctx.target_request_length = target_length;
    ctx.target_buffer_base = target_buffer.addr;
    ctx.target_buffer_length = target_buffer.length;
    const uint64_t request_source =
        reinterpret_cast<uint64_t>(request.source);
    if (use_full_buffer_extents) {
        auto local_segment = metadata_->segmentManager().getLocal();
        if (!local_segment ||
            local_segment->type != SegmentType::Memory) {
            return Status::InvalidArgument(
                "NCCL full paired source segment is not memory"
                LOC_MARK);
        }
        auto* source_buffer =
            local_segment->findBuffer(request_source, source_length);
        if (!source_buffer) {
            return Status::InvalidArgument(
                "NCCL full paired source is not in a registered buffer"
                LOC_MARK);
        }
        LocationParser source_location(source_buffer->location);
        if (source_location.type() != "cuda" ||
            source_location.index() != local_location.index()) {
            return Status::InvalidArgument(
                "NCCL full paired source buffer is on the wrong device"
                LOC_MARK);
        }
        ctx.source_buffer_base = source_buffer->addr;
        ctx.source_buffer_length = source_buffer->length;

        constexpr uint64_t kRequiredWindowAlignment =
            NCCL_WIN_REQUIRED_ALIGNMENT;
        static_assert((kRequiredWindowAlignment &
                       (kRequiredWindowAlignment - 1)) == 0);
        auto expand_to_allocation_alignment = [](
            uint64_t buffer_base, uint64_t buffer_length, const char* name,
            uint64_t& window_base, uint64_t& window_length) -> Status {
            constexpr uint64_t kAlignment = NCCL_WIN_REQUIRED_ALIGNMENT;
            window_base = buffer_base & ~(kAlignment - 1);
            const uint64_t prefix = buffer_base - window_base;
            if (buffer_length >
                std::numeric_limits<uint64_t>::max() - prefix) {
                return Status::InvalidArgument(
                    std::string(name) +
                    " full paired window size overflows" LOC_MARK);
            }
            window_length = prefix + buffer_length;
            return Status::OK();
        };

        // Torch tensors can begin at a suballocation offset that is not a
        // valid NCCL window start. The 4 KiB floor remains inside the owning
        // CUDA allocation while covering the complete logical buffer.
        CHECK_STATUS(expand_to_allocation_alignment(
            target_buffer.addr, target_buffer.length, "NCCL target",
            ctx.target_base, ctx.target_length));
        CHECK_STATUS(expand_to_allocation_alignment(
            source_buffer->addr, source_buffer->length, "NCCL source",
            ctx.source_base, ctx.source_length));
        ctx.target_offset = request.target_offset - ctx.target_base;
        ctx.source_offset = request_source - ctx.source_base;
    } else {
        constexpr uint64_t kNcclWindowAlignment = 64 * 1024;

        const uint64_t aligned_target_base =
            request.target_offset & ~(kNcclWindowAlignment - 1);
        const uint64_t target_offset =
            request.target_offset - aligned_target_base;
        if (target_length >
            std::numeric_limits<uint64_t>::max() - target_offset) {
            return Status::InvalidArgument(
                "NCCL target window size overflows" LOC_MARK);
        }
        const uint64_t target_end = target_offset + target_length;
        const uint64_t aligned_target_length =
            (target_end + kNcclWindowAlignment - 1) &
            ~(kNcclWindowAlignment - 1);
        if (!containsRange(target_buffer.addr, target_buffer.length,
                           aligned_target_base, aligned_target_length)) {
            return Status::InvalidArgument(
                "Aligned NCCL target window escapes registered buffer"
                LOC_MARK);
        }
        ctx.target_base = aligned_target_base;
        ctx.target_length = aligned_target_length;
        ctx.target_offset = target_offset;

        const uint64_t aligned_source_base =
            request_source & ~(kNcclWindowAlignment - 1);
        const uint64_t source_offset = request_source - aligned_source_base;
        if (source_length >
            std::numeric_limits<uint64_t>::max() - source_offset) {
            return Status::InvalidArgument(
                "NCCL source window size overflows" LOC_MARK);
        }
        const uint64_t source_end = source_offset + source_length;
        const uint64_t aligned_source_length =
            (source_end + kNcclWindowAlignment - 1) &
            ~(kNcclWindowAlignment - 1);
        ctx.source_base = aligned_source_base;
        ctx.source_length = aligned_source_length;
        ctx.source_offset = source_offset;
    }
    ctx.source_request_base = request_source;
    ctx.source_request_length = source_length;
    ctx.local_device = local_location.index();
    ctx.remote_device = remote_location.index();
    ctx.session_key = makeSessionKey(local_segment_name_,
                                     ctx.remote_segment_name,
                                     ctx.local_device, ctx.remote_device);
    ctx.window_key = makeWindowKey(ctx.session_key, "target", ctx.target_base,
                                   ctx.target_length);
    ctx.source_window_key = makeWindowKey(ctx.session_key, "source",
                                          ctx.source_base, ctx.source_length);
    return Status::OK();
}

void NcclTransport::startBackground(std::function<void()> fn) {
    std::lock_guard<std::mutex> lock(background_mutex_);
    background_threads_.emplace_back(std::move(fn));
}

Status NcclTransport::waitForComm(const std::string& session_key,
                                  std::shared_ptr<CommState>& state) {
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto it = comms_.find(session_key);
        if (it == comms_.end()) {
            return Status::InvalidArgument(
                "NCCL communicator session not found" LOC_MARK);
        }
        state = it->second;
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::ensureComm(const TransferContext& ctx,
                                 std::shared_ptr<CommState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto& entry = comms_[ctx.session_key];
        if (!entry) entry = std::make_shared<CommState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->status = Status::OK();
            state->initializing = true;
            state->device_index = ctx.local_device;
            state->local_rank = 0;
            state->peer_rank = 1;
            should_init = true;
        }
    }

    if (should_init) {
        const size_t lanes = params_.gin_lanes;
        ncclUniqueId unique_id{};
        Status status = ncclStatus(ncclGetUniqueId(&unique_id),
                                   "ncclGetUniqueId");
        if (status.ok()) {
            NcclBootstrapDesc request;
            request.session_key = ctx.session_key;
            request.unique_id = serializeUniqueId(unique_id);
            // For the single-communicator device-GIN path this field carries
            // the requested number of GIN contexts/stripe lanes.
            request.comm_count = static_cast<int>(lanes);
            request.device_index = ctx.remote_device;
            NcclBootstrapDesc response;
            status = ControlClient::bootstrapNccl(ctx.remote_rpc_addr,
                                                  request, response);
        }
        if (status.ok()) {
            int previous_device = 0;
            status = setCudaDevice(ctx.local_device, previous_device);
            if (status.ok()) {
                state->lanes = lanes;
                status = ncclStatus(
                    ncclCommInitRank(&state->comm, 2, unique_id, 0),
                    "ncclCommInitRank");
                if (status.ok()) {
                    state->peer_in_lsa = peerInLsaTeam(state->comm,
                                                       state->peer_rank);
                }
                if (status.ok() && (!state->peer_in_lsa || params_.force_gin)) {
                    ncclDevCommRequirements_t reqs =
                        NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
                    reqs.ginForceEnable = true;
                    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
                    reqs.ginContextCount = static_cast<int>(lanes);
                    // Data completion uses [0, lanes); wait-ack uses
                    // [lanes, 2 * lanes).
                    reqs.ginSignalCount = static_cast<int>(lanes * 2);
                    status = ncclStatus(ncclGroupStart(),
                                        "ncclGroupStart(dev comm)");
                    if (status.ok()) {
                        Status create_status = ncclStatus(
                            ncclDevCommCreate(state->comm, &reqs,
                                              &state->dev_comm),
                            "ncclDevCommCreate");
                        Status group_status = ncclStatus(ncclGroupEnd(),
                                                        "ncclGroupEnd(dev comm)");
                        status = create_status.ok() ? group_status : create_status;
                    }
                    state->dev_comm_created = status.ok();
                }
                if (status.ok() && (!state->peer_in_lsa || params_.force_gin) &&
                    static_cast<size_t>(state->dev_comm.ginContextCount) <
                        lanes) {
                    status = Status::InternalError(
                        "NCCL dev comm did not provide requested GIN contexts" LOC_MARK);
                }
                if (status.ok()) {
                    status = cudaStatus(
                        cudaStreamCreateWithFlags(&state->completion_stream,
                                                  cudaStreamNonBlocking),
                        "cudaStreamCreateWithFlags(completion)");
                }
                if (status.ok()) {
                    if (state->peer_in_lsa) {
                        LOG(INFO) << "NCCL LSA communicator ready: lanes="
                                  << lanes << " peer_in_lsa=1";
                    } else {
                        LOG(INFO) << "NCCL GIN single communicator ready: lanes="
                                  << lanes << " gin_connections="
                                  << static_cast<int>(
                                         state->dev_comm.ginConnectionCount)
                                  << " gin_contexts="
                                  << state->dev_comm.ginContextCount
                                  << " peer_in_lsa=0";
                        if (static_cast<size_t>(
                                state->dev_comm.ginConnectionCount) < lanes) {
                            LOG(WARNING)
                                << "NCCL GIN has fewer connections than contexts; "
                                   "set NCCL_GIN_NCONNECTIONS="
                                << lanes << " for peak striped bandwidth";
                        }
                    }
                }
                cudaSetDevice(previous_device);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    return waitForComm(ctx.session_key, state);
}

Status NcclTransport::ensureWindow(
    const TransferContext& ctx, const std::shared_ptr<CommState>& comm_state,
    std::shared_ptr<WindowState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto it = windows_.find(ctx.window_key);
        if (it != windows_.end() && it->second) {
            state = it->second;
        } else if (envFlagEnabled("MC_NCCL_CACHE_COMPACT_WINDOWS")) {
            for (auto& entry : windows_) {
                auto& candidate = entry.second;
                if (!candidate) continue;
                std::lock_guard<std::mutex> candidate_lock(candidate->mu);
                if (candidate->source_window ||
                    candidate->session_key != ctx.session_key ||
                    candidate->device_index != ctx.local_device ||
                    (!candidate->ready && !candidate->initializing) ||
                    !candidate->status.ok()) {
                    continue;
                }
                if (containsRange(candidate->base, candidate->length,
                                  ctx.target_base, ctx.target_length)) {
                    state = candidate;
                    break;
                }
            }
        }
        if (!state) {
            auto& entry = windows_[ctx.window_key];
            if (!entry) entry = std::make_shared<WindowState>();
            state = entry;
        }
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->status = Status::OK();
            state->base = ctx.target_base;
            state->length = ctx.target_length;
            state->source_window = false;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }

    if (should_init) {
        Status status;
        NcclWindowDesc request;
        request.session_key = ctx.session_key;
        request.window_key = ctx.window_key;
        request.addr = ctx.target_base;
        request.length = ctx.target_length;
        request.device_index = ctx.remote_device;
        request.win_flags = NCCL_WIN_COLL_SYMMETRIC;
        request.allocate_local = false;
        NcclWindowDesc response;
        status = ControlClient::registerNcclWindow(ctx.remote_rpc_addr,
                                                   request, response);

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(ctx.local_device, previous_device);
            device_changed = status.ok();
        }
        if (status.ok()) {
            status = ncclStatus(ncclMemAlloc(&state->local_buffer,
                                             ctx.target_length),
                                "ncclMemAlloc(window dummy)");
            state->owns_local_buffer = status.ok();
            if (!status.ok()) {
                LOG(ERROR) << "NCCL target window dummy allocation failed"
                           << " session=" << ctx.session_key
                           << " key=" << ctx.window_key
                           << " local_device=" << ctx.local_device
                           << " remote_device=" << ctx.remote_device
                           << " target_base=0x" << std::hex
                           << ctx.target_base << std::dec
                           << " target_length=" << ctx.target_length
                           << " target_offset=" << ctx.target_offset;
            }
        }
        if (status.ok()) {
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, state->local_buffer,
                                       ctx.target_length, &state->window,
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister");
            if (!status.ok()) {
                LOG(ERROR) << "NCCL target window register failed"
                           << " session=" << ctx.session_key
                           << " key=" << ctx.window_key
                           << " local_device=" << ctx.local_device
                           << " remote_device=" << ctx.remote_device
                           << " target_base=0x" << std::hex
                           << ctx.target_base << std::dec
                           << " target_length=" << ctx.target_length
                           << " target_offset=" << ctx.target_offset;
            }
        }
        if (device_changed) cudaSetDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::ensureSourceWindow(
    const TransferContext& ctx, const std::shared_ptr<CommState>& comm_state,
    std::shared_ptr<WindowState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto it = windows_.find(ctx.source_window_key);
        if (it != windows_.end() && it->second) {
            state = it->second;
        } else if (envFlagEnabled("MC_NCCL_CACHE_COMPACT_WINDOWS")) {
            for (auto& entry : windows_) {
                auto& candidate = entry.second;
                if (!candidate) continue;
                std::lock_guard<std::mutex> candidate_lock(candidate->mu);
                if (!candidate->source_window ||
                    candidate->session_key != ctx.session_key ||
                    candidate->device_index != ctx.local_device ||
                    (!candidate->ready && !candidate->initializing) ||
                    !candidate->status.ok()) {
                    continue;
                }
                if (containsRange(candidate->base, candidate->length,
                                  ctx.source_base, ctx.source_length)) {
                    state = candidate;
                    break;
                }
            }
        }
        if (!state) {
            auto& entry = windows_[ctx.source_window_key];
            if (!entry) entry = std::make_shared<WindowState>();
            state = entry;
        }
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->status = Status::OK();
            state->base = ctx.source_base;
            state->length = ctx.source_length;
            state->source_window = true;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }

    if (should_init) {
        Status status;
        NcclWindowDesc request;
        request.session_key = ctx.session_key;
        request.window_key = ctx.source_window_key;
        request.length = ctx.source_length;
        request.device_index = ctx.remote_device;
        request.win_flags = NCCL_WIN_COLL_SYMMETRIC;
        request.allocate_local = true;
        NcclWindowDesc response;
        status = ControlClient::registerNcclWindow(ctx.remote_rpc_addr,
                                                   request, response);

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(ctx.local_device, previous_device);
            device_changed = status.ok();
        }
        if (status.ok() && envFlagEnabled("MC_NCCL_STAGE_SOURCE_WINDOWS")) {
            status = ncclStatus(
                ncclMemAlloc(&state->local_buffer, ctx.source_length),
                "ncclMemAlloc(source staging)");
            state->owns_local_buffer = status.ok();
            if (!status.ok()) {
                LOG(ERROR) << "NCCL source staging allocation failed"
                           << " session=" << ctx.session_key
                           << " key=" << ctx.source_window_key
                           << " local_device=" << ctx.local_device
                           << " remote_device=" << ctx.remote_device
                           << " source_base=0x" << std::hex
                           << ctx.source_base << std::dec
                           << " source_length=" << ctx.source_length;
            }
        }
        if (status.ok()) {
            void* register_buffer =
                state->local_buffer
                    ? state->local_buffer
                    : reinterpret_cast<void*>(ctx.source_base);
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, register_buffer,
                                       ctx.source_length, &state->window,
                                       NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister(source)");
            if (!status.ok()) {
                LOG(ERROR) << "NCCL source window register failed"
                           << " session=" << ctx.session_key
                           << " key=" << ctx.source_window_key
                           << " local_device=" << ctx.local_device
                           << " remote_device=" << ctx.remote_device
                           << " source_base=0x" << std::hex
                           << ctx.source_base << std::dec
                           << " source_length=" << ctx.source_length
                           << " staged=" << (state->local_buffer != nullptr);
            }
        }
        if (!status.ok() && state->owns_local_buffer && state->local_buffer) {
            auto result = ncclMemFree(state->local_buffer);
            if (result != ncclSuccess) {
                LOG(WARNING) << "ncclMemFree(source staging after failure): "
                             << ncclGetErrorString(result);
            }
            state->local_buffer = nullptr;
            state->owns_local_buffer = false;
        }
        if (device_changed) cudaSetDevice(previous_device);

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] { return state->ready || !state->status.ok(); });
    return state->status;
}

Status NcclTransport::ensurePairedWindow(
    const TransferContext& ctx,
    const std::shared_ptr<CommState>& comm_state,
    std::shared_ptr<WindowState>& state) {
    bool should_init = false;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[ctx.window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;

        std::lock_guard<std::mutex> state_lock(state->mu);
        if (!state->ready && !state->initializing) {
            state->initializing = true;
            state->ready = false;
            state->status = Status::OK();
            state->base = ctx.source_base;
            state->length = ctx.source_length;
            state->source_window = true;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            should_init = true;
        }
    }

    if (should_init) {
        Status status;
        double rpc_ms = 0.0;
        double allocation_ms = 0.0;
        double registration_ms = 0.0;

        NcclWindowDesc request;
        request.session_key = ctx.session_key;
        request.window_key = ctx.window_key;
        request.addr = ctx.target_base;
        request.length = ctx.target_length;
        request.device_index = ctx.remote_device;
        request.win_flags = NCCL_WIN_COLL_SYMMETRIC;
        request.allocate_local = false;
        NcclWindowDesc response;

        auto rpc_start = SteadyClock::now();
        status = ControlClient::registerNcclWindow(
            ctx.remote_rpc_addr, request, response);
        rpc_ms = elapsedMs(rpc_start);

        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(ctx.local_device, previous_device);
            device_changed = status.ok();
        }
        if (status.ok() &&
            envFlagEnabled("MC_NCCL_STAGE_SOURCE_WINDOWS")) {
            auto allocation_start = SteadyClock::now();
            status = ncclStatus(
                ncclMemAlloc(&state->local_buffer, ctx.source_length),
                "ncclMemAlloc(paired source staging)");
            allocation_ms = elapsedMs(allocation_start);
            state->owns_local_buffer = status.ok();
        }
        if (status.ok()) {
            void* register_buffer =
                state->local_buffer
                    ? state->local_buffer
                    : reinterpret_cast<void*>(ctx.source_base);
            auto registration_start = SteadyClock::now();
            status = ncclStatus(
                ncclCommWindowRegister(
                    comm_state->comm, register_buffer, ctx.source_length,
                    &state->window, NCCL_WIN_COLL_SYMMETRIC),
                "ncclCommWindowRegister(paired)");
            registration_ms = elapsedMs(registration_start);
            if (!status.ok()) {
                LOG(ERROR) << "NCCL paired source window register failed"
                           << " session=" << ctx.session_key
                           << " source_base=0x" << std::hex
                           << ctx.source_base << std::dec
                           << " source_length=" << ctx.source_length
                           << " target_base=0x" << std::hex
                           << ctx.target_base << std::dec
                           << " target_length=" << ctx.target_length;
            }
        }
        if (!status.ok() && state->owns_local_buffer &&
            state->local_buffer) {
            auto result = ncclMemFree(state->local_buffer);
            if (result != ncclSuccess) {
                LOG(WARNING)
                    << "ncclMemFree(paired source staging after failure): "
                    << ncclGetErrorString(result);
            }
            state->local_buffer = nullptr;
            state->owns_local_buffer = false;
        }
        if (device_changed) cudaSetDevice(previous_device);

        if (envFlagEnabled("MC_NCCL_PROFILE_PAGED")) {
            LOG(WARNING) << "NCCL paired window profile"
                         << " source_length=" << ctx.source_length
                         << " target_length=" << ctx.target_length
                         << " rpc_ms=" << rpc_ms
                         << " allocation_ms=" << allocation_ms
                         << " registration_ms=" << registration_ms
                         << " status=" << status.ToString();
        }

        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    }

    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait(lock, [&] {
        return state->ready || !state->status.ok();
    });
    return state->status;
}

Status NcclTransport::ensurePagedWindowsBatch(
    const std::vector<TransferContext>& contexts,
    const std::shared_ptr<CommState>& comm_state,
    std::vector<std::shared_ptr<WindowState>>& target_states,
    std::vector<std::shared_ptr<WindowState>>& source_states) {
    target_states.resize(contexts.size());
    source_states.resize(contexts.size());
    if (contexts.empty()) return Status::OK();

    struct BatchEntry {
        const TransferContext* ctx = nullptr;
        std::shared_ptr<WindowState> state;
        bool source = false;
    };
    std::vector<BatchEntry> entries;
    entries.reserve(contexts.size() * 2);

    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto reserve_state = [&](const TransferContext& ctx, bool source,
                                 std::shared_ptr<WindowState>& state) {
            const std::string& key =
                source ? ctx.source_window_key : ctx.window_key;
            auto& slot = windows_[key];
            if (!slot) slot = std::make_shared<WindowState>();
            state = slot;

            std::lock_guard<std::mutex> state_lock(state->mu);
            if (state->ready || state->initializing) return;
            state->initializing = true;
            state->ready = false;
            state->status = Status::OK();
            state->base = source ? ctx.source_base : ctx.target_base;
            state->length =
                source ? ctx.source_length : ctx.target_length;
            state->source_window = source;
            state->device_index = ctx.local_device;
            state->session_key = ctx.session_key;
            entries.push_back(BatchEntry{&ctx, state, source});
        };

        for (size_t i = 0; i < contexts.size(); ++i) {
            reserve_state(contexts[i], false, target_states[i]);
            reserve_state(contexts[i], true, source_states[i]);
        }
    }

    Status status;
    int previous_device = 0;
    bool device_changed = false;
    if (!entries.empty()) {
        status = setCudaDevice(contexts.front().local_device, previous_device);
        device_changed = status.ok();
    }

    const bool stage_sources =
        envFlagEnabled("MC_NCCL_STAGE_SOURCE_WINDOWS");
    for (auto& entry : entries) {
        if (!status.ok()) break;
        const uint64_t length =
            entry.source ? entry.ctx->source_length
                         : entry.ctx->target_length;
        if (!entry.source || stage_sources) {
            const char* allocation_name =
                entry.source ? "ncclMemAlloc(paged batch source staging)"
                             : "ncclMemAlloc(paged batch target dummy)";
            status = ncclStatus(
                ncclMemAlloc(&entry.state->local_buffer, length),
                allocation_name);
            entry.state->owns_local_buffer = status.ok();
        }
    }

    if (status.ok() && !entries.empty()) {
        NcclWindowBatchDesc request;
        request.windows.reserve(entries.size());
        for (const auto& entry : entries) {
            NcclWindowDesc window;
            window.session_key = entry.ctx->session_key;
            window.window_key =
                entry.source ? entry.ctx->source_window_key
                             : entry.ctx->window_key;
            window.addr = entry.source ? 0 : entry.ctx->target_base;
            window.length =
                entry.source ? entry.ctx->source_length
                             : entry.ctx->target_length;
            window.device_index = entry.ctx->remote_device;
            window.win_flags = NCCL_WIN_COLL_SYMMETRIC;
            window.allocate_local = entry.source;
            request.windows.push_back(std::move(window));
        }

        NcclWindowBatchDesc response;
        status = ControlClient::registerNcclWindowBatch(
            contexts.front().remote_rpc_addr, request, response);
    }

    if (status.ok() && !entries.empty()) {
        status = ncclStatus(ncclGroupStart(),
                            "ncclGroupStart(paged window batch)");
        Status register_status;
        if (status.ok()) {
            for (auto& entry : entries) {
                void* buffer = entry.source
                                   ? (entry.state->local_buffer
                                          ? entry.state->local_buffer
                                          : reinterpret_cast<void*>(
                                                entry.ctx->source_base))
                                   : entry.state->local_buffer;
                auto next = ncclStatus(
                    ncclCommWindowRegister(
                        comm_state->comm, buffer,
                        entry.source ? entry.ctx->source_length
                                     : entry.ctx->target_length,
                        &entry.state->window, NCCL_WIN_COLL_SYMMETRIC),
                    "ncclCommWindowRegister(paged batch)");
                if (register_status.ok() && !next.ok()) {
                    register_status = next;
                }
            }
            auto group_status = ncclStatus(
                ncclGroupEnd(), "ncclGroupEnd(paged window batch)");
            status = register_status.ok() ? group_status : register_status;
        }
    }

    if (device_changed) cudaSetDevice(previous_device);

    for (auto& entry : entries) {
        {
            std::lock_guard<std::mutex> lock(entry.state->mu);
            entry.state->status = status;
            entry.state->ready = status.ok();
            entry.state->initializing = false;
        }
        entry.state->cv.notify_all();
    }

    if (!status.ok()) return status;

    auto wait_ready = [](const std::shared_ptr<WindowState>& state) {
        std::unique_lock<std::mutex> lock(state->mu);
        state->cv.wait(
            lock, [&] { return state->ready || !state->status.ok(); });
        return state->status;
    };
    for (size_t i = 0; i < contexts.size(); ++i) {
        status = wait_ready(target_states[i]);
        if (!status.ok()) return status;
        status = wait_ready(source_states[i]);
        if (!status.ok()) return status;
    }
    return Status::OK();
}

Status NcclTransport::copySourceWindowBytes(
    const TransferContext& ctx, const std::shared_ptr<WindowState>& state,
    bool to_window, cudaStream_t stream) {
    if (!state || !state->local_buffer || ctx.source_request_length == 0) {
        return Status::OK();
    }
    if (!containsRange(state->base, state->length, ctx.source_request_base,
                       ctx.source_request_length)) {
        return Status::InternalError(
            "NCCL staged source window does not contain request" LOC_MARK);
    }
    const size_t window_offset = static_cast<size_t>(
        ctx.source_request_base - state->base);
    auto* window_ptr = static_cast<char*>(state->local_buffer) + window_offset;
    void* request_ptr = reinterpret_cast<void*>(
        static_cast<uintptr_t>(ctx.source_request_base));
    auto err = cudaMemcpyAsync(to_window ? static_cast<void*>(window_ptr)
                                         : request_ptr,
                               to_window ? request_ptr
                                         : static_cast<void*>(window_ptr),
                               static_cast<size_t>(ctx.source_request_length),
                               cudaMemcpyDeviceToDevice, stream);
    return cudaStatus(err, to_window ? "cudaMemcpyAsync(NCCL source staging)"
                                     : "cudaMemcpyAsync(NCCL source unstaging)");
}

Status NcclTransport::releaseWindow(const std::string& window_key) {
    std::shared_ptr<WindowState> state;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto it = windows_.find(window_key);
        if (it == windows_.end() || !it->second) return Status::OK();
        state = it->second;
        windows_.erase(it);
    }

    std::unique_lock<std::mutex> state_lock(state->mu);
    state->cv.wait(state_lock, [&] { return !state->initializing; });

    Status status;
    int previous_device = 0;
    bool device_changed = false;
    const bool needs_device = state->window || state->local_buffer;
    if (needs_device && state->device_index >= 0) {
        status = setCudaDevice(state->device_index, previous_device);
        device_changed = status.ok();
    }

    // Grouped registration can fail after creating a subset of the handles.
    // A non-null handle still has to be deregistered even if the aggregate
    // state never reached ready.
    if (status.ok() && state->window) {
        std::shared_ptr<CommState> comm_state;
        ncclComm_t comm = nullptr;
        bool lsa_session = false;
        {
            std::lock_guard<std::mutex> comm_lock(comm_mutex_);
            auto comm_it = comms_.find(state->session_key);
            if (comm_it != comms_.end()) comm_state = comm_it->second;
        }
        if (comm_state) {
            std::lock_guard<std::mutex> comm_state_lock(comm_state->mu);
            if (comm_state->ready) {
                comm = comm_state->comm;
                lsa_session = comm_state->peer_in_lsa;
            }
        }
        if (lsa_session) {
            LOG(INFO) << "Deferring NCCL LSA window deregister for session "
                      << state->session_key << " key=" << window_key;
        } else if (!comm) {
            LOG(WARNING) << "Skipping NCCL window deregister without ready "
                            "communicator for session "
                         << state->session_key << " key=" << window_key;
        } else {
            auto result = ncclCommWindowDeregister(comm, state->window);
            if (result != ncclSuccess) {
                status = ncclStatus(result, "ncclCommWindowDeregister");
                LOG(WARNING) << "ncclCommWindowDeregister failed for key="
                             << window_key << ": "
                             << ncclGetErrorString(result);
            }
        }
        state->window = nullptr;
        state->ready = false;
    }

    if (status.ok() && state->owns_local_buffer && state->local_buffer) {
        auto result = ncclMemFree(state->local_buffer);
        if (result != ncclSuccess) {
            status = ncclStatus(result, "ncclMemFree(window buffer)");
            LOG(WARNING) << "ncclMemFree window buffer failed for key="
                         << window_key << ": " << ncclGetErrorString(result);
        }
        state->local_buffer = nullptr;
        state->owns_local_buffer = false;
    }
    if (device_changed) cudaSetDevice(previous_device);

    state->status = status;
    state->initializing = false;
    state->cv.notify_all();
    return status;
}

Status NcclTransport::releaseCachedWindowsForSession(
    const TransferContext& ctx) {
    std::vector<std::string> window_keys;
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        for (const auto& entry : windows_) {
            const auto& key = entry.first;
            const auto& state = entry.second;
            if (!state) continue;
            std::lock_guard<std::mutex> state_lock(state->mu);
            if (state->session_key != ctx.session_key ||
                state->device_index != ctx.local_device || state->base == 0 ||
                state->length == 0) {
                continue;
            }
            window_keys.push_back(key);
        }
    }

    if (window_keys.empty()) return Status::OK();

    LOG(WARNING) << "Evicting cached NCCL windows for retry"
                 << " session=" << ctx.session_key
                 << " local_device=" << ctx.local_device
                 << " remote_device=" << ctx.remote_device
                 << " count=" << window_keys.size();

    Status cleanup_status;
    auto merge_status = [&](const Status& next, const char* action,
                            const std::string& window_key) {
        if (next.ok()) return;
        LOG(WARNING) << action << " failed for key=" << window_key << ": "
                     << next.ToString();
        if (cleanup_status.ok()) cleanup_status = next;
    };

    for (const auto& window_key : window_keys) {
        merge_status(releaseRemoteWindow(ctx, window_key),
                     "NCCL remote cached window deregister", window_key);
        merge_status(releaseWindow(window_key),
                     "NCCL local cached window deregister", window_key);
    }
    return cleanup_status;
}

Status NcclTransport::releaseRemoteWindow(const TransferContext& ctx,
                                          const std::string& window_key) {
    NcclWindowDesc request;
    request.session_key = ctx.session_key;
    request.window_key = window_key;
    request.device_index = ctx.remote_device;
    NcclWindowDesc response;
    return ControlClient::deregisterNcclWindow(ctx.remote_rpc_addr, request,
                                               response);
}

Status NcclTransport::releaseRemoteWindowsBatch(
    const TransferContext& ctx,
    const std::vector<std::string>& window_keys) {
    if (window_keys.empty()) return Status::OK();

    NcclWindowBatchDesc request;
    request.windows.reserve(window_keys.size());
    for (const auto& window_key : window_keys) {
        NcclWindowDesc window;
        window.session_key = ctx.session_key;
        window.window_key = window_key;
        window.device_index = ctx.remote_device;
        request.windows.push_back(std::move(window));
    }
    NcclWindowBatchDesc response;
    return ControlClient::deregisterNcclWindowBatch(
        ctx.remote_rpc_addr, request, response);
}

Status NcclTransport::postRemoteWaitSignal(const TransferContext& ctx,
                                           uint64_t signal_value) {
    NcclSignalDesc request;
    request.session_key = ctx.session_key;
    request.peer = 0;
    request.op_count = static_cast<int>(signal_value);
    request.signal_value = signal_value;
    request.signal_index = 0;
    request.context = 0;
    request.device_index = ctx.remote_device;
    NcclSignalDesc response;
    return ControlClient::waitNcclSignal(ctx.remote_rpc_addr, request,
                                         response);
}

Status NcclTransport::onBootstrapNccl(const NcclBootstrapDesc& request,
                                      NcclBootstrapDesc& response) {
    response.session_key = request.session_key;
    response.unique_id = request.unique_id;
    response.unique_ids = request.unique_ids;
    response.comm_count = request.comm_count;
    response.device_index = request.device_index;

    bool should_init = false;
    std::shared_ptr<CommState> state;
    {
        std::lock_guard<std::mutex> lock(comm_mutex_);
        auto& entry = comms_[request.session_key];
        if (!entry) entry = std::make_shared<CommState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        state->status = Status::OK();
        if (state->ready || state->initializing) return Status::OK();
        state->initializing = true;
        state->device_index = request.device_index;
        state->local_rank = 1;
        state->peer_rank = 0;
        should_init = true;
    }

    if (!should_init) return Status::OK();

    startBackground([this, state, request]() {
        Status status;
        const size_t lanes = request.comm_count > 0
                                 ? static_cast<size_t>(request.comm_count)
                                 : size_t{1};
        ncclUniqueId unique_id{};
        const std::string& raw = request.unique_ids.empty()
                                     ? request.unique_id
                                     : request.unique_ids[0];
        status = deserializeUniqueId(raw, unique_id);
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        if (status.ok()) {
            state->lanes = lanes;
            status = ncclStatus(
                ncclCommInitRank(&state->comm, 2, unique_id, 1),
                "ncclCommInitRank(remote)");
            if (status.ok()) {
                state->peer_in_lsa = peerInLsaTeam(state->comm,
                                                   state->peer_rank);
            }
            if (status.ok() && (!state->peer_in_lsa || params_.force_gin)) {
                ncclDevCommRequirements_t reqs =
                    NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
                reqs.ginForceEnable = true;
                reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
                reqs.ginContextCount = static_cast<int>(lanes);
                // Data completion uses [0, lanes); wait-ack uses
                // [lanes, 2 * lanes).
                reqs.ginSignalCount = static_cast<int>(lanes * 2);
                status = ncclStatus(ncclGroupStart(),
                                    "ncclGroupStart(remote dev comm)");
                if (status.ok()) {
                    Status create_status = ncclStatus(
                        ncclDevCommCreate(state->comm, &reqs,
                                          &state->dev_comm),
                        "ncclDevCommCreate(remote)");
                    Status group_status = ncclStatus(ncclGroupEnd(),
                                                    "ncclGroupEnd(remote dev comm)");
                    status = create_status.ok() ? group_status : create_status;
                }
                state->dev_comm_created = status.ok();
            }
            if (status.ok() && (!state->peer_in_lsa || params_.force_gin) &&
                static_cast<size_t>(state->dev_comm.ginContextCount) < lanes) {
                status = Status::InternalError(
                    "NCCL remote dev comm did not provide requested GIN contexts" LOC_MARK);
            }
            if (status.ok()) {
                status = cudaStatus(
                    cudaStreamCreateWithFlags(&state->completion_stream,
                                              cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags(remote completion)");
            }
            if (status.ok()) {
                if (state->peer_in_lsa) {
                    LOG(INFO) << "NCCL LSA remote communicator ready: lanes="
                              << lanes << " peer_in_lsa=1";
                } else {
                    LOG(INFO) << "NCCL GIN remote single communicator ready: lanes="
                              << lanes << " gin_connections="
                              << static_cast<int>(
                                     state->dev_comm.ginConnectionCount)
                              << " gin_contexts="
                              << state->dev_comm.ginContextCount
                              << " peer_in_lsa=0";
                    if (static_cast<size_t>(
                            state->dev_comm.ginConnectionCount) < lanes) {
                        LOG(WARNING)
                            << "NCCL GIN has fewer connections than contexts; "
                               "set NCCL_GIN_NCONNECTIONS="
                            << lanes << " for peak striped bandwidth";
                    }
                }
            }
        }
        if (device_changed) cudaSetDevice(previous_device);
        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    });
    return Status::OK();
}

Status NcclTransport::onRegisterNcclWindow(const NcclWindowDesc& request,
                                           NcclWindowDesc& response) {
    response.session_key = request.session_key;
    response.window_key = request.window_key;
    response.addr = request.addr;
    response.length = request.length;
    response.device_index = request.device_index;
    response.win_flags = request.win_flags;
    response.allocate_local = request.allocate_local;

    bool should_init = false;
    std::shared_ptr<WindowState> state;
    const std::string local_window_key =
        makeRemoteWindowKey(request.window_key);
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        auto& entry = windows_[local_window_key];
        if (!entry) entry = std::make_shared<WindowState>();
        state = entry;
        std::lock_guard<std::mutex> state_lock(state->mu);
        if (state->ready || state->initializing) return Status::OK();
        state->initializing = true;
        state->status = Status::OK();
        state->length = request.length;
        state->device_index = request.device_index;
        state->session_key = request.session_key;
        should_init = true;
    }

    if (!should_init) return Status::OK();

    startBackground([this, state, request]() {
        std::shared_ptr<CommState> comm_state;
        Status status = waitForComm(request.session_key, comm_state);
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        if (status.ok() && request.allocate_local) {
            status = ncclStatus(ncclMemAlloc(&state->local_buffer,
                                             request.length),
                                "ncclMemAlloc(remote window dummy)");
            state->owns_local_buffer = status.ok();
        }
        if (status.ok()) {
            void* buffer = request.allocate_local
                               ? state->local_buffer
                               : reinterpret_cast<void*>(request.addr);
            status = ncclStatus(
                ncclCommWindowRegister(comm_state->comm, buffer,
                                       request.length, &state->window,
                                       request.win_flags),
                "ncclCommWindowRegister(remote)");
            if (!status.ok()) {
                LOG(ERROR) << "NCCL remote window register failed"
                           << " session=" << request.session_key
                           << " key=" << request.window_key
                           << " addr=0x" << std::hex << request.addr
                           << std::dec << " length=" << request.length
                           << " device=" << request.device_index;
            }
        }
        if (device_changed) cudaSetDevice(previous_device);
        {
            std::lock_guard<std::mutex> lock(state->mu);
            state->status = status;
            state->ready = status.ok();
            state->initializing = false;
        }
        state->cv.notify_all();
    });
    return Status::OK();
}


Status NcclTransport::onDeregisterNcclWindow(const NcclWindowDesc& request,
                                             NcclWindowDesc& response) {
    response.session_key = request.session_key;
    response.window_key = request.window_key;
    response.addr = request.addr;
    response.length = request.length;
    response.device_index = request.device_index;
    response.win_flags = request.win_flags;
    response.allocate_local = request.allocate_local;
    return releaseWindow(makeRemoteWindowKey(request.window_key));
}

Status NcclTransport::onRegisterNcclWindowBatch(
    const NcclWindowBatchDesc& request, NcclWindowBatchDesc& response) {
    response.windows = request.windows;
    if (request.windows.empty()) return Status::OK();

    const auto& first = request.windows.front();
    std::vector<std::string> local_keys;
    local_keys.reserve(request.windows.size());
    std::unordered_set<std::string> unique_keys;
    for (const auto& window : request.windows) {
        if (window.session_key != first.session_key ||
            window.device_index != first.device_index) {
            return Status::InvalidArgument(
                "NCCL window batch spans multiple sessions or devices"
                LOC_MARK);
        }
        if (window.length == 0 || window.window_key.empty()) {
            return Status::InvalidArgument(
                "NCCL window batch contains an empty window" LOC_MARK);
        }
        std::string local_key = makeRemoteWindowKey(window.window_key);
        if (!unique_keys.insert(local_key).second) {
            return Status::InvalidArgument(
                "NCCL window batch contains duplicate keys" LOC_MARK);
        }
        local_keys.push_back(std::move(local_key));
    }

    std::vector<std::shared_ptr<WindowState>> states;
    states.reserve(request.windows.size());
    {
        std::lock_guard<std::mutex> lock(window_mutex_);
        for (const auto& key : local_keys) {
            if (windows_.find(key) != windows_.end()) {
                return Status::InvalidArgument(
                    "NCCL window batch key already exists" LOC_MARK);
            }
        }
        for (size_t i = 0; i < request.windows.size(); ++i) {
            const auto& window = request.windows[i];
            auto state = std::make_shared<WindowState>();
            state->initializing = true;
            state->base = window.addr;
            state->length = window.length;
            state->source_window = window.allocate_local;
            state->device_index = window.device_index;
            state->session_key = window.session_key;
            windows_.emplace(local_keys[i], state);
            states.push_back(std::move(state));
        }
    }

    std::shared_ptr<CommState> comm_state;
    Status status = waitForComm(first.session_key, comm_state);
    int previous_device = 0;
    bool device_changed = false;
    if (status.ok()) {
        status = setCudaDevice(first.device_index, previous_device);
        device_changed = status.ok();
    }
    for (size_t i = 0; status.ok() && i < request.windows.size(); ++i) {
        if (!request.windows[i].allocate_local) continue;
        status = ncclStatus(
            ncclMemAlloc(&states[i]->local_buffer,
                         request.windows[i].length),
            "ncclMemAlloc(remote paged batch dummy)");
        states[i]->owns_local_buffer = status.ok();
    }

    auto finish_states = [states](const Status& result) {
        for (const auto& state : states) {
            {
                std::lock_guard<std::mutex> lock(state->mu);
                state->status = result;
                state->ready = result.ok();
                state->initializing = false;
            }
            state->cv.notify_all();
        }
    };

    if (!status.ok()) {
        if (device_changed) cudaSetDevice(previous_device);
        finish_states(status);
        return status;
    }
    if (device_changed) cudaSetDevice(previous_device);

    startBackground([comm_state, states, windows = request.windows,
                     finish_states]() {
        int thread_previous_device = 0;
        Status batch_status =
            setCudaDevice(windows.front().device_index,
                          thread_previous_device);
        bool thread_device_changed = batch_status.ok();
        if (batch_status.ok()) {
            batch_status = ncclStatus(
                ncclGroupStart(),
                "ncclGroupStart(remote paged window batch)");
        }
        Status register_status;
        if (batch_status.ok()) {
            for (size_t i = 0; i < windows.size(); ++i) {
                void* buffer =
                    windows[i].allocate_local
                        ? states[i]->local_buffer
                        : reinterpret_cast<void*>(windows[i].addr);
                auto next = ncclStatus(
                    ncclCommWindowRegister(
                        comm_state->comm, buffer, windows[i].length,
                        &states[i]->window, windows[i].win_flags),
                    "ncclCommWindowRegister(remote paged batch)");
                if (register_status.ok() && !next.ok()) {
                    register_status = next;
                }
            }
            auto group_status = ncclStatus(
                ncclGroupEnd(),
                "ncclGroupEnd(remote paged window batch)");
            batch_status =
                register_status.ok() ? group_status : register_status;
        }
        if (thread_device_changed) {
            cudaSetDevice(thread_previous_device);
        }
        finish_states(batch_status);
    });
    return Status::OK();
}

Status NcclTransport::onDeregisterNcclWindowBatch(
    const NcclWindowBatchDesc& request, NcclWindowBatchDesc& response) {
    response.windows = request.windows;
    Status status;
    for (const auto& window : request.windows) {
        auto next =
            releaseWindow(makeRemoteWindowKey(window.window_key));
        if (status.ok() && !next.ok()) status = next;
    }
    return status;
}

Status NcclTransport::onWaitNcclSignal(const NcclSignalDesc& request,
                                       NcclSignalDesc& response) {
    response.session_key = request.session_key;
    response.peer = request.peer;
    response.op_count = request.op_count;
    response.signal_index = request.signal_index;
    response.context = request.context;
    response.device_index = request.device_index;
    response.signal_value = request.signal_value;

    if (!request.put_signal) {
        std::shared_ptr<CommState> comm_state;
        Status status = waitForComm(request.session_key, comm_state);
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        if (status.ok()) {
            const uint64_t signal_value =
                request.signal_value ? request.signal_value
                                     : request.op_count;
            auto err = tentNcclGinLaunchWaitAck(
                comm_state->dev_comm, request.peer,
                static_cast<int>(comm_state->lanes),
                static_cast<unsigned long long>(signal_value),
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchWaitAck(remote)");
        }
        if (device_changed) cudaSetDevice(previous_device);
        return status;
    }


    startBackground([this, request]() {
        std::shared_ptr<CommState> comm_state;
        Status status = waitForComm(request.session_key, comm_state);
        std::shared_ptr<WindowState> window_state;
        std::shared_ptr<WindowState> source_window_state;
        if (status.ok() && request.put_signal) {
            {
                std::lock_guard<std::mutex> lock(window_mutex_);
                auto dst_it = windows_.find(
                    makeRemoteWindowKey(request.window_key));
                auto src_it = windows_.find(
                    makeRemoteWindowKey(request.source_window_key));
                if (dst_it == windows_.end() || src_it == windows_.end()) {
                    status = Status::InvalidArgument(
                        "NCCL device put window not found" LOC_MARK);
                } else {
                    window_state = dst_it->second;
                    source_window_state = src_it->second;
                }
            }
            if (status.ok()) {
                std::unique_lock<std::mutex> lock(window_state->mu);
                window_state->cv.wait(lock, [&] {
                    return window_state->ready || !window_state->status.ok();
                });
                status = window_state->status;
            }
            if (status.ok()) {
                std::unique_lock<std::mutex> lock(source_window_state->mu);
                source_window_state->cv.wait(lock, [&] {
                    return source_window_state->ready ||
                           !source_window_state->status.ok();
                });
                status = source_window_state->status;
            }
        }
        int previous_device = 0;
        bool device_changed = false;
        if (status.ok()) {
            status = setCudaDevice(request.device_index, previous_device);
            device_changed = status.ok();
        }
        const uint64_t signal_value =
            request.signal_value ? request.signal_value : request.op_count;
        if (status.ok() && request.put_signal) {
            auto err = tentNcclGinLaunchPut(
                comm_state->dev_comm, request.peer,
                static_cast<int>(comm_state->lanes), window_state->window,
                request.peer_window_offset, source_window_state->window,
                request.source_addr, request.length,
                static_cast<unsigned long long>(signal_value),
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchPut(remote)");
        } else if (status.ok()) {
            auto err = tentNcclGinLaunchWaitAck(
                comm_state->dev_comm, request.peer,
                static_cast<int>(comm_state->lanes),
                static_cast<unsigned long long>(signal_value),
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchWaitAck(remote)");
        }
        if (device_changed) cudaSetDevice(previous_device);
        if (!status.ok()) {
            LOG(WARNING) << "NCCL remote signal operation failed: "
                         << status.ToString();
        }
    });
    return Status::OK();
}

void NcclTransport::startTransfer(NcclTask* task, NcclSubBatch* batch) {
    if (!task || !batch) return;
    if (shutting_down_.load(std::memory_order_acquire)) {
        markFailed(*task, "NCCL transport shutting down");
        return;
    }

    TransferContext ctx;
    auto status = buildTransferContext(task->request, ctx);
    const bool is_read = task->request.opcode == Request::READ;
    std::shared_ptr<CommState> comm_state;
    if (status.ok()) status = ensureComm(ctx, comm_state);
    std::unique_lock<std::mutex> submission_lock;
    if (status.ok()) {
        submission_lock =
            std::unique_lock<std::mutex>(comm_state->submission_mu);
    }
    std::shared_ptr<WindowState> window_state;
    if (status.ok()) status = ensureWindow(ctx, comm_state, window_state);
    const bool use_lsa =
        status.ok() && comm_state->peer_in_lsa && !params_.force_gin;
    std::shared_ptr<WindowState> source_window_state;
    if (status.ok() && !use_lsa) {
        status = ensureSourceWindow(ctx, comm_state, source_window_state);
    }
    const bool wait_ack =
        params_.wait_ack && status.ok() && !use_lsa && !is_read;
    const uint64_t signal_value =
        wait_ack ? comm_state->signal_epoch.fetch_add(
                       1, std::memory_order_acq_rel) +
                       1
                 : 0;
    if (status.ok() && wait_ack) {
        status = postRemoteWaitSignal(ctx, signal_value);
    }
    if (status.ok()) {
        int previous_device = 0;
        status = setCudaDevice(ctx.local_device, previous_device);
        bool device_changed = status.ok();
        cudaEvent_t event = nullptr;
        if (status.ok()) {
            const int lanes = static_cast<int>(comm_state->lanes);
            size_t target_window_offset = 0;
            if (!containsRange(window_state->base, window_state->length,
                               ctx.target_request_base,
                               ctx.target_request_length)) {
                status = Status::InternalError(
                    "NCCL target window does not contain request" LOC_MARK);
            } else {
                target_window_offset = static_cast<size_t>(
                    ctx.target_request_base - window_state->base);
            }

            if (status.ok() && use_lsa) {
                void* peer_ptr = nullptr;
                status = getLsaPeerPointer(window_state->window,
                                           target_window_offset,
                                           comm_state->peer_rank,
                                           &peer_ptr);
                if (status.ok()) {
                    void* local_ptr = reinterpret_cast<void*>(
                        static_cast<uintptr_t>(ctx.source_request_base));
                    auto err = cudaMemcpyAsync(
                        is_read ? local_ptr : peer_ptr,
                        is_read ? peer_ptr : local_ptr, task->request.length,
                        cudaMemcpyDeviceToDevice,
                        comm_state->completion_stream);
                    status = cudaStatus(err, "cudaMemcpyAsync(NCCL LSA)");
                }
            } else if (status.ok()) {
                size_t source_window_offset = 0;
                if (!containsRange(source_window_state->base,
                                   source_window_state->length,
                                   ctx.source_request_base,
                                   ctx.source_request_length)) {
                    status = Status::InternalError(
                        "NCCL source window does not contain request"
                        LOC_MARK);
                } else {
                    source_window_offset = static_cast<size_t>(
                        ctx.source_request_base - source_window_state->base);
                }

                const bool source_staged =
                    source_window_state && source_window_state->local_buffer;
                if (status.ok() && source_staged && !is_read) {
                    status = copySourceWindowBytes(
                        ctx, source_window_state, true,
                        comm_state->completion_stream);
                }

                if (status.ok() && is_read) {
                    auto err = tentNcclGinLaunchGet(
                        comm_state->dev_comm, comm_state->peer_rank, lanes,
                        window_state->window, target_window_offset,
                        source_window_state->window, source_window_offset,
                        task->request.length, comm_state->completion_stream);
                    status = cudaStatus(err, "tentNcclGinLaunchGet");
                    if (status.ok() && source_staged) {
                        status = copySourceWindowBytes(
                            ctx, source_window_state, false,
                            comm_state->completion_stream);
                    }
                } else if (status.ok()) {
                    auto err = tentNcclGinLaunchPut(
                        comm_state->dev_comm, comm_state->peer_rank, lanes,
                        window_state->window, target_window_offset,
                        source_window_state->window, source_window_offset,
                        task->request.length,
                        static_cast<unsigned long long>(signal_value),
                        comm_state->completion_stream);
                    status = cudaStatus(err, "tentNcclGinLaunchPut");
                    if (status.ok() && wait_ack) {
                        err = tentNcclGinLaunchWaitSignal(
                            comm_state->dev_comm, lanes, lanes,
                            static_cast<unsigned long long>(signal_value),
                            comm_state->completion_stream);
                        status = cudaStatus(err,
                                            "tentNcclGinLaunchWaitSignal");
                    }
                }
            }
            if (status.ok()) {
                auto err = cudaEventCreateWithFlags(&event,
                                                    cudaEventDisableTiming);
                if (err != cudaSuccess) {
                    status = Status::InternalError(
                        std::string("cudaEventCreateWithFlags: ") +
                        cudaGetErrorString(err) + LOC_MARK);
                }
            }
            if (status.ok()) {
                auto err = cudaEventRecord(event,
                                           comm_state->completion_stream);
                if (err != cudaSuccess) {
                    status = Status::InternalError(
                        std::string("cudaEventRecord: ") +
                        cudaGetErrorString(err) + LOC_MARK);
                }
            }
        }
        if (device_changed) cudaSetDevice(previous_device);
        if (status.ok()) {
            task->completion_event.store(event, std::memory_order_release);
            event = nullptr;
        }
        if (event) cudaEventDestroy(event);
    }

    if (!status.ok()) markFailed(*task, status.ToString());
}

Status NcclTransport::transferPagedSync(
    const PagedTransferRequest& request) {
    if (!installed_ || !thread_pool_) {
        return Status::InternalError(
            "NCCL transport is not installed" LOC_MARK);
    }
    if (shutting_down_.load(std::memory_order_acquire)) {
        return Status::InvalidArgument(
            "NCCL transport is shutting down" LOC_MARK);
    }
    if (request.page_bytes == 0) {
        return Status::InvalidArgument(
            "Paged transfer page_bytes must be nonzero" LOC_MARK);
    }
    const size_t src_page_stride = request.src_page_stride_bytes
                                       ? request.src_page_stride_bytes
                                       : request.page_bytes;
    const size_t dst_page_stride = request.dst_page_stride_bytes
                                       ? request.dst_page_stride_bytes
                                       : request.page_bytes;
    if (src_page_stride < request.page_bytes ||
        dst_page_stride < request.page_bytes) {
        return Status::InvalidArgument(
            "Paged transfer page stride is smaller than page_bytes" LOC_MARK);
    }
    if (request.src_layer_ptrs.size() != request.dst_layer_ptrs.size()) {
        return Status::InvalidArgument(
            "Paged transfer source and destination layer counts differ"
            LOC_MARK);
    }
    if (request.src_page_indices.size() != request.dst_page_indices.size()) {
        return Status::InvalidArgument(
            "Paged transfer source and destination page tables differ"
            LOC_MARK);
    }

    const size_t layer_count = request.src_layer_ptrs.size();
    const size_t page_count = request.src_page_indices.size();
    if (layer_count == 0 || page_count == 0) return Status::OK();
    if (page_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidArgument(
            "Paged transfer page table is too large" LOC_MARK);
    }
    if (page_count > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        return Status::InvalidArgument(
            "Paged transfer page table byte size overflows" LOC_MARK);
    }

    bool has_valid_pages = false;
    std::unordered_set<int32_t> unique_dst_pages;
    unique_dst_pages.reserve(page_count);
    int32_t min_src_page = 0;
    int32_t min_dst_page = 0;
    int32_t max_src_page = 0;
    int32_t max_dst_page = 0;
    for (size_t i = 0; i < page_count; ++i) {
        const int32_t src_page = request.src_page_indices[i];
        const int32_t dst_page = request.dst_page_indices[i];
        if (src_page < 0 || dst_page < 0) continue;
        if (!unique_dst_pages.insert(dst_page).second) {
            return Status::InvalidArgument(
                "Paged transfer contains duplicate destination pages"
                LOC_MARK);
        }
        if (!has_valid_pages) {
            min_src_page = src_page;
            min_dst_page = dst_page;
            max_src_page = src_page;
            max_dst_page = dst_page;
            has_valid_pages = true;
        } else {
            min_src_page = std::min(min_src_page, src_page);
            min_dst_page = std::min(min_dst_page, dst_page);
            max_src_page = std::max(max_src_page, src_page);
            max_dst_page = std::max(max_dst_page, dst_page);
        }
    }
    if (!has_valid_pages) return Status::OK();

    const bool full_paired_windows_requested =
        envFlagEnabled("MC_NCCL_CACHE_FULL_PAIRED_WINDOWS");
    // Full-pool windows retain absolute page indices. Applying a compact
    // minimum-page base as well would count that offset twice.
    const bool compact_windows =
        !full_paired_windows_requested &&
        (request.page_bytes == 524288 || request.page_bytes == 65536 ||
         request.page_bytes == 16384);
    const bool cache_compact_windows =
        envFlagEnabled("MC_NCCL_CACHE_COMPACT_WINDOWS");

    if (full_paired_windows_requested &&
        envFlagEnabled("MC_NCCL_STAGE_SOURCE_WINDOWS")) {
        return Status::InvalidArgument(
            "NCCL full paired windows require direct source registration"
            LOC_MARK);
    }
    const bool retry_compact_window_register = envFlagEnabled(
        "MC_NCCL_EVICT_COMPACT_WINDOWS_ON_FAILURE");
    const bool profile_paged = envFlagEnabled("MC_NCCL_PROFILE_PAGED");
    const bool trace_paged = profile_paged || envFlagEnabled("MC_NCCL_TRACE_PAGED");
    if (trace_paged) {
        LOG(WARNING) << "NCCL paged trace begin"
                     << " target_id=" << request.target_id
                     << " page_bytes=" << request.page_bytes
                     << " layers=" << layer_count
                     << " pages=" << page_count
                     << " src_page_min=" << min_src_page
                     << " src_page_max=" << max_src_page
                     << " dst_page_min=" << min_dst_page
                     << " dst_page_max=" << max_dst_page
                     << " compact=" << compact_windows
                     << " cache_compact=" << cache_compact_windows;
    }

    auto page_window = [&](int32_t min_page, int32_t max_page,
                           size_t page_stride, const char* name,
                           size_t& base_offset,
                           size_t& span) -> Status {
        const size_t first_page = static_cast<size_t>(min_page);
        const size_t page_delta = static_cast<size_t>(max_page - min_page);
        if (first_page >
            std::numeric_limits<size_t>::max() / page_stride) {
            return Status::InvalidArgument(
                std::string(name) + " byte offset overflows" LOC_MARK);
        }
        if (page_delta > (std::numeric_limits<size_t>::max() -
                          request.page_bytes) / page_stride) {
            return Status::InvalidArgument(
                std::string(name) + " byte size overflows" LOC_MARK);
        }
        base_offset = first_page * page_stride;
        span = page_delta * page_stride + request.page_bytes;
        return Status::OK();
    };

    constexpr size_t kMinNcclWindowBytes = 64 * 1024;
    constexpr size_t kCachedCompactWindowBytes = 2 * 1024 * 1024;
    const size_t min_compact_window_bytes =
        cache_compact_windows ? kCachedCompactWindowBytes
                              : kMinNcclWindowBytes;
    int32_t window_min_src_page = compact_windows ? min_src_page : 0;
    int32_t window_min_dst_page = compact_windows ? min_dst_page : 0;
    int32_t window_max_src_page = max_src_page;
    int32_t window_max_dst_page = max_dst_page;

    auto expand_small_compact_window = [&](int32_t& min_page,
                                           int32_t& max_page, size_t page_stride,
                                           const char* name) -> Status {
        if (!compact_windows ||
            request.page_bytes >= min_compact_window_bytes) {
            return Status::OK();
        }
        const size_t remaining_bytes =
            min_compact_window_bytes - request.page_bytes;
        const size_t pages_per_window =
            1 + remaining_bytes / page_stride + (remaining_bytes % page_stride != 0);
        if (pages_per_window <= 1) return Status::OK();
        if (min_page < 0 || max_page < min_page) {
            return Status::InvalidArgument(
                std::string(name) + " compact page window is invalid"
                LOC_MARK);
        }
        const size_t min_page_value = static_cast<size_t>(min_page);
        const size_t aligned_min =
            (min_page_value / pages_per_window) * pages_per_window;
        if (aligned_min >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return Status::InvalidArgument(
                std::string(name) + " compact page window overflows"
                LOC_MARK);
        }
        if (aligned_min >
            std::numeric_limits<size_t>::max() - pages_per_window + 1) {
            return Status::InvalidArgument(
                std::string(name) + " compact page window overflows"
                LOC_MARK);
        }
        size_t expanded_max = aligned_min + pages_per_window - 1;
        expanded_max = std::max(expanded_max, static_cast<size_t>(max_page));
        if (expanded_max >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return Status::InvalidArgument(
                std::string(name) + " compact page window overflows"
                LOC_MARK);
        }
        min_page = static_cast<int32_t>(aligned_min);
        max_page = static_cast<int32_t>(expanded_max);
        return Status::OK();
    };

    CHECK_STATUS(expand_small_compact_window(
        window_min_src_page, window_max_src_page, src_page_stride, "Paged source window"));
    CHECK_STATUS(expand_small_compact_window(
        window_min_dst_page, window_max_dst_page, dst_page_stride, "Paged target window"));

    size_t source_base_offset = 0;
    size_t target_base_offset = 0;
    size_t source_span = 0;
    size_t target_span = 0;
    CHECK_STATUS(page_window(window_min_src_page, window_max_src_page,
                             src_page_stride, "Paged source window", source_base_offset,
                             source_span));
    CHECK_STATUS(page_window(window_min_dst_page, window_max_dst_page,
                             dst_page_stride, "Paged target window", target_base_offset,
                             target_span));

    std::vector<int32_t> src_page_indices;
    std::vector<int32_t> dst_page_indices;
    src_page_indices.reserve(page_count);
    dst_page_indices.reserve(page_count);
    for (size_t i = 0; i < page_count; ++i) {
        const int32_t src_page = request.src_page_indices[i];
        const int32_t dst_page = request.dst_page_indices[i];
        if (src_page < 0 || dst_page < 0) {
            src_page_indices.push_back(src_page);
            dst_page_indices.push_back(dst_page);
        } else if (compact_windows &&
                   !full_paired_windows_requested) {
            src_page_indices.push_back(src_page - window_min_src_page);
            dst_page_indices.push_back(dst_page - window_min_dst_page);
        } else {
            src_page_indices.push_back(src_page);
            dst_page_indices.push_back(dst_page);
        }
    }

    struct PagedRun {
        TransferContext ctx;
        std::vector<size_t> src_layer_offsets;
        std::vector<size_t> dst_layer_offsets;
    };

    constexpr size_t kMaxPagedWindowBytes =
        static_cast<size_t>(2) * 1024 * 1024 * 1024;

    auto checked_add = [](uint64_t base, size_t offset, const char* name,
                          uint64_t& out) -> Status {
        if (base > std::numeric_limits<uint64_t>::max() - offset) {
            return Status::InvalidArgument(
                std::string(name) + " byte offset overflows" LOC_MARK);
        }
        out = base + offset;
        return Status::OK();
    };

    auto layer_window = [&](size_t begin, size_t end, uint64_t& src_base,
                            size_t& src_window_span, uint64_t& dst_base,
                            size_t& dst_window_span) -> Status {
        if (begin >= end || end > layer_count) {
            return Status::InvalidArgument(
                "Paged transfer layer window is empty" LOC_MARK);
        }
        uint64_t min_src = std::numeric_limits<uint64_t>::max();
        uint64_t min_dst = std::numeric_limits<uint64_t>::max();
        uint64_t max_src = 0;
        uint64_t max_dst = 0;
        for (size_t i = begin; i < end; ++i) {
            if (!request.src_layer_ptrs[i]) {
                return Status::InvalidArgument(
                    "Paged transfer source layer pointer is null" LOC_MARK);
            }
            const uint64_t src_ptr = static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(request.src_layer_ptrs[i]));
            const uint64_t dst_ptr = request.dst_layer_ptrs[i];
            uint64_t src_start = 0;
            uint64_t dst_start = 0;
            uint64_t src_end = 0;
            uint64_t dst_end = 0;
            CHECK_STATUS(checked_add(src_ptr, source_base_offset,
                                     "Paged source", src_start));
            CHECK_STATUS(checked_add(dst_ptr, target_base_offset,
                                     "Paged target", dst_start));
            CHECK_STATUS(checked_add(src_start, source_span,
                                     "Paged source", src_end));
            CHECK_STATUS(checked_add(dst_start, target_span,
                                     "Paged target", dst_end));
            min_src = std::min(min_src, src_start);
            min_dst = std::min(min_dst, dst_start);
            max_src = std::max(max_src, src_end);
            max_dst = std::max(max_dst, dst_end);
        }
        if (max_src < min_src || max_dst < min_dst) {
            return Status::InvalidArgument(
                "Paged transfer layer window is invalid" LOC_MARK);
        }
        src_base = min_src;
        dst_base = min_dst;
        src_window_span = static_cast<size_t>(max_src - min_src);
        dst_window_span = static_cast<size_t>(max_dst - min_dst);
        return Status::OK();
    };

    std::vector<PagedRun> runs;
    runs.reserve(layer_count);
    size_t layer = 0;
    while (layer < layer_count) {
        size_t run_layers = 1;
        uint64_t run_source_base = 0;
        uint64_t run_target_base = 0;
        size_t run_source_span = 0;
        size_t run_target_span = 0;
        CHECK_STATUS(layer_window(layer, layer + run_layers, run_source_base,
                                  run_source_span, run_target_base,
                                  run_target_span));

        // The remote registry tracks DSV4 KV/state pools as separate buffer
        // entries. A compact window may start inside one entry and cross into
        // the next even when the virtual addresses are adjacent, so keep the
        // correctness path to one compact layer window per run.
        if (run_layers >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            return Status::InvalidArgument(
                "Paged transfer layer run is too large" LOC_MARK);
        }

        Request layer_request;
        layer_request.opcode = Request::WRITE;
        layer_request.source = reinterpret_cast<void*>(
            static_cast<uintptr_t>(run_source_base));
        layer_request.target_id = request.target_id;
        layer_request.target_offset = run_target_base;
        layer_request.length = std::max(run_source_span, run_target_span);

        TransferContext ctx;
        CHECK_STATUS(buildTransferContext(
            layer_request, run_source_span, run_target_span, ctx,
            full_paired_windows_requested));
        if (full_paired_windows_requested) {
            if (ctx.source_buffer_base == 0 ||
                ctx.source_buffer_length == 0 ||
                ctx.target_buffer_base == 0 ||
                ctx.target_buffer_length == 0) {
                return Status::InvalidArgument(
                    "NCCL full paired window extent is empty" LOC_MARK);
            }
            if (!containsRange(ctx.source_buffer_base,
                               ctx.source_buffer_length,
                               run_source_base, run_source_span) ||
                !containsRange(ctx.target_buffer_base,
                               ctx.target_buffer_length,
                               run_target_base, run_target_span)) {
                return Status::InvalidArgument(
                    "NCCL full paired request escapes registered buffer"
                    LOC_MARK);
            }
            if (!containsRange(ctx.source_base, ctx.source_length,
                               run_source_base, run_source_span) ||
                !containsRange(ctx.target_base, ctx.target_length,
                               run_target_base, run_target_span)) {
                return Status::InternalError(
                    "NCCL aligned full paired window misses request"
                    LOC_MARK);
            }
        } else if (compact_windows) {
            ctx.target_base = run_target_base;
            ctx.target_length = run_target_span;
            ctx.target_offset = 0;
            ctx.window_key = makeWindowKey(ctx.session_key, "paged-target",
                                           ctx.target_base, ctx.target_length);
            ctx.source_window_key = makeWindowKey(ctx.session_key, "paged-source",
                                                  ctx.source_base,
                                                  ctx.source_length);
        } else {
            // Match the legacy path for regular KV when the remote target
            // buffer is larger than the touched source span, but never shrink
            // the source window below the actual source request.
            ctx.source_length = std::max(ctx.source_length, ctx.target_length);
            ctx.window_key = makeWindowKey(ctx.session_key, "paged-target",
                                           ctx.target_base, ctx.target_length);
            ctx.source_window_key = makeWindowKey(ctx.session_key, "paged-source",
                                                  ctx.source_base,
                                                  ctx.source_length);
        }
        if (trace_paged) {
            LOG(INFO) << "NCCL paged transfer run"
                      << " compact=" << compact_windows
                      << " cache_compact=" << cache_compact_windows
                      << " page_bytes=" << request.page_bytes
                      << " layer_begin=" << layer
                      << " layer_count=" << run_layers
                      << " src_base=0x" << std::hex << ctx.source_base
                      << " dst_base=0x" << ctx.target_base << std::dec
                      << " src_length=" << ctx.source_length
                      << " dst_length=" << ctx.target_length
                      << " dst_offset=" << ctx.target_offset;
        }
        if (!runs.empty()) {
            const auto& first = runs.front().ctx;
            if (ctx.local_device != first.local_device ||
                ctx.remote_device != first.remote_device ||
                ctx.session_key != first.session_key) {
                return Status::InvalidArgument(
                    "Paged transfer layers must use one NCCL session"
                    LOC_MARK);
            }
        }

        PagedRun run;
        run.ctx = std::move(ctx);
        run.src_layer_offsets.reserve(run_layers);
        run.dst_layer_offsets.reserve(run_layers);
        for (size_t i = layer; i < layer + run_layers; ++i) {
            const uint64_t src_ptr = static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(request.src_layer_ptrs[i]));
            const uint64_t dst_ptr = request.dst_layer_ptrs[i];
            uint64_t src_start = 0;
            uint64_t dst_start = 0;
            CHECK_STATUS(checked_add(src_ptr, source_base_offset,
                                     "Paged source", src_start));
            CHECK_STATUS(checked_add(dst_ptr, target_base_offset,
                                     "Paged target", dst_start));
            run.src_layer_offsets.push_back(
                static_cast<size_t>(src_start - run_source_base));
            run.dst_layer_offsets.push_back(
                static_cast<size_t>(dst_start - run_target_base));
        }
        runs.push_back(std::move(run));
        layer += run_layers;
    }
    if (trace_paged &&
        (runs.size() != 1 ||
         runs.front().src_layer_offsets.size() != layer_count)) {
        LOG(INFO) << "NCCL paged transfer compact layers=" << layer_count
                  << " runs=" << runs.size()
                  << " cache_compact=" << cache_compact_windows
                  << " page_bytes=" << request.page_bytes
                  << " pages=" << page_count
                  << " max_window_bytes=" << kMaxPagedWindowBytes;
    }

    const bool pair_paged_windows_requested =
        full_paired_windows_requested ||
        envFlagEnabled("MC_NCCL_PAIR_PAGED_WINDOWS");
    bool pair_paged_windows =
        pair_paged_windows_requested &&
        (full_paired_windows_requested || !cache_compact_windows);
    if (pair_paged_windows) {
        for (const auto& run : runs) {
            if (run.ctx.source_length == 0 ||
                run.ctx.target_length == 0 ||
                (!full_paired_windows_requested &&
                 run.ctx.source_length != run.ctx.target_length)) {
                pair_paged_windows = false;
                break;
            }
        }
    }
    if (!pair_paged_windows && full_paired_windows_requested) {
        return Status::InvalidArgument(
            "NCCL full paired source or target extent is empty"
            LOC_MARK);
    }
    const bool full_paired_windows =
        full_paired_windows_requested && pair_paged_windows;
    if (pair_paged_windows) {
        for (auto& run : runs) {
            run.ctx.window_key = makePairedWindowKey(
                run.ctx.session_key, run.ctx.source_base,
                run.ctx.target_base, run.ctx.source_length,
                run.ctx.target_length);
        }
    } else if (pair_paged_windows_requested && trace_paged) {
        LOG(WARNING)
            << "NCCL paired paged windows unavailable"
            << " page_bytes=" << request.page_bytes
            << " cache_compact=" << cache_compact_windows
            << " runs=" << runs.size();
    }

    std::shared_ptr<CommState> comm_state;
    CHECK_STATUS(ensureComm(runs.front().ctx, comm_state));
    // Paged KV issues one transfer per page per layer. The LSA path becomes a
    // cudaMemcpyAsync page loop, which is much slower than fused GIN on GB200.
    // Keep LSA for regular contiguous transfers, but route paged transfers to GIN
    // even when the peer is in the local shared-address team.
    const bool use_lsa = false;
    if (!use_lsa && !comm_state->dev_comm_created) {
        return Status::InvalidArgument(
            "NCCL paged GIN requires a device communicator; set "
            "MC_NCCL_FORCE_GIN=1 and NCCL_CUMEM_ENABLE=1 for LSA peers"
            LOC_MARK);
    }
    const int lanes = static_cast<int>(comm_state->lanes);
    const bool batch_paged_windows =
        !use_lsa && !pair_paged_windows && !cache_compact_windows &&
        envFlagEnabled("MC_NCCL_BATCH_PAGED_WINDOWS");
    // Window registration and signal epochs are ordered collectively on one
    // communicator. Interleaving synchronous callers can otherwise deadlock a
    // later epoch ahead of the put that produces an earlier epoch.
    std::unique_lock<std::mutex> submission_lock(comm_state->submission_mu);
    const bool wait_ack =
        !envFlagEnabled("MC_NCCL_PAGED_UNSAFE_LOCAL_COMPLETION");
    uint64_t signal_value = 0;

    std::vector<TransferContext> batch_contexts;
    std::vector<std::shared_ptr<WindowState>> batch_target_states;
    std::vector<std::shared_ptr<WindowState>> batch_source_states;
    Status status = Status::OK();
    double batch_register_ms = 0.0;
    if (batch_paged_windows) {
        batch_contexts.reserve(runs.size());
        for (const auto& run : runs) {
            batch_contexts.push_back(run.ctx);
        }
        auto batch_register_start = SteadyClock::now();
        status = ensurePagedWindowsBatch(
            batch_contexts, comm_state, batch_target_states,
            batch_source_states);
        batch_register_ms = elapsedMs(batch_register_start);
    }

    int previous_device = 0;
    bool device_changed = false;
    if (status.ok()) {
        status = setCudaDevice(runs.front().ctx.local_device,
                               previous_device);
        device_changed = status.ok();
    }
    // Legacy cudaFree synchronizes the device. Keep temporary workspace
    // lifetime ordered on the completion stream when async allocation is on.
    const bool async_device_alloc =
        envFlagEnabled("MC_NCCL_PAGED_ASYNC_ALLOC");
    int32_t* d_src_pages = nullptr;
    int32_t* d_dst_pages = nullptr;
    TentNcclPagedTransferJob* d_jobs = nullptr;
    size_t* d_src_layer_offsets = nullptr;
    size_t* d_dst_layer_offsets = nullptr;
    cudaEvent_t event = nullptr;
    bool work_enqueued = false;
    double setup_ms = 0.0;
    double ensure_target_ms = 0.0;
    double ensure_source_ms = 0.0;
    double run_prepare_ms = 0.0;
    double launch_ms = 0.0;
    double sync_ms = 0.0;
    double cleanup_ms = 0.0;
    double cleanup_local_ms = 0.0;
    double cleanup_remote_wait_ms = 0.0;

    auto cleanup_device_allocations = [&]() {
        auto free_one = [&](void* ptr, const char* name) {
            if (!ptr) return;
            const auto err =
                async_device_alloc
                    ? cudaFreeAsync(ptr, comm_state->completion_stream)
                    : cudaFree(ptr);
            if (err != cudaSuccess) {
                if (status.ok()) {
                    status = cudaStatus(err, name);
                } else {
                    LOG(WARNING) << name << ": " << cudaGetErrorString(err);
                }
            }
        };
        free_one(d_dst_layer_offsets, "cudaFree(paged dst layer offsets)");
        free_one(d_src_layer_offsets, "cudaFree(paged src layer offsets)");
        free_one(d_jobs, "cudaFree(paged jobs)");
        free_one(d_dst_pages, "cudaFree(paged dst pages)");
        free_one(d_src_pages, "cudaFree(paged src pages)");
    };


    auto cleanup_paged_windows = [&]() -> Status {
        // Compact KV windows are small enough to cache and reuse across
        // requests. Non-compact paged transfers carry DSV4 extra state and can
        // be hundreds of MB per tensor, so caching them quickly exhausts NCCL's
        // symmetric memory space.
        if (status.ok() && full_paired_windows) return Status::OK();
        if (status.ok() && compact_windows && cache_compact_windows) return Status::OK();
        Status cleanup_status;
        auto merge_status = [&](const Status& next, const char* action,
                                const std::string& window_key) {
            if (next.ok()) return;
            LOG(WARNING) << action << " failed for key=" << window_key
                         << ": " << next.ToString();
            if (cleanup_status.ok()) cleanup_status = next;
        };
        if (pair_paged_windows || batch_paged_windows) {
            std::vector<std::string> window_keys;
            window_keys.reserve(
                runs.size() * (pair_paged_windows ? 1 : 2));
            for (const auto& run : runs) {
                window_keys.push_back(run.ctx.window_key);
                if (!pair_paged_windows) {
                    window_keys.push_back(run.ctx.source_window_key);
                }
            }

            Status remote_status;
            std::thread remote_thread;
            try {
                remote_thread = std::thread(
                    [this, ctx = runs.front().ctx, window_keys,
                     &remote_status]() {
                        remote_status =
                            releaseRemoteWindowsBatch(ctx, window_keys);
                    });
            } catch (const std::exception& e) {
                merge_status(
                    Status::InternalError(
                        std::string(
                            "spawn NCCL batch window deregister RPC: ") +
                        e.what() + LOC_MARK),
                    "NCCL remote batch window deregister", "batch");
            }

            auto local_start = SteadyClock::now();
            for (const auto& window_key : window_keys) {
                merge_status(releaseWindow(window_key),
                             "NCCL local batch window deregister",
                             window_key);
            }
            cleanup_local_ms += elapsedMs(local_start);

            auto remote_join_start = SteadyClock::now();
            if (remote_thread.joinable()) remote_thread.join();
            cleanup_remote_wait_ms += elapsedMs(remote_join_start);
            merge_status(remote_status,
                         "NCCL remote batch window deregister", "batch");
            return cleanup_status;
        }

        auto cleanup_pair = [&](const TransferContext& ctx,
                                const std::string& window_key) {
            Status remote_status;
            std::thread remote_thread;
            try {
                remote_thread = std::thread([this, ctx, window_key,
                                             &remote_status]() {
                    remote_status = releaseRemoteWindow(ctx, window_key);
                });
            } catch (const std::exception& e) {
                merge_status(Status::InternalError(
                                 std::string("spawn NCCL window deregister RPC: ") +
                                 e.what() + LOC_MARK),
                             "NCCL remote window deregister", window_key);
            }

            auto local_start = SteadyClock::now();
            Status local_status = releaseWindow(window_key);
            cleanup_local_ms += elapsedMs(local_start);
            merge_status(local_status, "NCCL local window deregister",
                         window_key);
            auto remote_join_start = SteadyClock::now();
            if (remote_thread.joinable()) remote_thread.join();
            cleanup_remote_wait_ms += elapsedMs(remote_join_start);
            merge_status(remote_status, "NCCL remote window deregister",
                         window_key);
        };


        for (const auto& run : runs) {
            cleanup_pair(run.ctx, run.ctx.window_key);
            cleanup_pair(run.ctx, run.ctx.source_window_key);
        }
        return cleanup_status;
    };

    if (layer_count > std::numeric_limits<size_t>::max() /
                          sizeof(size_t)) {
        return Status::InvalidArgument(
            "Paged transfer layer offset table byte size overflows"
            LOC_MARK);
    }
    if (runs.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidArgument(
            "Paged transfer job table is too large" LOC_MARK);
    }
    if (runs.size() > std::numeric_limits<size_t>::max() /
                          sizeof(TentNcclPagedTransferJob)) {
        return Status::InvalidArgument(
            "Paged transfer job table byte size overflows" LOC_MARK);
    }

    const size_t layer_offsets_bytes = layer_count * sizeof(size_t);
    const size_t jobs_bytes =
        runs.size() * sizeof(TentNcclPagedTransferJob);
    std::vector<size_t> run_layer_offset_starts;
    std::vector<size_t> flattened_src_layer_offsets;
    std::vector<size_t> flattened_dst_layer_offsets;
    run_layer_offset_starts.reserve(runs.size());
    flattened_src_layer_offsets.reserve(layer_count);
    flattened_dst_layer_offsets.reserve(layer_count);
    for (const auto& run : runs) {
        if (run.src_layer_offsets.size() !=
            run.dst_layer_offsets.size()) {
            return Status::InternalError(
                "Paged transfer run layer offset counts differ" LOC_MARK);
        }
        run_layer_offset_starts.push_back(
            flattened_src_layer_offsets.size());
        flattened_src_layer_offsets.insert(
            flattened_src_layer_offsets.end(),
            run.src_layer_offsets.begin(), run.src_layer_offsets.end());
        flattened_dst_layer_offsets.insert(
            flattened_dst_layer_offsets.end(),
            run.dst_layer_offsets.begin(), run.dst_layer_offsets.end());
    }
    if (flattened_src_layer_offsets.size() != layer_count ||
        flattened_dst_layer_offsets.size() != layer_count) {
        return Status::InternalError(
            "Paged transfer flattened layer count differs" LOC_MARK);
    }
    std::vector<TentNcclPagedTransferJob> jobs(runs.size());

    auto allocate_device = [&](void** ptr, size_t bytes,
                               const char* name) -> Status {
        const auto err =
            async_device_alloc
                ? cudaMallocAsync(ptr, bytes,
                                  comm_state->completion_stream)
                : cudaMalloc(ptr, bytes);
        return cudaStatus(err, name);
    };

    if (status.ok() && !use_lsa) {
        auto setup_start = SteadyClock::now();
        const size_t page_table_bytes = page_count * sizeof(int32_t);
        cudaError_t err = cudaSuccess;
        status = allocate_device(reinterpret_cast<void**>(&d_src_pages),
                                 page_table_bytes,
                                 "allocate(paged src pages)");
        if (status.ok()) {
            status = allocate_device(reinterpret_cast<void**>(&d_dst_pages),
                                     page_table_bytes,
                                     "allocate(paged dst pages)");
        }
        if (status.ok()) {
            status = allocate_device(reinterpret_cast<void**>(&d_jobs),
                                     jobs_bytes,
                                     "allocate(paged jobs)");
        }
        if (status.ok()) {
            status = allocate_device(
                reinterpret_cast<void**>(&d_src_layer_offsets),
                layer_offsets_bytes,
                "allocate(paged src layer offsets)");
        }
        if (status.ok()) {
            status = allocate_device(
                reinterpret_cast<void**>(&d_dst_layer_offsets),
                layer_offsets_bytes,
                "allocate(paged dst layer offsets)");
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(d_src_pages,
                                  src_page_indices.data(),
                                  page_table_bytes,
                                  cudaMemcpyHostToDevice,
                                  comm_state->completion_stream);
            status = cudaStatus(err, "cudaMemcpyAsync(paged src pages)");
            if (status.ok()) work_enqueued = true;
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(d_dst_pages,
                                  dst_page_indices.data(), page_table_bytes,
                                  cudaMemcpyHostToDevice,
                                  comm_state->completion_stream);
            status = cudaStatus(err, "cudaMemcpyAsync(paged dst pages)");
            if (status.ok()) work_enqueued = true;
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(
                d_src_layer_offsets,
                flattened_src_layer_offsets.data(), layer_offsets_bytes,
                cudaMemcpyHostToDevice,
                comm_state->completion_stream);
            status = cudaStatus(
                err, "cudaMemcpyAsync(paged src layer offsets)");
            if (status.ok()) work_enqueued = true;
        }
        if (status.ok()) {
            err = cudaMemcpyAsync(
                d_dst_layer_offsets,
                flattened_dst_layer_offsets.data(), layer_offsets_bytes,
                cudaMemcpyHostToDevice,
                comm_state->completion_stream);
            status = cudaStatus(
                err, "cudaMemcpyAsync(paged dst layer offsets)");
            if (status.ok()) work_enqueued = true;
        }
        setup_ms += elapsedMs(setup_start);
    }

    TentNcclPagedKvLayout layout;
    layout.page_bytes = request.page_bytes;
    layout.src_page_stride_bytes = src_page_stride;
    layout.dst_page_stride_bytes = dst_page_stride;

    for (size_t run_index = 0; status.ok() && run_index < runs.size();
         ++run_index) {
        const auto& run = runs[run_index];
        const auto& ctx = run.ctx;
        std::shared_ptr<WindowState> window_state;
        if (pair_paged_windows) {
            auto ensure_paired_start = SteadyClock::now();
            status = ensurePairedWindow(ctx, comm_state, window_state);
            ensure_target_ms += elapsedMs(ensure_paired_start);
        } else if (batch_paged_windows) {
            window_state = batch_target_states[run_index];
        } else {
            auto ensure_target_start = SteadyClock::now();
            status = ensureWindow(ctx, comm_state, window_state);
            ensure_target_ms += elapsedMs(ensure_target_start);
            if (!status.ok() && compact_windows &&
                cache_compact_windows &&
                retry_compact_window_register) {
                LOG(WARNING)
                    << "NCCL paged target window registration failed; "
                       "evicting cached windows and retrying once: "
                    << status.ToString();
                Status eviction_status =
                    releaseCachedWindowsForSession(ctx);
                if (!eviction_status.ok()) {
                    LOG(WARNING)
                        << "Cached NCCL window eviction before target "
                           "retry had errors: "
                        << eviction_status.ToString();
                }
                window_state.reset();
                ensure_target_start = SteadyClock::now();
                status = ensureWindow(ctx, comm_state, window_state);
                ensure_target_ms += elapsedMs(ensure_target_start);
            }
        }
        if (!status.ok()) break;

        if (use_lsa) {
            if (!containsRange(window_state->base, window_state->length,
                               ctx.target_request_base,
                               ctx.target_request_length)) {
                status = Status::InternalError(
                    "Paged target LSA window does not contain request"
                    LOC_MARK);
                break;
            }
            const size_t dst_base_offset = static_cast<size_t>(
                ctx.target_request_base - window_state->base);
            void* peer_ptr = nullptr;
            status = getLsaPeerPointer(window_state->window,
                                       dst_base_offset,
                                       comm_state->peer_rank, &peer_ptr);
            if (!status.ok()) break;
            const char* src_base = reinterpret_cast<const char*>(
                static_cast<uintptr_t>(ctx.source_request_base));
            char* dst_base = static_cast<char*>(peer_ptr);
            for (size_t run_layer = 0;
                 status.ok() && run_layer < run.src_layer_offsets.size();
                 ++run_layer) {
                for (size_t page = 0; page < page_count; ++page) {
                    const int32_t src_page = src_page_indices[page];
                    const int32_t dst_page = dst_page_indices[page];
                    if (src_page < 0 || dst_page < 0) continue;
                    auto err = cudaMemcpyAsync(
                        dst_base + run.dst_layer_offsets[run_layer] +
                            static_cast<size_t>(dst_page) *
                                dst_page_stride,
                        src_base + run.src_layer_offsets[run_layer] +
                            static_cast<size_t>(src_page) *
                                src_page_stride,
                        request.page_bytes, cudaMemcpyDeviceToDevice,
                        comm_state->completion_stream);
                    status = cudaStatus(err, "cudaMemcpyAsync(NCCL paged LSA)");
                    if (!status.ok()) break;
                    work_enqueued = true;
                }
            }
            continue;
        }

        std::shared_ptr<WindowState> source_window_state;
        if (pair_paged_windows) {
            source_window_state = window_state;
        } else if (batch_paged_windows) {
            source_window_state = batch_source_states[run_index];
        } else {
            auto ensure_source_start = SteadyClock::now();
            status = ensureSourceWindow(
                ctx, comm_state, source_window_state);
            ensure_source_ms += elapsedMs(ensure_source_start);
            if (!status.ok() && compact_windows &&
                cache_compact_windows &&
                retry_compact_window_register) {
                LOG(WARNING)
                    << "NCCL paged source window registration failed; "
                       "evicting cached windows and retrying once: "
                    << status.ToString();
                Status eviction_status =
                    releaseCachedWindowsForSession(ctx);
                if (!eviction_status.ok()) {
                    LOG(WARNING)
                        << "Cached NCCL window eviction before source "
                           "retry had errors: "
                        << eviction_status.ToString();
                }
                window_state.reset();
                source_window_state.reset();
                auto retry_target_start = SteadyClock::now();
                status = ensureWindow(
                    ctx, comm_state, window_state);
                ensure_target_ms += elapsedMs(retry_target_start);
                if (status.ok()) {
                    ensure_source_start = SteadyClock::now();
                    status = ensureSourceWindow(
                        ctx, comm_state, source_window_state);
                    ensure_source_ms +=
                        elapsedMs(ensure_source_start);
                }
            }
        }
        if (!status.ok()) break;

        size_t src_base_offset = 0;
        size_t dst_base_offset = 0;
        if (!containsRange(source_window_state->base,
                           source_window_state->length,
                           ctx.source_request_base,
                           ctx.source_request_length)) {
            LOG(ERROR) << "Paged source window does not contain request"
                       << " session=" << ctx.session_key
                       << " key=" << ctx.source_window_key
                       << " state_base=0x" << std::hex
                       << source_window_state->base
                       << " request_base=0x" << ctx.source_request_base
                       << std::dec
                       << " state_length=" << source_window_state->length
                       << " request_length="
                       << ctx.source_request_length
                       << " ctx_base=0x" << std::hex << ctx.source_base
                       << std::dec
                       << " ctx_length=" << ctx.source_length
                       << " staged="
                       << (source_window_state->local_buffer != nullptr);
            status = Status::InternalError(
                "Paged source window does not contain request" LOC_MARK);
            break;
        }
        const uint64_t target_window_base =
            pair_paged_windows ? ctx.target_base : window_state->base;
        const uint64_t target_window_length =
            pair_paged_windows ? ctx.target_length
                               : window_state->length;
        if (!containsRange(target_window_base, target_window_length,
                           ctx.target_request_base,
                           ctx.target_request_length)) {
            LOG(ERROR) << "Paged target window does not contain request"
                       << " session=" << ctx.session_key
                       << " key=" << ctx.window_key
                       << " state_base=0x" << std::hex
                       << target_window_base
                       << " request_base=0x" << ctx.target_request_base
                       << std::dec
                       << " state_length=" << target_window_length
                       << " request_length="
                       << ctx.target_request_length
                       << " ctx_base=0x" << std::hex << ctx.target_base
                       << std::dec
                       << " ctx_length=" << ctx.target_length;
            status = Status::InternalError(
                "Paged target window does not contain request" LOC_MARK);
            break;
        }
        src_base_offset = static_cast<size_t>(
            ctx.source_request_base - source_window_state->base);
        dst_base_offset = static_cast<size_t>(
            ctx.target_request_base - target_window_base);

        if (source_window_state->local_buffer) {
            if (trace_paged) {
                LOG(WARNING) << "NCCL paged trace source_stage begin"
                             << " session=" << ctx.session_key
                             << " src_request_base=0x" << std::hex
                             << ctx.source_request_base << std::dec
                             << " src_request_length=" << ctx.source_request_length
                             << " window_base=0x" << std::hex
                             << source_window_state->base << std::dec
                             << " window_length=" << source_window_state->length;
            }
            status = copySourceWindowBytes(ctx, source_window_state, true,
                                           comm_state->completion_stream);
            if (trace_paged) {
                LOG(WARNING) << "NCCL paged trace source_stage queued"
                             << " session=" << ctx.session_key
                             << " status=" << status.ToString();
            }
            if (status.ok()) work_enqueued = true;
        }
        if (!status.ok()) break;

        auto run_prepare_start = SteadyClock::now();
        const size_t run_layer_count = run.src_layer_offsets.size();
        auto& job = jobs[run_index];
        job.src_page_table = d_src_pages;
        job.dst_page_table = d_dst_pages;
        job.num_pages = static_cast<int>(page_count);
        job.layer_begin = 0;
        job.layer_end = static_cast<int>(run_layer_count);
        job.src_layer_stride = 0;
        job.dst_layer_stride = 0;
        job.src_base_offset = src_base_offset;
        job.dst_base_offset = dst_base_offset;
        job.src_layer_offsets =
            d_src_layer_offsets + run_layer_offset_starts[run_index];
        job.dst_layer_offsets =
            d_dst_layer_offsets + run_layer_offset_starts[run_index];
        job.dst_window = window_state->window;
        job.src_window = source_window_state->window;
        job.page_bytes = request.page_bytes;
        job.src_page_stride_bytes = src_page_stride;
        job.dst_page_stride_bytes = dst_page_stride;
        run_prepare_ms += elapsedMs(run_prepare_start);
    }

    if (status.ok() && !use_lsa) {
        const int uniform_job_num_pages = jobs.front().num_pages;
        const int uniform_job_num_layers =
            jobs.front().layer_end - jobs.front().layer_begin;
        const bool uniform_jobs =
            jobs.size() > 1 && uniform_job_num_pages > 0 &&
            uniform_job_num_layers > 0 &&
            std::all_of(jobs.begin(), jobs.end(), [&](const auto& job) {
                return job.num_pages == uniform_job_num_pages &&
                       job.layer_end - job.layer_begin ==
                           uniform_job_num_layers;
            });
        if (uniform_jobs) {
            layout.uniform_job_num_pages = uniform_job_num_pages;
            layout.uniform_job_num_layers = uniform_job_num_layers;
        }
        auto run_prepare_start = SteadyClock::now();
        auto err = cudaMemcpyAsync(
            d_jobs, jobs.data(), jobs_bytes, cudaMemcpyHostToDevice,
            comm_state->completion_stream);
        status = cudaStatus(err, "cudaMemcpyAsync(paged jobs)");
        if (status.ok()) work_enqueued = true;
        run_prepare_ms += elapsedMs(run_prepare_start);
    }

    const bool batch_paged_jobs =
        !use_lsa && jobs.size() > 1 &&
        (page_count == 1 ||
         (layout.uniform_job_num_pages > 0 &&
          layout.uniform_job_num_layers > 0) ||
         envFlagEnabled("MC_NCCL_BATCH_PAGED_JOBS"));
    if (status.ok() && !use_lsa) {
        if (wait_ack) {
            signal_value = comm_state->signal_epoch.fetch_add(
                               1, std::memory_order_acq_rel) +
                           1;
        }
        auto launch_start = SteadyClock::now();
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace launch begin"
                         << " session=" << runs.front().ctx.session_key
                         << " peer_rank=" << comm_state->peer_rank
                         << " lanes=" << lanes
                         << " jobs=" << jobs.size()
                         << " batch_jobs=" << batch_paged_jobs
                         << " layers=" << layer_count
                         << " pages=" << page_count;
        }
        const size_t launch_count =
            batch_paged_jobs ? 1 : jobs.size();
        for (size_t launch_index = 0;
             status.ok() && launch_index < launch_count;
             ++launch_index) {
            const size_t job_index =
                batch_paged_jobs ? 0 : launch_index;
            const unsigned long long completion_signal =
                launch_index + 1 == launch_count ? signal_value : 0;
            const int num_jobs = batch_paged_jobs
                                     ? static_cast<int>(jobs.size())
                                     : 1;
            auto err = tentNcclGinLaunchPagedPut(
                comm_state->dev_comm, comm_state->peer_rank, lanes,
                jobs[job_index].dst_window,
                jobs[job_index].src_window, layout,
                d_jobs + job_index, num_jobs, completion_signal,
                comm_state->completion_stream);
            status = cudaStatus(err, "tentNcclGinLaunchPagedPut");
            if (status.ok()) work_enqueued = true;
        }
        if (status.ok() && wait_ack) {
            status = postRemoteWaitSignal(runs.front().ctx, signal_value);
            if (status.ok()) {
                auto err = tentNcclGinLaunchWaitSignal(
                    comm_state->dev_comm, lanes, lanes,
                    static_cast<unsigned long long>(signal_value),
                    comm_state->completion_stream);
                status = cudaStatus(err, "tentNcclGinLaunchWaitSignal(paged ack)");
                if (status.ok()) work_enqueued = true;
            }
        }
        launch_ms += elapsedMs(launch_start);
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace launch done"
                         << " session=" << runs.front().ctx.session_key
                         << " status=" << status.ToString()
                         << " launch_ms=" << launch_ms;
        }
    }

    if (status.ok()) {
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace event create";
        }
        auto err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        status = cudaStatus(err, "cudaEventCreateWithFlags(paged sync)");
    }
    if (status.ok()) {
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace event record";
        }
        auto err = cudaEventRecord(event, comm_state->completion_stream);
        status = cudaStatus(err, "cudaEventRecord(paged sync)");
    }
    if (status.ok()) {
        auto sync_start = SteadyClock::now();
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace event sync begin";
        }
        auto err = cudaEventSynchronize(event);
        status = cudaStatus(err, "cudaEventSynchronize(paged sync)");
        sync_ms += elapsedMs(sync_start);
        if (trace_paged) {
            LOG(WARNING) << "NCCL paged trace event sync done"
                         << " status=" << status.ToString()
                         << " sync_ms=" << sync_ms;
        }
    } else if (work_enqueued && comm_state) {
        auto err = cudaStreamSynchronize(comm_state->completion_stream);
        if (err != cudaSuccess) {
            LOG(WARNING) << "cudaStreamSynchronize(paged cleanup): "
                         << cudaGetErrorString(err);
        }
    }

    auto cleanup_start = SteadyClock::now();
    Status paged_cleanup_status = cleanup_paged_windows();
    cleanup_ms = elapsedMs(cleanup_start);
    if (status.ok()) {
        status = paged_cleanup_status;
    } else if (!paged_cleanup_status.ok()) {
        LOG(WARNING) << "NCCL paged window cleanup after failure also "
                        "failed: "
                     << paged_cleanup_status.ToString();
    }

    if (profile_paged) {
        LOG(WARNING) << "NCCL paged profile"
                     << " compact=" << compact_windows
                     << " cache_compact=" << cache_compact_windows
                     << " async_alloc=" << async_device_alloc
                     << " paired_windows=" << pair_paged_windows
                     << " full_paired_windows=" << full_paired_windows
                     << " batch_windows=" << batch_paged_windows
                     << " batch_jobs=" << batch_paged_jobs
                     << " wait_ack=" << wait_ack
                     << " batch_register_ms=" << batch_register_ms
                     << " page_bytes=" << request.page_bytes
                     << " layers=" << layer_count
                     << " pages=" << page_count
                     << " runs=" << runs.size()
                     << " setup_ms=" << setup_ms
                     << " ensure_target_ms=" << ensure_target_ms
                     << " ensure_source_ms=" << ensure_source_ms
                     << " run_prepare_ms=" << run_prepare_ms
                     << " launch_ms=" << launch_ms
                     << " sync_ms=" << sync_ms
                     << " cleanup_ms=" << cleanup_ms
                     << " cleanup_local_ms=" << cleanup_local_ms
                     << " cleanup_remote_wait_ms=" << cleanup_remote_wait_ms
                     << " status=" << status.ToString();
    }
    if (trace_paged) {
        LOG(WARNING) << "NCCL paged trace end"
                     << " target_id=" << request.target_id
                     << " session=" << (runs.empty() ? std::string("") : runs.front().ctx.session_key)
                     << " local_device=" << (runs.empty() ? -1 : runs.front().ctx.local_device)
                     << " remote_device=" << (runs.empty() ? -1 : runs.front().ctx.remote_device)
                     << " page_bytes=" << request.page_bytes
                     << " layers=" << layer_count
                     << " pages=" << page_count
                     << " status=" << status.ToString();
    }

    if (event) cudaEventDestroy(event);
    cleanup_device_allocations();
    if (device_changed) cudaSetDevice(previous_device);
    return status;
}

Status NcclTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    if (!thread_pool_)
        return Status::InternalError("NCCL transport is not installed" LOC_MARK);
    if (request_list.size() + nccl_batch->task_list.size() >
        nccl_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);

    std::vector<NcclTask*> new_tasks;
    new_tasks.reserve(request_list.size());
    for (const auto& request : request_list) {
        nccl_batch->task_list.emplace_back();
        auto& task = nccl_batch->task_list.back();
        task.request = request;
        task.status_word.store(TransferStatusEnum::PENDING,
                               std::memory_order_release);
        task.transferred_bytes.store(0, std::memory_order_release);
        task.completion_event.store(nullptr, std::memory_order_release);

        new_tasks.push_back(&task);
    }

    if (new_tasks.empty()) return Status::OK();

    auto task_ptrs = std::make_shared<std::vector<NcclTask*>>(
        std::move(new_tasks));
    try {
        // Communicator and window setup can perform RPCs; keep it off the
        // transfer-engine submission path.
        thread_pool_->enqueue([this, nccl_batch, task_ptrs]() {
            for (auto* task : *task_ptrs) {
                try {
                    startTransfer(task, nccl_batch);
                } catch (const std::exception& e) {
                    markFailed(
                        *task, std::string("NCCL transfer worker failed: ") +
                                   e.what());
                } catch (...) {
                    markFailed(*task, "NCCL transfer worker failed");
                }
            }
        });
    } catch (const std::exception& e) {
        for (auto* task : *task_ptrs) {
            markFailed(
                *task, std::string("NCCL submit worker enqueue failed: ") +
                           e.what());
        }
        return Status::InternalError(
            std::string("NCCL submit worker enqueue failed: ") + e.what() +
            LOC_MARK);
    }

    return Status::OK();
}

Status NcclTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                        TransferStatus& status) {
    auto nccl_batch = dynamic_cast<NcclSubBatch*>(batch);
    if (!nccl_batch)
        return Status::InvalidArgument("Invalid NCCL sub-batch" LOC_MARK);
    if (task_id < 0 || task_id >= (int)nccl_batch->task_list.size()) {
        return Status::InvalidArgument("Invalid task id" LOC_MARK);
    }

    auto& task = nccl_batch->task_list[task_id];
    auto current = task.status_word.load(std::memory_order_acquire);
    auto event = task.completion_event.load(std::memory_order_acquire);
    if (current == TransferStatusEnum::PENDING && event) {
        auto err = cudaEventQuery(event);
        if (err == cudaSuccess) {
            task.transferred_bytes.store(task.request.length,
                                         std::memory_order_release);
            task.status_word.store(TransferStatusEnum::COMPLETED,
                                   std::memory_order_release);
            current = TransferStatusEnum::COMPLETED;
        } else if (err != cudaErrorNotReady) {
            task.status_word.store(TransferStatusEnum::FAILED,
                                   std::memory_order_release);
            current = TransferStatusEnum::FAILED;
        }
    }

    status.s = current;
    status.transferred_bytes =
        task.transferred_bytes.load(std::memory_order_acquire);
    return Status::OK();
}

Status NcclTransport::allocateLocalMemory(void** addr, size_t size,
                                          MemoryOptions& options) {
    LocationParser location(options.location);
    if (location.type() != "cuda") {
        return Status::InvalidArgument(
            "NCCL transport only allocates CUDA memory" LOC_MARK);
    }

    int previous_device = 0;
    CHECK_CUDA(cudaGetDevice(&previous_device));
    CHECK_CUDA(cudaSetDevice(location.index()));
    auto status = ncclStatus(ncclMemAlloc(addr, size), "ncclMemAlloc");
    cudaSetDevice(previous_device);
    if (!status.ok()) return status;

    std::lock_guard<std::mutex> lock(allocation_mutex_);
    nccl_allocations_.insert(reinterpret_cast<uint64_t>(*addr));
    return Status::OK();
}

Status NcclTransport::freeLocalMemory(void* addr, size_t size) {
    {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        auto it = nccl_allocations_.find(reinterpret_cast<uint64_t>(addr));
        if (it == nccl_allocations_.end()) {
            return Platform::getLoader().free(addr, size);
        }
        nccl_allocations_.erase(it);
    }
    CHECK_NCCL(ncclMemFree(addr));
    return Status::OK();
}

bool NcclTransport::isNcclAllocated(uint64_t addr) const {
    std::lock_guard<std::mutex> lock(allocation_mutex_);
    return nccl_allocations_.count(addr) != 0;
}

bool NcclTransport::isCudaLocation(const std::string& location) const {
    return LocationParser(location).type() == "cuda";
}

Status NcclTransport::addMemoryBuffer(BufferDesc& desc,
                                      const MemoryOptions& options) {
    if (!isCudaLocation(desc.location)) return Status::OK();

    const bool nccl_allocated = isNcclAllocated(desc.addr);
    if (!nccl_allocated && !allow_external_window_buffers_) {
        if (options.type == NCCL) {
            return Status::InvalidArgument(
                "NCCL host RMA requires ncclMemAlloc/VMM-compatible CUDA "
                "buffers; set transports/nccl/allow_external_window_buffers "
                "only if the caller guarantees window registration support" LOC_MARK);
        }
        return Status::OK();
    }

    desc.transports.push_back(TransportType::NCCL);
    desc.transport_attrs[TransportType::NCCL] =
        nccl_allocated ? "allocator=ncclMemAlloc;window=deferred"
                       : "allocator=external;window=deferred";
    return Status::OK();
}

Status NcclTransport::removeMemoryBuffer(BufferDesc& desc) {
    desc.transport_attrs.erase(TransportType::NCCL);
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
