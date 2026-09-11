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

#include "nccl_collective_executor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include <glog/logging.h>

#ifdef USE_NCCL_PG
#include <nccl.h>
#endif

namespace mooncake {
namespace {

#ifdef USE_NCCL_PG

PGError makeNcclError(const char* operation, ncclResult_t result) {
    return PGError{
        PGErrorCode::SystemError,
        std::string(operation) + " failed: " + ncclGetErrorString(result)};
}

std::optional<ncclDataType_t> toNcclDataType(DataType datatype) {
    switch (datatype) {
        case DataType::Int8:
            return ncclInt8;
        case DataType::Uint8:
        case DataType::Bool:
            return ncclUint8;
        case DataType::Int32:
            return ncclInt32;
        case DataType::Uint32:
            return ncclUint32;
        case DataType::Int64:
            return ncclInt64;
        case DataType::Uint64:
            return ncclUint64;
        case DataType::Float16:
            return ncclFloat16;
        case DataType::Float32:
            return ncclFloat32;
        case DataType::Float64:
            return ncclFloat64;
        case DataType::Bfloat16:
            return ncclBfloat16;
        case DataType::Float8e4m3fn:
            return ncclFloat8e4m3;
        case DataType::Float8e5m2:
            return ncclFloat8e5m2;
        case DataType::Int16:
        case DataType::Uint16:
        case DataType::Float8e4m3fnuz:
        case DataType::Float8e5m2fnuz:
        case DataType::Float8e8m0fnu:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<ncclRedOp_t> toNcclReduceOp(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:
            return ncclSum;
        case ReduceOp::Avg:
            return ncclAvg;
        case ReduceOp::Product:
            return ncclProd;
        case ReduceOp::Min:
            return ncclMin;
        case ReduceOp::Max:
            return ncclMax;
    }
    return std::nullopt;
}

#endif

#ifndef USE_NCCL_PG
auto notCompiled() {
    return makePGError(PGErrorCode::NotSupported,
                       "Mooncake PG was built without USE_NCCL_PG");
}
#endif

}  // namespace

struct NcclCollectiveExecutor::PendingOperation {
    std::shared_ptr<GpuCollectiveStatus> status =
        std::make_shared<GpuCollectiveStatus>();
#ifdef USE_NCCL_PG
    cudaEvent_t event = nullptr;
    bool recorded = false;
    bool captured = false;
    int device = -1;
    std::chrono::steady_clock::time_point started =
        std::chrono::steady_clock::now();

    ~PendingOperation() noexcept {
        if (!event) return;
        int previous_device = -1;
        const auto get_result = cudaGetDevice(&previous_device);
        if (device >= 0) (void)cudaSetDevice(device);
        const auto result = cudaEventDestroy(event);
        if (result != cudaSuccess) {
            LOG(ERROR) << "cudaEventDestroy(NCCL operation) failed: "
                       << cudaGetErrorString(result);
        }
        if (get_result == cudaSuccess && previous_device != device) {
            (void)cudaSetDevice(previous_device);
        }
    }

    bool completed() const noexcept {
        // Captured event records can run repeatedly. Do not latch a successful
        // replay as completion of every future execution of the same graph.
        return !captured && recorded && cudaEventQuery(event) == cudaSuccess;
    }
#endif
};

NcclCollectiveExecutor::NcclCollectiveExecutor() = default;

void NcclCollectiveExecutor::retireCompletedOperations() noexcept {
#ifdef USE_NCCL_PG
    std::erase_if(pending_operations_, [](const auto& operation) {
        // Eager operations still need a deadline even if the caller discarded
        // their status. Captured records have no per-replay completion event.
        return operation->completed() ||
               (operation->captured && operation->recorded &&
                operation->status.use_count() == 1);
    });
#endif
}

uint64_t NcclCollectiveExecutor::markPendingOperationsAborted() noexcept {
    uint64_t first_failed_operation = next_operation_;
#ifdef USE_NCCL_PG
    for (auto& operation : pending_operations_) {
        if (!operation->completed()) {
            operation->status->aborted.store(true, std::memory_order_release);
            // Graph replay is not a new host submission. Its Work remains
            // abort-sensitive, but capture order is not a replay sequence.
            if (!operation->captured) {
                first_failed_operation = std::min(first_failed_operation,
                                                  operation->status->sequence);
            }
        }
    }
#endif
    return first_failed_operation;
}

NcclCollectiveExecutor::~NcclCollectiveExecutor() {
    disable("executor destruction");
    if (watchdog_.joinable()) watchdog_.join();
#ifdef USE_NCCL_PG
    void* barrier_buffer = nullptr;
    int device_index = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        barrier_buffer = barrier_buffer_;
        barrier_buffer_ = nullptr;
        device_index = device_index_;
    }
    if (barrier_buffer) {
        int previous_device = -1;
        const auto get_result = cudaGetDevice(&previous_device);
        if (device_index >= 0) (void)cudaSetDevice(device_index);
        const auto free_result = cudaFree(barrier_buffer);
        if (free_result != cudaSuccess) {
            LOG(ERROR) << "cudaFree(NCCL barrier buffer) failed: "
                       << cudaGetErrorString(free_result);
        }
        if (get_result == cudaSuccess && previous_device >= 0 &&
            previous_device != device_index) {
            (void)cudaSetDevice(previous_device);
        }
    }
#endif
}

bool NcclCollectiveExecutor::isCompiled() noexcept {
#ifdef USE_NCCL_PG
    return true;
#else
    return false;
#endif
}

PGResult<NcclCollectiveExecutor::UniqueId>
NcclCollectiveExecutor::createUniqueId() {
#ifdef USE_NCCL_PG
    static_assert(NCCL_UNIQUE_ID_BYTES == kNcclUniqueIdBytes);
    ncclUniqueId id{};
    const auto result = ncclGetUniqueId(&id);
    if (result != ncclSuccess) {
        return makePGError(makeNcclError("ncclGetUniqueId", result));
    }
    UniqueId output{};
    std::memcpy(output.data(), id.internal, output.size());
    return output;
#else
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::initialize(
    const UniqueId& unique_id, int rank, int size, int device_index,
    const std::atomic<size_t>* timeout_us) {
#ifdef USE_NCCL_PG
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!communicator_ && !watchdog_.joinable(),
                      "NCCL executor is already initialized");
    PG_VALIDATE_ARG(rank >= 0 && rank < size, "invalid NCCL rank");
    PG_VALIDATE_ARG(size > 1, "NCCL communicator requires at least two ranks");
    PG_VALIDATE_ARG(timeout_us, "NCCL collective timeout is null");
    // NCCL's environment setting overrides config.blocking. Blocking calls
    // would prevent serialized timeout/abort handling from making progress.
    const char* blocking = std::getenv("NCCL_COMM_BLOCKING");
    PG_VALIDATE_ARG(!blocking || std::strcmp(blocking, "0") == 0,
                    "Mooncake PG requires NCCL_COMM_BLOCKING unset or 0");

    GpuDeviceGuard device_guard(device_index);
    ncclUniqueId id{};
    static_assert(sizeof(id.internal) == kNcclUniqueIdBytes);
    std::memcpy(id.internal, unique_id.data(), unique_id.size());

    // Nonblocking NCCL calls let us serialize abort against API calls without
    // getting stuck inside a host-side connection/setup operation.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclComm_t candidate = nullptr;
    auto result = ncclCommInitRankConfig(&candidate, size, id, rank, &config);
    const auto init_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (result == ncclInProgress &&
           std::chrono::steady_clock::now() < init_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ncclResult_t state = ncclInProgress;
        result = ncclCommGetAsyncError(candidate, &state);
        if (result == ncclSuccess) result = state;
    }
    if (result != ncclSuccess) {
        if (candidate) (void)ncclCommAbort(candidate);
        if (result == ncclInProgress) {
            return makePGError(PGErrorCode::Timeout,
                               "NCCL initialization timed out");
        }
        return makePGError(makeNcclError("ncclCommInitRankConfig", result));
    }

    void* barrier_buffer = nullptr;
    auto cuda_result = cudaMalloc(&barrier_buffer, sizeof(int32_t));
    if (cuda_result == cudaSuccess) {
        cuda_result = cudaMemset(barrier_buffer, 0, sizeof(int32_t));
    }
    if (cuda_result != cudaSuccess) {
        (void)ncclCommAbort(candidate);
        if (barrier_buffer) (void)cudaFree(barrier_buffer);
        return makePGError(PGErrorCode::SystemError,
                           std::string("NCCL barrier allocation failed: ") +
                               cudaGetErrorString(cuda_result));
    }

    communicator_ = candidate;
    barrier_buffer_ = barrier_buffer;
    device_index_ = device_index;
    size_ = size;
    timeout_us_ = timeout_us;
    unique_id_ = unique_id;
    generation_ready_.store(true, std::memory_order_release);
    active_.store(true, std::memory_order_release);
    watchdog_ = std::thread(&NcclCollectiveExecutor::watchdogLoop, this);
    LOG(INFO) << "Mooncake PG initialized NCCL collectives for rank " << rank
              << "/" << size << " on CUDA device " << device_index;
    return {};
#else
    (void)unique_id;
    (void)rank;
    (void)size;
    (void)device_index;
    (void)timeout_us;
    return notCompiled();
#endif
}

std::optional<NcclCollectiveFailure> NcclCollectiveExecutor::failedGeneration()
    const noexcept {
    if (isActive() && failed_.load(std::memory_order_acquire)) {
        return NcclCollectiveFailure{
            .unique_id = unique_id_,
            .first_failed_operation =
                failure_state_->first_failed_operation.load(
                    std::memory_order_acquire)};
    }
    return std::nullopt;
}

void NcclCollectiveExecutor::requestGroupAbort(
    const NcclCollectiveFailure& failure) noexcept {
    if (!generation_ready_.load(std::memory_order_acquire) ||
        failure.unique_id != unique_id_)
        return;
    // Update retained Work even if this executor has already been disabled by
    // a membership change. It still belongs to this same old generation.
    failure_state_->failFrom(failure.first_failed_operation);
    group_abort_requested_.store(true, std::memory_order_release);
    progress_.notify_all();
}

bool NcclCollectiveExecutor::supports(DataType datatype) const noexcept {
#ifdef USE_NCCL_PG
    return toNcclDataType(datatype).has_value();
#else
    (void)datatype;
    return false;
#endif
}

PGResult<NcclCollectiveFailure> NcclCollectiveExecutor::quiesceForRecovery() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        PG_VALIDATE_STATE(
            isActive() &&
                (failed_.load(std::memory_order_acquire) ||
                 group_abort_requested_.load(std::memory_order_acquire)),
            "NCCL recovery requires an observed communicator failure");
        abortLocked("NCCL communicator failure reported by the coordinator");
        PG_VALIDATE_STATE(abort_succeeded_,
                          "NCCL abort did not complete successfully");
    }
    // Do not wait for launch_mutex_ before abort: an ncclInProgress submission
    // holds it while waiting for the watchdog/abort to unblock it.
    std::lock_guard<std::mutex> submission(launch_mutex_);
    const auto failure = failedGeneration();
    PG_VALIDATE_STATE(failure.has_value(),
                      "NCCL backend changed during recovery");
    return *failure;
}

bool NcclCollectiveExecutor::supportsReduction(DataType datatype,
                                               ReduceOp op) const noexcept {
#ifdef USE_NCCL_PG
    return toNcclDataType(datatype).has_value() &&
           toNcclReduceOp(op).has_value();
#else
    (void)datatype;
    (void)op;
    return false;
#endif
}

template <typename Function>
PGResult<void> NcclCollectiveExecutor::launch(
    const char* operation, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status, Function&& function) {
#ifdef USE_NCCL_PG
    std::lock_guard<std::mutex> submission(launch_mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    if (group_abort_requested_.load(std::memory_order_acquire)) {
        abortLocked("NCCL communicator failure reported by the coordinator");
    }
    if (failure_reason_) return makePGError(failure_code_, failure_reason_);
    PG_VALIDATE_STATE(active_.load(std::memory_order_acquire) && communicator_,
                      "NCCL collective executor is inactive");
    const GpuDeviceGuard device_guard(device_index_);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    auto cuda_result = cudaStreamIsCapturing(stream, &capture_status);
    if (cuda_result != cudaSuccess) {
        return makePGError(PGErrorCode::SystemError,
                           cudaGetErrorString(cuda_result));
    }
    if (capture_status == cudaStreamCaptureStatusNone) {
        retireCompletedOperations();
    }
    auto pending = std::make_unique<PendingOperation>();
    pending->device = device_index_;
    pending->captured = capture_status != cudaStreamCaptureStatusNone;
    if (!pending->captured) {
        cuda_result =
            cudaEventCreateWithFlags(&pending->event, cudaEventDisableTiming);
        if (cuda_result != cudaSuccess) {
            return makePGError(PGErrorCode::SystemError,
                               cudaGetErrorString(cuda_result));
        }
    }
    auto* tracked = pending.get();
    tracked->status->failure_state = failure_state_;
    tracked->status->sequence = next_operation_++;
    // Track even legacy C API calls without a returned status: discarding a
    // handle must not remove the deadline for an outstanding GPU operation.
    pending_operations_.push_back(std::move(pending));
    if (status) *status = tracked->status;
    progress_.notify_all();
    auto result = static_cast<ncclResult_t>(function(communicator_));
    while (result == ncclInProgress) {
        // No CUDA event may be recorded until NCCL finishes enqueueing. Drop
        // mutex_ while waiting so the watchdog or a view update can abort.
        progress_.wait_for(lock, std::chrono::microseconds(100));
        if (!communicator_) {
            return makePGError(failure_code_, failure_reason_);
        }
        ncclResult_t state = ncclInProgress;
        result = ncclCommGetAsyncError(static_cast<ncclComm_t>(communicator_),
                                       &state);
        if (result == ncclSuccess) result = state;
    }
    if (result != ncclSuccess) {
        abortLocked("NCCL enqueue failed");
        return makePGError(makeNcclError(operation, result));
    }
    if (!tracked->captured) {
        const auto event_result = cudaEventRecord(tracked->event, stream);
        if (event_result != cudaSuccess) {
            abortLocked("NCCL completion event recording failed");
            return makePGError(PGErrorCode::SystemError,
                               cudaGetErrorString(event_result));
        }
    }
    tracked->recorded = true;
    return {};
#else
    (void)operation;
    (void)stream;
    (void)status;
    (void)function;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::broadcast(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    int root, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclBroadcast", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclBroadcast(send_buffer, recv_buffer, count, *type, root,
                          static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)root;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allReduce(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclAllReduce", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclAllReduce(send_buffer, recv_buffer, count, *type, *reduction,
                          static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)op;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allGather(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    cudaStream_t stream, std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclAllGather", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclAllGather(send_buffer, recv_buffer, count, *type,
                          static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::reduceScatter(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclReduceScatter", stream, status, [=](void* opaque) {
        return static_cast<int>(ncclReduceScatter(
            send_buffer, recv_buffer, count, *type, *reduction,
            static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)op;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allToAll(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    cudaStream_t stream, std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");

    const void* nccl_send_buffer = send_buffer;
    void* staging = nullptr;
    if (send_buffer == recv_buffer && count != 0) {
        const size_t element_size = elementSize(datatype);
        PG_VALIDATE_ARG(
            size_ > 0 && count <= std::numeric_limits<size_t>::max() /
                                      static_cast<size_t>(size_),
            "all-to-all element count overflows size_t");
        const size_t total_count = count * static_cast<size_t>(size_);
        PG_VALIDATE_ARG(
            total_count <= std::numeric_limits<size_t>::max() / element_size,
            "all-to-all byte count overflows size_t");
        const size_t bytes = total_count * element_size;
        auto cuda_result = cudaMallocAsync(&staging, bytes, stream);
        if (cuda_result == cudaSuccess) {
            cuda_result = cudaMemcpyAsync(staging, send_buffer, bytes,
                                          cudaMemcpyDeviceToDevice, stream);
        }
        if (cuda_result != cudaSuccess) {
            if (staging) (void)cudaFreeAsync(staging, stream);
            return makePGError(PGErrorCode::SystemError,
                               std::string("NCCL all-to-all staging failed: ") +
                                   cudaGetErrorString(cuda_result));
        }
        nccl_send_buffer = staging;
    }

    auto result = launch("ncclAlltoAll", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclAlltoAll(nccl_send_buffer, recv_buffer, count, *type,
                         static_cast<ncclComm_t>(opaque), stream));
    });
    if (staging) {
        const auto cuda_result = cudaFreeAsync(staging, stream);
        if (result.has_value() && cuda_result != cudaSuccess) {
            return makePGError(
                PGErrorCode::SystemError,
                std::string("NCCL all-to-all staging free failed: ") +
                    cudaGetErrorString(cuda_result));
        }
    }
    return result;
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::reduce(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, int root, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclReduce", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclReduce(send_buffer, recv_buffer, count, *type, *reduction, root,
                       static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)op;
    (void)root;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::gather(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    int root, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclGather", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclGather(send_buffer, recv_buffer, count, *type, root,
                       static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)root;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::scatter(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    int root, cudaStream_t stream,
    std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclScatter", stream, status, [=](void* opaque) {
        return static_cast<int>(
            ncclScatter(send_buffer, recv_buffer, count, *type, root,
                        static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)send_buffer;
    (void)recv_buffer;
    (void)count;
    (void)datatype;
    (void)root;
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::barrier(
    cudaStream_t stream, std::shared_ptr<GpuCollectiveStatus>* status) {
#ifdef USE_NCCL_PG
    return launch("NCCL barrier", stream, status, [=, this](void* opaque) {
        return static_cast<int>(
            ncclAllReduce(barrier_buffer_, barrier_buffer_, 1, ncclInt32,
                          ncclSum, static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)stream;
    (void)status;
    return notCompiled();
#endif
}

void NcclCollectiveExecutor::abortLocked(const char* reason, PGErrorCode code,
                                         bool report_failure) noexcept {
#ifdef USE_NCCL_PG
    if (communicator_) {
        failure_reason_ = reason;
        failure_code_ = code;
        // Snapshot completion before abort. Once abort releases GPU work, its
        // events alone cannot distinguish success from cancellation.
        const auto first_failed_operation = markPendingOperationsAborted();
        // Publish before teardown, which may wait on captured graph references.
        // Shutdown and membership-driven disable are not failure observations.
        if (report_failure) {
            failure_state_->failFrom(first_failed_operation);
            failed_.store(true, std::memory_order_release);
        }
        auto comm = static_cast<ncclComm_t>(communicator_);
        communicator_ = nullptr;
        const auto result = ncclCommAbort(comm);
        abort_succeeded_ = result == ncclSuccess;
        if (result != ncclSuccess) {
            LOG(ERROR) << "ncclCommAbort failed: "
                       << ncclGetErrorString(result);
        }
        LOG(INFO) << "Mooncake PG aborted NCCL collectives: " << reason;
    }
    pending_operations_.clear();
    progress_.notify_all();
    // Do not publish TE eligibility after a local timeout/error. Only a
    // coordinated view change or recovery commit may enable TE via disable().
#else
    (void)reason;
    (void)code;
    (void)report_failure;
#endif
}

void NcclCollectiveExecutor::watchdogLoop() noexcept {
#ifdef USE_NCCL_PG
    try {
        const GpuDeviceGuard device_guard(device_index_);
        std::unique_lock<std::mutex> lock(mutex_);
        while (communicator_) {
            progress_.wait_for(lock, std::chrono::milliseconds(1));
            if (!communicator_) break;
            if (group_abort_requested_.load(std::memory_order_acquire)) {
                abortLocked(
                    "NCCL communicator failure reported by the coordinator");
                break;
            }
            ncclResult_t state = ncclSuccess;
            const auto result = ncclCommGetAsyncError(
                static_cast<ncclComm_t>(communicator_), &state);
            if (result != ncclSuccess ||
                (state != ncclSuccess && state != ncclInProgress)) {
                LOG(ERROR) << "NCCL asynchronous error: "
                           << ncclGetErrorString(result != ncclSuccess ? result
                                                                       : state);
                abortLocked("NCCL asynchronous error");
                break;
            }

            const auto now = std::chrono::steady_clock::now();
            const size_t timeout_us =
                timeout_us_->load(std::memory_order_relaxed);
            const char* failure = nullptr;
            PGErrorCode code = PGErrorCode::SystemError;
            for (const auto& operation : pending_operations_) {
                // Capture is not execution. Only host enqueue is timed for
                // captured work; per-replay deadlines require graph
                // integration.
                if (operation->captured && operation->recorded) continue;
                if (operation->recorded) {
                    const auto event_result = cudaEventQuery(operation->event);
                    if (event_result == cudaSuccess) continue;
                    if (event_result != cudaErrorNotReady) {
                        failure = "NCCL completion event query failed";
                        break;
                    }
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        now - operation->started)
                        .count();
                if (static_cast<uint64_t>(elapsed) > timeout_us) {
                    failure = "NCCL collective timed out";
                    code = PGErrorCode::Timeout;
                    break;
                }
            }
            if (failure) {
                abortLocked(failure, code);
                break;
            }
            retireCompletedOperations();
        }
    } catch (const std::exception& error) {
        LOG(ERROR) << "NCCL watchdog failed: " << error.what();
        std::lock_guard<std::mutex> lock(mutex_);
        abortLocked("NCCL watchdog failed");
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        abortLocked("NCCL watchdog failed");
    }
#endif
}

void NcclCollectiveExecutor::disable(const char* reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLocked(reason ? reason : "NCCL executor disabled",
                PGErrorCode::SystemError, /*report_failure=*/false);
    // Publish TE eligibility only after abort completes while holding the same
    // lock used by launch(). This prevents a concurrent local operation from
    // routing to TE before this rank has aborted its old NCCL communicator.
    active_.store(false, std::memory_order_release);
}

}  // namespace mooncake
