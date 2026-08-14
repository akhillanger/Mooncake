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

NcclCollectiveExecutor::~NcclCollectiveExecutor() {
    disable("executor destruction");
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

PGResult<void> NcclCollectiveExecutor::initialize(const UniqueId& unique_id,
                                                  int rank, int size,
                                                  int device_index) {
#ifdef USE_NCCL_PG
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(!communicator_, "NCCL executor is already initialized");
    PG_VALIDATE_ARG(rank >= 0 && rank < size, "invalid NCCL rank");
    PG_VALIDATE_ARG(size > 1, "NCCL communicator requires at least two ranks");

    GpuDeviceGuard device_guard(device_index);
    ncclUniqueId id{};
    static_assert(sizeof(id.internal) == kNcclUniqueIdBytes);
    std::memcpy(id.internal, unique_id.data(), unique_id.size());

    // A blocking communicator makes every host API return only after its GPU
    // work has been enqueued. Mooncake can then record its completion event
    // immediately without polling ncclCommGetAsyncError on every collective.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    ncclComm_t candidate = nullptr;
    const auto result =
        ncclCommInitRankConfig(&candidate, size, id, rank, &config);
    if (result != ncclSuccess) {
        if (candidate) (void)ncclCommAbort(candidate);
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
    active_.store(true, std::memory_order_release);
    LOG(INFO) << "Mooncake PG initialized NCCL collectives for rank " << rank
              << "/" << size << " on CUDA device " << device_index;
    return {};
#else
    (void)unique_id;
    (void)rank;
    (void)size;
    (void)device_index;
    return notCompiled();
#endif
}

bool NcclCollectiveExecutor::supports(DataType datatype) const noexcept {
#ifdef USE_NCCL_PG
    return toNcclDataType(datatype).has_value();
#else
    (void)datatype;
    return false;
#endif
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
PGResult<void> NcclCollectiveExecutor::launch(const char* operation,
                                              Function&& function) {
#ifdef USE_NCCL_PG
    std::lock_guard<std::mutex> lock(mutex_);
    PG_VALIDATE_STATE(active_.load(std::memory_order_acquire) && communicator_,
                      "NCCL collective executor is inactive");
    auto comm = static_cast<ncclComm_t>(communicator_);
    const auto result = static_cast<ncclResult_t>(function(communicator_));
    if (result != ncclSuccess) {
        // Keep routing this group to the failed NCCL executor. Falling back to
        // TE on only one rank would mismatch the other ranks and can hang. A
        // coordinated membership view change calls disable() and is the only
        // transition that permits TE fallback.
        communicator_ = nullptr;
        (void)ncclCommAbort(comm);
        return makePGError(makeNcclError(operation, result));
    }
    return {};
#else
    (void)operation;
    (void)function;
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::broadcast(const void* send_buffer,
                                                 void* recv_buffer,
                                                 size_t count,
                                                 DataType datatype, int root,
                                                 cudaStream_t stream) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclBroadcast", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allReduce(const void* send_buffer,
                                                 void* recv_buffer,
                                                 size_t count,
                                                 DataType datatype, ReduceOp op,
                                                 cudaStream_t stream) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclAllReduce", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allGather(const void* send_buffer,
                                                 void* recv_buffer,
                                                 size_t count,
                                                 DataType datatype,
                                                 cudaStream_t stream) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclAllGather", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::reduceScatter(
    const void* send_buffer, void* recv_buffer, size_t count, DataType datatype,
    ReduceOp op, cudaStream_t stream) {
#ifdef USE_NCCL_PG
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclReduceScatter", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::allToAll(const void* send_buffer,
                                                void* recv_buffer, size_t count,
                                                DataType datatype,
                                                cudaStream_t stream) {
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

    auto result = launch("ncclAlltoAll", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::reduce(const void* send_buffer,
                                              void* recv_buffer, size_t count,
                                              DataType datatype, ReduceOp op,
                                              int root, cudaStream_t stream) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    const auto reduction = toNcclReduceOp(op);
    PG_VALIDATE_ARG(type.has_value() && reduction.has_value(),
                    "reduction is unsupported by NCCL");
    return launch("ncclReduce", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::gather(const void* send_buffer,
                                              void* recv_buffer, size_t count,
                                              DataType datatype, int root,
                                              cudaStream_t stream) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclGather", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::scatter(const void* send_buffer,
                                               void* recv_buffer, size_t count,
                                               DataType datatype, int root,
                                               cudaStream_t stream) {
#ifdef USE_NCCL_PG
    PG_VALIDATE_ARG(root >= 0 && root < size_,
                    "root is outside the NCCL communicator");
    const auto type = toNcclDataType(datatype);
    PG_VALIDATE_ARG(type.has_value(), "datatype is unsupported by NCCL");
    return launch("ncclScatter", [=](void* opaque) {
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
    return notCompiled();
#endif
}

PGResult<void> NcclCollectiveExecutor::barrier(cudaStream_t stream) {
#ifdef USE_NCCL_PG
    return launch("NCCL barrier", [=, this](void* opaque) {
        return static_cast<int>(
            ncclAllReduce(barrier_buffer_, barrier_buffer_, 1, ncclInt32,
                          ncclSum, static_cast<ncclComm_t>(opaque), stream));
    });
#else
    (void)stream;
    return notCompiled();
#endif
}

void NcclCollectiveExecutor::disable(const char* reason) noexcept {
#ifdef USE_NCCL_PG
    std::lock_guard<std::mutex> lock(mutex_);
    if (communicator_) {
        auto comm = static_cast<ncclComm_t>(communicator_);
        communicator_ = nullptr;
        const auto result = ncclCommAbort(comm);
        if (result != ncclSuccess) {
            LOG(ERROR) << "ncclCommAbort failed: "
                       << ncclGetErrorString(result);
        }
        LOG(INFO) << "Mooncake PG disabled NCCL collectives: "
                  << (reason ? reason : "unspecified reason");
    }
    // Publish TE eligibility only after abort completes while holding the same
    // lock used by launch(). This prevents a concurrent operation from routing
    // to TE while another rank can still enqueue on the old NCCL communicator.
    active_.store(false, std::memory_order_release);
#else
    (void)reason;
#endif
}

}  // namespace mooncake
