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

#ifndef MOONCAKE_PG_NCCL_COLLECTIVE_EXECUTOR_H
#define MOONCAKE_PG_NCCL_COLLECTIVE_EXECUTOR_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "comm_types.h"
#include "control_plane/control_types.h"
#include "error_types.h"
#include "gpu_runtime.h"

namespace mooncake {

// Enqueues NCCL collectives on the caller's stream while Mooncake retains
// ownership of group formation, membership, failure detection, and P2P.
class NcclCollectiveExecutor {
   public:
    using UniqueId = std::array<uint8_t, kNcclUniqueIdBytes>;

    NcclCollectiveExecutor() = default;
    ~NcclCollectiveExecutor();

    NcclCollectiveExecutor(const NcclCollectiveExecutor&) = delete;
    NcclCollectiveExecutor& operator=(const NcclCollectiveExecutor&) = delete;

    static bool isCompiled() noexcept;
    static PGResult<UniqueId> createUniqueId();

    PGResult<void> initialize(const UniqueId& unique_id, int rank, int size,
                              int device_index);
    bool isActive() const noexcept {
        return active_.load(std::memory_order_acquire);
    }
    bool supports(DataType datatype) const noexcept;
    bool supportsReduction(DataType datatype, ReduceOp op) const noexcept;

    PGResult<void> broadcast(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype, int root,
                             cudaStream_t stream);
    PGResult<void> allReduce(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype, ReduceOp op,
                             cudaStream_t stream);
    PGResult<void> allGather(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype,
                             cudaStream_t stream);
    PGResult<void> reduceScatter(const void* send_buffer, void* recv_buffer,
                                 size_t count, DataType datatype, ReduceOp op,
                                 cudaStream_t stream);
    PGResult<void> allToAll(const void* send_buffer, void* recv_buffer,
                            size_t count, DataType datatype,
                            cudaStream_t stream);
    PGResult<void> reduce(const void* send_buffer, void* recv_buffer,
                          size_t count, DataType datatype, ReduceOp op,
                          int root, cudaStream_t stream);
    PGResult<void> gather(const void* send_buffer, void* recv_buffer,
                          size_t count, DataType datatype, int root,
                          cudaStream_t stream);
    PGResult<void> scatter(const void* send_buffer, void* recv_buffer,
                           size_t count, DataType datatype, int root,
                           cudaStream_t stream);
    PGResult<void> barrier(cudaStream_t stream);

    // NCCL communicators have fixed membership. Abort this communicator before
    // Mooncake applies a different active-rank view; later operations then use
    // the existing Transfer Engine implementation.
    void disable(const char* reason) noexcept;

   private:
    template <typename Function>
    PGResult<void> launch(const char* operation, Function&& function);

    mutable std::mutex mutex_;
    std::atomic<bool> active_{false};
    void* communicator_ = nullptr;
    void* barrier_buffer_ = nullptr;
    int device_index_ = -1;
    int size_ = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_NCCL_COLLECTIVE_EXECUTOR_H
