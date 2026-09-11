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
#include <condition_variable>
#include <mutex>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

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

    NcclCollectiveExecutor();
    ~NcclCollectiveExecutor();

    NcclCollectiveExecutor(const NcclCollectiveExecutor&) = delete;
    NcclCollectiveExecutor& operator=(const NcclCollectiveExecutor&) = delete;

    static bool isCompiled() noexcept;
    static PGResult<UniqueId> createUniqueId();

    PGResult<void> initialize(const UniqueId& unique_id, int rank, int size,
                              int device_index,
                              const std::atomic<size_t>* timeout_us);
    bool isActive() const noexcept {
        return active_.load(std::memory_order_acquire);
    }
    std::optional<NcclCollectiveFailure> failedGeneration() const noexcept;
    // Called by the Agent, including while idle. Queue the abort for the
    // watchdog so NCCL teardown cannot block the Agent's heartbeat executor.
    void requestGroupAbort(const NcclCollectiveFailure& failure) noexcept;
    // Wait for abort completion without enabling TE. The caller must stop new
    // submissions and release captured graphs before entering recovery.
    PGResult<NcclCollectiveFailure> quiesceForRecovery();
    bool supports(DataType datatype) const noexcept;
    bool supportsReduction(DataType datatype, ReduceOp op) const noexcept;

    PGResult<void> broadcast(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype, int root,
                             cudaStream_t stream,
                             std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> allReduce(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype, ReduceOp op,
                             cudaStream_t stream,
                             std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> allGather(const void* send_buffer, void* recv_buffer,
                             size_t count, DataType datatype,
                             cudaStream_t stream,
                             std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> reduceScatter(const void* send_buffer, void* recv_buffer,
                                 size_t count, DataType datatype, ReduceOp op,
                                 cudaStream_t stream,
                                 std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> allToAll(const void* send_buffer, void* recv_buffer,
                            size_t count, DataType datatype,
                            cudaStream_t stream,
                            std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> reduce(const void* send_buffer, void* recv_buffer,
                          size_t count, DataType datatype, ReduceOp op,
                          int root, cudaStream_t stream,
                          std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> gather(const void* send_buffer, void* recv_buffer,
                          size_t count, DataType datatype, int root,
                          cudaStream_t stream,
                          std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> scatter(const void* send_buffer, void* recv_buffer,
                           size_t count, DataType datatype, int root,
                           cudaStream_t stream,
                           std::shared_ptr<GpuCollectiveStatus>* status);
    PGResult<void> barrier(cudaStream_t stream,
                           std::shared_ptr<GpuCollectiveStatus>* status);

    // NCCL communicators have fixed membership. Abort this communicator before
    // Mooncake applies a different active-rank view; later operations then use
    // the existing Transfer Engine implementation.
    void disable(const char* reason) noexcept;

   private:
    template <typename Function>
    PGResult<void> launch(const char* operation, cudaStream_t stream,
                          std::shared_ptr<GpuCollectiveStatus>* status,
                          Function&& function);

    struct PendingOperation;
    // Called with mutex_ held. Never query events belonging to captured work.
    void retireCompletedOperations() noexcept;
    uint64_t markPendingOperationsAborted() noexcept;
    void abortLocked(const char* reason,
                     PGErrorCode code = PGErrorCode::SystemError,
                     bool report_failure = true) noexcept;
    void watchdogLoop() noexcept;
    std::vector<std::unique_ptr<PendingOperation>> pending_operations_;

    // Serialize submissions across waits for nonblocking NCCL enqueue. The
    // watchdog/disable use only mutex_, so they can interrupt those waits.
    std::mutex launch_mutex_;
    mutable std::mutex mutex_;
    std::condition_variable progress_;
    std::thread watchdog_;
    const std::atomic<size_t>* timeout_us_ = nullptr;
    const char* failure_reason_ = nullptr;
    PGErrorCode failure_code_ = PGErrorCode::SystemError;
    std::atomic<bool> active_{false};
    // Written once before generation_ready_ is published; never reused.
    UniqueId unique_id_{};
    std::atomic<bool> generation_ready_{false};
    std::shared_ptr<GpuCollectiveFailureState> failure_state_ =
        std::make_shared<GpuCollectiveFailureState>();
    uint64_t next_operation_ = 0;  // protected by mutex_
    std::atomic<bool> failed_{false};
    std::atomic<bool> group_abort_requested_{false};
    void* communicator_ = nullptr;
    bool abort_succeeded_ = false;  // protected by mutex_
    void* barrier_buffer_ = nullptr;
    int device_index_ = -1;
    int size_ = 0;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_NCCL_COLLECTIVE_EXECUTOR_H
