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

// Test-only interposition. No fault injection is compiled into Mooncake PG.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <nccl.h>

namespace {
// 0: off, 1: async error, 2: enqueue stall, 3: mask native async errors
std::atomic<int> fault_mode{0};
std::atomic<ncclComm_t> target{nullptr};
std::atomic<uint64_t> fault_count{0};
std::atomic<uint64_t> abort_count{0};

template <typename Function>
Function findReal(const char* name) {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        static void* library = dlopen("libnccl.so.2", RTLD_LAZY | RTLD_NOLOAD);
        if (library) symbol = dlsym(library, name);
    }
    if (!symbol) std::abort();
    return reinterpret_cast<Function>(symbol);
}
}  // namespace

extern "C" void mooncakePgTestNcclSetFault(int mode) {
    fault_mode.store(mode, std::memory_order_release);
}

extern "C" uint64_t mooncakePgTestNcclFaultCount() {
    return fault_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t mooncakePgTestNcclAbortCount() {
    return abort_count.load(std::memory_order_acquire);
}

extern "C" ncclResult_t ncclCommAbort(ncclComm_t comm) {
    static auto real = findReal<decltype(&ncclCommAbort)>("ncclCommAbort");
    if (comm == target.load(std::memory_order_acquire)) {
        abort_count.fetch_add(1, std::memory_order_release);
    }
    return real(comm);
}

extern "C" ncclResult_t ncclAllReduce(const void* send, void* recv,
                                      size_t count, ncclDataType_t type,
                                      ncclRedOp_t op, ncclComm_t comm,
                                      cudaStream_t stream) {
    static auto real = findReal<decltype(&ncclAllReduce)>("ncclAllReduce");
    target.store(comm, std::memory_order_release);
    if (fault_mode.load(std::memory_order_acquire) == 2) {
        // Do not enqueue any CUDA work. Incorrectly recording a completion
        // event before NCCL leaves ncclInProgress would report success here.
        return ncclInProgress;
    }
    return real(send, recv, count, type, op, comm, stream);
}

extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,
                                              ncclResult_t* state) {
    static auto real =
        findReal<decltype(&ncclCommGetAsyncError)>("ncclCommGetAsyncError");
    const int mode = fault_mode.load(std::memory_order_acquire);
    if (mode && comm == target.load(std::memory_order_acquire)) {
        fault_count.fetch_add(1, std::memory_order_relaxed);
        if (mode == 3) {
            // Mask errors, not host-enqueue progress. Reporting success while
            // NCCL is still enqueueing would create a false completion event.
            const auto result = real(comm, state);
            if (result != ncclSuccess ||
                (*state != ncclSuccess && *state != ncclInProgress)) {
                *state = ncclSuccess;
            }
            return ncclSuccess;
        }
        *state = mode == 1 ? ncclRemoteError : ncclInProgress;
        return ncclSuccess;
    }
    return real(comm, state);
}
