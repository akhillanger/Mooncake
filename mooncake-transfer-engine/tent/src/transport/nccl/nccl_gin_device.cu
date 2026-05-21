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

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

namespace {

__global__ void ncclGinPutKernel(ncclDevComm comm, int peer,
                                 int context_index,
                                 ncclWindow_t dst_window,
                                 size_t dst_offset,
                                 ncclWindow_t src_window,
                                 size_t src_offset, size_t bytes,
                                 unsigned long long data_signal_value,
                                 unsigned long long ack_signal_value) {
#if __CUDA_ARCH__ >= 700
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, context_index);
    const ncclCoopCta coop = ncclCoopCta();
    gin.put(world, peer, dst_window, dst_offset, src_window, src_offset, bytes,
            ncclGin_StrongSignalInc{0}, ncclGin_None{}, coop);
    gin.waitSignal(coop, 1, ack_signal_value);
#endif
}

__global__ void ncclGinGetKernel(ncclDevComm comm, int peer,
                                 int context_index,
                                 ncclWindow_t remote_window,
                                 size_t remote_offset,
                                 ncclWindow_t local_window,
                                 size_t local_offset, size_t bytes) {
#if __CUDA_ARCH__ >= 700
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, context_index);
    const ncclCoopCta coop = ncclCoopCta();
    gin.get(world, peer, remote_window, remote_offset, local_window,
            local_offset, bytes, coop);
    gin.flush(coop);
#endif
}

__global__ void ncclGinWaitAckKernel(ncclDevComm comm, int peer,
                                     int context_index,
                                     unsigned long long data_signal_value,
                                     unsigned long long ack_signal_value) {
#if __CUDA_ARCH__ >= 700
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, context_index);
    const ncclCoopCta coop = ncclCoopCta();
    gin.waitSignal(coop, 0, data_signal_value);
    gin.signal(world, peer, ncclGin_StrongSignalInc{1}, coop);
#endif
}

}  // namespace

extern "C" cudaError_t tentNcclGinLaunchPut(ncclDevComm_t dev_comm, int peer,
                                             int context_index,
                                             ncclWindow_t dst_window,
                                             size_t dst_offset,
                                             ncclWindow_t src_window,
                                             size_t src_offset, size_t bytes,
                                             unsigned long long signal_value,
                                             cudaStream_t stream) {
    ncclGinPutKernel<<<1, 128, 0, stream>>>(
        dev_comm, peer, context_index, dst_window, dst_offset, src_window,
        src_offset, bytes, signal_value, signal_value);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchGet(ncclDevComm_t dev_comm, int peer,
                                             int context_index,
                                             ncclWindow_t remote_window,
                                             size_t remote_offset,
                                             ncclWindow_t local_window,
                                             size_t local_offset, size_t bytes,
                                             cudaStream_t stream) {
    ncclGinGetKernel<<<1, 128, 0, stream>>>(
        dev_comm, peer, context_index, remote_window, remote_offset,
        local_window, local_offset, bytes);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchWaitAck(ncclDevComm_t dev_comm,
                                                 int peer,
                                                 int context_index,
                                                 unsigned long long signal_value,
                                                 cudaStream_t stream) {
    ncclGinWaitAckKernel<<<1, 128, 0, stream>>>(
        dev_comm, peer, context_index, signal_value, signal_value);
    return cudaGetLastError();
}
