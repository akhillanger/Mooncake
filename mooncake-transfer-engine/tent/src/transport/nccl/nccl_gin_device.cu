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

__device__ size_t stripeOffset(size_t length, int lane, int lanes) {
    return (length * static_cast<size_t>(lane)) / static_cast<size_t>(lanes);
}

__global__ void ncclGinPutKernel(ncclDevComm comm, int peer, int lanes,
                                 ncclWindow_t dst_window,
                                 size_t dst_offset,
                                 ncclWindow_t src_window,
                                 size_t src_offset, size_t total_bytes,
                                 unsigned long long data_signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    const size_t begin = stripeOffset(total_bytes, lane, lanes);
    const size_t end = stripeOffset(total_bytes, lane + 1, lanes);
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    if (data_signal_value) {
        gin.put(world, peer, dst_window, dst_offset + begin, src_window,
                src_offset + begin, end - begin, ncclGin_StrongSignalInc{0},
                ncclGin_None{}, coop);
    } else {
        gin.put(world, peer, dst_window, dst_offset + begin, src_window,
                src_offset + begin, end - begin, ncclGin_None{},
                ncclGin_None{}, coop);
    }
    gin.flush(coop);
#endif
}

__global__ void ncclGinWaitSignalKernel(ncclDevComm comm, int lanes,
                                        int signal_index,
                                        unsigned long long signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    gin.waitSignal(coop, signal_index, signal_value);
#endif
}

__global__ void ncclGinGetKernel(ncclDevComm comm, int peer, int lanes,
                                 ncclWindow_t remote_window,
                                 size_t remote_offset,
                                 ncclWindow_t local_window,
                                 size_t local_offset, size_t total_bytes) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    const size_t begin = stripeOffset(total_bytes, lane, lanes);
    const size_t end = stripeOffset(total_bytes, lane + 1, lanes);
    const size_t bytes = end - begin;
    if (bytes == 0) return;
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    gin.get(world, peer, remote_window, remote_offset + begin, local_window,
            local_offset + begin, bytes, coop);
    gin.flush(coop);
#endif
}

__global__ void ncclGinWaitAckKernel(ncclDevComm comm, int peer, int lanes,
                                     unsigned long long data_signal_value,
                                     unsigned long long ack_signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    gin.waitSignal(coop, 0, data_signal_value);
    gin.signal(world, peer, ncclGin_StrongSignalInc{1}, coop);
#endif
}

}  // namespace

extern "C" cudaError_t tentNcclGinLaunchPut(ncclDevComm_t dev_comm, int peer,
                                             int lanes,
                                             ncclWindow_t dst_window,
                                             size_t dst_offset,
                                             ncclWindow_t src_window,
                                             size_t src_offset,
                                             size_t total_bytes,
                                             unsigned long long signal_value,
                                             cudaStream_t stream) {
    if (lanes <= 0) return cudaErrorInvalidValue;
    ncclGinPutKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, peer, lanes, dst_window, dst_offset, src_window, src_offset,
        total_bytes, signal_value);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchWaitSignal(
    ncclDevComm_t dev_comm, int lanes, int signal_index,
    unsigned long long signal_value, cudaStream_t stream) {
    if (lanes <= 0) return cudaErrorInvalidValue;
    ncclGinWaitSignalKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, lanes, signal_index, signal_value);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchGet(ncclDevComm_t dev_comm, int peer,
                                             int lanes,
                                             ncclWindow_t remote_window,
                                             size_t remote_offset,
                                             ncclWindow_t local_window,
                                             size_t local_offset,
                                             size_t total_bytes,
                                             cudaStream_t stream) {
    if (lanes <= 0) return cudaErrorInvalidValue;
    ncclGinGetKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, peer, lanes, remote_window, remote_offset, local_window,
        local_offset, total_bytes);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchWaitAck(ncclDevComm_t dev_comm,
                                                 int peer, int lanes,
                                                 unsigned long long signal_value,
                                                 cudaStream_t stream) {
    if (lanes <= 0) return cudaErrorInvalidValue;
    ncclGinWaitAckKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, peer, lanes, signal_value, signal_value);
    return cudaGetLastError();
}
