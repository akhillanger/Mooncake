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

#ifndef TENT_TRANSPORT_NCCL_PAGED_GIN_H_
#define TENT_TRANSPORT_NCCL_PAGED_GIN_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

struct TentNcclPagedKvLayout {
    size_t page_stride_bytes = 0;
    int page_size_tokens = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int dtype_bytes = 0;
    int gin_resource_sharing = NCCL_GIN_RESOURCE_SHARING_CTA;
    uint32_t gin_opt_flags = ncclGinOptFlagsDefault;
};

struct TentNcclPagedTransferJob {
    const int32_t* src_page_table = nullptr;
    const int32_t* dst_page_table = nullptr;
    int num_pages = 0;
    int layer_begin = 0;
    int layer_end = 0;
    size_t src_layer_stride = 0;
    size_t dst_layer_stride = 0;
    size_t src_base_offset = 0;
    size_t dst_base_offset = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// If signal_value is nonzero, the paged put increments one completion
// signal per GIN lane at signal indexes [0, lanes). Use
// tentNcclGinLaunchWaitSignal(..., signal_base = 0, signal_value, ...)
// on the receiver to wait for all lane-local completions.
cudaError_t tentNcclGinLaunchPagedPut(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t dst_window,
    ncclWindow_t src_window, TentNcclPagedKvLayout layout,
    const TentNcclPagedTransferJob* jobs, int num_jobs,
    unsigned long long signal_value, cudaStream_t stream);

cudaError_t tentNcclGinLaunchWaitSignal(
    ncclDevComm_t dev_comm, int lanes, int signal_base,
    unsigned long long signal_value, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENT_TRANSPORT_NCCL_PAGED_GIN_H_
