// Copyright 2026 KVCache.AI
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef MOONCAKE_EP_ENABLE_NCCL_JIT

#include <cstdint>

#include <cuda_runtime.h>

#include <elastic/mooncake_ep_elastic_launch.cuh>

namespace mooncake::elastic::jit {

struct DispatchSpec {
    int hidden_bytes = 0;
    int num_sf_packs = 0;
    int num_max_tokens_per_rank = 0;
    int num_experts = 0;
    int num_topk = 0;
    int num_sms = 0;
    int num_notify_warps = 0;
    int num_threads = 0;
    int smem_bytes = 0;
    int num_scaleout_ranks = 1;
    int num_scaleup_ranks = 1;
    bool reuse_slot_indices = false;
};

struct CombineSpec {
    int hidden = 0;
    int num_max_tokens_per_rank = 0;
    int num_experts = 0;
    int num_topk = 0;
    int num_sms = 0;
    int num_threads = 0;
    int smem_bytes = 0;
    int num_scaleout_ranks = 1;
    int num_scaleup_ranks = 1;
};

// This experimental switch is intentionally process-wide so AOT and JIT can
// be compared without changing the public ElasticBuffer API. It is sampled
// when an ElasticBuffer is constructed.
bool requested_by_environment();

void launch_dispatch(const DispatchSpec& spec, void* x, void* sf,
                     int64_t* topk_idx, float* topk_weights,
                     int64_t* copied_topk_idx,
                     int* cumulative_local_expert_recv_stats,
                     int* psum_num_recv_tokens_per_scaleup_rank,
                     int* psum_num_recv_tokens_per_expert,
                     int* dst_buffer_slot_idx, int* token_metadata_at_forward,
                     int num_tokens, int sf_token_stride, int sf_hidden_stride,
                     const transport::NcclContext& comm_ctx, void* buffer,
                     void* workspace, void* mapped_host_workspace,
                     int scaleout_rank_idx, int scaleup_rank_idx,
                     cudaStream_t stream);

void launch_combine(const CombineSpec& spec, void* x, float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward, int* channel_linked_list,
                    int num_reduced_tokens,
                    const transport::NcclContext& comm_ctx, void* buffer,
                    void* workspace, int scaleout_rank_idx,
                    int scaleup_rank_idx, cudaStream_t stream);

}  // namespace mooncake::elastic::jit

#endif  // MOONCAKE_EP_ENABLE_NCCL_JIT
