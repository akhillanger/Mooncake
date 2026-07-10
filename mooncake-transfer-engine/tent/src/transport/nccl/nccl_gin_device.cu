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

#include "tent/transport/nccl/paged_gin.h"

namespace {

struct ResolvedPagedJob {
    size_t page_bytes;
    size_t src_page_stride;
    size_t dst_page_stride;
    ncclWindow_t dst_window;
    ncclWindow_t src_window;
    int layer_count;
};

__device__ __forceinline__ bool resolvePagedJob(
    const TentNcclPagedKvLayout& layout,
    const TentNcclPagedTransferJob& job, ncclWindow_t default_dst_window,
    ncclWindow_t default_src_window, ResolvedPagedJob& resolved) {
    const size_t default_page_bytes =
        layout.page_bytes ? layout.page_bytes : layout.page_stride_bytes;
    const size_t default_src_page_stride =
        layout.src_page_stride_bytes ? layout.src_page_stride_bytes
                                     : layout.page_stride_bytes;
    const size_t default_dst_page_stride =
        layout.dst_page_stride_bytes ? layout.dst_page_stride_bytes
                                     : layout.page_stride_bytes;
    resolved.page_bytes =
        job.page_bytes
            ? job.page_bytes
            : (job.page_stride_bytes ? job.page_stride_bytes
                                     : default_page_bytes);
    resolved.src_page_stride =
        job.src_page_stride_bytes
            ? job.src_page_stride_bytes
            : (job.page_stride_bytes ? job.page_stride_bytes
                                     : default_src_page_stride);
    resolved.dst_page_stride =
        job.dst_page_stride_bytes
            ? job.dst_page_stride_bytes
            : (job.page_stride_bytes ? job.page_stride_bytes
                                     : default_dst_page_stride);
    resolved.dst_window =
        job.dst_window ? job.dst_window : default_dst_window;
    resolved.src_window =
        job.src_window ? job.src_window : default_src_window;
    resolved.layer_count = job.layer_end - job.layer_begin;
    return job.num_pages > 0 && resolved.layer_count > 0 &&
           job.layer_begin >= 0 && job.src_page_table && job.dst_page_table &&
           resolved.page_bytes != 0 && resolved.src_page_stride != 0 &&
           resolved.dst_page_stride != 0 &&
           resolved.src_page_stride >= resolved.page_bytes &&
           resolved.dst_page_stride >= resolved.page_bytes &&
           resolved.dst_window && resolved.src_window;
}

__device__ __forceinline__ void issuePagedPut(
    ncclGin& gin, ncclTeam world, int peer,
    const TentNcclPagedTransferJob& job, const ResolvedPagedJob& resolved,
    int page_slot, int layer, const ncclCoopThread& thread_coop,
    uint32_t gin_opt_flags) {
    const int src_page = job.src_page_table[page_slot];
    const int dst_page = job.dst_page_table[page_slot];
    if (src_page < 0 || dst_page < 0) return;

    const size_t src_layer_offset =
        job.src_layer_offsets
            ? job.src_layer_offsets[layer - job.layer_begin]
            : static_cast<size_t>(layer) * job.src_layer_stride;
    const size_t dst_layer_offset =
        job.dst_layer_offsets
            ? job.dst_layer_offsets[layer - job.layer_begin]
            : static_cast<size_t>(layer) * job.dst_layer_stride;
    const size_t src_offset =
        job.src_base_offset + src_layer_offset +
        static_cast<size_t>(src_page) * resolved.src_page_stride;
    const size_t dst_offset =
        job.dst_base_offset + dst_layer_offset +
        static_cast<size_t>(dst_page) * resolved.dst_page_stride;
    gin.put(world, peer, resolved.dst_window, dst_offset, resolved.src_window,
            src_offset, resolved.page_bytes, ncclGin_None{}, ncclGin_None{},
            thread_coop, ncclGin_None{}, cuda::thread_scope_thread,
            cuda::thread_scope_device, gin_opt_flags);
}

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
    const size_t bytes = end - begin;
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    if (data_signal_value) {
        if (bytes) {
            gin.put(world, peer, dst_window, dst_offset + begin, src_window,
                    src_offset + begin, bytes,
                    ncclGin_StrongSignalInc{
                        static_cast<unsigned int>(lane)},
                    ncclGin_None{}, coop);
        } else {
            gin.signal(world, peer,
                       ncclGin_StrongSignalInc{
                           static_cast<unsigned int>(lane)},
                       coop);
        }
    } else if (bytes) {
        gin.put(world, peer, dst_window, dst_offset + begin, src_window,
                src_offset + begin, bytes, ncclGin_None{}, ncclGin_None{},
                coop);
    }
    gin.flush(coop);
#endif
}

__global__ void ncclGinWaitSignalKernel(ncclDevComm comm, int lanes,
                                        int signal_base,
                                        unsigned long long signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    gin.waitSignal(coop, signal_base + lane, signal_value);
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
                                     unsigned long long data_signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    ncclTeam world = ncclTeamWorld(comm);
    ncclGin gin(comm, lane);
    const ncclCoopCta coop = ncclCoopCta();
    gin.waitSignal(coop, lane, data_signal_value);
    gin.signal(world, peer,
               ncclGin_StrongSignalInc{
                   static_cast<unsigned int>(lanes + lane)},
               coop);
#endif
}

__global__ void ncclGinPagedPutKernel(
    ncclDevComm comm, int peer, int lanes, ncclWindow_t dst_window,
    ncclWindow_t src_window, TentNcclPagedKvLayout layout,
    const TentNcclPagedTransferJob* jobs, int num_jobs,
    unsigned long long signal_value) {
#if __CUDA_ARCH__ >= 700
    const int lane = static_cast<int>(blockIdx.x);
    if (lane >= lanes) return;
    const bool layout_valid =
        layout.page_stride_bytes != 0 ||
        (layout.page_bytes != 0 && layout.src_page_stride_bytes != 0 &&
         layout.dst_page_stride_bytes != 0);
    if (!layout_valid) return;
    if (num_jobs > 0 && !jobs) return;

    ncclTeam world = ncclTeamWorld(comm);
    const ncclGinResourceSharingMode sharing =
        static_cast<ncclGinResourceSharingMode>(layout.gin_resource_sharing);
    ncclGin gin(comm, lane, sharing);
    const uint32_t gin_opt_flags = layout.gin_opt_flags;
    const ncclCoopThread thread_coop = ncclCoopThread();
    const ncclCoopCta cta_coop = ncclCoopCta();
    // Interleave lanes in the flattened rank. A lane-major rank leaves every
    // lane except lane 0 idle whenever a job has at most one CTA of pages.
    const unsigned long long thread_rank =
        static_cast<unsigned long long>(threadIdx.x) *
            static_cast<unsigned long long>(lanes) +
        static_cast<unsigned long long>(lane);
    const unsigned long long thread_count =
        static_cast<unsigned long long>(lanes) *
        static_cast<unsigned long long>(blockDim.x);
    const bool uniform_jobs =
        num_jobs > 1 && layout.uniform_job_num_pages > 0 &&
        layout.uniform_job_num_layers > 0;
    if (uniform_jobs) {
        const unsigned long long job_work =
            static_cast<unsigned long long>(layout.uniform_job_num_pages) *
            static_cast<unsigned long long>(layout.uniform_job_num_layers);
        const unsigned long long total_work =
            job_work * static_cast<unsigned long long>(num_jobs);
        for (unsigned long long work = thread_rank; work < total_work;
             work += thread_count) {
            const int job_index = static_cast<int>(work / job_work);
            const unsigned long long job_local_work = work % job_work;
            const TentNcclPagedTransferJob job = jobs[job_index];
            ResolvedPagedJob resolved;
            if (!resolvePagedJob(layout, job, dst_window, src_window,
                                 resolved) ||
                job.num_pages != layout.uniform_job_num_pages ||
                resolved.layer_count != layout.uniform_job_num_layers) {
                continue;
            }
            const int page_slot = static_cast<int>(
                job_local_work %
                static_cast<unsigned long long>(job.num_pages));
            const int layer =
                job.layer_begin +
                static_cast<int>(
                    job_local_work /
                    static_cast<unsigned long long>(job.num_pages));
            issuePagedPut(gin, world, peer, job, resolved, page_slot, layer,
                          thread_coop, gin_opt_flags);
        }
    } else {
        unsigned long long work_base_mod = 0;
        for (int job_index = 0; job_index < num_jobs; ++job_index) {
            const TentNcclPagedTransferJob job = jobs[job_index];
            ResolvedPagedJob resolved;
            if (!resolvePagedJob(layout, job, dst_window, src_window,
                                 resolved)) {
                continue;
            }

            const unsigned long long job_work =
                static_cast<unsigned long long>(job.num_pages) *
                static_cast<unsigned long long>(resolved.layer_count);
            // Treat all jobs as one flattened work list. Besides balancing
            // small jobs across lanes, this rotates one-page jobs across GIN
            // contexts.
            const unsigned long long first_work =
                (thread_rank + thread_count - work_base_mod) % thread_count;
            for (unsigned long long work = first_work; work < job_work;
                 work += thread_count) {
                const int page_slot = static_cast<int>(
                    work % static_cast<unsigned long long>(job.num_pages));
                const int layer =
                    job.layer_begin +
                    static_cast<int>(
                        work /
                        static_cast<unsigned long long>(job.num_pages));
                issuePagedPut(gin, world, peer, job, resolved, page_slot,
                              layer, thread_coop, gin_opt_flags);
            }
            work_base_mod =
                (work_base_mod + (job_work % thread_count)) % thread_count;
        }
    }

    // One CTA owns each GIN context. The CTA-wide flush orders all thread-scope
    // puts posted by that lane before the lane-local completion signal.
    gin.flush(cta_coop);
    if (signal_value) {
        gin.signal(world, peer,
                       ncclGin_StrongSignalInc{
                           static_cast<unsigned int>(lane)},
                       cta_coop);
    }
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
    ncclDevComm_t dev_comm, int lanes, int signal_base,
    unsigned long long signal_value, cudaStream_t stream) {
    if (lanes <= 0) return cudaErrorInvalidValue;
    ncclGinWaitSignalKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, lanes, signal_base, signal_value);
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
        dev_comm, peer, lanes, signal_value);
    return cudaGetLastError();
}

extern "C" cudaError_t tentNcclGinLaunchPagedPut(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t dst_window,
    ncclWindow_t src_window, TentNcclPagedKvLayout layout,
    const TentNcclPagedTransferJob* jobs, int num_jobs,
    unsigned long long signal_value, cudaStream_t stream) {
    const bool layout_valid =
        layout.page_stride_bytes != 0 ||
        (layout.page_bytes != 0 && layout.src_page_stride_bytes != 0 &&
         layout.dst_page_stride_bytes != 0);
    if (lanes <= 0 || num_jobs < 0 || !layout_valid ||
        (num_jobs > 0 && !jobs)) {
        return cudaErrorInvalidValue;
    }
    // Threads issue puts independently onto one CTA-shared GDAKI context.
    // Exclusive thread resource mode requires a distinct context per issuer.
    if (layout.gin_resource_sharing == NCCL_GIN_RESOURCE_SHARING_THREAD) {
        return cudaErrorNotSupported;
    }
    // Skipping credit checks can overrun the send queue for an unbounded page
    // list, while aggregation requires an explicit final doorbell per context.
    // Neither optimization is safe for this kernel yet.
    if (layout.gin_opt_flags != ncclGinOptFlagsDefault) {
        return cudaErrorNotSupported;
    }

    ncclGinPagedPutKernel<<<lanes, 128, 0, stream>>>(
        dev_comm, peer, lanes, dst_window, src_window, layout, jobs, num_jobs,
        signal_value);
    return cudaGetLastError();
}
