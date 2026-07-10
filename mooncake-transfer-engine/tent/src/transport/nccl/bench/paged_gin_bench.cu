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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "tent/transport/nccl/paged_gin.h"

extern "C" cudaError_t tentNcclGinLaunchPut(
    ncclDevComm_t dev_comm, int peer, int lanes, ncclWindow_t dst_window,
    size_t dst_offset, ncclWindow_t src_window, size_t src_offset,
    size_t total_bytes, unsigned long long signal_value, cudaStream_t stream);

extern "C" cudaError_t tentNcclGinLaunchWaitAck(
    ncclDevComm_t dev_comm, int peer, int lanes,
    unsigned long long signal_value, cudaStream_t stream);

namespace {

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CHECK_NCCL(call)                                                     \
    do {                                                                     \
        ncclResult_t _res = (call);                                          \
        if (_res != ncclSuccess) {                                           \
            std::cerr << "NCCL error: " << ncclGetErrorString(_res)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct Options {
    int device0 = 0;
    int device1 = 1;
    int lanes = 4;
    int layers = 8;
    int pages = 32;
    int physical_pages = 64;
    int page_size_tokens = 16;
    int num_kv_heads = 4;
    int head_dim = 64;
    int dtype_bytes = 2;
    int warmup = 5;
    int iters = 20;
    int window_offset_bytes = 0;
    int page_padding_bytes = 0;
    int gin_resource_sharing = NCCL_GIN_RESOURCE_SHARING_CTA;
    int gin_opt_flags = ncclGinOptFlagsDefault;
    bool validate = true;
    bool with_ack = false;
    bool baseline = true;
    bool split_jobs = false;
    bool identity_map = false;
    bool check_option_guard = false;
};

struct WindowPair {
    ncclWindow_t rank0 = nullptr;
    ncclWindow_t rank1 = nullptr;
};

struct BenchState {
    Options opts;
    TentNcclPagedKvLayout layout;
    TentNcclPagedTransferJob job_host;
    ncclComm_t comm0 = nullptr;
    ncclComm_t comm1 = nullptr;
    ncclDevComm_t dev_comm0{};
    ncclDevComm_t dev_comm1{};
    WindowPair pool_window;
    WindowPair staging_window;
    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    void* src_pool = nullptr;
    void* dst_pool = nullptr;
    void* src_staging = nullptr;
    void* dst_staging = nullptr;
    size_t src_pool_window_offset = 0;
    size_t dst_pool_window_offset = 0;
    size_t src_staging_window_offset = 0;
    size_t dst_staging_window_offset = 0;
    int32_t* src_table0 = nullptr;
    int32_t* dst_table0 = nullptr;
    int32_t* src_table1 = nullptr;
    int32_t* dst_table1 = nullptr;
    TentNcclPagedTransferJob* job0 = nullptr;
    TentNcclPagedTransferJob* job1 = nullptr;
    uint64_t* errors1 = nullptr;
    size_t page_bytes = 0;
    size_t page_stride_bytes = 0;
    size_t page_words = 0;
    size_t layer_stride_bytes = 0;
    size_t pool_bytes = 0;
    size_t staging_bytes = 0;
    unsigned long long next_signal = 1;
    unsigned long long ack_signal_count = 0;
    int fused_job_count = 1;
};

bool parseIntValue(const char* arg, const char* key, int& value) {
    const size_t len = std::strlen(key);
    if (std::strncmp(arg, key, len) != 0 || arg[len] != '=') return false;
    char* end = nullptr;
    const long parsed = std::strtol(arg + len + 1, &end, 10);
    if (!end || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid integer option: " << arg << std::endl;
        std::exit(EXIT_FAILURE);
    }
    value = static_cast<int>(parsed);
    return true;
}

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --device0=N --device1=N\n"
        << "  --lanes=N\n"
        << "  --layers=N --pages=N --physical-pages=N\n"
        << "  --page-size=N --kv-heads=N --head-dim=N --dtype-bytes=N\n"
        << "  --warmup=N --iters=N\n"
        << "  --page-padding-bytes=N --window-offset-bytes=N\n"
        << "  --gin-sharing=N (0=gpu, 1=cta, 2=thread)\n"
        << "  --aggregate-requests --skip-credit-check\n"
        << "  --check-option-guard\n"
        << "  --split-jobs --identity-map --with-ack\n"
        << "  --no-validate --no-baseline\n";
}

Options parseOptions(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (std::strcmp(arg, "--no-validate") == 0) {
            opts.validate = false;
        } else if (std::strcmp(arg, "--no-baseline") == 0) {
            opts.baseline = false;
        } else if (std::strcmp(arg, "--check-option-guard") == 0) {
            opts.check_option_guard = true;
        } else if (std::strcmp(arg, "--split-jobs") == 0) {
            opts.split_jobs = true;
        } else if (std::strcmp(arg, "--with-ack") == 0) {
            opts.with_ack = true;
        } else if (std::strcmp(arg, "--identity-map") == 0) {
            opts.identity_map = true;
        } else if (std::strcmp(arg, "--aggregate-requests") == 0) {
            opts.gin_opt_flags |= ncclGinOptFlagsAggregateRequests;
        } else if (std::strcmp(arg, "--skip-credit-check") == 0) {
            opts.gin_opt_flags |= ncclGinOptFlagsMaySkipCreditCheck;
        } else if (parseIntValue(arg, "--device0", opts.device0) ||
                   parseIntValue(arg, "--device1", opts.device1) ||
                   parseIntValue(arg, "--lanes", opts.lanes) ||
                   parseIntValue(arg, "--layers", opts.layers) ||
                   parseIntValue(arg, "--pages", opts.pages) ||
                   parseIntValue(arg, "--physical-pages", opts.physical_pages) ||
                   parseIntValue(arg, "--page-size", opts.page_size_tokens) ||
                   parseIntValue(arg, "--kv-heads", opts.num_kv_heads) ||
                   parseIntValue(arg, "--head-dim", opts.head_dim) ||
                   parseIntValue(arg, "--dtype-bytes", opts.dtype_bytes) ||
                   parseIntValue(arg, "--gin-sharing",
                                 opts.gin_resource_sharing) ||
                   parseIntValue(arg, "--warmup", opts.warmup) ||
                   parseIntValue(arg, "--page-padding-bytes",
                                 opts.page_padding_bytes) ||
                   parseIntValue(arg, "--window-offset-bytes",
                                 opts.window_offset_bytes) ||
                   parseIntValue(arg, "--iters", opts.iters)) {
            continue;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opts;
}

void validateOptions(const Options& opts) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "This benchmark needs at least two CUDA devices"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opts.device0 < 0 || opts.device0 >= device_count || opts.device1 < 0 ||
        opts.device1 >= device_count || opts.device0 == opts.device1) {
        std::cerr << "Invalid device pair: " << opts.device0 << ", "
                  << opts.device1 << " with device_count=" << device_count
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opts.lanes <= 0 || opts.layers <= 0 || opts.pages <= 0 ||
        opts.physical_pages < opts.pages || opts.page_size_tokens <= 0 ||
        opts.num_kv_heads <= 0 || opts.head_dim <= 0 || opts.dtype_bytes <= 0 ||
        opts.warmup < 0 || opts.iters <= 0 ||
        opts.page_padding_bytes < 0 ||
        opts.window_offset_bytes < 0) {
        std::cerr << "Invalid non-positive benchmark option" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opts.window_offset_bytes % sizeof(uint32_t) != 0) {
        std::cerr << "--window-offset-bytes must preserve uint32 alignment"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opts.page_padding_bytes % sizeof(uint32_t) != 0) {
        std::cerr << "--page-padding-bytes must preserve uint32 alignment"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (opts.gin_resource_sharing < NCCL_GIN_RESOURCE_SHARING_GPU ||
        opts.gin_resource_sharing > NCCL_GIN_RESOURCE_SHARING_THREAD) {
        std::cerr << "Invalid GIN resource sharing mode" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__host__ __device__ uint32_t expectedWord(uint64_t layer, uint64_t page,
                                          uint64_t word_in_page) {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    x ^= (layer + 0x100000001b3ull) * 0xbf58476d1ce4e5b9ull;
    x ^= (page + 0x94d049bb133111ebull) * 0x94d049bb133111ebull;
    x ^= word_in_page * 0x2545f4914f6cdd1dull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return static_cast<uint32_t>(x ^ (x >> 32));
}

__global__ void fillPoolKernel(uint32_t* pool, uint64_t total_words,
                               uint64_t layer_stride_words,
                               uint64_t page_stride_words) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = gridDim.x * blockDim.x;
    for (uint64_t idx = tid; idx < total_words; idx += stride) {
        const uint64_t layer = idx / layer_stride_words;
        const uint64_t rem = idx - layer * layer_stride_words;
        const uint64_t page = rem / page_stride_words;
        const uint64_t word = rem - page * page_stride_words;
        pool[idx] = expectedWord(layer, page, word);
    }
}

__global__ void gatherKernel(const uint32_t* src_pool, uint32_t* staging,
                             TentNcclPagedTransferJob job,
                             uint64_t page_words) {
    const int layer_count = job.layer_end - job.layer_begin;
    const uint64_t work_items =
        static_cast<uint64_t>(layer_count) * static_cast<uint64_t>(job.num_pages);
    for (uint64_t item = blockIdx.x; item < work_items; item += gridDim.x) {
        const int page_slot = static_cast<int>(item % job.num_pages);
        const int layer =
            job.layer_begin + static_cast<int>(item / job.num_pages);
        const int src_page = job.src_page_table[page_slot];
        if (src_page < 0) continue;
        const uint64_t src_offset_words =
            (static_cast<uint64_t>(layer) * job.src_layer_stride) /
                sizeof(uint32_t) +
            static_cast<uint64_t>(src_page) *
                (job.src_page_stride_bytes / sizeof(uint32_t));
        const uint64_t dst_offset_words = item * page_words;
        for (uint64_t word = threadIdx.x; word < page_words;
             word += blockDim.x) {
            staging[dst_offset_words + word] = src_pool[src_offset_words + word];
        }
    }
}

__global__ void scatterKernel(const uint32_t* staging, uint32_t* dst_pool,
                              TentNcclPagedTransferJob job,
                              uint64_t page_words) {
    const int layer_count = job.layer_end - job.layer_begin;
    const uint64_t work_items =
        static_cast<uint64_t>(layer_count) * static_cast<uint64_t>(job.num_pages);
    for (uint64_t item = blockIdx.x; item < work_items; item += gridDim.x) {
        const int page_slot = static_cast<int>(item % job.num_pages);
        const int layer =
            job.layer_begin + static_cast<int>(item / job.num_pages);
        const int dst_page = job.dst_page_table[page_slot];
        if (dst_page < 0) continue;
        const uint64_t src_offset_words = item * page_words;
        const uint64_t dst_offset_words =
            (static_cast<uint64_t>(layer) * job.dst_layer_stride) /
                sizeof(uint32_t) +
            static_cast<uint64_t>(dst_page) *
                (job.dst_page_stride_bytes / sizeof(uint32_t));
        for (uint64_t word = threadIdx.x; word < page_words;
             word += blockDim.x) {
            dst_pool[dst_offset_words + word] = staging[src_offset_words + word];
        }
    }
}

__global__ void verifyKernel(const uint32_t* dst_pool,
                             TentNcclPagedTransferJob job,
                             uint64_t page_words,
                             unsigned long long* errors) {
    const int layer_count = job.layer_end - job.layer_begin;
    const uint64_t work_items =
        static_cast<uint64_t>(layer_count) * static_cast<uint64_t>(job.num_pages);
    for (uint64_t item = blockIdx.x; item < work_items; item += gridDim.x) {
        const int page_slot = static_cast<int>(item % job.num_pages);
        const int layer =
            job.layer_begin + static_cast<int>(item / job.num_pages);
        const int src_page = job.src_page_table[page_slot];
        const int dst_page = job.dst_page_table[page_slot];
        if (src_page < 0 || dst_page < 0) continue;
        const uint64_t dst_offset_words =
            (static_cast<uint64_t>(layer) * job.dst_layer_stride) /
                sizeof(uint32_t) +
            static_cast<uint64_t>(dst_page) *
                (job.dst_page_stride_bytes / sizeof(uint32_t));
        for (uint64_t word = threadIdx.x; word < page_words;
             word += blockDim.x) {
            const uint32_t expected =
                expectedWord(layer, static_cast<uint64_t>(src_page), word);
            const uint32_t got = dst_pool[dst_offset_words + word];
            if (got != expected) atomicAdd(errors, 1ull);
        }
        const uint64_t dst_page_stride_words =
            job.dst_page_stride_bytes / sizeof(uint32_t);
        for (uint64_t word = page_words + threadIdx.x;
             word < dst_page_stride_words; word += blockDim.x) {
            if (dst_pool[dst_offset_words + word] != 0)
                atomicAdd(errors, 1ull);
        }
    }
}

int copyGrid(uint64_t work_items) {
    return static_cast<int>(std::min<uint64_t>(work_items, 65535));
}

void* ncclAllocOn(int device, size_t bytes, size_t offset_bytes) {
    CHECK_CUDA(cudaSetDevice(device));
    void* ptr = nullptr;
    CHECK_NCCL(ncclMemAlloc(&ptr, bytes + offset_bytes));
    return static_cast<void*>(static_cast<char*>(ptr) + offset_bytes);
}

size_t requiredWindowOffset(const void* ptr) {
    constexpr uintptr_t kAlignment = NCCL_WIN_REQUIRED_ALIGNMENT;
    static_assert((kAlignment & (kAlignment - 1)) == 0);
    return reinterpret_cast<uintptr_t>(ptr) & (kAlignment - 1);
}

void* requiredWindowBase(void* ptr) {
    return static_cast<void*>(static_cast<char*>(ptr) -
                              requiredWindowOffset(ptr));
}

void cudaMallocOn(int device, void** ptr, size_t bytes) {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaMalloc(ptr, bytes));
}

void createComms(BenchState& state) {
    ncclUniqueId id{};
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_NCCL(ncclCommInitRank(&state.comm0, 2, id, 0));
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_NCCL(ncclCommInitRank(&state.comm1, 2, id, 1));
    CHECK_NCCL(ncclGroupEnd());
}

const char* ginBackendName(uint8_t type) {
    switch (type) {
        case NCCL_NET_DEVICE_GIN_PROXY:
            return "PROXY";
        case NCCL_NET_DEVICE_GIN_GDAKI:
            return "GDAKI";
        case NCCL_NET_DEVICE_GIN_GPI:
            return "GPI";
        default:
            return "UNKNOWN";
    }
}

void printGinBackends(const ncclDevComm_t& dev_comm) {
    std::cout << " gin_backends=[";
    for (int i = 0; i < static_cast<int>(dev_comm.ginConnectionCount); ++i) {
        if (i) std::cout << ",";
        const uint8_t type = dev_comm.ginNetDeviceTypes[i];
        std::cout << ginBackendName(type) << "(" << static_cast<int>(type)
                  << ")";
    }
    std::cout << "]";
}

void createDevComms(BenchState& state) {
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginForceEnable = true;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    reqs.ginContextCount = state.opts.lanes;
    reqs.ginSignalCount = state.opts.lanes * 2;

    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_NCCL(ncclDevCommCreate(state.comm0, &reqs, &state.dev_comm0));
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_NCCL(ncclDevCommCreate(state.comm1, &reqs, &state.dev_comm1));
    CHECK_NCCL(ncclGroupEnd());

    const char* nconnections = std::getenv("NCCL_GIN_NCONNECTIONS");
    std::cout << "dev_comm rank0 contexts=" << state.dev_comm0.ginContextCount
              << " connections="
              << static_cast<int>(state.dev_comm0.ginConnectionCount)
              << " signals=" << state.dev_comm0.ginSignalCount
              << " NCCL_GIN_NCONNECTIONS="
              << (nconnections ? nconnections : "<unset>");
    printGinBackends(state.dev_comm0);
    std::cout << std::endl;
}

WindowPair registerWindowPair(const BenchState& state, void* rank0_ptr,
                              void* rank1_ptr, size_t bytes) {
    WindowPair pair;
    const size_t rank0_offset = requiredWindowOffset(rank0_ptr);
    const size_t rank1_offset = requiredWindowOffset(rank1_ptr);
    CHECK_NCCL(ncclGroupStart());
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_NCCL(ncclCommWindowRegister(state.comm0,
                                      requiredWindowBase(rank0_ptr),
                                      bytes + rank0_offset,
                                      &pair.rank0, NCCL_WIN_COLL_SYMMETRIC));
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_NCCL(ncclCommWindowRegister(state.comm1,
                                      requiredWindowBase(rank1_ptr),
                                      bytes + rank1_offset,
                                      &pair.rank1, NCCL_WIN_COLL_SYMMETRIC));
    CHECK_NCCL(ncclGroupEnd());
    return pair;
}

void enablePeerAccessOneWay(int device, int peer) {
    int can_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, device, peer));
    if (!can_access) return;
    CHECK_CUDA(cudaSetDevice(device));
    cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        CHECK_CUDA(cudaGetLastError());
        return;
    }
    CHECK_CUDA(err);
}

void enablePeerAccess(const BenchState& state) {
    enablePeerAccessOneWay(state.opts.device0, state.opts.device1);
    enablePeerAccessOneWay(state.opts.device1, state.opts.device0);
}

void copyPageTables(BenchState& state) {
    std::vector<int32_t> src_table(state.opts.pages);
    std::vector<int32_t> dst_table(state.opts.pages);
    for (int i = 0; i < state.opts.pages; ++i) {
        src_table[i] = i;
        dst_table[i] = state.opts.identity_map
                           ? i : state.opts.physical_pages - 1 - i;
    }

    const size_t table_bytes = src_table.size() * sizeof(int32_t);
    cudaMallocOn(state.opts.device0, reinterpret_cast<void**>(&state.src_table0),
                 table_bytes);
    cudaMallocOn(state.opts.device0, reinterpret_cast<void**>(&state.dst_table0),
                 table_bytes);
    cudaMallocOn(state.opts.device1, reinterpret_cast<void**>(&state.src_table1),
                 table_bytes);
    cudaMallocOn(state.opts.device1, reinterpret_cast<void**>(&state.dst_table1),
                 table_bytes);

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaMemcpy(state.src_table0, src_table.data(), table_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(state.dst_table0, dst_table.data(), table_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaMemcpy(state.src_table1, src_table.data(), table_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(state.dst_table1, dst_table.data(), table_bytes,
                          cudaMemcpyHostToDevice));
}

TentNcclPagedTransferJob makeJob(int32_t* src_table, int32_t* dst_table,
                                 const BenchState& state) {
    TentNcclPagedTransferJob job{};
    job.src_page_table = src_table;
    job.dst_page_table = dst_table;
    job.num_pages = state.opts.pages;
    job.layer_begin = 0;
    job.layer_end = state.opts.layers;
    job.src_layer_stride = state.layer_stride_bytes;
    job.dst_layer_stride = state.layer_stride_bytes;
    job.src_base_offset = state.src_pool_window_offset;
    job.dst_base_offset = state.dst_pool_window_offset;
    job.page_bytes = state.page_bytes;
    job.src_page_stride_bytes = state.page_stride_bytes;
    job.dst_page_stride_bytes = state.page_stride_bytes;
    return job;
}

void copyJobs(BenchState& state) {
    TentNcclPagedTransferJob job0 =
        makeJob(state.src_table0, state.dst_table0, state);
    TentNcclPagedTransferJob job1 =
        makeJob(state.src_table1, state.dst_table1, state);
    state.job_host = job0;
    state.fused_job_count = state.opts.split_jobs ? state.opts.layers : 1;
    std::vector<TentNcclPagedTransferJob> jobs0(state.fused_job_count, job0);
    std::vector<TentNcclPagedTransferJob> jobs1(state.fused_job_count, job1);
    if (state.opts.split_jobs) {
        for (int i = 0; i < state.fused_job_count; ++i) {
            jobs0[i].layer_begin = jobs1[i].layer_begin = i;
            jobs0[i].layer_end = jobs1[i].layer_end = i + 1;
        }
    }
    const size_t jobs_bytes = jobs0.size() * sizeof(jobs0.front());
    cudaMallocOn(state.opts.device0, reinterpret_cast<void**>(&state.job0),
                 jobs_bytes);
    cudaMallocOn(state.opts.device1, reinterpret_cast<void**>(&state.job1),
                 jobs_bytes);
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaMemcpy(state.job0, jobs0.data(), jobs_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaMemcpy(state.job1, jobs1.data(), jobs_bytes,
                          cudaMemcpyHostToDevice));
}

void initializeData(BenchState& state) {
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaStreamCreateWithFlags(&state.stream0, cudaStreamNonBlocking));
    fillPoolKernel<<<256, 256, 0, state.stream0>>>(
        reinterpret_cast<uint32_t*>(state.src_pool), state.pool_bytes / 4,
        state.layer_stride_bytes / 4, state.page_stride_bytes / 4);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaStreamCreateWithFlags(&state.stream1, cudaStreamNonBlocking));
    CHECK_CUDA(cudaMemsetAsync(state.dst_pool, 0, state.pool_bytes,
                               state.stream1));
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
}

BenchState setup(const Options& opts) {
    BenchState state;
    state.opts = opts;
    state.page_bytes = static_cast<size_t>(opts.page_size_tokens) *
                       static_cast<size_t>(opts.num_kv_heads) *
                       static_cast<size_t>(opts.head_dim) * 2u *
                       static_cast<size_t>(opts.dtype_bytes);
    if (state.page_bytes == 0 || state.page_bytes % sizeof(uint32_t) != 0) {
        std::cerr << "page_bytes must be nonzero and 4-byte aligned"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    const size_t page_padding_bytes =
        static_cast<size_t>(opts.page_padding_bytes);
    if (state.page_bytes >
        std::numeric_limits<size_t>::max() - page_padding_bytes) {
        std::cerr << "page stride overflows" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    state.page_stride_bytes = state.page_bytes + page_padding_bytes;
    state.page_words = state.page_bytes / sizeof(uint32_t);
    state.layer_stride_bytes =
        static_cast<size_t>(opts.physical_pages) * state.page_stride_bytes;
    state.pool_bytes =
        static_cast<size_t>(opts.layers) * state.layer_stride_bytes;
    state.staging_bytes =
        static_cast<size_t>(opts.layers) * static_cast<size_t>(opts.pages) *
        state.page_bytes;
    state.layout.page_bytes = state.page_bytes;
    state.layout.src_page_stride_bytes = state.page_stride_bytes;
    state.layout.dst_page_stride_bytes = state.page_stride_bytes;
    state.layout.page_size_tokens = opts.page_size_tokens;
    state.layout.num_kv_heads = opts.num_kv_heads;
    state.layout.head_dim = opts.head_dim;
    state.layout.dtype_bytes = opts.dtype_bytes;
    state.layout.gin_resource_sharing = opts.gin_resource_sharing;
    state.layout.gin_opt_flags = static_cast<uint32_t>(opts.gin_opt_flags);
    if (opts.split_jobs && opts.layers > 1) {
        state.layout.uniform_job_num_pages = opts.pages;
        state.layout.uniform_job_num_layers = 1;
    }

    createComms(state);
    createDevComms(state);
    enablePeerAccess(state);

    const size_t window_offset_bytes =
        static_cast<size_t>(opts.window_offset_bytes);
    state.src_pool =
        ncclAllocOn(opts.device0, state.pool_bytes, window_offset_bytes);
    state.dst_pool =
        ncclAllocOn(opts.device1, state.pool_bytes, window_offset_bytes);
    state.src_staging =
        ncclAllocOn(opts.device0, state.staging_bytes, window_offset_bytes);
    state.dst_staging =
        ncclAllocOn(opts.device1, state.staging_bytes, window_offset_bytes);
    state.src_pool_window_offset = requiredWindowOffset(state.src_pool);
    state.dst_pool_window_offset = requiredWindowOffset(state.dst_pool);
    state.src_staging_window_offset =
        requiredWindowOffset(state.src_staging);
    state.dst_staging_window_offset =
        requiredWindowOffset(state.dst_staging);
    state.pool_window =
        registerWindowPair(state, state.src_pool, state.dst_pool,
                           state.pool_bytes);
    state.staging_window =
        registerWindowPair(state, state.src_staging, state.dst_staging,
                           state.staging_bytes);

    copyPageTables(state);
    copyJobs(state);
    cudaMallocOn(opts.device1, reinterpret_cast<void**>(&state.errors1),
                 sizeof(uint64_t));
    initializeData(state);
    return state;
}

void resetDestination(BenchState& state) {
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaMemsetAsync(state.dst_pool, 0, state.pool_bytes,
                               state.stream1));
    if (state.opts.baseline) {
        CHECK_CUDA(cudaMemsetAsync(state.dst_staging, 0, state.staging_bytes,
                                   state.stream1));
    }
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
}

double runFused(BenchState& state) {
    const unsigned long long signal = state.next_signal++;
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(tentNcclGinLaunchPagedPut(
        state.dev_comm0, 1, state.opts.lanes, state.pool_window.rank0,
        state.pool_window.rank0, state.layout, state.job0,
        state.fused_job_count, signal, state.stream0));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(tentNcclGinLaunchWaitSignal(
        state.dev_comm1, state.opts.lanes, 0, signal, state.stream1));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double runFusedAck(BenchState& state) {
    if (state.ack_signal_count + 1 < state.next_signal) {
        CHECK_CUDA(cudaSetDevice(state.opts.device1));
        for (unsigned long long target = state.ack_signal_count + 1;
             target < state.next_signal; ++target) {
            CHECK_CUDA(tentNcclGinLaunchWaitAck(
                state.dev_comm1, 0, state.opts.lanes, target, state.stream1));
        }
        CHECK_CUDA(cudaStreamSynchronize(state.stream1));
        state.ack_signal_count = state.next_signal - 1;
    }

    const unsigned long long signal = state.next_signal++;
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(tentNcclGinLaunchPagedPut(
        state.dev_comm0, 1, state.opts.lanes, state.pool_window.rank0,
        state.pool_window.rank0, state.layout, state.job0,
        state.fused_job_count, signal, state.stream0));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(tentNcclGinLaunchWaitAck(
        state.dev_comm1, 0, state.opts.lanes, signal, state.stream1));

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(tentNcclGinLaunchWaitSignal(
        state.dev_comm0, state.opts.lanes, state.opts.lanes, signal,
        state.stream0));
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
    state.ack_signal_count = signal;

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double runBaseline(BenchState& state) {
    const unsigned long long signal = state.next_signal++;
    const int blocks = copyGrid(static_cast<uint64_t>(state.opts.layers) *
                                static_cast<uint64_t>(state.opts.pages));
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    gatherKernel<<<blocks, 256, 0, state.stream0>>>(
        reinterpret_cast<const uint32_t*>(state.src_pool),
        reinterpret_cast<uint32_t*>(state.src_staging), state.job_host,
        state.page_words);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(tentNcclGinLaunchWaitSignal(
        state.dev_comm1, state.opts.lanes, 0, signal, state.stream1));

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(tentNcclGinLaunchPut(
        state.dev_comm0, 1, state.opts.lanes, state.staging_window.rank0,
        state.dst_staging_window_offset, state.staging_window.rank0,
        state.src_staging_window_offset, state.staging_bytes, signal,
        state.stream0));

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    scatterKernel<<<blocks, 256, 0, state.stream1>>>(
        reinterpret_cast<const uint32_t*>(state.dst_staging),
        reinterpret_cast<uint32_t*>(state.dst_pool),
        makeJob(state.src_table1, state.dst_table1, state), state.page_words);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double runCoalescedLayerPut(BenchState& state) {
    const unsigned long long signal = state.next_signal++;
    const size_t layer_bytes = state.opts.pages * state.page_bytes;
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(tentNcclGinLaunchWaitSignal(
        state.dev_comm1, state.opts.lanes, 0, signal, state.stream1));

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    for (int layer = 0; layer < state.opts.layers; ++layer) {
        const size_t layer_offset =
            static_cast<size_t>(layer) * state.layer_stride_bytes;
        const bool last = layer + 1 == state.opts.layers;

        CHECK_CUDA(tentNcclGinLaunchPut(
            state.dev_comm0, 1, state.opts.lanes, state.pool_window.rank0,
            state.dst_pool_window_offset + layer_offset,
            state.pool_window.rank0,
            state.src_pool_window_offset + layer_offset, layer_bytes,
            last ? signal : 0, state.stream0));
    }

    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double runCoalescedPeerMemcpy(BenchState& state) {
    const size_t layer_bytes = state.opts.pages * state.page_bytes;
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    for (int layer = 0; layer < state.opts.layers; ++layer) {
        const size_t layer_offset =
            static_cast<size_t>(layer) * state.layer_stride_bytes;
        const char* src = static_cast<const char*>(state.src_pool) +
                          layer_offset;
        char* dst = static_cast<char*>(state.dst_pool) + layer_offset;
        CHECK_CUDA(cudaMemcpyPeerAsync(
            dst, state.opts.device1, src, state.opts.device0,
            layer_bytes, state.stream0));
    }
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double runPeerPageMemcpy(BenchState& state) {
    const auto start = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaSetDevice(state.opts.device0));
    for (int layer = 0; layer < state.opts.layers; ++layer) {
        const size_t layer_offset =
            static_cast<size_t>(layer) * state.layer_stride_bytes;
        for (int page_slot = 0; page_slot < state.opts.pages; ++page_slot) {
            const int src_page = page_slot;
            const int dst_page =
                state.opts.identity_map
                    ? page_slot : state.opts.physical_pages - 1 - page_slot;
            const char* src = static_cast<const char*>(state.src_pool) +
                              layer_offset +
                              static_cast<size_t>(src_page) * state.page_stride_bytes;
            char* dst = static_cast<char*>(state.dst_pool) + layer_offset +
                        static_cast<size_t>(dst_page) * state.page_stride_bytes;
            CHECK_CUDA(cudaMemcpyPeerAsync(dst, state.opts.device1, src,
                                           state.opts.device0, state.page_bytes,
                                           state.stream0));
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(state.stream0));

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

uint64_t verifyDestination(BenchState& state) {
    CHECK_CUDA(cudaSetDevice(state.opts.device1));
    CHECK_CUDA(cudaMemsetAsync(state.errors1, 0, sizeof(uint64_t),
                               state.stream1));
    const int blocks = copyGrid(static_cast<uint64_t>(state.opts.layers) *
                                static_cast<uint64_t>(state.opts.pages));
    verifyKernel<<<blocks, 256, 0, state.stream1>>>(
        reinterpret_cast<const uint32_t*>(state.dst_pool),
        makeJob(state.src_table1, state.dst_table1, state), state.page_words,
        reinterpret_cast<unsigned long long*>(state.errors1));
    CHECK_CUDA(cudaGetLastError());
    uint64_t errors = 0;
    CHECK_CUDA(cudaMemcpyAsync(&errors, state.errors1, sizeof(errors),
                               cudaMemcpyDeviceToHost, state.stream1));
    CHECK_CUDA(cudaStreamSynchronize(state.stream1));
    return errors;
}

double averageMs(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    double sum = 0.0;
    for (double sample : samples) sum += sample;
    return sum / static_cast<double>(samples.size());
}

double percentileMs(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double rank = p * static_cast<double>(sorted.size() - 1);
    return sorted[static_cast<size_t>(rank)];
}

void runAndReport(BenchState& state, const char* name,
                  double (*fn)(BenchState&)) {
    for (int i = 0; i < state.opts.warmup; ++i) fn(state);
    std::vector<double> samples;
    samples.reserve(state.opts.iters);
    for (int i = 0; i < state.opts.iters; ++i) {
        samples.push_back(fn(state));
    }
    std::sort(samples.begin(), samples.end());
    std::cout << std::left << std::setw(18) << name << std::right
              << " avg_ms=" << std::fixed << std::setprecision(3)
              << averageMs(samples) << " p50_ms=" << percentileMs(samples, 0.50)
              << " p95_ms=" << percentileMs(samples, 0.95) << std::endl;
}

std::string mib(size_t bytes) {
    const double value = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value << " MiB";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options opts = parseOptions(argc, argv);
    validateOptions(opts);

    if (opts.check_option_guard) {
        if (opts.gin_opt_flags == ncclGinOptFlagsDefault) {
            std::cerr << "--check-option-guard requires a non-default GIN flag"
                      << std::endl;
            return EXIT_FAILURE;
        }
        TentNcclPagedKvLayout layout;
        layout.page_stride_bytes = 1;
        layout.gin_opt_flags = static_cast<uint32_t>(opts.gin_opt_flags);
        ncclDevComm_t dev_comm{};
        const cudaError_t err = tentNcclGinLaunchPagedPut(
            dev_comm, 0, 1, nullptr, nullptr, layout, nullptr, 0, 0, nullptr);
        std::cout << "option_guard flags=" << opts.gin_opt_flags
                  << " result=" << cudaGetErrorName(err) << std::endl;
        return err == cudaErrorNotSupported ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    BenchState state = setup(opts);
    const bool coalesced_control =
        opts.identity_map && state.page_stride_bytes == state.page_bytes;
    const uint64_t work_items =
        static_cast<uint64_t>(opts.layers) * static_cast<uint64_t>(opts.pages);
    std::cout << "paged_gin_kv_bench devices=" << opts.device0 << ","
              << opts.device1 << " lanes=" << opts.lanes
              << " layers=" << opts.layers << " pages=" << opts.pages
              << " physical_pages=" << opts.physical_pages
              << " page_bytes=" << state.page_bytes
              << " page_stride_bytes=" << state.page_stride_bytes
              << " window_offset_bytes=" << opts.window_offset_bytes
              << " src_window_offset=" << state.src_pool_window_offset
              << " dst_window_offset=" << state.dst_pool_window_offset
              << " gin_sharing=" << opts.gin_resource_sharing
              << " gin_opt_flags=" << opts.gin_opt_flags
              << " jobs=" << state.fused_job_count
              << " map=" << (opts.identity_map ? "identity" : "reverse")
              << " total_page_copies=" << work_items << std::endl;
    std::cout << "pool_bytes_per_rank=" << mib(state.pool_bytes)
              << " staging_bytes_per_rank=" << mib(state.staging_bytes)
              << std::endl;

    if (opts.validate) {
        resetDestination(state);
        runFused(state);
        const uint64_t fused_errors = verifyDestination(state);
        std::cout << "fused_validation_errors=" << fused_errors << std::endl;
        if (fused_errors != 0) return EXIT_FAILURE;

        if (opts.with_ack) {
            resetDestination(state);
            runFusedAck(state);
            const uint64_t ack_errors = verifyDestination(state);
            std::cout << "ack_validation_errors=" << ack_errors << std::endl;
            if (ack_errors != 0) return EXIT_FAILURE;
        }

        if (opts.baseline) {
            resetDestination(state);
            runBaseline(state);
            const uint64_t baseline_errors = verifyDestination(state);
            std::cout << "baseline_validation_errors=" << baseline_errors
                      << std::endl;
            if (baseline_errors != 0) return EXIT_FAILURE;
        }

        if (coalesced_control) {
            resetDestination(state);
            runCoalescedLayerPut(state);
            const uint64_t coalesced_errors = verifyDestination(state);
            std::cout << "coalesced_validation_errors=" << coalesced_errors
                      << std::endl;
            if (coalesced_errors != 0) return EXIT_FAILURE;
            resetDestination(state);
            runCoalescedPeerMemcpy(state);
            const uint64_t coalesced_peer_errors = verifyDestination(state);
            std::cout << "coalesced_peer_validation_errors="
                      << coalesced_peer_errors << std::endl;
            if (coalesced_peer_errors != 0) return EXIT_FAILURE;

        }

        resetDestination(state);
        runPeerPageMemcpy(state);
        const uint64_t peer_errors = verifyDestination(state);
        std::cout << "peer_memcpy_validation_errors=" << peer_errors
                  << std::endl;
        if (peer_errors != 0) return EXIT_FAILURE;
    }

    resetDestination(state);
    runAndReport(state, "paged_gin_put", runFused);
    if (opts.with_ack) {
        resetDestination(state);
        runAndReport(state, "paged_gin_put_ack", runFusedAck);
    }

    if (opts.baseline) {
        resetDestination(state);
        runAndReport(state, "gather_put_scatter", runBaseline);
    }
    if (coalesced_control) {
        resetDestination(state);
        runAndReport(state, "coalesced_layer_put", runCoalescedLayerPut);
        resetDestination(state);
        runAndReport(state, "coalesced_peer_copy", runCoalescedPeerMemcpy);
    }
    resetDestination(state);
    runAndReport(state, "peer_page_memcpy", runPeerPageMemcpy);
    return EXIT_SUCCESS;
}
