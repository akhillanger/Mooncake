// Copyright 2026 KVCache.AI
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mooncake_ep_elastic_jit.h"

#ifdef MOONCAKE_EP_ENABLE_NCCL_JIT

#include <cuda.h>
#include <nccl.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <elastic/mooncake_ep_elastic_compiled.cuh>

// The compiler process, on-disk cache, and per-context loaded-kernel cache
// follow the design used by NVIDIA/nccl contrib/nccl_ep/device/jit. Mooncake
// emits an explicit instantiation of its existing __global__ template and uses
// CUDA's module function enumeration, so no kernel body is duplicated and the
// AOT and JIT paths compile the same source template.

#ifndef MOONCAKE_EP_JIT_EP_INCLUDE_DIR
#define MOONCAKE_EP_JIT_EP_INCLUDE_DIR ""
#endif
#ifndef MOONCAKE_EP_JIT_TE_INCLUDE_DIR
#define MOONCAKE_EP_JIT_TE_INCLUDE_DIR ""
#endif
#ifndef MOONCAKE_EP_JIT_NCCL_INCLUDE_DIR
#define MOONCAKE_EP_JIT_NCCL_INCLUDE_DIR ""
#endif
#ifndef MOONCAKE_EP_JIT_CUDA_INCLUDE_DIR
#define MOONCAKE_EP_JIT_CUDA_INCLUDE_DIR ""
#endif
#ifndef MOONCAKE_EP_JIT_NVCC
#define MOONCAKE_EP_JIT_NVCC "nvcc"
#endif

namespace mooncake::elastic::jit {
namespace {

constexpr int64_t kTimeoutCycles = 200000000000ll;
constexpr int kMinComputeCapability = 80;
constexpr std::string_view kBuildId = __DATE__ " " __TIME__;

struct ProcessResult {
    int exit_code = -1;
    std::string output;
};

struct KernelVariant {
    std::string family;
    std::string name;
    std::string source;
    int smem_bytes = 0;
};

struct LoadedKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

struct MemoryKey {
    CUcontext context = nullptr;
    std::array<int, 13> values{};

    bool operator==(const MemoryKey& other) const {
        return context == other.context && values == other.values;
    }
};

struct MemoryKeyHash {
    size_t operator()(const MemoryKey& key) const {
        size_t hash =
            std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.context));
        for (int value : key.values) {
            hash ^= std::hash<int>{}(value) + 0x9e3779b9u + (hash << 6) +
                    (hash >> 2);
        }
        return hash;
    }
};

std::string env_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

bool env_flag(const char* name, bool default_value) {
    const std::string value = env_value(name);
    if (value.empty()) return default_value;
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" ||
        value == "ON") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" ||
        value == "off" || value == "OFF") {
        return false;
    }
    throw std::invalid_argument(
        std::string(name) + " must be one of 0, 1, false, true, off, or on");
}

bool log_enabled() { return env_flag("MOONCAKE_EP_JIT_LOG", false); }

void jit_log(const std::string& message) {
    if (!log_enabled()) return;
    std::fprintf(stderr, "[Mooncake EP JIT] pid=%ld %s\n",
                 static_cast<long>(getpid()), message.c_str());
}

std::vector<std::string> split_flags(const std::string& flags) {
    std::vector<std::string> result;
    std::string current;
    char quote = '\0';
    bool escaped = false;
    for (char c : flags) {
        if (escaped) {
            current.push_back(c);
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (quote != '\0') {
            if (c == quote) {
                quote = '\0';
            } else {
                current.push_back(c);
            }
        } else if (c == '\'' || c == '"') {
            quote = c;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                result.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (escaped) current.push_back('\\');
    if (quote != '\0') {
        throw std::invalid_argument(
            "MOONCAKE_EP_JIT_EXTRA_FLAGS contains an unmatched quote");
    }
    if (!current.empty()) result.push_back(std::move(current));
    return result;
}

std::string command_string(const std::vector<std::string>& argv) {
    std::ostringstream out;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) out << ' ';
        if (argv[i].find_first_of(" \t\n\"'\\") == std::string::npos) {
            out << argv[i];
            continue;
        }
        out << '\'';
        for (char c : argv[i]) {
            if (c == '\'')
                out << "'\\''";
            else
                out << c;
        }
        out << '\'';
    }
    return out.str();
}

bool run_process(const std::vector<std::string>& argv, ProcessResult* result) {
    if (result == nullptr || argv.empty()) return false;
    result->exit_code = -1;
    result->output.clear();
    int pipe_fd[2] = {-1, -1};
    if (pipe(pipe_fd) != 0) {
        result->output = std::string("pipe failed: ") + std::strerror(errno);
        return false;
    }

    std::vector<char*> raw_argv;
    raw_argv.reserve(argv.size() + 1);
    for (const std::string& arg : argv)
        raw_argv.push_back(const_cast<char*>(arg.c_str()));
    raw_argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        const int saved_errno = errno;
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        result->output =
            std::string("fork failed: ") + std::strerror(saved_errno);
        return false;
    }
    if (pid == 0) {
        close(pipe_fd[0]);
        dup2(pipe_fd[1], STDOUT_FILENO);
        dup2(pipe_fd[1], STDERR_FILENO);
        close(pipe_fd[1]);
        execvp(raw_argv[0], raw_argv.data());
        _exit(127);
    }

    close(pipe_fd[1]);
    char buffer[4096];
    while (true) {
        const ssize_t size = read(pipe_fd[0], buffer, sizeof(buffer));
        if (size > 0) {
            result->output.append(buffer, static_cast<size_t>(size));
        } else if (size == 0) {
            break;
        } else if (errno != EINTR) {
            result->output +=
                std::string("\nread failed: ") + std::strerror(errno);
            break;
        }
    }
    close(pipe_fd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        result->output +=
            std::string("\nwaitpid failed: ") + std::strerror(errno);
        return false;
    }
    if (WIFEXITED(status)) {
        result->exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result->exit_code = 128 + WTERMSIG(status);
    }
    return result->exit_code == 0;
}

class ScopedFileLock {
   public:
    explicit ScopedFileLock(const std::filesystem::path& path) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            throw std::runtime_error("cannot create JIT cache directory: " +
                                     error.message());
        }
        fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open JIT lock " + path.string() +
                                     ": " + std::strerror(errno));
        }
        if (flock(fd_, LOCK_EX) != 0) {
            const std::string message = std::strerror(errno);
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot lock JIT cache: " + message);
        }
    }

    ScopedFileLock(const ScopedFileLock&) = delete;
    ScopedFileLock& operator=(const ScopedFileLock&) = delete;

    ~ScopedFileLock() {
        if (fd_ < 0) return;
        flock(fd_, LOCK_UN);
        close(fd_);
    }

   private:
    int fd_ = -1;
};

void write_file_atomic(const std::filesystem::path& path,
                       const std::string& contents) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        throw std::runtime_error("cannot create JIT cache directory: " +
                                 error.message());
    }
    const auto temporary =
        path.string() + ".tmp." + std::to_string(static_cast<long>(getpid()));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot write JIT file " + temporary);
        }
        output.write(contents.data(),
                     static_cast<std::streamsize>(contents.size()));
        if (!output) {
            throw std::runtime_error("short write to JIT file " + temporary);
        }
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot publish JIT file " + path.string() +
                                 ": " + error.message());
    }
}

std::string fnv1a_digest(const std::vector<std::string>& parts) {
    uint64_t hash = 1469598103934665603ull;
    for (const auto& part : parts) {
        for (unsigned char c : part) {
            hash ^= static_cast<uint64_t>(c);
            hash *= 1099511628211ull;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::string resolve_nvcc() {
    std::string path = env_value("MOONCAKE_EP_JIT_NVCC");
    if (!path.empty()) return path;
    path = env_value("NVCC");
    if (!path.empty()) return path;
    path = env_value("CUDA_HOME");
    if (path.empty()) path = env_value("CUDA_PATH");
    if (!path.empty())
        return (std::filesystem::path(path) / "bin/nvcc").string();
    if (std::string(MOONCAKE_EP_JIT_NVCC).size() != 0)
        return MOONCAKE_EP_JIT_NVCC;
    return "nvcc";
}

std::string compiler_id() {
    static const std::string id = [] {
        ProcessResult result;
        const std::string nvcc = resolve_nvcc();
        run_process({nvcc, "--version"}, &result);
        return nvcc + "\n" + result.output +
               "\nexit=" + std::to_string(result.exit_code);
    }();
    return id;
}

std::string cuda_error(CUresult status) {
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    return std::string(name == nullptr ? "CUDA_ERROR" : name) + ": " +
           (description == nullptr ? "unknown driver error" : description);
}

std::filesystem::path cache_root() {
    const std::string configured = env_value("MOONCAKE_EP_JIT_CACHE_DIR");
    if (!configured.empty()) return configured;
    return std::filesystem::path("/tmp") / "mooncake_ep" / "jit";
}

std::string architecture_for_device(int sm) {
    const std::string configured = env_value("MOONCAKE_EP_JIT_ARCH");
    if (!configured.empty()) return configured;
    return "sm_" + std::to_string(sm);
}

std::vector<std::string> compiler_options(const std::string& architecture) {
    std::vector<std::string> result = {
        "--std=c++20",
        "--expt-relaxed-constexpr",
        "-O3",
        "-arch=" + architecture,
        "-DUSE_CUDA=1",
        "-DUSE_NCCL_DEVICE=1",
        "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1",
        "-DMOONCAKE_EP_NCCL_JIT_ONLY=1",
        "-DEP_NUM_TOPK_IDX_BITS=64",
        "-DNDEBUG=1",
    };
    const auto add_include = [&result](const char* path) {
        if (path != nullptr && path[0] != '\0')
            result.emplace_back(std::string("-I") + path);
    };
    add_include(MOONCAKE_EP_JIT_EP_INCLUDE_DIR);
    add_include(MOONCAKE_EP_JIT_TE_INCLUDE_DIR);
    add_include(MOONCAKE_EP_JIT_NCCL_INCLUDE_DIR);
    const std::filesystem::path nccl_device =
        std::filesystem::path(MOONCAKE_EP_JIT_NCCL_INCLUDE_DIR) / "nccl_device";
    add_include(nccl_device.c_str());
    add_include(MOONCAKE_EP_JIT_CUDA_INCLUDE_DIR);
    const std::filesystem::path cccl =
        std::filesystem::path(MOONCAKE_EP_JIT_CUDA_INCLUDE_DIR) / "cccl";
    add_include(cccl.c_str());

    auto extra = split_flags(env_value("MOONCAKE_EP_JIT_EXTRA_FLAGS"));
    result.insert(result.end(), std::make_move_iterator(extra.begin()),
                  std::make_move_iterator(extra.end()));
    return result;
}

std::string dispatch_variant_name(const DispatchSpec& spec) {
    std::ostringstream name;
    name << (spec.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch")
         << "_hb" << spec.hidden_bytes << "_sf" << spec.num_sf_packs << "_max"
         << spec.num_max_tokens_per_rank << "_e" << spec.num_experts << "_k"
         << spec.num_topk << "_sms" << spec.num_sms << "_so"
         << spec.num_scaleout_ranks << "_su" << spec.num_scaleup_ranks
         << "_notify" << spec.num_notify_warps << "_reuse"
         << static_cast<int>(spec.reuse_slot_indices);
    return name.str();
}

std::string combine_variant_name(const CombineSpec& spec) {
    std::ostringstream name;
    name << (spec.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine")
         << "_h" << spec.hidden << "_max" << spec.num_max_tokens_per_rank
         << "_e" << spec.num_experts << "_k" << spec.num_topk << "_sms"
         << spec.num_sms << "_so" << spec.num_scaleout_ranks << "_su"
         << spec.num_scaleup_ranks;
    return name.str();
}

std::string dispatch_source(const DispatchSpec& spec) {
    std::ostringstream source;
    source << "#include <elastic/mooncake_ep_elastic_"
           << (spec.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch")
           << "_official.cuh>\n\n"
           << "using Ops = mooncake::elastic::transport::NcclOps;\n\n";
    if (spec.num_scaleout_ranks == 1) {
        source << "template __global__ void mooncake::elastic::dispatch_impl<\n"
               << "    Ops, true, false, "
               << (spec.reuse_slot_indices ? "true" : "false") << ", "
               << spec.num_sms << ", " << spec.num_notify_warps
               << ", Ops::kNumDispatchWarps, " << spec.num_scaleup_ranks << ", "
               << spec.hidden_bytes << ", " << spec.num_sf_packs << ", "
               << spec.num_max_tokens_per_rank << ", " << spec.num_experts
               << ", " << spec.num_topk << ", 1, Ops::kNumQPs, "
               << kTimeoutCycles << "ll>(\n"
               << "    void*, mooncake::sf_pack_t*, mooncake::topk_idx_t*, "
                  "float*,\n"
               << "    mooncake::topk_idx_t*, int*, int*, int*, int*, int, "
                  "int, int,\n"
               << "    const Ops::Context, void*, void*, void*, int);\n";
    } else {
        source
            << "template __global__ void "
               "mooncake::elastic::hybrid_dispatch_impl<\n"
            << "    Ops, false, "
            << (spec.reuse_slot_indices ? "true" : "false") << ", "
            << spec.num_sms << ", " << spec.num_notify_warps
            << ", Ops::kNumHybridScaleoutWarps, Ops::kNumHybridForwardWarps, "
            << spec.num_scaleout_ranks << ", " << spec.num_scaleup_ranks << ", "
            << spec.hidden_bytes << ", " << spec.num_sf_packs << ", "
            << spec.num_max_tokens_per_rank << ", " << spec.num_experts << ", "
            << spec.num_topk << ", 1, Ops::kNumQPs, " << kTimeoutCycles
            << "ll>(\n"
            << "    void*, mooncake::sf_pack_t*, mooncake::topk_idx_t*, "
               "float*,\n"
            << "    mooncake::topk_idx_t*, int*, int*, int*, int*, int*, int, "
               "int,\n"
            << "    int, const Ops::Context, void*, void*, void*, int, int);\n";
    }
    return source.str();
}

std::string combine_source(const CombineSpec& spec) {
    std::ostringstream source;
    source << "#include <elastic/mooncake_ep_elastic_"
           << (spec.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine")
           << "_official.cuh>\n\n"
           << "using Ops = mooncake::elastic::transport::NcclOps;\n\n";
    if (spec.num_scaleout_ranks == 1) {
        source << "template __global__ void mooncake::elastic::combine_impl<\n"
               << "    Ops, true, false, true, " << spec.num_sms
               << ", Ops::kNumCombineWarps, " << spec.num_scaleup_ranks << ", "
               << spec.hidden << ", " << spec.num_max_tokens_per_rank << ", "
               << spec.num_experts << ", " << spec.num_topk
               << ", Ops::kNumQPs, " << kTimeoutCycles << "ll>(\n"
               << "    nv_bfloat16*, float*, int*, int*, const Ops::Context, "
                  "void*,\n"
               << "    void*, int, int);\n";
    } else {
        source << "template __global__ void "
                  "mooncake::elastic::hybrid_combine_impl<\n"
               << "    Ops, false, true, " << spec.num_sms
               << ", Ops::kNumHybridScaleupWarps, Ops::kNumHybridForwardWarps, "
               << spec.num_scaleout_ranks << ", " << spec.num_scaleup_ranks
               << ", " << spec.hidden << ", " << spec.num_max_tokens_per_rank
               << ", " << spec.num_experts << ", " << spec.num_topk
               << ", Ops::kNumQPs, " << kTimeoutCycles << "ll>(\n"
               << "    nv_bfloat16*, float*, int*, int*, int*, int*,\n"
               << "    const Ops::Context, void*, void*, int, int, int);\n";
    }
    return source.str();
}

KernelVariant make_dispatch_variant(const DispatchSpec& spec) {
    return {spec.num_scaleout_ranks == 1 ? "dispatch" : "hybrid_dispatch",
            dispatch_variant_name(spec), dispatch_source(spec),
            spec.smem_bytes};
}

KernelVariant make_combine_variant(const CombineSpec& spec) {
    return {spec.num_scaleout_ranks == 1 ? "combine" : "hybrid_combine",
            combine_variant_name(spec), combine_source(spec), spec.smem_bytes};
}

std::string compile_key(const KernelVariant& variant, int sm,
                        const std::string& architecture,
                        const std::vector<std::string>& options) {
    std::vector<std::string> parts = {
        variant.family,
        variant.name,
        variant.source,
        std::to_string(sm),
        architecture,
        std::string(kBuildId),
        compiler_id(),
        std::to_string(NCCL_VERSION_CODE),
        std::to_string(CUDA_VERSION),
    };
    parts.insert(parts.end(), options.begin(), options.end());
    return fnv1a_digest(parts);
}

void compile_variant(const KernelVariant& variant,
                     const std::filesystem::path& source_path,
                     const std::filesystem::path& cubin_path,
                     const std::filesystem::path& log_path,
                     const std::vector<std::string>& options) {
    write_file_atomic(source_path, variant.source);
    const auto temporary_cubin = cubin_path.string() + ".tmp." +
                                 std::to_string(static_cast<long>(getpid()));
    std::vector<std::string> argv = {
        resolve_nvcc(), "--cubin", source_path.string(), "-o", temporary_cubin};
    argv.insert(argv.end(), options.begin(), options.end());

    ProcessResult result;
    const auto begin = std::chrono::steady_clock::now();
    const bool succeeded = run_process(argv, &result);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    std::ostringstream log;
    log << "$ " << command_string(argv) << '\n' << result.output;
    if (result.output.empty() || result.output.back() != '\n') log << '\n';
    log << "exit_code=" << result.exit_code << '\n'
        << "compiler_seconds=" << seconds << '\n';
    write_file_atomic(log_path, log.str());
    if (!succeeded) {
        std::filesystem::remove(temporary_cubin);
        throw std::runtime_error("nvcc failed for Mooncake EP JIT variant " +
                                 variant.name + "; see " + log_path.string() +
                                 "\n" + result.output);
    }

    std::error_code error;
    std::filesystem::rename(temporary_cubin, cubin_path, error);
    if (error) {
        std::filesystem::remove(temporary_cubin);
        throw std::runtime_error("cannot publish JIT cubin " +
                                 cubin_path.string() + ": " + error.message());
    }
    jit_log("family=" + variant.family + " variant=" + variant.name +
            " cache=miss compiler_seconds=" + std::to_string(seconds) +
            " cubin=" + cubin_path.string());
}

CUcontext current_context() {
    CUcontext context = nullptr;
    CUresult status = cuCtxGetCurrent(&context);
    if (status == CUDA_SUCCESS && context == nullptr) {
        cudaFree(nullptr);
        status = cuCtxGetCurrent(&context);
    }
    if (status != CUDA_SUCCESS || context == nullptr) {
        throw std::runtime_error("cannot obtain the current CUDA context: " +
                                 cuda_error(status));
    }
    return context;
}

MemoryKey memory_key(const DispatchSpec& spec, CUcontext context) {
    return {
        context,
        {spec.num_scaleout_ranks == 1 ? 1 : 2, spec.hidden_bytes,
         spec.num_sf_packs, spec.num_max_tokens_per_rank, spec.num_experts,
         spec.num_topk, spec.num_sms, spec.num_notify_warps, spec.num_threads,
         spec.smem_bytes, spec.num_scaleout_ranks, spec.num_scaleup_ranks,
         static_cast<int>(spec.reuse_slot_indices)}};
}

MemoryKey memory_key(const CombineSpec& spec, CUcontext context) {
    return {context,
            {spec.num_scaleout_ranks == 1 ? 3 : 4, spec.hidden,
             spec.num_max_tokens_per_rank, spec.num_experts, spec.num_topk,
             spec.num_sms, spec.num_threads, spec.smem_bytes,
             spec.num_scaleout_ranks, spec.num_scaleup_ranks, 0, 0, 0}};
}

class KernelCache {
   public:
    static KernelCache& instance() {
        static KernelCache cache;
        return cache;
    }

    CUfunction get(const DispatchSpec& spec) {
        const MemoryKey key = memory_key(spec, current_context());
        return get(key, [&] { return make_dispatch_variant(spec); });
    }

    CUfunction get(const CombineSpec& spec) {
        const MemoryKey key = memory_key(spec, current_context());
        return get(key, [&] { return make_combine_variant(spec); });
    }

   private:
    template <typename Factory>
    CUfunction get(const MemoryKey& memory_key, Factory&& make_variant) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto loaded = kernels_.find(memory_key);
        if (loaded != kernels_.end()) return loaded->second->function;

        const KernelVariant variant = make_variant();
        int device = -1;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            throw std::runtime_error(
                "cannot query the CUDA device for Mooncake EP JIT");
        }
        const int sm = properties.major * 10 + properties.minor;
        if (sm < kMinComputeCapability) {
            throw std::runtime_error("Mooncake EP JIT requires sm_80 or newer");
        }

        const std::string architecture = architecture_for_device(sm);
        const auto options = compiler_options(architecture);
        const std::string digest =
            compile_key(variant, sm, architecture, options);
        const auto root = cache_root();
        const auto source_path = root / (digest + ".cu");
        const auto cubin_path = root / (digest + ".cubin");
        const auto log_path = root / (digest + ".log");
        const auto lock_path = root / (digest + ".lock");
        bool cache_hit = false;
        {
            ScopedFileLock lock(lock_path);
            std::error_code error;
            cache_hit = std::filesystem::is_regular_file(cubin_path, error) &&
                        !error &&
                        std::filesystem::file_size(cubin_path, error) > 0;
            if (!cache_hit) {
                compile_variant(variant, source_path, cubin_path, log_path,
                                options);
            }
        }
        if (cache_hit) {
            jit_log("family=" + variant.family + " variant=" + variant.name +
                    " cache=hit cubin=" + cubin_path.string());
        }

        auto kernel = std::make_unique<LoadedKernel>();
        const auto load_begin = std::chrono::steady_clock::now();
        CUresult status = cuModuleLoad(&kernel->module, cubin_path.c_str());
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error("cannot load JIT cubin " +
                                     cubin_path.string() + ": " +
                                     cuda_error(status));
        }
        unsigned int function_count = 0;
        status = cuModuleGetFunctionCount(&function_count, kernel->module);
        if (status != CUDA_SUCCESS || function_count != 1) {
            cuModuleUnload(kernel->module);
            throw std::runtime_error(
                "Mooncake EP JIT cubin must contain exactly one kernel; got " +
                std::to_string(function_count));
        }
        status =
            cuModuleEnumerateFunctions(&kernel->function, 1, kernel->module);
        if (status == CUDA_SUCCESS) {
            status = cuFuncSetAttribute(
                kernel->function,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                variant.smem_bytes);
        }
        if (status != CUDA_SUCCESS) {
            cuModuleUnload(kernel->module);
            throw std::runtime_error("cannot configure JIT kernel " +
                                     variant.name + ": " + cuda_error(status));
        }
        const double load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          load_begin)
                .count();
        jit_log("family=" + variant.family + " variant=" + variant.name +
                " load_seconds=" + std::to_string(load_seconds));
        CUfunction function = kernel->function;
        kernels_.emplace(memory_key, std::move(kernel));
        return function;
    }

    // Modules deliberately remain loaded for the process lifetime. Unloading
    // them during static destruction can run after CUDA context teardown.
    std::mutex mutex_;
    std::unordered_map<MemoryKey, std::unique_ptr<LoadedKernel>, MemoryKeyHash>
        kernels_;
};

template <typename Spec>
void launch_kernel(const Spec& spec, void** arguments, cudaStream_t stream) {
    CUfunction function = KernelCache::instance().get(spec);
    const CUresult status = cuLaunchCooperativeKernel(
        function, spec.num_sms, 1, 1, spec.num_threads, 1, 1,
        static_cast<unsigned int>(spec.smem_bytes),
        reinterpret_cast<CUstream>(stream), arguments);
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("cannot launch Mooncake EP JIT kernel: " +
                                 cuda_error(status));
    }
}

}  // namespace

bool requested_by_environment() {
    return env_flag("MOONCAKE_EP_NCCL_JIT", false);
}

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
                     cudaStream_t stream) {
    static_assert(sizeof(topk_idx_t) == sizeof(int64_t));
    if (spec.num_scaleout_ranks == 1) {
        void* arguments[] = {
            &x,
            &sf,
            &topk_idx,
            &topk_weights,
            &copied_topk_idx,
            &cumulative_local_expert_recv_stats,
            &psum_num_recv_tokens_per_scaleup_rank,
            &psum_num_recv_tokens_per_expert,
            &dst_buffer_slot_idx,
            &num_tokens,
            &sf_token_stride,
            &sf_hidden_stride,
            const_cast<transport::NcclContext*>(&comm_ctx),
            &buffer,
            &workspace,
            &mapped_host_workspace,
            &scaleup_rank_idx,
        };
        launch_kernel(spec, arguments, stream);
        return;
    }

    void* arguments[] = {
        &x,
        &sf,
        &topk_idx,
        &topk_weights,
        &copied_topk_idx,
        &cumulative_local_expert_recv_stats,
        &psum_num_recv_tokens_per_scaleup_rank,
        &psum_num_recv_tokens_per_expert,
        &dst_buffer_slot_idx,
        &token_metadata_at_forward,
        &num_tokens,
        &sf_token_stride,
        &sf_hidden_stride,
        const_cast<transport::NcclContext*>(&comm_ctx),
        &buffer,
        &workspace,
        &mapped_host_workspace,
        &scaleout_rank_idx,
        &scaleup_rank_idx,
    };
    launch_kernel(spec, arguments, stream);
}

void launch_combine(const CombineSpec& spec, void* x, float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward, int* channel_linked_list,
                    int num_reduced_tokens,
                    const transport::NcclContext& comm_ctx, void* buffer,
                    void* workspace, int scaleout_rank_idx,
                    int scaleup_rank_idx, cudaStream_t stream) {
    if (spec.num_scaleout_ranks == 1) {
        void* arguments[] = {
            &x,
            &topk_weights,
            &src_metadata,
            &psum_num_recv_tokens_per_scaleup_rank,
            const_cast<transport::NcclContext*>(&comm_ctx),
            &buffer,
            &workspace,
            &scaleup_rank_idx,
            &num_reduced_tokens,
        };
        launch_kernel(spec, arguments, stream);
        return;
    }

    void* arguments[] = {
        &x,
        &topk_weights,
        &src_metadata,
        &psum_num_recv_tokens_per_scaleup_rank,
        &token_metadata_at_forward,
        &channel_linked_list,
        const_cast<transport::NcclContext*>(&comm_ctx),
        &buffer,
        &workspace,
        &scaleout_rank_idx,
        &scaleup_rank_idx,
        &num_reduced_tokens,
    };
    launch_kernel(spec, arguments, stream);
}

}  // namespace mooncake::elastic::jit

#endif  // MOONCAKE_EP_ENABLE_NCCL_JIT
