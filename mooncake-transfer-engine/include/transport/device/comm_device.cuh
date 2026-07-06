// Communication device API — top-level context and routing.
//
// CommCtx bundles all transport state the kernel needs.  The kernel
// constructs one from the raw pointers passed in and calls mc_route_put /
// mc_signal / mc_red_add instead of touching transport internals directly.
#pragma once

#include "transport/device/device_ops.cuh"
#include "transport/device/p2p_device.cuh"
#include "transport/device/ibgda_device.cuh"
#ifdef USE_NCCL_DEVICE
#include "transport/device/nccl_device.cuh"
#endif

namespace mooncake {
namespace device {

// ---------------------------------------------------------------------------
// CommCtx
// ---------------------------------------------------------------------------

struct CommCtx {
    P2PContext p2p;
    IbgdaContext ibgda;
#ifdef USE_NCCL_DEVICE
    NcclDeviceContext nccl;
    ncclWindow_t nccl_window;
    bool use_nccl;
#endif
    int rank;
    int num_ranks;
};

// Construct CommCtx from the raw kernel arguments.
// raddrs/rkeys/qp_devctxs may be nullptr on MUSA (ignored).
__device__ __forceinline__ CommCtx make_comm_ctx(
    void* gdr_buffer, const int32_t* nvlink_available,
    void* const* ipc_peer_ptrs, void* raddrs, void* rkeys, void* qp_devctxs,
    const void* rdma_send_signal_buffer, const void* rdma_recv_signal_buffer,
#ifdef USE_NCCL_DEVICE
    NcclDeviceContext nccl, ncclWindow_t nccl_window,
#endif
    int rank, int num_ranks, int num_qps) {
    CommCtx ctx;
    ctx.rank = rank;
    ctx.num_ranks = num_ranks;

    ctx.p2p.available = nvlink_available;
    ctx.p2p.peer_ptrs = ipc_peer_ptrs;
    ctx.p2p.local_base = gdr_buffer;

#ifdef MOONCAKE_EP_USE_MACA
    ctx.ibgda.qp_devctxs = qp_devctxs;
#else
    ctx.ibgda.qp_devctxs = reinterpret_cast<mlx5gda_qp_devctx*>(qp_devctxs);
#endif
    ctx.ibgda.raddrs = reinterpret_cast<const uint64_t*>(raddrs);
    ctx.ibgda.rkeys = reinterpret_cast<const uint32_t*>(rkeys);
    ctx.ibgda.local_atomic_base = rdma_send_signal_buffer;
    ctx.ibgda.remote_atomic_base = rdma_recv_signal_buffer;

#ifdef USE_NCCL_DEVICE
    ctx.nccl = nccl;
    ctx.nccl_window = nccl_window;
    ctx.use_nccl = nccl_window != nullptr;
#endif

    return ctx;
}

// ---------------------------------------------------------------------------
// Routing helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool mc_comm_p2p_available(const CommCtx& ctx,
                                                      int dst_rank) {
#ifdef USE_NCCL_DEVICE
    if (ctx.use_nccl) return mc_nccl_lsa_available(ctx.nccl, dst_rank);
#endif
    return mc_p2p_available(ctx.p2p, dst_rank);
}

// Translate a local GDR pointer to the peer's mapped VA.
__device__ __forceinline__ void* mc_comm_peer_ptr(const CommCtx& ctx,
                                                  int dst_rank,
                                                  const void* local_ptr) {
#ifdef USE_NCCL_DEVICE
    if (ctx.use_nccl) {
        size_t offset = reinterpret_cast<const char*>(local_ptr) -
                        reinterpret_cast<const char*>(ctx.p2p.local_base);
        return mc_nccl_lsa_ptr(ctx.nccl, ctx.nccl_window, offset, dst_rank);
    }
#endif
    return mc_p2p_peer_ptr(ctx.p2p, dst_rank, local_ptr);
}

__device__ __forceinline__ bool mc_comm_uses_gin(const CommCtx& ctx,
                                                 int peer) {
#ifdef USE_NCCL_DEVICE
    return ctx.use_nccl && peer != ctx.rank &&
           !mc_nccl_lsa_available(ctx.nccl, peer);
#else
    return false;
#endif
}

__device__ __forceinline__ int mc_comm_gin_context(
    const CommCtx& ctx, int channel, int peer) {
#ifdef USE_NCCL_DEVICE
    const unsigned int key =
        static_cast<unsigned int>(channel) * ctx.num_ranks + peer;
    return mc_nccl_gin_context(ctx.nccl, key);
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// mc_route_put
//
// Returns the destination pointer for a warp-cooperative copy:
//   - local rank:  recv_ptr itself (caller does UNROLLED_WARP_COPY)
//   - P2P rank:    peer-mapped recv_ptr (caller does UNROLLED_WARP_COPY)
//   - IBGDA rank:  nullptr (caller must stage data then call mc_rdma_put)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void* mc_route_put(const CommCtx& ctx, int dst_rank,
                                              void* recv_ptr) {
    if (dst_rank == ctx.rank) return recv_ptr;
    if (mc_comm_p2p_available(ctx, dst_rank))
        return mc_comm_peer_ptr(ctx, dst_rank, recv_ptr);
    return nullptr;  // IBGDA path
}

// Issue an IBGDA RDMA WRITE.  Call only when mc_route_put returned nullptr.
// lane_id: only lane 0 issues the WQE.
__device__ __forceinline__ void mc_rdma_put(
    const CommCtx& ctx, int channel, int dst_rank, int qps_per_rank,
    const void* send_ptr,
    void* recv_ptr,  // local VA of the recv slot (for raddr computation)
    uint32_t nbytes, int lane_id) {
    if (lane_id == 0) {
#ifdef USE_NCCL_DEVICE
        if (ctx.use_nccl) {
            const int gin_context = mc_comm_gin_context(ctx, channel, dst_rank);
            const size_t send_offset =
                reinterpret_cast<const char*>(send_ptr) -
                reinterpret_cast<const char*>(ctx.p2p.local_base);
            const size_t recv_offset = reinterpret_cast<const char*>(recv_ptr) -
                                       reinterpret_cast<const char*>(
                                           ctx.p2p.local_base);
            mc_nccl_gin_put(ctx.nccl, gin_context, dst_rank, ctx.nccl_window,
                            recv_offset, ctx.nccl_window, send_offset, nbytes);
            return;
        }
#endif
        uint64_t recv_raddr =
            ctx.ibgda.raddrs[dst_rank] +
            (reinterpret_cast<const char*>(recv_ptr) -
             reinterpret_cast<const char*>(ctx.p2p.local_base));
        mc_ibgda_put(ctx.ibgda, channel, dst_rank, ctx.rank, qps_per_rank,
                     send_ptr, recv_raddr, nbytes);
    }
}

// ---------------------------------------------------------------------------
// mc_signal / mc_red_add
//
// Route a signal (store) or reduction (atomic add) to dst_rank.
// sig_ptr is a local VA within the GDR buffer.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mc_signal(const CommCtx& ctx, int dst_rank,
                                          int channel, int qps_per_rank,
                                          int* sig_ptr, int32_t val,
                                          int staging_index = 0,
                                          uint64_t* completion_ptr = nullptr) {
    if (dst_rank == ctx.rank) {
        mc_st_release(sig_ptr, val);
        return;
    }
    if (mc_comm_p2p_available(ctx, dst_rank)) {
        auto* peer_sig = reinterpret_cast<int*>(
            mc_comm_peer_ptr(ctx, dst_rank, sig_ptr));
        mc_st_release(peer_sig, val);
    } else {
#ifdef USE_NCCL_DEVICE
        if (ctx.use_nccl) {
            // GIN consumes the source asynchronously, so each in-flight
            // expert needs a distinct staging slot.
            auto* local_signal_base = reinterpret_cast<int*>(
                const_cast<void*>(ctx.ibgda.local_atomic_base));
            int* local_sig_ptr = local_signal_base + staging_index;
            mc_st_release(local_sig_ptr, val);

            const int gin_context = mc_comm_gin_context(ctx, channel, dst_rank);
            const size_t local_offset =
                reinterpret_cast<const char*>(local_sig_ptr) -
                reinterpret_cast<const char*>(ctx.p2p.local_base);
            const size_t remote_offset =
                reinterpret_cast<const char*>(sig_ptr) -
                reinterpret_cast<const char*>(ctx.p2p.local_base);
            const size_t completion_offset =
                reinterpret_cast<const char*>(completion_ptr) -
                reinterpret_cast<const char*>(ctx.p2p.local_base);
            mc_nccl_gin_put_va_signal(
                ctx.nccl, gin_context, dst_rank, ctx.nccl_window,
                remote_offset, ctx.nccl_window, local_offset, sizeof(int32_t),
                ctx.nccl_window, completion_offset);
            mc_nccl_gin_flush(ctx.nccl, gin_context);
            return;
        }
#endif
        uint64_t recv_raddr =
            ctx.ibgda.raddrs[dst_rank] +
            (reinterpret_cast<const char*>(sig_ptr) -
             reinterpret_cast<const char*>(ctx.p2p.local_base));
        uint64_t laddr =
            ctx.ibgda.raddrs[ctx.rank] +
            (reinterpret_cast<const char*>(sig_ptr) -
             reinterpret_cast<const char*>(ctx.ibgda.remote_atomic_base)) +
            (reinterpret_cast<const char*>(ctx.ibgda.local_atomic_base) -
             reinterpret_cast<const char*>(ctx.p2p.local_base));
        mc_ibgda_red_add(ctx.ibgda, channel, dst_rank, ctx.rank, qps_per_rank,
                         laddr, recv_raddr, val);
    }
}

__device__ __forceinline__ void mc_red_add(const CommCtx& ctx, int dst_rank,
                                           int channel, int qps_per_rank,
                                           int* sig_ptr, int32_t val,
                                           int staging_index = 0,
                                           uint64_t* completion_ptr = nullptr) {
    mc_signal(ctx, dst_rank, channel, qps_per_rank, sig_ptr, val,
              staging_index, completion_ptr);
}

__device__ __forceinline__ bool mc_completion_ready(
    const CommCtx& ctx, int src_rank, int channel,
    const uint64_t* completion_ptr, uint64_t epoch) {
#ifdef USE_NCCL_DEVICE
    if (mc_comm_uses_gin(ctx, src_rank)) {
        const int gin_context = mc_comm_gin_context(ctx, channel, ctx.rank);
        const size_t completion_offset =
            reinterpret_cast<const char*>(completion_ptr) -
            reinterpret_cast<const char*>(ctx.p2p.local_base);
        return mc_nccl_gin_read_va_signal(ctx.nccl, gin_context,
                                          ctx.nccl_window,
                                          completion_offset) >= epoch;
    }
#endif
    return true;
}

}  // namespace device
}  // namespace mooncake
