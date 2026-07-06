#!/usr/bin/env python3
import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from mooncake.mooncake_ep_buffer import Buffer


def env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return int(value)
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Mooncake EP payload transports."
    )
    parser.add_argument(
        "--rank", type=int, default=env_int("RANK", "SLURM_PROCID", default=0)
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=env_int("WORLD_SIZE", "SLURM_NTASKS", default=1),
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=env_int("LOCAL_RANK", "SLURM_LOCALID", default=0),
    )
    parser.add_argument(
        "--master-addr", default=os.getenv("MASTER_ADDR", "127.0.0.1")
    )
    parser.add_argument(
        "--master-port", type=int, default=env_int("MASTER_PORT", default=29500)
    )
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def dequantize_fp8(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    hidden = values.shape[-1]
    value_blocks = values.reshape(-1, hidden // 128, 128).float()
    scale_blocks = scales.reshape(-1, hidden // 128, 1).float()
    return (value_blocks * scale_blocks).reshape(values.shape).to(
        torch.bfloat16
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("world size must be positive")
    if args.tokens <= 0 or args.hidden <= 0 or args.experts <= 0:
        raise ValueError("tokens, hidden, and experts must be positive")
    if args.topk <= 0 or args.topk > args.experts:
        raise ValueError("topk must be in [1, experts]")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if args.experts % args.world_size:
        raise ValueError("experts must be divisible by world size")
    if args.hidden % 128:
        raise ValueError("hidden must be divisible by 128")

    visible_devices = torch.cuda.device_count()
    if visible_devices <= 0:
        raise RuntimeError("no CUDA devices are visible")
    if visible_devices == 1:
        args.local_rank = 0
    elif args.local_rank >= visible_devices:
        raise ValueError("local rank exceeds the visible CUDA device count")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    dist.init_process_group(
        backend=args.backend,
        init_method=init_method,
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=5),
        device_id=device if args.backend == "nccl" else None,
    )
    group = dist.group.WORLD

    torch.manual_seed(args.seed + args.rank)
    x = torch.randn(
        (args.tokens, args.hidden), dtype=torch.bfloat16, device=device
    )
    scores = torch.randn(
        (args.tokens, args.experts), dtype=torch.float32, device=device
    )
    topk_idx = torch.topk(scores, args.topk, dim=-1).indices
    topk_weights = torch.softmax(
        torch.randn(
            (args.tokens, args.topk), dtype=torch.float32, device=device
        ),
        dim=-1,
    )
    active_ranks = torch.ones(
        (args.world_size,), dtype=torch.int32, device=device
    )
    combined_out = torch.empty_like(x)
    check_x_by_rank = None
    check_topk_by_rank = None
    if args.check:
        gather_device = (
            torch.device("cpu") if args.backend == "gloo" else device
        )
        gather_x = x.to(gather_device)
        gather_topk = topk_idx.to(gather_device)
        x_by_rank = [
            torch.empty_like(gather_x) for _ in range(args.world_size)
        ]
        topk_by_rank = [
            torch.empty_like(gather_topk) for _ in range(args.world_size)
        ]
        dist.all_gather(x_by_rank, gather_x, group=group)
        dist.all_gather(topk_by_rank, gather_topk, group=group)
        check_x_by_rank = torch.stack(x_by_rank).to(device)
        check_topk_by_rank = torch.stack(topk_by_rank).to(device)

    buffer_bytes = Buffer.get_ep_buffer_size_hint(
        args.tokens, args.hidden, args.world_size, args.experts
    )
    buffer = Buffer(group, buffer_bytes)

    def validate_dispatch_result(recv_value, handle) -> None:
        assert check_x_by_rank is not None
        assert check_topk_by_rank is not None
        recv_bf16 = (
            dequantize_fp8(recv_value[0], recv_value[1])
            if args.fp8
            else recv_value
        )
        src_info, layout_range = handle[0], handle[1]
        src_info_cpu = src_info.cpu()
        layout_cpu = layout_range.cpu()
        local_experts = args.experts // args.world_size
        bad_by_source = [0] * args.world_size
        details = []

        for local_expert in range(local_experts):
            global_expert = args.rank * local_experts + local_expert
            for source_rank in range(args.world_size):
                expected_ids = torch.nonzero(
                    (check_topk_by_rank[source_rank] == global_expert).any(
                        dim=1
                    ),
                    as_tuple=False,
                ).flatten()
                packed = int(layout_cpu[local_expert, source_rank].item())
                packed &= (1 << 64) - 1
                count = packed & 0xFFFFFFFF
                begin = (packed >> 32) & 0xFFFFFFFF
                actual_ids = src_info_cpu[
                    local_expert, begin : begin + count
                ].to(torch.int64)

                source_errors = 0
                if count != expected_ids.numel():
                    source_errors += abs(count - expected_ids.numel())
                elif not torch.equal(
                    torch.sort(actual_ids).values,
                    expected_ids.cpu(),
                ):
                    source_errors += count

                if count and torch.all(
                    (actual_ids >= 0) & (actual_ids < args.tokens)
                ):
                    expected_rows = check_x_by_rank[source_rank].index_select(
                        0, actual_ids.to(device)
                    )
                    actual_rows = recv_bf16[
                        local_expert, begin : begin + count
                    ]
                    if args.fp8:
                        row_bad = ~torch.isclose(
                            actual_rows.float(),
                            expected_rows.float(),
                            rtol=0.15,
                            atol=5e-3,
                        ).all(dim=1)
                    else:
                        row_bad = (
                            (
                                actual_rows.float()
                                - expected_rows.float()
                            )
                            .abs()
                            .amax(dim=1)
                            > 1e-3
                        )
                    source_errors += int(row_bad.sum().item())

                if source_errors:
                    bad_by_source[source_rank] += source_errors
                    if len(details) < 12:
                        details.append(
                            (
                                local_expert,
                                source_rank,
                                count,
                                int(expected_ids.numel()),
                                source_errors,
                            )
                        )

        if any(bad_by_source):
            print(
                f"DISPATCH_DIAGNOSTIC rank={args.rank} "
                f"bad_by_source={bad_by_source} details={details}",
                flush=True,
            )
            raise RuntimeError("Mooncake EP dispatch validation failed")

    def run_iteration(
        check_dispatch: bool = False,
    ) -> tuple[torch.Tensor, float, float]:
        torch.cuda.synchronize()
        dispatch_start = time.perf_counter()
        recv_x, _, handle, event, hook = buffer.dispatch(
            x,
            topk_idx,
            active_ranks,
            num_max_dispatch_tokens_per_rank=args.tokens,
            num_experts=args.experts,
            timeout_us=-1,
            use_fp8=args.fp8,
            async_finish=False,
            return_recv_hook=False,
        )
        if hook is not None:
            hook()
        if event.event is not None:
            event.current_stream_wait()
        torch.cuda.synchronize()
        dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0
        if check_dispatch:
            validate_dispatch_result(recv_x, handle)

        if args.fp8:
            expert_out = dequantize_fp8(recv_x[0], recv_x[1])
            torch.cuda.synchronize()
        else:
            expert_out = recv_x

        combine_start = time.perf_counter()
        combined_x, event, hook = buffer.combine(
            expert_out,
            topk_idx,
            topk_weights,
            active_ranks,
            timeout_us=-1,
            handle=handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
            out=combined_out,
        )
        if hook is not None:
            hook()
        if event.event is not None:
            event.current_stream_wait()
        torch.cuda.synchronize()
        combine_ms = (time.perf_counter() - combine_start) * 1000.0
        return combined_x, dispatch_ms, combine_ms

    dist.barrier(group)
    result, _, _ = run_iteration(check_dispatch=args.check)
    if args.check:
        try:
            torch.testing.assert_close(
                result,
                x,
                rtol=0.15 if args.fp8 else 5e-2,
                atol=5e-3 if args.fp8 else 1e-3,
            )
        except AssertionError:
            token_error = (result.float() - x.float()).abs().amax(dim=1)
            bad_tokens = torch.nonzero(
                token_error > (5e-3 if args.fp8 else 1e-3),
                as_tuple=False,
            ).flatten()
            owner = topk_idx // (args.experts // args.world_size)
            owner_counts = torch.bincount(
                owner[bad_tokens].flatten(), minlength=args.world_size
            )
            print(
                f"CHECK_DIAGNOSTIC rank={args.rank} "
                f"bad_tokens={bad_tokens.numel()} "
                f"owners={owner_counts.cpu().tolist()} "
                f"first_tokens={bad_tokens[:16].cpu().tolist()}",
                flush=True,
            )
            raise
    dist.barrier(group)

    for _ in range(args.warmup):
        run_iteration()
    dist.barrier(group)

    local_rows = []
    last_result = None
    for _ in range(args.iterations):
        last_result, dispatch_ms, combine_ms = run_iteration()
        local_rows.append(
            [dispatch_ms, combine_ms, dispatch_ms + combine_ms]
        )
    if args.check:
        assert last_result is not None
        torch.testing.assert_close(
            last_result,
            x,
            rtol=0.15 if args.fp8 else 5e-2,
            atol=5e-3 if args.fp8 else 1e-3,
        )

    reduction_device = (
        torch.device("cpu") if args.backend == "gloo" else device
    )
    timings = torch.tensor(local_rows, dtype=torch.float32, device=reduction_device)
    dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)

    if args.rank == 0:
        dispatch_values = timings[:, 0].cpu().tolist()
        combine_values = timings[:, 1].cpu().tolist()
        total_values = timings[:, 2].cpu().tolist()
        dispatch_stats = summarize(dispatch_values)
        combine_stats = summarize(combine_values)
        total_stats = summarize(total_values)

        routes = (
            args.tokens * args.hidden * args.topk * args.world_size
        )
        dispatch_bytes = routes * (
            1.0 + (4.0 / 128.0) if args.fp8 else 2.0
        )
        combine_bytes = routes * 2.0
        dispatch_gbps = dispatch_bytes / (
            dispatch_stats["p50_ms"] * 1.0e6
        )
        combine_gbps = combine_bytes / (
            combine_stats["p50_ms"] * 1.0e6
        )

        properties = getattr(buffer, "_nccl_properties", None)
        report = {
            "hostname": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "transport": os.getenv(
                "MOONCAKE_EP_DEVICE_TRANSPORT", "auto"
            ),
            "control_plane": args.backend,
            "timing_scope": "host_wall_synchronized",
            "nccl_mnnvl_enable": (
                os.getenv("NCCL_MNNVL_ENABLE", "default") if properties else None
            ),
            "nccl_gin_connection": (
                os.getenv("MOONCAKE_EP_NCCL_GIN_CONNECTION", "full")
                if properties
                else None
            ),
            "world_size": args.world_size,
            "tokens_per_rank": args.tokens,
            "hidden": args.hidden,
            "experts": args.experts,
            "topk": args.topk,
            "fp8_dispatch": args.fp8,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "nccl_properties": properties,
            "dispatch": dispatch_stats,
            "combine": combine_stats,
            "dispatch_plus_combine": total_stats,
            "logical_dispatch_gbps_p50": dispatch_gbps,
            "logical_combine_gbps_p50": combine_gbps,
        }
        print(json.dumps(report, sort_keys=True))
        print(
            "RESULT "
            f"transport={report['transport']} ranks={args.world_size} "
            f"tokens={args.tokens} hidden={args.hidden} "
            f"experts={args.experts} topk={args.topk} fp8={args.fp8} "
            f"dispatch_p50={dispatch_stats['p50_ms']:.3f}ms "
            f"combine_p50={combine_stats['p50_ms']:.3f}ms "
            f"total_p50={total_stats['p50_ms']:.3f}ms "
            f"dispatch_p95={dispatch_stats['p95_ms']:.3f}ms "
            f"combine_p95={combine_stats['p95_ms']:.3f}ms"
        )

    dist.barrier(group)
    del buffer
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
