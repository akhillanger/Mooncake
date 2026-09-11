# mooncake-pg pgbench

Python benchmark harness for mooncake-pg with nccl-tests style CLI and output.

## Quick Start (CPU)
```
PYTHONPATH=mooncake-pg \
python mooncake-pg/benchmark/pgbench.py \
  --collective all_reduce --backend mooncake-cpu --device cpu -g 2 -b 256 -e 256 -n 2 -w 1 -c 1
```

## Quick Start (CUDA)
```
PYTHONPATH=mooncake-pg \
python mooncake-pg/benchmark/pgbench.py \
  --collective all_reduce --backend mooncake --device cuda -g 8 -b 8 -e 128M -f 2
```

Set `MOONCAKE_PGTEST_DEVICE_FILTERS=mlx5_1,mlx5_2,...` to explicitly set HCA whitelist.

Build with `-DUSE_NCCL_PG=ON` to enable the NCCL collective data plane. Such a
build selects NCCL automatically for founding CUDA groups. Use
`--mooncake-gpu-collective-backend=transfer_engine` or `nccl` to compare the
two data planes explicitly:

```bash
python mooncake-pg/benchmark/pgbench.py --collective all_reduce \
  --backend mooncake --mooncake-gpu-collective-backend nccl -g 8
```

## Notes
- Use `-g/--ngpus` for the built-in single-node launcher. Multi-node runs can
  use `torchrun`; set `MOONCAKE_PG_HOST_IP` to a routable address on each node.
- Dtypes align with nccl-tests; use `-d all` to sweep supported types.
- Sizes accept suffixes like `K`, `M`, `G` (e.g., `-b 8 -e 128M -f 2`).
- Size/count columns are always human-readable.
- Unsupported collectives are ignored (not implemented).
- JSON output (`-J`) is not supported yet.
- Supported backends: `mooncake`, `mooncake-cpu`, `nccl`, `gloo`.
- NCCL changes only GPU collectives; Mooncake's control plane, fault detection,
  and P2P path remain active. Because an NCCL communicator has fixed membership,
  the first committed membership change aborts it and that process group uses
  the existing TE collective path for the rest of its lifetime.

## P2P Regular-k Bench

Use `p2p_regular_k_bench.py` for pure P2P traffic tests with configurable
regular-k topology (each rank sends to `+1..+k` peers and receives from
`-1..-k` peers modulo world size).

### Quick Start (CPU)
```
PYTHONPATH=mooncake-pg \
python mooncake-pg/benchmark/p2p_regular_k_bench.py \
  --backend mooncake-cpu --device cpu -g 8 -k 2 --tensor-bytes 4M -w 5 -n 30
```

### Quick Start (CUDA)
```
PYTHONPATH=mooncake-pg \
python mooncake-pg/benchmark/p2p_regular_k_bench.py \
  --backend mooncake --device cuda -g 8 -k 3 --tensor-bytes 8M -w 10 -n 50
```

### Output
- `lat_mean_us`, `lat_p50_us`, `lat_p95_us`, `lat_p99_us`
- `rank_bidir_GBps`: per-rank bidirectional bandwidth
- `cluster_bidir_GBps`: aggregate bidirectional bandwidth over all ranks
