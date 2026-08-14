import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mooncake import pg

from pg_test_utils import (
    MooncakePGCUDABackendTestCase,
    MooncakePGWorkerContext,
    wait_until,
)


def _extension_fallback_worker(
    ctx: MooncakePGWorkerContext,
    extend_event: mp.Event,
) -> None:
    initial_world_size = ctx.world_size - 1
    extension_rank = ctx.world_size - 1

    if ctx.proc_rank < initial_world_size:
        device = ctx.init_group(
            world_size=initial_world_size,
            max_group_size=ctx.world_size,
        )
        backend = ctx.get_backend()
        initial_backend = pg.get_gpu_collective_backend(backend)

        initial = torch.tensor([ctx.proc_rank + 1], dtype=torch.int32, device=device)
        dist.all_reduce(initial)

        if ctx.proc_rank == 0:
            extend_event.set()
        wait_until(
            lambda: pg.get_peer_state(backend, [extension_rank])[0],
            timeout_s=10.0,
            poll_interval_s=0.01,
            description=f"rank {ctx.proc_rank} waiting for extension rank",
        )
        response = pg.recover_ranks(backend, [extension_rank])
        if response.status != pg.ProposalStatus.Applied:
            raise AssertionError(
                f"recover_ranks failed with {response.status}: "
                f"{response.reject_reason}"
            )

        final = torch.tensor([ctx.proc_rank + 1], dtype=torch.int32, device=device)
        dist.all_reduce(final)
        ctx.record_result(
            {
                "role": "founder",
                "initial_backend": initial_backend,
                "final_backend": pg.get_gpu_collective_backend(backend),
                "initial_sum": int(initial.cpu().item()),
                "final_sum": int(final.cpu().item()),
            }
        )
        return

    if not extend_event.wait(timeout=30.0):
        raise TimeoutError("timed out waiting for founding group")
    device = ctx.init_group(
        rank=extension_rank,
        world_size=ctx.world_size,
        max_group_size=ctx.world_size,
        is_extension=True,
    )
    backend = ctx.get_backend()
    initial_backend = pg.get_gpu_collective_backend(backend)
    pg.join_group(backend)

    final = torch.tensor([extension_rank + 1], dtype=torch.int32, device=device)
    dist.all_reduce(final)
    ctx.record_result(
        {
            "role": "extension",
            "initial_backend": initial_backend,
            "final_backend": pg.get_gpu_collective_backend(backend),
            "final_sum": int(final.cpu().item()),
        }
    )


@unittest.skipUnless(
    os.getenv("MOONCAKE_PGTEST_NCCL") == "1",
    "requires a USE_NCCL_PG build",
)
class TestMooncakePGNccl(MooncakePGCUDABackendTestCase):
    world_size = 3
    spawn_timeout_s = 120.0

    def test_extension_falls_back_to_transfer_engine(self) -> None:
        spawn_ctx = mp.get_context("spawn")
        extend_event = spawn_ctx.Event()
        rows = self.spawn_backend_and_collect(
            _extension_fallback_worker,
            extend_event,
        )
        self.assert_all_ok(rows)

        expected_initial_sum = sum(range(1, self.world_size))
        expected_final_sum = sum(range(1, self.world_size + 1))
        for row in rows:
            self.assertEqual(row["final_backend"], "transfer_engine")
            self.assertEqual(row["final_sum"], expected_final_sum)
            if row["role"] == "founder":
                self.assertEqual(row["initial_backend"], "nccl")
                self.assertEqual(row["initial_sum"], expected_initial_sum)
            else:
                self.assertEqual(row["initial_backend"], "transfer_engine")


if __name__ == "__main__":
    unittest.main()
