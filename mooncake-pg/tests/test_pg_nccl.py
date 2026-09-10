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


def _abort_status_worker(ctx, selected_backend, operation, ready, departed):
    pg.set_gpu_collective_backend(selected_backend)
    pg.set_collective_timeout_us(1_000_000)
    device = ctx.init_group()
    backend = ctx.get_backend()
    assert pg.get_gpu_collective_backend(backend) == selected_backend
    departed_rank = ctx.world_size - 1
    values = [
        torch.ones(1, dtype=torch.int32, device=device)
        for _ in range(2 if selected_backend == "nccl" else 1)
    ]

    initial = torch.tensor([ctx.rank + 1], dtype=torch.int32, device=device)
    completed_work = dist.all_reduce(initial, async_op=True)
    completed_work.wait()
    ctx.synchronize()
    assert pg.get_local_success(completed_work)
    assert initial.item() == ctx.world_size * (ctx.world_size + 1) // 2
    ready.wait(timeout=30.0)

    pending = []
    if ctx.rank != departed_rank:
        for value in values:
            if operation == "barrier":
                pending.append(dist.barrier(async_op=True))
            else:
                pending.append(dist.all_reduce(value, async_op=True))
    # Do not let the peer exit until every survivor has posted its work. This
    # avoids a race where membership changes before the second NCCL launch.
    ready.wait(timeout=30.0)
    if ctx.rank == departed_rank:
        ctx.record_result({"role": "departed"})
        os._exit(0)

    # The parent signals actual process exit, not just intent to exit.
    if not departed.wait(timeout=30.0):
        raise TimeoutError("peer did not exit")
    for work in pending:
        work.wait()
        assert not pg.get_local_success(work), "aborted work reported success"
        expected_hint = [0] * ctx.world_size
        if selected_backend == "transfer_engine":
            expected_hint[departed_rank] = 1
        assert pg.get_failed_ranks_hint(work).tolist() == expected_hint

    wait_until(
        lambda: pg.get_active_ranks(backend).cpu().tolist()
        == [1] * departed_rank + [0],
        timeout_s=30.0,
        description="waiting for departed rank to become inactive",
    )
    assert pg.get_gpu_collective_backend(backend) == "transfer_engine"
    final = torch.ones(1, dtype=torch.int32, device=device)
    final_work = dist.all_reduce(final, async_op=True)
    final_work.wait()
    assert pg.get_local_success(final_work)
    assert final.item() == departed_rank
    dist.destroy_process_group()

    # The outcome belongs to Work, not the current communicator/backend.
    assert pg.get_local_success(completed_work)
    assert pg.get_local_success(final_work)
    for work in pending:
        assert not pg.get_local_success(work)
    ctx.record_result({"role": "survivor"})


def _completed_status_worker(ctx):
    pg.set_gpu_collective_backend("nccl")
    device = ctx.init_group()
    value = torch.tensor([ctx.rank + 1], dtype=torch.int32, device=device)
    output = torch.empty(ctx.world_size, dtype=torch.int32, device=device)
    work = dist.all_gather_into_tensor(output, value, async_op=True)
    work.wait()
    ctx.synchronize()
    # Do not query the work's status or submit another operation before
    # destruction: abort must itself recognize this operation has completed.
    dist.destroy_process_group()
    assert output.tolist() == list(range(1, ctx.world_size + 1))
    assert pg.get_local_success(work)
    assert pg.get_failed_ranks_hint(work).tolist() == [0] * ctx.world_size
    ctx.record_result({"success": True})


def _capture_replay_worker(ctx):
    pg.set_gpu_collective_backend("nccl")
    device = ctx.init_group()
    value = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(value)
    ctx.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work = dist.all_reduce(value, async_op=True)
    for _ in range(2):
        value.fill_(ctx.rank + 1)
        graph.replay()
        ctx.synchronize()
        assert value.item() == ctx.world_size * (ctx.world_size + 1) // 2
    # Retain Work through replay, but synchronize the graph rather than its
    # internal event: host-side captured-Work status queries are not supported.
    # NCCL teardown waits for captured graphs to release their communicator.
    graph.reset()
    del work
    dist.destroy_process_group()
    # Do not replay a graph after its NCCL communicator is destroyed.
    ctx.record_result({"success": True})


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

    def _check_abort_status(self, backend, operation):
        spawn_ctx = mp.get_context("spawn")
        ready = spawn_ctx.Barrier(self.world_size)
        departed = spawn_ctx.Event()
        rows = self.spawn_backend_and_collect(
            _abort_status_worker,
            backend,
            operation,
            ready,
            departed,
            process_exit_events={self.world_size - 1: departed},
        )
        self.assert_all_ok(rows)
        self.assertEqual({row["role"] for row in rows}, {"survivor", "departed"})

    def test_aborted_allreduce_reports_failure(self):
        self._check_abort_status("nccl", "allreduce")

    def test_aborted_barrier_reports_failure(self):
        self._check_abort_status("nccl", "barrier")

    def test_transfer_engine_failure_status_unchanged(self):
        self._check_abort_status("transfer_engine", "allreduce")

    def test_completed_work_survives_communicator_shutdown(self):
        self.assert_all_ok(
            self.spawn_backend_and_collect(_completed_status_worker, world_size=2)
        )

    def test_capture_and_replay_with_tracking(self):
        self.assert_all_ok(
            self.spawn_backend_and_collect(_capture_replay_worker, world_size=2)
        )

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
