import ctypes
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mooncake import pg

from pg_test_utils import (
    MooncakePGCUDABackendTestCase,
    MooncakePGWorkerContext,
    temporary_env,
    wait_until,
)


@contextmanager
def _preload_nccl_fault():
    with tempfile.TemporaryDirectory(prefix="mooncake-pg-nccl-fault-") as tmp:
        library = Path(tmp) / "libnccl_fault.so"
        command = [
            os.environ.get("CXX", "c++"),
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
        ]
        for variable in ("NCCL_ROOT", "CUDA_HOME"):
            if os.getenv(variable):
                command.append(f"-I{os.environ[variable]}/include")
        command += [
            str(Path(__file__).with_name("nccl_fault_preload.cpp")),
            "-ldl",
            "-o",
            str(library),
        ]
        # An explicitly enabled NCCL test run must not silently skip broken
        # injection. Supply NCCL_ROOT/CUDA_HOME if headers are not on the path.
        subprocess.run(command, check=True, capture_output=True, text=True)
        preload = f"{library} {os.environ.get('LD_PRELOAD', '')}".strip()
        with temporary_env({"LD_PRELOAD": preload}):
            yield str(library)


def _watchdog_worker(ctx, fault, library_path, ready):
    pg.set_gpu_collective_backend("nccl")
    pg.set_collective_timeout_us(60_000_000)
    device = ctx.init_group()
    backend = ctx.get_backend()
    value = torch.ones(1, dtype=torch.int32, device=device)
    completed = dist.all_reduce(value, async_op=True)
    completed.wait()
    ctx.synchronize()
    assert pg.get_local_success(completed)

    injected = None
    if library_path:
        injected = ctypes.CDLL(library_path)
        injected.mooncakePgTestNcclSetFault.argtypes = [ctypes.c_int]
        injected.mooncakePgTestNcclSetFault.restype = None
        injected.mooncakePgTestNcclFaultCount.argtypes = []
        injected.mooncakePgTestNcclFaultCount.restype = ctypes.c_uint64

    # Change the existing setting after initialization. Async-error injection
    # keeps a long deadline to prove it is the error, not the timeout, detected.
    if fault != "async_error":
        pg.set_collective_timeout_us(1_000_000)
    ready.wait(timeout=30.0)
    elapsed = None
    if ctx.rank != ctx.world_size - 1:
        started = time.monotonic()
        if fault == "enqueue_stall":
            injected.mooncakePgTestNcclSetFault(2)
            try:
                dist.all_reduce(value, async_op=True)
            except RuntimeError as error:
                assert "NCCL collective timed out" in str(error)
            else:
                raise AssertionError("unfinished NCCL enqueue returned Work")
        else:
            work = dist.all_reduce(value, async_op=True)
            if fault == "async_error":
                # Arm only after both submitters have returned Work, so an
                # early abort cannot turn the other rank's launch into an
                # immediate error instead of exercising asynchronous status.
                ready.wait(timeout=30.0)
                injected.mooncakePgTestNcclSetFault(1)
            work.wait()
            assert not pg.get_local_success(work)
            assert pg.get_failed_ranks_hint(work).tolist() == [0] * ctx.world_size
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"NCCL failure detection took {elapsed:.2f}s"
        if fault != "async_error":
            assert elapsed >= 0.75, f"timeout fired too early: {elapsed:.2f}s"
        if injected:
            assert injected.mooncakePgTestNcclFaultCount() > 0
            injected.mooncakePgTestNcclSetFault(0)
        assert pg.get_local_success(completed)
        # No rank died and no Coordinator view changed. Do not switch this
        # rank alone to TE: further NCCL submissions must fail promptly.
        assert all(pg.get_peer_state(backend, list(range(ctx.world_size))))
        assert pg.get_active_ranks(backend).cpu().tolist() == [1] * ctx.world_size
        assert pg.get_gpu_collective_backend(backend) == "nccl"
        try:
            dist.all_reduce(value, async_op=True)
        except RuntimeError as error:
            assert "NCCL" in str(error)
        else:
            raise AssertionError("failed NCCL communicator accepted new work")

    # The withheld participant remains alive, with its heartbeat thread
    # running, until the other ranks have observed their failures.
    if ctx.rank == ctx.world_size - 1 and fault == "async_error":
        ready.wait(timeout=30.0)
    ready.wait(timeout=30.0)
    dist.destroy_process_group()
    ctx.record_result({"elapsed": elapsed})


def _group_failure_worker(ctx, library_path, ready, recovery=None, returned=None):
    pg.set_gpu_collective_backend("nccl")
    # Other ranks must fail by notification, not their own timeout.
    pg.set_collective_timeout_us(60_000_000)
    device = ctx.init_group()
    healthy = torch.ones(1, dtype=torch.int32, device=device)
    healthy_work = dist.all_reduce(healthy, async_op=True)
    healthy_work.wait()
    ctx.synchronize()

    affected = dist.new_group(backend=ctx.backend_name)
    backend = ctx.get_backend(affected)
    # Int16 uses TE even while NCCL is selected. Leave its double-buffer
    # sequence nonzero and verify recovery preserves that sequence.
    if recovery:
        te_value = torch.ones(1, dtype=torch.int16, device=device)
        te_completed = dist.all_reduce(te_value, group=affected, async_op=True)
        te_completed.wait()
        ctx.synchronize()
        assert pg.get_local_success(te_completed)
        assert te_value.item() == ctx.world_size
    value = torch.ones(1, dtype=torch.int32, device=device)
    completed = dist.all_reduce(value, group=affected, async_op=True)
    completed.wait()
    ctx.synchronize()
    injected = ctypes.CDLL(library_path)
    injected.mooncakePgTestNcclSetFault.argtypes = [ctypes.c_int]
    injected.mooncakePgTestNcclSetFault.restype = None
    injected.mooncakePgTestNcclAbortCount.argtypes = []
    injected.mooncakePgTestNcclAbortCount.restype = ctypes.c_uint64
    injected.mooncakePgTestNcclFaultCount.argtypes = []
    injected.mooncakePgTestNcclFaultCount.restype = ctypes.c_uint64
    before = injected.mooncakePgTestNcclAbortCount()

    ready.wait(timeout=30.0)
    pending = None
    if ctx.rank != ctx.world_size - 1:
        pending = dist.all_reduce(value, group=affected, async_op=True)
    # Mask native NCCL errors only after host enqueue has completed, so this
    # tests Mooncake's control plane without altering enqueue progress.
    if ctx.rank != 0:
        injected.mooncakePgTestNcclSetFault(3)
    ready.wait(timeout=30.0)
    if pending is not None:
        assert (
            not pending.is_completed()
        ), "test collective was not pending at injection"
    ready.wait(timeout=30.0)
    started = time.monotonic()
    if ctx.rank == 0:
        injected.mooncakePgTestNcclSetFault(1)
    wait_until(
        lambda: injected.mooncakePgTestNcclAbortCount() > before,
        timeout_s=10.0,
        description="waiting for group-wide NCCL abort (including idle rank)",
    )
    elapsed = time.monotonic() - started
    if pending is not None:
        pending.wait()
        assert not pg.get_local_success(pending), "pending work incorrectly succeeded"
        assert (
            pg.get_failed_ranks_hint(pending).tolist() == [0] * ctx.world_size
        ), "fabricated peer hint"
    assert injected.mooncakePgTestNcclFaultCount() > 0, "injection not exercised"
    assert pg.get_local_success(completed), "completed affected-group work lost success"
    assert pg.get_local_success(healthy_work), "healthy-group work lost success"
    assert all(
        pg.get_peer_state(backend, list(range(ctx.world_size)))
    ), "peer state changed"
    assert (
        pg.get_active_ranks(backend).cpu().tolist() == [1] * ctx.world_size
    ), "membership changed"
    assert pg.get_gpu_collective_backend(backend) == "nccl", "backend switched to TE"
    try:
        dist.all_reduce(value, group=affected, async_op=True)
    except RuntimeError as error:
        assert "NCCL" in str(error)
        if ctx.rank != 0:
            assert "reported by the coordinator" in str(error), str(error)
    else:
        raise AssertionError("failed group accepted new NCCL work")
    injected.mooncakePgTestNcclSetFault(0)
    ready.wait(timeout=30.0)

    # A failed subgroup must not poison the world group using the same Agents.
    healthy.fill_(1)
    dist.all_reduce(healthy)
    assert healthy.item() == ctx.world_size
    if recovery:
        epoch = pg.get_current_epoch(backend)
        ready.wait(timeout=30.0)
        if recovery == "timeout":
            if ctx.rank == ctx.world_size - 1:
                # The barrier has a 20-second coordinator deadline.
                time.sleep(22.0)
            else:
                response = pg.sync_after_failure(backend)
                assert response.status == pg.SyncAfterFailureStatus.Rejected
                assert "timed out" in response.reject_reason
                assert pg.get_gpu_collective_backend(backend) == "nccl"
                # Even unsupported NCCL types must not bypass a failed barrier.
                try:
                    dist.all_reduce(te_value, group=affected, async_op=True)
                except RuntimeError as error:
                    assert "recovery is pending" in str(error)
                else:
                    raise AssertionError("failed recovery allowed a TE submission")
            ready.wait(timeout=30.0)

        started_recovery = time.monotonic()
        if ctx.rank == ctx.world_size - 1:
            time.sleep(1.0)
            assert not any(returned), "recovery returned before idle rank joined"
            assert pg.get_gpu_collective_backend(backend) == "nccl"
        response = pg.sync_after_failure(backend)
        recovery_elapsed = time.monotonic() - started_recovery
        assert (
            response.status == pg.SyncAfterFailureStatus.Reconciled
        ), response.reject_reason
        returned[ctx.rank] = 1
        assert pg.get_current_epoch(backend) == epoch, "recovery changed membership"
        assert pg.get_gpu_collective_backend(backend) == "transfer_engine"
        if ctx.rank != ctx.world_size - 1:
            assert recovery_elapsed >= 0.75, "recovery skipped the idle rank"
        for iteration in range(3):
            value.fill_(ctx.rank + 1 + iteration)
            work = dist.all_reduce(value, group=affected, async_op=True)
            work.wait()
            assert pg.get_local_success(work)
            assert value.item() == (
                ctx.world_size * (ctx.world_size + 1) // 2 + ctx.world_size * iteration
            )
        assert pg.get_local_success(te_completed)
        assert (
            pg.sync_after_failure(backend).status == pg.SyncAfterFailureStatus.NoPending
        )
        assert pg.get_gpu_collective_backend(ctx.get_backend()) == "nccl"
    dist.destroy_process_group(affected)
    ready.wait(timeout=30.0)
    replacement = dist.new_group(backend=ctx.backend_name)
    # Allow repeated heartbeats/late notifications from the old group to arrive.
    time.sleep(2.1)
    value.fill_(1)
    dist.all_reduce(value, group=replacement)
    assert value.item() == ctx.world_size
    assert pg.get_gpu_collective_backend(ctx.get_backend(replacement)) == "nccl"
    dist.destroy_process_group(replacement)
    dist.destroy_process_group()
    assert pg.get_local_success(completed)
    if pending is not None:
        assert not pg.get_local_success(pending)
    ctx.record_result({"elapsed": elapsed})


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
        timeout_s=45.0,
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
    pg.set_collective_timeout_us(200_000)
    time.sleep(0.5)  # An idle captured graph is not an outstanding execution.
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

    def _check_watchdog(self, fault, library_path=None):
        ready = mp.get_context("spawn").Barrier(self.world_size)
        self.assert_all_ok(
            self.spawn_backend_and_collect(_watchdog_worker, fault, library_path, ready)
        )

    def test_timeout_with_live_nonparticipating_peer(self):
        self._check_watchdog("timeout")

    def test_asynchronous_nccl_error(self):
        with _preload_nccl_fault() as library:
            self._check_watchdog("async_error", library)

    def test_timeout_during_nonblocking_enqueue(self):
        with _preload_nccl_fault() as library:
            self._check_watchdog("enqueue_stall", library)

    def test_one_rank_error_aborts_group_including_idle_rank(self):
        ready = mp.get_context("spawn").Barrier(self.world_size)
        with _preload_nccl_fault() as library:
            rows = self.spawn_backend_and_collect(_group_failure_worker, library, ready)
        self.assert_all_ok(rows)

    def _check_recovery(self, mode):
        spawn_ctx = mp.get_context("spawn")
        ready = spawn_ctx.Barrier(self.world_size)
        returned = spawn_ctx.Array("i", self.world_size)
        with _preload_nccl_fault() as library:
            rows = self.spawn_backend_and_collect(
                _group_failure_worker, library, ready, mode, returned
            )
        self.assert_all_ok(rows)

    def test_explicit_recovery_waits_for_idle_rank_then_uses_te(self):
        self._check_recovery("recover")

    def test_explicit_recovery_timeout_keeps_nccl_poisoned_then_retries(self):
        self._check_recovery("timeout")

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
