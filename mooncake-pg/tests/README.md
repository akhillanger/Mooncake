# PG Tests

## Environment Variables

- `MOONCAKE_PGTEST_DEVICE_FILTERS`
  Comma-separated NIC / IB device names passed to
  `pg.set_device_filter(...)`. Leave unset to use the backend default device
  selection.

- `MOONCAKE_PGTEST_MASTER_ADDR`
  Rendezvous address used for the local test process group. Defaults to
  `127.0.0.1`.

- `MOONCAKE_PGTEST_MASTER_PORT`
  Rendezvous port used for the local test process group. If unset, each test
  allocates a free local port automatically.

## Usage

```bash
# Run all test cases
python -m unittest discover -s mooncake-pg/tests -v

# Run CUDA test cases
python -m unittest discover -s mooncake-pg/tests -k CUDA -v

# Run CPU-only test cases
python -m unittest discover -s mooncake-pg/tests -k CPU -v
```

## NCCL failure handling

With a `USE_NCCL_PG=ON` build and at least three CUDA devices:

```bash
MOONCAKE_PGTEST_NCCL=1 python -m pytest -v mooncake-pg/tests/test_pg_nccl.py
```

The suite tests peer exit, successful TE fallback, completed-work status,
graph replay, and timeouts while a peer stays alive but does not participate.
Fault-injection tests compile a test-only `LD_PRELOAD` shim to inject an NCCL
asynchronous error or a host enqueue that remains `ncclInProgress`. Set `NCCL_ROOT` and
`CUDA_HOME` to the matching header installations if they are not on the
compiler's include path. `CXX` selects the compiler. No injection hooks are
built into the production backend.

The single-rank-error test keeps every process alive, masks native NCCL error
detection on recipients, and uses a long local timeout. It verifies that the
Coordinator notification aborts both a waiting rank and an idle rank, without
changing membership or choosing TE. It also checks that an unrelated group and
a subsequently created group remain usable.

The explicit-recovery tests reuse this injected failure and synchronize all
active ranks with `sync_after_failure`, delaying the idle rank to verify that
no peer switches to TE early. They run an Int16 TE collective before failure
and several TE all-reduces after recovery, checking that the existing TE task
sequence, membership, earlier Work status, and unrelated NCCL group survive.
Another test withholds a rank past the 20-second barrier deadline, checks that
rejection does not enable TE, then retries successfully with all ranks.

The hardware-free coordinator tests cover retries, stale sessions and tokens,
nonmember reports, unchanged views/health, and group destruction. With
`BUILD_UNIT_TESTS=ON`, run:

```bash
cmake --build build --target pg_nccl_failure_control_test
ctest --test-dir build -R '^pg_nccl_failure_control_test$' --output-on-failure
```

These tests also work with `USE_NCCL_PG=OFF`; they do not initialize CUDA or NCCL.
They also verify that delayed failure cutoffs update retained Work status and
that out-of-order reports cannot erase an earlier failed sequence.
Recovery coverage includes all-rank acknowledgement, duplicate requests, lost
commit replies, missing-rank deadlines, mismatched TE counters, stale
sessions/generations/views, and membership changes during recovery.
