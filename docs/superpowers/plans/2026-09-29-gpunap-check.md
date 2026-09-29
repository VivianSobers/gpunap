# gpunap check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gpunap check`, an installable, dependency-free suite that reproduces every verified FINDINGS result on the user's own driver.

**Architecture:** A ctypes driver wrapper, a time-limited worker process for driver calls, target processes driven over stdin and stdout, one module per check with a pure verdict function, and a runner that owns pre-flight, cleanup and reporting. See the spec.

**Tech Stack:** Python 3.9+ standard library, ctypes over `libcuda.so.1`, `nvidia-smi`, optional PyTorch, pytest for unit tests, setuptools for packaging.

**Spec:** `docs/superpowers/specs/2026-09-29-gpunap-check-design.md`

## Global Constraints

- Commit messages are `<type>: <4-6 word description>`, the user as sole author, no trailers, no body. One logical change per commit.
- No runtime dependencies outside the standard library. PyTorch is imported only inside PyTorch targets and checks.
- Python 3.9 compatible: no `match`, no `tomllib`, no `X | Y` types outside `from __future__ import annotations`.
- Never record hostnames, IPs, account names or home paths in results or committed files. Machines are labelled with `--label`.
- Every committed number comes from a run that executed.
- Unit tests must pass on a machine without an NVIDIA driver.

## Review Focus

1. A target that dies before printing READY (CUDA init failure, out of memory next to a co-tenant): the check returns `invalid` with the target's output, and cleanup still runs. Tested per check through the shared `start_target` helper (Task 7).
2. A driver call that hangs (stopped process): the worker is killed at its limit and the result says `hung`. Tested in Task 3.
3. A check that raises: the verdict is `error`, cleanup still resumes and kills its targets, and later checks still run. Tested in Task 9.
4. A PID reused by an unrelated process after a target exits: never signalled. Tested in Task 6.
5. Ctrl-C mid-run: cleanup runs, a partial report is written, exit code 130. Tested in Task 9.

---

Each task follows the same loop: write the tests named in the task, watch them fail, implement, watch them pass, run the whole unit suite, commit with the message given.

### Task 1: Package skeleton

Files: `pyproject.toml`, `src/gpunap/__init__.py`, `tests/test_version.py`, `LICENSE`, `.gitignore`.
- setuptools build backend, `src/` layout, name `gpunap`, version `0.1.0`, `requires-python >=3.9`, no dependencies, console script `gpunap = gpunap.cli:main`, pytest config (`testpaths = ["tests"]`).
- Test: `gpunap.__version__ == "0.1.0"`.
- LICENSE: Apache-2.0, as PLAN.md section 10 lays out.
- Commits: `build: add package skeleton` (pyproject, init, test, gitignore); `docs: add Apache 2.0 license`.

### Task 2: driver.py

Interfaces produced: `Call(op, result, code, start, end)` with `.ok` and `.seconds`; `Driver(lib=None)` with `available() -> bool`, `init() -> Call`, `version() -> int`, `lock(pid, timeout_ms=0)`, `checkpoint(pid)`, `restore(pid)`, `unlock(pid)`, `state(pid) -> Call` (result is `running|locked|checkpointed|failed` or an error name), `pause(pid, timeout_ms=0) -> list[Call]` (unlock after a failed checkpoint), `resume(pid) -> list[Call]`; `LockArgs` ctypes struct.
Tests with a fake library object: error names come from `cuGetErrorName`; unknown codes become `rc<N>`; state codes map to names; a failed checkpoint in `pause` is followed by an unlock; a failed lock skips the checkpoint; the lock timeout reaches the library in `LockArgs.timeoutMs`; `available()` is false when loading fails.
Commit: `feat: add checkpoint driver wrapper`.

### Task 3: worker.py

Interfaces: `call(op, pid, limit_s, timeout_ms=0, python=sys.executable) -> dict` with keys `op`, `result`, `code`, `start`, `end`, `hung`, `stderr`; command line `python -m gpunap.worker <op> <pid> [--timeout-ms N]` printing one JSON line; `run_json(argv, limit_s) -> dict` generic helper; `reserve` mode: `python -m gpunap.worker reserve <pid> <mb>` allocates `mb`, prints `READY`, waits for `GO` on stdin, frees, restores, unlocks, prints JSON.
Tests: `run_json` returns the parsed last JSON line; a child that sleeps past the limit is killed and returns `hung: True`; a child that prints nothing returns `result: "no-output"` with stderr kept; unknown op is rejected by the command line.
Commit: `feat: add time-limited driver worker`.

### Task 4: procinfo.py

Interfaces (all take `proc="/proc"`): `stat_state(pid)`, `start_time(pid)`, `cmdline(pid) -> list[str]`, `children(pid) -> list[int]`, `uvm_maps(pid)`, `thread_names(pid)`, `mem_available_mb(meminfo_text)`, `cgroup_path(pid)`, `cgroup_memory(pid, sys_fs="/sys/fs/cgroup") -> dict`.
Tests with fake trees: command names with spaces and parentheses; vanished processes return None; cgroup v2 `memory.max` of `max` is None.
Commit: `feat: add proc and cgroup probes`.

### Task 5: nvsmi.py, sysinfo.py and results.py

Interfaces: `nvsmi.parse_csv(text, fields)`, `nvsmi.gpu()` (name, driver, used, free, total), `nvsmi.apps() -> list[(pid, mb)]`; `sysinfo.module_kind(line)`, `sysinfo.collect() -> dict`; `results.Verdict` constants, `CheckResult(name, verdict, summary, data, seconds)`, `Report` (header, results, partial flag) with `to_dict()`, `results.claim(path)`, `results.default_name(label, when)`.
Tests: CSV parsing with `[N/A]`; module kind; claim suffixes; default name has no hostname and is filesystem-safe; report JSON round-trips.
Commits: `feat: add nvidia-smi query helpers`, `feat: add machine info collector`, `feat: add result and report types`.

### Task 6: check/owned.py

Interfaces: `OwnedProcesses(token, run_dir, proc="/proc")` with `start(argv, **popen) -> Target`, `is_ours(pid, start_time) -> bool`, `live() -> list[Target]`, `cleanup(driver_call, module_kind) -> list[dict]`, `save()`; `reconcile(run_dir, driver_call, module_kind, proc="/proc") -> list[dict]`; `Target(popen, pid, start_time)`; a pure `cleanup_steps(state, module_kind, was_resumed) -> list[str]`.
Tests: `cleanup_steps` for running, locked, checkpointed, failed, hung-state and open-module cases; `is_ours` rejects a changed start time and a missing token; `cleanup` with a fake driver call issues restore then unlock then SIGCONT then SIGKILL; `reconcile` ignores records whose process no longer matches.
Commit: `feat: add owned process tracking`.

### Task 7: check/cudactx.py, check/targets.py, check/base.py

Interfaces: `Ctx` (alloc, alloc_managed, fill, verify, sync, stream, stream_sync, event_sync, memcpy_sample); `targets.main(argv)` kinds `hold`, `managed`, `sync_first`, `exit_after`, `parent_child`, `hog`, `torch_sync_first`, `torch_nccl`, `torch_ipc`, `torch_train`; `base.Check(name, mode, needs, gpu_mb, host_mb, run, doc)`; `base.Context` (driver call helper, owned processes, label, token, python, out); `base.start_target(ctx, kind, *args) -> Target` that waits for READY or returns an invalid marker; `base.Session` stdin/stdout line protocol with a reader thread (`send`, `wait_for(prefix, timeout)`, `lines`).
Tests: the fill pattern generator and sample verifier are pure and tested; the line protocol is tested against a plain Python child that echoes; `start_target` returns invalid when the child exits before READY.
Commits: `feat: add driver API target context`, `feat: add target process programs`, `feat: add check session protocol`.

### Task 8: the checks

One module and one commit each, each with unit tests for its verdict function built from recorded data shapes:
`api` (`feat: add api check`), `roundtrip`, `errors`, `sync_first_call`, `managed_memory`, `sigstop`, `exit_while_paused`, `child_context`, `cgroup`, `nccl_single_gpu`, `ipc_shared_memory`, `restore_oom`, `size_sweep`, `controller_death`, `correctness`. Commit messages follow `feat: add <name> check` with the name written in words so that the description has 4 to 6 words.

### Task 9: runner and CLI

Interfaces: `runner.preflight(check, gpu, apps, mem_avail, module_kind, full, have) -> (ok, reason)` pure; `runner.run(checks, ctx, full, only) -> Report`; `cli.main(argv)`.
Tests: preflight skips on missing needs, memory budget and co-tenants under `--full`, and on the open module under `--full`; a check that raises gives `error` and cleanup still runs (fake check and fake owned processes); KeyboardInterrupt gives a partial report and exit 130; `--list` prints every check; `--only` selects.
Commits: `feat: add check runner`, `feat: add command line interface`.

### Task 10: GPU acceptance

Files: `tests/gpu/expectations.json`, `tests/gpu/compare.py`, results under `results/check/`.
Build a wheel, install it on gpu1 and gpu2, run `gpunap check --full --label gpuN` on idle GPUs and a default run next to any co-tenant, compare, check for leftovers, commit results.
Commits: `test: add GPU expectations table`, `test: add results comparison script`, `test: add gpu1 check results`, `test: add gpu2 check results`.

### Task 11: CI and docs

Files: `.github/workflows/ci.yml` (unit tests on Python 3.9 to 3.13), `README.md` (install and usage), `docs/checks.md` (what each check does and how to read a verdict).
Commits: `ci: run unit tests on push`, `docs: document gpunap check usage`, `docs: describe each check`.
