# gpunap check: design

Date: 29 September 2026. Status: approved in conversation, section by section, on 29 September.

## Goal

`gpunap check` tells the owner of any Linux GPU machine what NVIDIA's CUDA checkpoint API does on
their driver: which pause-and-resume operations are safe and which ones kill, crash or strand a job.
Each check reproduces one result from `validation/FINDINGS.md` that is backed by a result file. The
suite is the first part of gpunap. The pause scheduler described in `PLAN.md` will run it at install
time later (FINDINGS design rule 10).

## Decisions already made

- The suite lives in the gpunap package and runs as `gpunap check`.
- A default run is safe on a shared machine: under about 1 GB of GPU memory, only its own processes
  are signalled, and it takes 10 to 15 minutes. Heavy checks need `--full` and an idle GPU.
- The core depends only on the Python standard library and the NVIDIA driver, reached through
  ctypes. PyTorch checks run when PyTorch is importable and are reported as skipped otherwise.
- v1 is done when all unit tests pass, `gpunap check --full` on idle gpu1 (driver 595) and gpu2
  (driver 580) matches the expectations table below, a default run next to other people's jobs stays
  within its budget and leaves nothing behind, and the package installs with pip.

## Checks

Default run:

| Check | What it reproduces | Needs |
| --- | --- | --- |
| `api` | the driver exports the five checkpoint functions; driver version, kernel module type | libcuda |
| `roundtrip` | pause and resume a 256 MB process three times, data intact | libcuda |
| `errors` | calls in the wrong state, on a missing PID and on a process without CUDA return errors and change nothing | libcuda |
| `sync_first_call` | a synchronize as the first CUDA call after a checkpoint crashes; a copy, an event synchronize and a guarded synchronize wait for the resume | libcuda; PyTorch variant when available |
| `managed_memory` | checkpointing a process with managed memory is refused, and whether the process survives; `/dev/nvidia-uvm` mapping counts with and without managed memory | libcuda |
| `sigstop` | lock, state query and restore hang on a stopped process; it recovers after SIGCONT | libcuda |
| `exit_while_paused` | a process that exits without another CUDA call exits with code 0 while paused | libcuda |
| `child_context` | pausing a parent leaves a child's own CUDA context running; both can be paused and resumed | libcuda |
| `cgroup` | a process in a memory-limited cgroup is killed when the checkpoint copies its GPU memory into it | libcuda, `systemd-run --user` |
| `nccl_single_gpu` | pausing a single-GPU NCCL process group aborts the job or works; `pt_nccl_*` thread names | PyTorch |
| `ipc_shared_memory` | CUDA memory shared between processes breaks the checkpoint or the restore | PyTorch |

`--full` run (idle GPU required), in addition to the default checks:

| Check | What it reproduces |
| --- | --- |
| `restore_oom` | a restore into too little free memory leaves the process permanently unrestorable; claiming the memory first avoids it |
| `size_sweep` | pause and resume time against size, up to 20 GB, data intact |
| `controller_death` | the controller killed partway through a checkpoint and partway through a restore |
| `correctness` | deterministic PyTorch training stays bit-identical across 20 pauses |

Not in v1: the 30-minute pause and the 50-cycle leak test, the lock timeout (it needs a
multi-second kernel), and the state-query blocking time.

## Verdicts

Every check returns exactly one verdict:

- `ok`: the operation behaved safely.
- `hazard`: the operation killed, crashed, corrupted or stranded the check's own process.
- `skipped`: a requirement is missing; the reason is recorded.
- `invalid`: the check could not set up its own conditions, for example a call that landed before
  the checkpoint had finished. It says nothing about the driver.
- `error`: a bug in gpunap itself; the traceback is recorded.

Each check also records a one-line summary and its raw data: every driver call's result name, start
and end time, the target processes' exit codes and outputs, the repeat count and the configuration.
Each check's verdict rule is a pure function of that data.

## Architecture

Package `gpunap` under `src/`:

| Module | Responsibility |
| --- | --- |
| `driver.py` | ctypes wrapper over `cuCheckpointProcessLock`, `Checkpoint`, `Restore`, `Unlock` and `GetState`, plus `cuInit`, `cuDriverGetVersion` and `cuGetErrorName`. Loads `libcuda.so.1` on first use, and accepts an injected library for tests. `pause` unlocks again if the checkpoint fails. |
| `worker.py` | Runs one driver call in a child process with a time limit and returns its result as JSON; a hung call is killed and reported as hung. Also the `reserve` helper used by `restore_oom`. |
| `procinfo.py` | Read-only `/proc` and cgroup probes with an injectable root. |
| `nvsmi.py` | `nvidia-smi` queries and their parsers. |
| `sysinfo.py` | Driver, kernel module type, kernel, RAM, swap, PCIe link and package versions. |
| `results.py` | Verdicts, `CheckResult`, the run report, JSON output and never-overwrite file naming. |
| `check/cudactx.py` | Minimal driver-API context for target processes: allocate, fill, verify, synchronize, managed memory. |
| `check/targets.py` | The processes being paused, run as `python -m gpunap.check.targets <kind>`. They talk to the check over stdin and stdout. |
| `check/owned.py` | Starts targets and tracks them by PID, start time and a per-run token; cleans them up. |
| `check/base.py` | The check description (name, mode, needs, memory budget) and the context passed to each check. |
| `check/<name>.py` | One module per check. |
| `check/runner.py` | Pre-flight, sequential execution, cleanup, interrupt handling, report. |
| `cli.py` | `gpunap check [--full] [--only A,B] [--label NAME] [--out DIR] [--list]`, `gpunap sysinfo`, `gpunap --version`. |

`validation/` stays unchanged as the historical record behind FINDINGS. The checks re-implement its
logic in small tested units and do not import it.

### Target protocol

A target prints `READY` once its GPU state exists. Commands arrive one per line on stdin: `VERIFY`
makes it read back its GPU data and print `VERIFIED ok` or `VERIFIED bad <detail>`; `EXIT` makes it
exit with code 0. Single-shot targets (`sync_first` and `exit_after`) print `CALL <time>` right
before their one call and `DONE <time> rc=<code>` after it. The per-run token is on every target's
command line.

## Safety

- Pre-flight, per check: free GPU memory must cover the check's budget plus 1 GB, and available host
  memory must cover the largest pause plus 1 GB; otherwise the check is skipped. The number and
  memory of other GPU processes are recorded.
- `--full` refuses to run while any other process holds GPU memory, and each heavy check looks again
  before it starts.
- A process is signalled only if its PID, start time and command-line token all match a target this
  run started. Checks run one at a time.
- Every driver call that can hang runs through `worker.py` with a time limit.
- Cleanup runs after every check, including one that raised: for each live target, query its state
  (with a time limit), resume a checkpointed target or unlock a locked one, send SIGCONT, send SIGKILL.
- Ctrl-C runs the same cleanup, writes the results gathered so far marked partial, and exits 130.
- Each run records its targets in `~/.cache/gpunap/runs/<token>.json`. The next run cleans up
  targets left behind by a runner that was killed.
- Open kernel module: external source, NVIDIA issue #53 reports that killing a resumed process can
  hang the machine and that checkpointing it first avoids this. On the open module, cleanup
  checkpoints a resumed target again before killing it, and `--full` refuses to run.
- Results files never overwrite an earlier file. Only the label given with `--label` is recorded,
  never the hostname.

## Testing

Unit tests run without a GPU: every verdict rule, `driver.py` against a fake libcuda, the worker's
time limit, cleanup order with a fake driver (a check that raises, and an interrupt), PID
ownership, pre-flight decisions, parsers against fake `/proc` trees and `nvidia-smi` output, and
file naming.

GPU tests run by hand on gpu1 and gpu2. `tests/gpu/expectations.json` lists each check's expected
verdict per driver, with the FINDINGS evidence behind it, and `tests/gpu/compare.py` compares a
results file against it:

| Check | 595 | 580 |
| --- | --- | --- |
| `api`, `roundtrip`, `errors`, `exit_while_paused`, `child_context` | ok | ok |
| `sync_first_call` | hazard | hazard |
| `managed_memory` | hazard | ok |
| `nccl_single_gpu` | hazard | ok |
| `ipc_shared_memory`, `sigstop`, `cgroup`, `restore_oom` | hazard | hazard |
| `size_sweep`, `controller_death`, `correctness` | ok | ok |

After each GPU run, no process carrying the run token may remain, and GPU memory must return to
within 100 MB of where it started.

Installation: the lab machines have no DNS, so `pip install git+https://github.com/VivianSobers/gpunap`
is checked on a machine with internet access, and the lab machines install a wheel built from the
same commit.
