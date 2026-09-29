# What each check does

`gpunap check` starts its own small CUDA processes, pauses and resumes them through the driver's
checkpoint API, and records what happened. It never touches a process it did not start. Each check
reproduces one result from [validation/FINDINGS.md](../validation/FINDINGS.md), so a report tells you
which of those results hold on your driver.

## Reading a verdict

Every check ends with one verdict.

| Verdict | Meaning |
| --- | --- |
| `ok` | The operation behaved safely: the process survived and its data was intact. |
| `hazard` | The operation killed, crashed, corrupted or stranded the check's own process. On the drivers tested so far, several checks are expected to report this; see the table at the end. |
| `skipped` | Something the check needs is missing (PyTorch, `systemd-run`, `--full`, free memory, an idle GPU). The summary says what. |
| `invalid` | The check could not set up its own conditions, for example a call that landed before the checkpoint had finished. It says nothing about the driver. Run it again. |
| `error` | gpunap itself failed. The traceback is in the report. Please file it. |

A `hazard` is a finding about the driver, not a failure of the tool. The command exits 0 when every
check reached a verdict, 1 when any check ended in `error`, 2 for a usage error or a refused run, and
130 when interrupted.

The JSON report holds each check's summary and its raw data: every driver call's result, start and
end time, the target processes' exit codes and last output lines, and the settings used. The file
records the label you gave with `--label` and never the hostname. The home directory, user name,
host name and UID are replaced before the file is written.

## Default checks

These run next to other people's jobs. Each one checks free GPU memory and host memory before it
starts and is skipped when either would drop below its budget plus 1 GB.

`api` checks that `libcuda.so.1` exports the five checkpoint functions and that `cuInit` succeeds,
and records the driver version and whether the kernel module is the proprietary or the open one.

`roundtrip` pauses and resumes a process holding 256 MB three times and reads its data back after
each cycle.

`errors` makes calls in the wrong state (restore a running process, checkpoint twice, unlock twice
and so on), calls on a PID that does not exist and on a process without CUDA. Each must fail with an
error and leave the process as it was, and the process without CUDA must survive the calls.

`sync_first_call` starts a process that sleeps and then makes exactly one CUDA call, and checkpoints
it so that the call lands while it is checkpointed. The calls are `cuCtxSynchronize` and
`cuStreamSynchronize`, and as controls a copy, an event synchronize, and a synchronize that follows a
tiny kernel launch. It runs each call through the driver API, and through PyTorch when PyTorch is
installed, twice (three times with `--full`). FINDINGS: the two synchronize calls segfault inside
`libcuda`; the controls wait for the resume.

`managed_memory` pauses a process that holds 64 MB of CUDA managed memory next to 128 MB of ordinary
memory, and counts its `/dev/nvidia-uvm` mappings, the outside signal FINDINGS found for managed
memory. FINDINGS: the checkpoint is refused; on 595 the process is broken afterwards, on 580 it
carries on.

`sigstop` stops a running process with SIGSTOP and tries a lock with a 5 s timeout and a state
query. It then checkpoints a second process, stops it, and tries a restore and a state query. Each
call goes through a worker that is killed at its time limit (15 s, or 10 s for a state query).
Afterwards it sends SIGCONT, brings both processes back and checks their data. FINDINGS: all three calls hang while the process is stopped.

`exit_while_paused` checkpoints a process that is about to exit without making another CUDA call.
FINDINGS: it exits with code 0, so a vanished PID after a pause is normal.

`child_context` starts a process whose child has its own CUDA context, pauses the parent, checks
that the child is still running and holding its memory, then pauses and resumes each process
separately. FINDINGS: each process has to be paused on its own.

`cgroup` runs a process under `systemd-run --user --scope` with a `memory.max` above its running size
and below its size once paused, then pauses it. A checkpoint copies GPU memory into the process's
own memory, which counts against its cgroup. FINDINGS: the kernel kills it during the checkpoint.

`nccl_single_gpu` (PyTorch) pauses a job with a single-GPU NCCL process group that runs an
`all_reduce` every 20 ms, the setup torchrun and Accelerate create, and records its `pt_nccl_*`
thread names. FINDINGS: on 595 the checkpoint fails and the job aborts; on 580 it pauses cleanly.

`ipc_shared_memory` (PyTorch) shares a CUDA tensor with a child process through
`torch.multiprocessing`, then pauses and resumes the child and then the parent. FINDINGS: the
checkpoint or the restore breaks on both drivers. NVIDIA lists this as unsupported before driver 610.

## Checks that need `--full`

These fill GPU memory or run for minutes, so they only run with `--full`, and the heavy ones only when
the GPU is idle. Idle here means other processes hold at most 1 GB of GPU memory in total, which
leaves room for a desktop session. Each heavy check looks again just before it starts. `--full` is
refused on the open kernel module, because NVIDIA issue #53 reports that killing a resumed process
there can hang the machine.

`restore_oom` checkpoints a 1 GB process, starts a second process that fills the GPU until only 40 %
of the first one's size is free, and tries to restore. It then frees the memory and tries again
after 0 and 10 s. Afterwards it tests the workaround from FINDINGS on a fresh process: a helper holds
a placeholder of the process's size, frees it and restores straight away. FINDINGS: the first restore
runs out of memory and every later one fails, so the process can never be restored; the workaround
avoids it.

`size_sweep` runs a process at 0.5, 1, 2, 4, 8, 16 and 20 GB, as far as free GPU memory and host
memory allow, with three pause, resume and verify cycles at each size. It reports the median
checkpoint and restore time. The data must be intact after every cycle.

`controller_death` kills the controller with SIGKILL 0.25 s into a checkpoint of a process of up to
16 GB, and again 0.25 s into a restore. It polls the state for 10 s, brings the process back as the
state requires, and checks its data. FINDINGS: the checkpoint completes without its controller, and
the restore completes but leaves the process locked until another controller unlocks it.

`correctness` (PyTorch) trains a small MLP deterministically for 300 steps, printing each step's loss
as an exact hex float and a hash of the final weights. Two uninterrupted runs must match each other.
A third run is paused at 20 seeded random steps for 0.3 to 1.5 s each and must match them line for
line.

## Cleanup

Every target carries a random run token on its command line, and gpunap records each one's PID and
start time. A process is signalled only when all three still match. After every check, including one
that raised or was interrupted, gpunap sends each live target SIGCONT, queries its state, unlocks it
if it is locked, and sends SIGKILL. A checkpointed target is killed without being restored. Each run records
its targets in `~/.cache/gpunap/runs/`, so the next run cleans up after a runner that was killed.

## Expected verdicts

These come from FINDINGS and are the table `tests/gpu/compare.py` checks a report against.

| Check | Driver 595 | Driver 580 |
| --- | --- | --- |
| `api`, `roundtrip`, `errors`, `exit_while_paused`, `child_context` | ok | ok |
| `sync_first_call` | hazard | hazard |
| `managed_memory` | hazard | ok |
| `nccl_single_gpu` | hazard | ok |
| `ipc_shared_memory`, `sigstop`, `cgroup`, `restore_oom` | hazard | hazard |
| `size_sweep`, `controller_death`, `correctness` | ok | ok |
