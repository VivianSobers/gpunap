# gpunap validation: what holds and what breaks

Tests of the assumptions behind `../PLAN.md`, run on two lab machines on the evening of 27 September
2026, with reruns and new measurements on 29 September. The code is in this folder. Every result
file is in `results_gpu1/` or `results_gpu2/` and records the driver, the GPU and the full test
configuration.

How to read this document: each claim names the result files it rests on, or the count of runs
behind it. A statement no result file backs is labelled "Hypothesis" or "External source". "595"
and "580" are the driver versions on gpu1 and gpu2. A claim that holds on one machine only says so.

## Verdict

For ordinary single-process PyTorch jobs, pausing and resuming works and is exact. 33 runs across 9
workload types and both drivers went through 388 pause-and-resume cycles, and in every run each
step's loss and the final weights matched an uninterrupted run bit for bit (the correctness files,
`<kind>_p<pauses>_*.json` and `smoke_mlp.json`).

Pointing it at an arbitrary job is unsafe. I found seven conditions in which a pause or a resume
kills the job, crashes it, or leaves it stuck. Two of them (the synchronize crash and CUDA memory
shared between processes) gave no signal from outside the process in these tests, and a third
(restoring into too little memory) has to be prevented rather than detected. Two matter most:

- If a restore runs while other processes hold too much GPU memory, it fails, and from then on the
  job can never be restored and has to be killed. 7 of 7 runs: 4 on 595 and 3 on 580
  (`fault_short_far`, `fault_short_near`, `fault_retry_far`, and `fault_retry_near` on 595). External
  source: this is NVIDIA cuda-checkpoint issue #44.
- A job crashes with a segfault inside `libcuda` when a synchronize (`cuCtxSynchronize`,
  `cuStreamSynchronize`, or PyTorch's `torch.cuda.synchronize()` and `Stream.synchronize()`) is the
  first CUDA call it makes after being checkpointed. Every such run crashed: 18 of 18 through
  PyTorch and 12 of 12 through the driver API alone, on both drivers (`fault_repro_sync.json` on
  both machines, `fault_tail_ops_*` on 595). A synchronize that follows a call which waits at the
  lock does not crash (see the table below). A search of NVIDIA's cuda-checkpoint issues on
  29 September found no report of it.

Both have workarounds that passed small tests: claim the memory before restoring (1 run per driver),
and make every synchronize start with a tiny launch (16 runs). Neither closes the gap completely;
design rules 1 and 2 give the remaining windows. gpunap can be built, but only as an opt-in tool that
starts the jobs itself, refuses the jobs it cannot vouch for, and says plainly what it cannot
prevent. On a box where people also start GPU jobs outside gpunap it cannot be airtight.

External source: Google's llm-d project already ships a node daemon that swaps GPU memory to host
RAM in the same way. Details are under "Prior art, corrected" below.

## Setup

| | gpu1 | gpu2 | Evidence |
| --- | --- | --- | --- |
| GPU | RTX 4090 24 GB | RTX 4090 24 GB | `sysinfo.json` |
| PCIe link | Gen 4 x16 under a copy load (Gen 1 at idle) | Gen 4 x16, read at idle | `pcie_link.json` (gpu1), `sysinfo.json` |
| Driver | 595.84, proprietary kernel module | 580.178.04, proprietary kernel module | `sysinfo.json` |
| Linux kernel | 6.8.0-138-generic | 6.8.0-138-generic | `sysinfo.json` |
| Default PyTorch (NCCL) | 2.8.0 (2.27.3), Python 3.10.12 | 2.14.0 (2.30.7), Python 3.10.12 | `sysinfo.json` |
| Host RAM | 128,498 MB, no swap | 128,498 MB, no swap | `sysinfo.json` |

Each box also has a conda environment with Python 3.13 (3.13.12 on gpu1, 3.13.5 on gpu2, as recorded
in the result files), used for the `nccl1_conda` and `paged` runs. Its PyTorch version was not
recorded; my notes from the run say 2.13.0. The account is shared, non-root and has no sudo. Nothing
was installed on either box.

The controller calls the driver's checkpoint functions directly through ctypes (`cu.py`).

For each correctness case (`run_case.py`), the workload first runs twice without interruption, and
the two runs matched for every workload (`ref_deterministic` in each file). It then runs a third time
while the controller pauses it at randomly chosen step counts, waits 0.3 to 1.5 s, and resumes it.
The step at which each pause actually landed is recorded per pause. Every step writes its loss as an
exact float in hex. The paused run has to match the reference line for line, along with a hash of
the final weights. The memory tests (`bench_size.py`) put a block of random integers on the GPU and
check it after resuming, using exact integer sums over the whole block and a fixed sample of raw
values. Each fault test is one function in `faults.py`.

My own harness had three bugs that produced misleading results, and I fixed them before collecting
the numbers below. The process that fills the GPU was not waited for. The background launcher left
SIGINT ignored in every child process. And a stale log file let one test pause a job before the job
had started CUDA. Runs affected by these bugs were thrown away and repeated. Some of those runs'
files are still in `results_gpu1/`: the `fault_tail_ops_*` files whose lock returned
`CUDA_ERROR_NOT_INITIALIZED` never paused anything and are not counted.

A fourth bug turned up later. `faults.py` wrote each test's result to a fixed file name, so a rerun
replaced the earlier file, and that lost the first synchronize runs on 580. The scripts now keep every
file (`resultfile.py`). The synchronize counts include reruns on 29 September with `repro_sync.py`.

One harness behaviour is left as it is: `run_case.py` reuses the reference logs in `logs/` when logs
with the same kind, seed, steps, ballast and environment variables exist. It does not check the code
or the PyTorch version. A stale reference would make a correct run look wrong, not the other way
round, so it cannot produce a false pass.

## What worked

| Test | Result | Evidence |
| --- | --- | --- |
| Correctness, 9 workload types: MLP (2 seeds); transformer with fp16 autocast, GradScaler, gradient accumulation and dropout (2 seeds); CNN with cuDNN; `torch.compile`, including a pause during compilation; CUDA graph replay; DataLoader with 4 workers and pinned memory; two threads on separate streams; Qwen3-1.7B greedy generation; single-GPU NCCL process group (580 only). Variants: `expandable_segments` allocator, `cudaMallocAsync` allocator, transformer training next to a 12 GB block | 33 runs, 388 cycles, all bit-identical to the reference. Several runs set up for 20 pauses completed fewer, because the job finished before the later pause points; 388 counts only completed cycles | correctness files, both machines |
| 20 GB job (21,092 MB on the GPU) | pause 5.18 to 5.25 s, resume 3.01 to 3.05 s, block intact. 595 only | `bench_size.json` (gpu1) |
| 50 cycles on a 2 GB job | all 50 succeeded; process RSS unchanged; checkpoint 0.73 to 0.78 s on 595 and 2.30 to 2.40 s on 580; available host memory stayed within a 1 GB band | `fault_leak.json`, both |
| 30-minute pause | resumed with the block intact. 595 only | `fault_longpause.json` (gpu1) |
| Two jobs paused together, resumed in either order | both finished with their blocks intact | `fault_two_jobs.json`, both |
| Job whose child process has its own CUDA context | each process has to be paused separately. Pausing only the parent left the child running and holding its memory | `fault_child_cuda.json`, both |
| One controller pauses and exits, a second one resumes | works | `fault_handoff.json`, both |
| Controller killed 0.25 s into a 16 GB checkpoint | the checkpoint completed without it. 595 only | `fault_ctlkill_ckpt.json` (gpu1) |
| Controller killed during a restore | the restore completed, but the job stayed locked (frozen) until another controller unlocked it | `fault_ctlkill_restore.json`, both |
| Paused 4 GB job killed with SIGKILL or SIGTERM | exited after 0.26 s (SIGKILL) and 0.21 s (SIGTERM), and available host memory rose by 4.8 and 5.2 GB. 595 only, 1 run each | `fault_sigkill.json`, `fault_sigterm.json` (gpu1) |
| Job killed in the middle of a 16 GB checkpoint | the call returned an error, and host and GPU memory came back | `fault_killmid.json`, both |
| Calls in the wrong state, wrong PIDs | wrong state gives `CUDA_ERROR_ILLEGAL_STATE` with no effect; a missing PID gives `OPERATING_SYSTEM`; a process without CUDA gives `NOT_INITIALIZED` (both drivers); a zombie gives `NOT_INITIALIZED` (595 only) | `fault_errors.json`, both; `fault_zombie.json` (gpu1) |
| Lock with a 1 s timeout during a 7 s kernel | returned `NOT_READY` after 1.0 s and the job kept running. Without a timeout the lock waited 5.2 s for the kernel. 595 only | `fault_lock_timeout.json` (gpu1) |
| Job that exits while paused, making no further CUDA call | exit code 0: one explicit `sys.exit` (595), and 8 correctness runs paused after their last step whose process was gone at resume time (4 on each driver). A job whose exit path starts with a synchronize crashes instead: rc -11, 1 run on 595 | `fault_tail_ops_a.json` (`exit`); correctness files; `fault_exit_while_paused.json` (`cpu_only_tail`) |

## What breaks the job

| Condition | What happened | Count and evidence | Detectable from outside? |
| --- | --- | --- | --- |
| Restore with too little free GPU memory | Restore returns out-of-memory. Every later restore returns `OPERATING_SYSTEM`, even with plenty of memory free, after 10 s and after 30 s. The failed attempt also keeps the memory it managed to grab (1,832 MB, 4,200 MB and 1,832 MB in the three runs that measured it) until the job is killed | 4/4 on 595, 3/3 on 580. `fault_short_*`, `fault_retry_*` | Not needed: prevent it by claiming memory first (rule 1) |
| A synchronize as the first CUDA call after the checkpoint | Segfault inside `libcuda`, in the calling thread. A process with no PyTorch crashes the same way on `cuCtxSynchronize` and `cuStreamSynchronize`. On 595 both calls fault at the same instruction and the same address in `libcuda` (`backtrace_sync_*.json`). Calls that wait at the lock do not crash, and a synchronize after one of them is safe: copies, event record and synchronize, and a launch followed by a synchronize all waited for the resume (3/3 each, both APIs, both drivers). A synchronize already in progress when the lock arrived returned normally (2/2 on 595, `fault_tail_ops_d_inflight_*`) | 595: device sync 6/6 and stream sync 6/6 through PyTorch, 3/3 and 3/3 through the driver API. 580: 3/3 and 3/3 through PyTorch, 3/3 and 3/3 through the driver API. `fault_repro_sync.json`, `fault_tail_ops_*` | No |
| Managed (unified) memory | Checkpoint returns `NOT_SUPPORTED`. On 595 the driver-API job's next device-to-host copy failed with error 719, and a bitsandbytes `PagedAdamW32bit` job exited with an error (rc 1) at step 260 of 300; its error message was not saved. On 580 the driver-API job's data was intact afterwards | 2/2 lost on 595, 1/1 survived on 580. `fault_uvm.json`, `paged_p1.json` (gpu1) | Possibly. Processes without managed memory had one `/dev/nvidia-uvm` mapping (4 kinds, 15 runs; DataLoader workers had none) and processes with managed memory had two (2 programs, 8 runs). Mostly 595: 12 of the 15 and 7 of the 8 runs. Only 64 MB and 256 MB allocations were tried. `fault_uvm_maps.json` (gpu1), `fault_uvm_detect.json` (both) |
| NCCL process group on one GPU (what torchrun and Accelerate set up) | On 595 the checkpoint returned `OPERATING_SYSTEM` after 1.39 to 1.40 s (0.40 s in the conda environment) and the job aborted with SIGABRT (rc -6), also with `NCCL_CUMEM_ENABLE=0`. The abort message was not saved. On 580 it paused cleanly, in the default and the conda environment, with bit-identical results | 4/4 lost on 595; 2/2 runs (10 cycles) fine on 580. `nccl1_*.json` | Likely for PyTorch: threads named `pt_nccl_watchdg` and `pt_nccl_heartbt` were present in 3/3 single-GPU NCCL jobs and absent in 3/3 plain PyTorch jobs. 595 and PyTorch 2.8.0 only. `fault_thread_names.json` (gpu1) |
| CUDA memory shared between processes (a CUDA tensor passed to a child with `torch.multiprocessing`) | On 595 the child's checkpoint failed after 31.9 s and the parent's after 34.5 s; the child died and the parent segfaulted (rc -11). On 580 the child's checkpoint succeeded but its restore returned `INVALID_VALUE` and it stayed stuck. External source: NVIDIA lists this as unsupported before driver 610 | 1/1 on each driver. `fault_ipc.json` | No signal found |
| Job stopped with SIGSTOP (Ctrl-Z) | Lock (with a 5 s timeout), state query and restore all hung until the controller gave up (30 s, 10 s and 30 s). After SIGCONT everything recovered and the data was intact | 1 run of each case per driver. 595: `fault_sigstop-2.json` (rerun on 29 September); 580: `fault_sigstop.json`. The older 595 file `fault_sigstop.json` came from an earlier test version that did not query the state while the job was stopped | Yes: process state `T` |
| Job in a memory-limited cgroup (as under Slurm, Docker or systemd) | The job was killed (SIGKILL) during the checkpoint. The limits were 2,851 MB (595) and 2,977 MB (580) for a job with 4,708 MB on the GPU. Paused GPU memory is charged to the job's own process (see "Speed and memory"), which accounts for the kill | 1/1 on each driver. `fault_cgroup.json` | Yes: `memory.max` and `memory.current` |

Three more behaviours do not harm the job, but a scheduler has to plan around them:

- A state query blocks until an in-flight checkpoint or restore finishes. For a 16 GB job it took
  3.8 s (595) and 14.9 s (580) during the checkpoint, and 2.3 s and 7.7 s during the restore
  (`fault_state_blocking.json`, 1 run each).
- Ctrl-C (SIGINT) on a paused job did nothing until it was resumed, and then the job exited
  (`fault_sigint.json`, both). Hypothesis, not tested: a job's own SIGTERM handler would be delayed
  the same way, because its main thread is blocked in the driver.
- A job can exit while it is paused (see "What worked"), so the daemon has to treat a vanished PID
  as a normal event.

## Speed and memory

| GPU memory of job | gpu1 pause / resume (s) | gpu2 pause / resume (s) |
| --- | --- | --- |
| 1,124 MB | 0.31 / 0.22 | not run |
| 1,636 MB | 0.43 / 0.29 | 1.58 to 1.60 / 0.83 to 0.87 |
| 4,708 MB | 1.15 to 1.18 / 0.69 to 0.71 | 4.25 to 4.31 / 2.12 to 2.30 |
| 8,804 MB | 2.19 to 2.36 / 1.27 to 1.29 | not run |
| 16,996 MB | 4.22 to 4.85 / 2.45 to 2.48 | 15.30 to 15.54 / 7.48 to 8.25 |
| 21,092 MB | 5.18 to 5.25 / 3.01 to 3.05 | not run |

Three repeats per size (`bench_size.json`). gpu1 checkpoints at about 4 GB/s and restores at about
7 GB/s; gpu2 at about 1.1 and 2.2 GB/s. A plain PyTorch copy test (`bw.json`) gave 18.4 GB/s pinned
and 14.0 to 14.3 GB/s pageable device-to-host on gpu1, and 14.3 GB/s pinned and 3.5 to 4.1 GB/s
pageable on gpu2. At 16 GB, gpu2's checkpoint is about 3.5 times slower than gpu1's, close to the
3.7-fold gap in pageable copy speed. Both links run at PCIe Gen 4 x16 (Setup), so the link width and
generation do not explain the gap. Hypothesis, not tested: host memory speed sets the pace.

While a job is paused, its GPU memory sits in the job's own process as ordinary memory. For the
16,996 MB job, the process's resident size went to 17,748 MB and the machine's available memory fell
by 16.8 to 16.9 GB. None of it was locked (`VmLck` 0). Hypothesis, not tested: on a box with swap it
could be swapped out (neither box has swap). Across 403 pauses in the correctness runs, paused jobs
used a median of 0 CPU seconds per second, with a maximum of 0.98 when a pause landed during CPU-side
work.

Step time after resuming was not measured in a controlled way. The correctness files record the
median step time before the first pause and after the last resume, but on these shared GPUs the
ratio of the two ranged from 0.12 to 1.48 on gpu1 (22 runs) and from 0.11 to 6.13 on gpu2 (10 runs),
often in runs with warm-up effects such as `torch.compile`. These numbers cannot show whether
resuming changes a job's speed.

## Not tested

- Jobs owned by a different Unix user. There is no second account and no sudo on the boxes.
- vLLM, SGLang, JAX and TensorFlow. None is installed, and the boxes had no DNS, so nothing could be
  installed.
- Multi-GPU jobs.
- Killing a job that is running again after a restore. External source: NVIDIA issue #53 reports
  that this hard-hangs the whole machine on the open kernel module 595.71.05. I did not stress this
  on machines other people depend on. Four restored jobs here (2 on each machine) were later ended
  by SIGINT with no problem (`fault_sigint.json`, `fault_sigint_control.json`), and both boxes use the
  proprietary module. A real test needs a machine nobody else is using.
- The whole machine running out of RAM. Only the cgroup case was tested.
- Driver 610, which NVIDIA says adds support for shared CUDA memory (external source).
- Whether the two boxes differ on NCCL and managed memory because of the driver or because of
  something else. They have the same GPU model, kernel release and kernel module type
  (`sysinfo.json`), but different default PyTorch versions. The NCCL difference also appeared in the
  conda environments, whose PyTorch version was not recorded. Hypothesis: the driver is the cause.

## Prior art, corrected

External sources, read on 27 September 2026. When I recommended gpunap I said I had found no
open-source tool for a single shared machine. A GitHub code search for the checkpoint function names
found some that I had missed:

- llm-d-rl-time-slicing (Google, Apache-2.0, announced on the Google Cloud blog on 24 July 2026,
  11 stars, last updated 25 September). Its snapshot agent is a node daemon that pauses GPU processes
  into host RAM using `cuda-checkpoint`, and it has a standalone mode that runs without Kubernetes.
  Its scheduler needs Kubernetes and is cooperative: jobs call `acquire()` and `yield()` around their
  GPU phases. The blog says the agent "is designed to run standalone outside Kubernetes for bare metal
  and Slurm environments", and the README roadmap lists support outside Kubernetes. Read depth:
  README, deploy docs, the agent's and orchestrator's `main.go` flags, and a summary of the blog.
- cudackpt (25 stars, updated 23 September): an `LD_PRELOAD` shim plus CRIU that checkpoints to
  disk and restores. It does not time-share. Read depth: README.
- NVIDIA's own issue tracker already has #44 (restore after out-of-memory), #53 (machine hang), #58
  (lock segfault after NVML is unloaded) and #30 and #45 (hangs with multi-process NCCL). Read depth:
  issue text and comments for #44, #53, #58 and #29; titles for the rest.

So the mechanism is not uncrowded. What remains open is pausing unmodified jobs preemptively on one
machine, with a safety layer, and a public record of what breaks. Google's cooperative design avoids
most of the hazards above, because it only pauses at points the job chooses. Hypothesis: that may be
why they chose it.

## Design rules the results force

1. Claim before restoring. When enough memory looks free, the daemon allocates a placeholder the
   size of the job plus 256 MB. Only if that works does it free the placeholder and restore at once.
   Tested once on each driver with another process filling the rest of the GPU: the restore worked,
   with 1.3 ms (595) and 7.0 ms (580) between freeing and restoring (`fault_reserve.json`). An outside
   process that allocates inside that gap still wins the race.
2. Start jobs through gpunap and guard synchronize. For PyTorch, patch `torch.cuda.synchronize` and
   `Stream.synchronize` so that each first launches a one-element kernel, which waits at the lock while
   the job is paused. Tested 16 times without a crash, and the synchronize ran after the resume: 4 with
   `faults.py tail_ops` (3 on 595, 1 on 580) and 12 with `repro_sync.py` (3 through PyTorch and 3
   through the driver API on each driver). The guard does not close the race. If the checkpoint
   completes after the guard launch returns and before the synchronize starts, the synchronize is
   still the first call after the checkpoint. That needs the thread to stall between two adjacent
   calls for as long as a lock plus checkpoint takes, and the fastest of 540 successful checkpoints in
   these result files took 0.18 s. Hypothesis, not tested: this is rare, but possible on a heavily loaded CPU. Native code that
   calls synchronize directly is not covered at all.
3. Refuse to pause when the process is stopped (`T`), has more than one `/dev/nvidia-uvm` mapping,
   has PyTorch NCCL threads on a driver where that crashes, has less cgroup headroom than its GPU
   memory plus a margin, or when the machine's available memory is short by the same measure. The
   mapping and thread-name signals rest on the runs listed in the table above, mostly on 595. They
   are not yet verified on other drivers, PyTorch versions or allocation patterns.
4. Opt-in only. The owner declares that the job does not share CUDA memory between processes, since
   nothing detectable showed it.
5. Pause and resume every CUDA process in the job's process tree.
6. Make every driver call from a separate worker process with a timeout, so a hung call can be
   killed. Never block the daemon on a state query.
7. On daemon start, reconcile: unlock jobs left locked, and send checkpointed jobs through rule 1.
8. Treat a job that exits while paused as normal, and check each PID's start time so a reused PID
   is never paused.
9. `gpunap kill` should send SIGKILL, because Ctrl-C waits for a resume. External source: issue #53
   suggests checkpointing a restored job again before killing it.
10. Keep a table of what works on each driver, filled in by running this suite at install time.

## Recommendation

There are three ways forward:

1. Build gpunap with the rules above. It is still useful, but the promise shrinks from "pause any
   job" to "pause jobs that pass the checks", and Google may ship bare-metal support first.
2. Make the test suite the project: a public conformance and hazard suite for NVIDIA's checkpoint
   API across drivers and frameworks, with the bugs reported upstream. I found nothing like it
   published, and these runs found a crash with no NVIDIA issue (synchronize after a checkpoint).
3. Pick a different project.

I recommend combining the first two. Build the suite first, since it is reusable and citable and
feeds bug reports upstream. Then build gpunap as the safe tool on top of it. The first concrete step
is to report the synchronize crash to NVIDIA with a minimal reproduction. `repro_sync.py` is that
reproduction, and its driver-API cases need nothing but `libcuda`.

## Rerunning

Copy this folder to a box and run from inside it. Seeds and every setting are in each result file.
Set `GPUNAP_LABEL` (for example `GPUNAP_LABEL=gpu1`) so the files record a label instead of
"unlabelled"; the real hostname is never recorded.

```
python3 bench_size.py 0.5 1 2 4 8 16 20          # speed and integrity against size
python3 faults.py errors short_far retry_far reserve sigstop cgroup uvm ipc tail_ops   # any test names from faults.py
python3 faults.py uvm_maps thread_names            # detection signals, 3 runs per case
bash matrix.sh                                     # correctness cases, three lanes in parallel
python3 run_case.py --kind amp_tf --pauses 20      # one correctness case
python3 summarize.py                               # table of all correctness results
python3 repro_sync.py --label gpu1                 # synchronize after a checkpoint, driver API and PyTorch
python3 backtrace_sync.py --label gpu1 --op ctx_sync   # the same crash under gdb
python3 pcie_link.py --label gpu1                  # PCIe link at idle and under load
python3 sysinfo.py --label gpu1                    # driver, kernel module, kernel, RAM, packages
python3 -m pytest -q test_*.py                     # unit tests for the harness, no GPU needed
```

`tail_ops` reads the operations to try from `TAIL_OPS`, for example `TAIL_OPS=sync,guarded_sync`.
