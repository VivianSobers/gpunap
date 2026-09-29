# gpunap validation: what holds and what breaks

Tests of the assumptions behind `../PLAN.md`, run on both lab boxes on the evening of 27 September
2026. The code is in this folder. Every number below comes from a JSON file in `results_gpu1/` or
`results_gpu2/`, and each file records the driver, GPU and full test configuration.

## Verdict

For ordinary single-process PyTorch jobs, pausing and resuming works and is exact. 33 runs across 9
workload types and both drivers went through 388 pause-and-resume cycles, and in every run each
step's loss and the final weights matched an uninterrupted run bit for bit.

Pointing it at an arbitrary job is unsafe. I found seven conditions in which a pause or a resume
kills the job, crashes it, or leaves it stuck. Three of them cannot be detected from outside the
process with anything I found. Two matter most:

- If a restore runs while other processes hold too much GPU memory, it fails, and from then on the
  job can never be restored and has to be killed. This is NVIDIA issue #44, open since January 2026.
  It happened 7 times out of 7, on both drivers.
- A paused job that calls `torch.cuda.synchronize()` or a stream synchronize crashes with a
  segfault inside the driver. Every such run on disk crashed: 18 of 18 through PyTorch across both
  drivers, and 12 of 12 in a process that uses only the driver API (`cuCtxSynchronize` and
  `cuStreamSynchronize`, 6 on each driver). I found no NVIDIA issue for it.

Both have workarounds that passed small tests: claim the memory before restoring, and make every
synchronize start with a tiny kernel launch. Neither closes the gap completely. gpunap can be built,
but only as an opt-in tool that starts the jobs itself, refuses the jobs it cannot vouch for, and
says plainly what it cannot prevent. On a box where people also start GPU jobs outside gpunap it
cannot be airtight.

The prior-art picture has also changed. Google's llm-d project already ships a node daemon that
swaps GPU memory to host RAM in the same way. It has a standalone mode, and bare-metal and Slurm
support are on its roadmap. Details are under "Prior art, corrected" below.

## Setup

| | gpu1 | gpu2 |
| --- | --- | --- |
| GPU | RTX 4090 24 GB, PCIe 4.0 x16 | RTX 4090 24 GB, PCIe 4.0 x16 |
| Driver | 595.84, proprietary kernel module | 580.178.04, proprietary kernel module |
| PyTorch (NCCL) | 2.8.0+cu128 (2.27.3); also 2.13.0 (2.29.7) | 2.14.0+cu130 (2.30.7); also 2.13.0 (2.29.7) |
| Host RAM | 125 GB, no swap | 125 GB |
| Account | shared non-root account, no sudo | same |

The controller calls the driver's checkpoint functions directly through ctypes (`cu.py`). Nothing
was installed on either box.

For each correctness case (`run_case.py`), the workload first runs twice without interruption, which
confirmed that every workload is deterministic. It then runs a third time while the controller pauses
it at random step counts, waits 0.3 to 1.5 s, and resumes it. Every step writes its loss as an exact
float in hex. The paused run has to match the reference line for line, along with a hash of the
final weights. The memory tests (`bench_size.py`) put a block of random integers on the GPU and check
it after resuming, using exact integer sums over the whole block and a fixed sample of raw values.
Each fault test is one function in `faults.py`.

My own harness had three bugs that produced misleading results, and I fixed them before collecting the
numbers below. The process that fills the GPU was not waited for. The background launcher left SIGINT
ignored in every child process. And a stale log file let one test pause a job before the job had
started CUDA. Runs affected by these bugs were thrown away and repeated.

A fourth bug turned up later. `faults.py` wrote each test's result to a fixed file name, so a rerun
replaced the earlier file, and that lost the first synchronize runs on driver 580. The scripts now
keep every file (`resultfile.py`). Both drivers were rerun on 29 September with `repro_sync.py`, and
the synchronize counts below include those runs.

## What worked

| Test | Result | Driver |
| --- | --- | --- |
| Correctness, 9 workload types: MLP (2 seeds); transformer with fp16 autocast, GradScaler, gradient accumulation and dropout (2 seeds); CNN with cuDNN; `torch.compile`, including a pause during compilation; CUDA graph replay; DataLoader with 4 workers and pinned memory; two threads on separate streams; Qwen3-1.7B greedy generation; single-GPU NCCL process group (gpu2 only). Variants: `expandable_segments` allocator, `cudaMallocAsync` allocator, transformer training next to a 12 GB block | 33 runs, 388 cycles, all bit-identical to the reference | both |
| 20 GB job (21,092 MB on the GPU) | pause 5.2 s, resume 3.0 s, block intact | 595 |
| 50 cycles on a 2 GB job | job memory, host memory and pause times flat across all 50 cycles | both |
| 30-minute pause | resumed with the block intact | 595 |
| Two jobs paused together, resumed in either order | both intact | both |
| Job whose child process has its own CUDA context | each process has to be paused separately. Pausing only the parent leaves the child running and holding its memory | both |
| One controller pauses and exits, a second one resumes | works | both |
| Controller killed 0.25 s into a 16 GB checkpoint | the checkpoint completes without it | 595 |
| Controller killed during a restore | the restore completes, but the job stays locked (frozen) until another controller unlocks it | both |
| Paused job killed with SIGKILL or SIGTERM | exits within 0.3 s and host memory comes back | 595 |
| Job killed in the middle of a 16 GB checkpoint | the call returns an error, and host and GPU memory come back | both |
| Calls in the wrong state, wrong PIDs | wrong state gives `CUDA_ERROR_ILLEGAL_STATE` with no effect; a missing PID gives `OPERATING_SYSTEM`; a process without CUDA, or a zombie, gives `NOT_INITIALIZED` | both |
| Lock with a 1 s timeout during a 7 s kernel | returns `NOT_READY` after 1.0 s and the job keeps running. Without a timeout the lock waited 5.2 s for the kernel | 595 |
| Job that finishes and exits while paused | exit code 0 | both |
| Step time after resuming | step times between pauses match the uninterrupted run (transformer on gpu2: 47 to 48 ms, reference 49 ms). One exception I did not investigate: the DataLoader job's last segment ran at 27.6 ms per step against 6.0 ms in its reference | both |

## What breaks the job

| Condition | What happened | Count | Detectable from outside? |
| --- | --- | --- | --- |
| Restore with too little free GPU memory | Restore returns out-of-memory. Every later restore returns `OPERATING_SYSTEM`, even with plenty of memory free, after 10 s and after 30 s. The failed attempt also keeps the memory it managed to grab (1.8 GB and 4.2 GB in two runs) until the job is killed. NVIDIA issue #44 | 4/4 on 595, 3/3 on 580 | Not needed: prevent it by claiming memory first (rule 1 below) |
| `torch.cuda.synchronize()` or a stream synchronize while paused | Segfault inside `libcuda`. A process with no PyTorch crashes the same way on `cuCtxSynchronize` and `cuStreamSynchronize` (`repro_sync.py`). If the job is only locked and not checkpointed, the same calls return at once without waiting, so the lock never covered them. Event synchronize, copies, kernel launches and allocations wait correctly | 595: device sync 6/6 and stream sync 6/6 through PyTorch, 3/3 and 3/3 through the driver API. 580: 3/3 and 3/3 through PyTorch, 3/3 and 3/3 through the driver API | No |
| Managed (unified) memory | Checkpoint returns `NOT_SUPPORTED`. On 595 the job's CUDA context is dead afterwards: every later call fails, and a bitsandbytes `PagedAdamW32bit` job crashed on its next cuBLAS call. On 580 the job survived the refused pause | 2/2 lost on 595; 1/1 survived on 580 | Yes, heuristically: the normal CUDA processes checked (4 kinds) had at most one `/dev/nvidia-uvm` mapping; the two managed-memory jobs had 2 and 5 |
| NCCL process group on one GPU (what torchrun and Accelerate set up) | On 595 the checkpoint fails after 1.4 s and the job aborts ("unspecified launch failure" in the NCCL watchdog), also with `NCCL_CUMEM_ENABLE=0` and with PyTorch 2.13. On 580 it pauses cleanly with the same PyTorch 2.13 build | 4/4 lost on 595; 2/2 fine on 580 | Yes for PyTorch: threads named `pt_nccl_watchdg` and `pt_nccl_heartbt` |
| CUDA memory shared between processes (a CUDA tensor passed to a child with `torch.multiprocessing`) | On 595 the checkpoint hangs for 32 to 35 s and fails; the child died and the parent segfaulted. On 580 the child's checkpoint succeeded but its restore returned `INVALID_VALUE` and it stayed stuck. NVIDIA lists this as unsupported before driver 610 | 1/1 on each driver | No signal found |
| Job stopped with SIGSTOP (Ctrl-Z) | Lock, state query and restore all hang with no end; the lock's 5 s timeout does not apply. After SIGCONT everything recovers and the data is intact | 1/1 of each case on each driver | Yes: process state `T` |
| Job in a memory-limited cgroup (as under Slurm, Docker or systemd) | The checkpoint copies GPU memory into the job's own process, so the cgroup kills the job partway through. The limit was 2,851 MB for a job with 4,708 MB on the GPU | 1/1 on each driver | Yes: `memory.max` and `memory.current` |

Three more behaviours do not harm the job but a scheduler has to plan around them:

- A state query blocks until an in-flight checkpoint or restore finishes: 3.8 s on gpu1 and
  14.9 s on gpu2 for a 16 GB job.
- Ctrl-C on a paused job does nothing until it is resumed, and then the job exits. A job's own
  SIGTERM handler would be delayed the same way, because its main thread is blocked in the driver.
- A job can finish and exit while it is paused (exit code 0 above), so the daemon has to treat a
  vanished PID as a normal event.

## Speed and memory

| GPU memory of job | gpu1 pause / resume (s) | gpu2 pause / resume (s) |
| --- | --- | --- |
| 1,124 MB | 0.31 / 0.22 | not run |
| 1,636 MB | 0.43 / 0.29 | 1.58 to 1.60 / 0.83 to 0.87 |
| 4,708 MB | 1.15 to 1.18 / 0.69 to 0.71 | 4.25 to 4.31 / 2.12 to 2.30 |
| 8,804 MB | 2.19 to 2.36 / 1.27 to 1.29 | not run |
| 16,996 MB | 4.22 to 4.85 / 2.45 to 2.48 | 15.30 to 15.54 / 7.48 to 8.25 |
| 21,092 MB | 5.18 to 5.25 / 3.01 to 3.05 | not run |

Three repeats per size. gpu1 checkpoints at about 4 GB/s and restores at about 7 GB/s; gpu2
at about 1.1 and 2.2 GB/s. A plain PyTorch copy test (`bw.py`) gave 18.4 GB/s pinned and 14.0 to 14.3
GB/s pageable device-to-host on gpu1, and 14.3 GB/s pinned and 3.5 to 4.1 GB/s pageable on
gpu2. At 16 GB, gpu2's checkpoint is about 3.5 times slower than gpu1's, close to the
3.7-fold gap in pageable copy speed. That suggests host memory speed sets the pace, though I have not
proven it.

While a job is paused, its GPU memory sits in the job's own process as ordinary memory. For the
16,996 MB job, the process's resident size went to 17,748 MB and the machine's available memory fell
by 16.8 to 16.9 GB. None of it was locked (`VmLck` 0), so on a box with swap it could be swapped out,
and it counts against the job's cgroup, which is the cause of the cgroup kill above. Paused jobs used
almost no CPU: the median was 0, with short bursts when a pause landed during CPU-side work.

## Not tested

- Jobs owned by a different Unix user. There is no second account and no sudo on the boxes.
- vLLM, SGLang, JAX and TensorFlow. None is installed, and the boxes had no DNS, so nothing could be
  installed.
- Multi-GPU jobs.
- Killing a job that is running again after a restore. NVIDIA issue #53 reports that this hard-hangs
  the whole machine on the open kernel module 595.71.05. I did not stress this on machines other
  people depend on. Five restored jobs here (3 on gpu1, 2 on gpu2) were later ended by SIGINT
  with no problem, and both boxes use the proprietary module. A real test needs a machine nobody else
  is using.
- The whole machine running out of RAM. Only the cgroup case was tested.
- Driver 610, which NVIDIA says adds support for shared CUDA memory.
- Whether the two boxes differ on NCCL and managed memory because of the driver or because of
  something else about the machines. They have the same GPU, kernel and PyTorch build, so the driver
  is the likely cause.

## Prior art, corrected

When I recommended gpunap I said I had found no open-source tool for a single shared machine. A GitHub
code search for the checkpoint function names found some that I had missed:

- **llm-d-rl-time-slicing** (Google, Apache-2.0, announced on the Google Cloud blog on 24 July 2026,
  11 stars, last updated 25 September). Its snapshot agent is a node daemon that pauses GPU processes
  into host RAM using `cuda-checkpoint`, and it has a standalone mode that runs without Kubernetes.
  Its scheduler needs Kubernetes and is cooperative: jobs call `acquire()` and `yield()` around their
  GPU phases. The blog says the agent "is designed to run standalone outside Kubernetes for bare metal
  and Slurm environments", and the README roadmap lists support outside Kubernetes. Read depth:
  README, deploy docs, the agent's and orchestrator's `main.go` flags, and a summary of the blog.
- **cudackpt** (25 stars, updated 23 September): an `LD_PRELOAD` shim plus CRIU that checkpoints to
  disk and restores. It does not time-share. Read depth: README.
- NVIDIA's own issue tracker already has #44 (restore after out-of-memory), #53 (machine hang), #58
  (lock segfault after NVML is unloaded) and #30 and #45 (hangs with multi-process NCCL). Read depth:
  issue text and comments for #44, #53, #58 and #29; titles for the rest.

So the mechanism is not uncrowded. What remains open is pausing unmodified jobs preemptively on one
machine, with a safety layer, and a public record of what breaks. Google's cooperative design avoids
most of the hazards above, because it only pauses at points the job chooses. That may be why they
chose it.

## Design rules the results force

1. **Claim before restoring.** When enough memory looks free, the daemon allocates a placeholder the
   size of the job plus 256 MB. Only if that works does it free the placeholder and restore at once.
   Tested once on each driver with another process filling the rest of the GPU: the restore worked,
   with 1.3 ms (gpu1) and 7 ms (gpu2) between freeing and restoring. An outside process that
   allocates inside that gap still wins the race.
2. **Start jobs through gpunap and guard synchronize.** For PyTorch, patch
   `torch.cuda.synchronize` and `Stream.synchronize` so that each first launches a one-element
   kernel, which waits while the job is paused. Tested 4 times (3 on gpu1, 1 on gpu2), then 12 more
   times with `repro_sync.py`, 6 on each machine (half through PyTorch, half through the driver
   API): no crash, and the synchronize ran after the resume. Native code that calls synchronize directly is not covered.
3. **Refuse to pause** when the process is stopped (`T`), has more than one `/dev/nvidia-uvm`
   mapping, has PyTorch NCCL threads on a driver where that crashes, has less cgroup headroom than
   its GPU memory plus a margin, or when the machine's available memory is short by the same
   measure.
4. **Opt-in only.** The owner declares that the job does not share CUDA memory between processes,
   since nothing detectable shows it.
5. Pause and resume every CUDA process in the job's process tree.
6. Make every driver call from a separate worker process with a timeout, so a hung call can be
   killed. Never block the daemon on a state query.
7. On daemon start, reconcile: unlock jobs left locked, and send checkpointed jobs through rule 1.
8. Treat a job that exits while paused as normal, and check each PID's start time so a reused PID
   is never paused.
9. `gpunap kill` should send SIGKILL, because Ctrl-C and handled SIGTERM wait for a resume. Issue #53
   suggests checkpointing a restored job again before killing it.
10. Keep a table of what works on each driver, filled in by running this suite at install time.

## Recommendation

There are three ways forward:

1. Build gpunap with the rules above. It is still useful, but the promise shrinks from "pause any
   job" to "pause jobs that pass the checks", and Google may ship bare-metal support first.
2. Make the test suite the project: a public conformance and hazard suite for NVIDIA's checkpoint
   API across drivers and frameworks, with the bugs reported upstream. I found nothing like it
   published, and tonight's run already found a crash with no NVIDIA issue (synchronize while
   paused).
3. Pick a different project.

I recommend combining the first two. Build the suite first, since it is reusable and citable and
feeds bug reports upstream. Then build gpunap as the safe tool on top of it. The first concrete step
is to report the synchronize crash to NVIDIA with a minimal reproduction. `repro_sync.py` is that
reproduction, and its driver-API cases need nothing but `libcuda`.

## Rerunning

Copy this folder to a box and run from inside it. Seeds and every setting are in each result file.

```
python3 bench_size.py 0.5 1 2 4 8 16 20          # speed and integrity against size
python3 faults.py errors short_far retry_far reserve sigstop cgroup uvm ipc tail_ops   # any test names from faults.py
bash matrix.sh                                     # correctness cases, three lanes in parallel
python3 run_case.py --kind amp_tf --pauses 20      # one correctness case
python3 summarize.py                               # table of all correctness results
python3 repro_sync.py --label gpu1                 # synchronize while checkpointed, driver API and PyTorch
python3 -m pytest -q test_*.py                     # unit tests for the harness, no GPU needed
```

`tail_ops` reads the operations to try from `TAIL_OPS`, for example `TAIL_OPS=sync,guarded_sync`.
