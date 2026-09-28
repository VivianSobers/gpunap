# gpunap: pause GPU jobs to RAM so a shared GPU box can be time-shared

Working plan for a personal side project, written 27 September 2026. It shares nothing with the
capstone. `gpunap` is a placeholder name; check GitHub and PyPI before the first commit.

**Status after validation (27 September, evening).** Read `validation/FINDINGS.md` before anything
below. The pause mechanism is exact for ordinary PyTorch jobs, but seven conditions kill or strand a
job, and a close prior-art project (Google's llm-d time slicing) was missed in section 2. Sections 2,
7 and 13 of this plan are superseded where they disagree with the findings, in particular the
"no open-source tool" claim and the safety checks, which are replaced by the design rules there.

## 1. The idea in one paragraph

Many university labs and small teams share one or two GPU machines without a cluster scheduler. When
someone needs the GPU for an hour, the only options today are to wait, or to kill a long-running job
and lose its progress. NVIDIA's driver can now pause a running CUDA process: it finishes queued work,
copies the process's GPU memory into host RAM, and releases the GPU; later it copies everything back
to the same addresses and the process carries on as if nothing happened. gpunap turns that driver
feature into a tool for shared machines. It pauses and resumes jobs with no code changes, needs no root
and no Kubernetes, and adds a small priority scheduler that pauses low-priority jobs when a
high-priority one needs memory and resumes them when it finishes.

## 2. Why this is not crowded (checked 27 September 2026)

| What exists | What it covers | Why it does not cover this | Read depth |
| --- | --- | --- | --- |
| NVIDIA driver checkpoint API and the `cuda-checkpoint` utility | The pause and resume primitive; the docs name pausing a process so other applications can use the GPU | A primitive for one process at a time, with no policy, no scheduling and no safety checks | Search summary of NVIDIA docs and repo |
| Slurm | Cluster scheduling with preemption | A slurm-users thread says there is no suspend option for GPU jobs, only requeue, which kills them | Search summary of the mailing list |
| Run:ai (NVIDIA) | GPU memory swap to CPU RAM | Enterprise platform on Kubernetes | Search summary |
| KAI Scheduler + HAMi (open source) | GPU sharing and memory isolation on Kubernetes | Kubernetes only; sharing, not pausing | Search summary |
| CRIUgpu (arXiv 2502.16631), CRIU's CUDA plugin | Full process checkpoint to disk, including GPU state | Container and CRIU oriented, usually needs elevated privileges; checkpoint and migration, not time-sharing | Abstract and search summary |
| Cedana | Save, migrate and resume for containers | Its GPU plugin is proprietary | Search summary of their docs |
| vLLM (RFC 34303, PR 58183) | Using CUDA checkpoint for its own sleep mode | Inside one inference engine | Search summary |
| torch_memory_saver | Releases and restores PyTorch tensor memory | Inside your own process, needs code changes; an open issue reports corrupted weights with CPU backup | Search summary |

A blog post titled "GPU Pause, Resume, and Migration: The Missing Primitive in Cluster Scheduling"
argues the same gap exists. I first wrote that I found no open-source tool for the bare-metal case.
That was wrong: a GitHub code search on the API names later found llm-d-rl-time-slicing (Google,
Apache-2.0, July 2026), whose snapshot agent does the same swap to host RAM, runs standalone without
Kubernetes, and lists bare-metal and Slurm support on its roadmap; its scheduler is cooperative and
Kubernetes-only. It also found cudackpt (LD_PRELOAD plus CRIU, checkpoint to disk). See
`validation/FINDINGS.md`, "Prior art, corrected".

## 3. Feasibility, already tested

On 27 September 2026 a probe on gpu1 (RTX 4090, driver 595.84, torch 2.8, non-root account,
another 20 GB job on the same GPU) did the following. Files and full output are in `probe/`.

| Step | Result |
| --- | --- |
| Training job running at step 312 | 1,198 MB of GPU memory |
| Lock + checkpoint | 0.37 s; process state "checkpointed"; 0 MB of GPU memory; no steps for 8 s |
| Restore + unlock | 0.36 s; state "running"; 1,198 MB again |
| Job finishes | All 1,500 losses bit-identical to an uninterrupted run; final weight checksum equal |

This is one run of one small job. Everything about large jobs, other workloads and failure cases is
still to be measured, which is most of the project.

## 4. Goals and non-goals

Goals, in order:

1. `gpunap pause PID` and `gpunap resume PID` that work on any single-GPU CUDA process owned by the
   same user, with safety checks, clear errors and timings.
2. `gpunap run --priority low -- CMD` to launch jobs that may be paused, and
   `gpunap run --priority high --vram 18G -- CMD` to launch jobs that pause lower-priority jobs until
   they fit, then resume them afterwards.
3. Never pause a job that was not started through gpunap or explicitly registered with
   `gpunap adopt PID`. On a shared account, consent is the whole design.
4. Prove correctness: paused and resumed jobs produce the same results as uninterrupted ones, across
   a matrix of real workloads.
5. Publish measured numbers: pause and resume time against memory size, host RAM use, overhead while
   running (expected zero), and queue-wait reduction on a replayed lab workload.

Non-goals for version 1: multi-GPU jobs, moving a paused job to a different GPU, surviving a reboot,
Kubernetes, Windows.

## 5. How the driver feature works

From NVIDIA's CUDA driver API documentation (read through a search summary; read the page itself
before writing code):

- `cuCheckpointProcessLock(pid, args)`: blocks new CUDA calls in the target process and waits for
  submitted work to finish.
- `cuCheckpointProcessCheckpoint(pid, args)`: copies device memory to host memory managed by the
  driver and releases the GPU.
- `cuCheckpointProcessRestore(pid, args)`: re-acquires the GPU, copies memory back, and restores
  mappings at their original addresses, plus streams and contexts.
- `cuCheckpointProcessUnlock(pid, args)`: lets the process continue.
- `cuCheckpointProcessGetState(pid, &state)`: running, locked, checkpointed or failed. The probe
  saw 0 for running and 2 for checkpointed.

Things to find out and test, not assume: the minimum driver version; which features are unsupported
(for example unified memory, IPC handles, multi-process setups); what happens when restore finds too
little free VRAM; and what the `args` structures allow (timeouts, and possibly GPU remapping in newer
drivers).

## 6. Architecture

```
  gpunap CLI  ──unix socket──>  gpunapd (one per box, runs as the user, no root)
                                   |- registry: jobs started or adopted by gpunap (pid, priority, owner tag)
                                   |- probe: NVML for free VRAM and per-process memory; /proc/meminfo for host RAM
                                   |- policy: who to pause, when to resume
                                   |- actuator: driver checkpoint API through ctypes
                                   |- event log (SQLite) + /metrics for Prometheus
```

The daemon holds decisions; the driver holds process state. If the daemon crashes, paused jobs stay
paused and safe, and on restart the daemon rebuilds its view from the registry and each process's
checkpoint state.

## 7. Components

### 7.1 Actuator

A thin, well-tested wrapper over the five driver calls, with timings and error codes:

```python
import ctypes, time

class Driver:
    def __init__(self):
        self.cuda = ctypes.CDLL("libcuda.so.1")
        for fn in ("cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint",
                   "cuCheckpointProcessRestore", "cuCheckpointProcessUnlock", "cuCheckpointProcessGetState"):
            getattr(self.cuda, fn)            # fail at start-up if the driver lacks the API
        rc = self.cuda.cuInit(0)
        if rc != 0:
            raise RuntimeError(f"cuInit failed: {rc}")

    def state(self, pid: int) -> int:
        s = ctypes.c_int(-1)
        rc = self.cuda.cuCheckpointProcessGetState(pid, ctypes.byref(s))
        if rc != 0:
            raise RuntimeError(f"get state failed for {pid}: {rc}")
        return s.value

    def pause(self, pid: int) -> dict:
        t0 = time.perf_counter()
        for step in (self.cuda.cuCheckpointProcessLock, self.cuda.cuCheckpointProcessCheckpoint):
            rc = step(pid, None)
            if rc != 0:
                raise RuntimeError(f"{step.__name__} failed for {pid}: {rc}")
        return {"pid": pid, "seconds": time.perf_counter() - t0}

    def resume(self, pid: int) -> dict:
        t0 = time.perf_counter()
        for step in (self.cuda.cuCheckpointProcessRestore, self.cuda.cuCheckpointProcessUnlock):
            rc = step(pid, None)
            if rc != 0:
                raise RuntimeError(f"{step.__name__} failed for {pid}: {rc}")
        return {"pid": pid, "seconds": time.perf_counter() - t0}
```

Map CUDA error numbers to names (`cuGetErrorName`) so messages are readable. If a lock succeeds but
the checkpoint fails, unlock again so the job is never left frozen.

### 7.2 Safety checks before a pause

- The PID is in the registry (started or adopted through gpunap), is alive, and is in the running
  state.
- Host RAM: `MemAvailable` in `/proc/meminfo` must exceed the process's GPU memory plus a margin,
  because the driver copies device memory into host memory. Refuse otherwise.
- The process uses exactly one GPU (from NVML's per-process list). Multi-GPU jobs are out of scope.
- Child processes: if a job's children also hold CUDA contexts (some data loaders do), pause the whole
  group or refuse. Test what happens when only the parent is paused.

### 7.3 Safety checks before a resume

- Free VRAM (NVML) must exceed the job's recorded footprint plus a margin. If it does not, wait and
  retry, never attempt a doomed restore. Record what the driver does when you deliberately try
  (section 9).
- If restore fails, keep the job paused, log loudly, and alert the owner. Never kill it.

### 7.4 Policy

Priorities: `high`, `normal`, `low`. When a high-priority job asks for `--vram N`:

1. If free VRAM already covers N plus the margin, start it.
2. Otherwise choose lower-priority registered jobs to pause: the fewest jobs that free enough memory,
   preferring jobs that were paused least recently (so nobody starves).
3. Start the high-priority job; resume paused jobs in reverse order as memory frees.

Limits: a maximum pause time per job (for example 2 hours) after which the high-priority job must
yield or wait, and a daily pause budget per owner tag. Write the policy as a pure function over a
snapshot of state, and test it with hypothesis:

- never pause a job outside the registry;
- never pause more memory than needed plus one job;
- every paused job is eventually resumed once the demand goes away;
- a job's total pause time never exceeds its limit.

### 7.5 CLI

```
gpunap run --priority low --name sweep-17 -- python train.py --seed 17
gpunap run --priority high --vram 18G --name demo -- python serve.py
gpunap adopt 123456 --priority low        # opt in a job started without gpunap
gpunap ps                                 # jobs, state, GPU MB, paused time
gpunap pause sweep-17 / gpunap resume sweep-17
gpunap log sweep-17                       # pause and resume history with timings
```

### 7.6 Metrics

Gauges: GPU memory held by running jobs, memory parked in host RAM, number of paused jobs. Counters:
pauses, resumes and failures by reason. Histograms: pause and resume time, and queue wait for
high-priority jobs. One small Grafana board, committed as JSON.

## 8. Correctness test matrix

Every row: an uninterrupted run and a run paused at random points (once, and 20 times), comparing
losses or outputs exactly where the workload is deterministic, and within a stated tolerance where it
is not.

| Workload | Why it is in the matrix |
| --- | --- |
| MLP training, deterministic (the probe) | Baseline, already passes once |
| Small transformer training with mixed precision and gradient accumulation | cuBLAS, AMP scaler state |
| CNN training with cuDNN convolutions | cuDNN handles and workspaces |
| `torch.compile` model | Triton kernels in the process |
| Inference with CUDA graphs | Captured graphs must replay after restore |
| DataLoader with worker processes | Children, pinned memory |
| A process near the GPU's memory limit (for example 20 GB) | Large copies, host RAM pressure |
| Unified memory and IPC (expected unsupported) | gpunap must detect and refuse, not corrupt |

Also add fault injection: pause during a long kernel, kill the daemon while a job is paused, fill the
GPU with another process before a resume, run out of host RAM. The rule for all of them: a job may be
delayed, but never corrupted and never killed by gpunap.

## 9. Benchmarks

Report only numbers from runs you executed. Save every run's config, driver version, GPU, host RAM,
raw timings and git commit under `benchmarks/results/`.

| Benchmark | Method |
| --- | --- |
| Pause and resume time against GPU memory size | Jobs holding 0.5, 1, 2, 4, 8, 16 and 20 GB; 5 repeats each; plot time against size and report the implied copy bandwidth |
| Overhead while running | Throughput of a training job with gpunap watching against without; expected no difference, report what you measure |
| Restore with too little free VRAM | What the driver returns, and how gpunap waits |
| Lab scenario | Replay a day of submissions (low-priority sweeps plus short high-priority jobs): high-priority queue wait and total GPU utilisation with gpunap, against first-come-first-served without pausing |

The lab boxes are shared. Check `nvidia-smi` before each benchmark, record co-tenant memory use, and
run large-memory tests only when the GPU is free and after telling the other users.

## 10. Repository layout

```
gpunap/
  pyproject.toml  README.md  LICENSE (Apache-2.0)
  src/gpunap/
    driver.py        (ctypes actuator, error names)
    probe.py         (NVML and /proc readers; fake backend for tests)
    registry.py      (SQLite: jobs, owners, priorities, events)
    policy.py        (pure scheduling functions)
    daemon.py        (unix-socket server, main loop)
    cli.py           (Typer)
    metrics.py
  tests/
    test_policy_props.py          (hypothesis, runs in CI)
    test_driver_errors.py         (fake libcuda, runs in CI)
    gpu/test_bit_identical.py     (the correctness matrix, runs on a GPU box)
    gpu/test_faults.py
  benchmarks/
    size_sweep.py  overhead.py  lab_replay.py  results/
  probe/                          (the 27 September feasibility probe, kept as history)
  docs/
    how-it-works.md  limitations.md  decisions.md  benchmarks.md
  .github/workflows/ci.yml
```

## 11. Milestones

About eight weekends. Each ends with something that works and is tested.

| # | Milestone | Done when |
| --- | --- | --- |
| M0 | Repo, CI, the probe reproduced from a clean checkout | The probe passes again on a GPU box from the repo |
| M1 | `driver.py` with error names and safe unlock on failure; `gpunap pause/resume PID` | Pause and resume any CUDA process you own, with timings |
| M2 | Correctness matrix, first four rows | Bit-identical (or within stated tolerance) with 1 and 20 pauses |
| M3 | Registry, `run`, `adopt`, `ps`, daemon over a unix socket | Jobs survive a daemon restart; only registered jobs can be paused |
| M4 | Policy with property tests; `--priority high --vram` pauses and resumes automatically | Two-job demo: a sweep pauses for a high-priority job and resumes by itself |
| M5 | Rest of the matrix and fault injection; `docs/limitations.md` | Every unsupported case is refused with a clear message |
| M6 | Benchmarks: size sweep, overhead, lab replay | Numbers and plots in `docs/benchmarks.md` |
| M7 | Metrics, dashboard, README, demo video, v0.1.0 | A stranger can install it and run the demo from the README |

## 12. Decisions to record in docs/decisions.md

- Driver API through ctypes instead of shelling out to `cuda-checkpoint`: fewer moving parts, no
  download, precise errors.
- Opt-in registry instead of pausing any process: consent on a shared account.
- Pause to host RAM only (no disk) in version 1: fast, no privileges, but lost on reboot.
- Pure policy function plus property tests instead of ad-hoc rules.
- Refusing unsupported workloads instead of trying and hoping.

## 13. Risks

| Risk | What to do |
| --- | --- |
| NVIDIA or a big project ships the same tool | Move fast to v0.1; your correctness matrix and measurements stay valuable either way |
| Driver limitations bite common workloads | That is a finding: document it in `limitations.md` and in the README |
| Other lab users are uneasy | Opt-in only; announce it; start by pausing only your own jobs |
| A resume fails and a job is stuck | Never kill; keep paused; alert; test this path deliberately (section 8) |
| Host RAM runs out | Check `MemAvailable` before every pause; refuse when short |
| Scope creep into a full cluster scheduler | Non-goals in section 4 go in the README |

## 14. Stretch goals

- A Slurm SPANK plugin or a task-spooler integration so those schedulers gain a real GPU suspend,
  which is exactly what the slurm-users thread asked for.
- Resume on a different GPU of the same machine, if the driver supports remapping (check the docs).
- Checkpoint to disk with CRIU where privileges allow, so paused jobs survive a reboot.
- Rewrite the daemon in Rust or Go for a single static binary.

## 15. First evening checklist

1. Create the repo with the layout in section 10; copy `probe/` in.
2. On a GPU box, run `python3 trainer.py ref.log 1500`, then run `trainer.py` again in the background
   and `controller.py <pid> run.log` against it, as in `probe/RESULT.md`. Check that it passes
   from your checkout.
3. Read NVIDIA's CUDA checkpointing driver API page end to end and list every limitation it states in
   `docs/limitations.md`.
4. Search GitHub for "cuCheckpointProcess" and "cuda-checkpoint" once more, and note anything close.
5. Write `test_policy_props.py` with the four properties in section 7.4 before writing `policy.py`.

## 16. Resume line template

Fill the brackets only with numbers from `docs/benchmarks.md`:

> Built gpunap, an open-source tool that time-shares GPU machines by pausing CUDA jobs to host RAM
> through NVIDIA's driver checkpoint API, with no code changes, no root and no Kubernetes. Pause and
> resume took [a] to [b] seconds for [x] to [y] GB jobs; [n] workloads verified bit-identical across
> [k] forced pauses; cut high-priority queue wait from [p] to [q] minutes in a replayed lab workload.

## 17. Sources

- NVIDIA CUDA driver API, CUDA checkpointing:
  https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html
- NVIDIA cuda-checkpoint: https://github.com/NVIDIA/cuda-checkpoint
- NVIDIA blog, checkpointing CUDA applications with CRIU:
  https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu
- slurm-users, "Avoid killing GPU jobs during preemption":
  https://lists.schedmd.com/mailman3/hyperkitty/list/slurm-users@lists.schedmd.com/message/KLTRWHVO4PJ3O3GMOCR43NBC7RXZEMXB/
- CRIUgpu: https://arxiv.org/abs/2502.16631
- Cedana vs CRIUgpu: https://docs.cedana.ai/articles/cedana-vs.-criu-cuda-for-gpu-checkpoint-restore
- vLLM RFC 34303: https://github.com/vllm-project/vllm/issues/34303
- KAI Scheduler vs Run:ai: https://www.zenml.io/blog/kai-scheduler-vs-runai
- HAMi-core in KAI Scheduler: https://project-hami.io/blog/hami-core-adopted-by-nvidia-kai-scheduler
- torch_memory_saver: https://github.com/fzyzcjy/torch_memory_saver (issue 71 on weight corruption)
- "GPU Pause, Resume, and Migration: The Missing Primitive in Cluster Scheduling":
  https://yezhisheng.me/post/gpu-pause-resume-migration/
