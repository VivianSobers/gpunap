# gpunap

gpunap pauses CUDA jobs into host RAM so that people can take turns on a shared GPU machine.

NVIDIA's driver has a checkpoint API (`cuCheckpointProcessLock`, `Checkpoint`, `Restore`, `Unlock`) that can freeze a running CUDA process. It copies the process's GPU memory into host RAM and releases the GPU. Later it copies everything back to the same addresses, and the process carries on. gpunap is a plan to turn that into a tool for small labs that share a GPU box without a cluster scheduler. A low-priority job gets paused when someone needs the GPU and resumed when they are done. The job needs no code changes, and nobody needs root.

There is no tool yet. This repo has the plan, a first feasibility probe, and a validation suite that tested the driver mechanism on two machines. The full write-up is in [validation/FINDINGS.md](validation/FINDINGS.md).

## What the tests found

The tests ran on 27 September 2026 on two RTX 4090 (24 GB) machines, called gpu1 (driver 595.84) and gpu2 (driver 580.178.04), under an ordinary non-root account.

Pausing works and is exact for ordinary single-process PyTorch jobs. 33 runs went through 388 pause-and-resume cycles, covering 9 workload types:

- MLP
- transformer with fp16 autocast and gradient accumulation
- CNN with cuDNN
- `torch.compile`
- CUDA graph replay
- DataLoader with worker processes
- two threads on separate streams
- Qwen3-1.7B generation
- single-GPU NCCL (gpu2 only)

In every run, each step's loss and the final weights matched an uninterrupted run bit for bit. On gpu1 a job holding 21,092 MB paused in about 5.2 s and resumed in about 3.0 s. gpu2 was about 3.5 times slower at 16 GB.

Pausing an arbitrary job is unsafe. Seven conditions killed a job, crashed it, or left it stuck:

- Restoring while other processes hold too much GPU memory fails, and after that the job can never be restored (NVIDIA issue #44). This happened in 4 of 4 runs on driver 595 and 3 of 3 on driver 580.
- Calling `torch.cuda.synchronize()` or a stream synchronize while the job is paused segfaults inside `libcuda`. A process that uses only the driver API crashes the same way on `cuCtxSynchronize` and `cuStreamSynchronize`, which puts the fault in the driver. `validation/repro_sync.py` reproduces it.
- Checkpointing a job that uses managed (unified) memory is refused, and on driver 595 the job's CUDA context is dead afterwards.
- A single-GPU NCCL process group, as torchrun and Accelerate set up, makes the job abort on driver 595. It pauses cleanly on 580.
- CUDA memory shared between processes breaks the checkpoint or the restore on both drivers.
- A job stopped with SIGSTOP makes the lock and restore calls hang until it gets SIGCONT.
- A job in a memory-limited cgroup is killed partway through the checkpoint, because the copied GPU memory is charged to the job.

Two workarounds passed small tests: allocate a placeholder of the job's size before restoring, and make every synchronize start with a one-element kernel launch. Neither closes the gap completely. So gpunap can only be an opt-in tool that starts jobs itself and refuses the ones it cannot vouch for.

Google's llm-d-rl-time-slicing project already has a node agent that swaps GPU memory to host RAM in the same way. Its scheduler is cooperative and needs Kubernetes. FINDINGS.md recommends building the hazard suite first, since it is reusable and can feed bug reports to NVIDIA, and building gpunap on top of it.

## Caveats in the recorded results

- Every synchronize run on disk crashed. On driver 595 that is 3 of 3 device and 3 of 3 stream synchronize runs, all through PyTorch. On driver 580 it is 3 of 3 of each through PyTorch and 3 of 3 of each through the driver API. The driver-API reproduction has not run on 595 yet.
- Several runs labelled with 20 pauses completed fewer, because the job finished before the later pause points. The 388 total counts only cycles that completed.

## Layout

| Path | Contents |
| --- | --- |
| `PLAN.md` | The original design and milestones. Sections 2, 7 and 13 are superseded by FINDINGS.md. |
| `probe/` | The first feasibility run: a small MLP trainer, a controller that pauses it once, both loss logs, and `RESULT.md`. |
| `validation/cu.py` | ctypes wrapper over the driver checkpoint API, plus `/proc` and `nvidia-smi` probes. |
| `validation/target.py` | Deterministic workloads used as pause targets. |
| `validation/run_case.py`, `matrix.sh`, `summarize.py` | Correctness cases: two reference runs, then a run paused at random steps. |
| `validation/faults.py` | Fault-injection and edge-case tests, one function each. |
| `validation/bench_size.py`, `bw.py` | Pause and resume time against memory size, and raw copy bandwidth. |
| `validation/repro_sync.py` | Standalone reproduction of the synchronize crash, through the driver API and through PyTorch. |
| `validation/resultfile.py`, `test_*.py` | A helper that keeps earlier result files from being overwritten, and unit tests that run without a GPU. |
| `validation/results_gpu1/`, `results_gpu2/` | Raw JSON results. Each file records the driver, GPU and full test configuration. |

## Running the tests

You need Linux, an NVIDIA driver with the checkpoint API (580 and 595 were tested), Python with a CUDA build of PyTorch, `psutil`, and `nvidia-smi` on the path. The `hf` workload needs `transformers` and Qwen3-1.7B in the Hugging Face cache. The `paged` workload needs `bitsandbytes`, and the `cgroup` test needs `systemd-run --user`.

The fault tests fill GPU memory and kill processes, although only processes the scripts started themselves. Run them on a GPU you can take over. From inside `validation/`:

```
python3 bench_size.py 0.5 1 2 4 8 16 20          # speed and integrity against size
python3 faults.py errors short_far reserve sigstop cgroup uvm ipc tail_ops
bash matrix.sh                                     # correctness cases, three lanes in parallel
python3 run_case.py --kind amp_tf --pauses 20      # one correctness case
python3 summarize.py                               # table of all correctness results
python3 repro_sync.py --label mybox                # synchronize while checkpointed, 30 runs
python3 -m pytest -q test_*.py                     # unit tests, no GPU needed
```

New results go to `validation/results/` and logs to `validation/logs/`. A rerun never replaces an earlier file; it writes `name-2.json` next to it.
