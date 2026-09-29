# gpunap

gpunap pauses CUDA jobs into host RAM so that people can take turns on a shared GPU machine.

NVIDIA's driver has a checkpoint API (`cuCheckpointProcessLock`, `Checkpoint`, `Restore`, `Unlock`) that can freeze a running CUDA process. It copies the process's GPU memory into host RAM and releases the GPU. Later it copies everything back to the same addresses, and the process carries on. gpunap is a plan to turn that into a tool for small labs that share a GPU box without a cluster scheduler. A low-priority job gets paused when someone needs the GPU and resumed when they are done. The job needs no code changes, and nobody needs root.

The pausing tool itself does not exist yet. This repo has the plan, a first feasibility probe, a validation suite that tested the driver mechanism on two machines, and `gpunap check`, an installable command that reproduces those results on your own driver. The full write-up is in [validation/FINDINGS.md](validation/FINDINGS.md). Each claim there names the result file it rests on, and anything without one is labelled as a hypothesis or an external source.

## gpunap check

`gpunap check` runs 15 checks, each of which reproduces one FINDINGS result on the machine it runs on. It needs Linux, Python 3.9 or newer, an NVIDIA driver with the checkpoint API, and `nvidia-smi`. It has no Python dependencies. The PyTorch checks run when PyTorch is installed and are reported as skipped otherwise.

```
pip install git+https://github.com/VivianSobers/gpunap
gpunap check --label mybox            # 11 checks, several minutes, safe next to other jobs
gpunap check --full --label mybox     # adds 4 heavy checks; 3 of them need an idle GPU
gpunap check --list                   # what each check does
gpunap sysinfo                        # the machine facts a report records
```

Each check ends with a verdict: `ok`, `hazard` (the operation killed, crashed, corrupted or stranded the check's own process), `skipped`, `invalid` (the check could not set up its conditions; run it again) or `error` (a bug in gpunap). The run writes a JSON report to the current directory, or to `--out`. The report records the label you give and never the hostname, and the home directory, user name and UID are replaced before it is written. [docs/checks.md](docs/checks.md) describes every check and the verdicts expected on drivers 595 and 580.

The checks only signal processes they started, which they recognise by PID, start time and a per-run token. Before each check they make sure the GPU and the host have room for it plus 1 GB. Three of the `--full` checks (`restore_oom`, `size_sweep` and `controller_death`) can fill most of the GPU's memory, so they are skipped unless other processes hold at most 1 GB in total. `--full` is refused on the open kernel module, where NVIDIA issue #53 reports that killing a resumed process can hang the machine.

On 29 September 2026, `gpunap check --full` gave the expected verdict for all 15 checks on both machines: 8 `ok` and 7 `hazard` on gpu1 (driver 595) and 10 `ok` and 5 `hazard` on gpu2 (driver 580). After each run no process of the run was left, and GPU memory in use was within 100 MB of where it started. The reports are in `results/check/`, and `python tests/gpu/compare.py REPORT` compares a report with the expected table.

## What the tests found

The tests ran on 27 September 2026, with reruns on 29 September, on two RTX 4090 (24 GB) machines, called gpu1 (driver 595.84) and gpu2 (driver 580.178.04), under an ordinary non-root account.

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
- A synchronize (`torch.cuda.synchronize()`, a stream synchronize, or the driver's own `cuCtxSynchronize` and `cuStreamSynchronize`) segfaults inside `libcuda` when it is the first CUDA call the job makes after being checkpointed. All 30 such runs crashed, on both drivers; the driver-API runs had no PyTorch loaded. A synchronize that comes after a copy or a launch waits for the resume like other calls. `validation/repro_sync.py` reproduces the crash, and `validation/backtrace_sync.py` captures it under gdb.
- Checkpointing a job that uses managed (unified) memory is refused. On driver 595 the job failed afterwards (2 of 2 runs); on 580 it carried on (1 run).
- A single-GPU NCCL process group, as torchrun and Accelerate set up, makes the job abort on driver 595. It pauses cleanly on 580.
- CUDA memory shared between processes breaks the checkpoint or the restore on both drivers.
- A job stopped with SIGSTOP makes the lock, state query and restore calls hang until it gets SIGCONT.
- A job in a memory-limited cgroup is killed partway through the checkpoint, because the copied GPU memory is charged to the job.

Two workarounds passed small tests: allocate a placeholder of the job's size before restoring, and make every synchronize start with a one-element kernel launch. Neither closes the gap completely. Another process can still grab the memory in the moment between freeing the placeholder and restoring. And if a checkpoint completes between the guard launch and the synchronize, the synchronize still crashes, although that needs the thread to stall for at least as long as a checkpoint takes (0.18 s at the fastest here). So gpunap can only be an opt-in tool that starts jobs itself and refuses the ones it cannot vouch for.

Google's llm-d-rl-time-slicing project already has a node agent that swaps GPU memory to host RAM in the same way. Its scheduler is cooperative and needs Kubernetes. FINDINGS.md recommends building the hazard suite first, since it is reusable and can feed bug reports to NVIDIA, and building gpunap on top of it. `gpunap check` is that suite.

## Caveats in the recorded results

- Several runs labelled with 20 pauses completed fewer, because the job finished before the later pause points. The 388 total counts only cycles that completed.
- Some results rest on one machine or a single run. FINDINGS.md says which.

## Layout

| Path | Contents |
| --- | --- |
| `src/gpunap/` | The `gpunap` package: driver wrapper, time-limited driver worker, `/proc` and `nvidia-smi` probes, and `check/` with one module per check, the target processes, the runner and the process tracking. |
| `tests/` | Unit tests that run without a GPU, plus `tests/gpu/` with the expected verdicts per driver and the script that compares a report with them. |
| `results/check/` | `gpunap check` reports from gpu1 and gpu2. |
| `docs/checks.md` | What each check does and how to read a verdict. |
| `PLAN.md` | The original design and milestones. Sections 2, 7 and 13 are superseded by FINDINGS.md. |
| `probe/` | The first feasibility run: a small MLP trainer, a controller that pauses it once, both loss logs, and `RESULT.md`. |
| `validation/cu.py` | ctypes wrapper over the driver checkpoint API, plus `/proc` and `nvidia-smi` probes. |
| `validation/target.py` | Deterministic workloads used as pause targets. |
| `validation/run_case.py`, `matrix.sh`, `summarize.py` | Correctness cases: two reference runs, then a run paused at random steps. |
| `validation/faults.py` | Fault-injection and edge-case tests, one function each. |
| `validation/bench_size.py`, `bw.py` | Pause and resume time against memory size, and raw copy bandwidth. |
| `validation/repro_sync.py` | Standalone reproduction of the synchronize crash, through the driver API and through PyTorch. |
| `validation/backtrace_sync.py` | Runs the synchronize crash under gdb and saves the backtrace. |
| `validation/pcie_link.py`, `sysinfo.py` | PCIe link under load, and the machine facts the findings rely on (driver, kernel module, kernel, RAM, packages). |
| `validation/resultfile.py`, `procinfo.py`, `test_*.py` | Helpers that keep earlier result files and read `/proc`, and unit tests that run without a GPU. |
| `validation/results_gpu1/`, `results_gpu2/` | Raw JSON results. Each file records the driver, GPU and full test configuration. |

## Running the validation scripts

The package's own unit tests run from the repository root with `python -m pytest -q` and need no GPU.

The validation scripts are the older, broader suite behind FINDINGS. You need Linux, an NVIDIA driver with the checkpoint API (580 and 595 were tested), Python with a CUDA build of PyTorch, `psutil`, and `nvidia-smi` on the path. The `hf` workload needs `transformers` and Qwen3-1.7B in the Hugging Face cache. The `paged` workload needs `bitsandbytes`, and the `cgroup` test needs `systemd-run --user`. Set `GPUNAP_LABEL` to a name for the machine; the scripts never record the real hostname.

The fault tests fill GPU memory and kill processes, although only processes the scripts started themselves. Run them on a GPU you can take over. From inside `validation/`:

```
python3 bench_size.py 0.5 1 2 4 8 16 20          # speed and integrity against size
python3 faults.py errors short_far reserve sigstop cgroup uvm ipc tail_ops
bash matrix.sh                                     # correctness cases, three lanes in parallel
python3 run_case.py --kind amp_tf --pauses 20      # one correctness case
python3 summarize.py                               # table of all correctness results
python3 repro_sync.py --label mybox                # synchronize after a checkpoint, 30 runs
python3 faults.py uvm_maps thread_names            # detection signals, 3 runs per case
python3 sysinfo.py --label mybox                   # machine facts, no GPU work
python3 -m pytest -q test_*.py                     # unit tests, no GPU needed
```

New results go to `validation/results/` and logs to `validation/logs/`. A rerun never replaces an earlier file; it writes `name-2.json` next to it.
