# Feasibility probe, 27 September 2026, gpu1

RTX 4090, NVIDIA driver 595.84, torch 2.8, run as a shared non-root account. Another job
(about 20 GB) was using the same GPU at the time. No software was installed: `controller.py` calls
the driver's checkpoint functions in `libcuda.so.1` through ctypes.

`trainer.py` trains a small MLP with a fixed seed and deterministic algorithms, holds a 512 MB
extra tensor, and logs every step's loss with a timestamp. `ref.log` is an uninterrupted run.
`run.log` is a second run that `controller.py` paused at step 312 and resumed 10 seconds later.

Console output of the controller, copied from the run:

```
cuInit 0
before: state (0, 0) gpu_mb 1198 step 312
lock rc=0 0.001s  checkpoint rc=0 0.371s  state (0, 2)
suspended: gpu_mb 0  step advanced during 8s? False (step 312 -> 312)
restore rc=0 0.354s  unlock rc=0 0.002s  state (0, 0)  gpu_mb 1198
CHECKSUM -3.133426900074e+03
steps 1500 | all 1500 losses bit-identical to uninterrupted run: True | final checksum equal: True | longest gap between steps: 10.8 s
```

State 0 is running and state 2 is checkpointed. While paused, `nvidia-smi` showed no GPU memory for
the process. One run, one small job: this shows the mechanism works here, not how it behaves for
large jobs or other workloads.

An earlier attempt failed because the controller started before the trainer had created its log.
That run was never paused, and it also matched `ref.log` exactly, which confirms the determinism
baseline.
