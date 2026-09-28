import ctypes, subprocess, sys, time
cuda = ctypes.CDLL("libcuda.so.1")
for fn in ("cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint", "cuCheckpointProcessRestore",
           "cuCheckpointProcessUnlock", "cuCheckpointProcessGetState"):
    getattr(cuda, fn)                                            # raises if the driver lacks the API
print("cuInit", cuda.cuInit(0))
def state(pid):
    s = ctypes.c_int(-1); r = cuda.cuCheckpointProcessGetState(pid, ctypes.byref(s)); return r, s.value
def gpu_mb(pid):
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in q.strip().splitlines() if l.split(",")[0].strip() == str(pid))
def last_step(path):
    import os
    if not os.path.exists(path): return -1
    lines = [l for l in open(path) if l[0].isdigit()]; return int(lines[-1].split()[0]) if lines else -1
pid, log = int(sys.argv[1]), sys.argv[2]
while last_step(log) < 300: time.sleep(0.2)
print("before: state", state(pid), "gpu_mb", gpu_mb(pid), "step", last_step(log))
t0 = time.time(); r1 = cuda.cuCheckpointProcessLock(pid, None); t1 = time.time()
r2 = cuda.cuCheckpointProcessCheckpoint(pid, None); t2 = time.time()
print(f"lock rc={r1} {t1-t0:.3f}s  checkpoint rc={r2} {t2-t1:.3f}s  state {state(pid)}")
time.sleep(2); s_a = last_step(log); m = gpu_mb(pid); time.sleep(8); s_b = last_step(log)
print(f"suspended: gpu_mb {m}  step advanced during 8s? {s_b != s_a} (step {s_a} -> {s_b})")
t3 = time.time(); r3 = cuda.cuCheckpointProcessRestore(pid, None); t4 = time.time()
r4 = cuda.cuCheckpointProcessUnlock(pid, None); t5 = time.time()
print(f"restore rc={r3} {t4-t3:.3f}s  unlock rc={r4} {t5-t4:.3f}s  state {state(pid)}  gpu_mb {gpu_mb(pid)}")
