"""ctypes wrapper over the CUDA driver checkpoint API, plus /proc and nvidia-smi probes.

CLI: python cu.py {state,lock,checkpoint,restore,unlock,pause,resume} PID [lock_timeout_ms]
prints one JSON line.
"""
import ctypes, json, os, subprocess, sys, time

_cu = ctypes.CDLL("libcuda.so.1")
_INIT_RC = _cu.cuInit(0)


class LockArgs(ctypes.Structure):
    _fields_ = [("timeoutMs", ctypes.c_uint), ("reserved0", ctypes.c_uint),
                ("reserved1", ctypes.c_uint64 * 7)]


STATE = {0: "running", 1: "locked", 2: "checkpointed", 3: "failed"}


def err(rc):
    if rc == 0:
        return "OK"
    p = ctypes.c_char_p()
    if _cu.cuGetErrorName(rc, ctypes.byref(p)) == 0 and p.value:
        return p.value.decode()
    return f"rc{rc}"


def state(pid):
    s = ctypes.c_int(-1)
    rc = _cu.cuCheckpointProcessGetState(int(pid), ctypes.byref(s))
    return STATE.get(s.value, s.value) if rc == 0 else err(rc)


def _op(fn, pid, args=None):
    t = time.perf_counter()
    rc = fn(int(pid), args)
    return err(rc), round(time.perf_counter() - t, 4)


def lock(pid, timeout_ms=0):
    a = LockArgs(int(timeout_ms), 0)
    return _op(_cu.cuCheckpointProcessLock, pid, ctypes.byref(a))


def checkpoint(pid):
    return _op(_cu.cuCheckpointProcessCheckpoint, pid)


def restore(pid):
    return _op(_cu.cuCheckpointProcessRestore, pid)


def unlock(pid):
    return _op(_cu.cuCheckpointProcessUnlock, pid)


def pause(pid, timeout_ms=0):
    r = {"lock": lock(pid, timeout_ms)}
    if r["lock"][0] == "OK":
        r["checkpoint"] = checkpoint(pid)
        if r["checkpoint"][0] != "OK":
            r["unlock_after_fail"] = unlock(pid)
    r["state"] = state(pid)
    return r


def resume(pid):
    r = {"restore": restore(pid)}
    if r["restore"][0] == "OK":
        r["unlock"] = unlock(pid)
    r["state"] = state(pid)
    return r


# ---------- probes ----------

def gpu_apps():
    """{pid: MiB} from nvidia-smi's compute-apps list."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        p, m = [x.strip() for x in line.split(",")]
        d[int(p)] = d.get(int(p), 0) + (int(m) if m.isdigit() else 0)
    return d


def gpu_used_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    u, f = out.strip().splitlines()[0].split(",")
    return int(u), int(f)


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024


def proc_mem(pid):
    d = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k = line.split(":")[0]
            if k in ("VmRSS", "VmLck", "VmPin", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
                d[k] = int(line.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return d


def tree_cpu_seconds(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return None
    tot = 0.0
    for q in procs:
        try:
            t = q.cpu_times()
            tot += t.user + t.system
        except psutil.NoSuchProcess:
            pass
    return tot


def driver_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"gpu": out, "cuInit": err(_INIT_RC), "host": os.uname().nodename}


if __name__ == "__main__":
    cmd, pid = sys.argv[1], int(sys.argv[2])
    extra = [int(x) for x in sys.argv[3:]]
    fn = {"state": state, "lock": lock, "checkpoint": checkpoint, "restore": restore,
          "unlock": unlock, "pause": pause, "resume": resume}[cmd]
    print(json.dumps({"cmd": cmd, "pid": pid, "result": fn(pid, *extra)}), flush=True)
