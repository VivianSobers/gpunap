"""Minimal reproduction: one CUDA call made while the calling process is checkpointed.

python3 repro_sync.py --label gpu1 [--repeats 3] [--apis driver,torch]
                      [--ops ctx_sync,stream_sync,event_sync,memcpy,guarded_ctx_sync]

For each repeat, api and op, the parent starts a child that does a little GPU work, prints READY,
sleeps 3 s and then makes exactly one CUDA call. The parent locks and checkpoints the child 0.5 s
after READY, so the call lands while the child is checkpointed. If the child is still alive 10 s
after the pause, the parent restores and unlocks it. All runs go into one JSON file under results/.
"""
import argparse, ctypes, importlib.metadata, json, os, platform, signal, subprocess, sys, threading, time

import resultfile

SEED = 0
SLEEP_S, PAUSE_AT_S, WAIT_S, AFTER_RESUME_S = 3.0, 0.5, 10.0, 30.0
OPS = ("ctx_sync", "stream_sync", "event_sync", "memcpy", "guarded_ctx_sync")
STATE = {0: "running", 1: "locked", 2: "checkpointed", 3: "failed"}
_cu = None


def cuda():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
        _cu.cuInit(0)
    return _cu


def ck(rc, what):
    if rc != 0:
        sys.exit(f"{what} failed rc={rc}")


def err(rc):
    if rc == 0:
        return "OK"
    p = ctypes.c_char_p()
    return p.value.decode() if cuda().cuGetErrorName(rc, ctypes.byref(p)) == 0 and p.value else f"rc{rc}"


def call(fn, pid):
    t = time.time()
    rc = getattr(cuda(), fn)(pid, None)
    return {"result": err(rc), "start": round(t, 4), "end": round(time.time(), 4)}


def state(pid):
    s = ctypes.c_int(-1)
    rc = cuda().cuCheckpointProcessGetState(pid, ctypes.byref(s))
    return STATE.get(s.value, s.value) if rc == 0 else err(rc)


# ---------- child ----------

def child_driver(op):
    cu = cuda()
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    n = 64 * 2 ** 20  # 256 MB of 32-bit words
    buf, stream = ctypes.c_uint64(), ctypes.c_void_p()
    ck(cu.cuMemAlloc_v2(ctypes.byref(buf), ctypes.c_size_t(n * 4)), "cuMemAlloc")
    ck(cu.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
    ck(cu.cuMemsetD32Async(buf, ctypes.c_uint(0x1234ABCD), ctypes.c_size_t(n), stream), "cuMemsetD32Async")
    ck(cu.cuStreamSynchronize(stream), "cuStreamSynchronize")

    def run():
        if op == "ctx_sync":
            return cu.cuCtxSynchronize()
        if op == "stream_sync":
            return cu.cuStreamSynchronize(stream)
        if op == "event_sync":
            ev = ctypes.c_void_p()
            ck(cu.cuEventCreate(ctypes.byref(ev), 0), "cuEventCreate")
            ck(cu.cuEventRecord(ev, stream), "cuEventRecord")
            return cu.cuEventSynchronize(ev)
        if op == "memcpy":
            host = (ctypes.c_uint32 * 1024)()
            rc = cu.cuMemcpyDtoH_v2(host, buf, ctypes.c_size_t(4096))
            return rc if rc else (0 if all(v == 0x1234ABCD for v in host) else "data_mismatch")
        if op == "guarded_ctx_sync":  # mitigation: a tiny launch first, which waits while paused
            ck(cu.cuMemsetD32Async(buf, ctypes.c_uint(0x1234ABCD), ctypes.c_size_t(1), stream), "guard launch")
            return cu.cuCtxSynchronize()
        raise ValueError(op)
    return run


def child_torch(op):
    import torch
    torch.manual_seed(SEED)
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(30):
        x = torch.tanh(x @ x.T / 32)
    torch.cuda.synchronize()

    def run():
        if op == "ctx_sync":
            torch.cuda.synchronize()
        elif op == "stream_sync":
            torch.cuda.current_stream().synchronize()
        elif op == "event_sync":
            e = torch.cuda.Event()
            e.record()
            e.synchronize()
        elif op == "memcpy":
            x.cpu()
        elif op == "guarded_ctx_sync":
            torch.empty(1, device="cuda").add_(1)
            torch.cuda.synchronize()
        else:
            raise ValueError(op)
        return 0
    return run


def child(api, op):
    run = (child_driver if api == "driver" else child_torch)(op)
    print("READY", flush=True)
    time.sleep(SLEEP_S)
    print(f"CALL {time.time():.4f}", flush=True)
    rc = run()
    print(f"DONE {time.time():.4f} rc={rc}", flush=True)
    os._exit(0)


# ---------- parent ----------

def stamp(lines, tag):
    """(time, rc) from the first '<tag> <time> [rc=<rc>]' line, or (None, None)."""
    for line in lines:
        f = line.split()
        if f and f[0] == tag:
            rc = next((x[3:] for x in f[2:] if x.startswith("rc=")), None)
            return float(f[1]), rc
    return None, None


def verdict(state, ckpt_done, call_t, done_t, done_rc, resume_start, rc):
    """Classify one run. Times are unix seconds; rc is the child's return code or "timeout"."""
    if state != "checkpointed":
        return "invalid: pause did not checkpoint"
    if call_t is None:
        return "invalid: child never reached the call"
    if ckpt_done is None or call_t < ckpt_done:
        return "invalid: call came before the checkpoint"
    if rc == "timeout":
        return "hung after resume"
    if rc < 0:
        return f"crash: {signal.Signals(-rc).name}"
    if done_t is None:
        return f"error: exit {rc}"
    if resume_start is None or done_t < resume_start:
        return "returned while checkpointed" + ("" if done_rc == "0" else f" (rc={done_rc})")
    return "waited for resume"


def run_one(api, op):
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), "child", "--api", api, "--op", op],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, ready = [], threading.Event()

    def drain():
        for line in pr.stdout:
            out.append(line.rstrip())
            if line.startswith("READY"):
                ready.set()

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    r = {"api": api, "op": op}
    if not ready.wait(timeout=120):
        pr.kill()
        pr.wait()
        th.join(timeout=5)
        r.update(child_rc=pr.returncode, output=out[-20:], verdict="invalid: child never became ready")
        return r
    time.sleep(PAUSE_AT_S)
    r["lock"] = call("cuCheckpointProcessLock", pr.pid)
    if r["lock"]["result"] == "OK":
        r["checkpoint"] = call("cuCheckpointProcessCheckpoint", pr.pid)
        if r["checkpoint"]["result"] != "OK":
            r["unlock_after_fail"] = call("cuCheckpointProcessUnlock", pr.pid)
    r["state_after_pause"] = state(pr.pid) if pr.poll() is None else "exited"
    try:
        rc = pr.wait(timeout=WAIT_S)
        r["exited_while_paused"] = True
    except subprocess.TimeoutExpired:
        r["exited_while_paused"] = False
        r["restore"] = call("cuCheckpointProcessRestore", pr.pid)
        r["unlock"] = call("cuCheckpointProcessUnlock", pr.pid)
        try:
            rc = pr.wait(timeout=AFTER_RESUME_S)
        except subprocess.TimeoutExpired:
            pr.kill()
            pr.wait()
            rc = "timeout"
    th.join(timeout=5)
    call_t, _ = stamp(out, "CALL")
    done_t, done_rc = stamp(out, "DONE")
    r.update(child_rc=rc, output=out[-20:], verdict=verdict(
        r["state_after_pause"], r.get("checkpoint", {}).get("end"), call_t, done_t, done_rc,
        r.get("restore", {}).get("start"), rc))
    return r


def nvsmi(*args):
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="parent", choices=("parent", "child"))
    p.add_argument("--api", default="driver", choices=("driver", "torch"))
    p.add_argument("--op", default="ctx_sync", choices=OPS)
    p.add_argument("--label", help="machine label for the results, e.g. gpu1 (never the hostname)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--apis", default="driver,torch")
    p.add_argument("--ops", default=",".join(OPS))
    p.add_argument("--min_free_mb", type=int, default=2048)
    a = p.parse_args()
    if a.mode == "child":
        child(a.api, a.op)
    if not a.label:
        sys.exit("--label is required")
    name, drv, used, free = [x.strip() for x in nvsmi("--query-gpu=name,driver_version,memory.used,memory.free")[0].split(",")]
    if int(free) < a.min_free_mb:
        sys.exit(f"only {free} MiB free on the GPU, need {a.min_free_mb}")
    others = [int(x) for x in nvsmi("--query-compute-apps=used_memory") if x.strip().isdigit()]
    api_ver = ctypes.c_int()
    cuda().cuDriverGetVersion(ctypes.byref(api_ver))
    try:
        torch_ver = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_ver = None
    head = {"label": a.label, "gpu": name, "driver": drv, "cuda_driver_api_version": api_ver.value,
            "gpu_mb_used_before": int(used), "gpu_mb_free_before": int(free),
            "co_tenant_apps": len(others), "co_tenant_mb": sum(others),
            "python": platform.python_version(), "torch": torch_ver, "seed": SEED,
            "config": {"repeats": a.repeats, "apis": a.apis, "ops": a.ops, "sleep_s": SLEEP_S,
                       "pause_at_s": PAUSE_AT_S, "wait_s": WAIT_S, "after_resume_s": AFTER_RESUME_S},
            "started": round(time.time(), 1)}
    runs = []
    for rep in range(a.repeats):
        for api in a.apis.split(","):
            for op in a.ops.split(","):
                r = run_one(api, op)
                r["repeat"] = rep
                runs.append(r)
                print(json.dumps({k: r.get(k) for k in ("repeat", "api", "op", "child_rc", "verdict")}), flush=True)
    os.makedirs("results", exist_ok=True)
    path = resultfile.claim("results/fault_repro_sync.json")
    json.dump({**head, "runs": runs, "finished": round(time.time(), 1)}, open(path, "w"), indent=1)
    counts = {}
    for r in runs:
        key = (r["api"], r["op"], r["verdict"])
        counts[key] = counts.get(key, 0) + 1
    for (api, op, v), n in sorted(counts.items()):
        print(f"{api:6s} {op:17s} {n}x {v}")
    print("wrote", path)


if __name__ == "__main__":
    main()
