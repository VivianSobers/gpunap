# Synchronize-crash evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the synchronize-while-paused crash on disk as a minimal, rerunnable reproduction with complete results, so it can be reported to NVIDIA and counted honestly in FINDINGS.md.

**Architecture:** A small helper stops every validation script from overwriting an earlier run's result file. A standalone script, `repro_sync.py`, reproduces the crash with the CUDA driver API alone (no PyTorch) and with PyTorch. For each call it records whether the call landed while the process was checkpointed, so a run that never paused can't be counted. The script runs on gpu1 now, and on gpu2 once there is access.

**Tech Stack:** Python 3.10+ (3.10.12 and torch 2.8.0+cu128 on gpu1), ctypes over `libcuda.so.1`, pytest 9 locally.

**Spec:** `validation/FINDINGS.md` ("What breaks the job", synchronize row, and "Recommendation": report the crash with a minimal reproduction) and the "Caveats in the recorded results" section of `README.md`.

## Global Constraints

- Commit messages are `<type>: <4-6 word description>`, with the user as sole author, no trailers and no body. One logical change per commit.
- Committed files never contain the real hostnames, IPs, account names or home paths of the lab machines. They are called gpu1 (driver 595.84) and gpu2 (driver 580.178.04). `$GPU1` below means the ssh alias for gpu1 in `~/.ssh/config`; never write the alias into a committed file.
- Report only numbers from runs that executed. Each result file records the seed, the full config, the driver and the GPU.
- gpu1 is shared. Check `nvidia-smi` before running and record co-tenant memory use. This plan's tests use under 1 GB of GPU memory.
- The local machine has no NVIDIA GPU. Only pure-Python logic can be tested locally.
- `$PRIVATE_PATTERN` is a regex of the lab machines' real hostnames, IPs, account name and home paths. It is kept outside the repo.

## Review Focus

1. The checkpoint finishes after the child has already made its call (slow driver, busy GPU). The run must be marked invalid, not counted as a crash or a pass. Tested in Task 2 (`test_call_before_checkpoint_is_invalid`).
2. The child dies before it reaches the call (CUDA init failure, out of memory next to a co-tenant). The run is recorded as invalid and the script moves on. Tested in Task 2 (`test_no_call_line_is_invalid`).
3. The child hangs after resume. The parent kills it after 30 s and records "hung after resume". Tested in Task 2 (`test_timeout_is_hung`).
4. A result file with the same name already exists from an earlier run. The new run writes `name-2.json` and leaves the old file untouched. Tested in Task 1.
5. The results directory is missing. `claim` must raise, not loop forever. Tested in Task 1 (`test_missing_directory_raises`).

---

### Task 1: Stop overwriting earlier results

`faults.py` writes `results/fault_<name>.json`, `run_case.py` writes `results/<tag>.json` and `bench_size.py` writes `results/bench_size.json`, each replacing whatever an earlier run left there. That is how gpu2's synchronize-crash results were lost.

**Files:**
- Create: `validation/resultfile.py`
- Create: `validation/test_resultfile.py`
- Modify: `validation/faults.py:6` (import) and `validation/faults.py:711` (write)
- Modify: `validation/run_case.py:8` (import), `:69-70` and `:141` (writes)
- Modify: `validation/bench_size.py:3` (import) and `:48` (write)
- Modify: `.gitignore`

**Interfaces:**
- Produces: `resultfile.claim(path: str) -> str`. It creates the file exclusively and returns the path it created: `path` if that was free, otherwise `<root>-2<ext>`, `<root>-3<ext>` and so on.

- [ ] **Step 1: Write the failing test**

`validation/test_resultfile.py`:

```python
import pytest

from resultfile import claim


def test_free_path_is_claimed_as_is(tmp_path):
    p = tmp_path / "fault_x.json"
    assert claim(str(p)) == str(p)
    assert p.exists()


def test_existing_file_is_left_alone(tmp_path):
    p = tmp_path / "fault_x.json"
    p.write_text("earlier run")
    assert claim(str(p)) == str(tmp_path / "fault_x-2.json")
    assert p.read_text() == "earlier run"


def test_suffix_counts_up(tmp_path):
    (tmp_path / "r.json").write_text("1")
    (tmp_path / "r-2.json").write_text("2")
    assert claim(str(tmp_path / "r.json")) == str(tmp_path / "r-3.json")


def test_repeated_claims_never_collide(tmp_path):
    base = str(tmp_path / "r.json")
    assert len({claim(base) for _ in range(5)}) == 5


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        claim(str(tmp_path / "nope" / "r.json"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd validation && python3 -m pytest -q test_resultfile.py`
Expected: collection error, `ModuleNotFoundError: No module named 'resultfile'`

- [ ] **Step 3: Write the implementation**

`validation/resultfile.py`:

```python
"""Claim a result file without overwriting an earlier run's file."""
import os


def claim(path):
    """Create path exclusively and return it; if it exists, use <root>-2<ext>, <root>-3<ext>, ... instead."""
    root, ext = os.path.splitext(path)
    candidate, n = path, 1
    while True:
        try:
            with open(candidate, "x"):
                return candidate
        except FileExistsError:
            n += 1
            candidate = f"{root}-{n}{ext}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd validation && python3 -m pytest -q test_resultfile.py`
Expected: `5 passed`

- [ ] **Step 5: Use it in the three scripts**

`faults.py`: change `import cu` to `import cu, resultfile`, and in the `__main__` block replace

```python
        json.dump(out, open(f"results/fault_{name}.json", "w"), indent=1, default=str)
```

with

```python
        json.dump(out, open(resultfile.claim(f"results/fault_{name}.json"), "w"), indent=1, default=str)
```

`run_case.py`: change `import cu` to `import cu, resultfile`. Replace both `open(f"results/{tag}.json", "w")` with `open(resultfile.claim(f"results/{tag}.json"), "w")`. Only one of the two runs per invocation, because the first is followed by `sys.exit`.

`bench_size.py`: change `import cu` to `import cu, resultfile`, add `OUT = None` after `CYCLES = 3`, and replace the last line of the loop

```python
    json.dump(out, open("results/bench_size.json", "w"), indent=1, default=str)
```

with

```python
    OUT = OUT or resultfile.claim("results/bench_size.json")
    json.dump(out, open(OUT, "w"), indent=1, default=str)
```

The file is claimed on the first write and rewritten after each size, as before.

`.gitignore`: add the line `.pytest_cache/`.

- [ ] **Step 6: Check the scripts still parse and nothing else writes results directly**

Run: `cd validation && python3 -m py_compile faults.py run_case.py bench_size.py && grep -n 'open(f\?"results/' *.py`
Expected: no compile errors, and grep prints nothing. Every `results/` write now goes through `resultfile.claim(...)`, which the pattern doesn't match.

`summarize.py` still skips suffixed files correctly, because it matches on `/fault_` and `bench_size`.

- [ ] **Step 7: Commit**

```bash
git add .gitignore validation/resultfile.py validation/test_resultfile.py validation/faults.py validation/run_case.py validation/bench_size.py
git commit -m "fix: stop overwriting earlier results"
```

### Task 2: Minimal synchronize-crash reproduction

**Files:**
- Create: `validation/repro_sync.py`
- Create: `validation/test_repro_sync.py`

**Interfaces:**
- Consumes: `resultfile.claim` from Task 1.
- Produces: `python3 repro_sync.py --label gpu1 [--repeats 3] [--apis driver,torch] [--ops ...]`, which writes `results/fault_repro_sync.json` (or a suffixed name). Pure functions `verdict(state, ckpt_done, call_t, done_t, done_rc, resume_start, rc) -> str` and `stamp(lines, tag) -> (float | None, str | None)`.

Each run gets exactly one verdict:
- `invalid: pause did not checkpoint`
- `invalid: child never reached the call`
- `invalid: call came before the checkpoint`
- `hung after resume`
- `crash: SIGSEGV` (or another signal name)
- `error: exit N`
- `returned while checkpointed` (with ` (rc=X)` appended when the call returned an error code)
- `waited for resume`

- [ ] **Step 1: Write the failing test**

`validation/test_repro_sync.py`:

```python
from repro_sync import stamp, verdict


def test_not_checkpointed_is_invalid():
    assert verdict("locked", 10.0, 11.0, 11.1, "0", None, 0) == "invalid: pause did not checkpoint"


def test_no_call_line_is_invalid():
    assert verdict("checkpointed", 10.0, None, None, None, None, 1) == "invalid: child never reached the call"


def test_call_before_checkpoint_is_invalid():
    assert verdict("checkpointed", 10.0, 9.5, 9.6, "0", None, 0) == "invalid: call came before the checkpoint"


def test_timeout_is_hung():
    assert verdict("checkpointed", 10.0, 11.0, None, None, 20.0, "timeout") == "hung after resume"


def test_signal_is_crash():
    assert verdict("checkpointed", 10.0, 11.0, None, None, None, -11) == "crash: SIGSEGV"


def test_nonzero_exit_without_done_is_error():
    assert verdict("checkpointed", 10.0, 11.0, None, None, 20.0, 1) == "error: exit 1"


def test_done_before_resume_returned_while_checkpointed():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "0", 20.0, 0) == "returned while checkpointed"


def test_done_with_no_resume_returned_while_checkpointed():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "0", None, 0) == "returned while checkpointed"


def test_error_code_is_reported():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "999", None, 0) == "returned while checkpointed (rc=999)"


def test_done_after_resume_waited():
    assert verdict("checkpointed", 10.0, 11.0, 20.5, "0", 20.0, 0) == "waited for resume"


def test_stamp_reads_time_and_rc():
    out = ["READY", "CALL 11.2500", "DONE 20.5000 rc=0"]
    assert stamp(out, "CALL") == (11.25, None)
    assert stamp(out, "DONE") == (20.5, "0")
    assert stamp(out, "MISSING") == (None, None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd validation && python3 -m pytest -q test_repro_sync.py`
Expected: collection error, `ModuleNotFoundError: No module named 'repro_sync'`

- [ ] **Step 3: Write the implementation**

`validation/repro_sync.py`. `libcuda` is loaded lazily, so the module imports on a machine without a GPU and the tests run locally.

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd validation && python3 -m pytest -q test_repro_sync.py test_resultfile.py`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add validation/repro_sync.py validation/test_repro_sync.py
git commit -m "test: add synchronize crash reproduction"
```

### Task 3: Run the reproduction on gpu1

**Files:**
- Create: `validation/results_gpu1/fault_repro_sync.json` (copied back from gpu1)

- [ ] **Step 1: Check the GPU and copy the two files over**

```bash
ssh $GPU1 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
scp validation/repro_sync.py validation/resultfile.py $GPU1:~/gpunap_validation/
```

Expected: at least 2,048 MiB free. The script refuses to start otherwise. If another user's job leaves less, stop and tell the user.

- [ ] **Step 2: Smoke run, one repeat of one op per api**

```bash
ssh $GPU1 'cd ~/gpunap_validation && timeout 300 python3 repro_sync.py --label gpu1 --repeats 1 --ops memcpy'
```

Expected: `driver memcpy 1x waited for resume` and `torch memcpy 1x waited for resume`, matching the memcpy control in `results_gpu1/fault_tail_ops_a.json` (7.70 s, waited). If either run is `invalid: ...`, stop and debug with superpowers:systematic-debugging before the full run. Then delete the smoke result on gpu1 (`rm ~/gpunap_validation/results/fault_repro_sync.json`) so the full run gets the unsuffixed name.

- [ ] **Step 3: Full run**

```bash
ssh $GPU1 'cd ~/gpunap_validation && timeout 1800 python3 repro_sync.py --label gpu1 --repeats 3'
```

This is 30 runs of 15 to 20 s each. For torch `ctx_sync` and `stream_sync`, earlier results predict `crash: SIGSEGV`, and for torch `event_sync`, `memcpy` and `guarded_ctx_sync` they predict `waited for resume`. The driver-API rows are new, and whatever they show is the finding. Write down any row that disagrees with these predictions. Don't explain it away.

- [ ] **Step 4: Copy the result back and check it**

```bash
scp $GPU1:~/gpunap_validation/results/fault_repro_sync.json validation/results_gpu1/fault_repro_sync.json
python3 -c "import json; d=json.load(open('validation/results_gpu1/fault_repro_sync.json')); print(d['label'], d['driver'], len(d['runs']), d['co_tenant_mb'])"
grep -ciE "$PRIVATE_PATTERN" validation/results_gpu1/fault_repro_sync.json
```

Expected: `gpu1 595.84 30 <n>` and a grep count of 0.

- [ ] **Step 5: Commit**

```bash
git add validation/results_gpu1/fault_repro_sync.json
git commit -m "test: add gpu1 synchronize repro results"
```

### Task 4: Correct the documented evidence

**Files:**
- Modify: `validation/FINDINGS.md` (synchronize row of "What breaks the job", the Verdict bullet, "Rerunning")
- Modify: `README.md` (synchronize bullet and "Caveats in the recorded results")

- [ ] **Step 1: Rewrite the synchronize row from the results file**

Print the counts:

```bash
python3 -c "
import json, collections
d = json.load(open('validation/results_gpu1/fault_repro_sync.json'))
c = collections.Counter((r['api'], r['op'], r['verdict']) for r in d['runs'])
[print(k, n) for k, n in sorted(c.items())]"
```

In FINDINGS.md, replace the count cell `device sync 4/4 and stream sync 3/3 on 595; 2/2 and 2/2 on 580` with the counts that exist on disk. From the original run, that is `device sync 3/3 and stream sync 3/3 on 595 (results_gpu1/fault_tail_ops_*)`. Add the reproduction's torch and driver-API counts from the output above, and add `580: reported from the original run but the result files were overwritten; not rerun yet`. Add a sentence to the "What happened" cell stating whether the plain driver API (`cuCtxSynchronize`, `cuStreamSynchronize`) crashes too, using the driver rows. In the Verdict bullet, change "This happened 11 times out of 11 on both drivers" to the on-disk count and remove "on both drivers" unless gpu2 has been rerun.

- [ ] **Step 2: Add the reproduction to "Rerunning"**

Add to the code block in FINDINGS.md:

```
python3 repro_sync.py --label gpu1                 # synchronize while checkpointed, driver API and PyTorch
```

- [ ] **Step 3: Update README.md**

Change the synchronize bullet under "Pausing an arbitrary job is unsafe" to say whether the driver API crashes too, from the results. Replace the first caveat bullet with the corrected on-disk counts. Keep the 580 gap stated until gpu2 is rerun. Run the humanizer skill (embedded mode) over the changed README prose.

- [ ] **Step 4: Check the numbers**

Every count in the changed lines of both files must match the Step 1 output or a file under `results_gpu1/`. Read the diff line by line against them: `git diff -U0 README.md validation/FINDINGS.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md validation/FINDINGS.md
git commit -m "docs: correct synchronize crash evidence"
```

## Out of scope, needs the user

- **gpu2 (driver 580).** There is no ssh access to it from this machine. With access, the same two files and `python3 repro_sync.py --label gpu2` close the 580 gap.
- **Reporting to NVIDIA.** Filing the issue is public and is the user's call. `repro_sync.py` plus the results file is the attachment.
- **Pushing.** Push to `origin main` after Task 4, once the user has seen the results.
