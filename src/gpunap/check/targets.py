"""Target processes that checks pause and resume.

    python -m gpunap.check.targets <kind> --token T [options]

A target prints READY once its GPU state exists. Long-running kinds then read commands from stdin:
VERIFY reads the GPU data back and prints "VERIFIED ok" or "VERIFIED bad <why>", EXIT ends the
process with exit code 0. Single-shot kinds (sync_first, torch_sync_first, exit_after) print
"CALL <time>" right before their one call and "DONE <time> rc=<code>" after it.

Driver-API kinds need only libcuda. torch_* kinds need PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

MB = 2 ** 20


def say(*parts) -> None:
    print(" ".join(str(p) for p in parts), flush=True)


def serve(verify: Callable[[], Optional[str]], stdin=None, on_exit: Optional[Callable[[], None]] = None) -> int:
    for line in (stdin or sys.stdin):
        cmd = line.strip()
        if cmd == "VERIFY":
            try:
                bad = verify()
            except Exception as e:  # a failed read is a finding, not a crash of the protocol
                bad = f"{type(e).__name__}: {e}"
            say("VERIFIED", "ok" if bad is None else f"bad {bad}")
        elif cmd == "EXIT":
            break
        elif cmd:
            say("UNKNOWN", cmd)
    if on_exit:
        on_exit()
    return 0


def _both(*errs: Optional[str]) -> Optional[str]:
    bad = [e for e in errs if e]
    return "; ".join(bad) if bad else None


# ---------- driver-API kinds ----------

def k_hold(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n = a.mb * MB
    p = c.alloc(n)
    c.fill(p, n, a.seed)
    say("READY", os.getpid())
    return serve(lambda: c.verify(p, n, a.seed))


def k_hog(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n = a.mb * MB
    p = c.alloc(n)
    c.fill(p, n, a.seed)
    say("READY", os.getpid())
    return serve(lambda: None)


def k_managed(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n, m = a.mb * MB, a.managed_mb * MB
    p = c.alloc(n)
    q = c.alloc_managed(m)
    c.fill(p, n, a.seed)
    c.fill(q, m, a.seed + 1)
    say("READY", os.getpid())
    return serve(lambda: _both(c.verify(p, n, a.seed), c.verify(q, m, a.seed + 1)))


def _driver_op(c, op: str, p: int, s):
    from gpunap.check.cudactx import CudaError, check_words, pattern_value
    try:
        if op == "ctx_sync":
            return c.sync()
        if op == "stream_sync":
            return c.stream_sync(s)
        if op == "event_sync":
            return c.event_sync(s)
        if op == "memcpy":
            rc, data = c.read(p, 4096)
            return rc if rc else (check_words(data, pattern_value(0, 0)) or 0)
        if op == "guarded_ctx_sync":  # a tiny launch first: it waits at the lock while paused
            rc = c.memset_async(p, pattern_value(0, 0), 1, s)
            return rc if rc else c.sync()
    except CudaError as e:
        return str(e).replace(" ", "_")
    raise ValueError(op)


def k_sync_first(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n = 64 * MB
    p = c.alloc(n)
    c.fill(p, n, 0)
    s = c.stream()
    say("READY", os.getpid())
    time.sleep(a.sleep)
    say("CALL", f"{time.time():.4f}")
    rc = _driver_op(c, a.op, p, s)
    say("DONE", f"{time.time():.4f}", f"rc={rc}")
    os._exit(0)


def k_exit_after(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n = 64 * MB
    c.fill(c.alloc(n), n, 0)
    say("READY", os.getpid())
    time.sleep(a.sleep)
    say("EXITING", f"{time.time():.4f}")
    sys.exit(0)  # normal interpreter shutdown, no further CUDA call


def k_parent_child(a) -> int:
    from gpunap.check.cudactx import Ctx
    c = Ctx()
    n = a.mb * MB
    p = c.alloc(n)
    c.fill(p, n, a.seed)
    child = subprocess.Popen([sys.executable, "-m", "gpunap.check.targets", "hold", "--token", a.token,
                              "--mb", str(a.mb), "--seed", str(a.seed + 1)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in child.stdout:
        if line.startswith("READY"):
            break
    else:
        say("CHILDFAILED", child.wait())
        return 1
    say("CHILD", child.pid)
    say("READY", os.getpid())

    def verify():
        child.stdin.write("VERIFY\n")
        child.stdin.flush()
        reply = child.stdout.readline().strip()
        return _both(c.verify(p, n, a.seed), None if reply == "VERIFIED ok" else f"child: {reply or 'no reply'}")

    def stop():
        try:
            child.stdin.write("EXIT\n")
            child.stdin.flush()
        except OSError:
            pass
        child.wait(timeout=30)

    return serve(verify, on_exit=stop)


# ---------- PyTorch kinds ----------

def _torch_op(torch, op: str, x):
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


def k_torch_sync_first(a) -> int:
    import torch
    torch.manual_seed(a.seed)
    x = torch.randn(1024, 1024, device="cuda")
    for _ in range(30):
        x = torch.tanh(x @ x.T / 32)
    torch.cuda.synchronize()
    say("READY", os.getpid())
    time.sleep(a.sleep)
    say("CALL", f"{time.time():.4f}")
    rc = _torch_op(torch, a.op, x)
    say("DONE", f"{time.time():.4f}", f"rc={rc}")
    os._exit(0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def k_torch_nccl(a) -> int:
    """A single-GPU NCCL process group, as torchrun and Accelerate set up, with a collective every 20 ms."""
    import torch
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    try:
        dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
    except TypeError:
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=0, world_size=1)
    t = torch.arange(4096, device="cuda", dtype=torch.float32)
    expect = t.sum().item()
    stop, steps = threading.Event(), [0]

    def loop():
        while not stop.is_set():
            y = t.clone()
            dist.all_reduce(y)
            y.sum().item()
            steps[0] += 1
            time.sleep(0.02)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    while steps[0] < 5:
        time.sleep(0.01)
    say("READY", os.getpid())

    def verify():
        y = t.clone()
        dist.all_reduce(y)
        got = y.sum().item()
        before = steps[0]
        time.sleep(0.2)
        if not th.is_alive() or steps[0] == before:
            return "collective loop stopped"
        return None if got == expect else f"all_reduce sum {got} != {expect}"

    def finish():
        stop.set()
        th.join(timeout=10)
        dist.destroy_process_group()

    return serve(verify, on_exit=finish)


def _ipc_child(t, cmd_q, res_q, expect) -> None:
    import torch
    res_q.put("ready")
    while True:
        cmd = cmd_q.get()
        if cmd == "VERIFY":
            res_q.put(int(t.to(torch.int64).sum().item()) == expect)
        else:
            break


def k_torch_ipc(a) -> int:
    """A CUDA tensor shared with a child process through torch.multiprocessing (CUDA IPC)."""
    import torch
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    t = torch.arange(32 * MB, dtype=torch.int32, device="cuda") % 1000
    expect = int(t.to(torch.int64).sum().item())
    cmd_q, res_q = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_ipc_child, args=(t, cmd_q, res_q, expect))
    p.start()
    if res_q.get(timeout=120) != "ready":
        return 1
    say("CHILD", p.pid)
    say("READY", os.getpid())

    def verify():
        own = int(t.to(torch.int64).sum().item()) == expect
        cmd_q.put("VERIFY")
        try:
            child_ok = res_q.get(timeout=60)
        except Exception:
            child_ok = None
        return _both(None if own else "parent sum wrong",
                     None if child_ok else f"child: {'no reply' if child_ok is None else 'sum wrong'}")

    def finish():
        cmd_q.put("EXIT")
        p.join(timeout=30)

    return serve(verify, on_exit=finish)


def k_torch_train(a) -> int:
    """Deterministic MLP training. Prints every step's loss as an exact hex float, then a weight hash."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).cuda()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
    g = torch.Generator(device="cuda").manual_seed(a.seed + 7)
    say("READY", os.getpid())
    for step in range(a.steps):
        x = torch.randn(256, 1024, device="cuda", generator=g)
        loss = F.mse_loss(m(x), x.flip(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        say("STEP", step, loss.item().hex())
        time.sleep(a.step_sleep)
    h = hashlib.sha1()
    for p in m.state_dict().values():
        h.update(p.detach().float().cpu().numpy().tobytes())
    say("END", h.hexdigest())
    return 0


KINDS = {"hold": k_hold, "hog": k_hog, "managed": k_managed, "sync_first": k_sync_first,
         "exit_after": k_exit_after, "parent_child": k_parent_child, "torch_sync_first": k_torch_sync_first,
         "torch_nccl": k_torch_nccl, "torch_ipc": k_torch_ipc, "torch_train": k_torch_train}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpunap.check.targets")
    p.add_argument("kind", choices=sorted(KINDS))
    p.add_argument("--token", required=True)
    p.add_argument("--mb", type=int, default=256)
    p.add_argument("--managed-mb", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op", default="ctx_sync")
    p.add_argument("--sleep", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--step-sleep", type=float, default=0.01)
    a = p.parse_args(argv)
    return KINDS[a.kind](a)


if __name__ == "__main__":
    sys.exit(main())
