"""Deterministic GPU workloads used as pause/resume targets.

Every workload writes one line per step to --log: "<step> <value> <unix time>", where <value> is an
exact representation (float.hex or a hash), then a final "END <checksums>" line. Two runs with the
same arguments must produce identical <step> <value> columns; a paused run is compared against that.
"""
import argparse, hashlib, os, sys, threading, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ballast_gb", type=float, default=0.0)
p.add_argument("--step_sleep", type=float, default=0.0)
p.add_argument("--log", required=True)
p.add_argument("--stop_file", default="")
a = p.parse_args()

torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
LOG = open(a.log, "w", buffering=1)
_lk = threading.Lock()


def emit(step, val):
    with _lk:
        LOG.write(f"{step} {val} {time.time():.4f}\n")


# ---------- ballast: a large block of random int32 with exact integrity checks ----------
CH = 2 ** 24  # 64 MB of int32 per chunk


def make_ballast(gb):
    n = int(gb * 2 ** 30) // 4
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)
    chunks = []
    while n > 0:
        k = min(CH, n)
        chunks.append(torch.randint(-2 ** 31, 2 ** 31 - 1, (k,), dtype=torch.int32, device=dev, generator=g))
        n -= k
    return chunks


def ballast_sig(chunks):
    """Per-chunk int64 sums plus a fixed sample of 4096 raw values per chunk (exact)."""
    sums, samples = [], []
    for i, c in enumerate(chunks):
        sums.append(int(c.to(torch.int64).sum().item()))
        idx = torch.arange(0, c.numel(), max(1, c.numel() // 4096), device=dev)[:4096]
        samples.append(hashlib.sha1(c[idx].cpu().numpy().tobytes()).hexdigest())
    return hashlib.sha1(repr((sums, samples)).encode()).hexdigest()


ballast = make_ballast(a.ballast_gb) if a.ballast_gb > 0 else []
sig0 = ballast_sig(ballast) if ballast else "none"
torch.cuda.synchronize()


def params_sig(*mods):
    h = hashlib.sha1()
    for m in mods:
        for t in m.state_dict().values():
            h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def gen():
    return torch.Generator(device=dev).manual_seed(a.seed + 7)


def run_mlp(compile_=False):
    m = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(),
                      nn.Linear(2048, 1024)).to(dev)
    f = torch.compile(m) if compile_ else m
    opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
    g = gen()
    for s in range(a.steps):
        x = torch.randn(256, 1024, device=dev, generator=g)
        loss = F.mse_loss(f(x), x.flip(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        emit(s, loss.item().hex())
        if a.step_sleep:
            time.sleep(a.step_sleep)
    return params_sig(m)


def run_amp_tf():
    V, d = 4096, 512
    emb, head = nn.Embedding(V, d).to(dev), nn.Linear(d, V).to(dev)
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2048, dropout=0.1, batch_first=True),
                                4).to(dev)
    params = list(emb.parameters()) + list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    g = gen()
    for s in range(a.steps):
        for _ in range(2):  # gradient accumulation
            x = torch.randint(0, V, (16, 128), device=dev, generator=g)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head(enc(emb(x)))
                loss = F.cross_entropy(logits.float().view(-1, V), x.roll(1, 1).view(-1)) / 2
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        emit(s, f"{loss.item().hex()}|{scaler.get_scale()}")
    return params_sig(emb, enc, head)


def run_cnn():
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    m = nn.Sequential(block(3, 64), block(64, 128), block(128, 256), block(256, 256),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10)).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
    g = gen()
    for s in range(a.steps):
        x = torch.randn(64, 3, 64, 64, device=dev, generator=g)
        y = torch.randint(0, 10, (64,), device=dev, generator=g)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        emit(s, loss.item().hex())
    return params_sig(m)


def run_cudagraph():
    m = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096), nn.GELU(),
                      nn.Linear(4096, 1024)).to(dev).eval()
    static_in = torch.zeros(512, 1024, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            m(static_in)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_out = m(static_in)
    g = gen()
    for s in range(a.steps):
        static_in.copy_(torch.randn(512, 1024, device=dev, generator=g))
        graph.replay()
        emit(s, static_out.double().sum().item().hex())
    return params_sig(m)


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 10 ** 6

    def __getitem__(self, i):
        r = np.random.RandomState(i)
        return torch.from_numpy(r.randn(1024).astype(np.float32)), torch.from_numpy(r.randn(1024).astype(np.float32))


def run_loader():
    m = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    gl = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(DS(), batch_size=256, shuffle=True, generator=gl, num_workers=4,
                                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    for s in range(a.steps):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = F.mse_loss(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        emit(s, loss.item().hex())
    return params_sig(m)


def run_threads():
    mods, out = [nn.Linear(2048, 2048).to(dev) for _ in range(2)], {}  # init in main thread

    def worker(t):
        m = mods[t]
        opt = torch.optim.SGD(m.parameters(), lr=1e-4)
        g = torch.Generator(device=dev).manual_seed(a.seed + 100 + t)
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            for s in range(a.steps):
                x = torch.randn(512, 2048, device=dev, generator=g)
                loss = (m(x) - x).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                emit(f"{s}.{t}", loss.item().hex())
        out[t] = params_sig(m)

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return f"{out[0]}|{out[1]}"


def run_hf():
    import glob
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
    h = hashlib.sha1()
    for s in range(a.steps):
        ids = tok(f"Write one sentence about the number {s}.", return_tensors="pt").to(dev)
        with torch.no_grad():
            o = m.generate(**ids, max_new_tokens=40, do_sample=False)
        v = hashlib.sha1(o.cpu().numpy().tobytes()).hexdigest()[:16]
        h.update(v.encode())
        emit(s, v)
    return h.hexdigest()


def run_ballast():
    x = torch.randn(1024, 1024, device=dev, generator=gen())
    acc = torch.zeros((), device=dev, dtype=torch.float64)
    for s in range(a.steps):
        if a.stop_file and os.path.exists(a.stop_file):
            break
        x = torch.tanh(x @ x.T / 32)
        acc += x.double().sum()
        emit(s, acc.item().hex())
        if a.step_sleep:
            time.sleep(a.step_sleep)
    return acc.item().hex()


def run_longkernel():
    """Each step is one ~7 s fp64 GEMM, so a pause request arrives while a kernel is in flight."""
    n = 16384
    g = gen()
    x = torch.randn(n, n, device=dev, dtype=torch.float64, generator=g)
    y = torch.randn(n, n, device=dev, dtype=torch.float64, generator=g)
    for s in range(a.steps):
        emit(f"{s}", "launch")
        z = x @ y
        emit(s, z[::997, ::991].sum().item().hex())
    return "ok"


def run_managed():
    """Ballast job that also holds 64 MB of CUDA managed (unified) memory, allocated through the driver API."""
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    torch.zeros(1, device=dev)  # makes PyTorch's context current on this thread
    ptr = ctypes.c_uint64()
    rc = lib.cuMemAllocManaged(ctypes.byref(ptr), ctypes.c_size_t(64 * 2 ** 20), ctypes.c_uint(1))
    if rc != 0:
        raise RuntimeError(f"cuMemAllocManaged failed rc={rc}")
    rc = lib.cuMemsetD32_v2(ptr, ctypes.c_uint(7), ctypes.c_size_t(16 * 2 ** 20))
    if rc != 0:
        raise RuntimeError(f"cuMemsetD32 on managed memory failed rc={rc}")
    return run_ballast()


def run_parent_child():
    """Parent and a child process each hold their own CUDA context and ballast."""
    import subprocess
    child = subprocess.Popen([sys.executable, __file__, "--kind", "ballast", "--ballast_gb", "1",
                              "--steps", str(a.steps), "--step_sleep", str(a.step_sleep),
                              "--seed", str(a.seed + 1), "--log", a.log + ".child", "--stop_file", a.stop_file])
    open(a.log + ".childpid", "w").write(str(child.pid))
    r = run_ballast()
    child.wait()
    return f"{r}|child_rc={child.returncode}"


def run_paged():
    """bitsandbytes paged optimizer: its state lives in CUDA managed (unified) memory."""
    import bitsandbytes as bnb
    m = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).to(dev)
    opt = bnb.optim.PagedAdamW32bit(m.parameters(), lr=1e-4)
    g = gen()
    for s in range(a.steps):
        x = torch.randn(256, 1024, device=dev, generator=g)
        loss = F.mse_loss(m(x), x.flip(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        emit(s, loss.item().hex())
        if a.step_sleep:
            time.sleep(a.step_sleep)
        if a.stop_file and os.path.exists(a.stop_file):
            break
    return params_sig(m)


def run_tail():
    """A few GPU steps, then 3 s of CPU-only work; with TAIL_CUDA=1 one more CUDA call before exit."""
    x = torch.randn(1024, 1024, device=dev, generator=gen())
    for s in range(30):
        x = torch.tanh(x @ x.T / 32)
        emit(s, x.double().sum().item().hex())
    time.sleep(3)
    if os.environ.get("TAIL_CUDA") == "1":
        v = x.sum().item()
        with _lk:
            LOG.write(f"TAIL cuda_value={v} t={time.time():.3f}\n")
    return "tail"


def run_tail2():
    """30 GPU steps, 3 s of CPU work, then exactly one operation chosen by TAIL_OP, then exit."""
    x = torch.randn(1024, 1024, device=dev, generator=gen())
    for s in range(30):
        x = torch.tanh(x @ x.T / 32)
        emit(s, x.double().sum().item().hex())
    time.sleep(3)
    op = os.environ["TAIL_OP"]
    t = time.time()
    if op == "sync":
        torch.cuda.synchronize()
    elif op == "memcpy":
        x.cpu()
    elif op == "kernel_item":
        x.sum().item()
    elif op == "launch_only":
        y = x * 2  # asynchronous launch, no sync
    elif op == "alloc":
        torch.empty(256 * 2 ** 20, dtype=torch.uint8, device=dev)
    elif op == "guarded_sync":  # mitigation idea: a tiny launch first (it blocks while paused), then sync
        torch.empty(1, device=dev).add_(1)
        torch.cuda.synchronize()
    elif op == "stream_sync":
        torch.cuda.current_stream().synchronize()
    elif op == "event_sync":
        e = torch.cuda.Event()
        e.record()
        e.synchronize()
    elif op == "sync_in_flight":
        n = 16384
        a_ = torch.randn(n, n, device=dev, dtype=torch.float64)
        with _lk:
            LOG.write(f"TAIL launching t={time.time():.3f}\n")
        LOG.flush()
        b_ = a_ @ a_  # ~7 s kernel; the pause arrives while this thread waits in cudaDeviceSynchronize
        torch.cuda.synchronize()
    elif op == "exit":
        with _lk:
            LOG.write(f"TAIL op=exit returning t={time.time():.3f}\n")
        LOG.flush()
        sys.exit(0)  # normal interpreter shutdown, no explicit CUDA call
    with _lk:
        LOG.write(f"TAIL op={op} done after {time.time() - t:.2f}s at {time.time():.3f}\n")
    LOG.flush()
    if op == "sync_in_flight":  # stay alive, then touch the GPU once more
        time.sleep(5)
        v = x.sum().item()
        with _lk:
            LOG.write(f"TAIL after-sync kernel ok value={v} at {time.time():.3f}\n")
        LOG.flush()
    os._exit(0)


def run_nccl1():
    """Single-GPU job that still initialises torch.distributed with NCCL (as torchrun/Accelerate do)."""
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + a.seed + os.getpid() % 1000))
    dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
    m = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 1024)).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    g = gen()
    for s in range(a.steps):
        x = torch.randn(256, 1024, device=dev, generator=g)
        loss = F.mse_loss(m(x), x.flip(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p_ in m.parameters():
            dist.all_reduce(p_.grad)
        opt.step()
        emit(s, loss.item().hex())
    dist.destroy_process_group()
    return params_sig(m)


kinds = {"managed": run_managed, "nccl1": run_nccl1, "tail2": run_tail2, "tail": run_tail, "paged": run_paged, "mlp": run_mlp, "compile": lambda: run_mlp(True), "amp_tf": run_amp_tf, "cnn": run_cnn,
         "cudagraph": run_cudagraph, "loader": run_loader, "threads": run_threads, "hf": run_hf,
         "ballast": run_ballast, "longkernel": run_longkernel, "parent_child": run_parent_child}

if __name__ == "__main__":
    final = kinds[a.kind]()
    torch.cuda.synchronize()
    sig1 = ballast_sig(ballast) if ballast else "none"
    LOG.write(f"END final={final} ballast_ok={sig0 == sig1} ballast={sig1}\n")
    LOG.close()
