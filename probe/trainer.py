import os, sys, time, torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0); torch.use_deterministic_algorithms(True)
out = open(sys.argv[1], "w"); steps = int(sys.argv[2])
dev = "cuda"
model = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 1024)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ballast = torch.randn(128 * 1024 * 1024, device=dev)          # 512 MB of state that must survive
g = torch.Generator(device=dev); g.manual_seed(1)
for step in range(steps):
    x = torch.randn(256, 1024, device=dev, generator=g)
    loss = ((model(x) - x.flip(1)) ** 2).mean() + 1e-12 * ballast[step]
    opt.zero_grad(); loss.backward(); opt.step()
    out.write(f"{step} {loss.item():.10e} {time.time():.3f}\n"); out.flush()
    time.sleep(0.01)
chk = sum(p.double().sum().item() for p in model.parameters()) + ballast.double().sum().item()
out.write(f"CHECKSUM {chk:.12e}\n"); out.close()
