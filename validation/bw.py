"""Raw host<->device copy bandwidth (pinned and pageable), to compare against checkpoint/restore speed."""
import json, time, torch
GB = 4
x = torch.empty(GB * 2 ** 30, dtype=torch.uint8, device="cuda")
x.fill_(1)
out = {"torch": torch.__version__}
for pinned in (True, False):
    h = torch.empty(x.numel(), dtype=torch.uint8, pin_memory=pinned)
    h.fill_(2)
    r = []
    for _ in range(3):
        torch.cuda.synchronize(); t = time.perf_counter(); h.copy_(x); torch.cuda.synchronize(); d2h = time.perf_counter() - t
        t = time.perf_counter(); x.copy_(h); torch.cuda.synchronize(); h2d = time.perf_counter() - t
        r.append((round(GB * 1.073741824 / d2h, 2), round(GB * 1.073741824 / h2d, 2)))
    out["pinned" if pinned else "pageable"] = {"d2h_GBps,h2d_GBps": r}
    del h
print(json.dumps(out))
