"""Allocate N MiB on the GPU, print READY, and hold it until killed (used to fill VRAM)."""
import sys, time
import torch
x = torch.empty(int(sys.argv[1]) * 2 ** 20, dtype=torch.uint8, device="cuda")
x.fill_(1)
torch.cuda.synchronize()
print("READY", flush=True)
time.sleep(3600)
