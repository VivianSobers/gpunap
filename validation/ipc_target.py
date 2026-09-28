"""Parent shares a CUDA tensor with a child through torch.multiprocessing (legacy CUDA IPC)."""
import os, sys, time
import torch
import torch.multiprocessing as mp


def child(q, log, stop):
    t = q.get()
    f = open(log, "w", buffering=1)
    i = 0
    while not os.path.exists(stop):
        f.write(f"{i} {int(t.to(torch.int64).sum().item())} {time.time():.3f}\n")
        i += 1
        time.sleep(0.05)
    f.write("END\n")


if __name__ == "__main__":
    log, stop = sys.argv[1], sys.argv[2]
    ctx = mp.get_context("spawn")
    t = torch.arange(128 * 2 ** 20, dtype=torch.int32, device="cuda") % 1000
    expect = int(t.to(torch.int64).sum().item())
    q = ctx.Queue()
    c = ctx.Process(target=child, args=(q, log + ".child", stop))
    c.start()
    q.put(t)
    open(log + ".childpid", "w").write(str(c.pid))
    f = open(log, "w", buffering=1)
    i = 0
    while not os.path.exists(stop):
        f.write(f"{i} {int(t.to(torch.int64).sum().item())} {time.time():.3f}\n")
        i += 1
        time.sleep(0.05)
    c.join()
    f.write(f"END expect={expect} child_rc={c.exitcode}\n")
