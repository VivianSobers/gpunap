"""PCIe link generation and width at idle and under a host-to-device copy load.

python3 pcie_link.py --label gpu1

The link can train down when the GPU is idle, so the reading that matters is the one under load.
Writes results/pcie_link.json.
"""
import argparse, json, os, subprocess, threading, time

import resultfile

FIELDS = ("pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.gen.gpumax", "pcie.link.gen.hostmax",
          "pcie.link.width.current", "pcie.link.width.max")
SEED = 0


def parse(line):
    vals = [v.strip() for v in line.split(",")]
    return {k: int(v) if v.isdigit() else v for k, v in zip(FIELDS, vals)}


def query():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    return parse(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="machine label, e.g. gpu1 (never the hostname)")
    p.add_argument("--load_s", type=float, default=6.0)
    a = p.parse_args()
    import torch
    torch.manual_seed(SEED)
    name, drv = [x.strip() for x in subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines()[0].split(",")]
    r = {"label": a.label, "gpu": name, "driver": drv, "torch": torch.__version__, "seed": SEED,
         "config": {"load_s": a.load_s, "copy_mb": 1024, "pinned": True}, "idle": query()}
    x = torch.empty(2 ** 30, dtype=torch.uint8, device="cuda")
    h = torch.empty(2 ** 30, dtype=torch.uint8, pin_memory=True)
    stop, copies = threading.Event(), []

    def load():
        while not stop.is_set():
            t = time.perf_counter()
            x.copy_(h)
            torch.cuda.synchronize()
            copies.append(time.perf_counter() - t)

    th = threading.Thread(target=load)
    th.start()
    time.sleep(1.0)
    r["under_load"] = []
    t0 = time.time()
    while time.time() - t0 < a.load_s:
        r["under_load"].append(query())
        time.sleep(0.5)
    stop.set()
    th.join()
    r["h2d_pinned_GBps_median"] = round(1.073741824 / sorted(copies)[len(copies) // 2], 2)
    r["n_copies"] = len(copies)
    os.makedirs("results", exist_ok=True)
    path = resultfile.claim("results/pcie_link.json")
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps({"idle": r["idle"], "load_last": r["under_load"][-1], "h2d": r["h2d_pinned_GBps_median"]}))
    print("wrote", path)


if __name__ == "__main__":
    main()
