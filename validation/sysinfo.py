"""Record the machine facts the findings rely on, without touching the GPU.

python3 sysinfo.py --label gpu1

Kernel module type, driver, kernel release, RAM and swap, PCIe link (read at idle, where the link may
be trained down), and installed PyTorch/NCCL package versions. Writes results/sysinfo.json.
"""
import argparse, importlib.metadata, json, os, platform, subprocess

import resultfile

PCIE = ("pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max")


def module_kind(version_line):
    return "open" if "Open Kernel Module" in version_line else "proprietary"


def meminfo_mb(text):
    out = {}
    for line in text.splitlines():
        k = line.split(":")[0]
        if k in ("MemTotal", "SwapTotal"):
            out[k] = int(line.split()[1]) // 1024
    return out


def pkg(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="machine label, e.g. gpu1 (never the hostname)")
    a = p.parse_args()
    ver = open("/proc/driver/nvidia/version").readline().strip()
    smi = subprocess.run(["nvidia-smi", f"--query-gpu=name,driver_version,{','.join(PCIE)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
    r = {"label": a.label, "gpu": smi[0], "driver": smi[1], "kernel_module": module_kind(ver),
         "pcie_idle": dict(zip(PCIE, smi[2:])), "kernel_release": platform.release(),
         "memory_mb": meminfo_mb(open("/proc/meminfo").read()), "python": platform.python_version(),
         "packages": {n: pkg(n) for n in ("torch", "nvidia-nccl-cu12", "nvidia-nccl-cu13")}}
    os.makedirs("results", exist_ok=True)
    path = resultfile.claim("results/sysinfo.json")
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps(r))
    print("wrote", path)


if __name__ == "__main__":
    main()
