"""Read-only /proc probes that need no CUDA, so they can be unit-tested anywhere."""
import os


def uvm_maps(pid, proc="/proc"):
    """Mappings of /dev/nvidia-uvm in the process (a candidate signal for managed memory), or None if gone."""
    try:
        return [l.split()[0] + " " + l.split()[-1] for l in open(f"{proc}/{pid}/maps") if "nvidia-uvm" in l]
    except FileNotFoundError:
        return None


def thread_names(pid, proc="/proc"):
    """Sorted thread names (comm) of every task in the process, or None if gone."""
    try:
        tids = os.listdir(f"{proc}/{pid}/task")
    except FileNotFoundError:
        return None
    names = []
    for tid in tids:
        try:
            names.append(open(f"{proc}/{pid}/task/{tid}/comm").read().strip())
        except FileNotFoundError:
            pass
    return sorted(names)
