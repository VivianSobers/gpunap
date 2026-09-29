"""Capture a backtrace of the synchronize-while-checkpointed crash.

python3 backtrace_sync.py --label gpu1 [--op ctx_sync]

Runs `repro_sync.py child --api driver --op <op>` under gdb (so it works with ptrace_scope=1),
checkpoints the child once it prints READY, and saves gdb's output with the driver, GPU and pause
results to results/backtrace_sync_<op>.json.
"""
import argparse, json, os, subprocess, sys, threading, time

import repro_sync
import resultfile

GDB = ["gdb", "-q", "-nx", "-batch",
       "-ex", "set pagination off",
       "-ex", "handle SIGSEGV stop print nopass",
       "-ex", "handle SIGUSR1 SIGUSR2 SIGPIPE SIG32 SIG33 SIG34 SIG35 SIG36 nostop noprint pass",
       "-ex", "run",
       "-ex", "info registers rip",
       "-ex", "print $_siginfo._sifields._sigfault.si_addr",
       "-ex", "bt",
       "-ex", "thread apply all bt",
       "-ex", "kill"]


def children_of(pid, proc="/proc"):
    """PIDs whose parent is pid, read from <proc>/<n>/stat."""
    out = []
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            stat = open(os.path.join(proc, name, "stat")).read()
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
            out.append(int(name))
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="machine label, e.g. gpu1 (never the hostname)")
    p.add_argument("--op", default="ctx_sync", choices=repro_sync.OPS)
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = GDB + ["--args", sys.executable, os.path.join(here, "repro_sync.py"), "child", "--api", "driver", "--op", a.op]
    gdb = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, ready = [], threading.Event()

    def drain():
        for line in gdb.stdout:
            out.append(line.rstrip())
            if line.startswith("READY"):
                ready.set()

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    r = {"label": a.label, "op": a.op, "api": "driver"}
    if not ready.wait(timeout=120):
        gdb.kill()
        r["error"] = "child never printed READY"
    else:
        kids = children_of(gdb.pid)
        r["inferior_pids"] = kids
        pid = kids[0]
        time.sleep(repro_sync.PAUSE_AT_S)
        r["lock"] = repro_sync.call("cuCheckpointProcessLock", pid)
        if r["lock"]["result"] == "OK":
            r["checkpoint"] = repro_sync.call("cuCheckpointProcessCheckpoint", pid)
        r["state_after_pause"] = repro_sync.state(pid)
        try:
            gdb.wait(timeout=repro_sync.WAIT_S + 60)
        except subprocess.TimeoutExpired:
            r["restore"] = repro_sync.call("cuCheckpointProcessRestore", pid)
            r["unlock"] = repro_sync.call("cuCheckpointProcessUnlock", pid)
            gdb.wait(timeout=60)
    th.join(timeout=5)
    call_t, _ = repro_sync.stamp(out, "CALL")
    r["call_after_checkpoint_s"] = (round(call_t - r["checkpoint"]["end"], 3)
                                    if call_t and "checkpoint" in r else None)
    r["sigsegv_seen"] = any("SIGSEGV" in line for line in out)
    name, drv = [x.strip() for x in repro_sync.nvsmi("--query-gpu=name,driver_version")[0].split(",")]
    r.update(gpu=name, driver=drv, gdb=subprocess.run(["gdb", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
             gdb_commands=GDB, gdb_output=out)
    os.makedirs("results", exist_ok=True)
    path = resultfile.claim(f"results/backtrace_sync_{a.op}.json")
    json.dump(r, open(path, "w"), indent=1)
    print("sigsegv_seen", r["sigsegv_seen"], "call_after_checkpoint_s", r["call_after_checkpoint_s"])
    print("wrote", path)


if __name__ == "__main__":
    main()
