"""Fault-injection and edge-case tests for pause/resume. python faults.py NAME [NAME ...]
Each test writes results/fault_<NAME>.json (fault_<NAME>-2.json and so on if it exists). Only processes started here are ever signalled.
"""
import json, os, signal, subprocess, sys, threading, time

import cu, resultfile

PY = sys.executable
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
MINE = []  # every PID started here; cleanup touches only these


def start(name, gb=1.0, kind="ballast", sleep=0.02, steps=10 ** 7, script="target.py", extra=()):
    log, stop = f"logs/f_{name}.log", f"logs/f_{name}.stop"
    for f in (log, stop, log + ".child", log + ".childpid"):
        if os.path.exists(f):
            os.remove(f)
    if script == "target.py":
        c = [PY, script, "--kind", kind, "--ballast_gb", str(gb), "--steps", str(steps),
             "--step_sleep", str(sleep), "--log", log, "--stop_file", stop, *extra]
    else:
        c = [PY, script, log, stop]
    # restore default SIGINT: a shell's background job ignores it, and children would inherit that
    pr = subprocess.Popen(c, stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT,
                          preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    MINE.append(pr.pid)
    return pr, log, stop


def lines(path):
    try:
        return sum(1 for l in open(path) if l[:1].isdigit())
    except FileNotFoundError:
        return 0


def wait_lines(path, n, pr, timeout=300):
    t = time.time()
    while lines(path) < n:
        if pr.poll() is not None or time.time() - t > timeout:
            raise RuntimeError(f"target not progressing ({path}), rc={pr.poll()}")
        time.sleep(0.05)


def finish(pr, log, stop, timeout=120):
    open(stop, "w").close()
    try:
        rc = pr.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    end = [l.strip() for l in open(log) if l.startswith("END")] if os.path.exists(log) else []
    return {"rc": rc, "end": end[0] if end else None}


def ctl(*args, timeout=120):
    """Run one driver call in a separate controller process (so it can hang or be killed)."""
    t = time.time()
    try:
        out = subprocess.run([PY, "cu.py", *map(str, args)], capture_output=True, text=True, timeout=timeout)
        r = json.loads(out.stdout.strip().splitlines()[-1])["result"] if out.stdout.strip() else out.stderr[-500:]
    except subprocess.TimeoutExpired:
        r = f"controller timed out after {timeout}s"
    return {"call": args[0], "result": r, "wall_s": round(time.time() - t, 3)}


def hog(leave_mb):
    """Start a process that fills the GPU until only about leave_mb MiB stay free."""
    _, free = cu.gpu_used_free()
    want = free - leave_mb - 450  # the hog's own CUDA context takes roughly 450 MiB
    if want <= 0:
        return None, {"hog_mb": 0, "note": "not enough free memory to hog"}
    h = subprocess.Popen([PY, "hog.py", str(want)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    MINE.append(h.pid)
    seen = []
    for line in h.stdout:  # skip library warnings until the hog has really allocated
        seen.append(line.strip())
        if line.startswith("READY") or len(seen) > 50:
            break
    return h, {"hog_mb": want, "hog_ready": seen[-1] if seen else None, "free_after_hog": cu.gpu_used_free()[1]}


def kill(pr):
    if pr and pr.poll() is None:
        pr.kill()
        pr.wait()


def gpu_mb(pid):
    return cu.gpu_apps().get(pid, "absent")


# ------------------------------------------------------------------ tests

def t_errors():
    r = {}
    sl = subprocess.Popen(["sleep", "120"])
    MINE.append(sl.pid)
    r["nonexistent_pid"] = {op: ctl(op, 4194000, *([1000] if op == "lock" else [])) for op in ("state", "lock")}
    r["non_cuda_pid"] = {op: ctl(op, sl.pid, *([1000] if op == "lock" else [])) for op in ("state", "lock", "checkpoint")}
    kill(sl)
    pr, log, stop = start("errors", gb=1)
    wait_lines(log, 20, pr)
    pid = pr.pid
    seq = [("checkpoint",), ("restore",), ("unlock",), ("lock", 0), ("lock", 1000), ("checkpoint",),
           ("checkpoint",), ("lock", 1000), ("unlock",), ("restore",), ("unlock",), ("state",)]
    r["wrong_state_sequence"] = []
    for s in seq:
        before = cu.state(pid)
        res = ctl(s[0], pid, *s[1:])
        r["wrong_state_sequence"].append({"op": s, "state_before": before, "res": res["result"], "state_after": cu.state(pid)})
    r["finish"] = finish(pr, log, stop)
    return r


def t_restore_short(leave_frac):
    pr, log, stop = start(f"short{leave_frac}", gb=8)
    wait_lines(log, 20, pr)
    pid = pr.pid
    time.sleep(0.5)
    need = gpu_mb(pid)
    r = {"target_gpu_mb": need, "pause": cu.pause(pid), "free_after_pause": cu.gpu_used_free()[1]}
    h, r["hog"] = hog(int(need * leave_frac))
    r["restore_1"] = cu.restore(pid)
    r["state_after_restore_1"] = cu.state(pid)
    r["target_alive_after_1"] = pr.poll() is None
    r["gpu_mb_target_after_1"] = gpu_mb(pid)
    time.sleep(3)
    r["state_3s_later"] = cu.state(pid)
    kill(h)
    time.sleep(2)
    r["free_after_hog_killed"] = cu.gpu_used_free()[1]
    st = cu.state(pid)
    if st == "checkpointed":
        r["resume_2"] = cu.resume(pid)
    elif st == "locked":
        r["unlock_2"] = cu.unlock(pid)
    r["state_final"] = cu.state(pid)
    r["finish"] = finish(pr, log, stop)
    return r


def t_ctl_killed(phase):
    """Kill the controller process 0.25 s into a checkpoint (or restore) of a 16 GB job."""
    pr, log, stop = start(f"ctlkill_{phase}", gb=16)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"target_gpu_mb": gpu_mb(pid)}
    if phase == "restore":
        r["pause"] = cu.pause(pid)
    op = "pause" if phase == "checkpoint" else "resume"
    c = subprocess.Popen([PY, "cu.py", op, str(pid)], stdout=subprocess.PIPE, text=True)
    MINE.append(c.pid)
    time.sleep(0.25)
    c.kill()
    c.wait()
    r["states_after_kill"] = []
    for _ in range(20):
        r["states_after_kill"].append((round(time.time(), 2), cu.state(pid), gpu_mb(pid)))
        time.sleep(0.5)
    st = cu.state(pid)
    if st == "locked":
        r["recover"] = {"unlock": cu.unlock(pid)}
    elif st == "checkpointed":
        r["recover"] = cu.resume(pid)
    r["state_final"] = cu.state(pid)
    r["finish"] = finish(pr, log, stop)
    return r


def t_target_signal(sig):
    pr, log, stop = start(f"sig_{sig}", gb=4)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"pause": cu.pause(pid), "memavail_paused": cu.mem_available_mb()}
    t = time.time()
    os.kill(pid, getattr(signal, sig))
    try:
        rc = pr.wait(timeout=15)
        r["exited_while_paused"] = {"rc": rc, "after_s": round(time.time() - t, 2)}
    except subprocess.TimeoutExpired:
        r["exited_while_paused"] = None
        r["state_after_15s"] = cu.state(pid)
        r["resume"] = cu.resume(pid)
        try:
            r["exit_after_resume"] = {"rc": pr.wait(timeout=30)}
        except subprocess.TimeoutExpired:
            r["exit_after_resume"] = "still alive 30s after resume"
            r["finish"] = finish(pr, log, stop)
    time.sleep(2)
    r["memavail_after_exit"] = cu.mem_available_mb()
    r["gpu_mb_after_exit"] = gpu_mb(pid)
    r["out_tail"] = open(log + ".out").read()[-400:]
    return r


def t_target_killed_mid_checkpoint():
    pr, log, stop = start("killmid", gb=16)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"target_gpu_mb": gpu_mb(pid), "memavail_before": cu.mem_available_mb()}
    box = {}
    th = threading.Thread(target=lambda: box.update(cu.pause(pid)))
    th.start()
    time.sleep(0.25)
    os.kill(pid, signal.SIGKILL)
    th.join(timeout=120)
    r["pause_result"] = box or "pause call did not return in 120 s"
    pr.wait(timeout=30)
    time.sleep(3)
    r["memavail_after"] = cu.mem_available_mb()
    r["gpu_mb_after"] = gpu_mb(pid)
    r["gpu_used_free_after"] = cu.gpu_used_free()
    return r


def t_lock_timeout():
    pr, log, stop = start("longkernel", gb=0, kind="longkernel", steps=4)
    t0 = time.time()
    while "launch" not in open(log).read() if os.path.exists(log) else True:
        time.sleep(0.05)
        if time.time() - t0 > 300:
            raise RuntimeError("no launch")
    time.sleep(1.0)
    pid = pr.pid
    r = {"lock_1s_timeout": cu.lock(pid, 1000), "state_after_timeout": cu.state(pid)}
    t1 = time.time()
    r["lock_no_timeout"] = cu.lock(pid, 0)
    r["state_after_lock"] = cu.state(pid)
    r["checkpoint"] = cu.checkpoint(pid)
    r["resume"] = cu.resume(pid)
    r["total_s"] = round(time.time() - t1, 2)
    try:
        r["target_rc"] = pr.wait(timeout=200)
    except subprocess.TimeoutExpired:
        kill(pr)
        r["target_rc"] = "timeout"
    r["log"] = open(log).read()[-800:]
    return r


def t_sigstop():
    r = {}
    pr, log, stop = start("sigstop_a", gb=2)
    wait_lines(log, 20, pr)
    pid = pr.pid
    os.kill(pid, signal.SIGSTOP)
    r["A_lock_while_stopped_5s_timeout"] = ctl("lock", pid, 5000, timeout=30)
    r["A_state_while_stopped"] = ctl("state", pid, timeout=10)  # in-process GetState would hang
    r["A_state"] = r["A_state_while_stopped"]["result"]
    if r["A_state"] == "locked":
        r["A_checkpoint_while_stopped"] = ctl("checkpoint", pid, timeout=60)
        r["A_restore_while_stopped"] = ctl("restore", pid, timeout=60)
        r["A_unlock_while_stopped"] = ctl("unlock", pid, timeout=60)
    os.kill(pid, signal.SIGCONT)
    time.sleep(2)
    r["A_state_after_cont"] = cu.state(pid)
    if r["A_state_after_cont"] == "checkpointed":
        r["A_resume_after_cont"] = cu.resume(pid)
    elif r["A_state_after_cont"] == "locked":
        r["A_unlock_after_cont"] = cu.unlock(pid)
    r["A_finish"] = finish(pr, log, stop)

    pr, log, stop = start("sigstop_b", gb=2)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r["B_pause"] = cu.pause(pid)
    os.kill(pid, signal.SIGSTOP)
    r["B_restore_while_stopped"] = ctl("restore", pid, timeout=30)
    r["B_state_while_stopped"] = ctl("state", pid, timeout=10)
    os.kill(pid, signal.SIGCONT)
    time.sleep(2)
    r["B_state_after_cont"] = cu.state(pid)
    if r["B_state_after_cont"] == "checkpointed":
        r["B_resume_after_cont"] = cu.resume(pid)
    elif r["B_state_after_cont"] == "locked":
        r["B_unlock_after_cont"] = cu.unlock(pid)
    r["B_finish"] = finish(pr, log, stop)
    return r


def t_child_cuda():
    pr, log, stop = start("parentchild", gb=1, kind="parent_child")
    wait_lines(log, 20, pr)
    wait_lines(log + ".child", 20, pr)
    ppid, cpid = pr.pid, int(open(log + ".childpid").read())
    MINE.append(cpid)
    r = {"before": {"parent": gpu_mb(ppid), "child": gpu_mb(cpid)}}
    r["pause_parent"] = cu.pause(ppid)
    c0 = lines(log + ".child")
    time.sleep(2)
    r["parent_paused"] = {"parent": gpu_mb(ppid), "child": gpu_mb(cpid), "child_state": cu.state(cpid),
                          "child_lines_in_2s": lines(log + ".child") - c0}
    r["pause_child"] = cu.pause(cpid)
    r["both_paused"] = {"parent": gpu_mb(ppid), "child": gpu_mb(cpid)}
    r["resume_child"] = cu.resume(cpid)
    r["resume_parent"] = cu.resume(ppid)
    r["finish"] = finish(pr, log, stop)
    r["child_end"] = [l.strip() for l in open(log + ".child") if l.startswith("END")]
    return r


def t_ipc():
    pr, log, stop = start("ipc", script="ipc_target.py")
    t = time.time()
    while not os.path.exists(log + ".childpid") or lines(log + ".child") < 10:
        time.sleep(0.1)
        if time.time() - t > 120:
            raise RuntimeError("ipc target did not start")
    ppid, cpid = pr.pid, int(open(log + ".childpid").read())
    MINE.append(cpid)
    r = {"before": {"parent": gpu_mb(ppid), "child": gpu_mb(cpid)}}
    r["pause_child_importer"] = cu.pause(cpid)
    time.sleep(1)
    if r["pause_child_importer"]["state"] == "checkpointed":
        r["resume_child"] = cu.resume(cpid)
    r["child_state"] = cu.state(cpid)
    time.sleep(1)
    r["pause_parent_exporter"] = cu.pause(ppid)
    time.sleep(1)
    if r["pause_parent_exporter"]["state"] == "checkpointed":
        r["resume_parent"] = cu.resume(ppid)
    r["parent_state"] = cu.state(ppid)
    time.sleep(1)
    r["finish"] = finish(pr, log, stop, timeout=60)
    ch = [l.split() for l in open(log + ".child") if l[:1].isdigit()]
    pa = [l.split() for l in open(log) if l[:1].isdigit()]
    r["child_distinct_sums"] = sorted(set(x[1] for x in ch))
    r["parent_distinct_sums"] = sorted(set(x[1] for x in pa))
    r["child_alive_at_end"] = os.path.exists(f"/proc/{cpid}")
    return r


def t_uvm():
    pr, log, stop = start("uvm", script="uvm_target.py")
    wait_lines(log, 10, pr)
    pid = pr.pid
    r = {"gpu_mb": gpu_mb(pid), "pause": cu.pause(pid)}
    time.sleep(1)
    if r["pause"]["state"] == "checkpointed":
        r["resume"] = cu.resume(pid)
    r["state"] = cu.state(pid)
    r["finish"] = finish(pr, log, stop, timeout=120)
    r["out_tail"] = open(log + ".out").read()[-400:]
    return r


def t_two_jobs():
    a_ = start("twoA", gb=4)
    b_ = start("twoB", gb=4)
    wait_lines(a_[1], 20, a_[0])
    wait_lines(b_[1], 20, b_[0])
    A, B = a_[0].pid, b_[0].pid
    r = {"pause_A": cu.pause(A), "pause_B": cu.pause(B), "gpu_both_paused": [gpu_mb(A), gpu_mb(B)],
         "resume_A": cu.resume(A), "pause_A_again": cu.pause(A), "resume_B": cu.resume(B), "resume_A_2": cu.resume(A)}
    r["finish_A"] = finish(*a_)
    r["finish_B"] = finish(*b_)
    return r


def t_zombie():
    pr, log, stop = start("zombie", gb=0.5, steps=30, sleep=0.0)
    t = time.time()
    while not os.path.exists(f"/proc/{pr.pid}/stat") or open(f"/proc/{pr.pid}/stat").read().split()[2] != "Z":
        time.sleep(0.05)
        if time.time() - t > 120:
            break
    r = {"proc_state": open(f"/proc/{pr.pid}/stat").read().split()[2], "state": ctl("state", pr.pid),
         "lock": ctl("lock", pr.pid, 1000)}
    r["rc"] = pr.wait()
    return r


def t_handoff():
    pr, log, stop = start("handoff", gb=2)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"controller1_pause_then_exit": ctl("pause", pid), "controller2_state": ctl("state", pid),
         "controller3_resume": ctl("resume", pid)}
    r["finish"] = finish(pr, log, stop)
    return r


def t_leak(cycles=50):
    pr, log, stop = start("leak", gb=2)
    wait_lines(log, 20, pr)
    pid = pr.pid
    rows = []
    for i in range(cycles):
        p_ = cu.pause(pid)
        time.sleep(0.2)
        q_ = cu.resume(pid)
        time.sleep(0.2)
        rows.append({"i": i, "ok": p_["state"] == "checkpointed" and q_["state"] == "running",
                     "ckpt_s": p_.get("checkpoint", [None, None])[1], "restore_s": q_["restore"][1],
                     "memavail": cu.mem_available_mb(), "rss": cu.proc_mem(pid).get("VmRSS"), "gpu": gpu_mb(pid)})
    return {"rows": rows, "all_ok": all(x["ok"] for x in rows), "finish": finish(pr, log, stop)}


def t_longpause(seconds=1800):
    pr, log, stop = start("longpause", gb=1)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"pause": cu.pause(pid), "t_paused": time.time()}
    time.sleep(seconds)
    r["state_before_resume"] = cu.state(pid)
    r["resume"] = cu.resume(pid)
    r["paused_s"] = round(time.time() - r["t_paused"], 1)
    time.sleep(3)
    r["finish"] = finish(pr, log, stop)
    return r


def t_cgroup(limit_extra_mb=2048, gb=4):
    """Target in a memory-limited cgroup (like Slurm or Docker): checkpointed data is charged to the job."""
    log, stop = "logs/f_cgroup.log", "logs/f_cgroup.stop"
    for f in (log, stop):
        if os.path.exists(f):
            os.remove(f)
    # probe the RSS of an identical target first, to set a limit that fits running but not paused
    pr, log, stop = start("cgroup_probe", gb=gb)
    wait_lines(log, 20, pr)
    rss = cu.proc_mem(pr.pid).get("VmRSS", 0)
    finish(pr, log, stop)
    limit = rss + limit_extra_mb
    log, stop = "logs/f_cgroup.log", "logs/f_cgroup.stop"
    c = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={limit}M", "-p", "MemorySwapMax=0",
         PY, "target.py", "--kind", "ballast", "--ballast_gb", str(gb), "--steps", "10000000",
         "--step_sleep", "0.02", "--log", log, "--stop_file", stop]
    pr = subprocess.Popen(c, stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
    MINE.append(pr.pid)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {"probe_rss_mb": rss, "memory_max_mb": limit, "cmdline": open(f"/proc/{pid}/cmdline").read().replace("\0", " ")[:120],
         "cgroup": open(f"/proc/{pid}/cgroup").read().strip(), "gpu_mb": gpu_mb(pid)}
    r["pause"] = cu.pause(pid)
    time.sleep(1)
    r["alive_after_pause"] = pr.poll() is None
    r["state"] = cu.state(pid) if r["alive_after_pause"] else None
    if r["state"] == "checkpointed":
        r["resume"] = cu.resume(pid)
    r["finish"] = finish(pr, log, stop) if pr.poll() is None else {"rc": pr.poll()}
    r["dmesg_hint"] = "check journal for oom"
    return r


def t_sigint_control():
    """Does SIGINT (Ctrl-C) stop a target that was never paused, or one paused and resumed earlier?"""
    r = {}
    for label, cycles in (("never_paused", 0), ("paused_resumed_before", 1)):
        pr, log, stop = start(f"sigint_{label}", gb=1)
        wait_lines(log, 20, pr)
        for _ in range(cycles):
            cu.pause(pr.pid)
            time.sleep(0.5)
            cu.resume(pr.pid)
            time.sleep(0.5)
        os.kill(pr.pid, signal.SIGINT)
        try:
            r[label] = {"rc": pr.wait(timeout=20)}
        except subprocess.TimeoutExpired:
            r[label] = {"rc": "still alive 20 s after SIGINT"}
            r[label]["finish"] = finish(pr, log, stop)
        r[label]["out_tail"] = open(log + ".out").read()[-300:]
    return r


def t_state_blocking():
    """How long do GetState and nvidia-smi take while a 16 GB checkpoint and restore are in flight?"""
    pr, log, stop = start("stateblock", gb=16)
    wait_lines(log, 20, pr)
    pid = pr.pid
    r = {}
    for phase, op in (("during_checkpoint", cu.pause), ("during_restore", cu.resume)):
        box = {}
        th = threading.Thread(target=lambda: box.update(op(pid)))
        th.start()
        time.sleep(0.3)
        r[phase + "_state_call"] = ctl("state", pid, timeout=60)
        t = time.time()
        subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"], capture_output=True, timeout=60)
        r[phase + "_nvidia_smi_s"] = round(time.time() - t, 3)
        th.join()
        r[phase + "_op"] = box
    r["finish"] = finish(pr, log, stop)
    return r


def uvm_maps(pid):
    """Mappings of /dev/nvidia-uvm in the process: a candidate signal for managed memory."""
    try:
        return [l.split()[0] + " " + l.split()[-1] for l in open(f"/proc/{pid}/maps") if "nvidia-uvm" in l]
    except FileNotFoundError:
        return None


def t_uvm_detect():
    r = {}
    cases = [("plain_torch", dict(gb=1)), ("pinned_loader", dict(kind="loader", gb=0, sleep=0.0, steps=100000)),
             ("driver_api_managed", dict(script="uvm_target.py")), ("torch_ipc", dict(script="ipc_target.py"))]
    for name, kw in cases:
        pr, log, stop = start(f"uvmdet_{name}", **kw)
        wait_lines(log, 10, pr)
        time.sleep(1)
        m = uvm_maps(pr.pid)
        r[name] = {"n_uvm_maps": len(m) if m is not None else None, "sample": (m or [])[:4],
                   "fds_uvm": sum(1 for fd in os.listdir(f"/proc/{pr.pid}/fd")
                                  if "nvidia-uvm" in os.path.realpath(f"/proc/{pr.pid}/fd/{fd}"))}
        if os.path.exists(log + ".childpid"):
            cp = int(open(log + ".childpid").read())
            MINE.append(cp)
            mc = uvm_maps(cp)
            r[name]["child_n_uvm_maps"] = len(mc) if mc is not None else None
        r[name]["finish"] = finish(pr, log, stop, timeout=60)
        if r[name]["finish"]["rc"] == "timeout":
            kill(pr)
    return r


def t_restore_retry(leave_frac=0.4, gb=4):
    """After a restore fails for lack of VRAM, can the job still be restored once memory is free?"""
    pr, log, stop = start(f"retry{leave_frac}", gb=gb)
    wait_lines(log, 20, pr)
    pid = pr.pid
    time.sleep(0.5)
    need = gpu_mb(pid)
    r = {"target_gpu_mb": need, "used_before_pause": cu.gpu_used_free()[0], "pause": cu.pause(pid)}
    r["used_after_pause"] = cu.gpu_used_free()[0]
    h, r["hog"] = hog(int(need * leave_frac))
    r["hog"].pop("hog_ready", None)
    used_h = cu.gpu_used_free()[0]
    r["restore_1"] = cu.restore(pid)
    r["used_delta_from_failed_restore_mb"] = cu.gpu_used_free()[0] - used_h
    r["state_1"] = cu.state(pid)
    kill(h)
    time.sleep(2)
    r["used_after_hog_killed"] = cu.gpu_used_free()[0]
    r["attempts"] = []
    t0 = time.time()
    for wait in (0, 10, 30):
        time.sleep(max(0, wait - (time.time() - t0)))
        st = cu.state(pid)
        r["attempts"].append({"t": round(time.time() - t0, 1), "state": st,
                              "restore": cu.restore(pid) if st == "checkpointed" else None, "state_after": cu.state(pid)})
        if r["attempts"][-1]["state_after"] != "checkpointed":
            break
    st = cu.state(pid)
    if st == "checkpointed":
        r["try_checkpoint_again"] = cu.checkpoint(pid)
        r["try_unlock"] = cu.unlock(pid)
        r["try_restore_last"] = cu.restore(pid)
    st = cu.state(pid)
    if st == "locked":
        r["unlock"] = cu.unlock(pid)
    r["state_final"] = cu.state(pid)
    if r["state_final"] == "running":
        r["finish"] = finish(pr, log, stop)
    else:
        kill(pr)
        time.sleep(3)
        r["killed_target"] = True
    r["used_at_end"] = cu.gpu_used_free()[0]
    return r


def t_reserve(gb=4):
    """Mitigation for the restore-OOM trap: the controller holds a placeholder allocation of the paused
    job's size while others fill the GPU, frees it, and restores immediately."""
    import ctypes
    c = cu._cu
    pr, log, stop = start("reserve", gb=gb)
    wait_lines(log, 20, pr)
    pid = pr.pid
    time.sleep(0.5)
    need = gpu_mb(pid)
    r = {"target_gpu_mb": need, "pause": cu.pause(pid)}
    dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
    c.cuDeviceGet(ctypes.byref(dev), 0)
    c.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    c.cuCtxSetCurrent(ctx)
    r["placeholder_alloc"] = cu.err(c.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t((need + 256) * 2 ** 20)))
    h, r["hog"] = hog(200)  # an outsider takes everything else
    r["hog"].pop("hog_ready", None)
    t = time.perf_counter()
    r["placeholder_free"] = cu.err(c.cuMemFree_v2(ptr))
    r["free_to_restore_start_ms"] = round((time.perf_counter() - t) * 1000, 3)
    r["resume"] = cu.resume(pid)
    kill(h)
    c.cuDevicePrimaryCtxRelease(dev)
    r["finish"] = finish(pr, log, stop) if r["resume"]["state"] == "running" else "not resumed"
    return r


def t_exit_while_paused():
    """Pause a job during CPU-only work at its end: can it exit while checkpointed? And a job that
    needs CUDA once more: does that call wait for the resume?"""
    r = {}
    for label, tail_cuda in (("cpu_only_tail", "0"), ("cuda_call_in_tail", "1")):
        log = f"logs/f_exit_{label}.log"
        if os.path.exists(log):
            os.remove(log)
        env = dict(os.environ, TAIL_CUDA=tail_cuda)
        pr = subprocess.Popen([PY, "target.py", "--kind", "tail", "--log", log], env=env,
                              stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
        MINE.append(pr.pid)
        wait_lines(log, 30, pr)
        time.sleep(0.3)
        mem0 = cu.mem_available_mb()
        p = cu.pause(pr.pid)
        t = time.time()
        try:
            rc = pr.wait(timeout=10)
            r[label] = {"pause": p, "exited_while_paused": True, "rc": rc, "after_s": round(time.time() - t, 2)}
        except subprocess.TimeoutExpired:
            r[label] = {"pause": p, "exited_while_paused": False, "state_after_10s": cu.state(pr.pid),
                        "resume": cu.resume(pr.pid)}
            r[label]["rc_after_resume"] = pr.wait(timeout=30)
        time.sleep(1)
        r[label]["memavail_delta_mb"] = cu.mem_available_mb() - mem0
        r[label]["log_tail"] = open(log).read()[-200:]
    return r


def t_tail_ops():
    """Which single CUDA operation, issued while the job is checkpointed, blocks and which crashes?"""
    r = {}
    ops = os.environ.get("TAIL_OPS", "sync,memcpy,kernel_item,launch_only,alloc,exit").split(",")
    for op in ops:
        log = f"logs/f_tail_{op}.log"
        if os.path.exists(log):
            os.remove(log)  # a stale log would make wait_lines return before the new job starts
        pr = subprocess.Popen([PY, "target.py", "--kind", "tail2", "--log", log], env=dict(os.environ, TAIL_OP=op),
                              stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
        MINE.append(pr.pid)
        wait_lines(log, 30, pr)
        time.sleep(0.3 if op != "sync_in_flight" else 4.5)  # in-flight: 3 s sleep, then launch + sync
        t0p = time.time()
        p = cu.pause(pr.pid)
        try:
            rc = pr.wait(timeout=10)
            r[op] = {"pause": p, "ended_while_paused": True, "rc": rc, "t_pause_start": round(t0p, 3)}
        except subprocess.TimeoutExpired:
            r[op] = {"pause": p, "ended_while_paused": False, "t_pause_start": round(t0p, 3), "resume": cu.resume(pr.pid)}
            r[op]["rc_after_resume"] = pr.wait(timeout=30)
        r[op]["tail"] = [l.strip() for l in open(log) if l.startswith("TAIL")]
        r[op]["stderr_tail"] = open(log + ".out").read()[-200:]
    return r


def t_sync_locked_only():
    """Is a synchronize issued while the job is only LOCKED (not checkpointed) also fatal?"""
    r = {}
    for op in ("sync", "stream_sync"):
        log = f"logs/f_lockonly_{op}.log"
        if os.path.exists(log):
            os.remove(log)
        pr = subprocess.Popen([PY, "target.py", "--kind", "tail2", "--log", log], env=dict(os.environ, TAIL_OP=op),
                              stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
        MINE.append(pr.pid)
        wait_lines(log, 30, pr)
        time.sleep(0.3)
        lk = cu.lock(pr.pid)
        time.sleep(6)  # the job issues its synchronize about 2.7 s into this window
        alive = pr.poll() is None
        st = cu.state(pr.pid) if alive else None
        ul = cu.unlock(pr.pid) if alive else None
        try:
            rc = pr.wait(timeout=30)
        except subprocess.TimeoutExpired:
            kill(pr)
            rc = "timeout"
        r[op] = {"lock": lk, "alive_after_6s_locked": alive, "state": st, "unlock": ul, "rc": rc,
                 "tail": [l.strip() for l in open(log) if l.startswith("TAIL")]}
    return r


TESTS = {"errors": t_errors, "short_far": lambda: t_restore_short(0.4), "short_near": lambda: t_restore_short(0.93),
         "ctlkill_ckpt": lambda: t_ctl_killed("checkpoint"), "ctlkill_restore": lambda: t_ctl_killed("restore"),
         "sigkill": lambda: t_target_signal("SIGKILL"), "sigterm": lambda: t_target_signal("SIGTERM"),
         "sigint": lambda: t_target_signal("SIGINT"), "killmid": t_target_killed_mid_checkpoint,
         "lock_timeout": t_lock_timeout, "sigstop": t_sigstop, "child_cuda": t_child_cuda, "ipc": t_ipc,
         "uvm": t_uvm, "two_jobs": t_two_jobs, "zombie": t_zombie, "handoff": t_handoff, "leak": t_leak,
         "longpause": t_longpause, "cgroup": t_cgroup, "sigint_control": t_sigint_control, "state_blocking": t_state_blocking, "uvm_detect": t_uvm_detect, "retry_far": lambda: t_restore_retry(0.4),
         "retry_near": lambda: t_restore_retry(0.93), "retry_far_small": lambda: t_restore_retry(0.4, gb=1),
         "reserve": t_reserve, "exit_while_paused": t_exit_while_paused, "tail_ops": t_tail_ops, "sync_locked_only": t_sync_locked_only}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        t0 = time.time()
        try:
            res = TESTS[name]()
        except Exception as e:
            res = {"exception": repr(e)}
        finally:
            here = os.getcwd()
            for pid in MINE:  # tidy up only our own processes: same pid AND started in this directory
                try:
                    if (os.path.exists(f"/proc/{pid}") and os.readlink(f"/proc/{pid}/cwd") == here
                            and open(f"/proc/{pid}/stat").read().split()[2] != "Z"):
                        st = cu.state(pid)
                        if st == "checkpointed":
                            cu.resume(pid)
                        elif st == "locked":
                            cu.unlock(pid)
                        os.kill(pid, signal.SIGCONT)
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, FileNotFoundError, PermissionError):
                    pass
            MINE.clear()
        out = {"test": name, "driver": cu.driver_info(), "wall_s": round(time.time() - t0, 1), "result": res}
        json.dump(out, open(resultfile.claim(f"results/fault_{name}.json"), "w"), indent=1, default=str)
        print(json.dumps(out, default=str)[:1500], flush=True)
