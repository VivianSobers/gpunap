"""Correctness case: reference run, second reference run (determinism check), then a paused run.

python run_case.py --kind mlp --pauses 20 --seed 0 [--steps N] [--ballast_gb G] [--env K=V ...] [--tag T]
Writes results/<tag>.json with the full config, per-pause measurements and the verdict.
"""
import argparse, json, os, random, statistics, subprocess, sys, time

import cu

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True)
p.add_argument("--pauses", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--ballast_gb", type=float, default=0.0)
p.add_argument("--env", nargs="*", default=[])
p.add_argument("--tag", default=None)
p.add_argument("--hold", type=float, nargs=2, default=[0.3, 1.5])
p.add_argument("--early", action="store_true", help="also pause during the first 3 steps")
a = p.parse_args()
tag = a.tag or f"{a.kind}_p{a.pauses}_s{a.seed}"
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
env = dict(os.environ, **dict(kv.split("=", 1) for kv in a.env))
PY = sys.executable
T0 = time.time()


def cmd(log):
    return [PY, "target.py", "--kind", a.kind, "--steps", str(a.steps), "--seed", str(a.seed),
            "--ballast_gb", str(a.ballast_gb), "--log", log]


def rows(path):
    out = []
    for line in open(path):
        f = line.split()
        if f and f[0] != "END":
            out.append((f[0], f[1], float(f[2])))
    return out


def end_line(path):
    for line in open(path):
        if line.startswith("END"):
            return line.strip()
    return None


def keyed(path):
    return sorted((r[0], r[1]) for r in rows(path)) if a.kind == "threads" else [(r[0], r[1]) for r in rows(path)]


def progress(path):
    try:
        return sum(1 for line in open(path) if line[:1].isdigit())
    except FileNotFoundError:
        return 0


total_lines = a.steps * (2 if a.kind == "threads" else 1)
envkey = "_".join(sorted(a.env)).replace("=", "-").replace(":", "-").replace(",", "-")
refs = []
for r in ("ref1", "ref2"):
    log = f"logs/{a.kind}_s{a.seed}_n{a.steps}_b{a.ballast_gb}_{envkey}_{r}.log"
    if not (os.path.exists(log) and end_line(log)):
        rc = subprocess.run(cmd(log), env=env, stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT).returncode
        if rc != 0:
            json.dump({"tag": tag, "error": f"{r} exited {rc}", "out": open(log + ".out").read()[-3000:]},
                      open(f"results/{tag}.json", "w"), indent=1)
            sys.exit(f"{r} failed, see {log}.out")
    refs.append(log)
deterministic = keyed(refs[0]) == keyed(refs[1]) and end_line(refs[0]) == end_line(refs[1])

rng = random.Random(1000 + a.seed * 7 + a.pauses)
lo = 0 if a.early else max(3, total_lines // 20)
points = sorted(rng.sample(range(lo, int(total_lines * 0.9)), a.pauses))
if a.early:
    points[0] = 1
holds = [rng.uniform(*a.hold) for _ in points]

log = f"logs/{tag}_paused.log"
proc = subprocess.Popen(cmd(log), env=env, stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
pid = proc.pid
events = []
for i, (pt, hold) in enumerate(zip(points, holds)):
    while progress(log) < pt and proc.poll() is None:
        time.sleep(0.01)
    if proc.poll() is not None:
        events.append({"i": i, "error": "target exited before pause point"})
        break
    ev = {"i": i, "at_line": progress(log), "hold_s": round(hold, 3),
          "gpu_mb_before": cu.gpu_apps().get(pid, 0), "memavail_before": cu.mem_available_mb(),
          "procmem_before": cu.proc_mem(pid)}
    ev["pause"] = cu.pause(pid)
    t_paused = time.time()
    cpu0, line0 = cu.tree_cpu_seconds(pid), progress(log)
    ev["gpu_mb_paused"] = cu.gpu_apps().get(pid, "absent")
    ev["memavail_paused"] = cu.mem_available_mb()
    ev["procmem_paused"] = cu.proc_mem(pid)
    time.sleep(max(0.0, hold - (time.time() - t_paused)))
    cpu1 = cu.tree_cpu_seconds(pid)
    ev["lines_during_pause"] = progress(log) - line0
    ev["cpu_s_per_s_paused"] = round((cpu1 - cpu0) / max(1e-6, time.time() - t_paused), 3) if cpu0 is not None and cpu1 is not None else None
    ev["resume"] = cu.resume(pid)
    ev["gpu_mb_after"] = cu.gpu_apps().get(pid, 0)
    ev["memavail_after"] = cu.mem_available_mb()
    events.append(ev)
try:
    rc = proc.wait(timeout=3600)
except subprocess.TimeoutExpired:
    proc.kill()
    rc = "timeout"

paused_rows, ref_rows = rows(log), rows(refs[0])


def step_dt(rs):
    return [b[2] - a_[2] for a_, b in zip(rs, rs[1:])]


pre = step_dt([r for r in paused_rows[: max(2, points[0])]])
post = step_dt(paused_rows[points[-1] + 5:]) if len(paused_rows) > points[-1] + 7 else []
same_rows = keyed(log) == keyed(refs[0])
same_end = end_line(log) == end_line(refs[0])
all_ok = all(e.get("pause", {}).get("state") == "checkpointed" and e.get("resume", {}).get("state") == "running"
             for e in events) and len(events) == a.pauses
res = {
    "tag": tag, "config": vars(a), "driver": cu.driver_info(), "python": sys.version.split()[0],
    "target_pid": pid, "target_rc": rc, "ref_deterministic": deterministic,
    "rows_identical_to_ref": same_rows, "end_identical_to_ref": same_end,
    "all_pauses_ok": all_ok, "n_rows": len(paused_rows), "n_ref_rows": len(ref_rows),
    "end_paused": end_line(log), "end_ref": end_line(refs[0]),
    "median_step_s_before_first_pause": round(statistics.median(pre), 5) if pre else None,
    "median_step_s_after_last_resume": round(statistics.median(post), 5) if post else None,
    "events": events, "wall_s": round(time.time() - T0, 1),
}
if not same_rows:
    k1, k2 = keyed(log), keyed(refs[0])
    res["first_diff"] = next(((i, x, y) for i, (x, y) in enumerate(zip(k1, k2)) if x != y), ("len", len(k1), len(k2)))
json.dump(res, open(f"results/{tag}.json", "w"), indent=1)
lat = [e["pause"]["checkpoint"][1] for e in events if "checkpoint" in e.get("pause", {})]
print(f"{tag}: rc={rc} det_ref={deterministic} rows_same={same_rows} end_same={same_end} pauses_ok={all_ok} "
      f"n={len(events)} ckpt_s_max={max(lat) if lat else None} "
      f"cpu_paused={[e.get('cpu_s_per_s_paused') for e in events][:3]} "
      f"gpu_paused={[e.get('gpu_mb_paused') for e in events][:3]}", flush=True)
