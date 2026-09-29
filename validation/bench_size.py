"""Pause/resume time and host-memory effect against GPU memory size. python bench_size.py 0.5 1 2 4 8 16 20"""
import json, os, subprocess, sys, time
import cu, resultfile

PY = sys.executable
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)
CYCLES = 3
OUT = None
out = {"driver": cu.driver_info(), "cycles": CYCLES, "sizes": {}}
for gb in map(float, sys.argv[1:]):
    used, free = cu.gpu_used_free()
    others = cu.gpu_apps()
    if free < gb * 1024 + 2500:
        out["sizes"][gb] = {"skipped": f"only {free} MiB free", "others": others}
        print(gb, "skipped", free, flush=True)
        continue
    log, stop = f"logs/size_{gb}.log", f"logs/size_{gb}.stop"
    for f in (log, stop):
        if os.path.exists(f):
            os.remove(f)
    pr = subprocess.Popen([PY, "target.py", "--kind", "ballast", "--ballast_gb", str(gb), "--steps", "10000000",
                           "--step_sleep", "0.02", "--log", log, "--stop_file", stop],
                          stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
    pid = pr.pid
    while (not os.path.exists(log) or sum(1 for _ in open(log)) < 20) and pr.poll() is None:
        time.sleep(0.1)
    rec = {"gpu_free_before_start": free, "others_before": others, "cycles": []}
    for c in range(CYCLES):
        m0, pm0, g0 = cu.mem_available_mb(), cu.proc_mem(pid), cu.gpu_apps().get(pid, 0)
        pa = cu.pause(pid)
        m1, pm1, g1, uf1 = cu.mem_available_mb(), cu.proc_mem(pid), cu.gpu_apps().get(pid, "absent"), cu.gpu_used_free()
        time.sleep(1.0)
        re = cu.resume(pid)
        m2, pm2, g2 = cu.mem_available_mb(), cu.proc_mem(pid), cu.gpu_apps().get(pid, 0)
        rec["cycles"].append({"gpu_mb": g0, "pause": pa, "resume": re, "memavail": [m0, m1, m2],
                              "procmem": [pm0, pm1, pm2], "gpu_mb_paused": g1, "gpu_used_free_paused": uf1,
                              "gpu_mb_after": g2})
        time.sleep(1.0)
    open(stop, "w").close()
    rec["rc"] = pr.wait(timeout=300)
    rec["end"] = [l.strip() for l in open(log) if l.startswith("END")]
    out["sizes"][gb] = rec
    c = rec["cycles"]
    print(gb, "GB | gpu_mb", c[0]["gpu_mb"], "| ckpt_s", [x["pause"].get("checkpoint", ["", None])[1] for x in c],
          "| restore_s", [x["resume"]["restore"][1] for x in c], "| memavail drop MB",
          [x["memavail"][0] - x["memavail"][1] for x in c], "| rss paused", c[0]["procmem"][1].get("VmRSS"),
          "lck", c[0]["procmem"][1].get("VmLck"), "| ok", rec["end"], flush=True)
    OUT = OUT or resultfile.claim("results/bench_size.json")
    json.dump(out, open(OUT, "w"), indent=1, default=str)
