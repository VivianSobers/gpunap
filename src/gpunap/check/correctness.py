"""correctness (--full, PyTorch): deterministic training stays bit-identical across 20 pauses.

The workload (targets.k_torch_train) prints every step's loss as an exact hex float and a hash of the
final weights. It runs twice without interruption, and the two runs must match or the workload is not
deterministic here and the check is invalid. A third run is paused at 20 seeded random steps, left
paused for 0.3 to 1.5 s each time, and resumed; it must match the references line for line.
FINDINGS: every paused run matched its references on both drivers.
"""
from __future__ import annotations

import random
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from gpunap import results
from gpunap.check import base

STEPS, PAUSES, SEED, MARGIN = 300, 20, 0, 10
WAIT_S = (0.3, 1.5)
RUN_LIMIT_S = 600


def parse(lines: List[str]) -> Tuple[List[str], Optional[str]]:
    steps, end = [], None
    for line in lines:
        f = line.split()
        if len(f) == 3 and f[0] == "STEP":
            steps.append(f[2])
        elif len(f) == 2 and f[0] == "END":
            end = f[1]
    return steps, end


def schedule(steps: int, n: int, seed: int) -> List[Tuple[int, float]]:
    """(step to pause at, seconds to stay paused), sorted by step."""
    rng = random.Random(seed)
    at = sorted(rng.sample(range(MARGIN, steps - MARGIN), n))
    return [(s, round(rng.uniform(*WAIT_S), 3)) for s in at]


def _same(ref: Dict, run: Dict) -> Optional[str]:
    """None if run matches ref, else where it first differs."""
    for i, (a, b) in enumerate(zip(ref["steps"], run["steps"])):
        if a != b:
            return f"step {i} loss {b}, expected {a}"
    if len(run["steps"]) != len(ref["steps"]):
        return f"{len(run['steps'])} steps, expected {len(ref['steps'])}"
    if run["end"] != ref["end"]:
        return f"final weights hash {run['end']}, expected {ref['end']}"
    return None


def verdict(refs: List[Dict], paused: Dict) -> Tuple[str, str]:
    for i, r in enumerate(refs, 1):
        if r["rc"] != 0 or r["end"] is None:
            return results.INVALID, f"unpaused run {i} exited with {r['rc']}"
    diff = _same(refs[0], refs[1])
    if diff:
        return results.INVALID, f"the two unpaused runs differ ({diff}): the workload is not deterministic here"
    exited = None
    for i, p in enumerate(paused["pauses"], 1):
        if not base.ok(p["pause"]):
            return results.HAZARD, f"pause {i}: pause returned {p['pause'].get('result')}"
        if not base.ok(p["resume"]):
            if p.get("exited_while_paused") == 0 and i == len(paused["pauses"]):
                exited = i  # past its last CUDA call: it exits while paused (FINDINGS), no resume needed
                continue
            return results.HAZARD, f"pause {i}: resume returned {p['resume'].get('result')}"
    if paused["rc"] != 0:
        return results.HAZARD, f"paused run exited with {paused['rc']} after {len(paused['steps'])} steps"
    diff = _same(refs[0], paused)
    if diff:
        return results.HAZARD, f"paused run differs: {diff}"
    if not paused["pauses"]:
        return results.INVALID, "no pause landed during the run"
    tail = f"; the job finished its work and exited normally during pause {exited}" if exited else ""
    return results.OK, (f"{len(paused['steps'])} steps and the final weights bit-identical to two unpaused runs "
                        f"across {len(paused['pauses'])} pauses{tail}")


def _collect(s: base.Session, t, limit_s: float) -> Dict:
    try:
        rc = t.popen.wait(timeout=limit_s)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    s.wait_for("NEVER", 0.3)  # let the reader collect the last lines
    steps, end = parse([x for _, x in s.lines])
    return {"steps": steps, "end": end, "rc": rc, "tail": s.output(3)}


def _last_step(s: base.Session) -> Optional[int]:
    for _, x in reversed(s.lines):
        if x.startswith("STEP "):
            return int(x.split()[1])
    return None


def run_train(ctx: base.Context, plan: List[Tuple[int, float]]) -> Dict:
    t, s = base.start_target(ctx, "torch_train", "--steps", STEPS, "--seed", SEED)
    pauses = []
    for at, wait in plan:
        if s.wait_for(f"STEP {at} ", RUN_LIMIT_S) is None:
            break
        p = {"at_step": at, "wait_s": wait, "pause": base.pause(ctx, t)}
        p["landed_step"] = _last_step(s)
        if base.ok(p["pause"]):
            time.sleep(wait)
            p["resume"] = base.resume(ctx, t)
        else:
            p["resume"] = {"result": "not attempted"}
        pauses.append(p)
        if not base.ok(p["resume"]):
            try:
                p["exited_while_paused"] = t.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p["exited_while_paused"] = None
            break
    d = _collect(s, t, RUN_LIMIT_S)
    d["pauses"] = pauses
    return d


def run(ctx: base.Context) -> results.CheckResult:
    t0 = time.time()
    plan = schedule(STEPS, PAUSES, SEED)
    try:
        refs = [run_train(ctx, []) for _ in range(2)]
        paused = run_train(ctx, plan)
    except base.NotReady as e:
        return base.not_ready_result("correctness", e, time.time() - t0)
    v, summary = verdict(refs, paused)
    data = {"steps": STEPS, "seed": SEED, "schedule": plan, "wait_s": WAIT_S,
            "references": [{k: r[k] for k in ("rc", "end", "tail")} for r in refs],
            "paused": {k: paused[k] for k in ("rc", "end", "tail", "pauses")}}
    return base.result("correctness", v, summary, data, t0)


CHECK = base.Check("correctness", base.FULL, ("libcuda", "torch"), gpu_mb=1500, host_mb=3000, run=run,
                   doc="Deterministic PyTorch training stays bit-identical across 20 pauses.")
