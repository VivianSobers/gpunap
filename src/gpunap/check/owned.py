"""Processes a check run started, and nothing else.

A process counts as ours only while its PID and start time match what was recorded when it started
(FINDINGS design rule 8). Every run writes its targets to a run file, so a later run can clean up
after a runner that was killed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from gpunap import procinfo

DEFAULT_RUN_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gpunap", "runs")
Call = Callable[..., Dict]


@dataclass
class Target:
    pid: int
    start_time: int
    kind: str = ""
    popen: Optional[subprocess.Popen] = None
    via_parent: bool = False
    resumed: bool = False
    extra: Dict = field(default_factory=dict)

    def alive(self, proc: str = "/proc") -> bool:
        if self.popen is not None and self.popen.poll() is not None:
            return False
        return procinfo.stat_state(self.pid, proc) not in (None, "Z")


def cleanup_steps(state: str, module_kind: Optional[str], was_resumed: bool) -> List[str]:
    """Driver steps and the final kill for one target, given its checkpoint state.

    A checkpointed target is killed as it is: that frees its memory (FINDINGS) and needs no free
    GPU memory. A locked one is unlocked first. On the open kernel module a target that has been
    resumed is checkpointed again before the kill (external source: NVIDIA issue #53)."""
    steps: List[str] = []
    if state == "locked":
        steps.append("unlock")
    if module_kind == "open" and was_resumed and state in ("running", "locked"):
        steps += ["lock", "checkpoint"]
    steps.append("sigkill")
    return steps


def _signal(pid: int, sig: int) -> Optional[str]:
    try:
        os.kill(pid, sig)
        return None
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "permission denied"


def _clean_one(t: Target, call: Call, module_kind: Optional[str], limit_s: float) -> Dict:
    entry = {"pid": t.pid, "kind": t.kind, "sigcont": _signal(t.pid, signal.SIGCONT)}
    st = call("state", t.pid, limit_s)
    entry["state"] = st.get("result")
    steps = cleanup_steps(entry["state"], module_kind, t.resumed)
    entry["steps"] = steps
    entry["calls"] = []
    for step in steps[:-1]:
        r = call(step, t.pid, 60.0)
        entry["calls"].append({"op": step, "result": r.get("result")})
    entry["sigkill"] = _signal(t.pid, signal.SIGKILL)
    if t.popen is not None:
        try:
            t.popen.wait(timeout=30)
        except subprocess.TimeoutExpired:
            entry["wait"] = "still running 30 s after SIGKILL"
    else:
        end = time.time() + 10
        while time.time() < end and procinfo.stat_state(t.pid) not in (None, "Z"):
            time.sleep(0.05)
    return entry


class OwnedProcesses:
    def __init__(self, token: str, run_dir: str = DEFAULT_RUN_DIR, proc: str = "/proc"):
        self.token = token
        self.run_dir = run_dir
        self.proc = proc
        self.targets: List[Target] = []

    @property
    def run_file(self) -> str:
        return os.path.join(self.run_dir, f"{self.token}.json")

    def _start_time(self, pid: int) -> int:
        for _ in range(100):
            st = procinfo.start_time(pid, self.proc)
            if st is not None:
                return st
            time.sleep(0.01)
        raise RuntimeError(f"no /proc entry for new process {pid}")

    def start(self, argv: List[str], kind: str = "", **popen_kwargs) -> Target:
        if self.token not in argv:
            raise ValueError("a target's command line must carry the run token")
        popen_kwargs.setdefault("start_new_session", True)
        p = subprocess.Popen(argv, **popen_kwargs)
        t = Target(pid=p.pid, start_time=self._start_time(p.pid), kind=kind, popen=p)
        self.targets.append(t)
        self.save()
        return t

    def adopt(self, pid: int, kind: str = "") -> Target:
        """Record a process one of our targets started itself (a child)."""
        t = Target(pid=pid, start_time=self._start_time(pid), kind=kind, via_parent=True)
        self.targets.append(t)
        self.save()
        return t

    def is_ours(self, pid: int, start_time: int, via_parent: bool = False) -> bool:
        if procinfo.start_time(pid, self.proc) != start_time:
            return False
        return via_parent or self.token in (procinfo.cmdline(pid, self.proc) or [])

    def live(self) -> List[Target]:
        return [t for t in self.targets
                if t.alive(self.proc) and self.is_ours(t.pid, t.start_time, t.via_parent)]

    def cleanup(self, call: Call, module_kind: Optional[str], limit_s: float = 10.0) -> List[Dict]:
        """Unlock or re-checkpoint as needed, then SIGKILL every live target, children first."""
        log = [_clean_one(t, call, module_kind, limit_s) for t in reversed(self.live())]
        self.save()
        return log

    def save(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        rec = {"token": self.token, "targets": [
            {"pid": t.pid, "start_time": t.start_time, "kind": t.kind, "via_parent": t.via_parent}
            for t in self.targets]}
        with open(self.run_file, "w") as f:
            json.dump(rec, f)

    def close(self) -> None:
        try:
            os.remove(self.run_file)
        except FileNotFoundError:
            pass


def reconcile(run_dir: str, call: Call, module_kind: Optional[str], proc: str = "/proc") -> List[Dict]:
    """Clean up targets left by earlier runs whose runner died. Only processes whose PID, start time
    and token still match are touched."""
    log: List[Dict] = []
    if not os.path.isdir(run_dir):
        return log
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(run_dir, name)
        try:
            rec = json.load(open(path))
        except (OSError, ValueError):
            continue
        tracker = OwnedProcesses(rec.get("token", ""), run_dir, proc)
        for r in rec.get("targets", []):
            t = Target(pid=r["pid"], start_time=r["start_time"], kind=r.get("kind", ""),
                       via_parent=r.get("via_parent", False))
            if t.alive(proc) and tracker.is_ours(t.pid, t.start_time, t.via_parent):
                log.append(_clean_one(t, call, module_kind, 10.0))
        os.remove(path)
    return log
