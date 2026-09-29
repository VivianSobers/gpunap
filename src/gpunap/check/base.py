"""What a check is, the context it runs in, and the line protocol it uses to talk to its targets.

A target prints READY once its GPU state exists, then answers commands on stdin one line at a time
(see gpunap.check.targets).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from subprocess import PIPE, STDOUT, Popen
from typing import Callable, Dict, List, Optional, Tuple

from gpunap import results
from gpunap.check.owned import OwnedProcesses, Target
from gpunap.worker import child_env

__all__ = ["PIPE", "STDOUT", "Check", "Context", "Session", "NotReady", "start_target", "not_ready_result"]

DEFAULT, FULL = "default", "full"
NEEDS = ("libcuda", "torch", "systemd-run", "idle_gpu")


@dataclass
class Check:
    name: str
    mode: str
    needs: Tuple[str, ...]
    gpu_mb: int
    host_mb: int
    run: Callable[["Context"], results.CheckResult]
    doc: str


@dataclass
class Context:
    owned: OwnedProcesses
    call: Optional[Callable[..., Dict]]
    token: str
    python: str
    label: str
    module_kind: Optional[str]
    full: bool
    notes: List[str] = field(default_factory=list)

    def argv(self, kind: str, *args) -> List[str]:
        return [self.python, "-m", "gpunap.check.targets", kind, "--token", self.token, *map(str, args)]


class Session:
    """Line protocol over a target's stdin and stdout. A reader thread timestamps every line."""

    def __init__(self, popen: Popen):
        self.popen = popen
        self.lines: List[Tuple[float, str]] = []
        self.eof = False
        self._cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.popen.stdout:
            with self._cv:
                self.lines.append((time.time(), line.rstrip("\n")))
                self._cv.notify_all()
        with self._cv:
            self.eof = True
            self._cv.notify_all()

    def wait_for(self, prefix: str, timeout: float, start: int = 0) -> Optional[Tuple[float, str]]:
        """The first line from index start on that begins with prefix, or None on timeout or EOF."""
        end = time.time() + timeout
        with self._cv:
            while True:
                for t, text in self.lines[start:]:
                    if text.startswith(prefix):
                        return t, text
                left = end - time.time()
                if self.eof or left <= 0:
                    return None
                self._cv.wait(min(left, 0.5))

    def send(self, cmd: str) -> bool:
        try:
            self.popen.stdin.write(cmd + "\n")
            self.popen.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def request(self, cmd: str, reply: str, timeout: float) -> Optional[str]:
        n = len(self.lines)
        if not self.send(cmd):
            return None
        got = self.wait_for(reply, timeout, start=n)
        return got[1] if got else None

    def output(self, n: int = 40) -> List[str]:
        return [text for _, text in self.lines[-n:]]

    def first(self, prefix: str) -> Optional[Tuple[float, str]]:
        return next(((t, x) for t, x in self.lines if x.startswith(prefix)), None)


class NotReady(Exception):
    def __init__(self, output: List[str], rc: Optional[int]):
        super().__init__(f"target exited or stalled before READY (rc={rc})")
        self.output = output
        self.rc = rc


def start_target(ctx: Context, kind: str, *args, ready_timeout: float = 120) -> Tuple[Target, Session]:
    t = ctx.owned.start(ctx.argv(kind, *args), kind=kind, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=True,
                        bufsize=1, env=child_env())
    s = Session(t.popen)
    if s.wait_for("READY", ready_timeout) is None:
        if t.popen.poll() is None:
            t.popen.kill()
            t.popen.wait()
        time.sleep(0.1)
        raise NotReady(s.output(), t.popen.poll())
    return t, s


def not_ready_result(name: str, e: NotReady, seconds: float) -> results.CheckResult:
    return results.CheckResult(name, results.INVALID, "a target process never became ready",
                               {"output": e.output, "rc": e.rc}, seconds)
