"""ctypes wrapper over the CUDA driver's process checkpoint API.

libcuda.so.1 is loaded on first use, so importing this module never needs a GPU, and tests can
inject a fake library.
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

FUNCS = ("cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint", "cuCheckpointProcessRestore",
         "cuCheckpointProcessUnlock", "cuCheckpointProcessGetState")
STATES = {0: "running", 1: "locked", 2: "checkpointed", 3: "failed"}


class LockArgs(ctypes.Structure):
    """CUcheckpointLockArgs from cuda.h."""
    _fields_ = [("timeoutMs", ctypes.c_uint), ("reserved0", ctypes.c_uint),
                ("reserved1", ctypes.c_uint64 * 7)]


@dataclass
class Call:
    """One driver call: its name, the result name (OK, an error name, or a state), timings."""
    op: str
    result: str
    code: int
    start: float
    end: float

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 4)
        return d


class Driver:
    def __init__(self, lib=None, path: str = "libcuda.so.1"):
        self._lib = lib
        self._path = path
        self._init_code: Optional[int] = None

    def lib(self):
        if self._lib is None:
            self._lib = ctypes.CDLL(self._path)
        if self._init_code is None:
            self._init_code = self._lib.cuInit(0)
        return self._lib

    def init_code(self) -> Optional[int]:
        try:
            self.lib()
        except OSError:
            return None
        return self._init_code

    def missing(self) -> List[str]:
        lib = self.lib()
        return [f for f in FUNCS if not hasattr(lib, f)]

    def available(self) -> bool:
        try:
            return not self.missing() and self._init_code == 0
        except OSError:
            return False

    def version(self) -> int:
        v = ctypes.c_int(0)
        self.lib().cuDriverGetVersion(ctypes.byref(v))
        return v.value

    def error_name(self, code: int) -> str:
        if code == 0:
            return "OK"
        p = ctypes.c_char_p()
        if self.lib().cuGetErrorName(code, ctypes.byref(p)) == 0 and p.value:
            return p.value.decode()
        return f"rc{code}"

    def _call(self, op: str, fn: str, pid, args=None) -> Call:
        t = time.time()
        code = getattr(self.lib(), fn)(int(pid), args)
        e = time.time()
        return Call(op, self.error_name(code), code, t, e)

    def lock(self, pid, timeout_ms: int = 0) -> Call:
        args = LockArgs(int(timeout_ms), 0)
        return self._call("lock", "cuCheckpointProcessLock", pid, ctypes.byref(args))

    def checkpoint(self, pid) -> Call:
        return self._call("checkpoint", "cuCheckpointProcessCheckpoint", pid)

    def restore(self, pid) -> Call:
        return self._call("restore", "cuCheckpointProcessRestore", pid)

    def unlock(self, pid) -> Call:
        return self._call("unlock", "cuCheckpointProcessUnlock", pid)

    def state(self, pid) -> Call:
        s = ctypes.c_int(-1)
        t = time.time()
        code = self.lib().cuCheckpointProcessGetState(int(pid), ctypes.byref(s))
        e = time.time()
        result = STATES.get(s.value, f"state{s.value}") if code == 0 else self.error_name(code)
        return Call("state", result, code, t, e)

    def pause(self, pid, timeout_ms: int = 0) -> List[Call]:
        """Lock then checkpoint. A failed checkpoint is followed by an unlock, so the job is never
        left frozen."""
        calls = [self.lock(pid, timeout_ms)]
        if calls[-1].ok:
            calls.append(self.checkpoint(pid))
            if not calls[-1].ok:
                calls.append(self.unlock(pid))
        return calls

    def resume(self, pid) -> List[Call]:
        """Restore then unlock. After a failed restore the job stays checkpointed."""
        calls = [self.restore(pid)]
        if calls[-1].ok:
            calls.append(self.unlock(pid))
        return calls
