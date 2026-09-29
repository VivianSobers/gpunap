"""Test doubles shared by the unit tests."""


NAMES = {0: b"CUDA_SUCCESS", 3: b"CUDA_ERROR_NOT_INITIALIZED", 304: b"CUDA_ERROR_OPERATING_SYSTEM",
         401: b"CUDA_ERROR_ILLEGAL_STATE", 801: b"CUDA_ERROR_NOT_SUPPORTED"}


class FakeLib:
    """Stands in for libcuda.so.1: returns chosen codes and records every call."""

    def __init__(self, codes=None, state=0, init=0):
        self.codes = codes or {}
        self.state_value = state
        self.init = init
        self.calls = []
        self.lock_timeouts = []

    def cuInit(self, flags):
        return self.init

    def cuGetErrorName(self, code, ref):
        if code not in NAMES:
            return 1
        ref._obj.value = NAMES[code]
        return 0

    def cuDriverGetVersion(self, ref):
        ref._obj.value = 13020
        return 0

    def _op(self, name, pid):
        self.calls.append((name, pid))
        return self.codes.get(name, 0)

    def cuCheckpointProcessLock(self, pid, args):
        self.lock_timeouts.append(args._obj.timeoutMs)
        return self._op("lock", pid)

    def cuCheckpointProcessCheckpoint(self, pid, args):
        return self._op("checkpoint", pid)

    def cuCheckpointProcessRestore(self, pid, args):
        return self._op("restore", pid)

    def cuCheckpointProcessUnlock(self, pid, args):
        return self._op("unlock", pid)

    def cuCheckpointProcessGetState(self, pid, ref):
        self.calls.append(("state", pid))
        ref._obj.value = self.state_value
        return self.codes.get("state", 0)


class MissingLockLib(FakeLib):
    def __getattribute__(self, name):
        if name == "cuCheckpointProcessLock":
            raise AttributeError(name)
        return object.__getattribute__(self, name)
