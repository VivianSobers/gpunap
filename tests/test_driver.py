import ctypes

from gpunap.driver import Driver

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


def test_error_names_come_from_the_driver():
    d = Driver(lib=FakeLib(codes={"checkpoint": 801}))
    c = d.checkpoint(42)
    assert (c.op, c.result, c.code, c.ok) == ("checkpoint", "CUDA_ERROR_NOT_SUPPORTED", 801, False)


def test_unknown_error_code_is_named_by_number():
    assert Driver(lib=FakeLib(codes={"restore": 9999})).restore(1).result == "rc9999"


def test_success_is_named_ok():
    c = Driver(lib=FakeLib()).unlock(1)
    assert c.result == "OK" and c.ok and c.seconds >= 0


def test_state_codes_map_to_names():
    assert Driver(lib=FakeLib(state=2)).state(7).result == "checkpointed"
    assert Driver(lib=FakeLib(state=1)).state(7).result == "locked"
    assert Driver(lib=FakeLib(state=9)).state(7).result == "state9"


def test_state_error_is_reported_as_error_name():
    assert Driver(lib=FakeLib(codes={"state": 3})).state(7).result == "CUDA_ERROR_NOT_INITIALIZED"


def test_pause_unlocks_after_a_failed_checkpoint():
    lib = FakeLib(codes={"checkpoint": 304})
    calls = Driver(lib=lib).pause(5)
    assert [c.op for c in calls] == ["lock", "checkpoint", "unlock"]
    assert lib.calls == [("lock", 5), ("checkpoint", 5), ("unlock", 5)]


def test_pause_skips_checkpoint_when_lock_fails():
    lib = FakeLib(codes={"lock": 401})
    assert [c.op for c in Driver(lib=lib).pause(5)] == ["lock"]


def test_resume_does_not_unlock_after_a_failed_restore():
    lib = FakeLib(codes={"restore": 304})
    assert [c.op for c in Driver(lib=lib).resume(5)] == ["restore"]


def test_resume_unlocks_after_restore():
    assert [c.op for c in Driver(lib=FakeLib()).resume(5)] == ["restore", "unlock"]


def test_lock_timeout_reaches_the_library():
    lib = FakeLib()
    Driver(lib=lib).lock(3, timeout_ms=5000)
    assert lib.lock_timeouts == [5000]


def test_available_needs_all_functions_and_init():
    assert Driver(lib=FakeLib()).available()
    assert not Driver(lib=FakeLib(init=100)).available()
    assert Driver(lib=MissingLockLib()).missing() == ["cuCheckpointProcessLock"]


def test_available_is_false_when_the_library_cannot_load():
    assert not Driver(path="libdoes-not-exist.so.9").available()


def test_version_and_call_dict():
    d = Driver(lib=FakeLib())
    assert d.version() == 13020
    row = d.lock(1).to_dict()
    assert set(row) == {"op", "result", "code", "start", "end", "seconds"}


def test_pid_is_passed_as_int():
    lib = FakeLib()
    Driver(lib=lib).restore("12")
    assert lib.calls == [("restore", 12)]


def test_lockargs_layout_matches_the_header():
    from gpunap.driver import LockArgs
    assert ctypes.sizeof(LockArgs) == 64
