from gpunap.check import base
from gpunap.check.owned import Target


class FakeCtx:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def call(self, op, pid, limit_s, timeout_ms=0):
        self.seen.append((op, pid, limit_s, timeout_ms))
        return dict(self.reply, op=op)


def test_ok_needs_ok_result_and_no_hang():
    assert base.ok({"result": "OK", "hung": False})
    assert not base.ok({"result": "OK", "hung": True})
    assert not base.ok({"result": "CUDA_ERROR_OUT_OF_MEMORY"})
    assert not base.ok(None)


def test_resume_marks_the_target_resumed_only_on_success():
    t = Target(pid=5, start_time=1)
    base.resume(FakeCtx({"result": "CUDA_ERROR_OPERATING_SYSTEM", "hung": False}), t)
    assert t.resumed is False
    base.resume(FakeCtx({"result": "OK", "hung": False}), t)
    assert t.resumed is True


def test_pause_passes_the_lock_timeout():
    c = FakeCtx({"result": "OK"})
    base.pause(c, Target(pid=9, start_time=1), limit_s=15, timeout_ms=5000)
    assert c.seen == [("pause", 9, 15, 5000)]
