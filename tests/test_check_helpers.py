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


def test_call_seconds_finds_the_named_call():
    d = {"calls": [{"op": "lock", "seconds": 0.01}, {"op": "checkpoint", "seconds": 0.5}]}
    assert base.call_seconds(d, "checkpoint") == 0.5
    assert base.call_seconds(d, "restore") is None and base.call_seconds({}, "lock") is None


def test_cycle_problem_names_the_first_failing_step():
    good = {"pause": {"result": "OK"}, "resume": {"result": "OK"}, "verify": "ok"}
    assert base.cycle_problem([good, good]) is None
    assert base.cycle_problem([good, dict(good, resume={"result": "E"})]) == "cycle 2: resume returned E"
    assert base.cycle_problem([dict(good, verify="bad x")]) == "cycle 1: data check bad x"
