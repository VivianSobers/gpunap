from gpunap import results
from gpunap.check import sync_first_call as sf


def test_stamp_reads_time_and_rc():
    out = ["READY 1", "CALL 11.2500", "DONE 20.5000 rc=0"]
    assert sf.stamp(out, "CALL") == (11.25, None)
    assert sf.stamp(out, "DONE") == (20.5, "0")
    assert sf.stamp(out, "NOPE") == (None, None)


def test_run_labels():
    assert sf.label(False, 10.0, 11.0, None, None, None, 0) == "invalid: pause failed"
    assert sf.label(True, 10.0, None, None, None, None, 1) == "invalid: never reached the call"
    assert sf.label(True, 10.0, 9.5, 9.6, "0", None, 0) == "invalid: call came before the checkpoint"
    assert sf.label(True, 10.0, 11.0, None, None, 20.0, "timeout") == "hung after resume"
    assert sf.label(True, 10.0, 11.0, None, None, None, -11) == "crash: SIGSEGV"
    assert sf.label(True, 10.0, 11.0, None, None, 20.0, 1) == "error: exit 1"
    assert sf.label(True, 10.0, 11.0, 11.0, "0", None, 0) == "returned while checkpointed"
    assert sf.label(True, 10.0, 11.0, 20.5, "0", 20.0, 0) == "waited for resume"
    assert sf.label(True, 10.0, 11.0, 20.5, "719", 20.0, 0) == "waited for resume (rc=719)"
    assert sf.label(True, 10.0, 11.0, 11.0, "999", None, 0) == "returned while checkpointed (rc=999)"


def run(api, op, label):
    return {"api": api, "op": op, "label": label}


def test_any_crash_is_a_hazard():
    runs = [run("driver", "ctx_sync", "crash: SIGSEGV"), run("driver", "memcpy", "waited for resume")]
    v, summary = sf.verdict(runs)
    assert v == results.HAZARD and "driver ctx_sync: crash: SIGSEGV 1/1" in summary


def test_an_error_code_after_resume_is_a_hazard():
    assert sf.verdict([run("driver", "memcpy", "waited for resume (rc=719)")])[0] == results.HAZARD


def test_all_waiting_is_ok():
    runs = [run("driver", op, "waited for resume") for op in sf.OPS]
    assert sf.verdict(runs)[0] == results.OK


def test_only_invalid_runs_is_invalid():
    assert sf.verdict([run("driver", "ctx_sync", "invalid: pause failed")])[0] == results.INVALID


def test_invalid_runs_do_not_hide_a_crash():
    runs = [run("driver", "ctx_sync", "invalid: pause failed"), run("driver", "ctx_sync", "crash: SIGSEGV")]
    assert sf.verdict(runs)[0] == results.HAZARD
