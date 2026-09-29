from gpunap import results
from gpunap.check import child_context as cc

OK = {"result": "OK", "hung": False}


def test_all_steps_fine_is_ok_and_reports_child_state():
    v, why = cc.verdict([OK, OK, OK, OK], "ok", 0, "running")
    assert v == results.OK and "left the child running" in why


def test_a_failed_step_is_a_hazard():
    v, why = cc.verdict([OK, {"result": "CUDA_ERROR_OPERATING_SYSTEM"}, OK, OK], "ok", 0, "running")
    assert v == results.HAZARD and "pause child" in why


def test_lost_data_is_a_hazard():
    assert cc.verdict([OK] * 4, "bad child: no reply", 0, "running")[0] == results.HAZARD
