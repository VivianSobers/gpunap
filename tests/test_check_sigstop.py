from gpunap import results
from gpunap.check import sigstop as ss

HUNG = {"result": "hung", "hung": True}
FINE = {"result": "OK", "hung": False}


def case(calls, check="ok", rc=0):
    return {"calls": calls, "verify": check, "rc": rc}


def test_hangs_with_recovery_are_a_hazard():
    v, why = ss.verdict(case({"lock": HUNG, "state": HUNG}), case({"restore": HUNG, "state": HUNG}), True)
    assert v == results.HAZARD and "lock" in why and "restore" in why and "recovered" in why


def test_no_hangs_and_intact_data_is_ok():
    assert ss.verdict(case({"lock": FINE, "state": FINE}), case({"restore": FINE, "state": FINE}), True)[0] == results.OK


def test_lost_data_is_a_hazard():
    v, why = ss.verdict(case({"lock": FINE, "state": FINE}, check="bad x"), case({"restore": FINE, "state": FINE}), True)
    assert v == results.HAZARD and "data" in why


def test_failed_setup_pause_is_invalid():
    assert ss.verdict(case({"lock": HUNG}), case({}), False)[0] == results.INVALID
