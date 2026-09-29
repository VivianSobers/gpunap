from gpunap import results
from gpunap.check import roundtrip as rt

GOOD = {"pause": {"result": "OK", "hung": False}, "resume": {"result": "OK", "hung": False}, "verify": "ok"}


def test_all_cycles_good_is_ok():
    assert rt.verdict([GOOD] * 3, 0) == (results.OK, None)


def test_failed_pause_is_a_hazard():
    c = dict(GOOD, pause={"result": "CUDA_ERROR_NOT_SUPPORTED", "hung": False})
    assert rt.verdict([GOOD, c], 0) == (results.HAZARD, "cycle 2: pause returned CUDA_ERROR_NOT_SUPPORTED")


def test_hung_resume_is_a_hazard():
    c = dict(GOOD, resume={"result": "hung", "hung": True})
    assert rt.verdict([c], 0)[0] == results.HAZARD


def test_bad_data_is_a_hazard():
    c = dict(GOOD, verify="bad chunk 1 offset 0: word 0 is 0x0, expected 0x5")
    assert rt.verdict([c], 0) == (results.HAZARD, "cycle 1: data check bad chunk 1 offset 0: word 0 is 0x0, expected 0x5")


def test_nonzero_exit_is_a_hazard():
    assert rt.verdict([GOOD], -11) == (results.HAZARD, "target exited with -11")
