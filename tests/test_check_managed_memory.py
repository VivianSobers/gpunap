from gpunap import results
from gpunap.check import managed_memory as mm

REFUSED = {"result": "CUDA_ERROR_NOT_SUPPORTED", "hung": False}
OK = {"result": "OK", "hung": False}


def test_refused_and_survived_is_ok():
    v, why = mm.verdict(REFUSED, None, "ok", 0)
    assert v == results.OK and "CUDA_ERROR_NOT_SUPPORTED" in why


def test_refused_then_broken_is_a_hazard():
    assert mm.verdict(REFUSED, None, "bad cuMemcpyDtoH rc=719 at offset 0", 0)[0] == results.HAZARD
    assert mm.verdict(REFUSED, None, "ok", 1)[0] == results.HAZARD
    assert mm.verdict(REFUSED, None, "no reply", None)[0] == results.HAZARD


def test_a_hung_pause_is_a_hazard():
    assert mm.verdict({"result": "hung", "hung": True}, None, "ok", 0)[0] == results.HAZARD


def test_supported_and_restored_is_ok():
    assert mm.verdict(OK, OK, "ok", 0)[0] == results.OK


def test_supported_but_restore_fails_is_a_hazard():
    assert mm.verdict(OK, {"result": "CUDA_ERROR_INVALID_VALUE", "hung": False}, "ok", 0)[0] == results.HAZARD
