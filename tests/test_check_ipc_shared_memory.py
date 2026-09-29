from gpunap import results
from gpunap.check import ipc_shared_memory as ipc

OK = {"result": "OK", "hung": False}


def steps(**over):
    d = {"pause child": OK, "resume child": OK, "pause parent": OK, "resume parent": OK}
    d.update(over)
    return d


def test_everything_working_is_ok():
    assert ipc.verdict(steps(), "ok", 0) == (results.OK, "shared CUDA memory survived pausing child and parent")


def test_stuck_after_failed_restore_is_a_hazard():
    v, why = ipc.verdict(steps(**{"resume child": {"result": "CUDA_ERROR_INVALID_VALUE", "hung": False}}), "no reply", None)
    assert v == results.HAZARD and "resume child returned CUDA_ERROR_INVALID_VALUE" in why


def test_failed_checkpoint_that_kills_is_a_hazard():
    v, why = ipc.verdict(steps(**{"pause child": {"result": "CUDA_ERROR_OPERATING_SYSTEM", "hung": False},
                                  "resume child": None}), "no reply", -11)
    assert v == results.HAZARD and "exit -11" in why


def test_clean_refusal_is_ok():
    v, why = ipc.verdict(steps(**{"pause child": {"result": "CUDA_ERROR_NOT_SUPPORTED", "hung": False},
                                  "resume child": None}), "ok", 0)
    assert v == results.OK and "refused" in why


def test_hang_is_a_hazard():
    assert ipc.verdict(steps(**{"pause child": {"result": "hung", "hung": True}, "resume child": None}), "ok", 0)[0] == results.HAZARD
