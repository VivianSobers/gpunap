from gpunap import results
from gpunap.check import nccl_single_gpu as nc

OK = {"result": "OK", "hung": False}
FAIL = {"result": "CUDA_ERROR_OPERATING_SYSTEM", "hung": False}


def cyc(pause=OK, resume=OK, verify="ok"):
    return {"pause": pause, "resume": resume, "verify": verify}


def test_abort_after_failed_checkpoint_is_a_hazard():
    v, why = nc.verdict([cyc(pause=FAIL, resume=None, verify="no reply")], -6)
    assert v == results.HAZARD and "SIGABRT" in why and "CUDA_ERROR_OPERATING_SYSTEM" in why


def test_clean_cycles_are_ok():
    assert nc.verdict([cyc()] * 3, 0)[0] == results.OK


def test_refused_but_alive_is_ok():
    v, why = nc.verdict([cyc(pause=FAIL, resume=None)], 0)
    assert v == results.OK and "refused" in why


def test_bad_collective_is_a_hazard():
    assert nc.verdict([cyc(verify="bad collective loop stopped")], 0)[0] == results.HAZARD


def test_nccl_threads_detected():
    assert nc.nccl_threads(["cuda-EvtHandlr", "pt_nccl_heartbt", "pt_nccl_watchdg", "python3"]) == ["pt_nccl_heartbt", "pt_nccl_watchdg"]
    assert nc.nccl_threads(None) == []
