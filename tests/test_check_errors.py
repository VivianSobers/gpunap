from gpunap import results
from gpunap.check import errors as er

SEQ_OK = [(op, want, "OK" if want else "CUDA_ERROR_ILLEGAL_STATE") for op, want in er.SEQUENCE]
OTHERS = {"missing_pid_state": "CUDA_ERROR_OPERATING_SYSTEM", "missing_pid_lock": "CUDA_ERROR_OPERATING_SYSTEM",
          "no_cuda_state": "CUDA_ERROR_NOT_INITIALIZED", "no_cuda_lock": "CUDA_ERROR_NOT_INITIALIZED",
          "no_cuda_checkpoint": "CUDA_ERROR_NOT_INITIALIZED"}


def test_errors_everywhere_and_nothing_harmed_is_ok():
    assert er.verdict(SEQ_OK, "running", "ok", 0, OTHERS, True) == (results.OK, None)


def test_a_wrong_state_call_that_succeeds_is_a_hazard():
    seq = list(SEQ_OK)
    seq[0] = ("checkpoint", False, "OK")
    assert er.verdict(seq, "running", "ok", 0, OTHERS, True)[0] == results.HAZARD


def test_a_failed_setup_call_is_invalid():
    seq = [(op, want, "CUDA_ERROR_ILLEGAL_STATE") for op, want in er.SEQUENCE]
    assert er.verdict(seq, "running", "ok", 0, OTHERS, True)[0] == results.INVALID


def test_a_call_on_a_process_without_cuda_that_succeeds_is_a_hazard():
    assert er.verdict(SEQ_OK, "running", "ok", 0, dict(OTHERS, no_cuda_lock="OK"), True)[0] == results.HAZARD


def test_harm_to_the_target_or_the_bystander_is_a_hazard():
    assert er.verdict(SEQ_OK, "locked", "ok", 0, OTHERS, True)[0] == results.HAZARD
    assert er.verdict(SEQ_OK, "running", "bad x", 0, OTHERS, True)[0] == results.HAZARD
    assert er.verdict(SEQ_OK, "running", "ok", 0, OTHERS, False)[0] == results.HAZARD


def test_unused_pid_is_not_in_proc(tmp_path):
    (tmp_path / "99").mkdir()
    (tmp_path / "98").mkdir()
    assert er.unused_pid(100, str(tmp_path)) == 97
