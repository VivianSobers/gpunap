from gpunap import results
from gpunap.check import exit_while_paused as ew


def test_clean_exit_while_checkpointed_is_ok():
    v, why = ew.verdict(True, 10.0, 12.0, True, 0)
    assert v == results.OK and "while checkpointed" in why


def test_exit_that_waits_for_the_resume_is_ok():
    assert ew.verdict(True, 10.0, 12.0, False, 0)[0] == results.OK


def test_crash_is_a_hazard():
    assert ew.verdict(True, 10.0, 12.0, True, -11) == (results.HAZARD, "exit code -11 (SIGSEGV)")


def test_invalid_cases():
    assert ew.verdict(False, None, 12.0, True, 0)[0] == results.INVALID
    assert ew.verdict(True, 10.0, 9.0, True, 0)[0] == results.INVALID
    assert ew.verdict(True, 10.0, None, True, 0)[0] == results.INVALID
