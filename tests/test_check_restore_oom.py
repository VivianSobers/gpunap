from gpunap import results
from gpunap.check import restore_oom as ro

OOM = {"result": "CUDA_ERROR_OUT_OF_MEMORY", "hung": False}
OSE = {"result": "CUDA_ERROR_OPERATING_SYSTEM", "hung": False}
OK = {"result": "OK", "hung": False}


def test_unrestorable_after_oom_is_a_hazard():
    v, why = ro.verdict(OOM, [OSE, OSE], False, {"result": "OK", "verify": "ok"})
    assert v == results.HAZARD and "unrestorable" in why and "workaround worked" in why


def test_recovered_after_oom_is_ok():
    v, why = ro.verdict(OOM, [OK], True, {"result": "OK", "verify": "ok"})
    assert v == results.OK


def test_restore_that_succeeds_next_to_the_hog_is_invalid():
    assert ro.verdict(OK, [], True, None)[0] == results.INVALID


def test_failed_workaround_is_reported():
    v, why = ro.verdict(OOM, [OSE], False, {"result": "CUDA_ERROR_OUT_OF_MEMORY", "verify": "not checked"})
    assert v == results.HAZARD and "workaround failed" in why


def test_hog_size_leaves_forty_percent_of_the_job():
    assert ro.hog_mb(free_mb=20000, job_mb=1500) == 20000 - 600 - ro.HOG_CONTEXT_MB
