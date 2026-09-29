from gpunap import results
from gpunap.check import size_sweep as ss

OK = {"result": "OK", "hung": False}


def _cycle(ck=1.0, rs=1.0, verify="ok"):
    return {"pause": dict(OK, calls=[{"op": "lock", "seconds": 0.01}, {"op": "checkpoint", "seconds": ck}]),
            "resume": dict(OK, calls=[{"op": "restore", "seconds": rs}, {"op": "unlock", "seconds": 0.01}]),
            "verify": verify}


def test_fits_needs_gpu_and_host_headroom():
    assert ss.fits(1024, free_mb=1024 + ss.GPU_MARGIN_MB, host_mb=1024 + ss.HOST_MARGIN_MB) is None
    assert "GPU" in ss.fits(1024, free_mb=1024 + ss.GPU_MARGIN_MB - 1, host_mb=10 ** 6)
    assert "host" in ss.fits(1024, free_mb=10 ** 6, host_mb=1024 + ss.HOST_MARGIN_MB - 1)


def test_all_sizes_intact_is_ok_with_times():
    sizes = [{"mb": 512, "cycles": [_cycle(0.2, 0.3)] * 3, "rc": 0},
             {"mb": 1024, "cycles": [_cycle(0.4, 0.5)] * 3, "rc": 0}]
    v, why = ss.verdict(sizes)
    assert v == results.OK
    assert "0.5 GB" in why and "1 GB" in why


def test_skipped_sizes_do_not_fail_the_check():
    sizes = [{"mb": 512, "cycles": [_cycle()] * 3, "rc": 0}, {"mb": 20480, "skipped": "not enough free GPU memory"}]
    v, why = ss.verdict(sizes)
    assert v == results.OK and "20 GB skipped" in why


def test_no_size_run_is_skipped():
    assert ss.verdict([{"mb": 512, "skipped": "not enough free GPU memory"}])[0] == results.SKIPPED


def test_bad_data_at_one_size_is_a_hazard():
    sizes = [{"mb": 512, "cycles": [_cycle(), _cycle(verify="bad chunk 3")], "rc": 0}]
    v, why = ss.verdict(sizes)
    assert v == results.HAZARD and "0.5 GB cycle 2: data check bad chunk 3" in why


def test_failed_resume_is_a_hazard():
    c = _cycle()
    c["resume"] = {"result": "CUDA_ERROR_OUT_OF_MEMORY", "hung": False}
    assert ss.verdict([{"mb": 2048, "cycles": [c], "rc": 0}])[0] == results.HAZARD


def test_target_that_never_started_is_an_error_entry():
    v, why = ss.verdict([{"mb": 512, "not_ready": {"rc": 1, "output": []}}])
    assert v == results.INVALID


def test_rates_are_gigabytes_per_second():
    assert ss.rate(2048, 0.5) == 4.0
    assert ss.rate(2048, None) is None
    assert ss.rate(2048, 0) is None


def test_timings_take_the_median_of_cycles():
    t = ss.timings(1024, [_cycle(0.1, 0.4), _cycle(0.3, 0.2), _cycle(0.2, 0.3)])
    assert t["checkpoint_s"] == 0.2 and t["restore_s"] == 0.3
    assert t["checkpoint_gb_s"] == 5.0
