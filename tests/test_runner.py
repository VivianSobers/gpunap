import pytest

from gpunap import results
from gpunap.check import base, runner

GPU = {"name": "GPU", "driver_version": "595.1", "memory.used": 400, "memory.free": 23000, "memory.total": 24000}
HAVE = {"libcuda", "torch", "systemd-run"}


def chk(name="c", mode=base.DEFAULT, needs=("libcuda",), gpu_mb=1000, host_mb=1000, run=None):
    return base.Check(name, mode, needs, gpu_mb, host_mb,
                      run or (lambda ctx: results.CheckResult(name, results.OK, "fine")), doc="")


def pf(check, gpu=GPU, apps=(), mem=64000, kind="proprietary", full=False, have=HAVE):
    return runner.preflight(check, gpu, list(apps), mem, kind, full, have)


def test_preflight_passes_with_room():
    assert pf(chk()) == (True, None)


def test_preflight_skips_missing_needs():
    assert pf(chk(needs=("libcuda", "torch")), have={"libcuda"}) == (False, "PyTorch is not installed")
    ok, why = pf(chk(), have=set())
    assert not ok and "checkpoint API" in why
    assert pf(chk(needs=("systemd-run",)), have={"libcuda"})[1] == "systemd-run is not available"


def test_preflight_needs_a_gpu():
    assert pf(chk(), gpu=None) == (False, "nvidia-smi reports no GPU")


def test_preflight_keeps_a_gigabyte_of_gpu_headroom():
    assert pf(chk(gpu_mb=1000), gpu=dict(GPU, **{"memory.free": 2024}))[0]
    ok, why = pf(chk(gpu_mb=1000), gpu=dict(GPU, **{"memory.free": 2023}))
    assert not ok and "2024 MB of free GPU memory" in why


def test_preflight_keeps_a_gigabyte_of_host_headroom():
    assert pf(chk(host_mb=1000), mem=2024)[0]
    ok, why = pf(chk(host_mb=1000), mem=2023)
    assert not ok and "host memory" in why


def test_idle_gpu_tolerates_small_steady_users():
    c = chk(needs=("libcuda", "idle_gpu"))
    assert pf(c, apps=[{"pid": 5, "used_mb": 392}])[0]
    ok, why = pf(c, apps=[{"pid": 5, "used_mb": 392}, {"pid": 6, "used_mb": 700}])
    assert not ok and "2 other processes use 1092 MB" in why


def test_full_mode_is_refused_on_the_open_module():
    ok, why = pf(chk(), kind="open", full=True)
    assert not ok and "open kernel module" in why
    assert pf(chk(), kind="open", full=False)[0]


def test_select_keeps_order_and_rejects_unknown_names():
    a, b = chk("a"), chk("b", mode=base.FULL)
    assert runner.select([a, b], only=None) == [a, b]
    assert runner.select([a, b], only=["b"]) == [b]
    with pytest.raises(ValueError):
        runner.select([a, b], only=["nope"])


class FakeOwned:
    run_dir = "/nonexistent-gpunap-test"

    def __init__(self):
        self.cleanups, self.closed = 0, False

    def cleanup(self, call, module_kind):
        self.cleanups += 1
        return []

    def close(self):
        self.closed = True


class FakeProbe:
    def __init__(self, apps=()):
        self._apps = list(apps)

    def gpu(self):
        return GPU

    def apps(self):
        return self._apps

    def mem_available_mb(self):
        return 64000

    def have(self):
        return HAVE

    def sysinfo(self):
        return {"gpu": "GPU"}


def ctx(full=False):
    return base.Context(owned=FakeOwned(), call=None, token="gpunap-t", python="python", label="x",
                        module_kind="proprietary", full=full)


def test_a_check_that_raises_is_an_error_and_later_checks_still_run():
    def boom(ctx):
        raise RuntimeError("bad")
    c = ctx()
    rep = runner.run([chk("a", run=boom), chk("b")], c, probe=FakeProbe())
    assert [(r.name, r.verdict) for r in rep.results] == [("a", results.ERROR), ("b", results.OK)]
    assert "RuntimeError: bad" in rep.results[0].data["traceback"]
    assert c.owned.cleanups == 2 and c.owned.closed and not rep.partial


def test_interrupt_cleans_up_and_marks_the_report_partial():
    def interrupted(ctx):
        raise KeyboardInterrupt
    c = ctx()
    rep = runner.run([chk("a"), chk("b", run=interrupted), chk("c")], c, probe=FakeProbe())
    assert rep.partial and [r.name for r in rep.results] == ["a"]
    assert c.owned.cleanups == 3 and c.owned.closed


def test_full_checks_are_skipped_without_full():
    rep = runner.run([chk("a", mode=base.FULL)], ctx(), probe=FakeProbe())
    assert rep.results[0].verdict == results.SKIPPED and "--full" in rep.results[0].summary


def test_preflight_failures_are_recorded_as_skipped():
    rep = runner.run([chk("a", needs=("libcuda", "idle_gpu"))], ctx(), probe=FakeProbe([{"pid": 1, "used_mb": 5000}]))
    assert rep.results[0].verdict == results.SKIPPED


def test_header_records_the_run_and_the_co_tenants():
    rep = runner.run([chk("a")], ctx(), probe=FakeProbe([{"pid": 1, "used_mb": 300}]))
    h = rep.header
    assert h["label"] == "x" and h["token"] == "gpunap-t" and h["full"] is False
    assert h["other_gpu_processes"] == {"count": 1, "mb": 300}
    assert h["system"] == {"gpu": "GPU"} and "gpu_memory_used_mb_after" in h


def test_results_are_reported_as_they_arrive():
    seen = []
    runner.run([chk("a"), chk("b")], ctx(), probe=FakeProbe(), on_result=seen.append)
    assert [r.name for r in seen] == ["a", "b"]
