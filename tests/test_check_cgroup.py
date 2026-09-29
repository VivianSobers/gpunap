from gpunap import results
from gpunap.check import cgroup as cg

LIMITED = {"max_bytes": 900 * 2 ** 20, "limit_at": "/user.slice/run-x.scope"}


def test_killed_during_checkpoint_is_a_hazard():
    v, why = cg.verdict(LIMITED, False, "no reply", -9)
    assert v == results.HAZARD and "SIGKILL" in why


def test_survived_with_intact_data_is_ok():
    assert cg.verdict(LIMITED, True, "ok", 0)[0] == results.OK


def test_no_limit_is_invalid():
    assert cg.verdict({"max_bytes": None}, True, "ok", 0)[0] == results.INVALID
    assert cg.verdict(None, True, "ok", 0)[0] == results.INVALID


def test_limit_sits_between_running_and_paused_size():
    assert cg.limit_mb(rss_mb=300, gpu_mb=512) == 556


def test_redact_drops_the_user_path():
    got = cg.redact({"path": "/user.slice/user-4242.slice/user@4242.service/app.slice/run-r1.scope",
                     "max_bytes": 5, "limit_at": "/user.slice/user-4242.slice/user@4242.service/app.slice/run-r1.scope",
                     "current_bytes": 3})
    assert got == {"scope": "run-r1.scope", "max_bytes": 5, "limit_on_own_cgroup": True, "current_bytes": 3}
    assert "4242" not in repr(got)
