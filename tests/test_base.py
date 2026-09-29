import sys

import pytest

from gpunap.check import base, owned

PY = sys.executable
ECHO = ("import sys\nprint('READY', flush=True)\nfor line in sys.stdin:\n    c = line.strip()\n"
        "    if c == 'PING':\n        print('PONG 1', flush=True)\n    elif c == 'EXIT':\n        sys.exit(0)\n")


def ctx(tmp_path):
    return base.Context(owned=owned.OwnedProcesses("tok9", str(tmp_path)), call=None, token="tok9",
                        python=PY, label="test", module_kind="proprietary", full=False)


def test_session_request_and_reply(tmp_path):
    c = ctx(tmp_path)
    t = c.owned.start([PY, "-c", ECHO, "tok9"], stdin=base.PIPE, stdout=base.PIPE, stderr=base.STDOUT, text=True, bufsize=1)
    s = base.Session(t.popen)
    assert s.wait_for("READY", 10) is not None
    assert s.request("PING", "PONG", 10) == "PONG 1"
    s.send("EXIT")
    assert t.popen.wait(timeout=10) == 0
    assert s.wait_for("NEVER", 1) is None
    assert s.send("PING") is False


def test_target_argv_carries_the_token(tmp_path):
    argv = ctx(tmp_path).argv("hold", "--mb", 64)
    assert argv[1:] == ["-m", "gpunap.check.targets", "hold", "--token", "tok9", "--mb", "64"]


def test_start_target_raises_not_ready_when_the_child_dies_first(tmp_path):
    c = ctx(tmp_path)
    c.python = PY
    with pytest.raises(base.NotReady) as e:
        base.start_target(c, "no-such-kind", ready_timeout=30)
    assert e.value.rc not in (None, 0) and e.value.output


def test_not_ready_becomes_an_invalid_result():
    r = base.not_ready_result("roundtrip", base.NotReady(["boom"], 1), 1.5)
    assert r.verdict == "invalid" and r.data["output"] == ["boom"] and r.data["rc"] == 1
