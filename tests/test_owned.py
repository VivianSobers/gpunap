import json
import os
import subprocess
import sys
import time

import pytest

from gpunap import procinfo
from gpunap.check import owned

PY = sys.executable


def sleeper(token):
    return [PY, "-c", "import time, sys; time.sleep(60)", "--token", token]


class FakeCall:
    def __init__(self, state):
        self.state = state
        self.ops = []

    def __call__(self, op, pid, limit_s, timeout_ms=0):
        self.ops.append(op)
        return {"op": op, "result": self.state if op == "state" else "OK", "hung": False}


def gone(pid, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if procinfo.stat_state(pid) in (None, "Z"):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("state,module,resumed,expect", [
    ("running", "proprietary", False, ["sigkill"]),
    ("running", "proprietary", True, ["sigkill"]),
    ("running", "open", True, ["lock", "checkpoint", "sigkill"]),
    ("running", "open", False, ["sigkill"]),
    ("locked", "proprietary", False, ["unlock", "sigkill"]),
    ("locked", "open", True, ["unlock", "lock", "checkpoint", "sigkill"]),
    ("checkpointed", "proprietary", True, ["sigkill"]),
    ("checkpointed", "open", True, ["sigkill"]),
    ("hung", "open", True, ["sigkill"]),
    ("CUDA_ERROR_NOT_INITIALIZED", "proprietary", False, ["sigkill"]),
])
def test_cleanup_steps(state, module, resumed, expect):
    assert owned.cleanup_steps(state, module, resumed) == expect


def test_start_refuses_argv_without_the_token(tmp_path):
    with pytest.raises(ValueError):
        owned.OwnedProcesses("tok123", str(tmp_path)).start([PY, "-c", "pass"])


def test_is_ours_needs_matching_start_time(tmp_path):
    op = owned.OwnedProcesses("tok123", str(tmp_path))
    t = op.start(sleeper("tok123"))
    try:
        assert op.is_ours(t.pid, t.start_time)
        assert not op.is_ours(t.pid, t.start_time + 1)
        assert not op.is_ours(os.getpid(), procinfo.start_time(os.getpid()))
    finally:
        t.popen.kill()
        t.popen.wait()


def test_cleanup_unlocks_a_locked_target_then_kills_it(tmp_path):
    op = owned.OwnedProcesses("tok123", str(tmp_path))
    t = op.start(sleeper("tok123"))
    call = FakeCall("locked")
    log = op.cleanup(call, "proprietary")
    assert call.ops == ["state", "unlock"]
    assert gone(t.pid) and log[0]["pid"] == t.pid and log[0]["steps"][-1] == "sigkill"


def test_cleanup_leaves_exited_targets_alone(tmp_path):
    op = owned.OwnedProcesses("tok123", str(tmp_path))
    t = op.start([PY, "-c", "pass", "--token", "tok123"])
    t.popen.wait()
    call = FakeCall("running")
    assert op.cleanup(call, "proprietary") == [] and call.ops == []


def test_run_file_records_targets_and_is_removed_by_close(tmp_path):
    op = owned.OwnedProcesses("tok123", str(tmp_path))
    t = op.start(sleeper("tok123"))
    rec = json.load(open(tmp_path / "tok123.json"))
    assert rec["targets"][0]["pid"] == t.pid
    op.cleanup(FakeCall("running"), "proprietary")
    op.close()
    assert not (tmp_path / "tok123.json").exists()


def test_reconcile_kills_leftovers_and_ignores_reused_pids(tmp_path):
    left = subprocess.Popen(sleeper("oldtok"))
    time.sleep(0.2)
    me = os.getpid()
    (tmp_path / "oldtok.json").write_text(json.dumps({"token": "oldtok", "targets": [
        {"pid": left.pid, "start_time": procinfo.start_time(left.pid), "kind": "hold", "via_parent": False},
        {"pid": me, "start_time": procinfo.start_time(me) + 7, "kind": "hold", "via_parent": False}]}))
    log = owned.reconcile(str(tmp_path), FakeCall("checkpointed"), "proprietary")
    left.wait(timeout=5)
    assert [x["pid"] for x in log] == [left.pid]
    assert not (tmp_path / "oldtok.json").exists()
