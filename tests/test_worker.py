import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

from fakes import FakeLib
from gpunap import worker
from gpunap.driver import Driver

PY = sys.executable
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_run_json_returns_the_last_json_line():
    d = worker.run_json([PY, "-c", "print('noise'); print('{\"a\": 1}'); print('{\"a\": 2}')"], 10)
    assert d["a"] == 2 and d["hung"] is False


def test_a_call_that_outlives_its_limit_is_killed_and_reported_hung():
    d = worker.run_json([PY, "-c", "import time; time.sleep(30)"], 0.5)
    assert d["hung"] is True and d["result"] == "hung" and d["wall_s"] < 10


def test_no_output_keeps_stderr_and_exit_code():
    d = worker.run_json([PY, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], 10)
    assert d["result"] == "no-output" and "boom" in d["stderr"] and d["rc"] == 3


def test_unknown_op_is_rejected():
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([PY, "-m", "gpunap.worker", "bogus", "1"], capture_output=True, text=True, env=env)
    assert p.returncode == 2


def test_single_op_prints_one_json_line():
    out = io.StringIO()
    with redirect_stdout(out):
        rc = worker.main(["unlock", "5"], driver=Driver(lib=FakeLib()))
    d = json.loads(out.getvalue().strip().splitlines()[-1])
    assert rc == 0 and d["op"] == "unlock" and d["result"] == "OK"


def test_pause_reports_every_call_and_the_last_result():
    out = io.StringIO()
    with redirect_stdout(out):
        worker.main(["pause", "5", "--timeout-ms", "700"], driver=Driver(lib=FakeLib(codes={"checkpoint": 304})))
    d = json.loads(out.getvalue().strip().splitlines()[-1])
    assert [c["op"] for c in d["calls"]] == ["lock", "checkpoint", "unlock"]
    assert d["result"] == "CUDA_ERROR_OPERATING_SYSTEM"


def test_call_builds_a_worker_command_that_finds_the_package(monkeypatch):
    seen = {}

    def fake_run_json(argv, limit_s, env=None):
        seen.update(argv=argv, limit=limit_s, env=env)
        return {"result": "OK", "hung": False}

    monkeypatch.setattr(worker, "run_json", fake_run_json)
    d = worker.call("lock", 12, 7.5, timeout_ms=5000)
    assert seen["argv"][1:] == ["-m", "gpunap.worker", "lock", "12", "--timeout-ms", "5000"]
    assert seen["limit"] == 7.5 and d["op"] == "lock"
    assert SRC in seen["env"]["PYTHONPATH"].split(os.pathsep)


def test_announce_prints_a_calling_line_before_the_result():
    out = io.StringIO()
    with redirect_stdout(out):
        worker.main(["checkpoint", "5", "--announce"], driver=Driver(lib=FakeLib()))
    first, last = [json.loads(x) for x in out.getvalue().strip().splitlines()]
    assert first["event"] == "CALLING" and first["op"] == "checkpoint"
    assert first["time"] <= last["start"] and last["op"] == "checkpoint"
