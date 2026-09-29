import json
import socket

import pytest

from gpunap import results as r


def test_claim_never_overwrites(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("old")
    assert r.claim(str(p)) == str(tmp_path / "x-2.json")
    assert r.claim(str(p)) == str(tmp_path / "x-3.json")
    assert p.read_text() == "old"


def test_claim_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        r.claim(str(tmp_path / "nope" / "x.json"))


def test_default_name_is_safe_and_has_no_hostname():
    name = r.default_name("gpu 1/../x", 0.0)
    assert name == "gpunap-check-gpu-1-..-x-19700101T000000Z.json"
    assert "/" not in name and socket.gethostname() not in name


def test_check_result_rejects_unknown_verdicts():
    with pytest.raises(ValueError):
        r.CheckResult("api", "fine", "")


def test_report_counts_and_round_trips(tmp_path):
    rep = r.Report(header={"label": "gpu1"}, results=[r.CheckResult("api", r.OK, "5 functions", {"x": 1}, 0.1),
                                                       r.CheckResult("sigstop", r.HAZARD, "lock hung", {}, 30.0)])
    path = r.write(rep, str(tmp_path), "gpu1", 0.0)
    d = json.load(open(path))
    assert d["counts"] == {"ok": 1, "hazard": 1, "skipped": 0, "invalid": 0, "error": 0}
    assert d["results"][1]["verdict"] == "hazard" and d["partial"] is False and d["gpunap"] == "0.1.0"


def test_write_redacts_home_user_host_and_uid(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_identity", lambda: {"home": "/home/alice", "user": "alice", "host": "lab-box-7", "uid": "4242"})
    rep = r.Report(header={"label": "gpu1"}, results=[r.CheckResult("x", r.OK, "ran on lab-box-7", {
        "trace": 'File "/home/alice/src/gpunap/x.py", line 3', "who": "alice ran it",
        "cg": "/user.slice/user-4242.slice/user@4242.service/app.slice", "n": 4242})])
    text = open(r.write(rep, str(tmp_path), "gpu1", 0.0)).read()
    assert "alice" not in text and "lab-box-7" not in text and "user-4242" not in text and "user@4242" not in text
    assert "~/src/gpunap/x.py" in text and "HOST" in text and '"n": 4242' in text
