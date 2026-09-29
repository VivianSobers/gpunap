import json
import os

import pytest

import gpunap
from gpunap import cli, results
from gpunap.check import registry, runner


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and gpunap.__version__ in capsys.readouterr().out


def test_list_prints_every_check(capsys):
    assert cli.main(["check", "--list"]) == 0
    out = capsys.readouterr().out
    for c in registry.CHECKS:
        assert c.name in out
    assert "full" in out


def test_unknown_only_name_is_rejected(capsys):
    assert cli.main(["check", "--only", "nope"]) == 2
    assert "unknown check: nope" in capsys.readouterr().err


def _fake_run(verdicts, partial=False, seen=None):
    def run(checks, ctx, probe=None, on_result=None):
        if seen is not None:
            seen.update(names=[c.name for c in checks], full=ctx.full, label=ctx.label)
        rep = results.Report({"label": ctx.label})
        for c, v in zip(checks, verdicts):
            r = results.CheckResult(c.name, v, f"{c.name} summary")
            rep.results.append(r)
            on_result(r)
        rep.partial = partial
        return rep
    return run


def test_check_writes_a_report_and_prints_each_result(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(runner, "run", _fake_run([results.OK, results.HAZARD], seen=seen))
    rc = cli.main(["check", "--only", "api,roundtrip", "--label", "box", "--out", str(tmp_path)])
    assert rc == 0 and seen == {"names": ["api", "roundtrip"], "full": False, "label": "box"}
    out = capsys.readouterr().out
    assert "2 checks" in out and "roundtrip" in out and "hazard" in out
    files = os.listdir(tmp_path)
    assert len(files) == 1 and files[0].startswith("gpunap-check-box-")
    d = json.load(open(tmp_path / files[0]))
    assert d["counts"]["hazard"] == 1


def test_an_error_verdict_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "run", _fake_run([results.ERROR]))
    assert cli.main(["check", "--only", "api", "--out", str(tmp_path)]) == 1
    assert ": 1 check\n" in capsys.readouterr().out


def test_a_partial_run_exits_130(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run", _fake_run([results.OK], partial=True))
    assert cli.main(["check", "--only", "api", "--out", str(tmp_path)]) == 130
    assert len(os.listdir(tmp_path)) == 1


def test_full_is_refused_on_the_open_module(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "module_kind", lambda: "open")
    monkeypatch.setattr(runner, "run", _fake_run([]))
    assert cli.main(["check", "--full", "--out", str(tmp_path)]) == 2
    assert "open kernel module" in capsys.readouterr().err
    assert os.listdir(tmp_path) == []


def test_sysinfo_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(cli.sysinfo, "collect", lambda driver=None: {"gpu": "X"})
    assert cli.main(["sysinfo"]) == 0
    assert json.loads(capsys.readouterr().out) == {"gpu": "X"}


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 2
    assert "check" in capsys.readouterr().out


def test_python_dash_m_runs_the_cli():
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    p = subprocess.run([sys.executable, "-m", "gpunap", "--version"], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=src))
    assert p.returncode == 0 and gpunap.__version__ in p.stdout


def test_sigterm_during_a_run_is_handled_like_ctrl_c(tmp_path, monkeypatch):
    import signal
    seen = {}

    def run(checks, ctx, probe=None, on_result=None):
        handler = signal.getsignal(signal.SIGTERM)
        try:
            handler(signal.SIGTERM, None)
        except KeyboardInterrupt:
            seen["interrupt"] = True
        return results.Report({})
    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(runner, "run", run)
    cli.main(["check", "--only", "api", "--out", str(tmp_path)])
    assert seen == {"interrupt": True} and signal.getsignal(signal.SIGTERM) is before
