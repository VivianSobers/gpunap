import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("compare", os.path.join(HERE, "gpu", "compare.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

EXPECT = {"checks": {"a": {"595": "ok", "580": "ok"}, "b": {"595": "hazard", "580": "ok"}}}


def report(verdicts, driver="595.58.03", before=400, after=420, partial=False):
    return {"header": {"system": {"driver": driver}, "gpu_memory_used_mb_before": before,
                       "gpu_memory_used_mb_after": after}, "partial": partial,
            "results": [{"name": n, "verdict": v} for n, v in verdicts.items()]}


def test_branch_is_the_major_version():
    assert compare.branch("595.58.03") == "595" and compare.branch("580.95.05") == "580"
    assert compare.branch(None) is None


def test_matching_report_has_no_problems():
    rows, problems = compare.compare(report({"a": "ok", "b": "hazard"}), EXPECT)
    assert problems == [] and [r[3] for r in rows] == ["match", "match"]


def test_driver_decides_the_expected_column():
    _, problems = compare.compare(report({"a": "ok", "b": "hazard"}, driver="580.95.05"), EXPECT)
    assert problems == ["b: expected ok on 580, got hazard"]


def test_missing_and_skipped_checks_are_problems():
    rows, problems = compare.compare(report({"a": "skipped"}), EXPECT)
    assert "a: skipped" in problems[0] and "b: not in the report" in problems[1]
    _, problems = compare.compare(report({"a": "skipped", "b": "hazard"}), EXPECT, allow_skipped=True)
    assert problems == []


def test_leftover_gpu_memory_and_partial_runs_are_problems():
    _, problems = compare.compare(report({"a": "ok", "b": "hazard"}, before=400, after=501), EXPECT)
    assert problems == ["GPU memory used went from 400 MB to 501 MB (more than 100 MB)"]
    _, problems = compare.compare(report({"a": "ok", "b": "hazard"}, partial=True), EXPECT)
    assert problems == ["the report is partial"]


def test_unknown_driver_branch_is_a_problem():
    _, problems = compare.compare(report({"a": "ok"}, driver="610.1"), EXPECT)
    assert problems == ["no expectations for driver branch 610"]
