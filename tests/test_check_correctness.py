from gpunap import results
from gpunap.check import correctness as co

OK = {"result": "OK", "hung": False}
STEPS = [f"0x1.{i:04x}p-1" for i in range(5)]


def _run(steps=STEPS, end="abc", rc=0, pauses=None):
    return {"steps": list(steps), "end": end, "rc": rc, "pauses": pauses or []}


def _pause(step=2):
    return {"at_step": step, "landed_step": step, "pause": OK, "resume": OK}


def test_parse_reads_steps_in_order_and_the_end_hash():
    lines = ["READY 5", "STEP 0 0x1.8p-1", "STEP 1 0x1.4p-1", "END deadbeef"]
    assert co.parse(lines) == (["0x1.8p-1", "0x1.4p-1"], "deadbeef")


def test_parse_without_end():
    assert co.parse(["READY 5", "STEP 0 0x1p0"]) == (["0x1p0"], None)


def test_schedule_is_seeded_sorted_and_inside_the_run():
    a, b = co.schedule(300, 20, seed=0), co.schedule(300, 20, seed=0)
    assert a == b and len(a) == 20 and a == sorted(a)
    assert all(co.MARGIN <= s < 300 - co.MARGIN for s, _ in a)
    assert all(co.WAIT_S[0] <= w <= co.WAIT_S[1] for _, w in a)


def test_identical_runs_are_ok():
    v, why = co.verdict([_run(), _run()], _run(pauses=[_pause()] * 3))
    assert v == results.OK and "3 pauses" in why and "bit-identical" in why


def test_references_that_differ_are_invalid():
    other = _run(steps=STEPS[:4] + ["0x1p0"])
    assert co.verdict([_run(), other], _run(pauses=[_pause()]))[0] == results.INVALID


def test_a_different_loss_is_a_hazard_naming_the_step():
    paused = _run(steps=STEPS[:3] + ["0x1p0", STEPS[4]], pauses=[_pause()])
    v, why = co.verdict([_run(), _run()], paused)
    assert v == results.HAZARD and "step 3" in why


def test_a_different_weight_hash_is_a_hazard():
    v, why = co.verdict([_run(), _run()], _run(end="fff", pauses=[_pause()]))
    assert v == results.HAZARD and "weights" in why


def test_a_crashed_paused_run_is_a_hazard():
    v, why = co.verdict([_run(), _run()], _run(steps=STEPS[:2], end=None, rc=-11, pauses=[_pause()]))
    assert v == results.HAZARD and "-11" in why


def test_a_failed_pause_is_a_hazard():
    p = dict(_pause(), resume={"result": "CUDA_ERROR_ILLEGAL_STATE", "hung": False})
    assert co.verdict([_run(), _run()], _run(pauses=[p]))[0] == results.HAZARD


def test_no_pause_landed_is_invalid():
    assert co.verdict([_run(), _run()], _run(pauses=[]))[0] == results.INVALID
