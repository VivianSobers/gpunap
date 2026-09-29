from gpunap import results
from gpunap.check import controller_death as cd

OK = {"result": "OK", "hung": False}


def _phase(phase, states, recover=OK, verify="ok", killed_during=True, rc=0):
    return {"phase": phase, "killed_during_call": killed_during, "states": states,
            "recover": recover, "verify": verify, "rc": rc}


def test_pick_mb_takes_the_largest_size_that_fits():
    assert cd.pick_mb(free_mb=30000, host_mb=60000) == 16384
    assert cd.pick_mb(free_mb=12000, host_mb=60000) == 8192
    assert cd.pick_mb(free_mb=1000, host_mb=60000) is None


def test_outcome_names_what_the_driver_did_without_its_controller():
    assert cd.outcome("checkpoint", ["checkpointed"] * 3) == "the checkpoint completed without the controller"
    assert cd.outcome("restore", ["locked"] * 3) == "the restore completed, the job stayed locked"
    assert cd.outcome("checkpoint", ["locked"]) == "the checkpoint was abandoned, the job stayed locked"
    assert cd.outcome("restore", ["running"]) == "the job is running"
    assert cd.outcome("restore", ["failed"]) == "the job is in the failed state"


def test_recovered_phases_are_ok_and_describe_the_outcome():
    v, why = cd.verdict([_phase("checkpoint", ["checkpointed"]), _phase("restore", ["locked"])])
    assert v == results.OK
    assert "checkpoint: the checkpoint completed without the controller" in why
    assert "restore: the restore completed, the job stayed locked" in why


def test_a_call_that_finished_before_the_kill_is_invalid():
    v, why = cd.verdict([_phase("checkpoint", ["checkpointed"], killed_during=False)])
    assert v == results.INVALID and "finished before the kill" in why


def test_failed_state_is_a_hazard():
    v, _ = cd.verdict([_phase("restore", ["failed"], recover={"result": "CUDA_ERROR_ILLEGAL_STATE", "hung": False})])
    assert v == results.HAZARD


def test_bad_data_after_recovery_is_a_hazard():
    v, why = cd.verdict([_phase("checkpoint", ["checkpointed"], verify="bad chunk 2")])
    assert v == results.HAZARD and "data check bad chunk 2" in why


def test_failed_recovery_is_a_hazard():
    v, why = cd.verdict([_phase("checkpoint", ["checkpointed"], recover={"result": "hung", "hung": True})])
    assert v == results.HAZARD and "recovery returned hung" in why


def test_no_size_fits_is_skipped():
    assert cd.verdict([{"phase": "checkpoint", "skipped": "not enough free GPU memory"}])[0] == results.SKIPPED


def test_recovery_step_follows_the_state():
    assert cd.recovery("checkpointed") == "resume"
    assert cd.recovery("locked") == "unlock"
    assert cd.recovery("running") is None
    assert cd.recovery("failed") is None
