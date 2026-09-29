from repro_sync import stamp, verdict


def test_not_checkpointed_is_invalid():
    assert verdict("locked", 10.0, 11.0, 11.1, "0", None, 0) == "invalid: pause did not checkpoint"


def test_no_call_line_is_invalid():
    assert verdict("checkpointed", 10.0, None, None, None, None, 1) == "invalid: child never reached the call"


def test_call_before_checkpoint_is_invalid():
    assert verdict("checkpointed", 10.0, 9.5, 9.6, "0", None, 0) == "invalid: call came before the checkpoint"


def test_timeout_is_hung():
    assert verdict("checkpointed", 10.0, 11.0, None, None, 20.0, "timeout") == "hung after resume"


def test_signal_is_crash():
    assert verdict("checkpointed", 10.0, 11.0, None, None, None, -11) == "crash: SIGSEGV"


def test_nonzero_exit_without_done_is_error():
    assert verdict("checkpointed", 10.0, 11.0, None, None, 20.0, 1) == "error: exit 1"


def test_done_before_resume_returned_while_checkpointed():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "0", 20.0, 0) == "returned while checkpointed"


def test_done_with_no_resume_returned_while_checkpointed():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "0", None, 0) == "returned while checkpointed"


def test_error_code_is_reported():
    assert verdict("checkpointed", 10.0, 11.0, 11.0, "999", None, 0) == "returned while checkpointed (rc=999)"


def test_done_after_resume_waited():
    assert verdict("checkpointed", 10.0, 11.0, 20.5, "0", 20.0, 0) == "waited for resume"


def test_stamp_reads_time_and_rc():
    out = ["READY", "CALL 11.2500", "DONE 20.5000 rc=0"]
    assert stamp(out, "CALL") == (11.25, None)
    assert stamp(out, "DONE") == (20.5, "0")
    assert stamp(out, "MISSING") == (None, None)


def test_error_after_resume_is_reported():
    assert verdict("checkpointed", 10.0, 11.0, 20.5, "999", 20.0, 0) == "waited for resume (rc=999)"


def test_data_mismatch_after_resume_is_reported():
    assert verdict("checkpointed", 10.0, 11.0, 20.5, "data_mismatch", 20.0, 0) == "waited for resume (rc=data_mismatch)"
