import io
from contextlib import redirect_stdout

import pytest

from gpunap.check import targets


def run_serve(text, verify):
    out, calls = io.StringIO(), []
    with redirect_stdout(out):
        rc = targets.serve(verify, io.StringIO(text), on_exit=lambda: calls.append("exit"))
    return rc, out.getvalue().splitlines(), calls


def test_verify_reports_ok_and_bad():
    results = iter([None, "chunk 3 offset 0: word 0 is 0x0"])
    rc, lines, calls = run_serve("VERIFY\nVERIFY\nEXIT\nVERIFY\n", lambda: next(results))
    assert rc == 0 and calls == ["exit"]
    assert lines == ["VERIFIED ok", "VERIFIED bad chunk 3 offset 0: word 0 is 0x0"]


def test_end_of_input_ends_the_loop():
    rc, lines, calls = run_serve("", lambda: None)
    assert rc == 0 and lines == [] and calls == ["exit"]


def test_unknown_commands_are_echoed_not_fatal():
    rc, lines, _ = run_serve("DANCE\nVERIFY\n", lambda: None)
    assert lines == ["UNKNOWN DANCE", "VERIFIED ok"]


def test_a_verify_that_raises_reports_bad():
    def boom():
        raise RuntimeError("cuMemcpyDtoH failed rc=719")
    _, lines, _ = run_serve("VERIFY\n", boom)
    assert lines == ["VERIFIED bad RuntimeError: cuMemcpyDtoH failed rc=719"]


def test_kind_list_and_token_required():
    assert {"hold", "managed", "hog", "sync_first", "exit_after", "parent_child", "torch_sync_first",
            "torch_nccl", "torch_ipc", "torch_train"} <= set(targets.KINDS)
    with pytest.raises(SystemExit):
        targets.main(["hold"])
