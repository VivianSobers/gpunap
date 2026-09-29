from backtrace_sync import children_of


def fake_proc(root, pid, ppid, cmd):
    d = root / str(pid)
    d.mkdir()
    (d / "stat").write_text(f"{pid} (python3) S {ppid} 1 1 0 -1\n")
    (d / "cmdline").write_text("\0".join(cmd) + "\0")


def test_finds_the_child_of_gdb(tmp_path):
    fake_proc(tmp_path, 100, 1, ["gdb"])
    fake_proc(tmp_path, 101, 100, ["python3", "repro_sync.py", "child"])
    fake_proc(tmp_path, 102, 1, ["python3", "repro_sync.py", "child"])
    assert children_of(100, str(tmp_path)) == [101]


def test_ignores_non_process_entries_and_vanished_ones(tmp_path):
    fake_proc(tmp_path, 100, 1, ["gdb"])
    (tmp_path / "self").mkdir()
    (tmp_path / "200").mkdir()  # exited between listdir and read: no stat file
    assert children_of(100, str(tmp_path)) == []


def test_command_names_with_spaces_and_parens(tmp_path):
    d = tmp_path / "101"
    d.mkdir()
    (d / "stat").write_text("101 (my (odd) proc) S 100 1 1 0 -1\n")
    assert children_of(100, str(tmp_path)) == [101]
