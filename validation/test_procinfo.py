from procinfo import thread_names, uvm_maps


def make_task(root, pid, tid, comm):
    d = root / str(pid) / "task" / str(tid)
    d.mkdir(parents=True)
    (d / "comm").write_text(comm + "\n")


def test_thread_names_lists_every_task(tmp_path):
    make_task(tmp_path, 10, 10, "python3")
    make_task(tmp_path, 10, 11, "pt_nccl_watchdg")
    make_task(tmp_path, 10, 12, "cuda-EvtHandlr")
    assert thread_names(10, str(tmp_path)) == ["cuda-EvtHandlr", "pt_nccl_watchdg", "python3"]


def test_thread_names_of_missing_process_is_none(tmp_path):
    assert thread_names(99, str(tmp_path)) is None


def test_uvm_maps_counts_only_uvm_lines(tmp_path):
    (tmp_path / "10").mkdir()
    (tmp_path / "10" / "maps").write_text(
        "7f00-7f10 rw-s 00000000 00:05 12 /dev/nvidia-uvm\n"
        "7f10-7f20 rw-s 00000000 00:05 13 /dev/nvidiactl\n"
        "7f20-7f30 rw-s 00000000 00:05 12 /dev/nvidia-uvm\n")
    assert len(uvm_maps(10, str(tmp_path))) == 2


def test_uvm_maps_of_missing_process_is_none(tmp_path):
    assert uvm_maps(99, str(tmp_path)) is None
