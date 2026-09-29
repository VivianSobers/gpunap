from gpunap import procinfo as pi


def proc_entry(root, pid, comm="python3", state="S", ppid=1, start=12345, cmd=("python3",), maps="", cgroup="0::/user.slice/app.scope"):
    d = root / str(pid)
    (d / "task" / str(pid)).mkdir(parents=True)
    rest = [state, str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 5
    (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(rest) + "\n")
    (d / "cmdline").write_text("\0".join(cmd) + "\0")
    (d / "maps").write_text(maps)
    (d / "cgroup").write_text(cgroup + "\n")
    (d / "task" / str(pid) / "comm").write_text(comm + "\n")
    return d


def test_state_and_start_time(tmp_path):
    proc_entry(tmp_path, 10, state="T", start=999)
    assert pi.stat_state(10, str(tmp_path)) == "T"
    assert pi.start_time(10, str(tmp_path)) == 999


def test_command_names_with_spaces_and_parentheses(tmp_path):
    proc_entry(tmp_path, 11, comm="my (odd) name", state="R", ppid=10, start=5)
    assert pi.stat_state(11, str(tmp_path)) == "R"
    assert pi.children(10, str(tmp_path)) == [11]


def test_vanished_process_gives_none(tmp_path):
    for f in (pi.stat_state, pi.start_time, pi.cmdline, pi.uvm_maps, pi.thread_names, pi.cgroup_path):
        assert f(404, str(tmp_path)) is None


def test_cmdline_splits_on_nul(tmp_path):
    proc_entry(tmp_path, 12, cmd=("python3", "-m", "gpunap.check.targets", "hold", "--token", "abc"))
    assert pi.cmdline(12, str(tmp_path))[-1] == "abc"


def test_children_ignores_non_numeric_and_broken_entries(tmp_path):
    proc_entry(tmp_path, 20, ppid=1)
    proc_entry(tmp_path, 21, ppid=20)
    (tmp_path / "self").mkdir()
    (tmp_path / "99").mkdir()
    assert pi.children(20, str(tmp_path)) == [21]


def test_uvm_maps_counts_only_uvm_lines(tmp_path):
    proc_entry(tmp_path, 30, maps="7f00-7f10 rw-s 0 00:05 12 /dev/nvidia-uvm\n"
                                   "7f10-7f20 rw-s 0 00:05 13 /dev/nvidiactl\n"
                                   "7f20-7f30 rw-s 0 00:05 12 /dev/nvidia-uvm\n")
    assert len(pi.uvm_maps(30, str(tmp_path))) == 2


def test_thread_names_sorted(tmp_path):
    d = proc_entry(tmp_path, 40, comm="python3")
    (d / "task" / "41").mkdir()
    (d / "task" / "41" / "comm").write_text("pt_nccl_watchdg\n")
    assert pi.thread_names(40, str(tmp_path)) == ["pt_nccl_watchdg", "python3"]


def test_mem_available_from_meminfo_text():
    assert pi.mem_available_mb("MemTotal: 100 kB\nMemAvailable:   2097152 kB\n") == 2048


def test_cgroup_limit_is_the_tightest_on_the_path(tmp_path):
    proc = tmp_path / "proc"
    sysfs = tmp_path / "cg"
    proc_entry(proc, 50, cgroup="0::/user.slice/run-x.scope")
    (sysfs / "user.slice" / "run-x.scope").mkdir(parents=True)
    (sysfs / "memory.max").write_text("max\n")
    (sysfs / "user.slice" / "memory.max").write_text("8589934592\n")
    (sysfs / "user.slice" / "run-x.scope" / "memory.max").write_text("2147483648\n")
    (sysfs / "user.slice" / "run-x.scope" / "memory.current").write_text("1073741824\n")
    m = pi.cgroup_memory(50, str(proc), str(sysfs))
    assert m == {"path": "/user.slice/run-x.scope", "max_bytes": 2147483648, "limit_at": "/user.slice/run-x.scope",
                 "current_bytes": 1073741824}


def test_cgroup_without_limit(tmp_path):
    proc = tmp_path / "proc"
    sysfs = tmp_path / "cg"
    proc_entry(proc, 51, cgroup="0::/a")
    (sysfs / "a").mkdir(parents=True)
    (sysfs / "a" / "memory.max").write_text("max\n")
    m = pi.cgroup_memory(51, str(proc), str(sysfs))
    assert m["max_bytes"] is None and m["current_bytes"] is None


def test_vm_rss_mb(tmp_path):
    d = proc_entry(tmp_path, 60)
    (d / "status").write_text("Name:\tpython3\nVmRSS:\t  819200 kB\nVmLck:\t0 kB\n")
    assert pi.vm_rss_mb(60, str(tmp_path)) == 800
    assert pi.vm_rss_mb(61, str(tmp_path)) is None
