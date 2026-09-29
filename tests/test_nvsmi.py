from gpunap import nvsmi


def test_parse_csv_maps_fields_and_numbers():
    rows = nvsmi.parse_csv("NVIDIA GeForce RTX 4090, 595.84, 694, 23381\n", ("name", "driver", "used", "free"))
    assert rows == [{"name": "NVIDIA GeForce RTX 4090", "driver": "595.84", "used": 694, "free": 23381}]


def test_parse_csv_keeps_not_available_as_text():
    rows = nvsmi.parse_csv("1234, [N/A]\n5678, 512\n", ("pid", "used_mb"))
    assert rows == [{"pid": 1234, "used_mb": "[N/A]"}, {"pid": 5678, "used_mb": 512}]


def test_parse_csv_of_nothing_is_empty():
    assert nvsmi.parse_csv("", ("a",)) == [] and nvsmi.parse_csv("\n\n", ("a",)) == []


def test_queries_return_nothing_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(nvsmi, "EXE", "nvidia-smi-does-not-exist")
    assert nvsmi.gpu() is None and nvsmi.apps() == [] and nvsmi.pcie() is None


def test_apps_totals(monkeypatch):
    monkeypatch.setattr(nvsmi, "query", lambda args: "11, 300\n12, [N/A]\n13, 700\n")
    assert nvsmi.apps() == [{"pid": 11, "used_mb": 300}, {"pid": 12, "used_mb": "[N/A]"}, {"pid": 13, "used_mb": 700}]
    assert nvsmi.apps_mb() == 1000
