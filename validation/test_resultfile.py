import pytest

from resultfile import claim


def test_free_path_is_claimed_as_is(tmp_path):
    p = tmp_path / "fault_x.json"
    assert claim(str(p)) == str(p)
    assert p.exists()


def test_existing_file_is_left_alone(tmp_path):
    p = tmp_path / "fault_x.json"
    p.write_text("earlier run")
    assert claim(str(p)) == str(tmp_path / "fault_x-2.json")
    assert p.read_text() == "earlier run"


def test_suffix_counts_up(tmp_path):
    (tmp_path / "r.json").write_text("1")
    (tmp_path / "r-2.json").write_text("2")
    assert claim(str(tmp_path / "r.json")) == str(tmp_path / "r-3.json")


def test_repeated_claims_never_collide(tmp_path):
    base = str(tmp_path / "r.json")
    assert len({claim(base) for _ in range(5)}) == 5


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        claim(str(tmp_path / "nope" / "r.json"))
