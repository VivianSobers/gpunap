from pcie_link import FIELDS, parse


def test_parse_maps_fields_to_ints():
    line = "4, 4, 4, 5, 16, 16"
    assert parse(line) == dict(zip(FIELDS, [4, 4, 4, 5, 16, 16]))


def test_parse_keeps_non_numeric_values_as_text():
    line = "[N/A], 4, 4, 5, 16, 16"
    assert parse(line)[FIELDS[0]] == "[N/A]"
