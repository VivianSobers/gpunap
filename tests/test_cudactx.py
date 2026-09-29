from gpunap.check import cudactx as cc


def test_pattern_values_differ_by_chunk_and_seed_and_are_never_zero():
    vals = {cc.pattern_value(s, i) for s in range(3) for i in range(50)}
    assert len(vals) == 150 and 0 not in vals
    assert all(0 < v < 2 ** 32 for v in vals)


def test_chunk_plan_covers_the_buffer_exactly():
    plan = cc.chunk_plan(150 * 2 ** 20, 64 * 2 ** 20)
    assert [(off, words) for off, words in plan] == [(0, 16 * 2 ** 20), (64 * 2 ** 20, 16 * 2 ** 20), (128 * 2 ** 20, 22 * 2 ** 18)]
    assert sum(w * 4 for _, w in plan) == 150 * 2 ** 20


def test_sample_offsets_stay_inside_the_chunk():
    offs = cc.sample_offsets(64 * 2 ** 20, 16 * 2 ** 20, 4096)
    assert offs[0] == 64 * 2 ** 20
    assert all(64 * 2 ** 20 <= o and o + 4096 <= 80 * 2 ** 20 for o in offs)
    assert len(set(offs)) == 3


def test_sample_offsets_for_a_tiny_chunk():
    assert cc.sample_offsets(0, 1024, 4096) == [0]


def test_check_words():
    word = (7).to_bytes(4, "little")
    assert cc.check_words(word * 8, 7) is None
    assert "word 2" in cc.check_words(word * 2 + (8).to_bytes(4, "little") + word, 7)
