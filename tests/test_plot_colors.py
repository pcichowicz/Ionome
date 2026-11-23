"""Group colours: generic for any study and stable across plots."""
from analysis.utils.plotting import GROUP_PALETTE, QC_COLOR, group_colors


def test_qc_is_always_grey_and_other_groups_get_distinct_colours():
    c = group_colors(["tumour", "QC", "control", "tumour", None])
    assert c["QC"] == QC_COLOR
    assert c["control"] != c["tumour"] and QC_COLOR not in (c["control"], c["tumour"])
    assert set(c) == {"QC", "control", "tumour"}              # None is ignored


def test_colours_do_not_depend_on_which_groups_a_plot_shows():
    both = group_colors(["a", "b", "c"])
    assert group_colors(["c", "a", "b"]) == both                # input order is irrelevant
    assert group_colors(["a", "b", "c", "QC"])["b"] == both["b"]  # adding QC does not move the others


def test_more_groups_than_colours_repeat_the_palette():
    groups = [f"g{i:02d}" for i in range(len(GROUP_PALETTE) + 3)]
    c = group_colors(groups)
    assert c[groups[len(GROUP_PALETTE)]] == c[groups[0]]


def test_groups_take_the_palette_in_sorted_order():
    # fixed contract: the first group (alphabetically) gets the first palette colour, and so on; QC stays grey
    c = group_colors(["b_group", "a_group", "c_group", "QC"])
    assert c == {"a_group": GROUP_PALETTE[0], "b_group": GROUP_PALETTE[1], "c_group": GROUP_PALETTE[2], "QC": QC_COLOR}
