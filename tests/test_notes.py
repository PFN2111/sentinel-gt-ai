"""Notes interpretation: negation, unit naming, first-out."""

from __future__ import annotations

import pytest

from sentinel.notes import first_out_is_spread, parse_notes


@pytest.mark.parametrize(
    "text,flag,expected",
    [
        ("No TC or harness work since that outage.", "tc_work", False),
        ("Nothing unusual but TC12 was replaced last month.", "tc_work", True),
        ("TC harness repaired yesterday.", "tc_work", True),
        ("No load changes or fuel transfers reported.", "load_change", False),
        ("Load ramp from 30 to 48 MW just before the alarm.", "load_change", True),
        ("Unit was re-tuned last month; dynamics are back again.", "recent_tuning", True),
        ("Unit was re-tuned last month; dynamics are back again.", "repeat", True),
        ("Not tuned since commissioning.", "recent_tuning", False),
        ("Operators heard a humming tone.", "operator_reports_noise", True),
    ],
)
def test_flags_respect_negation(text, flag, expected):
    assert parse_notes(text).has(flag) is expected


@pytest.mark.parametrize("text,unit", [("Unit 2 at baseload", "Unit 2"), ("GT3 tripped", "GT3"), ("CT 2 hum", "CT2"), ("no unit given", None)])
def test_unit_naming(text, unit):
    assert parse_notes(text).unit == unit


def test_first_out_and_tc_numbers():
    info = parse_notes("GT1 7FA.03 first-out: high exhaust spread. TC12 and TC 13 replaced.")
    assert first_out_is_spread(info) is True
    assert info.frame == "7FA.03"
    assert info.tcs_mentioned == [12, 13]
    assert first_out_is_spread(parse_notes("first out was flame loss")) is False
    assert first_out_is_spread(parse_notes("no first out given")) is None
