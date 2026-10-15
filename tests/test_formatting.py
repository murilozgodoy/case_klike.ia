"""Testes dos formatadores numéricos usados nos gráficos e no log de limpeza."""

import pytest

from klike.formatting import fmt_compact, fmt_number, fmt_pct, fmt_signed, fmt_signed_pct


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [(1234.5, 1, "1.234,5"), (1_000_000, 0, "1.000.000"), (0.15, 2, "0,15")],
)
def test_fmt_number_uses_brazilian_separators(value, decimals, expected):
    assert fmt_number(value, decimals) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (0.001, "0,001"),
        (0.05, "0,05"),
        (1, "1"),
        (1_000, "1 mil"),
        (1_500, "1,5 mil"),
        (120_000, "120 mil"),
        (1_000_000, "1 mi"),
        (14_000_000, "14 mi"),
    ],
)
def test_fmt_compact_drops_useless_zeros(value, expected):
    assert fmt_compact(value) == expected


def test_fmt_pct_shows_decimals_only_when_they_matter():
    assert fmt_pct(0.15) == "15%"
    assert fmt_pct(0.025) == "2,5%"
    assert fmt_pct(0.092, decimals=1) == "9,2%"


def test_fmt_signed_shows_direction():
    assert fmt_signed_pct(0.51) == "+51%"
    assert fmt_signed_pct(-0.39) == "−39%"
    assert fmt_signed(-5.1, 1, " pts") == "−5,1 pts"
    assert fmt_signed_pct(0.001) == "0%"
