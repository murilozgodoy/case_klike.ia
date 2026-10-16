"""Testes do suporte histórico: contagem de campanhas parecidas em qualquer recorte.

O recommendations engine vai usar essas contagens para não extrapolar. Os testes conferem
os casos de pouco ou nenhum suporte conhecidos da Parte 1.
"""

import pandas as pd
import pytest

from klike.config import PERFORMANCE_COLS, TARGET
from klike.data import load_clean
from klike.support import SUPPORT_COLUMNS, HistoricalSupport, campaign_profiles

CAMPAIGN = {
    "platform": "LinkedIn",
    "category": "SaaS",
    "objective": "conversions",
    "target_audience_age": "35-44",
    "is_retargeting": False,
    "video_duration_s": 45,
    "format": "vertical",
    "has_subtitle": None,
    "has_cta": True,
    "has_hook": True,
    "has_face": False,
    "text_density": "low",
    "music_voice_ratio": 0.2,
}


@pytest.fixture(scope="module")
def clean_data():
    return load_clean()


@pytest.fixture(scope="module")
def support(clean_data):
    return HistoricalSupport.from_campaigns(clean_data)


def test_support_keeps_only_pre_flight_attributes(support):
    assert list(support.profiles.columns) == SUPPORT_COLUMNS
    assert TARGET not in support.profiles.columns
    assert not set(PERFORMANCE_COLS) & set(support.profiles.columns)
    assert len(support.profiles) == support.count() == 500


def test_counts_any_subset_of_attributes(support):
    assert support.count(platform="TikTok", format="horizontal") == 0
    assert support.count(platform="LinkedIn", format="vertical") == 11
    assert support.count(platform="TikTok", duration_band="mais de 30 s") == 5
    assert support.count(platform="Meta", has_hook=True) + support.count(
        platform="Meta", has_hook=False
    ) == support.count(platform="Meta")


def test_missing_values_have_their_own_band(support, clean_data):
    missing = clean_data[["video_duration_s", "music_voice_ratio", "has_subtitle"]].isna().sum()
    assert support.count(duration_band="desconhecida") == missing["video_duration_s"]
    assert support.count(music_band="desconhecida") == missing["music_voice_ratio"]
    assert support.count(subtitle_status="desconhecido") == missing["has_subtitle"]


def test_tables_include_combinations_without_campaigns(support):
    table = support.table("platform", "format")
    assert len(table) == 9
    assert table.loc[("TikTok", "horizontal")] == 0
    assert table.sum() == 500


def test_unknown_columns_and_values_are_rejected(support):
    with pytest.raises(ValueError, match="Coluna de suporte desconhecida: 'duration'"):
        support.count(duration=30)
    with pytest.raises(ValueError, match="platform: valor desconhecido 'YouTube'"):
        support.count(platform="YouTube")
    with pytest.raises(ValueError, match="Coluna de suporte desconhecida"):
        support.table("klike_score")


def test_counts_for_a_campaign_widen_the_context_step_by_step(support):
    levels = [
        ["platform", "format", "category", "objective"],
        ["platform", "format", "category"],
        ["platform", "format"],
        ["platform"],
        [],
    ]
    counts = support.counts_for(CAMPAIGN, levels)
    assert list(counts.index) == [
        "platform + format + category + objective",
        "platform + format + category",
        "platform + format",
        "platform",
        "todas",
    ]
    assert counts.is_monotonic_increasing
    assert counts["platform + format"] == 11
    assert counts["todas"] == 500


def test_profiles_validate_the_campaign_like_the_prediction():
    with pytest.raises(ValueError, match="platform: valor desconhecido"):
        campaign_profiles({**CAMPAIGN, "platform": "YouTube"})
    profile = campaign_profiles(CAMPAIGN).iloc[0]
    assert profile["duration_band"] == "mais de 30 s"
    assert profile["music_band"] == "predomina a voz"
    assert profile["subtitle_status"] == "desconhecido"
    assert isinstance(campaign_profiles([CAMPAIGN, CAMPAIGN]), pd.DataFrame)
