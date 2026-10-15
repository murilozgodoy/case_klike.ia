"""Testes da limpeza de dados.

Dois níveis:
- testes sintéticos, que injetam um erro conhecido numa campanha "saudável" e
  verificam se a regra certa o corrige;
- testes de regressão no dataset real, que travam o comportamento esperado
  (quais registros são corrigidos e que as identidades do funil passam a fechar).
"""

import numpy as np
import pandas as pd
import pytest

from klike.config import RAW_DATA_PATH
from klike.data import IDENTITY, PLAUSIBILITY, TOL_IMPRESSIONS_REL, clean, load_raw


def _campaign(**overrides) -> dict:
    """Campanha sintética em que todas as identidades do funil fecham."""
    base = {
        "campaign_id": "T-001",
        "date": pd.Timestamp("2024-01-01"),
        "platform": "Meta",
        "category": "E-commerce",
        "objective": "traffic",
        "target_audience_age": "25-34",
        "is_retargeting": False,
        "video_duration_s": 15.0,
        "format": "vertical",
        "has_subtitle": True,
        "has_cta": True,
        "has_hook": True,
        "has_face": True,
        "text_density": "medium",
        "music_voice_ratio": 0.4,
        "impressions": 10_000,
        "clicks": 500,
        "ctr": 0.05,
        "cpc": 0.80,
        "spend": 400.0,
        "conversions": 20,
        "revenue": 1200.0,
        "roas": 3.0,
        "avg_watch_time_s": 8.0,
        "engagement_rate": 0.02,
        "klike_score": 60.0,
    }
    base.update(overrides)
    return base


def _healthy_meta_campaigns(n: int = 5) -> list[dict]:
    """Campanhas saudáveis da Meta, que servem de referência de faixa observada."""
    return [_campaign(campaign_id=f"REF-{i}") for i in range(n)]


def _varied_meta_campaigns() -> list[dict]:
    """Campanhas saudáveis com CPC e CTR variados, para criar faixas largas."""
    return [
        _campaign(campaign_id=f"VAR-{i}", cpc=cpc, spend=500 * cpc, ctr=ctr, impressions=500 / ctr)
        for i, (cpc, ctr) in enumerate([(0.5, 0.02), (0.8, 0.05), (1.2, 0.08), (2.0, 0.10)])
    ]


def _single_fix(df: pd.DataFrame, campaign_id: str) -> pd.Series:
    result = clean(df)
    return result.data.set_index("campaign_id").loc[campaign_id], result


# --------------------------------------------------------------------------- #
# Regras, com dados sintéticos
# --------------------------------------------------------------------------- #
def test_inflated_impressions_are_rebuilt_from_clicks_and_ctr():
    bad = _campaign(campaign_id="BAD", impressions=5_000_000)
    df = pd.DataFrame([*_healthy_meta_campaigns(), bad])
    row, result = _single_fix(df, "BAD")
    assert row["impressions"] == 10_000
    assert result.log["rule"].tolist() == ["Impressões infladas"]
    # Nenhuma conta separa impressões de CTR: a escolha é por plausibilidade.
    assert result.log["basis"].tolist() == [PLAUSIBILITY]


def test_ambiguous_impressions_go_to_review():
    # Impressões em dobro: o CTR alternativo, 2,5%, também seria plausível.
    bad = _campaign(campaign_id="BAD", impressions=20_000)
    df = pd.DataFrame([*_varied_meta_campaigns(), bad])
    row, result = _single_fix(df, "BAD")
    assert row["impressions"] == 20_000
    assert result.log.empty
    assert result.unresolved["campaign_id"].tolist() == ["BAD"]


def test_inconsistent_ctr_goes_to_review_when_clicks_cannot_be_confirmed():
    # Sem CPC não há segunda identidade para confirmar os cliques: não corrigimos.
    df = pd.DataFrame([_campaign(campaign_id="BAD", impressions=5_000_000, cpc=np.nan)])
    row, result = _single_fix(df, "BAD")
    assert row["impressions"] == 5_000_000
    assert result.unresolved["campaign_id"].tolist() == ["BAD"]


def test_wrong_cpc_is_identified_through_roas():
    df = pd.DataFrame([_campaign(campaign_id="BAD", cpc=25.0)])
    row, result = _single_fix(df, "BAD")
    assert row["cpc"] == pytest.approx(0.80)
    assert row["spend"] == pytest.approx(400.0)
    assert result.log["rule"].tolist() == ["CPC fora de escala"]
    assert result.log["basis"].tolist() == [IDENTITY]


def test_wrong_spend_is_identified_through_roas():
    df = pd.DataFrame([_campaign(campaign_id="BAD", spend=90_000.0)])
    row, result = _single_fix(df, "BAD")
    assert row["spend"] == pytest.approx(400.0)
    assert row["cpc"] == pytest.approx(0.80)
    assert result.log["rule"].tolist() == ["Gasto fora de escala"]
    assert result.log["basis"].tolist() == [IDENTITY]


def test_zero_revenue_falls_back_to_platform_cpc_range():
    # Branding não tem receita, então o ROAS não desempata entre gasto e CPC.
    bad = _campaign(campaign_id="BAD", category="Branding", revenue=0.0, roas=0.0, spend=50_000.0)
    df = pd.DataFrame([*_healthy_meta_campaigns(), bad])
    row, result = _single_fix(df, "BAD")
    assert row["spend"] == pytest.approx(400.0)
    assert row["cpc"] == pytest.approx(0.80)
    assert result.log["basis"].tolist() == [PLAUSIBILITY]


def test_ambiguous_spend_cpc_conflict_goes_to_review():
    # Sem receita, e o CPC implícito no gasto, R$ 1,60, também é plausível na Meta.
    bad = _campaign(campaign_id="BAD", category="Branding", revenue=0.0, roas=0.0, spend=800.0)
    df = pd.DataFrame([*_varied_meta_campaigns(), bad])
    row, result = _single_fix(df, "BAD")
    assert row["spend"] == pytest.approx(800.0)
    assert row["cpc"] == pytest.approx(0.80)
    assert result.log.empty
    assert result.unresolved["campaign_id"].tolist() == ["BAD"]


def test_missing_cpc_and_revenue_are_recovered_from_identities():
    df = pd.DataFrame([_campaign(campaign_id="GAP", cpc=np.nan, revenue=np.nan)])
    row, result = _single_fix(df, "GAP")
    assert row["cpc"] == pytest.approx(0.80)
    assert row["revenue"] == pytest.approx(1200.0)
    assert result.log["basis"].tolist() == [IDENTITY, IDENTITY]


def test_healthy_campaign_is_left_untouched():
    df = pd.DataFrame(_healthy_meta_campaigns())
    result = clean(df)
    assert result.log.empty
    assert result.unresolved.empty


def test_flags_are_missing_when_source_value_is_missing():
    # Sem o valor de origem, não dá para afirmar que a campanha está abaixo do teto.
    df = pd.DataFrame(
        [
            _campaign(campaign_id="CAP", engagement_rate=0.10, avg_watch_time_s=15.0),
            _campaign(campaign_id="GAP", engagement_rate=np.nan, avg_watch_time_s=np.nan),
        ]
    )
    flags = clean(df).data.set_index("campaign_id")
    assert flags.loc["CAP", "flag_engagement_at_cap"]
    assert flags.loc["CAP", "flag_full_watch"]
    assert pd.isna(flags.loc["GAP", "flag_engagement_at_cap"])
    assert pd.isna(flags.loc["GAP", "flag_full_watch"])


def test_clean_does_not_mutate_its_input():
    df = pd.DataFrame([_campaign(campaign_id="BAD", impressions=5_000_000)])
    snapshot = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, snapshot)


# --------------------------------------------------------------------------- #
# Regressão no dataset real
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cleaned():
    return clean(load_raw(RAW_DATA_PATH))


def test_known_errors_in_dataset_are_fixed(cleaned):
    fixed = cleaned.log.groupby("rule")["campaign_id"].apply(set).to_dict()
    assert fixed["Impressões infladas"] == {"KLK-0148", "KLK-0232", "KLK-0258", "KLK-0413"}
    assert fixed["CPC fora de escala"] == {"KLK-0100", "KLK-0240", "KLK-0416"}
    assert fixed["Gasto fora de escala"] == {"KLK-0109", "KLK-0224", "KLK-0259"}
    assert len(fixed["CPC faltante recuperado"]) == 28
    assert len(fixed["Receita faltante recuperada"]) == 25
    assert cleaned.unresolved.empty


def test_only_ambiguous_field_choices_rest_on_plausibility(cleaned):
    # As 4 impressões e o KLK-0224 (Branding, sem receita) não têm segunda conta
    # que diga qual campo está errado. Todas as outras correções têm.
    by_basis = cleaned.log.groupby("basis")["campaign_id"].apply(set).to_dict()
    assert by_basis[PLAUSIBILITY] == {"KLK-0148", "KLK-0232", "KLK-0258", "KLK-0413", "KLK-0224"}
    assert len(cleaned.log) == 63
    assert (cleaned.log["basis"] == IDENTITY).sum() == 58


def test_funnel_identities_hold_after_cleaning(cleaned):
    d = cleaned.data
    impressions_gap = (d["impressions"] / (d["clicks"] / d["ctr"]) - 1).abs()
    spend_gap = (d["clicks"] * d["cpc"] - d["spend"]).abs() / d["spend"]
    roas_gap = (d["revenue"] / d["spend"] - d["roas"]).abs()
    assert impressions_gap.max() <= TOL_IMPRESSIONS_REL
    assert spend_gap.max() < 0.001
    assert roas_gap.max() <= 0.0051


def test_only_unrecoverable_columns_remain_missing(cleaned):
    # As marcações ficam ausentes de propósito quando falta o valor de origem.
    original = cleaned.data[[c for c in cleaned.data.columns if not c.startswith("flag_")]]
    still_missing = set(original.columns[original.isna().any()])
    assert still_missing == {
        "video_duration_s",
        "has_subtitle",
        "music_voice_ratio",
        "avg_watch_time_s",
        "engagement_rate",
    }


def test_cleaning_is_idempotent(cleaned):
    flags = [c for c in cleaned.data.columns if c.startswith("flag_")]
    second_pass = clean(cleaned.data.drop(columns=flags))
    assert second_pass.log.empty


def test_load_raw_rejects_unexpected_categories(tmp_path):
    raw = pd.read_csv(RAW_DATA_PATH)
    raw.loc[0, "platform"] = "Snapchat"
    path = tmp_path / "broken.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="platform"):
        load_raw(path)
