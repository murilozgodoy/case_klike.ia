"""Testes das métricas derivadas e da estimativa de efeitos."""

import numpy as np
import pandas as pd
import pytest

from klike.analysis import (
    add_evidence,
    add_rate_metrics,
    count_best_practices,
    estimate_effects,
    fit_model,
    fit_negative_binomial,
    fit_poisson,
    interaction_pvalue,
    tidy_effects,
)
from klike.data import load_clean

HOOK = "Hook nos 3 primeiros segundos"


def _synthetic_campaigns(n: int = 600, hook_effect: float = 0.4, seed: int = 0) -> pd.DataFrame:
    """Base sintética em que só o hook muda o CTR, com efeito conhecido na escala log."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "platform": rng.choice(["Meta", "TikTok", "LinkedIn"], n),
            "category": rng.choice(["E-commerce", "SaaS"], n),
            "objective": rng.choice(["traffic", "conversions"], n),
            "target_audience_age": rng.choice(["18-24", "25-34"], n),
            "is_retargeting": rng.random(n) < 0.3,
            "has_hook": rng.random(n) < 0.5,
            "has_face": rng.random(n) < 0.5,
            "has_subtitle": pd.array(rng.random(n) < 0.5, dtype="boolean"),
            "has_cta": rng.random(n) < 0.5,
            "text_density": rng.choice(["low", "medium", "high"], n),
        }
    )
    noise = rng.normal(0, 0.2, n)
    df["ctr"] = 0.05 * np.exp(hook_effect * df["has_hook"] + noise)
    return df


def _synthetic_counts(n: int = 800, hook_effect: float = 0.4, seed: int = 3) -> pd.DataFrame:
    """Conversões com muitos zeros e sobredispersão, com efeito conhecido do hook.

    A taxa de cada campanha varia por fatores não observados (gama com média 1 e
    variância 0,5), o que gera uma binomial negativa com alpha = 0,5.
    """
    df = _synthetic_campaigns(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    df["impressions"] = rng.integers(500, 20_000, n)
    expected = df["impressions"] * 0.0004 * np.exp(hook_effect * df["has_hook"])
    df["conversions"] = rng.poisson(expected * rng.gamma(shape=2.0, scale=0.5, size=n))
    df["conversions_per_mille"] = df["conversions"] / df["impressions"] * 1000
    return df


def test_estimate_effects_recovers_a_known_effect():
    effects = estimate_effects(_synthetic_campaigns(), "ctr").set_index("atributo")
    hook = effects.loc["Hook nos 3 primeiros segundos"]
    true_effect = np.exp(0.4) - 1  # +49%
    assert hook["ic_inferior"] < true_effect < hook["ic_superior"]
    assert hook["significativo"]


def test_estimate_effects_does_not_invent_an_effect():
    effects = estimate_effects(_synthetic_campaigns(), "ctr").set_index("atributo")
    face = effects.loc["Rosto humano"]
    assert face["ic_inferior"] < 0 < face["ic_superior"]


def test_estimate_effects_in_original_units():
    df = _synthetic_campaigns()
    df["klike_score"] = 50 + 12 * df["has_hook"] + np.random.default_rng(1).normal(0, 3, len(df))
    effects = estimate_effects(df, "klike_score", log=False).set_index("atributo")
    assert effects.loc["Hook nos 3 primeiros segundos", "efeito"] == pytest.approx(12, abs=1)


def test_interaction_pvalue_detects_an_effect_that_differs_by_platform():
    df = _synthetic_campaigns()
    rng = np.random.default_rng(2)
    # Hook só funciona no TikTok.
    on_tiktok = (df["platform"] == "TikTok") & df["has_hook"]
    df["ctr"] = 0.05 * np.exp(0.6 * on_tiktok + rng.normal(0, 0.2, len(df)))
    assert interaction_pvalue(df, "has_hook", "ctr") < 0.01


def test_interaction_pvalue_accepts_an_effect_equal_everywhere():
    # Na base sintética padrão, o hook tem o mesmo efeito em todas as plataformas.
    assert interaction_pvalue(_synthetic_campaigns(), "has_hook", "ctr") > 0.05


def test_log_model_drops_zeros_while_poisson_keeps_them():
    df = _synthetic_counts()
    zeros = int((df["conversions"] == 0).sum())
    assert zeros > 50  # o cenário precisa ter zeros de verdade
    assert fit_model(df, "conversions_per_mille").nobs == len(df) - zeros
    assert fit_poisson(df, "conversions_per_mille").nobs == len(df)


def test_fit_poisson_recovers_a_known_effect_with_zeros():
    fit = fit_poisson(_synthetic_counts(), "conversions_per_mille")
    hook = tidy_effects(fit, {"has_hook": HOOK}).set_index("atributo").loc[HOOK]
    assert hook["ic_inferior"] < np.exp(0.4) - 1 < hook["ic_superior"]


def test_fit_negative_binomial_recovers_a_known_effect_and_overdispersion():
    fit = fit_negative_binomial(_synthetic_counts(), "conversions", "impressions")
    hook = tidy_effects(fit, {"has_hook": HOOK}).set_index("atributo").loc[HOOK]
    assert hook["ic_inferior"] < np.exp(0.4) - 1 < hook["ic_superior"]
    assert fit.params["alpha"] == pytest.approx(0.5, abs=0.2)


def test_add_evidence_separates_robust_results_from_weak_ones():
    out = add_evidence(pd.DataFrame({"p_valor": [0.001, 0.02, 0.04, 0.30]}))
    # Benjamini-Hochberg: p x número de testes / posição no ranking.
    assert out["q_valor"].tolist() == pytest.approx([0.004, 0.04, 0.04 * 4 / 3, 0.30])
    assert out["evidencia"].tolist() == ["robusto", "robusto", "indício", "inconclusivo"]


def test_conversion_conclusions_do_not_depend_on_dropping_zeros():
    # O log descarta as 19 campanhas sem conversão; os modelos que as mantêm concordam.
    df = add_rate_metrics(load_clean())
    terms = {
        "has_hook": "hook",
        "has_face": "rosto",
        "text_density[T.high]": "texto alto",
        "is_retargeting": "retargeting",
    }
    fits = [
        fit_model(df, "conversions_per_mille"),
        fit_poisson(df, "conversions_per_mille"),
        fit_negative_binomial(df, "conversions", "impressions"),
    ]
    for fit in fits:
        table = tidy_effects(fit, terms)
        assert table["significativo"].all()
        assert np.sign(table["efeito"]).tolist() == [1, 1, -1, 1]


def test_add_rate_metrics():
    df = pd.DataFrame(
        {
            "avg_watch_time_s": [30.0, np.nan, 5.0],
            "video_duration_s": [20.0, 10.0, 10.0],
            "conversions": [10, 0, 5],
            "clicks": [100, 50, 20],
            "impressions": [10_000, 1_000, 2_000],
            "roas": [2.0, 0.0, 3.0],
            "flag_roas_applicable": [True, False, True],
        }
    )
    out = add_rate_metrics(df)
    assert out["completion_rate"].tolist()[0] == 1.0  # limitada a 100%
    assert np.isnan(out["completion_rate"].iloc[1])
    assert out["conversion_rate"].tolist() == [0.1, 0.0, 0.25]
    assert out["conversions_per_mille"].tolist() == [1.0, 0.0, 2.5]
    assert np.isnan(out["roas_non_branding"].iloc[1])  # Branding: ROAS não se aplica


def test_count_best_practices_leaves_unknown_out():
    df = pd.DataFrame(
        {
            "has_hook": [True, True, False],
            "has_face": [True, False, False],
            "has_subtitle": pd.array([True, None, False], dtype="boolean"),
            "has_cta": [True, True, False],
        }
    )
    counts = count_best_practices(df)
    assert counts.iloc[0] == 4
    assert np.isnan(counts.iloc[1])  # legenda desconhecida: não dá para contar
    assert counts.iloc[2] == 0
