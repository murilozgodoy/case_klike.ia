"""Testes das evidências do recommendations engine.

Conferem que cada contraste mede exatamente a transição sugerida, que a família de hipóteses
e a correção de Benjamini-Hochberg são as documentadas, que as sensibilidades marcam os casos
conhecidos e que as diferenças de score vêm de reajustar todo o pipeline congelado.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from klike.analysis import fit_model
from klike.data import load_clean
from klike.evidence import (
    PERFORMANCE_METRICS,
    PERFORMANCE_PATH,
    PERFORMANCE_ROUNDING,
    SCORE_PATH,
    SCORE_ROUNDING,
    TRANSITIONS,
    _refit,
    load_performance_evidence,
    load_score_differences,
    performance_evidence,
    save_evidence,
    score_differences,
    transition_effect,
)
from klike.modeling import prepare_features
from klike.predictor import train_final_model

BY_KEY = {transition.key: transition for transition in TRANSITIONS}


@pytest.fixture(scope="module")
def clean_data():
    return load_clean()


@pytest.fixture(scope="module")
def evidence(clean_data):
    return performance_evidence(clean_data)


@pytest.fixture(scope="module")
def score(clean_data):
    return score_differences(clean_data, n_resamples=30)


def row(table: pd.DataFrame, transition: str, metric: str) -> pd.Series:
    return table.set_index(["transicao", "metrica"]).loc[(transition, metric)]


def _synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Campanhas sintéticas com efeitos conhecidos, na escala log, sobre o CTR."""
    rng = np.random.default_rng(seed)
    hook = rng.integers(0, 2, n).astype(bool)
    text = rng.choice(["low", "medium", "high"], n)
    duration = rng.uniform(5, 60, n)
    frame = pd.DataFrame(
        {
            "platform": rng.choice(["Meta", "TikTok", "LinkedIn"], n),
            "category": rng.choice(["E-commerce", "SaaS"], n),
            "objective": rng.choice(["traffic", "conversions"], n),
            "target_audience_age": rng.choice(["18-24", "25-34"], n),
            "is_retargeting": rng.integers(0, 2, n).astype(bool),
            "has_hook": hook,
            "has_face": rng.integers(0, 2, n).astype(bool),
            "has_cta": rng.integers(0, 2, n).astype(bool),
            "has_subtitle": rng.integers(0, 2, n).astype(bool),
            "text_density": text,
            "video_duration_s": duration,
        }
    )
    text_effect = np.select([text == "medium", text == "high"], [0.1, -0.3], 0.0)
    noise = rng.normal(0, 0.15, n)
    frame["ctr"] = np.exp(-3.5 + 0.4 * hook + text_effect + 0.2 * np.log(duration) + noise)
    return frame


# --------------------------------------------------------------------------- #
# Contrastes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("hook", np.expm1(0.4)),
        ("texto_alto_medio", np.expm1(0.1 - (-0.3))),
        ("texto_alto_baixo", np.expm1(0.3)),
        ("texto_baixo_medio", np.expm1(0.1)),
    ],
)
def test_each_transition_measures_its_exact_contrast(key, expected):
    fit = fit_model(_synthetic(), "ctr")
    effect, low, high, _ = transition_effect(fit, BY_KEY[key])
    assert low < expected < high
    assert effect == pytest.approx(expected, abs=0.04)


def test_doubling_the_duration_uses_the_log_scale():
    fit = fit_model(_synthetic(), "ctr", extra="np.log(video_duration_s)")
    effect, *_ = transition_effect(fit, BY_KEY["duracao"])
    assert effect == pytest.approx(2**0.2 - 1, abs=0.03)  # dobrar soma log(2) ao log


def test_effects_can_stay_in_the_outcome_unit():
    frame = _synthetic().assign(score=lambda d: 50 + 12 * d["has_hook"])
    fit = fit_model(frame, "score", log=False)
    effect, *_ = transition_effect(fit, BY_KEY["hook"], log=False)
    assert effect == pytest.approx(12, abs=1e-6)


# --------------------------------------------------------------------------- #
# Família, correção e sensibilidades
# --------------------------------------------------------------------------- #
def test_family_has_the_seventy_documented_hypotheses(evidence):
    assert len(TRANSITIONS) == 14
    assert len(evidence) == 14 * len(PERFORMANCE_METRICS) == 70
    assert not evidence.duplicated(["transicao", "metrica"]).any()
    # Inclui as transições que o engine descarta, como duração e música.
    assert {"duracao", "musica", "formato_meta_quadrado"} <= set(evidence["transicao"])


def test_correction_covers_the_whole_family(evidence):
    q = stats.false_discovery_control(evidence["p_valor"], method="bh")
    np.testing.assert_allclose(evidence["q_valor"], q)
    robust = evidence["evidencia"] == "robusto"
    hint = evidence["evidencia"] == "indício"
    assert (evidence.loc[robust, "q_valor"] < 0.05).all()
    assert ((evidence.loc[hint, "p_valor"] < 0.05) & (evidence.loc[hint, "q_valor"] >= 0.05)).all()
    assert (evidence.loc[~robust & ~hint, "p_valor"] >= 0.05).all()


def test_reproduces_the_audited_contrasts_for_high_to_medium_text(evidence):
    # Valores conferidos na auditoria independente do planejamento.
    assert row(evidence, "texto_alto_medio", "completion_rate")["efeito"] == pytest.approx(
        0.291, abs=5e-4
    )
    assert row(evidence, "texto_alto_medio", "conversions_per_mille")["efeito"] == pytest.approx(
        0.359, abs=5e-4
    )
    assert row(evidence, "texto_alto_medio", "roas_non_branding")["efeito"] == pytest.approx(
        0.208, abs=5e-4
    )


def test_sensitivities_flag_the_known_fragile_results(evidence):
    conversions = "conversions_per_mille"
    subtitle = row(evidence, "legenda", conversions)
    assert subtitle["evidencia"] == "inconclusivo"
    assert subtitle["evidencia_binomial_negativa"] == "robusto"  # o modelo de contagem discorda
    assert not subtitle["estavel"]
    text = row(evidence, "texto_alto_medio", conversions)
    assert text["evidencia"] == "robusto"
    assert text["evidencia_sem_plausibilidade"] == "indício"
    assert not text["estavel"]
    assert row(evidence, "hook", "ctr")["estavel"]
    linkedin = row(evidence, "formato_linkedin_vertical", conversions)
    assert linkedin["evidencia_binomial_negativa"] == "não estimável"
    others = evidence[evidence["metrica"] != conversions]
    assert (others["evidencia_binomial_negativa"] == "não se aplica").all()
    assert others["efeito_binomial_negativa"].isna().all()


def test_group_sizes_count_both_sides_of_each_comparison(evidence):
    hook = row(evidence, "hook", "ctr")
    assert hook["n_de"] + hook["n_para"] == hook["n"] == 500
    subtitle = row(evidence, "legenda", "ctr")
    assert subtitle["n_de"] + subtitle["n_para"] < subtitle["n"]  # legenda ausente não conta
    assert row(evidence, "formato_linkedin_vertical", "ctr")[["n_de", "n_para"]].tolist() == [
        11,
        44,
    ]
    assert pd.isna(row(evidence, "musica", "ctr")["n_de"])  # número contínuo: sem dois grupos


def test_limitations_are_counted_in_the_data(evidence):
    ctr = row(evidence, "hook", "ctr")["limitacoes"]
    assert ctr == "teto de registro: 55 campanhas com CTR de 15%"
    conversions = row(evidence, "hook", "conversions_per_mille")["limitacoes"]
    assert conversions.startswith("19 campanhas sem conversão")


# --------------------------------------------------------------------------- #
# Diferenças de score
# --------------------------------------------------------------------------- #
def test_each_resample_relearns_the_whole_pipeline(clean_data):
    features, target = prepare_features(clean_data), clean_data["klike_score"]
    rows = np.random.default_rng(1).integers(0, len(features), len(features))
    duration = _refit(features, target, rows).named_steps["features"].named_transformers_["duracao"]
    assert duration[0].statistics_[0] == features["video_duration_s"].iloc[rows].median()
    # Sem os vídeos quadrados do TikTok no treino, a interação deles deixa de existir.
    tiktok_square = (clean_data["platform"] == "TikTok") & (clean_data["format"] == "quadrado")
    without = np.flatnonzero(~tiktok_square.to_numpy())
    interactions = (
        _refit(features, target, without)
        .named_steps["features"]
        .named_transformers_["plataforma_formato"]
    )
    assert ("TikTok", "quadrado") not in interactions.combinations_


def test_score_differences_match_the_final_predictor(score, clean_data):
    # O modelo é aditivo: a diferença de uma edição é a mesma em qualquer campanha real,
    # desde que a previsão não encoste nos limites de 0 a 100.
    predictor = train_final_model()
    differences = score.set_index("transicao")["diferenca"]

    def gain(campaign: pd.DataFrame, **edit) -> float:
        return (
            predictor.predict(campaign.assign(**edit)).iloc[0] - predictor.predict(campaign).iloc[0]
        )

    without_hook = clean_data[~clean_data["has_hook"].astype(bool)].iloc[[0]]
    assert gain(without_hook, has_hook=True) == pytest.approx(differences["hook"], abs=1e-6)
    linkedin = clean_data["platform"] == "LinkedIn"
    vertical = clean_data[linkedin & (clean_data["format"] == "vertical")].iloc[[0]]
    assert gain(vertical, format="horizontal") == pytest.approx(
        differences["formato_linkedin_vertical"], abs=1e-6
    )


def test_score_intervals_are_reproducible(clean_data, score):
    again = score_differences(clean_data, n_resamples=30)
    pd.testing.assert_frame_equal(again, score)
    other = score_differences(clean_data, n_resamples=30, random_state=7)
    assert not np.allclose(other["ic_inferior"], score["ic_inferior"])
    assert (score["ic_superior"] > score["ic_inferior"]).all()


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #
def test_saved_tables_round_trip(evidence, score, tmp_path):
    save_evidence(evidence, score, tmp_path / "evidencias.csv", tmp_path / "score.csv")
    loaded = load_performance_evidence(tmp_path / "evidencias.csv")
    pd.testing.assert_frame_equal(loaded, evidence.round(PERFORMANCE_ROUNDING), check_dtype=False)
    pd.testing.assert_frame_equal(
        load_score_differences(tmp_path / "score.csv"),
        score.round(SCORE_ROUNDING),
        check_dtype=False,
    )


def test_exported_tables_match_a_fresh_computation(evidence, clean_data):
    # Os CSVs em reports/ são gerados pelo notebook 04; recalcular tem de dar o mesmo.
    pd.testing.assert_frame_equal(
        load_performance_evidence(PERFORMANCE_PATH),
        evidence.round(PERFORMANCE_ROUNDING),
        check_dtype=False,
    )
    saved = load_score_differences(SCORE_PATH)
    fresh = score_differences(clean_data, n_resamples=int(saved["reamostragens"].iloc[0]))
    pd.testing.assert_frame_equal(saved, fresh.round(SCORE_ROUNDING), check_dtype=False)
