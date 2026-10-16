"""Testes do modelo final: previsão para campanhas novas, persistência e reprodutibilidade.

Usam a interface pública, ``KlikeScorePredictor``, como faria o recommendations engine:
uma campanha com os 13 campos de pré-publicação entra, e o score previsto sai, sem chamar
``prepare_features`` nem conhecer o pré-processamento.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from klike.config import PERFORMANCE_COLS, RAW_DATA_PATH, ROOT_DIR, TARGET
from klike.data import load_clean
from klike.modeling import (
    PRE_FLIGHT_INPUTS,
    SELECTED_MODEL,
    SELECTED_PARAMS,
    SELECTED_SPEC,
    bootstrap_metrics,
    build_pipeline,
    prepare_features,
    split_development_test,
    tuned,
)
from klike.predictor import (
    CERTIFIED_METRICS,
    KlikeScorePredictor,
    data_sha256,
    train_final_model,
)
from klike.train import main

CAMPAIGN = {
    "platform": "TikTok",
    "category": "E-commerce",
    "objective": "conversions",
    "target_audience_age": "18-24",
    "is_retargeting": False,
    "video_duration_s": 15,
    "format": "vertical",
    "has_subtitle": True,
    "has_cta": True,
    "has_hook": True,
    "has_face": True,
    "text_density": "medium",
    "music_voice_ratio": 0.4,
}


@pytest.fixture(scope="module")
def predictor():
    return train_final_model()


@pytest.fixture(scope="module")
def clean_data():
    return load_clean()


def score(predictor, campaign) -> float:
    return float(predictor.predict(campaign).iloc[0])


# --------------------------------------------------------------------------- #
# Campanhas novas: só pré-publicação
# --------------------------------------------------------------------------- #
def test_new_campaign_needs_only_the_pre_flight_fields(predictor):
    assert set(CAMPAIGN) == set(PRE_FLIGHT_INPUTS)
    result = predictor.predict(CAMPAIGN)
    assert isinstance(result, pd.Series)
    assert result.name == "klike_score_previsto"
    assert len(result) == 1
    assert 0 <= result.iloc[0] <= 100


def test_performance_metrics_are_neither_needed_nor_used(predictor):
    with_history = {
        **CAMPAIGN,
        **{col: 1.0 for col in PERFORMANCE_COLS},
        "ctr": 0.9,
        "cpc": 99.0,
        "roas": 50.0,
        "revenue": 1e6,
        TARGET: 3.0,
        "campaign_id": "NOVA-001",
        "date": "2026-10-15",
        "flag_ctr_at_cap": True,
    }
    assert score(predictor, with_history) == score(predictor, CAMPAIGN)


def test_optional_fields_may_be_empty_or_omitted(predictor, clean_data):
    empty = {**CAMPAIGN, "video_duration_s": None, "music_voice_ratio": np.nan}
    omitted = {
        k: v for k, v in CAMPAIGN.items() if k not in ("video_duration_s", "music_voice_ratio")
    }
    assert score(predictor, empty) == pytest.approx(score(predictor, omitted))
    # Sem duração e música/voz, o pipeline imputa as medianas das 500 campanhas de treino.
    medians = {
        **CAMPAIGN,
        "video_duration_s": clean_data["video_duration_s"].median(),
        "music_voice_ratio": clean_data["music_voice_ratio"].median(),
    }
    assert score(predictor, empty) == pytest.approx(score(predictor, medians))


def test_unmarked_subtitle_is_its_own_case(predictor):
    unknown = score(predictor, {**CAMPAIGN, "has_subtitle": None})
    omitted = score(predictor, {k: v for k, v in CAMPAIGN.items() if k != "has_subtitle"})
    assert unknown == omitted
    assert unknown not in (
        score(predictor, {**CAMPAIGN, "has_subtitle": True}),
        score(predictor, {**CAMPAIGN, "has_subtitle": False}),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"platform": "YouTube"}, "platform: valor desconhecido ['YouTube']"),
        ({"format": "story"}, "format: valor desconhecido ['story']"),
        ({"has_hook": "sim"}, "has_hook: esperado verdadeiro/falso"),
        ({"video_duration_s": -5}, "video_duration_s: esperado um número maior que zero"),
        ({"music_voice_ratio": 1.5}, "music_voice_ratio: esperado um número entre 0 e 1"),
        ({"category": None}, "category: valor ausente"),
        ({"music_ratio": 0.3}, "campos desconhecidos ['music_ratio']"),
    ],
)
def test_invalid_inputs_raise_clear_messages(predictor, change, message):
    with pytest.raises(ValueError, match=r"^Entrada inválida: ") as error:
        predictor.predict({**CAMPAIGN, **change})
    assert message in str(error.value)


def test_absent_required_fields_are_listed(predictor):
    campaign = {k: v for k, v in CAMPAIGN.items() if k not in ("platform", "has_face")}
    expected = r"faltam as colunas obrigatórias \['platform', 'has_face'\]"
    with pytest.raises(ValueError, match=expected):
        predictor.predict(campaign)


def test_field_order_and_input_format_do_not_change_the_prediction(predictor):
    expected = score(predictor, CAMPAIGN)
    reordered = dict(reversed(list(CAMPAIGN.items())))
    assert score(predictor, reordered) == expected
    frame = pd.DataFrame([CAMPAIGN])
    assert score(predictor, frame[frame.columns[::-1]]) == expected
    batch = predictor.predict([{**CAMPAIGN, "has_hook": False}, reordered])
    assert batch.iloc[1] == expected


def test_readme_example_gives_the_documented_score(predictor):
    # O README mostra esta mesma campanha; se o modelo mudar, a documentação precisa mudar.
    assert round(score(predictor, CAMPAIGN), 1) == 88.3


def test_dataframe_index_is_kept(predictor):
    frame = pd.DataFrame([CAMPAIGN, CAMPAIGN], index=["a", "b"])
    assert list(predictor.predict(frame).index) == ["a", "b"]


# --------------------------------------------------------------------------- #
# A interface aplica todas as etapas
# --------------------------------------------------------------------------- #
def test_predict_applies_validation_preparation_and_the_whole_pipeline(predictor):
    expected = np.clip(predictor.pipeline.predict(prepare_features(CAMPAIGN)), 0, 100)
    np.testing.assert_allclose(predictor.predict(CAMPAIGN).to_numpy(), expected)
    features = predictor.pipeline.named_steps["features"]
    assert set(features.named_transformers_) == {
        "categorias",
        "binarias",
        "duracao",
        "numericas",
        "plataforma_formato",
    }
    assert features.named_transformers_["plataforma_formato"].combinations_ == [
        ("TikTok", "quadrado"),
        ("LinkedIn", "quadrado"),
        ("LinkedIn", "horizontal"),
    ]
    assert predictor.pipeline.named_steps["model"].alpha == SELECTED_PARAMS["alpha"]


def test_format_effect_depends_on_the_platform(predictor):
    def horizontal_minus_vertical(platform):
        base = {**CAMPAIGN, "platform": platform}
        return score(predictor, {**base, "format": "horizontal"}) - score(
            predictor, {**base, "format": "vertical"}
        )

    # A interação F3 está ativa: o horizontal perde pontos na Meta e ganha no LinkedIn.
    assert horizontal_minus_vertical("Meta") < 0 < horizontal_minus_vertical("LinkedIn")


def test_predictions_are_limited_to_the_score_scale(predictor):
    class Extrapolating:
        def predict(self, features):
            return np.array([-12.0, 50.0, 130.0])

    limited = KlikeScorePredictor(Extrapolating(), predictor.support, predictor.metadata)
    result = limited.predict([CAMPAIGN] * 3)
    assert result.tolist() == [0.0, 50.0, 100.0]


def test_every_valid_combination_stays_within_the_scale(predictor):
    rng = np.random.default_rng(0)
    campaigns = [
        {
            **CAMPAIGN,
            "platform": rng.choice(["Meta", "TikTok", "LinkedIn"]),
            "format": rng.choice(["vertical", "quadrado", "horizontal"]),
            "has_hook": bool(rng.integers(2)),
            "has_face": bool(rng.integers(2)),
            "has_cta": bool(rng.integers(2)),
            "text_density": rng.choice(["low", "medium", "high"]),
            "video_duration_s": float(rng.uniform(1, 180)),
            "music_voice_ratio": float(rng.uniform(0, 1)),
        }
        for _ in range(300)
    ]
    scores = predictor.predict(campaigns)
    assert scores.between(0, 100).all()


# --------------------------------------------------------------------------- #
# Treino final, persistência e metadados
# --------------------------------------------------------------------------- #
def test_final_model_refits_the_frozen_specification_on_all_campaigns(predictor, clean_data):
    direct = build_pipeline(SELECTED_MODEL, SELECTED_SPEC, **SELECTED_PARAMS).fit(
        prepare_features(clean_data), clean_data[TARGET]
    )
    assert predictor.metadata["treino"]["campanhas"] == len(clean_data) == 500
    np.testing.assert_allclose(
        predictor.pipeline.named_steps["model"].coef_, direct.named_steps["model"].coef_
    )


def test_save_and_load_preserve_predictions_metadata_and_support(predictor, tmp_path):
    path = predictor.save(tmp_path / "modelo.joblib")
    loaded = KlikeScorePredictor.load(path)
    campaigns = [CAMPAIGN, {**CAMPAIGN, "platform": "LinkedIn", "format": "horizontal"}]
    pd.testing.assert_series_equal(loaded.predict(campaigns), predictor.predict(campaigns))
    assert loaded.metadata == predictor.metadata
    pd.testing.assert_frame_equal(loaded.support.profiles, predictor.support.profiles)


def test_loading_explains_how_to_recreate_a_missing_or_unknown_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m klike.train"):
        KlikeScorePredictor.load(tmp_path / "nao_existe.joblib")
    joblib.dump({"formato": 999}, tmp_path / "antigo.joblib")
    with pytest.raises(ValueError, match="formato desconhecido"):
        KlikeScorePredictor.load(tmp_path / "antigo.joblib")


def test_loading_warns_when_scikit_learn_changed(predictor, tmp_path):
    path = predictor.save(tmp_path / "modelo.joblib")
    content = joblib.load(path)
    content["metadados"] = {**content["metadados"], "versoes": {"scikit-learn": "0.0.1"}}
    joblib.dump(content, path)
    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        KlikeScorePredictor.load(path)


def test_metadata_documents_specification_data_versions_and_official_metrics(predictor):
    metadata = predictor.metadata
    as_text = json.dumps(metadata, ensure_ascii=False)  # só tipos simples
    assert str(ROOT_DIR) not in as_text  # nenhum caminho local
    assert metadata["especificacao"] == {
        "algoritmo": "ridge",
        "missing_indicators": False,
        "log_duration": False,
        "platform_format": True,
        "alpha": SELECTED_PARAMS["alpha"],
    }
    assert metadata["treino"]["arquivo"] == RAW_DATA_PATH.name
    assert metadata["treino"]["sha256"] == data_sha256(RAW_DATA_PATH)
    assert metadata["versoes"]["scikit-learn"] == sklearn.__version__
    official = metadata["metricas_oficiais"]
    assert {k: official[k] for k in CERTIFIED_METRICS} == CERTIFIED_METRICS
    assert "não tem métrica própria" in official["observacao"]


def test_data_fingerprint_does_not_depend_on_line_endings(tmp_path):
    (tmp_path / "lf.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "crlf.csv").write_bytes(b"a,b\r\n1,2\r\n")
    assert data_sha256(tmp_path / "lf.csv") == data_sha256(tmp_path / "crlf.csv")


def test_certified_metrics_match_a_reproduction_of_the_certification(clean_data):
    # Refaz a certificação da Parte 2 só para conferir os números guardados nos metadados.
    development, test = split_development_test(clean_data)
    certified = tuned(SELECTED_MODEL, SELECTED_SPEC).fit(
        prepare_features(development), development[TARGET]
    )
    assert certified.best_params_ == {"model__alpha": SELECTED_PARAMS["alpha"]}
    metrics = bootstrap_metrics(test[TARGET], certified.predict(prepare_features(test))).round(2)
    assert metrics.loc["mae", "estimativa"] == CERTIFIED_METRICS["mae"]
    assert (
        metrics.loc["mae", ["ic_inferior", "ic_superior"]].tolist() == CERTIFIED_METRICS["mae_ic95"]
    )
    for metric in ("rmse", "r2", "spearman"):
        assert metrics.loc[metric, "estimativa"] == CERTIFIED_METRICS[metric]
    dummy = build_pipeline("dummy").fit(prepare_features(development), development[TARGET])
    dummy_mae = np.abs(test[TARGET] - dummy.predict(prepare_features(test))).mean()
    assert round(dummy_mae, 2) == CERTIFIED_METRICS["mae_dummy"]


def test_training_command_recreates_a_usable_file(tmp_path, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = main(["--saida", str(tmp_path / "modelo.joblib")])
    assert path.exists()
    assert "Métricas oficiais, da certificação: MAE 6,43" in capsys.readouterr().out
    assert 0 <= score(KlikeScorePredictor.load(path), CAMPAIGN) <= 100
