"""Testes da fundação da modelagem: features declaradas, validação, divisão e pipelines.

Garantem que o modelo A só usa informação de pré-publicação, que uma campanha nova é
preparada exatamente como o histórico e que nada aprendido com os dados vaza entre
partições. Testes de métrica usam dados sintéticos: o desempenho real só é medido nos
experimentos formais.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import FunctionTransformer

from klike.config import (
    CATEGORY_ORDERS,
    CONTEXT_COLS,
    CREATIVE_COLS,
    DATE_COL,
    ID_COL,
    PERFORMANCE_COLS,
    TARGET,
)
from klike.data import load_clean, load_raw
from klike.modeling import (
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    PARAM_GRIDS,
    POST_FLIGHT_RATES,
    PRE_FLIGHT_INPUTS,
    SELECTED_MODEL,
    SELECTED_SPEC,
    FeatureSpec,
    PlatformFormatInteractions,
    build_pipeline,
    compare_on_splits,
    evaluate_on_splits,
    outer_splits,
    prepare_features,
    prepare_post_flight_features,
    split_development_test,
    tuned,
)

NEW_CAMPAIGN = {
    "platform": "TikTok",
    "category": "E-commerce",
    "objective": "conversions",
    "target_audience_age": "18-24",
    "is_retargeting": False,
    "video_duration_s": 15.0,
    "format": "vertical",
    "has_subtitle": True,
    "has_cta": True,
    "has_hook": True,
    "has_face": False,
    "text_density": "low",
    "music_voice_ratio": 0.4,
}


@pytest.fixture(scope="module")
def clean_data():
    return load_clean()


@pytest.fixture(scope="module")
def development(clean_data):
    return split_development_test(clean_data)[0]


@pytest.fixture(scope="module")
def fitted_ridge(development):
    return build_pipeline("ridge").fit(prepare_features(development), development[TARGET])


def _synthetic(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Campanhas sintéticas em que o score depende só do hook: +12 pontos."""
    rng = np.random.default_rng(seed)
    campaigns = pd.DataFrame({col: rng.choice(CATEGORY_ORDERS[col], n) for col in CATEGORY_ORDERS})
    for col in ["is_retargeting", "has_cta", "has_hook", "has_face", "has_subtitle"]:
        campaigns[col] = rng.random(n) < 0.5
    campaigns["video_duration_s"] = rng.uniform(5, 60, n)
    campaigns["music_voice_ratio"] = rng.uniform(0, 1, n)
    score = 50 + 12 * campaigns["has_hook"] + rng.normal(0, 3, n)
    return prepare_features(campaigns), score


def _feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["features"].get_feature_names_out())


# --------------------------------------------------------------------------- #
# Features declaradas
# --------------------------------------------------------------------------- #
def test_model_a_uses_only_pre_flight_columns():
    assert PRE_FLIGHT_INPUTS == CONTEXT_COLS + CREATIVE_COLS
    assert len(PRE_FLIGHT_INPUTS) == 13
    forbidden = set(PERFORMANCE_COLS) | {TARGET, ID_COL, DATE_COL}
    assert not forbidden & set(PRE_FLIGHT_INPUTS)
    assert not forbidden & set(MODEL_A_FEATURES)


def test_model_b_adds_only_performance_rates():
    assert MODEL_B_FEATURES == MODEL_A_FEATURES + POST_FLIGHT_RATES
    assert TARGET not in MODEL_B_FEATURES


def test_post_flight_features_mark_roas_as_not_applicable_in_branding(clean_data):
    features = prepare_post_flight_features(clean_data)
    assert list(features.columns) == MODEL_B_FEATURES
    assert (features.loc[clean_data["category"] == "Branding", "roas"] == 0).all()
    assert features["roas"].notna().all()


# --------------------------------------------------------------------------- #
# Preparação: a mesma no histórico e numa campanha nova
# --------------------------------------------------------------------------- #
def test_new_campaign_with_only_pre_flight_fields_gets_a_score(fitted_ridge):
    features = prepare_features(NEW_CAMPAIGN)
    assert list(features.columns) == MODEL_A_FEATURES
    score = fitted_ridge.predict(features)
    assert score.shape == (1,)
    assert np.isfinite(score[0])


def test_post_flight_columns_do_not_change_the_prediction(clean_data, fitted_ridge):
    rows = clean_data.head(20)
    scrambled = rows.assign(
        **{col: rows[col].sample(frac=1, random_state=0).to_numpy() for col in PERFORMANCE_COLS}
    )
    without = rows.drop(columns=[*PERFORMANCE_COLS, TARGET])
    expected = fitted_ridge.predict(prepare_features(rows))
    for variant in (scrambled, without):
        np.testing.assert_allclose(fitted_ridge.predict(prepare_features(variant)), expected)


def test_raw_csv_and_cleaned_data_prepare_identically(clean_data):
    # A limpeza da Parte 1 só corrige métricas de performance: nada que o modelo A use.
    pd.testing.assert_frame_equal(prepare_features(load_raw()), prepare_features(clean_data))


def test_optional_fields_may_be_missing_or_absent(fitted_ridge):
    campaign = {
        **NEW_CAMPAIGN,
        "video_duration_s": None,
        "music_voice_ratio": np.nan,
        "has_subtitle": None,
    }
    features = prepare_features(campaign)
    assert features.loc[0, "subtitle_status"] == "desconhecido"
    assert features[["video_duration_s", "music_voice_ratio"]].isna().all(axis=None)
    absent = {k: v for k, v in campaign.items() if v is not None and not pd.isna(v)}
    pd.testing.assert_frame_equal(prepare_features(absent), features)
    # A imputação acontece dentro do pipeline, com o que foi aprendido no treino.
    assert np.isfinite(fitted_ridge.predict(features)[0])


@pytest.mark.parametrize("field", ["platform", "category", "has_hook", "is_retargeting"])
def test_required_fields_cannot_be_missing(field):
    with pytest.raises(ValueError, match=field):
        prepare_features({**NEW_CAMPAIGN, field: None})
    with pytest.raises(ValueError, match=field):
        prepare_features({k: v for k, v in NEW_CAMPAIGN.items() if k != field})


@pytest.mark.parametrize(
    ("field", "value"),
    [("platform", "Snapchat"), ("platform", "Tiktok"), ("format", "story"), ("text_density", "x")],
)
def test_unknown_categories_are_rejected(field, value):
    # Ignorar a categoria faria um "Tiktok" digitado errado virar a plataforma de referência.
    with pytest.raises(ValueError, match="valor desconhecido"):
        prepare_features({**NEW_CAMPAIGN, field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_duration_s", 0),
        ("video_duration_s", -5),
        ("video_duration_s", "15s"),
        ("music_voice_ratio", 1.5),
        ("has_hook", "sim"),
        ("has_subtitle", "talvez"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        prepare_features({**NEW_CAMPAIGN, field: value})


def test_errors_list_every_problem_at_once():
    with pytest.raises(ValueError) as error:
        prepare_features({**NEW_CAMPAIGN, "platform": "Snapchat", "music_voice_ratio": 2})
    assert "platform" in str(error.value)
    assert "music_voice_ratio" in str(error.value)


def test_errors_include_absent_fields_and_invalid_values_together():
    campaign = {k: v for k, v in NEW_CAMPAIGN.items() if k != "platform"}
    with pytest.raises(ValueError) as error:
        prepare_features({**campaign, "format": "story", "music_voice_ratio": 1.5})
    message = str(error.value)
    assert "faltam as colunas obrigatórias ['platform']" in message
    assert "format: valor desconhecido" in message
    assert "music_voice_ratio" in message


def test_column_order_and_input_format_do_not_matter():
    expected = prepare_features(NEW_CAMPAIGN)
    reordered = dict(reversed(list(NEW_CAMPAIGN.items())))
    pd.testing.assert_frame_equal(prepare_features(reordered), expected)
    pd.testing.assert_frame_equal(prepare_features([NEW_CAMPAIGN]), expected)
    pd.testing.assert_frame_equal(prepare_features(pd.DataFrame([NEW_CAMPAIGN])), expected)


def test_booleans_accept_zero_and_one():
    as_numbers = {**NEW_CAMPAIGN, "has_hook": 1, "has_face": 0, "is_retargeting": 0}
    pd.testing.assert_frame_equal(prepare_features(as_numbers), prepare_features(NEW_CAMPAIGN))


# --------------------------------------------------------------------------- #
# Divisão e partições
# --------------------------------------------------------------------------- #
def test_split_is_deterministic_disjoint_and_stratified(clean_data):
    development, test = split_development_test(clean_data)
    _, test_again = split_development_test(clean_data.sample(frac=1, random_state=1))
    assert (len(development), len(test)) == (400, 100)
    assert set(development[ID_COL]).isdisjoint(test[ID_COL])
    assert list(test[ID_COL]) == list(test_again[ID_COL])  # não depende da ordem das linhas
    share = clean_data["platform"].value_counts(normalize=True)
    test_share = test["platform"].value_counts(normalize=True)
    assert (test_share - share).abs().max() < 0.01


def test_outer_splits_are_fixed_and_cover_the_development_set(development):
    splits = outer_splits(development)
    assert len(splits) == 25
    for (train, valid), (_, valid_again) in zip(splits, outer_splits(development), strict=True):
        np.testing.assert_array_equal(valid, valid_again)
        assert set(train).isdisjoint(valid)
    for repeat in range(5):
        folds = splits[repeat * 5 : (repeat + 1) * 5]
        covered = np.sort(np.concatenate([valid for _, valid in folds]))
        np.testing.assert_array_equal(covered, np.arange(len(development)))


# --------------------------------------------------------------------------- #
# Pipelines sem vazamento
# --------------------------------------------------------------------------- #
def test_imputation_is_learned_only_from_training_rows(development):
    features = prepare_features(development)
    short = features["video_duration_s"] < 15
    pipeline = build_pipeline("ridge").fit(features[short], development[TARGET][short])
    branch = pipeline.named_steps["features"].named_transformers_["duracao"]
    learned = branch.named_steps["simpleimputer"].statistics_[0]
    assert learned == pytest.approx(features.loc[short, "video_duration_s"].median())
    assert learned < features["video_duration_s"].median()


def test_encodings_do_not_depend_on_which_rows_were_in_training(development):
    features = prepare_features(development)
    target = development[TARGET]
    no_linkedin = features["platform"] != "LinkedIn"
    full = build_pipeline("ridge").fit(features, target)
    partial = build_pipeline("ridge").fit(features[no_linkedin], target[no_linkedin])
    assert _feature_names(full) == _feature_names(partial)
    trees = build_pipeline("hgb").fit(features[no_linkedin], target[no_linkedin])
    encoded = trees.named_steps["features"].transform(features[~no_linkedin])
    assert (encoded["platform"] == 2).all()  # LinkedIn é o terceiro da lista fixa
    assert np.isfinite(trees.predict(features[~no_linkedin])).all()


def test_feature_options_add_the_declared_columns():
    features, score = _synthetic()
    base = _feature_names(build_pipeline("ridge").fit(features, score))
    extended = _feature_names(
        build_pipeline("ridge", FeatureSpec(True, False, True)).fit(features, score)
    )
    added = [name for name in extended if name not in base]
    assert added == [
        "missingindicator_video_duration_s",
        "missingindicator_music_voice_ratio",
        "TikTok × quadrado",
        "TikTok × horizontal",
        "LinkedIn × quadrado",
        "LinkedIn × horizontal",
    ]
    logged = build_pipeline("ridge", FeatureSpec(log_duration=True)).fit(features, score)
    first_step = logged.named_steps["features"].named_transformers_["duracao"].steps[0][1]
    assert isinstance(first_step, FunctionTransformer)
    assert first_step.func is np.log
    trees = build_pipeline("hgb", FeatureSpec(missing_indicators=True, platform_format=True))
    assert "platform_format" in _feature_names(trees.fit(features, score))


def test_interactions_exist_only_for_combinations_with_support(development):
    features = prepare_features(development)
    spec = FeatureSpec(platform_format=True)
    pipeline = build_pipeline("ridge", spec).fit(features, development[TARGET])
    interactions = [name for name in _feature_names(pipeline) if "×" in name]
    # TikTok horizontal não aparece na base: não vira coluna, então não ganha coeficiente.
    assert interactions == ["TikTok × quadrado", "LinkedIn × quadrado", "LinkedIn × horizontal"]


def test_interaction_support_is_counted_only_in_the_training_rows():
    train = pd.DataFrame(
        {
            "platform": ["TikTok"] * 3 + ["LinkedIn"] * 2,
            "format": ["quadrado"] * 3 + ["horizontal"] * 2,
        }
    )
    interactions = PlatformFormatInteractions(min_support=3).fit(train)
    assert list(interactions.get_feature_names_out()) == ["TikTok × quadrado"]
    new = pd.DataFrame({"platform": ["LinkedIn", "TikTok"], "format": ["horizontal", "quadrado"]})
    # A combinação sem suporte no treino não recebe efeito próprio na previsão.
    np.testing.assert_array_equal(interactions.transform(new), [[0], [1]])


def test_trees_handle_a_combination_never_seen_in_training(development):
    features = prepare_features(development)
    spec = FeatureSpec(platform_format=True)
    trees = build_pipeline("hgb", spec).fit(features, development[TARGET])
    unseen = prepare_features({**NEW_CAMPAIGN, "platform": "TikTok", "format": "horizontal"})
    assert np.isfinite(trees.predict(unseen)).all()


def test_options_without_meaning_are_rejected():
    with pytest.raises(ValueError, match="árvores"):
        build_pipeline("hgb", FeatureSpec(log_duration=True))
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        build_pipeline("svm")
    with pytest.raises(ValueError, match="Cenário desconhecido"):
        build_pipeline("ridge", scenario="durante")


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def test_dummy_baseline_predicts_the_training_mean():
    features, score = _synthetic()
    pipeline = build_pipeline("dummy").fit(features.iloc[:100], score.iloc[:100])
    assert pipeline.predict(features.iloc[100:105]) == pytest.approx(score.iloc[:100].mean())


def test_evaluate_on_splits_compares_candidates_on_the_same_partitions():
    features, score = _synthetic()
    splits = list(KFold(5, shuffle=True, random_state=0).split(features))
    ridge = evaluate_on_splits(build_pipeline("ridge"), features, score, splits)
    dummy = evaluate_on_splits(build_pipeline("dummy"), features, score, splits)
    assert list(ridge.columns) == ["particao", "mae", "rmse", "r2", "spearman", "hiperparametros"]
    assert len(ridge) == len(dummy) == 5
    assert (ridge["rmse"] >= ridge["mae"]).all()
    assert (ridge["mae"] < dummy["mae"]).all()  # o efeito do hook existe e é aprendido
    assert dummy["spearman"].isna().all()  # previsão constante não ordena nada
    assert dummy["hiperparametros"].isna().all()  # nada foi ajustado


def test_tuned_candidates_record_the_hyperparameters_of_each_partition():
    features, score = _synthetic()
    splits = list(KFold(3, shuffle=True, random_state=0).split(features))
    folds = evaluate_on_splits(tuned("ridge"), features, score, splits)
    grid = PARAM_GRIDS["ridge"]["model__alpha"]
    assert all(params["model__alpha"] in grid for params in folds["hiperparametros"])


def test_compare_on_splits_reports_paired_gains():
    reference = pd.DataFrame({"particao": [0, 1, 2, 3], "mae": [10.0, 10.0, 10.0, 10.0]})
    candidate = pd.DataFrame({"particao": [0, 1, 2, 3], "mae": [9.0, 9.5, 10.5, 9.0]})
    result = compare_on_splits(candidate, reference)
    assert result["melhora"] == pytest.approx(0.5)
    assert result["melhora_relativa"] == pytest.approx(0.05)
    assert result["vitorias"] == pytest.approx(0.75)


def test_compare_on_splits_requires_the_same_partitions_and_an_error_metric():
    reference = pd.DataFrame({"particao": [0, 1], "mae": [1.0, 1.0], "r2": [0.5, 0.5]})
    with pytest.raises(ValueError, match="mesmas partições"):
        compare_on_splits(reference.assign(particao=[1, 2]), reference)
    with pytest.raises(ValueError, match="Métrica de erro"):
        compare_on_splits(reference, reference, metric="r2")


def test_ridge_recovers_a_known_effect_in_points():
    features, score = _synthetic(n=400)
    pipeline = build_pipeline("ridge", alpha=0.01).fit(features, score)
    coefficients = dict(zip(_feature_names(pipeline), pipeline[-1].coef_, strict=True))
    assert coefficients["has_hook"] == pytest.approx(12, abs=1)


def test_selected_specification_builds_a_linear_model_with_supported_interactions():
    features, score = _synthetic()
    search = tuned(SELECTED_MODEL, SELECTED_SPEC).fit(features, score)
    names = _feature_names(search.best_estimator_)
    assert SELECTED_MODEL == "ridge"
    assert FeatureSpec(platform_format=True) == SELECTED_SPEC
    assert "LinkedIn × horizontal" in names
    assert not any(name.startswith("missingindicator") for name in names)


def test_hyperparameters_are_tuned_within_a_small_declared_grid():
    assert len(ParameterGrid(PARAM_GRIDS["ridge"])) == 5
    assert len(ParameterGrid(PARAM_GRIDS["hgb"])) == 8
    features, score = _synthetic()
    search = tuned("ridge").fit(features, score)
    assert search.best_params_["model__alpha"] in PARAM_GRIDS["ridge"]["model__alpha"]
