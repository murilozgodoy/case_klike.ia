"""Modelagem do Klike Score: features declaradas, divisão dos dados, validação e pipelines.

Dois cenários, separados pelo momento em que a informação existe:

- Modelo A, pré-publicação (principal): só o contexto da campanha e os atributos do
  criativo, conhecidos antes da veiculação. É o que pode pontuar um criativo novo e
  apoiar o recommendations engine.
- Modelo B, pós-publicação (comparativo): acrescenta taxas de performance. Mede quanto
  do score acompanha o que só se sabe depois do gasto. Não serve para avaliar um
  criativo que ainda não foi veiculado.

A preparação das features é determinística e é a mesma no treino e na inferência. Tudo
o que aprende com os dados (imputação, escala, codificação) fica dentro do pipeline e é
ajustado só no treino de cada partição.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from klike.analysis import add_rate_metrics
from klike.config import CATEGORY_ORDERS, CONTEXT_COLS, CREATIVE_COLS, ID_COL

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 400 campanhas de desenvolvimento e 100 de teste final
N_SPLITS = 5
N_REPEATS = 5  # 25 partições externas, as mesmas para todos os candidatos
INNER_FOLDS = 3  # ajuste de hiperparâmetros dentro do treino de cada partição externa

# --------------------------------------------------------------------------- #
# Features declaradas
# --------------------------------------------------------------------------- #
# Entradas do modelo A: as 13 colunas do dataset que existem antes da veiculação.
PRE_FLIGHT_INPUTS = CONTEXT_COLS + CREATIVE_COLS
# Podem faltar, como no histórico. As demais são obrigatórias.
OPTIONAL_INPUTS = ["video_duration_s", "has_subtitle", "music_voice_ratio"]

# Colunas que o modelo A recebe depois da preparação, em ordem fixa.
CATEGORICAL_FEATURES = [
    "platform",
    "category",
    "objective",
    "target_audience_age",
    "format",
    "text_density",
    "subtitle_status",
]
BINARY_FEATURES = ["is_retargeting", "has_hook", "has_face", "has_cta"]
NUMERIC_FEATURES = ["video_duration_s", "music_voice_ratio"]
MODEL_A_FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# Legenda ausente é uma categoria: o criativo não foi marcado, o que acontece também
# em produção. A primeira categoria de cada lista é a referência do modelo linear.
SUBTITLE_STATUS = ["não", "sim", "desconhecido"]
CATEGORIES = {
    **{col: CATEGORY_ORDERS[col] for col in CATEGORICAL_FEATURES if col != "subtitle_status"},
    "subtitle_status": SUBTITLE_STATUS,
}
PLATFORM_FORMATS = [f"{p} | {f}" for p in CATEGORIES["platform"] for f in CATEGORIES["format"]]

# Taxas que o modelo B acrescenta. Volumes (impressões, gasto) ficam de fora: medem verba.
POST_FLIGHT_RATES = [
    "ctr",
    "engagement_rate",
    "completion_rate",
    "conversion_rate",
    "conversions_per_mille",
    "cpc",
    "roas",
]
MODEL_B_FEATURES = MODEL_A_FEATURES + POST_FLIGHT_RATES


@dataclass(frozen=True)
class FeatureSpec:
    """Opções de features testadas nos experimentos F1 a F3. Todas desligadas na base."""

    missing_indicators: bool = False  # F1: indicadores de ausência de duração e música
    log_duration: bool = False  # F2: duração em log (só faz sentido no modelo linear)
    platform_format: bool = False  # F3: interação entre plataforma e formato


BASE_SPEC = FeatureSpec()

# Espaços de hiperparâmetros, pequenos de propósito (ver o protocolo no notebook 03).
PARAM_GRIDS = {
    "ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "hgb": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [4, 8],
        "model__min_samples_leaf": [10, 30],
    },
}
# Iterações fixas, sem early stopping: ele reservaria parte de um treino que já é pequeno
# (320 campanhas por partição) para a validação interna. A semente garante a reprodução.
HGB_FIXED = {"max_iter": 200, "early_stopping": False}
# Mínimo de campanhas no treino para uma combinação de plataforma e formato virar coluna
# de interação (F3). Abaixo disso, o coeficiente dependeria de poucas campanhas.
MIN_INTERACTION_SUPPORT = 10


# --------------------------------------------------------------------------- #
# Preparação e validação
# --------------------------------------------------------------------------- #
def prepare_features(campaigns: pd.DataFrame | Mapping | Sequence[Mapping]) -> pd.DataFrame:
    """Converte campanhas no formato do modelo A, validando cada campo.

    Aceita um dicionário (uma campanha), uma lista de dicionários ou um DataFrame. Usa
    só as 13 colunas de pré-publicação: colunas extras, como métricas de performance,
    são ignoradas. Não depende da limpeza do histórico, que exige métricas que uma
    campanha nova ainda não tem. Entradas inválidas geram um erro que lista os problemas.
    """
    frame = _as_frame(campaigns)
    _validate(frame)
    out = pd.DataFrame(index=frame.index)
    for col in CATEGORICAL_FEATURES:
        if col == "subtitle_status":
            values = _optional(frame, "has_subtitle").map(_subtitle_status)
        else:
            values = frame[col].astype(object)
        out[col] = pd.Categorical(values, categories=CATEGORIES[col])
    for col in BINARY_FEATURES:
        out[col] = frame[col].astype(object).map(bool).astype(int)
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(_optional(frame, col), errors="coerce").astype(float)
    return out


def prepare_post_flight_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Modelo B (comparativo): as features do modelo A mais taxas de performance.

    Exige a base histórica limpa, porque as taxas vêm de métricas corrigidas na Parte 1.
    ROAS não se aplica a Branding: ali ele vale 0, e a categoria Branding, que já está no
    modelo, absorve a diferença de nível. Assim, a relação com o ROAS é aprendida só com
    as campanhas que medem receita.
    """
    rates = add_rate_metrics(cleaned)
    out = prepare_features(cleaned)
    for col in POST_FLIGHT_RATES:
        if col == "roas":
            out[col] = rates["roas"].where(rates["flag_roas_applicable"], 0.0).astype(float)
        else:
            out[col] = rates[col].astype(float)
    return out


def _as_frame(campaigns: pd.DataFrame | Mapping | Sequence[Mapping]) -> pd.DataFrame:
    if isinstance(campaigns, pd.DataFrame):
        return campaigns
    if isinstance(campaigns, Mapping):
        return pd.DataFrame([campaigns])
    return pd.DataFrame(list(campaigns))


def _optional(frame: pd.DataFrame, col: str) -> pd.Series:
    """Coluna opcional; se não vier na entrada, conta como ausente em todas as linhas."""
    if col in frame.columns:
        return frame[col].astype(object)
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _is_missing(value) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _is_bool(value) -> bool:
    """Aceita True/False e 0/1, como vêm de CSV, JSON ou do tipo boolean do pandas."""
    if isinstance(value, bool | np.bool_):
        return True
    return isinstance(value, int | np.integer | float | np.floating) and value in (0, 1)


def _subtitle_status(value) -> str:
    if _is_missing(value):
        return "desconhecido"
    return "sim" if bool(value) else "não"


def _number(value) -> float | None:
    if isinstance(value, bool | np.bool_ | str):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _validate(frame: pd.DataFrame) -> None:
    """Junta todos os problemas da entrada numa única mensagem de erro."""
    problems: list[str] = []
    required = [col for col in PRE_FLIGHT_INPUTS if col not in OPTIONAL_INPUTS]
    absent = [col for col in required if col not in frame.columns]
    if absent:
        problems.append(f"faltam as colunas obrigatórias {absent}")

    for col in required:
        if col not in frame.columns:
            continue
        values = frame[col].astype(object)
        missing = values.map(_is_missing)
        if missing.any():
            problems.append(f"{col}: valor ausente em {int(missing.sum())} campanha(s)")
        present = values[~missing]
        if col in CATEGORY_ORDERS:
            unknown = sorted({str(v) for v in present if v not in CATEGORY_ORDERS[col]})
            if unknown:
                allowed = ", ".join(CATEGORY_ORDERS[col])
                problems.append(f"{col}: valor desconhecido {unknown} (aceitos: {allowed})")
        elif not present.map(_is_bool).all():
            problems.append(f"{col}: esperado verdadeiro/falso")

    subtitle = _optional(frame, "has_subtitle")
    if not subtitle[~subtitle.map(_is_missing)].map(_is_bool).all():
        problems.append("has_subtitle: esperado verdadeiro/falso ou ausente")

    for col, valid, rule in [
        ("video_duration_s", lambda x: x > 0, "maior que zero"),
        ("music_voice_ratio", lambda x: 0 <= x <= 1, "entre 0 e 1"),
    ]:
        values = _optional(frame, col)
        numbers = values[~values.map(_is_missing)].map(_number)
        if numbers.isna().any() or not numbers.dropna().map(valid).all():
            problems.append(f"{col}: esperado um número {rule}, ou ausente")

    if problems:
        raise ValueError("Entrada inválida: " + "; ".join(problems) + ".")


# --------------------------------------------------------------------------- #
# Divisão dos dados e partições
# --------------------------------------------------------------------------- #
def split_development_test(
    data: pd.DataFrame, *, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa desenvolvimento (80%) e teste final (20%), estratificado por plataforma.

    Ordena por campaign_id antes do sorteio, para a divisão não depender da ordem das
    linhas. O teste só é consultado uma vez, depois que modelo, features e
    hiperparâmetros estiverem decididos.
    """
    ordered = data.sort_values(ID_COL, kind="stable").reset_index(drop=True)
    development, test = train_test_split(
        ordered,
        test_size=test_size,
        stratify=ordered["platform"].astype(str),
        random_state=random_state,
    )
    return (
        development.sort_values(ID_COL).reset_index(drop=True),
        test.sort_values(ID_COL).reset_index(drop=True),
    )


def outer_splits(
    development: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """As partições externas, as mesmas para todos os candidatos, estratificadas por plataforma.

    Partições repetidas compartilham campanhas: a variação entre elas mostra
    estabilidade, mas elas não são amostras independentes.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(development, development["platform"].astype(str)))


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build_pipeline(
    model: str, spec: FeatureSpec = BASE_SPEC, *, scenario: str = "pre", **params
) -> Pipeline:
    """Pipeline completo de um candidato: preparação aprendida no treino e modelo.

    - "dummy": prevê a média do treino, o piso que qualquer modelo precisa superar;
    - "ridge": regressão linear com regularização, a candidata interpretável;
    - "hgb": gradient boosting de histogramas, a candidata não linear.

    ``scenario="post"`` monta o modelo B, com as taxas de performance.
    """
    if scenario not in ("pre", "post"):
        raise ValueError(f"Cenário desconhecido: {scenario!r} (use 'pre' ou 'post').")
    rates = POST_FLIGHT_RATES if scenario == "post" else []
    if model == "dummy":
        return Pipeline([("model", DummyRegressor(strategy="mean"))])
    if model == "ridge":
        ridge = Ridge(**{"alpha": 1.0, **params})
        return Pipeline([("features", _linear_preprocessor(spec, rates)), ("model", ridge)])
    if model == "hgb":
        if spec.log_duration:
            raise ValueError(
                "Log da duração não muda um modelo de árvores: os cortes são os mesmos "
                "para qualquer transformação que preserve a ordem."
            )
        categorical = CATEGORICAL_FEATURES + (["platform_format"] if spec.platform_format else [])
        hgb = HistGradientBoostingRegressor(
            categorical_features=categorical,
            random_state=RANDOM_STATE,
            **{**HGB_FIXED, **params},
        )
        features = _tree_preprocessor(spec, rates).set_output(transform="pandas")
        return Pipeline([("features", features), ("model", hgb)])
    raise ValueError(f"Modelo desconhecido: {model!r} (use 'dummy', 'ridge' ou 'hgb').")


def tuned(model: str, spec: FeatureSpec = BASE_SPEC, *, scenario: str = "pre") -> GridSearchCV:
    """Candidato com hiperparâmetros escolhidos só dentro do treino de cada partição externa.

    Usado como estimador na validação cruzada externa, forma a validação aninhada: o
    ajuste nunca vê a parte de validação da partição externa.
    """
    return GridSearchCV(
        build_pipeline(model, spec, scenario=scenario),
        PARAM_GRIDS[model],
        cv=KFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error",
    )


def _linear_preprocessor(spec: FeatureSpec, rates: list[str]) -> ColumnTransformer:
    """Categorias em colunas 0/1 (contra a referência), binárias como vêm, números padronizados.

    As categorias usam a lista fixa do config, então toda partição gera as mesmas colunas,
    mesmo que alguma categoria falte no treino.
    """
    duration_steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if spec.log_duration:
        duration_steps.insert(0, FunctionTransformer(np.log, feature_names_out="one-to-one"))
    branches = [
        (
            "categorias",
            OneHotEncoder(
                categories=[CATEGORIES[col] for col in CATEGORICAL_FEATURES],
                drop="first",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
        ("binarias", "passthrough", BINARY_FEATURES),
        ("duracao", make_pipeline(*duration_steps), ["video_duration_s"]),
        (
            "numericas",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            ["music_voice_ratio", *rates],
        ),
    ]
    if spec.missing_indicators:
        branches.append(("ausencia", MissingIndicator(features="all"), NUMERIC_FEATURES))
    if spec.platform_format:
        branches.append(
            ("plataforma_formato", PlatformFormatInteractions(), ["platform", "format"])
        )
    return ColumnTransformer(branches, verbose_feature_names_out=False)


def _tree_preprocessor(spec: FeatureSpec, rates: list[str]) -> ColumnTransformer:
    """Categorias como códigos fixos para o suporte nativo do HGB; números como vêm.

    Valores ausentes seguem para o modelo, que aprende para que lado mandá-los. Os códigos
    vêm da lista fixa do config, então treino e inferência codificam igual.
    """
    branches = [
        (
            "categorias",
            OrdinalEncoder(categories=[CATEGORIES[col] for col in CATEGORICAL_FEATURES]),
            CATEGORICAL_FEATURES,
        ),
        ("binarias", "passthrough", BINARY_FEATURES),
        ("numericas", "passthrough", [*NUMERIC_FEATURES, *rates]),
    ]
    if spec.missing_indicators:
        branches.append(("ausencia", MissingIndicator(features="all"), NUMERIC_FEATURES))
    if spec.platform_format:
        branches.append(
            (
                "plataforma_formato",
                make_pipeline(
                    FunctionTransformer(
                        _platform_format_combined, feature_names_out=_combined_name
                    ),
                    OrdinalEncoder(categories=[PLATFORM_FORMATS]),
                ),
                ["platform", "format"],
            )
        )
    return ColumnTransformer(branches, verbose_feature_names_out=False)


class PlatformFormatInteractions(TransformerMixin, BaseEstimator):
    """Colunas 0/1 de plataforma × formato, só nas combinações com suporte no treino.

    As combinações são contadas no treino de cada partição. Uma combinação com menos de
    ``min_support`` campanhas, como TikTok horizontal, que não aparece na base, não vira
    coluna: o modelo não finge ter aprendido algo sobre ela. As referências de
    plataforma e de formato ficam de fora, como nos efeitos principais.
    """

    def __init__(self, min_support: int = MIN_INTERACTION_SUPPORT):
        self.min_support = min_support

    def fit(self, X: pd.DataFrame, y=None):
        pairs = Counter(zip(X["platform"].astype(str), X["format"].astype(str), strict=True))
        self.combinations_ = [
            (p, f)
            for p in CATEGORIES["platform"][1:]
            for f in CATEGORIES["format"][1:]
            if pairs[(p, f)] >= self.min_support
        ]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "combinations_")
        platform, fmt = X["platform"].astype(str).to_numpy(), X["format"].astype(str).to_numpy()
        columns = [((platform == p) & (fmt == f)).astype(int) for p, f in self.combinations_]
        return np.column_stack(columns) if columns else np.zeros((len(X), 0), dtype=int)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "combinations_")
        return np.array([f"{p} × {f}" for p, f in self.combinations_], dtype=object)


def _platform_format_combined(frame: pd.DataFrame) -> pd.DataFrame:
    """Plataforma e formato numa só categoria, para as árvores.

    Uma combinação que não aparece no treino, como TikTok horizontal, é tratada pelo
    modelo como valor ausente na previsão: não existe efeito aprendido para ela.
    """
    combined = frame["platform"].astype(str) + " | " + frame["format"].astype(str)
    return pd.DataFrame({"platform_format": combined}, index=frame.index)


def _combined_name(_, __) -> list[str]:
    return ["platform_format"]


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def evaluate_on_splits(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Ajusta o candidato em cada partição e mede o erro na parte de validação.

    Devolve uma linha por partição, com MAE (a métrica principal), RMSE, R² e Spearman.
    Como todos os candidatos usam as mesmas partições, as comparações são pareadas.
    """
    rows = []
    for number, (train, valid) in enumerate(splits):
        fitted = clone(estimator).fit(features.iloc[train], target.iloc[train])
        predicted = fitted.predict(features.iloc[valid])
        actual = target.iloc[valid].to_numpy()
        rows.append(
            {
                "particao": number,
                "mae": mean_absolute_error(actual, predicted),
                "rmse": root_mean_squared_error(actual, predicted),
                "r2": r2_score(actual, predicted),
                "spearman": _spearman(actual, predicted),
            }
        )
    return pd.DataFrame(rows)


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Correlação de postos; indefinida quando a previsão é constante, como no baseline."""
    if np.ptp(predicted) == 0:
        return np.nan
    return float(stats.spearmanr(actual, predicted).statistic)
