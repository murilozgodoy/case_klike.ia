"""Evidências do recommendations engine: contrastes de performance e diferenças de score.

O engine mantém três camadas separadas, e este módulo calcula as duas primeiras com a
terceira ao lado:

- associação histórica com performance: o contraste exato de cada transição sugerida, como
  texto alto → médio, nos modelos da Parte 1, em variação percentual e com intervalo de 95%;
- qualidade dessa evidência: correção de Benjamini-Hochberg sobre a família inteira de
  contrastes avaliados e análises de sensibilidade, sem as correções por plausibilidade e com
  binomial negativa para conversões, que mantém as campanhas sem conversão;
- diferença prevista de Klike Score: o que o modelo final prevê ao editar só a característica,
  com intervalo por bootstrap de todo o treino da especificação congelada.

Os contrastes são associações ajustadas, não efeitos causais. A família foi definida depois da
EDA: a correção controla a multiplicidade dentro dela, mas não torna a análise confirmatória.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from klike.analysis import (
    CONTEXT_FORMULA,
    CONTEXT_WITHIN_PLATFORM,
    add_evidence,
    add_rate_metrics,
    fit_model,
    fit_negative_binomial,
)
from klike.config import CTR_CAP, ENGAGEMENT_CAP, ID_COL, REPORTS_DIR, TARGET
from klike.data import PLAUSIBILITY, clean, load_clean, load_raw
from klike.formatting import fmt_compact
from klike.modeling import (
    RANDOM_STATE,
    SELECTED_MODEL,
    SELECTED_PARAMS,
    SELECTED_SPEC,
    build_pipeline,
    prepare_features,
)

PERFORMANCE_PATH = REPORTS_DIR / "recommendation_evidence.csv"
SCORE_PATH = REPORTS_DIR / "score_differences.csv"

# Métricas de performance estimadas para cada transição. A escolha da métrica principal de
# cada objetivo de campanha fica com o engine.
PERFORMANCE_METRICS = {
    "ctr": "CTR",
    "engagement_rate": "engajamento",
    "completion_rate": "retenção",
    "conversions_per_mille": "conversões por mil impressões",
    "roas_non_branding": "ROAS",
}
NEGATIVE_BINOMIAL_METRIC = "conversions_per_mille"  # a única com contagem e exposição
NOT_APPLICABLE = "não se aplica"
NOT_ESTIMABLE = "não estimável"

TEXT = "text_density"
FORMAT = "C(format, Treatment('vertical'))"


@dataclass(frozen=True)
class Transition:
    """Uma edição do criativo e o contraste que a mede nos modelos da Parte 1.

    ``current`` e ``target`` são os valores do campo antes e depois da edição. Para formato,
    o efeito é estimado dentro de ``platform``. Para música/voz e duração, que são números,
    não há dois grupos de campanhas a contar, e ``scale`` converte o coeficiente na edição
    descrita: dobrar a duração soma log(2) ao log da duração.
    """

    key: str
    label: str
    attribute: str
    current: object
    target: object
    contrast: str
    extra: str | None = None
    platform: str | None = None
    scale: float = 1.0

    @property
    def has_groups(self) -> bool:
        return self.attribute not in ("music_voice_ratio", "video_duration_s")


TRANSITIONS = [
    Transition("hook", "hook: não → sim", "has_hook", False, True, "has_hook"),
    Transition("rosto", "rosto humano: não → sim", "has_face", False, True, "has_face"),
    Transition("cta", "CTA: não → sim", "has_cta", False, True, "has_cta"),
    Transition(
        "legenda", "legenda: não → sim", "has_subtitle", False, True, "subtitle_status[T.sim]"
    ),
    Transition(
        "texto_alto_medio",
        "texto: alto → médio",
        TEXT,
        "high",
        "medium",
        f"{TEXT}[T.medium] - {TEXT}[T.high]",
    ),
    Transition("texto_alto_baixo", "texto: alto → baixo", TEXT, "high", "low", f"-{TEXT}[T.high]"),
    Transition(
        "texto_baixo_medio", "texto: baixo → médio", TEXT, "low", "medium", f"{TEXT}[T.medium]"
    ),
    Transition(
        "musica",
        "música/voz: +0,1 (de 0,4 para 0,5)",
        "music_voice_ratio",
        0.4,
        0.5,
        "musica_decimos",
        extra="musica_decimos",
    ),
    Transition(
        "duracao",
        "duração: dobrar (de 15 para 30 s)",
        "video_duration_s",
        15,
        30,
        "np.log(video_duration_s)",
        extra="np.log(video_duration_s)",
        scale=float(np.log(2)),
    ),
    Transition(
        "formato_meta_horizontal",
        "formato na Meta: horizontal → vertical",
        "format",
        "horizontal",
        "vertical",
        f"-{FORMAT}[T.horizontal]",
        extra=FORMAT,
        platform="Meta",
    ),
    Transition(
        "formato_meta_quadrado",
        "formato na Meta: quadrado → vertical",
        "format",
        "quadrado",
        "vertical",
        f"-{FORMAT}[T.quadrado]",
        extra=FORMAT,
        platform="Meta",
    ),
    Transition(
        "formato_tiktok_quadrado",
        "formato no TikTok: quadrado → vertical",
        "format",
        "quadrado",
        "vertical",
        f"-{FORMAT}[T.quadrado]",
        extra=FORMAT,
        platform="TikTok",
    ),
    Transition(
        "formato_linkedin_vertical",
        "formato no LinkedIn: vertical → horizontal",
        "format",
        "vertical",
        "horizontal",
        f"{FORMAT}[T.horizontal]",
        extra=FORMAT,
        platform="LinkedIn",
    ),
    Transition(
        "formato_linkedin_quadrado",
        "formato no LinkedIn: quadrado → horizontal",
        "format",
        "quadrado",
        "horizontal",
        f"{FORMAT}[T.horizontal] - {FORMAT}[T.quadrado]",
        extra=FORMAT,
        platform="LinkedIn",
    ),
]

# Campanha de referência para medir a diferença de score de cada edição. O modelo é aditivo:
# a diferença não depende dos outros campos, só da plataforma, no caso do formato.
REFERENCE_CAMPAIGN = {
    "platform": "Meta",
    "category": "E-commerce",
    "objective": "conversions",
    "target_audience_age": "25-34",
    "is_retargeting": False,
    "video_duration_s": 15,
    "format": "vertical",
    "has_subtitle": False,
    "has_cta": False,
    "has_hook": False,
    "has_face": False,
    "text_density": "low",
    "music_voice_ratio": 0.4,
}


# --------------------------------------------------------------------------- #
# Associação histórica com performance
# --------------------------------------------------------------------------- #
def performance_data(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Base limpa com as taxas da Parte 1 e a música/voz em décimos, como na EDA."""
    data = add_rate_metrics(cleaned)
    data["musica_decimos"] = data["music_voice_ratio"] * 10
    return data


def performance_evidence(cleaned: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabela de evidências: uma linha por transição e métrica, com BH e sensibilidades.

    O modelo principal é o da Parte 1: regressão em log com erros robustos, controlando o
    contexto da campanha e os outros atributos do criativo; para formato, dentro da
    plataforma. A correção de Benjamini-Hochberg cobre a família inteira, inclusive as
    transições que o engine descarta. As sensibilidades refazem a família sem as campanhas
    corrigidas por plausibilidade e, para conversões, com binomial negativa.
    """
    cleaned = load_clean() if cleaned is None else cleaned
    data = performance_data(cleaned)
    plausibility = _plausibility_ids(cleaned)

    main = _family(data)
    without = _family(data[~data[ID_COL].isin(plausibility)])
    count_model = _negative_binomial_family(data, main)

    table = main.assign(
        efeito_sem_plausibilidade=without["efeito"],
        evidencia_sem_plausibilidade=without["evidencia"],
        efeito_binomial_negativa=count_model["efeito"],
        evidencia_binomial_negativa=count_model["evidencia"],
    )
    count_level = table["evidencia_binomial_negativa"]
    comparable = ~count_level.isin([NOT_APPLICABLE, NOT_ESTIMABLE])
    table["estavel"] = (table["evidencia"] == table["evidencia_sem_plausibilidade"]) & (
        ~comparable | (count_level == table["evidencia"])
    )
    table["limitacoes"] = table["metrica"].map(_limitations(data))
    return table[PERFORMANCE_COLUMNS].astype(GROUP_SIZES)


PERFORMANCE_COLUMNS = [
    "transicao",
    "descricao",
    "atributo",
    "de",
    "para",
    "plataforma",
    "metrica",
    "metrica_nome",
    "efeito",
    "ic_inferior",
    "ic_superior",
    "p_valor",
    "q_valor",
    "evidencia",
    "efeito_sem_plausibilidade",
    "evidencia_sem_plausibilidade",
    "efeito_binomial_negativa",
    "evidencia_binomial_negativa",
    "estavel",
    "n",
    "n_de",
    "n_para",
    "limitacoes",
]


def _family(data: pd.DataFrame) -> pd.DataFrame:
    """Estima a família inteira num cenário de dados e classifica com BH."""
    fits: dict[tuple, object] = {}
    rows = []
    for transition in TRANSITIONS:
        for metric, name in PERFORMANCE_METRICS.items():
            key = (metric, transition.extra, transition.platform)
            if key not in fits:
                fits[key] = fit_model(
                    _scope(data, transition),
                    metric,
                    controls=_controls(transition),
                    extra=transition.extra,
                )
            fit = fits[key]
            effect, low, high, p = transition_effect(fit, transition)
            n_from, n_to = _group_sizes(data, fit, transition)
            rows.append(
                {
                    "transicao": transition.key,
                    "descricao": transition.label,
                    "atributo": transition.attribute,
                    "de": _readable(transition.current),
                    "para": _readable(transition.target),
                    "plataforma": transition.platform or "todas",
                    "metrica": metric,
                    "metrica_nome": name,
                    "efeito": effect,
                    "ic_inferior": low,
                    "ic_superior": high,
                    "p_valor": p,
                    "n": int(fit.nobs),
                    "n_de": n_from,
                    "n_para": n_to,
                }
            )
    return add_evidence(pd.DataFrame(rows))


def _negative_binomial_family(data: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Conversões com binomial negativa: troca só essas linhas e reclassifica a família.

    A binomial negativa mantém as campanhas sem conversão, que a regressão em log descarta.
    Onde ela não converge, a linha fica como não estimável e a família usa o valor do log,
    para que o tamanho da família não mude entre os cenários.
    """
    fits: dict[tuple, object] = {}
    family = main[["metrica", "transicao", "efeito", "p_valor"]].copy()
    family["efeito"] = np.nan
    estimable = pd.Series(False, index=family.index)
    conversions = family["metrica"] == NEGATIVE_BINOMIAL_METRIC
    for transition in TRANSITIONS:
        row = family.index[conversions & (family["transicao"] == transition.key)][0]
        key = (transition.extra, transition.platform)
        if key not in fits:
            fits[key] = _fit_negative_binomial(data, transition)
        if fits[key] is None:
            continue
        effect, _, _, p = transition_effect(fits[key], transition)
        family.loc[row, ["efeito", "p_valor"]] = [effect, p]
        estimable[row] = True
    classified = add_evidence(family)
    level = classified["evidencia"].where(estimable, NOT_ESTIMABLE)
    level = level.where(family["metrica"] == NEGATIVE_BINOMIAL_METRIC, NOT_APPLICABLE)
    return pd.DataFrame({"efeito": family["efeito"], "evidencia": level})


def _fit_negative_binomial(data: pd.DataFrame, transition: Transition):
    # Com poucas campanhas, o otimizador emite avisos numéricos antes de falhar; a falha é
    # tratada como não estimável, então os avisos intermediários não acrescentam nada.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return fit_negative_binomial(
                _scope(data, transition),
                "conversions",
                "impressions",
                controls=_controls(transition),
                extra=transition.extra,
            )
        except RuntimeError:
            return None


def _readable(value) -> str:
    """Valor do campo como texto, para a tabela exportada: sim/não, categorias e números."""
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, int | float):
        return fmt_compact(value)
    return str(value)


def _scope(data: pd.DataFrame, transition: Transition) -> pd.DataFrame:
    if transition.platform is None:
        return data
    return data[data["platform"] == transition.platform]


def _controls(transition: Transition) -> str:
    return CONTEXT_FORMULA if transition.platform is None else CONTEXT_WITHIN_PLATFORM


def transition_effect(
    fit, transition: Transition, *, log: bool = True
) -> tuple[float, float, float, float]:
    """Efeito da transição num modelo ajustado: estimativa, intervalo de 95% e p-valor.

    Com ``log=True``, o resultado do modelo está em log e o efeito vira variação percentual
    (0,29 = +29%). Com ``log=False``, fica na unidade do resultado, como pontos de score.
    """
    test = fit.t_test(f"{transition.contrast} = 0")
    estimate = float(np.squeeze(test.effect)) * transition.scale
    low, high = (np.squeeze(test.conf_int()) * transition.scale).tolist()
    if log:
        estimate, low, high = np.expm1([estimate, low, high])
    return float(estimate), float(low), float(high), float(np.squeeze(test.pvalue))


def _group_sizes(data: pd.DataFrame, fit, transition: Transition) -> tuple[float, float]:
    """Campanhas do ajuste com o valor atual e com o valor sugerido."""
    if not transition.has_groups:
        return np.nan, np.nan
    values = data.loc[fit.model.data.row_labels, transition.attribute]
    return float((values == transition.current).sum()), float((values == transition.target).sum())


def _plausibility_ids(cleaned: pd.DataFrame) -> set[str]:
    """Campanhas cuja correção na Parte 1 se apoiou em plausibilidade, não em identidade."""
    log = clean(load_raw()).log
    ids = set(log.loc[log["basis"] == PLAUSIBILITY, ID_COL])
    return ids & set(cleaned[ID_COL])


def _limitations(data: pd.DataFrame) -> dict[str, str]:
    """Fragilidades conhecidas de cada métrica, contadas na própria base."""
    no_conversion = int((data["conversions"] == 0).sum())
    branding = int((~data["flag_roas_applicable"]).sum())
    return {
        "ctr": f"teto de registro: {int((data['ctr'] >= CTR_CAP).sum())} campanhas com CTR de 15%",
        "engagement_rate": (
            f"teto de registro: {int((data['engagement_rate'] >= ENGAGEMENT_CAP).sum())} "
            f"campanhas com 10%; {int(data['engagement_rate'].isna().sum())} sem registro"
        ),
        "completion_rate": (
            f"limitada a 100%: {int((data['completion_rate'] >= 1).sum())} campanhas; "
            f"{int(data['completion_rate'].isna().sum())} sem tempo assistido ou duração"
        ),
        "conversions_per_mille": (
            f"{no_conversion} campanhas sem conversão ficam fora do modelo em log; "
            "a binomial negativa as mantém"
        ),
        "roas_non_branding": f"não se aplica a Branding ({branding} campanhas)",
    }


# --------------------------------------------------------------------------- #
# Diferença prevista de Klike Score
# --------------------------------------------------------------------------- #
def score_differences(
    cleaned: pd.DataFrame | None = None,
    *,
    n_resamples: int = 1000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Diferença de score prevista para cada edição, com intervalo de 95% por bootstrap.

    A estimativa vem do modelo final, a especificação congelada treinada nas 500 campanhas.
    O intervalo reamostra as campanhas e refaz todo o treino a cada amostra, inclusive o que
    o pipeline aprende (imputação, padronização e quais combinações de plataforma e formato
    viram interação), sem nova escolha de hiperparâmetros ou de features. Ele mede a
    variabilidade da estimativa preditiva, não o efeito causal de editar um vídeo, e não usa
    o MAE do modelo, que mede o erro de prever o score de uma campanha.
    """
    cleaned = load_clean() if cleaned is None else cleaned
    features, target = prepare_features(cleaned), cleaned[TARGET]
    before, after = _edited_reference()
    estimate = _differences(_refit(features, target), before, after)
    rng = np.random.default_rng(random_state)
    samples = rng.integers(0, len(features), (n_resamples, len(features)))
    draws = np.array(
        [_differences(_refit(features, target, rows), before, after) for rows in samples]
    )
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "transicao": [t.key for t in TRANSITIONS],
            "descricao": [t.label for t in TRANSITIONS],
            "plataforma": [t.platform or "todas" for t in TRANSITIONS],
            "diferenca": estimate,
            "ic_inferior": low,
            "ic_superior": high,
            "reamostragens": n_resamples,
        }
    )


def _refit(features: pd.DataFrame, target: pd.Series, rows: np.ndarray | None = None):
    """Treina a especificação congelada do zero nas linhas pedidas (todas, se nenhuma)."""
    if rows is not None:
        features, target = features.iloc[rows], target.iloc[rows]
    return build_pipeline(SELECTED_MODEL, SELECTED_SPEC, **SELECTED_PARAMS).fit(features, target)


def _edited_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A campanha de referência antes e depois de cada edição, já preparada."""
    before, after = [], []
    for transition in TRANSITIONS:
        context = {"platform": transition.platform} if transition.platform else {}
        before.append({**REFERENCE_CAMPAIGN, **context, transition.attribute: transition.current})
        after.append({**REFERENCE_CAMPAIGN, **context, transition.attribute: transition.target})
    return prepare_features(before), prepare_features(after)


def _differences(pipeline, before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(after) - pipeline.predict(before)


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #
def save_evidence(
    performance: pd.DataFrame,
    score: pd.DataFrame,
    performance_path: str | Path = PERFORMANCE_PATH,
    score_path: str | Path = SCORE_PATH,
) -> None:
    """Grava as duas tabelas em CSV, com a precisão que a leitura precisa."""
    performance.round(PERFORMANCE_ROUNDING).to_csv(performance_path, index=False)
    score.round(SCORE_ROUNDING).to_csv(score_path, index=False)


def load_performance_evidence(path: str | Path = PERFORMANCE_PATH) -> pd.DataFrame:
    """Tabela de evidências de performance gravada por ``save_evidence``."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""]).astype(GROUP_SIZES)


def load_score_differences(path: str | Path = SCORE_PATH) -> pd.DataFrame:
    """Diferenças de score com intervalo, gravadas por ``save_evidence``."""
    return pd.read_csv(path)


# Variações relativas com 4 casas (0,5130 = +51,30%); pontos de score com 2. Os p-valores
# ficam com precisão completa: arredondados, os muito pequenos virariam zero.
PERFORMANCE_ROUNDING = {
    "efeito": 4,
    "ic_inferior": 4,
    "ic_superior": 4,
    "efeito_sem_plausibilidade": 4,
    "efeito_binomial_negativa": 4,
}
GROUP_SIZES = {"n_de": "Int64", "n_para": "Int64"}
SCORE_ROUNDING = {"diferenca": 2, "ic_inferior": 2, "ic_superior": 2}
