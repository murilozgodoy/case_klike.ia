"""Métricas comparáveis entre campanhas e estimativa do efeito de atributos do criativo.

Métricas de volume (impressões, cliques, conversões) crescem com o orçamento, então
comparar criativos por elas mede verba, não qualidade. Por isso trabalhamos com taxas.

O efeito de um atributo é estimado por regressão com controles de contexto
(plataforma, categoria, objetivo, faixa etária e retargeting): compara campanhas que
diferem no atributo, mantendo o resto igual. É uma associação, não prova de causa.

Com dezenas de testes, alguns passam de p < 0,05 só por acaso. Por isso cada família
de testes, o conjunto que responde a uma mesma pergunta, recebe o q-valor de
Benjamini-Hochberg, que separa resultados robustos de indícios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.base.wrapper import ResultsWrapper

BEST_PRACTICES = ["has_hook", "has_face", "has_subtitle", "has_cta"]

# Modelo padrão: atributos do criativo + contexto da campanha.
CREATIVE_FORMULA = "has_hook + has_face + subtitle_status + has_cta + text_density"
CONTEXT_FORMULA = (
    "C(platform) + C(category) + C(objective) + C(target_audience_age) + is_retargeting"
)
# Dentro de uma plataforma, a plataforma deixa de ser controle.
CONTEXT_WITHIN_PLATFORM = "C(category) + C(objective) + C(target_audience_age) + is_retargeting"

CREATIVE_TERMS = {
    "has_hook": "Hook nos 3 primeiros segundos",
    "has_face": "Rosto humano",
    "subtitle_status[T.sim]": "Legenda",
    "has_cta": "Call-to-action",
    "text_density[T.medium]": "Texto médio (vs. baixo)",
    "text_density[T.high]": "Texto alto (vs. baixo)",
}

# Níveis de evidência, do mais forte ao mais fraco (ver add_evidence).
EVIDENCE_LEVELS = ["robusto", "indício", "inconclusivo"]


def add_rate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta taxas que permitem comparar campanhas de tamanhos diferentes.

    - completion_rate: tempo médio assistido / duração, limitado a 100%;
    - conversion_rate: conversões por clique;
    - conversions_per_mille: conversões por mil impressões (CTR x conversão);
    - roas_non_branding: ROAS só onde ele se aplica, ausente em Branding.
    """
    out = df.copy()
    out["completion_rate"] = (out["avg_watch_time_s"] / out["video_duration_s"]).clip(upper=1)
    out["conversion_rate"] = out["conversions"] / out["clicks"]
    out["conversions_per_mille"] = out["conversions"] / out["impressions"] * 1000
    out["roas_non_branding"] = out["roas"].where(out["flag_roas_applicable"])
    return out


def count_best_practices(df: pd.DataFrame) -> pd.Series:
    """Quantas das 4 boas práticas (hook, rosto, legenda e CTA) o criativo tem.

    Fica ausente quando alguma delas não foi marcada, para não contar um
    "desconhecido" como "não tem".
    """
    practices = df[BEST_PRACTICES].astype("boolean")
    count = practices.fillna(False).astype(int).sum(axis=1).astype(float)
    return count.where(practices.notna().all(axis=1))


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as colunas para o formato que a fórmula da regressão espera."""
    out = df.copy()
    for col in ["has_hook", "has_face", "has_cta", "is_retargeting"]:
        out[col] = out[col].astype(int)
    out["subtitle_status"] = pd.Categorical(
        out["has_subtitle"].map({True: "sim", False: "não"}).fillna("desconhecido"),
        categories=["não", "sim", "desconhecido"],
    )
    out["text_density"] = pd.Categorical(
        out["text_density"].astype(str), categories=["low", "medium", "high"]
    )
    return out


def _without_unused_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Remove categorias sem nenhuma campanha no recorte.

    Elas virariam colunas zeradas e deixariam o modelo indeterminado, como acontece
    com uma categoria de anunciante ausente numa plataforma.
    """
    categorical = data.select_dtypes("category").columns
    return data.assign(**{col: data[col].cat.remove_unused_categories() for col in categorical})


def _rhs(controls: str | None, extra: str | None) -> str:
    """Lado direito da fórmula: atributos do criativo, controles e termos extras."""
    return " + ".join(part for part in [CREATIVE_FORMULA, controls, extra] if part)


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    *,
    log: bool = True,
    controls: str | None = CONTEXT_FORMULA,
    extra: str | None = None,
) -> ResultsWrapper:
    """Regressão do resultado nos atributos do criativo, com erros robustos (HC3).

    Com ``log=True`` o resultado entra em log: cada coeficiente vira uma variação
    percentual, o que deixa comparáveis métricas de escalas diferentes. Linhas sem o
    resultado ficam de fora e, na escala log, as com resultado zero também. Para
    resultados com zeros, ``fit_poisson`` e ``fit_negative_binomial`` servem de
    checagem de sensibilidade.
    """
    data = modeling_frame(df).dropna(subset=[outcome])
    if log:
        data = data[data[outcome] > 0]
    data = _without_unused_categories(data)
    lhs = f"np.log({outcome})" if log else outcome
    return smf.ols(f"{lhs} ~ {_rhs(controls, extra)}", data=data).fit(cov_type="HC3")


def fit_poisson(
    df: pd.DataFrame,
    outcome: str,
    *,
    exposure: str | None = None,
    controls: str | None = CONTEXT_FORMULA,
    extra: str | None = None,
) -> ResultsWrapper:
    """Regressão de Poisson por pseudo-verossimilhança, com erros robustos.

    Alternativa ao log para resultados com zeros: o coeficiente também vira variação
    percentual, agora sobre a média do resultado, e as campanhas com zero ficam no
    modelo. O resultado não precisa ser uma contagem; os erros robustos cobrem a
    variância que a distribuição de Poisson não prevê.

    Sem ``exposure``, o resultado é a própria taxa e cada campanha pesa igual, como
    na regressão em log. Com ``exposure`` (ex.: conversões com impressões), o modelo
    descreve a taxa agregada, e campanhas com mais volume pesam mais.
    """
    required = [outcome, exposure] if exposure else [outcome]
    data = _without_unused_categories(modeling_frame(df).dropna(subset=required))
    offset = np.log(data[exposure]) if exposure else None
    model = smf.glm(
        f"{outcome} ~ {_rhs(controls, extra)}",
        data=data,
        family=sm.families.Poisson(),
        offset=offset,
    )
    return model.fit(cov_type="HC0")


def fit_negative_binomial(
    df: pd.DataFrame,
    count: str,
    exposure: str,
    *,
    controls: str | None = CONTEXT_FORMULA,
    extra: str | None = None,
) -> ResultsWrapper:
    """Binomial negativa para contagens com exposição, como conversões por impressão.

    Mantém os zeros e acomoda a sobredispersão: quando as campanhas diferem em coisas
    não observadas, a variância das contagens cresce mais rápido que a média. Campanhas
    com poucas conversões, cuja taxa é mais ruidosa, pesam menos na estimativa.
    """
    data = _without_unused_categories(modeling_frame(df).dropna(subset=[count, exposure]))
    model = smf.negativebinomial(
        f"{count} ~ {_rhs(controls, extra)}", data=data, offset=np.log(data[exposure])
    )
    fit = model.fit(method="newton", maxiter=100, disp=0, cov_type="HC0")
    if not fit.mle_retvals["converged"]:
        raise RuntimeError(f"A binomial negativa não convergiu para {count!r}.")
    return fit


def tidy_effects(fit: ResultsWrapper, terms: dict[str, str], *, log: bool = True) -> pd.DataFrame:
    """Tabela com efeito, intervalo de confiança de 95% e significância de cada termo.

    Na escala log, o efeito é a variação percentual (0,51 = +51%). Na escala
    original, é a diferença em unidades do resultado, como pontos de score.
    ``significativo`` olha cada teste isoladamente (o intervalo não cruza o zero);
    para uma família de testes, ``add_evidence`` aplica a correção.
    """
    ci = fit.conf_int()
    rows = []
    for term, label in terms.items():
        if term not in fit.params:
            continue
        estimate = np.array([fit.params[term], ci.loc[term, 0], ci.loc[term, 1]])
        if log:
            estimate = np.exp(estimate) - 1
        rows.append(
            {
                "atributo": label,
                "efeito": estimate[0],
                "ic_inferior": estimate[1],
                "ic_superior": estimate[2],
                "p_valor": fit.pvalues[term],
            }
        )
    table = pd.DataFrame(rows)
    table["significativo"] = (table["ic_inferior"] > 0) | (table["ic_superior"] < 0)
    table.attrs["n"] = int(fit.nobs)
    return table


def estimate_effects(
    df: pd.DataFrame,
    outcome: str,
    *,
    log: bool = True,
    controls: str | None = CONTEXT_FORMULA,
    terms: dict[str, str] = CREATIVE_TERMS,
) -> pd.DataFrame:
    """Atalho: ajusta o modelo padrão e devolve a tabela de efeitos."""
    return tidy_effects(fit_model(df, outcome, log=log, controls=controls), terms, log=log)


def add_evidence(table: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Acrescenta o q-valor de Benjamini-Hochberg e o nível de evidência de cada teste.

    A tabela inteira é tratada como uma família. O q-valor controla a proporção
    esperada de falsos positivos entre os resultados declarados. Níveis:

    - robusto: q < alpha, resiste à correção para múltiplos testes;
    - indício: p < alpha, mas q >= alpha, então pode ser acaso;
    - inconclusivo: p >= alpha.
    """
    out = table.copy()
    out["q_valor"] = stats.false_discovery_control(out["p_valor"], method="bh")
    out["evidencia"] = np.select(
        [out["q_valor"] < alpha, out["p_valor"] < alpha],
        EVIDENCE_LEVELS[:2],
        default=EVIDENCE_LEVELS[2],
    )
    return out


def interaction_pvalue(
    df: pd.DataFrame,
    attribute: str,
    outcome: str,
    *,
    moderator: str = "C(platform)",
    log: bool = True,
) -> float:
    """p-valor do teste de que o efeito do atributo é o mesmo em todos os grupos.

    Comparar a significância do efeito em cada grupo, separadamente, engana: um
    efeito pode ser significativo numa plataforma e não na outra só porque a
    segunda tem menos campanhas. O teste certo compara os efeitos diretamente,
    com termos de interação atributo x grupo, e verifica se são todos zero.
    """
    terms_in_model = [part.strip() for part in CREATIVE_FORMULA.split("+")]
    main_effect = None if attribute in terms_in_model else attribute
    interaction = f"{attribute}:{moderator}"
    extra = " + ".join(part for part in [main_effect, interaction] if part)
    fit = fit_model(df, outcome, log=log, extra=extra)
    names = list(fit.params.index)
    terms = [t for t in names if t.startswith(attribute) and f":{moderator}" in t]
    restrictions = np.zeros((len(terms), len(names)))
    for row, term in enumerate(terms):
        restrictions[row, names.index(term)] = 1
    return float(fit.wald_test(restrictions, scalar=True).pvalue)
