"""Avaliação do recommendations engine nas campanhas históricas.

Rodar o engine nas 500 campanhas da base é uma avaliação funcional: mede a cobertura, a
composição das recomendações, a coerência das saídas e quanto a ordem depende dos critérios
escolhidos. Não valida o impacto das edições: as evidências e o modelo vieram dessas mesmas
campanhas, e só um teste A/B mede o efeito de uma edição.

O módulo transforma relatórios do engine em tabelas e confere a coerência deles. Não recalcula
evidências nem previsões.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from klike.config import CONTEXT_COLS, REPORTS_DIR
from klike.engine import (
    ACTIONS,
    BORDERLINE_SCORE,
    CAVEATS,
    EXECUTIVE_SIZE,
    MUSIC_MAX_TARGET,
    MUSIC_STEP,
    NO_CAVEAT,
    OBJECTIVE_METRICS,
    SUPPORT_CAVEATS,
    Association,
    Recommendation,
    RecommendationEngine,
    RecommendationReport,
)
from klike.evidence import PERFORMANCE_METRICS
from klike.modeling import PRE_FLIGHT_INPUTS

EVALUATION_PATH = REPORTS_DIR / "engine_recommendations.csv"
CLASS_ORDER = {"A": 0, "B": 1, "C": 2}
RISK_ORDER = {"baixo": 0, "moderado": 1, "alto": 2}


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def recommendation_table(reports: Sequence[RecommendationReport]) -> pd.DataFrame:
    """Uma linha por recomendação: contexto da campanha, classe, associação, score e suporte.

    As marcações saem dos campos estruturados, não do texto dos avisos; ``audit`` confere que
    os avisos dizem o mesmo.
    """
    rows = []
    for position, report in enumerate(reports):
        campaign = report.campaign
        for r in report.recommendations:
            a = r.association
            estimate, low, high = r.model_change
            rows.append(
                {
                    "campanha_id": _label(report, position),
                    "plataforma": campaign["platform"],
                    "categoria": campaign["category"],
                    "objetivo": campaign["objective"],
                    "formato": campaign["format"],
                    "prioridade": r.priority,
                    "acao": r.action,
                    "de": r.current_label,
                    "para": r.suggested_label,
                    "classe": r.recommendation_class,
                    "metrica": r.metric,
                    "efeito": a.effect if a else math.nan,
                    "ic_inferior": a.low if a else math.nan,
                    "ic_superior": a.high if a else math.nan,
                    "evidencia": a.evidence if a else None,
                    "estavel": a.stable if a else None,
                    "score_antes": r.score_before,
                    "score_depois": r.score_after,
                    "diferenca_score": r.score_change,
                    "score_modelo": estimate,
                    "score_modelo_inferior": low,
                    "score_modelo_superior": high,
                    "recorte": r.support.level,
                    "campanhas_valor_atual": r.support.n_current,
                    "campanhas_valor_sugerido": r.support.n_target,
                    "risco": r.support.risk,
                    "sinal_desfavoravel": _unfavorable(a),
                    "limitrofe": 0 < low < BORDERLINE_SCORE,
                    "extrapolacao": report.combination_support < r.support.minimum,
                    "outras_metricas_favoraveis": ", ".join(
                        other.metric_name for other in r.other_associations if other.effect > 0
                    ),
                    "avisos": len(r.warnings),
                    "ressalvas": " · ".join(r.caveats),
                }
            )
    return pd.DataFrame(rows)


def campaign_table(reports: Sequence[RecommendationReport]) -> pd.DataFrame:
    """Uma linha por campanha: recomendações por classe, a primeira e os avisos da campanha."""
    rows = []
    for position, report in enumerate(reports):
        campaign, recs = report.campaign, report.recommendations
        classes = [r.recommendation_class for r in recs]
        rows.append(
            {
                "campanha_id": _label(report, position),
                "plataforma": campaign["platform"],
                "categoria": campaign["category"],
                "objetivo": campaign["objective"],
                "formato": campaign["format"],
                "score_previsto": report.predicted_score,
                "recomendacoes": len(recs),
                **{klass: classes.count(klass) for klass in CLASS_ORDER},
                "primeira": recs[0].action if recs else None,
                "nao_recomendadas": len(report.not_recommended),
                "avisos_da_campanha": len(report.notes),
            }
        )
    return pd.DataFrame(rows)


def coverage(campaigns: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Campanhas, recomendações e a parcela de campanhas com 0, 1, 2 e 3 ou mais.

    ``campaigns`` é a saída de ``campaign_table``; ``by`` agrupa por uma coluna dela. As duas
    últimas colunas dizem quantas campanhas têm alguma recomendação com evidência na métrica do
    objetivo (A ou B) e quantas só recebem oportunidades de score (C).
    """
    groups = [("todas", campaigns)] if by is None else campaigns.groupby(by, sort=False)
    rows = {}
    for name, group in groups:
        n = group["recomendacoes"]
        rows[name] = {
            "campanhas": len(group),
            "recomendacoes": int(n.sum()),
            "media": n.mean(),
            "nenhuma": (n == 0).mean(),
            "uma": (n == 1).mean(),
            "duas": (n == 2).mean(),
            "tres_ou_mais": (n >= 3).mean(),
            "com_A_ou_B": (group["A"] + group["B"] > 0).mean(),
            "so_C": ((group["C"] == n) & (n > 0)).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def action_funnel(reports: Sequence[RecommendationReport]) -> pd.DataFrame:
    """Para cada edição do catálogo, quantas campanhas poderiam recebê-la e quantas recebem.

    Na plataforma e nas métricas da edição, a campanha tem o valor ausente, outro valor (como
    quem já tem hook, ou música/voz acima de 0,57) ou o valor de partida. Só as do último grupo
    são avaliadas: viram recomendação de alguma classe ou são rejeitadas, com o motivo
    registrado no relatório.
    """
    rows = []
    for action in ACTIONS:
        counts = Counter()
        for report in reports:
            campaign = report.campaign
            if action.platform and campaign["platform"] != action.platform:
                continue
            if not action.evaluated_for(report.metric):
                continue
            counts["campanhas"] += 1
            if campaign[action.field] is None:
                counts["valor_ausente"] += 1
            elif not action.applies_to(campaign):
                counts["outro_valor"] += 1
            else:
                counts["avaliadas"] += 1
                klass = next(
                    (
                        r.recommendation_class
                        for r in report.recommendations
                        if r.action == action.key
                    ),
                    "rejeitada",
                )
                counts[klass] += 1
        metrics = (
            ", ".join(PERFORMANCE_METRICS[m] for m in action.metrics) if action.metrics else "todas"
        )
        rows.append(
            {
                "acao": action.key,
                "plataforma": action.platform or "todas",
                "metricas": metrics,
                **counts,
            }
        )
    columns = ["campanhas", "valor_ausente", "outro_valor", "avaliadas", *CLASS_ORDER, "rejeitada"]
    funnel = pd.DataFrame(rows).reindex(columns=["acao", "plataforma", "metricas", *columns])
    return funnel.fillna(0).astype(dict.fromkeys(columns, int)).set_index("acao")


def rejections(reports: Sequence[RecommendationReport]) -> pd.DataFrame:
    """Edições aplicáveis que o engine avaliou e não recomendou, com o motivo."""
    rows = [
        {"campanha_id": _label(report, position), "acao": key, "motivo": reason}
        for position, report in enumerate(reports)
        for key, reason in report.not_recommended
    ]
    return pd.DataFrame(rows, columns=["campanha_id", "acao", "motivo"])


def save_recommendations(table: pd.DataFrame, path: Path = EVALUATION_PATH) -> Path:
    """Grava a tabela de recomendações, com os números arredondados em 4 casas."""
    table.round(4).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Ordem: alternativas e conflitos
# --------------------------------------------------------------------------- #
def engine_order(row) -> tuple:
    """A ordem do engine, refeita a partir de uma linha de ``recommendation_table``.

    Classe primeiro; dentro de A e de B, a estimativa na métrica do objetivo; dentro de C, o
    limite inferior do intervalo do score do modelo. A chave da ação desempata.
    """
    if row.classe == "C":
        return (2, -row.score_modelo_inferior, -row.score_modelo, row.acao)
    return (CLASS_ORDER[row.classe], -row.efeito, -row.ic_inferior, -row.diferenca_score, row.acao)


def _descending(column: str) -> Callable:
    """Uma coluna só, em ordem decrescente e sem olhar a classe; sem valor, vai para o fim."""

    def key(row) -> tuple:
        value = getattr(row, column)
        return (math.inf if pd.isna(value) else -value, row.acao)

    return key


def _sensitive_as_a(row) -> tuple:
    """Como o engine, mas a evidência robusta e sensível à análise conta como A."""
    if row.classe == "B" and row.evidencia == "robusto":
        row = row._replace(classe="A")
    return engine_order(row)


def _support_first(row) -> tuple:
    """Como o engine, mas, dentro de cada classe, o suporte mais específico vem antes."""
    order = engine_order(row)
    return (order[0], RISK_ORDER[row.risco], *order[1:])


def _c_by_estimate(row) -> tuple:
    """Como o engine, mas as oportunidades de score seguem a estimativa pontual do modelo."""
    if row.classe == "C":
        return (2, -row.score_modelo, -row.score_modelo_inferior, row.acao)
    return engine_order(row)


# Ordens plausíveis para o diagnóstico de robustez. A ordem do engine continua a padrão.
ALTERNATIVE_ORDERS = {
    "só impacto": ("estimativa na métrica do objetivo, sem olhar a classe", _descending("efeito")),
    "conservadora": (
        "limite inferior do intervalo na métrica do objetivo, sem olhar a classe",
        _descending("ic_inferior"),
    ),
    "só score": ("ganho de score previsto para a campanha", _descending("diferenca_score")),
    "sensível como A": (
        "como a do engine, mas a evidência robusta e sensível à análise conta como A",
        _sensitive_as_a,
    ),
    "suporte primeiro": (
        "como a do engine, mas dentro de cada classe o risco de extrapolação menor vem antes",
        _support_first,
    ),
    "C pela estimativa": (
        "como a do engine, mas a classe C segue a estimativa pontual do score",
        _c_by_estimate,
    ),
}


def reorder(table: pd.DataFrame, key: Callable) -> pd.Series:
    """Prioridade de cada recomendação numa ordem alternativa, campanha a campanha."""
    priority = {}
    for _, group in table.groupby("campanha_id", sort=False):
        for position, row in enumerate(sorted(group.itertuples(), key=key), start=1):
            priority[row.Index] = position
    return pd.Series(priority, name="prioridade").reindex(table.index)


def order_robustness(
    table: pd.DataFrame, orders: dict = ALTERNATIVE_ORDERS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantas campanhas mudariam a primeira recomendação, ou a ordem, em cada alternativa.

    Só contam campanhas com duas ou mais recomendações. Devolve o resumo por alternativa, com
    uma última linha para qualquer uma delas, e as trocas da primeira recomendação: de qual
    edição para qual, em quantas campanhas.
    """
    multi = table[table.groupby("campanha_id")["acao"].transform("size") >= 2]
    first = multi.loc[multi["prioridade"] == 1].set_index("campanha_id")["acao"]
    summary, swaps = [], []
    any_first, any_order = set(), set()
    for name, (criterion, key) in orders.items():
        priority = reorder(multi, key)
        new_first = multi.loc[priority == 1].set_index("campanha_id")["acao"].reindex(first.index)
        changed = first != new_first
        reordered = (priority != multi["prioridade"]).groupby(multi["campanha_id"]).any()
        any_first |= set(first.index[changed])
        any_order |= set(reordered.index[reordered])
        summary.append(
            {
                "ordem": name,
                "criterio": criterion,
                "campanhas": len(first),
                "primeira_muda": int(changed.sum()),
                "ordem_muda": int(reordered.sum()),
            }
        )
        pairs = Counter(zip(first[changed], new_first[changed], strict=True))
        swaps += [
            {"ordem": name, "de": old, "para": new, "campanhas": n}
            for (old, new), n in pairs.most_common()
        ]
    summary.append(
        {
            "ordem": "qualquer uma",
            "criterio": "alguma das alternativas acima",
            "campanhas": len(first),
            "primeira_muda": len(any_first),
            "ordem_muda": len(any_order),
        }
    )
    columns = ["ordem", "de", "para", "campanhas"]
    return pd.DataFrame(summary).set_index("ordem"), pd.DataFrame(swaps, columns=columns)


def conflicts(table: pd.DataFrame) -> pd.DataFrame:
    """Pares da mesma campanha em que outro critério inverteria a ordem do engine.

    Em cada par, uma recomendação vem antes da outra. A de baixo pode ter: estimativa maior na
    métrica do objetivo, com classe pior (classe × magnitude); ganho de score maior (score ×
    associação); suporte num recorte mais específico (impacto × suporte); ou, entre
    oportunidades de score, estimativa pontual maior com limite inferior menor (incerteza).
    """
    pairs = table.merge(table, on="campanha_id", suffixes=("_acima", "_abaixo"))
    pairs = pairs[pairs["prioridade_acima"] < pairs["prioridade_abaixo"]]
    worse_class = pairs["classe_acima"].map(CLASS_ORDER) < pairs["classe_abaixo"].map(CLASS_ORDER)
    both_c = (pairs["classe_acima"] == "C") & (pairs["classe_abaixo"] == "C")
    rules = {
        "classe × magnitude": worse_class & (pairs["efeito_abaixo"] > pairs["efeito_acima"]),
        "score × associação": pairs["diferenca_score_abaixo"] > pairs["diferenca_score_acima"],
        "impacto × suporte": (
            pairs["risco_acima"].map(RISK_ORDER) > pairs["risco_abaixo"].map(RISK_ORDER)
        ),
        "incerteza na classe C": both_c
        & (pairs["score_modelo_abaixo"] > pairs["score_modelo_acima"]),
    }
    rows = []
    for name, mask in rules.items():
        hits = pairs[mask]
        common = Counter(zip(hits["acao_acima"], hits["acao_abaixo"], strict=True)).most_common(1)
        rows.append(
            {
                "conflito": name,
                "pares": len(hits),
                "campanhas": hits["campanha_id"].nunique(),
                "par_mais_comum": f"{common[0][0][0]} antes de {common[0][0][1]}"
                if common
                else None,
                "pares_mais_comum": common[0][1] if common else 0,
            }
        )
    return pd.DataFrame(rows).set_index("conflito")


# --------------------------------------------------------------------------- #
# Verificação de ponta a ponta
# --------------------------------------------------------------------------- #
_HEADER = re.compile(r"^(\d+)\. (.+)  \[classe ([ABC]): ")
_IMPACT = "   Impacto estimado: "


def audit(
    engine: RecommendationEngine, campaigns: pd.DataFrame, *, metric: str | None = None
) -> pd.DataFrame:
    """Roda o engine nas campanhas e confere a coerência das saídas, de ponta a ponta.

    Uma linha por verificação, com os itens examinados e as falhas. Além das regras de cada
    relatório, roda tudo de novo em ordem inversa (determinismo) e só com os campos de
    pré-publicação (as métricas observadas não entram no cálculo). ``metric`` troca a métrica
    de todas as campanhas, como na chamada do engine.
    """
    reports = engine.recommend_all(campaigns, metric=metric)
    results: dict[str, list[int]] = {}

    def check(name: str, ok: bool) -> None:
        items, failures = results.get(name, (0, 0))
        results[name] = [items + 1, failures + (not ok)]

    for report in reports:
        recs = report.recommendations
        check("JSON, tabela e texto mostram as mesmas recomendações", _formats_agree(report))
        check("resumo coerente com a lista", _summary_agrees(report))
        check("resumo executivo igual às primeiras recomendações", _executive_agrees(report))
        check("métrica escolhida manualmente declarada", _metric_declared(report))
        evaluated = [r.action for r in recs] + [key for key, _ in report.not_recommended]
        check(
            "música avaliada só na retenção",
            report.metric == "completion_rate" or "musica" not in evaluated,
        )
        check("prioridades de 1 a n", [r.priority for r in recs] == list(range(1, len(recs) + 1)))
        check("no máximo uma edição por campo", len({r.field for r in recs}) == len(recs))
        check(
            "contexto da campanha nunca é editado", all(r.field not in CONTEXT_COLS for r in recs)
        )
        check(
            "nenhuma oportunidade só de score numa combinação sem histórico",
            report.combination_support > 0 or all(r.recommendation_class != "C" for r in recs),
        )
        for r in recs:
            check(
                "cada edição parte do valor atual da campanha",
                report.campaign[r.field] == r.current_value != r.suggested_value,
            )
            check(
                "score antes e depois entre 0 e 100",
                0
                <= min(r.score_before, r.score_after)
                <= max(r.score_before, r.score_after)
                <= 100,
            )
            check("classe coerente com a evidência", _class_agrees(r))
            check("números iguais aos das tabelas de evidência", _tables_agree(engine, r))
            check(
                "suporte com o mínimo nos dois lados",
                min(r.support.n_current, r.support.n_target) >= engine.min_support,
            )
            check("avisos coerentes com os campos", _warnings_agree(report, r, engine.min_support))
            check(
                "ressalvas curtas iguais aos avisos", _caveats_agree(report, r, engine.min_support)
            )
            if r.action == "musica":
                check("música só na retenção, +0,1 e até 0,67", _music_agrees(r))
    again = engine.recommend_all(campaigns.iloc[::-1], metric=metric)
    for report, repeated in zip(reports, reversed(again), strict=True):
        check(
            "mesma saída ao rodar de novo, em outra ordem", report.to_dict() == repeated.to_dict()
        )
    pre_flight = campaigns[[col for col in PRE_FLIGHT_INPUTS if col in campaigns]]
    blind_reports = engine.recommend_all(pre_flight, metric=metric)
    for report, blind in zip(reports, blind_reports, strict=True):
        check(
            "mesma saída sem as métricas observadas",
            _without_history(report) == _without_history(blind),
        )
    rows = [{"verificacao": k, "itens": items, "falhas": f} for k, (items, f) in results.items()]
    return pd.DataFrame(rows).set_index("verificacao")


def _formats_agree(report: RecommendationReport) -> bool:
    """JSON, tabela e texto: mesmas recomendações, ordem, classe, impacto e avisos."""
    recs = report.recommendations
    expected = [(r.priority, r.title, r.recommendation_class, r.estimated_impact) for r in recs]
    payload = report.to_dict()
    from_json = [
        (d["prioridade"], d["titulo"], d["classe"], d["impacto_estimado"])
        for d in payload["recomendacoes"]
    ]
    frame = report.to_frame()
    from_frame = (
        []
        if frame.empty
        else list(
            zip(
                frame.index,
                frame["recomendação"],
                frame["classe"],
                frame["impacto estimado"],
                strict=True,
            )
        )
    )
    caveats = [" · ".join(r.caveats) for r in recs]
    same_caveats = [" · ".join(d["ressalvas"]) for d in payload["recomendacoes"]] == caveats and (
        frame.empty or list(frame["ressalvas"]) == caveats
    )
    lines = report.to_text().splitlines()
    headers = [m for m in map(_HEADER.match, lines) if m]
    impacts = [line.removeprefix(_IMPACT) for line in lines if line.startswith(_IMPACT)]
    if len(headers) != len(impacts):
        return False
    from_text = [
        (int(m[1]), m[2], m[3], impact) for m, impact in zip(headers, impacts, strict=True)
    ]
    warnings = sum(len(r.warnings) for r in recs)
    return (
        expected == from_json == from_frame == from_text
        and same_caveats
        and (frame.empty or frame["avisos"].sum() == warnings)
        and sum(line.startswith("   Aviso: ") for line in lines) == warnings
        and [(d["acao"], d["motivo"]) for d in payload["nao_recomendadas"]]
        == list(report.not_recommended)
        and report.summary in lines
    )


def _summary_agrees(report: RecommendationReport) -> bool:
    n = len(report.recommendations)
    if n == 0:
        return report.summary.startswith("Nenhuma edição do catálogo tem base")
    return report.summary.startswith(f"{n} recomenda") and ("menos de três" in report.summary) == (
        n < 3
    )


def _class_agrees(r: Recommendation) -> bool:
    """A classe segue a regra: A robusta e estável; B indício ou robusta e sensível; C o resto."""
    a = r.association
    if r.recommendation_class == "A":
        return a is not None and a.effect > 0 and a.evidence == "robusto" and a.stable
    if r.recommendation_class == "B":
        sensitive = a is not None and a.evidence == "robusto" and not a.stable
        return a is not None and a.effect > 0 and (a.evidence == "indício" or sensitive)
    inconclusive = a is None or a.evidence == "inconclusivo"
    return inconclusive and r.model_change[1] > 0 and r.score_change > 0


def _tables_agree(engine: RecommendationEngine, r: Recommendation) -> bool:
    a = r.association
    row = engine.performance.loc[(r.action, r.metric)]
    association = a is None or (a.effect, a.low, a.high, a.evidence, a.stable) == (
        row["efeito"],
        row["ic_inferior"],
        row["ic_superior"],
        row["evidencia"],
        bool(row["estavel"]),
    )
    model = engine.score_differences.loc[r.action, ["diferenca", "ic_inferior", "ic_superior"]]
    return association and r.model_change == tuple(model)


# Início de cada aviso e o rótulo curto que ele recebe no resumo executivo. O de pouca base
# depende do risco do suporte (``SUPPORT_CAVEATS``).
_WARNING_CAVEATS = {
    "Extrapolação": "extrapolação",
    "Sem associação mensurável": "sem associação mensurável",
    "A associação com": "estimativa desfavorável e inconclusiva",
    "A classe depende da análise": "classe sensível à análise",
    "Associada a": "desfavorável em outra métrica",
    "Ganho de score limítrofe": "ganho de score limítrofe",
    "O score previsto cai": "score previsto cai",
    "A previsão encosta no limite": "score no limite da escala",
    "Sem evidência suficiente de ganho": "sem evidência na métrica",
}


def _expected_warnings(
    report: RecommendationReport, r: Recommendation, minimum: int
) -> dict[str, bool]:
    """Quais avisos a recomendação deve ter, pelos campos estruturados, não pelo texto."""
    a, change = r.association, r.score_change
    return {
        "Extrapolação": report.combination_support < minimum,
        "Sem associação mensurável": a is None,
        "A associação com": _unfavorable(a),
        "A classe depende da análise": a is not None and a.sensitivity is not None,
        "Associada a": any(other.effect < 0 for other in r.other_associations),
        "Ganho de score limítrofe": 0 < r.model_change[1] < BORDERLINE_SCORE,
        "Pouca base no contexto": r.support.risk != "baixo",
        "O score previsto cai": change < 0,
        "A previsão encosta no limite": abs(change - r.model_change[0]) > 0.05,
        "Sem evidência suficiente de ganho": r.recommendation_class == "C",
    }


def _warnings_agree(report: RecommendationReport, r: Recommendation, minimum: int) -> bool:
    """Cada aviso aparece exatamente quando o campo correspondente pede."""
    expected = _expected_warnings(report, r, minimum)
    present = {prefix: any(w.startswith(prefix) for w in r.warnings) for prefix in expected}
    scenario_first = not expected["Extrapolação"] or r.warnings[0].startswith("Extrapolação")
    return scenario_first and present == expected


def _caveats_agree(report: RecommendationReport, r: Recommendation, minimum: int) -> bool:
    """As ressalvas curtas são os rótulos dos avisos esperados, na ordem de precedência."""
    expected = _expected_warnings(report, r, minimum)
    labels = {
        SUPPORT_CAVEATS[r.support.risk]
        if prefix == "Pouca base no contexto"
        else _WARNING_CAVEATS[prefix]
        for prefix, present in expected.items()
        if present
    }
    return r.caveats == (tuple(label for label in CAVEATS if label in labels) or (NO_CAVEAT,))


def _executive_agrees(report: RecommendationReport) -> bool:
    """O resumo executivo mostra as primeiras recomendações, iguais no texto e no JSON."""
    recs, payload = report.recommendations, report.to_dict()
    top = recs[:EXECUTIVE_SIZE]
    fields = ["prioridade", "titulo", "classe", "impacto_estimado", "ressalvas"]
    same_json = len(payload["principais"]) == len(top) and all(
        [entry[f] for f in fields] == [detail[f] for f in fields]
        for entry, detail in zip(payload["principais"], payload["recomendacoes"], strict=False)
    )
    counts = Counter(r.recommendation_class for r in recs)
    composition = payload["composicao"]
    same_counts = (
        composition["recomendacoes"] == len(recs)
        and composition["com_evidencia_na_metrica"] == counts["A"] + counts["B"]
        and composition["hipoteses_de_score"] == counts["C"]
    )
    lines = report.to_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Principais recomendações")]
    block = [
        line
        for r in top
        for line in (
            f"   {r.priority}. {r.title} · classe {r.recommendation_class}",
            f"      {r.headline}",
            f"      Ressalvas: {' · '.join(r.caveats)}",
        )
    ]
    if not top:
        same_text = not starts
    else:
        same_text = len(starts) == 1 and lines[starts[0] + 1 : starts[0] + 1 + len(block)] == block
    return same_json and same_counts and same_text


def _metric_declared(report: RecommendationReport) -> bool:
    """Com a métrica trocada, a saída diz isso e nunca chama a métrica de "do objetivo"."""
    manual = report.metric != OBJECTIVE_METRICS[report.objective]
    payload = report.to_dict()
    text = report.to_text() + json.dumps(payload, ensure_ascii=False)
    wrong_name = "métrica do objetivo" if manual else "métrica selecionada"
    return (
        payload["metrica_selecionada_manualmente"] == manual
        and payload["metrica_principal_do_objetivo"] == OBJECTIVE_METRICS[report.objective]
        and ("selecionada manualmente" in text) == manual
        and wrong_name not in text
    )


def _music_agrees(r: Recommendation) -> bool:
    """A música só aparece na retenção, soma 0,1 ao valor exato e não passa de 0,67."""
    return (
        r.metric == "completion_rate"
        and math.isclose(r.suggested_value - r.current_value, MUSIC_STEP, abs_tol=1e-9)
        and r.suggested_value <= MUSIC_MAX_TARGET + 1e-9
    )


def _unfavorable(a: Association | None) -> bool:
    """Associação inconclusiva com estimativa desfavorável na métrica do objetivo."""
    return a is not None and a.evidence == "inconclusivo" and a.effect < 0


def _without_history(report: RecommendationReport) -> dict:
    payload = report.to_dict()
    return {k: v for k, v in payload.items() if k not in ("campanha_id", "observado")}


def _label(report: RecommendationReport, position: int) -> str:
    """O campaign_id ou, numa campanha nova, um rótulo pela posição."""
    return report.campaign_id or f"nova {position + 1}"
