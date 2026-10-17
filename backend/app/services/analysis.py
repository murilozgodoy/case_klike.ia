"""Análise de campanhas e comparação de cenários, sempre pelo engine do pacote ``klike``.

O relatório é o ``RecommendationReport.to_dict()`` do engine, sem nenhuma remontagem. A
comparação de cenários roda o mesmo engine na campanha original e na versão com uma única
edição do criativo, e só atribui evidência de performance quando a edição é exatamente uma
ação do catálogo.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from app.services.campaigns import history_row, pre_flight
from app.services.errors import InvalidInputError
from app.services.serialization import jsonable
from klike.config import COLUMN_LABELS, CONTEXT_COLS, CREATIVE_COLS
from klike.engine import ACTIONS, RecommendationEngine, RecommendationReport
from klike.evidence import PERFORMANCE_METRICS
from klike.modeling import PRE_FLIGHT_INPUTS

SCENARIO_NOTE = (
    "Esta comparação apresenta a diferença estimada pelo modelo de Klike Score. Não representa "
    "o efeito causal de editar o vídeo nem uma previsão de aumento de conversões, e não é um "
    "teste A/B."
)


def _recommend(engine: RecommendationEngine, campaign) -> RecommendationReport:
    try:
        return engine.recommend(campaign)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error


def analyze_history(engine: RecommendationEngine, history, campaign_id: str) -> dict:
    return jsonable(_recommend(engine, history_row(history, campaign_id)).to_dict())


def analyze_new(engine: RecommendationEngine, campaign: Mapping[str, Any]) -> dict:
    return jsonable(_recommend(engine, dict(campaign)).to_dict())


def compare_scenario(
    engine: RecommendationEngine,
    history,
    *,
    field: str,
    value: Any,
    campaign_id: str | None = None,
    campaign: Mapping[str, Any] | None = None,
) -> dict:
    """A campanha original e a mesma campanha com uma única edição do criativo.

    Numa campanha histórica, a versão modificada recebe só os 13 atributos de
    pré-publicação: as métricas observadas pertencem à campanha real, não à hipotética.
    """
    if field in CONTEXT_COLS:
        raise InvalidInputError(
            f"{field} é contexto da campanha: a comparação só altera atributos do criativo."
        )
    if field not in CREATIVE_COLS:
        raise InvalidInputError(f"{field}: atributo desconhecido.")
    if value is None:
        raise InvalidInputError(f"{field}: a alteração precisa de um valor.")

    if campaign_id is not None:
        row = history_row(history, campaign_id)
        base, attributes = row, pre_flight(row.iloc[0])
    else:
        base = attributes = {col: campaign.get(col) for col in PRE_FLIGHT_INPUTS}
    original = _recommend(engine, base)
    modified = _recommend(engine, {**attributes, field: value})
    before, after = original.campaign[field], modified.campaign[field]
    if before == after:
        raise InvalidInputError(f"{field}: o valor novo é igual ao atual.")

    return {
        "alteracao": {
            "campo": field,
            "rotulo": COLUMN_LABELS[field],
            "valor_original": jsonable(before),
            "valor_novo": jsonable(after),
        },
        "score": {
            "original": original.predicted_score,
            "modificado": modified.predicted_score,
            "diferenca": modified.predicted_score - original.predicted_score,
        },
        "evidencia": catalog_evidence(original, field, after),
        "nota": SCENARIO_NOTE,
        "original": jsonable(original.to_dict()),
        "modificado": jsonable(modified.to_dict()),
    }


def catalog_evidence(report: RecommendationReport, field: str, value: Any) -> dict:
    """A evidência do engine para a edição, se ela for exatamente uma ação do catálogo.

    Usa as próprias ações do engine para reconhecer a edição. Uma edição livre, como trocar
    para um formato que o catálogo não sugere ou mudar a música em outro valor, não herda a
    evidência de nenhuma ação parecida.
    """
    campaign = report.campaign
    for action in ACTIONS:
        if action.field != field or not action.applies_to(campaign):
            continue
        target = action.for_campaign(campaign).target
        if not _same(target, value):
            continue
        recommended = next((r for r in report.recommendations if r.action == action.key), None)
        if recommended is not None:
            return {
                "situacao": "recomendada",
                "acao": action.key,
                "recomendacao": jsonable(recommended.to_dict()),
            }
        reason = dict(report.not_recommended).get(action.key)
        if reason is not None:
            return {"situacao": "avaliada_nao_recomendada", "acao": action.key, "motivo": reason}
        metrics = ", ".join(PERFORMANCE_METRICS[m] for m in action.metrics or ())
        return {
            "situacao": "fora_do_escopo",
            "acao": action.key,
            "motivo": (
                f"Edição do catálogo avaliada só em {metrics}; a métrica desta campanha é "
                f"{report.metric_name}."
            ),
        }
    return {
        "situacao": "sem_evidencia",
        "acao": None,
        "motivo": (
            "Esta alteração não é uma edição do catálogo estudada nas evidências: o modelo prevê "
            "o score, mas não há associação histórica com performance estimada para ela."
        ),
    }


def _same(target: Any, value: Any) -> bool:
    if isinstance(target, float) and isinstance(value, int | float) and not isinstance(value, bool):
        return math.isclose(target, value, abs_tol=1e-9)
    return target == value
