"""Consulta das campanhas históricas: listagem com filtros e detalhe."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.errors import InvalidInputError, NotFoundError
from app.services.serialization import jsonable, record
from klike.config import CATEGORY_ORDERS, ID_COL, PERFORMANCE_COLS, TARGET
from klike.modeling import PRE_FLIGHT_INPUTS
from klike.predictor import KlikeScorePredictor

CLASSES = ["A", "B", "C"]
SUMMARY_COLUMNS = [
    ID_COL,
    "platform",
    "category",
    "objective",
    "format",
    "score_observado",
    "score_previsto",
    "recomendacoes",
    *CLASSES,
]


def build_catalog(
    history: pd.DataFrame, predictor: KlikeScorePredictor, evaluation: pd.DataFrame
) -> pd.DataFrame:
    """Uma linha por campanha: contexto, score observado e previsto e recomendações por classe.

    O score previsto vem do preditor; as contagens vêm de ``engine_recommendations.csv``,
    a avaliação do engine nas 500 campanhas, que um teste do projeto mantém igual ao engine.
    """
    counts = (
        evaluation.groupby("campanha_id")["classe"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(index=history.index, columns=CLASSES, fill_value=0)
    )
    catalog = history[[ID_COL, "platform", "category", "objective", "format"]].copy()
    catalog["score_observado"] = history[TARGET]
    catalog["score_previsto"] = predictor.predict(history[PRE_FLIGHT_INPUTS]).to_numpy()
    catalog[CLASSES] = counts[CLASSES].astype(int)
    catalog["recomendacoes"] = counts.sum(axis=1).astype(int)
    return catalog[SUMMARY_COLUMNS].sort_index()


def list_campaigns(
    catalog: pd.DataFrame,
    *,
    platform: str | None = None,
    objective: str | None = None,
    category: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    selected = catalog
    for column, value in [("platform", platform), ("objective", objective), ("category", category)]:
        if value is None:
            continue
        if value not in CATEGORY_ORDERS[column]:
            accepted = ", ".join(CATEGORY_ORDERS[column])
            raise InvalidInputError(f"{column}: valor desconhecido {value!r} (aceitos: {accepted})")
        selected = selected[selected[column] == value]
    if search:
        selected = selected[selected[ID_COL].str.contains(search.strip(), case=False, regex=False)]
    start = (page - 1) * page_size
    rows = selected.iloc[start : start + page_size]
    return {
        "total": len(selected),
        "pagina": page,
        "por_pagina": page_size,
        "campanhas": [record(row, SUMMARY_COLUMNS) for _, row in rows.iterrows()],
    }


def history_row(history: pd.DataFrame, campaign_id: str) -> pd.DataFrame:
    """A campanha histórica como DataFrame de uma linha, como o engine a recebe."""
    if campaign_id not in history.index:
        raise NotFoundError(f"A campanha {campaign_id!r} não está no histórico.")
    return history.loc[[campaign_id]]


def campaign_detail(history: pd.DataFrame, catalog: pd.DataFrame, campaign_id: str) -> dict:
    """Atributos de pré-publicação, métricas observadas e o resumo da listagem."""
    row = history_row(history, campaign_id).iloc[0]
    return {
        "campaign_id": campaign_id,
        "atributos": pre_flight(row),
        "observado": record(row, [TARGET, *PERFORMANCE_COLS]),
        "resumo": record(catalog.loc[campaign_id], SUMMARY_COLUMNS),
    }


def pre_flight(row: pd.Series) -> dict[str, Any]:
    """Os 13 atributos de pré-publicação, com ausentes como nulos."""
    return jsonable({col: None if pd.isna(row[col]) else row[col] for col in PRE_FLIGHT_INPUTS})
