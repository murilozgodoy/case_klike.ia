"""Recursos carregados uma única vez por processo e compartilhados entre as requisições.

O preditor vem do arquivo gerado por ``python -m klike.train``. As evidências e a avaliação
do engine são lidas dos relatórios do projeto. Nada é treinado nem recalculado por
requisição.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import pandas as pd
from fastapi import Depends, Request

from app.services.campaigns import build_catalog
from app.services.metadata import build_metadata
from klike.config import ID_COL, MODEL_PATH
from klike.data import load_clean
from klike.engine import RecommendationEngine
from klike.engine_evaluation import EVALUATION_PATH
from klike.evidence import load_performance_evidence, load_score_differences
from klike.predictor import KlikeScorePredictor, train_final_model


@dataclass(frozen=True, eq=False)
class Resources:
    engine: RecommendationEngine
    history: pd.DataFrame  # base limpa da Parte 1, indexada por campaign_id
    catalog: pd.DataFrame  # uma linha por campanha, para a listagem
    metadata: dict
    model_source: str


def load_resources() -> Resources:
    """Carrega o modelo salvo ou, sem arquivo, treina a especificação congelada uma vez."""
    if MODEL_PATH.exists():
        predictor = KlikeScorePredictor.load(MODEL_PATH)
        source = "arquivo gerado por python -m klike.train"
    else:
        predictor = train_final_model()
        source = "especificação congelada, treinada na inicialização"
    engine = RecommendationEngine(predictor, load_performance_evidence(), load_score_differences())
    history = load_clean().set_index(ID_COL, drop=False)
    evaluation = pd.read_csv(EVALUATION_PATH)
    catalog = build_catalog(history, predictor, evaluation)
    return Resources(
        engine=engine,
        history=history,
        catalog=catalog,
        metadata=build_metadata(catalog, evaluation),
        model_source=source,
    )


def get_resources(request: Request) -> Resources:
    return request.app.state.resources


ResourcesDep = Annotated[Resources, Depends(get_resources)]
