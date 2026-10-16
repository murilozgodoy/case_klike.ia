"""Modelo final do Klike Score: treino, persistência e previsão para campanhas novas.

A especificação é a congelada na Parte 2: Ridge com a interação plataforma × formato e a
penalização da certificação. Depois da certificação, ela é reajustada nas 500 campanhas.
As métricas oficiais continuam sendo as da certificação, do modelo treinado nas 400
campanhas de desenvolvimento: o modelo reajustado não tem métrica própria, porque não sobra
nenhuma campanha para avaliá-lo de forma independente.

Uma única chamada valida a campanha, prepara as features, aplica o pipeline e limita o score
à escala de 0 a 100. Quem usa o modelo não precisa chamar ``prepare_features``::

    predictor = KlikeScorePredictor.load()
    predictor.predict({"platform": "TikTok", "category": "E-commerce", ...})

Para treinar e salvar de novo: ``python -m klike.train``.
"""

from __future__ import annotations

import hashlib
import platform
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from klike.config import MODEL_PATH, RAW_DATA_PATH, TARGET
from klike.data import load_clean
from klike.modeling import (
    OPTIONAL_INPUTS,
    PRE_FLIGHT_INPUTS,
    SELECTED_MODEL,
    SELECTED_PARAMS,
    SELECTED_SPEC,
    build_pipeline,
    prepare_features,
)
from klike.support import HistoricalSupport

FILE_FORMAT = 1  # muda quando a estrutura do arquivo salvo mudar
SCORE_RANGE = (0.0, 100.0)
RETRAIN_HINT = "Gere o arquivo de novo com: python -m klike.train"

# Métricas oficiais: a certificação da Parte 2 (notebook 03, seção 7), com o modelo treinado
# nas 400 campanhas de desenvolvimento e avaliado uma única vez nas 100 reservadas. Um teste
# refaz a certificação e confere estes valores.
CERTIFIED_METRICS = {
    "mae": 6.43,
    "mae_ic95": [5.55, 7.33],
    "rmse": 7.92,
    "r2": 0.70,
    "spearman": 0.83,
    "mae_dummy": 11.67,
}


@dataclass(eq=False)
class KlikeScorePredictor:
    """Modelo final pronto para uso: valida, prepara, prevê e limita o score de 0 a 100."""

    pipeline: Pipeline
    support: HistoricalSupport
    metadata: dict

    def predict(self, campaigns: pd.DataFrame | Mapping | Sequence[Mapping]) -> pd.Series:
        """Klike Score previsto para cada campanha, na escala de 0 a 100.

        Aceita um dicionário, uma lista de dicionários ou um DataFrame com os 13 campos de
        pré-publicação, em qualquer ordem. Duração, legenda e música/voz podem faltar ou vir
        vazias. Colunas do histórico, como métricas de performance, são ignoradas. Qualquer
        outro campo, categoria desconhecida ou valor inválido gera um erro que lista todos os
        problemas da entrada.
        """
        features = prepare_features(campaigns, strict=True)
        scores = np.clip(self.pipeline.predict(features), *SCORE_RANGE)
        return pd.Series(scores, index=features.index, name="klike_score_previsto")

    def save(self, path: str | Path = MODEL_PATH) -> Path:
        """Salva pipeline, suporte histórico e metadados num único arquivo joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "formato": FILE_FORMAT,
            "pipeline": self.pipeline,
            "suporte": self.support.profiles,
            "metadados": self.metadata,
        }
        joblib.dump(content, path)
        return path

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> KlikeScorePredictor:
        """Carrega um modelo salvo por ``save``.

        O joblib usa pickle, então só carregue arquivos gerados por você. Se a versão do
        scikit-learn mudou desde o treino, um aviso recomenda gerar o arquivo de novo.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {path}. {RETRAIN_HINT}")
        content = joblib.load(path)
        if not isinstance(content, dict) or content.get("formato") != FILE_FORMAT:
            raise ValueError(f"Arquivo de modelo em formato desconhecido. {RETRAIN_HINT}")
        trained_with = content["metadados"]["versoes"]["scikit-learn"]
        if trained_with != sklearn.__version__:
            warnings.warn(
                f"O modelo foi salvo com scikit-learn {trained_with}, e este ambiente tem "
                f"{sklearn.__version__}. {RETRAIN_HINT}",
                stacklevel=2,
            )
        return cls(
            pipeline=content["pipeline"],
            support=HistoricalSupport(content["suporte"]),
            metadata=content["metadados"],
        )


def train_final_model(data_path: str | Path = RAW_DATA_PATH) -> KlikeScorePredictor:
    """Reajusta a especificação congelada em todas as campanhas do histórico.

    Usa a limpeza da Parte 1 e a mesma preparação da validação. Não calcula métrica: as
    oficiais são as da certificação (``CERTIFIED_METRICS``).
    """
    data_path = Path(data_path)
    campaigns = load_clean(data_path)
    pipeline = build_pipeline(SELECTED_MODEL, SELECTED_SPEC, **SELECTED_PARAMS)
    pipeline.fit(prepare_features(campaigns), campaigns[TARGET])
    return KlikeScorePredictor(
        pipeline=pipeline,
        support=HistoricalSupport.from_campaigns(campaigns),
        metadata=_metadata(pipeline, len(campaigns), data_path),
    )


def _metadata(pipeline: Pipeline, n_campaigns: int, data_path: Path) -> dict:
    """O que identifica o modelo salvo, só com tipos simples (serializável em JSON)."""
    interactions = pipeline.named_steps["features"].named_transformers_["plataforma_formato"]
    return {
        "modelo": "Ridge com interação plataforma × formato (modelo A, pré-publicação)",
        "especificacao": {"algoritmo": SELECTED_MODEL, **asdict(SELECTED_SPEC), **SELECTED_PARAMS},
        "entradas": {
            "obrigatorias": [col for col in PRE_FLIGHT_INPUTS if col not in OPTIONAL_INPUTS],
            "opcionais": list(OPTIONAL_INPUTS),
        },
        "treino": {
            "campanhas": n_campaigns,
            "arquivo": data_path.name,
            "sha256": data_sha256(data_path),
            "interacoes_plataforma_formato": [f"{p} × {f}" for p, f in interactions.combinations_],
        },
        "metricas_oficiais": {
            "origem": (
                "certificação da Parte 2: treino nas 400 campanhas de desenvolvimento e uma "
                "única avaliação nas 100 reservadas"
            ),
            **CERTIFIED_METRICS,
            "observacao": "o modelo reajustado nas 500 campanhas não tem métrica própria",
        },
        "previsao": {"escala": list(SCORE_RANGE)},
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def data_sha256(path: str | Path) -> str:
    """Impressão digital do arquivo de dados, igual em qualquer sistema operacional.

    As quebras de linha são normalizadas antes: o git pode gravar o CSV com CRLF no
    Windows e com LF no Linux, sem que o conteúdo mude.
    """
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()
