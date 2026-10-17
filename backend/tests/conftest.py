"""Fixtures da API: um cliente com o ciclo de vida completo e um engine independente.

O engine de referência é montado direto do pacote, com a especificação congelada treinada
na hora: as respostas da API precisam coincidir com ele, não com a própria instância da API.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

from klike.data import load_clean  # noqa: E402
from klike.engine import RecommendationEngine  # noqa: E402
from klike.evidence import load_performance_evidence, load_score_differences  # noqa: E402
from klike.predictor import train_final_model  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def reference():
    """Engine do pacote, independente do que a API carregou."""
    return RecommendationEngine(
        train_final_model(), load_performance_evidence(), load_score_differences()
    )


@pytest.fixture(scope="session")
def history():
    return load_clean().set_index("campaign_id", drop=False)
