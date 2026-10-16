"""Testes da documentação: os trechos de saída no README vêm da saída real do engine."""

import pytest

from klike.config import ROOT_DIR
from klike.data import load_clean
from klike.engine import RecommendationEngine
from klike.evidence import load_performance_evidence, load_score_differences
from klike.predictor import train_final_model


@pytest.fixture(scope="module")
def engine():
    return RecommendationEngine(
        train_final_model(), load_performance_evidence(), load_score_differences()
    )


@pytest.fixture(scope="module")
def history():
    return load_clean().set_index("campaign_id", drop=False)


def test_readme_excerpt_is_taken_from_the_real_output(engine, history):
    # O trecho da KLK-0024 no README é uma seleção de linhas da saída real, na mesma ordem.
    lines = (ROOT_DIR / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Trecho da saída para a KLK"))
    opening = lines.index("```", start)
    excerpt = [line for line in lines[opening + 1 : lines.index("```", opening + 1)] if line]
    output = iter(engine.recommend(history.loc[["KLK-0024"]]).to_text().splitlines())
    assert len(excerpt) > 10
    assert all(line in output for line in excerpt)  # cada linha, na ordem em que aparece
