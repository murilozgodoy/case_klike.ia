"""Testes da API: as respostas precisam coincidir com a execução direta do pacote ``klike``.

Não basta o status 200: cada teste compara o conteúdo com o engine de referência, ou confere
a regra que a API promete (uma edição por vez, só atributos do criativo, nenhuma métrica
observada no cenário hipotético, JSON estrito).
"""

import json

import pandas as pd
import pytest
from app.main import create_app
from app.services.serialization import jsonable
from fastapi.testclient import TestClient

from klike.config import CATEGORY_ORDERS, CONTEXT_COLS, CREATIVE_COLS
from klike.engine_evaluation import EVALUATION_PATH
from klike.modeling import OPTIONAL_INPUTS, PRE_FLIGHT_INPUTS
from klike.predictor import CERTIFIED_METRICS

NEW = {
    "platform": "TikTok",
    "category": "E-commerce",
    "objective": "conversions",
    "target_audience_age": "18-24",
    "is_retargeting": False,
    "video_duration_s": 22,
    "format": "quadrado",
    "has_subtitle": False,
    "has_cta": False,
    "has_hook": False,
    "has_face": True,
    "text_density": "high",
    "music_voice_ratio": 0.35,
}
AWARENESS = NEW | {
    "objective": "awareness",
    "format": "vertical",
    "has_subtitle": True,
    "has_cta": True,
    "has_hook": True,
    "text_density": "medium",
}


def strict_json(response) -> dict:
    """O corpo como JSON, recusando NaN e infinitos, que não são JSON válido."""

    def refuse(constant):
        raise AssertionError(f"valor não permitido em JSON: {constant}")

    return json.loads(response.text, parse_constant=refuse)


def expected(report) -> dict:
    """O relatório do pacote, depois da mesma ida e volta por JSON que a API faz."""
    return json.loads(json.dumps(jsonable(report.to_dict()), ensure_ascii=False))


def actions(report: dict) -> list[str]:
    return [r["acao"] for r in report["recomendacoes"]]


# --------------------------------------------------------------------------- #
# Estado e metadados
# --------------------------------------------------------------------------- #
def test_health_reports_the_loaded_engine(client):
    body = strict_json(client.get("/api/health"))
    assert body["status"] == "ok"
    assert body["engine_carregado"] is True
    assert body["campanhas"] == 500


def test_metadata_reads_fields_and_options_from_the_package(client):
    body = strict_json(client.get("/api/metadata"))
    fields = {f["campo"]: f for f in body["campos"]}
    assert list(fields) == PRE_FLIGHT_INPUTS
    assert body["atributos_contexto"] == CONTEXT_COLS
    assert body["atributos_criativo"] == CREATIVE_COLS
    for col, options in CATEGORY_ORDERS.items():
        assert fields[col]["opcoes"] == options
    assert {c for c, f in fields.items() if not f["obrigatorio"]} == set(OPTIONAL_INPUTS)


def test_indicators_come_from_the_existing_results(client):
    indicators = strict_json(client.get("/api/metadata"))["indicadores"]
    assert indicators["campanhas"] == 500
    assert indicators["modelo"]["mae"] == CERTIFIED_METRICS["mae"] == 6.43
    # Recontagem direta da avaliação salva do engine.
    evaluation = pd.read_csv(EVALUATION_PATH)
    per_campaign = evaluation.groupby("campanha_id").size()
    with_evidence = evaluation[evaluation["classe"].isin(["A", "B"])].groupby("campanha_id").size()
    engine = indicators["avaliacao_engine"]
    assert engine["recomendacoes"] == len(evaluation) == 1257
    assert engine["classes"] == evaluation["classe"].value_counts().to_dict()
    assert engine["campanhas_tres_ou_mais"] == (per_campaign >= 3).sum() == 247
    assert engine["campanhas_sem_recomendacao"] == 500 - len(per_campaign) == 20
    assert engine["campanhas_com_evidencia"] == len(with_evidence) == 395
    assert engine["campanhas_tres_ou_mais_com_evidencia"] == (with_evidence >= 3).sum() == 34
    assert engine["parcela_tres_ou_mais"] == pytest.approx(247 / 500)


def test_guided_examples_show_what_they_promise(client):
    examples = {e["campaign_id"]: e for e in strict_json(client.get("/api/metadata"))["exemplos"]}
    full = strict_json(client.get("/api/campaigns/KLK-0024/analysis"))
    assert {r["classe"] for r in full["recomendacoes"]} == {"A", "B", "C"}
    empty = strict_json(client.get("/api/campaigns/KLK-0291/analysis"))
    assert empty["recomendacoes"] == []
    assert strict_json(client.get("/api/campaigns/KLK-0291"))["observado"]["klike_score"] == 89.1
    music = strict_json(client.get("/api/campaigns/KLK-0001/analysis"))
    rec = next(r for r in music["recomendacoes"] if r["acao"] == "musica")
    assert rec["diferenca_score"] < 0 and "score previsto cai" in rec["ressalvas"]
    assert set(examples) == {"KLK-0024", "KLK-0291", "KLK-0001"}


# --------------------------------------------------------------------------- #
# Campanhas históricas
# --------------------------------------------------------------------------- #
def test_listing_filters_searches_and_paginates(client, history, reference):
    body = strict_json(
        client.get("/api/campaigns", params={"platform": "Meta", "objective": "conversions"})
    )
    meta = history[(history["platform"] == "Meta") & (history["objective"] == "conversions")]
    assert body["total"] == len(meta) == 84
    assert len(body["campanhas"]) == 20
    assert all(c["platform"] == "Meta" for c in body["campanhas"])
    found = strict_json(client.get("/api/campaigns", params={"busca": "0024"}))["campanhas"]
    assert [c["campaign_id"] for c in found] == ["KLK-0024"]
    # Score previsto e contagens iguais ao pacote.
    summary = found[0]
    report = reference.recommend(history.loc[["KLK-0024"]])
    assert summary["score_previsto"] == pytest.approx(report.predicted_score)
    assert summary["recomendacoes"] == len(report.recommendations)
    last = strict_json(client.get("/api/campaigns", params={"pagina": 25, "por_pagina": 20}))
    assert (last["total"], len(last["campanhas"])) == (500, 20)
    assert client.get("/api/campaigns", params={"platform": "YouTube"}).status_code == 422


def test_detail_separates_attributes_from_observed_metrics(client, history):
    body = strict_json(client.get("/api/campaigns/KLK-0024"))
    assert list(body["atributos"]) == PRE_FLIGHT_INPUTS
    assert body["observado"]["klike_score"] == history.loc["KLK-0024", "klike_score"]
    missing = client.get("/api/campaigns/KLK-9999")
    assert missing.status_code == 404
    assert "KLK-9999" in missing.json()["detail"]


def test_historical_analysis_matches_the_package(client, history, reference):
    body = strict_json(client.get("/api/campaigns/KLK-0024/analysis"))
    assert body == expected(reference.recommend(history.loc[["KLK-0024"]]))
    assert actions(body) == ["hook", "rosto", "texto_alto_medio", "formato_meta_horizontal"]


def test_campaign_without_recommendations_is_reported_honestly(client, history, reference):
    body = strict_json(client.get("/api/campaigns/KLK-0291/analysis"))
    assert body == expected(reference.recommend(history.loc[["KLK-0291"]]))
    assert body["recomendacoes"] == [] and body["principais"] == []
    assert body["resumo"].startswith("Nenhuma edição do catálogo tem base")


# --------------------------------------------------------------------------- #
# Campanha nova
# --------------------------------------------------------------------------- #
def test_new_campaign_matches_the_package(client, reference):
    body = strict_json(client.post("/api/analyze", json=NEW))
    assert body == expected(reference.recommend(NEW))
    assert body["campanha_id"] is None and body["observado"] is None


def test_missing_required_field_is_rejected(client):
    response = client.post("/api/analyze", json={k: v for k, v in NEW.items() if k != "has_hook"})
    assert response.status_code == 422
    assert "has_hook: campo obrigatório" in response.json()["detail"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"platform": "YouTube"}, "platform: valor desconhecido"),
        ({"text_density": "enorme"}, "text_density: valor desconhecido"),
        ({"music_voice_ratio": 1.5}, "music_voice_ratio: esperado um número entre 0 e 1"),
        ({"video_duration_s": -3}, "video_duration_s: esperado um número maior que zero"),
        ({"has_hook": "talvez"}, "has_hook: esperado verdadeiro ou falso"),
        ({"music_ratio": 0.3}, "music_ratio: campo desconhecido"),
    ],
)
def test_invalid_values_are_rejected_with_clear_messages(client, change, message):
    response = client.post("/api/analyze", json=NEW | change)
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_optional_fields_may_be_missing(client, reference):
    partial = {k: v for k, v in NEW.items() if k not in OPTIONAL_INPUTS}
    body = strict_json(client.post("/api/analyze", json=partial))
    assert body == expected(reference.recommend(partial))
    assert "legenda" not in actions(body)  # sem marcação não vira "sem legenda"
    assert any("Duração não informada" in note for note in body["avisos"])


def test_music_recommendation_keeps_the_falling_score_visible(client, reference):
    body = strict_json(client.post("/api/analyze", json=AWARENESS))
    assert body == expected(reference.recommend(AWARENESS))
    music = next(r for r in body["recomendacoes"] if r["acao"] == "musica")
    assert music["classe"] == "A" and music["diferenca_score"] < 0
    assert music["ressalvas"][0] == "score previsto cai"


def test_unseen_combination_carries_the_extrapolation_warning(client, reference):
    unseen = NEW | {"format": "horizontal"}  # TikTok horizontal: nenhuma campanha no histórico
    body = strict_json(client.post("/api/analyze", json=unseen))
    assert body == expected(reference.recommend(unseen))
    assert body["campanhas_na_mesma_combinacao"] == 0
    assert any("nenhuma campanha no histórico" in note for note in body["avisos"])
    assert all(r["avisos"][0].startswith("Extrapolação") for r in body["recomendacoes"])


# --------------------------------------------------------------------------- #
# Laboratório de cenários
# --------------------------------------------------------------------------- #
def test_scenario_compares_one_creative_edit(client, history, reference):
    body = strict_json(
        client.post(
            "/api/scenarios/compare",
            json={"campaign_id": "KLK-0024", "changes": {"has_hook": True}},
        )
    )
    base = {col: history.loc["KLK-0024", col] for col in PRE_FLIGHT_INPUTS}
    scores = reference.predictor.predict([base, base | {"has_hook": True}])
    assert body["score"]["original"] == pytest.approx(scores.iloc[0])
    assert body["score"]["modificado"] == pytest.approx(scores.iloc[1])
    assert body["score"]["diferenca"] == pytest.approx(scores.iloc[1] - scores.iloc[0])
    assert body["alteracao"] == {
        "campo": "has_hook",
        "rotulo": "Hook nos 3 primeiros segundos",
        "valor_original": False,
        "valor_novo": True,
    }
    # O cenário original é o mesmo relatório da análise; a edição é a recomendação do engine.
    assert body["original"] == strict_json(client.get("/api/campaigns/KLK-0024/analysis"))
    assert body["evidencia"]["situacao"] == "recomendada"
    assert body["evidencia"]["recomendacao"] == body["original"]["recomendacoes"][0]
    assert "não é um teste A/B" in body["nota"]


def test_hypothetical_scenario_carries_no_observed_performance(client):
    body = strict_json(
        client.post(
            "/api/scenarios/compare",
            json={"campaign_id": "KLK-0024", "changes": {"has_face": True}},
        )
    )
    assert body["original"]["observado"] is not None
    assert body["modificado"]["observado"] is None
    assert body["modificado"]["campanha_id"] is None
    assert "hook" in actions(body["modificado"])  # o engine roda de novo na versão hipotética
    assert "rosto" not in actions(body["modificado"])


def test_free_edits_do_not_inherit_evidence(client):
    def evidence(campaign, changes):
        payload = {"campaign": campaign, "changes": changes}
        return strict_json(client.post("/api/scenarios/compare", json=payload))["evidencia"]

    # Formato fora do catálogo da plataforma, e música diferente de +0,1: sem evidência.
    assert evidence(NEW | {"format": "vertical"}, {"format": "quadrado"})["situacao"] == (
        "sem_evidencia"
    )
    assert evidence(AWARENESS, {"music_voice_ratio": 0.55})["situacao"] == "sem_evidencia"
    # A edição exata do catálogo herda a recomendação do engine.
    exact = evidence(AWARENESS, {"music_voice_ratio": 0.45})
    assert (exact["situacao"], exact["acao"]) == ("recomendada", "musica")
    # Fora da retenção, a música é do catálogo, mas não é avaliada.
    assert evidence(NEW, {"music_voice_ratio": 0.45})["situacao"] == "fora_do_escopo"


def test_scenario_into_an_unseen_combination_keeps_the_warning(client):
    body = strict_json(
        client.post(
            "/api/scenarios/compare", json={"campaign": NEW, "changes": {"format": "horizontal"}}
        )
    )
    assert body["modificado"]["campanhas_na_mesma_combinacao"] == 0
    assert any("nenhuma campanha no histórico" in n for n in body["modificado"]["avisos"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"campaign_id": "KLK-0024", "changes": {"has_hook": True, "has_face": True}},
            "uma alteração por vez",
        ),
        (
            {"campaign_id": "KLK-0024", "changes": {"platform": "TikTok"}},
            "platform é contexto da campanha",
        ),
        ({"campaign_id": "KLK-0024", "changes": {}}, "uma alteração por vez"),
        ({"campaign_id": "KLK-0024", "changes": {"has_hook": False}}, "igual ao atual"),
        ({"campaign_id": "KLK-0024", "changes": {"has_hook": None}}, "precisa de um valor"),
        (
            {"campaign_id": "KLK-0024", "changes": {"format": "triangular"}},
            "format: valor desconhecido",
        ),
        (
            {"campaign": NEW, "campaign_id": "KLK-0024", "changes": {"has_hook": True}},
            "um dos dois",
        ),
    ],
)
def test_scenario_rejects_invalid_edits(client, payload, message):
    response = client.post("/api/scenarios/compare", json=payload)
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_unknown_campaign_in_scenario_is_not_found(client):
    payload = {"campaign_id": "KLK-9999", "changes": {"has_hook": True}}
    assert client.post("/api/scenarios/compare", json=payload).status_code == 404


# --------------------------------------------------------------------------- #
# JSON estrito e erros internos
# --------------------------------------------------------------------------- #
def test_missing_values_become_null_not_nan(client, history):
    # KLK-0044 tem legenda e música/voz ausentes no histórico.
    body = strict_json(client.get("/api/campaigns/KLK-0044"))
    assert body["atributos"]["has_subtitle"] is None
    assert body["atributos"]["music_voice_ratio"] is None
    assert jsonable({"a": float("nan"), "b": [float("inf"), 1.0]}) == {"a": None, "b": [None, 1.0]}
    strict_json(client.get("/api/campaigns/KLK-0044/analysis"))


def test_internal_errors_do_not_leak_details(monkeypatch):
    with TestClient(create_app(), raise_server_exceptions=False) as local:
        engine = local.app.state.resources.engine

        def broken(*args, **kwargs):
            raise RuntimeError("detalhe interno")

        monkeypatch.setattr(engine, "recommend", broken)
        response = local.get("/api/campaigns/KLK-0024/analysis")
    assert response.status_code == 500
    assert response.json() == {"detail": "Erro interno ao processar a requisição."}
