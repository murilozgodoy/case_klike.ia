"""Testes da avaliação do recommendations engine.

Conferem que as tabelas descrevem os relatórios sem perder nada, que a ordem refeita pela
tabela é a do engine, que o funil respeita o escopo de métrica de cada edição, que a
auditoria aponta falhas de verdade e que a avaliação salva em ``reports/`` corresponde ao
engine atual.
"""

import dataclasses

import pandas as pd
import pytest

from klike.data import load_clean
from klike.engine import NO_CAVEAT, RecommendationEngine
from klike.engine_evaluation import (
    EVALUATION_PATH,
    action_funnel,
    audit,
    campaign_table,
    conflicts,
    coverage,
    engine_order,
    order_robustness,
    recommendation_table,
    reorder,
)
from klike.evidence import load_performance_evidence, load_score_differences
from klike.predictor import train_final_model

# Campanhas com situações diferentes: as três classes, nenhuma recomendação, só oportunidades
# de score, legenda sem marcação e o formato de cada plataforma.
SAMPLE = [
    "KLK-0024",  # Meta, conversions: A, A, B e C
    "KLK-0003",  # criativo completo: nenhuma
    "KLK-0013",  # TikTok quadrado: só C
    "KLK-0029",  # legenda sem marcação
    "KLK-0030",  # Meta horizontal, engagement: formato A em primeiro
    "KLK-0198",  # LinkedIn vertical: só o formato
    "KLK-0203",  # awareness: texto A, rosto B
]
NEW = {
    "platform": "TikTok",
    "category": "E-commerce",
    "objective": "conversions",
    "target_audience_age": "18-24",
    "is_retargeting": False,
    "video_duration_s": 22,
    "format": "horizontal",  # nenhuma campanha no histórico
    "has_subtitle": False,
    "has_cta": False,
    "has_hook": False,
    "has_face": True,
    "text_density": "high",
    "music_voice_ratio": 0.35,
}


@pytest.fixture(scope="module")
def engine():
    return RecommendationEngine(
        train_final_model(), load_performance_evidence(), load_score_differences()
    )


@pytest.fixture(scope="module")
def history():
    return load_clean().set_index("campaign_id", drop=False)


@pytest.fixture(scope="module")
def reports(engine, history):
    return engine.recommend_all(history.loc[SAMPLE])


def test_tables_describe_every_recommendation_and_campaign(engine, reports):
    table, campaigns = recommendation_table(reports), campaign_table(reports)
    assert len(table) == sum(len(report.recommendations) for report in reports)
    assert list(campaigns["campanha_id"]) == SAMPLE
    assert campaigns["recomendacoes"].eq(campaigns[["A", "B", "C"]].sum(axis=1)).all()
    text = table[(table["campanha_id"] == "KLK-0024") & (table["acao"] == "texto_alto_medio")]
    columns = ["prioridade", "classe", "evidencia", "estavel"]
    assert text[columns].iloc[0].tolist() == [3, "B", "robusto", False]
    assert table.loc[table["acao"] == "formato_tiktok_quadrado", "limitrofe"].all()
    assert coverage(campaigns).loc["todas", "nenhuma"] == pytest.approx(1 / len(SAMPLE))
    # Campanhas novas não se misturam: cada uma ganha um rótulo pela posição.
    new = campaign_table(engine.recommend_all(pd.DataFrame([NEW, NEW])))
    assert list(new["campanha_id"]) == ["nova 1", "nova 2"]


def test_the_table_reproduces_the_engine_order(reports):
    table = recommendation_table(reports)
    assert reorder(table, engine_order).eq(table["prioridade"]).all()


def test_alternative_orders_report_what_would_change(reports):
    summary, swaps = order_robustness(recommendation_table(reports))
    assert summary["campanhas"].eq((campaign_table(reports)["recomendacoes"] >= 2).sum()).all()
    # KLK-0030: o formato vem primeiro pela associação com engajamento; pelo score, seria o hook.
    swapped = set(zip(swaps["ordem"], swaps["de"], swaps["para"], strict=True))
    assert ("só score", "formato_meta_horizontal", "hook") in swapped
    # KLK-0024: o texto, robusto mas sensível, passaria o rosto se contasse como A.
    assert summary.loc["sensível como A", "ordem_muda"] >= 1


def test_conflicts_name_the_pairs_another_criterion_would_swap(reports):
    found = conflicts(recommendation_table(reports[:1]))  # KLK-0024
    for conflict in ["classe × magnitude", "score × associação"]:
        assert found.loc[conflict, "pares"] == 1
        assert found.loc[conflict, "par_mais_comum"] == "rosto antes de texto_alto_medio"


def test_funnel_accounts_for_every_campaign_and_edit(reports):
    funnel = action_funnel(reports)
    parts = funnel[["valor_ausente", "outro_valor", "avaliadas"]].sum(axis=1)
    assert funnel["campanhas"].eq(parts).all()
    assert funnel["avaliadas"].eq(funnel[["A", "B", "C", "rejeitada"]].sum(axis=1)).all()
    assert funnel.loc["hook", "campanhas"] == len(SAMPLE)
    assert funnel.loc["legenda", "valor_ausente"] == 1  # KLK-0029


def test_audit_passes_on_real_and_new_campaigns(engine, history):
    for campaigns in [history.loc[SAMPLE], pd.DataFrame([NEW, NEW | {"format": "quadrado"}])]:
        result = audit(engine, campaigns)
        assert result["itens"].gt(0).all()
        assert result["falhas"].eq(0).all()


class _Tampered(RecommendationEngine):
    """Engine adulterado: quando recebe métricas observadas, devolve a lista invertida."""

    def recommend(self, campaign, **kwargs):
        report = super().recommend(campaign, **kwargs)
        if report.observed is None:
            return report
        return dataclasses.replace(report, recommendations=report.recommendations[::-1])


def test_audit_flags_reports_that_break_the_rules(engine, history):
    tampered = _Tampered(engine.predictor, load_performance_evidence(), load_score_differences())
    failures = audit(tampered, history.loc[SAMPLE])["falhas"]
    broken = ["prioridades de 1 a n", "mesma saída sem as métricas observadas"]
    assert failures[broken].gt(0).all()
    assert failures.drop(broken).eq(0).all()


def test_saved_evaluation_matches_the_current_engine(engine):
    saved = pd.read_csv(EVALUATION_PATH).fillna({"outras_metricas_favoraveis": ""})
    current = recommendation_table(engine.recommend_all(load_clean())).round(4)
    pd.testing.assert_frame_equal(saved, current, check_dtype=False)


def test_funnel_respects_the_metric_scope_of_each_edit(reports):
    music = action_funnel(reports).loc["musica"]
    # Só KLK-0203 é de awareness na amostra; a música nem é avaliada nas outras.
    assert (music["metricas"], music["campanhas"], music["avaliadas"], music["A"]) == (
        "retenção",
        1,
        1,
        1,
    )
    assert action_funnel(reports).loc["hook", "metricas"] == "todas"


def test_audit_covers_music_manual_metric_and_exact_values(engine):
    new = pd.DataFrame(
        [
            NEW | {"objective": "awareness", "format": "quadrado", "music_voice_ratio": value}
            for value in [0.333, 0.57, 0.571]
        ]
        + [NEW | {"objective": "traffic", "format": "quadrado", "music_voice_ratio": 0.2}]
    )
    for metric in [None, "completion_rate"]:
        result = audit(engine, new, metric=metric)
        assert result["itens"].gt(0).all()
        assert result["falhas"].eq(0).all()
    # Com retenção escolhida para todas: 0,333, 0,57 e a de traffic; 0,571 passaria de 0,67.
    assert result.loc["música só na retenção, +0,1 e até 0,67", "itens"] == 3


class _HiddenCaveats(RecommendationEngine):
    """Engine adulterado: troca as ressalvas curtas pela ressalva geral, escondendo avisos."""

    def recommend(self, campaign, **kwargs):
        report = super().recommend(campaign, **kwargs)
        recs = tuple(dataclasses.replace(r, caveats=(NO_CAVEAT,)) for r in report.recommendations)
        return dataclasses.replace(report, recommendations=recs)


def test_audit_flags_short_caveats_that_hide_warnings(engine, history):
    tampered = _HiddenCaveats(
        engine.predictor, load_performance_evidence(), load_score_differences()
    )
    failures = audit(tampered, history.loc[SAMPLE])["falhas"]
    assert failures["ressalvas curtas iguais aos avisos"] > 0
    assert failures.drop("ressalvas curtas iguais aos avisos").eq(0).all()
