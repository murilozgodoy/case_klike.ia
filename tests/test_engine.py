"""Testes do recommendations engine.

Conferem as regras de elegibilidade, a separação entre score, associação, classe e suporte,
a ordem das recomendações e os casos difíceis: dados ausentes, combinações sem histórico,
ganho limítrofe, pouca base, conflito entre score e performance, a edição contínua de
música/voz, a métrica escolhida manualmente e o resumo executivo.
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from klike.data import load_clean
from klike.engine import ACTIONS, CAVEATS, MIN_SUPPORT, NO_CAVEAT, RecommendationEngine
from klike.evidence import load_performance_evidence, load_score_differences
from klike.predictor import KlikeScorePredictor, train_final_model
from klike.recommend import main

CAMPAIGN = {
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
# Já tem tudo o que o catálogo sugere, no formato mais usado da plataforma.
COMPLETE = {
    **CAMPAIGN,
    "format": "vertical",
    "has_subtitle": True,
    "has_cta": True,
    "has_hook": True,
    "text_density": "medium",
}


@pytest.fixture(scope="module")
def predictor():
    return train_final_model()


@pytest.fixture(scope="module")
def engine(predictor):
    return RecommendationEngine(predictor, load_performance_evidence(), load_score_differences())


@pytest.fixture(scope="module")
def history():
    return load_clean().set_index("campaign_id", drop=False)


def actions(report) -> list[str]:
    return [r.action for r in report.recommendations]


def by_action(report, key):
    return next(r for r in report.recommendations if r.action == key)


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def test_new_campaign_needs_only_the_pre_flight_fields(engine):
    report = engine.recommend(CAMPAIGN)
    assert report.campaign_id is None
    assert report.observed is None
    assert report.campaign == {**CAMPAIGN, "video_duration_s": 22.0}
    assert report.metric == "conversions_per_mille"
    json.dumps(report.to_dict())  # saída serializável, para integração


def test_historical_campaign_keeps_observed_values_apart(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    assert report.campaign_id == "KLK-0024"
    assert report.observed["klike_score"] == pytest.approx(43.1)
    row = history.loc["KLK-0024"]
    assert report.observed["conversions_per_mille"] == pytest.approx(
        row["conversions"] / row["impressions"] * 1000
    )
    # O valor observado não entra no cálculo: a mesma campanha sem histórico dá o mesmo resultado.
    pre_flight = {col: row[col] for col in report.campaign}
    assert actions(engine.recommend(pre_flight)) == actions(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"platform": "YouTube"}, "platform: valor desconhecido"),
        ({"has_hook": "talvez"}, "has_hook: esperado verdadeiro/falso"),
        ({"music_ratio": 0.3}, "campos desconhecidos"),
    ],
)
def test_invalid_inputs_raise_clear_errors(engine, change, message):
    with pytest.raises(ValueError, match=message):
        engine.recommend({**CAMPAIGN, **change})


def test_engine_analyzes_one_campaign_at_a_time(engine):
    with pytest.raises(ValueError, match="uma campanha por vez"):
        engine.recommend(pd.DataFrame([CAMPAIGN, CAMPAIGN]))
    with pytest.raises(ValueError, match="Métrica desconhecida"):
        engine.recommend(CAMPAIGN, metric="cliques")


# --------------------------------------------------------------------------- #
# Elegibilidade
# --------------------------------------------------------------------------- #
def test_never_recommends_what_the_campaign_already_has(engine):
    report = engine.recommend(CAMPAIGN)
    assert "rosto" not in actions(report)  # a campanha já mostra rosto
    complete = engine.recommend(COMPLETE)
    assert complete.recommendations == ()
    assert complete.summary.startswith("Nenhuma edição do catálogo tem base")


def test_each_edit_changes_only_its_own_field():
    for action in ACTIONS:
        # A música não tem valor fixo de partida: ele vem de cada campanha.
        start = 0.35 if action.current is None else action.current
        campaign = {**CAMPAIGN, "platform": action.platform or "TikTok", action.field: start}
        assert action.applies_to(campaign)
        edited = action.for_campaign(campaign).apply(campaign)
        assert {k for k in campaign if campaign[k] != edited[k]} == {action.field}


def test_catalog_never_offers_two_edits_of_the_same_field():
    # Edições do mesmo campo partem de valores diferentes ou valem em plataformas diferentes,
    # então nunca se aplicam juntas: não há edição repetida nem incompatível.
    for a, b in itertools.combinations(ACTIONS, 2):
        if a.field == b.field:
            other_platform = a.platform and b.platform and a.platform != b.platform
            assert a.current != b.current or other_platform, (a.key, b.key)


def test_scores_come_from_the_public_predictor_for_each_scenario(engine, predictor):
    report = engine.recommend(CAMPAIGN)
    before = predictor.predict(CAMPAIGN).iloc[0]
    for r in report.recommendations:
        after = predictor.predict({**CAMPAIGN, r.field: r.suggested_value}).iloc[0]
        assert r.score_before == pytest.approx(before)
        assert r.score_after == pytest.approx(after)
    assert report.context == {
        col: CAMPAIGN[col]
        for col in ["platform", "category", "objective", "target_audience_age", "is_retargeting"]
    }


def test_missing_values_are_not_treated_as_absences(engine):
    unknown = engine.recommend({**CAMPAIGN, "has_subtitle": None, "video_duration_s": None})
    assert "legenda" not in actions(unknown)
    assert any("Legenda sem marcação" in note for note in unknown.notes)
    assert any("Duração não informada" in note for note in unknown.notes)
    assert "legenda" in actions(engine.recommend(CAMPAIGN))  # marcada como sem legenda


def test_combination_without_history_is_flagged(engine):
    report = engine.recommend({**COMPLETE, "format": "horizontal"})
    assert not any(r.field == "format" for r in report.recommendations)
    assert any("nenhuma campanha no histórico" in note for note in report.notes)


def test_engine_needs_enough_campaigns_on_both_sides(predictor):
    strict = RecommendationEngine(
        predictor, load_performance_evidence(), load_score_differences(), min_support=300
    )
    report = strict.recommend(CAMPAIGN)
    assert report.recommendations == ()
    reasons = dict(report.not_recommended)
    # O formato de destino é checado antes: no TikTok, só 152 vídeos verticais.
    expected = "só 152 campanhas com vídeo vertical no TikTok, menos que o mínimo de 300"
    assert reasons.pop("formato_tiktok_quadrado") == expected
    assert all("menos de 300 campanhas de cada lado" in reason for reason in reasons.values())


# --------------------------------------------------------------------------- #
# Classes, conflitos e avisos
# --------------------------------------------------------------------------- #
def test_historical_case_has_every_class_in_order(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    assert actions(report) == ["hook", "rosto", "texto_alto_medio", "formato_meta_horizontal"]
    assert [r.recommendation_class for r in report.recommendations] == ["A", "A", "B", "C"]
    assert [r.priority for r in report.recommendations] == [1, 2, 3, 4]


def test_priority_can_differ_from_estimated_impact(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    face, text = by_action(report, "rosto"), by_action(report, "texto_alto_medio")
    assert text.association.effect > face.association.effect
    assert text.priority > face.priority
    assert "impacto estimado maior" in text.priority_note
    assert "sensível à análise" in text.estimated_impact
    assert any("depende da análise" in w for w in text.warnings)


def test_score_opportunities_follow_the_lower_bound_of_the_model_interval(engine):
    report = engine.recommend(
        {
            **COMPLETE,
            "platform": "LinkedIn",
            "format": "vertical",
            "has_cta": False,
            "text_density": "low",
        }  # fmt: skip
    )
    assert actions(report) == ["cta", "texto_baixo_medio", "formato_linkedin_vertical"]
    assert {r.recommendation_class for r in report.recommendations} == {"C"}
    linkedin = by_action(report, "formato_linkedin_vertical")
    assert linkedin.model_change[0] > by_action(report, "texto_baixo_medio").model_change[0]
    assert "mais incerto" in linkedin.priority_note


def test_tiktok_square_to_vertical_is_a_borderline_experiment(engine):
    report = engine.recommend(COMPLETE | {"format": "quadrado"})
    tiktok = by_action(report, "formato_tiktok_quadrado")
    assert tiktok.recommendation_class == "C"
    assert 0 < tiktok.model_change[1] < 1
    assert any("limítrofe" in w and "experimental" in w for w in tiktok.warnings)


def test_linkedin_format_warns_about_little_support(engine):
    report = engine.recommend(COMPLETE | {"platform": "LinkedIn", "format": "vertical"})
    linkedin = by_action(report, "formato_linkedin_vertical")
    assert linkedin.support.level == "plataforma"
    assert (linkedin.support.n_current, linkedin.support.n_target) == (11, 44)
    assert linkedin.support.risk == "moderado"
    assert any("Pouca base" in w for w in linkedin.warnings)


def test_conflict_between_score_and_performance_is_explicit(engine):
    meta = COMPLETE | {"platform": "Meta", "format": "horizontal"}
    traffic = by_action(
        engine.recommend(meta | {"objective": "traffic"}), "formato_meta_horizontal"
    )
    assert traffic.recommendation_class == "C"
    assert traffic.score_change > 0
    assert any("sinal desfavorável" in w for w in traffic.warnings)
    engagement = engine.recommend(meta | {"category": "Lead Gen", "objective": "engagement"})
    assert by_action(engagement, "formato_meta_horizontal").recommendation_class == "A"


def test_metric_override_changes_the_priority(engine):
    meta = COMPLETE | {"platform": "Meta", "format": "horizontal", "has_face": False}
    default = engine.recommend(meta | {"objective": "traffic"})
    engagement = engine.recommend(meta | {"objective": "traffic"}, metric="engagement_rate")
    assert actions(default)[0] == "rosto"
    assert actions(engagement)[0] == "formato_meta_horizontal"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, 0),
        ({"has_cta": False}, 1),
        ({"has_cta": False, "has_subtitle": False}, 2),
        ({"has_cta": False, "has_subtitle": False, "has_hook": False}, 3),
    ],
)
def test_reports_fewer_than_three_improvements_honestly(engine, change, expected):
    report = engine.recommend(COMPLETE | change)
    assert len(report.recommendations) == expected
    assert ("menos de três" in report.summary) == (0 < expected < 3)


def test_ranking_is_deterministic(engine):
    first = engine.recommend(CAMPAIGN).to_dict()
    reordered = dict(reversed(list(CAMPAIGN.items())))
    assert engine.recommend(reordered).to_dict() == first
    assert engine.recommend(CAMPAIGN).to_dict() == first


# --------------------------------------------------------------------------- #
# Escala do score
# --------------------------------------------------------------------------- #
class _Fixed:
    """Pipeline substituto: prevê ``before`` para a campanha e ``after`` para cada edição."""

    def __init__(self, before: float, after: float):
        self.before, self.after = before, after

    def predict(self, features):
        return np.array([self.before] + [self.after] * (len(features) - 1))


def test_score_limits_are_respected_and_flagged(predictor):
    def engine_with(before, after):
        stub = KlikeScorePredictor(_Fixed(before, after), predictor.support, predictor.metadata)
        return RecommendationEngine(stub, load_performance_evidence(), load_score_differences())

    near_top = engine_with(96, 130).recommend(CAMPAIGN)
    hook = by_action(near_top, "hook")
    assert hook.score_after == 100
    assert any("limite da escala" in w for w in hook.warnings)
    at_top = engine_with(120, 140).recommend(COMPLETE | {"has_cta": False})
    assert at_top.recommendations == ()
    reasons = dict(at_top.not_recommended)
    assert reasons["cta"].startswith("sem evidência suficiente na métrica do objetivo")
    assert reasons["cta"].endswith("o score previsto já está no limite da escala")


# --------------------------------------------------------------------------- #
# Consistência com as evidências
# --------------------------------------------------------------------------- #
def test_output_matches_the_evidence_tables(engine, history):
    performance = load_performance_evidence().set_index(["transicao", "metrica"])
    score = load_score_differences().set_index("transicao")
    for r in engine.recommend(history.loc[["KLK-0024"]]).recommendations:
        row = performance.loc[(r.action, r.metric)]
        assert r.association.effect == row["efeito"]
        assert (r.association.low, r.association.high) == (row["ic_inferior"], row["ic_superior"])
        assert r.association.evidence == row["evidencia"]
        assert r.association.stable == row["estavel"]
        assert r.model_change == tuple(
            score.loc[r.action, ["diferenca", "ic_inferior", "ic_superior"]]
        )


def test_support_is_the_first_level_with_enough_campaigns_on_each_side(engine, predictor):
    report = engine.recommend(CAMPAIGN)
    for r in report.recommendations:
        column, current, target = next(a for a in ACTIONS if a.key == r.action).support_condition
        counts = [
            predictor.support.count(**r.support.context, **{column: value})
            for value in (current, target)
        ]
        assert counts == [r.support.n_current, r.support.n_target]
        assert min(counts) >= MIN_SUPPORT


# --------------------------------------------------------------------------- #
# Linha de comando
# --------------------------------------------------------------------------- #
def test_command_line_for_a_historical_campaign(capsys):
    report = main(["--campanha", "KLK-0024"])
    out = capsys.readouterr().out
    assert out.startswith("Campanha KLK-0024 · Meta")
    assert len(report.recommendations) == 4


def test_command_line_reads_a_new_campaign_and_writes_json(capsys, tmp_path):
    path = tmp_path / "campanha.json"
    path.write_text(json.dumps(CAMPAIGN), encoding="utf-8")
    main(["--arquivo", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["campanha_id"] is None
    assert len(payload["recomendacoes"]) >= 3
    assert payload["recomendacoes"][0]["classe"] == "A"


def test_command_line_rejects_unknown_or_invalid_campaigns(capsys, tmp_path):
    with pytest.raises(SystemExit) as unknown:
        main(["--campanha", "KLK-9999"])
    assert unknown.value.code == 2
    path = tmp_path / "invalida.json"
    path.write_text(json.dumps({**CAMPAIGN, "platform": "YouTube"}), encoding="utf-8")
    with pytest.raises(SystemExit) as invalid:
        main(["--arquivo", str(path)])
    assert invalid.value.code == 1
    assert "platform: valor desconhecido" in capsys.readouterr().err


def test_command_line_prints_the_same_report_as_the_api(engine, history, capsys):
    report = engine.recommend(history.loc[["KLK-0024"]])
    main(["--campanha", "KLK-0024", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    main(["--campanha", "KLK-0024"])
    assert capsys.readouterr().out == report.to_text() + "\n"


def test_example_file_demonstrates_at_least_three_recommendations(capsys):
    report = main(["--arquivo", "examples/campanha_nova.json"])
    assert len(report.recommendations) >= 3
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# Linguagem das explicações
# --------------------------------------------------------------------------- #
def test_explanations_describe_general_adjusted_associations(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    hook = by_action(report, "hook")
    assert hook.rationale.startswith("Associação geral, ajustada pelo contexto")
    assert "não prova que ele seja igual em todas" in hook.rationale
    meta = COMPLETE | {"platform": "Meta", "format": "horizontal"}
    engagement = engine.recommend(meta | {"category": "Lead Gen", "objective": "engagement"})
    fmt = by_action(engagement, "formato_meta_horizontal")
    assert fmt.rationale.startswith("Associação nas campanhas da plataforma Meta")
    text, payload = report.to_text(), json.dumps(report.to_dict(), ensure_ascii=False)
    for old in ["Em campanhas parecidas", "Base de comparação"]:
        assert old not in text and old not in payload
    assert "Suporte histórico contextual" in text
    assert report.to_dict()["nota_metodologica"] in text
    assert "não uma estimativa do efeito" in hook.to_dict()["suporte"]["observacao"]


def test_estimates_near_zero_keep_their_sign(engine):
    # −0,1% e +0,2% não podem virar "0%", ainda mais num aviso de sinal desfavorável.
    text = engine.recommend(COMPLETE | {"objective": "engagement", "text_density": "low"})
    warning = next(w for w in by_action(text, "texto_baixo_medio").warnings if "desfavor" in w)
    assert warning.endswith("sinal desfavorável: −0,1% (de −18% a +21%).")
    face = engine.recommend(COMPLETE | {"objective": "awareness", "has_face": False})
    assert "retenção +11% (de +0,2% a +23%), só indício" in by_action(face, "rosto").rationale


def test_score_opportunities_speak_of_the_objective_metric(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    fmt = by_action(report, "formato_meta_horizontal")  # C em conversões, A em engajamento
    assert fmt.recommendation_class == "C"
    assert fmt.estimated_impact.endswith(
        "sem evidência suficiente em conversões por mil impressões"
    )
    warning = next(w for w in fmt.warnings if "a métrica do objetivo" in w)
    assert "Em outras métricas, a base mostra associação favorável: engajamento" in warning
    assert fmt.next_step.endswith("a base não mostra ganho suficiente nessa métrica.")
    assert "sem evidência de performance" not in json.dumps(report.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Combinações sem histórico e métricas incompatíveis
# --------------------------------------------------------------------------- #
def test_unobserved_combination_keeps_only_evidence_based_edits(engine):
    report = engine.recommend(CAMPAIGN | {"format": "horizontal"})  # TikTok horizontal: 0 campanhas
    assert actions(report) == ["hook", "texto_alto_medio"]
    for r in report.recommendations:
        assert r.warnings[0].startswith("Extrapolação: a combinação TikTok com vídeo horizontal")
        assert "não foram observados nessa combinação" in r.warnings[0]
    reasons = dict(report.not_recommended)
    assert set(reasons) == {"cta", "legenda"}
    assert all("combinação sem histórico" in reason for reason in reasons.values())
    assert report.to_dict()["campanhas_na_mesma_combinacao"] == 0


def test_observed_combinations_keep_their_recommendations(engine):
    # A mesma campanha num formato com histórico mantém as oportunidades de score.
    report = engine.recommend(CAMPAIGN)
    assert {"cta", "legenda"} <= set(actions(report))
    assert not any(w.startswith("Extrapolação") for r in report.recommendations for w in r.warnings)


def test_rare_combination_warns_without_dropping_score_opportunities(predictor):
    # Com mínimo de 12, os 11 vídeos verticais do LinkedIn viram uma combinação rara, não vazia.
    engine = RecommendationEngine(
        predictor, load_performance_evidence(), load_score_differences(), min_support=12
    )
    linkedin = COMPLETE | {"platform": "LinkedIn", "format": "vertical", "has_cta": False}
    report = engine.recommend(linkedin)
    cta = by_action(report, "cta")
    assert cta.recommendation_class == "C"
    assert cta.warnings[0].startswith(
        "Extrapolação: a combinação LinkedIn com vídeo vertical tem 11"
    )
    assert "se apoiam em poucas campanhas dessa combinação" in cta.warnings[0]


def test_roas_is_rejected_for_branding(engine, tmp_path):
    branding = COMPLETE | {"category": "Branding", "objective": "awareness"}
    with pytest.raises(ValueError, match="ROAS não se aplica a campanhas de Branding"):
        engine.recommend(branding, metric="roas_non_branding")
    path = tmp_path / "branding.json"
    path.write_text(json.dumps(branding), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--arquivo", str(path), "--metrica", "roas_non_branding"])
    assert error.value.code == 1


# --------------------------------------------------------------------------- #
# Música/voz: edição contínua, avaliada só na retenção
# --------------------------------------------------------------------------- #
# Criativo completo numa campanha de awareness: a única edição aplicável é a música.
AWARENESS = COMPLETE | {"objective": "awareness"}
MUSIC = next(action for action in ACTIONS if action.key == "musica")


def test_music_is_a_retention_edit_with_its_caveats(engine):
    report = engine.recommend(AWARENESS)
    assert actions(report) == ["musica"]
    music = report.recommendations[0]
    assert music.recommendation_class == "A"
    assert (music.current_value, music.suggested_value) == (0.35, 0.45)
    assert music.title == "Aumentar a proporção música/voz de 0,35 para 0,45"
    assert music.score_change == pytest.approx(-0.47, abs=0.01)
    assert any(w.startswith("O score previsto cai 0,5 pontos") for w in music.warnings)
    assert music.caveats[0] == "score previsto cai"
    # Abrangência: a amostra da regressão, separada do suporte; plataformas sem afirmar igualdade.
    assert "A regressão usa as 404 campanhas da base, de todos os objetivos" in music.rationale
    assert "não se confunde com as contagens do suporte histórico" in music.rationale
    assert "o que não prova que ele seja igual em todas" in music.rationale
    assert "uma proxy de atenção" in music.rationale
    assert "não uma proporção ideal" in music.rationale
    assert "amostra suficiente" in music.next_step
    assert "narração" in music.next_step
    # O impacto é na retenção; o score aparece caindo, nunca como melhora.
    assert music.estimated_impact.startswith("retenção +4%")
    assert music.headline.endswith("· score −0,5")


def test_music_support_counts_integer_windows_around_the_values(engine, history):
    music = engine.recommend(AWARENESS).recommendations[0]
    # Contagem manual, em centésimos inteiros, no recorte que o engine usou.
    rows = history
    for column, value in music.support.context.items():
        rows = rows[rows[column] == value]
    hundredths = (rows["music_voice_ratio"].dropna() * 100).round().astype(int)
    assert music.support.n_current == hundredths.between(30, 39).sum()
    assert music.support.n_target == hundredths.between(40, 49).sum()
    assert music.support.current_label == "com música/voz entre 0,30 e 0,39"
    assert music.support.target_label == "com música/voz entre 0,40 e 0,49"
    assert "janelas de 0,10" in music.support.describe()
    assert music.to_dict()["suporte"]["aproximacao"] == music.support.note


@pytest.mark.parametrize("objective", ["traffic", "engagement", "conversions", "app_install"])
def test_music_is_not_evaluated_outside_retention(engine, objective):
    report = engine.recommend(AWARENESS | {"objective": objective})
    assert report.recommendations == ()
    assert "musica" not in dict(report.not_recommended)


def test_manual_retention_metric_evaluates_music_and_says_so(engine):
    report = engine.recommend(AWARENESS | {"objective": "traffic"}, metric="completion_rate")
    assert actions(report) == ["musica"]
    assert report.metric_is_manual
    text, payload = report.to_text(), report.to_dict()
    assert (
        "Métrica avaliada: retenção (selecionada manualmente; a métrica principal de traffic é CTR)"
    ) in text
    assert payload["metrica_selecionada_manualmente"] is True
    assert payload["metrica_principal_do_objetivo"] == "ctr"
    assert "métrica do objetivo" not in text + json.dumps(payload, ensure_ascii=False)
    default = engine.recommend(AWARENESS)
    assert not default.metric_is_manual
    assert default.to_dict()["metrica_selecionada_manualmente"] is False
    assert "Métrica do objetivo: retenção" in default.to_text()


def test_manual_metric_changes_every_reference_to_the_objective_metric(engine):
    report = engine.recommend(
        AWARENESS | {"objective": "traffic", "text_density": "low"}, metric="completion_rate"
    )
    text = by_action(report, "texto_baixo_medio")
    assert text.recommendation_class == "C"
    assert text.to_dict()["classe_descricao"].endswith("na métrica selecionada")
    assert any("retenção, a métrica selecionada" in w for w in text.warnings)
    assert text.priority_note.endswith("com evidência na métrica selecionada.")
    assert "com evidência na métrica selecionada" in report.summary


@pytest.mark.parametrize(
    ("value", "applies"),
    [
        (0.0, True),
        (0.57, True),
        (0.47 + 0.1, True),  # 0,5700000000000001: o resíduo binário não tira a edição
        (0.571, False),
        (0.574, False),  # arredondada para centésimos, passaria
        (0.58, False),
        (None, False),
    ],
)
def test_music_eligibility_uses_the_exact_value(value, applies):
    assert MUSIC.applies_to(AWARENESS | {"music_voice_ratio": value}) is applies


def test_new_campaigns_keep_the_exact_music_value(engine, predictor):
    campaign = AWARENESS | {"music_voice_ratio": 0.333}
    music = by_action(engine.recommend(campaign), "musica")
    assert (music.current_value, music.suggested_value) == (0.333, 0.433)
    after = predictor.predict(campaign | {"music_voice_ratio": 0.433}).iloc[0]
    assert music.score_after == pytest.approx(after)
    assert music.title.endswith("de 0,333 para 0,433")
    # Os centésimos só centram as janelas do suporte: 0,333 vira 33.
    assert music.support.current_label == "com música/voz entre 0,28 e 0,37"
    assert MUSIC.for_campaign(campaign | {"music_voice_ratio": 0.571}).title.endswith(
        "de 0,571 para 0,671"
    )


def test_missing_music_is_neither_suggested_nor_treated_as_voice(engine):
    report = engine.recommend(AWARENESS | {"music_voice_ratio": None})
    assert report.recommendations == ()
    assert any("o engine não sugere mudá-la" in note for note in report.notes)


def test_music_has_no_favorable_evidence_outside_retention():
    # Salvaguarda do escopo: se as evidências mudarem, o escopo da música precisa ser revisto.
    evidence = load_performance_evidence()
    music = evidence[evidence["transicao"] == "musica"].set_index("metrica")
    favorable = music[(music["efeito"] > 0) & music["evidencia"].isin(["robusto", "indício"])]
    assert list(favorable.index) == ["completion_rate"]
    assert MUSIC.metrics == ("completion_rate",)


def test_rare_combination_keeps_music_with_extrapolation_first(predictor):
    # Com mínimo de 12, os 11 vídeos verticais do LinkedIn viram uma combinação rara.
    engine = RecommendationEngine(
        predictor, load_performance_evidence(), load_score_differences(), min_support=12
    )
    report = engine.recommend(AWARENESS | {"platform": "LinkedIn", "format": "vertical"})
    music = by_action(report, "musica")
    assert music.warnings[0].startswith("Extrapolação")
    assert music.caveats[:2] == ("extrapolação", "score previsto cai")


# --------------------------------------------------------------------------- #
# Resumo executivo
# --------------------------------------------------------------------------- #
def test_executive_summary_shows_the_first_three_with_short_caveats(engine, history):
    report = engine.recommend(history.loc[["KLK-0024"]])
    lines = report.to_text().splitlines()
    start = lines.index("Principais recomendações (3 de 4):")
    assert lines[start + 1 : start + 4] == [
        "   1. Incluir um hook nos 3 primeiros segundos · classe A",
        "      conversões por mil impressões +48% (de +26% a +75%), evidência robusta · "
        "score +17,0",
        f"      Ressalvas: {NO_CAVEAT}",
    ]
    assert lines[start + 9] == "      Ressalvas: classe sensível à análise"  # o texto, B
    payload = report.to_dict()
    assert [entry["acao"] for entry in payload["principais"]] == [
        "hook",
        "rosto",
        "texto_alto_medio",
    ]
    assert payload["composicao"] == {
        "recomendacoes": 4,
        "com_evidencia_na_metrica": 3,
        "hipoteses_de_score": 1,
        "por_classe": {"A": 2, "B": 1, "C": 1},
        "menos_de_tres": False,
    }
    assert report.summary == (
        "4 recomendações: 3 com evidência na métrica do objetivo (2 de classe A e 1 de classe B) "
        "e 1 hipótese de score (classe C)."
    )
    frame = report.to_frame()
    assert list(frame["ressalvas"]) == [" · ".join(r.caveats) for r in report.recommendations]


def test_executive_summary_never_calls_score_opportunities_improvements(engine, history):
    report = engine.recommend(history.loc[["KLK-0013"]])  # três recomendações C
    assert report.summary == (
        "3 recomendações, todas hipóteses de score (classe C), sem evidência suficiente na "
        "métrica do objetivo."
    )
    text = report.to_text()
    block = text[text.index("Principais recomendações") : text.index("Em detalhe:")]
    assert "melhoria" not in block
    assert "melhoria" not in report.summary
    assert all("sem evidência na métrica" in r.caveats for r in report.recommendations)


def test_fewer_than_three_recommendations_are_stated_in_the_summary(engine):
    report = engine.recommend(COMPLETE | {"has_cta": False})
    assert report.summary.startswith(
        "1 recomendação, hipótese de score (classe C), sem evidência suficiente na métrica do "
        "objetivo. Há menos de três recomendações defensáveis"
    )
    assert "Principais recomendações:" in report.to_text()  # cabem todas, sem "(n de m)"
    assert report.to_dict()["composicao"]["menos_de_tres"] is True


def test_caveats_follow_the_fixed_precedence_without_hiding_warnings(engine, history):
    rank = {label: position for position, label in enumerate(CAVEATS)}
    for cid in ["KLK-0013", "KLK-0203", "KLK-0024"]:
        for r in engine.recommend(history.loc[[cid]]).recommendations:
            if r.warnings:
                # Cada aviso completo tem um rótulo, na ordem de precedência, sem repetição.
                assert len(r.caveats) == len(r.warnings)
                assert [rank[c] for c in r.caveats] == sorted(rank[c] for c in r.caveats)
            else:
                assert r.caveats == (NO_CAVEAT,)
    text = by_action(engine.recommend(history.loc[["KLK-0013"]]), "texto_baixo_medio")
    assert text.caveats == (
        "sem evidência na métrica",
        "estimativa desfavorável e inconclusiva",
        "pouca base no contexto (só na plataforma)",
    )


def test_support_caveat_says_how_wide_the_comparison_had_to_go(engine, history):
    # Mesmo risco e mesmas contagens: o rótulo só diz onde o engine achou base.
    moderate = by_action(engine.recommend(history.loc[["KLK-0001"]]), "musica")
    assert moderate.support.risk == "moderado"
    assert moderate.caveats[-1] == "pouca base no contexto (só na plataforma)"
    high = engine.recommend(history.loc[["KLK-0352"]])
    music = by_action(high, "musica")
    assert (music.priority, music.support.risk) == (1, "alto")
    assert music.caveats == ("score previsto cai", "pouca base no contexto (só na base inteira)")
    assert "Ressalvas: score previsto cai · pouca base no contexto (só na base inteira)" in (
        high.to_text()
    )


def test_registered_half_values_are_not_rounded_twice(engine, history):
    # O rosto no CTR está registrado como 0,2350 (estimativa exata: 23,496%): sai +23,5%.
    face = by_action(
        engine.recommend(COMPLETE | {"objective": "traffic", "has_face": False}), "rosto"
    )
    assert face.estimated_impact.startswith("CTR +23,5% (de +7% a +42%)")
    hook = by_action(engine.recommend(history.loc[["KLK-0024"]]), "rosto")
    assert "CTR +23,5% (de +7% a +42%)" in hook.rationale
    assert "+24%" not in hook.rationale


# --------------------------------------------------------------------------- #
# Score nos limites da escala e intervalo do modelo
# --------------------------------------------------------------------------- #
def test_score_fall_and_scale_limit_warnings_are_independent(predictor):
    def music_with(before, after):
        stub = KlikeScorePredictor(_Fixed(before, after), predictor.support, predictor.metadata)
        engine = RecommendationEngine(stub, load_performance_evidence(), load_score_differences())
        return by_action(engine.recommend(AWARENESS), "musica")

    near_floor = music_with(0.3, -0.17)  # a queda de 0,47 é cortada em 0
    assert near_floor.score_after == 0
    assert near_floor.caveats[:2] == ("score previsto cai", "score no limite da escala")
    assert any(w.startswith("O score previsto cai 0,3 pontos") for w in near_floor.warnings)
    assert any("menor, em módulo, que a do modelo" in w for w in near_floor.warnings)
    at_top = music_with(120, 119.53)  # as duas previsões ficam em 100
    assert at_top.score_change == 0
    assert "score previsto cai" not in at_top.caveats
    assert at_top.caveats[0] == "score no limite da escala"
    unclipped = music_with(50, 49.53)
    assert "score no limite da escala" not in unclipped.caveats


def test_score_opportunity_needs_a_model_interval_above_zero(predictor):
    # Tabela de score substituta: intervalo do CTA cruzando o zero e todo negativo.
    for low, high in [(-0.5, 2.0), (-2.0, -0.5)]:
        scores = load_score_differences()
        scores.loc[scores["transicao"] == "cta", ["ic_inferior", "ic_superior"]] = [low, high]
        engine = RecommendationEngine(predictor, load_performance_evidence(), scores)
        report = engine.recommend(COMPLETE | {"has_cta": False})
        assert report.recommendations == ()
        assert dict(report.not_recommended)["cta"] == (
            "sem evidência suficiente na métrica do objetivo, e o intervalo do score do modelo "
            "não fica acima de zero"
        )
