"""Recommendations engine: edições sugeridas para um criativo, com evidência e prioridade.

Recebe uma campanha, com os 13 atributos de pré-publicação; métricas observadas de uma campanha
histórica são opcionais. Devolve as edições do vídeo que a base sustenta, em ordem de
prioridade. Para cada edição, quatro informações ficam separadas:

- diferença prevista de score: o preditor da Parte 2 calcula o score da campanha antes e depois
  da edição, dentro da escala de 0 a 100; o intervalo do modelo vem de ``score_differences.csv``;
- associação com performance: o contraste da edição na métrica do objetivo da campanha, lido de
  ``recommendation_evidence.csv``, sem recalcular nada;
- classe de evidência: A, B ou C, pela regra registrada no notebook 04;
- suporte histórico: quantas campanhas parecidas têm o valor atual e o sugerido.

Nada disso é efeito causal: cada recomendação é uma hipótese para validar com teste A/B.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from klike.config import CONTEXT_COLS, TARGET
from klike.evidence import PERFORMANCE_METRICS, load_performance_evidence, load_score_differences
from klike.formatting import fmt_number, fmt_signed, fmt_signed_pct
from klike.modeling import PRE_FLIGHT_INPUTS, prepare_features
from klike.predictor import KlikeScorePredictor, train_final_model

# Métrica principal de cada objetivo. É uma premissa de produto: pode ser trocada por parâmetro.
OBJECTIVE_METRICS = {
    "traffic": "ctr",
    "engagement": "engagement_rate",
    "awareness": "completion_rate",
    "conversions": "conversions_per_mille",
    "app_install": "conversions_per_mille",
}
# Campanhas de cada lado da comparação para usar um recorte de suporte. Decisão operacional,
# a mesma das interações da Parte 2; a avaliação testa 5 e 20.
MIN_SUPPORT = 10
# Abaixo deste limite inferior, em pontos, o ganho de score do modelo é tratado como limítrofe.
BORDERLINE_SCORE = 1.0
SIGNIFICANCE = 0.05

CLASS_LABELS = {
    "A": "melhoria com evidência de performance",
    "B": "melhoria com evidência fraca ou sensível",
    "C": "oportunidade de score, sem evidência suficiente na métrica do objetivo",
}
# Como ler qualquer saída do engine: o que cada número é, e o que ele não é.
METHOD_NOTE = (
    "As associações com performance são gerais: estimadas na base inteira, com ajuste pelo "
    "contexto da campanha e pelos outros atributos do criativo (o formato, dentro de cada "
    "plataforma). Não são efeitos medidos só em campanhas como esta. O suporte histórico "
    "contextual conta campanhas comparáveis no recorte indicado: descreve a base de "
    "comparação, não uma estimativa do efeito nesse recorte. O score é a previsão do modelo. "
    "Nada disso é efeito causal."
)
EVIDENCE_PHRASES = {
    "robusto": "evidência robusta",
    "indício": "só indício",
    "inconclusivo": "inconclusiva",
}
# Recortes de suporte, do mais específico ao mais amplo, e o risco de extrapolação de cada um.
SUPPORT_LEVELS = [
    ("plataforma, categoria e objetivo", ("platform", "category", "objective"), "baixo"),
    ("plataforma e objetivo", ("platform", "objective"), "baixo"),
    ("plataforma", ("platform",), "moderado"),
    ("todas as campanhas", (), "alto"),
]
# No suporte histórico, a legenda aparece como sim, não ou desconhecido.
SUBTITLE_SUPPORT = {False: "não", True: "sim"}


@dataclass(frozen=True)
class Action:
    """Uma edição do catálogo: o campo, o valor atual e o sugerido, e como descrevê-la.

    Os rótulos completam a frase "N campanhas ..." (``sem hook``, ``com texto alto``); a
    ``phrase`` descreve a edição numa justificativa.
    """

    key: str  # transição nas tabelas de evidência
    title: str
    field: str
    current: object
    target: object
    current_label: str
    target_label: str
    phrase: str
    platform: str | None = None
    alternative: str | None = None  # outra transição, citada na justificativa

    def applies_to(self, campaign: Mapping) -> bool:
        """A campanha tem exatamente o valor atual, na plataforma da edição, se houver uma.

        Um valor ausente nunca se aplica: legenda sem marcação não é o mesmo que sem legenda.
        """
        if self.platform is not None and campaign["platform"] != self.platform:
            return False
        value = campaign[self.field]
        return value is not None and value == self.current

    def apply(self, campaign: Mapping) -> dict:
        """A mesma campanha com só este campo editado."""
        return {**campaign, self.field: self.target}

    @property
    def support_condition(self) -> tuple[str, object, object]:
        """Coluna do suporte histórico e os dois valores comparados."""
        if self.field == "has_subtitle":
            return "subtitle_status", SUBTITLE_SUPPORT[self.current], SUBTITLE_SUPPORT[self.target]
        return self.field, self.current, self.target


def _add(key: str, title: str, field: str, feature: str, phrase: str) -> Action:
    """Edição de uma característica binária: de ausente para presente."""
    return Action(key, title, field, False, True, f"sem {feature}", f"com {feature}", phrase)


def _change(
    key: str, title: str, field: str, values: tuple[str, str], labels: tuple[str, str], **extra
) -> Action:
    """Edição de uma categoria para outra, como texto alto → médio."""
    current_label, target_label = labels
    phrase = f"usar {target_label} em vez de {current_label.split(' ', 1)[1]}"
    return Action(
        key,
        title,
        field,
        *values,
        f"com {current_label}",
        f"com {target_label}",
        phrase,
        **extra,
    )


ACTIONS = [
    _add(
        "hook",
        "Incluir um hook nos 3 primeiros segundos",
        "has_hook",
        "hook",
        "ter um hook nos 3 primeiros segundos",
    ),
    _add("rosto", "Mostrar um rosto humano", "has_face", "rosto humano", "mostrar um rosto humano"),
    _add("cta", "Incluir um call-to-action", "has_cta", "call-to-action", "ter call-to-action"),
    _add("legenda", "Incluir legenda", "has_subtitle", "legenda", "ter legenda"),
    _change(
        "texto_alto_medio",
        "Reduzir o texto na tela de alto para médio",
        "text_density",
        ("high", "medium"),
        ("texto alto", "texto médio"),
        alternative="texto_alto_baixo",
    ),
    _change(
        "texto_baixo_medio",
        "Aumentar o texto na tela de baixo para médio",
        "text_density",
        ("low", "medium"),
        ("texto baixo", "texto médio"),
    ),
    _change(
        "formato_meta_horizontal",
        "Trocar o vídeo horizontal por vertical",
        "format",
        ("horizontal", "vertical"),
        ("vídeo horizontal", "vídeo vertical"),
        platform="Meta",
    ),
    _change(
        "formato_tiktok_quadrado",
        "Trocar o vídeo quadrado por vertical",
        "format",
        ("quadrado", "vertical"),
        ("vídeo quadrado", "vídeo vertical"),
        platform="TikTok",
    ),
    _change(
        "formato_linkedin_vertical",
        "Trocar o vídeo vertical por horizontal",
        "format",
        ("vertical", "horizontal"),
        ("vídeo vertical", "vídeo horizontal"),
        platform="LinkedIn",
    ),
    _change(
        "formato_linkedin_quadrado",
        "Trocar o vídeo quadrado por horizontal",
        "format",
        ("quadrado", "horizontal"),
        ("vídeo quadrado", "vídeo horizontal"),
        platform="LinkedIn",
    ),
]
# Edições avaliadas na Etapa 1 que o engine não recomenda (notebook 04, seção 8).
DISCARDED = {
    "texto_alto_baixo": "citada como alternativa na recomendação de texto alto → médio",
    "musica": "efeito pequeno, com score e retenção em direções opostas",
    "duracao": "nenhuma associação, com intervalos estreitos",
    "formato_meta_quadrado": "sem associação e sem diferença de score",
}


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Association:
    """Associação histórica da edição com uma métrica de performance (variação relativa)."""

    metric: str
    metric_name: str
    effect: float
    low: float
    high: float
    evidence: str
    stable: bool
    p_value: float
    q_value: float
    n: int
    sensitivity: str | None
    limitations: str

    def interval(self) -> str:
        return f"{_pct(self.effect)} (de {_pct(self.low)} a {_pct(self.high)})"

    def describe(self) -> str:
        sensitive = "" if self.stable else ", mas sensível à análise"
        return f"{self.metric_name} {self.interval()}, {EVIDENCE_PHRASES[self.evidence]}{sensitive}"

    def to_dict(self) -> dict:
        return {
            "metrica": self.metric,
            "metrica_nome": self.metric_name,
            "variacao": round(self.effect, 4),
            "ic_inferior": round(self.low, 4),
            "ic_superior": round(self.high, 4),
            "unidade": "variação relativa",
            "evidencia": self.evidence,
            "estavel": self.stable,
            "p_valor": self.p_value,
            "q_valor": self.q_value,
            "campanhas": self.n,
            "sensibilidade": self.sensitivity,
            "limitacoes": self.limitations,
        }


@dataclass(frozen=True)
class Support:
    """Suporte histórico contextual: campanhas com o valor atual e com o sugerido, no recorte."""

    level: str
    context: dict
    n_current: int
    n_target: int
    risk: str
    minimum: int

    def to_dict(self) -> dict:
        return {
            "recorte": self.level,
            "contexto": self.context,
            "campanhas_valor_atual": self.n_current,
            "campanhas_valor_sugerido": self.n_target,
            "minimo_por_lado": self.minimum,
            "risco_extrapolacao": self.risk,
            "observacao": (
                "campanhas comparáveis no recorte: descrevem a base de comparação, não uma "
                "estimativa do efeito nesse recorte"
            ),
        }


@dataclass(frozen=True)
class Recommendation:
    """Uma edição recomendada, com prioridade, evidência, score e suporte separados."""

    priority: int
    action: str
    title: str
    field: str
    current_value: object
    suggested_value: object
    current_label: str
    suggested_label: str
    recommendation_class: str
    metric: str
    metric_name: str
    association: Association | None
    other_associations: tuple[Association, ...]
    score_before: float
    score_after: float
    model_change: tuple[float, float, float]  # estimativa e intervalo, de score_differences.csv
    support: Support
    rationale: str
    priority_note: str | None
    warnings: tuple[str, ...]
    next_step: str

    @property
    def score_change(self) -> float:
        return self.score_after - self.score_before

    @property
    def estimated_impact(self) -> str:
        """O impacto estimado, separado da prioridade: na métrica do objetivo ou só no score."""
        if self.recommendation_class in ("A", "B"):
            return self.association.describe()
        change = fmt_signed(self.score_change, 1)
        return f"Klike Score {change} pontos; sem evidência suficiente em {self.metric_name}"

    def to_dict(self) -> dict:
        estimate, low, high = self.model_change
        return {
            "prioridade": self.priority,
            "acao": self.action,
            "titulo": self.title,
            "campo": self.field,
            "valor_atual": self.current_value,
            "valor_sugerido": self.suggested_value,
            "de": self.current_label,
            "para": self.suggested_label,
            "classe": self.recommendation_class,
            "classe_descricao": CLASS_LABELS[self.recommendation_class],
            "impacto_estimado": self.estimated_impact,
            "metrica": self.metric,
            "metrica_nome": self.metric_name,
            "associacao": self.association.to_dict() if self.association else None,
            "outras_associacoes": [a.to_dict() for a in self.other_associations],
            "score_antes": round(self.score_before, 2),
            "score_depois": round(self.score_after, 2),
            "diferenca_score": round(self.score_change, 2),
            "diferenca_score_modelo": {
                "estimativa": estimate,
                "ic_inferior": low,
                "ic_superior": high,
                "observacao": "variabilidade da estimativa do modelo, não efeito causal",
            },
            "suporte": self.support.to_dict(),
            "justificativa": self.rationale,
            "nota_de_prioridade": self.priority_note,
            "avisos": list(self.warnings),
            "proximo_passo": self.next_step,
        }


@dataclass(frozen=True)
class RecommendationReport:
    """Resultado do engine para uma campanha."""

    campaign: dict
    objective: str
    metric: str
    metric_name: str
    predicted_score: float
    observed: dict | None
    recommendations: tuple[Recommendation, ...]
    not_recommended: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    summary: str
    campaign_id: str | None = None
    combination_support: int | None = None  # campanhas com a mesma plataforma e formato
    context: dict = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", {col: self.campaign[col] for col in CONTEXT_COLS})

    def to_dict(self) -> dict:
        return {
            "campanha_id": self.campaign_id,
            "campanha": self.campaign,
            "contexto": self.context,
            "objetivo": self.objective,
            "metrica": self.metric,
            "metrica_nome": self.metric_name,
            "score_previsto": round(self.predicted_score, 2),
            "observado": self.observed,
            "campanhas_na_mesma_combinacao": self.combination_support,
            "resumo": self.summary,
            "nota_metodologica": METHOD_NOTE,
            "avisos": list(self.notes),
            "recomendacoes": [r.to_dict() for r in self.recommendations],
            "nao_recomendadas": [
                {"acao": key, "motivo": reason} for key, reason in self.not_recommended
            ],
        }

    def to_frame(self) -> pd.DataFrame:
        """Uma linha por recomendação, para ler num notebook."""
        rows = [
            {
                "prioridade": r.priority,
                "recomendação": r.title,
                "classe": r.recommendation_class,
                "impacto estimado": r.estimated_impact,
                "score": (
                    f"{fmt_number(r.score_before, 1)} → {fmt_number(r.score_after, 1)} "
                    f"({fmt_signed(r.score_change, 1)})"
                ),
                "suporte histórico contextual": (
                    f"{r.support.n_current} {r.current_label}, {r.support.n_target} "
                    f"{r.suggested_label} ({r.support.level})"
                ),
                "avisos": len(r.warnings),
            }
            for r in self.recommendations
        ]
        return pd.DataFrame(rows).set_index("prioridade") if rows else pd.DataFrame()

    def to_text(self) -> str:
        """Relatório em texto, para a linha de comando."""
        c = self.campaign
        header = " · ".join(
            [
                f"Campanha {self.campaign_id}" if self.campaign_id else "Campanha nova",
                c["platform"],
                c["category"],
                c["objective"],
                f"público {c['target_audience_age']}",
                "com retargeting" if c["is_retargeting"] else "sem retargeting",
            ]
        )
        score = f"Score previsto: {fmt_number(self.predicted_score, 1)}"
        if self.observed and self.observed.get(TARGET) is not None:
            score += f" (observado: {fmt_number(self.observed[TARGET], 1)})"
        lines = [header, f"Métrica do objetivo: {self.metric_name}", score, "", self.summary]
        for r in self.recommendations:
            _, low, high = r.model_change
            lines += [
                "",
                f"{r.priority}. {r.title}  [classe {r.recommendation_class}: "
                f"{CLASS_LABELS[r.recommendation_class]}]",
                f"   Impacto estimado: {r.estimated_impact}",
                f"   Score previsto: {fmt_number(r.score_before, 1)} → "
                f"{fmt_number(r.score_after, 1)} ({fmt_signed(r.score_change, 1)}; intervalo do "
                f"modelo: {_points(low)} a {_points(high)})",
                f"   Suporte histórico contextual: {r.support.n_current} campanhas "
                f"{r.current_label} e {r.support.n_target} {r.suggested_label} (recorte: "
                f"{r.support.level})",
                f"   Por quê: {r.rationale}",
            ]
            if r.priority_note:
                lines.append(f"   Prioridade: {r.priority_note}")
            lines += [f"   Aviso: {w}" for w in r.warnings]
            lines.append(f"   Próximo passo: {r.next_step}")
        if self.not_recommended:
            lines += ["", "Avaliadas e não recomendadas:"]
            lines += [f"   - {key}: {reason}" for key, reason in self.not_recommended]
        if self.notes:
            lines += ["", "Avisos da campanha:"]
            lines += [f"   - {note}" for note in self.notes]
        lines += ["", f"Como ler este relatório: {METHOD_NOTE}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RecommendationEngine:
    """Junta preditor, evidências e suporte histórico, sem recalcular nenhum deles."""

    def __init__(
        self,
        predictor: KlikeScorePredictor,
        performance: pd.DataFrame,
        score_differences: pd.DataFrame,
        *,
        min_support: int = MIN_SUPPORT,
    ):
        self.predictor = predictor
        self.performance = performance.set_index(["transicao", "metrica"])
        self.score_differences = score_differences.set_index("transicao")
        self.min_support = min_support

    @classmethod
    def default(cls, *, min_support: int = MIN_SUPPORT) -> RecommendationEngine:
        """Modelo salvo em ``models/``, se existir; senão, treina o modelo final na hora."""
        try:
            predictor = KlikeScorePredictor.load()
        except FileNotFoundError:
            predictor = train_final_model()
        return cls(
            predictor,
            load_performance_evidence(),
            load_score_differences(),
            min_support=min_support,
        )

    def recommend(
        self, campaign: Mapping | pd.Series | pd.DataFrame, *, metric: str | None = None
    ) -> RecommendationReport:
        """Recomendações para uma campanha, em ordem de prioridade.

        ``metric`` troca a métrica do objetivo por outra de ``PERFORMANCE_METRICS``.
        """
        frame = _single_campaign(campaign)
        current = _canonical(prepare_features(frame, strict=True).iloc[0])
        metric = metric or OBJECTIVE_METRICS[current["objective"]]
        if metric not in PERFORMANCE_METRICS:
            raise ValueError(
                f"Métrica desconhecida: {metric!r} (use {', '.join(PERFORMANCE_METRICS)})."
            )
        if metric == "roas_non_branding" and current["category"] == "Branding":
            raise ValueError(
                "ROAS não se aplica a campanhas de Branding, que não registram receita; "
                "escolha outra métrica."
            )

        applicable = [action for action in ACTIONS if action.applies_to(current)]
        scenarios = [current] + [action.apply(current) for action in applicable]
        scores = self.predictor.predict(scenarios).to_numpy()
        combination = self.predictor.support.count(
            platform=current["platform"], format=current["format"]
        )
        candidates, skipped = [], []
        for action, after in zip(applicable, scores[1:], strict=True):
            outcome = self._evaluate(
                action, current, metric, float(scores[0]), float(after), combination
            )
            if isinstance(outcome, str):
                skipped.append((action.key, outcome))
            else:
                candidates.append(outcome)
        ranked = _rank(candidates)
        return RecommendationReport(
            campaign=current,
            objective=current["objective"],
            metric=metric,
            metric_name=PERFORMANCE_METRICS[metric],
            predicted_score=float(scores[0]),
            observed=_observed(frame.iloc[0], metric, current),
            recommendations=ranked,
            not_recommended=tuple(skipped),
            notes=self._campaign_notes(current),
            summary=_summary(ranked, current),
            campaign_id=_campaign_id(frame.iloc[0]),
            combination_support=combination,
        )

    def recommend_all(self, campaigns: pd.DataFrame, **kwargs) -> list[RecommendationReport]:
        """Uma análise por linha, para avaliar o engine em muitas campanhas."""
        return [self.recommend(campaigns.iloc[[i]], **kwargs) for i in range(len(campaigns))]

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        action: Action,
        current: dict,
        metric: str,
        before: float,
        after: float,
        combination: int,
    ):
        """Classifica uma edição aplicável ou devolve o motivo de não recomendá-la.

        ``combination`` é o número de campanhas do histórico com a mesma plataforma e o mesmo
        formato da campanha. Numa combinação nunca observada, uma sugestão só de score (C) não
        tem base nenhuma: sem evidência na métrica do objetivo, o ganho viria só de uma previsão
        fora do histórico. As classes A e B ficam, com aviso de extrapolação.
        """
        if action.platform:
            available = self.predictor.support.count(platform=action.platform, format=action.target)
            if available < self.min_support:
                return (
                    f"só {available} campanhas {action.target_label} no {action.platform}, "
                    f"menos que o mínimo de {self.min_support}"
                )
        association = self._association(action.key, metric, current)
        model = self._model_change(action.key)
        favorable = association is not None and association.effect > 0
        if association and association.effect < 0 and association.p_value < SIGNIFICANCE:
            return f"associação desfavorável: {association.describe()}"
        if favorable and association.evidence == "robusto":
            klass = "A" if association.stable else "B"
        elif favorable and association.evidence == "indício":
            klass = "B"
        elif model[1] <= 0:
            return (
                "sem evidência suficiente na métrica do objetivo, e o intervalo do score do "
                "modelo inclui zero"
            )
        elif after <= before:
            return (
                "sem evidência suficiente na métrica do objetivo, e o score previsto já está no "
                "limite da escala"
            )
        else:
            klass = "C"
        if klass == "C" and combination == 0:
            return (
                f"sugestão só de score numa combinação sem histórico ({current['platform']} com "
                f"vídeo {current['format']}): sem evidência na métrica do objetivo, o ganho "
                "previsto não tem base observada"
            )
        support = self._support(action, current)
        if support is None:
            return f"menos de {self.min_support} campanhas de cada lado mesmo no recorte mais amplo"
        return _Candidate(
            action=action,
            klass=klass,
            metric=metric,
            association=association,
            others=self._other_associations(action.key, metric, current),
            before=before,
            after=after,
            model=model,
            support=support,
            alternative=self._alternative(action, metric, current, model),
            scenario=_scenario_warning(current, combination, self.min_support),
        )

    def _association(self, key: str, metric: str, campaign: dict) -> Association | None:
        if metric == "roas_non_branding" and campaign["category"] == "Branding":
            return None
        row = self.performance.loc[(key, metric)]
        return Association(
            metric=metric,
            metric_name=PERFORMANCE_METRICS[metric],
            effect=float(row["efeito"]),
            low=float(row["ic_inferior"]),
            high=float(row["ic_superior"]),
            evidence=row["evidencia"],
            stable=bool(row["estavel"]),
            p_value=float(row["p_valor"]),
            q_value=float(row["q_valor"]),
            n=int(row["n"]),
            sensitivity=_sensitivity(row),
            limitations=row["limitacoes"],
        )

    def _other_associations(self, key: str, metric: str, campaign: dict) -> tuple:
        """Associações com evidência nas outras métricas, como contexto da recomendação."""
        others = []
        for other in PERFORMANCE_METRICS:
            association = None if other == metric else self._association(key, other, campaign)
            if association and association.evidence in ("robusto", "indício"):
                others.append(association)
        return tuple(others)

    def _alternative(self, action: Action, metric: str, campaign: dict, model) -> str | None:
        """Frase sobre a alternativa da edição, como texto baixo no lugar de médio."""
        if action.alternative is None:
            return None
        association = self._association(action.alternative, metric, campaign)
        alternative_score = self._model_change(action.alternative)[0]
        detail = f"{association.describe()}; " if association else ""
        return (
            f"Reduzir para texto baixo também é opção: {detail}no modelo, o score sobe "
            f"{fmt_signed(alternative_score, 1)} pontos, contra {fmt_signed(model[0], 1)} do "
            "texto médio."
        )

    def _model_change(self, key: str) -> tuple[float, float, float]:
        row = self.score_differences.loc[key]
        return float(row["diferenca"]), float(row["ic_inferior"]), float(row["ic_superior"])

    def _support(self, action: Action, campaign: dict) -> Support | None:
        """Primeiro recorte, do mais específico ao mais amplo, com o mínimo de cada lado.

        O formato é comparado sempre dentro da plataforma, então para no recorte da plataforma.
        """
        column, current_value, target_value = action.support_condition
        levels = SUPPORT_LEVELS[:-1] if action.platform else SUPPORT_LEVELS
        for name, columns, risk in levels:
            context = {col: campaign[col] for col in columns}
            n_current = self.predictor.support.count(**context, **{column: current_value})
            n_target = self.predictor.support.count(**context, **{column: target_value})
            if min(n_current, n_target) >= self.min_support:
                return Support(name, context, n_current, n_target, risk, self.min_support)
        return None

    def _campaign_notes(self, campaign: dict) -> tuple[str, ...]:
        notes = []
        for col, label in [
            ("video_duration_s", "Duração"),
            ("music_voice_ratio", "Proporção música/voz"),
        ]:
            if campaign[col] is None:
                notes.append(
                    f"{label} não informada: o modelo usa a mediana das campanhas de treino."
                )
        if campaign["has_subtitle"] is None:
            notes.append(
                "Legenda sem marcação: o engine não sabe se o vídeo tem legenda e não sugere "
                "incluí-la."
            )
        combination = self.predictor.support.count(
            platform=campaign["platform"], format=campaign["format"]
        )
        if combination < self.min_support:
            notes.append(
                f"A combinação {campaign['platform']} com vídeo {campaign['format']} "
                f"{_history(combination)}: o score previsto é extrapolação."
            )
        return tuple(notes)


@dataclass(frozen=True)
class _Candidate:
    action: Action
    klass: str
    metric: str
    association: Association | None
    others: tuple[Association, ...]
    before: float
    after: float
    model: tuple[float, float, float]
    support: Support
    alternative: str | None
    scenario: str | None  # aviso de extrapolação, quando a combinação tem pouco histórico

    def sort_key(self) -> tuple:
        """Classe primeiro; dentro dela, a estimativa na métrica do objetivo.

        Para oportunidades de score (C), a ordem segue o limite inferior do intervalo do
        modelo, que respeita a incerteza. A chave da ação desempata, para o ranking ser
        determinístico.
        """
        order = {"A": 0, "B": 1, "C": 2}[self.klass]
        if self.klass in ("A", "B"):
            a = self.association
            return (order, -a.effect, -a.low, -(self.after - self.before), self.action.key)
        return (order, -self.model[1], -self.model[0], self.action.key)


def _rank(candidates: list[_Candidate]) -> tuple[Recommendation, ...]:
    ordered = sorted(candidates, key=_Candidate.sort_key)
    return tuple(
        _recommendation(position, candidate, ordered[: position - 1])
        for position, candidate in enumerate(ordered, start=1)
    )


def _recommendation(priority: int, c: _Candidate, above: list[_Candidate]) -> Recommendation:
    action = c.action
    return Recommendation(
        priority=priority,
        action=action.key,
        title=action.title,
        field=action.field,
        current_value=action.current,
        suggested_value=action.target,
        current_label=action.current_label,
        suggested_label=action.target_label,
        recommendation_class=c.klass,
        metric=c.metric,
        metric_name=PERFORMANCE_METRICS[c.metric],
        association=c.association,
        other_associations=c.others,
        score_before=c.before,
        score_after=c.after,
        model_change=c.model,
        support=c.support,
        rationale=_rationale(c),
        priority_note=_priority_note(c, above),
        warnings=_warnings(c),
        next_step=_next_step(c),
    )


# --------------------------------------------------------------------------- #
# Textos
# --------------------------------------------------------------------------- #
def _rationale(c: _Candidate) -> str:
    action, a = c.action, c.association
    metric = PERFORMANCE_METRICS[c.metric]
    if c.klass in ("A", "B") and action.platform:
        parts = [
            f"Associação nas campanhas da plataforma {action.platform}, ajustada pelo restante "
            f"do contexto: {action.phrase} se associa a {a.describe()}."
        ]
    elif c.klass in ("A", "B"):
        parts = [
            f"Associação geral, ajustada pelo contexto: na base inteira, {action.phrase} se "
            f"associa a {a.describe()}. Não é um efeito estimado só para campanhas como esta; a "
            "base não mostrou diferença robusta entre plataformas, o que não prova que ele seja "
            "igual em todas."
        ]
    elif a is not None:
        parts = [
            f"O modelo prevê mais score, mas não há evidência suficiente de ganho em {metric}, "
            f"a métrica do objetivo: {a.interval()}, {EVIDENCE_PHRASES[a.evidence]}."
        ]
    else:
        parts = [f"O modelo prevê mais score; não há como medir a associação com {metric}."]
    if c.others:
        parts.append(
            "Em outras métricas: " + "; ".join(other.describe() for other in c.others) + "."
        )
    if c.alternative:
        parts.append(c.alternative)
    return " ".join(parts)


def _priority_note(c: _Candidate, above: list[_Candidate]) -> str | None:
    """Explica quando a prioridade não acompanha o impacto estimado."""
    if c.klass in ("A", "B"):
        smaller = [
            other.action.title.lower()
            for other in above
            if other.association and other.association.effect < c.association.effect
        ]
        if smaller:
            return (
                f"Tem impacto estimado maior que {_join(smaller)}, mas fica depois porque a "
                f"evidência é de classe {c.klass}: {CLASS_LABELS[c.klass]}."
            )
        return None
    if any(other.klass in ("A", "B") for other in above):
        return (
            "Oportunidade de score: fica depois das recomendações com evidência na métrica do "
            "objetivo."
        )
    larger = [
        other.action.title.lower()
        for other in above
        if other.klass == "C" and other.model[0] < c.model[0]
    ]
    if larger:
        return (
            f"O ganho de score estimado é maior que o de {_join(larger)}, mas mais incerto: entre "
            "oportunidades de score, a ordem segue o limite inferior do intervalo do modelo."
        )
    return None


def _warnings(c: _Candidate) -> tuple[str, ...]:
    warnings = [c.scenario] if c.scenario else []
    a = c.association
    if a is None:
        metric = PERFORMANCE_METRICS[c.metric]
        warnings.append(f"Sem associação mensurável com {metric} nesta campanha.")
    elif a.evidence == "inconclusivo" and a.effect < 0:
        warnings.append(
            f"A associação com {a.metric_name} é inconclusiva e de sinal desfavorável: "
            f"{a.interval()}."
        )
    if a and a.sensitivity:
        warnings.append(f"A classe depende da análise: {a.sensitivity}.")
    warnings += [f"Associada a {other.describe()}." for other in c.others if other.effect < 0]
    estimate, low, high = c.model
    if 0 < low < BORDERLINE_SCORE:
        warnings.append(
            f"Ganho de score limítrofe: o intervalo do modelo vai de {_points(low)} a "
            f"{_points(high)} pontos. É uma candidata experimental, não uma melhoria de "
            "alta confiança."
        )
    if c.support.risk != "baixo":
        warnings.append(
            f"Pouca base no contexto da campanha: só no recorte {c.support.level} há pelo menos "
            f"{c.support.minimum} campanhas de cada lado da comparação."
        )
    change = c.after - c.before
    if change < 0:
        warnings.append(f"O score previsto cai {fmt_number(-change, 1)} pontos com esta edição.")
    elif abs(change - estimate) > 0.05:
        warnings.append(
            "A previsão encosta no limite da escala: a diferença nesta campanha é menor que a do "
            "modelo."
        )
    if c.klass == "C":
        metric = PERFORMANCE_METRICS[c.metric]
        warning = (
            f"Sem evidência suficiente de ganho em {metric}, a métrica do objetivo: é uma "
            "oportunidade de score, não um ganho comprovado nessa métrica."
        )
        favorable = [other.metric_name for other in c.others if other.effect > 0]
        if favorable:
            warning += (
                f" Em outras métricas, a base mostra associação favorável: {_join(favorable)}."
            )
        warnings.append(warning)
    return tuple(warnings)


def _next_step(c: _Candidate) -> str:
    metric = PERFORMANCE_METRICS[c.metric]
    if c.klass == "C":
        return (
            f"Testar em A/B antes de adotar, medindo {metric}: a base não mostra ganho suficiente "
            "nessa métrica."
        )
    return f"Teste A/B da versão {c.action.target_label} contra a atual, medindo {metric}."


def _scenario_warning(campaign: dict, combination: int, minimum: int) -> str | None:
    """Aviso forte quando a própria campanha está numa combinação com pouco histórico."""
    if combination >= minimum:
        return None
    basis = (
        "não foram observados nessa combinação"
        if combination == 0
        else "se apoiam em poucas campanhas dessa combinação"
    )
    return (
        f"Extrapolação: a combinação {campaign['platform']} com vídeo {campaign['format']} "
        f"{_history(combination)}. O score antes e depois e a associação geral {basis}: trate "
        "a recomendação com cautela redobrada."
    )


def _summary(recommendations: tuple[Recommendation, ...], campaign: dict) -> str:
    total = len(recommendations)
    present = _practices_present(campaign)
    if total == 0:
        return f"Nenhuma edição do catálogo tem base para esta campanha. {present}"
    counts = {k: sum(r.recommendation_class == k for r in recommendations) for k in "ABC"}
    parts = [f"{counts[k]} de classe {k}" for k in "ABC" if counts[k]]
    noun = "recomendação" if total == 1 else "recomendações"
    text = f"{total} {noun}: {_join(parts)}."
    if total < 3:
        text += (
            " Há menos de três melhorias defensáveis para esta campanha, e o engine não completa "
            f"a lista com sugestões sem base. {present}"
        )
    return text


def _practices_present(campaign: dict) -> str:
    present = [
        label
        for flag, label in [
            (campaign["has_hook"], "hook"),
            (campaign["has_face"], "rosto humano"),
            (campaign["has_subtitle"] is True, "legenda"),
            (campaign["has_cta"], "call-to-action"),
        ]
        if flag
    ]
    if present:
        text = f"O criativo já tem {_join(present)}"
    else:
        text = "O criativo não tem nenhuma das práticas do catálogo"
    if campaign["text_density"] != "high":
        text += f" e usa texto {'médio' if campaign['text_density'] == 'medium' else 'baixo'}"
    return text + "."


def _points(value: float) -> str:
    """Pontos de score com sinal; perto de zero, duas casas, para não parecer zero."""
    return fmt_signed(value, 2 if 0 < abs(value) < 0.05 else 1)


def _pct(value: float) -> str:
    """Variação percentual com sinal; perto de zero, uma casa, para não parecer zero."""
    return fmt_signed_pct(value, 1 if 0 < abs(value) < 0.005 else 0)


def _history(n: int) -> str:
    """Quantas campanhas do histórico têm a combinação, numa frase completa."""
    if n == 0:
        return "não tem nenhuma campanha no histórico"
    return f"tem {n} campanha{'' if n == 1 else 's'} no histórico"


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} e {items[-1]}"


def _sensitivity(row: pd.Series) -> str | None:
    if bool(row["estavel"]):
        return None
    parts = []
    if row["evidencia_sem_plausibilidade"] != row["evidencia"]:
        level = EVIDENCE_PHRASES[row["evidencia_sem_plausibilidade"]]
        parts.append(f"sem as correções por plausibilidade da Parte 1, fica {level}")
    count_level = row["evidencia_binomial_negativa"]
    if count_level in EVIDENCE_PHRASES and count_level != row["evidencia"]:
        parts.append(
            "com binomial negativa, que mantém as campanhas sem conversão, a associação é de "
            f"{_pct(row['efeito_binomial_negativa'])}, {EVIDENCE_PHRASES[count_level]}"
        )
    return "; ".join(parts) or None


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def _single_campaign(campaign: Mapping | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(campaign, pd.DataFrame):
        frame = campaign
    elif isinstance(campaign, pd.Series):
        frame = campaign.to_frame().T
    else:
        frame = pd.DataFrame([dict(campaign)])
    if len(frame) != 1:
        raise ValueError("O engine analisa uma campanha por vez; use recommend_all para várias.")
    return frame


def _canonical(features: pd.Series) -> dict:
    """Os 13 campos de pré-publicação, já validados, em tipos simples."""
    numbers = {
        col: None if pd.isna(features[col]) else float(features[col])
        for col in ["video_duration_s", "music_voice_ratio"]
    }
    values = {
        **{
            col: str(features[col])
            for col in ["platform", "category", "objective", "target_audience_age"]
        },
        **{col: bool(features[col]) for col in ["is_retargeting", "has_cta", "has_hook"]},
        "has_face": bool(features["has_face"]),
        "format": str(features["format"]),
        "text_density": str(features["text_density"]),
        "has_subtitle": {"sim": True, "não": False}.get(features["subtitle_status"]),
        **numbers,
    }
    return {col: values[col] for col in PRE_FLIGHT_INPUTS}


def _campaign_id(row: pd.Series) -> str | None:
    value = row.get("campaign_id")
    return None if value is None or pd.isna(value) else str(value)


def _observed(row: pd.Series, metric: str, campaign: dict) -> dict | None:
    """Métricas observadas, quando a entrada é uma campanha histórica. Não entram no cálculo."""

    def value(col: str) -> float | None:
        if col not in row.index:
            return None
        number = pd.to_numeric(row[col], errors="coerce")
        return None if pd.isna(number) else float(number)

    watch, conversions, impressions = (
        value(col) for col in ["avg_watch_time_s", "conversions", "impressions"]
    )
    duration = campaign["video_duration_s"]
    rates = {
        "ctr": value("ctr"),
        "engagement_rate": value("engagement_rate"),
        "completion_rate": min(watch / duration, 1.0) if watch is not None and duration else None,
        "conversions_per_mille": (
            conversions / impressions * 1000 if conversions is not None and impressions else None
        ),
        "roas_non_branding": value("roas") if campaign["category"] != "Branding" else None,
    }
    observed = {TARGET: value(TARGET), metric: rates[metric]}
    return observed if any(v is not None for v in observed.values()) else None


# --------------------------------------------------------------------------- #
# Atalho
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _default_engine() -> RecommendationEngine:
    return RecommendationEngine.default()


def recommend(
    campaign: Mapping | pd.Series | pd.DataFrame, *, metric: str | None = None
) -> RecommendationReport:
    """Recomendações para uma campanha, com o engine padrão (modelo final e evidências salvas)."""
    return _default_engine().recommend(campaign, metric=metric)
