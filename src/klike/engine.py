"""Recommendations engine: edições sugeridas para um criativo, com evidência e prioridade.

Recebe uma campanha, com os 13 atributos de pré-publicação; métricas observadas de uma campanha
histórica são opcionais. Devolve as edições do vídeo que a base sustenta, em ordem de
prioridade. Para cada edição, quatro informações ficam separadas:

- diferença prevista de score: o preditor da Parte 2 calcula o score da campanha antes e depois
  da edição, dentro da escala de 0 a 100; o intervalo do modelo vem de ``score_differences.csv``;
- associação com performance: o contraste da edição na métrica do objetivo da campanha, ou na
  métrica escolhida, lido de ``recommendation_evidence.csv``, sem recalcular nada;
- classe de evidência: A, B ou C, pela regra registrada no notebook 04;
- suporte histórico: quantas campanhas comparáveis têm o valor atual e o sugerido.

O relatório abre com um resumo executivo: as três primeiras recomendações, com as ressalvas de
cada uma em rótulos curtos. Os avisos completos continuam na parte detalhada.

Nada disso é efeito causal: cada recomendação é uma hipótese para validar com teste A/B.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache

import pandas as pd

from klike.config import CONTEXT_COLS, TARGET
from klike.evidence import PERFORMANCE_METRICS, load_performance_evidence, load_score_differences
from klike.formatting import fmt_number, fmt_signed, fmt_signed_pct
from klike.modeling import PRE_FLIGHT_INPUTS, prepare_features
from klike.predictor import KlikeScorePredictor, train_final_model
from klike.support import MUSIC_HUNDREDTHS, HistoricalSupport, to_hundredths

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
# Música/voz: a edição soma 0,1 à proporção, o contraste exato da família de evidências. O valor
# sugerido não passa de 0,67, o percentil 90 da base: um limite contra extrapolação, não uma
# proporção ideal.
MUSIC_STEP = 0.1
MUSIC_MAX_TARGET = 0.67
# Quantas recomendações o resumo executivo mostra, no início do relatório.
EXECUTIVE_SIZE = 3

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
# Rótulos curtos dos avisos, na ordem em que o resumo executivo os mostra. Cada aviso completo
# tem exatamente um rótulo, gerado junto com ele; sem nenhum aviso, vale a ressalva geral.
CAVEATS = (
    "extrapolação",
    "score previsto cai",
    "score no limite da escala",
    "sem evidência na métrica",
    "sem associação mensurável",
    "classe sensível à análise",
    "estimativa desfavorável e inconclusiva",
    "desfavorável em outra métrica",
    "ganho de score limítrofe",
    "pouca base no contexto (só na plataforma)",
    "pouca base no contexto (só na base inteira)",
)
NO_CAVEAT = "associação, não causa (validar em A/B)"
# O rótulo de pouca base diz o recorte mais amplo que foi preciso usar, pelo risco do suporte.
SUPPORT_CAVEATS = {
    "moderado": "pouca base no contexto (só na plataforma)",
    "alto": "pouca base no contexto (só na base inteira)",
}


def class_label(klass: str, manual_metric: bool = False) -> str:
    """Descrição da classe. Com a métrica trocada manualmente, a C fala da métrica selecionada."""
    if klass == "C":
        return f"oportunidade de score, sem evidência suficiente na {_metric_ref(manual_metric)}"
    return CLASS_LABELS[klass]


def _metric_ref(manual_metric: bool) -> str:
    """Como os textos chamam a métrica avaliada: a do objetivo ou a escolhida na chamada."""
    return "métrica selecionada" if manual_metric else "métrica do objetivo"


@dataclass(frozen=True)
class Action:
    """Uma edição do catálogo: o campo, o valor atual e o sugerido, e como descrevê-la.

    Os rótulos completam a frase "N campanhas ..." (``sem hook``, ``com texto alto``); a
    ``phrase`` descreve a edição numa justificativa. ``metrics`` limita as métricas em que a
    edição é avaliada; sem ele, vale para todas.
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
    metrics: tuple[str, ...] | None = None

    def evaluated_for(self, metric: str) -> bool:
        return self.metrics is None or metric in self.metrics

    def for_campaign(self, campaign: Mapping) -> Action:
        """A edição com os valores desta campanha. Numa edição categórica, ela mesma."""
        return self

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

    def support_counts(self, support: HistoricalSupport, context: Mapping) -> tuple[int, int]:
        """Campanhas do recorte com o valor atual e com o sugerido."""
        column, current, target = self.support_condition
        return (
            support.count(**context, **{column: current}),
            support.count(**context, **{column: target}),
        )

    @property
    def support_labels(self) -> tuple[str, str]:
        """Como o suporte descreve os dois grupos de campanhas comparados."""
        return self.current_label, self.target_label

    @property
    def support_note(self) -> str | None:
        """Como o suporte aproxima os dois lados, quando não conta valores exatos."""
        return None

    def rationale_note(self, association: Association | None) -> str | None:
        """Ressalva própria da edição, acrescentada à justificativa."""
        return None

    def next_step_note(self) -> str | None:
        """Cuidado próprio da edição no teste A/B."""
        return None


@dataclass(frozen=True)
class MusicIncrease(Action):
    """Somar ``step`` à proporção música/voz, sem passar de ``maximum``.

    É a única edição contínua: o valor atual e o sugerido vêm de cada campanha. O valor exato
    decide se a edição se aplica e entra na previsão. Os centésimos inteiros servem só para
    centrar as janelas do suporte histórico, uma aproximação dos dois lados da comparação: o
    valor atual e o sugerido raramente aparecem exatos na base.
    """

    step: float = MUSIC_STEP
    maximum: float = MUSIC_MAX_TARGET

    def applies_to(self, campaign: Mapping) -> bool:
        """Música/voz informada e, no valor exato, até ``maximum - step``.

        As seis casas tiram só o resíduo binário: 0,570 se aplica; 0,571 e 0,574 não.
        """
        value = campaign[self.field]
        limit = round((self.maximum - self.step) * 100, 6)
        return value is not None and round(value * 100, 6) <= limit

    def apply(self, campaign: Mapping) -> dict:
        return {**campaign, self.field: _plus(campaign[self.field], self.step)}

    def for_campaign(self, campaign: Mapping) -> MusicIncrease:
        current = campaign[self.field]
        target = _plus(current, self.step)
        return replace(
            self,
            title=f"Aumentar a proporção música/voz de {_ratio(current)} para {_ratio(target)}",
            current=current,
            target=target,
            current_label=f"com música/voz {_ratio(current)}",
            target_label=f"com música/voz {_ratio(target)}",
        )

    def support_counts(self, support: HistoricalSupport, context: Mapping) -> tuple[int, int]:
        (low, high), (target_low, target_high) = self._windows()
        return (
            support.count_between(MUSIC_HUNDREDTHS, low, high, **context),
            support.count_between(MUSIC_HUNDREDTHS, target_low, target_high, **context),
        )

    @property
    def support_labels(self) -> tuple[str, str]:
        current, target = (
            f"com música/voz entre {_hundredths(low)} e {_hundredths(high)}"
            for low, high in self._windows()
        )
        return current, target

    @property
    def support_note(self) -> str:
        return f"janelas de {_hundredths(self._width())} em torno do valor atual e do sugerido"

    def rationale_note(self, association: Association | None) -> str:
        return (
            f"A regressão usa as {association.n} campanhas da base, de todos os objetivos, com "
            f"{association.metric_name} e música/voz registradas, e trata a música como termo "
            "linear. Essa amostra não se confunde com as contagens do suporte histórico, que "
            "só descrevem a base de comparação no recorte. Retenção é a fração média do vídeo "
            "assistida (tempo médio assistido ÷ duração, até 100%), uma proxy de atenção. O "
            f"valor sugerido nunca passa de {_ratio(self.maximum)}, o percentil 90 da base: é "
            "um limite contra extrapolação, não uma proporção ideal."
        )

    def next_step_note(self) -> str:
        return (
            "O ganho associado é pequeno: o teste precisa de amostra suficiente para detectá-lo. "
            "Confira também se a narração, com menos espaço, continua passando a mensagem."
        )

    def _width(self) -> int:
        return round(self.step * 100)

    def _windows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Janelas adjacentes, em centésimos: [x − 5, x + 4] e [x + 5, x + 14] para 0,1."""
        width = self._width()
        start = to_hundredths(self.current) - width // 2
        return (start, start + width - 1), (start + width, start + 2 * width - 1)


def _plus(value: float, step: float) -> float:
    """Soma sem o resíduo binário: 0,57 + 0,1 dá 0,6699999999999999 em ponto flutuante."""
    return round(value + step, 10)


def _ratio(value: float) -> str:
    """Música/voz com as casas que o valor tem, no mínimo duas: 0,40; 0,39; 0,571."""
    decimals = max(2, len(f"{value:.6f}".rstrip("0").split(".")[1]))
    return fmt_number(value, decimals)


def _hundredths(value: int) -> str:
    """Centésimos inteiros como proporção: 37 → 0,37. Abaixo de zero, a janela começa em 0."""
    return fmt_number(max(value, 0) / 100, 2)


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
    # Só na retenção, a única métrica em que a música tem associação; os valores vêm de cada
    # campanha (``for_campaign``).
    MusicIncrease(
        "musica",
        "Aumentar a proporção música/voz em 0,1",
        "music_voice_ratio",
        None,
        None,
        "com a música/voz atual",
        "com 0,1 a mais de música",
        "aumentar a proporção música/voz em 0,1",
        metrics=("completion_rate",),
    ),
]
# Edições avaliadas na Etapa 1 que o engine não recomenda (notebook 04, seção 8).
DISCARDED = {
    "texto_alto_baixo": "citada como alternativa na recomendação de texto alto → médio",
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
    current_label: str  # os dois grupos comparados, como "sem hook" e "com hook"
    target_label: str
    note: str | None = None  # como os grupos aproximam os valores, quando não são exatos

    def describe(self) -> str:
        """Os dois lados da comparação e o recorte, numa frase."""
        note = f"; {self.note}" if self.note else ""
        return (
            f"{self.n_current} campanhas {self.current_label} e {self.n_target} "
            f"{self.target_label} (recorte: {self.level}{note})"
        )

    def to_dict(self) -> dict:
        payload = {
            "recorte": self.level,
            "contexto": self.context,
            "campanhas_valor_atual": self.n_current,
            "campanhas_valor_sugerido": self.n_target,
            "grupo_valor_atual": self.current_label,
            "grupo_valor_sugerido": self.target_label,
            "minimo_por_lado": self.minimum,
            "risco_extrapolacao": self.risk,
            "observacao": (
                "campanhas comparáveis no recorte: descrevem a base de comparação, não uma "
                "estimativa do efeito nesse recorte"
            ),
        }
        if self.note:
            payload["aproximacao"] = self.note
        return payload


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
    caveats: tuple[str, ...] = (NO_CAVEAT,)  # rótulos curtos dos avisos, para o resumo
    manual_metric: bool = False

    @property
    def score_change(self) -> float:
        return self.score_after - self.score_before

    @property
    def estimated_impact(self) -> str:
        """O impacto estimado, separado da prioridade: na métrica avaliada ou só no score."""
        if self.recommendation_class in ("A", "B"):
            return self.association.describe()
        change = fmt_signed(self.score_change, 1)
        return f"Klike Score {change} pontos; sem evidência suficiente em {self.metric_name}"

    @property
    def headline(self) -> str:
        """Impacto e score numa linha, para o resumo executivo."""
        if self.recommendation_class == "C":
            return self.estimated_impact  # a classe C já se mede pelo score
        return f"{self.estimated_impact} · score {fmt_signed(self.score_change, 1)}"

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
            "classe_descricao": class_label(self.recommendation_class, self.manual_metric),
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
            "ressalvas": list(self.caveats),
            "proximo_passo": self.next_step,
        }

    def executive_entry(self) -> dict:
        """A recomendação no resumo executivo: o essencial e as ressalvas curtas."""
        return {
            "prioridade": self.priority,
            "acao": self.action,
            "titulo": self.title,
            "classe": self.recommendation_class,
            "metrica_nome": self.metric_name,
            "impacto_estimado": self.estimated_impact,
            "diferenca_score": round(self.score_change, 2),
            "ressalvas": list(self.caveats),
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

    @property
    def objective_metric(self) -> str:
        """A métrica principal do objetivo da campanha, pela premissa de ``OBJECTIVE_METRICS``."""
        return OBJECTIVE_METRICS[self.objective]

    @property
    def metric_is_manual(self) -> bool:
        """A métrica avaliada foi escolhida na chamada, no lugar da principal do objetivo."""
        return self.metric != self.objective_metric

    @property
    def executive(self) -> tuple[Recommendation, ...]:
        """As recomendações do resumo executivo: as primeiras, na ordem do engine."""
        return self.recommendations[:EXECUTIVE_SIZE]

    def composition(self) -> dict:
        """Quantas recomendações há, e quantas têm evidência na métrica avaliada."""
        counts = {k: sum(r.recommendation_class == k for r in self.recommendations) for k in "ABC"}
        total = len(self.recommendations)
        return {
            "recomendacoes": total,
            "com_evidencia_na_metrica": counts["A"] + counts["B"],
            "hipoteses_de_score": counts["C"],
            "por_classe": counts,
            "menos_de_tres": total < 3,
        }

    def metric_line(self) -> str:
        """A métrica avaliada, dizendo quando ela não é a principal do objetivo."""
        if not self.metric_is_manual:
            return f"Métrica do objetivo: {self.metric_name}"
        principal = PERFORMANCE_METRICS[self.objective_metric]
        return (
            f"Métrica avaliada: {self.metric_name} (selecionada manualmente; a métrica principal "
            f"de {self.objective} é {principal})"
        )

    def to_dict(self) -> dict:
        return {
            "campanha_id": self.campaign_id,
            "campanha": self.campaign,
            "contexto": self.context,
            "objetivo": self.objective,
            "metrica": self.metric,
            "metrica_nome": self.metric_name,
            "metrica_principal_do_objetivo": self.objective_metric,
            "metrica_selecionada_manualmente": self.metric_is_manual,
            "score_previsto": round(self.predicted_score, 2),
            "observado": self.observed,
            "campanhas_na_mesma_combinacao": self.combination_support,
            "resumo": self.summary,
            "composicao": self.composition(),
            "principais": [r.executive_entry() for r in self.executive],
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
                    f"{r.support.n_current} {r.support.current_label}, {r.support.n_target} "
                    f"{r.support.target_label} ({r.support.level})"
                ),
                "ressalvas": " · ".join(r.caveats),
                "avisos": len(r.warnings),
            }
            for r in self.recommendations
        ]
        return pd.DataFrame(rows).set_index("prioridade") if rows else pd.DataFrame()

    def to_text(self) -> str:
        """Relatório em texto, para a linha de comando: resumo executivo e depois o detalhe."""
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
        lines = [header, self.metric_line(), score, "", self.summary]
        if self.executive:
            total, shown = len(self.recommendations), len(self.executive)
            count = "" if shown == total else f" ({shown} de {total})"
            lines += ["", f"Principais recomendações{count}:"]
            for r in self.executive:
                lines += [
                    f"   {r.priority}. {r.title} · classe {r.recommendation_class}",
                    f"      {r.headline}",
                    f"      Ressalvas: {' · '.join(r.caveats)}",
                ]
            lines += ["", "Em detalhe:"]
        for r in self.recommendations:
            _, low, high = r.model_change
            lines += [
                "",
                f"{r.priority}. {r.title}  [classe {r.recommendation_class}: "
                f"{class_label(r.recommendation_class, r.manual_metric)}]",
                f"   Impacto estimado: {r.estimated_impact}",
                f"   Score previsto: {fmt_number(r.score_before, 1)} → "
                f"{fmt_number(r.score_after, 1)} ({fmt_signed(r.score_change, 1)}; intervalo do "
                f"modelo: {_points(low)} a {_points(high)})",
                f"   Suporte histórico contextual: {r.support.describe()}",
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

        ``metric`` troca a métrica do objetivo por outra de ``PERFORMANCE_METRICS``, e a saída
        diz que ela foi escolhida manualmente. Uma edição com escopo de métrica, como a
        música/voz, só é avaliada nas métricas do escopo.
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

        applicable = [
            action.for_campaign(current)
            for action in ACTIONS
            if action.evaluated_for(metric) and action.applies_to(current)
        ]
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
            summary=_summary(ranked, current, metric != OBJECTIVE_METRICS[current["objective"]]),
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
        manual = metric != OBJECTIVE_METRICS[current["objective"]]
        ref = _metric_ref(manual)
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
                f"sem evidência suficiente na {ref}, e o intervalo do score do modelo não fica "
                "acima de zero"
            )
        elif after <= before:
            return (
                f"sem evidência suficiente na {ref}, e o score previsto já está no limite da escala"
            )
        else:
            klass = "C"
        if klass == "C" and combination == 0:
            return (
                f"sugestão só de score numa combinação sem histórico ({current['platform']} com "
                f"vídeo {current['format']}): sem evidência na {ref}, o ganho previsto não tem "
                "base observada"
            )
        support = self._support(action, current)
        if support is None:
            return f"menos de {self.min_support} campanhas de cada lado mesmo no recorte mais amplo"
        return _Candidate(
            action=action,
            klass=klass,
            metric=metric,
            manual_metric=manual,
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
        A música/voz é contada em janelas em torno do valor atual e do sugerido.
        """
        levels = SUPPORT_LEVELS[:-1] if action.platform else SUPPORT_LEVELS
        for name, columns, risk in levels:
            context = {col: campaign[col] for col in columns}
            n_current, n_target = action.support_counts(self.predictor.support, context)
            if min(n_current, n_target) >= self.min_support:
                return Support(
                    name,
                    context,
                    n_current,
                    n_target,
                    risk,
                    self.min_support,
                    *action.support_labels,
                    note=action.support_note,
                )
        return None

    def _campaign_notes(self, campaign: dict) -> tuple[str, ...]:
        notes = []
        if campaign["video_duration_s"] is None:
            notes.append("Duração não informada: o modelo usa a mediana das campanhas de treino.")
        if campaign["music_voice_ratio"] is None:
            notes.append(
                "Proporção música/voz não informada: o modelo usa a mediana das campanhas de "
                "treino, e o engine não sugere mudá-la."
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
    manual_metric: bool  # a métrica foi escolhida na chamada, não pelo objetivo
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
    warnings = _warnings(c)
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
        warnings=tuple(text for _, text in warnings),
        next_step=_next_step(c),
        caveats=_caveats(label for label, _ in warnings),
        manual_metric=c.manual_metric,
    )


def _caveats(labels: Iterable[str]) -> tuple[str, ...]:
    """Rótulos curtos, sem repetição e na ordem de ``CAVEATS``; sem aviso, a ressalva geral."""
    present = set(labels)
    return tuple(label for label in CAVEATS if label in present) or (NO_CAVEAT,)


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
            f"a {_metric_ref(c.manual_metric)}: {a.interval()}, {EVIDENCE_PHRASES[a.evidence]}."
        ]
    else:
        parts = [f"O modelo prevê mais score; não há como medir a associação com {metric}."]
    note = action.rationale_note(a)
    if note:
        parts.append(note)
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
            "Oportunidade de score: fica depois das recomendações com evidência na "
            f"{_metric_ref(c.manual_metric)}."
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


def _warnings(c: _Candidate) -> list[tuple[str, str]]:
    """Avisos da recomendação, cada um com o seu rótulo curto (``CAVEATS``)."""
    warnings = [("extrapolação", c.scenario)] if c.scenario else []
    a = c.association
    metric = PERFORMANCE_METRICS[c.metric]
    if a is None:
        warnings.append(
            ("sem associação mensurável", f"Sem associação mensurável com {metric} nesta campanha.")
        )
    elif a.evidence == "inconclusivo" and a.effect < 0:
        warnings.append(
            (
                "estimativa desfavorável e inconclusiva",
                f"A associação com {a.metric_name} é inconclusiva e de sinal desfavorável: "
                f"{a.interval()}.",
            )
        )
    if a and a.sensitivity:
        warnings.append(
            ("classe sensível à análise", f"A classe depende da análise: {a.sensitivity}.")
        )
    warnings += [
        ("desfavorável em outra métrica", f"Associada a {other.describe()}.")
        for other in c.others
        if other.effect < 0
    ]
    estimate, low, high = c.model
    if 0 < low < BORDERLINE_SCORE:
        warnings.append(
            (
                "ganho de score limítrofe",
                f"Ganho de score limítrofe: o intervalo do modelo vai de {_points(low)} a "
                f"{_points(high)} pontos. É uma candidata experimental, não uma melhoria de "
                "alta confiança.",
            )
        )
    if c.support.risk != "baixo":
        warnings.append(
            (
                SUPPORT_CAVEATS[c.support.risk],
                f"Pouca base no contexto da campanha: só no recorte {c.support.level} há pelo "
                f"menos {c.support.minimum} campanhas de cada lado da comparação.",
            )
        )
    # Os dois avisos de score são independentes: perto de 0, uma queda pode ser cortada pela
    # escala e continuar sendo queda.
    change = c.after - c.before
    if change < 0:
        warnings.append(
            (
                "score previsto cai",
                f"O score previsto cai {_magnitude(change)} pontos com esta edição: o modelo do "
                f"Klike Score e a associação com {metric} apontam em direções opostas.",
            )
        )
    if abs(change - estimate) > 0.05:
        warnings.append(
            (
                "score no limite da escala",
                "A previsão encosta no limite da escala: nesta campanha, a diferença de score é "
                "menor, em módulo, que a do modelo.",
            )
        )
    if c.klass == "C":
        warning = (
            f"Sem evidência suficiente de ganho em {metric}, a {_metric_ref(c.manual_metric)}: "
            "é uma oportunidade de score, não um ganho comprovado nessa métrica."
        )
        favorable = [other.metric_name for other in c.others if other.effect > 0]
        if favorable:
            warning += (
                f" Em outras métricas, a base mostra associação favorável: {_join(favorable)}."
            )
        warnings.append(("sem evidência na métrica", warning))
    return warnings


def _next_step(c: _Candidate) -> str:
    metric = PERFORMANCE_METRICS[c.metric]
    if c.klass == "C":
        return (
            f"Testar em A/B antes de adotar, medindo {metric}: a base não mostra ganho suficiente "
            "nessa métrica."
        )
    step = f"Teste A/B da versão {c.action.target_label} contra a atual, medindo {metric}."
    note = c.action.next_step_note()
    return f"{step} {note}" if note else step


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


def _summary(
    recommendations: tuple[Recommendation, ...], campaign: dict, manual_metric: bool = False
) -> str:
    """Composição da lista: quantas têm evidência na métrica avaliada e quantas são só de score.

    Uma oportunidade de score (C) nunca é chamada de melhoria.
    """
    total = len(recommendations)
    present = _practices_present(campaign)
    if total == 0:
        return f"Nenhuma edição do catálogo tem base para esta campanha. {present}"
    counts = {k: sum(r.recommendation_class == k for r in recommendations) for k in "ABC"}
    evidence, scores = counts["A"] + counts["B"], counts["C"]
    ref = _metric_ref(manual_metric)
    noun = "recomendação" if total == 1 else "recomendações"
    if evidence:
        classes = _join([f"{counts[k]} de classe {k}" for k in "AB" if counts[k]])
    if evidence and scores:
        hypotheses = "hipótese" if scores == 1 else "hipóteses"
        text = (
            f"{total} {noun}: {evidence} com evidência na {ref} ({classes}) e {scores} "
            f"{hypotheses} de score (classe C)."
        )
    elif evidence:
        every = "todas " if total > 1 else ""
        text = f"{total} {noun}, {every}com evidência na {ref} ({classes})."
    else:
        every = "todas hipóteses" if total > 1 else "hipótese"
        text = f"{total} {noun}, {every} de score (classe C), sem evidência suficiente na {ref}."
    if total < 3:
        text += (
            " Há menos de três recomendações defensáveis para esta campanha, e o engine não "
            f"completa a lista com sugestões sem base. {present}"
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


def _magnitude(value: float) -> str:
    """Pontos de score sem sinal; perto de zero, duas casas, para não parecer zero."""
    return fmt_number(abs(value), 2 if 0 < abs(value) < 0.05 else 1)


def _pct(value: float) -> str:
    """Variação percentual com sinal, sem arredondar duas vezes.

    Perto de zero, uma casa, para não parecer zero. Quando o valor registrado cai exatamente
    na metade, ganha uma casa: o rosto no CTR, registrado como 0,2350, sai +23,5%, e não +24%,
    porque a estimativa exata é 23,496%.
    """
    decimals = 1 if 0 < abs(value) < 0.005 else 0
    if round(abs(value) * 100 * 10**decimals, 6) % 1 == 0.5:
        decimals += 1
    return fmt_signed_pct(value, decimals)


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
