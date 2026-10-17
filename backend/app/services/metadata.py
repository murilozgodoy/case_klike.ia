"""Metadados para a interface: campos, opções, métricas, indicadores e exemplos guiados.

As opções vêm das listas do próprio pacote, e os indicadores, dos resultados já calculados:
a certificação do modelo (Parte 2) e a avaliação do engine nas 500 campanhas (Parte 3).
"""

from __future__ import annotations

import pandas as pd

from klike.config import CATEGORY_ORDERS, COLUMN_LABELS, CONTEXT_COLS, CREATIVE_COLS
from klike.engine import CLASS_LABELS, METHOD_NOTE, OBJECTIVE_METRICS, class_label
from klike.evidence import PERFORMANCE_METRICS
from klike.modeling import OPTIONAL_INPUTS, PRE_FLIGHT_INPUTS
from klike.predictor import CERTIFIED_METRICS

# Limites numéricos para a interface; a validação definitiva é a do pacote (prepare_features).
NUMBER_LIMITS = {
    "video_duration_s": {"minimo": 0, "minimo_exclusivo": True, "unidade": "segundos"},
    "music_voice_ratio": {"minimo": 0, "maximo": 1, "passo": 0.01},
}

# Campanhas reais da base simulada que mostram comportamentos diferentes do engine.
# Um teste da API confere que cada uma continua mostrando o que a descrição diz.
GUIDED_EXAMPLES = [
    {
        "campaign_id": "KLK-0024",
        "titulo": "As três classes de recomendação",
        "descricao": (
            "Campanha de conversões na Meta: hook e rosto com evidência robusta (A), texto "
            "com evidência sensível (B) e o formato vertical só como hipótese de score (C)."
        ),
    },
    {
        "campaign_id": "KLK-0291",
        "titulo": "Score alto, nenhuma recomendação",
        "descricao": (
            "O criativo já segue o catálogo e o score observado é 89,1: o engine diz que não "
            "há edição com base, em vez de completar a lista."
        ),
    },
    {
        "campaign_id": "KLK-0001",
        "titulo": "Música/voz: retenção e score em direções opostas",
        "descricao": (
            "Campanha de awareness em que mais música se associa a mais retenção, enquanto o "
            "score previsto cai: a recomendação mostra as duas coisas."
        ),
    },
]


def _field(col: str) -> dict:
    if col in CATEGORY_ORDERS:
        kind = "categoria"
    elif col in NUMBER_LIMITS:
        kind = "numero"
    else:
        kind = "booleano"
    return {
        "campo": col,
        "rotulo": COLUMN_LABELS[col],
        "grupo": "contexto" if col in CONTEXT_COLS else "criativo",
        "tipo": kind,
        "opcoes": CATEGORY_ORDERS.get(col),
        "obrigatorio": col not in OPTIONAL_INPUTS,
        **NUMBER_LIMITS.get(col, {}),
    }


def _engine_evaluation(catalog: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    n = catalog["recomendacoes"]
    with_evidence = catalog["A"] + catalog["B"]
    return {
        "descricao": (
            "Avaliação funcional do engine nas 500 campanhas da base: mede cobertura e "
            "composição, não o impacto das edições."
        ),
        "recomendacoes": len(evaluation),
        "campanhas_tres_ou_mais": int((n >= 3).sum()),
        "parcela_tres_ou_mais": float((n >= 3).mean()),
        "campanhas_com_evidencia": int((with_evidence >= 1).sum()),
        "campanhas_tres_ou_mais_com_evidencia": int((with_evidence >= 3).sum()),
        "campanhas_sem_recomendacao": int((n == 0).sum()),
        "classes": {k: int((evaluation["classe"] == k).sum()) for k in CLASS_LABELS},
    }


def build_metadata(catalog: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    return {
        "campos": [_field(col) for col in PRE_FLIGHT_INPUTS],
        "atributos_contexto": CONTEXT_COLS,
        "atributos_criativo": CREATIVE_COLS,
        "metricas": PERFORMANCE_METRICS,
        "metrica_por_objetivo": OBJECTIVE_METRICS,
        "classes": {k: class_label(k) for k in CLASS_LABELS},
        "nota_metodologica": METHOD_NOTE,
        "indicadores": {
            "campanhas": len(catalog),
            "modelo": {
                "descricao": (
                    "Certificação da Parte 2: modelo treinado nas 400 campanhas de "
                    "desenvolvimento e avaliado uma única vez nas 100 reservadas."
                ),
                "mae": CERTIFIED_METRICS["mae"],
                "mae_ic95": CERTIFIED_METRICS["mae_ic95"],
                "mae_dummy": CERTIFIED_METRICS["mae_dummy"],
                "r2": CERTIFIED_METRICS["r2"],
            },
            "avaliacao_engine": _engine_evaluation(catalog, evaluation),
        },
        "exemplos": GUIDED_EXAMPLES,
    }
