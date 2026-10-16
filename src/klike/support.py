"""Suporte histórico: quantas campanhas do treino têm cada combinação de atributos.

O modelo prevê um score para qualquer combinação válida de atributos, mesmo as que quase
não aparecem no histórico, como vídeo horizontal no TikTok. Nesses casos, a previsão é uma
extrapolação. O suporte guarda os atributos de pré-publicação das campanhas de treino, sem
score nem métricas, para contar quantas campanhas parecidas existem.

A contagem não exige que as 13 features coincidam: pode olhar só a plataforma e o formato,
ou o contexto inteiro, e alargar o recorte quando houver poucas campanhas parecidas. É a
base para o recommendations engine (Parte 3) sinalizar incerteza e evitar recomendações
excessivamente confiantes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from klike.modeling import BINARY_FEATURES, CATEGORICAL_FEATURES, prepare_features

# Os números viram faixas, porque campanhas parecidas raramente têm a mesma duração exata.
# Duração: as mesmas faixas da Parte 1 (notebook 02). Música/voz: terços, de 0 (só voz) a
# 1 (só música).
DURATION_BANDS = (0, 10, 15, 20, 30, np.inf)
DURATION_LABELS = ["até 10 s", "11 a 15 s", "16 a 20 s", "21 a 30 s", "mais de 30 s"]
MUSIC_BANDS = (0, 1 / 3, 2 / 3, 1)
MUSIC_LABELS = ["predomina a voz", "equilibrado", "predomina a música"]
UNKNOWN_BAND = "desconhecida"

SUPPORT_COLUMNS = [*CATEGORICAL_FEATURES, *BINARY_FEATURES, "duration_band", "music_band"]
# Colunas numéricas, guardadas em inteiros e contadas por intervalo (``count_between``). A
# música/voz em centésimos permite janelas em torno de um valor sem os erros de ponto
# flutuante nas fronteiras: 0,17 − 0,05 dá 0,12000000000000001.
MUSIC_HUNDREDTHS = "musica_centesimos"
RANGE_COLUMNS = [MUSIC_HUNDREDTHS]


def to_hundredths(value: float) -> int:
    """Música/voz em centésimos inteiros, com a metade arredondada para cima: 0,325 → 33.

    O arredondamento a 6 casas antes tira só o resíduo binário (0,29 × 100 dá
    28,999999999999996). Os valores do histórico têm duas casas e ficam exatos.
    """
    return math.floor(round(value * 100, 6) + 0.5)


def campaign_profiles(campaigns: pd.DataFrame | Mapping | Sequence[Mapping]) -> pd.DataFrame:
    """Atributos de pré-publicação, prontos para contar campanhas parecidas.

    Valida a entrada como na previsão. A legenda aparece como sim, não ou desconhecido; as
    binárias como verdadeiro ou falso; duração e música/voz em faixas, com uma faixa própria
    para o valor ausente. Essas colunas são categóricas com a lista completa de valores, então
    as contagens mostram também as combinações sem nenhuma campanha. A música/voz também fica
    em centésimos inteiros, ausente quando não informada, para as contagens por intervalo.
    """
    features = prepare_features(campaigns, strict=True)
    profiles = features[CATEGORICAL_FEATURES].copy()
    for col in BINARY_FEATURES:
        profiles[col] = pd.Categorical(features[col].astype(bool), categories=[False, True])
    profiles["duration_band"] = _band(features["video_duration_s"], DURATION_BANDS, DURATION_LABELS)
    profiles["music_band"] = _band(features["music_voice_ratio"], MUSIC_BANDS, MUSIC_LABELS)
    music = features["music_voice_ratio"]
    profiles[MUSIC_HUNDREDTHS] = pd.array(
        [None if pd.isna(value) else to_hundredths(value) for value in music], dtype="Int64"
    )
    return profiles[[*SUPPORT_COLUMNS, *RANGE_COLUMNS]]


def _band(values: pd.Series, bins: Sequence[float], labels: list[str]) -> pd.Categorical:
    bands = pd.cut(values, bins=list(bins), labels=labels, include_lowest=True)
    return bands.cat.add_categories(UNKNOWN_BAND).fillna(UNKNOWN_BAND)


@dataclass(frozen=True, eq=False)
class HistoricalSupport:
    """Atributos das campanhas de treino, para contar campanhas parecidas em qualquer recorte."""

    profiles: pd.DataFrame

    @classmethod
    def from_campaigns(
        cls, campaigns: pd.DataFrame | Mapping | Sequence[Mapping]
    ) -> HistoricalSupport:
        return cls(campaign_profiles(campaigns).reset_index(drop=True))

    def count(self, **conditions) -> int:
        """Campanhas do treino com todos os valores pedidos. Sem condições, o total do treino.

        Exemplo: ``count(platform="TikTok", format="horizontal")``.
        """
        return int(self._matches(conditions).sum())

    def count_between(self, column: str, low: int, high: int, **conditions) -> int:
        """Campanhas com ``column`` entre ``low`` e ``high``, inclusive, e os valores pedidos.

        Para as colunas numéricas, em inteiros. Valores ausentes ficam de fora. Exemplo, a
        música/voz entre 0,27 e 0,36 no TikTok:
        ``count_between("musica_centesimos", 27, 36, platform="TikTok")``.
        """
        if column not in RANGE_COLUMNS:
            raise ValueError(
                f"{column!r} não é contada por intervalo (aceitas: {', '.join(RANGE_COLUMNS)})"
            )
        values = self.profiles[column]
        inside = ((values >= low) & (values <= high)).fillna(False).to_numpy(dtype=bool)
        return int((self._matches(conditions) & inside).sum())

    def table(self, *columns: str) -> pd.Series:
        """Campanhas por combinação das colunas pedidas, inclusive as que não têm nenhuma."""
        for column in columns:
            self._check_column(column)
        return self.profiles.groupby(list(columns), observed=False).size().rename("campanhas")

    def counts_for(self, campaign: Mapping, levels: Sequence[Sequence[str]]) -> pd.Series:
        """Campanhas iguais a esta em cada recorte, do mais específico ao mais amplo.

        Cada nível lista as colunas que precisam coincidir; uma lista vazia é o treino
        inteiro. Serve para alargar o contexto quando há poucas campanhas parecidas.
        """
        profile = campaign_profiles(campaign).iloc[0]
        counts = {
            " + ".join(level) or "todas": self.count(**{col: profile[col] for col in level})
            for level in levels
        }
        return pd.Series(counts, name="campanhas")

    def _matches(self, conditions: Mapping) -> np.ndarray:
        matches = np.ones(len(self.profiles), dtype=bool)
        for column, value in conditions.items():
            self._check_value(column, value)
            matches &= (self.profiles[column] == value).to_numpy()
        return matches

    def _check_column(self, column: str) -> None:
        if column in RANGE_COLUMNS:
            raise ValueError(f"{column!r} é numérica: conte por intervalo com count_between")
        if column not in SUPPORT_COLUMNS:
            accepted = ", ".join(SUPPORT_COLUMNS)
            raise ValueError(f"Coluna de suporte desconhecida: {column!r} (aceitas: {accepted})")

    def _check_value(self, column: str, value) -> None:
        self._check_column(column)
        allowed = list(self.profiles[column].cat.categories)
        if value not in allowed:
            raise ValueError(f"{column}: valor desconhecido {value!r} (aceitos: {allowed})")
