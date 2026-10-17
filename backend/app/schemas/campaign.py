"""Contratos de campanha: a entrada de uma campanha nova e as respostas da listagem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CampaignInput(BaseModel):
    """Os 13 atributos de pré-publicação. Valores das categorias são validados pelo pacote.

    Duração, legenda e música/voz podem faltar, como no histórico; legenda nula quer dizer
    sem marcação, e não "sem legenda".
    """

    model_config = ConfigDict(extra="forbid")

    platform: str
    category: str
    objective: str
    target_audience_age: str
    is_retargeting: bool
    video_duration_s: float | None = None
    format: str
    has_subtitle: bool | None = None
    has_cta: bool
    has_hook: bool
    has_face: bool
    text_density: str
    music_voice_ratio: float | None = None


class CampaignSummary(BaseModel):
    campaign_id: str
    platform: str
    category: str
    objective: str
    format: str
    score_observado: float
    score_previsto: float
    recomendacoes: int = Field(description="recomendações na avaliação do engine")
    A: int
    B: int
    C: int


class CampaignPage(BaseModel):
    total: int
    pagina: int
    por_pagina: int
    campanhas: list[CampaignSummary]


class CampaignDetail(BaseModel):
    campaign_id: str
    atributos: dict[str, Any] = Field(description="os 13 atributos de pré-publicação")
    observado: dict[str, Any] = Field(description="score e métricas observados na campanha real")
    resumo: CampaignSummary
