"""Contratos de análise e de comparação de cenários.

O relatório do engine segue o formato de ``RecommendationReport.to_dict()`` e é devolvido
como está: a API não redefine a estrutura de classes, recomendações ou ressalvas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.campaign import CampaignInput
from klike.config import CONTEXT_COLS, CREATIVE_COLS

Report = dict[str, Any]


class ScenarioRequest(BaseModel):
    """Uma campanha-base, do histórico ou nova, e exatamente uma edição do criativo."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str | None = None
    campaign: CampaignInput | None = None
    changes: dict[str, Any] = Field(
        description="um único atributo do criativo e o valor novo, como {'has_hook': true}"
    )

    @model_validator(mode="after")
    def one_base_and_one_creative_edit(self) -> ScenarioRequest:
        if (self.campaign_id is None) == (self.campaign is None):
            raise ValueError("informe campaign_id ou campaign, um dos dois")
        if len(self.changes) != 1:
            raise ValueError(
                f"a comparação aplica uma alteração por vez; foram enviadas {len(self.changes)}"
            )
        field = next(iter(self.changes))
        if field in CONTEXT_COLS:
            raise ValueError(
                f"{field} é contexto da campanha: a comparação só altera atributos do criativo"
            )
        if field not in CREATIVE_COLS:
            raise ValueError(
                f"{field}: atributo desconhecido (aceitos: {', '.join(CREATIVE_COLS)})"
            )
        return self


class ScenarioResponse(BaseModel):
    alteracao: dict[str, Any]
    score: dict[str, float] = Field(description="scores previstos pelo mesmo preditor")
    evidencia: dict[str, Any] = Field(
        description="evidência do engine, só quando a edição é uma ação do catálogo"
    )
    nota: str
    original: Report
    modificado: Report
