"""Campanhas históricas: listagem, detalhe e análise pelo engine."""

from fastapi import APIRouter, Query

from app.core.resources import ResourcesDep
from app.schemas.analysis import Report
from app.schemas.campaign import CampaignDetail, CampaignPage
from app.services.analysis import analyze_history
from app.services.campaigns import campaign_detail, list_campaigns

router = APIRouter(prefix="/campaigns", tags=["campanhas"])


@router.get("", response_model=CampaignPage)
def campaigns(
    resources: ResourcesDep,
    platform: str | None = None,
    objective: str | None = None,
    category: str | None = None,
    busca: str | None = Query(None, max_length=20, description="trecho do campaign_id"),
    pagina: int = Query(1, ge=1),
    por_pagina: int = Query(20, ge=1, le=100),
) -> dict:
    return list_campaigns(
        resources.catalog,
        platform=platform,
        objective=objective,
        category=category,
        search=busca,
        page=pagina,
        page_size=por_pagina,
    )


@router.get("/{campaign_id}", response_model=CampaignDetail)
def campaign(campaign_id: str, resources: ResourcesDep) -> dict:
    return campaign_detail(resources.history, resources.catalog, campaign_id)


@router.get("/{campaign_id}/analysis", response_model=Report)
def analysis(campaign_id: str, resources: ResourcesDep) -> dict:
    """O relatório do engine para a campanha histórica, como ``RecommendationReport.to_dict()``."""
    return analyze_history(resources.engine, resources.history, campaign_id)
