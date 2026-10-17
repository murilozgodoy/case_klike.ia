"""Análise de uma campanha nova e comparação de cenários."""

from fastapi import APIRouter

from app.core.resources import ResourcesDep
from app.schemas.analysis import Report, ScenarioRequest, ScenarioResponse
from app.schemas.campaign import CampaignInput
from app.services.analysis import analyze_new, compare_scenario

router = APIRouter(tags=["análise"])


@router.post("/analyze", response_model=Report)
def analyze(campaign: CampaignInput, resources: ResourcesDep) -> dict:
    """O relatório do engine para uma campanha nova, que não é guardada."""
    return analyze_new(resources.engine, campaign.model_dump())


@router.post("/scenarios/compare", response_model=ScenarioResponse)
def compare(request: ScenarioRequest, resources: ResourcesDep) -> dict:
    """A campanha original e a mesma campanha com uma única edição do criativo."""
    ((field, value),) = request.changes.items()
    return compare_scenario(
        resources.engine,
        resources.history,
        field=field,
        value=value,
        campaign_id=request.campaign_id,
        campaign=request.campaign.model_dump() if request.campaign else None,
    )
