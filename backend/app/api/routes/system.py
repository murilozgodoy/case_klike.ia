"""Estado do serviço e metadados para a interface."""

from fastapi import APIRouter

from app.core.resources import ResourcesDep
from app.schemas.system import Health

router = APIRouter(tags=["sistema"])


@router.get("/health", response_model=Health)
def health(resources: ResourcesDep) -> Health:
    return Health(
        status="ok",
        engine_carregado=True,
        modelo=resources.model_source,
        campanhas=len(resources.history),
    )


@router.get("/metadata")
def metadata(resources: ResourcesDep) -> dict:
    """Campos e opções do formulário, métricas, indicadores do projeto e exemplos guiados."""
    return resources.metadata
