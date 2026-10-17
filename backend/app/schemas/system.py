"""Contratos de estado do serviço."""

from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    engine_carregado: bool
    modelo: str
    campanhas: int
