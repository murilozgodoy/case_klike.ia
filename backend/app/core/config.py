"""Configuração da API por variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


@dataclass(frozen=True)
class Settings:
    """CORS: as origens do frontend, separadas por vírgula, e um padrão opcional.

    ``CORS_ORIGIN_REGEX`` serve para as URLs de pré-visualização da Vercel, como
    ``https://.*-seu-usuario\\.vercel\\.app``.
    """

    cors_origins: tuple[str, ...]
    cors_origin_regex: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS)
        return cls(
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            cors_origin_regex=os.getenv("CORS_ORIGIN_REGEX") or None,
        )
