"""API do Klike Interactive Lab: expõe o pacote ``klike`` para a interface web.

A API não tem lógica científica própria. Ela valida a entrada, chama o preditor e o
recommendations engine do pacote e devolve as saídas deles em JSON. O modelo e as evidências
são carregados uma única vez, na inicialização.

Para rodar localmente, a partir da raiz do repositório::

    uvicorn app.main:app --app-dir backend --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, campaigns, system
from app.core.config import Settings
from app.core.resources import load_resources
from app.services.errors import InvalidInputError, NotFoundError

logger = logging.getLogger("klike.api")

# Mensagens do Pydantic em português, para a interface mostrar ao usuário.
VALIDATION_MESSAGES = {
    "missing": "campo obrigatório",
    "extra_forbidden": "campo desconhecido",
    "bool_parsing": "esperado verdadeiro ou falso",
    "bool_type": "esperado verdadeiro ou falso",
    "float_parsing": "esperado um número",
    "float_type": "esperado um número",
    "string_type": "esperado um texto",
    "dict_type": "esperado um objeto",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.resources = load_resources()
    logger.info("Engine carregado: %s", app.state.resources.model_source)
    yield


def _readable(errors: list[dict]) -> list[str]:
    messages = []
    for error in errors:
        location = ".".join(str(part) for part in error["loc"] if part not in ("body", "query"))
        message = VALIDATION_MESSAGES.get(error["type"], error["msg"])
        message = message.removeprefix("Value error, ")
        messages.append(f"{location}: {message}" if location else message)
    return messages


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Klike Interactive Lab API",
        version="1.0.0",
        description=(
            "Camada de integração entre a interface e o pacote klike: o Klike Score vem do "
            "KlikeScorePredictor, e as recomendações, do RecommendationEngine."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    for router in (system.router, campaigns.router, analysis.router):
        app.include_router(router, prefix="/api")

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, error: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(InvalidInputError)
    async def invalid_input(_: Request, error: InvalidInputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        messages = _readable(error.errors())
        return JSONResponse(
            status_code=422,
            content={"detail": "Entrada inválida: " + "; ".join(messages) + ".", "erros": messages},
        )

    @app.exception_handler(Exception)
    async def unexpected(_: Request, error: Exception) -> JSONResponse:
        logger.exception("Erro inesperado", exc_info=error)
        return JSONResponse(
            status_code=500, content={"detail": "Erro interno ao processar a requisição."}
        )

    return app


app = create_app()
