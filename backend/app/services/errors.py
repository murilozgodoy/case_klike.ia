"""Erros que a API traduz em respostas HTTP."""


class NotFoundError(LookupError):
    """Recurso inexistente: vira 404."""


class InvalidInputError(ValueError):
    """Entrada recusada pela validação do pacote ou da API: vira 422, com a mensagem."""
