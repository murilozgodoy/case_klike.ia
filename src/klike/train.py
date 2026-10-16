"""Treina o modelo final e salva o arquivo para uso: ``python -m klike.train``.

Reajusta a especificação congelada nas 500 campanhas e salva pipeline, suporte histórico e
metadados em ``models/``. O arquivo não vai para o repositório: este comando o recria.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from klike.config import MODEL_PATH, RAW_DATA_PATH
from klike.formatting import fmt_compact, fmt_number
from klike.predictor import train_final_model


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m klike.train",
        description="Treina o modelo final do Klike Score e salva o arquivo joblib.",
    )
    parser.add_argument(
        "--dados", type=Path, default=RAW_DATA_PATH, help="CSV com o histórico de campanhas"
    )
    parser.add_argument("--saida", type=Path, default=MODEL_PATH, help="arquivo .joblib de saída")
    args = parser.parse_args(argv)

    predictor = train_final_model(args.dados)
    path = predictor.save(args.saida)
    metadata = predictor.metadata
    training, metrics = metadata["treino"], metadata["metricas_oficiais"]
    low, high = metrics["mae_ic95"]
    print(f"Modelo salvo em {_relative(path)}")
    print(
        f"Treino: {training['campanhas']} campanhas de {training['arquivo']} | "
        f"alpha {fmt_compact(metadata['especificacao']['alpha'])} | "
        f"interações: {', '.join(training['interacoes_plataforma_formato'])}"
    )
    print(
        f"Métricas oficiais, da certificação: MAE {fmt_number(metrics['mae'], 2)} "
        f"(intervalo de 95%: {fmt_number(low, 2)} a {fmt_number(high, 2)})"
    )
    return path


def _relative(path: Path) -> Path:
    """Caminho a partir da pasta atual, quando o arquivo está dentro dela."""
    try:
        return path.resolve().relative_to(Path.cwd())
    except ValueError:
        return path


if __name__ == "__main__":
    main()
