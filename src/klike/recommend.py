"""Recomendações para uma campanha pela linha de comando: ``python -m klike.recommend``.

Exemplos::

    python -m klike.recommend --campanha KLK-0024
    python -m klike.recommend --arquivo examples/campanha_nova.json
    python -m klike.recommend --campanha KLK-0024 --metrica ctr --json

A campanha pode vir do histórico, pelo ``campaign_id``, ou de um JSON com os 13 atributos de
pré-publicação. O relatório sai em texto ou, com ``--json``, no formato para integração.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from klike.data import load_clean
from klike.engine import RecommendationEngine, RecommendationReport
from klike.evidence import PERFORMANCE_METRICS


def main(argv: Sequence[str] | None = None) -> RecommendationReport:
    parser = argparse.ArgumentParser(
        prog="python -m klike.recommend",
        description="Recomendações de edição para o criativo de uma campanha.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campanha", help="campaign_id de uma campanha do histórico (KLK-0024)")
    source.add_argument(
        "--arquivo", type=Path, help="JSON com os 13 atributos de pré-publicação de uma campanha"
    )
    parser.add_argument(
        "--metrica",
        choices=list(PERFORMANCE_METRICS),
        help="troca a métrica do objetivo da campanha por outra",
    )
    parser.add_argument("--json", action="store_true", help="saída em JSON, para integração")
    args = parser.parse_args(argv)

    if args.campanha:
        history = load_clean()
        campaign = history[history["campaign_id"] == args.campanha]
        if campaign.empty:
            parser.error(f"a campanha {args.campanha!r} não está no histórico")
    else:
        campaign = json.loads(args.arquivo.read_text(encoding="utf-8"))
    try:
        report = RecommendationEngine.default().recommend(campaign, metric=args.metrica)
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    # Saída sempre em UTF-8: no Windows, um redirecionamento usaria a codificação local, e o
    # JSON, os acentos e as setas sairiam corrompidos.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.to_text())
    return report


if __name__ == "__main__":
    main()
