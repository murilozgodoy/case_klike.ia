"""Formatação de números no padrão brasileiro (1.234,5; 15%; 1,2 mil).

Fica num módulo próprio porque é usada tanto nos gráficos quanto nos textos de
evidência do log de limpeza.
"""

from __future__ import annotations

import numpy as np


def fmt_number(value: float, decimals: int = 0) -> str:
    """1234.5 -> '1.234,5' (separadores brasileiros)."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _short(value: float) -> str:
    """Número curto sem zeros inúteis: 1.0 -> '1'; 1.5 -> '1,5'; 0.001 -> '0,001'."""
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    if magnitude >= 100:
        decimals = 0
    elif magnitude >= 1:
        decimals = 1
    else:  # dois algarismos significativos
        decimals = min(6, -int(np.floor(np.log10(magnitude))) + 1)
    text = fmt_number(value, decimals)
    return text.rstrip("0").rstrip(",") if "," in text else text


def fmt_compact(value: float) -> str:
    """Números de forma curta: 1.200 -> '1,2 mil'; 3.400.000 -> '3,4 mi'; 0,05 -> '0,05'."""
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{_short(value / 1_000_000)} mi"
    if magnitude >= 1_000:
        return f"{_short(value / 1_000)} mil"
    return _short(value)


def fmt_pct(value: float, decimals: int | None = None) -> str:
    """0.15 -> '15%'; 0.025 -> '2,5%'. Sem casas decimais, a menos que façam diferença."""
    pct = value * 100
    if decimals is None:
        decimals = 0 if abs(pct - round(pct)) < 1e-9 else 1
    return f"{fmt_number(pct, decimals)}%"


def fmt_signed(value: float, decimals: int = 0, suffix: str = "") -> str:
    """Variação com sinal explícito: (5.1, 1) -> '+5,1'; (-5.1, 1, ' pts') -> '−5,1 pts'.

    Usa o sinal de menos tipográfico, mais legível em gráficos. Zero sai sem sinal.
    """
    text = fmt_number(abs(value), decimals)
    if float(text.replace(".", "").replace(",", ".")) == 0:
        return f"0{suffix}"
    sign = "+" if value > 0 else "−"
    return f"{sign}{text}{suffix}"


def fmt_signed_pct(value: float, decimals: int = 0) -> str:
    """Variação percentual com sinal: 0.51 -> '+51%'; -0.39 -> '−39%'."""
    return fmt_signed(value * 100, decimals, suffix="%")
