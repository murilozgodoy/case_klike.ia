"""Conversão para JSON estrito: tipos do numpy viram tipos do Python, e NaN vira nulo."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def record(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    """Os campos pedidos de uma linha, prontos para JSON."""
    return jsonable({col: row[col] for col in columns})
