"""Estilo visual único para todos os gráficos do projeto.

Regras do sistema:
- cada plataforma tem uma cor fixa em todos os gráficos: a cor segue a entidade;
- cinza para contexto, azul de destaque para o que importa;
- vermelho reservado para sinalizar problema, como um erro de dado;
- marcas finas, grade discreta e títulos que contam a conclusão;
- números no formato brasileiro (1.000; 0,15; 10 mil).

As três cores das plataformas passam no validador de daltonismo usado como
referência. O verde-água do LinkedIn tem pouco contraste com o fundo, então
gráficos por plataforma sempre levam legenda ou rótulo direto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FuncFormatter

from klike.config import FIGURES_DIR
from klike.formatting import fmt_compact, fmt_number, fmt_pct

# Superfície e tinta
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Papéis de cor
ACCENT = "#2a78d6"  # série única ou destaque
CONTEXT = "#c3c2b7"  # marcas de contexto, que ficam em segundo plano
CRITICAL = "#d03b3b"  # status: problema (sempre acompanhado de rótulo)
GOOD = "#0ca30c"  # status: positivo (sempre acompanhado de rótulo)

PLATFORM_COLORS = {"Meta": "#2a78d6", "TikTok": "#eb6834", "LinkedIn": "#1baf7a"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Divergente para efeitos: vermelho (piora), cinza neutro no zero, azul (melhora).
DIVERGING = LinearSegmentedColormap.from_list("klike_diverging", ["#b8322f", "#f0efec", "#1c5cab"])


def ink_on(color) -> str:
    """Cor de texto legível sobre um fundo: branco no escuro, tinta no claro."""
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.5 else INK


def _pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def set_style() -> None:
    """Aplica o estilo do projeto a todos os gráficos seguintes."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.family": _pick_font(),
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


# --------------------------------------------------------------------------- #
# Títulos e exportação
# --------------------------------------------------------------------------- #
def add_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    """Título que conta a conclusão + subtítulo que diz o que está sendo medido.

    Reserva uma faixa fixa no topo da figura e organiza os painéis abaixo dela.
    """
    height_in = fig.get_figheight()
    header_in = 0.78 if subtitle else 0.5
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / height_in))
    fig.text(
        0.01,
        1 - 0.16 / height_in,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        fig.text(
            0.01,
            1 - 0.46 / height_in,
            subtitle,
            ha="left",
            va="top",
            fontsize=9.5,
            color=INK_SECONDARY,
        )


def save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Salva a figura em PNG na pasta de relatórios e devolve o caminho."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    return path


# --------------------------------------------------------------------------- #
# Formatadores de eixo
# --------------------------------------------------------------------------- #
compact_formatter = FuncFormatter(lambda x, _: fmt_compact(x))
pct_formatter = FuncFormatter(lambda x, _: fmt_pct(x))


def decimal_formatter(decimals: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda x, _: fmt_number(x, decimals))
