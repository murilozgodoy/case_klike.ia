"""Carregamento e limpeza do dataset de campanhas.

A limpeza se apoia em três identidades contábeis do funil de mídia:

    ctr   = clicks / impressions
    spend = clicks * cpc
    roas  = revenue / spend

Como essas métricas são redundantes entre si, cada identidade funciona como um
"checksum". Quando uma delas quebra, as outras ajudam a apontar qual campo está
errado e permitem reconstruí-lo. Cada correção registra a base da decisão:

- identidade: duas contas independentes confirmam qual campo está errado, ou o
  valor faltante sai diretamente de uma conta;
- plausibilidade: as contas mostram o erro, mas não dizem qual campo está errado.
  Só corrigimos quando uma versão cai dentro da faixa observada nas campanhas
  saudáveis e a outra cai fora dela. Essas correções ficam marcadas no log para
  revisão com a fonte dos dados.

O que não puder ser resolvido assim vai para revisão manual, sem correção.
Toda alteração fica registrada num log de auditoria, uma linha por correção.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from klike.config import (
    BOOL_COLS,
    CATEGORY_ORDERS,
    CTR_CAP,
    DATE_COL,
    ENGAGEMENT_CAP,
    ID_COL,
    ORDERED_CATEGORIES,
    RAW_DATA_PATH,
)
from klike.formatting import fmt_compact, fmt_number

# Tolerâncias das identidades. As métricas vêm arredondadas no CSV
# (CTR com 4 casas, CPC e ROAS com 2), então nunca exigimos igualdade exata.
TOL_IMPRESSIONS_REL = 0.10  # impressões registradas vs. cliques / CTR
TOL_SPEND_REL = 0.02  # gasto registrado vs. cliques x CPC
TOL_ROAS_ABS = 0.01  # ROAS registrado vs. receita / gasto

# Base da decisão de cada correção (ver docstring do módulo).
IDENTITY = "identidade"
PLAUSIBILITY = "plausibilidade"

LOG_COLUMNS = [ID_COL, "column", "old_value", "new_value", "rule", "basis", "evidence"]


@dataclass
class CleaningResult:
    """Resultado da limpeza: dados corrigidos, log de correções e pendências."""

    data: pd.DataFrame
    log: pd.DataFrame
    unresolved: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Quantidade de registros corrigidos por regra e por base da decisão."""
        if self.log.empty:
            return pd.DataFrame(columns=["rule", "basis", "registros", "colunas"])
        return (
            self.log.groupby(["rule", "basis"], sort=False)
            .agg(
                registros=(ID_COL, "nunique"),
                colunas=("column", lambda s: ", ".join(s.unique())),
            )
            .reset_index()
        )


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV original com os tipos corretos.

    Booleanos viram o tipo ``boolean`` do pandas, que aceita valor ausente
    (``has_subtitle`` tem faltantes). Categorias ganham ordem fixa, e valores
    fora do esperado interrompem a leitura em vez de passarem em silêncio.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    for col in BOOL_COLS:
        df[col] = df[col].astype("boolean")

    for col, categories in CATEGORY_ORDERS.items():
        unexpected = set(df[col].dropna().unique()) - set(categories)
        if unexpected:
            raise ValueError(f"Valores inesperados em {col!r}: {sorted(unexpected)}")
        df[col] = pd.Categorical(df[col], categories=categories, ordered=col in ORDERED_CATEGORIES)

    if df[ID_COL].duplicated().any():
        raise ValueError("Há campaign_id duplicado no dataset.")
    return df


def load_clean(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Atalho para os notebooks: dados carregados e já limpos."""
    return clean(load_raw(path)).data


# --------------------------------------------------------------------------- #
# Limpeza
# --------------------------------------------------------------------------- #
def clean(raw: pd.DataFrame) -> CleaningResult:
    """Aplica as regras de limpeza na ordem certa e devolve dados, log e pendências.

    A ordem importa: primeiro corrigimos os valores que quebram as contas, depois
    recuperamos faltantes. Assim, a receita faltante é reconstruída a partir do gasto
    já corrigido, e não do gasto inconsistente. O DataFrame de entrada não é alterado.
    """
    df = raw.copy()
    log: list[dict] = []
    unresolved: list[dict] = []

    _fix_inflated_impressions(df, log, unresolved)
    _fix_spend_cpc_conflicts(df, log, unresolved)
    _recover_missing_cpc(df, log)
    _recover_missing_revenue(df, log)
    _add_flags(df)

    return CleaningResult(
        data=df,
        log=pd.DataFrame(log, columns=LOG_COLUMNS),
        unresolved=pd.DataFrame(unresolved, columns=[ID_COL, "issue"]),
    )


def _record(
    log: list[dict],
    df: pd.DataFrame,
    idx,
    column: str,
    new_value,
    rule: str,
    basis: str,
    evidence: str,
) -> None:
    """Aplica uma correção e registra o antes, o depois e a base da decisão no log."""
    log.append(
        {
            ID_COL: df.at[idx, ID_COL],
            "column": column,
            "old_value": df.at[idx, column],
            "new_value": new_value,
            "rule": rule,
            "basis": basis,
            "evidence": evidence,
        }
    )
    df.at[idx, column] = new_value


def _pct(value: float) -> str:
    """Percentual curto para o log: 0.0022 -> '0,22%'; 0.00004 -> '0,004%'."""
    return f"{fmt_compact(value * 100)}%"


def _spend_identity_holds(df: pd.DataFrame) -> pd.Series:
    """True quando gasto = cliques x CPC, dentro da tolerância (False se faltar CPC)."""
    implied_spend = df["clicks"] * df["cpc"]
    return (implied_spend - df["spend"]).abs() / df["spend"] <= TOL_SPEND_REL


def _fix_inflated_impressions(df: pd.DataFrame, log: list[dict], unresolved: list[dict]) -> None:
    """Impressões incompatíveis com cliques / CTR.

    Se cliques x CPC bate com o gasto, os cliques estão confirmados por uma
    segunda identidade. Sobram impressões e CTR como suspeitos, e nenhuma outra
    conta separa os dois. A decisão é por plausibilidade: as impressões são
    reconstruídas como cliques / CTR só quando o CTR registrado cai dentro da faixa
    das campanhas em que a conta fecha e o CTR alternativo (cliques / impressões
    registradas) cai fora dela. Nos outros casos, a linha vai para revisão manual.
    """
    implied = df["clicks"] / df["ctr"]
    broken = (df["impressions"] / implied - 1).abs() > TOL_IMPRESSIONS_REL
    clicks_confirmed = _spend_identity_holds(df)
    low, high = df.loc[~broken, "ctr"].min(), df.loc[~broken, "ctr"].max()
    alternative_ctr = df["clicks"] / df["impressions"]
    clear_choice = df["ctr"].between(low, high) & ~alternative_ctr.between(low, high)

    for idx in df.index[broken & clicks_confirmed & clear_choice]:
        factor = df.at[idx, "impressions"] / implied[idx]
        _record(
            log,
            df,
            idx,
            "impressions",
            int(round(implied[idx])),
            rule="Impressões infladas",
            basis=PLAUSIBILITY,
            evidence=(
                f"impressões {fmt_number(factor)}x acima de cliques/CTR; cliques confirmados "
                f"por gasto = cliques x CPC; o CTR alternativo, cliques/impressões, seria "
                f"{_pct(alternative_ctr[idx])}, fora da faixa observada ({_pct(low)} a "
                f"{_pct(high)})"
            ),
        )
    for idx in df.index[broken & ~clicks_confirmed]:
        unresolved.append(
            {ID_COL: df.at[idx, ID_COL], "issue": "CTR inconsistente sem segunda identidade"}
        )
    for idx in df.index[broken & clicks_confirmed & ~clear_choice]:
        unresolved.append(
            {ID_COL: df.at[idx, ID_COL], "issue": "impressões e CTR discordam, sem desempate"}
        )


def _fix_spend_cpc_conflicts(df: pd.DataFrame, log: list[dict], unresolved: list[dict]) -> None:
    """Gasto e cliques x CPC discordam: qual dos dois está errado?

    1. Se há receita, o ROAS desempata, porque foi calculado com o gasto verdadeiro.
       Duas contas independentes concordam, então a base é a identidade.
    2. Se a receita é zero (caso de Branding), o ROAS não informa nada. Aí a decisão
       é por plausibilidade: fica a versão de CPC que cai dentro da faixa da
       plataforma, nas campanhas em que a conta fecha, desde que a outra caia fora.
       Se as duas caírem do mesmo lado da faixa, a linha vai para revisão manual.
    """
    consistent = _spend_identity_holds(df)
    broken = df["cpc"].notna() & ~consistent
    cpc_range = df.loc[consistent].groupby("platform", observed=True)["cpc"].agg(["min", "max"])

    for idx in df.index[broken]:
        clicks, cpc = df.at[idx, "clicks"], df.at[idx, "cpc"]
        spend, revenue, roas = df.at[idx, "spend"], df.at[idx, "revenue"], df.at[idx, "roas"]
        spend_from_cpc = round(clicks * cpc, 2)
        cpc_from_spend = round(spend / clicks, 2)
        cpc_txt, implied_txt = fmt_number(cpc, 2), fmt_number(cpc_from_spend, 2)

        culprit, basis = None, IDENTITY
        if pd.notna(revenue) and revenue > 0:
            if abs(revenue / spend - roas) <= TOL_ROAS_ABS:
                culprit = "cpc"
                roas_txt = fmt_number(revenue / spend, 2)
                evidence = (
                    f"gasto confirmado pelo ROAS (receita/gasto = {roas_txt}); "
                    f"CPC {cpc_txt} não bate com gasto/cliques = {implied_txt}"
                )
            elif abs(revenue / spend_from_cpc - roas) <= TOL_ROAS_ABS:
                culprit = "spend"
                evidence = (
                    "cliques x CPC confirmado pelo ROAS (receita/(cliques x CPC) = "
                    f"{fmt_number(revenue / spend_from_cpc, 2)}); gasto registrado "
                    f"{fmt_number(spend / spend_from_cpc)}x maior"
                )

        if culprit is None:
            platform = df.at[idx, "platform"]
            if platform not in cpc_range.index:
                unresolved.append(
                    {ID_COL: df.at[idx, ID_COL], "issue": "gasto e CPC discordam, sem referência"}
                )
                continue
            low, high = cpc_range.at[platform, "min"], cpc_range.at[platform, "max"]
            recorded_ok, implied_ok = low <= cpc <= high, low <= cpc_from_spend <= high
            if recorded_ok == implied_ok:
                unresolved.append(
                    {ID_COL: df.at[idx, ID_COL], "issue": "gasto e CPC discordam, sem desempate"}
                )
                continue
            culprit, basis = ("spend" if recorded_ok else "cpc"), PLAUSIBILITY
            evidence = (
                f"ROAS não desempata (receita zero ou ausente); faixa de CPC da plataforma "
                f"{fmt_number(low, 2)} a {fmt_number(high, 2)}: CPC registrado {cpc_txt} "
                f"{'dentro' if recorded_ok else 'fora'}, gasto/cliques {implied_txt} "
                f"{'dentro' if implied_ok else 'fora'}"
            )

        if culprit == "cpc":
            _record(log, df, idx, "cpc", cpc_from_spend, "CPC fora de escala", basis, evidence)
        else:
            _record(log, df, idx, "spend", spend_from_cpc, "Gasto fora de escala", basis, evidence)


def _recover_missing_cpc(df: pd.DataFrame, log: list[dict]) -> None:
    """CPC ausente é recuperável exatamente: CPC = gasto / cliques."""
    for idx in df.index[df["cpc"].isna() & (df["clicks"] > 0)]:
        _record(
            log,
            df,
            idx,
            "cpc",
            round(df.at[idx, "spend"] / df.at[idx, "clicks"], 2),
            rule="CPC faltante recuperado",
            basis=IDENTITY,
            evidence="identidade gasto = cliques x CPC",
        )


def _recover_missing_revenue(df: pd.DataFrame, log: list[dict]) -> None:
    """Receita ausente é recuperável: receita = ROAS x gasto.

    É uma aproximação, porque o ROAS vem com 2 casas decimais. O erro máximo é
    de 0,5% do gasto, desprezível para a análise.
    """
    for idx in df.index[df["revenue"].isna()]:
        _record(
            log,
            df,
            idx,
            "revenue",
            round(df.at[idx, "roas"] * df.at[idx, "spend"], 2),
            rule="Receita faltante recuperada",
            basis=IDENTITY,
            evidence="identidade ROAS = receita / gasto (ROAS com 2 casas decimais)",
        )


def _add_flags(df: pd.DataFrame) -> None:
    """Marca situações que não são erro, mas mudam a forma de analisar o dado.

    As comparações usam o tipo numérico que aceita ausência (Float64). Assim, quando
    o valor de origem falta, a marcação fica ausente, e não falsa: não dá para
    afirmar que uma campanha sem engajamento registrado está abaixo do teto.
    """
    # Possível teto de registro: muitos valores exatamente no limite e nenhum acima.
    # Se for censura, o valor real pode ser maior. Mantemos e sinalizamos.
    df["flag_ctr_at_cap"] = df["ctr"] >= CTR_CAP
    df["flag_engagement_at_cap"] = df["engagement_rate"].astype("Float64") >= ENGAGEMENT_CAP
    # Tempo médio assistido igual à duração exigiria que todos vissem o vídeo inteiro:
    # provavelmente um limite da métrica. Mantemos e sinalizamos.
    df["flag_full_watch"] = df["avg_watch_time_s"].astype("Float64") >= df[
        "video_duration_s"
    ].astype("Float64")
    # Branding não registra receita: ROAS = 0 ali significa "não se aplica", não fracasso.
    df["flag_roas_applicable"] = df["category"] != "Branding"
