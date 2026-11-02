"""Prévisions financières : tendances historiques + scénarios (pessimiste / central / optimiste)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .analysis import cagr


def historical_drivers(df: pd.DataFrame) -> dict:
    rev = df.loc["revenue"].dropna()
    g = cagr(rev)
    last = df.columns[-1]
    take = lambda k: df.at[k, last] if k in df.index else np.nan
    mean_ratio = lambda a, b: float((df.loc[a] / df.loc[b]).replace([np.inf, -np.inf], np.nan).dropna().tail(3).mean())
    return {
        "growth": 0.03 if g != g else float(g),
        "ebitda_margin": mean_ratio("ebitda", "revenue"),
        "ebit_margin": mean_ratio("ebit", "revenue"),
        "da_pct": mean_ratio("da", "revenue"),
        "capex_pct": mean_ratio("capex", "revenue"),
        "net_margin": mean_ratio("net_income", "revenue"),
        "last_revenue": float(take("revenue")),
        "last_year": int(last),
    }


def trend_projection(series: pd.Series, horizon: int = 5) -> pd.Series:
    """Régression log-linéaire (croissance exponentielle) sur l'historique."""
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 2:
        return pd.Series(dtype=float)
    x = np.array(s.index, dtype=float)
    b, a = np.polyfit(x, np.log(s.values), 1)
    fut = np.arange(int(x[-1]) + 1, int(x[-1]) + horizon + 1)
    return pd.Series(np.exp(a + b * fut), index=fut)


def scenarios(drv: dict, g_central: float, margin_target: float, tax: float, horizon: int = 5,
              spread_g: float = 0.03, spread_m: float = 0.02) -> dict[str, pd.DataFrame]:
    out = {}
    for name, dg, dm in [("Pessimiste", -spread_g, -spread_m), ("Central", 0, 0), ("Optimiste", spread_g, spread_m)]:
        rev = drv["last_revenue"]
        m0 = drv["ebit_margin"] if drv["ebit_margin"] == drv["ebit_margin"] else margin_target
        rows = []
        for t in range(1, horizon + 1):
            g = g_central + dg
            rev *= (1 + g)
            m = m0 + (margin_target + dm - m0) * t / horizon
            ebit = rev * m
            da = rev * (drv["da_pct"] if drv["da_pct"] == drv["da_pct"] else 0.04)
            capex = rev * (drv["capex_pct"] if drv["capex_pct"] == drv["capex_pct"] else 0.05)
            nopat = ebit * (1 - tax)
            rows.append({"Année": drv["last_year"] + t, "Chiffre d'affaires": rev, "Croissance": g, "EBITDA": ebit + da,
                         "EBIT": ebit, "Marge EBIT": m, "NOPAT": nopat, "Capex": capex, "D&A": da,
                         "FCFF simplifié": nopat + da - capex})
        out[name] = pd.DataFrame(rows).set_index("Année")
    return out
