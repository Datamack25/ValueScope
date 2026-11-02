"""Diagnostic financier : ratios, décomposition DuPont, ROIC vs WACC, scores de solidité,
cycle de vie (Damodaran, « The Corporate Life Cycle ») et commentaires automatiques."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _s(df, k):
    return df.loc[k] if k in df.index else pd.Series(np.nan, index=df.columns)


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.Series) else out


def effective_tax(df) -> pd.Series:
    t = _div(_s(df, "tax"), _s(df, "pretax_income"))
    return t.clip(0, 0.5)


def invested_capital(df) -> pd.Series:
    """Capital investi = Capitaux propres + Dette financière − Trésorerie (approche financement)."""
    return _s(df, "equity") + _s(df, "total_debt").fillna(0) - _s(df, "cash").fillna(0)


def ratios(df: pd.DataFrame, marginal_tax: float = 0.25) -> pd.DataFrame:
    rev, ebit, ni = _s(df, "revenue"), _s(df, "ebit"), _s(df, "net_income")
    eq, ta, debt, cash = _s(df, "equity"), _s(df, "total_assets"), _s(df, "total_debt"), _s(df, "cash")
    t = effective_tax(df).fillna(marginal_tax)
    ic = invested_capital(df)
    nopat = ebit * (1 - t)
    fcff = nopat + _s(df, "da").fillna(0) - _s(df, "capex").fillna(0) - (_wcr(df).diff().fillna(0))
    rows = {
        # Croissance & rentabilité
        "Croissance du CA (%)": rev.pct_change() * 100,
        "Marge brute (%)": _div(_s(df, "gross_profit"), rev) * 100,
        "Marge d'EBITDA (%)": _div(_s(df, "ebitda"), rev) * 100,
        "Marge opérationnelle (%)": _div(ebit, rev) * 100,
        "Marge nette (%)": _div(ni, rev) * 100,
        "Taux d'IS effectif (%)": t * 100,
        "ROE (%)": _div(ni, eq) * 100,
        "ROA (%)": _div(ni, ta) * 100,
        "ROIC après impôt (%)": _div(nopat, ic.shift(1).fillna(ic)) * 100,
        # DuPont
        "DuPont – Rotation de l'actif (x)": _div(rev, ta),
        "DuPont – Levier (Actif/CP) (x)": _div(ta, eq),
        "Ventes / Capital investi (x)": _div(rev, ic),
        # Structure & solvabilité
        "Gearing (Dette nette/CP) (%)": _div(debt - cash, eq) * 100,
        "Dette nette / EBITDA (x)": _div(debt - cash, _s(df, "ebitda")),
        "Couverture des intérêts EBIT/Int (x)": _div(ebit, _s(df, "interest_expense")),
        "Autonomie financière CP/Total bilan (%)": _div(eq, ta) * 100,
        # Liquidité & BFR
        "Liquidité générale (x)": _div(_s(df, "current_assets"), _s(df, "current_liabilities")),
        "Liquidité réduite (x)": _div(_s(df, "current_assets") - _s(df, "inventory").fillna(0), _s(df, "current_liabilities")),
        "DSO – Délai clients (jours)": _div(_s(df, "receivables"), rev) * 365,
        "DIO – Rotation stocks (jours)": _div(_s(df, "inventory"), _s(df, "cogs")) * 365,
        "DPO – Délai fournisseurs (jours)": _div(_s(df, "payables"), _s(df, "cogs")) * 365,
        "BFR d'exploitation / CA (%)": _div(_wcr(df), rev) * 100,
        # Cash-flow
        "Conversion cash CFO/Résultat net (x)": _div(_s(df, "cfo"), ni),
        "Capex / CA (%)": _div(_s(df, "capex"), rev) * 100,
        "Capex / D&A (x)": _div(_s(df, "capex"), _s(df, "da")),
        "Free cash-flow (CFO − Capex)": _s(df, "cfo") - _s(df, "capex"),
        "FCFF (NOPAT + D&A − Capex − ΔBFR)": fcff,
        "Taux de réinvestissement (%)": _div(_s(df, "capex") - _s(df, "da").fillna(0) + _wcr(df).diff().fillna(0), nopat) * 100,
        "Taux de distribution (%)": _div(_s(df, "dividends") + _s(df, "buybacks").fillna(0), ni) * 100,
    }
    return pd.DataFrame(rows).T


def _wcr(df):
    """BFR d'exploitation non cash = Clients + Stocks − Fournisseurs."""
    return _s(df, "receivables").fillna(0) + _s(df, "inventory").fillna(0) - _s(df, "payables").fillna(0)


def altman_z(df, market_cap: float | None = None, listed: bool = True) -> dict:
    y = df.columns[-1]
    g = lambda k: float(df.at[k, y]) if k in df.index and pd.notna(df.at[k, y]) else np.nan
    ta = g("total_assets")
    if not ta or np.isnan(ta):
        return {"score": np.nan, "zone": "n.d.", "model": ""}
    wc = (g("current_assets") or 0) - (g("current_liabilities") or 0)
    re_ = g("retained_earnings")
    re_ = re_ if not np.isnan(re_) else g("equity") * 0.5
    tl = ta - (g("equity") or 0)
    if listed and market_cap:
        z = 1.2 * wc / ta + 1.4 * re_ / ta + 3.3 * g("ebit") / ta + 0.6 * market_cap / max(tl, 1e-9) + 1.0 * g("revenue") / ta
        zone = "Sûre" if z > 2.99 else ("Grise" if z > 1.81 else "Détresse")
        return {"score": z, "zone": zone, "model": "Altman Z (1968, sociétés cotées)"}
    z = 6.56 * wc / ta + 3.26 * re_ / ta + 6.72 * g("ebit") / ta + 1.05 * g("equity") / max(tl, 1e-9)
    zone = "Sûre" if z > 2.6 else ("Grise" if z > 1.1 else "Détresse")
    return {"score": z, "zone": zone, "model": "Altman Z'' (non cotées / hors industrie)"}


def piotroski(df) -> dict:
    if df.shape[1] < 2:
        return {"score": np.nan, "details": {}}
    c, p = df.columns[-1], df.columns[-2]
    g = lambda k, y: df.at[k, y] if k in df.index else np.nan
    roa = lambda y: g("net_income", y) / g("total_assets", y)
    d = {}
    d["ROA > 0"] = roa(c) > 0
    d["CFO > 0"] = g("cfo", c) > 0
    d["ROA en hausse"] = roa(c) > roa(p)
    d["CFO > Résultat net (qualité)"] = g("cfo", c) > g("net_income", c)
    d["Levier en baisse"] = g("long_debt", c) / g("total_assets", c) <= g("long_debt", p) / g("total_assets", p) \
        if pd.notna(g("long_debt", c)) else g("total_debt", c) / g("total_assets", c) <= g("total_debt", p) / g("total_assets", p)
    d["Liquidité en hausse"] = g("current_assets", c) / g("current_liabilities", c) > g("current_assets", p) / g("current_liabilities", p)
    d["Pas de dilution"] = not (g("shares", c) > g("shares", p) * 1.01) if pd.notna(g("shares", c)) else True
    d["Marge brute en hausse"] = g("gross_profit", c) / g("revenue", c) > g("gross_profit", p) / g("revenue", p)
    d["Rotation de l'actif en hausse"] = g("revenue", c) / g("total_assets", c) > g("revenue", p) / g("total_assets", p)
    d = {k: bool(v) if v == v else False for k, v in d.items()}
    return {"score": sum(d.values()), "details": d}


def life_cycle(df) -> tuple[str, str]:
    """Stade du cycle de vie (Damodaran) à partir de la croissance, des marges et du réinvestissement."""
    rev = _s(df, "revenue").dropna()
    n = len(rev)
    g = (rev.iloc[-1] / rev.iloc[0]) ** (1 / max(n - 1, 1)) - 1 if n >= 2 and rev.iloc[0] > 0 else 0
    m = float((_s(df, "ebit") / _s(df, "revenue")).dropna().iloc[-1]) if _s(df, "ebit").notna().any() else 0
    if g > 0.30 and m < 0:
        return "Start-up / jeune croissance", "Valeur portée par la croissance future : narrative essentielle, DCF avec convergence des marges et probabilité de défaillance."
    if g > 0.15:
        return "Forte croissance", "Valeur = croissance rentable. Le levier clé est le ratio ventes/capital (efficacité du réinvestissement)."
    if g > 0.05:
        return "Croissance mature", "Équilibre croissance / rentabilité. ROIC vs WACC détermine si la croissance crée de la valeur."
    if g > -0.02:
        return "Maturité", "Cash cow : valeur dominée par les flux actuels, politique de retour à l'actionnaire et dette."
    return "Déclin", "Valeur = cash-flows décroissants + valeur de liquidation. Attention aux réinvestissements destructeurs."


def cagr(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1


def commentary(df, r: pd.DataFrame, wacc: float | None = None) -> list[tuple[str, str]]:
    """Retourne des (niveau, texte) : niveau ∈ {'ok','warn','bad','info'}."""
    out = []
    last = r.columns[-1]
    v = lambda k: r.at[k, last] if k in r.index else np.nan
    g = cagr(_s(df, "revenue"))
    if g == g:
        out.append(("ok" if g > 0.05 else "warn" if g > 0 else "bad", f"TCAM du chiffre d'affaires : {g*100:.1f} % sur la période."))
    m = v("Marge opérationnelle (%)")
    if m == m:
        out.append(("ok" if m > 10 else "warn" if m > 3 else "bad", f"Marge opérationnelle de {m:.1f} % sur le dernier exercice."))
    roic = v("ROIC après impôt (%)")
    if roic == roic and wacc:
        spread = roic - wacc * 100
        lvl = "ok" if spread > 1 else "warn" if spread > -1 else "bad"
        out.append((lvl, f"ROIC {roic:.1f} % vs coût du capital {wacc*100:.1f} % → "
                         f"{'création' if spread > 0 else 'destruction'} de valeur (spread {spread:+.1f} pts)."))
    nd = v("Dette nette / EBITDA (x)")
    if nd == nd:
        out.append(("ok" if nd < 2 else "warn" if nd < 3.5 else "bad", f"Dette nette / EBITDA : {nd:.2f}x."))
    cov = v("Couverture des intérêts EBIT/Int (x)")
    if cov == cov:
        out.append(("ok" if cov > 5 else "warn" if cov > 2 else "bad", f"Couverture des intérêts : {cov:.1f}x."))
    cc = v("Conversion cash CFO/Résultat net (x)")
    if cc == cc:
        out.append(("ok" if cc > 0.9 else "warn" if cc > 0.6 else "bad",
                    f"Conversion du résultat en cash : {cc:.2f}x {'(qualité des résultats satisfaisante)' if cc > 0.9 else '(à surveiller : accruals élevés)'}."))
    lq = v("Liquidité générale (x)")
    if lq == lq:
        out.append(("ok" if lq > 1.2 else "warn" if lq > 0.9 else "bad", f"Liquidité générale : {lq:.2f}x."))
    return out
