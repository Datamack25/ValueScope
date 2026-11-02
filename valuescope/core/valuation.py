"""Moteur de valorisation inspiré d'Aswath Damodaran (NYU Stern) et des pratiques de place.

Méthodes :
  1. DCF FCFF à 2 phases (« Ginzu » simplifié) : croissance → convergence de marge → ventes/capital
     → croissance stable ≤ taux sans risque, ROIC terminal, WACC convergent, probabilité de défaillance.
  2. DCF actionnaire (FCFE / dividendes potentiels) en 2 phases, payout stable = 1 − g/ROE.
  3. Revenu résiduel (Ohlson / EVA) sur fonds propres.
  4. Multiples (EV/EBITDA, EV/CA, PER, P/B) — secteur ou comparables fournis.
  5. Actif net réévalué (ANR).
  + Monte Carlo, DCF inversé (croissance implicite du marché), synthèse et verdict.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd


@dataclass
class DCFInputs:
    revenue0: float
    margin0: float              # marge EBIT actuelle (décimal)
    g_high: float               # croissance années 1-5
    g_stable: float             # croissance perpétuelle (≤ rf)
    target_margin: float        # marge opérationnelle cible
    convergence_year: int       # année où la marge cible est atteinte
    sales_to_capital: float     # réinvestissement : ΔCA / réinvestissement
    tax_eff: float              # IS effectif actuel
    tax_marginal: float         # IS marginal (converge en fin de période)
    wacc: float                 # coût du capital initial
    wacc_stable: float          # coût du capital en régime stable
    roic_stable: float          # ROIC en régime stable
    cash: float
    debt: float
    minority: float = 0.0
    non_operating: float = 0.0
    shares: float = 1.0
    price: float | None = None
    p_failure: float = 0.0      # probabilité de défaillance
    distress_pct: float = 0.5   # % de la valeur récupérée en cas de défaillance
    years: int = 10
    options_value: float = 0.0  # stock-options / instruments dilutifs (valeur)


def _dcf_core(revenue0, margin0, g_high, g_stable, target_margin, conv_year, s2c, tax_eff, tax_marg,
              wacc, wacc_st, roic_st, years=10):
    """Vectorisé : chaque paramètre peut être un scalaire ou un tableau (N,1)."""
    t = np.arange(1, years + 1)
    half = years // 2
    # croissance : g_high sur la phase 1, puis convergence linéaire vers g_stable
    growth = np.where(t <= half, g_high, g_high - (g_high - g_stable) * (t - half) / (years - half))
    growth = np.broadcast_to(growth, np.broadcast(g_high, growth).shape).astype(float)
    rev = revenue0 * np.cumprod(1 + growth, axis=-1)
    w = np.clip(t / np.maximum(conv_year, 1), 0, 1)
    margin = margin0 + (target_margin - margin0) * w
    ebit = rev * margin
    tax = np.where(t <= half, tax_eff, tax_eff + (tax_marg - tax_eff) * (t - half) / (years - half))
    tax = np.where(ebit > 0, tax, 0.0)           # pas d'impôt sur pertes (simplification)
    nopat = ebit * (1 - tax)
    prev_rev = np.concatenate([np.broadcast_to(revenue0, rev[..., :1].shape), rev[..., :-1]], axis=-1)
    reinv = (rev - prev_rev) / s2c
    fcff = nopat - reinv
    rate = np.where(t <= half, wacc, wacc - (wacc - wacc_st) * (t - half) / (years - half))
    disc = np.cumprod(1 / (1 + rate), axis=-1)
    pv = fcff * disc
    # valeur terminale
    rev_t1 = rev[..., -1:] * (1 + g_stable)
    ebit_t1 = rev_t1 * target_margin
    nopat_t1 = ebit_t1 * (1 - tax_marg)
    reinv_rate = np.where(roic_st > 0, g_stable / roic_st, 0)
    fcff_t1 = nopat_t1 * (1 - reinv_rate)
    tv = fcff_t1 / (wacc_st - g_stable)
    pv_tv = tv * disc[..., -1:]
    return dict(t=t, growth=growth, rev=rev, margin=margin, ebit=ebit, tax=tax, nopat=nopat, reinv=reinv,
                fcff=fcff, rate=rate, disc=disc, pv=pv, tv=tv, pv_tv=pv_tv, fcff_t1=fcff_t1, reinv_rate=reinv_rate,
                op_value=pv.sum(axis=-1, keepdims=True) + pv_tv)


def dcf_fcff(i: DCFInputs) -> dict:
    r = _dcf_core(i.revenue0, i.margin0, i.g_high, i.g_stable, i.target_margin, i.convergence_year,
                  i.sales_to_capital, i.tax_eff, i.tax_marginal, i.wacc, i.wacc_stable, i.roic_stable, i.years)
    op = float(np.ravel(r["op_value"])[0])
    op_adj = op * (1 - i.p_failure) + op * i.distress_pct * i.p_failure
    equity = op_adj + i.cash + i.non_operating - i.debt - i.minority - i.options_value
    per_share = equity / i.shares if i.shares else np.nan
    table = pd.DataFrame({
        "Croissance CA": r["growth"], "Chiffre d'affaires": r["rev"], "Marge EBIT": r["margin"], "EBIT": r["ebit"],
        "Taux d'IS": r["tax"], "NOPAT": r["nopat"], "Réinvestissement": r["reinv"], "FCFF": r["fcff"],
        "Coût du capital": r["rate"], "Facteur d'actualisation": r["disc"], "VA des FCFF": r["pv"],
    }, index=[f"A+{k}" for k in r["t"]]).T
    # ROIC implicite (capital investi croissant avec le réinvestissement)
    return dict(table=table, op_value=op, op_value_adj=op_adj, tv=float(np.ravel(r["tv"])[0]), pv_tv=float(np.ravel(r["pv_tv"])[0]),
                pv_fcff=float(r["pv"].sum()), equity=equity, per_share=per_share,
                tv_share=float(np.ravel(r["pv_tv"])[0]) / op if op else np.nan, fcff_t1=float(np.ravel(r["fcff_t1"])[0]),
                reinv_rate_terminal=float(np.ravel(r["reinv_rate"])[0]))


def monte_carlo(i: DCFInputs, n=5000, sd_growth=0.03, sd_margin=0.02, sd_wacc=0.01, sd_s2c=0.25, seed=42):
    rng = np.random.default_rng(seed)
    g = rng.normal(i.g_high, sd_growth, (n, 1))
    m = rng.normal(i.target_margin, sd_margin, (n, 1))
    w = np.clip(rng.normal(i.wacc, sd_wacc, (n, 1)), i.g_stable + 0.01, None)
    ws = np.clip(w - (i.wacc - i.wacc_stable), i.g_stable + 0.01, None)
    s = np.clip(rng.normal(i.sales_to_capital, i.sales_to_capital * sd_s2c, (n, 1)), 0.2, None)
    r = _dcf_core(i.revenue0, i.margin0, g, i.g_stable, m, i.convergence_year, s, i.tax_eff, i.tax_marginal,
                  w, ws, i.roic_stable, i.years)
    op = r["op_value"].ravel()
    op = op * (1 - i.p_failure) + op * i.distress_pct * i.p_failure
    eq = op + i.cash + i.non_operating - i.debt - i.minority - i.options_value
    return eq / i.shares


def reverse_dcf(i: DCFInputs, price: float) -> float | None:
    """Croissance annuelle (phase 1) implicitement intégrée dans le cours actuel."""
    if not price or not i.shares:
        return None
    f = lambda g: dcf_fcff(DCFInputs(**{**asdict(i), "g_high": g}))["per_share"] - price
    lo, hi = -0.30, 0.80
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def sensitivity(i: DCFInputs, waccs, gs) -> pd.DataFrame:
    out = {}
    for w in waccs:
        out[f"{w*100:.1f} %"] = [
            dcf_fcff(DCFInputs(**{**asdict(i), "wacc": w, "wacc_stable": max(w - (i.wacc - i.wacc_stable), g + 0.005),
                                  "g_stable": g}))["per_share"] for g in gs]
    return pd.DataFrame(out, index=[f"{g*100:.1f} %" for g in gs])


def sensitivity_growth_margin(i: DCFInputs, gs, ms) -> pd.DataFrame:
    out = {}
    for m in ms:
        out[f"{m*100:.1f} %"] = [dcf_fcff(DCFInputs(**{**asdict(i), "g_high": g, "target_margin": m}))["per_share"] for g in gs]
    return pd.DataFrame(out, index=[f"{g*100:.1f} %" for g in gs])


def fcfe_ddm(ni0, roe, payout0, ke, g_stable, shares, years=5, roe_stable=None):
    """DCF actionnaire 2 phases. Croissance fondamentale g = ROE × (1 − payout)."""
    roe_stable = roe_stable if roe_stable else ke
    g1 = float(np.clip(roe * (1 - payout0), -0.05, 0.40))
    ni = ni0
    pv = 0.0
    rows = []
    for t in range(1, years + 1):
        ni *= (1 + g1)
        fcfe = ni * payout0
        d = fcfe / (1 + ke) ** t
        pv += d
        rows.append((t, ni, fcfe, d))
    payout_st = 1 - g_stable / roe_stable if roe_stable > 0 else 1
    tv = ni * (1 + g_stable) * payout_st / (ke - g_stable)
    pv_tv = tv / (1 + ke) ** years
    eq = pv + pv_tv
    table = pd.DataFrame(rows, columns=["Année", "Résultat net", "FCFE / dividende potentiel", "Valeur actuelle"]).set_index("Année").T
    return dict(equity=eq, per_share=eq / shares if shares else np.nan, g1=g1, payout_stable=payout_st, tv=tv, table=table)


def residual_income(bv0, roe0, ke, g_stable, payout, shares, years=10):
    """Revenu résiduel : VA = BV0 + Σ (ROE_t − ke)·BV_{t-1} / (1+ke)^t + VT. ROE converge vers ke + 1/3 du spread."""
    bv, pv, rows = bv0, 0.0, []
    roe_end = ke + (roe0 - ke) / 3
    for t in range(1, years + 1):
        roe = roe0 + (roe_end - roe0) * t / years
        ni = roe * bv
        ri = (roe - ke) * bv
        pv += ri / (1 + ke) ** t
        rows.append((t, bv, roe, ni, ri))
        bv = bv + ni * (1 - payout)
    tv = (roe_end - ke) * bv / (ke - g_stable)
    pv_tv = tv / (1 + ke) ** years
    eq = bv0 + pv + pv_tv
    table = pd.DataFrame(rows, columns=["Année", "Fonds propres début", "ROE", "Résultat net", "Revenu résiduel"]).set_index("Année").T
    return dict(equity=eq, per_share=eq / shares if shares else np.nan, table=table)


def multiples(metrics: dict, mult: dict, net_debt: float, minority: float, shares: float) -> pd.DataFrame:
    """metrics : ebitda, revenue, net_income, equity ; mult : EV_EBITDA, EV_Sales, PER, P_B."""
    rows = []
    def add(name, base, m, ev_based):
        if base is None or not np.isfinite(base) or m is None or not np.isfinite(m) or base <= 0:
            return
        v = base * m
        eq = v - net_debt - minority if ev_based else v
        rows.append((name, m, base, v if ev_based else np.nan, eq, eq / shares if shares else np.nan))
    add("EV / EBITDA", metrics.get("ebitda"), mult.get("EV_EBITDA"), True)
    add("EV / Chiffre d'affaires", metrics.get("revenue"), mult.get("EV_Sales"), True)
    add("PER (Cours / Bénéfice)", metrics.get("net_income"), mult.get("PER"), False)
    add("P/B (Cours / Valeur comptable)", metrics.get("equity"), mult.get("P_B"), False)
    return pd.DataFrame(rows, columns=["Méthode", "Multiple", "Agrégat", "Valeur d'entreprise", "Valeur des fonds propres", "Par action"])


def verdict(value: float, price: float | None, band: float = 0.15) -> tuple[str, str, float | None]:
    if not price or not np.isfinite(value):
        return "Valeur intrinsèque calculée", "info", None
    up = value / price - 1
    if up > band:
        return "SOUS-ÉVALUÉE", "ok", up
    if up < -band:
        return "SURÉVALUÉE", "bad", up
    return "CORRECTEMENT VALORISÉE", "warn", up
