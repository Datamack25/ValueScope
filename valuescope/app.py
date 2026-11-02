"""ValueScope — Plateforme d'analyse financière et de valorisation d'entreprises.
Inspirée de la démarche d'Aswath Damodaran (NYU Stern) : « Narrative & Numbers ».
Lancement : streamlit run app.py"""
from __future__ import annotations
import io
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.schema import FIELDS, SECTIONS
from core import parser as P
from core import analysis as A
from core import market as M
from core import valuation as V
from core import forecast as F
from core import reports as RP

APP_DIR = Path(__file__).resolve().parent      # chemins indépendants du dossier de lancement (Streamlit Cloud)
APP_VERSION = "4.1 · 02/11/2026"

st.set_page_config(page_title="ValueScope · Analyse & Valorisation", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")

# Couleurs de séries lisibles sur fond clair ET sombre (le thème clair/sombre est choisi par l'utilisateur)
NAVY, GOLD, TEAL, RED = "#3D6FB6", "#C9A227", "#1B998B", "#D1495B"

# ------------------------------------------------------------------ STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, input, textarea, button { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, .vs-title { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; max-width: 1320px; }
section[data-testid="stSidebar"] { background: #0B1F3A; }
section[data-testid="stSidebar"] * { color: #E9E4D8 !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #13294a !important; color:#fff !important; border-color:#2a4268 !important; }
section[data-testid="stSidebar"] hr { border-color: #2a4268; }
.vs-brand { font-family: 'Fraunces', serif; font-size: 1.75rem; font-weight: 700; color: #fff !important; line-height: 1; }
.vs-brand span { color: #C9A227 !important; }
.vs-tag { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #9fb0c8 !important; margin-top: .3rem; }
.vs-hero { background: linear-gradient(120deg, #0B1F3A 0%, #15325c 60%, #1f4677 100%); border-radius: 18px; padding: 2.2rem 2.4rem;
           color: #fff; position: relative; overflow: hidden; margin-bottom: 1.2rem; }
.vs-hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
                 border: 38px solid rgba(201,162,39,.18); }
.vs-hero h1 { color: #fff !important; font-size: 2.3rem; margin: 0 0 .4rem 0; }
.vs-hero p { color: #d5dcea; font-size: 1.02rem; max-width: 760px; margin: 0; position: relative; z-index: 1; }
.vs-hero h1, .vs-kicker { position: relative; z-index: 1; }
.vs-kicker { color: #C9A227; text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; font-weight: 600; margin-bottom: .5rem; }
/* Composants : couleurs neutres translucides + texte hérité du thème → lisibles en mode clair et sombre */
.vs-card { background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
           padding: 1rem 1.15rem; height: 100%; }
.vs-muted, .vs-kpi-l, .vs-kpi-s { opacity: .72; }
.vs-kpi-l { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 600; }
.vs-kpi-v { font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; margin-top: .2rem; }
.vs-kpi-s { font-size: .8rem; }
.vs-step { display:flex; gap:.8rem; align-items:flex-start; }
.vs-num { font-family:'Fraunces',serif; font-size:1.6rem; color:#C9A227; font-weight:700; line-height:1; }
.vs-pill { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.75rem; font-weight:600; border:1px solid; }
.vs-pill.ok { background:rgba(27,153,139,.16); color:#1B998B; border-color:rgba(27,153,139,.45); }
.vs-pill.warn { background:rgba(201,162,39,.16); color:#B8921A; border-color:rgba(201,162,39,.5); }
.vs-pill.bad { background:rgba(209,73,91,.15); color:#D1495B; border-color:rgba(209,73,91,.45); }
.vs-pill.info { background:rgba(61,111,182,.15); color:#4F82C9; border-color:rgba(61,111,182,.45); }
.vs-comment { padding:.55rem .8rem; border-radius:10px; margin-bottom:.4rem; font-size:.92rem; border-left:4px solid; }
.vs-comment.ok { background:rgba(27,153,139,.12); border-color:#1B998B; }
.vs-comment.warn { background:rgba(201,162,39,.13); border-color:#C9A227; }
.vs-comment.bad { background:rgba(209,73,91,.12); border-color:#D1495B; }
.vs-comment.info { background:rgba(61,111,182,.11); border-color:#3D6FB6; }
.vs-comment code { background: transparent; }
.vs-verdict { border-radius:18px; padding:1.6rem 1.8rem; color:#fff; }
.vs-verdict h2 { color:#fff !important; margin:0; font-size:2rem; }
.vs-verdict .big { font-family:'IBM Plex Mono',monospace; font-size:2.6rem; }
.vs-quote { font-family:'Fraunces',serif; font-style:italic; font-size:1.05rem; border-left:3px solid #C9A227; padding-left:1rem; }
div[data-testid="stTabs"] button p { font-weight:600; }
.stDownloadButton button, .stButton button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

ss = st.session_state
UNIT_FACTOR = {"Unités": 1e-6, "Milliers": 1e-3, "Millions": 1.0, "Milliards": 1e3}


# ------------------------------------------------------------------ HELPERS
from ui_kit import kpi, fnum, plot_layout, GRID  # noqa: E402
import ui_markets as UM  # noqa: E402
import ui_ml as UML  # noqa: E402
from core import store as STORE  # noqa: E402
from core import ml as ML  # noqa: E402


def has_data():
    return "df" in ss and ss.df is not None and not ss.df.empty


def load_df(df: pd.DataFrame, name: str | None = None, extra: dict | None = None):
    """Charge les données puis demande un rerun : les widgets de la barre latérale sont mis à jour en début de script."""
    ss.df = P.complete(df)
    pend = dict(extra or {})
    if name:
        pend["name"] = name
    ss._pending = pend
    ss._need_defaults = True


def set_defaults(force=False):
    """Hypothèses par défaut déduites des données, du pays et du secteur (à la Damodaran)."""
    if not has_data() or (ss.get("_defaults_set") and not force):
        return
    df = ss.df
    y = df.columns[-1]
    g = lambda k: float(df.at[k, y]) if pd.notna(df.at[k, y]) else np.nan
    cur, crp, tmarg = M.COUNTRIES[ss.country]
    cur = ss.get("cur_override") or cur          # société cotée : devise des comptes
    ind = M.INDUSTRY.loc[ss.industry]
    rf = M.CURRENCIES.get(cur, (4.0, ""))[0] / 100
    rev, ebit = g("revenue"), g("ebit")
    m0 = ebit / rev if rev else 0.1
    hist_g = A.cagr(df.loc["revenue"])
    ic = A.invested_capital(df)[y]
    s2c = rev / ic if ic and ic > 0 else ind.Sales_to_capital
    teff = A.effective_tax(df)[y]
    d = dict(rf=rf, erp=M.MATURE_ERP / 100, crp=crp / 100, lam=1.0, tax_marg=tmarg / 100,
             tax_eff=float(teff) if teff == teff else tmarg / 100,
             g_high=float(np.clip(hist_g if hist_g == hist_g else 0.04, -0.05, 0.40)),
             g_stable=float(min(rf, 0.025)), target_margin=float(m0 if m0 > 0 else ind.Marge_op / 100),
             conv=5, s2c=float(np.clip(s2c, 0.3, 5.0)), beta_u=float(ind.Beta_u), beta_mode="Bottom-up sectoriel",
             beta_manual=1.0, size_prem=0.0, p_fail=0.0, distress=0.5, roic_mode="ROIC terminal = coût du capital",
             roic_st=float(ind.ROIC / 100), non_op=0.0, options=0.0,
             m_EV_EBITDA=float(ind.EV_EBITDA), m_EV_Sales=float(ind.EV_Sales), m_PER=float(ind.PER), m_P_B=float(ind.P_B),
             anr_adj=0.0, w_dcf=40, w_fcfe=15, w_ri=15, w_mult=25, w_anr=5, band=15,
             sd_g=3.0, sd_m=2.0, sd_w=1.0, large=True)
    sh = g("shares")
    if sh == sh and sh > 0 and (force or not ss.get("shares_user")):
        ss["shares"] = sh
    stage = A.life_cycle(df)[0]
    if "croissance" in stage.lower() and "mature" not in stage.lower() or "Start-up" in stage:
        d.update(w_dcf=50, w_fcfe=10, w_ri=10, w_mult=30, w_anr=0)      # croissance : la valeur est dans le futur
    elif stage == "Déclin":
        d.update(w_dcf=30, w_fcfe=10, w_ri=10, w_mult=20, w_anr=30)     # déclin : la valeur liquidative compte
    for k, v in d.items():
        if force or k not in ss:
            ss[k] = v
    ss._defaults_set = True
    if force:   # resynchronise les champs de saisie en %
        for k in ("_rf", "_erp", "_crp", "_tmg", "_g1", "_gs", "_m", "_gh", "_tm"):
            ss.pop(k, None)


# ------------------------------------------------------------------ ÉTAT (avant tout widget)
PERSIST = ["rf", "erp", "crp", "lam", "tax_marg", "tax_eff", "g_high", "g_stable", "target_margin", "conv", "s2c", "beta_u",
           "beta_mode", "beta_manual", "size_prem", "p_fail", "distress", "roic_mode", "roic_st", "non_op", "options",
           "m_EV_EBITDA", "m_EV_Sales", "m_PER", "m_P_B", "anr_adj", "w_dcf", "w_fcfe", "w_ri", "w_mult", "w_anr", "band",
           "sd_g", "sd_m", "sd_w", "large"]
for k, v in dict(name="Ma société", country="France", industry="Marché total (moyenne)", unit="Millions",
                 price=0.0, shares=10.0, listed=True, page="Accueil", data_mode="En ligne (enregistre en CSV)").items():
    ss.setdefault(k, v)
_p = ss.pop("_pending", None)
if _p:
    ss.update(_p)
if ss.pop("_need_defaults", False):
    set_defaults(force=True)
else:
    set_defaults()
for k in PERSIST:           # évite la suppression des valeurs quand la page du widget n'est pas affichée
    if k in ss:
        ss[k] = ss[k]

# ------------------------------------------------------------------ SIDEBAR

with st.sidebar:
    st.markdown("<div class='vs-brand'>Value<span>Scope</span></div><div class='vs-tag'>Analyse · Prévision · Valorisation</div>",
                unsafe_allow_html=True)
    st.write("")
    pages = ["Accueil", "🏢 Grandes entreprises", "🌍 Analyse pays", "🛢 Matières premières", "🤖 Machine learning", "💾 Base de données", "1 · Import des états", "2 · Diagnostic", "3 · Prévisions", "4 · Coût du capital",
             "5 · Valorisation", "6 · Verdict & rapports", "Méthodologie"]
    if ss.page not in pages:
        ss.page = "Accueil"
    st.radio("Navigation", pages, key="page", label_visibility="collapsed")
    st.divider()
    st.text_input("Société", key="name")
    c1, c2 = st.columns(2)
    c1.selectbox("Pays", list(M.COUNTRIES), key="country",
                 on_change=lambda: (ss.pop("cur_override", None), set_defaults(force=True)))
    c2.selectbox("Unité", list(UNIT_FACTOR), key="unit", help="Unité des montants des états financiers")
    st.selectbox("Secteur (référentiel Damodaran)", list(M.INDUSTRY.index), key="industry",
                 on_change=lambda: set_defaults(force=True))
    st.toggle("Société cotée", key="listed")
    c1, c2 = st.columns(2)
    c1.number_input("Cours actuel", min_value=0.0, step=0.5, key="price", help="0 = non coté / pas de comparaison")
    c2.number_input("Actions (millions)", min_value=0.0001, step=0.1, key="shares", format="%.4f",
                    on_change=lambda: ss.__setitem__("shares_user", True))
    st.radio("Source des données marchés", ["En ligne (enregistre en CSV)", "Hors-ligne (base CSV uniquement)"],
             key="data_mode", help="En ligne : données en direct, sauvegardées dans la base CSV. Hors-ligne : uniquement la base CSV.")
    with st.expander("Récupérer cours (Yahoo Finance)"):
        tk = st.text_input("Ticker (ex. MC.PA, AAPL)", key="ticker")
        if st.button("Charger le cours", width="stretch") and tk:
            r = M.fetch_market(tk)
            if r.get("ok") and r.get("price"):
                pend = {"price": float(r["price"])}
                if r.get("shares"):
                    pend.update(shares=float(r["shares"]), shares_user=True)
                if r.get("name"):
                    pend["name"] = r["name"]
                ss._pending = pend
                ss._flash = f"{r.get('name')} : cours {r['price']} {r.get('currency','')}"
                st.rerun()
            else:
                st.error("Données indisponibles (connexion ou ticker).")
    cur = ss.get("cur_override") or M.COUNTRIES[ss.country][0]
    st.caption(f"ValueScope v{APP_VERSION}")
    st.caption(f"Devise de référence : **{cur}** · Montants en **{ss.unit.lower()}**")


# ------------------------------------------------------------------ ENGINE
def compute() -> dict:
    df = ss.df
    y = df.columns[-1]
    g = lambda k: float(df.at[k, y]) if pd.notna(df.at[k, y]) else np.nan
    to_m = UNIT_FACTOR[ss.unit]       # montants -> millions (actions en millions)
    price = ss.price if ss.price and ss.price > 0 else None
    shares = ss.shares
    cash, debt, minority = g("cash") or 0, g("total_debt") or 0, g("minority") or 0
    equity_bv = g("equity")
    # --- coût du capital
    mkt_cap = price * shares / to_m if price else equity_bv   # dans l'unité des états
    de = debt / mkt_cap if mkt_cap and mkt_cap > 0 else 0
    if ss.beta_mode == "Bottom-up sectoriel":
        beta = M.lever_beta(ss.beta_u, de, ss.tax_marg)
    else:
        beta = ss.beta_manual
    ke = ss.rf + beta * ss.erp + ss.lam * ss.crp + ss.size_prem / 100
    cov = g("ebit") / g("interest_expense") if g("interest_expense") and g("interest_expense") > 0 else float("inf")
    rating, spread = M.synthetic_rating(cov, ss.large)
    kd = ss.rf + spread / 100 + (ss.crp if ss.crp > 0 else 0) * 0.7
    wd = debt / (debt + mkt_cap) if (debt + mkt_cap) > 0 else 0
    wacc = ke * (1 - wd) + kd * (1 - ss.tax_marg) * wd
    ke_st = ss.rf + 1.0 * ss.erp + ss.lam * ss.crp
    wacc_st = ke_st * (1 - wd) + kd * (1 - ss.tax_marg) * wd
    wacc_st = max(wacc_st, ss.g_stable + 0.01)
    roic_st = wacc_st if ss.roic_mode.startswith("ROIC terminal = coût") else ss.roic_st

    r = A.ratios(df, ss.tax_marg)
    rev, ebit = g("revenue"), g("ebit")
    inp = V.DCFInputs(revenue0=rev, margin0=ebit / rev, g_high=ss.g_high, g_stable=ss.g_stable,
                      target_margin=ss.target_margin, convergence_year=int(ss.conv), sales_to_capital=ss.s2c,
                      tax_eff=ss.tax_eff, tax_marginal=ss.tax_marg, wacc=wacc, wacc_stable=wacc_st, roic_stable=roic_st,
                      cash=cash, debt=debt, minority=minority, non_operating=ss.non_op, shares=shares / to_m,
                      price=price, p_failure=ss.p_fail, distress_pct=ss.distress, options_value=ss.options)
    dcf = V.dcf_fcff(inp)
    mc = V.monte_carlo(inp, sd_growth=ss.sd_g / 100, sd_margin=ss.sd_m / 100, sd_wacc=ss.sd_w / 100)
    ni = g("net_income")
    roe = ni / equity_bv if equity_bv and equity_bv > 0 else 0.1
    payout = float(np.clip(((g("dividends") or 0) + (g("buybacks") or 0)) / ni, 0, 1)) if ni and ni > 0 else 0.4
    payout_eq = max(payout, 1 - (inp.g_high / roe if roe > 0 else 0))  # FCFE potentiel ≥ dividendes versés
    fcfe = V.fcfe_ddm(ni, roe, float(np.clip(payout_eq, 0.1, 1)), ke, ss.g_stable, shares / to_m) if ni and ni > 0 else None
    ri = V.residual_income(equity_bv, roe, ke, ss.g_stable, payout, shares / to_m) if equity_bv and equity_bv > 0 else None
    mult = V.multiples(dict(ebitda=g("ebitda"), revenue=rev, net_income=ni, equity=equity_bv),
                       dict(EV_EBITDA=ss.m_EV_EBITDA, EV_Sales=ss.m_EV_Sales, PER=ss.m_PER, P_B=ss.m_P_B),
                       debt - cash, minority, shares / to_m)
    anr = (equity_bv + ss.anr_adj) / (shares / to_m) if equity_bv else np.nan

    rows = []
    p10, p90 = np.percentile(mc, [10, 90])
    rows.append(("DCF FCFF (Damodaran)", p10, dcf["per_share"], p90, ss.w_dcf))
    if fcfe:
        rows.append(("DCF actionnaire (FCFE/DDM)", fcfe["per_share"] * .85, fcfe["per_share"], fcfe["per_share"] * 1.15, ss.w_fcfe))
    if ri:
        rows.append(("Revenu résiduel (EVA/Ohlson)", ri["per_share"] * .85, ri["per_share"], ri["per_share"] * 1.15, ss.w_ri))
    if not mult.empty:
        rows.append(("Multiples (moyenne)", mult["Par action"].min(), mult["Par action"].mean(), mult["Par action"].max(), ss.w_mult))
    if anr == anr:
        rows.append(("Actif net réévalué (ANR)", anr * .9, anr, anr * 1.1, ss.w_anr))
    summ = pd.DataFrame(rows, columns=["Méthode", "Bas", "Par action", "Haut", "Poids"])
    valid = summ[np.isfinite(summ["Par action"]) & (summ["Poids"] > 0)]
    fair = float(np.average(valid["Par action"], weights=valid["Poids"])) if not valid.empty else dcf["per_share"]
    verdict, lvl, up = V.verdict(fair, price, ss.band / 100)
    lc = A.life_cycle(df)
    scen = F.scenarios(F.historical_drivers(df), ss.g_high, ss.target_margin, ss.tax_marg)
    sens = V.sensitivity(inp, [wacc - .02, wacc - .01, wacc, wacc + .01, wacc + .02],
                         [ss.g_stable - .01, ss.g_stable - .005, ss.g_stable, ss.g_stable + .005, min(ss.g_stable + .01, ss.rf)])
    implied = V.reverse_dcf(inp, price) if price else None
    narrative = (f"{ss.name} se situe au stade « {lc[0]} » de son cycle de vie. Notre récit : un chiffre d'affaires de "
                 f"{fnum(rev,0)} ({ss.unit.lower()}) qui croît de {ss.g_high*100:.1f} % par an sur 5 ans puis converge vers "
                 f"{ss.g_stable*100:.1f} % ; une marge opérationnelle qui passe de {ebit/rev*100:.1f} % à {ss.target_margin*100:.1f} % ; "
                 f"une croissance financée à raison de {ss.s2c:.2f} € de ventes supplémentaires par € réinvesti. Le coût du capital "
                 f"ressort à {wacc*100:.1f} % (coût des fonds propres {ke*100:.1f} %, bêta {beta:.2f}). "
                 f"La valeur terminale pèse {dcf['tv_share']*100:.0f} % de la valeur opérationnelle."
                 + (f" Le cours actuel ({price:.2f}) intègre une croissance implicite de {implied*100:.1f} % par an." if implied is not None else ""))
    return dict(df=df, ratios=r, wacc=wacc, ke=ke, kd=kd, beta=beta, rating=rating, cov=cov, wd=wd, de=de, wacc_st=wacc_st,
                inp=inp, dcf=dcf, mc=mc, fcfe=fcfe, ri=ri, mult=mult, anr=anr, summary=summ, fair_value=fair,
                verdict=verdict, verdict_lvl=lvl, upside=up, price=price, lifecycle=lc, scen=scen, sens=sens,
                implied=implied, narrative=narrative, name=ss.name, unit=f"{ss.unit} {cur}", country=ss.country,
                industry=ss.industry, altman=A.altman_z(df, mkt_cap if price else None, ss.listed and bool(price)),
                piotroski=A.piotroski(df)["score"], comments=A.commentary(df, r, wacc), roe=roe, payout=payout,
                wacc_detail={"Taux sans risque": fnum(ss.rf, 2, True), "Prime de risque actions (ERP)": fnum(ss.erp, 2, True),
                             "Prime de risque pays (CRP × λ)": fnum(ss.crp * ss.lam, 2, True), "Bêta endetté": f"{beta:.2f}",
                             "Coût des fonds propres": fnum(ke, 2, True), "Notation synthétique": f"{rating} (couverture {fnum(cov,1)}x)",
                             "Coût de la dette avant impôt": fnum(kd, 2, True), "Poids de la dette": fnum(wd, 1, True),
                             "WACC": fnum(wacc, 2, True), "WACC en régime stable": fnum(wacc_st, 2, True)},
                assumptions={"Croissance phase 1 (A1-A5)": fnum(ss.g_high, 1, True), "Croissance perpétuelle": fnum(ss.g_stable, 1, True),
                             "Marge EBIT actuelle": fnum(ebit / rev, 1, True), "Marge EBIT cible": fnum(ss.target_margin, 1, True),
                             "Année de convergence": str(ss.conv), "Ventes / capital": f"{ss.s2c:.2f}",
                             "IS effectif → marginal": f"{ss.tax_eff*100:.1f} % → {ss.tax_marg*100:.1f} %",
                             "ROIC terminal": fnum(roic_st, 1, True), "Probabilité de défaillance": fnum(ss.p_fail, 0, True),
                             "Trésorerie": fnum(cash, 1), "Dette (y.c. locative)": fnum(debt, 1), "Actions (millions)": fnum(shares, 3)})


# ------------------------------------------------------------------ PAGES
def page_home():
    st.markdown("""<div class='vs-hero'><div class='vs-kicker'>Narrative & Numbers</div>
    <h1>Toute valorisation est une histoire racontée en chiffres.</h1>
    <p>Importez les états financiers d'une société française, américaine ou d'ailleurs — Excel, CSV, PDF, Word ou image.
    ValueScope reconstruit les comptes, diagnostique la performance, projette les flux et confronte cinq méthodes de valorisation
    au cours de marché pour dire si l'action est sous- ou surévaluée.</p></div>""", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [("01", "Importer", "Excel, CSV, PDF, Word, image (OCR). Libellés PCG / liasse et US GAAP / IFRS reconnus."),
             ("02", "Diagnostiquer", "30 ratios, DuPont, ROIC vs WACC, Altman, Piotroski, cycle de vie."),
             ("03", "Valoriser", "DCF FCFF, FCFE, revenu résiduel, multiples, ANR, Monte Carlo, DCF inversé."),
             ("04", "Livrer", "Rapport PDF, Word, classeur Excel et graphiques PNG prêts à partager.")]
    for c, (n, t, d) in zip(cols, steps):
        c.markdown(f"<div class='vs-card'><div class='vs-step'><div class='vs-num'>{n}</div><div><b>{t}</b><br>"
                   f"<span class='vs-muted' style='font-size:.9rem'>{d}</span></div></div></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("### Explorer les marchés")
    m = st.columns(5)
    for col, (pg, t, d) in zip(m, [
        ("🏢 Grandes entreprises", "🏢 Grandes entreprises", "CAC 40, États-Unis, Europe, Asie : comptes publiés, multiples, valorisation express."),
        ("🌍 Analyse pays", "🌍 Analyse pays", "Dette publique, déficits, soutenabilité, croissance, inflation — données FMI & Banque mondiale."),
        ("🛢 Matières premières", "🛢 Matières premières", "Pétrole, gaz, or, cuivre, lithium, blé, cacao… tendances, volatilité, corrélations."),
        ("🤖 Machine learning", "🤖 Machine learning", "Typologies, vrais comparables, anomalies, prévisions testées, risque de dégradation."),
        ("💾 Base de données", "💾 Base de données", "Base CSV des comptes, séries macro, cours et analyses — fonctionne hors-ligne.")]):
        col.markdown(f"<div class='vs-card'><b>{t}</b><br><span class='vs-muted' style='font-size:.9rem'>{d}</span></div>",
                     unsafe_allow_html=True)
        if col.button(f"Ouvrir", key=f"go_{pg}", width="stretch"):
            ss._pending = {"page": pg}; st.rerun()
    st.write("")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("### Analyser une société en 30 secondes")
        st.write("Testez la plateforme avec une société fictive, ou passez directement à l'import de vos propres états.")
        b1, b2 = st.columns(2)
        if b1.button("Exemple 🇫🇷 Atelier Méridien SA", width="stretch"):
            df, _ = P.parse_any(open(APP_DIR / "data" / "exemple_societe_francaise.csv", "rb").read(), "x.csv")
            load_df(df, "Atelier Méridien SA", dict(country="France", industry="Industrie diversifiée", unit="Millions",
                                                    price=82.0, shares_user=True))
            ss._pending = {**ss.get("_pending", {}), "page": "2 · Diagnostic"}; st.rerun()
        if b2.button("Exemple 🇺🇸 Northwind Software", width="stretch"):
            df, _ = P.parse_any(open(APP_DIR / "data" / "sample_us_company.xlsx", "rb").read(), "x.xlsx")
            load_df(df, "Northwind Software Inc.", dict(country="États-Unis", industry="Logiciels & SaaS", unit="Millions",
                                                        price=95.0, shares_user=True))
            ss._pending = {**ss.get("_pending", {}), "page": "2 · Diagnostic"}; st.rerun()
        if st.button("→ Importer mes états financiers", type="primary"):
            ss._pending = {**ss.get("_pending", {}), "page": "1 · Import des états"}; st.rerun()
    with c2:
        st.markdown("<div class='vs-quote'>« Valuation is not an exercise in precision. It is about being less wrong "
                    "than the market, and having a story that holds together with the numbers. »</div>"
                    "<div class='vs-muted' style='font-size:.8rem;margin-top:.4rem'>Esprit de la démarche d'A. Damodaran — paraphrase</div>",
                    unsafe_allow_html=True)
        st.write("")
        st.caption(f"Référence marché pré-chargée : ERP implicite S&P 500 au 1er oct. 2026 = {M.MATURE_ERP:.2f} % ; "
                   f"T-bond US 10 ans {M.CURRENCIES['USD'][0]:.2f} %. Toutes les hypothèses restent modifiables.")


def page_import():
    st.markdown("## Import des états financiers")
    st.caption("Plusieurs fichiers possibles (ex. bilan en PDF + compte de résultat en Excel) : ils sont fusionnés par année.")
    files = st.file_uploader("Déposez vos fichiers", type=["xlsx", "xls", "xlsm", "csv", "txt", "pdf", "docx", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
                             accept_multiple_files=True)
    c1, c2 = st.columns([1, 1])
    tmpl = pd.DataFrame({"Poste": [v[0] for v in FIELDS.values()], "2022": "", "2023": "", "2024": "", "2025": ""})
    b = io.BytesIO(); tmpl.to_excel(b, index=False)
    c1.download_button("⬇ Modèle Excel vierge", b.getvalue(), "modele_etats_financiers.xlsx", width="stretch")
    if c2.button("Partir d'un tableau vide (saisie manuelle)", width="stretch"):
        e = pd.DataFrame(np.nan, index=list(FIELDS), columns=[2023, 2024, 2025]); load_df(e); st.rerun()
    if files and st.button("Analyser les fichiers", type="primary"):
        frames, logs = [], []
        for f in files:
            with st.spinner(f"Lecture de {f.name}…"):
                df, msg = P.parse_any(f.getvalue(), f.name)
            n = int(df.notna().any(axis=1).sum()) if not df.empty else 0
            logs.append((f.name, n, msg))
            if not df.empty:
                frames.append(df)
        for name, n, msg in logs:
            lvl = "ok" if n >= 10 else "warn" if n > 0 else "bad"
            st.markdown(f"<span class='vs-pill {lvl}'>{n} postes</span> &nbsp; {name}" +
                        (f" — <i>{msg[:180]}</i>" if msg and n == 0 else ""), unsafe_allow_html=True)
        if frames:
            load_df(P._merge(frames), extra={"shares_user": False})
            ss._flash = "Données importées. Vérifiez et corrigez ci-dessous si nécessaire."
            st.rerun()
    if has_data():
        df = ss.df
        cover = df.notna().any(axis=1)
        essentials = ["revenue", "ebit", "net_income", "equity", "total_assets", "total_debt", "cash", "capex", "cfo"]
        miss = [FIELDS[k][0] for k in essentials if not cover.get(k, False)]
        k1, k2, k3 = st.columns(3)
        kpi(k1, "Exercices", f"{len(df.columns)}", " · ".join(str(c) for c in df.columns))
        kpi(k2, "Postes renseignés", f"{int(cover.sum())}/{len(FIELDS)}", "après reconstitution des agrégats")
        kpi(k3, "Postes clés manquants", f"{len(miss)}", ", ".join(miss[:4]) or "Aucun ✓")
        st.write("")
        st.markdown("#### Vérifier / corriger les données")
        st.caption("Les charges sont saisies en positif. Ajoutez une année : bouton ci-dessous. Les montants sont dans l'unité choisie à gauche.")
        ed = st.data_editor(P.to_editor(df), width="stretch", height=560, disabled=["Poste"], key=f"ed_{id(df)}")
        c1, c2 = st.columns(2)
        if c1.button("✓ Valider les corrections", type="primary", width="stretch"):
            new = P.from_editor(ed); new.index = list(FIELDS); load_df(new); st.rerun()
        if c2.button("+ Ajouter l'exercice suivant", width="stretch"):
            new = df.copy(); new[int(df.columns[-1]) + 1] = np.nan; ss.df = new; st.rerun()


def page_diag(R):
    df, r = R["df"], R["ratios"]
    y = df.columns[-1]
    st.markdown(f"## Diagnostic financier — {R['name']}")
    st.markdown(f"<span class='vs-pill info'>Cycle de vie : {R['lifecycle'][0]}</span> &nbsp; "
                f"<span class='vs-pill {'ok' if R['altman']['zone']=='Sûre' else 'warn' if R['altman']['zone']=='Grise' else 'bad'}'>"
                f"Altman {fnum(R['altman']['score'],2)} · {R['altman']['zone']}</span> &nbsp; "
                f"<span class='vs-pill {'ok' if R['piotroski']>=7 else 'warn' if R['piotroski']>=4 else 'bad'}'>Piotroski {R['piotroski']}/9</span>",
                unsafe_allow_html=True)
    st.write("")
    c = st.columns(5)
    kpi(c[0], f"CA {y}", fnum(df.at["revenue", y], 0), f"TCAM {fnum(A.cagr(df.loc['revenue']),1,True)}")
    em = fnum(r.at["Marge d'EBITDA (%)", y], 1)
    kpi(c[1], "Marge EBIT", fnum(r.at["Marge opérationnelle (%)", y], 1) + " %", f"EBITDA {em} %")
    kpi(c[2], "ROIC", fnum(r.at["ROIC après impôt (%)", y], 1) + " %", f"WACC {fnum(R['wacc'],1,True)}")
    kpi(c[3], "Dette nette/EBITDA", fnum(r.at["Dette nette / EBITDA (x)", y], 2) + "x", f"Gearing {fnum(r.at['Gearing (Dette nette/CP) (%)', y],0)} %")
    kpi(c[4], "Free cash-flow", fnum(r.at["Free cash-flow (CFO − Capex)", y], 0), f"CFO/RN {fnum(r.at['Conversion cash CFO/Résultat net (x)', y],2)}x")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Lecture de l'analyste", "Graphiques", "Tous les ratios", "Scores détaillés"])
    with t1:
        st.markdown(f"<div class='vs-quote'>{R['lifecycle'][1]}</div>", unsafe_allow_html=True)
        st.write("")
        for lvl, txt in R["comments"]:
            st.markdown(f"<div class='vs-comment {lvl}'>{txt}</div>", unsafe_allow_html=True)
    with t2:
        yrs = [str(c) for c in df.columns]
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=yrs, y=df.loc["revenue"], name="CA", marker_color=NAVY)
        fig.add_bar(x=yrs, y=df.loc["ebitda"], name="EBITDA", marker_color=GOLD)
        fig.add_bar(x=yrs, y=df.loc["net_income"], name="Résultat net", marker_color=TEAL)
        a.plotly_chart(plot_layout(fig, title="Activité et résultats"), width="stretch")
        fig = go.Figure()
        for k, col in [("ROE (%)", NAVY), ("ROIC après impôt (%)", TEAL), ("Marge opérationnelle (%)", GOLD)]:
            fig.add_scatter(x=yrs, y=r.loc[k], name=k, mode="lines+markers", line=dict(color=col, width=3))
        fig.add_hline(y=R["wacc"] * 100, line_dash="dash", line_color=RED, annotation_text="WACC")
        b.plotly_chart(plot_layout(fig, title="Rentabilité vs coût du capital (%)"), width="stretch")
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_bar(x=yrs, y=df.loc["cfo"], name="CFO", marker_color=NAVY)
        fig.add_bar(x=yrs, y=-df.loc["capex"], name="Capex", marker_color=RED)
        fig.add_scatter(x=yrs, y=r.loc["Free cash-flow (CFO − Capex)"], name="FCF", mode="lines+markers", line=dict(color=GOLD, width=3))
        a.plotly_chart(plot_layout(fig, title="Génération de trésorerie"), width="stretch")
        fig = go.Figure()
        for k, col in [("DSO – Délai clients (jours)", NAVY), ("DIO – Rotation stocks (jours)", GOLD), ("DPO – Délai fournisseurs (jours)", TEAL)]:
            fig.add_scatter(x=yrs, y=r.loc[k], name=k.split("–")[1].split("(")[0], mode="lines+markers", line=dict(color=col, width=3))
        b.plotly_chart(plot_layout(fig, title="Cycle d'exploitation (jours)"), width="stretch")
    with t3:
        rr = r.copy(); rr.columns = [str(c) for c in rr.columns]
        st.dataframe(rr.style.format("{:,.2f}", na_rep="—"), width="stretch", height=820)
    with t4:
        a, b = st.columns(2)
        a.markdown(f"**{R['altman']['model']}** : {fnum(R['altman']['score'],2)} → zone **{R['altman']['zone']}**")
        a.caption("Z (cotées) : > 2,99 sûre ; 1,81–2,99 grise ; < 1,81 détresse. Z'' : > 2,6 / 1,1–2,6 / < 1,1.")
        det = A.piotroski(df)["details"]
        b.markdown(f"**Piotroski F-score : {R['piotroski']}/9**")
        for k, v in det.items():
            b.markdown(f"{'✅' if v else '❌'} {k}")


def page_forecast(R):
    st.markdown("## Prévisions")
    st.caption("Trois scénarios cohérents avec les hypothèses du DCF (croissance phase 1 et marge cible), plus une tendance statistique.")
    c1, c2, c3 = st.columns(3)
    c1.slider("Croissance annuelle centrale (%)", -10.0, 60.0, float(ss.g_high * 100), 0.5, key="_gh",
              on_change=lambda: ss.__setitem__("g_high", ss._gh / 100))
    c2.slider("Marge EBIT cible (%)", -20.0, 60.0, float(ss.target_margin * 100), 0.5, key="_tm",
              on_change=lambda: ss.__setitem__("target_margin", ss._tm / 100))
    drv = F.historical_drivers(R["df"])
    kpi(c3, "TCAM historique", fnum(drv["growth"], 1, True), f"Marge EBIT moy. 3 ans {fnum(drv['ebit_margin'],1,True)}")
    df = R["df"]
    hist = df.loc["revenue"].dropna()
    trend = F.trend_projection(hist)
    fig = go.Figure()
    fig.add_scatter(x=[str(i) for i in hist.index], y=hist, name="Historique", mode="lines+markers", line=dict(color=NAVY, width=3))
    cols = {"Pessimiste": RED, "Central": GOLD, "Optimiste": TEAL}
    for n, sc in R["scen"].items():
        x = [str(hist.index[-1])] + [str(i) for i in sc.index]
        yv = [hist.iloc[-1]] + list(sc["Chiffre d'affaires"])
        fig.add_scatter(x=x, y=yv, name=n, mode="lines+markers", line=dict(color=cols[n], width=2.5, dash="dot" if n != "Central" else None))
    if not trend.empty:
        fig.add_scatter(x=[str(i) for i in trend.index], y=trend, name="Tendance log-linéaire", line=dict(color="#8A94A6", dash="dash"))
    st.plotly_chart(plot_layout(fig, 420, "Chiffre d'affaires : historique et projections"), width="stretch")
    tabs = st.tabs(list(R["scen"]))
    for t, (n, sc) in zip(tabs, R["scen"].items()):
        s = sc.copy(); s.index = [str(i) for i in s.index]
        t.dataframe(s.T.style.format(lambda v: f"{v*100:.1f} %" if abs(v) < 1 and v != 0 else f"{v:,.1f}"), width="stretch")


def page_wacc(R):
    st.markdown("## Coût du capital")
    st.caption("Approche Damodaran : taux sans risque de la devise + bêta bottom-up sectoriel ré-endetté × ERP mature + λ × prime de risque pays ; "
               "coût de la dette par notation synthétique (couverture des intérêts).")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Fonds propres**")
        st.number_input("Taux sans risque (%)", value=ss.rf * 100, step=0.05, key="_rf", on_change=lambda: ss.__setitem__("rf", ss._rf / 100))
        st.caption(M.CURRENCIES.get(M.COUNTRIES[ss.country][0], (0, ""))[1])
        st.number_input("Prime de risque actions mature (%)", value=ss.erp * 100, step=0.05, key="_erp", on_change=lambda: ss.__setitem__("erp", ss._erp / 100))
        st.number_input("Prime de risque pays CRP (%)", value=ss.crp * 100, step=0.05, key="_crp", on_change=lambda: ss.__setitem__("crp", ss._crp / 100))
        st.slider("Exposition λ au risque pays", 0.0, 2.0, key="lam", step=0.05,
                  help="Part des revenus/opérations exposés au pays (1 = exposition moyenne)")
        st.number_input("Prime de taille / spécifique (%)", min_value=0.0, step=0.25, key="size_prem")
    with b:
        st.markdown("**Bêta**")
        st.radio("Méthode", ["Bottom-up sectoriel", "Saisie manuelle (régression, Yahoo…)"], key="beta_mode")
        st.number_input("Bêta désendetté du secteur", step=0.05, key="beta_u")
        st.number_input("Bêta manuel (endetté)", step=0.05, key="beta_manual")
        st.number_input("IS marginal (%)", value=ss.tax_marg * 100, step=0.5, key="_tmg", on_change=lambda: ss.__setitem__("tax_marg", ss._tmg / 100))
    with c:
        st.markdown("**Dette**")
        st.toggle("Grande entreprise (barème notation)", key="large")
        kpi(st, "Notation synthétique", R["rating"], f"Couverture EBIT/intérêts {fnum(R['cov'],1)}x")
        st.write("")
        kpi(st, "D / (D+E)", fnum(R["wd"], 1, True), "en valeurs de marché" if R["price"] else "en valeurs comptables (non coté)")
    st.write("")
    k = st.columns(5)
    kpi(k[0], "Bêta endetté", f"{R['beta']:.2f}")
    kpi(k[1], "Coût des FP", fnum(R["ke"], 2, True))
    kpi(k[2], "Coût de la dette", fnum(R["kd"], 2, True), "avant impôt")
    kpi(k[3], "WACC", fnum(R["wacc"], 2, True))
    kpi(k[4], "WACC stable", fnum(R["wacc_st"], 2, True), "bêta → 1")
    with st.expander("Référentiel sectoriel indicatif (modifiable via la sélection du secteur)"):
        st.dataframe(M.INDUSTRY, width="stretch")
        st.caption("Ordres de grandeur inspirés des tables globales de Damodaran. Mettez-les à jour depuis pages.stern.nyu.edu/~adamodar (Data).")


def page_valuation(R):
    st.markdown("## Valorisation")
    t1, t2, t3, t4, t5 = st.tabs(["DCF FCFF (Damodaran)", "Méthodes actionnaires", "Multiples", "Risque & Monte Carlo", "Sensibilités"])
    with t1:
        a, b, c = st.columns(3)
        a.number_input("Croissance A1–A5 (%)", value=ss.g_high * 100, step=0.5, key="_g1", on_change=lambda: ss.__setitem__("g_high", ss._g1 / 100))
        a.number_input("Croissance perpétuelle (%)", value=min(ss.g_stable, ss.rf) * 100, step=0.1, max_value=ss.rf * 100, key="_gs",
                       on_change=lambda: ss.__setitem__("g_stable", ss._gs / 100), help="Plafonnée au taux sans risque (règle de Damodaran)")
        b.number_input("Marge EBIT cible (%)", value=ss.target_margin * 100, step=0.5, key="_m",
                       on_change=lambda: ss.__setitem__("target_margin", ss._m / 100))
        b.slider("Année de convergence de la marge", 1, 10, key="conv")
        c.number_input("Ventes / capital investi", step=0.05, min_value=0.1, key="s2c", help="€ de CA supplémentaire par € réinvesti")
        c.selectbox("ROIC en régime stable", ["ROIC terminal = coût du capital", "ROIC terminal personnalisé"], key="roic_mode")
        if ss.roic_mode != "ROIC terminal = coût du capital":
            c.number_input("ROIC terminal (décimal)", step=0.01, key="roic_st")
        d = R["dcf"]
        k = st.columns(4)
        kpi(k[0], "VA des FCFF (10 ans)", fnum(d["pv_fcff"], 0))
        kpi(k[1], "VA valeur terminale", fnum(d["pv_tv"], 0), f"{d['tv_share']*100:.0f} % de la valeur")
        kpi(k[2], "Valeur des fonds propres", fnum(d["equity"], 0), f"+ tréso − dette − minoritaires")
        kpi(k[3], "Valeur / action", fnum(d["per_share"], 2), f"Cours {fnum(R['price'],2)}" if R["price"] else "")
        st.write("")
        t = d["table"].copy()
        pct_rows = ["Croissance CA", "Marge EBIT", "Taux d'IS", "Coût du capital"]
        st.dataframe(t.style.format(lambda v: f"{v:,.1f}").format(lambda v: f"{v*100:.2f} %", subset=pd.IndexSlice[pct_rows, :])
                     .format("{:.4f}", subset=pd.IndexSlice[["Facteur d'actualisation"], :]), width="stretch")
        fig = go.Figure(go.Waterfall(x=["VA FCFF", "VA valeur terminale", "Défaillance", "Trésorerie", "Actifs hors exploit.", "Dette", "Minoritaires & options", "Fonds propres"],
                                     measure=["relative"] * 7 + ["total"],
                                     y=[d["pv_fcff"], d["pv_tv"], d["op_value_adj"] - d["op_value"], R["inp"].cash, R["inp"].non_operating,
                                        -R["inp"].debt, -(R["inp"].minority + R["inp"].options_value), 0],
                                     increasing=dict(marker_color=TEAL), decreasing=dict(marker_color=RED), totals=dict(marker_color=NAVY)))
        fig = plot_layout(fig, 380, "Du FCFF à la valeur des fonds propres")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
    with t2:
        a, b = st.columns(2)
        with a:
            st.markdown("#### DCF actionnaire (FCFE / dividendes potentiels)")
            if R["fcfe"]:
                f = R["fcfe"]
                st.markdown(f"Croissance fondamentale ROE × rétention = **{fnum(f['g1'],1,True)}** · payout stable "
                            f"1 − g/ROE = **{fnum(f['payout_stable'],1,True)}** · Ke **{fnum(R['ke'],2,True)}**")
                kpi(st, "Valeur / action (FCFE)", fnum(f["per_share"], 2))
                st.dataframe(f["table"].style.format("{:,.1f}"), width="stretch")
            else:
                st.info("Résultat net négatif : méthode non applicable.")
        with b:
            st.markdown("#### Revenu résiduel (Ohlson / EVA)")
            if R["ri"]:
                kpi(st, "Valeur / action (RI)", fnum(R["ri"]["per_share"], 2), f"ROE {fnum(R['roe'],1,True)} vs Ke {fnum(R['ke'],1,True)}")
                st.dataframe(R["ri"]["table"].style.format("{:,.3f}"), width="stretch")
            st.markdown("#### Actif net réévalué")
            st.number_input("Ajustements de réévaluation (± plus-values latentes, unité des états)", step=1.0, key="anr_adj")
            kpi(st, "ANR / action", fnum(R["anr"], 2))
    with t3:
        st.markdown("#### Multiples de référence")
        c = st.columns(4)
        c[0].number_input("EV / EBITDA", step=0.5, key="m_EV_EBITDA")
        c[1].number_input("EV / CA", step=0.1, key="m_EV_Sales")
        c[2].number_input("PER", step=0.5, key="m_PER")
        c[3].number_input("P / B", step=0.1, key="m_P_B")
        with st.expander("🤖 Comparables par machine learning (plus proches voisins dans la base CSV)"):
            scr = STORE.load_screener()
            if scr.empty or len(ML.company_matrix(scr)) < 6:
                st.info("Base CSV insuffisante : consultez des sociétés dans 🏢 Grandes entreprises ou actualisez la base.")
            else:
                df_, y_ = R["df"], R["df"].columns[-1]
                gv = lambda k: float(df_.at[k, y_]) if pd.notna(df_.at[k, y_]) else np.nan
                to_m = UNIT_FACTOR[ss.unit]
                mcap_md = (R["price"] * ss.shares / 1000) if R["price"] else gv("equity") * to_m / 1000
                target = {"Croissance CA %": R["ratios"].at["Croissance du CA (%)", y_],
                          "Marge opé. %": R["ratios"].at["Marge opérationnelle (%)", y_],
                          "Marge nette %": R["ratios"].at["Marge nette (%)", y_], "ROE %": R["ratios"].at["ROE (%)", y_],
                          "Dette/CP %": gv("total_debt") / gv("equity") * 100 if gv("equity") else np.nan,
                          "Taille (log cap.)": np.log10(max(mcap_md, 0.01)) if mcap_md == mcap_md else np.nan}
                kk = st.slider("Nombre de comparables", 3, 12, 6, key="ml_k")
                peers = ML.nearest_peers(target, scr, kk)
                if peers.empty:
                    st.info("Pas assez de variables renseignées pour trouver des comparables.")
                else:
                    st.dataframe(peers.style.format({c: "{:,.2f}" for c in peers.columns if c not in ("Ticker", "Société")}, na_rep="—"),
                                 width="stretch", hide_index=True)
                    med = peers[[m for m in ML.MULTIPLES if m in peers]].median()
                    st.caption("Médianes : " + " · ".join(f"{k} {v:.1f}x" for k, v in med.items() if v == v))
                    if st.button("Appliquer les médianes des comparables ML"):
                        mp = {"PER": "m_PER", "EV/EBITDA": "m_EV_EBITDA", "EV/CA": "m_EV_Sales", "P/B": "m_P_B"}
                        ss._pending = {mp[k]: float(v) for k, v in med.items() if v == v and v > 0}
                        st.rerun()
        with st.expander("Ou calculer les multiples à partir de comparables (médiane)"):
            peers = st.data_editor(pd.DataFrame({"Comparable": ["", "", ""], "EV_EBITDA": [np.nan] * 3, "EV_Sales": [np.nan] * 3,
                                                 "PER": [np.nan] * 3, "P_B": [np.nan] * 3}), num_rows="dynamic", width="stretch", key="peers")
            if st.button("Appliquer la médiane des comparables"):
                med = peers[["EV_EBITDA", "EV_Sales", "PER", "P_B"]].apply(pd.to_numeric, errors="coerce").median()
                for k, v in med.items():
                    if v == v:
                        ss[f"m_{k}"] = float(v)
                st.rerun()
        m = R["mult"]
        st.dataframe(m.style.format({"Multiple": "{:.2f}", "Agrégat": "{:,.1f}", "Valeur d'entreprise": "{:,.1f}",
                                     "Valeur des fonds propres": "{:,.1f}", "Par action": "{:,.2f}"}, na_rep="—"), width="stretch")
        st.caption("Damodaran rappelle qu'un multiple est un DCF qui ne dit pas son nom : comparez toujours les fondamentaux "
                   "(croissance, risque, ROIC) de la société à ceux des comparables.")
    with t4:
        a, b = st.columns([1, 2])
        with a:
            st.markdown("**Risque de défaillance**")
            st.slider("Probabilité de défaillance", 0.0, 0.6, key="p_fail", step=0.01)
            st.slider("Valeur récupérée en cas de défaillance", 0.0, 1.0, key="distress", step=0.05)
            st.number_input("Actifs hors exploitation", step=1.0, key="non_op")
            st.number_input("Stock-options / dilution (valeur)", step=1.0, key="options")
            st.markdown("**Incertitude (écarts-types, pts)**")
            st.slider("Croissance", 0.0, 10.0, key="sd_g", step=0.5)
            st.slider("Marge", 0.0, 8.0, key="sd_m", step=0.5)
            st.slider("WACC", 0.0, 3.0, key="sd_w", step=0.25)
            st.caption("Astuce : 🤖 Machine learning › Risque de dégradation peut estimer cette probabilité à partir des données.")
        with b:
            mc = R["mc"]
            lo, hi = np.percentile(mc, [1, 99])
            mcc = mc[(mc > lo) & (mc < hi)]
            fig = go.Figure(go.Histogram(x=mcc, nbinsx=70, marker_color=NAVY, opacity=.85))
            if R["price"]:
                fig.add_vline(x=R["price"], line_color=RED, line_width=3, annotation_text="Cours")
            fig.add_vline(x=float(np.median(mc)), line_color=GOLD, line_width=3, annotation_text="Médiane")
            st.plotly_chart(plot_layout(fig, 380, "5 000 simulations de la valeur par action"), width="stretch")
            k = st.columns(4)
            p5, p50, p95 = np.percentile(mc, [5, 50, 95])
            kpi(k[0], "P5", fnum(p5, 2)); kpi(k[1], "Médiane", fnum(p50, 2)); kpi(k[2], "P95", fnum(p95, 2))
            kpi(k[3], "P(valeur > cours)", fnum(float((mc > R["price"]).mean()), 0, True) if R["price"] else "n.d.")
    with t5:
        st.markdown("#### Valeur par action — WACC (colonnes) × croissance perpétuelle (lignes)")
        st.dataframe(R["sens"].style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=None), width="stretch")
        inp = R["inp"]
        gm = V.sensitivity_growth_margin(inp, [inp.g_high + d for d in (-.04, -.02, 0, .02, .04)],
                                         [inp.target_margin + d for d in (-.04, -.02, 0, .02, .04)])
        st.markdown("#### Valeur par action — marge cible (colonnes) × croissance A1-A5 (lignes)")
        st.dataframe(gm.style.format("{:,.2f}").background_gradient(cmap="RdYlGn", axis=None), width="stretch")
        if R["implied"] is not None:
            st.info(f"**DCF inversé** : pour justifier le cours de {R['price']:.2f}, le marché anticipe une croissance de "
                    f"**{R['implied']*100:.1f} %** par an sur 5 ans (vs {ss.g_high*100:.1f} % dans notre scénario), toutes autres hypothèses égales.")


def page_verdict(R):
    ss._last_R = R
    lvl_col = {"ok": "linear-gradient(120deg,#13776c,#1B998B)", "bad": "linear-gradient(120deg,#8f2233,#D1495B)",
               "warn": "linear-gradient(120deg,#8a6d0f,#C9A227)", "info": "linear-gradient(120deg,#0B1F3A,#1f4677)"}
    up = R["upside"]
    st.markdown(f"""<div class='vs-verdict' style='background:{lvl_col[R['verdict_lvl']]}'>
      <div class='vs-kicker' style='color:#fff;opacity:.8'>Verdict · {R['name']}</div>
      <h2>{R['verdict']}</h2>
      <div style='display:flex;gap:3rem;flex-wrap:wrap;margin-top:.8rem'>
        <div><div style='opacity:.8;font-size:.8rem'>VALEUR INTRINSÈQUE / ACTION</div><div class='big'>{fnum(R['fair_value'],2)}</div></div>
        <div><div style='opacity:.8;font-size:.8rem'>COURS</div><div class='big'>{fnum(R['price'],2) if R['price'] else '—'}</div></div>
        <div><div style='opacity:.8;font-size:.8rem'>POTENTIEL</div><div class='big'>{f"{up*100:+.1f} %" if up is not None else '—'}</div></div>
      </div></div>""", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.4, 1])
    with a:
        s = R["summary"]
        fig = go.Figure()
        fig.add_bar(y=s["Méthode"], x=s["Haut"] - s["Bas"], base=s["Bas"], orientation="h", marker_color=GOLD, opacity=.55, name="Fourchette")
        fig.add_scatter(y=s["Méthode"], x=s["Par action"], mode="markers", marker=dict(color=NAVY, size=12), name="Valeur centrale")
        if R["price"]:
            fig.add_vline(x=R["price"], line_color=RED, line_width=3, annotation_text="Cours",
                          annotation_position="top left" if R["price"] < R["fair_value"] else "top right")
        fig.add_vline(x=R["fair_value"], line_color=TEAL, line_dash="dash", line_width=2, annotation_text="Valeur pondérée",
                      annotation_position="top right" if (R["price"] or 0) < R["fair_value"] else "top left")
        fig = plot_layout(fig, 360, "Football field")
        fig.update_yaxes(type="category"); fig.update_xaxes(type="linear", showgrid=True, gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Pondération des méthodes**")
        c1, c2 = st.columns(2)
        c1.slider("DCF FCFF", 0, 100, key="w_dcf"); c2.slider("FCFE", 0, 100, key="w_fcfe")
        c1.slider("Revenu résiduel", 0, 100, key="w_ri"); c2.slider("Multiples", 0, 100, key="w_mult")
        c1.slider("ANR", 0, 100, key="w_anr"); c2.slider("Bande de neutralité (± %)", 5, 40, key="band")
    st.dataframe(R["summary"].style.format({"Bas": "{:,.2f}", "Par action": "{:,.2f}", "Haut": "{:,.2f}"}), width="stretch", hide_index=True)
    st.markdown("#### Le récit derrière les chiffres")
    st.markdown(f"<div class='vs-quote'>{R['narrative']}</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("💾 Enregistrer cette analyse dans le journal CSV"):
        STORE.log_analysis({"societe": R["name"], "pays": R["country"], "secteur": R["industry"], "unite": R["unit"],
                            "cours": R["price"], "valeur_intrinseque": round(R["fair_value"], 4),
                            "potentiel_pct": round(R["upside"] * 100, 2) if R["upside"] is not None else None,
                            "verdict": R["verdict"], "wacc_pct": round(R["wacc"] * 100, 3), "ke_pct": round(R["ke"] * 100, 3),
                            "croissance_pct": round(ss.g_high * 100, 2), "g_stable_pct": round(ss.g_stable * 100, 2),
                            "marge_cible_pct": round(ss.target_margin * 100, 2), "dcf_par_action": round(R["dcf"]["per_share"], 4),
                            "cycle_de_vie": R["lifecycle"][0]})
        STORE.save_financials("PERSO:" + R["name"], R["name"], R["unit"], R["df"])
        st.toast("Analyse et états financiers enregistrés dans la base CSV.", icon="💾")
    st.markdown("#### Télécharger les livrables")
    c = st.columns(4)
    base = "".join(ch if ch.isalnum() else "_" for ch in R["name"])[:40]
    with st.spinner("Préparation des livrables…"):
        try:
            c[0].download_button("📄 Rapport PDF", RP.pdf(R), f"ValueScope_{base}.pdf", "application/pdf", width="stretch")
        except Exception as e:
            c[0].error(f"PDF : {e}")
        c[1].download_button("📝 Rapport Word", RP.word(R), f"ValueScope_{base}.docx", width="stretch")
        c[2].download_button("📊 Classeur Excel", RP.excel(R), f"ValueScope_{base}.xlsx", width="stretch")
        c[3].download_button("🖼 Graphiques PNG (zip)", RP.charts_zip(R), f"ValueScope_{base}_graphiques.zip", width="stretch")
    st.caption("Analyse automatisée fondée sur les données importées et des hypothèses modifiables — ne constitue pas un conseil en investissement.")


def page_method():
    st.markdown("## Méthodologie")
    st.markdown("""
**Philosophie.** ValueScope suit la démarche « Narrative & Numbers » d'Aswath Damodaran (NYU Stern) : chaque hypothèse chiffrée
doit correspondre à un élément du récit (taille de marché, avantage concurrentiel, besoin de réinvestissement, risque).

| Brique | Méthode | Points clés |
|---|---|---|
| Coût des fonds propres | CAPM étendu | rf de la devise + βL × ERP mature + λ × CRP (+ prime de taille) |
| Bêta | Bottom-up | βU sectoriel ré-endetté : βL = βU × (1 + (1 − t) × D/E) |
| Coût de la dette | Notation synthétique | Couverture EBIT/intérêts → notation → spread |
| DCF FCFF | 2 phases, 10 ans | Croissance → convergence de marge → réinvestissement = ΔCA / (ventes/capital) → IS effectif → marginal |
| Valeur terminale | Croissance stable | g ≤ rf, réinvestissement = g / ROIC stable, WACC convergé (β → 1) |
| Défaillance | Probabilité | Valeur = (1 − p) × DCF + p × valeur de liquidation |
| FCFE / DDM | 2 phases | g = ROE × rétention ; payout stable = 1 − g / ROE |
| Revenu résiduel | Ohlson / EVA | Valeur = FP + VA des (ROE − Ke) × FP, convergence du ROE |
| Multiples | Relative | EV/EBITDA, EV/CA, PER, P/B — secteur ou comparables |
| ANR | Patrimoniale | Capitaux propres ± réévaluations |
| Incertitude | Monte Carlo | 5 000 tirages sur croissance, marge, WACC, ventes/capital |
| Marché | DCF inversé | Croissance implicite contenue dans le cours |

**Import.** Libellés reconnus en français (PCG, liasse 2050-2053, IFRS) et en anglais (US GAAP 10-K, IFRS). Les agrégats manquants
(EBITDA, EBIT, dette totale, actif circulant…) sont reconstitués. Toujours vérifier l'écran d'édition après un import PDF ou image.

**Grandes entreprises.** États financiers annuels publiés et indicateurs de marché via Yahoo Finance (différé). La valorisation
express applique le même moteur avec les hypothèses par défaut (70 % DCF, 30 % multiples sectoriels) ; le bouton « Analyse complète »
charge la société dans les pages 2 à 6 pour affiner chaque hypothèse.

**Pays.** FMI — World Economic Outlook (dette brute et nette, soldes budgétaire et primaire, recettes, dépenses, croissance, inflation,
chômage, balance courante, PIB), projections incluses ; Banque mondiale — réserves, charge d'intérêts, dette extérieure.
Soutenabilité : d(t+1) = d(t) × (1 + r)/(1 + g) − solde primaire ; solde stabilisant = d × (r − g)/(1 + g).
Le score de risque est une grille transparente et indicative, pas une notation d'agence.

**Matières premières.** Contrats à terme continus (Yahoo Finance) ; lithium, uranium et terres rares via ETF spécialisés (proxies).
Indicateurs : performances, volatilité annualisée, replis, moyennes mobiles 50/200 jours, corrélations hebdomadaires sur 3 ans,
ratios or/argent et cuivre/or, écart Brent − WTI.

**Machine learning (scikit-learn).** Typologie des sociétés et des pays : K-means (nombre de groupes choisi par silhouette) et ACP ;
comparables par plus proches voisins sur croissance, marges, ROE, endettement et taille. Anomalies : Isolation Forest.
Prévision des matières premières : rendement à 4 semaines par Gradient Boosting, Random Forest ou Ridge, validation glissante
(TimeSeriesSplit, écart de 4 semaines) comparée à la marche aléatoire, intervalles par régression quantile. Risque de dégradation :
régression logistique ou Gradient Boosting sur le panel société-année (perte, couverture < 1,5x ou fonds propres négatifs l'année
suivante) ou sur un CSV étiqueté, AUC en validation croisée stratifiée. Chaque modèle affiche sa qualité hors échantillon.

**Base CSV.** Toute donnée récupérée en ligne est enregistrée dans <code>data/store/</code> ; le mode hors-ligne n'utilise que la base.
Le journal conserve chaque valorisation enregistrée (hypothèses, valeur, verdict).

**Limites.** Banques et assurances : préférer FCFE / revenu résiduel (la dette y est une matière première). Les paramètres de marché
pré-remplis sont indicatifs : mettez-les à jour depuis le site de Damodaran (onglet Data) avant toute décision.
""")


# ------------------------------------------------------------------ ROUTER
if ss.get("_flash"):
    st.toast(ss.pop("_flash"), icon="✅")
page = ss.page
if page == "Accueil":
    page_home()
elif page == "1 · Import des états":
    page_import()
elif page == "🏢 Grandes entreprises":
    UM.page_companies(load_df)
elif page == "🌍 Analyse pays":
    UM.page_countries()
elif page == "🛢 Matières premières":
    UM.page_commodities()
elif page == "🤖 Machine learning":
    UML.page_ml()
elif page == "💾 Base de données":
    UML.page_data()
elif page == "Méthodologie":
    page_method()
elif not has_data():
    st.markdown("## Aucune donnée")
    st.info("Commencez par importer des états financiers ou charger un exemple depuis l'accueil.")
    if st.button("→ Aller à l'import", type="primary"):
        ss._pending = {**ss.get("_pending", {}), "page": "1 · Import des états"}; st.rerun()
else:
    set_defaults()
    if ss.df.loc["revenue"].isna().all() or ss.df.loc["ebit"].isna().all():
        st.warning("Le chiffre d'affaires ou le résultat d'exploitation est manquant : complétez les données dans l'onglet Import.")
    else:
        ss.g_stable = min(ss.g_stable, ss.rf)
        try:
            R = compute()
        except Exception as e:
            st.error(f"Calcul impossible avec les données actuelles ({e}). Vérifiez les postes clés dans l'onglet Import.")
            st.stop()
        {"2 · Diagnostic": page_diag, "3 · Prévisions": page_forecast, "4 · Coût du capital": page_wacc,
         "5 · Valorisation": page_valuation, "6 · Verdict & rapports": page_verdict}[page](R)
