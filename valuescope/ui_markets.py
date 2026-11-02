"""Pages « marchés » : grandes entreprises cotées, analyse pays, matières premières."""
from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import companies as CO
from core import countries as CT
from core import commodities as CM
from core import analysis as A
from core import market as M
from core import valuation as V
from core.schema import FIELDS
from ui_kit import kpi, fnum, plot_layout, geo_layout, comment, hero, offline, NAVY, GOLD, TEAL, RED, GREY, PALETTE, GRID, DIVERGING

ss = st.session_state
THIS_YEAR = date.today().year


def _time_layout(fig, h=380, title=None):
    plot_layout(fig, h, title)
    fig.update_xaxes(type="date", showgrid=False)
    return fig


def _year_layout(fig, h=380, title=None):
    plot_layout(fig, h, title)
    fig.update_xaxes(type="linear", showgrid=False)
    return fig


# ====================================================================== DONNÉES : EN LIGNE → CSV, REPLI CSV
from core import store as STORE  # noqa: E402
from core import net as NET  # noqa: E402


def offline_mode() -> bool:
    return ss.get("data_mode", "").startswith("Hors-ligne")


def _save_once(key: str, fn, *args):
    """Écrit dans la base CSV une seule fois par jeu de données récupéré (évite les écritures à chaque interaction)."""
    done = ss.setdefault("_saved", set())
    if key in done:
        return
    try:
        fn(*args)
        done.add(key)
    except Exception as e:
        st.toast(f"Écriture CSV impossible : {e}", icon="⚠️")


def source_badge(src: str, when: str | None = None):
    if src == "live":
        st.markdown("<span class='vs-pill ok'>● Données en direct · enregistrées dans la base CSV</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='vs-pill warn'>● Base CSV locale{(' · ' + when) if when else ''} — mode hors-ligne ou source injoignable</span>",
                    unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _screener_live(tickers: tuple):
    df = CO.fetch_screener(list(tickers))
    if df.empty or df["Cours"].isna().all():        # ne pas mettre en cache un résultat vide
        raise RuntimeError("Yahoo Finance : aucun cours reçu. " + " | ".join(f"{k} : {v[1]}" for k, v in NET.LOG.items() if not v[0]))
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _company_live(ticker: str):
    return CO.fetch_company(ticker)


@st.cache_data(ttl=43200, show_spinner=False)
def _imf_live():
    """FMI d'abord ; chaque indicateur manquant est complété par son équivalent Banque mondiale."""
    out, wb_used, errors = CT.fetch_all_macro()
    if not out:
        raise RuntimeError(" | ".join(errors[:6]))
    out["_meta"] = pd.DataFrame({"wb_fallback": [", ".join(wb_used)], "errors": [" | ".join(errors[:6])]})
    return out


@st.cache_data(ttl=43200, show_spinner=False)
def _wb_live(iso: str):
    out = {}
    for code, (label, scale) in CT.WB_IND.items():
        try:
            s = CT.fetch_wb(code, iso)
            if not s.empty:
                out[label] = s * scale
        except Exception:
            pass
    if not out:
        raise RuntimeError("Banque mondiale injoignable")
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _prices_live():
    return CM.fetch_prices(list(CM.COMMODITIES), period="10y", interval="1d")


def get_screener(tickers: tuple):
    if not offline_mode():
        try:
            df = _screener_live(tickers)
            if not df.empty and df["Cours"].notna().any():
                _save_once(f"scr:{tickers}", STORE.save_screener, df)
                return df, "live", None
        except Exception as e:
            ss["_live_err"] = str(e)
    df = STORE.load_screener(list(tickers))
    when = df["Mis à jour"].max() if "Mis à jour" in df else None
    return df, "csv", when


def get_company(tk: str):
    if not offline_mode():
        try:
            c = _company_live(tk)
            if not c["df"].empty:
                name = c["info"].get("longName") or CO.ALL_TICKERS.get(tk, tk)
                _save_once(f"fin:{tk}", STORE.save_financials, tk, name, c["fin_currency"], c["df"])
                _save_once(f"inf:{tk}", STORE.save_screener, pd.DataFrame([CO.row_from_info(tk, c["info"])]))
                return c, "live", None
        except Exception as e:
            ss["_live_err"] = str(e)
    df, meta = STORE.load_financials(tk)
    if df.empty:
        raise RuntimeError(f"{tk} : source en ligne en échec ({ss.get('_live_err', 'mode hors-ligne')}) "
                           f"et société absente de la base CSV.")
    row = STORE.load_screener([tk])
    info = CO.info_from_row(row.iloc[0].to_dict()) if not row.empty else {"longName": meta.get("name")}
    price = info.get("currentPrice")
    shares = row.iloc[0].get("Actions (M)") if not row.empty else None
    if not shares or shares != shares:
        shares = float(df.loc["shares"].dropna().iloc[-1]) if "shares" in df.index and df.loc["shares"].notna().any() else None
    cur = info.get("currency")
    if cur in ("GBp", "GBX", "ZAc", "ILA") and price:
        price, cur = price / 100, {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}[cur]
    same = (cur == (info.get("financialCurrency") or meta.get("currency")))
    c = dict(info=info, df=df, hist=None, price=price, price_fin=price if same else None, currency=cur,
             fin_currency=info.get("financialCurrency") or meta.get("currency"), shares_m=shares, fx=1.0)
    return c, "csv", meta.get("updated")


def get_imf():
    if not offline_mode():
        try:
            d = dict(_imf_live())
            meta = d.pop("_meta", None)
            if meta is not None and meta["wb_fallback"].iloc[0]:
                st.info("Le FMI n'a pas répondu pour certaines séries : complétées par la Banque mondiale "
                        "(périmètre parfois différent, ex. dette de l'administration centrale) et sans projections.")
            _save_once("imf", STORE.save_imf, d)
            return d, "live", None
        except Exception as e:
            ss["_live_err"] = str(e)
    d = STORE.load_imf()
    if not d:
        raise RuntimeError(f"FMI et Banque mondiale injoignables ({ss.get('_live_err', 'mode hors-ligne')}) et base CSV vide.")
    return d, "csv", None


def get_wb(iso: str):
    if not offline_mode():
        try:
            d = _wb_live(iso)
            _save_once(f"wb:{iso}", STORE.save_wb, iso, d)
            return d
        except Exception:
            pass
    return STORE.load_wb(iso)


def get_prices():
    if not offline_mode():
        try:
            px_ = _prices_live()
            if not px_.empty:
                fred_cols = list(px_.attrs.get("fred", {}))      # unités différentes : non mélangées à l'historique CSV
                _save_once("px", STORE.save_prices, px_.drop(columns=fred_cols))
                return px_, "live", None
        except Exception as e:
            ss["_live_err"] = str(e)
    px_ = STORE.load_prices()
    if px_.empty:
        raise RuntimeError(f"Yahoo Finance injoignable ({ss.get('_live_err', 'mode hors-ligne')}) et base CSV vide.")
    return px_, "csv", str(px_.index.max().date())


# ====================================================================== ENTREPRISES
def quick_value(df: pd.DataFrame, country: str, industry: str, price: float | None, shares_m: float | None,
                currency: str | None) -> dict | None:
    """Valorisation express (mêmes règles que le moteur complet, hypothèses par défaut)."""
    try:
        d = A_complete(df)
        y = d.columns[-1]
        g = lambda k: float(d.at[k, y]) if pd.notna(d.at[k, y]) else np.nan
        cur, crp, tmarg = M.COUNTRIES.get(country, ("USD", 0.0, 25.0))
        cur = currency if currency in M.CURRENCIES else cur
        rf = M.CURRENCIES.get(cur, (4.0, ""))[0] / 100
        ind = M.INDUSTRY.loc[industry]
        t = tmarg / 100
        rev, ebit = g("revenue"), g("ebit")
        if not (rev > 0) or not np.isfinite(ebit) or not shares_m:
            return None
        cash, debt = np.nan_to_num(g("cash")), np.nan_to_num(g("total_debt"))
        mcap = price * shares_m if price else g("equity")
        de = debt / mcap if mcap and mcap > 0 else 0
        beta = M.lever_beta(ind.Beta_u, de, t)
        ke = rf + beta * M.MATURE_ERP / 100 + crp / 100
        cov = ebit / g("interest_expense") if g("interest_expense") and g("interest_expense") > 0 else float("inf")
        _, spread = M.synthetic_rating(cov, True)
        kd = rf + spread / 100
        wd = debt / (debt + mcap) if (debt + mcap) > 0 else 0
        wacc = ke * (1 - wd) + kd * (1 - t) * wd
        wacc_st = max((rf + M.MATURE_ERP / 100 + crp / 100) * (1 - wd) + kd * (1 - t) * wd, min(rf, .025) + .01)
        ic = A.invested_capital(d)[y]
        s2c = float(np.clip(rev / ic if ic and ic > 0 else ind.Sales_to_capital, 0.3, 5))
        hg = A.cagr(d.loc["revenue"])
        teff = A.effective_tax(d)[y]
        inp = V.DCFInputs(revenue0=rev, margin0=ebit / rev, g_high=float(np.clip(hg if hg == hg else .04, -.05, .30)),
                          g_stable=min(rf, .025), target_margin=max(ebit / rev, 0.01), convergence_year=5,
                          sales_to_capital=s2c, tax_eff=float(teff) if teff == teff else t, tax_marginal=t, wacc=wacc,
                          wacc_stable=wacc_st, roic_stable=wacc_st, cash=cash, debt=debt,
                          minority=np.nan_to_num(g("minority")), shares=shares_m, price=price)
        dcf = V.dcf_fcff(inp)
        mult = V.multiples(dict(ebitda=g("ebitda"), revenue=rev, net_income=g("net_income"), equity=g("equity")),
                           dict(EV_EBITDA=ind.EV_EBITDA, EV_Sales=ind.EV_Sales, PER=ind.PER, P_B=ind.P_B),
                           debt - cash, np.nan_to_num(g("minority")), shares_m)
        m = float(mult["Par action"].median()) if not mult.empty else np.nan
        fair = 0.7 * dcf["per_share"] + 0.3 * m if np.isfinite(m) else dcf["per_share"]
        verdict, lvl, up = V.verdict(fair, price)
        return dict(dcf=dcf["per_share"], mult=m, fair=fair, verdict=verdict, lvl=lvl, up=up, wacc=wacc,
                    g=inp.g_high, margin=inp.margin0, implied=V.reverse_dcf(inp, price) if price else None)
    except Exception:
        return None


def A_complete(df):
    from core.parser import complete
    return complete(df)


def page_companies(load_df):
    hero("Marchés · Entreprises", "Les grandes entreprises, chiffres à l'appui.",
         "États financiers publiés, multiples de marché et valorisation express pour les leaders français, américains, "
         "européens et mondiaux — puis analyse complète en un clic dans le moteur ValueScope.")
    c1, c2, c3 = st.columns([1.1, 2.4, 1])
    region = c1.selectbox("Univers", list(CO.UNIVERSE))
    names = CO.UNIVERSE[region]
    pick = c2.multiselect("Sociétés", list(names), default=list(names)[:10], format_func=lambda t: f"{names[t]} ({t})")
    extra = c3.text_input("Autre ticker Yahoo", placeholder="ex. AC.PA, UBER")
    tickers = tuple(pick + ([extra.strip().upper()] if extra.strip() else []))
    if not tickers:
        st.info("Sélectionnez au moins une société.")
        return
    try:
        with st.spinner("Récupération des données de marché…"):
            scr, src, when = get_screener(tickers)
    except Exception as e:
        offline(e, "ui_markets_1"); return
    if scr.empty or scr["Cours"].isna().all():
        offline("Aucune donnée : Yahoo Finance n'a renvoyé aucun cours et ces sociétés sont absentes de la base CSV. "
                + " | ".join(f"{k} : {v[1]}" for k, v in NET.LOG.items() if not v[0]), "ui_markets_2"); return
    source_badge(src, when)
    if src == "csv" and len(scr) < len(tickers):
        st.caption(f"{len(tickers) - len(scr)} société(s) absente(s) de la base CSV.")

    t1, t2 = st.tabs(["📋 Tableau de bord comparatif", "🔎 Fiche société & valorisation"])
    with t1:
        k = st.columns(4)
        kpi(k[0], "Sociétés", str(len(scr)), region)
        kpi(k[1], "PER médian", fnum(scr["PER"].median(), 1) + "x", f"prévisionnel {fnum(scr['PER prévisionnel'].median(),1)}x")
        kpi(k[2], "EV/EBITDA médian", fnum(scr["EV/EBITDA"].median(), 1) + "x")
        kpi(k[3], "Marge opé. médiane", fnum(scr["Marge opé. %"].median(), 1) + " %", f"ROE médian {fnum(scr['ROE %'].median(),1)} %")
        st.write("")
        num = list(scr.select_dtypes("number").columns)
        st.dataframe(scr.style.format({c: "{:,.1f}" for c in num}, na_rep="—")
                     .background_gradient(subset=["Perf. 52 sem. %"], cmap="RdYlGn", vmin=-40, vmax=40)
                     .background_gradient(subset=["ROE %"], cmap="Greens"),
                     width="stretch", hide_index=True, height=min(60 + 36 * len(scr), 640))
        a, b = st.columns(2)
        s2 = scr.dropna(subset=["Capitalisation (Md)"]).sort_values("Capitalisation (Md)")
        fig = go.Figure(go.Bar(x=s2["Capitalisation (Md)"], y=s2["Société"], orientation="h", marker_color=NAVY,
                               text=[f"{v:,.0f}" for v in s2["Capitalisation (Md)"]], textposition="outside"))
        fig.update_traces(cliponaxis=False)
        fig = plot_layout(fig, 420, "Capitalisation boursière (milliards, devise de cotation)")
        fig.update_layout(margin=dict(r=50))
        fig.update_xaxes(type="linear"); fig.update_yaxes(type="category")
        a.plotly_chart(fig, width="stretch")
        s3 = scr.dropna(subset=["Marge opé. %", "EV/EBITDA"])
        s3 = s3[(s3["EV/EBITDA"] > 0) & (s3["EV/EBITDA"] < 80)]
        if not s3.empty:
            fig = px.scatter(s3, x="Marge opé. %", y="EV/EBITDA", size=s3["Capitalisation (Md)"].fillna(1).clip(lower=1),
                             text="Société", color="Croissance CA %", color_continuous_scale="Tealgrn", size_max=45)
            fig.update_traces(textposition="top center", textfont_size=10)
            fig = plot_layout(fig, 420, "Qualité vs prix : marge opérationnelle et EV/EBITDA")
            fig.update_xaxes(type="linear", showgrid=True, gridcolor=GRID)
            b.plotly_chart(fig, width="stretch")
            b.caption("En haut à gauche : multiples élevés pour une rentabilité modeste (à questionner). "
                      "En bas à droite : rentabilité élevée pour un multiple modeste (à creuser).")
        st.caption("Source : Yahoo Finance (données publiées et consensus), différé. Devise de cotation propre à chaque marché.")

    with t2:
        tk = st.selectbox("Société à analyser", list(scr["Ticker"]), format_func=lambda t: f"{CO.ALL_TICKERS.get(t, t)} ({t})")
        try:
            with st.spinner("Chargement des états financiers…"):
                c, csrc, cwhen = get_company(tk)
        except Exception as e:
            offline(e, "ui_markets_3"); return
        if csrc == "csv":
            source_badge(csrc, cwhen)
        elif c["info"].get("_fin_source") or c["info"].get("_source"):
            st.markdown("<span class='vs-pill warn'>● Yahoo Finance limité : cours via l'API chart"
                        + (" · comptes officiels SEC EDGAR" if c["info"].get("_fin_source") else "") + "</span>", unsafe_allow_html=True)
        info, df = c["info"], c["df"]
        name = info.get("longName") or CO.ALL_TICKERS.get(tk, tk)
        country = CO.COUNTRY_MAP.get(info.get("country"), "Autre (saisie libre)")
        industry = CO.map_sector(info.get("industry"), info.get("sector"))
        fin = CO.is_financial(info.get("industry"), info.get("sector"))
        st.markdown(f"### {name}")
        st.markdown(f"<span class='vs-pill info'>{info.get('sector') or '—'} · {info.get('industry') or '—'}</span> &nbsp;"
                    f"<span class='vs-pill info'>{info.get('country') or '—'}</span> &nbsp;"
                    f"<span class='vs-pill info'>Comptes en {c['fin_currency']} · cours en {c['currency']}</span>",
                    unsafe_allow_html=True)
        st.write("")
        k = st.columns(5)
        kpi(k[0], "Cours", fnum(c["price"], 2), c["currency"] or "")
        kpi(k[1], "Capitalisation", fnum((info.get("marketCap") or np.nan) / 1e9, 1) + " Md")
        kpi(k[2], "PER", fnum(info.get("trailingPE"), 1) + "x", f"prévisionnel {fnum(info.get('forwardPE'),1)}x")
        kpi(k[3], "EV / EBITDA", fnum(info.get("enterpriseToEbitda"), 1) + "x")
        kpi(k[4], "Objectif analystes", fnum(info.get("targetMeanPrice"), 2), f"{info.get('numberOfAnalystOpinions') or '—'} analystes · {info.get('recommendationKey') or ''}")
        st.write("")
        if df.empty or "revenue" not in df.index:
            st.warning("États financiers indisponibles pour ce titre sur Yahoo Finance.")
            return
        full = A_complete(df)
        r = A.ratios(full)
        qv = None if fin else quick_value(df, country, industry, c["price_fin"], c["shares_m"], c["fin_currency"])
        a, b = st.columns([1.35, 1])
        with a:
            h = c["hist"]
            if h is not None and not h.empty:
                hh = h["Close"]
                fig = go.Figure(go.Scatter(x=hh.index, y=hh, line=dict(color=NAVY, width=2.2), fill="tozeroy",
                                           fillcolor="rgba(61,111,182,.12)", name="Cours"))
                fig.add_scatter(x=hh.index, y=hh.rolling(40).mean(), line=dict(color=GOLD, width=1.6), name="Moy. 40 sem.")
                st.plotly_chart(_time_layout(fig, 340, "Cours sur 5 ans"), width="stretch")
        with b:
            if fin:
                st.info("Établissement financier : le DCF « FCFF » n'est pas pertinent (la dette est une matière première). "
                        "Utilisez le revenu résiduel et les multiples P/B, PER dans l'analyse complète.")
            elif qv:
                col = {"ok": TEAL, "bad": RED, "warn": GOLD, "info": NAVY}[qv["lvl"]]
                st.markdown(f"""<div class='vs-verdict' style='background:{col};padding:1.2rem 1.4rem'>
                <div class='vs-kicker' style='color:#fff;opacity:.85'>Valorisation express</div>
                <h2 style='font-size:1.5rem'>{qv['verdict']}</h2>
                <div style='display:flex;gap:2rem;margin-top:.5rem;flex-wrap:wrap'>
                <div><div style='opacity:.8;font-size:.72rem'>VALEUR / ACTION</div><div style="font-family:'IBM Plex Mono';font-size:1.6rem">{fnum(qv['fair'],2)}</div></div>
                <div><div style='opacity:.8;font-size:.72rem'>POTENTIEL</div><div style="font-family:'IBM Plex Mono';font-size:1.6rem">{f"{qv['up']*100:+.0f} %" if qv['up'] is not None else '—'}</div></div>
                </div></div>""", unsafe_allow_html=True)
                st.caption(f"DCF {fnum(qv['dcf'],2)} (70 %) · multiples sectoriels {fnum(qv['mult'],2)} (30 %) · "
                           f"WACC {fnum(qv['wacc'],1,True)} · croissance {fnum(qv['g'],1,True)} · en {c['fin_currency']}"
                           + (f" · le cours intègre {qv['implied']*100:.1f} %/an de croissance" if qv.get("implied") is not None else ""))
            else:
                st.warning("Valorisation express impossible (données incomplètes).")
            if st.button("🔬 Analyse et valorisation complètes", type="primary", width="stretch"):
                load_df(df, name, dict(country=country, industry=industry, unit="Millions", price=float(c["price_fin"] or 0),
                                       shares=float(c["shares_m"] or 10), shares_user=True, listed=True,
                                       cur_override=c["fin_currency"] if c["fin_currency"] in M.CURRENCIES else None,
                                       page="2 · Diagnostic"))
                st.rerun()
            st.caption(f"Pays de référence : {country} · Secteur Damodaran : {industry} (modifiables ensuite).")
        x1, x2, x3 = st.tabs(["Comptes publiés", "Ratios clés", "Lecture de l'analyste"])
        with x1:
            yrs = [str(y) for y in full.columns]
            fig = go.Figure()
            fig.add_bar(x=yrs, y=full.loc["revenue"], name="Chiffre d'affaires", marker_color=NAVY)
            fig.add_bar(x=yrs, y=full.loc["ebit"], name="Résultat opérationnel", marker_color=GOLD)
            fig.add_bar(x=yrs, y=full.loc["net_income"], name="Résultat net", marker_color=TEAL)
            fig.add_scatter(x=yrs, y=full.loc["cfo"] - full.loc["capex"], name="Free cash-flow", mode="lines+markers",
                            line=dict(color=RED, width=3))
            st.plotly_chart(plot_layout(fig, 360, f"Activité et résultats (millions {c['fin_currency']})"), width="stretch")
            show = full.dropna(how="all").copy()
            show.index = [FIELDS[i][0] for i in show.index]
            show.columns = yrs
            st.dataframe(show.style.format("{:,.0f}", na_rep="—"), width="stretch", height=520)
        with x2:
            keep = ["Croissance du CA (%)", "Marge brute (%)", "Marge opérationnelle (%)", "Marge nette (%)", "ROE (%)",
                    "ROIC après impôt (%)", "Dette nette / EBITDA (x)", "Couverture des intérêts EBIT/Int (x)",
                    "Liquidité générale (x)", "Conversion cash CFO/Résultat net (x)", "Capex / CA (%)", "Taux de distribution (%)"]
            rr = r.loc[[k for k in keep if k in r.index]].copy()
            rr.columns = [str(y) for y in rr.columns]
            st.dataframe(rr.style.format("{:,.2f}", na_rep="—"), width="stretch")
        with x3:
            lc = A.life_cycle(full)
            st.markdown(f"<span class='vs-pill info'>Cycle de vie : {lc[0]}</span>", unsafe_allow_html=True)
            st.write("")
            for lvl, txt in A.commentary(full, r, qv["wacc"] if qv else None):
                comment(lvl, txt)
            az = A.altman_z(full, (info.get("marketCap") or 0) / 1e6 / (c["fx"] or 1), True)
            comment("info", f"{az['model']} : {fnum(az['score'],2)} — zone {az['zone']} · Piotroski {A.piotroski(full)['score']}/9")


# ====================================================================== PAYS
def page_countries():
    hero("Macro · Souverains", "La santé financière des États.",
         "Dette publique, déficits, charge d'intérêts, croissance, inflation, balance courante et réserves — historiques "
         "et projections du FMI (World Economic Outlook), complétés par la Banque mondiale.")
    try:
        with st.spinner("Chargement des données FMI…"):
            imf, isrc, _ = get_imf()
    except Exception as e:
        offline(e, "ui_markets_4"); return
    source_badge(isrc)
    labels = {code: lab for code, (lab, _) in CT.IMF_IND.items() if code in imf}
    names = {iso: v[0] for iso, v in CT.COUNTRIES.items()}

    t1, t2, t3 = st.tabs(["🗺 Vue mondiale", "🔎 Fiche pays", "⚖ Comparaison"])
    with t1:
        c1, c2 = st.columns([2, 1])
        code = c1.selectbox("Indicateur", list(labels), format_func=lambda c: labels[c])
        yrs = sorted(imf[code].index)
        year = c2.select_slider("Année", options=yrs, value=min(THIS_YEAR, max(yrs)))
        row = imf[code].loc[year].dropna() if year in imf[code].index else pd.Series(dtype=float)
        dfm = pd.DataFrame({"iso": row.index, "valeur": row.values, "pays": [names.get(i, i) for i in row.index]})
        rev = code in ("GGXWDG_NGDP", "GGXWDN_NGDP", "PCPIPCH", "LUR", "GGX_NGDP")
        fig = px.choropleth(dfm, locations="iso", color="valeur", hover_name="pays",
                            color_continuous_scale="RdYlGn_r" if rev else "RdYlGn", projection="natural earth")
        geo_layout(fig, 460)
        fig.update_layout(coloraxis_colorbar=dict(title=""))
        st.plotly_chart(fig, width="stretch")
        if year > THIS_YEAR - 1:
            st.caption(f"{year} : estimation / projection du FMI.")
        d2 = dfm.sort_values("valeur")
        fig = go.Figure(go.Bar(x=d2["pays"], y=d2["valeur"], marker_color=[RED if (v > 90 if rev and code.startswith("GGXWD") else False) else NAVY for v in d2["valeur"]]))
        if code == "GGXWDG_NGDP":
            fig.add_hline(y=60, line_dash="dash", line_color=GOLD, annotation_text="Critère de Maastricht 60 %")
        if code == "GGXCNL_NGDP":
            fig.add_hline(y=-3, line_dash="dash", line_color=GOLD, annotation_text="Critère de Maastricht −3 %")
        st.plotly_chart(plot_layout(fig, 380, f"Classement — {labels[code]} ({year})"), width="stretch")

    with t2:
        c1, c2 = st.columns([1, 2])
        iso = c1.selectbox("Pays", list(CT.COUNTRIES), format_func=lambda i: names[i], index=0)
        prof = CT.profile(iso, imf)
        if prof.empty:
            st.warning("Pas de données FMI pour ce pays."); return
        wb = get_wb(iso)
        last_obs = THIS_YEAR - 1
        get = lambda lab, yr=THIS_YEAR: CT.latest(prof.loc[lab], yr)[0] if lab in prof.index else np.nan
        m = dict(debt=get("Dette publique brute (% PIB)"), balance=get("Solde budgétaire (% PIB)"),
                 growth=get("Croissance du PIB réel (%)"), inflation=get("Inflation (%)"), unemp=get("Chômage (%)"),
                 ca=get("Balance courante (% PIB)"))
        ir = wb.get("Charge d'intérêts (% des recettes)")
        m["interest_rev"] = CT.latest(ir, THIS_YEAR)[0] if ir is not None else np.nan
        score, lab, notes = CT.risk_score(m)
        k = st.columns(6)
        kpi(k[0], "Dette / PIB", fnum(m["debt"], 1) + " %", f"{THIS_YEAR} (FMI)")
        kpi(k[1], "Solde public", fnum(m["balance"], 1) + " %", "du PIB")
        kpi(k[2], "Croissance réelle", fnum(m["growth"], 1) + " %")
        kpi(k[3], "Inflation", fnum(m["inflation"], 1) + " %")
        kpi(k[4], "Chômage", fnum(m["unemp"], 1) + " %")
        kpi(k[5], "PIB", fnum(get("PIB (milliards USD)"), 0) + " Md$", f"{fnum(get('PIB par habitant (USD)'),0)} $/hab.")
        st.write("")
        a, b = st.columns([1, 1.6])
        with a:
            col = {"Faible": "ok", "Modéré": "warn", "Élevé": "bad", "Très élevé": "bad"}.get(lab, "info")
            st.markdown(f"#### Risque macro-financier : <span class='vs-pill {col}'>{lab} · {fnum(score,0)}/100</span>",
                        unsafe_allow_html=True)
            for lvl, txt in notes:
                comment(lvl, txt)
            crp = M.COUNTRIES.get(names[iso])
            if crp:
                comment("info", f"Prime de risque pays utilisée dans ValueScope (Damodaran, indicatif) : {crp[1]:.2f} %.")
            st.caption("Score indicatif et transparent (dette, déficit, charge d'intérêts, inflation, extérieur, croissance, emploi) — "
                       "ce n'est pas une notation d'agence.")
        with b:
            fig = go.Figure()
            if "Dette publique brute (% PIB)" in prof.index:
                s = prof.loc["Dette publique brute (% PIB)"].dropna()
                hist, proj = s[s.index < THIS_YEAR], s[s.index >= THIS_YEAR - 1]
                fig.add_scatter(x=hist.index, y=hist, name="Dette brute", line=dict(color=NAVY, width=3))
                fig.add_scatter(x=proj.index, y=proj, name="Projection FMI", line=dict(color=NAVY, width=3, dash="dot"))
            if "Dette publique nette (% PIB)" in prof.index:
                s = prof.loc["Dette publique nette (% PIB)"].dropna()
                fig.add_scatter(x=s.index, y=s, name="Dette nette", line=dict(color=GREY, width=2))
            fig.add_hline(y=60, line_dash="dash", line_color=GOLD, annotation_text="60 %")
            st.plotly_chart(_year_layout(fig, 360, "Dette publique (% du PIB)"), width="stretch")
        x1, x2, x3, x4 = st.tabs(["Finances publiques", "Soutenabilité de la dette", "Croissance & prix", "Extérieur & données"])
        with x1:
            fig = go.Figure()
            for lab_, colr in [("Solde budgétaire (% PIB)", RED), ("Solde primaire (% PIB)", GOLD)]:
                if lab_ in prof.index:
                    s = prof.loc[lab_].dropna(); s = s[s.index >= 2000]
                    fig.add_bar(x=s.index, y=s, name=lab_, marker_color=colr)
            fig.add_hline(y=-3, line_dash="dash", line_color=GREY, annotation_text="−3 %")
            st.plotly_chart(_year_layout(fig, 340, "Soldes publics (% du PIB)"), width="stretch")
            fig = go.Figure()
            for lab_, colr in [("Recettes publiques (% PIB)", TEAL), ("Dépenses publiques (% PIB)", RED)]:
                if lab_ in prof.index:
                    s = prof.loc[lab_].dropna(); s = s[s.index >= 2000]
                    fig.add_scatter(x=s.index, y=s, name=lab_, line=dict(color=colr, width=3))
            st.plotly_chart(_year_layout(fig, 320, "Recettes et dépenses publiques (% du PIB)"), width="stretch")
        with x2:
            st.markdown("Équation de dynamique : **d(t+1) = d(t) × (1 + r) / (1 + g) − solde primaire**, avec r le taux "
                        "d'intérêt apparent et g la croissance nominale. Si r > g, un excédent primaire est nécessaire pour stabiliser la dette.")
            gn_def = (np.nanmean([get("Croissance du PIB réel (%)", y) for y in range(THIS_YEAR, THIS_YEAR + 5)]) +
                      np.nanmean([get("Inflation (%)", y) for y in range(THIS_YEAR, THIS_YEAR + 5)]))
            gn_def = float(gn_def) if np.isfinite(gn_def) else 3.0
            cur = M.COUNTRIES.get(names[iso], ("USD", 0, 0))[0]
            r_def = M.CURRENCIES.get(cur, (4.0,))[0]
            c1, c2, c3 = st.columns(3)
            r_ = c1.slider("Taux d'intérêt apparent r (%)", 0.0, 20.0, float(round(r_def, 1)), 0.1)
            g_ = c2.slider("Croissance nominale g (%)", -5.0, 25.0, float(round(gn_def, 1)), 0.1)
            sp_now = get("Solde primaire (% PIB)")
            sp_ = c3.slider("Solde primaire supposé (% PIB)", -10.0, 8.0, float(round(sp_now if np.isfinite(sp_now) else -1, 1)), 0.1)
            d0 = m["debt"]
            if np.isfinite(d0):
                star = CT.stabilizing_primary(d0, r_ / 100, g_ / 100)
                k = st.columns(3)
                kpi(k[0], "Écart r − g", f"{r_-g_:+.1f} pts", "défavorable si positif" if r_ > g_ else "favorable")
                kpi(k[1], "Solde primaire stabilisant", fnum(star, 2) + " %", "du PIB")
                kpi(k[2], "Effort requis", f"{star - sp_:+.2f} pts", "vs solde supposé")
                path = CT.debt_dynamics(d0, sp_, r_ / 100, g_ / 100, 10)
                fig = go.Figure()
                fig.add_scatter(x=[THIS_YEAR + i for i in path.index], y=path, name="Scénario", line=dict(color=NAVY, width=3))
                fig.add_scatter(x=[THIS_YEAR + i for i in path.index],
                                y=CT.debt_dynamics(d0, star, r_ / 100, g_ / 100, 10), name="Solde stabilisant",
                                line=dict(color=TEAL, dash="dot"))
                fig.add_scatter(x=[THIS_YEAR + i for i in path.index],
                                y=CT.debt_dynamics(d0, sp_, (r_ + 2) / 100, g_ / 100, 10), name="Choc de taux +2 pts",
                                line=dict(color=RED, dash="dash"))
                st.plotly_chart(_year_layout(fig, 360, "Trajectoire de la dette sur 10 ans (% du PIB)"), width="stretch")
        with x3:
            fig = go.Figure()
            xmax = THIS_YEAR + 5
            for lab_, colr in [("Croissance du PIB réel (%)", NAVY), ("Inflation (%)", RED), ("Chômage (%)", GOLD)]:
                if lab_ in prof.index:
                    s = prof.loc[lab_].dropna(); s = s[s.index >= 2000]
                    fig.add_scatter(x=s.index, y=s, name=lab_, mode="lines+markers", line=dict(color=colr, width=2.5))
                    xmax = max(xmax, int(s.index.max()) if len(s) else xmax)
            fig.add_vrect(x0=THIS_YEAR - .5, x1=xmax + .5, fillcolor=GOLD, opacity=.07,
                          line_width=0, annotation_text="Projections", annotation_position="top left")
            st.plotly_chart(_year_layout(fig, 380, "Croissance, inflation et chômage (%)"), width="stretch")
        with x4:
            fig = go.Figure()
            if "Balance courante (% PIB)" in prof.index:
                s = prof.loc["Balance courante (% PIB)"].dropna(); s = s[s.index >= 2000]
                fig.add_bar(x=s.index, y=s, marker_color=[TEAL if v >= 0 else RED for v in s], name="Balance courante")
            st.plotly_chart(_year_layout(fig, 300, "Balance courante (% du PIB)"), width="stretch")
            if wb:
                cols = st.columns(len(wb))
                for col_, (lab_, s) in zip(cols, wb.items()):
                    v, yr = CT.latest(s, THIS_YEAR)
                    kpi(col_, lab_, fnum(v, 1), f"Banque mondiale · {yr or '—'}")
                res = wb.get("Réserves de change totales (Md USD)")
                if res is not None and len(res):
                    fig = go.Figure(go.Scatter(x=res.index, y=res, line=dict(color=GOLD, width=3), fill="tozeroy"))
                    st.plotly_chart(_year_layout(fig, 280, "Réserves de change (Md USD, or inclus)"), width="stretch")
            tbl = prof.loc[:, [c for c in prof.columns if 2015 <= c <= THIS_YEAR + 5]]
            tbl.columns = [f"{c}{'p' if c >= THIS_YEAR else ''}" for c in tbl.columns]
            st.dataframe(tbl.style.format("{:,.1f}", na_rep="—"), width="stretch")
            st.caption("« p » = projection FMI. Sources : FMI World Economic Outlook (DataMapper), Banque mondiale (WDI).")

    with t3:
        sel = st.multiselect("Pays à comparer", list(CT.COUNTRIES), default=["FRA", "DEU", "ITA", "USA", "GBR", "JPN"],
                             format_func=lambda i: names[i])
        code = st.selectbox("Indicateur ", list(labels), format_func=lambda c: labels[c], key="cmp_ind")
        if sel:
            d = imf[code][[i for i in sel if i in imf[code].columns]]
            d = d[d.index >= 2000]
            fig = go.Figure()
            for j, i in enumerate(d.columns):
                fig.add_scatter(x=d.index, y=d[i], name=names[i], line=dict(color=PALETTE[j % len(PALETTE)], width=2.6))
            fig.add_vline(x=THIS_YEAR - .5, line_dash="dot", line_color=GREY, annotation_text="projections →")
            st.plotly_chart(_year_layout(fig, 420, labels[code]), width="stretch")
            snap = pd.DataFrame({labels[c]: [CT.latest(imf[c][i], THIS_YEAR)[0] if i in imf[c].columns else np.nan for i in sel]
                                 for c in labels}, index=[names[i] for i in sel])
            st.dataframe(snap.style.format("{:,.1f}", na_rep="—").background_gradient(axis=0, cmap="RdYlGn_r",
                         subset=[c for c in ["Dette publique brute (% PIB)", "Inflation (%)", "Chômage (%)"] if c in snap.columns]),
                         width="stretch")
            st.caption(f"Valeurs {THIS_YEAR} (estimations FMI).")


# ====================================================================== MATIÈRES PREMIÈRES
def page_commodities():
    hero("Marchés · Matières premières", "Ce qui fait tourner le monde.",
         "Énergie, métaux précieux et industriels, métaux de la transition, produits agricoles : prix, tendances, "
         "volatilité, corrélations et facteurs fondamentaux.")
    try:
        with st.spinner("Chargement des cours…"):
            px_, psrc, pwhen = get_prices()
    except Exception as e:
        offline(e, "ui_markets_5"); return
    source_badge(psrc, pwhen)
    if px_.empty:
        offline("Aucune donnée renvoyée par Yahoo Finance. " + " | ".join(f"{k} : {v[1]}" for k, v in NET.LOG.items() if not v[0]), "ui_markets_6"); return
    perf = CM.perf_table(px_)
    if px_.attrs.get("fred"):
        st.markdown("<div class='vs-comment warn'>Yahoo Finance n'a pas répondu pour certaines matières premières : "
                    f"<b>{len(px_.attrs['fred'])}</b> série(s) proviennent de la FRED (prix spot ou prix mensuels du FMI ; "
                    "unités indiquées dans le tableau). Or, argent, platine, palladium, lithium et terres rares n'y ont pas d'équivalent.</div>",
                    unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📊 Panorama", "🔎 Fiche matière première", "🔗 Comparaisons & corrélations"])
    with t1:
        cats = st.multiselect("Catégories", CM.CATEGORIES, default=CM.CATEGORIES[:-1])
        p = perf[perf["Catégorie"].isin(cats)]
        k = st.columns(4)
        for col_, tk in zip(k, ["BZ=F", "GC=F", "HG=F", "ZW=F"]):
            row = perf[perf["Ticker"] == tk]
            if not row.empty:
                rr = row.iloc[0]
                kpi(col_, rr["Matière première"], fnum(rr["Dernier"], 2), f"{rr['Unité']} · 1 an {fnum(rr['1 an %'],1)} %")
        st.write("")
        pcts = [c for c in p.columns if c.endswith("%")]
        st.dataframe(p.drop(columns=["Ticker"]).style.format({**{c: "{:+.1f}" for c in pcts}, "Volatilité 1 an %": "{:.1f}",
                                                               "Dernier": "{:,.2f}", "Plus haut 52s": "{:,.2f}", "Plus bas 52s": "{:,.2f}"}, na_rep="—")
                     .background_gradient(subset=["1 mois %", "YTD %", "1 an %"], cmap="RdYlGn", vmin=-30, vmax=30),
                     width="stretch", hide_index=True, height=min(60 + 36 * len(p), 820))
        p2 = p.sort_values("1 an %")
        fig = go.Figure(go.Bar(x=p2["1 an %"], y=p2["Matière première"], orientation="h",
                               marker_color=[TEAL if v >= 0 else RED for v in p2["1 an %"]]))
        fig = plot_layout(fig, 30 * len(p2) + 80, "Performance sur 1 an (%)")
        fig.update_xaxes(type="linear", showgrid=True, gridcolor=GRID); fig.update_yaxes(type="category")
        st.plotly_chart(fig, width="stretch")
        st.caption("Sources : Yahoo Finance — contrats à terme continus (front month) ; lithium, uranium et terres rares via ETF "
                   "spécialisés (proxies, pas des prix physiques). Cours en différé.")
    with t2:
        avail = [t for t in CM.COMMODITIES if t in px_.columns and px_[t].notna().sum() > 50]
        tk = st.selectbox("Matière première", avail, format_func=lambda t: CM.COMMODITIES[t][0])
        nm, cat, unit, uses, prod, drivers = CM.COMMODITIES[tk]
        unit = px_.attrs.get("fred", {}).get(tk, unit)
        hor = st.radio("Horizon", ["1 an", "3 ans", "5 ans", "10 ans"], index=2, horizontal=True)
        n = {"1 an": 365, "3 ans": 365 * 3, "5 ans": 365 * 5, "10 ans": 3660}[hor]
        s = px_[tk].dropna()
        s = s[s > 0]
        view = s[s.index >= s.index[-1] - pd.Timedelta(days=n)]
        row = perf[perf["Ticker"] == tk].iloc[0]
        k = st.columns(5)
        kpi(k[0], "Dernier", fnum(row["Dernier"], 2), unit)
        kpi(k[1], "1 mois", f"{row['1 mois %']:+.1f} %"); kpi(k[2], "1 an", f"{row['1 an %']:+.1f} %")
        kpi(k[3], "Volatilité 1 an", fnum(row["Volatilité 1 an %"], 1) + " %")
        kpi(k[4], "Écart au plus haut", f"{row['Écart au plus haut %']:+.1f} %", "sur 10 ans")
        a, b = st.columns([1.7, 1])
        with a:
            fig = go.Figure()
            fig.add_scatter(x=view.index, y=view, name=nm, line=dict(color=NAVY, width=2))
            fig.add_scatter(x=view.index, y=s.rolling(50).mean().reindex(view.index), name="MM 50 j", line=dict(color=GOLD, width=1.5))
            fig.add_scatter(x=view.index, y=s.rolling(200).mean().reindex(view.index), name="MM 200 j", line=dict(color=RED, width=1.5))
            st.plotly_chart(_time_layout(fig, 400, f"{nm} — {unit}"), width="stretch")
            dd = (view / view.cummax() - 1) * 100
            fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1), fillcolor="rgba(209,73,91,.18)"))
            st.plotly_chart(_time_layout(fig, 220, "Repli depuis le plus haut (%)"), width="stretch")
        with b:
            st.markdown(f"<div class='vs-card'><div class='vs-kpi-l'>{cat}</div><h3 style='margin:.2rem 0 .6rem'>{nm}</h3>"
                        f"<b>Usages</b><br><span class='vs-muted'>{uses}</span><br><br>"
                        f"<b>Principaux producteurs</b><br><span class='vs-muted'>{prod}</span><br><br>"
                        f"<b>Ce qui fait le prix</b><br><span class='vs-muted'>{drivers}</span></div>", unsafe_allow_html=True)
            st.write("")
            for lvl, txt in CM.signals(s):
                comment(lvl, txt)
            yr = s.groupby(s.index.year).last().pct_change().dropna() * 100
            fig = go.Figure(go.Bar(x=[str(y) for y in yr.index], y=yr, marker_color=[TEAL if v >= 0 else RED for v in yr]))
            st.plotly_chart(plot_layout(fig, 260, "Performance annuelle (%)"), width="stretch")
    with t3:
        avail = [t for t in CM.COMMODITIES if t in px_.columns]
        sel = st.multiselect("Comparer (base 100)", avail, default=[t for t in ["BZ=F", "GC=F", "HG=F", "ZW=F", "CC=F"] if t in avail],
                             format_func=lambda t: CM.COMMODITIES[t][0])
        hor = st.radio("Période", ["1 an", "3 ans", "5 ans", "10 ans"], index=2, horizontal=True, key="cmp_h")
        n = {"1 an": 365, "3 ans": 365 * 3, "5 ans": 365 * 5, "10 ans": 3660}[hor]
        if sel:
            d = px_[sel].ffill()
            d = d[d.index >= d.index[-1] - pd.Timedelta(days=n)].dropna(how="all")
            base = d.bfill().iloc[0]
            norm = d / base * 100
            fig = go.Figure()
            for j, t in enumerate(sel):
                fig.add_scatter(x=norm.index, y=norm[t], name=CM.COMMODITIES[t][0], line=dict(color=PALETTE[j % len(PALETTE)], width=2))
            fig.add_hline(y=100, line_color=GREY, line_dash="dot")
            st.plotly_chart(_time_layout(fig, 420, "Évolution comparée (base 100)"), width="stretch")
        with st.container():
            w = px_.resample("W").last()
            w = w[w.index >= w.index[-1] - pd.Timedelta(days=365 * 3)]
            cor = np.log(w).diff().corr()
            lab = [CM.COMMODITIES[t][0].split(" (")[0] for t in cor.columns]
            fig = go.Figure(go.Heatmap(z=cor.values, x=lab, y=lab, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                                       text=np.round(cor.values, 2), texttemplate="%{text}", textfont=dict(size=9, color="#111")))
            fig = plot_layout(fig, 760, "Corrélations des rendements hebdomadaires (3 ans)")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")
        with st.container():
            def ratio(a_, b_, f=1.0):
                if a_ in px_.columns and b_ in px_.columns:
                    return (px_[a_] / px_[b_] * f).dropna()
                return pd.Series(dtype=float)
            gs = ratio("GC=F", "SI=F")
            if len(gs):
                fig = go.Figure(go.Scatter(x=gs.index, y=gs, line=dict(color=GOLD, width=1.8)))
                fig.add_hline(y=gs.mean(), line_dash="dot", line_color=GREY, annotation_text="moyenne 10 ans")
                st.plotly_chart(_time_layout(fig, 280, "Ratio or / argent (onces d'argent pour une once d'or)"), width="stretch")
            cg = ratio("HG=F", "GC=F", 1000)
            if len(cg):
                fig = go.Figure(go.Scatter(x=cg.index, y=cg, line=dict(color=TEAL, width=1.8)))
                st.plotly_chart(_time_layout(fig, 280, "Ratio cuivre / or (×1000) — baromètre cycle vs refuge"), width="stretch")
            sp = (px_["BZ=F"] - px_["CL=F"]).dropna() if {"BZ=F", "CL=F"} <= set(px_.columns) else pd.Series(dtype=float)
            if len(sp):
                fig = go.Figure(go.Scatter(x=sp.index, y=sp, line=dict(color=NAVY, width=1.5)))
                st.plotly_chart(_time_layout(fig, 280, "Écart Brent − WTI (USD / baril)"), width="stretch")
            st.caption("Ratio or/argent élevé = prudence des investisseurs ; ratio cuivre/or en hausse = anticipations de croissance.")
