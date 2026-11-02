"""Composants d'interface partagés."""
from __future__ import annotations
import numpy as np
import streamlit as st

# Couleurs choisies pour rester lisibles sur fond clair comme sur fond sombre :
# le texte des graphiques n'est jamais forcé, il suit le thème Streamlit (clair ou sombre) de l'utilisateur.
NAVY, GOLD, TEAL, RED, GREY = "#3D6FB6", "#C9A227", "#1B998B", "#D1495B", "#8A94A6"
GRID = "rgba(128,128,128,0.25)"
PALETTE = [NAVY, GOLD, TEAL, RED, "#8E6CC0", "#E07A5F", "#4CAF50", GREY, "#5BC0EB", "#B5838D", "#F4A259", "#6D9DC5"]
DIVERGING = [[0, "#5B8FD6"], [0.5, "#EDEDED"], [1, "#E07A7A"]]      # texte noir lisible sur toute l'échelle


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='vs-card'><div class='vs-kpi-l'>{label}</div><div class='vs-kpi-v'>{value}</div>"
                 f"<div class='vs-kpi-s'>{sub}</div></div>", unsafe_allow_html=True)


def fnum(v, d=1, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n.d."
    if pct:
        return f"{v*100:,.{d}f} %".replace(",", " ")
    return f"{v:,.{d}f}".replace(",", " ")


def plot_layout(fig, h=360, title=None):
    """Mise en page commune : fond transparent, police et couleur de texte héritées du thème (clair/sombre)."""
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=56 if title else 14, b=10),
                      title=dict(text=title, x=0, xanchor="left", y=.98, yanchor="top", font=dict(family="Fraunces, Georgia, serif", size=16)) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif"),
                      legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"),
                      modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False, type="category", gridcolor=GRID)
    return fig


def geo_layout(fig, h=460):
    """Cartes : fond transparent, terres neutres lisibles dans les deux thèmes."""
    fig.update_geos(showcountries=True, countrycolor="rgba(128,128,128,.5)", showcoastlines=False, showland=True,
                    landcolor="rgba(128,128,128,.12)", showocean=False, showlakes=False, bgcolor="rgba(0,0,0,0)",
                    showframe=False)
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif"),
                      legend=dict(orientation="h", y=-0.05, bgcolor="rgba(0,0,0,0)"), modebar=dict(orientation="v"))
    return fig


def comment(lvl, txt):
    st.markdown(f"<div class='vs-comment {lvl}'>{txt}</div>", unsafe_allow_html=True)


def hero(kicker, title, text):
    st.markdown(f"<div class='vs-hero'><div class='vs-kicker'>{kicker}</div><h1>{title}</h1><p>{text}</p></div>",
                unsafe_allow_html=True)


def diagnose() -> "pd.DataFrame":
    """Teste chaque source de données et renvoie un tableau d'état (statut, durée, détail)."""
    import time
    import pandas as pd
    from core import net, fallback as FB, countries as CT

    def yf_test():
        import yfinance as yf
        i = yf.Ticker("AAPL").info or {}
        if not (i.get("currentPrice") or i.get("regularMarketPrice")):
            raise RuntimeError("réponse vide (limitation de débit probable)")
        return f"Apple {i.get('currentPrice') or i.get('regularMarketPrice')}"

    tests = [
        ("Yahoo Finance — yfinance (fondamentaux)", yf_test),
        ("Yahoo Finance — API chart (cours, repli)", lambda: f"Or {FB.chart('GC=F', '5d', '1d')[0].iloc[-1]:.1f}"),
        ("SEC EDGAR (comptes US, repli)", lambda: f"CIK Apple {FB._cik('AAPL')}"),
        ("FMI DataMapper (macro)", lambda: f"{len(CT.fetch_imf('NGDP_RPCH', ['FRA', 'USA']).columns)} pays"),
        ("Banque mondiale (macro, repli)", lambda: f"{len(CT.fetch_wb('NY.GDP.MKTP.CD', 'FRA'))} années"),
    ]
    rows = []
    for name, fn in tests:
        t0 = time.time()
        try:
            msg = fn()
            rows.append((name, "✅ OK", f"{time.time()-t0:.1f} s", msg, ""))
        except Exception as e:
            rows.append((name, "❌ Échec", f"{time.time()-t0:.1f} s", str(e)[:200], net.explain(e)))
    return pd.DataFrame(rows, columns=["Source", "Statut", "Durée", "Détail", "Conseil"])


def offline(err, key: str | None = None):
    """Message d'erreur explicite : cause probable, conseil, détail technique et test des sources."""
    from core import net
    st.error("Les données n'ont pas pu être chargées.")
    st.markdown(f"<div class='vs-comment warn'><b>Cause probable :</b> {net.explain(err)}</div>", unsafe_allow_html=True)
    with st.expander("Détail technique", expanded=True):
        st.code(str(err)[:1500] or "—")
    c1, c2 = st.columns(2)
    if c1.button("🔄 Réessayer", key=f"retry_{key or id(err)}"):
        st.cache_data.clear()
        st.rerun()
    if c2.button("🩺 Tester toutes les sources", key=f"diag_{key or id(err)}"):
        with st.spinner("Test des sources…"):
            st.dataframe(diagnose(), width="stretch", hide_index=True)
    st.caption("Astuce : en mode « Hors-ligne (base CSV) », la page utilise les données déjà enregistrées.")
