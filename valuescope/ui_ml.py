"""Pages « Machine learning » et « Base de données CSV »."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import ml as ML
from core import store as STORE
from core import companies as CO
from core import countries as CT
from core import commodities as CM
from core.parser import complete
from ui_kit import kpi, fnum, plot_layout, geo_layout, comment, hero, offline, NAVY, GOLD, TEAL, RED, GREY, PALETTE, GRID, DIVERGING
import ui_markets as UM

ss = st.session_state


def _scatter_layout(fig, h=460, title=None):
    plot_layout(fig, h, title)
    fig.update_xaxes(type="linear", showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ====================================================================== MACHINE LEARNING
def page_ml():
    hero("Intelligence · Machine learning", "Laisser les données parler — et mesurer ce qu'elles valent.",
         "Typologies et vrais comparables, détection d'anomalies, prévisions de matières premières testées hors échantillon, "
         "typologie des pays et modèle de risque de dégradation financière. Chaque modèle affiche sa qualité mesurée.")
    t1, t2, t3, t4, t5 = st.tabs(["🧭 Typologie & comparables", "🚨 Anomalies", "📈 Prévision matières premières",
                                  "🌍 Typologie des pays", "⚠️ Risque de dégradation"])
    scr = STORE.load_screener()
    with t1:
        _tab_clusters(scr)
    with t2:
        _tab_anomalies(scr)
    with t3:
        _tab_forecast()
    with t4:
        _tab_countries()
    with t5:
        _tab_distress()


def _need_companies(scr, n=8):
    if scr.empty or len(ML.company_matrix(scr)) < n:
        st.info(f"La base CSV contient {0 if scr.empty else len(scr)} société(s) : il en faut au moins {n}. "
                "Ouvrez l'onglet 🏢 Grandes entreprises (chaque consultation enrichit la base) ou utilisez "
                "« Actualiser l'univers » dans 💾 Base de données.")
        return True
    return False


def _tab_clusters(scr):
    if _need_companies(scr):
        return
    X = ML.company_matrix(scr)
    c1, c2 = st.columns([1, 3])
    auto = c1.toggle("Nombre de groupes automatique (silhouette)", value=True)
    k = None if auto else c1.slider("Nombre de groupes", 2, min(8, len(X) - 1), 4)
    res = ML.cluster(X, k)
    lab = res["labels"].merge(scr[["Ticker", "Société", "Capitalisation (Md)"]], left_on="id", right_on="Ticker")
    with c1:
        kpi(st, "Groupes", str(res["k"]), f"silhouette {fnum(res['silhouette'],2)}")
        st.caption("Silhouette : > 0,5 groupes nets ; 0,25–0,5 structure modérée ; < 0,25 groupes peu séparés.")
        st.caption(f"ACP : les 2 axes expliquent {res['explained'].sum()*100:.0f} % de la variance.")
    fig = px.scatter(lab, x="pc1", y="pc2", color="cluster", text="Société", size=lab["Capitalisation (Md)"].fillna(5).clip(lower=5),
                     size_max=38, color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="top center", textfont_size=9)
    c2.plotly_chart(_scatter_layout(fig, 520, "Carte des profils financiers (ACP)"), width="stretch")
    st.markdown("#### Profil médian de chaque groupe")
    st.dataframe(res["profile"].style.format("{:,.1f}"), width="stretch")
    st.markdown("#### Comparables les plus proches (k plus proches voisins)")
    a, b = st.columns([1, 3])
    tk = a.selectbox("Société", list(X.index), format_func=lambda t: f"{CO.ALL_TICKERS.get(t, t)} ({t})")
    kk = a.slider("Nombre de comparables", 3, 12, 6)
    row = scr.set_index("Ticker").loc[tk]
    target = {c: row.get(c, np.nan) for c in ML.COMPANY_FEATURES}
    target["Taille (log cap.)"] = np.log10(max(row.get("Capitalisation (Md)", np.nan) or 0.01, 0.01))
    peers = ML.nearest_peers(target, scr, kk, exclude=tk)
    if not peers.empty:
        b.dataframe(peers.style.format({c: "{:,.2f}" for c in peers.columns if c not in ("Ticker", "Société")}, na_rep="—"),
                    width="stretch", hide_index=True)
        med = peers[[m for m in ML.MULTIPLES if m in peers]].median()
        own = row[[m for m in ML.MULTIPLES if m in row.index]]
        cmp_ = pd.DataFrame({"Société": own, "Médiane des comparables ML": med, "Écart %": (own / med - 1) * 100})
        a.dataframe(cmp_.style.format("{:,.1f}", na_rep="—"), width="stretch")
        st.caption("Les comparables sont choisis sur les fondamentaux (croissance, marges, ROE, endettement, taille), "
                   "pas sur le secteur déclaré : c'est l'esprit de Damodaran — un multiple n'a de sens qu'entre sociétés "
                   "de risque, croissance et rentabilité comparables.")


def _tab_anomalies(scr):
    if _need_companies(scr):
        return
    X = ML.company_matrix(scr)
    cont = st.slider("Part attendue de profils atypiques", 0.03, 0.30, 0.10, 0.01)
    an = ML.anomalies(X, cont).merge(scr[["Ticker", "Société", "Secteur"]], left_on="id", right_on="Ticker")
    a, b = st.columns([1.2, 1])
    a.dataframe(an[["Société", "Secteur", "Score d'anomalie", "Atypique", "Principaux écarts"]]
                .style.format({"Score d'anomalie": "{:.3f}"}).apply(
                    lambda r: ["background-color: rgba(209,73,91,.22)" if r["Atypique"] else "" for _ in r], axis=1),
                width="stretch", hide_index=True, height=520)
    d = X.reset_index().merge(an[["id", "Atypique"]], left_on="Ticker", right_on="id")
    d = d.merge(scr[["Ticker", "Société"]], on="Ticker")
    if {"Marge opé. %", "Dette/CP %"} <= set(d.columns):
        fig = px.scatter(d, x="Marge opé. %", y="Dette/CP %", color=d["Atypique"].map({True: "Atypique", False: "Normal"}),
                         text="Société", color_discrete_map={"Atypique": RED, "Normal": NAVY})
        fig.update_traces(textposition="top center", textfont_size=9)
        b.plotly_chart(_scatter_layout(fig, 520, "Rentabilité vs endettement"), width="stretch")
    comment("info", "Isolation Forest isole les profils qui se distinguent de l'ensemble. Un profil atypique n'est pas une fraude : "
                    "c'est une invitation à lire les comptes (modèle économique particulier, retraitement, événement exceptionnel). "
                    "En LCB-FT, la même logique sert à prioriser les revues.")


def _tab_forecast():
    try:
        px_, src, when = UM.get_prices()
    except Exception as e:
        offline(e, "ui_ml_1"); return
    avail = [t for t in CM.COMMODITIES if t in px_.columns and px_[t].notna().sum() > 700]
    if not avail:
        st.info("Historique insuffisant dans la base. Ouvrez 🛢 Matières premières avec une connexion active."); return
    c1, c2 = st.columns(2)
    tk = c1.selectbox("Matière première", avail, format_func=lambda t: CM.COMMODITIES[t][0], key="ml_cm")
    model = c2.selectbox("Modèle", ["Gradient Boosting", "Random Forest", "Ridge (linéaire)"])
    with st.spinner("Entraînement et validation glissante…"):
        try:
            r = ML.commodity_forecast(px_[tk].dropna(), 4, model)
        except Exception as e:
            st.warning(str(e)); return
    unit = CM.COMMODITIES[tk][2]
    k = st.columns(5)
    kpi(k[0], "Dernier cours", fnum(r["last"], 2), f"{unit} · {r['date'].date()}")
    kpi(k[1], "Prévision à 4 semaines", fnum(r["price"], 2), f"{r['ret']*100:+.1f} %")
    kpi(k[2], "Intervalle 10 %–90 %", f"{fnum(r['low'],1)} – {fnum(r['high'],1)}", "régression quantile")
    kpi(k[3], "Gain vs marche aléatoire", f"{r['skill']*100:+.1f} %", "réduction de l'erreur (RMSE)")
    kpi(k[4], "Bon sens de variation", f"{r['hit']*100:.0f} %", f"référence naïve {r['base_hit']*100:.0f} % · {r['n_test']} tests")
    st.write("")
    if r["skill"] <= 0.01 or r["hit"] <= r["base_hit"] + 0.02:
        comment("bad", "Hors échantillon, le modèle ne fait pas mieux qu'une marche aléatoire ou qu'un pari naïf : "
                       "sa prévision ne doit pas guider une décision. C'est le résultat le plus fréquent sur les marchés de matières premières.")
    elif r["skill"] < 0.05:
        comment("warn", "Pouvoir prédictif faible mais positif hors échantillon : à utiliser comme signal d'appoint, jamais seul.")
    else:
        comment("ok", "Le modèle améliore la prévision hors échantillon par rapport à la marche aléatoire. Restez prudent : "
                      "les régimes de marché changent et les performances passées ne garantissent rien.")
    a, b = st.columns([1.6, 1])
    with a:
        bt = r["backtest"].sort_index().iloc[::4]          # périodes de 4 semaines sans chevauchement
        cum = pd.DataFrame({"Stratégie long/short du modèle": np.exp(bt["Stratégie"].cumsum()) - 1,
                            "Détention simple": np.exp(bt["Réalisé"].cumsum()) - 1})
        fig = go.Figure()
        for j, col in enumerate(cum.columns):
            fig.add_scatter(x=cum.index, y=cum[col] * 100, name=col, line=dict(color=[NAVY, GOLD][j], width=2.4))
        fig = plot_layout(fig, 360, "Backtest glissant — rendements cumulés (%, hors frais, périodes de 4 semaines)")
        fig.update_xaxes(type="date")
        st.plotly_chart(fig, width="stretch")
    with b:
        if r["importance"] is not None:
            imp = r["importance"].head(8)[::-1]
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=TEAL))
            fig = plot_layout(fig, 360, "Variables les plus utilisées")
            fig.update_xaxes(type="linear"); fig.update_yaxes(type="category")
            st.plotly_chart(fig, width="stretch")
    st.caption("Méthode : variables construites uniquement avec le passé (rendements 1 à 26 semaines, volatilité, distance aux moyennes, "
               "position dans la fourchette 52 semaines, saisonnalité) ; validation TimeSeriesSplit à 5 plis avec un écart de 4 semaines "
               "entre apprentissage et test, pour éviter toute fuite d'information.")


def _tab_countries():
    try:
        imf, src, _ = UM.get_imf()
    except Exception as e:
        offline(e, "ui_ml_2"); return
    from datetime import date
    yrs = sorted(imf.get("GGXWDG_NGDP", pd.DataFrame()).index)
    if not yrs:
        st.info("Données FMI absentes."); return
    c1, c2 = st.columns([1, 3])
    year = c1.select_slider("Année de référence (moyenne 3 ans)", options=[y for y in yrs if y >= 2000],
                            value=min(date.today().year, max(yrs)))
    X = ML.country_matrix(imf, year)
    if len(X) < 8:
        st.info("Pas assez de pays renseignés."); return
    auto = c1.toggle("Nombre de groupes automatique", value=True, key="ctry_auto")
    k = None if auto else c1.slider("Nombre de groupes", 2, 8, 4, key="ctry_k")
    res = ML.cluster(X, k)
    names = {iso: v[0] for iso, v in CT.COUNTRIES.items()}
    lab = res["labels"].assign(pays=lambda d: d["id"].map(names))
    kpi(c1, "Groupes", str(res["k"]), f"silhouette {fnum(res['silhouette'],2)}")
    fig = px.choropleth(lab, locations="id", color="cluster", hover_name="pays", projection="natural earth",
                        color_discrete_sequence=PALETTE)
    geo_layout(fig, 430)
    c2.plotly_chart(fig, width="stretch")
    a, b = st.columns([1.3, 1])
    a.dataframe(res["profile"].style.format("{:,.1f}"), width="stretch")
    a.caption("PIB/hab. en log10 (4 = 10 000 $, 4,7 ≈ 50 000 $).")
    fig = px.scatter(lab, x="pc1", y="pc2", color="cluster", text="pays", color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="top center", textfont_size=8)
    b.plotly_chart(_scatter_layout(fig, 420, "Proximité des profils (ACP)"), width="stretch")


def _distress_template() -> bytes:
    t = pd.DataFrame(columns=["Societe", "Annee"] + ML.DISTRESS_FEATURES + ["defaut"])
    t.loc[0] = ["Exemple SA", 2024, .08, .04, 2.1, 6.5, 1.3, .38, .05, .07, .12, .2, 0]
    return t.to_csv(index=False, sep=";", decimal=",").encode("utf-8")


def _tab_distress():
    st.markdown("Le modèle apprend, sur des couples société-année, quels profils de ratios précèdent une **dégradation financière "
                "l'année suivante** (perte nette, couverture des intérêts < 1,5x ou fonds propres négatifs) — ou un **défaut** "
                "si vous fournissez vos propres données étiquetées.")
    src = st.radio("Données d'apprentissage", ["Panel de la base CSV (sociétés enregistrées)", "Mon fichier CSV étiqueté"], horizontal=True)
    target = "Dégradation N+1"
    if src.startswith("Panel"):
        panel = ML.build_panel(STORE.all_financials())
        if panel.empty:
            st.info("Aucun état financier dans la base. Consultez des fiches sociétés ou actualisez l'univers."); return
    else:
        st.download_button("⬇ Modèle de fichier", _distress_template(), "modele_risque_credit.csv")
        up = st.file_uploader("CSV avec les colonnes du modèle et une colonne cible 0/1", type=["csv"], key="dist_up")
        if not up:
            return
        raw = up.getvalue().decode("utf-8", errors="ignore")
        sep = ";" if raw.count(";") > raw.count(",") else ","
        panel = pd.read_csv(io.StringIO(raw), sep=sep, decimal="," if sep == ";" else ".")
        target = st.selectbox("Colonne cible (1 = défaut / dégradation)", list(panel.columns), index=len(panel.columns) - 1)
    feats_all = [c for c in ML.DISTRESS_FEATURES if c in panel.columns] or \
        [c for c in panel.select_dtypes("number").columns if c != target]
    c1, c2 = st.columns([2, 1])
    feats = c1.multiselect("Variables explicatives", feats_all, default=feats_all)
    model = c2.selectbox("Algorithme", ["Régression logistique", "Gradient Boosting"])
    if not feats:
        return
    k = st.columns(3)
    yv = pd.to_numeric(panel[target], errors="coerce") if target in panel else pd.Series(dtype=float)
    kpi(k[0], "Observations", str(int(yv.notna().sum())))
    kpi(k[1], "Cas positifs", str(int((yv == 1).sum())))
    kpi(k[2], "Sociétés", str(panel["Ticker"].nunique() if "Ticker" in panel else panel.iloc[:, 0].nunique()))
    if st.button("Entraîner le modèle", type="primary"):
        try:
            panel[target] = yv
            m = ML.train_classifier(panel, feats, target, model)
            ss._distress = m
            ss._distress_src = src
        except Exception as e:
            st.warning(str(e)); return
    m = ss.get("_distress")
    if not m:
        return
    st.write("")
    k = st.columns(4)
    kpi(k[0], "AUC (validation croisée)", fnum(m["auc"], 3), "0,5 = hasard · > 0,75 = bon")
    kpi(k[1], "Taux de base", fnum(m["base_rate"], 1, True), f"{m['pos']} cas / {m['n']}")
    kpi(k[2], "Modèle", "Logit" if "logistique" in str(model).lower() else "GBM")
    kpi(k[3], "Variables", str(len(m["features"])))
    if m["auc"] < 0.65:
        comment("bad", "Pouvoir discriminant faible : n'utilisez pas ce modèle pour décider ; enrichissez la base (plus de sociétés et d'années).")
    elif m["auc"] < 0.75:
        comment("warn", "Pouvoir discriminant moyen : utile pour classer les dossiers à revoir en priorité.")
    else:
        comment("ok", "Bon pouvoir discriminant hors échantillon. Vérifiez la stabilité avec davantage de données avant tout usage en crédit.")
    a, b = st.columns(2)
    imp = m["importance"].head(10)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=[TEAL if v >= 0 else RED for v in imp.values]))
    fig = plot_layout(fig, 360, "Poids des variables (coefficients standardisés ou importance)")
    fig.update_xaxes(type="linear"); fig.update_yaxes(type="category")
    a.plotly_chart(fig, width="stretch")
    if "Ticker" in panel and "Année" in panel:
        latest = panel.sort_values("Année").groupby("Ticker").tail(1).copy()
        latest["Probabilité de dégradation"] = ML.predict_proba(m, latest)
        latest = latest.sort_values("Probabilité de dégradation", ascending=False)
        b.dataframe(latest[["Société", "Année", "Probabilité de dégradation"]].style.format({"Probabilité de dégradation": "{:.1%}"}),
                    width="stretch", hide_index=True, height=360)
    if "df" in ss and ss.df is not None and not ss.df.empty:
        cur = ML.company_panel(complete(ss.df)).iloc[[-1]]
        pr = float(ML.predict_proba(m, cur)[0])
        st.markdown(f"#### Société en cours d'analyse : {ss.get('name')}")
        comment("warn" if pr > m["base_rate"] * 1.5 else "info",
                f"Probabilité estimée de dégradation l'an prochain : <b>{pr*100:.1f} %</b> (taux de base {m['base_rate']*100:.1f} %).")
        if st.button("Utiliser cette probabilité comme risque de défaillance dans le DCF"):
            ss._pending = {"p_fail": float(round(min(pr, 0.6), 2))}
            ss._flash = "Probabilité appliquée dans Valorisation › Risque & Monte Carlo."
            st.rerun()
        st.caption("Une dégradation n'est pas une faillite : ajustez la valeur dans l'onglet Valorisation si besoin.")


# ====================================================================== BASE DE DONNÉES
def page_data():
    hero("Données · Base CSV", "Vos données essentielles, conservées.",
         "Chaque consultation en ligne alimente une base CSV locale : comptes et multiples des sociétés, séries FMI et Banque "
         "mondiale, cours des matières premières, journal de vos valorisations. La plateforme peut ensuite fonctionner hors-ligne.")
    st.dataframe(STORE.status(), width="stretch", hide_index=True)
    comment("warn", "Sur Streamlit Community Cloud, le disque est effacé à chaque redémarrage de l'application. "
                    "Téléchargez régulièrement la base (zip) et placez les CSV dans <code>data/store/</code> de votre dépôt GitHub : "
                    "ils seront rechargés à chaque démarrage.")
    a, b, c = st.columns(3)
    a.download_button("⬇ Télécharger toute la base (zip)", STORE.export_zip(), "valuescope_base_csv.zip", width="stretch")
    with b.popover("⬆ Restaurer / importer des CSV", width="stretch"):
        ups = st.file_uploader("Zip de la base ou fichiers CSV portant les noms attendus", type=["zip", "csv"],
                               accept_multiple_files=True, key="store_up")
        if ups and st.button("Importer", type="primary"):
            ok = STORE.import_files([(u.name, u.getvalue()) for u in ups])
            st.success(f"{len(ok)} fichier(s) restauré(s) : {', '.join(ok) or 'aucun nom reconnu'}")
    with c.popover("🗑 Vider un jeu de données", width="stretch"):
        k = st.selectbox("Jeu de données", list(STORE.LABELS), format_func=lambda x: STORE.LABELS[x])
        if st.button("Supprimer définitivement"):
            STORE.clear(k); st.rerun()

    st.markdown("### Actualiser la base")
    st.caption("Nécessite une connexion Internet. L'univers complet compte environ 90 sociétés : comptez quelques minutes avec les comptes détaillés.")
    c1, c2, c3 = st.columns(3)
    groups = c1.multiselect("Univers", list(CO.UNIVERSE), default=list(CO.UNIVERSE))
    with_fin = c1.checkbox("Inclure les états financiers détaillés (pour le machine learning)", value=True)
    if c1.button("Actualiser les sociétés", type="primary", width="stretch"):
        tks = [t for g in groups for t in CO.UNIVERSE[g]]
        bar = st.progress(0.0, text="Démarrage…")
        rows, okf = [], 0
        for i, tk in enumerate(tks):
            bar.progress((i + 1) / len(tks), text=f"{CO.ALL_TICKERS.get(tk, tk)} ({i+1}/{len(tks)})")
            try:
                if with_fin:
                    cdat = CO.fetch_company(tk)
                    rows.append(CO.row_from_info(tk, cdat["info"]))
                    if not cdat["df"].empty:
                        STORE.save_financials(tk, cdat["info"].get("longName") or CO.ALL_TICKERS.get(tk, tk),
                                              cdat["fin_currency"], cdat["df"]); okf += 1
                else:
                    rows.append(CO.fetch_screener([tk]).iloc[0].to_dict())
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty and df["Cours"].notna().any():
            STORE.save_screener(df)
            st.success(f"{int(df['Cours'].notna().sum())} sociétés mises à jour" + (f", {okf} jeux d'états financiers." if with_fin else "."))
        else:
            offline("Aucune donnée reçue.", "ui_ml_3")
    if c2.button("Actualiser les pays (FMI + Banque mondiale)", width="stretch"):
        try:
            with st.spinner("FMI…"):
                STORE.save_imf({k: v for k, v in UM._imf_live().items() if k != "_meta"})
            bar = st.progress(0.0)
            for i, iso in enumerate(CT.COUNTRIES):
                bar.progress((i + 1) / len(CT.COUNTRIES), text=CT.COUNTRIES[iso][0])
                try:
                    STORE.save_wb(iso, UM._wb_live(iso))
                except Exception:
                    pass
            st.success("Séries pays enregistrées.")
        except Exception as e:
            offline(e, "ui_ml_4")
    if c3.button("Actualiser les matières premières", width="stretch"):
        try:
            with st.spinner("Cours…"):
                STORE.save_prices(UM._prices_live())
            st.success("Cours enregistrés.")
        except Exception as e:
            offline(e, "ui_ml_5")

    st.markdown("### 🩺 Diagnostic des sources de données")
    st.caption("Vérifie, depuis le serveur, l'accès à chaque source et propose une solution en cas d'échec.")
    if st.button("Tester les sources", key="diag_page"):
        with st.spinner("Test en cours (jusqu'à 1 minute)…"):
            from ui_kit import diagnose
            st.dataframe(diagnose(), width="stretch", hide_index=True)
    from core import net as NET
    if NET.LOG:
        st.dataframe(pd.DataFrame([(k, "✅" if v[0] else "❌", v[2], v[1]) for k, v in NET.LOG.items()],
                                  columns=["Source", "Dernier statut", "Heure", "Message"]), width="stretch", hide_index=True)

    st.markdown("### Explorer un jeu de données")
    k = st.selectbox("Fichier", list(STORE.FILES), format_func=lambda x: STORE.LABELS[x], key="peek")
    p = STORE.ROOT / STORE.FILES[k]
    if p.exists():
        d = pd.read_csv(p, nrows=5000, index_col=0 if k == "commodities_prices" else None)
        st.dataframe(d.tail(500) if k == "commodities_prices" else d, width="stretch", height=420)
        st.download_button(f"⬇ {STORE.FILES[k]}", p.read_bytes(), STORE.FILES[k], mime="text/csv")
    else:
        st.caption("Fichier vide pour l'instant.")
