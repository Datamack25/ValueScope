"""Machine learning appliqué à l'analyse financière (scikit-learn).

1. Typologie des entreprises (K-means + ACP) et comparables par plus proches voisins.
2. Détection d'anomalies (Isolation Forest) sur les profils financiers.
3. Prévision des matières premières : rendement à 4 semaines, validation glissante (walk-forward)
   comparée à un modèle naïf, intervalles par régression quantile.
4. Typologie des pays (K-means sur les fondamentaux macro du FMI).
5. Risque de dégradation financière : classification supervisée sur le panel société-année de la base
   (ou sur un CSV étiqueté fourni par l'utilisateur), AUC en validation croisée.

Principe : chaque modèle affiche sa qualité mesurée hors échantillon ; aucune prévision n'est présentée sans elle."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42

# ====================================================================== 1. TYPOLOGIE & COMPARABLES
COMPANY_FEATURES = ["Croissance CA %", "Marge opé. %", "Marge nette %", "ROE %", "Dette/CP %", "Taille (log cap.)"]
MULTIPLES = ["PER", "EV/EBITDA", "EV/CA", "P/B"]


def company_matrix(scr: pd.DataFrame) -> pd.DataFrame:
    d = scr.copy()
    d["Taille (log cap.)"] = np.log10(d["Capitalisation (Md)"].clip(lower=0.01)) if "Capitalisation (Md)" in d else np.nan
    X = d.set_index("Ticker")[[c for c in COMPANY_FEATURES if c in d.columns]].astype(float)
    X = X.clip(X.quantile(.02), X.quantile(.98), axis=1)       # limite l'effet des valeurs extrêmes
    return X[X.notna().sum(axis=1) >= 3]


def _prep(X: pd.DataFrame):
    pipe = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    return pipe.fit_transform(X), pipe


def _name_cluster(centroid: pd.Series, overall: pd.Series, spread: pd.Series) -> str:
    z = ((centroid - overall) / spread.replace(0, np.nan)).dropna()
    words = {"Croissance CA %": ("forte croissance", "croissance faible"), "Marge opé. %": ("marges élevées", "marges faibles"),
             "ROE %": ("rentabilité élevée", "rentabilité faible"), "Dette/CP %": ("endettement élevé", "peu endetté"),
             "Taille (log cap.)": ("méga-capitalisations", "plus petites capitalisations"),
             "Dette publique": ("dette élevée", "dette faible"), "Solde budgétaire": ("budgets équilibrés", "déficits marqués"),
             "Croissance": ("croissance dynamique", "croissance atone"), "Inflation": ("inflation élevée", "inflation maîtrisée"),
             "Chômage": ("chômage élevé", "chômage bas"), "Balance courante": ("excédents extérieurs", "déficits extérieurs"),
             "PIB/hab. (log)": ("économies riches", "économies émergentes")}
    top = z.abs().sort_values(ascending=False).head(2).index
    parts = [words.get(k, (k + " ↑", k + " ↓"))[0 if z[k] > 0 else 1] for k in top if abs(z[k]) > .35]
    return " · ".join(parts).capitalize() if parts else "Profil médian"


def cluster(X: pd.DataFrame, k: int | None = None) -> dict:
    """K-means avec choix de k par silhouette si non fourni ; projection ACP 2D."""
    Z, _ = _prep(X)
    n = len(X)
    if n < 4:
        raise ValueError("Au moins 4 observations sont nécessaires.")
    if k is None:
        best = (2, -1)
        for kk in range(2, min(7, n - 1) + 1):
            lab = KMeans(kk, n_init=10, random_state=SEED).fit_predict(Z)
            s = silhouette_score(Z, lab)
            if s > best[1]:
                best = (kk, s)
        k = best[0]
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    lab = km.labels_
    sil = silhouette_score(Z, lab) if k > 1 and k < n else np.nan
    pca = PCA(2, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    Xf = X.fillna(X.median())
    names = {}
    for c in range(k):
        names[c] = _name_cluster(Xf[lab == c].mean(), Xf.median(), Xf.std())
    prof = Xf.groupby(lab).median()
    prof.index = [f"{c+1}. {names[c]}" for c in prof.index]
    prof.insert(0, "Effectif", pd.Series(lab).value_counts().sort_index().values)
    out = pd.DataFrame({"id": X.index, "cluster": [f"{c+1}. {names[c]}" for c in lab], "pc1": P[:, 0], "pc2": P[:, 1]})
    return dict(labels=out, k=k, silhouette=sil, profile=prof, explained=pca.explained_variance_ratio_)


def nearest_peers(target: dict, scr: pd.DataFrame, k: int = 6, exclude: str | None = None) -> pd.DataFrame:
    """Comparables par k plus proches voisins sur les fondamentaux standardisés."""
    X = company_matrix(scr)
    if exclude:
        X = X.drop(index=[exclude], errors="ignore")
    if len(X) < 3:
        return pd.DataFrame()
    t = pd.Series({c: target.get(c, np.nan) for c in X.columns}, dtype=float)
    avail = [c for c in X.columns if np.isfinite(t[c])]
    if len(avail) < 2:
        return pd.DataFrame()
    Xa = X[avail].fillna(X[avail].median())
    mu, sd = Xa.mean(), Xa.std().replace(0, 1)
    dist = np.sqrt((((Xa - mu) / sd - (t[avail] - mu) / sd) ** 2).sum(axis=1))
    idx = dist.sort_values().head(k).index
    peers = scr.set_index("Ticker").loc[idx, ["Société"] + [c for c in avail if c in scr.columns] +
                                         [m for m in MULTIPLES if m in scr.columns]].copy()
    peers.insert(1, "Distance", dist.loc[idx].round(2))
    return peers.reset_index()


# ====================================================================== 2. ANOMALIES
def anomalies(X: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    Z, _ = _prep(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED).fit(Z)
    score = -iso.score_samples(Z)
    zdf = pd.DataFrame(Z, index=X.index, columns=X.columns)
    expl = [", ".join(f"{c} {'↑' if zdf.loc[i, c] > 0 else '↓'}" for c in zdf.loc[i].abs().sort_values(ascending=False).index[:2])
            for i in X.index]
    out = pd.DataFrame({"id": X.index, "Score d'anomalie": score, "Atypique": iso.predict(Z) == -1,
                        "Principaux écarts": expl}).sort_values("Score d'anomalie", ascending=False)
    return out


# ====================================================================== 3. PRÉVISION MATIÈRES PREMIÈRES
def _features(s: pd.Series) -> pd.DataFrame:
    w = s.resample("W-FRI").last().dropna()
    lr = np.log(w).diff()
    f = pd.DataFrame(index=w.index)
    for l in (1, 2, 4, 8, 13, 26):
        f[f"ret_{l}s"] = np.log(w / w.shift(l))
    f["vol_13s"] = lr.rolling(13).std()
    f["vol_52s"] = lr.rolling(52).std()
    f["dist_mm26"] = np.log(w / w.rolling(26).mean())
    f["dist_mm52"] = np.log(w / w.rolling(52).mean())
    f["pos_52s"] = (w - w.rolling(52).min()) / (w.rolling(52).max() - w.rolling(52).min())
    f["mois_sin"] = np.sin(2 * np.pi * w.index.month / 12)
    f["mois_cos"] = np.cos(2 * np.pi * w.index.month / 12)
    f["y"] = np.log(w.shift(-4) / w)          # rendement des 4 semaines suivantes
    f["prix"] = w
    return f


def commodity_forecast(s: pd.Series, horizon_weeks: int = 4, model: str = "Gradient Boosting") -> dict:
    f0 = _features(s)
    f = f0.dropna(subset=[c for c in f0.columns if c != "y"])
    train = f.dropna(subset=["y"])
    if len(train) < 120:
        raise ValueError("Historique trop court (au moins ~2,5 ans de données hebdomadaires).")
    feats = [c for c in f.columns if c not in ("y", "prix")]
    X, y = train[feats].values, train["y"].values

    def make():
        if model == "Ridge (linéaire)":
            return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        if model == "Random Forest":
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=8, max_features=.5, random_state=SEED, n_jobs=-1)
        return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=.03, subsample=.8, random_state=SEED)

    # validation glissante : on ne prédit jamais avec des données du futur (écart de 4 semaines entre train et test)
    tss = TimeSeriesSplit(n_splits=5, gap=horizon_weeks)
    preds, actual, dates = [], [], []
    for tr, te in tss.split(X):
        m = make().fit(X[tr], y[tr])
        preds.append(m.predict(X[te])); actual.append(y[te]); dates.append(train.index[te])
    p, a, d = np.concatenate(preds), np.concatenate(actual), np.concatenate(dates)
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    rmse_naive = float(np.sqrt(np.mean(a ** 2)))                       # naïf : rendement nul (marche aléatoire)
    hit = float(np.mean(np.sign(p) == np.sign(a)))
    base_hit = float(max(np.mean(a > 0), np.mean(a < 0)))              # toujours parier sur le sens majoritaire
    strat = np.sign(p) * a                                              # stratégie long/short indicative (sans coûts)
    final = make().fit(X, y)
    x_last = f[feats].iloc[[-1]].values
    mu = float(final.predict(x_last)[0])
    q = {}
    for alpha in (.1, .9):
        gq = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=250, max_depth=2,
                                       learning_rate=.03, subsample=.8, random_state=SEED).fit(X, y)
        q[alpha] = float(gq.predict(x_last)[0])
    last = float(f["prix"].iloc[-1])
    imp = None
    if hasattr(final, "feature_importances_"):
        imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    elif hasattr(final, "named_steps"):
        imp = pd.Series(np.abs(final.named_steps["ridge"].coef_), index=feats).sort_values(ascending=False)
    return dict(last=last, date=f.index[-1], ret=mu, price=last * np.exp(mu), low=last * np.exp(min(q[.1], mu)),
                high=last * np.exp(max(q[.9], mu)), rmse=rmse, rmse_naive=rmse_naive, skill=1 - rmse / rmse_naive,
                hit=hit, base_hit=base_hit, n_test=len(a),
                backtest=pd.DataFrame({"Prévu": p, "Réalisé": a, "Stratégie": strat}, index=pd.to_datetime(d)),
                importance=imp)


# ====================================================================== 4. TYPOLOGIE DES PAYS
COUNTRY_MAP = {"GGXWDG_NGDP": "Dette publique", "GGXCNL_NGDP": "Solde budgétaire", "NGDP_RPCH": "Croissance",
               "PCPIPCH": "Inflation", "LUR": "Chômage", "BCA_NGDPD": "Balance courante", "NGDPDPC": "PIB/hab. (log)"}


def country_matrix(imf: dict[str, pd.DataFrame], year: int, window: int = 3) -> pd.DataFrame:
    """Moyenne sur `window` années jusqu'à `year` (lisse les chocs ponctuels)."""
    cols = {}
    for code, lab in COUNTRY_MAP.items():
        df = imf.get(code)
        if df is None:
            continue
        sub = df[(df.index > year - window) & (df.index <= year)]
        v = sub.mean()
        cols[lab] = np.log10(v.clip(lower=1)) if code == "NGDPDPC" else v
    X = pd.DataFrame(cols)
    X["Inflation"] = X["Inflation"].clip(upper=60) if "Inflation" in X else np.nan
    return X[X.notna().sum(axis=1) >= 4]


# ====================================================================== 5. RISQUE DE DÉGRADATION
DISTRESS_FEATURES = ["Marge opérationnelle", "ROA", "Dette nette / EBITDA", "Couverture des intérêts", "Liquidité générale",
                     "Autonomie financière", "Croissance du CA", "Conversion cash", "BFR / Total actif", "Réserves / Total actif"]


def company_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par exercice : ratios de solidité + événement de dégradation l'année suivante."""
    g = lambda k: df.loc[k] if k in df.index else pd.Series(np.nan, index=df.columns)
    with np.errstate(all="ignore"):
        rev, ebit, ni, ta = g("revenue"), g("ebit"), g("net_income"), g("total_assets")
        cov = ebit / g("interest_expense").replace(0, np.nan)
        p = pd.DataFrame({
            "Marge opérationnelle": ebit / rev,
            "ROA": ni / ta,
            "Dette nette / EBITDA": (g("total_debt") - g("cash")) / g("ebitda").where(g("ebitda") > 0),
            "Couverture des intérêts": cov.clip(-20, 50),
            "Liquidité générale": g("current_assets") / g("current_liabilities"),
            "Autonomie financière": g("equity") / ta,
            "Croissance du CA": rev.pct_change(),
            "Conversion cash": (g("cfo") / ta),
            "BFR / Total actif": (g("current_assets") - g("current_liabilities")) / ta,
            "Réserves / Total actif": g("retained_earnings") / ta,
        })
        stress = (ni < 0) | (cov < 1.5) | (g("equity") <= 0)
    p["Dégradation N+1"] = stress.shift(-1).astype(float).where(stress.shift(-1).notna())
    p.index.name = "Année"
    return p.replace([np.inf, -np.inf], np.nan)


def build_panel(financials: dict[str, tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    rows = []
    for tk, (df, meta) in financials.items():
        p = company_panel(df).reset_index()
        p.insert(0, "Société", meta.get("name", tk))
        p.insert(0, "Ticker", tk)
        rows.append(p)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def train_classifier(data: pd.DataFrame, features: list[str], target: str, model: str = "Régression logistique") -> dict:
    d = data.dropna(subset=[target])
    d = d[d[features].notna().sum(axis=1) >= max(2, len(features) // 2)]
    y = d[target].astype(int).values
    pos, n = int(y.sum()), len(y)
    if n < 30 or pos < 5 or n - pos < 5:
        raise ValueError(f"Données insuffisantes pour un apprentissage fiable : {n} observations dont {pos} cas positifs "
                         f"(minimum 30 observations et 5 cas de chaque classe).")
    X = d[features].astype(float)
    if model == "Gradient Boosting":
        clf = make_pipeline(SimpleImputer(strategy="median"),
                            GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=.05, random_state=SEED))
    else:
        clf = make_pipeline(SimpleImputer(strategy="median"), RobustScaler(),
                            LogisticRegression(C=.5, class_weight="balanced", max_iter=2000))
    cv = StratifiedKFold(n_splits=min(5, pos, n - pos), shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    clf.fit(X, y)
    last = clf.steps[-1][1]
    if hasattr(last, "coef_"):
        imp = pd.Series(last.coef_[0], index=features).sort_values(key=np.abs, ascending=False)
    else:
        imp = pd.Series(last.feature_importances_, index=features).sort_values(ascending=False)
    return dict(model=clf, features=features, auc=auc, n=n, pos=pos, base_rate=pos / n, importance=imp,
                oof=pd.DataFrame({"proba": proba, "y": y}, index=d.index))


def predict_proba(m: dict, X: pd.DataFrame) -> np.ndarray:
    return m["model"].predict_proba(X[m["features"]].astype(float))[:, 1]
