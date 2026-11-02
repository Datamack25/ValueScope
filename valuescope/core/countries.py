"""Analyse macro-financière des pays : finances publiques, dette, croissance, inflation, extérieur.
Sources ouvertes, sans clé d'API :
  • FMI — World Economic Outlook via DataMapper  (https://www.imf.org/external/datamapper/api/v1/)
  • Banque mondiale — WDI                         (https://api.worldbank.org/v2/)
Les séries FMI incluent les projections du WEO (années futures)."""
from __future__ import annotations
import numpy as np
import pandas as pd

IMF_URL = "https://www.imf.org/external/datamapper/api/v1/{ind}/{ctry}"
WB_URL = "https://api.worldbank.org/v2/country/{ctry}/indicator/{ind}?format=json&per_page=200&date=1990:2030"
HEADERS = {"User-Agent": "ValueScope/1.0 (analyse financière)"}

IMF_IND = {
    "GGXWDG_NGDP": ("Dette publique brute (% PIB)", "%"),
    "GGXWDN_NGDP": ("Dette publique nette (% PIB)", "%"),
    "GGXCNL_NGDP": ("Solde budgétaire (% PIB)", "%"),
    "GGXONLB_NGDP": ("Solde primaire (% PIB)", "%"),
    "GGR_NGDP": ("Recettes publiques (% PIB)", "%"),
    "GGX_NGDP": ("Dépenses publiques (% PIB)", "%"),
    "NGDP_RPCH": ("Croissance du PIB réel (%)", "%"),
    "PCPIPCH": ("Inflation (%)", "%"),
    "LUR": ("Chômage (%)", "%"),
    "BCA_NGDPD": ("Balance courante (% PIB)", "%"),
    "NGDPD": ("PIB (milliards USD)", "Md$"),
    "NGDPDPC": ("PIB par habitant (USD)", "$"),
}
WB_IND = {
    "FI.RES.TOTL.CD": ("Réserves de change totales (Md USD)", 1e-9),
    "GC.XPN.INTP.RV.ZS": ("Charge d'intérêts (% des recettes)", 1),
    "DT.DOD.DECT.GN.ZS": ("Dette extérieure (% RNB)", 1),
    "FR.INR.LEND": ("Taux débiteur bancaire (%)", 1),
}

COUNTRIES = {  # ISO3 : (nom FR, région)
    "FRA": ("France", "Europe"), "DEU": ("Allemagne", "Europe"), "ITA": ("Italie", "Europe"), "ESP": ("Espagne", "Europe"),
    "NLD": ("Pays-Bas", "Europe"), "BEL": ("Belgique", "Europe"), "PRT": ("Portugal", "Europe"), "GRC": ("Grèce", "Europe"),
    "IRL": ("Irlande", "Europe"), "AUT": ("Autriche", "Europe"), "FIN": ("Finlande", "Europe"), "GBR": ("Royaume-Uni", "Europe"),
    "CHE": ("Suisse", "Europe"), "SWE": ("Suède", "Europe"), "NOR": ("Norvège", "Europe"), "DNK": ("Danemark", "Europe"),
    "POL": ("Pologne", "Europe"), "ROU": ("Roumanie", "Europe"), "HUN": ("Hongrie", "Europe"), "TUR": ("Turquie", "Europe"),
    "USA": ("États-Unis", "Amériques"), "CAN": ("Canada", "Amériques"), "MEX": ("Mexique", "Amériques"), "BRA": ("Brésil", "Amériques"),
    "ARG": ("Argentine", "Amériques"), "CHL": ("Chili", "Amériques"), "COL": ("Colombie", "Amériques"), "PER": ("Pérou", "Amériques"),
    "HTI": ("Haïti", "Amériques"), "DOM": ("Rép. dominicaine", "Amériques"), "JAM": ("Jamaïque", "Amériques"),
    "JPN": ("Japon", "Asie-Pacifique"), "CHN": ("Chine", "Asie-Pacifique"), "IND": ("Inde", "Asie-Pacifique"),
    "KOR": ("Corée du Sud", "Asie-Pacifique"), "IDN": ("Indonésie", "Asie-Pacifique"), "VNM": ("Vietnam", "Asie-Pacifique"),
    "AUS": ("Australie", "Asie-Pacifique"), "SGP": ("Singapour", "Asie-Pacifique"),
    "SAU": ("Arabie saoudite", "Moyen-Orient & Afrique"), "ARE": ("Émirats arabes unis", "Moyen-Orient & Afrique"),
    "EGY": ("Égypte", "Moyen-Orient & Afrique"), "MAR": ("Maroc", "Moyen-Orient & Afrique"), "DZA": ("Algérie", "Moyen-Orient & Afrique"),
    "TUN": ("Tunisie", "Moyen-Orient & Afrique"), "NGA": ("Nigeria", "Moyen-Orient & Afrique"), "ZAF": ("Afrique du Sud", "Moyen-Orient & Afrique"),
    "KEN": ("Kenya", "Moyen-Orient & Afrique"), "CIV": ("Côte d'Ivoire", "Moyen-Orient & Afrique"), "SEN": ("Sénégal", "Moyen-Orient & Afrique"),
    "CMR": ("Cameroun", "Moyen-Orient & Afrique"), "GHA": ("Ghana", "Moyen-Orient & Afrique"),
}
NAME_TO_ISO = {v[0]: k for k, v in COUNTRIES.items()}


def fetch_imf(indicator: str, isos: list[str]) -> pd.DataFrame:
    """DataFrame index = années, colonnes = ISO3. Une requête « tous pays » (plus robuste qu'une URL à 50 segments)."""
    from . import net
    js = net.get(f"https://www.imf.org/external/datamapper/api/v1/{indicator}", "FMI (DataMapper)", timeout=40, retries=2)
    vals = (js.get("values") or {}).get(indicator) or {}
    keep = set(isos)
    df = pd.DataFrame({iso: {int(y): v for y, v in d.items() if v is not None} for iso, d in vals.items() if iso in keep})
    if df.empty:
        raise RuntimeError(f"FMI : série {indicator} vide")
    return df.sort_index()


# Équivalents Banque mondiale (repli si le FMI est injoignable) : code WB, facteur d'échelle
WB_EQUIV = {
    "GGXWDG_NGDP": ("GC.DOD.TOTL.GD.ZS", 1),      # dette de l'administration centrale (% PIB) — périmètre plus étroit que le FMI
    "GGXCNL_NGDP": ("GC.NLD.TOTL.GD.ZS", 1),
    "GGR_NGDP": ("GC.REV.XGRT.GD.ZS", 1),
    "GGX_NGDP": ("GC.XPN.TOTL.GD.ZS", 1),
    "NGDP_RPCH": ("NY.GDP.MKTP.KD.ZG", 1),
    "PCPIPCH": ("FP.CPI.TOTL.ZG", 1),
    "LUR": ("SL.UEM.TOTL.ZS", 1),
    "BCA_NGDPD": ("BN.CAB.XOKA.GD.ZS", 1),
    "NGDPD": ("NY.GDP.MKTP.CD", 1e-9),
    "NGDPDPC": ("NY.GDP.PCAP.CD", 1),
}


def fetch_wb_multi(indicator: str, isos: list[str], scale: float = 1.0) -> pd.DataFrame:
    from . import net
    js = net.get(f"https://api.worldbank.org/v2/country/{';'.join(isos)}/indicator/{indicator}",
                 "Banque mondiale", params={"format": "json", "per_page": 20000, "date": "1990:2030"}, timeout=40)
    if not isinstance(js, list) or len(js) < 2 or not js[1]:
        return pd.DataFrame()
    rows = [(d.get("countryiso3code"), int(d["date"]), d["value"] * scale) for d in js[1] if d.get("value") is not None]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["iso", "annee", "v"]).pivot_table(index="annee", columns="iso", values="v").sort_index()


def fetch_wb(indicator: str, iso: str) -> pd.Series:
    from . import net
    js = net.get(f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}", "Banque mondiale",
                 params={"format": "json", "per_page": 200, "date": "1990:2030"})
    if not isinstance(js, list) or len(js) < 2 or not js[1]:
        return pd.Series(dtype=float)
    s = pd.Series({int(d["date"]): d["value"] for d in js[1] if d.get("value") is not None}, dtype=float)
    return s.sort_index()


def profile(iso: str, imf_all: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tableau indicateurs × années pour un pays."""
    rows = {}
    for code, (label, _) in IMF_IND.items():
        df = imf_all.get(code)
        if df is not None and iso in df.columns:
            rows[label] = df[iso]
    return pd.DataFrame(rows).T.sort_index(axis=1) if rows else pd.DataFrame()


def latest(s: pd.Series, year: int):
    s = s.dropna()
    s = s[s.index <= year]
    return (float(s.iloc[-1]), int(s.index[-1])) if len(s) else (np.nan, None)


def debt_dynamics(debt: float, primary: float, r: float, g: float, years: int = 10) -> pd.Series:
    """Dynamique de la dette : d_{t+1} = d_t × (1+r)/(1+g) − solde primaire (en points de PIB, r et g nominaux)."""
    d, path = debt, [debt]
    for _ in range(years):
        d = d * (1 + r) / (1 + g) - primary
        path.append(d)
    return pd.Series(path, index=range(0, years + 1))


def stabilizing_primary(debt: float, r: float, g: float) -> float:
    """Solde primaire (% PIB) qui stabilise le ratio de dette : sp* = d × (r − g) / (1 + g)."""
    return debt * (r - g) / (1 + g)


def risk_score(m: dict) -> tuple[float, str, list[tuple[str, str]]]:
    """Score 0-100 (100 = risque élevé) à partir des fondamentaux. Grille transparente et indicative."""
    pts, notes = [], []

    def add(val, good, bad, w, label, show=None):
        if val is None or not np.isfinite(val):
            return
        x = float(np.clip((val - good) / (bad - good), 0, 1))
        pts.append((x, w))
        lvl = "ok" if x < .33 else "warn" if x < .66 else "bad"
        notes.append((lvl, f"{label} : {(val if show is None else show):.1f}"))

    g = lambda k: m.get(k) if m.get(k) is not None and np.isfinite(m.get(k)) else None
    if g("debt") is not None:
        add(g("debt"), 40, 120, 3, "Dette publique (% PIB)")
    if g("balance") is not None:
        add(-g("balance"), 0, 7, 2.5, "Solde budgétaire (% PIB)", g("balance"))
    if g("interest_rev") is not None:
        add(g("interest_rev"), 5, 25, 2, "Charge d'intérêts (% des recettes)")
    if g("inflation") is not None:
        add(abs(g("inflation") - 2), 0, 10, 1.5, "Inflation (%)", g("inflation"))
    if g("ca") is not None:
        add(-g("ca"), -2, 6, 1.5, "Balance courante (% PIB)", g("ca"))
    if g("growth") is not None:
        add(-g("growth"), -3, 1, 1, "Croissance réelle (%)", g("growth"))
    if g("unemp") is not None:
        add(g("unemp"), 4, 15, 1, "Chômage (%)")
    if not pts:
        return np.nan, "n.d.", notes
    score = 100 * sum(x * w for x, w in pts) / sum(w for _, w in pts)
    lab = "Faible" if score < 30 else "Modéré" if score < 50 else "Élevé" if score < 70 else "Très élevé"
    return score, lab, notes


def fetch_all_macro() -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Toutes les séries macro : FMI d'abord, chaque indicateur manquant complété par la Banque mondiale.
    Retourne (séries, indicateurs servis par la Banque mondiale, erreurs)."""
    isos = list(COUNTRIES)
    out, errors, wb_used = {}, [], []
    for code in IMF_IND:
        try:
            out[code] = fetch_imf(code, isos)
        except Exception as e:
            errors.append(f"FMI {code}: {e}")
    for code, (wb_code, scale) in WB_EQUIV.items():
        if code in out:
            continue
        try:
            d = fetch_wb_multi(wb_code, isos, scale)
            if not d.empty:
                out[code] = d
                wb_used.append(code)
        except Exception as e:
            errors.append(f"Banque mondiale {wb_code}: {e}")
    return out, wb_used, errors

