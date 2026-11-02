"""Paramètres de marché « à la Damodaran » : taux sans risque, prime de risque actions (ERP),
primes de risque pays (CRP), notation synthétique, données sectorielles indicatives.

⚠️ Valeurs INDICATIVES pré-remplies, toutes modifiables dans l'interface. Pour une valorisation
professionnelle, mettez-les à jour depuis pages.stern.nyu.edu/~adamodar (onglet Data) ou importez
le fichier sectoriel de Damodaran dans l'onglet « Coût du capital »."""
from __future__ import annotations
import pandas as pd

# ERP implicite S&P 500 au 1er oct. 2026 (site de A. Damodaran) : 4,20 % (T12M, payout ajusté)
MATURE_ERP = 4.20
US_DEFAULT_SPREAD = 0.22

CURRENCIES = {
    # devise : (taux sans risque indicatif %, libellé)
    "USD": (5.29, "US Treasury 10 ans (Damodaran, oct. 2026)"),
    "EUR": (2.80, "Bund allemand 10 ans (indicatif — à vérifier)"),
    "GBP": (4.60, "Gilt 10 ans (indicatif)"),
    "CHF": (0.70, "Confédération 10 ans (indicatif)"),
    "CAD": (3.40, "Canada 10 ans (indicatif)"),
    "JPY": (1.50, "JGB 10 ans (indicatif)"),
    "XOF": (6.50, "Approche EUR + différentiel d'inflation (indicatif)"),
    "XAF": (6.50, "Approche EUR + différentiel d'inflation (indicatif)"),
    "BRL": (13.00, "Brésil 10 ans (indicatif)"),
    "MAD": (3.90, "Maroc 10 ans (indicatif)"),
    "DKK": (2.50, "Danemark 10 ans (indicatif)"),
    "SEK": (2.50, "Suède 10 ans (indicatif)"),
    "NOK": (3.90, "Norvège 10 ans (indicatif)"),
    "AUD": (4.30, "Australie 10 ans (indicatif)"),
    "INR": (6.80, "Inde 10 ans (indicatif)"),
    "KRW": (3.00, "Corée 10 ans (indicatif)"),
    "TWD": (1.60, "Taïwan 10 ans (indicatif)"),
    "CNY": (1.90, "Chine 10 ans (indicatif)"),
    "HKD": (3.80, "Hong Kong 10 ans (indicatif)"),
    "SAR": (5.00, "Arabie saoudite 10 ans (indicatif)"),
    "ZAR": (10.00, "Afrique du Sud 10 ans (indicatif)"),
    "MXN": (9.50, "Mexique 10 ans (indicatif)"),
}

# Pays : (devise, prime de risque pays CRP %, taux IS marginal %)
COUNTRIES = {
    "France": ("EUR", 0.85, 25.0),
    "États-Unis": ("USD", 0.00, 25.0),
    "Allemagne": ("EUR", 0.00, 30.0),
    "Royaume-Uni": ("GBP", 0.85, 25.0),
    "Suisse": ("CHF", 0.00, 19.6),
    "Belgique": ("EUR", 0.85, 25.0),
    "Pays-Bas": ("EUR", 0.00, 25.8),
    "Espagne": ("EUR", 1.70, 25.0),
    "Italie": ("EUR", 2.40, 24.0),
    "Canada": ("CAD", 0.00, 26.5),
    "Japon": ("JPY", 0.85, 30.6),
    "Chine": ("CNY", 0.85, 25.0),
    "Hong Kong": ("HKD", 0.85, 16.5),
    "Taïwan": ("TWD", 0.85, 20.0),
    "Corée du Sud": ("KRW", 0.70, 24.0),
    "Inde": ("INR", 2.60, 25.2),
    "Australie": ("AUD", 0.00, 30.0),
    "Danemark": ("DKK", 0.00, 22.0),
    "Suède": ("SEK", 0.00, 20.6),
    "Norvège": ("NOK", 0.00, 22.0),
    "Irlande": ("EUR", 0.85, 15.0),
    "Arabie saoudite": ("SAR", 0.85, 20.0),
    "Afrique du Sud": ("ZAR", 3.60, 27.0),
    "Mexique": ("MXN", 2.60, 30.0),
    "Brésil": ("BRL", 3.90, 34.0),
    "Maroc": ("MAD", 2.60, 31.0),
    "Côte d'Ivoire": ("XOF", 4.30, 25.0),
    "Sénégal": ("XOF", 5.20, 30.0),
    "Cameroun": ("XAF", 8.00, 33.0),
    "Haïti": ("USD", 13.00, 30.0),
    "Autre (saisie libre)": ("USD", 0.00, 25.0),
}

# Notation synthétique (Damodaran) : (seuil de couverture EBIT/intérêts, notation, spread %)
RATING_LARGE = [(8.5, "AAA", 0.59), (6.5, "AA", 0.70), (5.5, "A+", 0.92), (4.25, "A", 1.07), (3.0, "A-", 1.21),
                (2.5, "BBB", 1.47), (2.25, "BB+", 1.83), (2.0, "BB", 2.20), (1.75, "B+", 2.94), (1.5, "B", 3.66),
                (1.25, "B-", 4.40), (0.8, "CCC", 5.88), (0.65, "CC", 8.62), (0.2, "C", 11.72), (-1e9, "D", 17.97)]
RATING_SMALL = [(12.5, "AAA", 0.59), (9.5, "AA", 0.70), (7.5, "A+", 0.92), (6.0, "A", 1.07), (4.5, "A-", 1.21),
                (4.0, "BBB", 1.47), (3.5, "BB+", 1.83), (3.0, "BB", 2.20), (2.5, "B+", 2.94), (2.0, "B", 3.66),
                (1.5, "B-", 4.40), (1.25, "CCC", 5.88), (0.8, "CC", 8.62), (0.5, "C", 11.72), (-1e9, "D", 17.97)]


def synthetic_rating(coverage: float, large: bool = True):
    table = RATING_LARGE if large else RATING_SMALL
    if coverage is None or coverage != coverage:
        coverage = 3.0
    if coverage == float("inf"):
        return "AAA", table[0][2]
    for th, r, s in table:
        if coverage > th:
            return r, s
    return "D", table[-1][2]


# Données sectorielles indicatives (ordre de grandeur des tables Damodaran « Global »).
# beta désendetté, marge opé. %, ventes/capital, ROIC %, EV/EBITDA, EV/Sales, PER, P/B
_IND = [
    ("Aéronautique & Défense", 0.95, 9.5, 1.3, 11, 15.0, 1.6, 26, 4.5),
    ("Agroalimentaire", 0.65, 11.0, 1.5, 12, 11.5, 1.4, 19, 2.8),
    ("Automobile & équipementiers", 0.95, 6.5, 1.6, 9, 7.0, 0.6, 9, 1.2),
    ("Banques (non applicable DCF FCFF)", 0.55, 30.0, 0.3, 10, 9.0, 3.0, 10, 1.1),
    ("BTP & Ingénierie", 0.85, 6.0, 2.2, 12, 9.0, 0.6, 15, 2.0),
    ("Chimie", 0.90, 11.0, 1.0, 9, 9.5, 1.2, 18, 2.0),
    ("Distribution / Retail", 0.80, 5.5, 2.5, 12, 10.0, 0.6, 20, 3.0),
    ("E-commerce", 1.20, 7.0, 2.4, 14, 20.0, 1.8, 35, 6.0),
    ("Énergie (pétrole & gaz)", 0.85, 14.0, 0.8, 10, 5.5, 1.1, 10, 1.5),
    ("Énergies renouvelables", 0.80, 18.0, 0.4, 5, 12.0, 3.5, 22, 1.6),
    ("Hôtellerie & Restauration", 0.95, 13.0, 1.0, 10, 13.0, 2.2, 25, 5.0),
    ("Immobilier (foncières)", 0.60, 35.0, 0.2, 5, 18.0, 7.0, 25, 1.0),
    ("Industrie diversifiée", 0.90, 10.5, 1.3, 11, 12.0, 1.5, 21, 3.0),
    ("Logiciels & SaaS", 1.20, 22.0, 1.2, 25, 22.0, 6.5, 35, 9.0),
    ("Luxe & Habillement", 1.00, 18.0, 1.2, 18, 15.0, 3.0, 25, 5.0),
    ("Matériaux & Mines", 0.95, 15.0, 0.7, 10, 7.0, 1.6, 13, 1.7),
    ("Médias & Divertissement", 0.95, 12.0, 1.0, 9, 11.0, 2.0, 20, 2.5),
    ("Pharmacie & Biotech", 1.00, 22.0, 0.9, 15, 14.0, 4.0, 25, 4.0),
    ("Santé — services & équipements", 0.85, 12.0, 1.2, 11, 15.0, 2.0, 28, 3.5),
    ("Semi-conducteurs", 1.30, 25.0, 0.8, 22, 22.0, 7.0, 35, 7.0),
    ("Services aux entreprises / Conseil", 0.90, 10.0, 2.5, 18, 14.0, 1.6, 24, 5.0),
    ("Services aux collectivités (utilities)", 0.45, 18.0, 0.4, 6, 10.0, 2.6, 18, 1.8),
    ("Fintech & Paiements", 1.15, 18.0, 1.3, 14, 18.0, 4.5, 30, 4.5),
    ("Télécoms", 0.65, 15.0, 0.6, 7, 6.5, 1.8, 16, 1.8),
    ("Transport & Logistique", 0.90, 8.0, 1.5, 9, 8.0, 0.9, 16, 2.2),
    ("Tech hardware & électronique", 1.10, 12.0, 1.4, 16, 14.0, 2.0, 25, 5.0),
    ("Marché total (moyenne)", 0.90, 12.0, 1.2, 11, 12.0, 2.0, 20, 2.8),
]
INDUSTRY = pd.DataFrame(_IND, columns=["Secteur", "Beta_u", "Marge_op", "Sales_to_capital", "ROIC",
                                       "EV_EBITDA", "EV_Sales", "PER", "P_B"]).set_index("Secteur")


def lever_beta(beta_u: float, de: float, tax: float) -> float:
    """Formule de Hamada (Damodaran) : βL = βU × (1 + (1 − t) × D/E)."""
    return beta_u * (1 + (1 - tax) * max(de, 0))


def fetch_market(ticker: str) -> dict:
    """Cours, nb d'actions et beta : yfinance, puis repli sur l'API chart de Yahoo (cours seul)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            return {"price": price, "shares": (info.get("sharesOutstanding") or 0) / 1e6 or None, "beta": info.get("beta"),
                    "currency": info.get("currency"), "name": info.get("longName") or info.get("shortName"), "ok": True}
    except Exception:
        pass
    try:
        from .fallback import info_from_chart
        i, _ = info_from_chart(ticker)
        return {"price": i.get("currentPrice"), "shares": None, "beta": None, "currency": i.get("currency"),
                "name": i.get("longName"), "ok": bool(i.get("currentPrice"))}
    except Exception as e:
        return {"ok": False, "error": str(e)}
