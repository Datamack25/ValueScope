"""Grandes entreprises cotées : univers de référence + récupération des états financiers via Yahoo Finance
(yfinance) et conversion vers le modèle standard de ValueScope (montants en millions de la devise de reporting)."""
from __future__ import annotations
import numpy as np
import pandas as pd

UNIVERSE = {
    "🇫🇷 France (CAC 40)": {
        "MC.PA": "LVMH", "OR.PA": "L'Oréal", "RMS.PA": "Hermès", "TTE.PA": "TotalEnergies", "SAN.PA": "Sanofi",
        "AIR.PA": "Airbus", "SU.PA": "Schneider Electric", "AI.PA": "Air Liquide", "SAF.PA": "Safran",
        "EL.PA": "EssilorLuxottica", "DG.PA": "Vinci", "KER.PA": "Kering", "ORA.PA": "Orange", "CAP.PA": "Capgemini",
        "DSY.PA": "Dassault Systèmes", "BN.PA": "Danone", "RI.PA": "Pernod Ricard", "STLAP.PA": "Stellantis",
        "RNO.PA": "Renault", "ENGI.PA": "Engie", "SGO.PA": "Saint-Gobain", "CA.PA": "Carrefour",
        "BNP.PA": "BNP Paribas", "CS.PA": "AXA",
    },
    "🇺🇸 États-Unis": {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta",
        "TSLA": "Tesla", "AVGO": "Broadcom", "JPM": "JPMorgan Chase", "V": "Visa", "JNJ": "Johnson & Johnson",
        "XOM": "ExxonMobil", "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola", "MCD": "McDonald's",
        "NKE": "Nike", "DIS": "Disney", "NFLX": "Netflix", "BA": "Boeing", "PFE": "Pfizer", "COST": "Costco",
        "ORCL": "Oracle", "AMD": "AMD", "INTC": "Intel", "CAT": "Caterpillar",
    },
    "🇪🇺 Europe": {
        "ASML.AS": "ASML", "SAP.DE": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz", "NOVO-B.CO": "Novo Nordisk",
        "NESN.SW": "Nestlé", "ROG.SW": "Roche", "NOVN.SW": "Novartis", "SHEL.L": "Shell", "AZN.L": "AstraZeneca",
        "ULVR.L": "Unilever", "BP.L": "BP", "ITX.MC": "Inditex", "IBE.MC": "Iberdrola", "ENEL.MI": "Enel",
        "RACE.MI": "Ferrari", "VOW3.DE": "Volkswagen", "BMW.DE": "BMW", "MBG.DE": "Mercedes-Benz",
        "ADYEN.AS": "Adyen", "VOLV-B.ST": "Volvo", "HSBA.L": "HSBC",
    },
    "🌏 Asie & reste du monde": {
        "7203.T": "Toyota", "6758.T": "Sony", "9984.T": "SoftBank Group", "2330.TW": "TSMC",
        "005930.KS": "Samsung Electronics", "9988.HK": "Alibaba", "0700.HK": "Tencent", "BHP.AX": "BHP",
        "RELIANCE.NS": "Reliance Industries", "INFY.NS": "Infosys", "VALE3.SA": "Vale", "PETR4.SA": "Petrobras",
        "SHOP.TO": "Shopify", "RY.TO": "Royal Bank of Canada", "2222.SR": "Saudi Aramco", "NPN.JO": "Naspers",
    },
}

ALL_TICKERS = {t: n for g in UNIVERSE.values() for t, n in g.items()}

# Libellés Yahoo Finance -> clés ValueScope (par ordre de priorité)
YF_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "da": ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement", "Depreciation Amortization Depletion"],
    "ebit": ["Operating Income", "EBIT"],
    "rnd": ["Research And Development"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "pretax_income": ["Pretax Income"],
    "tax": ["Tax Provision"],
    "net_income": ["Net Income Common Stockholders", "Net Income", "Net Income From Continuing Operation Net Minority Interest"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "inventory": ["Inventory"],
    "current_assets": ["Current Assets"],
    "ppe": ["Net PPE"],
    "intangibles": ["Goodwill And Other Intangible Assets"],
    "total_assets": ["Total Assets"],
    "payables": ["Accounts Payable", "Payables"],
    "current_liabilities": ["Current Liabilities"],
    "short_debt": ["Current Debt"],
    "long_debt": ["Long Term Debt"],
    "lease_liabilities": ["Long Term Capital Lease Obligation"],
    "total_debt": ["Total Debt"],
    "minority": ["Minority Interest"],
    "equity": ["Stockholders Equity", "Common Stock Equity"],
    "retained_earnings": ["Retained Earnings"],
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "buybacks": ["Repurchase Of Capital Stock", "Common Stock Payments"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
}

COUNTRY_MAP = {"France": "France", "United States": "États-Unis", "Germany": "Allemagne", "United Kingdom": "Royaume-Uni",
               "Switzerland": "Suisse", "Belgium": "Belgique", "Netherlands": "Pays-Bas", "Spain": "Espagne",
               "Italy": "Italie", "Canada": "Canada", "Japan": "Japon", "China": "Chine", "Hong Kong": "Hong Kong",
               "Taiwan": "Taïwan", "Denmark": "Danemark", "Sweden": "Suède", "Norway": "Norvège", "Australia": "Australie",
               "India": "Inde", "South Korea": "Corée du Sud", "Saudi Arabia": "Arabie saoudite", "South Africa": "Afrique du Sud",
               "Ireland": "Irlande", "Mexico": "Mexique", "Brazil": "Brésil", "Morocco": "Maroc"}

SECTOR_MAP = [  # (mot-clé dans industry/sector Yahoo, secteur ValueScope)
    ("semiconductor", "Semi-conducteurs"), ("software", "Logiciels & SaaS"), ("internet retail", "E-commerce"),
    ("internet content", "Médias & Divertissement"), ("entertainment", "Médias & Divertissement"),
    ("luxury", "Luxe & Habillement"), ("apparel", "Luxe & Habillement"), ("footwear", "Luxe & Habillement"),
    ("household & personal", "Agroalimentaire"), ("beverages", "Agroalimentaire"), ("packaged foods", "Agroalimentaire"),
    ("confectioners", "Agroalimentaire"), ("drug manufacturers", "Pharmacie & Biotech"), ("biotechnology", "Pharmacie & Biotech"),
    ("medical", "Santé — services & équipements"), ("aerospace", "Aéronautique & Défense"), ("auto", "Automobile & équipementiers"),
    ("oil & gas", "Énergie (pétrole & gaz)"), ("utilities", "Services aux collectivités (utilities)"),
    ("specialty chemicals", "Chimie"), ("chemicals", "Chimie"), ("banks", "Banques (non applicable DCF FCFF)"),
    ("insurance", "Banques (non applicable DCF FCFF)"), ("credit services", "Fintech & Paiements"),
    ("information technology services", "Services aux entreprises / Conseil"), ("telecom", "Télécoms"),
    ("engineering & construction", "BTP & Ingénierie"), ("building", "BTP & Ingénierie"), ("discount stores", "Distribution / Retail"),
    ("grocery", "Distribution / Retail"), ("restaurants", "Hôtellerie & Restauration"), ("consumer electronics", "Tech hardware & électronique"),
    ("electrical equipment", "Industrie diversifiée"), ("farm & heavy", "Industrie diversifiée"), ("conglomerates", "Industrie diversifiée"),
    ("metals & mining", "Matériaux & Mines"), ("industrial metals", "Matériaux & Mines"), ("steel", "Matériaux & Mines"),
    ("communication services", "Télécoms"), ("technology", "Tech hardware & électronique"),
]


def map_sector(industry: str | None, sector: str | None) -> str:
    txt = f"{industry or ''} | {sector or ''}".lower()
    for kw, s in SECTOR_MAP:
        if kw in txt:
            return s
    return "Marché total (moyenne)"


def is_financial(industry: str | None, sector: str | None) -> bool:
    return any(k in f"{industry} {sector}".lower() for k in ("bank", "insurance", "financial services", "capital markets"))


def statements_to_df(inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les 3 états Yahoo (colonnes = dates de clôture) en DataFrame ValueScope (millions)."""
    frames = [f for f in (inc, bs, cf) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="first")]
    years = {}
    for c in raw.columns:
        y = c.year if hasattr(c, "year") else int(str(c)[:4])
        years.setdefault(y, c)            # une colonne par exercice
    out = {}
    for key, labels in YF_MAP.items():
        for lab in labels:
            if lab in raw.index and raw.loc[lab].notna().any():
                s = pd.to_numeric(raw.loc[lab], errors="coerce")
                out[key] = {y: s[c] for y, c in years.items()}
                break
    df = pd.DataFrame(out).T
    df = df[sorted(df.columns)] / 1e6
    df = df.dropna(axis=1, thresh=5)       # supprime les exercices quasi vides
    return df


def _yf_company(ticker: str):
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    if not (info.get("currentPrice") or info.get("regularMarketPrice") or info.get("longName")):
        raise RuntimeError(f"yfinance : informations vides pour {ticker} (limitation probable)")
    df = statements_to_df(t.income_stmt, t.balance_sheet, t.cashflow)
    try:
        hist = t.history(period="5y", interval="1wk", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()
    return info, df, hist


def fetch_company(ticker: str) -> dict:
    """Infos + états financiers. Chaîne de sources : yfinance → API chart Yahoo (cours) + SEC EDGAR (comptes US)."""
    from . import net, fallback as FB
    errors = []
    info, df, hist = {}, pd.DataFrame(), pd.DataFrame()
    try:
        info, df, hist = _yf_company(ticker)
        net.record("Yahoo (yfinance)", True, "OK")
    except Exception as e:
        errors.append(f"yfinance : {e}")
        net.record("Yahoo (yfinance)", False, str(e))
    if not info or hist is None or hist.empty:
        try:
            ci, s = FB.info_from_chart(ticker)
            info = {**ci, **{k: v for k, v in info.items() if v is not None}}
            hist = pd.DataFrame({"Close": s})
        except Exception as e:
            errors.append(f"API chart : {e}")
    if (df.empty or "revenue" not in df.index) and FB.is_us_ticker(ticker):
        try:
            df, nm = FB.sec_financials(ticker)
            info.setdefault("longName", nm)
            info.setdefault("financialCurrency", "USD")
            info["_fin_source"] = "SEC EDGAR"
        except Exception as e:
            errors.append(f"SEC : {e}")
    if not info and df.empty:
        raise RuntimeError(" | ".join(errors) or f"Aucune donnée pour {ticker}")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or \
        (float(hist["Close"].dropna().iloc[-1]) if hist is not None and not hist.empty else None)
    cur, fin_cur = info.get("currency"), info.get("financialCurrency") or info.get("currency")
    if cur in ("GBp", "GBX", "ZAc", "ILA"):          # cotations en centimes
        price = price / 100 if price else price
        cur = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}[cur]
    fx = 1.0
    if price and cur and fin_cur and cur != fin_cur:  # ex. cours en HKD, comptes en CNY
        try:
            s, _ = FB.chart(f"{cur}{fin_cur}=X", "5d", "1d")
            fx = float(s.iloc[-1]) if len(s) else 1.0
        except Exception:
            fx = 1.0
    shares = info.get("sharesOutstanding")
    if (not shares) and "shares" in df.index and df.loc["shares"].notna().any():
        shares = df.loc["shares"].dropna().iloc[-1] * 1e6
    if shares and price and not info.get("marketCap"):
        info["marketCap"] = shares * price
    return dict(info=info, df=df, hist=hist, price=price, price_fin=price * fx if price else None, currency=cur,
                fin_currency=fin_cur, shares_m=(shares / 1e6) if shares else None, fx=fx, errors=errors)


def row_from_info(tk: str, i: dict) -> dict:
    g = i.get
    return {
        "Ticker": tk, "Société": ALL_TICKERS.get(tk, g("shortName") or tk), "Pays": g("country"),
        "Secteur": g("sector"), "Industrie": g("industry"), "Devise": g("currency"),
        "Devise comptes": g("financialCurrency") or g("currency"),
        "Cours": g("currentPrice") or g("regularMarketPrice"),
        "Capitalisation (Md)": (g("marketCap") or np.nan) / 1e9,
        "Actions (M)": (g("sharesOutstanding") or np.nan) / 1e6,
        "PER": g("trailingPE"), "PER prévisionnel": g("forwardPE"), "EV/EBITDA": g("enterpriseToEbitda"),
        "EV/CA": g("enterpriseToRevenue"), "P/B": g("priceToBook"),
        "Marge opé. %": _pct(g("operatingMargins")), "Marge nette %": _pct(g("profitMargins")),
        "ROE %": _pct(g("returnOnEquity")), "Croissance CA %": _pct(g("revenueGrowth")),
        "Dette/CP %": g("debtToEquity"), "Rendement div. %": _div_yield(g("dividendYield")),
        "Perf. 52 sem. %": _pct(g("52WeekChange")), "Bêta": g("beta"),
        "Objectif analystes": g("targetMeanPrice"), "Nb analystes": g("numberOfAnalystOpinions"),
        "Reco.": g("recommendationKey"),
    }


def info_from_row(r: dict) -> dict:
    """Reconstruit un dictionnaire « info » minimal à partir d'une ligne de la base CSV (mode hors-ligne)."""
    v = lambda k: None if r.get(k) is None or (isinstance(r.get(k), float) and r.get(k) != r.get(k)) else r.get(k)
    mc = v("Capitalisation (Md)")
    return {"longName": v("Société"), "country": v("Pays"), "sector": v("Secteur"), "industry": v("Industrie"),
            "currency": v("Devise"), "financialCurrency": v("Devise comptes") or v("Devise"),
            "currentPrice": v("Cours"), "marketCap": mc * 1e9 if mc else None, "trailingPE": v("PER"),
            "forwardPE": v("PER prévisionnel"), "enterpriseToEbitda": v("EV/EBITDA"),
            "targetMeanPrice": v("Objectif analystes"), "numberOfAnalystOpinions": v("Nb analystes"),
            "recommendationKey": v("Reco.")}


def fetch_screener(tickers: list[str]) -> pd.DataFrame:
    """Indicateurs clés. yfinance d'abord ; en cas d'échec, cours et variation via l'API chart (autres champs vides)."""
    from . import net, fallback as FB
    import time
    try:
        import yfinance as yf
    except Exception as e:
        yf = None
        net.record("Yahoo (yfinance)", False, f"Module absent : {e}")
    rows, fails = [], 0
    for tk in tickers:
        i = {}
        if yf is not None and fails < 3:          # après 3 échecs d'affilée, on cesse de solliciter yfinance
            try:
                i = yf.Ticker(tk).info or {}
                if i.get("currentPrice") or i.get("regularMarketPrice"):
                    fails = 0
                    net.record("Yahoo (yfinance)", True, "OK")
                else:
                    raise RuntimeError("réponse vide")
            except Exception as e:
                fails += 1
                i = {}
                net.record("Yahoo (yfinance)", False, str(e))
        if not (i.get("currentPrice") or i.get("regularMarketPrice")):
            try:
                ci, _ = FB.info_from_chart(tk)
                i = {**ci, **{k: v for k, v in i.items() if v is not None}}
            except Exception:
                pass
            time.sleep(0.2)
        rows.append(row_from_info(tk, i))
    return pd.DataFrame(rows)


def _pct(v):
    return v * 100 if isinstance(v, (int, float)) else np.nan


def _div_yield(v):
    if not isinstance(v, (int, float)):
        return np.nan
    return v if v > 1 else v * 100       # yfinance a changé d'échelle selon les versions
