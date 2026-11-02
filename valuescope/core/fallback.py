"""Sources de repli quand yfinance échoue (limitation de débit, blocage) :
  • API « chart » publique de Yahoo Finance (cours, devise, nom, historique) — sans jeton ;
  • SEC EDGAR « companyfacts » (états financiers XBRL officiels des sociétés américaines, gratuits et sans clé)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from . import net

CHART_HOSTS = ["https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"]
SEC_HEADERS = {"User-Agent": "ValueScope research tool admin@valuescope.app", "Accept": "application/json"}


# ====================================================================== YAHOO CHART
def chart(symbol: str, range_: str = "10y", interval: str = "1d") -> tuple[pd.Series, dict]:
    last_err = None
    for host in CHART_HOSTS:
        try:
            js = net.get(f"{host}/v8/finance/chart/{symbol}", "Yahoo (API chart)",
                         params={"range": range_, "interval": interval, "includeAdjustedClose": "true", "events": "div,splits"},
                         retries=2)
            res = (js.get("chart") or {}).get("result") or []
            if not res:
                raise net.SourceError(f"Yahoo chart : aucun résultat pour {symbol}")
            r = res[0]
            ts = r.get("timestamp") or []
            ind = r.get("indicators") or {}
            adj = (ind.get("adjclose") or [{}])[0].get("adjclose")
            close = adj or (ind.get("quote") or [{}])[0].get("close") or []
            s = pd.Series(close, index=pd.to_datetime(ts, unit="s"), dtype=float).dropna()
            s.index = s.index.tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")]
            return s, r.get("meta") or {}
        except Exception as e:
            last_err = e
    raise net.SourceError(str(last_err))


def info_from_chart(symbol: str) -> tuple[dict, pd.Series]:
    """Informations minimales (nom, devise, cours, variation 52 semaines) à partir de l'API chart."""
    s, m = chart(symbol, "5y", "1wk")
    price = m.get("regularMarketPrice") or (float(s.iloc[-1]) if len(s) else None)
    y1 = s[s.index >= s.index[-1] - pd.Timedelta(days=365)] if len(s) else s
    info = {"longName": m.get("longName") or m.get("shortName") or symbol, "shortName": m.get("shortName"),
            "currency": m.get("currency"), "financialCurrency": m.get("currency"), "currentPrice": price,
            "52WeekChange": (price / float(y1.iloc[0]) - 1) if price and len(y1) else None, "_source": "Yahoo chart"}
    return info, s


# ====================================================================== SEC EDGAR
_CIK: dict[str, int] = {}

SEC_MAP = {  # clé ValueScope -> balises us-gaap par priorité ; "I" = valeur de bilan (instant)
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                 "RevenueFromContractWithCustomerIncludingAssessedTax"], "D"),
    "cogs": (["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "D"),
    "gross_profit": (["GrossProfit"], "D"),
    "rnd": (["ResearchAndDevelopmentExpense"], "D"),
    "da": (["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"], "D"),
    "ebit": (["OperatingIncomeLoss"], "D"),
    "interest_expense": (["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"], "D"),
    "pretax_income": (["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"], "D"),
    "tax": (["IncomeTaxExpenseBenefit"], "D"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "D"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "I"),
    "receivables": (["AccountsReceivableNetCurrent"], "I"),
    "inventory": (["InventoryNet"], "I"),
    "current_assets": (["AssetsCurrent"], "I"),
    "ppe": (["PropertyPlantAndEquipmentNet"], "I"),
    "intangibles": (["Goodwill"], "I"),
    "total_assets": (["Assets"], "I"),
    "payables": (["AccountsPayableCurrent"], "I"),
    "current_liabilities": (["LiabilitiesCurrent"], "I"),
    "short_debt": (["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"], "I"),
    "long_debt": (["LongTermDebtNoncurrent", "LongTermDebt"], "I"),
    "lease_liabilities": (["OperatingLeaseLiabilityNoncurrent"], "I"),
    "minority": (["MinorityInterest"], "I"),
    "equity": (["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "I"),
    "retained_earnings": (["RetainedEarningsAccumulatedDeficit"], "I"),
    "cfo": (["NetCashProvidedByUsedInOperatingActivities"], "D"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "D"),
    "dividends": (["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "D"),
    "buybacks": (["PaymentsForRepurchaseOfCommonStock"], "D"),
}


def is_us_ticker(tk: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", tk))


def _cik(ticker: str) -> int:
    if not _CIK:
        js = net.get("https://www.sec.gov/files/company_tickers.json", "SEC EDGAR", headers=SEC_HEADERS)
        for v in js.values():
            _CIK[str(v["ticker"]).upper()] = int(v["cik_str"])
    t = ticker.upper().replace("-", ".")
    if t not in _CIK and ticker.upper() not in _CIK:
        raise net.SourceError(f"SEC : ticker {ticker} introuvable")
    return _CIK.get(t) or _CIK[ticker.upper()]


def sec_financials(ticker: str, years: int = 5) -> tuple[pd.DataFrame, str]:
    """États financiers annuels (millions USD) depuis les données XBRL officielles de la SEC."""
    cik = _cik(ticker)
    js = net.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", "SEC EDGAR", headers=SEC_HEADERS)
    gaap = (js.get("facts") or {}).get("us-gaap") or {}
    out = {}
    for key, (tags, kind) in SEC_MAP.items():
        for tag in tags:
            units = (gaap.get(tag) or {}).get("units") or {}
            vals = units.get("USD")
            if not vals:
                continue
            pat = re.compile(r"^CY(\d{4})Q4I$" if kind == "I" else r"^CY(\d{4})$")
            ser = {}
            for v in vals:
                m = pat.match(v.get("frame") or "")
                if m:
                    ser[int(m.group(1))] = v["val"] / 1e6
            if ser:
                out[key] = ser
                break
    if not out or "revenue" not in out:
        raise net.SourceError(f"SEC : données XBRL insuffisantes pour {ticker}")
    yrs = sorted(out["revenue"])[-years:]
    df = pd.DataFrame(out).T.reindex(columns=yrs)
    # actions en circulation (dei) : valeur publiée la plus proche après la clôture de chaque exercice
    dei = (js.get("facts") or {}).get("dei") or {}
    sh = ((dei.get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}).get("shares") or []
    if sh:
        pts = sorted((v["end"], v["val"] / 1e6) for v in sh)
        row = {}
        for y in yrs:
            nxt = [val for end, val in pts if end[:4] == str(y + 1)]
            same = [val for end, val in pts if end[:4] == str(y)]
            row[y] = nxt[0] if nxt else (same[-1] if same else np.nan)
        if np.isnan(row[yrs[-1]]):
            row[yrs[-1]] = pts[-1][1]
        df.loc["shares"] = pd.Series(row)
    if "total_debt" not in df.index:
        df.loc["total_debt"] = df.reindex(["short_debt", "long_debt"]).sum(min_count=1)
    return df, js.get("entityName") or ticker
