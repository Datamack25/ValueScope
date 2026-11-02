"""Modèle de données standardisé + dictionnaire de libellés FR (PCG / liasse) & US GAAP / IFRS."""
from __future__ import annotations
import re
import unicodedata

# Clé interne -> (libellé FR affiché, section)
FIELDS = {
    # Compte de résultat
    "revenue": ("Chiffre d'affaires", "CR"),
    "cogs": ("Coût des ventes / achats consommés", "CR"),
    "gross_profit": ("Marge brute", "CR"),
    "personnel": ("Charges de personnel", "CR"),
    "ebitda": ("EBITDA / EBE", "CR"),
    "da": ("Dotations aux amortissements (D&A)", "CR"),
    "ebit": ("Résultat d'exploitation (EBIT)", "CR"),
    "rnd": ("Recherche & développement", "CR"),
    "interest_expense": ("Charges financières (intérêts)", "CR"),
    "pretax_income": ("Résultat avant impôt", "CR"),
    "tax": ("Impôt sur les sociétés", "CR"),
    "net_income": ("Résultat net", "CR"),
    # Bilan actif
    "cash": ("Trésorerie & équivalents", "BIL"),
    "receivables": ("Créances clients", "BIL"),
    "inventory": ("Stocks", "BIL"),
    "current_assets": ("Actif circulant", "BIL"),
    "ppe": ("Immobilisations corporelles", "BIL"),
    "intangibles": ("Immobilisations incorporelles / goodwill", "BIL"),
    "total_assets": ("Total actif", "BIL"),
    # Bilan passif
    "payables": ("Dettes fournisseurs", "BIL"),
    "current_liabilities": ("Passif circulant", "BIL"),
    "short_debt": ("Dettes financières CT", "BIL"),
    "long_debt": ("Dettes financières LT", "BIL"),
    "lease_liabilities": ("Dettes locatives (IFRS 16)", "BIL"),
    "total_debt": ("Dette financière totale", "BIL"),
    "minority": ("Intérêts minoritaires", "BIL"),
    "equity": ("Capitaux propres", "BIL"),
    "retained_earnings": ("Réserves / résultats non distribués", "BIL"),
    # Flux
    "cfo": ("Flux de trésorerie d'exploitation", "TFT"),
    "capex": ("Investissements (Capex)", "TFT"),
    "dividends": ("Dividendes versés", "TFT"),
    "buybacks": ("Rachats d'actions", "TFT"),
    # Marché
    "shares": ("Nombre d'actions (millions)", "MKT"),
}

SECTIONS = {"CR": "Compte de résultat", "BIL": "Bilan", "TFT": "Flux de trésorerie", "MKT": "Marché"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Synonymes (déjà normalisés à l'usage). L'ordre compte : les plus spécifiques d'abord.
SYNONYMS = [
    ("gross_profit", ["marge brute", "gross profit", "gross margin"]),
    ("cogs", ["cout des ventes", "cout des marchandises vendues", "achats consommes", "achats de marchandises",
              "cost of revenue", "cost of revenues", "cost of goods sold", "cost of sales", "cogs"]),
    ("revenue", ["chiffre d affaires net", "chiffre d affaires", "ca net", "ventes nettes", "produits des activites ordinaires",
                 "total revenue", "total revenues", "net revenue", "net revenues", "net sales", "revenue", "revenues", "sales", "turnover"]),
    ("personnel", ["charges de personnel", "salaires et traitements", "frais de personnel", "personnel expenses", "staff costs"]),
    ("ebitda", ["excedent brut d exploitation", "ebe", "ebitda"]),
    ("da", ["dotations aux amortissements", "dotations aux amortissements et provisions", "amortissements",
            "depreciation and amortization", "depreciation amortization", "depreciation"]),
    ("rnd", ["recherche et developpement", "frais de recherche", "research and development", "r d"]),
    ("ebit", ["resultat d exploitation", "resultat operationnel courant", "resultat operationnel", "operating income",
              "operating profit", "income from operations", "ebit"]),
    ("interest_expense", ["charges d interets", "interets et charges assimilees", "charges financieres", "cout de l endettement financier",
                          "interest expense", "interest expenses", "finance costs"]),
    ("pretax_income", ["resultat avant impot", "resultat courant avant impots", "income before income taxes",
                       "income before taxes", "pretax income", "pre tax income", "profit before tax", "earnings before taxes"]),
    ("tax", ["impots sur les benefices", "impot sur les societes", "impot sur le resultat", "charge d impot",
             "provision for income taxes", "income tax expense", "income taxes", "tax expense"]),
    ("net_income", ["resultat net part du groupe", "resultat net de l exercice", "resultat net", "benefice ou perte", "benefice net",
                    "net income attributable", "net income", "net earnings", "profit for the year", "net profit"]),
    ("cash", ["tresorerie et equivalents de tresorerie", "disponibilites", "tresorerie", "cash and cash equivalents",
              "cash and equivalents", "cash"]),
    ("receivables", ["creances clients et comptes rattaches", "creances clients", "clients et comptes rattaches",
                     "accounts receivable", "trade receivables", "receivables"]),
    ("inventory", ["stocks et en cours", "stocks", "inventories", "inventory"]),
    ("current_assets", ["total actif circulant", "actif circulant", "actifs courants", "total current assets", "current assets"]),
    ("ppe", ["immobilisations corporelles", "property plant and equipment", "property and equipment", "ppe", "pp e"]),
    ("intangibles", ["immobilisations incorporelles", "ecarts d acquisition", "goodwill", "intangible assets"]),
    ("total_assets", ["total de l actif", "total actif", "total bilan", "total assets"]),
    ("payables", ["dettes fournisseurs et comptes rattaches", "dettes fournisseurs", "fournisseurs", "accounts payable",
                  "trade payables", "payables"]),
    ("current_liabilities", ["total passif circulant", "passif circulant", "passifs courants", "total current liabilities",
                             "current liabilities"]),
    ("lease_liabilities", ["dettes locatives", "obligations locatives", "lease liabilities", "operating lease liabilities"]),
    ("short_debt", ["dettes financieres a court terme", "emprunts a court terme", "concours bancaires courants",
                    "short term debt", "short term borrowings", "current portion of long term debt", "current debt"]),
    ("long_debt", ["emprunts et dettes aupres des etablissements de credit", "dettes financieres a long terme",
                   "emprunts obligataires", "emprunts et dettes financieres", "long term debt", "long term borrowings", "non current debt"]),
    ("total_debt", ["dette financiere totale", "endettement financier brut", "dettes financieres", "total debt"]),
    ("minority", ["interets minoritaires", "participations ne donnant pas le controle", "noncontrolling interest",
                  "non controlling interests", "minority interest"]),
    ("retained_earnings", ["report a nouveau", "reserves", "retained earnings"]),
    ("equity", ["total capitaux propres", "capitaux propres part du groupe", "capitaux propres",
                "total stockholders equity", "total shareholders equity", "stockholders equity", "shareholders equity", "total equity", "equity"]),
    ("cfo", ["flux net de tresorerie genere par l activite", "flux de tresorerie lies a l activite", "flux de tresorerie d exploitation",
             "net cash provided by operating activities", "cash flow from operations", "operating cash flow", "cash from operations"]),
    ("capex", ["acquisitions d immobilisations corporelles et incorporelles", "acquisitions d immobilisations", "investissements corporels",
               "capital expenditures", "capital expenditure", "purchases of property and equipment", "capex"]),
    ("dividends", ["dividendes verses", "dividendes", "dividends paid", "dividends"]),
    ("buybacks", ["rachats d actions propres", "rachat d actions", "repurchases of common stock", "share buybacks", "buybacks"]),
    ("shares", ["nombre d actions", "actions en circulation", "shares outstanding", "diluted shares", "weighted average shares", "shares"]),
]


def match_label(label: str) -> str | None:
    n = _norm(label)
    if not n or len(n) < 2:
        return None
    if n in FIELDS:
        return n
    # correspondance exacte puis "commence par" puis "contient"
    for key, syns in SYNONYMS:
        if n in syns:
            return key
    for key, syns in SYNONYMS:
        for s in syns:
            if n.startswith(s) and len(s) >= 4:
                return key
    for key, syns in SYNONYMS:
        for s in syns:
            if len(s) >= 6 and f" {s} " in f" {n} ":
                return key
    return None


def parse_number(x) -> float | None:
    """Gère '1 234,5', '1,234.5', '(123)', '-', '12 %', '1.234.567', 'k€'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x else None
    s = str(x).strip().replace(" ", " ").replace(" ", " ")
    if s in ("", "-", "—", "–", "n/a", "na", "nm", "nan", "None"):
        return None
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("−")
    s = re.sub(r"[^0-9,\.]", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if (len(parts) > 2 or len(parts[-1]) == 3) else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v
