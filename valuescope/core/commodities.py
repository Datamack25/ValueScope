"""Matières premières : contrats à terme (Yahoo Finance) et ETF proxies pour les métaux de la transition."""
from __future__ import annotations
import numpy as np
import pandas as pd

# ticker : (nom, catégorie, unité de cotation, usages, producteurs principaux, facteurs de prix)
COMMODITIES = {
    "BZ=F": ("Pétrole Brent", "Énergie", "USD / baril",
             "Référence mondiale du brut : carburants, pétrochimie, plastiques, transport.",
             "États-Unis, Arabie saoudite, Russie, Canada, Irak, Chine",
             "Quotas OPEP+, demande chinoise, stocks américains, géopolitique (Moyen-Orient, Russie), dollar."),
    "CL=F": ("Pétrole WTI", "Énergie", "USD / baril",
             "Brut de référence américain (livré à Cushing, Oklahoma).",
             "États-Unis (schiste)", "Production de schiste, stocks EIA, capacité de raffinage, saison de conduite."),
    "NG=F": ("Gaz naturel (Henry Hub)", "Énergie", "USD / MMBtu",
             "Production d'électricité, chauffage, engrais, industrie ; exporté sous forme de GNL.",
             "États-Unis, Russie, Iran, Qatar, Canada", "Météo, stocks, exportations de GNL, production de schiste."),
    "HO=F": ("Fioul / Diesel (ULSD)", "Énergie", "USD / gallon",
             "Transport routier, maritime, agriculture, chauffage.", "Raffineries mondiales",
             "Marges de raffinage, prix du brut, demande industrielle."),
    "RB=F": ("Essence (RBOB)", "Énergie", "USD / gallon", "Carburant automobile.", "Raffineries américaines",
             "Saison de conduite estivale, raffinage, brut."),
    "GC=F": ("Or", "Métaux précieux", "USD / once troy",
             "Valeur refuge, réserves des banques centrales, joaillerie, électronique.",
             "Chine, Australie, Russie, Canada, États-Unis, Ghana",
             "Taux réels, dollar, achats des banques centrales, aversion au risque, inflation."),
    "SI=F": ("Argent", "Métaux précieux", "USD / once troy",
             "Photovoltaïque, électronique, joaillerie, investissement.", "Mexique, Chine, Pérou",
             "Demande solaire et industrielle, comportement de l'or, investisseurs."),
    "PL=F": ("Platine", "Métaux précieux", "USD / once troy", "Pots catalytiques (diesel), joaillerie, hydrogène.",
             "Afrique du Sud, Russie, Zimbabwe", "Industrie automobile, offre sud-africaine (électricité, mines)."),
    "PA=F": ("Palladium", "Métaux précieux", "USD / once troy", "Pots catalytiques essence, électronique.",
             "Russie, Afrique du Sud", "Ventes de véhicules thermiques, électrification, offre russe."),
    "HG=F": ("Cuivre", "Métaux industriels", "USD / livre",
             "Réseaux électriques, bâtiment, véhicules électriques, électronique — baromètre de l'économie (« Dr Copper »).",
             "Chili, Pérou, RD Congo, Chine", "Croissance chinoise, électrification, stocks LME/COMEX, grèves minières."),
    "ALI=F": ("Aluminium", "Métaux industriels", "USD / tonne",
             "Transport, emballage, construction, câbles.", "Chine, Inde, Russie, Canada",
             "Coût de l'énergie (fonderie très électro-intensive), politique industrielle chinoise."),
    "LIT": ("Lithium & batteries (ETF proxy)", "Métaux de la transition", "USD / part d'ETF",
            "Batteries lithium-ion (véhicules électriques, stockage).", "Australie, Chili, Chine, Argentine",
            "Ventes de VE, capacités de raffinage chinoises, nouveaux projets miniers."),
    "URA": ("Uranium (ETF proxy)", "Métaux de la transition", "USD / part d'ETF",
            "Combustible des centrales nucléaires.", "Kazakhstan, Canada, Namibie, Australie",
            "Relance du nucléaire, contrats long terme, géopolitique (Russie, Niger)."),
    "REMX": ("Terres rares & métaux stratégiques (ETF proxy)", "Métaux de la transition", "USD / part d'ETF",
             "Aimants permanents (moteurs, éoliennes), défense, électronique.", "Chine (largement dominante), Australie, États-Unis",
             "Quotas et restrictions d'export chinoises, demande VE/éolien, défense."),
    "ZW=F": ("Blé", "Agricoles", "US cents / boisseau", "Alimentation humaine (pain, pâtes), alimentation animale.",
             "Chine, Inde, Russie, UE (France), États-Unis, Ukraine", "Météo, récoltes mer Noire, stocks mondiaux, politiques d'export."),
    "ZC=F": ("Maïs", "Agricoles", "US cents / boisseau", "Alimentation animale, éthanol, amidon.",
             "États-Unis, Chine, Brésil, Argentine, Ukraine", "Météo US/Brésil, mandats éthanol, demande chinoise."),
    "ZS=F": ("Soja", "Agricoles", "US cents / boisseau", "Tourteaux (élevage), huile alimentaire, biodiesel.",
             "Brésil, États-Unis, Argentine", "Demande chinoise, récoltes Amérique du Sud, biocarburants."),
    "KC=F": ("Café Arabica", "Agricoles", "US cents / livre", "Boisson.", "Brésil, Vietnam (robusta), Colombie, Éthiopie",
             "Météo brésilienne (gel, sécheresse), stocks certifiés, réal brésilien."),
    "CC=F": ("Cacao", "Agricoles", "USD / tonne", "Chocolat, confiserie, cosmétique.", "Côte d'Ivoire, Ghana, Équateur, Cameroun",
             "Météo et maladies en Afrique de l'Ouest, prix bord champ réglementés, broyages."),
    "SB=F": ("Sucre n°11", "Agricoles", "US cents / livre", "Alimentation, éthanol.", "Brésil, Inde, Thaïlande",
             "Arbitrage sucre/éthanol au Brésil, mousson indienne, exportations."),
    "CT=F": ("Coton", "Agricoles", "US cents / livre", "Textile.", "Inde, Chine, États-Unis, Brésil", "Demande textile, météo, stocks chinois."),
    "LE=F": ("Bovins vivants", "Élevage", "US cents / livre", "Viande bovine.", "États-Unis, Brésil, UE", "Cycle du cheptel, coût de l'alimentation animale."),
    "DX-Y.NYB": ("Indice dollar (DXY) — référence", "Référence marché", "Points",
                 "La plupart des matières premières sont cotées en dollars : un dollar fort pèse souvent sur leurs prix.",
                 "—", "Politique de la Fed, différentiels de taux, aversion au risque."),
}
CATEGORIES = ["Énergie", "Métaux précieux", "Métaux industriels", "Métaux de la transition", "Agricoles", "Élevage", "Référence marché"]


# Repli FRED (Réserve fédérale de St. Louis, CSV public sans clé) — séries quotidiennes ou mensuelles (prix FMI)
FRED = {
    "BZ=F": ("DCOILBRENTEU", "USD / baril (quotidien, FRED)"),
    "CL=F": ("DCOILWTICO", "USD / baril (quotidien, FRED)"),
    "NG=F": ("DHHNGSP", "USD / MMBtu (quotidien, FRED)"),
    "HO=F": ("DHOILNYH", "USD / gallon (quotidien, FRED)"),
    "RB=F": ("DGASNYH", "USD / gallon (quotidien, FRED)"),
    "HG=F": ("PCOPPUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "ALI=F": ("PALUMUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "ZW=F": ("PWHEAMTUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "ZC=F": ("PMAIZMTUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "ZS=F": ("PSOYBUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "KC=F": ("PCOFFOTMUSDM", "US cents / livre (mensuel, FMI via FRED)"),
    "CC=F": ("PCOCOUSDM", "USD / tonne (mensuel, FMI via FRED)"),
    "SB=F": ("PSUGAISAUSDM", "US cents / livre (mensuel, FMI via FRED)"),
    "CT=F": ("PCOTTINDUSDM", "US cents / livre (mensuel, FMI via FRED)"),
    "LE=F": ("PBEEFUSDM", "US cents / livre (mensuel, FMI via FRED)"),
    "URA": ("PURANUSDM", "USD / livre U3O8 (mensuel, FMI via FRED)"),
    "DX-Y.NYB": ("DTWEXBGS", "Indice dollar large (quotidien, FRED)"),
}


def fetch_fred(series: str, years: int = 10) -> pd.Series:
    import io
    from . import net
    raw = net.get("https://fred.stlouisfed.org/graph/fredgraph.csv", "FRED (St. Louis Fed)",
                  params={"id": series}, expect_json=False, retries=2)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "v"]
    s = pd.Series(pd.to_numeric(df["v"], errors="coerce").values, index=pd.to_datetime(df["date"])).dropna()
    return s[s.index >= s.index.max() - pd.Timedelta(days=365 * years + 30)]


def periods_per_year(s: pd.Series) -> int:
    gap = pd.Series(s.dropna().index).diff().dt.days.median() if len(s.dropna()) > 3 else 1
    return 252 if gap <= 3 else 52 if gap <= 8 else 12


def fetch_prices(tickers: list[str], period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """yfinance (téléchargement groupé) puis, pour les séries manquantes, API chart de Yahoo."""
    from . import net, fallback as FB
    close = pd.DataFrame()
    try:
        import yfinance as yf
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
        if data is not None and not data.empty:
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(columns={"Close": tickers[0]})
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.dropna(how="all", axis=1).dropna(how="all")
            net.record("Yahoo (yfinance)", True, f"{close.shape[1]} séries")
    except Exception as e:
        net.record("Yahoo (yfinance)", False, str(e))
    missing = [t for t in tickers if t not in close.columns or close[t].notna().sum() < 50]
    got = {}
    for t in missing:
        try:
            s, _ = FB.chart(t, period, interval)
            if len(s) > 50:
                got[t] = s
        except Exception:
            continue
    fred_used = {}
    for t in [t for t in missing if t not in got and t in FRED]:
        try:
            s = fetch_fred(FRED[t][0])
            if len(s) > 24:
                got[t] = s
                fred_used[t] = FRED[t][1]
        except Exception:
            continue
    if got:
        extra = pd.DataFrame(got)
        close = extra if close.empty else close.drop(columns=[c for c in got if c in close.columns]).join(extra, how="outer")
    close.attrs["fred"] = fred_used
    if close.empty:
        raise RuntimeError("Yahoo Finance injoignable (yfinance et API chart) : " + net.LOG.get("Yahoo (API chart)", (0, "", ""))[1])
    res = close.sort_index().dropna(how="all")
    res.attrs["fred"] = fred_used
    return res


def perf_table(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tk in px.columns:
        s = px[tk].dropna()
        s = s[s > 0]
        if len(s) < 20:
            continue
        last, d = s.iloc[-1], s.index[-1]

        def ret(days=None, start=None):
            ref = s[s.index <= (start if start is not None else d - pd.Timedelta(days=days))]
            return (last / ref.iloc[-1] - 1) * 100 if len(ref) else np.nan

        r1y = s[s.index >= d - pd.Timedelta(days=365)]
        lr = np.log(s).diff().dropna()
        vol = lr[lr.index >= d - pd.Timedelta(days=365)].std() * np.sqrt(periods_per_year(s)) * 100
        dd = (s / s.cummax() - 1).iloc[-1] * 100
        info = COMMODITIES.get(tk, (tk, "", "", "", "", ""))
        unit = px.attrs.get("fred", {}).get(tk, info[2])
        rows.append({"Ticker": tk, "Matière première": info[0], "Catégorie": info[1], "Unité": unit, "Dernier": last,
                     "1 sem. %": ret(7), "1 mois %": ret(30), "YTD %": ret(start=pd.Timestamp(d.year - 1, 12, 31)),
                     "1 an %": ret(365), "5 ans %": ret(365 * 5), "Plus haut 52s": r1y.max(), "Plus bas 52s": r1y.min(),
                     "Volatilité 1 an %": vol, "Écart au plus haut %": dd, "Date": d.date()})
    return pd.DataFrame(rows)


def signals(s: pd.Series) -> list[tuple[str, str]]:
    """Lecture technique simple : tendance (MM50/200), position dans la fourchette 52 sem., momentum."""
    s = s.dropna()
    out = []
    if len(s) < 210:
        return out
    ma50, ma200 = s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
    last = s.iloc[-1]
    if last > ma50 > ma200:
        out.append(("ok", "Tendance haussière : cours > MM50 > MM200."))
    elif last < ma50 < ma200:
        out.append(("bad", "Tendance baissière : cours < MM50 < MM200."))
    else:
        out.append(("warn", "Tendance mixte : moyennes mobiles non alignées."))
    y = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
    pos = (last - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.nan
    if pos == pos:
        out.append(("info", f"Position dans la fourchette 52 semaines : {pos*100:.0f} % (0 % = plus bas, 100 % = plus haut)."))
    z = (last - s.iloc[-252 * 5:].mean()) / s.iloc[-252 * 5:].std()
    out.append(("warn" if abs(z) > 1.5 else "info", f"Écart à la moyenne 5 ans : {z:+.1f} écart-type(s)."))
    return out
