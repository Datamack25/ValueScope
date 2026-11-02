"""Base de données locale en CSV (dossier data/store/).

Fichiers :
  companies_market.csv      instantané des indicateurs de marché par société (dernière valeur par ticker)
  companies_financials.csv  états financiers annuels, format long (ticker, année, poste, valeur)
  countries_imf.csv         séries FMI WEO, format long (indicateur, pays ISO3, année, valeur)
  countries_wb.csv          séries Banque mondiale, format long (pays, indicateur, année, valeur)
  commodities_prices.csv    cours de clôture quotidiens (date × ticker)
  analyses_log.csv          journal des valorisations réalisées dans la plateforme

Sur Streamlit Community Cloud, le disque est réinitialisé au redémarrage : exportez la base (zip)
et versionnez les CSV dans le dépôt GitHub pour les conserver durablement."""
from __future__ import annotations
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "store"
FILES = {
    "companies_market": "companies_market.csv",
    "companies_financials": "companies_financials.csv",
    "countries_imf": "countries_imf.csv",
    "countries_wb": "countries_wb.csv",
    "commodities_prices": "commodities_prices.csv",
    "analyses_log": "analyses_log.csv",
}
LABELS = {
    "companies_market": "Indicateurs de marché des sociétés",
    "companies_financials": "États financiers des sociétés",
    "countries_imf": "Séries macro FMI",
    "countries_wb": "Séries Banque mondiale",
    "commodities_prices": "Cours des matières premières",
    "analyses_log": "Journal des analyses",
}


def _path(key: str) -> Path:
    return ROOT / FILES[key]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _read(key: str, **kw) -> pd.DataFrame:
    p = _path(key)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kw)
    except Exception:
        return pd.DataFrame()


def _write(key: str, df: pd.DataFrame, index=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    p = _path(key)
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, p)


# ------------------------------------------------------------------ sociétés
def save_screener(df: pd.DataFrame):
    if df is None or df.empty:
        return
    new = df.copy()
    new = new[new["Cours"].notna()] if "Cours" in new else new
    if new.empty:
        return
    new["Mis à jour"] = _now()
    old = _read("companies_market")
    out = pd.concat([old[~old["Ticker"].isin(new["Ticker"])], new]) if not old.empty else new
    _write("companies_market", out)


def load_screener(tickers: list[str] | None = None) -> pd.DataFrame:
    df = _read("companies_market")
    if df.empty or tickers is None:
        return df
    return df[df["Ticker"].isin(tickers)].reset_index(drop=True)


def save_financials(ticker: str, name: str, currency: str | None, df: pd.DataFrame):
    if df is None or df.empty:
        return
    long = df.copy()
    long.index.name = "poste"
    long = long.reset_index().melt(id_vars="poste", var_name="annee", value_name="valeur").dropna(subset=["valeur"])
    long.insert(0, "devise", currency)
    long.insert(0, "societe", name)
    long.insert(0, "ticker", ticker)
    long["mis_a_jour"] = _now()
    old = _read("companies_financials")
    out = pd.concat([old[old["ticker"] != ticker], long]) if not old.empty else long
    _write("companies_financials", out)


def load_financials(ticker: str) -> tuple[pd.DataFrame, dict]:
    d = _read("companies_financials")
    if d.empty:
        return pd.DataFrame(), {}
    d = d[d["ticker"] == ticker]
    if d.empty:
        return pd.DataFrame(), {}
    df = d.pivot_table(index="poste", columns="annee", values="valeur", aggfunc="last")
    df.columns = [int(c) for c in df.columns]
    meta = {"name": d["societe"].iloc[-1], "currency": d["devise"].iloc[-1], "updated": d["mis_a_jour"].iloc[-1]}
    return df, meta


def all_financials() -> dict[str, tuple[pd.DataFrame, dict]]:
    d = _read("companies_financials")
    out = {}
    if d.empty:
        return out
    for tk, g in d.groupby("ticker"):
        df = g.pivot_table(index="poste", columns="annee", values="valeur", aggfunc="last")
        df.columns = [int(c) for c in df.columns]
        out[tk] = (df, {"name": g["societe"].iloc[-1], "currency": g["devise"].iloc[-1]})
    return out


# ------------------------------------------------------------------ pays
def save_imf(imf: dict[str, pd.DataFrame]):
    rows = []
    for ind, df in imf.items():
        if df is None or df.empty:
            continue
        m = df.copy()
        m.index.name = "annee"
        m = m.reset_index().melt(id_vars="annee", var_name="iso", value_name="valeur").dropna()
        m.insert(0, "indicateur", ind)
        rows.append(m)
    if not rows:
        return
    new = pd.concat(rows)
    new["mis_a_jour"] = _now()
    old = _read("countries_imf")
    if not old.empty:
        keep = old.set_index(["indicateur", "iso", "annee"]).index.difference(new.set_index(["indicateur", "iso", "annee"]).index)
        old = old.set_index(["indicateur", "iso", "annee"]).loc[keep].reset_index()
        new = pd.concat([old, new])
    _write("countries_imf", new)


def load_imf() -> dict[str, pd.DataFrame]:
    d = _read("countries_imf")
    if d.empty:
        return {}
    out = {}
    for ind, g in d.groupby("indicateur"):
        out[ind] = g.pivot_table(index="annee", columns="iso", values="valeur", aggfunc="last").sort_index()
        out[ind].index = out[ind].index.astype(int)
    return out


def save_wb(iso: str, wb: dict[str, pd.Series]):
    if not wb:
        return
    new = pd.concat([pd.DataFrame({"iso": iso, "indicateur": k, "annee": s.index, "valeur": s.values}) for k, s in wb.items()])
    new["mis_a_jour"] = _now()
    old = _read("countries_wb")
    out = pd.concat([old[old["iso"] != iso], new]) if not old.empty else new
    _write("countries_wb", out)


def load_wb(iso: str) -> dict[str, pd.Series]:
    d = _read("countries_wb")
    if d.empty:
        return {}
    d = d[d["iso"] == iso]
    return {k: g.set_index("annee")["valeur"].sort_index() for k, g in d.groupby("indicateur")}


# ------------------------------------------------------------------ matières premières
def save_prices(px: pd.DataFrame):
    if px is None or px.empty:
        return
    old = load_prices()
    out = px.combine_first(old) if not old.empty else px
    out.index.name = "date"
    _write("commodities_prices", out.sort_index().round(4), index=True)


def load_prices() -> pd.DataFrame:
    d = _read("commodities_prices", index_col=0, parse_dates=True)
    return d.sort_index() if not d.empty else d


# ------------------------------------------------------------------ journal
def log_analysis(rec: dict):
    rec = {"date": _now(), **rec}
    old = _read("analyses_log")
    out = pd.concat([old, pd.DataFrame([rec])], ignore_index=True)
    _write("analyses_log", out)


def load_log() -> pd.DataFrame:
    return _read("analyses_log")


# ------------------------------------------------------------------ gestion
def status() -> pd.DataFrame:
    rows = []
    for k, f in FILES.items():
        p = ROOT / f
        if p.exists():
            try:
                n = sum(1 for _ in open(p, encoding="utf-8")) - 1
            except Exception:
                n = np.nan
            rows.append((LABELS[k], f, n, round(p.stat().st_size / 1024, 1),
                         datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y %H:%M")))
        else:
            rows.append((LABELS[k], f, 0, 0.0, "—"))
    return pd.DataFrame(rows, columns=["Jeu de données", "Fichier", "Lignes", "Taille (Ko)", "Dernière écriture"])


def export_zip() -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES.values():
            p = ROOT / f
            if p.exists():
                z.write(p, f)
    return b.getvalue()


def import_files(files: list[tuple[str, bytes]]) -> list[str]:
    """Restaure des CSV (ou un zip de la base). Seuls les noms de fichiers attendus sont acceptés."""
    ROOT.mkdir(parents=True, exist_ok=True)
    ok = []
    allowed = set(FILES.values())
    for name, data in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist():
                    base = os.path.basename(n)
                    if base in allowed:
                        (ROOT / base).write_bytes(z.read(n))
                        ok.append(base)
        elif os.path.basename(name) in allowed:
            (ROOT / os.path.basename(name)).write_bytes(data)
            ok.append(os.path.basename(name))
    return ok


def clear(key: str):
    p = _path(key)
    if p.exists():
        p.unlink()
