"""Remplit la base CSV (data/store/) depuis VOTRE ordinateur, où Yahoo Finance, le FMI et la Banque mondiale
répondent normalement (contrairement aux serveurs partagés de Streamlit Cloud, souvent bloqués).

Utilisation (dans le dossier du projet) :
    pip install -r requirements.txt
    python scripts/actualiser_base.py                 # tout : sociétés + pays + matières premières
    python scripts/actualiser_base.py --societes      # seulement les sociétés
    python scripts/actualiser_base.py --pays --matieres

Ensuite, envoyez sur GitHub les fichiers de data/store/*.csv : l'application en ligne les utilisera
automatiquement chaque fois qu'une source en direct est indisponible."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from core import companies as CO, countries as CT, commodities as CM, store as STORE, net  # noqa: E402


def societes():
    tks = list(CO.ALL_TICKERS)
    rows, ok_fin = [], 0
    print(f"\n=== Sociétés ({len(tks)}) ===")
    for i, tk in enumerate(tks, 1):
        try:
            c = CO.fetch_company(tk)
            rows.append(CO.row_from_info(tk, c["info"]))
            if not c["df"].empty:
                STORE.save_financials(tk, c["info"].get("longName") or CO.ALL_TICKERS[tk], c["fin_currency"], c["df"])
                ok_fin += 1
            print(f"  [{i:>3}/{len(tks)}] {tk:<12} ✓ cours {c['price']}  comptes {'✓' if not c['df'].empty else '—'}")
        except Exception as e:
            print(f"  [{i:>3}/{len(tks)}] {tk:<12} ✗ {str(e)[:120]}")
        time.sleep(0.6)                     # ménage les serveurs de Yahoo
    if rows:
        STORE.save_screener(pd.DataFrame(rows))
    print(f"→ {len(rows)} sociétés, {ok_fin} jeux d'états financiers enregistrés.")


def pays():
    print("\n=== Pays (FMI + Banque mondiale) ===")
    out, wb_used, errors = CT.fetch_all_macro()
    if out:
        STORE.save_imf(out)
        print(f"→ {len(out)} indicateurs macro enregistrés" + (f" (dont {len(wb_used)} via la Banque mondiale)" if wb_used else ""))
    for e in errors[:5]:
        print("  ! " + e[:150])
    n = 0
    for iso in CT.COUNTRIES:
        d = {}
        for code, (label, scale) in CT.WB_IND.items():
            try:
                s = CT.fetch_wb(code, iso)
                if not s.empty:
                    d[label] = s * scale
            except Exception:
                pass
        if d:
            STORE.save_wb(iso, d)
            n += 1
    print(f"→ séries Banque mondiale complémentaires pour {n} pays.")


def matieres():
    print("\n=== Matières premières ===")
    px = CM.fetch_prices(list(CM.COMMODITIES), period="10y", interval="1d")
    fred = list(px.attrs.get("fred", {}))
    STORE.save_prices(px.drop(columns=fred))
    print(f"→ {px.shape[1] - len(fred)} séries, {len(px)} dates enregistrées"
          + (f" ({len(fred)} séries FRED non enregistrées : unités différentes)" if fred else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Actualise la base CSV de ValueScope")
    ap.add_argument("--societes", action="store_true")
    ap.add_argument("--pays", action="store_true")
    ap.add_argument("--matieres", action="store_true")
    a = ap.parse_args()
    tout = not (a.societes or a.pays or a.matieres)
    for flag, fn in [(a.societes, societes), (a.pays, pays), (a.matieres, matieres)]:
        if tout or flag:
            try:
                fn()
            except Exception as e:
                print(f"  ✗ {fn.__name__} : {e}")
    print("\nÉtat de la base :")
    print(STORE.status().to_string(index=False))
    if net.LOG:
        print("\nDernier statut des sources :")
        for k, (ok, msg, t) in net.LOG.items():
            print(f"  {'✓' if ok else '✗'} {k} [{t}] {msg[:120]}")
    print(f"\nFichiers : {STORE.ROOT}\nÀ envoyer sur GitHub dans le même dossier data/store/.")
