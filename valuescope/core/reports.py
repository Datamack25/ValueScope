"""Génération des livrables : graphiques PNG, classeur Excel, rapport Word, rapport PDF."""
from __future__ import annotations
import io
import os
import zipfile
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .schema import FIELDS

NAVY, GOLD, TEAL, RED, GREY = "#0B1F3A", "#C9A227", "#1B998B", "#D1495B", "#8A94A6"
ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(colors="#333")


def _png(fig) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return b.getvalue()


def charts(R: dict) -> dict[str, bytes]:
    df, out = R["df"], {}
    yrs = [str(c) for c in df.columns]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(yrs, df.loc["revenue"], color=NAVY, label="Chiffre d'affaires")
    ax2 = ax.twinx()
    ax2.plot(yrs, df.loc["ebit"] / df.loc["revenue"] * 100, color=GOLD, marker="o", lw=2.5, label="Marge EBIT (%)")
    ax2.spines["top"].set_visible(False)
    _style(ax, "Croissance et rentabilité opérationnelle")
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88), frameon=False, fontsize=8)
    out["01_croissance_marges.png"] = _png(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    rr = R["ratios"]
    for k, c in [("ROE (%)", NAVY), ("ROIC après impôt (%)", TEAL)]:
        ax.plot(yrs, rr.loc[k].values, marker="o", lw=2.4, color=c, label=k)
    if R.get("wacc"):
        ax.axhline(R["wacc"] * 100, color=RED, ls="--", lw=1.6, label=f"Coût du capital {R['wacc']*100:.1f} %")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Rentabilité des capitaux vs coût du capital")
    out["02_roic_vs_wacc.png"] = _png(fig)

    dcf = R["dcf"]["table"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(dcf.columns, dcf.loc["FCFF"], color=TEAL, label="FCFF")
    ax.bar(dcf.columns, dcf.loc["VA des FCFF"], color=NAVY, width=.45, label="Valeur actuelle")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Flux de trésorerie disponibles projetés (DCF)")
    out["03_fcff_projetes.png"] = _png(fig)

    s = R["summary"]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(s) + 1.2))
    names = list(s["Méthode"])[::-1]
    low, high, mid = s["Bas"].values[::-1], s["Haut"].values[::-1], s["Par action"].values[::-1]
    ax.barh(names, high - low, left=low, color=GOLD, alpha=.55, height=.55)
    ax.scatter(mid, names, color=NAVY, zorder=3, s=36)
    if R.get("price"):
        ax.axvline(R["price"], color=RED, lw=2, label=f"Cours {R['price']:.2f}")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.axvline(R["fair_value"], color=TEAL, lw=2, ls="--")
    _style(ax, "Football field — fourchettes de valeur par action")
    ax.grid(axis="x", alpha=.25)
    out["04_football_field.png"] = _png(fig)

    if R.get("mc") is not None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        mc = R["mc"]
        mc = mc[(mc > np.percentile(mc, 1)) & (mc < np.percentile(mc, 99))]
        ax.hist(mc, bins=60, color=NAVY, alpha=.85)
        if R.get("price"):
            ax.axvline(R["price"], color=RED, lw=2, label="Cours actuel")
        ax.axvline(np.median(mc), color=GOLD, lw=2, label="Médiane")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, "Distribution Monte Carlo de la valeur par action")
        out["05_monte_carlo.png"] = _png(fig)
    return out


# ---------------------------------------------------------------- Excel
def excel(R: dict) -> bytes:
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.drawing.image import Image as XLImage
    b = io.BytesIO()
    df = R["df"].copy()
    df.index = [FIELDS[k][0] for k in df.index]
    with pd.ExcelWriter(b, engine="openpyxl") as w:
        syn = pd.DataFrame({
            "Élément": ["Société", "Devise / unité", "Date", "Valeur intrinsèque / action", "Cours", "Potentiel", "Verdict",
                        "Coût des fonds propres", "WACC", "Stade du cycle de vie"],
            "Valeur": [R["name"], R["unit"], date.today().isoformat(), round(R["fair_value"], 2),
                       R.get("price") or "n.d.", f"{R['upside']*100:.1f} %" if R.get("upside") is not None else "n.d.",
                       R["verdict"], f"{R['ke']*100:.2f} %", f"{R['wacc']*100:.2f} %", R["lifecycle"][0]]})
        syn.to_excel(w, "Synthèse", index=False)
        R["summary"].to_excel(w, "Synthèse", index=False, startrow=13)
        df.to_excel(w, "États financiers")
        R["ratios"].round(4).to_excel(w, "Ratios")
        R["dcf"]["table"].to_excel(w, "DCF FCFF")
        pd.DataFrame(R["assumptions"].items(), columns=["Hypothèse", "Valeur"]).to_excel(w, "Hypothèses", index=False)
        R["mult"].to_excel(w, "Multiples", index=False)
        R["sens"].to_excel(w, "Sensibilité WACC×g")
        for n, sc in R["scen"].items():
            sc.to_excel(w, f"Prévision {n}")
        wb = w.book
        head = PatternFill("solid", fgColor="0B1F3A")
        for ws in wb.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = head
                cell.alignment = Alignment(horizontal="center")
            ws.column_dimensions["A"].width = 42
            for col in "BCDEFGHIJKLM":
                ws.column_dimensions[col].width = 16
            for row in ws.iter_rows(min_row=2):
                for c in row[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "#,##0.00"
        ws = wb.create_sheet("Graphiques")
        r0 = 1
        for name, png in charts(R).items():
            img = XLImage(io.BytesIO(png))
            img.width, img.height = img.width * 0.5, img.height * 0.5
            ws.add_image(img, f"A{r0}")
            r0 += int(img.height / 19) + 2
    return b.getvalue()


# ---------------------------------------------------------------- Word
def word(R: dict) -> bytes:
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2)
    t = d.add_heading(f"Rapport d'analyse et de valorisation — {R['name']}", 0)
    d.add_paragraph(f"Date : {date.today():%d/%m/%Y} · Unité : {R['unit']} · Pays : {R['country']} · Secteur : {R['industry']}")
    d.add_heading("1. Conclusion", 1)
    p = d.add_paragraph()
    run = p.add_run(f"{R['verdict']} — valeur intrinsèque {R['fair_value']:,.2f} par action"
                    + (f" vs cours {R['price']:,.2f} (potentiel {R['upside']*100:+.1f} %)." if R.get("upside") is not None else "."))
    run.bold = True
    run.font.color.rgb = RGBColor(0x0B, 0x1F, 0x3A)
    d.add_paragraph(R["narrative"])
    d.add_heading("2. Diagnostic financier", 1)
    for lvl, txt in R["comments"]:
        d.add_paragraph(txt, style="List Bullet")
    d.add_paragraph(f"Cycle de vie : {R['lifecycle'][0]} — {R['lifecycle'][1]}")
    d.add_paragraph(f"Altman : {R['altman']['score']:.2f} ({R['altman']['zone']}, {R['altman']['model']}) · "
                    f"Piotroski F-score : {R['piotroski']}/9")
    ch = charts(R)
    for k in ["01_croissance_marges.png", "02_roic_vs_wacc.png"]:
        d.add_picture(io.BytesIO(ch[k]), width=Cm(16))
    d.add_heading("3. Coût du capital", 1)
    _table(d, pd.DataFrame(R["wacc_detail"].items(), columns=["Composante", "Valeur"]))
    d.add_heading("4. Valorisation", 1)
    d.add_picture(io.BytesIO(ch["03_fcff_projetes.png"]), width=Cm(16))
    _table(d, R["summary"].round(2))
    d.add_picture(io.BytesIO(ch["04_football_field.png"]), width=Cm(16))
    if "05_monte_carlo.png" in ch:
        d.add_picture(io.BytesIO(ch["05_monte_carlo.png"]), width=Cm(16))
    d.add_heading("5. Hypothèses clés", 1)
    _table(d, pd.DataFrame(R["assumptions"].items(), columns=["Hypothèse", "Valeur"]))
    d.add_heading("Avertissement", 2)
    d.add_paragraph("Analyse générée automatiquement à partir des états financiers fournis et d'hypothèses modifiables. "
                    "Elle ne constitue pas un conseil en investissement.")
    b = io.BytesIO()
    d.save(b)
    return b.getvalue()


def _table(d, df: pd.DataFrame):
    t = d.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = f"{v:,.2f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v is None or (isinstance(v, float) and v != v) else str(v))


# ---------------------------------------------------------------- PDF
def pdf(R: dict) -> bytes:
    from fpdf import FPDF
    p = FPDF()
    p.set_auto_page_break(True, 15)
    p.add_font("DV", "", os.path.join(ASSETS, "DejaVuSans.ttf"))
    p.add_font("DV", "B", os.path.join(ASSETS, "DejaVuSans-Bold.ttf"))
    p.add_page()
    p.set_fill_color(11, 31, 58)
    p.rect(0, 0, 210, 38, "F")
    p.set_text_color(255, 255, 255)
    p.set_font("DV", "B", 18)
    p.set_xy(12, 10)
    p.cell(0, 9, f"ValueScope — {R['name']}")
    p.set_font("DV", "", 10)
    p.set_xy(12, 21)
    p.cell(0, 6, f"Analyse financière & valorisation · {date.today():%d/%m/%Y} · {R['unit']} · {R['country']}")
    p.set_text_color(20, 20, 20)
    p.set_y(46)
    col = {"ok": (27, 153, 139), "bad": (209, 73, 91), "warn": (201, 162, 39), "info": (11, 31, 58)}[R["verdict_lvl"]]
    p.set_fill_color(*col)
    p.set_text_color(255, 255, 255)
    p.set_font("DV", "B", 13)
    txt = f"{R['verdict']}  ·  Valeur {R['fair_value']:,.2f}" + (f"  ·  Cours {R['price']:,.2f}  ·  {R['upside']*100:+.1f} %" if R.get("upside") is not None else "")
    p.cell(0, 11, txt, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    p.set_text_color(20, 20, 20)
    p.ln(3)
    p.set_font("DV", "", 9.5)
    p.multi_cell(0, 5, R["narrative"], new_x="LMARGIN", new_y="NEXT")
    p.ln(2)
    _h(p, "Diagnostic")
    for _, txt in R["comments"]:
        p.multi_cell(0, 5, "• " + txt, new_x="LMARGIN", new_y="NEXT")
    p.multi_cell(0, 5, f"• Cycle de vie : {R['lifecycle'][0]}. Altman {R['altman']['score']:.2f} ({R['altman']['zone']}). Piotroski {R['piotroski']}/9.", new_x="LMARGIN", new_y="NEXT")
    ch = charts(R)
    for k in ["01_croissance_marges.png", "02_roic_vs_wacc.png"]:
        p.image(io.BytesIO(ch[k]), w=175)
    p.add_page()
    _h(p, "Synthèse des méthodes de valorisation (par action)")
    _ptable(p, R["summary"][["Méthode", "Bas", "Par action", "Haut", "Poids"]])
    p.image(io.BytesIO(ch["04_football_field.png"]), w=175)
    p.image(io.BytesIO(ch["03_fcff_projetes.png"]), w=175)
    if "05_monte_carlo.png" in ch:
        p.image(io.BytesIO(ch["05_monte_carlo.png"]), w=175)
    p.add_page()
    _h(p, "Coût du capital")
    _ptable(p, pd.DataFrame(R["wacc_detail"].items(), columns=["Composante", "Valeur"]))
    _h(p, "Hypothèses du DCF")
    _ptable(p, pd.DataFrame(R["assumptions"].items(), columns=["Hypothèse", "Valeur"]))
    p.ln(4)
    p.set_font("DV", "", 7.5)
    p.multi_cell(0, 4, "Document généré automatiquement. Ne constitue pas un conseil en investissement.", new_x="LMARGIN", new_y="NEXT")
    return bytes(p.output())


def _h(p, t):
    p.set_font("DV", "B", 12)
    p.set_text_color(11, 31, 58)
    p.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
    p.set_text_color(20, 20, 20)
    p.set_font("DV", "", 9)


def _ptable(p, df):
    p.set_font("DV", "", 8.5)
    p.set_fill_color(255, 255, 255)
    p.set_draw_color(215, 210, 198)
    fmt = lambda v: f"{v:,.2f}" if isinstance(v, (float, np.floating, int)) and not isinstance(v, bool) and np.isfinite(v) else str(v)
    with p.table(col_widths=None, text_align="LEFT", line_height=5.5, cell_fill_color=(246, 244, 239), cell_fill_mode="ROWS",
                 headings_style=__import__("fpdf").fonts.FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(11, 31, 58))) as t:
        r = t.row()
        for c in df.columns:
            r.cell(str(c))
        for _, row in df.iterrows():
            r = t.row()
            for v in row:
                r.cell(fmt(v))
    p.ln(3)


def charts_zip(R: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, png in charts(R).items():
            z.writestr(n, png)
    return b.getvalue()
