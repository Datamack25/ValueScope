"""Import d'états financiers depuis Excel/CSV, PDF, Word ou image (OCR).
Sortie : DataFrame index = clés standard (schema.FIELDS), colonnes = années (int), valeurs en unités choisies."""
from __future__ import annotations
import io
import re
import pandas as pd
import numpy as np
from .schema import FIELDS, match_label, parse_number

YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
NUM_TOKEN = re.compile(r"\(?-?−?\d[\d\s  .,]*\)?")


def _year_of(v) -> int | None:
    if isinstance(v, (int, np.integer)) and 1950 <= int(v) <= 2100:
        return int(v)
    if isinstance(v, float) and v == v and float(v).is_integer() and 1950 <= v <= 2100:
        return int(v)
    if hasattr(v, "year"):
        return int(v.year)
    m = YEAR_RE.search(str(v))
    return int(m.group(1)) if m else None


def _rows_to_frame(rows: list[list]) -> pd.DataFrame:
    """Transforme des lignes brutes [libellé, v1, v2...] en DataFrame standard.
    Détecte la ligne d'en-tête contenant des années."""
    years: list[int] | None = None
    year_cols: list[int] = []
    data: dict[str, dict[int, float]] = {}
    for row in rows:
        cells = ["" if c is None or (isinstance(c, float) and c != c) else c for c in row]
        if not cells:
            continue
        ys = [(_year_of(c), i) for i, c in enumerate(cells) if i > 0]
        ys = [(y, i) for y, i in ys if y]
        if len(ys) >= 1 and (years is None or len(ys) >= len(years)) and all(
            parse_number(c) is None or _year_of(c) for c in cells[1:] if str(c).strip()
        ):
            years = [y for y, _ in ys]
            year_cols = [i for _, i in ys]
            continue
        label = next((str(c) for c in cells if str(c).strip() and parse_number(c) is None), None)
        if not label:
            continue
        key = match_label(label)
        if not key or key in data:
            continue
        if years:
            vals = [parse_number(cells[i]) if i < len(cells) else None for i in year_cols]
        else:
            vals = [parse_number(c) for c in cells[1:]]
            vals = [v for v in vals if v is not None]
        if not any(v is not None for v in vals):
            continue
        data[key] = {}
        cols = years if years else list(range(len(vals)))
        for y, v in zip(cols, vals):
            if v is not None:
                data[key][y] = v
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).T
    if not years:  # pas d'années -> on suppose les plus récentes à gauche
        n = df.shape[1]
        df.columns = [2025 - i for i in range(n)]
    df = df[sorted(df.columns)]
    return df


def parse_excel(file_bytes: bytes, name: str) -> pd.DataFrame:
    frames = []
    if name.lower().endswith((".csv", ".txt")):
        text = file_bytes.decode("utf-8", errors="ignore")
        sep = ";" if text.count(";") > text.count(",") else ","
        sheets = {"csv": pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str)}
    else:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    for _, sh in sheets.items():
        sh = sh.dropna(how="all").dropna(axis=1, how="all")
        if sh.empty:
            continue
        # Format "transposé" (années en lignes) : on détecte si la 1re colonne contient des années
        first_col_years = sum(1 for v in sh.iloc[:, 0] if _year_of(v)) >= max(2, len(sh) * 0.6)
        if first_col_years:
            sh = sh.T
        f = _rows_to_frame(sh.values.tolist())
        if not f.empty:
            frames.append(f)
    return _merge(frames)


def _text_lines_to_rows(text: str) -> list[list]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # nombre de colonnes attendu = nb d'années de l'en-tête
    n_exp = 0
    for l in lines:
        ys = YEAR_RE.findall(l)
        if len(ys) >= 2 and len(re.sub(r"[^0-9]", "", l)) <= 4 * len(ys) + 2:
            n_exp = len(ys)
            break
    rows = []
    for line in lines:
        nums = list(NUM_TOKEN.finditer(line))
        if not nums:
            continue
        label = line[:nums[0].start()].strip(" .:\t")
        tail = line[nums[0].start():]
        parts = [p for p in re.split(r"\s{2,}|\t", tail) if p.strip()]
        if len(parts) == 1:
            toks = tail.split()
            parts = toks if n_exp and len(toks) == n_exp else _split_numbers(tail)
        if not label:
            if all(_year_of(p) for p in parts):
                rows.append(["années"] + parts)
            continue
        rows.append([label] + parts)
    return rows


def _split_numbers(s: str) -> list[str]:
    """Regroupe les milliers séparés par des espaces ('12 345,6') sans fusionner deux nombres décimaux."""
    out, cur = [], ""
    for t in s.split():
        cur_closed = bool(re.search(r"[.,]\d{1,2}\)?$", cur)) or _year_of(cur) is not None
        if cur and not cur_closed and re.fullmatch(r"\d{3}([.,]\d+)?\)?", t) and re.fullmatch(r"\(?-?\d{1,3}( \d{3})*", cur):
            cur += " " + t
        else:
            if cur:
                out.append(cur)
            cur = t
    if cur:
        out.append(cur)
    return out


def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    import pdfplumber
    frames, all_rows = [], []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages[:60]:
            for t in page.extract_tables() or []:
                f = _rows_to_frame(t)
                if not f.empty:
                    frames.append(f)
            all_rows += _text_lines_to_rows(page.extract_text() or "")
    f = _rows_to_frame(all_rows)
    if not f.empty:
        frames.append(f)
    return _merge(frames)


def parse_docx(file_bytes: bytes) -> pd.DataFrame:
    import docx
    d = docx.Document(io.BytesIO(file_bytes))
    frames = []
    for t in d.tables:
        rows = [[c.text for c in r.cells] for r in t.rows]
        f = _rows_to_frame(rows)
        if not f.empty:
            frames.append(f)
    text = "\n".join(p.text for p in d.paragraphs)
    f = _rows_to_frame(_text_lines_to_rows(text))
    if not f.empty:
        frames.append(f)
    return _merge(frames)


def parse_image(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        if img.width < 1500:
            img = img.resize((img.width * 2, img.height * 2))
        text = pytesseract.image_to_string(img, lang="fra+eng", config="--psm 6")
    except Exception as e:  # tesseract absent ou langue manquante
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), config="--psm 6")
        except Exception:
            return pd.DataFrame(), f"OCR indisponible ({e.__class__.__name__}). Installez Tesseract ou saisissez les chiffres manuellement."
    return _rows_to_frame(_text_lines_to_rows(text)), text


def _merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = frames[0].copy()
    for f in frames[1:]:
        out = out.combine_first(f)
    out = out[sorted(out.columns)]
    return out


def parse_any(file_bytes: bytes, name: str) -> tuple[pd.DataFrame, str]:
    n = name.lower()
    if n.endswith((".xlsx", ".xlsm", ".xls", ".csv", ".txt")):
        return parse_excel(file_bytes, name), ""
    if n.endswith(".pdf"):
        return parse_pdf(file_bytes), ""
    if n.endswith(".docx"):
        return parse_docx(file_bytes), ""
    if n.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        return parse_image(file_bytes)
    return pd.DataFrame(), "Format non pris en charge."


def complete(df: pd.DataFrame) -> pd.DataFrame:
    """Complète les agrégats manquants et normalise les signes (charges en positif)."""
    df = df.copy()
    for k in FIELDS:
        if k not in df.index:
            df.loc[k] = np.nan
    df = df.loc[list(FIELDS)].astype(float)
    pos = ["cogs", "personnel", "da", "interest_expense", "tax", "capex", "dividends", "buybacks", "rnd"]
    for k in pos:
        df.loc[k] = df.loc[k].abs()

    def fill(k, s):
        df.loc[k] = df.loc[k].where(df.loc[k].notna(), s)

    r = df.loc
    fill("gross_profit", r["revenue"] - r["cogs"])
    fill("ebit", r["ebitda"] - r["da"])
    fill("ebitda", r["ebit"] + r["da"].fillna(0))
    fill("pretax_income", r["ebit"] - r["interest_expense"].fillna(0))
    fill("net_income", r["pretax_income"] - r["tax"].fillna(0))
    fill("tax", r["pretax_income"] - r["net_income"])
    td = r["short_debt"].fillna(0) + r["long_debt"].fillna(0) + r["lease_liabilities"].fillna(0)
    fill("total_debt", td.where(r["short_debt"].notna() | r["long_debt"].notna()))
    ca = r["cash"].fillna(0) + r["receivables"].fillna(0) + r["inventory"].fillna(0)
    fill("current_assets", ca.where(r["cash"].notna() | r["receivables"].notna()))
    cl = r["payables"].fillna(0) + r["short_debt"].fillna(0)
    fill("current_liabilities", cl.where(r["payables"].notna()))
    for k in ["minority", "lease_liabilities", "dividends", "buybacks", "rnd"]:
        df.loc[k] = df.loc[k].fillna(0)
    return df


def to_editor(df: pd.DataFrame) -> pd.DataFrame:
    e = df.copy()
    e.columns = [str(c) for c in e.columns]
    e.insert(0, "Poste", [FIELDS[k][0] for k in e.index])
    return e


def from_editor(e: pd.DataFrame) -> pd.DataFrame:
    df = e.drop(columns=["Poste"]).copy()
    cols = []
    for c in df.columns:
        y = _year_of(c)
        cols.append(y if y else c)
    df.columns = cols
    df = df.apply(pd.to_numeric, errors="coerce")
    return df[sorted([c for c in df.columns if isinstance(c, int)])]
