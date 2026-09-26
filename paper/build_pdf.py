"""Render the CS2RB paper (a Markdown subset) to a self-contained PDF.

Usage:
  python paper/build_pdf.py                         # paper/CS2RB.md -> paper/CS2RB.pdf
  python paper/build_pdf.py --src FILE.md --out FILE.pdf
  python paper/build_pdf.py --allow-placeholders    # build while {{RELEASE}} fields are unfilled

Covers exactly the Markdown the paper uses: `#`/`##`/`###` headings, paragraphs
with **bold**, *italic*, `code` and [links](https://...), pipe tables, `- `
bullet lists, `1. ` numbered lists, `![alt](figures/x.png)` images on their own
line (scaled to the text width), and a reference list of `[n] ...` paragraphs.
HTML comment lines (the table-generator markers) are skipped.

Fonts are embedded as TrueType subsets (arXiv requires embedded outline
fonts), so the PDF renders identically everywhere. The defaults point at the
Windows font directory (Cambria for text, Consolas for code); on another
machine pass --font-dir with files of the same names, or edit FONTS below.
Every non-ASCII character in the source is checked against the body font
before rendering, so a missing glyph fails the build instead of printing a box.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "CS2RB.md"
DEFAULT_OUT = HERE / "CS2RB.pdf"
DEFAULT_FONT_DIR = Path(r"C:\Windows\Fonts")

# family -> (regular, bold, italic, bold-italic); "file#N" selects face N of a .ttc
FONTS = {
    "body": ("cambria.ttc#0", "cambriab.ttf", "cambriai.ttf", "cambriaz.ttf"),
    "mono": ("consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"),
}
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
PAGE_W, PAGE_H = letter
MARGIN = 1.0 * inch
TEXT_W = PAGE_W - 2 * MARGIN


# ----------------------------------------------------------------- fonts

def register_fonts(font_dir: Path) -> None:
    for family, specs in FONTS.items():
        names = [family, f"{family}-Bold", f"{family}-Italic", f"{family}-BoldItalic"]
        for name, spec in zip(names, specs):
            fname, _, idx = spec.partition("#")
            path = font_dir / fname
            if not path.exists():
                raise SystemExit(f"font file not found: {path}  (pass --font-dir)")
            font = TTFont(name, str(path), subfontIndex=int(idx)) if idx else TTFont(name, str(path))
            pdfmetrics.registerFont(font)
        pdfmetrics.registerFontFamily(family, normal=names[0], bold=names[1],
                                      italic=names[2], boldItalic=names[3])


def check_glyphs(text: str) -> None:
    face = pdfmetrics.getFont("body").face
    missing = sorted({c for c in text if ord(c) > 127 and face.charToGlyph.get(ord(c), 0) == 0})
    if missing:
        names = ", ".join(f"{c} U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in missing)
        raise SystemExit(f"body font lacks glyphs for: {names}")


# ---------------------------------------------------------------- styles

def styles() -> dict:
    base = dict(fontName="body", fontSize=10.5, leading=14.2)
    return {
        "title": ParagraphStyle("title", fontName="body-Bold", fontSize=17, leading=21,
                                alignment=TA_CENTER, spaceAfter=10),
        "author": ParagraphStyle("author", fontName="body", fontSize=11.5, leading=15,
                                 alignment=TA_CENTER, spaceAfter=2),
        "meta": ParagraphStyle("meta", fontName="body", fontSize=9.5, leading=13,
                               alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                               spaceAfter=14),
        "h2": ParagraphStyle("h2", fontName="body-Bold", fontSize=13, leading=16,
                             spaceBefore=14, spaceAfter=5, keepWithNext=1),
        "h3": ParagraphStyle("h3", fontName="body-Bold", fontSize=11, leading=14,
                             spaceBefore=9, spaceAfter=3, keepWithNext=1),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "abstract": ParagraphStyle("abstract", alignment=TA_JUSTIFY, leftIndent=0.35 * inch,
                                   rightIndent=0.35 * inch, spaceAfter=6, fontName="body",
                                   fontSize=10, leading=13.4),
        "quote": ParagraphStyle("quote", fontName="body-Italic", fontSize=10, leading=13.4,
                                leftIndent=0.3 * inch, rightIndent=0.3 * inch, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", leftIndent=20, bulletIndent=6, spaceAfter=3, bulletFontName="body",
                                 alignment=TA_JUSTIFY, **base),
        "ref": ParagraphStyle("ref", fontName="body", fontSize=9.5, leading=12.4,
                              leftIndent=22, firstLineIndent=-22, spaceAfter=3),
        "cell": ParagraphStyle("cell", fontName="body", fontSize=9, leading=11, alignment=TA_LEFT),
        "cellc": ParagraphStyle("cellc", fontName="body", fontSize=9, leading=11, alignment=TA_CENTER),
        "head": ParagraphStyle("head", fontName="body-Bold", fontSize=9, leading=11, alignment=TA_LEFT),
        "headc": ParagraphStyle("headc", fontName="body-Bold", fontSize=9, leading=11, alignment=TA_CENTER),
    }


# ---------------------------------------------------------------- markup

def inline(s: str) -> str:
    """Markdown inline -> ReportLab paragraph markup."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r'<font face="mono" size="9">\1</font>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#145d7f">\1</link>', s)
    return s


def parse_blocks(text: str) -> list[tuple[str, object]]:
    """Return [(kind, payload)] in document order."""
    blocks: list[tuple[str, object]] = []
    para: list[str] = []
    table: list[str] = []
    seen_title = False
    after_title = 0          # counts the paragraphs of the title block
    in_abstract = False

    def flush_para() -> None:
        nonlocal para, after_title
        if not para:
            return
        joined = " ".join(x.strip() for x in para)
        para = []
        if seen_title and after_title < 2 and not blocks_has_h2():
            blocks.append(("author" if after_title == 0 else "meta", joined))
            after_title += 1
        elif re.match(r"^\[\d+\]\s", joined):
            blocks.append(("ref", joined))
        else:
            blocks.append(("abstract" if in_abstract else "p", joined))

    def blocks_has_h2() -> bool:
        return any(k == "h2" for k, _ in blocks)

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append(("table", list(table)))
            table = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue
        img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line.strip())
        if img:
            flush_para()
            flush_table()
            frac = img.group(1).rpartition("|")[2]           # "![Figure 1|0.65](x.png)" = 65% width
            blocks.append(("image", (img.group(2), float(frac) if re.fullmatch(r"0?\.\d+", frac) else 1.0)))
            continue
        if line.startswith("|"):
            flush_para()
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_para()
            continue
        if line.startswith("# ") and not seen_title:
            flush_para()
            blocks.append(("title", line[2:].strip()))
            seen_title = True
            continue
        if line.startswith("## "):
            flush_para()
            head = line[3:].strip()
            in_abstract = head.lower() == "abstract"
            blocks.append(("h2", head))
            continue
        if line.startswith("### "):
            flush_para()
            blocks.append(("h3", line[4:].strip()))
            continue
        if line.strip() == "---":
            flush_para()
            continue
        if line.startswith("> "):
            flush_para()
            blocks.append(("quote", line[2:].strip()))
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m and not m.group(1):
            flush_para()
            marker = m.group(2)
            blocks.append(("num" if marker[0].isdigit() else "bullet", [marker, m.group(3)]))
            continue
        if line.startswith("   ") and blocks and blocks[-1][0] in ("bullet", "num") and not para:
            blocks[-1][1][1] += " " + line.strip()      # list-item continuation line
            continue
        para.append(line)
    flush_para()
    flush_table()
    return blocks


def build_table(lines: list[str], st: dict, caption=None):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue                                    # the |---| separator row
        rows.append(cells)
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    # column widths proportional to the longest cell, first column left-aligned
    longest = [max(len(re.sub(r"[*`]", "", r[j])) for r in rows) for j in range(ncol)]
    weights = [max(4, min(l, 40)) for l in longest]
    weights[0] = max(weights[0] + 3, 10)            # keep row labels ("Overpass") on one line
    total = sum(weights)
    widths = [TEXT_W * w / total for w in weights]
    data = []
    for i, r in enumerate(rows):
        para_row = []
        for j, c in enumerate(r):
            style = (st["head"] if j == 0 else st["headc"]) if i == 0 else (st["cell"] if j == 0 else st["cellc"])
            para_row.append(Paragraph(inline(c), style))
        data.append(para_row)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "body"),     # TableStyle defaults to (unembedded) Helvetica
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether(([caption] if caption is not None else []) + [Spacer(1, 3), t, Spacer(1, 8)])


def story_from(blocks, st: dict, base: Path = HERE) -> tuple[list, str]:
    story, title = [], ""
    caption = None          # a "**Table ...**" paragraph waiting for the table it describes
    for kind, payload in blocks:
        if caption is not None and kind != "table":
            story.append(caption)
            caption = None
        if kind == "image":
            rel, frac = payload
            path = (base / rel).resolve()
            w, h = ImageReader(str(path)).getSize()
            width = TEXT_W * frac
            story += [Spacer(1, 4), Image(str(path), width=width, height=width * h / w), Spacer(1, 2)]
            continue
        if kind == "title":
            title = payload
            story.append(Paragraph(inline(payload), st["title"]))
        elif kind == "p" and payload.startswith("**Table"):
            # table captions sit above their table and are kept on the same page as it
            caption = Paragraph(inline(payload), st["body"])
        elif kind in ("author", "meta", "h2", "h3", "p", "abstract", "quote", "ref"):
            story.append(Paragraph(inline(payload), st["body" if kind == "p" else kind]))
        elif kind == "table":
            story.append(build_table(payload, st, caption))
            caption = None
        elif kind in ("bullet", "num"):
            marker, text = payload
            bullet = "\u2022" if kind == "bullet" else marker
            story.append(Paragraph(inline(text), st["bullet"], bulletText=bullet))
    return story, title


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("body", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(PAGE_W / 2, 0.62 * inch, str(doc.page))
    canvas.restoreState()


# ------------------------------------------------------------------ main

def build(src: Path, out: Path, font_dir: Path, author: str, allow_placeholders: bool) -> Path:
    text = src.read_text(encoding="utf-8")
    left = sorted(set(PLACEHOLDER.findall(text)))
    if left and not allow_placeholders:
        raise SystemExit("unfilled release fields " + ", ".join(left)
                         + " -- run paper/release.py first, or pass --allow-placeholders")
    register_fonts(font_dir)
    from reportlab import rl_config
    rl_config.canvas_basefontname = "body"     # keep Helvetica out of the page preamble
    check_glyphs(text)
    st = styles()
    story, title = story_from(parse_blocks(text), st, src.resolve().parent)
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title=title, author=author,
                            subject="Counter-Strike 2 round-level win prediction benchmark",
                            creator="CS2RB paper/build_pdf.py (ReportLab)",
                            initialFontName="body")   # keeps Helvetica out of the page preamble
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    if left:
        print(f"[build_pdf] WARNING: built with unfilled fields {left}")
    print(f"[build_pdf] wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    ap.add_argument("--author", default="William Bishop")
    ap.add_argument("--allow-placeholders", action="store_true")
    a = ap.parse_args(argv)
    build(a.src, a.out, a.font_dir, a.author, a.allow_placeholders)


if __name__ == "__main__":
    sys.exit(main())
