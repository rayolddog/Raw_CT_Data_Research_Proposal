#!/usr/bin/env python3
"""
md_to_pdf.py — render a (lightly-marked-up) Markdown file to a clean PDF with
ReportLab. Supports: #/##/### headings, **bold**, *italic*, `code`, bullet
lists (- ), numbered lists (1.), '---' horizontal rules, and blank-line-
separated paragraphs. Purpose-built for the grant proposal, but generic enough
to reuse.

Usage: md_to_pdf.py INPUT.md OUTPUT.pdf
"""
import html
import re
import sys

from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable, ListFlowable, ListItem, Image)
from reportlab.lib.utils import ImageReader

# DejaVu has full Greek + math glyphs (integral, <=, theta, omega) that the
# built-in Helvetica/WinAnsi fonts lack — required for the mathematics section.
_DEJAVU = "/usr/share/fonts/truetype/dejavu"
BODY_FONT, BOLD_FONT, ITAL_FONT, MONO_FONT = ("Helvetica", "Helvetica-Bold",
                                              "Helvetica-Oblique", "Courier")
try:
    pdfmetrics.registerFont(TTFont("DejaVu", f"{_DEJAVU}/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", f"{_DEJAVU}/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVuMono", f"{_DEJAVU}/DejaVuSansMono.ttf"))
    # this DejaVu install ships no separate Oblique face — map italic to the
    # regular face (journal-name italics are cosmetic; glyph coverage matters more)
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                       italic="DejaVu", boldItalic="DejaVu-Bold")
    BODY_FONT, BOLD_FONT, ITAL_FONT, MONO_FONT = ("DejaVu", "DejaVu-Bold",
                                                  "DejaVu", "DejaVuMono")
except Exception as e:                                    # fall back to core fonts
    print(f"[md_to_pdf] DejaVu unavailable ({e}); using Helvetica")


def inline(text):
    """Markdown inline -> ReportLab mini-HTML. Escape first, then re-introduce tags."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)          # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)  # italic
    text = re.sub(r"`(.+?)`", rf'<font face="{MONO_FONT}">\1</font>', text)    # code
    return text


def build_styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName=BODY_FONT,
                          fontSize=10, leading=14, alignment=TA_JUSTIFY,
                          spaceAfter=6)
    styles = {
        "title": ParagraphStyle("title", parent=ss["Title"], fontName=BOLD_FONT,
                                 fontSize=16, leading=20, spaceAfter=10,
                                 alignment=TA_CENTER),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD_FONT,
                             fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5,
                             textColor="#12305a"),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD_FONT,
                             fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3,
                             textColor="#12305a"),
        "body": body,
        "eq": ParagraphStyle("eq", parent=body, fontName=BODY_FONT, alignment=TA_CENTER,
                             leftIndent=12, spaceBefore=3, spaceAfter=5,
                             textColor="#111111"),
        "meta": ParagraphStyle("meta", parent=body, alignment=TA_CENTER,
                               textColor="#444444", spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=body, alignment=TA_CENTER,
                                  fontSize=9, leading=12, textColor="#333333",
                                  spaceBefore=3, spaceAfter=4),
    }
    return styles


def parse(md_path, styles):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    flow = []
    i = 0
    n = len(lines)
    # leading metadata block: **Key:** value lines right after the H1
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), styles["title"]))
            i += 1
            # consume the immediate **meta** lines
            while i < n and lines[i].strip().startswith("**") and ":**" in lines[i]:
                flow.append(Paragraph(inline(lines[i].strip()), styles["meta"]))
                i += 1
            flow.append(Spacer(1, 6))
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), styles["h3"]))
            i += 1
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), styles["h2"]))
            i += 1
        elif line.startswith("@@ "):                    # centered equation line
            flow.append(Paragraph(inline(line[3:]), styles["eq"]))
            i += 1
        elif re.match(r"^!\[.*\]\(.+?\)(?:\{[0-9.]+\})?\s*$", line):  # ![cap](path){frac}
            m = re.match(r"^!\[(.*)\]\((.+?)\)(?:\{([0-9.]+)\})?\s*$", line)
            cap, path, frac = m.group(1), m.group(2), m.group(3)
            iw, ih = ImageReader(path).getSize()
            avail = letter[0] - 1.8 * inch               # page width minus margins
            w = min(avail, iw) * (float(frac) if frac else 1.0)
            flow.append(Spacer(1, 4))
            flow.append(Image(path, width=w, height=w * ih / iw))
            if cap:
                flow.append(Paragraph(inline(cap), styles["caption"]))
            flow.append(Spacer(1, 4))
            i += 1
        elif line.strip() == "---":
            flow.append(Spacer(1, 2))
            flow.append(HRFlowable(width="100%", thickness=0.6, color="#bbbbbb",
                                   spaceBefore=2, spaceAfter=6))
            i += 1
        elif re.match(r"^\s*[-*] ", line) or re.match(r"^\s*\d+\.\s", line):
            items = []
            numbered = bool(re.match(r"^\s*\d+\.\s", line))
            while i < n and (re.match(r"^\s*[-*] ", lines[i]) or
                             re.match(r"^\s*\d+\.\s", lines[i])):
                txt = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i])
                items.append(ListItem(Paragraph(inline(txt), styles["body"]),
                                      leftIndent=14))
                i += 1
            flow.append(ListFlowable(items,
                                     bulletType="1" if numbered else "bullet",
                                     start="1" if numbered else None,
                                     bulletFontSize=9, leftIndent=16))
            flow.append(Spacer(1, 4))
        else:
            # paragraph: gather until blank line
            para = [line]
            i += 1
            while i < n and lines[i].strip() and not re.match(
                    r"^(#{1,3} |\s*[-*] |\s*\d+\.\s|---)", lines[i]):
                para.append(lines[i].rstrip())
                i += 1
            flow.append(Paragraph(inline(" ".join(para)), styles["body"]))
    return flow


def main():
    src, dst = sys.argv[1], sys.argv[2]
    styles = build_styles()
    doc = SimpleDocTemplate(dst, pagesize=letter,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="Sinogram ICH Detection — Research Proposal",
                            author="John B. Bramble, MD")
    doc.build(parse(src, styles))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
