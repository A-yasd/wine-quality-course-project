"""Render the generated DOCX to a standalone A4 PDF with ReportLab.

This path is useful when Microsoft Word COM automation is unavailable. The PDF
is built from the same DOCX body, tables and figures used for the submission.
"""

from __future__ import annotations

import html
import io
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph as PdfParagraph,
    SimpleDocTemplate,
    Spacer,
    Table as PdfTable,
    TableStyle,
)


PROJECT_OUTPUTS = Path(__file__).resolve().parents[2]
DOCX_PATH = PROJECT_OUTPUTS / "类别不平衡优质红酒筛选_课程设计.docx"
PDF_PATH = PROJECT_OUTPUTS / "类别不平衡优质红酒筛选_课程设计.pdf"


def register_fonts():
    pdfmetrics.registerFont(TTFont("SimSun", r"C:\Windows\Fonts\simsun.ttc"))
    pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))


def styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "TitleCN",
            fontName="SimHei",
            fontSize=17,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            fontName="SimSun",
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "info": ParagraphStyle(
            "InfoCN",
            fontName="SimSun",
            fontSize=9,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "abstract_label": ParagraphStyle(
            "AbstractLabel",
            fontName="SimHei",
            fontSize=10.5,
            leading=15,
            spaceBefore=1,
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            fontName="SimSun",
            fontSize=9.2,
            leading=14.2,
            alignment=TA_JUSTIFY,
            firstLineIndent=18,
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            fontName="SimSun",
            fontSize=8.8,
            leading=13.5,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1CN",
            fontName="SimHei",
            fontSize=13.2,
            leading=18,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            fontName="SimHei",
            fontSize=11.2,
            leading=16,
            spaceBefore=4,
            spaceAfter=2,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyCN",
            fontName="SimSun",
            fontSize=10.2,
            leading=15.2,
            alignment=TA_JUSTIFY,
            firstLineIndent=20.4,
            spaceAfter=2.3,
            wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "EquationCN",
            fontName="SimSun",
            fontSize=9.8,
            leading=14,
            alignment=TA_CENTER,
            spaceBefore=1,
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            fontName="SimSun",
            fontSize=8.3,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#454545"),
            spaceBefore=1,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "ReferenceCN",
            fontName="SimSun",
            fontSize=8.4,
            leading=11.8,
            alignment=TA_JUSTIFY,
            spaceAfter=1,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCN",
            fontName="SimSun",
            fontSize=7.5,
            leading=9.4,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            fontName="SimHei",
            fontSize=7.5,
            leading=9.4,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def is_heading(text: str) -> bool:
    return (
        text in {"参考资料"}
        or (
            len(text) >= 3
            and text[0].isdigit()
            and "  " in text[:4]
        )
    )


def paragraph_kind(paragraph: Paragraph, index: int) -> str:
    text = paragraph.text.strip()
    if index == 0:
        return "title"
    if index == 1:
        return "subtitle"
    if text.startswith("学号："):
        return "info"
    if text == "摘  要":
        return "abstract_label"
    if text.startswith("关键词："):
        return "keywords"
    if is_heading(text):
        return "h1"
    if text.startswith(("表 ", "图 ")):
        return "caption"
    if text.startswith(("p(x)", "L =", "tau*")):
        return "equation"
    if text.startswith("[") and "] " in text:
        return "reference"
    if text.startswith("复现与学术说明："):
        return "reference"
    return "body"


def extract_images(paragraph: Paragraph, document: Document, max_width: float):
    images = []
    for blip in paragraph._p.xpath(".//a:blip"):
        relationship_id = blip.get(qn("r:embed"))
        if not relationship_id:
            continue
        image_part = document.part.related_parts[relationship_id]
        stream = io.BytesIO(image_part.blob)
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(image_part.blob)) as probe:
            pixel_width, pixel_height = probe.size
        width = max_width
        height = width * pixel_height / pixel_width
        images.append(Image(stream, width=width, height=height))
    return images


def table_widths(table: Table, max_width: float):
    widths = [cell.width for cell in table.rows[0].cells]
    if all(width is not None and width > 0 for width in widths):
        values = [float(width) for width in widths]
        factor = max_width / sum(values)
        return [value * factor for value in values]
    return [max_width / len(widths)] * len(widths)


def create_table(table: Table, style: dict[str, ParagraphStyle], max_width: float):
    data = []
    for row_index, row in enumerate(table.rows):
        rendered_row = []
        for cell in row.cells:
            text = html.escape(cell.text.replace("\n", "<br/>"))
            text_style = style["table_header"] if row_index == 0 else style["table"]
            rendered_row.append(PdfParagraph(text, text_style))
        data.append(rendered_row)
    pdf_table = PdfTable(
        data,
        colWidths=table_widths(table, max_width),
        repeatRows=1,
        hAlign="CENTER",
    )
    pdf_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#6B6B6B")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return pdf_table


def draw_page(canvas, document):
    page_width, page_height = A4
    canvas.saveState()
    canvas.setFont("SimSun", 7.8)
    canvas.setFillColor(colors.HexColor("#6E6E6E"))
    canvas.drawRightString(page_width - 1.9 * cm, page_height - 0.85 * cm, "深度学习课程设计 · 2026 秋季")
    canvas.drawCentredString(page_width / 2, 0.72 * cm, f"第 {canvas.getPageNumber()} 页")
    canvas.setStrokeColor(colors.HexColor("#BFBFBF"))
    canvas.setLineWidth(0.35)
    canvas.line(1.9 * cm, page_height - 1.05 * cm, page_width - 1.9 * cm, page_height - 1.05 * cm)
    canvas.restoreState()


def build_pdf():
    register_fonts()
    report_styles = styles()
    source = Document(DOCX_PATH)
    author = source.core_properties.author or "黄海星"
    pdf = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.9 * cm,
        rightMargin=1.9 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title="类别不平衡场景下基于多层感知机的优质红酒筛选",
        author=author,
    )
    max_width = A4[0] - 1.9 * cm * 2
    story = []
    paragraph_index = 0
    pending_caption = None

    for child in source.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, source)
            text = paragraph.text.strip()
            images = extract_images(paragraph, source, max_width)
            if images:
                if pending_caption is not None:
                    story.append(pending_caption)
                    pending_caption = None
                story.extend(images)
                paragraph_index += 1
                continue
            if not text:
                continue
            kind = paragraph_kind(paragraph, paragraph_index)
            rendered = PdfParagraph(html.escape(text), report_styles[kind])
            if kind == "caption":
                if story and isinstance(story[-1], (Image, PdfTable)):
                    story[-1] = KeepTogether([story[-1], rendered])
                else:
                    pending_caption = rendered
                paragraph_index += 1
                continue
            if pending_caption is not None:
                story.append(pending_caption)
                pending_caption = None
            story.append(rendered)
            paragraph_index += 1
        elif child.tag == qn("w:tbl"):
            table = Table(child, source)
            story.append(create_table(table, report_styles, max_width))
            # The following caption is added by the ordinary paragraph branch.

    if pending_caption is not None:
        story.append(pending_caption)

    pdf.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
