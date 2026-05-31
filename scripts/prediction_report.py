from __future__ import annotations

from datetime import datetime
import base64
import re
from typing import Any
from io import BytesIO
import re as _re
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle, Image
from PIL import Image as PILImage


MODEL_LIMITATIONS = [
    "Does not model pests, diseases, fertilizer access, conflict, or management differences.",
    "Best used for seasonal planning and relative comparison, not field-level insurance or payout decisions.",
    "Predictions reflect climate patterns seen in the training data and may be unreliable for extreme out-of-distribution years.",
]

_EMOJI_RE = _re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=_re.UNICODE,
)


def _format_value(value: Any, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    if isinstance(value, (int, float, np.floating, np.integer)):
        if signed:
            return f"{float(value):+.{digits}f}"
        return f"{float(value):,.{digits}f}"
    return str(value)


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "prediction_report"


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", str(text))


def _markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No data available._"

    display_df = df.copy()
    if columns:
        existing = [col for col in columns if col in display_df.columns]
        display_df = display_df[existing]
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    headers = list(display_df.columns)
    if not headers:
        return "_No data available._"

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display_df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                cells.append(_format_value(value, digits=2))
            elif isinstance(value, (np.floating, np.integer)):
                cells.append(_format_value(value, digits=2))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_prediction_report_markdown(
    *,
    crop: str,
    region: str,
    year: int,
    y_pred: float,
    y_lower: float,
    y_upper: float,
    uncertainty_pct: float,
    model_confidence: float,
    uncertainty_band: str,
    model_members: int,
    sequence_df: pd.DataFrame,
    driver_df: pd.DataFrame | None = None,
    historical_summary: dict[str, Any] | None = None,
    anomalies_df: pd.DataFrame | None = None,
    images: dict[str, bytes] | None = None,
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.now()
    historical_summary = historical_summary or {}
    

    parts: list[str] = []
    parts.append(f"# Yield Prediction Report")
    parts.append(f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")
    parts.append("## Scenario Summary")
    parts.append(
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "Crop": crop,
                        "Region": region,
                        "Year": year,
                        "Prediction (kg/ha)": _format_value(y_pred, digits=0),
                        "95% CI Low": _format_value(y_lower, digits=0),
                        "95% CI High": _format_value(y_upper, digits=0),
                        "Model Confidence (%)": _format_value(model_confidence, digits=1),
                        "Uncertainty (%)": _format_value(uncertainty_pct, digits=1),
                        "Uncertainty Band": uncertainty_band,
                        "Ensemble Members": model_members,
                    }
                ]
            )
        )
    )

    parts.append("")
    parts.append("## Input Parameters")
    parts.append("This is the 12-month climate sequence that was used as model input.")
    parts.append("")
    parts.append(_markdown_table(sequence_df.reset_index().rename(columns={sequence_df.index.name or "index": "Month"})))

    parts.append("")
    parts.append("## Prediction Context")
    parts.append(f"- Expected yield: **{_format_value(y_pred, digits=0)} kg/ha**")
    parts.append(f"- 95% confidence interval: **{_format_value(y_lower, digits=0)} to {_format_value(y_upper, digits=0)} kg/ha**")
    parts.append(f"- Ensemble confidence: **{_format_value(model_confidence, digits=1)}%**")
    parts.append(f"- Uncertainty band: **{uncertainty_band}**")

    # Embed prediction chart (PNG) into Markdown if provided
    if images and images.get("prediction_chart"):
        try:
            b64 = base64.b64encode(images["prediction_chart"]).decode("ascii")
            parts.append("")
            parts.append(f"![Prediction chart](data:image/png;base64,{b64})")
        except Exception:
            pass

    if historical_summary:
        parts.append("")
        parts.append("## Temporal Comparison vs Historical Baseline")
        hist_mean = historical_summary.get("mean")
        hist_std = historical_summary.get("std")
        hist_count = historical_summary.get("count")
        pct_diff = historical_summary.get("pct_diff")
        z_score = historical_summary.get("z_score")
        percentile_rank = historical_summary.get("percentile_rank")

        parts.append(f"- Historical mean yield: **{_format_value(hist_mean, digits=0)} kg/ha**")
        if hist_std is not None:
            parts.append(f"- Historical standard deviation: **{_format_value(hist_std, digits=0)} kg/ha**")
        if hist_count is not None:
            parts.append(f"- Historical sample size: **{hist_count}**")
        if pct_diff is not None:
            parts.append(f"- Percent difference vs historical mean: **{_format_value(pct_diff, digits=1, signed=True)}%**")
        if z_score is not None:
            parts.append(f"- Z-score: **{_format_value(z_score, digits=2, signed=True)}σ**")
        if percentile_rank is not None:
            parts.append(f"- Percentile rank: **{_format_value(percentile_rank, digits=0)}th percentile**")

        if percentile_rank is not None:
            if percentile_rank >= 90:
                parts.append("- Interpretation: this result sits in the top 10% of historical seasons.")
            elif percentile_rank <= 10:
                parts.append("- Interpretation: this result sits in the bottom 10% of historical seasons.")

    # Monthly climate anomalies section removed: anomalies are no longer included in reports.

    if driver_df is not None and not driver_df.empty:
        parts.append("")
        parts.append("## Drivers Behind This Prediction")
        driver_view = driver_df.copy()
        keep_columns = [
            col
            for col in [
                "Rank",
                "Feature",
                "Parameter",
                "User_Mean",
                "Baseline_Mean",
                "User_vs_Baseline_Delta",
                "Yield_Impact_kg_ha",
                "Normalized_Impact_kg_ha",
                "Interpretation",
            ]
            if col in driver_view.columns
        ]
        parts.append(_markdown_table(driver_view[keep_columns]))

        # Embed driver influence chart (PNG) into Markdown if provided
        if images and images.get("driver_chart"):
            try:
                b64 = base64.b64encode(images["driver_chart"]).decode("ascii")
                parts.append("")
                parts.append(f"![Driver influence chart](data:image/png;base64,{b64})")
            except Exception:
                pass

    # Agronomic risk details and climate stress score omitted from report by design.

    parts.append("")
    parts.append("## Confidence Explanation")
    parts.append(
        "Model ensemble confidence reflects how closely the trained models agree. Higher values mean a tighter prediction band and stronger model consensus."
    )
    parts.append("")
    parts.append("## Model Limitations")
    for item in MODEL_LIMITATIONS:
        parts.append(f"- {item}")
    parts.append("")
    parts.append("## Notes")
    parts.append("- Precipitation is treated as monthly total rainfall in mm/month.")
    parts.append("- Driver impacts are sensitivity-based and are best used for ranking, not exact arithmetic decomposition.")
    parts.append("- This report is designed to be shared directly or exported to PDF from a Markdown editor if needed.")

    return "\n".join(parts).strip() + "\n"


def build_prediction_report_filename(crop: str, region: str, year: int, suffix: str = "md") -> str:
    return f"prediction_report_{_sanitize_filename(crop)}_{_sanitize_filename(region)}_{_sanitize_filename(year)}.{suffix}"


def _markdown_inline_to_html(text: str) -> str:
    text = _strip_emojis(str(text))
    text = _xml_escape(text)
    text = _re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = _re.sub(r"\*(?!\s)([^*]+?)\*", r"<i>\1</i>", text)
    return text


def _parse_markdown_table_rows(table_lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_lines:
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        rows.append(cells)
    return rows


def _build_reportlab_table(table_lines: list[str], styles, available_width: float) -> Table | None:
    if len(table_lines) < 2:
        return None

    rows = _parse_markdown_table_rows(table_lines)
    if len(rows) < 2:
        return None

    headers = rows[0]
    body_rows = rows[2:] if len(rows) > 2 else []
    if not headers:
        return None

    data = [headers] + body_rows if body_rows else [headers]
    col_count = max(len(row) for row in data)
    if col_count == 0:
        return None

    normalized: list[list[str]] = []
    for row in data:
        normalized.append(row + [""] * (col_count - len(row)))

    cell_style = styles["BodyText"].clone("PDFTableCell")
    cell_style.fontSize = 8.5
    cell_style.leading = 10
    header_style = styles["BodyText"].clone("PDFTableHeaderCell")
    header_style.fontSize = 8.5
    header_style.leading = 10
    header_style.textColor = colors.white
    header_style.fontName = "Helvetica-Bold"
    header_style.alignment = 1

    table_data = []
    for row_index, row in enumerate(normalized):
        table_row = []
        for cell in row:
            clean_cell = _strip_emojis(cell)
            if row_index == 0:
                table_row.append(Paragraph(_markdown_inline_to_html(clean_cell), header_style))
            else:
                table_row.append(Paragraph(_markdown_inline_to_html(clean_cell), cell_style))
        table_data.append(table_row)

    # Prefer a wider column for long 'Interpretation' text when present
    headers_lower = [h.lower() for h in headers]
    if "interpretation" in headers_lower and col_count > 1:
        interp_idx = headers_lower.index("interpretation")
        interp_width = max(available_width * 0.35, available_width / col_count)
        other_width = (available_width - interp_width) / (col_count - 1)
        col_widths = [other_width] * col_count
        col_widths[interp_idx] = interp_width
    else:
        col_widths = [available_width / col_count] * col_count

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3a5f")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                # Add extra horizontal padding for Interpretation column when present
                ("LEFTPADDING", (interp_idx if 'interp_idx' in locals() else 0, 0), (interp_idx if 'interp_idx' in locals() else 0, -1), 8),
                ("RIGHTPADDING", (interp_idx if 'interp_idx' in locals() else 0, 0), (interp_idx if 'interp_idx' in locals() else 0, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _markdown_to_pdf_story(markdown_text: str, title: str = "Yield Prediction Report", images: dict[str, bytes] | None = None) -> list:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=22, leading=26, spaceAfter=12))
    styles.add(ParagraphStyle(name="ReportHeading1", parent=styles["Heading1"], fontSize=17, leading=20, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportHeading2", parent=styles["Heading2"], fontSize=13.5, leading=16, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=10.5, leading=13, spaceAfter=3))
    styles.add(ParagraphStyle(name="ReportBullet", parent=styles["BodyText"], leftIndent=12, firstLineIndent=-8, fontSize=10.5, leading=13, spaceAfter=2))

    story = [Paragraph(_markdown_inline_to_html(title), styles["ReportTitle"]), Spacer(1, 0.15 * inch)]
    table_block: list[str] = []
    MAX_IMAGE_WIDTH = 9 * inch

    def flush_table_block() -> None:
        nonlocal table_block
        if not table_block:
            return
        table_text = "\n".join(table_block)
        story.append(Preformatted(table_text, styles["Code"]))
        story.append(Spacer(1, 0.1 * inch))
        table_block = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_table_block()
            story.append(Spacer(1, 0.08 * inch))
            continue

        if stripped.startswith("|"):
            table_block.append(stripped)
            continue

        flush_table_block()

        if stripped.startswith("# "):
            story.append(Paragraph(_markdown_inline_to_html(stripped[2:].strip()), styles["ReportHeading1"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_markdown_inline_to_html(stripped[3:].strip()), styles["ReportHeading2"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(f"• {_markdown_inline_to_html(stripped[2:].strip())}", styles["ReportBullet"]))
        else:
            story.append(Paragraph(_markdown_inline_to_html(stripped), styles["ReportBody"]))

    flush_table_block()
    return story


def build_prediction_report_pdf_bytes(markdown_text: str, images: dict[str, bytes] | None = None, title: str = "Yield Prediction Report") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
        author="Copilot",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="ReportHeading1", parent=styles["Heading1"], fontSize=15, leading=18, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="ReportHeading2", parent=styles["Heading2"], fontSize=12, leading=14, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=9.5, leading=12, spaceAfter=3))
    styles.add(ParagraphStyle(name="ReportBullet", parent=styles["BodyText"], leftIndent=12, firstLineIndent=-8, fontSize=9.5, leading=12, spaceAfter=2))
    styles.add(ParagraphStyle(name="ReportImageCaption", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=6, alignment=1, italic=True))

    story = [Paragraph(_markdown_inline_to_html(title), styles["ReportTitle"]), Spacer(1, 0.15 * inch)]
    table_block: list[str] = []
    available_image_width = doc.width
    available_image_height = doc.height
    max_image_width = min(9 * inch, available_image_width)
    max_image_height = min(9 * inch, available_image_height * 0.7)

    def _make_fitted_image(img_bytes: bytes) -> Image:
        try:
            pil = PILImage.open(BytesIO(img_bytes))
            orig_w, orig_h = pil.size
            # aspect = height / width
            aspect = float(orig_h) / float(orig_w) if orig_w else 1.0
        except Exception:
            # fallback: create image with max width and let reportlab preserve aspect
            return Image(BytesIO(img_bytes), width=max_image_width)

        # start by fitting to max width
        width_pt = max_image_width
        height_pt = width_pt * aspect

        # if too tall, fit to max height instead
        if height_pt > max_image_height:
            height_pt = max_image_height
            width_pt = height_pt / aspect if aspect != 0 else max_image_width

        return Image(BytesIO(img_bytes), width=width_pt, height=height_pt)

    def flush_table_block() -> None:
        nonlocal table_block
        if not table_block:
            return
        table = _build_reportlab_table(table_block, styles, doc.width)
        if table is not None:
            story.append(table)
        else:
            story.append(Paragraph(_markdown_inline_to_html(_strip_emojis(" ".join(table_block))), styles["ReportBody"]))
        story.append(Spacer(1, 0.12 * inch))
        table_block = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_table_block()
            story.append(Spacer(1, 0.08 * inch))
            continue

        if stripped.startswith("|"):
            table_block.append(stripped)
            continue

        flush_table_block()

        clean_line = _strip_emojis(stripped)
        # If explicit images dict provided, detect alt-text markers and insert images
        if images:
            if clean_line.startswith("![") and "Prediction chart" in clean_line and images.get("prediction_chart"):
                try:
                    img_obj = _make_fitted_image(images["prediction_chart"])
                    story.append(img_obj)
                    # caption from alt text
                    m_alt = _re.match(r"!\[([^\]]*)\]", clean_line)
                    if m_alt:
                        caption = m_alt.group(1).strip()
                        if caption:
                            story.append(Paragraph(_markdown_inline_to_html(caption), styles["ReportImageCaption"]))
                    story.append(Spacer(1, 0.12 * inch))
                    continue
                except Exception:
                    pass

            if clean_line.startswith("![") and ("Driver influence chart" in clean_line or "Driver influence" in clean_line) and images.get("driver_chart"):
                try:
                    img_obj = _make_fitted_image(images["driver_chart"])
                    story.append(img_obj)
                    m_alt = _re.match(r"!\[([^\]]*)\]", clean_line)
                    if m_alt:
                        caption = m_alt.group(1).strip()
                        if caption:
                            story.append(Paragraph(_markdown_inline_to_html(caption), styles["ReportImageCaption"]))
                    story.append(Spacer(1, 0.12 * inch))
                    continue
                except Exception:
                    pass

        # detect embedded PNG data-URI images and render them as a fallback
        if clean_line.startswith("![") and "data:image/png;base64," in clean_line:
            m = _re.search(r"!\[[^\]]*\]\(data:image/png;base64,([^\)]+)\)", clean_line)
            if m:
                try:
                    img_b64 = m.group(1)
                    img_bytes = base64.b64decode(img_b64)
                    img_obj = _make_fitted_image(img_bytes)
                    story.append(img_obj)
                    # extract alt text caption
                    m_alt = _re.match(r"!\[([^\]]*)\]", clean_line)
                    if m_alt:
                        caption = m_alt.group(1).strip()
                        if caption:
                            story.append(Paragraph(_markdown_inline_to_html(caption), styles["ReportImageCaption"]))
                    story.append(Spacer(1, 0.12 * inch))
                    continue
                except Exception:
                    # fall back to rendering the alt text
                    pass
        if clean_line.startswith("# "):
            story.append(Paragraph(_markdown_inline_to_html(clean_line[2:].strip()), styles["ReportHeading1"]))
        elif clean_line.startswith("## "):
            story.append(Paragraph(_markdown_inline_to_html(clean_line[3:].strip()), styles["ReportHeading2"]))
        elif clean_line.startswith("- "):
            story.append(Paragraph(f"• {_markdown_inline_to_html(clean_line[2:].strip())}", styles["ReportBullet"]))
        else:
            story.append(Paragraph(_markdown_inline_to_html(clean_line), styles["ReportBody"]))

    flush_table_block()

    def add_page_number(canvas, doc_obj):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc_obj.pagesize[0] - 0.7 * inch, 0.45 * inch, f"Page {doc_obj.page}")

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()
