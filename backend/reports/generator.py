import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from sqlalchemy.orm import Session

from db.crud import get_prediction, create_report
from db.models import Prediction

REPORTS_DIR = os.path.join("uploads", "reports")


def _confidence_color(confidence: float) -> HexColor:
    if confidence > 0.80:
        return HexColor("#2e7d32")  # green
    if confidence >= 0.50:
        return HexColor("#f9a825")  # yellow/amber
    return HexColor("#c62828")  # red


def generate_report(prediction_id: int, db: Session) -> str:
    """Generate a single-page PDF diagnostic report for a prediction.

    Returns the relative path to the generated PDF file.
    Raises ValueError if prediction not found.
    """
    prediction: Prediction | None = get_prediction(db, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    image = prediction.image
    if image is None:
        raise ValueError(f"Image for prediction {prediction_id} not found")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    pdf_filename = f"report_{prediction_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    width, height = A4
    c = Canvas(pdf_path, pagesize=A4)

    # ── Header ───────────────────────────────────────────────────────
    y = height - 40 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, "Secure Medical Image Processing")
    y -= 7 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, y, "Diagnostic Report")

    # separator line
    y -= 5 * mm
    c.setStrokeColor(HexColor("#1565c0"))
    c.setLineWidth(1.5)
    c.line(30 * mm, y, width - 30 * mm, y)

    # ── Patient / Image Section ──────────────────────────────────────
    y -= 12 * mm
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(HexColor("#1565c0"))
    c.drawString(30 * mm, y, "Patient / Image Information")
    c.setFillColor(HexColor("#000000"))

    y -= 8 * mm
    c.setFont("Helvetica", 10)
    fields = [
        ("Image ID", str(image.id)),
        ("Filename", image.filename),
        ("Disease Type", image.image_type.value if image.image_type else "N/A"),
        ("Upload Time", image.upload_time.strftime("%Y-%m-%d %H:%M:%S") if image.upload_time else "N/A"),
    ]
    for label, value in fields:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(35 * mm, y, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(75 * mm, y, value)
        y -= 6 * mm

    # ── Prediction Section ───────────────────────────────────────────
    y -= 6 * mm
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(HexColor("#1565c0"))
    c.drawString(30 * mm, y, "Prediction Results")
    c.setFillColor(HexColor("#000000"))

    y -= 10 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(35 * mm, y, f"Label:  {prediction.prediction_label}")

    y -= 8 * mm
    confidence_pct = prediction.confidence * 100
    c.setFont("Helvetica-Bold", 10)
    c.drawString(35 * mm, y, "Confidence:")
    c.setFillColor(_confidence_color(prediction.confidence))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(75 * mm, y, f"{confidence_pct:.1f}%")
    c.setFillColor(HexColor("#000000"))

    y -= 7 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(35 * mm, y, "Model:")
    c.setFont("Helvetica", 10)
    c.drawString(75 * mm, y, prediction.model_name)

    if prediction.processing_time_ms is not None:
        y -= 6 * mm
        c.setFont("Helvetica-Bold", 10)
        c.drawString(35 * mm, y, "Processing:")
        c.setFont("Helvetica", 10)
        c.drawString(75 * mm, y, f"{prediction.processing_time_ms} ms")

    # ── Grad-CAM Heatmap ─────────────────────────────────────────────
    if prediction.gradcam_path and os.path.isfile(prediction.gradcam_path):
        y -= 12 * mm
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(HexColor("#1565c0"))
        c.drawString(30 * mm, y, "Grad-CAM Heatmap")
        c.setFillColor(HexColor("#000000"))
        y -= 2 * mm
        img_size = 120
        try:
            c.drawImage(
                prediction.gradcam_path,
                35 * mm,
                y - img_size,
                width=img_size,
                height=img_size,
                preserveAspectRatio=True,
            )
            y -= img_size + 5 * mm
        except Exception:
            y -= 6 * mm
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(35 * mm, y, "(Heatmap image could not be loaded)")

    # ── Footer ───────────────────────────────────────────────────────
    footer_y = 25 * mm
    c.setStrokeColor(HexColor("#bdbdbd"))
    c.setLineWidth(0.5)
    c.line(30 * mm, footer_y + 5 * mm, width - 30 * mm, footer_y + 5 * mm)

    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#616161"))
    c.drawCentredString(width / 2, footer_y, f"Generated by SMIP System \u2014 {now_iso}")
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(
        width / 2,
        footer_y - 4 * mm,
        "This report is auto-generated and should be reviewed by a qualified medical professional.",
    )

    c.save()

    # persist report record
    create_report(db, prediction_id=prediction_id, pdf_path=pdf_path)

    return pdf_path
