"""Report generation and management endpoints."""
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps import get_current_user
from db.database import get_db
from db.models import Image, Prediction, Report, User
from reports.generator import generate_report

router = APIRouter()


class GenerateReportRequest(BaseModel):
    prediction_id: int


class ReportListItem(BaseModel):
    id: int
    prediction_id: int
    pdf_path: str
    created_at: str
    prediction_label: str | None = None
    model_name: str | None = None


@router.get("/health")
async def health():
    return {"status": "ok", "module": "reports"}


@router.get("", response_model=list[ReportListItem])
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all reports owned by the current user (joined via prediction → image)."""
    rows = (
        db.query(Report, Prediction)
        .join(Prediction, Report.prediction_id == Prediction.id)
        .join(Image, Prediction.image_id == Image.id)
        .filter(Image.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )
    return [
        ReportListItem(
            id=r.id,
            prediction_id=r.prediction_id,
            pdf_path=r.pdf_path,
            created_at=r.created_at.isoformat() if r.created_at else "",
            prediction_label=p.prediction_label,
            model_name=p.model_name,
        )
        for r, p in rows
    ]


@router.post("/generate", response_model=ReportListItem, status_code=status.HTTP_201_CREATED)
async def create_report_endpoint(
    req: GenerateReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a PDF diagnostic report for a prediction the user owns."""
    prediction = (
        db.query(Prediction)
        .join(Image, Prediction.image_id == Image.id)
        .filter(Prediction.id == req.prediction_id, Image.user_id == current_user.id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        pdf_path = generate_report(req.prediction_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

    # Fetch the just-created report row
    report = (
        db.query(Report)
        .filter(Report.prediction_id == req.prediction_id, Report.pdf_path == pdf_path)
        .order_by(Report.created_at.desc())
        .first()
    )
    if not report:
        raise HTTPException(status_code=500, detail="Report record missing after generation")

    return ReportListItem(
        id=report.id,
        prediction_id=report.prediction_id,
        pdf_path=report.pdf_path,
        created_at=report.created_at.isoformat() if report.created_at else "",
        prediction_label=prediction.prediction_label,
        model_name=prediction.model_name,
    )


@router.get("/{report_id}/download")
async def download_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download a report's PDF (must be owned by current user)."""
    row = (
        db.query(Report, Prediction)
        .join(Prediction, Report.prediction_id == Prediction.id)
        .join(Image, Prediction.image_id == Image.id)
        .filter(Report.id == report_id, Image.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    report, _ = row
    if not os.path.isfile(report.pdf_path):
        raise HTTPException(status_code=410, detail="PDF file is missing on disk")
    return FileResponse(
        report.pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(report.pdf_path),
    )


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a report (PDF + record). Must be owned by current user."""
    row = (
        db.query(Report)
        .join(Prediction, Report.prediction_id == Prediction.id)
        .join(Image, Prediction.image_id == Image.id)
        .filter(Report.id == report_id, Image.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        if os.path.isfile(row.pdf_path):
            os.remove(row.pdf_path)
    except OSError:
        pass
    db.delete(row)
    db.commit()
    return None
