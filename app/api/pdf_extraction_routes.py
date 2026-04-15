from flask import Blueprint, jsonify

from app.services.pdf_text_extraction_service import PdfTextExtractionService

pdf_extraction_bp = Blueprint(
    "pdf_extraction",
    __name__,
    url_prefix="/api/v1/candidate-sources"
)

service = PdfTextExtractionService()


@pdf_extraction_bp.post("/<source_id>/extract-text")
def extract_text_from_pdf(source_id: str):
    result = service.extract_from_source(source_id)

    return jsonify({
        "success": True,
        "message": "PDF extraction processed",
        "data": result,
    }), 200