from flask import Blueprint, jsonify

from app.services.pdf_text_extraction_service import PdfTextExtractionService

pdf_extraction_bp = Blueprint("pdf_extraction", __name__)
service = PdfTextExtractionService()


@pdf_extraction_bp.route(
    "/api/v1/candidate-sources/<source_id>/extract-text",
    methods=["POST"],
)
def extract_pdf_text(source_id):
    result = service.extract_from_source(source_id)

    return jsonify(
        {
            "success": True,
            "message": "PDF extraction processed",
            "data": result,
        }
    ), 200