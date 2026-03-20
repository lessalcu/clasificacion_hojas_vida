from flask import Blueprint, current_app, jsonify, request

from app.errors.exceptions import ValidationError
from app.services.cv_upload_service import CvUploadService

cv_upload_bp = Blueprint("cv_upload", __name__, url_prefix="/api/v1/candidates")


def build_service():
    return CvUploadService(
        max_size_bytes=current_app.config["MAX_CV_FILE_SIZE_BYTES"],
        max_batch_files=current_app.config["MAX_BATCH_FILES"],
    )


@cv_upload_bp.post("/upload")
def upload_single_cv():
    file = request.files.get("file")
    if not file:
        raise ValidationError("Field 'file' is required")

    service = build_service()
    result = service.upload_single(file)

    return jsonify({
        "success": True,
        "message": "CV uploaded successfully",
        "data": result
    }), 201


@cv_upload_bp.post("/upload-batch")
def upload_batch_cv():
    files = request.files.getlist("files")
    files = [file for file in files if file and (file.filename or "").strip()]

    if not files:
        raise ValidationError("Field 'files' is required")

    service = build_service()
    result = service.upload_batch(files)

    return jsonify({
        "success": True,
        "message": "Batch upload processed",
        "data": result
    }), 201