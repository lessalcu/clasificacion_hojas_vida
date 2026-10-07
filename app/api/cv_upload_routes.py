from flask import Blueprint, current_app, jsonify, request

from app.errors.exceptions import ValidationError
from app.services.cv_upload_service import CvUploadService

cv_upload_bp = Blueprint("cv_upload", __name__, url_prefix="/api/v1/candidates")


def build_service():
    return CvUploadService(
        max_size_bytes=current_app.config["MAX_CV_FILE_SIZE_BYTES"],
        max_batch_files=current_app.config["MAX_BATCH_FILES"],
    )


@cv_upload_bp.post("/validate")
def validate_single_cv():
    file = request.files.get("file")
    if not file:
        raise ValidationError("El campo 'file' es obligatorio")

    result = build_service().validate_single(file)

    return (
        jsonify(
            {
                "success": True,
                "message": "Validación completada",
                "data": result,
            }
        ),
        200,
    )


@cv_upload_bp.post("/validate-batch")
def validate_batch_cv():
    files = request.files.getlist("files")
    files = [file for file in files if file and (file.filename or "").strip()]

    if not files:
        raise ValidationError("El campo 'files' es obligatorio")

    result = build_service().validate_batch(files)

    return (
        jsonify(
            {
                "success": True,
                "message": "Validación del lote completada",
                "data": result,
            }
        ),
        200,
    )


@cv_upload_bp.post("/upload")
def upload_single_cv():
    file = request.files.get("file")
    if not file:
        raise ValidationError("El campo 'file' es obligatorio")

    result = build_service().upload_single(file)

    return (
        jsonify(
            {
                "success": True,
                "message": "Hoja de vida cargada correctamente",
                "data": result,
            }
        ),
        201,
    )


@cv_upload_bp.post("/upload-batch")
def upload_batch_cv():
    files = request.files.getlist("files")
    files = [file for file in files if file and (file.filename or "").strip()]

    if not files:
        raise ValidationError("El campo 'files' es obligatorio")

    result = build_service().upload_batch(files)

    return (
        jsonify(
            {"success": True, "message": "Carga por lote procesada", "data": result}
        ),
        201,
    )
