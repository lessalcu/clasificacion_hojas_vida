from flask import Blueprint, jsonify, request

from app.services.model_validation_service import ModelValidationService

model_validation_bp = Blueprint(
    "model_validation",
    __name__,
    url_prefix="/api/v1/model-validation",
)

service = ModelValidationService()


@model_validation_bp.post("/job-profiles/<job_profile_id>/cross-validation")
def validate_job_profile_with_cross_validation(job_profile_id: str):
    payload = request.get_json() or {}
    folds = payload.get("folds")

    result = service.validate_job_profile_with_cross_validation(
        job_profile_id=job_profile_id,
        folds=int(folds) if folds else None,
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Cross validation completed successfully",
                "data": result,
            }
        ),
        200,
    )
