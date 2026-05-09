from flask import Blueprint, jsonify, request

from app.services.model_training_service import ModelTrainingService
from app.validators.training_validator import validate_training_payload

model_training_bp = Blueprint(
    "model_training",
    __name__,
    url_prefix="/api/v1/model-training",
)

service = ModelTrainingService()


@model_training_bp.post("/job-profiles/<job_profile_id>/train")
def train_job_profile(job_profile_id: str):
    payload = validate_training_payload(request.get_json())

    result = service.train_job_profile(
        job_profile_id=job_profile_id,
        created_by=payload["created_by"],
        dataset_version=payload["dataset_version"],
        persist_to_storage=payload["persist_to_storage"],
        auto_build_dataset=payload["auto_build_dataset"],
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Model training completed successfully",
                "data": result,
            }
        ),
        200,
    )
