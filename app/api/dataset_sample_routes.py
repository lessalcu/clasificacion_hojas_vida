from flask import Blueprint, jsonify, request

from app.services.dataset_builder_service import DatasetBuilderService

dataset_sample_bp = Blueprint(
    "dataset_sample",
    __name__,
    url_prefix="/api/v1/dataset-samples",
)

service = DatasetBuilderService()


@dataset_sample_bp.post("/build")
def build_dataset_samples():
    payload = request.get_json() or {}

    job_profile_id = payload.get("job_profile_id")
    if not job_profile_id:
        raise ValueError("Field 'job_profile_id' is required")

    result = service.build_for_job_profile(
        job_profile_id=job_profile_id,
        created_by=payload.get("created_by"),
        dataset_version=payload.get("dataset_version", "manual-build"),
        source_type=payload.get("source_type", "manual-build"),
        limit=int(payload.get("limit", 500)),
        overwrite_auto_labels=bool(payload.get("overwrite_auto_labels", True)),
        label_strategy=payload.get("label_strategy"),
        positive_ratio=payload.get("positive_ratio"),
        match_threshold=payload.get("match_threshold"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Dataset samples built successfully",
                "data": result,
            }
        ),
        201,
    )


@dataset_sample_bp.post("/ensure-for-training")
def ensure_dataset_for_training():
    payload = request.get_json() or {}

    job_profile_id = payload.get("job_profile_id")
    if not job_profile_id:
        raise ValueError("Field 'job_profile_id' is required")

    result = service.ensure_dataset_for_training(
        job_profile_id=job_profile_id,
        created_by=payload.get("created_by"),
        dataset_version=payload.get("dataset_version", "manual-ensure"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Dataset checked successfully",
                "data": result,
            }
        ),
        200,
    )
