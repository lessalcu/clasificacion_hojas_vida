from flask import Blueprint, jsonify, request

from app.services.auto_training_service import AutoTrainingService

auto_training_bp = Blueprint(
    "auto_training",
    __name__,
    url_prefix="/api/v1/auto-training",
)

service = AutoTrainingService()


@auto_training_bp.post("/job-profiles/<job_profile_id>/check-and-run")
def check_and_run_auto_training(job_profile_id: str):
    payload = request.get_json() or {}

    result = service.check_and_train_if_needed(
        job_profile_id=job_profile_id,
        created_by=payload.get("created_by"),
        enabled=bool(payload.get("enabled", True)),
        force=bool(payload.get("force", False)),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Auto training check completed successfully",
                "data": result,
            }
        ),
        200,
    )
