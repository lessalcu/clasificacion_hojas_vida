from flask import Blueprint, jsonify, request

from app.services.model_training_service import ModelTrainingService
from app.services.training_report_service import TrainingReportService
from app.validators.training_validator import validate_training_payload

model_training_bp = Blueprint(
    "model_training",
    __name__,
    url_prefix="/api/v1/model-training",
)

service = ModelTrainingService()
training_report_service = TrainingReportService()


@model_training_bp.post("/job-profiles/<job_profile_id>/train")
def train_job_profile(job_profile_id: str):
    raw_payload = request.get_json() or {}
    payload = validate_training_payload(raw_payload)

    result = service.train_job_profile(
        job_profile_id=job_profile_id,
        created_by=payload["created_by"],
        dataset_version=payload["dataset_version"],
        persist_to_storage=payload["persist_to_storage"],
        auto_build_dataset=payload["auto_build_dataset"],
        random_state=raw_payload.get("random_state"),
        iteration_number=raw_payload.get("iteration_number", 1),
        experiment_run_id=raw_payload.get("experiment_run_id"),
        save_training_report=bool(raw_payload.get("save_training_report", True)),
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


@model_training_bp.post("/job-profiles/<job_profile_id>/train-iterations")
def train_job_profile_iterations(job_profile_id: str):
    payload = request.get_json() or {}

    result = service.train_job_profile_iterations(
        job_profile_id=job_profile_id,
        iterations=payload.get("iterations", 50),
        created_by=payload.get("created_by"),
        dataset_version_prefix=payload.get("dataset_version_prefix", "experiment"),
        persist_to_storage=bool(payload.get("persist_to_storage", True)),
        auto_build_dataset=bool(payload.get("auto_build_dataset", True)),
        base_random_state=payload.get("base_random_state"),
        start_iteration=payload.get("start_iteration", 1),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Training iterations completed successfully",
                "data": result,
            }
        ),
        200,
    )


@model_training_bp.get("/job-profiles/<job_profile_id>/all-executions")
def get_all_training_executions(job_profile_id: str):
    limit = request.args.get("limit")

    result = training_report_service.build_all_executions_report(
        job_profile_id=job_profile_id,
        persist_to_storage=False,
        limit=int(limit) if limit else None,
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Training executions retrieved successfully",
                "data": result,
            }
        ),
        200,
    )


@model_training_bp.post("/job-profiles/<job_profile_id>/all-executions/report")
def generate_all_training_executions_report(job_profile_id: str):
    payload = request.get_json() or {}

    result = training_report_service.build_all_executions_report(
        job_profile_id=job_profile_id,
        persist_to_storage=bool(payload.get("persist_to_storage", True)),
        limit=payload.get("limit"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Training executions summary report generated successfully",
                "data": result,
            }
        ),
        200,
    )
