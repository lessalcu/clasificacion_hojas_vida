from flask import Blueprint, jsonify, request

from app.services.model_versioning_service import ModelVersioningService

model_version_bp = Blueprint(
    "model_version",
    __name__,
    url_prefix="/api/v1/model-versions",
)

service = ModelVersioningService()


@model_version_bp.get("/job-profiles/<job_profile_id>")
def list_versions_by_job_profile(job_profile_id: str):
    versions = service.list_versions_by_job_profile(job_profile_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Model versions retrieved successfully",
                "data": versions,
            }
        ),
        200,
    )


@model_version_bp.get("/job-profiles/<job_profile_id>/active")
def get_active_model(job_profile_id: str):
    active_model = service.get_active_model(job_profile_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Active model retrieved successfully",
                "data": active_model,
            }
        ),
        200,
    )


@model_version_bp.post("/<model_version_id>/activate")
def activate_model(model_version_id: str):
    payload = request.get_json() or {}

    active_model = service.activate_model(
        model_version_id=model_version_id,
        created_by=payload.get("created_by"),
        reason=payload.get("reason"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Model version activated successfully",
                "data": active_model,
            }
        ),
        200,
    )


@model_version_bp.get("/<model_version_id>/audit")
def list_audit_events(model_version_id: str):
    events = service.list_audit_events(model_version_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Model version audit events retrieved successfully",
                "data": events,
            }
        ),
        200,
    )
