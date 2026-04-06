from flask import Blueprint, jsonify, request

from app.services.job_profile_service import JobProfileService

job_profile_bp = Blueprint("job_profile", __name__, url_prefix="/api/v1/job-profiles")
service = JobProfileService()


@job_profile_bp.post("")
def create_job_profile():
    payload = request.get_json() or {}
    result = service.create_job_profile(payload)

    return jsonify({
        "success": True,
        "message": "Job profile created successfully",
        "data": result
    }), 201


@job_profile_bp.get("")
def get_job_profiles():
    limit = request.args.get("limit", default=10, type=int)
    result = service.get_job_profiles(limit=limit)

    return jsonify({
        "success": True,
        "data": result
    }), 200


@job_profile_bp.get("/<item_id>")
def get_job_profile_by_id(item_id: str):
    result = service.get_job_profile_by_id(item_id)

    return jsonify({
        "success": True,
        "data": result
    }), 200


@job_profile_bp.put("/<item_id>")
def update_job_profile(item_id: str):
    payload = request.get_json() or {}
    result = service.update_job_profile(item_id, payload)

    return jsonify({
        "success": True,
        "message": "Job profile updated successfully",
        "data": result
    }), 200


@job_profile_bp.delete("/<item_id>")
def delete_job_profile(item_id: str):
    result = service.delete_job_profile(item_id)

    return jsonify({
        "success": True,
        "message": "Job profile deleted successfully",
        "data": result
    }), 200