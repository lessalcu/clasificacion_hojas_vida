from flask import Blueprint, jsonify

from app.services.candidate_profile_service import CandidateProfileService

candidate_profile_bp = Blueprint(
    "candidate_profile",
    __name__,
    url_prefix="/api/v1/candidate-sources"
)

service = CandidateProfileService()


@candidate_profile_bp.post("/<source_id>/normalize-profile")
def normalize_profile(source_id: str):
    result = service.generate_profile_from_source(source_id)

    return jsonify({
        "success": True,
        "message": "Candidate profile generated successfully",
        "data": result,
    }), 200


@candidate_profile_bp.get("/<source_id>/profile")
def get_profile(source_id: str):
    result = service.get_profile_by_source(source_id)

    return jsonify({
        "success": True,
        "data": result,
    }), 200