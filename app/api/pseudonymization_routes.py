from flask import Blueprint, jsonify

from app.services.pseudonymization_service import PseudonymizationService

pseudonymization_bp = Blueprint("pseudonymization", __name__)
service = PseudonymizationService()


@pseudonymization_bp.route(
    "/api/v1/candidate-sources/<source_id>/pseudonymize",
    methods=["POST"],
)
def pseudonymize_candidate_source(source_id):
    result = service.pseudonymize_source(source_id)

    return jsonify(
        {
            "success": True,
            "message": "Candidate source pseudonymized successfully",
            "data": result,
        }
    ), 200