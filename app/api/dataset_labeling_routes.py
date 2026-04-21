from flask import Blueprint, jsonify, request

from app.services.dataset_labeling_service import DatasetLabelingService

dataset_labeling_bp = Blueprint("dataset_labeling", __name__)
service = DatasetLabelingService()


@dataset_labeling_bp.route(
    "/api/v1/job-profiles/<job_profile_id>/candidates/<candidate_id>/label",
    methods=["POST"],
)
def label_candidate(job_profile_id, candidate_id):
    body = request.get_json(silent=True) or {}

    result = service.label_candidate(
        job_profile_id=job_profile_id,
        candidate_id=candidate_id,
        label=body.get("label"),
        rubric=body.get("rubric"),
        notes=body.get("notes"),
        label_reason=body.get("label_reason"),
        split_name=body.get("split_name", "train"),
        labeled_by=body.get("labeled_by"),
        processing_run_id=body.get("processing_run_id"),
    )

    return jsonify(
        {
            "success": True,
            "message": "Candidate labeled successfully",
            "data": result,
        }
    ), 201