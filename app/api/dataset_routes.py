from flask import Blueprint, jsonify, request

from app.services.dataset_builder_service import DatasetBuilderService

dataset_bp = Blueprint("dataset", __name__)
service = DatasetBuilderService()


@dataset_bp.route("/api/v1/job-profiles/<job_profile_id>/datasets/build", methods=["POST"])
def build_dataset(job_profile_id):
    body = request.get_json(silent=True) or {}
    export_format = body.get("format", "csv")

    result = service.build_dataset(job_profile_id, export_format=export_format)

    return jsonify(
        {
            "success": True,
            "message": "Dataset built successfully",
            "data": result,
        }
    ), 201