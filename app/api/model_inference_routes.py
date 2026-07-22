from flask import Blueprint, jsonify, request

from app.services.candidate_results_service import (
    CandidateResultsService,
)
from app.services.model_inference_service import (
    ModelInferenceService,
)

model_inference_bp = Blueprint(
    "model_inference",
    __name__,
    url_prefix="/api/v1/inference",
)

service = ModelInferenceService()
results_service = CandidateResultsService()

@model_inference_bp.post(
    "/job-profiles/<job_profile_id>/candidate-profiles/<candidate_profile_id>"
)
def classify_candidate_profile(job_profile_id: str, candidate_profile_id: str):
    payload = request.get_json() or {}

    result = service.classify_candidate_profile(
        job_profile_id=job_profile_id,
        candidate_profile_id=candidate_profile_id,
        created_by=payload.get("created_by"),
        auto_retrain=bool(payload.get("auto_retrain", True)),
        persist_result=bool(payload.get("persist_result", True)),
        top_k=payload.get("top_k"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Candidate classified successfully",
                "data": result,
            }
        ),
        200,
    )


@model_inference_bp.post("/job-profiles/<job_profile_id>/batch")
def classify_candidate_profiles(job_profile_id: str):
    payload = request.get_json() or {}

    result = service.classify_candidate_profiles(
        job_profile_id=job_profile_id,
        candidate_profile_ids=payload.get("candidate_profile_ids") or [],
        created_by=payload.get("created_by"),
        auto_retrain=bool(payload.get("auto_retrain", True)),
        persist_result=bool(payload.get("persist_result", True)),
        top_k=payload.get("top_k"),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Candidates classified successfully",
                "data": result,
            }
        ),
        200,
    )


@model_inference_bp.post("/job-profiles/<job_profile_id>/rank-all")
def rank_all_candidate_profiles(job_profile_id: str):
    payload = request.get_json() or {}

    result = service.rank_all_candidate_profiles(
        job_profile_id=job_profile_id,
        created_by=payload.get("created_by"),
        auto_retrain=bool(payload.get("auto_retrain", False)),
        persist_result=bool(payload.get("persist_result", True)),
        top_k=payload.get("top_k"),
        max_candidates=payload.get("max_candidates", 1000),
        reuse_existing=bool(payload.get("reuse_existing", True)),
        force_new_run=bool(payload.get("force_new_run", False)),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Candidate ranking generated successfully",
                "data": result,
            }
        ),
        200,
    )


@model_inference_bp.get("/processing-runs/<processing_run_id>/ranking")
def get_ranking_by_processing_run(processing_run_id: str):
    limit = request.args.get("limit") or request.args.get("top_k")

    result = service.get_ranking_by_processing_run(
        processing_run_id=processing_run_id,
        limit=limit,
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Ranking retrieved successfully",
                "data": result,
            }
        ),
        200,
    )

@model_inference_bp.get(
    "/processing-runs/"
    "<processing_run_id>/results"
)
def get_enriched_results_by_processing_run(
    processing_run_id: str,
):
    limit = (
        request.args.get("limit")
        or request.args.get("top_k")
    )

    result = (
        results_service
        .get_processing_run_results(
            processing_run_id=(
                processing_run_id
            ),
            limit=limit,
        )
    )

    return jsonify(
        {
            "success": True,
            "message": (
                "Candidate results retrieved successfully"
            ),
            "data": result,
        }
    ), 200