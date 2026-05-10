from flask import Blueprint, jsonify, request

from app.services.execution_report_service import ExecutionReportService

report_bp = Blueprint(
    "reports",
    __name__,
    url_prefix="/api/v1/reports",
)

service = ExecutionReportService()


@report_bp.get("/processing-runs/<processing_run_id>/trace")
def get_execution_trace(processing_run_id: str):
    result = service.get_execution_trace(processing_run_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Execution trace retrieved successfully",
                "data": result,
            }
        ),
        200,
    )


@report_bp.post("/processing-runs/<processing_run_id>/generate")
def generate_execution_report(processing_run_id: str):
    payload = request.get_json() or {}

    result = service.generate_execution_report(
        processing_run_id=processing_run_id,
        created_by=payload.get("created_by"),
        report_format=payload.get("report_format", "json"),
        persist_to_storage=bool(payload.get("persist_to_storage", True)),
    )

    return (
        jsonify(
            {
                "success": True,
                "message": "Execution report generated successfully",
                "data": result,
            }
        ),
        200,
    )


@report_bp.get("/processing-runs/<processing_run_id>")
def list_reports_by_processing_run(processing_run_id: str):
    result = service.list_reports_by_processing_run(processing_run_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Execution reports retrieved successfully",
                "data": result,
            }
        ),
        200,
    )


@report_bp.get("/<report_id>")
def get_report_by_id(report_id: str):
    result = service.get_report_by_id(report_id)

    return (
        jsonify(
            {
                "success": True,
                "message": "Execution report retrieved successfully",
                "data": result,
            }
        ),
        200,
    )
