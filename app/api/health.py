from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__, url_prefix="/api/v1")


@health_bp.get("/health")
def healthcheck():
    return jsonify({
        "success": True,
        "message": "API is running",
        "service": "cv-classifier-api"
    }), 200