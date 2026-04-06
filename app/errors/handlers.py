from flask import jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.errors.exceptions import NotFoundError, ValidationError


def register_error_handlers(app):
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": 400,
                "name": "BadRequest",
                "message": str(error)
            }
        }), 400

    @app.errorhandler(NotFoundError)
    def handle_not_found_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": 404,
                "name": "NotFound",
                "message": str(error)
            }
        }), 404

    @app.errorhandler(ValueError)
    def handle_value_error(error):
        return jsonify({
            "success": False,
            "error": {
                "code": 400,
                "name": "BadRequest",
                "message": str(error)
            }
        }), 400

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": {
                "code": 413,
                "name": "RequestEntityTooLarge",
                "message": "Request size exceeds the configured limit"
            }
        }), 413

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        return jsonify({
            "success": False,
            "error": {
                "code": error.code,
                "name": error.name,
                "message": error.description
            }
        }), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        app.logger.exception("Unexpected error: %s", str(error))
        return jsonify({
            "success": False,
            "error": {
                "code": 500,
                "name": type(error).__name__,
                "message": str(error)
            }
        }), 500