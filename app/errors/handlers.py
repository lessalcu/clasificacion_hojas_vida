from flask import jsonify
from werkzeug.exceptions import HTTPException


def register_error_handlers(app):
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