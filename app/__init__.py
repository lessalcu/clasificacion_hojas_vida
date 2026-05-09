import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from app.api.candidate_profile_routes import candidate_profile_bp
from app.api.cv_upload_routes import cv_upload_bp
from app.api.dataset_sample_routes import dataset_sample_bp
from app.api.health import health_bp
from app.api.job_profile_routes import job_profile_bp
from app.api.model_training_routes import model_training_bp
from app.api.model_validation_routes import model_validation_bp
from app.api.pdf_extraction_routes import pdf_extraction_bp
from app.api.model_version_routes import model_version_bp
from app.config import config_by_name
from app.errors.handlers import register_error_handlers
from app.utils.logger import configure_logging


def create_app():
    load_dotenv()

    env = os.getenv("FLASK_ENV", "development")
    config_class = config_by_name.get(env, config_by_name["development"])

    configure_logging()

    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, resources={r"/api/*": {"origins": app.config["FRONTEND_URL"]}})

    app.register_blueprint(health_bp)
    app.register_blueprint(job_profile_bp)
    app.register_blueprint(cv_upload_bp)
    app.register_blueprint(pdf_extraction_bp)
    app.register_blueprint(candidate_profile_bp)
    app.register_blueprint(dataset_sample_bp)
    app.register_blueprint(model_training_bp)
    app.register_blueprint(model_validation_bp)
    app.register_blueprint(model_version_bp)

    register_error_handlers(app)

    @app.get("/")
    def home():
        return {"success": True, "message": "Backend base initialized"}, 200

    app.logger.info("Application started in %s mode", env)

    return app
