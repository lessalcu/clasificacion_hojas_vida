import os


class Config:
    APP_NAME = "cv-classifier-api"
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
    SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")

    MAX_CV_FILE_SIZE_MB = int(os.getenv("MAX_CV_FILE_SIZE_MB", 6))
    MAX_CV_FILE_SIZE_BYTES = MAX_CV_FILE_SIZE_MB * 1024 * 1024
    MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", 20))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_REQUEST_SIZE_MB", 30)) * 1024 * 1024

class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}