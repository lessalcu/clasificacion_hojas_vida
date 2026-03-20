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