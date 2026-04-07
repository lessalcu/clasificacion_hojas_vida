import os

from dotenv import load_dotenv

load_dotenv()


def _get_tuple_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw_value = os.getenv(name)
    if not raw_value:
        return default

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        return default

    return int(parts[0]), int(parts[1])


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


    MODEL_ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "artifacts")
    MODEL_ARTIFACT_BUCKET = os.getenv("MODEL_ARTIFACT_BUCKET", "model-artifacts")
    REPORTS_BUCKET = os.getenv("REPORTS_BUCKET", "reports")

    TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", 0.20))
    TRAINING_RANDOM_STATE = int(os.getenv("TRAINING_RANDOM_STATE", 42))
    MIN_TRAINING_ROWS = int(os.getenv("MIN_TRAINING_ROWS", 10))
    TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
    TFIDF_NGRAM_RANGE = _get_tuple_env("TFIDF_NGRAM_RANGE", (1, 2))
    KNN_DEFAULT_NEIGHBORS = int(os.getenv("KNN_DEFAULT_NEIGHBORS", 5))
    TREE_MAX_DEPTH = int(os.getenv("TREE_MAX_DEPTH", 20))
    TREE_MIN_SAMPLES_LEAF = int(os.getenv("TREE_MIN_SAMPLES_LEAF", 2))


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
