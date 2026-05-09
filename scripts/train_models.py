import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app import create_app
from app.services.model_training_service import ModelTrainingService


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: py scripts/train_models.py <job_profile_id> [created_by]")

    job_profile_id = sys.argv[1]
    created_by = sys.argv[2] if len(sys.argv) > 2 else None

    app = create_app()
    with app.app_context():
        service = ModelTrainingService()
        result = service.train_job_profile(
            job_profile_id=job_profile_id,
            created_by=created_by,
            dataset_version="manual-script-run",
            persist_to_storage=True,
        )
        print(result)


if __name__ == "__main__":
    main()
