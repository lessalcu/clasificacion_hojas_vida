import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app import create_app
from app.services.model_training_service import ModelTrainingService


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Uso: py scripts/train_iterations.py <job_profile_id> [iterations] [created_by]"
        )

    job_profile_id = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    created_by = sys.argv[3] if len(sys.argv) > 3 else None

    app = create_app()
    with app.app_context():
        service = ModelTrainingService()
        result = service.train_job_profile_iterations(
            job_profile_id=job_profile_id,
            iterations=iterations,
            created_by=created_by,
            dataset_version_prefix="manual-iterations",
            persist_to_storage=True,
            auto_build_dataset=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
