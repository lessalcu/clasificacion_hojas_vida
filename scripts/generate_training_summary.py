import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app import create_app
from app.services.training_report_service import TrainingReportService


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Uso: py scripts/generate_training_summary.py <job_profile_id>")

    job_profile_id = sys.argv[1]

    app = create_app()
    with app.app_context():
        service = TrainingReportService()
        result = service.build_all_executions_report(
            job_profile_id=job_profile_id,
            persist_to_storage=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
