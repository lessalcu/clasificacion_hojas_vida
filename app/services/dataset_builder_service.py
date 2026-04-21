import csv
import io
import json
from datetime import datetime, timezone

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.job_profile_repository import JobProfileRepository
from app.repositories.processing_run_repository import ProcessingRunRepository
from app.services.storage_service import StorageService


class DatasetBuilderService:
    DATASET_BUCKET = "datasets"

    def __init__(self):
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()
        self.processing_run_repository = ProcessingRunRepository()
        self.storage_service = StorageService()

    def build_dataset(self, job_profile_id: str, export_format: str = "csv"):
        export_format = export_format.lower().strip()
        if export_format not in {"csv", "json"}:
            raise ValidationError("export_format must be csv or json")

        job_profile = self.job_profile_repository.get_by_id(job_profile_id)
        if not job_profile:
            raise NotFoundError("Job profile not found")

        run = self.processing_run_repository.create(
            {
                "job_profile_id": job_profile_id,
                "model_version_id": None,
                "created_by": None,
                "input_type": "candidate_profile_dataset",
                "status": "running",
                "total_candidates": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "artifact_bucket": None,
                "artifact_path": None,
                "error_message": None,
            }
        )

        try:
            profiles = self.candidate_profile_repository.list_all(limit=5000)
            if not profiles:
                raise ValidationError("No candidate profiles available")

            rows = [
                self._build_row(profile, job_profile)
                for profile in profiles
                if profile.get("pseudonym_code")
            ]

            if not rows:
                raise ValidationError(
                    "No candidate profiles with pseudonym_code available"
                )

            artifact_bytes, extension, content_type = self._serialize_rows(
                rows,
                export_format,
            )

            artifact_path = f"{job_profile_id}/{run['id']}/dataset_v1.{extension}"

            upload_result = self.storage_service.upload_bytes(
                bucket_name=self.DATASET_BUCKET,
                remote_path=artifact_path,
                data=artifact_bytes,
                content_type=content_type,
                upsert=True,
            )

            updated = self.processing_run_repository.update(
                run["id"],
                {
                    "status": "completed",
                    "total_candidates": len(rows),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "artifact_bucket": self.DATASET_BUCKET,
                    "artifact_path": artifact_path,
                    "error_message": None,
                },
            )

            return {
                "status": "completed",
                "processing_run": updated,
                "dataset": {
                    "bucket": self.DATASET_BUCKET,
                    "path": artifact_path,
                    "rows": len(rows),
                    "format": export_format,
                    "full_path": getattr(upload_result, "full_path", None),
                },
                "preview": rows[:5],
            }

        except Exception as error:
            self.processing_run_repository.update(
                run["id"],
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": str(error),
                },
            )
            raise

    def _build_row(self, profile: dict, job_profile: dict):
        return {
            "job_profile_id": job_profile["id"],
            "candidate_id": profile["candidate_id"],
            "source_id": profile.get("source_id"),
            "pseudonym_code": profile.get("pseudonym_code"),
            "source_type": profile.get("source_type"),
            "candidate_skills_text": self._join_named_items(profile.get("skills")),
            "candidate_technologies_text": self._join_named_items(profile.get("technologies")),
            "candidate_languages_text": self._join_language_items(profile.get("languages")),
            "candidate_keywords_text": self._join_plain_items(profile.get("keywords")),
            "candidate_experience_summary": profile.get("experience_summary") or "",
            "candidate_education_summary": profile.get("education_summary") or "",
            "candidate_normalized_text": profile.get("normalized_text") or "",
            "job_title": job_profile.get("title") or "",
            "job_required_skills_text": self._join_plain_items(job_profile.get("required_skills")),
            "job_technologies_text": self._join_plain_items(job_profile.get("technologies")),
            "job_languages_text": self._join_plain_items(job_profile.get("languages")),
            "job_experience_requirement": job_profile.get("experience_requirement") or "",
            "job_education_requirement": job_profile.get("education_requirement") or "",
        }

    def _join_named_items(self, items):
        if not items:
            return ""
        values = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("canonical")
                if name:
                    values.append(str(name))
            else:
                values.append(str(item))
        return " | ".join(values)

    def _join_language_items(self, items):
        if not items:
            return ""
        values = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("canonical")
                level = item.get("level")
                if name and level:
                    values.append(f"{name}:{level}")
                elif name:
                    values.append(str(name))
            else:
                values.append(str(item))
        return " | ".join(values)

    def _join_plain_items(self, items):
        if not items:
            return ""
        if isinstance(items, list):
            return " | ".join(str(x) for x in items)
        return str(items)

    def _serialize_rows(self, rows: list[dict], export_format: str):
        if export_format == "json":
            return (
                json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
                "json",
                "application/json",
            )

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

        return (
            buffer.getvalue().encode("utf-8"),
            "csv",
            "text/csv",
        )