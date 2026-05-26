from datetime import datetime, timezone
from pathlib import Path
import json

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.standard_cv_extractor import StandardCvExtractor
from app.services.storage_service import StorageService


class PdfTextExtractionService:
    def __init__(self):
        self.storage_service = StorageService()
        self.candidate_source_repository = CandidateSourceRepository()
        self.standard_cv_config = self._load_standard_cv_config()
        self.extractor = StandardCvExtractor(self.standard_cv_config)

    def extract_from_source(self, source_id: str):
        source = self.candidate_source_repository.get_by_id(source_id)

        if not source:
            raise NotFoundError("Candidate source not found")

        if source.get("source_type") != "pdf":
            raise ValidationError("Only PDF sources can be extracted")

        if not source.get("storage_bucket") or not source.get("storage_path"):
            raise ValidationError("Candidate source does not have a source file")

        pdf_bytes = self.storage_service.download_file(
            source["storage_bucket"],
            source["storage_path"],
        )

        extraction_result = self.extractor.extract(pdf_bytes)

        if not extraction_result["valid"]:
            self._mark_extraction_failed(
                source_id=source_id,
                error_message=extraction_result["message"],
            )
            raise ValidationError(extraction_result["message"])

        artifact_payload = extraction_result["artifact_payload"]

        artifact = {
            "candidate_id": source["candidate_id"],
            "candidate_source_id": source["id"],
            "source_type": source["source_type"],
            "original_filename": source.get("original_filename"),
            **artifact_payload,
        }

        artifact_path = f"{source['candidate_id']}/{source['id']}/v1.json"

        self.storage_service.upload_json(
            bucket_name="cv-normalized",
            remote_path=artifact_path,
            payload=artifact,
            upsert=True,
        )

        updated_source = self.candidate_source_repository.update(
            source["id"],
            {
                "extraction_status": "processed",
                "extracted_text_length": artifact["text_length"],
                "normalized_bucket": "cv-normalized",
                "normalized_path": artifact_path,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "extraction_error": None,
            },
        )

        return {
            "status": "processed",
            "candidate_source": updated_source,
            "artifact": {
                "bucket": "cv-normalized",
                "path": artifact_path,
                "page_count": artifact["page_count"],
                "text_length": artifact["text_length"],
                "quality_score": artifact["quality_score"],
                "sections_found": list(artifact["sections"].keys()),
            },
        }

    def _load_standard_cv_config(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "normalization_config.json"
        )

        if not config_path.exists():
            raise ValidationError(
                f"Normalization config not found at: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            full_config = json.load(file)

        standard_cv = full_config.get("standard_cv")
        if not standard_cv:
            raise ValidationError("standard_cv configuration is required")

        return standard_cv

    def _mark_extraction_failed(self, source_id: str, error_message: str):
        self.candidate_source_repository.update(
            source_id,
            {
                "extraction_status": "failed",
                "extracted_text_length": 0,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "extraction_error": error_message,
            },
        )