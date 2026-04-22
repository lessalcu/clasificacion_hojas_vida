import uuid
from werkzeug.utils import secure_filename

from app.errors.exceptions import ValidationError
from app.repositories.candidate_repository import CandidateRepository
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.storage_service import StorageService
from app.validators.cv_upload_validator import validate_pdf_file


class CvUploadService:
    def __init__(self, max_size_bytes: int, max_batch_files: int):
        self.max_size_bytes = max_size_bytes
        self.max_batch_files = max_batch_files
        self.storage_service = StorageService()
        self.candidate_repository = CandidateRepository()
        self.candidate_source_repository = CandidateSourceRepository()

    def upload_single(self, file_storage):
        validated = validate_pdf_file(file_storage, self.max_size_bytes)

        if hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
            file_storage.stream.seek(0)
            
        candidate = self.candidate_repository.create({
            "source_label": "pdf_upload"
        })

        if not candidate:
            raise ValidationError("Candidate could not be created")

        candidate_id = candidate["id"]

        safe_name = secure_filename(validated["original_filename"])
        unique_name = f"{uuid.uuid4()}_{safe_name}"
        storage_path = f"{candidate_id}/{unique_name}"

        upload_result = self.storage_service.upload_fileobj(
            bucket_name="cv-raw",
            remote_path=storage_path,
            file_obj=file_storage,
            content_type=validated["mime_type"],
            upsert=False,
        )
        candidate_source = self.candidate_source_repository.create({
            "candidate_id": candidate_id,
            "source_type": "pdf",
            "original_filename": validated["original_filename"],
            "storage_bucket": "cv-raw",
            "storage_path": storage_path,
            "mime_type": validated["mime_type"],
            "size_bytes": validated["file_size_bytes"],
            "extraction_status": "pending",
        })

        return {
            "candidate": candidate,
            "candidate_source": candidate_source,
            "storage": {
                "path": getattr(upload_result, "path", storage_path),
                "full_path": getattr(upload_result, "full_path", None),
            },
        }

    def upload_batch(self, files):
        if not files:
            raise ValidationError("At least one file is required")

        if len(files) > self.max_batch_files:
            raise ValidationError(
                f"Batch exceeds the maximum allowed number of files: {self.max_batch_files}"
            )

        results = []
        for file_storage in files:
            try:
                results.append({
                    "success": True,
                    "result": self.upload_single(file_storage)
                })
            except Exception as error:
                results.append({
                    "success": False,
                    "filename": getattr(file_storage, "filename", None),
                    "error": str(error)
                })

        return results