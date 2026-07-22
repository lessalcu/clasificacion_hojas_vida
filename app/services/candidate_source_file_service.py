from app.errors.exceptions import (
    NotFoundError,
    ValidationError,
)
from app.repositories.candidate_source_repository import (
    CandidateSourceRepository,
)
from app.services.storage_service import StorageService


class CandidateSourceFileService:
    def __init__(self):
        self.candidate_source_repository = (
            CandidateSourceRepository()
        )
        self.storage_service = StorageService()

    def get_pdf_file(self, source_id: str) -> dict:
        source = (
            self.candidate_source_repository.get_by_id(
                source_id
            )
        )

        if not source:
            raise NotFoundError(
                "Candidate source not found"
            )

        if source.get("source_type") != "pdf":
            raise ValidationError(
                "Only PDF candidate sources can be downloaded"
            )

        storage_bucket = source.get("storage_bucket")
        storage_path = source.get("storage_path")

        if not storage_bucket or not storage_path:
            raise ValidationError(
                "Candidate source does not have "
                "a valid storage location"
            )

        file_bytes = self.storage_service.download_file(
            bucket_name=storage_bucket,
            remote_path=storage_path,
        )

        if not isinstance(
            file_bytes,
            (bytes, bytearray),
        ):
            raise ValidationError(
                "The stored PDF could not be downloaded"
            )

        filename = (
            source.get("original_filename")
            or "candidate-cv.pdf"
        ).strip()

        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        return {
            "source": source,
            "filename": filename,
            "content_type": (
                source.get("mime_type")
                or "application/pdf"
            ),
            "file_bytes": bytes(file_bytes),
        }