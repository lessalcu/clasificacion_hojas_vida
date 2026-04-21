from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.dataset_sample_repository import DatasetSampleRepository
from app.repositories.job_profile_repository import JobProfileRepository


class DatasetLabelingService:
    VALID_LABELS = {
        "relevante",
        "no_relevante",
        "preseleccionado",
        "no_preseleccionado",
    }

    VALID_SPLITS = {
        "train",
        "validation",
        "test",
    }

    def __init__(self):
        self.dataset_sample_repository = DatasetSampleRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()

    def label_candidate(
        self,
        job_profile_id: str,
        candidate_id: str,
        label: str,
        rubric: dict | None = None,
        notes: str | None = None,
        label_reason: str | None = None,
        split_name: str | None = "train",
        labeled_by: str | None = None,
        processing_run_id: str | None = None,
    ):
        label = (label or "").strip().lower()
        if label not in self.VALID_LABELS:
            raise ValidationError("Invalid label")

        split_name = (split_name or "train").strip().lower()
        if split_name not in self.VALID_SPLITS:
            raise ValidationError("Invalid split_name")

        job_profile = self.job_profile_repository.get_by_id(job_profile_id)
        if not job_profile:
            raise NotFoundError("Job profile not found")

        profile = self.candidate_profile_repository.get_by_candidate_id(candidate_id)
        if not profile:
            raise NotFoundError("Candidate profile not found")

        payload = {
            "candidate_profile_id": profile["id"],
            "job_profile_id": job_profile_id,
            "candidate_id": candidate_id,
            "source_id": profile.get("source_id"),
            "processing_run_id": processing_run_id,
            "pseudonym_code": profile.get("pseudonym_code"),
            "label": label,
            "label_reason": label_reason,
            "split_name": split_name,
            "rubric": rubric or {},
            "notes": notes,
            "labeled_by": labeled_by,
        }

        sample = self.dataset_sample_repository.upsert_label(
            job_profile_id=job_profile_id,
            candidate_profile_id=profile["id"],
            payload=payload,
        )

        return {
            "status": "labeled",
            "dataset_sample": sample,
        }