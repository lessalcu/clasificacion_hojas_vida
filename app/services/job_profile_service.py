from app.repositories.job_profile_repository import JobProfileRepository
from app.validators.job_profile_validator import validate_job_profile_payload


class JobProfileService:
    def __init__(self):
        self.repository = JobProfileRepository()

    def create_job_profile(self, payload: dict):
        validated_payload = validate_job_profile_payload(payload, partial=False)
        created = self.repository.create(validated_payload)

        if not created:
            raise ValueError("Job profile could not be created")

        return created

    def get_job_profiles(self, limit: int = 10):
        return self.repository.get_all(limit=limit)

    def get_job_profile_by_id(self, item_id: str):
        job_profile = self.repository.get_by_id(item_id)

        if not job_profile:
            raise ValueError("Job profile not found")

        return job_profile

    def update_job_profile(self, item_id: str, payload: dict):
        existing = self.repository.get_by_id(item_id)

        if not existing:
            raise ValueError("Job profile not found")

        validated_payload = validate_job_profile_payload(payload, partial=True)

        if not validated_payload:
            raise ValueError("At least one valid field must be provided for update")

        updated = self.repository.update(item_id, validated_payload)

        if not updated:
            raise ValueError("Job profile could not be updated")

        return updated

    def delete_job_profile(self, item_id: str):
        existing = self.repository.get_by_id(item_id)

        if not existing:
            raise ValueError("Job profile not found")

        deleted = self.repository.delete(item_id)

        if not deleted:
            raise ValueError("Job profile could not be deleted")

        return deleted