from app.repositories.job_profile_repository import JobProfileRepository


class JobProfileService:
    def __init__(self):
        self.repository = JobProfileRepository()

    def create_job_profile(self, payload: dict):
        self._validate_payload(payload)
        return self.repository.create(payload)

    def get_job_profiles(self, limit: int = 10):
        return self.repository.get_all(limit=limit)

    def get_job_profile_by_id(self, item_id: str):
        return self.repository.get_by_id(item_id)

    def update_job_profile(self, item_id: str, payload: dict):
        return self.repository.update(item_id, payload)

    def delete_job_profile(self, item_id: str):
        return self.repository.delete(item_id)

    def _validate_payload(self, payload: dict):
        if not payload.get("title"):
            raise ValueError("Field 'title' is required")

        payload.setdefault("required_skills", [])
        payload.setdefault("technologies", [])
        payload.setdefault("languages", [])