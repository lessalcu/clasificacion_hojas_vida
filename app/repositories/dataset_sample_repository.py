from app.services.supabase_client import get_supabase


class DatasetSampleRepository:
    TABLE_NAME = "dataset_sample"

    def get_by_job_and_candidate_profile(
        self,
        job_profile_id: str,
        candidate_profile_id: str,
    ):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .eq("candidate_profile_id", candidate_profile_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def create(self, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .insert(payload)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def update_by_id(self, sample_id: str, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .update(payload)
            .eq("id", sample_id)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def upsert_label(
        self,
        job_profile_id: str,
        candidate_profile_id: str,
        payload: dict,
    ):
        existing = self.get_by_job_and_candidate_profile(
            job_profile_id,
            candidate_profile_id,
        )
        if existing:
            return self.update_by_id(existing["id"], payload)
        return self.create(payload)