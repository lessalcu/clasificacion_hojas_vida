from app.services.supabase_client import get_supabase


class DatasetSampleRepository:
    table_name = "dataset_sample"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return response.data[0] if response.data else None

    def update(self, item_id: str, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name).update(payload).eq("id", item_id).execute()
        )
        return response.data[0] if response.data else None

    def find_by_candidate_profile_and_job(
        self,
        candidate_profile_id: str,
        job_profile_id: str,
    ) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("candidate_profile_id", candidate_profile_id)
            .eq("job_profile_id", job_profile_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def get_rows(
        self,
        job_profile_id: str | None = None,
        split_names: list[str] | None = None,
        ready_only: bool = False,
        labeled_only: bool = False,
    ) -> list[dict]:
        supabase = get_supabase()
        query = supabase.table(self.table_name).select("*")

        if job_profile_id:
            query = query.eq("job_profile_id", job_profile_id)

        if split_names:
            query = query.in_("split_name", split_names)

        if ready_only:
            query = query.eq("quality_status", "ready")

        response = query.execute()
        rows = response.data or []

        if labeled_only:
            rows = [row for row in rows if row.get("label") is not None]

        return rows

    def count_ready_labeled_by_job(self, job_profile_id: str) -> int:
        rows = self.get_rows(
            job_profile_id=job_profile_id,
            ready_only=True,
            labeled_only=True,
        )
        return len(rows)
