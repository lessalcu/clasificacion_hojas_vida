from app.services.supabase_client import get_supabase


class DatasetSampleRepository:
    table_name = "dataset_sample"

    def get_rows(
        self,
        job_profile_id: str | None = None,
        split_names: list[str] | None = None,
    ) -> list[dict]:
        supabase = get_supabase()
        query = supabase.table(self.table_name).select("*")

        if job_profile_id:
            query = query.eq("job_profile_id", job_profile_id)

        if split_names:
            query = query.in_("split_name", split_names)

        response = query.execute()
        return response.data or []
