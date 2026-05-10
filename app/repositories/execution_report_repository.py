from app.services.supabase_client import get_supabase


class ExecutionReportRepository:
    table_name = "execution_report"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return self._normalize_response(response.data)

    def get_by_id(self, item_id: str) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("id", item_id)
            .limit(1)
            .execute()
        )
        return self._normalize_response(response.data)

    def list_by_processing_run(self, processing_run_id: str) -> list[dict]:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("processing_run_id", processing_run_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []

    def list_by_job_profile(self, job_profile_id: str, limit: int = 100) -> list[dict]:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
