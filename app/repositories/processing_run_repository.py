from datetime import UTC, datetime

from app.services.supabase_client import get_supabase


class ProcessingRunRepository:
    table_name = "processing_run"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return self._normalize_response(response.data)

    def update(self, item_id: str, payload: dict) -> dict | None:
        supabase = get_supabase()
        payload["updated_at"] = datetime.now(UTC).isoformat()

        response = (
            supabase.table(self.table_name).update(payload).eq("id", item_id).execute()
        )
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

    def list_completed_rank_all(
        self,
        job_profile_id: str,
        model_version_id: str,
        limit: int = 20,
    ) -> list[dict]:
        supabase = get_supabase()

        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .eq("model_version_id", model_version_id)
            .eq("input_type", "rank_all")
            .eq("status", "completed")
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
