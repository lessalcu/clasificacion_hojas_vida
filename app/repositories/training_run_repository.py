from datetime import UTC, datetime

from app.services.supabase_client import get_supabase


class TrainingRunRepository:
    table_name = "training_run"

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

    def list_by_job_profile(self, job_profile_id: str) -> list[dict]:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []

    def get_latest_by_job_profile(self, job_profile_id: str) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return self._normalize_response(response.data)

    def get_by_model_version_ids(self, model_version_ids: list[str]) -> list[dict]:
        if not model_version_ids:
            return []

        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .in_("model_version_id", model_version_ids)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []

    def mark_selected_by_model_version(
        self,
        job_profile_id: str,
        model_version_id: str,
    ) -> list[dict]:
        supabase = get_supabase()
        now = datetime.now(UTC).isoformat()

        supabase.table(self.table_name).update(
            {
                "is_selected": False,
                "updated_at": now,
            }
        ).eq("job_profile_id", job_profile_id).execute()

        response = (
            supabase.table(self.table_name)
            .update(
                {
                    "is_selected": True,
                    "updated_at": now,
                }
            )
            .eq("job_profile_id", job_profile_id)
            .eq("model_version_id", model_version_id)
            .execute()
        )

        return response.data or []

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data