from datetime import UTC, datetime

from app.services.supabase_client import get_supabase


class ModelVersionRepository:
    table_name = "model_version"

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

    def list_by_job_profile_and_dataset_version(
        self,
        job_profile_id: str,
        dataset_version: str,
    ) -> list[dict]:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .eq("dataset_version", dataset_version)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []

    def get_active_by_job_profile(self, job_profile_id: str) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("job_profile_id", job_profile_id)
            .eq("status", "active")
            .order("activated_at", desc=True)
            .limit(1)
            .execute()
        )
        return self._normalize_response(response.data)

    def archive_active_by_job_profile(self, job_profile_id: str) -> list[dict]:
        supabase = get_supabase()
        now = datetime.now(UTC).isoformat()

        response = (
            supabase.table(self.table_name)
            .update(
                {
                    "status": "archived",
                    "archived_at": now,
                    "updated_at": now,
                }
            )
            .eq("job_profile_id", job_profile_id)
            .in_("status", ["active", "selected"])
            .execute()
        )
        return response.data or []

    def mark_as_active(
        self,
        model_version_id: str,
        selection_reason: str | None = None,
    ) -> dict | None:
        return self.update(
            model_version_id,
            {
                "status": "active",
                "activated_at": datetime.now(UTC).isoformat(),
                "selection_reason": selection_reason,
            },
        )

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
