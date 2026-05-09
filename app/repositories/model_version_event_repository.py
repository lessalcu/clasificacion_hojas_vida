from app.services.supabase_client import get_supabase


class ModelVersionEventRepository:
    table_name = "model_version_event"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return self._normalize_response(response.data)

    def list_by_model_version(self, model_version_id: str) -> list[dict]:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("model_version_id", model_version_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
