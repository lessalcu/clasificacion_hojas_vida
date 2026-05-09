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

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
