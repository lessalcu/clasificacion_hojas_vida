from app.services.supabase_client import get_supabase


class TrainingRunRepository:
    table_name = "training_run"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return self._normalize_response(response.data)

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
