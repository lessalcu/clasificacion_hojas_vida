from app.services.supabase_client import get_supabase


class ProcessingRunRepository:
    TABLE_NAME = "processing_run"

    def create(self, payload: dict):
        supabase = get_supabase()
        response = supabase.table(self.TABLE_NAME).insert(payload).execute()
        data = response.data or []
        return data[0] if data else None

    def update(self, run_id: str, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .update(payload)
            .eq("id", run_id)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def get_by_id(self, run_id: str):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .select("*")
            .eq("id", run_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None