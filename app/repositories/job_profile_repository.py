from app.services.supabase_client import get_supabase


class JobProfileRepository:
    table_name = "job_profile"

    def create(self, payload: dict):
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return response.data[0] if response.data else None

    def get_all(self, limit: int = 10):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    def get_by_id(self, item_id: str):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("id", item_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def update(self, item_id: str, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .update(payload)
            .eq("id", item_id)
            .execute()
        )
        return response.data[0] if response.data else None

    def delete(self, item_id: str):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .delete()
            .eq("id", item_id)
            .execute()
        )
        return response.data[0] if response.data else None