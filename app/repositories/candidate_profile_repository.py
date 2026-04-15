from app.services.supabase_client import get_supabase


class CandidateProfileRepository:
    table_name = "candidate_profile"

    def get_by_source_id(self, source_id: str):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("source_id", source_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def create(self, payload: dict):
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return response.data[0] if response.data else None

    def update_by_source_id(self, source_id: str, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .update(payload)
            .eq("source_id", source_id)
            .execute()
        )
        return response.data[0] if response.data else None