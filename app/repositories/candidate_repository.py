from app.services.supabase_client import get_supabase


class CandidateRepository:
    table_name = "candidate"

    def create(self, payload: dict):
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return response.data[0] if response.data else None