from app.services.supabase_client import get_supabase


class CandidatePiiRepository:
    TABLE_NAME = "candidate_pii"

    def get_by_candidate_id(self, candidate_id: str):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .select("*")
            .eq("candidate_id", candidate_id)
            .limit(1)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def create(self, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .insert(payload)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def update_by_candidate_id(self, candidate_id: str, payload: dict):
        supabase = get_supabase()
        response = (
            supabase.table(self.TABLE_NAME)
            .update(payload)
            .eq("candidate_id", candidate_id)
            .execute()
        )
        data = response.data or []
        return data[0] if data else None

    def upsert_by_candidate_id(self, candidate_id: str, payload: dict):
        existing = self.get_by_candidate_id(candidate_id)
        if existing:
            return self.update_by_candidate_id(candidate_id, payload)
        return self.create(payload)