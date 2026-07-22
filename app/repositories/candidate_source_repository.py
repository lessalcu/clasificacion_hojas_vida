from app.services.supabase_client import get_supabase


class CandidateSourceRepository:
    table_name = "candidate_source"

    def create(self, payload: dict):
        supabase = get_supabase()

        response = (
            supabase.table(self.table_name)
            .insert(payload)
            .execute()
        )

        return response.data[0] if response.data else None

    def get_by_id(self, source_id: str):
        supabase = get_supabase()

        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("id", source_id)
            .limit(1)
            .execute()
        )

        return response.data[0] if response.data else None

    def get_by_ids(self, source_ids: list[str]) -> list[dict]:
        if not source_ids:
            return []

        unique_source_ids = list(dict.fromkeys(source_ids))

        supabase = get_supabase()

        response = (
            supabase.table(self.table_name)
            .select("*")
            .in_("id", unique_source_ids)
            .execute()
        )

        return response.data or []

    def update(self, source_id: str, payload: dict):
        supabase = get_supabase()

        response = (
            supabase.table(self.table_name)
            .update(payload)
            .eq("id", source_id)
            .execute()
        )

        return response.data[0] if response.data else None