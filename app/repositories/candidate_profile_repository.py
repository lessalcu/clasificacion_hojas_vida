from app.services.supabase_client import get_supabase


class CandidateProfileRepository:
    table_name = "candidate_profile"

    def get_by_ids(self, item_ids: list[str]):
        if not item_ids:
            return []

        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .in_("id", item_ids)
            .execute()
        )
        return response.data or []
