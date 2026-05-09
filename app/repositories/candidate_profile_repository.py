from app.services.supabase_client import get_supabase


class CandidateProfileRepository:
    table_name = "candidate_profile"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return response.data[0] if response.data else None

    def update(self, item_id: str, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name).update(payload).eq("id", item_id).execute()
        )
        return response.data[0] if response.data else None

    def update_by_source_id(self, source_id: str, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .update(payload)
            .eq("source_id", source_id)
            .execute()
        )
        return response.data[0] if response.data else None

    def get_by_id(self, item_id: str) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("id", item_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def get_by_source_id(self, source_id: str) -> dict | None:
        supabase = get_supabase()
        response = (
            supabase.table(self.table_name)
            .select("*")
            .eq("source_id", source_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def get_by_ids(self, item_ids: list[str]) -> list[dict]:
        if not item_ids:
            return []

        supabase = get_supabase()
        response = (
            supabase.table(self.table_name).select("*").in_("id", item_ids).execute()
        )
        return response.data or []

    def get_profiles_for_dataset(self, limit: int = 500) -> list[dict]:
        supabase = get_supabase()
        response = supabase.table(self.table_name).select("*").limit(limit).execute()

        profiles = response.data or []

        valid_profiles = []

        for profile in profiles:
            if self._is_valid_profile(profile):
                valid_profiles.append(profile)

        return valid_profiles

    def get_valid_profiles_created_after(
        self,
        created_after: str | None = None,
        limit: int = 10000,
    ) -> list[dict]:
        supabase = get_supabase()

        query = (
            supabase.table(self.table_name)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )

        if created_after:
            query = query.gt("created_at", created_after)

        response = query.execute()
        profiles = response.data or []

        valid_profiles = []

        for profile in profiles:
            if self._is_valid_profile(profile):
                valid_profiles.append(profile)

        return valid_profiles

    def count_valid_profiles_created_after(
        self,
        created_after: str | None = None,
        limit: int = 10000,
    ) -> int:
        profiles = self.get_valid_profiles_created_after(
            created_after=created_after,
            limit=limit,
        )

        return len(profiles)

    @staticmethod
    def _is_valid_profile(profile: dict) -> bool:
        raw_text = (profile.get("raw_text") or "").strip()
        normalized_text = (profile.get("normalized_text") or "").strip()
        experience_summary = (profile.get("experience_summary") or "").strip()
        education_summary = (profile.get("education_summary") or "").strip()

        skills = profile.get("skills") or []
        technologies = profile.get("technologies") or []
        languages = profile.get("languages") or []
        keywords = profile.get("keywords") or []

        has_text = bool(raw_text or normalized_text)
        has_summary = bool(experience_summary or education_summary)
        has_structured_data = bool(skills or technologies or languages or keywords)

        return has_text or has_summary or has_structured_data
