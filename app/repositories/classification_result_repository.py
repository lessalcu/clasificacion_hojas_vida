from app.services.supabase_client import get_supabase


class ClassificationResultRepository:
    table_name = "classification_result"

    def create(self, payload: dict) -> dict | None:
        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payload).execute()
        return self._normalize_response(response.data)

    def bulk_create(self, payloads: list[dict]) -> list[dict]:
        if not payloads:
            return []

        supabase = get_supabase()
        response = supabase.table(self.table_name).insert(payloads).execute()
        return response.data or []

    def list_by_processing_run(
        self,
        processing_run_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        supabase = get_supabase()

        query = (
            supabase.table(self.table_name)
            .select("*")
            .eq("processing_run_id", processing_run_id)
            .order("rank_position", desc=False)
        )

        if limit is not None and limit > 0:
            query = query.limit(limit)

        response = query.execute()
        return response.data or []

    def count_by_processing_run(self, processing_run_id: str) -> int:
        results = self.list_by_processing_run(processing_run_id=processing_run_id)
        return len(results)

    @staticmethod
    def _normalize_response(data):
        if isinstance(data, list):
            return data[0] if data else None
        return data
