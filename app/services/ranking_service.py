from __future__ import annotations


class RankingService:
    def generate_ranking(self, results: list[dict]) -> list[dict]:
        if not results:
            return []

        prepared_results = [
            self._prepare_item(item=item, input_order=index)
            for index, item in enumerate(results, start=1)
        ]

        valid_results = [item for item in prepared_results if not item.get("error")]

        self._normalize_scores(valid_results)

        ranked_results = sorted(
            prepared_results,
            key=self._ranking_key,
        )

        for position, item in enumerate(ranked_results, start=1):
            item["rank_position"] = position
            item.pop("_input_order", None)

        return ranked_results

    def _prepare_item(self, item: dict, input_order: int) -> dict:
        prepared = dict(item)

        score_detail = dict(prepared.get("score_detail") or {})

        raw_score = self._to_score(
            prepared.get("raw_score_0_100")
            if prepared.get("raw_score_0_100") is not None
            else prepared.get("score_0_100")
        )

        prepared["raw_score_0_100"] = raw_score
        prepared["score_0_100"] = raw_score
        prepared["_input_order"] = input_order

        score_detail["raw_final_score_0_100"] = raw_score

        prepared["score_detail"] = score_detail
        prepared["ranking_detail"] = {
            "input_order": input_order,
            "raw_score_0_100": raw_score,
            "normalization_method": "pending",
        }

        return prepared

    def _normalize_scores(self, valid_results: list[dict]) -> None:
        if not valid_results:
            return

        raw_scores = [
            self._to_score(item.get("raw_score_0_100")) for item in valid_results
        ]

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if len(valid_results) == 1 or min_score == max_score:
            for item in valid_results:
                raw_score = self._to_score(item.get("raw_score_0_100"))
                item["score_0_100"] = raw_score
                item["score_detail"]["normalized_score_0_100"] = raw_score
                item["score_detail"]["normalization_method"] = "raw_score_preserved"
                item["ranking_detail"]["normalized_score_0_100"] = raw_score
                item["ranking_detail"]["normalization_method"] = "raw_score_preserved"
            return

        for item in valid_results:
            raw_score = self._to_score(item.get("raw_score_0_100"))

            normalized_score = round(
                ((raw_score - min_score) / (max_score - min_score)) * 100,
                2,
            )

            item["score_0_100"] = normalized_score
            item["score_detail"]["normalized_score_0_100"] = normalized_score
            item["score_detail"]["normalization_method"] = "min_max_by_processing_run"
            item["ranking_detail"]["normalized_score_0_100"] = normalized_score
            item["ranking_detail"]["normalization_method"] = "min_max_by_processing_run"

    def _ranking_key(self, item: dict):
        has_error = 1 if item.get("error") else 0
        score = self._to_score(item.get("score_0_100"))

        predicted_label_priority = 1 if item.get("predicted_label") is True else 0

        score_detail = item.get("score_detail") or {}

        text_similarity_score = self._to_score(
            score_detail.get("text_similarity_score_0_100")
        )

        model_score = self._to_score(score_detail.get("model_score_0_100"))

        input_order = item.get("_input_order") or 0

        return (
            has_error,
            -score,
            -predicted_label_priority,
            -text_similarity_score,
            -model_score,
            input_order,
        )

    @staticmethod
    def _to_score(value) -> float:
        try:
            score = float(value or 0)
        except (TypeError, ValueError):
            score = 0.0

        if score < 0:
            return 0.0

        if score > 100:
            return 100.0

        return round(score, 2)
