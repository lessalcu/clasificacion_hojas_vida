from __future__ import annotations

from typing import Any

from app.errors.exceptions import (
    NotFoundError,
    ValidationError,
)
from app.repositories.candidate_profile_repository import (
    CandidateProfileRepository,
)
from app.repositories.candidate_source_repository import (
    CandidateSourceRepository,
)
from app.repositories.classification_result_repository import (
    ClassificationResultRepository,
)
from app.repositories.job_profile_repository import (
    JobProfileRepository,
)
from app.repositories.processing_run_repository import (
    ProcessingRunRepository,
)


class CandidateResultsService:
    def __init__(self):
        self.processing_run_repository = (
            ProcessingRunRepository()
        )

        self.classification_result_repository = (
            ClassificationResultRepository()
        )

        self.candidate_profile_repository = (
            CandidateProfileRepository()
        )

        self.candidate_source_repository = (
            CandidateSourceRepository()
        )

        self.job_profile_repository = (
            JobProfileRepository()
        )

    def get_processing_run_results(
        self,
        processing_run_id: str,
        limit: int | str | None = None,
    ) -> dict[str, Any]:
        processing_run = (
            self.processing_run_repository.get_by_id(
                processing_run_id
            )
        )

        if not processing_run:
            raise NotFoundError(
                "Processing run not found"
            )

        normalized_limit = self._normalize_limit(
            limit
        )

        classification_results = (
            self.classification_result_repository
            .list_by_processing_run(
                processing_run_id=processing_run_id,
                limit=normalized_limit,
            )
        )

        candidate_profile_ids = [
            result["candidate_profile_id"]
            for result in classification_results
            if result.get("candidate_profile_id")
        ]

        candidate_profiles = (
            self.candidate_profile_repository.get_by_ids(
                candidate_profile_ids
            )
        )

        candidate_profile_map = {
            profile["id"]: profile
            for profile in candidate_profiles
            if profile.get("id")
        }

        source_ids = [
            profile["source_id"]
            for profile in candidate_profiles
            if profile.get("source_id")
        ]

        candidate_sources = (
            self.candidate_source_repository.get_by_ids(
                source_ids
            )
        )

        candidate_source_map = {
            source["id"]: source
            for source in candidate_sources
            if source.get("id")
        }

        job_profile = None

        job_profile_id = processing_run.get(
            "job_profile_id"
        )

        if job_profile_id:
            job_profile = (
                self.job_profile_repository.get_by_id(
                    job_profile_id
                )
            )

        enriched_results = []

        for result in classification_results:
            candidate_profile = (
                candidate_profile_map.get(
                    result.get(
                        "candidate_profile_id"
                    )
                )
            )

            candidate_source = None

            if candidate_profile:
                candidate_source = (
                    candidate_source_map.get(
                        candidate_profile.get(
                            "source_id"
                        )
                    )
                )

            relevant_matches = (
                self._build_relevant_matches(
                    job_profile=job_profile or {},
                    candidate_profile=(
                        candidate_profile or {}
                    ),
                )
            )

            enriched_results.append(
                {
                    "classification_result_id": (
                        result.get("id")
                    ),
                    "processing_run_id": (
                        processing_run_id
                    ),
                    "candidate_profile_id": (
                        result.get(
                            "candidate_profile_id"
                        )
                    ),
                    "predicted_label": (
                        result.get(
                            "predicted_label"
                        )
                    ),
                    "score_0_100": float(
                        result.get(
                            "score_0_100"
                        )
                        or 0
                    ),
                    "rank_position": (
                        result.get(
                            "rank_position"
                        )
                    ),
                    "match_summary": (
                        result.get(
                            "match_summary"
                        )
                        or {}
                    ),
                    "relevant_matches": (
                        relevant_matches
                    ),
                    "candidate_profile": (
                        self
                        ._serialize_candidate_profile(
                            candidate_profile
                        )
                    ),
                    "candidate_source": (
                        self
                        ._serialize_candidate_source(
                            candidate_source
                        )
                    ),
                }
            )

        scores = [
            item["score_0_100"]
            for item in enriched_results
        ]

        recommended_count = len(
            [
                item
                for item in enriched_results
                if item.get(
                    "predicted_label"
                )
                is True
            ]
        )

        not_recommended_count = len(
            [
                item
                for item in enriched_results
                if item.get(
                    "predicted_label"
                )
                is False
            ]
        )

        return {
            "processing_run": processing_run,
            "job_profile": job_profile,
            "total_results": len(
                enriched_results
            ),
            "returned_results": len(
                enriched_results
            ),
            "summary": {
                "max_score_0_100": (
                    max(scores)
                    if scores
                    else 0
                ),
                "min_score_0_100": (
                    min(scores)
                    if scores
                    else 0
                ),
                "average_score_0_100": (
                    round(
                        sum(scores)
                        / len(scores),
                        2,
                    )
                    if scores
                    else 0
                ),
                "recommended_count": (
                    recommended_count
                ),
                "not_recommended_count": (
                    not_recommended_count
                ),
            },
            "results": enriched_results,
        }

    @staticmethod
    def _normalize_limit(
        limit: int | str | None,
    ) -> int | None:
        if limit in (None, ""):
            return None

        try:
            normalized = int(limit)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValidationError(
                "limit must be a valid integer"
            ) from exc

        if normalized <= 0:
            raise ValidationError(
                "limit must be greater than zero"
            )

        return min(normalized, 1000)

    @staticmethod
    def _serialize_candidate_profile(
        profile: dict | None,
    ) -> dict | None:
        if not profile:
            return None

        return {
            "id": profile.get("id"),
            "candidate_id": profile.get(
                "candidate_id"
            ),
            "source_id": profile.get(
                "source_id"
            ),
            "source_type": profile.get(
                "source_type"
            ),
            "pseudonym_code": profile.get(
                "pseudonym_code"
            ),
            "skills": (
                profile.get("skills")
                or []
            ),
            "technologies": (
                profile.get(
                    "technologies"
                )
                or []
            ),
            "experience_summary": (
                profile.get(
                    "experience_summary"
                )
                or ""
            ),
            "education_summary": (
                profile.get(
                    "education_summary"
                )
                or ""
            ),
            "languages": (
                profile.get("languages")
                or []
            ),
            "keywords": (
                profile.get("keywords")
                or []
            ),
            "created_at": profile.get(
                "created_at"
            ),
            "updated_at": profile.get(
                "updated_at"
            ),
        }

    @staticmethod
    def _serialize_candidate_source(
        source: dict | None,
    ) -> dict | None:
        if not source:
            return None

        return {
            "id": source.get("id"),
            "candidate_id": source.get(
                "candidate_id"
            ),
            "source_type": source.get(
                "source_type"
            ),
            "original_filename": (
                source.get(
                    "original_filename"
                )
            ),
            "mime_type": source.get(
                "mime_type"
            ),
            "size_bytes": source.get(
                "size_bytes"
            ),
            "extraction_status": (
                source.get(
                    "extraction_status"
                )
            ),
            "created_at": source.get(
                "created_at"
            ),
        }

    def _build_relevant_matches(
        self,
        job_profile: dict,
        candidate_profile: dict,
    ) -> dict[str, list[str]]:
        required_skills = (
            self._normalize_text_collection(
                job_profile.get(
                    "required_skills"
                )
            )
        )

        required_technologies = (
            self._normalize_text_collection(
                job_profile.get(
                    "technologies"
                )
            )
        )

        required_languages = (
            self._normalize_text_collection(
                job_profile.get(
                    "languages"
                )
            )
        )

        candidate_skills = (
            self._normalize_profile_collection(
                candidate_profile.get(
                    "skills"
                )
            )
        )

        candidate_technologies = (
            self._normalize_profile_collection(
                candidate_profile.get(
                    "technologies"
                )
            )
        )

        candidate_languages = (
            self._normalize_profile_collection(
                candidate_profile.get(
                    "languages"
                )
            )
        )

        candidate_keywords = (
            self._normalize_text_collection(
                candidate_profile.get(
                    "keywords"
                )
            )
        )

        candidate_terms = {
            **candidate_skills,
            **candidate_technologies,
            **candidate_languages,
            **candidate_keywords,
        }

        return {
            "skills": self._find_matches(
                required_skills,
                candidate_terms,
            ),
            "technologies": (
                self._find_matches(
                    required_technologies,
                    candidate_terms,
                )
            ),
            "languages": (
                self._find_matches(
                    required_languages,
                    candidate_terms,
                )
            ),
        }

    @staticmethod
    def _find_matches(
        required_terms: dict[str, str],
        candidate_terms: dict[str, str],
    ) -> list[str]:
        matches = []

        for (
            normalized,
            display,
        ) in required_terms.items():
            if normalized in candidate_terms:
                matches.append(display)

        return matches

    @staticmethod
    def _normalize_text_collection(
        value: Any,
    ) -> dict[str, str]:
        if value is None:
            return {}

        if isinstance(value, str):
            raw_items = [
                item.strip()
                for item in value.split(",")
            ]

        elif isinstance(value, list):
            raw_items = value

        else:
            return {}

        result = {}

        for item in raw_items:
            if isinstance(item, dict):
                display = str(
                    item.get("name")
                    or item.get("display")
                    or item.get(
                        "canonical"
                    )
                    or ""
                ).strip()
            else:
                display = str(
                    item
                ).strip()

            if display:
                normalized = (
                    CandidateResultsService
                    ._normalize_term(
                        display
                    )
                )

                result[normalized] = display

        return result

    @staticmethod
    def _normalize_profile_collection(
        value: Any,
    ) -> dict[str, str]:
        if not isinstance(value, list):
            return {}

        result = {}

        for item in value:
            if isinstance(item, dict):
                display = str(
                    item.get("name")
                    or item.get("display")
                    or item.get(
                        "canonical"
                    )
                    or ""
                ).strip()

                canonical = str(
                    item.get(
                        "canonical"
                    )
                    or display
                ).strip()

            else:
                display = str(
                    item
                ).strip()

                canonical = display

            if display:
                normalized = (
                    CandidateResultsService
                    ._normalize_term(
                        canonical
                    )
                )

                result[normalized] = display

        return result

    @staticmethod
    def _normalize_term(
        value: str,
    ) -> str:
        return " ".join(
            value.strip().lower().split()
        )