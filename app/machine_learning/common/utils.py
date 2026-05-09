from __future__ import annotations

import json
from typing import Any


def normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []

        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, list):
                return [str(item).strip() for item in loaded if str(item).strip()]
        except json.JSONDecodeError:
            pass

        return [part.strip() for part in stripped.split(",") if part.strip()]

    return [str(value).strip()]


def build_job_profile_text(job_profile: dict) -> str:
    parts: list[str] = [
        f"title: {job_profile.get('title', '')}",
        f"description: {job_profile.get('description', '')}",
        f"required_skills: {' '.join(normalize_text_list(job_profile.get('required_skills')))}",
        f"technologies: {' '.join(normalize_text_list(job_profile.get('technologies')))}",
        f"experience_requirement: {job_profile.get('experience_requirement', '')}",
        f"education_requirement: {job_profile.get('education_requirement', '')}",
        f"languages: {' '.join(normalize_text_list(job_profile.get('languages')))}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith(': ')).strip()


def build_candidate_text(candidate_profile: dict) -> str:
    normalized_text = (candidate_profile.get("normalized_text") or "").strip()
    raw_text = (candidate_profile.get("raw_text") or "").strip()

    if normalized_text:
        return normalized_text

    parts: list[str] = [
        f"skills: {' '.join(normalize_text_list(candidate_profile.get('skills')))}",
        f"technologies: {' '.join(normalize_text_list(candidate_profile.get('technologies')))}",
        f"experience: {candidate_profile.get('experience_summary', '')}",
        f"education: {candidate_profile.get('education_summary', '')}",
        f"languages: {' '.join(normalize_text_list(candidate_profile.get('languages')))}",
        f"keywords: {' '.join(normalize_text_list(candidate_profile.get('keywords')))}",
        f"raw_text: {raw_text}",
    ]
    return " | ".join(part for part in parts if part and not part.endswith(': ')).strip()


def build_training_text(job_profile: dict, candidate_profile: dict) -> str:
    job_text = build_job_profile_text(job_profile)
    candidate_text = build_candidate_text(candidate_profile)
    return f"[JOB_PROFILE] {job_text} [CANDIDATE_PROFILE] {candidate_text}".strip()
