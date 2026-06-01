from __future__ import annotations

import json
from typing import Any


def normalize_text_list(value: Any) -> list[str]:
    values: list[str] = []
    _collect_text_values(value, values)

    cleaned_values: list[str] = []
    seen = set()

    for item in values:
        text = str(item).strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        cleaned_values.append(text)
        seen.add(key)

    return cleaned_values


def _collect_text_values(value: Any, output: list[str]) -> None:
    if value is None:
        return

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return

        try:
            loaded = json.loads(stripped)
            _collect_text_values(loaded, output)
            return
        except json.JSONDecodeError:
            pass

        if "," in stripped:
            for part in stripped.split(","):
                _collect_text_values(part, output)
            return

        output.append(stripped)
        return

    if isinstance(value, (int, float, bool)):
        output.append(str(value))
        return

    if isinstance(value, dict):
        preferred_keys = (
            "name",
            "canonical",
            "value",
            "label",
            "title",
            "level",
        )

        nested_keys = (
            "raw_values",
            "aliases",
            "keywords",
            "items",
        )

        found_value = False

        for key in preferred_keys:
            if key in value and value.get(key) not in (None, ""):
                _collect_text_values(value.get(key), output)
                found_value = True

        for key in nested_keys:
            if key in value and value.get(key) not in (None, ""):
                _collect_text_values(value.get(key), output)
                found_value = True

        if not found_value:
            for item in value.values():
                _collect_text_values(item, output)

        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_text_values(item, output)
        return

    output.append(str(value))


def _join_terms(value: Any) -> str:
    return " ".join(normalize_text_list(value))


def build_job_profile_text(job_profile: dict) -> str:
    parts: list[str] = [
        f"title: {job_profile.get('title', '')}",
        f"description: {job_profile.get('description', '')}",
        f"required_skills: {_join_terms(job_profile.get('required_skills'))}",
        f"technologies: {_join_terms(job_profile.get('technologies'))}",
        f"experience_requirement: {job_profile.get('experience_requirement', '')}",
        f"education_requirement: {job_profile.get('education_requirement', '')}",
        f"languages: {_join_terms(job_profile.get('languages'))}",
    ]

    return " | ".join(
        part for part in parts if part and not part.endswith(": ")
    ).strip()


def build_candidate_text(candidate_profile: dict) -> str:
    normalized_text = (candidate_profile.get("normalized_text") or "").strip()
    raw_text = (candidate_profile.get("raw_text") or "").strip()

    parts: list[str] = [
        f"normalized_text: {normalized_text}",
        f"skills: {_join_terms(candidate_profile.get('skills'))}",
        f"technologies: {_join_terms(candidate_profile.get('technologies'))}",
        f"experience: {candidate_profile.get('experience_summary', '')}",
        f"education: {candidate_profile.get('education_summary', '')}",
        f"languages: {_join_terms(candidate_profile.get('languages'))}",
        f"keywords: {_join_terms(candidate_profile.get('keywords'))}",
        f"raw_text: {raw_text}",
    ]

    return " | ".join(
        part for part in parts if part and not part.endswith(": ")
    ).strip()


def build_training_text(job_profile: dict, candidate_profile: dict) -> str:
    job_text = build_job_profile_text(job_profile)
    candidate_text = build_candidate_text(candidate_profile)

    return f"[JOB_PROFILE] {job_text} [CANDIDATE_PROFILE] {candidate_text}".strip()
