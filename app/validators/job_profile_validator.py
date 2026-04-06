def _validate_text_field(value, field_name, min_length=1, max_length=255):
    if not isinstance(value, str):
        raise ValueError(f"Field '{field_name}' must be a string")

    value = value.strip()

    if len(value) < min_length:
        raise ValueError(
            f"Field '{field_name}' must have at least {min_length} characters"
        )

    if len(value) > max_length:
        raise ValueError(
            f"Field '{field_name}' must have at most {max_length} characters"
        )

    return value


def _validate_string_list(value, field_name, required=False, min_items=1, max_items=50):
    if value is None:
        if required:
            raise ValueError(f"Field '{field_name}' is required")
        return []

    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list")

    cleaned = []
    seen = set()

    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"All items in '{field_name}' must be strings")

        item = item.strip()

        if not item:
            continue

        normalized = item.lower()
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(item)

    if required and len(cleaned) < min_items:
        raise ValueError(
            f"Field '{field_name}' must contain at least {min_items} item(s)"
        )

    if len(cleaned) > max_items:
        raise ValueError(
            f"Field '{field_name}' must contain at most {max_items} item(s)"
        )

    return cleaned


def validate_job_profile_payload(payload: dict, partial: bool = False):
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    allowed_fields = {
        "title",
        "description",
        "required_skills",
        "technologies",
        "experience_requirement",
        "education_requirement",
        "languages",
    }

    unknown_fields = set(payload.keys()) - allowed_fields
    if unknown_fields:
        raise ValueError(f"Unknown fields: {', '.join(sorted(unknown_fields))}")

    validated = {}

    required_fields = ["title", "required_skills", "technologies"]

    if not partial:
        for field in required_fields:
            if field not in payload:
                raise ValueError(f"Field '{field}' is required")

    if "title" in payload:
        validated["title"] = _validate_text_field(
            payload["title"],
            "title",
            min_length=3,
            max_length=120,
        )

    if "description" in payload:
        validated["description"] = _validate_text_field(
            payload["description"],
            "description",
            min_length=3,
            max_length=1000,
        )

    if "required_skills" in payload:
        validated["required_skills"] = _validate_string_list(
            payload["required_skills"],
            "required_skills",
            required=not partial,
            min_items=1,
            max_items=30,
        )

    if "technologies" in payload:
        validated["technologies"] = _validate_string_list(
            payload["technologies"],
            "technologies",
            required=not partial,
            min_items=1,
            max_items=30,
        )

    if "experience_requirement" in payload:
        validated["experience_requirement"] = _validate_text_field(
            payload["experience_requirement"],
            "experience_requirement",
            min_length=2,
            max_length=250,
        )

    if "education_requirement" in payload:
        validated["education_requirement"] = _validate_text_field(
            payload["education_requirement"],
            "education_requirement",
            min_length=2,
            max_length=250,
        )

    if "languages" in payload:
        validated["languages"] = _validate_string_list(
            payload["languages"],
            "languages",
            required=False,
            min_items=0,
            max_items=20,
        )

    return validated