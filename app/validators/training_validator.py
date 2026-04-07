def validate_training_payload(payload: dict | None) -> dict:
    payload = payload or {}

    return {
        "created_by": payload.get("created_by"),
        "dataset_version": payload.get("dataset_version"),
        "persist_to_storage": bool(payload.get("persist_to_storage", True)),
    }
