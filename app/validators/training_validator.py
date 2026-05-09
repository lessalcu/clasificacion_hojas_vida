def validate_training_payload(payload: dict | None) -> dict:
    payload = payload or {}

    return {
        "created_by": payload.get("created_by"),
        "dataset_version": payload.get("dataset_version") or "api-training-run",
        "persist_to_storage": bool(payload.get("persist_to_storage", True)),
        "auto_build_dataset": bool(payload.get("auto_build_dataset", True)),
    }
