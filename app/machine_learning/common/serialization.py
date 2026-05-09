from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.machine_learning.common.schemas import ModelEvaluation


def persist_model_artifacts(
    evaluation: ModelEvaluation,
    artifact_dir: str,
    training_metadata: dict,
) -> dict[str, str]:
    target_dir = Path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = target_dir / "bundle.joblib"
    metrics_path = target_dir / "metrics.json"
    metadata_path = target_dir / "training_metadata.json"

    bundle = {
        "algorithm": evaluation.algorithm,
        "version_tag": evaluation.version_tag,
        "vectorizer": evaluation.vectorizer,
        "estimator": evaluation.estimator,
    }
    joblib.dump(bundle, bundle_path)

    metrics_payload = evaluation.to_metrics_dict()
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(training_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "bundle": str(bundle_path),
        "metrics": str(metrics_path),
        "metadata": str(metadata_path),
    }
