from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingDatasetRow:
    dataset_sample_id: str
    candidate_profile_id: str
    job_profile_id: str
    label: bool
    split_name: str
    source_type: str
    candidate_text: str
    job_profile_text: str
    training_text: str
    candidate_profile: dict[str, Any] = field(default_factory=dict)
    job_profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluation:
    algorithm: str
    estimator: Any
    vectorizer: Any
    model_version_id: str
    recall: float
    f1_score: float
    precision: float
    accuracy: float
    confusion_matrix: list[list[int]]
    labels: list[int]
    duration_ms: int
    predictions: list[int]
    probabilities: list[float]
    test_size: int
    train_size: int
    version_tag: str
    cross_validation: dict[str, Any] | None = None
    local_artifact_dir: str | None = None
    artifact_bucket: str | None = None
    artifact_path: str | None = None

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "model_version_id": self.model_version_id,
            "algorithm": self.algorithm,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "duration_ms": self.duration_ms,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "version_tag": self.version_tag,
            "cross_validation": self.cross_validation or {"enabled": False},
            "artifact_bucket": self.artifact_bucket,
            "artifact_path": self.artifact_path,
            "local_artifact_dir": self.local_artifact_dir,
        }
