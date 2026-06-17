from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import current_app

from app.services.storage_service import StorageService


METRICS = ["accuracy", "precision", "recall", "f1_score", "duration_ms"]
CV_METRICS = ["accuracy", "precision", "recall", "f1_score"]


class TrainingReportService:
    """Creates compact JSON reports for training iterations and summaries."""

    def __init__(self):
        self.storage_service = StorageService()

    def save_iteration_report(
        self,
        training_result: dict[str, Any],
        created_by: str | None = None,
        persist_to_storage: bool = True,
    ) -> dict[str, Any]:
        report = self.build_iteration_report(
            training_result=training_result,
            created_by=created_by,
        )

        local_path = self._write_local_report(
            job_profile_id=report["job_profile_id"],
            filename=report["report_filename"],
            payload=report,
            folder="individual",
        )

        artifact_bucket = None
        artifact_path = None

        if persist_to_storage:
            artifact_bucket = current_app.config["REPORTS_BUCKET"]
            artifact_path = self._build_storage_path(
                job_profile_id=report["job_profile_id"],
                filename=report["report_filename"],
                folder="individual",
            )
            self.storage_service.upload_bytes(
                bucket_name=artifact_bucket,
                remote_path=artifact_path,
                file_bytes=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json",
                upsert=True,
            )

        report_location = {
            "local_path": str(local_path),
            "artifact_bucket": artifact_bucket,
            "artifact_path": artifact_path,
        }

        report["report_location"] = report_location
        return report

    def build_iteration_report(
        self,
        training_result: dict[str, Any],
        created_by: str | None = None,
    ) -> dict[str, Any]:
        job_profile_id = training_result["job_profile_id"]
        dataset_version = training_result["dataset_version"]
        experiment_run_id = training_result.get("experiment_run_id")
        iteration_number = int(training_result.get("iteration_number") or 1)
        random_state = int(training_result.get("random_state") or current_app.config["TRAINING_RANDOM_STATE"])

        compact_models = {}
        for model in training_result.get("all_models") or []:
            algorithm = model.get("algorithm")
            if algorithm not in ["knn", "decision_tree"]:
                continue
            compact_models[algorithm] = self._compact_model_metrics(model)

        selected_model = training_result.get("selected_model") or {}

        report = {
            "report_type": "training_iteration",
            "job_profile_id": job_profile_id,
            "dataset_version": dataset_version,
            "experiment_run_id": experiment_run_id,
            "iteration_number": iteration_number,
            "execution_label": f"Ejecución {iteration_number}",
            "random_state": random_state,
            "created_by": created_by,
            "generated_at": datetime.now(UTC).isoformat(),
            "selection_criteria": training_result.get("selection_criteria"),
            "selected_algorithm": selected_model.get("algorithm"),
            "selected_model_version_id": selected_model.get("model_version_id"),
            "dataset_summary": self._compact_dataset_summary(training_result.get("dataset_summary") or {}),
            "models": compact_models,
        }

        report["report_filename"] = self._build_iteration_filename(
            experiment_run_id=experiment_run_id,
            iteration_number=iteration_number,
            dataset_version=dataset_version,
        )

        return report

    def build_all_executions_report(
        self,
        job_profile_id: str,
        persist_to_storage: bool = True,
        limit: int | str | None = None,
    ) -> dict[str, Any]:
        normalized_limit = self._normalize_limit(limit)
        reports = self.list_iteration_reports(job_profile_id=job_profile_id, limit=normalized_limit)
        reports = sorted(
            reports,
            key=lambda item: (
                str(item.get("experiment_run_id") or ""),
                int(item.get("iteration_number") or 0),
                str(item.get("generated_at") or ""),
            ),
        )

        executions = []
        for index, report in enumerate(reports, start=1):
            executions.append(
                {
                    "execution_number": index,
                    "execution_label": f"Ejecución {index}",
                    "source_iteration_number": report.get("iteration_number"),
                    "experiment_run_id": report.get("experiment_run_id"),
                    "dataset_version": report.get("dataset_version"),
                    "random_state": report.get("random_state"),
                    "selected_algorithm": report.get("selected_algorithm"),
                    "models": report.get("models") or {},
                }
            )

        summary = self._build_descriptive_statistics(executions)

        aggregate_report = {
            "report_type": "training_all_executions_summary",
            "job_profile_id": job_profile_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_executions": len(executions),
            "statistics": summary,
            "executions": executions,
        }

        filename = f"all-executions-summary-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.json"
        local_path = self._write_local_report(
            job_profile_id=job_profile_id,
            filename=filename,
            payload=aggregate_report,
            folder="summary",
        )

        aggregate_report["report_location"] = {
            "local_path": str(local_path),
            "artifact_bucket": None,
            "artifact_path": None,
        }

        if persist_to_storage:
            artifact_bucket = current_app.config["REPORTS_BUCKET"]
            artifact_path = self._build_storage_path(
                job_profile_id=job_profile_id,
                filename=filename,
                folder="summary",
            )
            self.storage_service.upload_bytes(
                bucket_name=artifact_bucket,
                remote_path=artifact_path,
                file_bytes=json.dumps(aggregate_report, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json",
                upsert=True,
            )
            aggregate_report["report_location"] = {
                "local_path": str(local_path),
                "artifact_bucket": artifact_bucket,
                "artifact_path": artifact_path,
            }

        return aggregate_report

    def list_iteration_reports(
        self,
        job_profile_id: str,
        limit: int | str | None = None,
    ) -> list[dict[str, Any]]:
        local_reports = self._load_local_iteration_reports(job_profile_id=job_profile_id)
        storage_reports = self._load_storage_iteration_reports(job_profile_id=job_profile_id)

        merged_by_key = {}
        for report in [*storage_reports, *local_reports]:
            key = (
                report.get("experiment_run_id"),
                report.get("iteration_number"),
                report.get("dataset_version"),
            )
            merged_by_key[key] = report

        reports = list(merged_by_key.values())
        reports = sorted(
            reports,
            key=lambda item: str(item.get("generated_at") or ""),
            reverse=True,
        )

        normalized_limit = self._normalize_limit(limit)
        if normalized_limit and normalized_limit > 0:
            return reports[:normalized_limit]

        return reports

    @staticmethod
    def _normalize_limit(limit: int | str | None) -> int | None:
        if limit in (None, ""):
            return None
        try:
            normalized = int(limit)
        except (TypeError, ValueError):
            return None
        return normalized if normalized > 0 else None

    @staticmethod
    def _compact_model_metrics(model: dict[str, Any]) -> dict[str, Any]:
        cross_validation = model.get("cross_validation") or {"enabled": False}
        cv_compact = {"enabled": bool(cross_validation.get("enabled"))}

        if cross_validation.get("enabled"):
            cv_compact.update(
                {
                    "method": cross_validation.get("method"),
                    "folds": cross_validation.get("folds"),
                    "random_state": cross_validation.get("random_state"),
                    "class_distribution": cross_validation.get("class_distribution"),
                    "accuracy_mean": cross_validation.get("accuracy_mean"),
                    "accuracy_std": cross_validation.get("accuracy_std"),
                    "precision_mean": cross_validation.get("precision_mean"),
                    "precision_std": cross_validation.get("precision_std"),
                    "recall_mean": cross_validation.get("recall_mean"),
                    "recall_std": cross_validation.get("recall_std"),
                    "f1_score_mean": cross_validation.get("f1_score_mean"),
                    "f1_score_std": cross_validation.get("f1_score_std"),
                    "fold_scores": cross_validation.get("fold_scores"),
                }
            )
        else:
            cv_compact["reason"] = cross_validation.get("reason")

        return {
            "model_version_id": model.get("model_version_id"),
            "algorithm": model.get("algorithm"),
            "accuracy": model.get("accuracy"),
            "precision": model.get("precision"),
            "recall": model.get("recall"),
            "f1_score": model.get("f1_score"),
            "duration_ms": model.get("duration_ms"),
            "train_size": model.get("train_size"),
            "test_size": model.get("test_size"),
            "confusion_matrix": model.get("confusion_matrix"),
            "cross_validation": cv_compact,
        }

    @staticmethod
    def _compact_dataset_summary(dataset_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_profile_id": dataset_summary.get("job_profile_id"),
            "dataset_version": dataset_summary.get("dataset_version"),
            "total_rows": dataset_summary.get("total_rows"),
            "train_size": dataset_summary.get("train_size"),
            "test_size": dataset_summary.get("test_size"),
            "class_distribution": dataset_summary.get("class_distribution"),
            "train_distribution": dataset_summary.get("train_distribution"),
            "test_distribution": dataset_summary.get("test_distribution"),
            "generated_at": dataset_summary.get("generated_at"),
        }

    def _build_descriptive_statistics(self, executions: list[dict[str, Any]]) -> dict[str, Any]:
        by_algorithm: dict[str, dict[str, list[float]]] = {
            "knn": {metric: [] for metric in METRICS},
            "decision_tree": {metric: [] for metric in METRICS},
        }
        cv_by_algorithm: dict[str, dict[str, list[float]]] = {
            "knn": {metric: [] for metric in CV_METRICS},
            "decision_tree": {metric: [] for metric in CV_METRICS},
        }
        selected_counts = {"knn": 0, "decision_tree": 0}

        for execution in executions:
            selected_algorithm = execution.get("selected_algorithm")
            if selected_algorithm in selected_counts:
                selected_counts[selected_algorithm] += 1

            models = execution.get("models") or {}
            for algorithm in ["knn", "decision_tree"]:
                model = models.get(algorithm) or {}
                for metric in METRICS:
                    value = self._to_float_or_none(model.get(metric))
                    if value is not None:
                        by_algorithm[algorithm][metric].append(value)

                cross_validation = model.get("cross_validation") or {}
                if cross_validation.get("enabled"):
                    for metric in CV_METRICS:
                        value = self._to_float_or_none(cross_validation.get(f"{metric}_mean"))
                        if value is not None:
                            cv_by_algorithm[algorithm][metric].append(value)

        return {
            "selection_counts": selected_counts,
            "holdout_metrics": {
                algorithm: {
                    metric: self._describe_values(values)
                    for metric, values in metrics.items()
                }
                for algorithm, metrics in by_algorithm.items()
            },
            "cross_validation_metrics": {
                algorithm: {
                    metric: self._describe_values(values)
                    for metric, values in metrics.items()
                }
                for algorithm, metrics in cv_by_algorithm.items()
            },
            "boxplot_data": {
                "holdout": by_algorithm,
                "cross_validation": cv_by_algorithm,
            },
            "recommended_algorithm": self._recommend_algorithm(cv_by_algorithm, by_algorithm),
            "recommendation_criteria": (
                "Se prioriza el mayor promedio de Recall, luego F1-Score, "
                "y finalmente la menor dispersión en F1-Score."
            ),
        }

    def _recommend_algorithm(
        self,
        cv_by_algorithm: dict[str, dict[str, list[float]]],
        by_algorithm: dict[str, dict[str, list[float]]],
    ) -> str | None:
        candidates = []

        for algorithm in ["knn", "decision_tree"]:
            source = cv_by_algorithm[algorithm]
            if not source["recall"]:
                source = by_algorithm[algorithm]

            recall = self._safe_mean(source.get("recall") or [])
            f1 = self._safe_mean(source.get("f1_score") or [])
            f1_std = self._safe_std(source.get("f1_score") or [])
            candidates.append((algorithm, recall, f1, f1_std))

        candidates = sorted(candidates, key=lambda item: (-item[1], -item[2], item[3]))
        return candidates[0][0] if candidates else None

    @staticmethod
    def _describe_values(values: list[float]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "q1": None,
                "q3": None,
                "max": None,
                "iqr": None,
            }

        sorted_values = sorted(values)
        q1 = TrainingReportService._percentile(sorted_values, 25)
        q3 = TrainingReportService._percentile(sorted_values, 75)

        return {
            "count": len(values),
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "max": round(max(values), 4),
            "iqr": round(q3 - q1, 4),
        }

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: int) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (len(sorted_values) - 1) * (percentile / 100)
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = position - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    @staticmethod
    def _to_float_or_none(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_local_iteration_reports(self, job_profile_id: str) -> list[dict[str, Any]]:
        folder = self._local_report_root() / job_profile_id / "individual"
        if not folder.exists():
            return []

        reports = []
        for file_path in folder.glob("*.json"):
            try:
                reports.append(json.loads(file_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return reports

    def _load_storage_iteration_reports(self, job_profile_id: str) -> list[dict[str, Any]]:
        try:
            bucket_name = current_app.config["REPORTS_BUCKET"]
            folder = f"training-executions/{job_profile_id}/individual"
            files = self.storage_service.list_files(bucket_name=bucket_name, folder=folder)
        except Exception:
            return []

        reports = []
        for item in files or []:
            name = item.get("name") if isinstance(item, dict) else None
            if not name or not name.endswith(".json"):
                continue

            remote_path = f"training-executions/{job_profile_id}/individual/{name}"
            try:
                file_bytes = self.storage_service.download_file(
                    bucket_name=bucket_name,
                    remote_path=remote_path,
                )
                reports.append(json.loads(file_bytes.decode("utf-8")))
            except Exception:
                continue

        return reports

    def _write_local_report(
        self,
        job_profile_id: str,
        filename: str,
        payload: dict[str, Any],
        folder: str,
    ) -> Path:
        target_folder = self._local_report_root() / job_profile_id / folder
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = target_folder / filename
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path

    @staticmethod
    def _local_report_root() -> Path:
        configured = current_app.config.get("TRAINING_REPORTS_DIR")
        if configured:
            return Path(configured)
        return Path(current_app.config["MODEL_ARTIFACTS_DIR"]) / "training_reports"

    @staticmethod
    def _build_storage_path(job_profile_id: str, filename: str, folder: str) -> str:
        return f"training-executions/{job_profile_id}/{folder}/{filename}"

    @staticmethod
    def _build_iteration_filename(
        experiment_run_id: str | None,
        iteration_number: int,
        dataset_version: str,
    ) -> str:
        safe_experiment = experiment_run_id or dataset_version.replace("/", "-")
        return f"{safe_experiment}-iteration-{iteration_number:03d}-report.json"
