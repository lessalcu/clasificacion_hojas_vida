from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from io import StringIO
from typing import Any

from flask import current_app

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.classification_result_repository import (
    ClassificationResultRepository,
)
from app.repositories.execution_report_repository import ExecutionReportRepository
from app.repositories.job_profile_repository import JobProfileRepository
from app.repositories.model_version_repository import ModelVersionRepository
from app.repositories.processing_run_repository import ProcessingRunRepository
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.storage_service import StorageService


class ExecutionReportService:
    def __init__(self):
        self.processing_run_repository = ProcessingRunRepository()
        self.classification_result_repository = ClassificationResultRepository()
        self.execution_report_repository = ExecutionReportRepository()
        self.job_profile_repository = JobProfileRepository()
        self.model_version_repository = ModelVersionRepository()
        self.training_run_repository = TrainingRunRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.storage_service = StorageService()

    def get_execution_trace(self, processing_run_id: str) -> dict[str, Any]:
        processing_run = self.processing_run_repository.get_by_id(processing_run_id)

        if not processing_run:
            raise NotFoundError(f"Processing run '{processing_run_id}' was not found")

        job_profile = None
        active_model = None

        job_profile_id = processing_run.get("job_profile_id")
        model_version_id = processing_run.get("model_version_id")

        if job_profile_id:
            job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if model_version_id:
            active_model = self.model_version_repository.get_by_id(model_version_id)

        classification_results = (
            self.classification_result_repository.list_by_processing_run(
                processing_run_id
            )
        )

        candidate_profile_ids = [
            item["candidate_profile_id"]
            for item in classification_results
            if item.get("candidate_profile_id")
        ]

        candidate_profiles = self.candidate_profile_repository.get_by_ids(
            candidate_profile_ids
        )

        candidate_profile_map = {
            profile["id"]: profile
            for profile in candidate_profiles
            if profile.get("id")
        }

        enriched_results = []

        for item in classification_results:
            candidate_profile = candidate_profile_map.get(
                item.get("candidate_profile_id")
            )

            enriched_results.append(
                {
                    "classification_result_id": item.get("id"),
                    "candidate_profile_id": item.get("candidate_profile_id"),
                    "candidate_pseudonym_code": (
                        candidate_profile.get("pseudonym_code")
                        if candidate_profile
                        else None
                    ),
                    "predicted_label": item.get("predicted_label"),
                    "score_0_100": self._to_float(item.get("score_0_100")),
                    "rank_position": item.get("rank_position"),
                    "match_summary": item.get("match_summary") or {},
                    "created_at": item.get("created_at"),
                }
            )

        total_candidates = int(processing_run.get("total_candidates") or 0)
        successful_candidates = len(classification_results)
        failed_candidates = max(total_candidates - successful_candidates, 0)

        scores = [
            item["score_0_100"]
            for item in enriched_results
            if item.get("score_0_100") is not None
        ]

        model_benchmark = self._build_model_benchmark(
            job_profile_id=job_profile_id,
            active_model=active_model,
        )

        summary = {
            "processing_run_id": processing_run_id,
            "execution_status": processing_run.get("status"),
            "input_type": processing_run.get("input_type"),
            "started_at": processing_run.get("started_at"),
            "finished_at": processing_run.get("finished_at"),
            "created_at": processing_run.get("created_at"),
            "job_profile_id": job_profile_id,
            "job_profile_title": job_profile.get("title") if job_profile else None,
            "model_version_id": model_version_id,
            "model_algorithm": active_model.get("algorithm") if active_model else None,
            "model_version_tag": (
                active_model.get("version_tag") if active_model else None
            ),
            "model_status": active_model.get("status") if active_model else None,
            "total_candidates": total_candidates,
            "successful_candidates": successful_candidates,
            "failed_candidates": failed_candidates,
            "error_message": processing_run.get("error_message"),
            "max_score_0_100": max(scores) if scores else 0,
            "min_score_0_100": min(scores) if scores else 0,
            "average_score_0_100": (
                round(sum(scores) / len(scores), 2) if scores else 0
            ),
            "model_selection": {
                "selected_algorithm": model_benchmark.get("selected_algorithm"),
                "selected_model_version_id": model_benchmark.get(
                    "selected_model_version_id"
                ),
                "selection_reason": model_benchmark.get("selection_reason"),
                "selection_criteria": model_benchmark.get("selection_criteria"),
            },
            "benchmark_summary": self._build_benchmark_summary(
                model_benchmark.get("models") or []
            ),
        }

        trace = {
            "summary": summary,
            "processing_run": processing_run,
            "job_profile": job_profile,
            "active_model": active_model,
            "model_version": active_model,
            "model_benchmark": model_benchmark,
            "results": enriched_results,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        self.processing_run_repository.update(
            processing_run_id,
            {
                "trace_summary": summary,
            },
        )

        return trace

    def generate_execution_report(
        self,
        processing_run_id: str,
        created_by: str | None = None,
        report_format: str = "json",
        persist_to_storage: bool = True,
    ) -> dict[str, Any]:
        report_format = report_format.lower().strip()

        if report_format not in ["json", "csv"]:
            raise ValidationError("report_format must be json or csv")

        trace = self.get_execution_trace(processing_run_id)
        summary = trace["summary"]

        file_bytes = self._build_report_bytes(
            trace=trace,
            report_format=report_format,
        )

        artifact_bucket = None
        artifact_path = None

        if persist_to_storage:
            artifact_bucket = current_app.config["REPORTS_BUCKET"]
            timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            artifact_path = (
                f"execution-reports/{processing_run_id}/"
                f"technical-report-{timestamp}.{report_format}"
            )

            content_type = "application/json" if report_format == "json" else "text/csv"

            self.storage_service.upload_bytes(
                bucket_name=artifact_bucket,
                remote_path=artifact_path,
                file_bytes=file_bytes,
                content_type=content_type,
                upsert=True,
            )

        report = self.execution_report_repository.create(
            {
                "processing_run_id": processing_run_id,
                "job_profile_id": summary.get("job_profile_id"),
                "model_version_id": summary.get("model_version_id"),
                "created_by": created_by,
                "report_type": "technical_execution",
                "report_format": report_format,
                "total_candidates": summary.get("total_candidates") or 0,
                "successful_candidates": summary.get("successful_candidates") or 0,
                "failed_candidates": summary.get("failed_candidates") or 0,
                "artifact_bucket": artifact_bucket,
                "artifact_path": artifact_path,
                "metadata": {
                    "summary": summary,
                    "model_benchmark": trace.get("model_benchmark"),
                    "generated_at": trace.get("generated_at"),
                },
            }
        )

        if artifact_bucket and artifact_path:
            self.processing_run_repository.update(
                processing_run_id,
                {
                    "report_bucket": artifact_bucket,
                    "report_path": artifact_path,
                },
            )

        return {
            "report": report,
            "trace": trace,
        }

    def list_reports_by_processing_run(self, processing_run_id: str) -> list[dict]:
        return self.execution_report_repository.list_by_processing_run(
            processing_run_id
        )

    def get_report_by_id(self, report_id: str) -> dict | None:
        report = self.execution_report_repository.get_by_id(report_id)

        if not report:
            raise NotFoundError(f"Execution report '{report_id}' was not found")

        return report

    def _build_model_benchmark(
        self,
        job_profile_id: str | None,
        active_model: dict | None,
    ) -> dict[str, Any]:
        if not job_profile_id:
            return {
                "selection_criteria": None,
                "selected_algorithm": None,
                "selected_model_version_id": None,
                "selection_reason": None,
                "dataset_version": None,
                "models": [],
            }

        dataset_version = active_model.get("dataset_version") if active_model else None

        if dataset_version:
            candidate_versions = (
                self.model_version_repository.list_by_job_profile_and_dataset_version(
                    job_profile_id=job_profile_id,
                    dataset_version=dataset_version,
                )
            )
        else:
            candidate_versions = []

        if not candidate_versions:
            candidate_versions = self.model_version_repository.list_by_job_profile(
                job_profile_id
            )

        latest_by_algorithm = {}

        for version in candidate_versions:
            algorithm = version.get("algorithm")

            if algorithm not in ["knn", "decision_tree"]:
                continue

            if algorithm not in latest_by_algorithm:
                latest_by_algorithm[algorithm] = version

        if active_model and active_model.get("algorithm"):
            latest_by_algorithm[active_model["algorithm"]] = active_model

        model_versions = list(latest_by_algorithm.values())
        model_version_ids = [
            version["id"] for version in model_versions if version.get("id")
        ]

        training_runs = self.training_run_repository.get_by_model_version_ids(
            model_version_ids
        )

        training_run_map = {}

        for run in training_runs:
            model_version_id = run.get("model_version_id")

            if model_version_id and model_version_id not in training_run_map:
                training_run_map[model_version_id] = run

        models = []

        for version in model_versions:
            model_version_id = version.get("id")
            training_run = training_run_map.get(model_version_id)

            item = self._build_model_benchmark_item(
                model_version=version,
                training_run=training_run,
                active_model=active_model,
            )
            models.append(item)

        models = sorted(
            models,
            key=lambda item: {
                "knn": 1,
                "decision_tree": 2,
            }.get(item.get("algorithm"), 99),
        )

        selected_model = None

        for item in models:
            if item.get("is_selected"):
                selected_model = item
                break

        if not selected_model and active_model:
            selected_model = {
                "algorithm": active_model.get("algorithm"),
                "model_version_id": active_model.get("id"),
                "recall": self._to_float(
                    (active_model.get("metrics") or {}).get("recall")
                ),
                "f1_score": self._to_float(
                    (active_model.get("metrics") or {}).get("f1_score")
                ),
                "duration_ms": self._to_int(
                    (active_model.get("metrics") or {}).get("duration_ms")
                ),
            }

        for item in models:
            item["selection_comment"] = self._build_selection_comment(
                item=item,
                selected_model=selected_model,
            )

        return {
            "selection_criteria": (
                "Mayor Recall, luego mayor F1-Score y finalmente menor duración."
            ),
            "selected_algorithm": (
                active_model.get("algorithm") if active_model else None
            ),
            "selected_model_version_id": (
                active_model.get("id") if active_model else None
            ),
            "selection_reason": (
                active_model.get("selection_reason") if active_model else None
            ),
            "dataset_version": dataset_version,
            "models": models,
        }

    def _build_model_benchmark_item(
        self,
        model_version: dict,
        training_run: dict | None,
        active_model: dict | None,
    ) -> dict[str, Any]:
        metrics = self._ensure_dict(model_version.get("metrics"))
        training_metrics = self._ensure_dict(
            training_run.get("metrics") if training_run else {}
        )

        merged_metrics = {
            **training_metrics,
            **metrics,
        }

        model_version_id = model_version.get("id")
        active_model_id = active_model.get("id") if active_model else None

        dataset_summary = self._compact_dataset_summary(
            self._ensure_dict(model_version.get("dataset_summary"))
            or self._ensure_dict(
                training_run.get("dataset_summary") if training_run else {}
            )
        )

        return {
            "model_version_id": model_version_id,
            "algorithm": model_version.get("algorithm"),
            "model_name": model_version.get("model_name"),
            "version_tag": model_version.get("version_tag"),
            "status": model_version.get("status"),
            "dataset_version": model_version.get("dataset_version")
            or (training_run.get("dataset_version") if training_run else None),
            "accuracy": self._to_float(
                merged_metrics.get("accuracy")
                or (training_run.get("accuracy") if training_run else None)
            ),
            "precision": self._to_float(
                merged_metrics.get("precision")
                or (training_run.get("precision") if training_run else None)
            ),
            "recall": self._to_float(
                merged_metrics.get("recall")
                or (training_run.get("recall") if training_run else None)
            ),
            "f1_score": self._to_float(
                merged_metrics.get("f1_score")
                or (training_run.get("f1_score") if training_run else None)
            ),
            "duration_ms": self._to_int(
                merged_metrics.get("duration_ms")
                or (training_run.get("duration_ms") if training_run else None)
            ),
            "confusion_matrix": (
                merged_metrics.get("confusion_matrix")
                or (training_run.get("confusion_matrix") if training_run else None)
            ),
            "train_size": self._to_int(
                merged_metrics.get("train_size")
                or model_version.get("train_size")
                or (training_run.get("train_size") if training_run else None)
            ),
            "test_size": self._to_int(
                merged_metrics.get("test_size")
                or model_version.get("test_size")
                or (training_run.get("test_size") if training_run else None)
            ),
            "artifact_bucket": model_version.get("artifact_bucket"),
            "artifact_path": model_version.get("artifact_path"),
            "model_parameters": self._ensure_dict(
                model_version.get("model_parameters")
            ),
            "dataset_summary": dataset_summary,
            "is_selected": model_version_id == active_model_id,
        }

    @staticmethod
    def _build_selection_comment(
        item: dict,
        selected_model: dict | None,
    ) -> str:
        if item.get("is_selected"):
            return (
                "Seleccionado como modelo activo porque obtuvo el mejor criterio "
                "según Recall, F1-Score y duración."
            )

        if not selected_model:
            return "No seleccionado."

        item_recall = item.get("recall") or 0
        selected_recall = selected_model.get("recall") or 0

        item_f1 = item.get("f1_score") or 0
        selected_f1 = selected_model.get("f1_score") or 0

        item_duration = item.get("duration_ms") or 0
        selected_duration = selected_model.get("duration_ms") or 0

        if item_recall < selected_recall:
            return (
                "No seleccionado porque su Recall fue menor; recuperó menos "
                "candidatos aptos que el modelo activo."
            )

        if item_f1 < selected_f1:
            return (
                "No seleccionado porque su F1-Score fue menor frente al modelo activo."
            )

        if item_duration > selected_duration:
            return (
                "No seleccionado porque tuvo mayor duración ante métricas equivalentes."
            )

        return "No seleccionado por el criterio global de comparación."

    def _build_report_bytes(
        self,
        trace: dict[str, Any],
        report_format: str,
    ) -> bytes:
        if report_format == "json":
            return json.dumps(
                trace,
                ensure_ascii=False,
                indent=2,
                default=str,
            ).encode("utf-8")

        return self._build_csv_report_bytes(trace)

    @staticmethod
    def _build_csv_report_bytes(trace: dict[str, Any]) -> bytes:
        output = StringIO()

        fieldnames = [
            "rank_position",
            "candidate_profile_id",
            "candidate_pseudonym_code",
            "predicted_label",
            "score_0_100",
            "selected_algorithm",
            "selection_reason",
            "benchmark_summary",
            "model_algorithm",
            "model_version_tag",
            "job_profile_title",
            "processing_run_id",
            "created_at",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        summary = trace.get("summary") or {}
        model_selection = summary.get("model_selection") or {}

        for item in trace.get("results") or []:
            writer.writerow(
                {
                    "rank_position": item.get("rank_position"),
                    "candidate_profile_id": item.get("candidate_profile_id"),
                    "candidate_pseudonym_code": item.get("candidate_pseudonym_code"),
                    "predicted_label": item.get("predicted_label"),
                    "score_0_100": item.get("score_0_100"),
                    "selected_algorithm": model_selection.get("selected_algorithm"),
                    "selection_reason": model_selection.get("selection_reason"),
                    "benchmark_summary": summary.get("benchmark_summary"),
                    "model_algorithm": summary.get("model_algorithm"),
                    "model_version_tag": summary.get("model_version_tag"),
                    "job_profile_title": summary.get("job_profile_title"),
                    "processing_run_id": summary.get("processing_run_id"),
                    "created_at": item.get("created_at"),
                }
            )

        return output.getvalue().encode("utf-8")

    @staticmethod
    def _build_benchmark_summary(models: list[dict]) -> str:
        parts = []

        for item in models:
            parts.append(
                (
                    f"{item.get('algorithm')}: "
                    f"accuracy={item.get('accuracy')}, "
                    f"precision={item.get('precision')}, "
                    f"recall={item.get('recall')}, "
                    f"f1={item.get('f1_score')}, "
                    f"duration_ms={item.get('duration_ms')}"
                )
            )

        return " | ".join(parts)

    @staticmethod
    def _compact_dataset_summary(dataset_summary: dict) -> dict:
        if not dataset_summary:
            return {}

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

    @staticmethod
    def _ensure_dict(value) -> dict:
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            try:
                loaded = json.loads(value)
                return loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}

        return {}

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0

        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _to_int(value) -> int:
        if value is None:
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0