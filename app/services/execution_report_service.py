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
from app.services.storage_service import StorageService


class ExecutionReportService:
    def __init__(self):
        self.processing_run_repository = ProcessingRunRepository()
        self.classification_result_repository = ClassificationResultRepository()
        self.execution_report_repository = ExecutionReportRepository()
        self.job_profile_repository = JobProfileRepository()
        self.model_version_repository = ModelVersionRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.storage_service = StorageService()

    def get_execution_trace(self, processing_run_id: str) -> dict[str, Any]:
        processing_run = self.processing_run_repository.get_by_id(processing_run_id)

        if not processing_run:
            raise NotFoundError(f"Processing run '{processing_run_id}' was not found")

        job_profile = None
        model_version = None

        job_profile_id = processing_run.get("job_profile_id")
        model_version_id = processing_run.get("model_version_id")

        if job_profile_id:
            job_profile = self.job_profile_repository.get_by_id(job_profile_id)

        if model_version_id:
            model_version = self.model_version_repository.get_by_id(model_version_id)

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
            "model_algorithm": (
                model_version.get("algorithm") if model_version else None
            ),
            "model_version_tag": (
                model_version.get("version_tag") if model_version else None
            ),
            "model_status": model_version.get("status") if model_version else None,
            "total_candidates": total_candidates,
            "successful_candidates": successful_candidates,
            "failed_candidates": failed_candidates,
            "error_message": processing_run.get("error_message"),
            "max_score_0_100": max(scores) if scores else 0,
            "min_score_0_100": min(scores) if scores else 0,
            "average_score_0_100": (
                round(sum(scores) / len(scores), 2) if scores else 0
            ),
        }

        trace = {
            "summary": summary,
            "processing_run": processing_run,
            "job_profile": job_profile,
            "model_version": model_version,
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
            "model_algorithm",
            "model_version_tag",
            "job_profile_title",
            "processing_run_id",
            "created_at",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        summary = trace.get("summary") or {}

        for item in trace.get("results") or []:
            writer.writerow(
                {
                    "rank_position": item.get("rank_position"),
                    "candidate_profile_id": item.get("candidate_profile_id"),
                    "candidate_pseudonym_code": item.get("candidate_pseudonym_code"),
                    "predicted_label": item.get("predicted_label"),
                    "score_0_100": item.get("score_0_100"),
                    "model_algorithm": summary.get("model_algorithm"),
                    "model_version_tag": summary.get("model_version_tag"),
                    "job_profile_title": summary.get("job_profile_title"),
                    "processing_run_id": summary.get("processing_run_id"),
                    "created_at": item.get("created_at"),
                }
            )

        return output.getvalue().encode("utf-8")

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0

        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return 0.0
