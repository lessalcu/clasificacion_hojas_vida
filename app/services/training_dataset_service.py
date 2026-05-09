from __future__ import annotations

from collections import Counter

import pandas as pd

from app.machine_learning.common.schemas import TrainingDatasetRow
from app.machine_learning.common.utils import build_candidate_text, build_job_profile_text, build_training_text
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.dataset_sample_repository import DatasetSampleRepository
from app.repositories.job_profile_repository import JobProfileRepository


class TrainingDatasetService:
    def __init__(self):
        self.dataset_sample_repository = DatasetSampleRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.job_profile_repository = JobProfileRepository()

    def build_training_dataframe(self, job_profile_id: str | None = None) -> pd.DataFrame:
        dataset_rows = self.dataset_sample_repository.get_rows(job_profile_id=job_profile_id)

        if not dataset_rows:
            return pd.DataFrame()

        candidate_profile_ids = list({row["candidate_profile_id"] for row in dataset_rows})
        job_profile_ids = list({row["job_profile_id"] for row in dataset_rows})

        candidate_profiles = self.candidate_profile_repository.get_by_ids(candidate_profile_ids)
        job_profiles = self.job_profile_repository.get_by_ids(job_profile_ids)

        candidate_profile_map = {row["id"]: row for row in candidate_profiles}
        job_profile_map = {row["id"]: row for row in job_profiles}

        normalized_rows: list[TrainingDatasetRow] = []

        for row in dataset_rows:
            candidate_profile = candidate_profile_map.get(row["candidate_profile_id"])
            job_profile = job_profile_map.get(row["job_profile_id"])

            if not candidate_profile or not job_profile:
                continue

            normalized_rows.append(
                TrainingDatasetRow(
                    dataset_sample_id=row["id"],
                    candidate_profile_id=row["candidate_profile_id"],
                    job_profile_id=row["job_profile_id"],
                    label=bool(row["label"]),
                    split_name=row.get("split_name") or "train",
                    source_type=candidate_profile.get("source_type", "unknown"),
                    candidate_text=build_candidate_text(candidate_profile),
                    job_profile_text=build_job_profile_text(job_profile),
                    training_text=build_training_text(job_profile, candidate_profile),
                    candidate_profile=candidate_profile,
                    job_profile=job_profile,
                )
            )

        return pd.DataFrame([row.__dict__ for row in normalized_rows])

    @staticmethod
    def summarize_dataframe(dataframe: pd.DataFrame) -> dict:
        if dataframe.empty:
            return {
                "rows": 0,
                "split_distribution": {},
                "label_distribution": {},
            }

        return {
            "rows": int(len(dataframe)),
            "split_distribution": dict(Counter(dataframe["split_name"].tolist())),
            "label_distribution": dict(Counter(dataframe["label"].astype(int).tolist())),
        }
