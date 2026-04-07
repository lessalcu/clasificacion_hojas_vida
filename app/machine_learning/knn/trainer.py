from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


class KNNTrainer:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def build_estimator(self, sample_size: int) -> KNeighborsClassifier:
        neighbors = max(1, min(self.n_neighbors, sample_size))
        return KNeighborsClassifier(
            n_neighbors=neighbors,
            metric="cosine",
            weights="distance",
            algorithm="brute",
        )
