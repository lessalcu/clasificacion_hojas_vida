from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


class KNNTrainer:
    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "cosine",
        weights: str = "distance",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def build_estimator(self, sample_size: int) -> KNeighborsClassifier:
        neighbors = max(1, min(self.n_neighbors, sample_size))

        return KNeighborsClassifier(
            n_neighbors=neighbors,
            metric=self.metric,
            weights=self.weights,
            algorithm="brute",
        )
