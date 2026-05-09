from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


class DecisionTreeTrainer:
    def __init__(self, max_depth: int = 20, min_samples_leaf: int = 2, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def build_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
