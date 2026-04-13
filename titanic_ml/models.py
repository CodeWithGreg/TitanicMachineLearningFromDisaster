"""Model registry module providing a unified interface for training and predicting
with multiple classification algorithms.

Supports: logistic_regression, decision_tree, random_forest, gradient_boosting, svm, knn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
}

# Models that accept a `random_state` parameter.
_MODELS_WITH_RANDOM_STATE = {
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svm",
}


class ModelRegistry:
    """Unified interface for training and predicting with supported classifiers.

    Parameters
    ----------
    model_type : str
        Key from ``SUPPORTED_MODELS``.
    hyperparams : dict | None
        Optional keyword arguments forwarded to the sklearn estimator constructor.
    random_seed : int
        Seed for reproducibility; injected as ``random_state`` where applicable.

    Raises
    ------
    ValueError
        If *model_type* is not one of the supported keys.
    """

    def __init__(
        self,
        model_type: str,
        hyperparams: dict | None = None,
        random_seed: int = 42,
    ) -> None:
        if model_type not in SUPPORTED_MODELS:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported types: {supported}"
            )

        self.model_type = model_type
        self.random_seed = random_seed
        self._trained = False

        params: dict = dict(hyperparams) if hyperparams else {}

        # Inject random_state for models that support it.
        if model_type in _MODELS_WITH_RANDOM_STATE and "random_state" not in params:
            params["random_state"] = random_seed

        cls = SUPPORTED_MODELS[model_type]
        self.model = cls(**params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ModelRegistry:
        """Fit the model on training data.

        Returns *self* so calls can be chained.
        """
        self.model.fit(X_train, y_train)
        self._trained = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 or 1).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        self._check_trained()
        return self.model.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return probability estimates for each class.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        AttributeError
            If the underlying model does not support probability estimation
            (e.g. SVM with ``probability=False``).
        """
        self._check_trained()

        if not hasattr(self.model, "predict_proba") or (
            isinstance(self.model, SVC) and not self.model.probability
        ):
            raise AttributeError(
                f"Model type '{self.model_type}' does not support "
                "probability estimation. For SVM, pass "
                "hyperparams={'probability': True} to enable it."
            )

        return self.model.predict_proba(X_test)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Model has not been trained")
