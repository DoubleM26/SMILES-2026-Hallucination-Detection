"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._members = []
        self._threshold = 0.5
        self._random_seed = 42

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return torch.from_numpy(np.log(probs / (1.0 - probs)).astype(np.float32))

    def _slices(self, n_features):
        geo_dim = 144
        if n_features > geo_dim and (n_features - geo_dim) % 3 == 0:
            block = (n_features - geo_dim) // 3
            return [
                (slice(0, block), "lr", 64),
                (slice(block, 2 * block), "lr", 64),
                (slice(2 * block, 3 * block), "lr", 64),
                (slice(3 * block, n_features), "ridge", 16),
            ]
        return [(slice(0, n_features), "lr", 64)]

    def _fit_one(self, X, y, model_name, pca_dim):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(pca_dim, X.shape[1], max(1, len(X) - 1))
        pca = PCA(n_components=n_components, random_state=self._random_seed)
        X_pca = pca.fit_transform(X_scaled)

        if model_name == "ridge":
            model = Ridge(alpha=100.0, random_state=self._random_seed)
        else:
            model = LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=3000,
                random_state=self._random_seed,
            )
        model.fit(X_pca, y)
        return scaler, pca, model, model_name

    def _score_one(self, X, feature_slice, member):
        scaler, pca, model, model_name = member
        X_part = X[:, feature_slice]
        X_pca = pca.transform(scaler.transform(X_part))
        if model_name == "ridge":
            return np.clip(model.predict(X_pca), 0.0, 1.0)
        return model.predict_proba(X_pca)[:, 1]

    def _scores(self, X):
        scores = [self._score_one(X, feature_slice, member) for feature_slice, member in self._members]
        return np.mean(np.vstack(scores), axis=0)

    def _best_threshold(self, y, scores):
        candidates = np.unique(np.concatenate([scores, np.linspace(0.0, 1.0, 501)]))
        best_t = 0.5
        best_acc = -1.0
        for t in candidates:
            acc = accuracy_score(y, (scores >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)
        return best_t

    def _oof_threshold(self, X, y):
        class_counts = np.bincount(y, minlength=2)
        n_splits = min(5, int(class_counts.min()))
        if n_splits < 2:
            return 0.5

        oof = np.zeros(len(y), dtype=np.float32)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self._random_seed)
        specs = self._slices(X.shape[1])

        for train_idx, val_idx in splitter.split(X, y):
            fold_scores = []
            for feature_slice, model_name, pca_dim in specs:
                member = self._fit_one(X[train_idx, feature_slice], y[train_idx], model_name, pca_dim)
                fold_scores.append(self._score_one(X[val_idx], feature_slice, member))
            oof[val_idx] = np.mean(np.vstack(fold_scores), axis=0)

        return self._best_threshold(y, oof)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._threshold = self._oof_threshold(X, y) if len(y) >= 20 else 0.5
        self._members = []
        for feature_slice, model_name, pca_dim in self._slices(X.shape[1]):
            member = self._fit_one(X[:, feature_slice], y, model_name, pca_dim)
            self._members.append((feature_slice, member))
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.int64)
        self._threshold = self._best_threshold(y_val, self.predict_proba(X_val)[:, 1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X = np.asarray(X, dtype=np.float32)
        prob_pos = np.clip(self._scores(X), 0.0, 1.0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

