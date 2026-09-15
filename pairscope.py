"""Renderer-observable paired residual coordinates for PairScope.

The renderer is represented by its support-averaged motion Gram matrix
G = mean(J_t.T @ W_t @ J_t). PairScope extracts residual modes that are large
under this image-space metric and predicts only their coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def renderer_gram(jacobians: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Accumulate J^T W J without retaining a pixel-sized feature covariance."""
    jacobians = np.asarray(jacobians, dtype=np.float64)
    if jacobians.ndim == 2:
        jacobians = jacobians[None]
    if jacobians.ndim != 3:
        raise ValueError("jacobians must have shape [T, K, D] or [K, D]")
    if weights is None:
        gram = np.einsum("tki,tkj->ij", jacobians, jacobians)
        scale = jacobians.shape[0] * jacobians.shape[1]
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights[None], jacobians.shape[:2])
        if weights.shape != jacobians.shape[:2]:
            raise ValueError("weights must have shape [K] or [T, K]")
        if np.any(weights < 0):
            raise ValueError("renderer feature weights must be nonnegative")
        gram = np.einsum("tk,tki,tkj->ij", weights, jacobians, jacobians)
        scale = float(weights.sum())
    if scale <= 0:
        return np.zeros((jacobians.shape[-1], jacobians.shape[-1]))
    return 0.5 * (gram + gram.T) / scale


@dataclass
class PairScopeDiagnostics:
    active: bool
    requested_rank: int
    effective_rank: int
    observable_energy: float
    retained_energy_fraction: float
    metric_orthogonality_error: float


class PairScope:
    """Fit an audio-to-residual map in a renderer-observable motion chart."""

    def __init__(self, rank: int = 8, ridge: float = 10.0, eigen_floor: float = 1e-8):
        if rank <= 0:
            raise ValueError("rank must be positive")
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.eigen_floor = float(eigen_floor)
        self.basis_: np.ndarray | None = None
        self.metric_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.system_: np.ndarray | None = None
        self.support_normalized_: np.ndarray | None = None
        self.support_leverage_: np.ndarray | None = None
        self.diagnostics_: PairScopeDiagnostics | None = None

    def fit(
        self,
        support_features: np.ndarray,
        support_base: np.ndarray,
        support_real: np.ndarray,
        metric: np.ndarray,
    ) -> "PairScope":
        x = np.asarray(support_features, dtype=np.float64)
        base = np.asarray(support_base, dtype=np.float64)
        real = np.asarray(support_real, dtype=np.float64)
        metric = np.asarray(metric, dtype=np.float64)
        if x.ndim != 2 or base.ndim != 2 or real.shape != base.shape:
            raise ValueError("features, base, and real must be aligned 2D arrays")
        if len(x) != len(base):
            raise ValueError("support features and motion must have equal length")
        if metric.shape != (base.shape[1], base.shape[1]):
            raise ValueError("metric shape must match the motion dimension")

        metric = 0.5 * (metric + metric.T)
        metric_eigenvalues = np.linalg.eigvalsh(metric)
        if metric_eigenvalues.min() < -1e-8 * max(1.0, metric_eigenvalues.max()):
            raise ValueError("renderer metric must be positive semidefinite")

        residual = real - base

        metric_values, metric_vectors = np.linalg.eigh(metric)
        metric_floor = self.eigen_floor * max(
            1.0, float(metric_values[-1]) if len(metric_values) else 0.0,
        )
        observable = metric_values > metric_floor
        metric_projector = metric_vectors[:, observable] @ metric_vectors[:, observable].T

        kernel = residual @ metric @ residual.T
        kernel = 0.5 * (kernel + kernel.T)
        values, vectors = np.linalg.eigh(kernel)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        total = float(np.maximum(values, 0).sum())
        floor = self.eigen_floor * max(1.0, float(values[0]) if len(values) else 0.0)
        keep = np.flatnonzero(values > floor)[: self.rank]

        self.metric_ = metric
        if len(keep) == 0:
            self.basis_ = np.zeros((base.shape[1], 0))
            self.feature_mean_ = x.mean(axis=0)
            self.feature_scale_ = np.ones(x.shape[1])
            self.coef_ = np.zeros((x.shape[1], 0))
            self.intercept_ = np.zeros(0)
            self.system_ = None
            self.support_normalized_ = None
            self.support_leverage_ = np.zeros(len(x))
            self.diagnostics_ = PairScopeDiagnostics(
                active=False,
                requested_rank=self.rank,
                effective_rank=0,
                observable_energy=total,
                retained_energy_fraction=0.0,
                metric_orthogonality_error=0.0,
            )
            return self

        selected_values = values[keep]
        selected_vectors = vectors[:, keep]
        # Remove metric-null components, which the renderer cannot identify and
        # which otherwise leak arbitrary support noise into predicted motion.
        basis = (
            metric_projector
            @ residual.T
            @ selected_vectors
            @ np.diag(1.0 / np.sqrt(selected_values))
        )
        target = residual @ metric @ basis

        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale[feature_scale < 1e-8] = 1.0
        normalized = (x - feature_mean) / feature_scale
        intercept = target.mean(axis=0)
        target_centered = target - intercept
        system = normalized.T @ normalized + self.ridge * np.eye(normalized.shape[1])
        coef = np.linalg.solve(system, normalized.T @ target_centered)

        identity = basis.T @ metric @ basis
        self.basis_ = basis
        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.coef_ = coef
        self.intercept_ = intercept
        self.system_ = system
        self.support_normalized_ = normalized
        self.support_leverage_ = None
        self.diagnostics_ = PairScopeDiagnostics(
            active=True,
            requested_rank=self.rank,
            effective_rank=len(keep),
            observable_energy=total,
            retained_energy_fraction=float(selected_values.sum() / max(total, 1e-12)),
            metric_orthogonality_error=float(np.max(np.abs(identity - np.eye(len(keep))))),
        )
        return self

    def predict_residual(self, query_features: np.ndarray) -> np.ndarray:
        if self.diagnostics_ is None or self.basis_ is None:
            raise RuntimeError("PairScope must be fitted before prediction")
        x = np.asarray(query_features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("query_features must be a 2D array")
        if not self.diagnostics_.active:
            return np.zeros((len(x), self.metric_.shape[0]))
        normalized = (x - self.feature_mean_) / self.feature_scale_
        coordinates = normalized @ self.coef_ + self.intercept_
        return coordinates @ self.basis_.T

    def leverage(self, query_features: np.ndarray) -> np.ndarray:
        """Return ridge leverage relative to the paired support feature cloud."""
        if self.diagnostics_ is None or self.feature_mean_ is None:
            raise RuntimeError("PairScope must be fitted before leverage")
        if self.system_ is None:
            return np.zeros(len(np.asarray(query_features)))
        x = np.asarray(query_features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("query_features must be a 2D array")
        normalized = (x - self.feature_mean_) / self.feature_scale_
        solved = np.linalg.solve(self.system_, normalized.T).T
        return np.maximum(0.0, np.einsum("ti,ti->t", normalized, solved))

    def trust_weights(
        self,
        query_features: np.ndarray,
        quantile: float | None = 1.0,
        power: float = 0.5,
    ) -> np.ndarray:
        """Attenuate residuals for query features outside paired support.

        The threshold is estimated from support leverage only. ``None`` is an
        explicit no-gate setting; the exact fallback is all ones.
        """
        if quantile is None:
            return np.ones(len(np.asarray(query_features)))
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if power <= 0.0:
            raise ValueError("power must be positive")
        if (
            self.system_ is None
            or self.support_normalized_ is None
            or len(self.support_normalized_) == 0
        ):
            return np.ones(len(np.asarray(query_features)))
        return self.trust_from_leverage(
            self.leverage(query_features), quantile=quantile, power=power,
        )

    def trust_from_leverage(
        self,
        scores: np.ndarray,
        quantile: float,
        power: float = 0.5,
    ) -> np.ndarray:
        """Map precomputed query leverage to support-calibrated trust."""
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if power <= 0.0:
            raise ValueError("power must be positive")
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError("leverage scores must be a 1D array")
        if self.support_leverage_ is None:
            solved = np.linalg.solve(self.system_, self.support_normalized_.T).T
            self.support_leverage_ = np.einsum(
                "ti,ti->t", self.support_normalized_, solved,
            )
        threshold = float(np.quantile(self.support_leverage_, quantile))
        ratio = threshold / np.maximum(scores, 1e-12)
        return np.minimum(1.0, ratio**power)

    def predict(self, query_features: np.ndarray, query_base: np.ndarray) -> np.ndarray:
        base = np.asarray(query_base, dtype=np.float64)
        residual = self.predict_residual(query_features)
        if residual.shape != base.shape:
            raise ValueError("query features and base motion must have equal length")
        return base + residual
