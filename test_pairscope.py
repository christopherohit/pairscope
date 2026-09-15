import unittest

import numpy as np

from pairscope import PairScope, renderer_gram


def ridge_predict(train_x, train_y, query_x, ridge=1e-3):
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.0
    x = (train_x - mean) / scale
    q = (query_x - mean) / scale
    intercept = train_y.mean(0)
    coef = np.linalg.solve(
        x.T @ x + ridge * np.eye(x.shape[1]),
        x.T @ (train_y - intercept),
    )
    return q @ coef + intercept


class PairScopeTest(unittest.TestCase):
    def test_renderer_gram_matches_explicit_accumulation(self):
        rng = np.random.default_rng(1)
        jacobians = rng.normal(size=(7, 5, 4))
        weights = rng.uniform(size=(7, 5))
        expected = sum(
            jacobians[t].T @ np.diag(weights[t]) @ jacobians[t]
            for t in range(len(jacobians))
        ) / weights.sum()
        np.testing.assert_allclose(renderer_gram(jacobians, weights), expected)

    def test_zero_observable_residual_is_exact_fallback(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(20, 3))
        base = rng.normal(size=(20, 6))
        model = PairScope(rank=3).fit(x, base, base, np.eye(6))
        query_base = rng.normal(size=(9, 6))
        np.testing.assert_allclose(model.predict(rng.normal(size=(9, 3)), query_base), query_base)
        self.assertFalse(model.diagnostics_.active)

    def test_constant_observable_residual_is_retained(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20, 3))
        base = rng.normal(size=(20, 6))
        offset = np.array([0.4, -0.2, 0.1, 0.0, 0.0, 0.0])
        metric = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        model = PairScope(rank=3, ridge=1e-3).fit(x, base, base + offset, metric)
        predicted = model.predict_residual(rng.normal(size=(9, 3)))
        np.testing.assert_allclose(predicted, np.broadcast_to(offset, predicted.shape), atol=1e-10)
        self.assertTrue(model.diagnostics_.active)

    def test_basis_is_orthonormal_under_renderer_metric(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 5))
        base = np.zeros((40, 8))
        real = rng.normal(size=(40, 8))
        j = rng.normal(size=(12, 8))
        metric = renderer_gram(j)
        model = PairScope(rank=4).fit(x, base, real, metric)
        got = model.basis_.T @ metric @ model.basis_
        np.testing.assert_allclose(got, np.eye(model.diagnostics_.effective_rank), atol=1e-7)

    def test_unobservable_noise_is_suppressed(self):
        rng = np.random.default_rng(4)
        n_train, n_test, feature_dim, motion_dim = 36, 200, 6, 30
        train_x = rng.normal(size=(n_train, feature_dim))
        test_x = rng.normal(size=(n_test, feature_dim))
        mapping = rng.normal(size=(feature_dim, 3))
        visible_basis = np.zeros((motion_dim, 3))
        visible_basis[:3] = np.eye(3)
        train_visible = train_x @ mapping @ visible_basis.T
        test_visible = test_x @ mapping @ visible_basis.T
        train_residual = train_visible.copy()
        train_residual[:, 3:] += 4.0 * rng.normal(size=(n_train, motion_dim - 3))
        metric = np.diag([1.0] * 3 + [0.0] * (motion_dim - 3))

        scoped = PairScope(rank=3, ridge=1e-3).fit(
            train_x, np.zeros_like(train_residual), train_residual, metric,
        ).predict_residual(test_x)
        full = ridge_predict(train_x, train_residual, test_x)
        scoped_visible_error = np.mean((scoped[:, :3] - test_visible[:, :3]) ** 2)
        full_visible_error = np.mean((full[:, :3] - test_visible[:, :3]) ** 2)
        self.assertLess(scoped_visible_error, full_visible_error + 1e-10)
        np.testing.assert_allclose(scoped[:, 3:], 0.0, atol=1e-10)

    def test_support_leverage_trust_attenuates_out_of_distribution_queries(self):
        rng = np.random.default_rng(11)
        support_x = rng.normal(size=(40, 4))
        support_base = np.zeros((40, 6))
        support_real = np.column_stack([
            support_x[:, :2],
            np.zeros((40, 4)),
        ])
        model = PairScope(rank=2, ridge=1.0).fit(
            support_x, support_base, support_real, np.eye(6),
        )
        in_domain = support_x[:8]
        out_domain = support_x[:8] + 20.0
        in_trust = model.trust_weights(in_domain, quantile=1.0, power=1.0)
        out_trust = model.trust_weights(out_domain, quantile=1.0, power=1.0)
        self.assertGreaterEqual(float(np.min(in_trust)), 0.99)
        self.assertLess(float(np.mean(out_trust)), 0.1)
        np.testing.assert_allclose(
            model.trust_weights(out_domain, quantile=None), 1.0,
        )


if __name__ == "__main__":
    unittest.main()
