# PairScope

PairScope is a motion-space research operator that constructs a
target-specific correction chart from two pieces of support evidence:

1. paired real-minus-base motion residuals; and
2. a positive-semidefinite metric describing which motion directions are
   visible to a renderer or renderer-feature map.

It predicts only residual coordinates that survive both sources of evidence
and can attenuate corrections for query features with high support leverage.

## Research Status

**PairScope is not a validated rendering improvement.** Its evidence is mixed:

- A seven-identity generic-FLAME geometry proxy showed `+0.0028` proxy-MSE
  improvement over paired low-rank and a `+0.0052` aligned-minus-shuffled gap.
- The proxy failed the stricter all-identity content guard and lost utility to
  a full-dimensional XLS-R residual baseline.
- A stable Gram extracted from one actual EmoTaG renderer failed both the
  untouched motion-proxy certificate and the 25-frame image certificate.
- Actual certificate PSNR was `20.604175` versus `20.604496` for the selected
  base, with MSE benefit `-6.40e-7`.

The actual-renderer experiment therefore uses the exact frozen base for query
inference. The positive generic proxy is mechanism evidence only; it must not be
reported as image quality, renderer success, cross-identity generalization, or
proof of novelty. See [docs/architecture.md](docs/architecture.md) for the
complete experiments and claim boundary.

## Method

For support residuals `E = real - base` and renderer metric `G`, PairScope
forms the paired observable kernel:

```text
K = E G E^T
```

From its retained positive eigenpairs `(A, Lambda)`, it constructs:

```text
U = E^T A Lambda^(-1/2)
```

Numerically, `U^T G U` should be the identity. `U` is the correction chart.
Support features are regressed to chart coordinates and mapped back to motion:

```text
coordinates = Ridge(features -> E G U)
residual    = coordinates U^T
prediction  = base + residual
```

Before chart construction, residual components in the null space of `G` are
removed. If no observable residual eigenvalue survives the configured floor,
PairScope becomes inactive and returns an exact zero residual.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

NumPy is the only runtime dependency.

## Input Contract

| Argument | Shape | Description |
|---|---|---|
| `jacobians` | `[T, K, D]` or `[K, D]` | Renderer-feature derivatives with respect to motion |
| `weights` | compatible with `K` | Optional non-negative renderer-feature weights |
| `support_features` | `[T, F]` | Frozen audio or conditioning features |
| `support_base` | `[T, D]` | Frozen support motion |
| `support_real` | `[T, D]` | Paired tracked real motion |
| `metric` | `[D, D]` | Positive-semidefinite renderer motion metric |
| `query_features` | `[Tq, F]` | Query conditioning features |
| `query_base` | `[Tq, D]` | Frozen query motion |

`renderer_gram` accepts one Jacobian or a frame stack, accumulates `J^T W J`,
symmetrizes it, and normalizes by the number of frame-feature samples or by the
sum of supplied weights. The caller is responsible for declaring the renderer
feature, mask, frame sample, and Jacobian method.

## Quick Start

```python
from pairscope import PairScope, renderer_gram

metric = renderer_gram(support_jacobians)

model = PairScope(
    rank=8,
    ridge=10.0,
    eigen_floor=1e-8,
).fit(
    support_features,
    support_base,
    support_real,
    metric,
)

residual = model.predict_residual(query_features)
trust = model.trust_weights(query_features, quantile=0.99, power=0.5)
motion = query_base + trust[:, None] * residual

print(model.diagnostics_)
```

`PairScope.predict(query_features, query_base)` adds the unattenuated residual.
Use explicit trust weighting, as above, when the evaluation protocol includes
the support-leverage safety component.

## Public API

### `renderer_gram(jacobians, weights=None)`

Constructs a normalized PSD motion metric. A metric derived from generic FLAME
geometry must be labeled a proxy; only a Jacobian from the actual frozen
personalized renderer and declared feature map is renderer evidence.

### `PairScope(rank=8, ridge=10.0, eigen_floor=1e-8)`

- `rank`: maximum paired observable chart rank.
- `ridge`: feature-to-chart regression penalty.
- `eigen_floor`: relative numerical floor for metric and kernel eigenvalues.

After `fit`, `diagnostics_` reports whether the chart is active, requested and
effective rank, observable energy, retained-energy fraction, and maximum
metric-orthogonality error.

### `predict_residual(...)` and `predict(...)`

`predict_residual` returns zero exactly when no observable paired direction was
retained. `predict` checks shape compatibility and adds the residual to the
query base. Neither method performs an aligned-versus-shuffled certificate.

### `leverage(...)` and `trust_weights(...)`

`leverage` measures query feature leverage against the support ridge system.
`trust_weights` calibrates a threshold from support leverage only and maps high
query leverage toward zero. Pass `quantile=None` for an explicit all-ones
no-gate ablation.

## Certificate and Integration Boundary

The module implements the chart, not the complete experiment. Before applying
it to query rendering, an external support-only protocol should:

1. split support into fit, selection, and untouched certificate segments;
2. freeze metric extraction, rank, ridge, gain, and trust settings;
3. compare aligned support with equal-budget shifted or shuffled support;
4. compare identity and matched-spectrum random metrics;
5. evaluate the actual renderer on the reserved certificate frames;
6. return the exact frozen base if the certificate fails.

Query images and query real motion are metrics-only. They must not select the
chart, metric, gain, trust threshold, or acceptance decision.

## Testing

```bash
python -m unittest -v
```

The tests verify:

- explicit and vectorized renderer-Gram agreement;
- exact fallback for zero observable residual;
- retention of constant observable offsets;
- `G`-orthonormality of the learned chart;
- suppression of noise in metric-null motion directions;
- attenuation of out-of-distribution query features by leverage trust.

These tests establish numerical invariants on synthetic arrays. They do not
validate a renderer or perceptual claim.

## Limitations

- A renderer Gram measures local sensitivity, not the signed direction needed
  to reduce image error; PairLift and PairQuotient study that separate issue.
- Finite-difference Jacobians need perturbation-size and frame-sampling audits.
- A low-rank chart can discard useful motion even when it is numerically stable.
- A generic FLAME metric is not interchangeable with a personalized renderer.
- Motion-proxy improvements cannot be presented as PSNR, SSIM, LPIPS, sync, or
  human-preference improvements.
- The actual-renderer result currently covers one identity and checkpoint.

## Repository Layout

- `pairscope.py` - Gram construction, chart fit, leverage, trust, prediction.
- `test_pairscope.py` - metric, fallback, orthogonality, noise, and OOD tests.
- `docs/architecture.md` - proxy and actual-renderer studies, controls, sources.

## License and Asset Responsibility

Code and documentation are MIT licensed. No renderer, checkpoints, identity
media, tracked motion, datasets, Jacobians, extracted features, or licensed
FLAME assets are distributed. Users must obtain and govern those assets under
their original terms, including consent, privacy, and biometric-data duties.
