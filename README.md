# PairScope

PairScope extracts paired residual coordinates that are observable under a
renderer-derived motion Gram matrix, then predicts only those coordinates from
support audio features.

## Status

A generic-FLAME proxy showed a positive mechanism signal, but the stricter
content/utility gates and the actual-renderer image certificate failed. The
implementation therefore remains an experiment design with exact fallback
behavior, not a validated rendering improvement. See `docs/architecture.md`.

## Install And Test

```bash
pip install -e .
python -m unittest -v
```

## Minimal Use

```python
from pairscope import PairScope, renderer_gram

metric = renderer_gram(support_jacobians)
model = PairScope(rank=8).fit(features, base_motion, real_motion, metric)
motion = model.predict(query_features, query_base)
```

No renderer, model weights, media, or FLAME assets are distributed. Licensed
under MIT.
