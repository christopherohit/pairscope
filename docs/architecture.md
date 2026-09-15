# PairScope: Renderer-Observable Same-Utterance Adaptation

> This document is a dated research record. The executable module in this
> repository is `pairscope.py`; references to `probe/`, renderer checkouts, and
> unpublished artifacts are provenance only.

## Status

PairScope is a proposed target-specific correction chart for personalized
audio-driven 3D talking heads. The completed multi-identity pilot is a
**motion-space plus generic-FLAME geometry proxy**. The proxy mechanism
shows a positive post-hoc FLAME-proxy-MSE improvement over paired low-rank
calibration (`+0.0028`, identity-bootstrap 95% interval `[+0.0004,+0.0058]`,
`p=0.031250`) and a positive aligned-versus-shuffled gap (`+0.0052`). It
fails the stricter content-across-all-identities and utility-versus-full-D
XLS-R gates. No 3DGS integration is authorized by this result.

An additional single-identity pilot now extracts a stable Gram from the actual
personalized EmoTaG renderer. PairScope fails both its untouched motion-proxy
certificate and its 25-frame image certificate, so its query output is the
exact frozen EmoTaG fallback. The architecture remains an experiment design
with a partially supported proxy mechanism, not a conference-ready claim.

A support-leverage OOD trust ablation raises mean proxy-pilot content from
`0.9543` to `0.9600` and repairs Jae-in, but still loses full-D utility and
leaves May below the content guard. It is retained as an optional safety
component, not as a successful architecture.

## Problem Boundary

Let `a_t` be audio features, `p_t` a frozen audio-to-motion teacher output,
`B_S(p_t)` a target-specific paired base calibration fitted on the support
clip, and `y_t` tracked target motion. The paired residual is

```text
E_t = y_t - B_S(p_t).
```

The standard residual adapter predicts all 53 motion coordinates in the same
coordinate system. PairScope instead asks a narrower question: which paired
residual directions are observable by the *target renderer* around the support
motion? It learns only those directions and has an exact base fallback when no
observable residual survives.

PairScope does not claim that a generic FLAME metric is a renderer metric. The
generic metric is used only to test whether the chart idea is distinguishable
from identity-space and matched-spectrum controls before the expensive renderer
experiment.

## Renderer-Observable Chart

Let `R_theta(m)` be a frozen personalized renderer and `phi` a frozen image or
renderer-feature map. For support frame `t`, define the motion Jacobian

```text
J_t = d phi(R_theta(m_t)) / d m_t.
```

With non-negative feature weights `W_t`, accumulate a support Gram matrix

```text
G = mean_t J_t^T W_t J_t.
```

`G` is positive semidefinite and measures first-order renderer-visible motion
energy. It is not required to be full rank: motion changes that do not change
the selected renderer features should not become correction coordinates.

For a support residual matrix `E_c` (rows are frames and columns are motion
coordinates), form the paired observable kernel

```text
K = E_c G E_c^T.
```

Let `(A, Lambda)` be the retained positive eigenpairs of `K`. Define

```text
U = E_c^T A Lambda^(-1/2).
```

Up to numerical tolerance,

```text
U^T G U = I.
```

`U` is a target-specific, paired, renderer-observable correction chart. A
small predictor maps frozen audio/teacher context `x_t` to chart coefficients:

```text
h(x_t) = Ridge(x_t -> E_t G U)
delta_t = U h(x_t)
m_hat_t = B_S(p_t) + gamma * delta_t.
```

The implementation keeps the raw support residual rather than subtracting its
mean, so a constant renderer-visible offset is retained. Before basis
construction it projects away the null space of `G`; arbitrary support noise in
unobservable coordinates therefore cannot leak into the prediction.

If the observable residual energy is zero, or no eigenvalue exceeds the
declared floor, `PairScope` returns exactly `B_S(p_t)`. The implementation and
invariants are in `probe/pairscope.py` and `probe/test_pairscope.py`.

## Support Certificate

The adapter is not enabled solely because support fit error decreases. The
support clip is split chronologically at 25 FPS:

1. 100 frames (4 seconds) fit alignment, the base calibrator, metric state, and
   candidate chart models.
2. 25 frames (1 second) select rank, ridge, residual scale, and any trust
   threshold.
3. 15 seconds of disjoint query frames are used only for the final report.

The renderer-stage certificate should compare base and corrected renders on the
reserved support-validation frames. For every predeclared non-zero circular
shift of the audio/teacher conditioning, rerun the same search and compute a
counterfactual null benefit. Enable the chart only if:

```text
benefit_aligned > 0
and
benefit_aligned > quantile_0.9(benefit_shifted_nulls).
```

The certificate is a correspondence test, not a causal proof and not a human
preference test. If it fails, inference returns the frozen base render exactly.

## Motion-Proxy Protocol Completed

The first implementation uses a generic FLAME surface finite-difference Gram:
`probe/flame_geometric_metric.py` decodes the licensed generic FLAME model,
perturbs each of the 53 SMIRK expression/jaw coordinates around support frames,
and averages vertex-displacement Jacobian Grams over the face mask. It reports
`renderer_evidence: false` deliberately. Controls are:

- identity metric `G=I`;
- random metric with the same eigenvalue spectrum as the FLAME Gram;
- same-utterance support correspondence versus a support-feature permutation;
- paired low-rank base;
- full-D strict XLS-R residual adapter.

The runner is `probe/run_pairscope_proxy.py`; the summary is
`probe/results/pairscope_proxy_7id_3seed.md`.

### Proxy result

Values below are identity-level means after averaging three seeds. Positive
values favor PairScope FLAME geometry; lower-error metrics are converted to
positive improvements.

| Comparison | Correlation | Motion MSE | FLAME-proxy MSE | Proxy wins | Mean content |
|---|---:|---:|---:|---:|---:|
| vs paired low-rank | +0.0051 | +0.0015 | +0.0028 | 6/7 | 0.9543 |
| vs full-D strict XLS-R | -0.0075 | -0.0018 | +0.0003 | 4/7 | 0.9543 |
| vs shuffled correspondence | +0.0067 | +0.0034 | +0.0052 | 6/7 | 0.9543 |
| vs identity metric | -0.0080 | -0.0019 | +0.0009 | 5/7 | 0.9543 |
| vs random matched spectrum | +0.0004 | +0.0000 | +0.0024 | 7/7 | 0.9543 |

The post-hoc proxy mechanism gate fails because Jae-in and May fall below the
`0.95` content guard. Utility also fails: correlation and motion MSE do not
match the full-D strict XLS-R residual baseline. The proxy is useful evidence
that metric-aware paired directions are not identical to arbitrary coordinate
selection, but it is not evidence of image-quality improvement.

## Actual Renderer Experiment

The first actual-renderer pilot is complete for the neutral Obama EmoTaG 20k
checkpoint. A standalone extractor finite-differences downsampled composited
RGB with respect to all 53 motion coordinates without modifying the dirty
EmoTaG checkout. The fit-only Gram uses frames `0,18,37,56,74`; the final Gram
uses `0,31,62,93,124`. Central-difference Grams at epsilon `0.001` and `0.002`
have cosine `0.99991` and `0.99821`, respectively, so extraction instability
is not the observed failure.

The experiment uses `75 fit / 25 selection / 25 untouched support certificate / 250 query`.
Selection chooses raw EmoTaG over the older paired low-rank base.
On the untouched motion certificate, PairScope's renderer-proxy MSE is
`0.8454149` versus `0.8374637` for raw EmoTaG; shuffled correspondence returns
the raw baseline exactly. On the actual 25-frame image certificate, PSNR is
`20.604175` versus `20.604496` for the selected base and the MSE benefit is
`-6.40e-7`. LPIPS is slightly lower (`0.068706` versus `0.068744`) but the
predeclared MSE/correspondence gate still fails. Query inference therefore
uses the exact base fallback and no query render is authorized.

Full details are in `research/PairScope_actual_renderer_obama.md`. This is one
identity and one seed; it establishes a negative mechanism result, not a
general conclusion about all renderers.

Any future multi-identity expansion should repeat the following protocol:

1. Train or load the verified neutral Obama 20k EmoTaG checkpoint without
   changing the documented support/query split.
2. Freeze Gaussian parameters, camera, FLAME identity/shape, and renderer
   feature extraction.
3. For each support frame, finite-difference or autodifferentiate the selected
   renderer feature map with respect to the 53-dimensional FLAME motion.
4. Accumulate `G` with a declared feature mask and normalization. Store the
   Jacobian/Gram hash and the feature definition in the result manifest.
5. Use a three-way support split for fitting, selection, and an untouched image
   certificate; refit only after the certificate passes.
6. Run aligned, shuffled, identity-metric, random-spectrum, and zero-energy
   controls with the same search budget.
7. Report PSNR/SSIM/LPIPS and motion metrics separately. A motion result must
   never be presented as image-quality evidence.

Finite differences are acceptable only as a validation path; a production
implementation should use the differentiable renderer graph where possible.
The final correction must be injected before FLAME deformation, with a
zero-initialized projector or scalar so rejection recovers the baseline exactly.

## Novelty Boundary

The following mechanisms are already occupied and must not be used as the sole
novelty claim:

| Adjacent work | Occupied idea | PairScope boundary |
|---|---|---|
| VOCA, Cudeiro et al. (2019) | Audio-driven 3D face motion with identity/style factors | PairScope is not a generic style-conditioned teacher. |
| DFRF, Shen et al. (ECCV 2022) | Few-shot personalized neural radiance fields | PairScope defines a paired renderer-observable correction chart. |
| Imitator, Thambiraja et al. (2023) | Short-reference identity-specific style optimization | PairScope does not optimize a persistent style embedding. |
| StyleTalk, Ma et al. (2023) | Reference-video style code and style-aware decoder | PairScope is not reference-style encoding. |
| Mimic, Fu et al. (2023) | Style/content disentanglement in 3D facial animation | PairScope is not a generic disentangled latent space. |
| TalkLoRA, Saunders & Namboodiri (2024) | Low-rank adaptation of speech-driven animation | Low rank is only an implementation detail; the chart is defined by paired renderer observability. |
| GaussianSpeech, Aneja et al. (2024) | Audio-conditioned 3D Gaussian avatar generation | PairScope does not claim a new Gaussian representation. |
| PC-Talk (2025) | Lip alignment and emotion control | PairScope is not word-level or emotion-control editing. |
| Perceptually Accurate 3D Talking Head (2025) | Speech-mesh representation and perceptual metrics | PairScope may use a feature map but its contribution is the paired chart/certificate. |
| SubtleTalk (2026) | Residual flow matching and weakly correlated regional dynamics | PairScope is not a universal residual-flow prior. |
| TT-SAC (2026) | Generic parameter-free test-time generator feedback | PairScope uses target paired support and a renderer-observability metric. |
| EmoTaG (CVPR 2026) | Few-shot 3DGS personalization, residual/gate, AdaIN | PairScope must not rename or reproduce these components; it is a conditional upstream motion correction. |

The local full-text audit found no exact operational method that constructs a
target/avatar-specific correction chart from paired residuals and the actual
personalized renderer sensitivity. This is a scoped negative search result,
not proof of novelty. TT-SAC discusses generator-feature Jacobians in a
theoretical fixed-point analysis; PairScope must distinguish that from an
operational support-time renderer Jacobian used to choose motion coordinates.

## Failure Policy

- If the actual renderer Gram is numerically unstable, freeze the baseline and
  report the failed Jacobian extraction; do not substitute a favorable proxy
  without relabeling it.
- If the support certificate fails, render the exact base output.
- If content falls below the declared threshold on validation, scale to zero or
  abstain; do not use query ground truth to rescue the episode.
- If shuffled correspondence performs similarly to aligned support, close the
  correspondence claim.
- If the actual image-space query gain does not beat the paired low-rank base,
  PairScope remains a diagnostic architecture even if motion proxy metrics
  improve.

## Sources

1. D. Cudeiro, T. Bolkart, C. Laidlaw, A. Ranjan, and M. J. Black, “Capture, Learning, and Synthesis of 3D Speaking Styles,” arXiv:1905.03079, 2019. https://arxiv.org/abs/1905.03079
2. S. Shen, W. Li, Z. Zhu, Y. Duan, J. Zhou, and J. Lu, “Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis,” ECCV 2022, arXiv:2207.11770. https://arxiv.org/abs/2207.11770
3. B. Thambiraja, I. Habibie, S. Aliakbarian, D. Cosker, and C. Theobalt, “Imitator: Personalized Speech-driven 3D Facial Animation,” arXiv:2301.00023, 2023. https://arxiv.org/abs/2301.00023
4. Y. Ma et al., “StyleTalk: One-shot Talking Head Generation with Controllable Speaking Styles,” arXiv:2301.01081, 2023. https://arxiv.org/abs/2301.01081
5. H. Fu et al., “Mimic: Speaking Style Disentanglement for Speech-Driven 3D Facial Animation,” arXiv:2312.10877, 2023. https://arxiv.org/abs/2312.10877
6. J. Saunders and V. P. Namboodiri, “TalkLoRA: Low-Rank Adaptation for Speech-Driven Animation,” arXiv:2408.13714, 2024. https://arxiv.org/abs/2408.13714
7. S. Aneja et al., “GaussianSpeech: Audio-Driven Gaussian Avatars,” arXiv:2411.18675, 2024. https://arxiv.org/abs/2411.18675
8. B. Wang et al., “PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation,” arXiv:2503.14295, 2025. https://arxiv.org/abs/2503.14295
9. L. Chae-Yeon et al., “Perceptually Accurate 3D Talking Head Generation: New Definitions, Speech-Mesh Representation, and Evaluation Metrics,” arXiv:2503.20308, 2025. https://arxiv.org/abs/2503.20308
10. M. Ding et al., “SubtleTalk: Generating Controllable Weakly-correlated Facial Dynamics for 3D Talking Heads,” arXiv:2608.06408, 2026. https://arxiv.org/abs/2608.06408
11. “TT-SAC: Test-Time Self-Adaptive Conditioning for Audio-Driven Human Animation,” arXiv:2605.25488, 2026. https://arxiv.org/abs/2605.25488
12. H. Xu et al., “EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization,” arXiv:2603.21332, 2026. https://arxiv.org/abs/2603.21332
