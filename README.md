# flash-moba-mlx

A small independent MLX implementation of *Optimizing Mixture of Block Attention*
(arXiv:2511.11571), plus a toy decoder-only language model trained on a synthetic
multi-key recall task to try the mechanism out, a Metal kernel for the block-sparse
forward pass, and a tile-schedule visualisation of what the kernel actually
schedules. It runs on Apple silicon CPU/GPU through MLX.

Two papers are involved and they are not the same paper. **Mixture of Block
Attention (MoBA) is from Lu et al., arXiv:2502.13189**: the block partitioning, the
centroid router `s_i = q^T k~_i`, the top-k selection and the causal masking are
that paper's scheme, and the paper implemented here reviews them in its Section 2.
**arXiv:2511.11571 is about optimising that scheme.** What this repo takes from
2511.11571 specifically is: the statistical model of the router as a
needle-in-a-haystack problem and the signal-to-noise ratio
`SNR = Delta_mu_eff * sqrt(d / 2B)` derived from it (Section 3); the two design
principles that fall out of it, namely smaller blocks and a short depthwise causal
convolution on keys (Section 3.3 and Appendix C); and the FlashMoBA kernel design
of Section 4, a tiled top-k followed by a gather-and-densify forward pass with an
online softmax instead of a materialised score matrix. The original MoBA design is
not attributed to 2511.11571 anywhere here.

This is not a reproduction. It is 160k parameters on 128-token sequences trained
for 4000 steps, it measures nothing about the paper's 340M/1B models, its 100B-token
training runs, its 8K-512K contexts or its 14.7x kernel speedup, and none of its
numbers are comparable to the paper's. What it does do is implement the mechanism
end to end on MLX, test the invariants that would catch a mistake, and report what
happens at toy scale — including two results that are worse than the paper's story.

## What is in here

**Block statistics** (`moba/blocks.py`). `K = L / B` contiguous key blocks and the
centroid `k~_j` are the paper's Section 2. The router must never score a block that
contains only future keys, so a query at position `t` has candidates `j <= t // B`
and block `t // B` is scored with the *causal prefix mean* of its own keys,
`(1/(t - B*t//B + 1)) sum_{s=B*(t//B)}^{t} k_s`, not the whole block mean. That is
the one place where the paper's description ("each query always attends to its
current block with causal masking") is ambiguous about whether the current block's
*score* may see future keys inside it; here it may not, so no key after `t` can
influence any selection. `causal_block_scores` computes the same numbers as
`causal_block_centroids` in O(LK) instead of O(LKd) by one matmul against all
centroids plus a diagonal overwrite.

**Router** (`moba/router.py`). `topk_blocks` masks non-candidates with `-inf`,
argsorts the negated scores once and takes the first `k`. Rows with fewer than `k`
candidates repeat index 0 in the unused slots with `selected=False`, which is how
the reference path, the tiled path and the Metal kernel all know to ignore them.
The router is per head: every (batch, head) pair routes independently, matching the
paper's kernel, which it says is fully parallelised over batches and heads.

**Four attention paths** (`moba/attention.py`, `moba/tiled.py`, `moba/kernel.py`).

* `dense_attention` is ordinary causal attention, the baseline and the target the
  `k = K` case has to hit.
* `moba_reference_attention` is MoBA written the obvious way: build the full L x L
  score matrix, apply the block-selection mask as an additive `-inf` bias, softmax,
  reduce. It is O(L^2) in time and memory and reads every key whether selected or
  not, which is exactly the inefficiency the paper's Section 4 is about. It is the
  correctness oracle.
* `tiled_attention` (`moba/tiled.py`) is the block-sparse path. For each of the `k`
  selected slots it gathers *only* that block's `B` keys and values and folds them
  into a running max / running sum / accumulator triple — the same online softmax
  the paper's gather-and-densify forward pass uses, with a per-query rescale between
  blocks. `_gather_rows` is the only operation that reads KV rows, so "only the
  selected blocks are ever loaded" is enforced in one place and tested by perturbing
  unselected blocks and checking nothing moves. This is the differentiable path, so
  it is also what the toy model trains through.
* `metal_moba_attention` (`moba/kernel.py`) is an `mx.fast.metal_kernel` for the
  same forward pass: one thread per query row, one sequential walk over each
  selected block, the same running max/sum/accumulator in registers. It is the toy
  counterpart of the paper's forward kernel.

`moba_attention(..., impl=...)` dispatches across all of them.

**Key convolution** (`moba/kconv.py`). Appendix C's short depthwise causal
convolution on keys: left-pad by `kernel_size - 1`, one kernel per (tap, channel),
SiLU, residual. The paper gives kernel sizes 3 and 5 and does not give the
initialisation or the residual form, so the tap weights start at near-identity
(tap 0 = 1, the rest 0) and the layer starts at `x + SiLU(x)`.

**Model** (`moba/model.py`). A pre-norm decoder-only transformer: RMSNorm, a GELU
MLP, rotary positions on queries and keys, and one MoBA attention layer per block.
RoPE is not incidental — an earlier version used learned absolute positional
embeddings and could not express "attend to the previous token" at all, because the
score between two positions would be `p_t^T W p_{t-1}`, which no single bilinear
form makes uniformly large; the recall task needs that shift head, and training
stalled at the trivial solution until positions became rotary.

**Task, metrics, training, plots** (`moba/task.py`, `moba/metrics.py`,
`moba/train.py`, `moba/plots.py`, `moba/snr.py`). The task is described below.
`metrics.py` counts loaded blocks, KV bytes and the logical query-block x key-block
tile schedule. `snr.py` implements the paper's Eq. 1-3 algebra and simulates the
statistical model they describe.

## The router has no gradient, and had to be given one

This is the most important design note in the repo. Block selection is discrete:
`topk_blocks` produces integer indices, the tiled path gathers with them, and the
indices are `stop_gradient`ed because MLX (correctly) refuses to build a VJP for a
gather's indices. The consequence is that **nothing in the forward pass depends on
the router's scores**, so the router receives no gradient at all and stays at its
initialisation. A first version of this repo trained exactly that: a router stuck
at random, with recall@k at chance for every k.

The fix implemented here is `MoBAAttention.router_loss`, an auxiliary objective
active when `ModelConfig.router_distill > 0` (the experiment uses 0.1). For the
*same* q and k it takes the dense causal attention distribution, sums it into block
masses `sum_{s in block j} p(t, s)`, and cross-entropies the router's softmax over
candidate blocks against that label-free target. It costs one extra L x L softmax
per layer during training and nothing at inference. This is my choice, not the
paper's; the paper does not describe a router training objective, and its own
MoBA description mentions a "gating mechanism" whose gradient path this repo does
not implement (see the differences section).

## Checks

`pytest` runs 153 tests. The ones that matter most:

- **The headline identity.** With `k = K` MoBA is full causal attention:
  `tests/test_tiled.py::test_tiled_with_every_block_selected_equals_dense_attention`,
  the same at model level in `tests/test_model.py`, and again through the Metal
  kernel in `tests/test_kernel.py`. Measured `max |moba(k=K) - dense|` = `1.13e-06`
  over three batches on untrained weights (`results/experiment.json`,
  `dense_equivalence`).
- **`k = 1` is exactly the argmax block.**
  `test_moba_reference_with_k_one_attends_to_exactly_one_block` rebuilds the answer
  by hand from the single argmax block and compares, and
  `test_moba_reference_with_k_above_the_block_count_equals_dense_attention` covers
  `k > K`, where the extra slots are padded and ignored.
- **The fast path equals the reference.** `tiled` and `compiled` are checked against
  the mask reference across `k` in {1,2,3,4} and block sizes {4,8,16} at 1e-5;
  `tests/test_kernel.py` does the same for the Metal kernel plus non-power-of-two
  head dimensions and float16. At experiment scale the worst observed difference is
  `8.6e-06` (`results/experiment.json`, `equivalence`).
- **Selection is causal and deterministic.** No query ever selects a block that
  contains only future keys, the query's own block is only scored on its own causal
  prefix, `topk` repeats identically across calls even under ties, and changing a
  key after position `t` leaves every selection at or before `t` unchanged.
- **Only selected blocks are read.** Perturbing the values of an unselected block
  leaves the tiled output and the Metal output bit-identical, and the gradients with
  respect to unselected keys and values are exactly zero.
- **Traffic tracks `k / K`.** Loaded blocks are counted per query, and the test
  asserts both the monotone trend and that once a query has `k` candidates behind it
  the fraction is *exactly* `k / K`.
- **The SNR model.** `tests/test_snr.py` checks that the simulated SNR matches
  `Delta_mu_eff * sqrt(d / 2B)` within 0.15 across `B` in {4,...,64}, that
  `Delta_mu_eff` is independent of `B`, that the SNR is linear in `sqrt(d/B)`, and
  that recall@1 improves as blocks shrink.
- **Gradients and shapes.** Finite gradients through the whole model and through the
  tiled path; a zero gradient for unselected blocks; dtype and shape contracts
  including a float16 Metal path.

## Experiment

`scripts/run_experiment.py`, 577 seconds wall clock on this machine, 2 seeds per
configuration trained concurrently. Everything below is from
`results/run_log.txt` and `results/experiment.json`.

**Task.** `L = 128` tokens, `B = 16`, `K = 8` blocks, vocabulary 96. `n_pairs = 2`
blocks each start with a `(key, value)` binding so both live in one block; every
other position is filler or a random noise key. The last two positions are copies
of the two binding keys, and at each of those positions the target is the bound
value. Because the binding keys are excluded from the noise, each query key appears
exactly twice and the task is unambiguous — a first version let distractors reuse
binding keys, which made it ambiguous rather than hard, and it is worth saying that
the failure looked like a training problem for a long time. Signal dilution is the
paper's `Delta_mu_eff / B`: the matching key is one key among 16 in its block's
centroid.

**Setup.** 4000 steps, batch 16, AdamW at lr 1e-3, weight decay 0.01, answer-position
cross-entropy, `router_distill = 0.1`, `kconv_size = 5`, 160,192 parameters
(`d_model = 64`, 4 heads of 16, 3 layers, `d_ff = 256`). Selection ran at `k` in
{1,2,4,8}; `k = 8 = K` *is* the dense causal baseline.

| config | loaded blocks / dense | loaded blocks / all | blocks per query | held-out loss | answer accuracy | router recall@k (layer 0) | train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| k = 1 | 0.222 | 0.125 | 1.000 | 0.8927 +- 0.0580 | 0.4707 +- 0.0059 | 0.163 | 96 |
| k = 2 | 0.417 | 0.234 | 1.875 | 0.7994 +- 0.0136 | 0.5020 +- 0.0059 | 0.304 | 126 |
| k = 4 | 0.722 | 0.406 | 3.250 | 0.7731 +- 0.0072 | 0.4844 +- 0.0078 | 0.616 | 222 |
| k = 8 (= K, dense) | 1.000 | 0.562 | 4.500 | 0.7704 +- 0.0090 | 0.5000 +- 0.0156 | 1.000 | 282 |

Chance for the recall column is `k / K`: 0.125, 0.25, 0.5 and 1.0. Answer accuracy
chance is `1 / n_pairs = 0.5`, which is also the ceiling of the shortcut discussed
below. `loaded blocks / all` uses `K` as the denominator and is the quantity that
tracks `k / K`; `loaded blocks / dense` uses the causal triangle, which is why
`k = K` is 1.000 there and 0.562 here. The "train s" column is the wall clock for
one configuration, and two seeds of each ran concurrently on four workers.

**Training-free sweep.** Re-running the router of one dense-trained model at every
`k`, with no retraining:

| k | held-out loss | answer accuracy | router recall@k | loaded / dense |
|---:|---:|---:|---:|---:|
| 1 | 0.9037 | 0.4766 | 0.245 | 0.222 |
| 2 | 0.8655 | 0.4805 | 0.380 | 0.417 |
| 4 | 0.8468 | 0.5430 | 0.670 | 0.722 |
| 8 | 0.8424 | 0.5391 | 1.000 | 1.000 |

**Wall clock, one forward pass on 8 x 128 tokens, median of 10 after 2 warm-ups.**

| k | dense | mask reference | tiled (MLX loop) | compiled tiled | Metal kernel |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.258 ms | 0.404 ms | 0.551 ms | 1.163 ms | 0.253 ms |
| 2 | 0.361 ms | 0.446 ms | 0.723 ms | 1.565 ms | 0.203 ms |
| 4 | 0.239 ms | 0.400 ms | 1.118 ms | 1.279 ms | 0.214 ms |
| 8 | 0.250 ms | 0.406 ms | 1.995 ms | 2.608 ms | 0.203 ms |

End-to-end tiled forward for the whole 3-layer model: 2.57 ms at `k = 1`, 3.22 at
`k = 2`, 4.42 at `k = 4`, 7.25 at `k = 8`, against 1.28 ms for dense causal.

**SNR simulation** (paper's Section 3 model, `d = 16`, 16 blocks, 4000 trials per
row). `Delta_mu_eff` is taken from the measured mean of `D` times `B`, so the
prediction is tested on its scaling rather than on a fitted constant:

| B | empirical SNR | `Delta_mu_eff sqrt(d/2B)` | recall@1 |
|---:|---:|---:|---:|
| 4 | 1.251 | 1.136 | 0.453 |
| 8 | 0.833 | 0.800 | 0.294 |
| 16 | 0.601 | 0.586 | 0.213 |
| 32 | 0.428 | 0.414 | 0.154 |
| 64 | 0.289 | 0.289 | 0.117 |

Figures in `results/`: `tile_schedule.png` (the logical query-block x key-block grid
for `k` = 1, 2, 4 and dense, coloured by the share of each query block's rows that
use the tile, with the live-tile fraction in each title — 78% of the causal tiles
are live at `k = 1` against 100% at `k = 2`), `selection_matrix.png` (which query
positions picked which key blocks), `accuracy_vs_k.png`, `traffic_vs_k.png`,
`snr_scaling.png` and `snr_recall.png`.

What the numbers mean:

- **The traffic claim holds.** Loaded blocks track `k / K` closely: 0.125, 0.234 and
  0.406 of all blocks against `k / K` of 0.125, 0.25 and 0.5, and the shortfall at
  `k = 4` is the causal boundary (early queries have fewer than `k` candidates).
  The KV-byte fractions are identical by construction, so they are not an
  independent measurement.
- **Smaller `k` costs quality, and `k = 1` costs the most.** Loss rises from 0.7704
  at `k = K` to 0.8927 at `k = 1`, outside the seed spread, and the router's recall
  falls to 0.163 against a 0.125 chance rate. `k = 2` and `k = 4` are inside about
  0.03 of dense.
- **The accuracy column is not informative and I am not dressing it up.** All four
  configurations sit at 0.47-0.50, and 0.50 is exactly `1 / n_pairs`: with only two
  values ever present in a sequence, a model that narrows to "one of the values I
  have seen somewhere" reaches 0.5 without doing any retrieval. The loss and the
  router recall separate the configurations; the accuracy does not. I confirmed this
  by adding value noise to the filler (so that every value type appears in every
  sequence) and re-training: accuracy collapsed to chance, which is what a model
  that was never retrieving should do. The published run uses
  `noise_value_rate = 0.0` precisely because it is the version that learns anything
  at all; that is a weakness of the experiment, not a finding.
- **The router is weak.** Even with the distillation objective it only beats chance:
  recall@1 is 0.163 against 0.125, recall@2 is 0.304 against 0.25, recall@4 is 0.616
  against 0.5. Above chance, consistently, and far from the paper's story of a
  router that reliably finds the signal block.
- **There is no speedup here and the MLX loop is slower than dense.** The tiled
  path does fewer floating point operations but pays for `k` separate gathers and
  the online-softmax bookkeeping, and at 128 tokens the dense matmul is already one
  kernel. The Metal kernel is the only path that keeps up with dense (0.20-0.25 ms
  against 0.24-0.36 ms) and even that is a wash, not the paper's 14.7x. These are
  wall-clock numbers from this laptop on a toy model, not a speedup claim.
- **The design principle reproduces.** The simulated SNR matches the paper's Eq. 3
  to within 0.02 once `Delta_mu_eff` is taken from the data, the measured
  `Delta_mu_eff` is constant across `B`, and recall@1 falls from 0.453 at `B = 4` to
  0.117 at `B = 64`. Smaller blocks do route better, exactly as Section 3.3 says.

## Where this may differ from the paper

- **The router's training objective is mine.** The paper describes a router that is
  trained, and mentions a "gating mechanism" in its Section 2 review of MoBA, but
  does not give a routing loss. With a hard top-k and no gate there is no gradient
  path to the router at all, so I added the block-mass distillation term described
  above, weighted 0.1. In a paired probe at `k = 4` it raised layer-0 recall from
  about 0.51 to 0.62, but that probe is not part of the committed run and the number
  is not in `results/`; what is committed is the trained-with-distillation result in
  the table above. It is not the paper's method, and the paper's own gate may be why
  its router does better.
- **No gate.** MoBA's gating is not implemented: the core attention is a single
  softmax over the union of the selected blocks. That is what the paper's Eq. in
  Section 2 says, and it is what makes `k = K` reduce *exactly* to full causal
  attention rather than approximately.
- **The causality reading.** Candidates are `j <= t // B`, the current block is
  scored on its causal prefix mean, and the current block is *not* automatically
  added to the selection. The paper says "each query always attends to its current
  block with causal masking", which could instead mean the current block is always
  in the set and the top-k is over strictly earlier blocks; under that reading
  `k = 1` would attend to two blocks and the `k = 1` argmax test would not hold.
- **The Metal kernel is forward only and is one kernel, not three.** There is no
  fused tiled top-k (the router runs in MLX), no varlen index reformatting, no
  `Br x Bc` physical tiling of Q into SRAM, no grouped-query attention, and **no
  backward pass**: MLX custom Metal kernels are not autodifferentiable, so training
  goes through `moba/tiled.py` instead and the kernel is pinned to it by test rather
  than by sharing code. The paper's Section 4.2 gather-and-densify forward pass and
  its recompute-based backward pass are only partly represented.
- **No key convolution ablation.** The paper's second design principle is validated
  against a `kconv3`/`kconv5` baseline; here the key convolution is implemented,
  unit tested and enabled in the published run (`kconv_size = 5`), but I did not
  train matched runs with and without it, so nothing here measures its benefit. The
  paper's Appendix C also does not specify the initialisation, the bias, the
  activation or the residual form; those are my choices.
- **The SNR result is a simulation of the paper's model, not a measurement of a
  trained router.** It draws the keys the paper's assumptions describe and checks
  the algebra. The empirical SNR on this repo's trained router is not reported
  because the router is too weak for it to mean anything.
- **Scale, task and metrics.** 128 tokens against the paper's 8K-512K, 8 blocks
  against thousands, 160k parameters against 340M/1B, 4000 steps against 100B
  tokens, a synthetic recall task with two bindings against WikiText perplexity,
  RULER S-NIAH, OBQA, PIQA, HellaSwag, Lambada, ARC, TriviaQA and Winogrande. `k = 2`
  and `k = 4` correspond to no configuration the paper reports (it uses `B` in
  {512,256,128} with `k` in {2,4,8} at constant 7/8 sparsity), and the toy model
  selects a much larger fraction of blocks than 7/8 because `K` is small.
- **The tile-schedule figure is a schedule of the MLX loop, not a trace of the
  Metal kernel.** The Metal kernel does not iterate physical tiles; it walks one
  query row per thread. The figure describes the logical grid the paper's Algorithm
  1 iterates, computed from the selection, and is labelled as such.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/run_experiment.py \
    --steps 4000 --seeds 2 --workers 4 --ks 1 2 4 8 --router-distill 0.1
```

The first command takes about 20 seconds. The second takes about 10 minutes on this
machine and rewrites `results/experiment.json`, `results/run_log.txt` and the
figures. Useful flags: `--steps`, `--seeds`, `--workers`, `--ks`, `--block-size`,
`--seq-len`, `--n-pairs`, `--n-queries`, `--router-distill`, `--kconv-size`,
`--retrain` (ignore the cached routing model in `artifacts/`). A `--steps 200` run
finishes in under a minute and is useful for checking the plumbing.
