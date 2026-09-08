# CASO: Content-Aware Surrogate Optimizer

[Project overview](../../README.md) · [PromoSim](../promosim/README.md) · [Data contracts](../../data/README.md) · [Baselines](../../baselines/README.md)

CASO learns to estimate the total acceptance of a movie–seed intervention, then uses that estimate to search for initial promoters. Its inputs are seed placement, static user–movie content match, and the social graph. Its final policy contains exactly the requested number of distinct users.

This guide covers the public feature, training, search, and prediction commands. Run commands from the **repository root**. For an example requiring no collected trajectories or existing model, start with `mosaic demo --output-dir artifacts/demo`.

## The three operations

| Operation | Question it answers | Output |
| --- | --- | --- |
| `mosaic train` | How does acceptance depend on seeds, content, and graph structure? | Fitted CASO checkpoint |
| `mosaic optimize` | Which users should be selected for this movie and budget? | Exact-budget seed policies |
| `mosaic predict` | How many users are expected to accept this movie for an existing seed set? | Predicted acceptance per policy |

`predict` uses the fitted surrogate and makes no LLM calls. A value such as `280.5` estimates an average total number of accepting users under the modeled environment, including the seed labels. It is not a per-user probability or an observed replay count. [PromoSim evaluation](../promosim/README.md#compare-policies-in-one-paired-run) supplies the actual simulation outcomes.

## Model and training design

1. **Static content match.** A frozen BGE-M3 encoder embeds the user profile and movie description. Their cosine similarity gives one content feature per user.
2. **Seed flow.** A budget-conditioned rectified flow learns a continuous representation of binary seed masks. Synthetic masks with budgets 0–10 supply its training examples without behavioral labels.
3. **Acceptance predictor.** Local GIN message passing and global self-attention process seed/content features with graph positional encodings. Mean/max pooling and a scalar readout predict total acceptance in `[0, N]` for `N` users.
4. **Policy search.** Latent optimization produces an exact-budget candidate after top-*k* recovery. CASO also constructs a surrogate-greedy candidate, refines both through cardinality-preserving swaps, and selects the higher-scoring set.

Stage two freezes the seed flow and trains only the acceptance predictor against replay-averaged totals. The flow provides the continuous seed representation; PromoSim outcomes provide the behavioral supervision.

## Requirements and inputs

Install the [core package](../../README.md#quick-start). Add the `encoder` extra for content generation:

```bash
python -m pip install -c constraints-py312.txt -e '.[encoder]'
```

| Command | Required inputs | Device |
| --- | --- | --- |
| `content-match` | User and movie CSVs; BGE-M3 weights | CUDA |
| `train-flow` | Node count; synthetic masks generated internally | CUDA by default; CPU supported |
| `train` | Replay conditions, graph, static content catalog; optional flow checkpoint | CUDA by default; CPU supported |
| `optimize` | Fitted CASO checkpoint, graph, content catalog, movies, budgets | CUDA |
| `predict` | Fitted CASO checkpoint, graph, content catalog, policy CSV | CUDA by default; CPU supported |

The graph, replay columns, feature columns, and checkpoint node coordinates must share the same user ordering. A checkpoint for 1,000 users is not a drop-in model for a different graph or population. The source release contains input CSVs; training trajectories, content embeddings, and fitted models must be generated separately.

## Prepare content features

```bash
mosaic content-match \
  --users data/promosim/user_1000.csv \
  --items data/promosim/item.csv \
  --movie Heat Jumanji Waiting_to_Exhale GoldenEye id:6 Nixon \
  --output artifacts/caso/content_match.npz
```

Include every movie used in training, validation, or inference. Features read static CSV profiles and movie metadata; they use no acceptance labels. Pass `--encoder /path/to/bge-m3` or set `MOSAIC_TEXT_ENCODER` to use a local snapshot.

The result is a versioned NPZ and a JSON provenance file. Unique titles retain their familiar keys. Ambiguous names require an explicit movie ID, such as `id:6` for `Sabrina__id_6`. See the [content catalog schema](../../data/README.md#static-content-catalog).

## Train a model

### Run both stages

```bash
mosaic train \
  --records data/promosim_records \
  --relationship data/promosim/relationship_1000.csv \
  --content-match artifacts/caso/content_match.npz \
  --output-dir artifacts/caso/train01 --seed 2026
```

Without `--flow-checkpoint`, training creates a new flow checkpoint under the run's `flow/` directory, then fits the acceptance predictor. The default public loader expects complete 30-round trajectories, including the initial row. Collect enough distinct conditions for nonempty training and validation sets.

### Reuse a separately trained flow

```bash
mosaic train-flow --num-nodes 1000 --updates 40000 --batch-size 256 \
  --output-dir artifacts/caso/flow01 --seed 2026

mosaic train \
  --flow-checkpoint artifacts/caso/flow01/flow_pretrained.pt \
  --records data/promosim_records \
  --relationship data/promosim/relationship_1000.csv \
  --content-match artifacts/caso/content_match.npz \
  --output-dir artifacts/caso/train02 --seed 2026
```

| Stage | Default settings |
| --- | --- |
| Flow | 40,000 updates; batch size 256; shared coordinate network of width 64; budget conditioning; rank coupling |
| Flow decoding | Dequantization width 0.1; EMA 0.995; 64-step RK4 |
| Predictor | Hidden width 64; two layers; four attention heads; eight positional dimensions |
| Predictor fitting | Up to 200 epochs; batch size 8; learning rate `3e-4`; patience 40; validation fraction 0.1 |

`mosaic train --help` and `mosaic train-flow --help` list the supported overrides. Use fresh output directories to preserve previous runs.

### Splits and checkpoint semantics

Eligible conditions are split within each movie. [DataSplit](../config.py) excludes the configured OOD panel and the *Gamma Rays* new-item target from predictor training and validation, including qualified variants of held-out titles. Validation total-acceptance MAE selects the predictor checkpoint.

The optional `train --checkpoint` argument reuses a CASO model's architecture and saved condition split. The predictor is still initialized randomly; this argument does not resume predictor weights. `--flow-checkpoint` specifically reuses the pretrained flow. Standalone `train-flow --resume` starts from saved flow weights; it does not restore a full optimizer/scheduler/RNG training session.

| Training artifact | Contents |
| --- | --- |
| `best.pt` | CASO architecture, model state, and run metadata |
| `condition_split.csv` | Movie, condition, budget, replay count, and train/validation assignment |
| `training_history.csv` | Epoch-level fitting history |
| `validation_predictions.csv` | Actual and predicted validation acceptance totals |
| `report.json` | Selected metrics, configuration, split keys, input hashes, and timing |
| `flow/flow_pretrained.pt` | Stage-one weights when trained within this run |

Standalone flow runs also save `best_so_far.pt`, `history.csv`, and `report.json`. The flow archive format is `mosaic-seed-flow-v1`; the full CASO checkpoint format is `mosaic-caso-v1`. Their roles are different, so supply each to its corresponding argument.

## Optimize seed sets

```bash
mosaic optimize \
  --checkpoint artifacts/caso/train01/best.pt \
  --relationship data/promosim/relationship_1000.csv \
  --content-match artifacts/caso/content_match.npz \
  --movie Heat GoldenEye --budget 5 10 \
  --output-dir artifacts/caso/search01 --seed 2026
```

Budgets must be in 1–10. The default search uses eight latent restarts, up to 1,000 updates, a learning rate of `5e-4`, eight flow steps during latent search, a stability patience of 50, and up to five swap passes. These values are configurable through the search CLI and [SearchConfig](../config.py).

`caso_policies.csv` contains the final CASO rows for replay. `caso_search.csv` includes candidate methods, predicted acceptance, candidate provenance, and search times. Per-movie JSON files also record configuration and input hashes. Use a new search directory for each experiment so results from different configurations are not mixed or overwritten.

## Predict acceptance for supplied seeds

```bash
mosaic predict \
  --checkpoint artifacts/caso/train01/best.pt \
  --relationship data/promosim/relationship_1000.csv \
  --content-match artifacts/caso/content_match.npz \
  --policy-csv artifacts/caso/search01/caso_policies.csv \
  --output artifacts/caso/predictions.csv --device cpu
```

The input may contain CASO, baseline, or manually specified policies. The output retains `movie,movie_id,method,budget,seeds` and adds `predicted_acceptance`. `--movie`, `--budget`, and `--method` filter rows; method names containing spaces should be quoted. The default inference batch size is one. Zero-seed control rows are accepted.

Keep the output path distinct from the input CSV. Predictions support rapid screening, while a final behavioral comparison should use fresh paired PromoSim replays.

## Common input problems

| Symptom | What to check |
| --- | --- |
| No usable training split | Complete trajectories, at least two conditions per training movie, and held-out exclusions |
| Missing content entry | Encode every eligible training/validation movie and inference target |
| Ambiguous movie | Use an explicit ID; regenerate ambiguous legacy content features |
| Node-count or seed-range error | Align the graph, selected user rows, replay columns, and feature vectors |
| Unsupported checkpoint format | Supply a flow checkpoint to `--flow-checkpoint` and a CASO checkpoint to search/prediction |
| CUDA unavailable | Use `--device cpu` for small training/prediction runs; content encoding and search require CUDA |

## Implementation map

| File | Responsibility |
| --- | --- |
| [content.py](content.py), [data.py](data.py) | Static features and validated replay records |
| [models.py](models.py), [graph.py](graph.py) | Flow, predictor, graph normalization, positional features |
| [flow_training.py](flow_training.py), [train.py](train.py) | Two-stage fitting and validation |
| [search.py](search.py), [runner.py](runner.py) | Latent/discrete seed search and exports |
| [inference.py](inference.py), [checkpoints.py](checkpoints.py) | Policy scoring and versioned model I/O |

Paper-specific adaptation and ablation drivers are outside this release. The public workflow is content preparation, two-stage training, seed optimization, and prediction, connected to [PromoSim](../promosim/README.md) for behavioral evidence and evaluation.
