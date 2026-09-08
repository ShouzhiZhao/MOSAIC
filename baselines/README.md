# Baseline seed selection

[Project overview](../README.md) · [CASO guide](../mosaic/caso/README.md) · [Data formats](../data/README.md)

This directory implements the comparison policies used alongside CASO. Given a directed social graph and a budget, the baseline runner returns an ordered sequence of seed users and exports its prefixes for budgets 1 through *k*. The same policy CSVs can be scored by CASO or evaluated with PromoSim.

Run all commands below **from the repository root**, after [installing the core package](../README.md#quick-start).

## Start with topology policies

```bash
mosaic baselines --algo degree discount voterank clusterrank \
  --movie Heat GoldenEye --k 10 --model IC --seed 2026 \
  --results-dir artifacts/baselines/topology01
```

This runs on the CPU using the bundled relationship CSV. It needs no encoder, behavioral trajectories, API key, or neural checkpoint. Four methods, two movies, and ten budgets produce 80 rows in `baseline_policies_IC.csv`. The topology methods reuse the same seeds across movies; the movie column makes them ready for content-specific evaluation.

Supply `--algo` explicitly to choose a subset. Omitting it attempts every registered method, including those that require CUDA or external weights.

## Available methods

| Family | `--algo` names | Selection objective | Compute and prerequisites |
| --- | --- | --- | --- |
| Topology | `random`, `degree`, `discount`, `voterank`, `clusterrank` | Graph structure | CPU; no outcome labels |
| Greedy diffusion | `naive`, `celf` | IC or LT spread | CPU; Monte Carlo simulation |
| Reverse-reachable sets | `tim`, `imm`, `dssa` | IC spread | CPU; sampled RR sets |
| Learned policies | `s2v_dqn`, `touplegdd` | Learned IC objective | CUDA; reference checkpoints |
| Learned policy | `gcomb` | IC objective | CUDA; fits a pruner and Q network on the target graph |
| Supplementary surrogate | `gnn_greedy` | IC spread | CUDA; fits a GCN surrogate on the target graph |
| Static content | Separate `mosaic content-baseline` command | Content match and weighted degree | CPU once the content catalog exists |

The paper's 16 main baselines count Naive Greedy and CELF separately under IC and LT and include ContentMatch-Degree. GNN-Greedy is an additional comparator. IC-only methods are skipped when `--model LT` is selected.

### Classical diffusion policies

```bash
mosaic baselines --algo naive celf --model LT \
  --mc 1000 --workers 8 --k 10 --movie Heat GoldenEye \
  --results-dir artifacts/baselines/lt01

mosaic baselines --algo tim imm dssa --model IC \
  --workers 8 --k 10 --movie Heat GoldenEye \
  --results-dir artifacts/baselines/rr01
```

`--mc` controls Monte Carlo counts for the greedy diffusion algorithms. RR methods use their own settings: the current runner uses 10,000 sets for TIM and `epsilon=0.5` for IMM and D-SSA. `--workers` controls CPU parallelism. The parent process assigns explicit random streams to Monte Carlo and RR workers, avoiding duplicated inherited RNG state.

### ContentMatch-Degree

Generate a [content catalog](../mosaic/caso/README.md#prepare-content-features) first, then run:

```bash
mosaic content-baseline --movie Heat GoldenEye --k 10 \
  --content-match artifacts/caso/content_match.npz \
  --output artifacts/baselines/content01.csv
```

The score multiplies min–max-normalized content similarity by min–max-normalized `log(1 + weighted out-degree)`. Ties are resolved by user ID. This method needs static user–movie features but no acceptance labels or fitted CASO model.

### Reference neural policies

S2V-DQN and ToupleGDD use native PyTorch ports of the reference architectures identified in [third_party/NOTICE.md](third_party/NOTICE.md). The checkpoint tensors are loaded into those architectures; they are not included in this source release.

Obtain the reference files named in the notice and place them in a local directory:

| Reference filename | Local filename |
| --- | --- |
| `s2vdqn.ckpt` | `s2vdqn.ckpt` |
| `tripling.ckpt` | `touplegdd.ckpt` |

```bash
export MOSAIC_BASELINE_CHECKPOINT_DIR=/path/to/baseline_checkpoints

mosaic baselines --algo s2v_dqn touplegdd --k 10 --movie Heat \
  --results-dir artifacts/baselines/neural01
```

The fallback location is `baselines/checkpoints/`, which is ignored by Git and excluded from release archives. ToupleGDD initializes source/target embeddings on the graph and scores nodes in double precision. Both reference wrappers select the top users from one scoring pass. GCOMB instead performs target-graph fitting during the run; its cost is part of policy construction.

## Graph and movie inputs

`--data-path` selects a relationship CSV. The runner reads `user_1`, `user_2`, and `closeness` into a directed NetworkX graph; `relationship` labels are used by PromoSim and are not needed here.

The edge weight is `min(closeness * scale, 1.0)`, with `--scale 0.1` by default. Supply finite, nonnegative closeness values. IC and LT use these weights as their diffusion parameters. Since the graph is constructed from edges, users absent from every edge are not included by this baseline loader. Custom graph IDs must still align with the user positions used by content features and PromoSim.

`--movie` associates policies with entries in the configured item catalog. Unique titles such as `Heat` work directly; use `id:6` or `Sabrina__id_6` for an ambiguous title. Graph-only methods do not change their seed sets with the movie. ContentMatch-Degree does.

## Outputs and timing

A topology run creates:

```text
artifacts/baselines/topology01/
├── all_seeds_k1-10_IC.csv
├── baseline_policies_IC.csv
├── Degree/k_10_IC.json
├── Degree Discount/k_10_IC.json
├── VoteRank/k_10_IC.json
└── ClusterRank/k_10_IC.json
```

| Output | Contents |
| --- | --- |
| `all_seeds_k1-<k>_<model>.csv` | `algorithm,k,seeds,time_seconds`; one row per prefix |
| `baseline_policies_<model>.csv` | `movie,movie_id,method,budget,seeds`; written when `--movie` is supplied |
| `<method>/k_<k>_<model>.json` | Full ordered seed list, selection model, and construction time |
| Content baseline CSV | Shared policy columns plus `time_seconds` |

Prefix rows reuse the construction time for the full *k*-seed sequence. They are not independent timing measurements for each smaller budget. Timings include any fitting performed by the method during that run.

Use a new results directory for each configuration. Reusing one updates the selected methods in the graph-only CSV, overwrites their JSON files, and rewrites the canonical policy CSV with only the methods in the current invocation.

## Predict or replay the selected policies

A fitted CASO checkpoint can estimate the expected acceptance of these seed sets without calling the LLM:

```bash
mosaic predict \
  --policy-csv artifacts/baselines/topology01/baseline_policies_IC.csv \
  --checkpoint artifacts/caso/train01/best.pt \
  --content-match artifacts/caso/content_match.npz \
  --output artifacts/baselines/topology01/predictions.csv
```

For observed acceptance inside the simulator, install/configure [PromoSim](../mosaic/promosim/README.md) and run:

```bash
mosaic evaluate \
  --policy-csv artifacts/baselines/topology01/baseline_policies_IC.csv \
  --replications 10 --seed 2026 \
  --output-dir artifacts/promosim_eval/topology01
```

For a paired CASO-versus-baseline comparison, merge the policy rows into one file and evaluate them together. See the [paired evaluation guide](../mosaic/promosim/README.md#compare-policies-in-one-paired-run). IC/LT labels describe how a policy was selected; PromoSim supplies its watching-based evaluation outcome.

## Implementation map

| Path | Responsibility |
| --- | --- |
| [run_baselines.py](run_baselines.py) | Graph loading, algorithm registry, seeds, and exports |
| [run_content_baseline.py](run_content_baseline.py) | Movie-specific content baseline |
| [algorithms/](algorithms/) | Selection algorithms and neural wrappers |
| [diffusion_model/](diffusion_model/) | IC/LT Monte Carlo evaluators |
| [third_party/NOTICE.md](third_party/NOTICE.md) | Reference revisions, checkpoint names, and attribution |

For missing weights, inspect the checkpoint directory and filenames above. For CUDA errors, choose topology/classical algorithms for a CPU run or configure a GPU environment. For ambiguous movies or invalid seed IDs, follow the [shared identity contract](../data/README.md#identity-and-ordering).
