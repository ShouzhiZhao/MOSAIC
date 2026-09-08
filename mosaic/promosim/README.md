# PromoSim: behavioral simulation and policy replay

[Project overview](../../README.md) · [CASO guide](../caso/README.md) · [Data contracts](../../data/README.md) · [Release notes](../../RELEASE.md)

PromoSim models content consumption through interacting social and recommendation channels. LLM-based agents use profiles, memories, and observations to choose bounded actions. A dynamic SimGCL recommender and social messages update the context available to later decisions.

Two public commands use this simulator:

| Command | Purpose | Intervention |
| --- | --- | --- |
| `mosaic promosim` | Collect behavioral trajectories for CASO training | One randomly sampled movie–seed condition per invocation |
| `mosaic evaluate` | Compare supplied seed policies through fresh replays | Seed sets from a policy CSV |

Run all examples **from the repository root**. Both commands require CUDA, encoder weights, and a configured LLM backend. The [CPU example](../../README.md#run-the-cpu-example) demonstrates the CASO interfaces without running PromoSim.

## Replay and outcome definition

![PromoSim reset, intervention, 30-round dynamics, and acceptance aggregation.](../../assets/promosim-protocol.png)

At time zero, every seed user receives an acceptance label and broadcasts the same target-movie post to their contacts. The seed label is a measurement convention; it does not populate that agent's internal watched history. All other users must choose to watch the target movie during the replay to be counted as accepting it.

The default horizon is 30 one-day rounds. Agents' recommendation and social actions update memories, watched/heard histories, content representations, and recommender interactions. A user who watches repeatedly still contributes only one terminal acceptance. Merely seeing, mentioning, or retransmitting the movie does not count.

The total acceptance of a replay is the number of users with at least one target watching event or an initial seed label. Repetition means estimate the expected total under the configured environment.

## Install and configure

Install a CUDA-compatible PyTorch build, then the PromoSim extra in a dedicated Python environment:

```bash
python -m pip install -c constraints-py312.txt -e '.[promosim]'
```

The reference dependency stack is documented in [RELEASE.md](../../RELEASE.md#environment-and-installation). PromoSim uses LangChain 0.0.352 and Pydantic 1; sharing its environment with applications requiring Pydantic 2 can cause conflicts.

### LLM service

```bash
export PROMOSIM_API_BASE="https://your-model-endpoint/v1"
export PROMOSIM_API_KEY="your-api-key"
export PROMOSIM_MODEL="your-served-model-name"
```

These environment variables override the endpoint, credential, and model name in [config/default.yaml](config/default.yaml). The endpoint must already serve the named model. The custom HTTP adapter sends `top_k` and `enable_thinking` as extra fields; use a compatible backend. The GPT adapter uses temperature and top-p. Credentials in the effective configuration are redacted in run metadata.

### Encoder and input files

BGE-M3 is downloaded on first use. To use a local snapshot:

```bash
export MOSAIC_TEXT_ENCODER=/path/to/bge-m3
```

The simulator and the content-feature workflow should use the same frozen encoder snapshot. The bundled movie, profile, and relationship CSVs are read from `data/promosim/`. For other inputs, set `MOSAIC_PROMOSIM_DATA_DIR` or pass `--data-dir` to the generation command. See the [identity and ordering rules](../../data/README.md#identity-and-ordering) before changing the user block or catalog.

The FAISS index is created automatically. Use a fresh cache path when changing the encoder or catalog. Avatar images and OpenCV belong to the legacy interactive interface and are unnecessary for these command-line workflows.

## Collect training conditions

```bash
mosaic promosim --target-movie-id 5 \
  --config mosaic/promosim/config/default.yaml \
  --replications 10 --seed 2026
```

Catalog ID `5` selects *Heat*. The generator samples a budget uniformly from 0–10, then distinct users without replacement, and rejects already-used movie–seed conditions. Every repetition reconstructs the simulator and runs the fixed intervention for the configured horizon.

One invocation collects one condition. Run generation again to collect another condition for the same movie, or change `--target-movie-id` for another target. Model training requires multiple distinct conditions; this command does not recreate the paper's full corpus in one call.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--target-movie-id` | Required | Integer ID from the item catalog |
| `--config` | Packaged `default.yaml` | Simulation settings |
| `--data-dir` | Default input directory | Override the directory for relative CSV/cache paths |
| `--output-dir` | `data/promosim_records` | Root under which movie directories are created |
| `--replications` | 10 | Complete replays requested per condition |
| `--seed` | 2026 | Condition-sampling and replay-seed base |
| `--resume-condition` | Automatic selection | Explicit condition ID to resume |

Outputs include individual replay NPZs, per-replay logs, a condition manifest, attempt records, initial descriptions, and a completion marker. The [data guide](../../data/README.md#generated-condition-layout) specifies their exact layout and fields.

### Resume interrupted generation

Rerunning the original command automatically resumes the first matching incomplete condition. To select a condition explicitly:

```bash
mosaic promosim --target-movie-id 5 \
  --config mosaic/promosim/config/default.yaml \
  --replications 10 --seed 2026 \
  --resume-condition <condition-id>
```

Replace `<condition-id>` with the directory name under `data/promosim_records/Heat/temp/`. Preserve the original configuration, input bytes, base seed, and replication count. The generator validates existing replay identity, array contents, and recorded hashes, then generates only missing replays. Array publication is atomic, and a Linux process lock prevents concurrent generators from writing conditions for the same movie output directory.

`complete.json` is written after the requested repetitions and aggregate file are complete. New versioned conditions enter CASO training only then. Legacy directories without `condition.json` cannot be resumed through this command.

If resume reports an input or checksum mismatch, investigate that condition and use a separate output root for intentionally changed experiments. Do not rename ambiguous archives or mark incomplete data as complete to bypass validation.

## Compare policies in one paired run

Evaluation accepts the [shared policy CSV](../../data/README.md#policy-csv), including rows from CASO, graph baselines, ContentMatch-Degree, and manually supplied seed sets. Seeds must be unique and match their stated budget; zero-seed controls are supported.

After generating CASO and baseline policies, combine their rows so all methods are evaluated together:

```python
from pathlib import Path
from mosaic.policies import read_policies, write_policies

caso = read_policies(Path("artifacts/caso/search01/caso_policies.csv"))
baselines = read_policies(
    Path("artifacts/baselines/topology01/baseline_policies_IC.csv")
)
write_policies(Path("artifacts/policies/comparison.csv"), [*caso, *baselines])
```

Then run the paired comparison:

```bash
mosaic evaluate --policy-csv artifacts/policies/comparison.csv \
  --config mosaic/promosim/config/default.yaml \
  --budget 5 10 --replications 10 --seed 2026 \
  --output-dir artifacts/promosim_eval/comparison01
```

`--movie`, `--method`, and `--budget` can restrict the evaluated rows. Quote method names containing spaces, for example `--method "Degree Discount" CASO`.

Pairing operates within each **movie, budget, and replication** cell. The first method initializes interests and recommender tensors; subsequent methods reuse copies of that initialization while constructing fresh agent memory and interaction state. All methods in the cell use the same derived local RNG seed and simulated start time. The preintervention state digest must match before the seed intervention is applied.

Running methods in separate evaluation directories with the same `--seed` does not share the realized LLM-generated interests. A single combined run provides that shared initialization. Remote LLM responses and concurrent scheduling can still vary within subsequent trajectories; pairing controls the start and local random streams, not every external response.

### Evaluation artifacts

| File or directory | Contents |
| --- | --- |
| `policy_replay_outcomes.csv` | Movie, method, budget, seeds, replay index, RNG seed, acceptance, and initial-state digest |
| `policy_replay_summary.csv` | Mean, sample standard deviation, and completed count per policy |
| `policy_replay_integrity.jsonl` | Initialization/completion/failure records, digests, configuration, and snapshot hashes |
| `initializations/` | Saved interests and recommender tensors for each paired cell |
| `policy_replay_environment.json` | Environment and run summary after successful completion |

Each completed outcome is flushed to disk. An interrupted run can therefore leave partial outcomes and audit records, but the summary files are produced only after successful completion. Evaluation requires a new output directory and currently does not support continuing an interrupted policy comparison. Generation's `--resume-condition` applies only to training-data collection.

CASO's `predicted_acceptance` estimates and PromoSim's measured `acceptance` are separate outputs. Final policy comparisons use the replay outcomes.

## Configuration reference

The [default YAML](config/default.yaml) defines the complete simulator configuration. Frequently adjusted settings include:

| Setting | Default | Effect |
| --- | --- | --- |
| `agent_num`, `agent_start_offset` | `1000`, `0` | Selected profile block |
| `round`, `interval` | `30`, `1d` | Horizon and simulated time step |
| `rec_model`, `rec_train` | `SimGCL`, `true` | Dynamic recommendation behavior |
| `generate_interests`, `rate_initial_tags` | `true`, `false` | LLM interest initialization and optional diagnostic ratings |
| `temperature`, `top_p`, `top_k` | `0.7`, `0.8`, `20` | LLM sampling settings |
| `enable_thinking`, `max_token` | `false`, `8000` | Thinking mode and response-token limit |
| `execution_mode`, `parallel_workers` | `parallel`, `40` | Agent execution concurrency |
| `llm_max_concurrency`, `llm_request_timeout` | `40`, `60` | HTTP concurrency limit and timeout in seconds |
| `max_retries`, `broadcast_workers` | `2`, `16` | Request retries and social-broadcast workers |

For controlled comparisons, hold these settings fixed across methods. The CLI `--seed` controls replay random streams. Changing the LLM model, encoder, agent inputs, or sampling settings creates a different simulation environment. The public CASO training CLI expects the default 30-round trajectory shape.

## Common run problems

| Symptom | Action |
| --- | --- |
| CUDA unavailable | Configure a supported GPU/PyTorch environment; use the CPU demo for interface exploration |
| Unknown LLM model or authentication failure | Check the served model name, endpoint URL, and API credential |
| HTTP throttling or timeouts | Review service capacity, concurrency limits, timeout, and retry settings |
| Existing evaluation output | Select a new directory; evaluation does not append to or resume prior comparisons |
| Resume hash/configuration mismatch | Match the original inputs/settings or start a separate collection root |
| Paired initial-state mismatch | Inspect the integrity log and initialization; the comparison stops before intervention |
| Missing or ambiguous movie | Use the correct catalog ID and regenerate unidentified legacy features |

## Implementation map and provenance

| Path | Responsibility |
| --- | --- |
| [generation.py](generation.py) | Random conditions, interventions, trajectories, and recovery |
| [simulator.py](simulator.py) | Agent lifecycle and coupled simulation rounds |
| [policy_replay.py](policy_replay.py), [integrity.py](integrity.py) | Shared initialization, RNGs, integrity checks, and policy replay |
| [evaluate.py](evaluate.py) | Public evaluation CLI and outcome aggregation |
| [agents/](agents/) | Profiles, actions, and memory |
| [recommender/](recommender/) | Input loading and recommendation models |
| [llm/](llm/) | HTTP and local language-model adapters |

PromoSim derives from RecAgent/YuLan-Rec; attribution and its license are retained in [NOTICE.md](../../NOTICE.md) and [third_party/LICENSE.RecAgent](third_party/LICENSE.RecAgent). The source release contains the simulator and processed inputs. Collected trajectories, model weights, and paper-specific experiment orchestration remain outside it.
