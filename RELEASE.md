# Public source release

This release contains the MOSAIC/CASO implementation, baseline implementations,
and the three processed input CSVs listed in `data/manifest.json`. It does not
distribute collected LLM trajectories, fitted models, content embeddings,
paper experiment outputs, private experiment drivers, or local regression tests.
Building a wheel or source archive also excludes those private modules.
See `NOTICE.md` and `data/SOURCES.md` for attribution and data provenance.

## Environment and installation

The reference environment is Linux with Python 3.12 and PyTorch 2.9.1+cu130.
The package declares Python >=3.10; other Python/CUDA combinations have not
been verified by this release check. `constraints-py312.txt` records the main
dependency versions used for local verification, rather than a complete lockfile.
Use a separate virtual environment: PromoSim still uses LangChain 0.0.352 and
Pydantic 1, which can conflict with applications requiring Pydantic 2.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-py312.txt -e .
mosaic --help
mosaic demo --output-dir /tmp/mosaic-demo
```

For PromoSim, install a PyTorch build suitable for your CUDA driver, then:

```bash
python -m pip install -c constraints-py312.txt -e '.[promosim]'
```

`mosaic demo` runs a small CPU workflow using synthetic inputs and newly fitted
weights. It requires neither an API key nor downloaded encoder weights. Its
outputs illustrate the interfaces; they are not paper results or useful fitted
models. Use a new output directory for each run.

PromoSim and content encoding require CUDA. Both use BGE-M3; its weights are
downloaded on first use unless `MOSAIC_TEXT_ENCODER` points to a local model
snapshot. Record the exact snapshot used when reproducing a run. LLM access is
configured with `PROMOSIM_API_KEY`, `PROMOSIM_API_BASE`, and `PROMOSIM_MODEL`
(see the packaged configuration). No hosted inference service or API credits
are included. Avatar assets belong to the legacy interactive interface and are
not distributed; command-line workflows do not require OpenCV or these images.

## Artifacts and commands

| Command | Inputs beyond installed source |
| --- | --- |
| `baselines` | Relationship CSV; the checkout includes the default graph |
| `train-flow` | No simulation archive; generates synthetic seed masks |
| `promosim` | Three CSVs, encoder weights, CUDA, LLM endpoint |
| `content-match` | User and movie CSVs, encoder weights, CUDA |
| `train` | Flow checkpoint, complete replay conditions, content catalog, graph |
| `optimize`, `predict` | Fitted CASO checkpoint, content catalog, graph |
| `evaluate` | Policy CSV plus the PromoSim requirements above |

Run from the source checkout to use its input CSVs. Wheels contain the Python
packages and packaged configuration/licenses; point `MOSAIC_PROMOSIM_DATA_DIR`
to the three CSVs if using a wheel elsewhere. `MOSAIC_SIMULATION_DATA_DIR` and
`MOSAIC_ARTIFACTS_DIR` choose generated data and model/output locations.
Every command exposes its arguments through `--help`.

## Movie identity and archive migration

Movies are identified by catalog IDs. Unique titles retain their existing keys.
Same-name movies use qualified keys, for example `Sabrina__id_6` and
`Sabrina__id_903`; pass `--movie id:6` to select an explicit ID. Policy CSVs now
include `movie_id`. Ambiguous title-only policies or old content embeddings are
rejected: regenerate the latter from the desired ID. Renaming an ambiguous
archive cannot establish which movie was originally encoded.

Content catalog version 2 includes `movie_ids` and `ambiguous_titles` alongside
`format`, `movies`, and `matches`. Unique-title version 1 catalogs remain readable.
All variants of a held-out title remain excluded from training.

## Replay integrity and recovery

Generation accepts `--seed`, `--replications`, and `--resume-condition`.
Each condition has a `condition.json` with effective redacted configuration,
movie ID, seed IDs, input hashes, round count, and replication count. Each new
replay contains `X`, `sim`, `movie_id`, `rng_seed`, and `format`; arrays are
published atomically. Attempts and file hashes are recorded in `attempts.jsonl`.
The default loader expects 31 rows (30 rounds plus the intervention), binary
watch events, finite similarities, and consistent seeds and node counts.

An interrupted generation automatically resumes the first matching incomplete
condition, or use `--resume-condition <condition-id>` to select one. Completed
replays are validated and retained; missing ones are generated. An explicit
resume with changed configuration or input hashes fails. Legacy unversioned
conditions are readable if valid but cannot be resumed. New versioned conditions
enter training only after `complete.json` is written. Generation uses a Linux
process lock to prevent concurrent writers for the same movie.

Paired policy evaluation reuses initial interests and recommender tensors within
each movie/budget/replication cell, while constructing fresh agent histories and
recommender interaction state. Initializations are saved under `initializations/`;
state digests and effective configuration are recorded in the integrity log.
Different methods in a cell share the same local random seed. This controls
initial state and local RNGs; remote LLM responses and concurrent scheduling can
still vary. Evaluation preserves existing result directories and requires a new
directory for a new run; it does not implement mid-evaluation resume.

## Configuration and maintenance

Training and search CLI defaults use the shared configuration dataclasses.
Serialized `ModelConfig` fallbacks retain legacy architecture compatibility;
new two-stage training obtains its flow architecture from the flow checkpoint.
HTTP and local LLM sampling use the configured temperature, top-p, top-k, and
thinking setting where supported; the GPT adapter uses temperature and top-p.
The default temperature 0.7 documents the formerly hard-coded effective value.
IC/LT Monte Carlo and RR-set workers receive explicit random streams derived in
the parent process, so worker partitioning does not duplicate inherited RNG state.

Local release verification uses regression checks, a CPU demo, and wheel/source
archive inspection. CI checks lint, compilation, packaging, and the installed CPU
example. Tests remain private; this release has no public regression suite
or automated paper-reproduction guarantee. Generated artifacts and checkpoints
record environment/configuration or input hashes as appropriate to their stage.
