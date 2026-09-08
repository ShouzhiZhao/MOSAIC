# Data and artifact formats

[Project overview](../README.md) · [PromoSim guide](../mosaic/promosim/README.md) · [CASO guide](../mosaic/caso/README.md)

This directory contains the fixed input environment for MOSAIC and the default location for locally generated simulation records. The contracts below keep movie identity, user ordering, and behavioral outcomes consistent across generation, training, prediction, and evaluation.

All command examples assume the repository root as the working directory.

## Included inputs

| File | Rows | Role |
| --- | ---: | --- |
| [promosim/item.csv](promosim/item.csv) | 3,883 | Movie catalog, genres, and descriptions |
| [promosim/user_1000.csv](promosim/user_1000.csv) | 1,000 | Agent profiles and initial interests |
| [promosim/relationship_1000.csv](promosim/relationship_1000.csv) | 20,846 | Directed social relationships |

[SOURCES.md](SOURCES.md) documents provenance and the limits of reconstructing these processed inputs. [manifest.json](manifest.json) records their exact byte sizes, row counts, column names, and SHA-256 checksums. These files are a processed MovieLens-based environment; they are not an upstream MovieLens distribution, and the software MIT license does not assign them a new data license.

Trajectories, embedding indexes, content catalogs, interaction logs, and fitted weights are generated or obtained separately and remain outside the source release.

## CSV schemas

PromoSim's CSV readers consume fields in the order shown. Preserve the headers and column order, and quote fields containing commas.

| File | Columns, in order |
| --- | --- |
| `item.csv` | `id,title,genre,description` |
| `user_1000.csv` | `id,name,gender,age,traits,status,interest,feature,group` |
| `relationship_1000.csv` | `user_1,user_2,relationship,closeness` |

The movie catalog also accepts `item_id` as the first column's header.

Movie `genre` and `description` supply semantic context. User `traits`, `status`, and `interest` contribute to the agent profile; `feature` is interpreted by the simulator's feature-description mapping, and `group` controls initial interest-selection behavior. The `relationship` string supplies the social relationship label.

`closeness` is a numeric, finite, nonnegative edge weight. CASO uses weighted graph structure; the baseline runner converts closeness to IC/LT weights with `min(closeness * scale, 1)`, using a default scale of 0.1. The same column should not be assumed to be a calibrated probability of watching a movie.

## Identity and ordering

### Movies

Use unique, consecutive, zero-based movie IDs in ascending CSV row order for the full simulator. Movie embeddings are indexed by these IDs. The processed IDs are not documented as original MovieLens IDs.

Titles are display metadata. Commands accept a unique title, its underscore-separated key, or an explicit `id:<integer>` selector. Same-name entries require explicit identity:

| Movie selector | Catalog ID | Stored key |
| --- | ---: | --- |
| `Heat` or `id:5` | 5 | `Heat` |
| `id:6` | 6 | `Sabrina__id_6` |
| `id:903` | 903 | `Sabrina__id_903` |

Canonical keys replace spaces and slashes with underscores. A duplicate title adds `__id_<id>`. Plain `Sabrina` is ambiguous and is rejected. Policy CSVs carry `movie_id` alongside the key.

### Users and graph nodes

Internal user IDs are **zero-based row positions in the selected profile block**. The user CSV's `id` field remains source metadata. `agent_start_offset` selects the beginning of the block and `agent_num` determines its length.

The following must refer to the same internal user ordering:

- `user_1` and `user_2` in the relationship CSV;
- seed IDs in policy CSVs;
- columns of replay arrays and content-match arrays;
- node coordinates in the CASO checkpoint.

Changing the selected profile block requires a correspondingly aligned graph and feature catalog. The content CLI reads the first `--num-users` profiles from its `--users` file; it does not apply PromoSim's `agent_start_offset`. Supply a CSV containing the selected block when using a nonzero offset.

## Generated condition layout

With default paths, a completed condition looks like this:

```text
data/promosim_records/
└── Heat/
    ├── <condition-id>.npz          # Aggregate across the condition's replays
    ├── .generation.lock           # Process lock for this movie
    └── temp/
        └── <condition-id>/
            ├── condition.json     # Configuration, identity, seeds, input hashes
            ├── attempts.jsonl     # Started/completed/failed replay attempts
            ├── profiles.json      # Initial descriptions from the first completed replay
            ├── 0.npz              # One complete replay
            ├── 1.npz
            ├── ...
            ├── replay-0.log
            ├── interaction.csv    # Working recommender interactions
            └── complete.json      # Completion marker and aggregate hash
```

A **condition** is a movie plus a fixed seed set. A **replay** is one stochastic run of that condition. Replays within a condition share the seed vector; their subsequent watching behavior can differ.

### Replay NPZ: `mosaic-replay-v2`

Let `T` be the number of rounds and `N` the number of users. The default shape is `(31, 1000)`.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `format` | String scalar | `mosaic-replay-v2` |
| `movie_id` | Integer scalar | Target catalog movie |
| `rng_seed` | Integer scalar | Derived replay RNG seed |
| `X` | Boolean `(T + 1, N)` | Initial seed labels, then per-round target watching |
| `sim` | Numeric `(T + 1, N)` | Dynamic user–target cosine similarities |

Row zero records the intervention. Later rows mark users who watched the target in that round; `X` is not a cumulative array. Terminal acceptance for one replay is computed by taking `X.any(axis=0)` and counting the accepted users. This counts each user once, including the initial seeds.

After generating a condition, inspect one replay without loading an LLM:

```python
from pathlib import Path
import numpy as np

path = next(Path("data/promosim_records/Heat/temp").glob("*/*.npz"))
with np.load(path, allow_pickle=False) as replay:
    print("Movie ID:", int(replay["movie_id"]))
    print("Shape:", replay["X"].shape)
    print("Initial seeds:", np.flatnonzero(replay["X"][0]).tolist())
    print("Terminal acceptance:", int(replay["X"].any(axis=0).sum()))
```

### Condition metadata and aggregation

`condition.json` uses format `mosaic-condition-v2` and records the movie key/ID, seed IDs, round and node counts, replication count, base seed, redacted configuration, input hashes, and environment metadata. `attempts.jsonl` records replay attempts and completed-file hashes. Resume checks use these records to preserve matching completed replays and reject changed inputs or corrupted output.

The top-level `<condition-id>.npz` stacks `X` and `sim` into `(replays, T + 1, N)` arrays and includes `movie_id`, `seed_agents`, `user_descriptions`, and `item_description`. The training loader reads the individual replay files under `temp/`; the aggregate is a convenience artifact.

For training, each user's acceptance label is the mean of their terminal Boolean outcome over the condition's retained replays. Summing these empirical frequencies yields the scalar total-acceptance target.

### Admission and recovery

The public training CLI expects 30 rounds. The loader checks matching array dimensions, binary `X`, finite values, cosine similarities within numerical tolerance of `[-1, 1]`, and a common seed vector and node count. Budgets outside 0–10 are excluded by default.

New versioned conditions are admitted only after `complete.json` exists, and their replay count must match the manifest. Legacy conditions without a manifest remain readable when their complete arrays pass validation; the default minimum is one complete replay. Legacy conditions cannot be resumed by the versioned generator. This distinction supports older archives while requiring newly generated conditions to finish their requested repetitions.

See the [PromoSim recovery guide](../mosaic/promosim/README.md#resume-interrupted-generation) for commands and failure handling.

## Static content catalog

`mosaic content-match` writes a label-free archive for CASO and ContentMatch-Degree. For `M` movies and `N` users, its version 2 fields are:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `format` | String scalar | `mosaic-content-match-v2` |
| `movies` | String `(M,)` | Canonical movie keys |
| `movie_ids` | Integer `(M,)` | Corresponding catalog IDs |
| `ambiguous_titles` | String vector | Same-name titles in the source catalog |
| `matches` | Float32 `(M, N)` | Frozen user–movie embedding cosine similarities |

A JSON file beside the NPZ records the encoder source, input hashes, environment, selected movies, and construction method. These features use the static profile CSV and movie metadata. They do not use replay outcomes or later `sim` rows.

Version 1 content catalogs remain supported for unambiguous movies. Same-name legacy features must be regenerated from a known movie ID; renaming a file or adding a suffix cannot recover the original encoded identity.

## Policy CSV

All selection, prediction, and replay commands share:

```csv
movie,movie_id,method,budget,seeds
Heat,5,Example,3,"[15, 969, 826]"
Heat,5,Control,0,[]
Sabrina__id_6,6,Example,2,"[15, 826]"
```

These rows illustrate the schema. `seeds` is a JSON list of distinct nonnegative integers, with length equal to `budget`. Movie and method names must be nonempty; duplicate movie/method/budget rows are rejected. Prediction and replay accept the zero-seed control. CASO search supports budgets 1–10.

Historical `algorithm,k,seeds` files are accepted at the input boundary. Supply `--movie` when a graph-only file has no movie column. A canonical file may include extra columns such as predicted scores or timing; policy readers use the shared fields.

## Paths and local caches

| Variable | Default | Used for |
| --- | --- | --- |
| `MOSAIC_DATA_DIR` | `<workspace>/data` | Parent of the default input/record directories |
| `MOSAIC_PROMOSIM_DATA_DIR` | `<data>/promosim` | Three input CSVs and the default FAISS cache |
| `MOSAIC_SIMULATION_DATA_DIR` | `<data>/promosim_records` | Condition archives |
| `MOSAIC_ARTIFACTS_DIR` | `<workspace>/artifacts` | Default checkpoints and result locations |
| `MOSAIC_TEXT_ENCODER` | `BAAI/bge-m3` | Encoder identifier or local snapshot |

Set environment variables before launching commands. Explicit CLI paths select individual inputs or outputs. PromoSim creates its FAISS index automatically; use a fresh `index_name` when changing the catalog or encoder so a previous index is not reused with different inputs.

The original preprocessing pipeline, behavioral archive, and model weights are outside this release. The included [manifest](manifest.json) identifies the public inputs independently of any locally generated files.
