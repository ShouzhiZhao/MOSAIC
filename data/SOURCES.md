# Input data provenance and identity

This source release includes the three processed CSV files listed in
`manifest.json`. The hashes identify the exact included bytes, independently
of later local simulation output. They are the fixed MovieLens-based input
environment used by this project, not an upstream MovieLens distribution.

- `item.csv`: processed movie names, genres and descriptions; `id` identifies
  the catalog row. It is not documented as the original MovieLens movie ID.
- `user_1000.csv`: processed agent profiles. Internal simulator IDs are zero-based
  row positions; the CSV `id` is retained as source metadata.
- `relationship_1000.csv`: directed internal-user edges, relationship labels and
  closeness values. Do not interpret this graph as observed MovieLens friendships.

The original sampling/enrichment scripts, upstream file checksums, and full
source-to-processed ID mapping are not part of the available release. Therefore
these CSVs are versioned inputs; reconstructing them from raw upstream data is
not a supported command. No new data license is asserted by the software MIT
license. Upstream MovieLens information is at https://grouplens.org/datasets/movielens/.

## Movie identity

Use `id:<integer>` in `--movie` arguments, or include a `movie_id` column in a
policy CSV. Unique titles remain accepted. Ambiguous titles are rejected with
candidate IDs, never overwritten. In the bundled catalog, use `id:6` for the
1995 reporting selection Sabrina and `id:903` for the entry described as 1954.
Qualified directory/feature keys are `Sabrina__id_6` and `Sabrina__id_903`.
Legacy title-only Sabrina features must be regenerated; they cannot establish
which movie description was encoded. Existing archives are not renamed.
