#!/usr/bin/env bash
set -euo pipefail

# Supply --target-movie-id and configure the serving endpoint through env vars.
exec python -m mosaic promosim "$@"
