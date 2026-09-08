#!/usr/bin/env bash
set -euo pipefail

# Forward training options; --output-dir is required to identify the new run.
exec python -m mosaic train "$@"
