#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

exec uv run python -m unittest discover -s tests -p "test_*.py" -v "$@"
