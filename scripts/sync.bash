#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: sync.bash [OPTIONS]

  Run time sync (retargeting sync time) for every demo under output/vicon_tables
  that has a matching GVHMR directory under output/gvhmr.

  Each demo writes unified.npz to output/synced/<demo_name>/ by default.

Options:
  --plot       Pass --plot to retargeting (matplotlib foot-speed alignment; blocks per demo).
  -h, --help   Show this message.

Environment overrides (optional):
  VICON_TABLES_DIR   Default: output/vicon_tables
  GVHMR_DIR          Default: output/gvhmr
  SYNC_OUTPUT_ROOT   Default: output/synced
EOF
}

PLOT=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --plot)
      PLOT=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

VICON_TABLES_DIR="${VICON_TABLES_DIR:-output/vicon_tables}"
GVHMR_DIR="${GVHMR_DIR:-output/gvhmr}"
SYNC_OUTPUT_ROOT="${SYNC_OUTPUT_ROOT:-output/synced}"

if [ ! -d "$VICON_TABLES_DIR" ]; then
  echo "Error: '$VICON_TABLES_DIR' is not a directory" >&2
  exit 1
fi

if [ ! -d "$GVHMR_DIR" ]; then
  echo "Error: '$GVHMR_DIR' is not a directory" >&2
  exit 1
fi

mkdir -p "$SYNC_OUTPUT_ROOT"

EXTRA_ARGS=()
if [ "$PLOT" = true ]; then
  EXTRA_ARGS+=(--plot)
fi

shopt -s nullglob
DEMOS=("$VICON_TABLES_DIR"/*)
shopt -u nullglob

if [ ${#DEMOS[@]} -eq 0 ]; then
  echo "No demos found under '$VICON_TABLES_DIR'." >&2
  exit 1
fi

for DEMO_VICON in "${DEMOS[@]}"; do
  if [ ! -d "$DEMO_VICON" ]; then
    continue
  fi

  DEMO_NAME="${DEMO_VICON##*/}"
  DEMO_GVHMR="$GVHMR_DIR/$DEMO_NAME"
  OUT_DIR="$SYNC_OUTPUT_ROOT/$DEMO_NAME"

  if [ ! -f "$DEMO_VICON/merged.npz" ]; then
    echo "Skipping '$DEMO_NAME': missing merged.npz under vicon tables." >&2
    continue
  fi

  if [ ! -d "$DEMO_GVHMR" ]; then
    echo "Skipping '$DEMO_NAME': no GVHMR directory at '$DEMO_GVHMR'." >&2
    continue
  fi

  mkdir -p "$OUT_DIR"
  echo "Syncing demo: $DEMO_NAME -> $OUT_DIR"

  uv run retargeting sync time "$DEMO_VICON" "$DEMO_GVHMR" -o "$OUT_DIR" "${EXTRA_ARGS[@]}"
done
