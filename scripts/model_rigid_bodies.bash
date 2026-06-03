#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: model_rigid_bodies.bash [OPTIONS]

  Batch-run rigid body modeling for each demo under output/vicon_tables,
  writing Pickled models under output/rigid_models.

Options:
  --plot       Pass --plot to motion-sync (show rigid body model plots).
  -v, --verbose   Pass --verbose to motion-sync (extra console output).
  -h, --help   Show this message.
EOF
}

PLOT=false
VERBOSE=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --plot)
      PLOT=true
      shift
      ;;
    -v | --verbose)
      VERBOSE=true
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

EXTRA_ARGS=()
if [ "$PLOT" = true ]; then
  EXTRA_ARGS+=(--plot)
fi
if [ "$VERBOSE" = true ]; then
  EXTRA_ARGS+=(--verbose)
fi

VICON_TABLES_DIR="output/vicon_tables"
OUTPUT_DIR="output/rigid_models"

if [ ! -d "$VICON_TABLES_DIR" ]; then
  echo "Error: '$VICON_TABLES_DIR' is not a directory" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Error: '$OUTPUT_DIR' is not a directory" >&2
  exit 1
fi

for DEMO in "$VICON_TABLES_DIR"/*; do
  if [ ! -d "$DEMO" ]; then
    echo "Error: '$DEMO' is not a directory" >&2
    exit 1
  fi

  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    uv run motion-sync model bodies "$DEMO" "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
  else
    uv run motion-sync model bodies "$DEMO" "$OUTPUT_DIR"
  fi
done