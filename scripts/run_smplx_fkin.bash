#!/usr/bin/env bash
# Run SMPL-X forward kinematics on every GVHMR output under a root directory.
# Each immediate subdirectory that contains hmr4d_results.pt gets joints.npz and vertices.npz.
#
# Usage:
#   ./scripts/run_smplx_fkin.bash [gvhmr_root] [config_path]
# Example:
#   ./scripts/run_smplx_fkin.bash outputs

set -euo pipefail

GVHMR_ROOT="${1:-outputs}"
CONFIG_PATH="${2:-}"

if [ ! -d "$GVHMR_ROOT" ]; then
  echo "Error: not a directory: $GVHMR_ROOT" >&2
  exit 1
fi

shopt -s nullglob
found_any=0
for DEMO_DIR in "$GVHMR_ROOT"/*; do
  if [ ! -d "$DEMO_DIR" ]; then
    continue
  fi
  if [ ! -f "$DEMO_DIR/hmr4d_results.pt" ]; then
    echo "Skipping (no hmr4d_results.pt): $DEMO_DIR"
    continue
  fi
  found_any=1
  echo "Processing: $DEMO_DIR"
  if [ -n "$CONFIG_PATH" ]; then
    uv run motion-sync fkin run "$DEMO_DIR" --config-path "$CONFIG_PATH"
  else
    uv run motion-sync fkin run "$DEMO_DIR"
  fi
done

if [ "$found_any" -eq 0 ]; then
  echo "No subdirectories with hmr4d_results.pt under $GVHMR_ROOT" >&2
  exit 1
fi
