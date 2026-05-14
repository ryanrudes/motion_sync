#!/usr/bin/env bash
set -euo pipefail

BAGS_DIR="${1:-data/bags}"
OUTPUT_DIR="${2:-output/vicon_tables}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate through all ROS 2 bag folders in the bags directory
for BAG_DIR in "$BAGS_DIR"/*; do
  if [ -d "$BAG_DIR" ]; then
    echo "Processing: $BAG_DIR"
    uv run retargeting convert bag "$BAG_DIR" "$OUTPUT_DIR"
  fi
done