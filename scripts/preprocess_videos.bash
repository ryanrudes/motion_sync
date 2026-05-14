#!/usr/bin/env bash
set -euo pipefail

# Usage: ./process_videos.sh /path/to/folder
DIR="${1:-.}"
GVHMR_PATH="${2:-.}"
GVHMR_OUTPUT_DIR="${3:-outputs}"

if [ ! -d "$DIR" ]; then
  echo "Error: '$DIR' is not a directory" >&2
  exit 1
fi

if [ ! -d "$GVHMR_PATH" ]; then
  echo "Error: '$GVHMR_PATH' is not a directory" >&2
  exit 1
fi

# Make the output directory if it doesn't exist
mkdir -p "$GVHMR_OUTPUT_DIR"

shopt -s nullglob
MOV_FILES=("$DIR"/*.MOV)

if [ ${#MOV_FILES[@]} -eq 0 ]; then
  echo "No .MOV files found in '$DIR'."
  exit 0
fi

for DEMO in "${MOV_FILES[@]}"; do
  base="${DEMO##*/}"
  base_no_ext="${base%.*}"

  echo "Processing: $DEMO"

  # Step 2: Get focal length (35mm equivalent, in mm)
  F_MM=$(exiftool -G1 -a -s "$DEMO" \
    | awk '/\[Keys\][[:space:]]+CameraFocalLength35mmEquivalent/ {print $NF}' \
    | tail -n 1)
  if [ -z "$F_MM" ]; then
    F_MM=$(exiftool -G1 -a -s "$DEMO" \
      | awk '/\[VideoKeys\][[:space:]]+FocalLengthIn35mmFormat/ {print $NF}' \
      | tail -n 1)
  fi
  echo "Focal length: ${F_MM} mm"

  # Step 3: Get frame rate
  FPS=$(exiftool -G1 -a -s "$DEMO" \
    | awk -F': ' '/VideoFrameRate/ {print $2}' \
    | tail -n 1)
  echo "Frame rate: ${FPS} fps"

  # Step 4: Convert to MP4 in full quality, no audio, preserving frame rate
  ffmpeg -y -i "$DEMO" -c:v libx264 -crf 0 -an -r "$FPS" "${DIR}/${base_no_ext}.mp4"

  # Step 5: Run GVHMR on the video
  python "$GVHMR_PATH/tools/demo/demo.py" --video "${DIR}/${base_no_ext}.mp4" -s --f_mm "$F_MM" --output_root "$GVHMR_OUTPUT_DIR"

  echo "Done: ${DIR}/${base_no_ext}.mp4"
  echo "-----------------------------"
done