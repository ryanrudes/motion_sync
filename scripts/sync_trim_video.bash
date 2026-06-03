#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: sync_trim_video.bash VIDEO_DIR [DEMO_NAME] [OPTIONS]

  Trim source demo videos to the synced time window in output/synced/<demo>/synced.npz
  (motion-sync sync video). Writes video_trimmed.mp4 next to each synced.npz by default.

Arguments:
  VIDEO_DIR    Directory containing per-demo source videos (see resolution below).
  DEMO_NAME    Optional: process only this demo instead of every entry under output/synced.

Options:
  --force      Re-trim even if video_trimmed.mp4 already exists.
  --no-ffmpeg  Pass --no-ffmpeg to motion-sync (OpenCV VideoWriter only).
  -h, --help   Show this message.

Source video resolution (first match wins):
  VIDEO_DIR/<demo>.{mp4,mov,MOV,MP4}
  VIDEO_DIR/<demo>/<demo>.{mp4,mov,MOV,MP4}

Environment overrides (optional):
  VIDEO_DIR          Same as positional VIDEO_DIR if the argument is omitted.
  SYNC_OUTPUT_ROOT   Default: output/synced
  SYNC_TRIM_BASENAME Default output filename: video_trimmed.mp4
EOF
}

_resolve_source_video() {
  local demo="$1"
  local root="$2"
  local ext cand
  for ext in mp4 mov MOV MP4; do
    cand="$root/${demo}.${ext}"
    if [[ -f "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
    cand="$root/${demo}/${demo}.${ext}"
    if [[ -f "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

FORCE=false
NO_FFMPEG=false
DEMO_FILTER=""
VIDEO_DIR_ARG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE=true
      shift
      ;;
    --no-ffmpeg)
      NO_FFMPEG=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ -z "$VIDEO_DIR_ARG" ]; then
        VIDEO_DIR_ARG="$1"
      elif [ -z "$DEMO_FILTER" ]; then
        DEMO_FILTER="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      shift
      ;;
  esac
done

VIDEO_DIR="${VIDEO_DIR_ARG:-${VIDEO_DIR:-}}"
if [ -z "$VIDEO_DIR" ]; then
  echo "Error: VIDEO_DIR is required (positional argument or VIDEO_DIR env var)." >&2
  usage >&2
  exit 1
fi

if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: '$VIDEO_DIR' is not a directory" >&2
  exit 1
fi

SYNC_OUTPUT_ROOT="${SYNC_OUTPUT_ROOT:-output/synced}"
SYNC_TRIM_BASENAME="${SYNC_TRIM_BASENAME:-video_trimmed.mp4}"

if [ ! -d "$SYNC_OUTPUT_ROOT" ]; then
  echo "Error: '$SYNC_OUTPUT_ROOT' is not a directory (run scripts/sync.bash first)." >&2
  exit 1
fi

EXTRA_ARGS=()
if [ "$NO_FFMPEG" = true ]; then
  EXTRA_ARGS+=(--no-ffmpeg)
fi

if [ -n "$DEMO_FILTER" ]; then
  SYNC_DIRS=("$SYNC_OUTPUT_ROOT/$DEMO_FILTER")
else
  shopt -s nullglob
  SYNC_DIRS=("$SYNC_OUTPUT_ROOT"/*)
  shopt -u nullglob
fi

if [ ${#SYNC_DIRS[@]} -eq 0 ]; then
  echo "No synced demos found under '$SYNC_OUTPUT_ROOT'." >&2
  exit 1
fi

processed=0
skipped=0
failed=0

for SYNC_DIR in "${SYNC_DIRS[@]}"; do
  if [ ! -d "$SYNC_DIR" ]; then
    continue
  fi

  DEMO_NAME="${SYNC_DIR##*/}"
  SYNCED="$SYNC_DIR/synced.npz"
  OUT_VIDEO="$SYNC_DIR/$SYNC_TRIM_BASENAME"

  if [ ! -f "$SYNCED" ]; then
    echo "Skipping '$DEMO_NAME': missing $SYNCED (run sync first)." >&2
    skipped=$((skipped + 1))
    continue
  fi

  if [ -f "$OUT_VIDEO" ] && [ "$FORCE" != true ]; then
    echo "Skipping '$DEMO_NAME': $OUT_VIDEO already exists (use --force to overwrite)."
    skipped=$((skipped + 1))
    continue
  fi

  SOURCE_VIDEO="$(_resolve_source_video "$DEMO_NAME" "$VIDEO_DIR" || true)"
  if [ -z "$SOURCE_VIDEO" ]; then
    echo "Skipping '$DEMO_NAME': no source video under '$VIDEO_DIR'." >&2
    skipped=$((skipped + 1))
    continue
  fi

  echo "Trimming '$DEMO_NAME': $SOURCE_VIDEO -> $OUT_VIDEO"
  if uv run motion-sync sync video "$SYNCED" "$SOURCE_VIDEO" "$OUT_VIDEO" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; then
    processed=$((processed + 1))
  else
    echo "Failed '$DEMO_NAME' (exit $?)" >&2
    failed=$((failed + 1))
  fi
done

echo "Done: processed=$processed skipped=$skipped failed=$failed"
if [ "$failed" -gt 0 ]; then
  exit 1
fi
