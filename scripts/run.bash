set -euo pipefail
TWOFOOT="${TWOFOOT:-$HOME/GitHub/twofoot_retargeting}"   # change if your clone path differs
DEMO="pushoff8_twoshoes"
HOLOS="$TWOFOOT/holosoma"
HOLOS_PY="$HOLOS/.venv/bin/python"
# setuptools / pyproject root (contains pyproject.toml, setup.py)
PKG="$HOLOS/src/holosoma_retargeting"
# package tree with models/ and examples/ (nested holosoma_retargeting/)
INNER="$PKG/holosoma_retargeting"

# 0) Time-sync → unified.npz (skip if output/synced/$DEMO/unified.npz already exists)
cd "$TWOFOOT"
uv run retargeting sync time \
  "output/vicon_tables/$DEMO" \
  "output/gvhmr/$DEMO" \
  --output "output/synced/$DEMO"

# 1) Package motion for Holosoma (writes NPZ next to holosoma package)
mkdir -p "$PKG/holosoma_retargeting_data"
uv run retargeting holosoma object-npz \
  "output/synced/$DEMO/unified.npz" \
  "$PKG/holosoma_retargeting_data/${DEMO}.npz"

# 2) Retarget (MuJoCo + interaction mesh; cwd must be INNER so models/ paths resolve)
mkdir -p "$PKG/holosoma_retargeting_results/$DEMO"
cd "$INNER"
"$HOLOS_PY" examples/robot_retarget.py \
  --data-path "$PKG/holosoma_retargeting_data" \
  --task-type object_interaction \
  --task-name "$DEMO" \
  --data-format smplh \
  --task-config.object-name skateboard \
  --save-dir "$PKG/holosoma_retargeting_results/$DEMO"

# 3) Playback in Viser (open the printed URL; Ctrl+C to quit)
"$HOLOS_PY" viser_player.py \
  --qpos-npz "$PKG/holosoma_retargeting_results/$DEMO/${DEMO}_original.npz" \
  --robot-urdf models/g1/g1_29dof.urdf \
  --object-urdf models/skateboard/skateboard.urdf \
  --assume-object-in-qpos True