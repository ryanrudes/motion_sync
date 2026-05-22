# Two-foot retargeting

Tools to ingest Vicon mocap and GVHMR video motion, align them in time, and write a single **`unified.npz`** per demo. Optional helpers trim source video to the synced window and visualize alignment.

---

## What’s in the repo

| Component | Role |
|-----------|------|
| `retargeting convert` | ROS 2 bags → `merged.npz` |
| `retargeting fkin` | GVHMR `hmr4d_results.pt` → `joints.npy` / `vertices.npy` (required before sync) |
| `retargeting sync` | Cross-correlate foot speeds, build `unified.npz`, trim video, debug viewer |
| `retargeting model` | Optional rigid-body fits from marker clouds → `output/rigid_models/` |
| `configs/retargeting.yaml` | Video/mocap rates, sync solver, marker names |

Batch drivers under `scripts/` wrap the same CLI for every demo under `output/`.

---

## Requirements

- Python **3.12**, [uv](https://docs.astral.sh/uv/)
- `ffmpeg` — video trim / transcode
- `exiftool` — focal length and FPS from `.MOV` (preprocess script only)
- [GVHMR](https://github.com/zju3dv/GVHMR) — run separately; outputs go under `output/gvhmr/<demo>/`
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) neutral weights at `data/smplx_models/smplx/SMPLX_NEUTRAL.npz` (or `.pkl`)
- Vicon ROS 2 bags in `data/bags/<demo>/` (large; gitignored)

---

## Install

```bash
cd twofoot_retargeting
uv sync
```

---

## Data layout

```text
data/
  bags/<demo>/                 # Vicon ROS 2 bags
  videos/<demo>.{mp4,mov}      # source video (gitignored)
  smplx_models/smplx/          # SMPL-X weights (gitignored)
  msg/*.msg                    # ROS types for bag conversion

output/
  vicon_tables/<demo>/merged.npz
  gvhmr/<demo>/
    hmr4d_results.pt
    joints.npy                 # from retargeting fkin
    vertices.npy
  synced/<demo>/
    unified.npz
    video_trimmed.mp4          # optional
  rigid_models/                # optional, from retargeting model
```

Use the **same demo name** in each tree (e.g. `pushoff8_twoshoes`).

---

## Pipeline

### 1. Vicon: bags → tables

```bash
./scripts/convert_bags.bash data/bags output/vicon_tables

# or one demo:
uv run retargeting convert bag data/bags/<demo> output/vicon_tables/<demo>
```

### 2. Video: GVHMR (+ optional batch preprocess)

Transcode and run GVHMR on all `.MOV` files in a folder:

```bash
chmod +x scripts/preprocess_videos.bash
./scripts/preprocess_videos.bash data/videos /path/to/GVHMR output/gvhmr
```

Ensure each demo directory under `output/gvhmr/<demo>/` contains `hmr4d_results.pt`.

### 3. SMPL-X forward kinematics

Sync loads `joints.npy` and `vertices.npy` from each GVHMR folder:

```bash
./scripts/run_smplx_fkin.bash output/gvhmr

# or one demo:
uv run retargeting fkin run output/gvhmr/<demo>
```

### 4. Time sync → `unified.npz`

One demo:

```bash
uv run retargeting sync time \
  output/vicon_tables/<demo> \
  output/gvhmr/<demo> \
  -o output/synced/<demo>
```

All demos with matching Vicon + GVHMR:

```bash
./scripts/sync.bash
./scripts/sync.bash --plot    # foot-speed overlay per demo (blocks)
```

Common flags:

| Flag | Meaning |
|------|---------|
| `--crop support` | Default: keep overlap where all sources have support |
| `--crop valid` | Stricter finiteness (often shorter clips) |
| `--crop none` | Full timeline, NaNs outside overlap |
| `--target-timeline video` | One row per video frame (default: `vicon`) |
| `--plot` / `--plot-file` | Foot-speed alignment figure |

**Lag convention:** `t_vicon_unified = t_vicon - lag`. Plots use mocap at **`t_mocap - lag`** on the video-clock axis.

Tune sync in `configs/retargeting.yaml` (`time_sync_solver`: `min_correlation`, `max_abs_lag_seconds`, `motion_weighted_sync`).

### 5. Trim video (optional)

```bash
./scripts/sync_trim_video.bash data/videos
./scripts/sync_trim_video.bash data/videos <demo> --force

# or:
uv run retargeting sync video \
  output/synced/<demo>/unified.npz \
  data/videos/<demo>.mp4 \
  output/synced/<demo>/video_trimmed.mp4
```

Trims to `t[0]` … `t[-1]` from `unified.npz` (video-clock seconds).

### 6. Visualize sync (optional)

```bash
uv run retargeting sync visualize \
  output/synced/<demo>/unified.npz \
  data/videos/<demo>.mp4
```

Side-by-side video and OptiTrack markers (`q` quit, space pause).

### 7. Optional: rigid-body marker models

```bash
./scripts/model_rigid_bodies.bash
# writes output/rigid_models/<demo>/*.pkl — not used by sync
```

---

## `unified.npz` contents

- **`t`** — timeline in **video-clock** seconds  
- **`video__*`** — resampled GVHMR / SMPL streams (`joints`, `vertices`, `body_pose`, `transl`, `global_orient`, `betas`, …)  
- **`vicon__*`** — resampled Vicon (`body_pos`, `body_quat`, `marker_pos`, …)  
- **`lag`**, **`corr`** — sync metadata  

**Axes:** GVHMR joints in `video__*` are **Y-up**; Vicon `vicon__body_pos` is **Z-up**.

**Quaternions:** raw `merged.npz` uses wxyz from Vicon `/tf`. In `unified.npz`, `vicon__body_quat` is **xyzw** `[qx, qy, qz, qw]` (do not read as wxyz).

---

## CLI reference

```bash
uv run retargeting --help
```

| Group | Command | Description |
|-------|---------|-------------|
| `convert` | `bag <bag_dir> <out_dir>` | ROS 2 bag → CSV/NPZ + `merged.npz` |
| `fkin` | `run <gvhmr_dir>` | Write `joints.npy`, `vertices.npy` |
| `sync` | `time <vicon_tables> <gvhmr> -o <dir>` | Build `unified.npz` |
| `sync` | `video <unified.npz> <src> <out>` | Trim video to sync window |
| `sync` | `visualize <unified.npz> <src>` | Debug player |
| `model` | `bodies <demo_tables> <out_dir>` | Optional rigid-body fits |

`retargeting detect` is registered but not implemented yet.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_bags.bash` | All bags → `output/vicon_tables` |
| `scripts/run_smplx_fkin.bash` | FK for each `output/gvhmr/*` with `hmr4d_results.pt` |
| `scripts/sync.bash` | Batch `sync time` |
| `scripts/sync_trim_video.bash` | Batch `sync video` |
| `scripts/preprocess_videos.bash` | MOV → MP4 + GVHMR |
| `scripts/model_rigid_bodies.bash` | Batch `model bodies` |

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Wrong sync lag | `--plot`; adjust `min_correlation` / `max_abs_lag_seconds` in YAML |
| Very short `unified.npz` with `--crop valid` | Use `--crop support` or `none` |
| Missing `joints.npy` | Run `./scripts/run_smplx_fkin.bash output/gvhmr` before sync |
| `Skipping … no hmr4d_results.pt` | Finish GVHMR for that demo name |
| Quaternion bugs downstream | Use xyzw for `vicon__body_quat` in `unified.npz` |

---

## Development

```bash
uv sync --extra dev
./scripts/test.bash
# or: uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Optional local git hooks (runs the same unittest suite on every commit):

```bash
pre-commit install
pre-commit run --all-files
```

Contributor notes: `AGENTS.md`.
