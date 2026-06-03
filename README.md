# motion-sync

Ingest Vicon mocap and GVHMR video motion, align them in time, and write a single **`synced.npz`** per demo. Downstream code loads a typed **`SyncClip`** (bodies, markers, video joints, contact layers)—not raw NPZ keys. Optional helpers trim source video to the synced window and visualize alignment.

**Full documentation** (ecosystem overview, custom schemas, API reference, retarget wiring) lives in the [retarget](https://github.com/ryanrudes/retarget) docs site under **Ecosystem** and **API → motion_sync**. Clone retarget with `git clone --recurse-submodules` for the docs site; clone this repo beside retarget for pipeline and editable installs.

---

## What’s in the repo

| Component | Role |
|-----------|------|
| `motion-sync convert` | ROS 2 bags → `vicon.npz` (Vicon-only, mocap clock) |
| `motion-sync fkin` | GVHMR `hmr4d_results.pt` → `joints.npy` / `vertices.npy` (required before sync) |
| `motion-sync sync` | Cross-correlate foot speeds, build `synced.npz`, trim video, debug viewer |
| `motion-sync model` | Optional rigid-body fits from marker clouds → `output/rigid_models/` |
| `motion-sync detect` | Foot support (and derived contacts) on `synced.npz` |
| `configs/motion_sync.yaml` | Video/mocap rates, sync solver, marker names |

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
cd motion_sync
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
  vicon_tables/<demo>/vicon.npz
  gvhmr/<demo>/
    hmr4d_results.pt
    joints.npy                 # from motion_sync fkin
    vertices.npy
  synced/<demo>/
    synced.npz
    video_trimmed.mp4          # optional
  rigid_models/                # optional, from motion_sync model
```

Use the **same demo name** in each tree (e.g. `pushoff8_twoshoes`).

---

## Pipeline

### 1. Vicon: bags → tables

```bash
./scripts/convert_bags.bash data/bags output/vicon_tables

# or one demo:
uv run motion-sync convert bag data/bags/<demo> output/vicon_tables/<demo>
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
uv run motion-sync fkin run output/gvhmr/<demo>
```

### 4. Time sync → `synced.npz`

One demo:

```bash
uv run motion-sync sync time \
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

**Lag convention:** `t_vicon_synced = t_vicon - lag`. Plots use mocap at **`t_mocap - lag`** on the video-clock axis.

Tune sync in `configs/motion_sync.yaml` (`time_sync_solver`: `min_correlation`, `max_abs_lag_seconds`, `motion_weighted_sync`).

### 5. Trim video (optional)

```bash
./scripts/sync_trim_video.bash data/videos
./scripts/sync_trim_video.bash data/videos <demo> --force

# or:
uv run motion-sync sync video \
  output/synced/<demo>/synced.npz \
  data/videos/<demo>.mp4 \
  output/synced/<demo>/video_trimmed.mp4
```

Trims to `t[0]` … `t[-1]` from `synced.npz` (video-clock seconds).

### 6. Visualize sync (optional)

```bash
uv run motion-sync sync visualize \
  output/synced/<demo>/synced.npz \
  data/videos/<demo>.mp4
```

Side-by-side video and OptiTrack markers (`q` quit, space pause).

### 7. Optional: rigid-body marker models

```bash
./scripts/model_rigid_bodies.bash
# writes output/rigid_models/<demo>/*.pkl — not used by sync
```

---

## Loading a clip in Python

Synced clips are persisted internally as compressed NPZ; application code should use **`SyncClip`** only (not ``vicon__*`` array keys). For skate trials, register everything in one step:

```python
from motion_sync import SyncClip
from motion_sync.contacts.foot_support import FootSupportState
from motion_sync.schemas.skateboarding import (
    Bodies,
    SKATE_FOOT_SUPPORT,
    SKATE_SESSION,
    SmplxCoreJoints,
)

clip = SyncClip.load("output/synced/pushoff5_twoshoes", session=SKATE_SESSION)

if not clip.contact_is_fresh(SKATE_FOOT_SUPPORT):
    clip = clip.detect(SKATE_FOOT_SUPPORT).save("output/synced/pushoff5_twoshoes")

foot = clip.contact(SKATE_FOOT_SUPPORT)
left = foot.track(Bodies.LEFT_SHOE)
on_board = left.states == FootSupportState.SKATEBOARD

left_foot = clip.joint(SmplxCoreJoints.L_FOOT).positions  # Y-up SMPL-X FK
core = clip.core_joint_positions()  # (frames, 20, 3) retarget smplx order
```

Vicon poses are **Z-up** with body quaternions **xyzw**. Video/SMPL streams are **Y-up**. Use `clip.export_vicon_bodies()` when an algorithm expects `(t, body_names, body_pos)` arrays.

---

## Clip contents (via `SyncClip`)

- **`clip.time_s`** — timeline in **video-clock** seconds  
- **`clip.vicon`** — resampled Vicon bodies and markers (**Z-up**, body quaternions **xyzw**)  
- **`clip.video`** — resampled GVHMR / SMPL streams (**Y-up**)  
- **`clip.metadata`** — `lag_s`, optional `correlation`  
- optional **`clip.valid`** mask  

---

## CLI reference

```bash
uv run motion-sync --help
```

| Group | Command | Description |
|-------|---------|-------------|
| `convert` | `bag <bag_dir> <out_dir>` | ROS 2 bag → CSV/NPZ + `vicon.npz` |
| `fkin` | `run <gvhmr_dir>` | Write `joints.npy`, `vertices.npy` |
| `sync` | `time <vicon_tables> <gvhmr> -o <dir>` | Build `synced.npz` |
| `sync` | `video <synced.npz> <src> <out>` | Trim video to sync window |
| `sync` | `visualize <synced.npz> <src>` | Debug player |
| `model` | `bodies <demo_tables> <out_dir>` | Optional rigid-body fits |
| `detect` | `foot-support <demo> [--force] [--plot]` | Classify foot support on `synced.npz` |

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
| Very short `synced.npz` with `--crop valid` | Use `--crop support` or `none` |
| Missing `joints.npy` | Run `./scripts/run_smplx_fkin.bash output/gvhmr` before sync |
| `Skipping … no hmr4d_results.pt` | Finish GVHMR for that demo name |
| Quaternion bugs downstream | Use xyzw for `vicon__body_quat` in `synced.npz` |

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
