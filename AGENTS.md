## Learned User Preferences

- When adding repo bash drivers (`*.bash`), prefer iterating all demos under standard output roots (`output/vicon_tables`, `output/gvhmr`, `output/synced`) with optional flags (for example `--plot`), not only single-demo paths, unless the task is explicitly one-off.
- For GVHMR batch scripts, loop each immediate subdirectory of the GVHMR root that contains the expected artifact (`hmr4d_results.pt`), skipping directories that lack it with a clear log line.
- Prefer explicit warnings or hard failures over silent continuation when data do not meet minimum constraints (for example too few samples for a robust fit, or zero rows after sync crop).
- Prefer a single shared definition for duplicated types and configuration-driven behavior over parallel copies; generalize modeling code from hard-coded body lists to the subjects and rigid bodies that actually appear in the data.
- For per-demo reruns, chain the existing CLI (`sync time`, `sync video`, `sync visualize`) or the batch scripts under `scripts/`; avoid one-off hard-coded demo names in new drivers unless the task is explicitly a single-demo experiment.
- For sync quality checks, use `retargeting sync time --plot` or `scripts/sync.bash --plot` before trusting `unified.npz`; inspect lag and correlation printed to the terminal.
- For hard or mysterious sync bugs (wrong lag, empty crop, misaligned feet in `sync visualize`), trace step-by-step with quantitative evidence (lag/corr, overlap lengths, foot-speed curves at `t_mocap - lag`), not ad-hoc lag tweaks.
- Bash drivers must resolve `_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` at the top before any `cd`; paths like `"$_SCRIPT_DIR/other_script.bash"` break if computed after `cd` changes cwd.

## Learned Workspace Facts

- **Scope:** This repo ingests Vicon + GVHMR, runs SMPL-X FK, time-syncs streams, and writes `unified.npz`.
- Primary CLI: `uv run retargeting …`. Groups: `convert`, `fkin`, `sync`, `model` (optional). `detect` is registered but not implemented.
- Common on-disk layout: `output/vicon_tables/<demo>/merged.npz`, `output/gvhmr/<demo>/` (`hmr4d_results.pt`, `joints.npy`, `vertices.npy`), `output/synced/<demo>/unified.npz`, source videos at `data/videos/<demo>.mp4` (or `.mov`). Demo names must match across trees.
- **FK before sync:** `load_gvhmr_data` requires `joints.npy` and `vertices.npy` in each GVHMR folder (`retargeting fkin run` or `scripts/run_smplx_fkin.bash`).
- Batch scripts: `convert_bags.bash`, `run_smplx_fkin.bash`, `sync.bash`, `sync_trim_video.bash`, `preprocess_videos.bash`, `model_rigid_bodies.bash` (optional rigid-body fits → `output/rigid_models/`).
- Time sync lag convention: `t_vicon_unified = t_vicon - lag`. Foot-speed overlays must plot mocap at `t_mocap - lag` on the video-clock axis (`retargeting sync time --plot`, `build_unified_dataset`), not `+ lag`.
- Time sync quality is configured under `time_sync_solver` in `configs/retargeting.yaml` (correlation floor, max absolute lag window, motion-weighted scoring). False peaks from aligning long low-motion overlaps are a known failure mode on several two-shoes demos.
- `build_unified_dataset` / CLI default crop is `support` (overlap where all sources have support). `crop=valid` intersects finiteness on required channels (shoe bodies only for `vicon/body_pos` via `resolve_vicon_schema_for_sync`); often yields shorter clips.
- `merged.npz` `body_quat` is wxyz from Vicon `/tf`. After sync, `unified.npz` `vicon__body_quat` is scipy xyzw `[qx, qy, qz, qw]` (SLERP in `syncer.py`). Downstream code that unpacks as `w, x, y, z = q[0..3]` is wrong for unified files.
- Axis conventions in `unified.npz`: GVHMR/SMPL FK (`video__joints`, `video__transl`, …) are **Y-up**; Vicon `vicon__body_pos` is **Z-up**. Consumers that need a single world frame must transform explicitly (no in-repo exporter).
- `unified.npz` keys use double underscores for slashes (`video__joints`, `vicon__body_pos`). Metadata arrays: `lag`, optional `corr`, timeline `t` in video-clock seconds.
- CLI sync helpers: `sync time` (build `unified.npz`), `sync video` (trim to `t[0]…t[-1]`; ffmpeg preferred), `sync visualize` (debug player). Batch trim: `scripts/sync_trim_video.bash data/videos [demo]` → `output/synced/<demo>/video_trimmed.mp4`.
- Config marker/body definitions for shoes and skateboard live in `configs/retargeting.yaml` under `bodies` and `time_sync_solver.smplx_joints` (foot-speed sync uses SMPL-X joint names mapped to Vicon shoe rigid bodies).
- **Tests:** `./scripts/test.bash` or `uv run python -m unittest discover -s tests -p "test_*.py" -v`. CI and pre-commit run the same command (synthetic fixtures only; no `output/` required).
