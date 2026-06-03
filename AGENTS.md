## Learned User Preferences

- When adding repo bash drivers (`*.bash`), prefer iterating all demos under standard output roots (`output/vicon_tables`, `output/gvhmr`, `output/synced`) with optional flags (for example `--plot`), not only single-demo paths, unless the task is explicitly one-off.
- For GVHMR batch scripts, loop each immediate subdirectory of the GVHMR root that contains the expected artifact (`hmr4d_results.pt`), skipping directories that lack it with a clear log line.
- Prefer explicit warnings or hard failures over silent continuation when data do not meet minimum constraints (for example too few samples for a robust fit, or zero rows after sync crop).
- Prefer a single shared definition for duplicated types and configuration-driven behavior over parallel copies; generalize modeling code from hard-coded body lists to the subjects and rigid bodies that actually appear in the data.
- For per-demo reruns, chain the existing CLI (`sync time`, `sync video`, `sync visualize`) or the batch scripts under `scripts/`; avoid one-off hard-coded demo names in new drivers unless the task is explicitly a single-demo experiment.
- For sync quality checks, use `motion-sync sync time --plot` or `scripts/sync.bash --plot` before trusting `synced.npz`; inspect lag and correlation printed to the terminal.
- For hard or mysterious sync bugs (wrong lag, empty crop, misaligned feet in `sync visualize`), trace step-by-step with quantitative evidence (lag/corr, overlap lengths, foot-speed curves at `t_mocap - lag`), not ad-hoc lag tweaks.
- Bash drivers must resolve `_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` at the top before any `cd`; paths like `"$_SCRIPT_DIR/other_script.bash"` break if computed after `cd` changes cwd.
- Bodies use one `StrEnum`; each body has its own marker `StrEnum` in `MocapSchema.body_markers` so logical names (e.g. `HEEL`) can repeat across feet. Values remain unique Vicon strings.
- Do not keep legacy `merged.npz` / `unified.npz` read paths after renames to `vicon.npz` / `synced.npz`; regenerate pipeline outputs instead of compatibility shims.

## Learned Workspace Facts

- **Scope:** Ingest Vicon + GVHMR, run SMPL-X FK, time-sync streams, and write `synced.npz` per demo under `output/synced/<demo>/`.
- Primary CLI: `uv run motion-sync …` (`convert`, `fkin`, `sync`, `detect`, optional `model`).
- On-disk layout: `output/vicon_tables/<demo>/vicon.npz` (Vicon-only, mocap clock), `output/gvhmr/<demo>/` (`hmr4d_results.pt`, `joints.npy`, `vertices.npy`), `output/synced/<demo>/synced.npz`, videos at `data/videos/<demo>.mp4` (or `.mov`). Demo names must match across trees.
- **FK before sync:** each GVHMR folder needs `joints.npy` and `vertices.npy` (`motion-sync fkin run` or `scripts/run_smplx_fkin.bash`).
- Batch scripts: `convert_bags.bash`, `run_smplx_fkin.bash`, `sync.bash`, `sync_trim_video.bash`, `preprocess_videos.bash`, `model_rigid_bodies.bash` (optional rigid-body fits → `output/rigid_models/`).
- Time sync: `t_vicon_synced = t_vicon - lag`; foot-speed overlays plot mocap at `t_mocap - lag` on the video-clock axis. Quality knobs live under `time_sync_solver` in `configs/motion_sync.yaml` (correlation floor, max lag, motion-weighted scoring).
- Default sync crop is `support`; `crop=valid` intersects finiteness on required channels and is often shorter. `vicon.npz` `body_quat` is wxyz; synced `vicon__body_quat` is scipy xyzw. GVHMR/SMPL channels are Y-up; Vicon `vicon__body_pos` is Z-up—transform explicitly for a single world frame.
- Downstream Python uses `SyncClip.load(path)` / `clip.save(path)` and `ViconRecording.load(path)`; NPZ key layout is internal (`motion_sync._storage`). `path` may be a demo directory or `synced.npz` file; marker names attach from sibling Vicon mocap when present.
- Prefer `SyncClip.load(demo, session=SKATE_SESSION)` to register mocap + contacts. Per-body markers: `clip.markers_for_body(Bodies.LEFT_SHOE)`, `clip.marker(LeftShoeMarkers.HEEL)`.
- Contact: `SKATE_SESSION` on load; types `SKATE_FOOT_SUPPORT`, `SKATE_SHOE_BOARD_GRIP` in `SKATE_CONTACTS`. Bases: `CategoricalContact`, `BinaryContact`. `clip.detect(...)` skips if `clip.contact_is_fresh(...)`; stale layers warn on read. CLI: `motion-sync detect foot-support [--force]`.
- Video: `SKATE_VIDEO` / `SmplxCoreJoints` on `SKATE_SESSION`; `clip.joint(...)`, `clip.core_joint_positions()` (Y-up FK).
- Marker/body definitions for shoes and skateboard live in `configs/motion_sync.yaml` under `bodies` and `time_sync_solver.smplx_joints`.
- CLI sync helpers: `sync time` (build `synced.npz`), `sync video` (trim to `t[0]…t[-1]`), `sync visualize` (debug player). Batch trim: `scripts/sync_trim_video.bash data/videos [demo]` → `output/synced/<demo>/video_trimmed.mp4`.
- **Tests:** `./scripts/test.bash` or `uv run python -m unittest discover -s tests -p "test_*.py" -v` (synthetic fixtures only; no `output/` required).
