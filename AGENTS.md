## Learned User Preferences

- When adding repo bash drivers (`*.bash`), prefer iterating all demos under standard output roots (`output/vicon_tables`, `output/gvhmr`, `output/synced`) with optional flags (for example `--plot`), not only single-demo paths, unless the task is explicitly one-off.
- For GVHMR batch scripts, loop each immediate subdirectory of the GVHMR root that contains the expected artifact (for example `hmr4d_results.pt`), skipping directories that lack it with a clear log line.
- Prefer explicit warnings or hard failures over silent continuation when data do not meet minimum constraints (for example too few samples for a robust fit).
- Prefer a single shared definition for duplicated types and configuration-driven behavior over parallel copies; generalize modeling code from hard-coded body lists to the subjects and rigid bodies that actually appear in the data.

## Learned Workspace Facts

- Primary CLI entry is `uv run retargeting …`; common on-disk layout uses `output/vicon_tables/<demo>/merged.npz`, `output/gvhmr/<demo>/`, and `output/synced/<demo>/unified.npz`.
- Time sync lag convention is `t_vicon_unified = t_vicon - lag`; foot-speed overlays must plot mocap at `t_mocap - lag` on the video-clock axis so they match `build_unified_dataset` and `retargeting sync time --plot`, not `+ lag`.
- Time sync quality is configured under `time_sync_solver` in `configs/retargeting.yaml` (correlation floor, max absolute lag window, motion-weighted scoring); false peaks from aligning long low-motion overlaps are a known failure mode on several two-shoes demos.
- With `crop="valid"`, the stitcher intersects finiteness masks only for required schema features; optional partial streams (for example markers) do not gate cropping.
- `resolve_vicon_schema_for_sync` can limit `vicon/body_pos` finiteness checks for `crop="valid"` to shoe rigid bodies so gaps in other trackers do not wipe the timeline; `build_unified_dataset` defaults `crop` to `support` for more reliable overlap length versus strict all-channel finiteness on every row.
