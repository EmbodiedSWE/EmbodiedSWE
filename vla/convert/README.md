# convert — raw episodes → training datasets

One canonical intermediate, many conventions as pure projections of it. Every raw
source — a sim episode today, a real-robot log tomorrow — is read into the same
`Episode` (episode.py); a convention (conventions.py) projects it to (state, action)
at a control rate; convert.py streams the result into a LeRobotDataset that both
π/openpi and GR00T consume. Sim-real co-training data goes through identical code by
construction; a new source costs one reader, a new label costs one pure function.

    episode.py       the canonical Episode + the sim reader
    conventions.py   label conventions (projections)
    filters.py       opt-in tick filters (idle removal)
    convert.py       the bake CLI

## Bake

    ~/Documents/Research/lerobot/.venv/bin/python vla/convert/convert.py \
        <…/data_gen/<gen_name>> --repo-id cosigen/bulb_franka_osc \
        [--control_space joint_vel] [--control_freq 15]  (default space: joint_target) [--batches …] [--cams front wrist] \
        [--root <out>] [--task "…"] [--include-failures] \
        [--workers auto] [--max-video-file-seconds 800]

Views come from each episode's `render_<view>.json` (`--cams` narrows); successful
episodes only by default; the dataset lands at `<gen_root>/datasets/<repo_id>` —
datasets stay with the campaign that produced them. Videos are decoded sequentially,
an episode never sits in RAM.

**Parallelism.** A bake costs ~2 min/episode (decode → PNG staging → h264 encode).
`--workers N` (or `auto` = `SLURM_CPUS_PER_TASK` / `os.cpu_count()`) splits the episodes
across N worker processes, each baking a temporary shard (`<root>.shards/wNN`), then
merges them with lerobot's `aggregate_datasets` into `--root`, applies the video-duration
cap (caveat 7) to the merged packing, verifies it, and deletes the shards. One command,
one dataset, whether you have 1 CPU or 48: `sbatch -c 32 … convert.py … --workers auto`.
Sequential mode (`--workers 1`, the default) uses lerobot's async image writer.
Multi-node scale-out = run the sharded bake per node and merge the same way
(`hpc/bulb_ik/merge_shards.py` is the template).

## The two decisions

Everything else is derived; a bake is fully specified by:

**`--control_freq`** — the label control frequency. Omitted = the native control rate (one tick
per latch; precisely, the video fps — equal to the solver rate for matched-regime
renders). A lower rate must divide the fps exactly and be an integer (60 → 30, 20,
15, 12, 10).

**`--control_space`** — what the action column means (default: `joint_target`). State is
always the same (`[arm q, gripper]` at the tick whose image the policy sees):

| convention  | action at tick t                       | semantics |
|-------------|----------------------------------------|-----------|
| `joint_pos` | achieved `q` at t+1/rate + gripper     | absolute tracking target; GR00T / LeRobot school; the sim-real workhorse — any joint PD executes it |
| `joint_vel` | `(q_next − q)·rate` + gripper          | π₀.₅-DROID exactly (15 Hz). NOT the measured `joint_vel`: the finite difference — "the average velocity that reaches the next pose", the form a tracker integrates |
| `joint_target` | the COMMANDED joint targets in force at t + commanded gripper closedness (UNCLAMPED: >1 = squeeze) | THE DEFAULT — controller intent: presses/squeezes survive as sustained target offsets that achieved labels flatten. Position-mode campaigns only (joint/diff_ik/pink_ik; the recorder omits the channel under a torque-mode arm, and this bake then refuses loudly — use joint_pos/joint_vel/raw_cmd for osc campaigns) |
| `raw_cmd`   | the recorded controller command, verbatim | LIBERO/MimicGen school; the sim-only matched-controller benchmark arm — deploy through the SAME controller |

At a fixed rate these are the same information (v = Δq·rate), so conventions are
dialects, not different data. Whatever the convention, sim bakes also carry the
verbatim command as a `raw_command` column: contact intent (a press = a sustained
offset that achieved kinematics hide) survives every projection, and later
relabelings (e.g. commanded joint targets via IK) never need the raw files again.

**Gripper** is one scalar CLOSEDNESS in [0, 1] (0 = fully open, 1 = fully closed),
normalized by each finger joint's travel (`episode.FINGER_TRAVEL`). It is achieved,
not commanded — squeeze force lives only in `raw_command`.

## Dataset schema

Keys are the superset-friendly ones: `observation.images.<view>` (video, native
render resolution — training transforms resize), `observation.state`,
`action`, `raw_command` (sim only), `task` (the scene's Goal sentence, `--task`
overrides). This satisfies the strict consumer natively and the flexible one
trivially:

- **GR00T** hardcodes these key names and needs `meta/modality.json` (written by the
  bake: named parts as start/end slices of the concatenated vectors, video key map).
- **π/openpi** is key-agnostic: each fine-tune declares a ~10-line repack transform
  mapping any keys to the model's inputs. Its other artifact, `norm_stats.json`,
  is generated by openpi's `compute_norm_stats` at training setup — derived, not
  authored; the dataset needs nothing extra.

The bake also writes `meta/bake.json` — THE provenance stamp: convention, rate,
part maps, gripper convention, the recorded control law, source episodes + git
shas. The stamp is the single source of truth an eval bridge reads to build the
matching executor (`raw_cmd` → the same OSC env.step; `joint_pos`/`joint_vel` → one
joint-PD tracking wrapper at the stamped rate), so train/eval convention drift is
structurally impossible.

## Video encoding

Dataset videos are written by lerobot's encoder; the bake exposes its knobs and picks
defaults for *training data that many frameworks will read*, which differ from lerobot's
own storage default (`libsvtav1`, `crf 30`, `g=2`):

| flag | default | why |
|---|---|---|
| `--vcodec` | `h264` | decodes everywhere (torchcodec, pyav, decord, every NVDEC generation); AV1 needs dav1d / Ampere+ |
| `--crf` | `23` | x264's standard quality point (18 ≈ visually lossless, ~35 % more bytes) |
| `--gop-seconds` | `0.25` | keyframe interval in **seconds** (`g = round(fps·s)`, min 1): worst-case seek cost is constant across control rates — g=15 at 60 Hz, g=4 at 15 Hz |
| `--pix-fmt` | `yuv420p` | the universally decodable layout |

Measured (torchcodec, 1 thread, 640×480 @ 60 fps, one 175 s episode, front camera):

| encode | size | random-access decode | sequential decode |
|---|---|---|---|
| lerobot default `libsvtav1 crf30 g=2` | ~90 MB | 330 fps | 850 fps |
| `h264 crf23 g=2` | 75 MB | 355 fps | 1020 fps |
| **`h264 crf23 g=15` (this default)** | **17 MB** | 264 fps | 2200 fps |
| `h264 crf18` long GOP (render masters) | 11 MB | 47 fps | 1790 fps |

Random access is governed by the GOP, not the codec: the render masters' long GOP is what
makes them slow to sample, and lerobot's g=2 buys the last 20 % of seek speed with 4–5× the
bytes. Policies read short clips (a few observation frames + an action chunk), where the
GOP-15 h264 is the fastest of all. The chosen encoder is stamped into `meta/bake.json`
(`video_encoder`) and lerobot's own `meta/info.json`. Pass `--vcodec libsvtav1 --gop-seconds 0.034`
to reproduce lerobot's default exactly.

Decoder note: in a conda env torchcodec needs the env's FFmpeg on the loader path
(`export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`) or lerobot silently falls back
to pyav — same frames, lower throughput.

## The control law travels with the data

`raw_cmd` numbers only mean anything under the controller that interpreted them
(pos/rot scale, re-anchoring, kp/kd, gripper drive stiffness). Generation stamps
that law as a `controller` block in each episode's `meta.json` (since 2026-08-18);
the bake copies it into `bake.json` and REFUSES pools whose blocks differ —
mixed presets or sim+real teleop must be separate bakes, staying partitionable.
Earlier batches lack the block (`controller: null`); recovering their law means
reading the batch's own solve.py, because solves override preset defaults live
(the bulb campaign sets rot_scale 0.15 over the preset's 0.097 — the exact
silent drift stamping exists to catch; kp 150/600, pos_scale 0.02 and gripper
drive stiffness 8000 came from setup writes too).

## Extending

- **New gripper**: one `FINGER_TRAVEL` entry (joint name → travel). Unknown finger
  joints are a hard error — a guessed travel silently corrupts every gripper label.
  Fingers are found by name marker ("finger"/"gripper"); an oddly named vendor
  joint means switching that lookup to an explicit per-robot registry entry.
- **New arm**: free — joints, dims and names flow from the render contract.
- **New source** (real robot): one reader returning `Episode`; nothing downstream
  changes.
- **New convention**: one pure function `Episode → Projected` in conventions.py
  (dexterous hands would enter here: per-finger parts instead of the closedness
  scalar — the named-parts machinery already supports it).

## Caveats

1. **Contact flattening**: during a press, achieved q barely moves — `joint_pos`
   says "stay", `joint_vel` says ~0, while the intent is *push*. `raw_command` is
   the recovery path; a joint-convention policy stalling exactly at contact phases
   is this. Same for gripper squeeze force.
2. **Idle/settle phases** are kept by default. `--filter-idle` (opt-in) drops the
   DEAD ticks only: robot static AND every object static (pose delta + recorded
   velocities ~0) AND no command intent. The intent guard (|EE offset| >= 0.1)
   keeps static-but-pushing press ticks; active settling has nonzero velocities
   and is kept; the first tick of each idle run stays as the arrive-and-settle
   cue. ~25% of the bulb episode is dead ticks. The classic BC freeze failure
   mode is the reason to turn it on; survivorship of presses is why it's safe.
3. The **last tick's action is "stay"** by construction (clamped lookahead).
4. **Norm stats**: near-constant dims explode after quantile normalization (openpi
   warns about this) — inspect `norm_stats.json` before the first training run.
5. **GR00T container version**: our lerobot writes the v3 layout; GR00T docs
   describe v2. Verify (or down-convert) before the first GR00T run.
6. **Success-only** by default — survivor bias, correct for BC
   (`--include-failures` exists).
7. **Video file duration vs LeRobot's timestamp check.** LeRobot stores frame
   timestamps as float32 and verifies each decoded frame within `tolerance_s=1e-4`.
   Past 1024 s into a video file the float32 step is 1.2e-4 s > the tolerance, so
   two roundings of the same instant fail the check and training dies mid-run
   (`FrameTimestampError`, ~50 min in; a short smoke never draws such a frame).
   LeRobot only caps files by MB (default 200 MB ≈ 30 min at 15 fps / 640×480 h264),
   so `--max-video-file-seconds` (default 800) measures the bitrate on the first
   episode, derives the MB cap from it, and an ffprobe pass fails the bake if any
   file still exceeds 1000 s. The merge step (`aggregate_datasets`) repacks and
   must be given the same MB cap (see the hpc merge script). Consumers should still
   pass `--tolerance_s=0.005` (13× finer than the 66.7 ms frame period) as a belt.
