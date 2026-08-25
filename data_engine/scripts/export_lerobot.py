"""Export a data_gen campaign to a LeRobot v2 dataset (pi0.5 BC schema).

Maps verified episodes — traj.npz (states/actions) + the multiply pass's rendered
frames (imgs/, imgs_draw<k>/ per visual draw) — onto vla_training.lerobot_dataset's
create_dataset()/add_step() (image, wrist_image, state, actions, task). Every
(episode, visual draw) pair becomes ONE dataset episode: visual multiplication
compounds directly into dataset size, physics stays the verified original.

Frame-step alignment comes from the render contract (imgs*/render_<view>.json:
frame_indices[i] = the trajectory step of frame i); state/action rows are sampled
at those steps. The base view is the first non-ego camera rendered; the wrist view
is the first ego (link-mounted) camera — episodes missing either are skipped with
a warning (rendering owns visual truth; the exporter never re-renders).

Needs openpi's pinned lerobot (run under the openpi venv with HF_LEROBOT_HOME
set); generation pods don't carry it, so this runs post-hoc wherever that env
exists:

    python export_lerobot.py <gen_root> --repo-id simgen/dgen13_v7 --task "thread the nut …"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vla_training.lerobot_dataset import add_step, create_dataset  # noqa: E402


def frame_files(view_dir: Path) -> list[Path]:
    return sorted(view_dir.glob("frame_*.jpg"))


def load_view(imgs: Path, ego: bool) -> tuple[str, dict] | None:
    """The first rendered view of the requested kind (ego = wrist) in this imgs dir."""
    for meta_p in sorted(imgs.glob("render_*.json")):
        meta = json.loads(meta_p.read_text())
        if bool(meta.get("link")) == ego and (imgs / meta["camera"]).is_dir():
            return meta["camera"], meta
    return None


def resize(img: np.ndarray, hw: int = 256) -> np.ndarray:
    import cv2

    return cv2.resize(img, (hw, hw), interpolation=cv2.INTER_AREA)


def select_vector(
    value: np.ndarray, *, expected: int, indices: tuple[int, ...] | None, label: str
) -> np.ndarray:
    flat = value.astype(np.float32).ravel()
    if indices is None:
        if len(flat) != expected:
            raise ValueError(
                f"{label} has {len(flat)} values but the export schema requires "
                f"{expected}; provide an explicit --{label}-indices mapping"
            )
        return flat
    if len(indices) != expected:
        raise ValueError(
            f"--{label}-indices contains {len(indices)} entries; expected {expected}"
        )
    if min(indices) < 0 or max(indices) >= len(flat):
        raise ValueError(
            f"--{label}-indices {indices} is out of bounds for {len(flat)} values"
        )
    return flat[list(indices)]


def export_episode(
    ds,
    ep_dir: Path,
    imgs: Path,
    task: str,
    state_indices: tuple[int, ...] | None,
    action_indices: tuple[int, ...] | None,
) -> bool:
    import imageio.v2 as imageio

    base = load_view(imgs, ego=False)
    wrist = load_view(imgs, ego=True)
    if base is None or wrist is None:
        print(f"[export] {ep_dir.name}/{imgs.name}: missing "
              f"{'base' if base is None else 'wrist'} view — skipped")
        return False
    with np.load(ep_dir / "traj.npz") as d:
        action = d["action"]
        jp_key = next((k for k in d.files if k.endswith("joint_pos")), None)
        if jp_key is None:
            print(f"[export] {ep_dir.name}: no joint_pos in traj.npz — skipped")
            return False
        joint_pos = d[jp_key]
    steps = base[1]["frame_indices"]
    base_frames = frame_files(imgs / base[0])
    wrist_frames = frame_files(imgs / wrist[0])
    lengths = {
        "frame_indices": len(steps),
        f"{base[0]} frames": len(base_frames),
        f"{wrist[0]} frames": len(wrist_frames),
    }
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"{ep_dir}/{imgs.name}: render streams are misaligned: {lengths}"
        )
    for i in range(len(steps)):
        t = int(steps[i])
        if t < 0 or t >= len(action) or t >= len(joint_pos):
            raise IndexError(
                f"{ep_dir}/{imgs.name}: frame {i} references trajectory step {t}, "
                f"but action/state lengths are {len(action)}/{len(joint_pos)}"
            )
        state = select_vector(
            joint_pos[t], expected=8, indices=state_indices, label="state"
        )
        act = select_vector(
            action[t], expected=7, indices=action_indices, label="action"
        )
        add_step(ds,
                 image=resize(imageio.imread(base_frames[i])),
                 wrist_image=resize(imageio.imread(wrist_frames[i])),
                 state=state, actions=act, task=task)
    ds.save_episode()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gen_root", type=Path)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", required=True, help="language instruction for every episode")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--include-diagnostics", action="store_true",
                    help="also export nominal/probe/agent-session test batches")
    ap.add_argument("--allow-partial-render", action="store_true",
                    help="export before the orchestrator's multiply pass is complete")
    ap.add_argument("--state-indices", type=lambda s: tuple(map(int, s.split(","))),
                    help="explicit 8-index mapping when source joint_pos is not 8-D")
    ap.add_argument("--action-indices", type=lambda s: tuple(map(int, s.split(","))),
                    help="explicit 7-index mapping when source action is not 7-D")
    args = ap.parse_args()

    multiply_done = args.gen_root / ".multiply_done"
    if not args.allow_partial_render:
        marker = json.loads(multiply_done.read_text()) if multiply_done.is_file() else {}
        if not marker.get("ok"):
            raise SystemExit(
                f"visual multiply is not complete ({multiply_done} missing or not ok); "
                "pass --allow-partial-render only for an intentional partial export"
            )

    manifest_path = args.gen_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    ds = create_dataset(args.repo_id)
    n_out = 0
    source_successes = 0
    expected_visual_episodes = 0
    for meta_p in sorted(args.gen_root.glob("data/batch_*/ep_*/meta.json")):
        batch = meta_p.parent.parent.name
        if (not args.include_diagnostics
                and not batch.startswith(("batch_farm_", "batch_compound_"))):
            continue
        meta = json.loads(meta_p.read_text())
        if not (meta.get("success") or args.include_failures):
            continue
        if meta.get("success"):
            source_successes += 1
        ep_dir = meta_p.parent
        image_dirs = sorted(p for p in ep_dir.glob("imgs*") if p.is_dir())
        expected_visual_episodes += len(image_dirs)
        for imgs in image_dirs:
            if imgs.is_dir() and export_episode(
                ds, ep_dir, imgs, args.task,
                args.state_indices, args.action_indices
            ):
                n_out += 1
    if manifest is not None and not args.include_diagnostics and not args.include_failures:
        expected_sources = int(manifest.get("successful_episodes", -1))
        if source_successes != expected_sources:
            raise RuntimeError(
                f"manifest/filesystem mismatch: manifest indexes {expected_sources} "
                f"successful source episodes, exporter found {source_successes}"
            )
    if n_out != expected_visual_episodes:
        raise RuntimeError(
            f"export incomplete: wrote {n_out} dataset episodes from "
            f"{expected_visual_episodes} rendered episode/draw directories"
        )
    print(f"[export] {n_out} dataset episodes -> {args.repo_id}")


if __name__ == "__main__":
    main()
