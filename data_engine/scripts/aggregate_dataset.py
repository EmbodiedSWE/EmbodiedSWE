#!/usr/bin/env python3
"""Aggregate finished data_gen campaigns into one release tree (for upload).

    python3 data_engine/scripts/aggregate_dataset.py <dgen_out> <release_dir> --gen gen_o50_v3 \\
        --tasks coffee,pc_gpu,...            # default: every task whose status.json says DONE

For each task the release holds EXACTLY the delivered dataset — the physics_set.json episodes
(the ladder's top rung, each replay-verified) with their traj.npz, meta.json and rendered
videos — plus what is needed to interpret them:

    <release_dir>/<task>/
      README.md                     counts, preset, controller, cameras/draws, how to read traj.npz
      gen.yaml  status.json  run.json           campaign facts, seed provenance
      scene_set.json strategy_set.json phase_set.json physics_set.json render_manifest.json
      episodes.jsonl                one row per episode: path, cell, steps, seed, draws, ...
      scenes/                       the cells' source (scene.py, grader, solve.py, phases) — no caches/assets
      data/<batch>/ep_NNNN/         traj.npz, meta.json, imgs/*.mp4 + imgs/render_*.json
    <release_dir>/index.json        per-task summary

Episode paths are kept verbatim so every *_set.json and render_manifest.json still resolves.
Only intermediate render shards (imgs/*.part.mp4) are left out. Idempotent: existing files of
equal size are skipped, so a rerun after more tasks finish only adds.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _copy(src: Path, dst: Path) -> int:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


def aggregate_task(task_dir: Path, gen: str, out: Path) -> dict:
    g = task_dir / "data_gen" / gen
    status = json.loads((g / "status.json").read_text())
    if status.get("stage") != "DONE":
        raise SystemExit(f"{task_dir.name}: stage is {status.get('stage')!r}, not DONE")
    physics = json.loads((g / "physics_set.json").read_text())
    eps = physics["episodes"]
    render = json.loads((g / "render_manifest.json").read_text()) if (g / "render_manifest.json").is_file() else {}
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in ("gen.yaml", "status.json", "scene_set.json", "strategy_set.json", "phase_set.json",
                 "physics_set.json", "render_manifest.json"):
        if (g / name).is_file():
            copied += _copy(g / name, out / name)
    if (task_dir / "run.json").is_file():
        copied += _copy(task_dir / "run.json", out / "run.json")
    # cell source: every regular file under scenes/ except caches and the asset symlinks
    for p in (g / "scenes").rglob("*"):
        if p.is_symlink() or not p.is_file() or "__pycache__" in p.parts:
            continue
        copied += _copy(p, out / p.relative_to(g))

    rows, steps, cells, cams, draws_seen = [], [], set(), set(), set()
    render_by_ep: dict[str, list] = {}
    for it in render.get("items", []):
        render_by_ep.setdefault(it["episode"], []).append(it)
    for ep in eps:
        src = g / ep
        meta = json.loads((src / "meta.json").read_text())
        copied += _copy(src / "traj.npz", out / ep / "traj.npz")
        copied += _copy(src / "meta.json", out / ep / "meta.json")
        vids = []
        for f in sorted((src / "imgs").glob("*")) if (src / "imgs").is_dir() else []:
            if f.name.endswith(".part.mp4") or not f.is_file():
                continue
            copied += _copy(f, out / ep / "imgs" / f.name)
            if f.suffix == ".mp4":
                vids.append(f.name)
                stem = f.stem                       # <cam>[_drawN|_probe]
                cam, _, draw = stem.partition("_draw")
                if "_probe" in stem:
                    cams.add(stem.replace("_probe", ""))
                else:
                    cams.add(cam)
                    draws_seen.add(int(draw) if draw else 0)
        with np.load(src / "traj.npz") as tr:
            leaves = sorted(tr.files)
            action_dim = int(tr["action"].shape[1]) if "action" in tr.files else None
        cells.add(meta.get("cell", ""))
        steps.append(int(meta.get("steps", 0)))
        rows.append({
            "path": ep, "cell": meta.get("cell"), "scene": (meta.get("cell") or "").split("/")[0],
            "steps": meta.get("steps"), "success_step": meta.get("success_step"), "seed": meta.get("seed"),
            "env_index": meta.get("env_index"), "score": meta.get("score"),
            "replay_verified": bool((meta.get("replay") or {}).get("success")),
            "noise": meta.get("noise") or {}, "physical_params": (meta.get("parameters") or {}).get("physical", {}),
            "controller_changes": len(meta.get("controller_changes") or []),
            "renders": [{"pass": it["pass"], "frames": it["frames"], "complete": it["complete"]}
                        for it in render_by_ep.get(ep, [])],
            "videos": vids, "traj_leaves": leaves, "action_dim": action_dim,
        })
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    first = json.loads((g / eps[0] / "meta.json").read_text())
    ctrl = first.get("controller") or {}
    summary = {
        "task": task_dir.name, "preset": first.get("preset"), "gen": gen,
        "episodes": len(eps), "replay_verified": sum(r["replay_verified"] for r in rows),
        "cells": sorted(cells), "scenes": sorted({c.split("/")[0] for c in cells}),
        "steps_min": min(steps), "steps_median": int(np.median(steps)), "steps_max": max(steps),
        "sim_dt": first.get("sim_dt"), "decimation": first.get("decimation"),
        "controller": {"class": ctrl.get("class"),
                       "leaves": [{"class": l.get("class"), "control_period": l.get("control_period")}
                                  for l in ctrl.get("leaves", [])]},
        "cameras": sorted(cams), "draws_per_episode": len(draws_seen),
        "render_items": len(render.get("items", [])),
        "render_complete": sum(1 for it in render.get("items", []) if it.get("complete")),
        "traj_leaves": rows[0]["traj_leaves"], "action_dim": rows[0]["action_dim"],
        "ladder": status.get("ladder"), "code_hash": physics.get("code_hash"),
        "seed_solution": json.loads((task_dir / "run.json").read_text()).get("source_run") if (task_dir / "run.json").is_file() else None,
        "size_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
    }
    (out / "README.md").write_text(_readme(summary))
    summary["files_copied_this_run"] = copied
    return summary


def _readme(s: dict) -> str:
    gb = s["size_bytes"] / 1e9
    return f"""# {s['task']} — {s['episodes']} demonstrations ({s['preset']})

Generated by the CoSiGen data_gen ladder (`{s['gen']}`): {s['ladder']} -> physics set of
{s['episodes']} episodes, every one replay-verified ({s['replay_verified']}/{s['episodes']}): the recorded
actions, fed back open-loop into the rebuilt world under the recorded control law, reproduce the
task's success. Seed solution: `{s['seed_solution']}`.

- scenes/cells: {len(s['scenes'])} scene variants, {len(s['cells'])} cells (scene/strategy[/phase]); sources under `scenes/`
- sim dt {s['sim_dt']} s, decimation {s['decimation']}; controller {s['controller']['class']} with leaves {s['controller']['leaves']}
- episode length: {s['steps_min']} / {s['steps_median']} / {s['steps_max']} control steps (min / median / max)
- renders: {s['render_complete']}/{s['render_items']} items = {s['draws_per_episode']} looks x {s['episodes']} episodes; cameras {s['cameras']}
- size: {gb:.1f} GB

## Layout

    physics_set.json            the {s['episodes']} delivered episodes (paths below); scene/strategy/phase_set.json = the nested rungs
    episodes.jsonl              one row per episode (path, cell, steps, seed, replay_verified, physical_params, videos, ...)
    render_manifest.json        per (episode, look) render record
    data/<batch>/ep_NNNN/
        meta.json               cell, seed, steps, success_step, controller (the law the episode ran under), controller_changes, noise, replay verdict
        traj.npz                per-step arrays, one row per control step, state recorded BEFORE the step:
                                {s['traj_leaves']}
                                `action` ({s['action_dim']}-dim) is the commanded action at that step
        imgs/<camera>.mp4, <camera>_draw1.mp4, <camera>_draw2.mp4   the episode rendered under each look
        imgs/render_<camera>[_drawN].json                            render parameters of that video

## Reading an episode

    import json, numpy as np
    meta = json.load(open("data/<batch>/ep_0000/meta.json"))
    tr = dict(np.load("data/<batch>/ep_0000/traj.npz"))
    tr["action"].shape, tr["robot/joint_pos"].shape   # (T, {s['action_dim']}), (T, n_joints)
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dgen_out"); ap.add_argument("release_dir")
    ap.add_argument("--gen", default="gen_o50_v3")
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    root, rel = Path(a.dgen_out), Path(a.release_dir)
    tasks = [t for t in a.tasks.split(",") if t] or sorted(
        p.name for p in root.iterdir()
        if (p / "data_gen" / a.gen / "status.json").is_file()
        and json.loads((p / "data_gen" / a.gen / "status.json").read_text()).get("stage") == "DONE")
    index = {"gen": a.gen, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tasks": {}}
    if (rel / "index.json").is_file():
        index["tasks"] = json.loads((rel / "index.json").read_text()).get("tasks", {})
    for t in tasks:
        s = aggregate_task(root / t, a.gen, rel / t)
        index["tasks"][t] = s
        print(f"{t:15s} {s['episodes']} episodes, {s['render_complete']}/{s['render_items']} renders, "
              f"{s['size_bytes']/1e9:.1f} GB, {s['files_copied_this_run']} files copied", flush=True)
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    tot = sum(v["episodes"] for v in index["tasks"].values())
    print(f"index: {len(index['tasks'])} tasks, {tot} episodes, "
          f"{sum(v['size_bytes'] for v in index['tasks'].values())/1e9:.1f} GB -> {rel}", flush=True)


if __name__ == "__main__":
    main()
