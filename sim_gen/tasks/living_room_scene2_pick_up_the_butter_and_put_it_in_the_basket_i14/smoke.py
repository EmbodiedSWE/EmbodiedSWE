"""smoke — REJECTION battery for the ButterHopperScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — the real Franka run — already proves the
rubric ACCEPTS the correct outcome). Every check here CONSTRUCTS a wrong outcome as a
settled/pinned state (teleports are instrumentation; the tray is only ever pose-written
as the joint FOLLOWER, which is safe on this stack) and asserts the rubric REJECTS it.
No probe ever reaches success() — a global flag watches every substep.

  1. settle/no-NaN     — reset layout settles finite: butter seated in the shaft on the
                         tray, tray closed, score 0, no success;
  2. randomization     — READBACK: basket position/yaw and butter jitter differ across
                         seeded resets;
  3. null policy       — 240 idle steps -> score ~0, no success;
  4. seed strategy     — the seed's end state (butter placed straight into the basket,
                         dispenser untouched/still sealed) constructed by teleport:
                         containment reads TRUE yet success stays False (the tray-open
                         clause is load-bearing — no honest delivery leaves it sealed);
  5. near-miss open    — butter in the basket but the tray pinned 20 mm SHORT of the
                         open threshold -> success False (tolerance is load-bearing);
  6. wrong order A     — tray pulled fully open with the basket still at spawn: the
                         butter drops onto the bare floor -> success False;
  7. wrong order B     — the out-of-order END state: basket then moved onto the drop
                         zone (butter already on the floor) -> still False;
  8. tray honesty      — after the pin is released the tray STAYS open (no hidden
                         restoring force sneaks the scene back toward a solvable state);
  9. rim topple        — butter perched on the basket rim (tray genuinely open) topples
                         OUTSIDE -> not contained, success False;
 10. flipped basket    — basket upside-down on the drop zone, butter resting on its
                         upturned bottom (tray open) -> success False (upright clause);
 11. staged partials   — reset < staged (0.30) < staged+part-open < 1.0, monotone,
                         never success;
 12. never-success     — global: no probe in this battery ever reached success().

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i14.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
from simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i14 import (  # noqa: E402
    scene as scene_mod,  # noqa: F401
)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_hopper")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    hx, hy = c.hopper_xy

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.75, -1.35, 0.85)) + o),
                                tuple(np.array((0.05, 0.0, 0.20)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    ever_success = {"v": False}

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            ever_success["v"] |= bool(scene.success()[0])
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        bu = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        print(f"[smoke] {tag:18s} open={float(scene.opening()[0])*1000:5.1f}mm "
              f"butter_z={float(bu[2]):.3f} contained={bool(scene.contained()[0])} "
              f"staged={bool(scene._staged[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- probe constructors (instrumentation, not a solution) ------------------------------
    def pin_tray(opening: float, n_sub: int) -> None:
        """Kinematically hold the tray at `opening` for n_sub substeps (FOLLOWER-only
        pose write — the housing anchor never moves; safe on this stack)."""
        for _ in range(n_sub):
            st = torch.zeros(1, 13, device=device)
            st[0, 0], st[0, 1], st[0, 2] = hx, hy - opening, c.tray_z
            st[0, 3] = 1.0
            st[0, 0:3] += scene.env_origins[0]
            scene.tray.write_root_state_to_sim(st, ids)
            step(1)

    def put_basket(dxy: tuple, yaw: float = 0.0, flipped: bool = False,
                   settle: int = 180) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1] = dxy[0], dxy[1]
        if flipped:
            st[0, 2] = c.basket_floor_t + c.basket_wall_h + 0.004
            st[0, 3], st[0, 4] = math.cos(math.pi / 2), math.sin(math.pi / 2)  # roll pi
        else:
            st[0, 2] = 0.004
            st[0, 3], st[0, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        st[0, 0:3] += scene.env_origins[0]
        scene.basket.write_root_state_to_sim(st, ids)
        step(settle)

    def put_butter(pos, quat=(1.0, 0.0, 0.0, 0.0), settle: int = 240) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = pos
        st[0, 3:7] = torch.tensor(quat, device=device)
        st[0, 0:3] += scene.env_origins[0]
        scene.butter.write_root_state_to_sim(st, ids)
        step(settle)

    def basket_xy() -> tuple:
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        return float(bp[0]), float(bp[1])

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bu = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    check("settle: states finite, butter seated in the shaft on the tray, tray closed",
          bool(torch.isfinite(scene.butter.data.root_state_w).all())
          and bool(torch.isfinite(scene.tray.data.root_state_w).all())
          and abs(float(bu[2]) - c.butter_home_z) < 0.008
          and float((bu[:2] - torch.tensor([hx, hy], device=device)).norm()) < 0.02
          and float(scene.opening()[0]) < 0.005)
    check("settle: score ~0 at reset, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        env.reset(seed=s)
        step(5)
        bp = (scene.basket.data.root_pos_w - scene.env_origins)[0]
        q = scene.basket.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * float(q[2] * q[2] + q[3] * q[3]))
        but = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(bp[0]), float(bp[1]), yaw, float(but[0]), float(but[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (basket_x, basket_y, basket_yaw, "
          f"butter_x, butter_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: basket pose and butter jitter differ across seeded resets "
          "(readback)",
          (spread[0] + spread[1]) > 0.05 and spread[2] > 0.3
          and (spread[3] + spread[4]) > 0.0015)

    # =========================== 3. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4. seed strategy: butter straight into basket ==============
    env.reset(seed=41)
    step(60)
    bx, by = basket_xy()
    put_butter((bx, by, c.basket_floor_t + c.butter_size[2] / 2 + 0.006))
    report("seed-strategy")
    check("seed strategy: butter placed straight into the basket with the dispenser "
          "still sealed reads contained=True yet NEVER succeeds (tray-open clause)",
          bool(scene.contained()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 1.0)

    # =========================== 5. near-miss: tray 20 mm short of open =====================
    env.reset(seed=51)
    step(60)
    bx, by = basket_xy()
    put_butter((bx, by, c.basket_floor_t + c.butter_size[2] / 2 + 0.006), settle=60)
    pin_tray(c.open_min - 0.020, 180)
    report("open-near-miss")
    check("near-miss: butter contained but tray pinned 20 mm short of open_min "
          "never succeeds (open tolerance load-bearing)",
          bool(scene.contained()[0]) and float(scene.opening()[0]) < c.open_min
          and not bool(scene.success()[0]))

    # =========================== 6./7./8. wrong order + honesty =============================
    env.reset(seed=61)
    step(60)
    pin_tray(0.185, 220)   # dispense with the basket still at spawn
    report("wrong-order-A")
    bu = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    check("wrong order: tray opened before staging drops the butter on the bare floor "
          "-> no success",
          float(bu[2]) < 0.060 and not bool(scene.contained()[0])
          and not bool(scene.success()[0]))
    put_basket((hx, hy), settle=240)   # the out-of-order END state
    report("wrong-order-B")
    check("out-of-order end state: basket moved onto the drop zone AFTER the drop "
          "(even capturing the floored slab inside it) still fails — the floor "
          "latch makes the wrong order irreversible",
          bool(scene._floored[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.65)
    op0 = float(scene.opening()[0])
    step(120)
    report("tray-honesty")
    check("tray honesty: released tray stays open (no hidden restoring force)",
          float(scene.opening()[0]) >= min(op0 - 0.005, 0.14)
          and float(scene.opening()[0]) >= 0.14)

    # =========================== 9. rim topple ==============================================
    env.reset(seed=71)
    step(60)
    pin_tray(c.open_min + 0.015, 160)  # open clause TRUE; shaft butter falls to floor
    # park the fallen butter far away FIRST (the basket teleports onto the drop zone
    # next — overlapping it with the floor butter would depenetration-kick both)
    put_butter((hx - 0.05, hy + 0.45, c.butter_size[2] / 2 + 0.003), settle=90)
    put_basket((hx, hy), settle=180)
    # now perch the butter on the basket rim, biased outward -> it topples OUTSIDE
    put_butter((hx + c.basket_inner_half + c.basket_wall_t / 2 + 0.012, hy,
                c.basket_floor_t + c.basket_wall_h + c.butter_size[2] / 2 + 0.004),
               settle=300)
    report("rim-topple")
    check("near-miss: butter perched on the rim topples OUTSIDE the basket "
          "(tray genuinely open) -> not contained, no success",
          not bool(scene.contained()[0]) and not bool(scene.success()[0]))

    # =========================== 10. flipped basket =========================================
    env.reset(seed=81)
    step(60)
    pin_tray(c.open_min + 0.015, 160)
    put_basket((hx, hy), flipped=True, settle=180)
    bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0][2])
    put_butter((hx, hy, bz + c.basket_floor_t / 2 + c.butter_size[2] / 2 + 0.004),
               settle=240)
    report("flipped-basket")
    check("wrong outcome: basket upside-down on the drop zone with butter resting on "
          "its upturned bottom (tray open) never succeeds (upright clause)",
          not bool(scene.success()[0]) and not bool(scene.contained()[0]))

    # =========================== 11. staged partials, monotone ==============================
    env.reset(seed=91)
    step(60)
    s0 = float(scene.score()[0])
    put_basket((hx, hy), settle=200)
    s1 = float(scene.score()[0])
    pin_tray(0.070, 140)   # butter still fully supported at this opening — stays in shaft
    s2 = float(scene.score()[0])
    bu = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] staged partials: reset={s0:.2f} staged={s1:.2f} "
          f"part-open={s2:.2f} butter_z={float(bu[2]):.3f}", flush=True)
    check("staged partials monotone: 0 <= reset < staged (0.30) < staged+part-open "
          "< 1.0, butter still in the shaft, never success",
          s0 <= 0.005 and 0.28 <= s1 <= 0.35 and s1 < s2 <= 0.65
          and abs(float(bu[2]) - c.butter_home_z) < 0.010
          and not bool(scene.success()[0]))

    # =========================== 12. never-success global ===================================
    check("never-success: no probe in this battery ever reached success()",
          not ever_success["v"])

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.butter_hopper")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
