"""Smoke / rubric-REJECTION battery for PanDomeEggScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i244`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — hover-drop the egg, contact-lower the inverted pan —
is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and
a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite and still (balls in the bowl,
                            pan flat right-side-up), score ~0 at rest;
  3-4.  randomization     — READBACK over 8 seeded resets: pan xy+yaw, bowl xy, egg xy,
                            shelf xy+yaw all move AND the green side flips;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy      — the seed's literal plan (carry the PAN flat right-side-up
                            under the shelf, egg untouched): ~0, no success;
  7.   wrong object       — the red TOMATO seated on the green pad and covered by the
                            upturned pan: only the flip-prep credit, no success;
  8.   wrong place        — the egg seated + covered on the GRAY decoy pad: no seat/
                            cover credit, no success;
  9.   near-miss uncover  — egg seated, inverted pan parked rim-down BESIDE the pad:
                            partial credit strictly inside (0.40, 0.60), no success;
  10.  tilted rim-perch   — pan dropped rim-onto an off-axis egg: never a settled
                            covered state (rest-height window / enclosure), no success;
  11.  right-side-up pan  — egg lying INSIDE the un-flipped pan on the green pad: the
                            seat z window + inversion gate reject, no success;
  12.  latched credit     — carrying the egg back to the bowl after seating leaves the
                            latched score unchanged (credit does not evaporate);
  13.  egg on top of dome — egg balanced ON the upturned pan bottom over the pad: the
                            seat z window rejects, no success;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i244.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pan_dome_egg")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.75)) + o),
                                tuple(np.array((0.02, 0.05, 0.05)) + o),
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

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def yaw_of(q: torch.Tensor) -> float:
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def shelf_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
        return sp, yaw_of(scene.shelf.data.root_quat_w[0])

    def report(tag: str) -> None:
        egg = (scene.egg.data.root_pos_w - scene.env_origins)[0]
        pan_z = float((scene.pan.data.root_pos_w - scene.env_origins)[0, 2])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | egg=({float(egg[0]):+.3f},{float(egg[1]):+.3f},"
              f"{float(egg[2]):.3f}) pan_z={pan_z:.3f} up_z={float(scene._pan_up_z()[0]):+.2f} "
              f"on_pad={bool(scene.egg_on_pad()[0])} dome={bool(scene.pan_dome()[0])} "
              f"covered={bool(scene.covered()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def inv_quat(yaw: float) -> tuple[float, float, float, float]:
        """qz(yaw) * qx(pi): pan cavity down, handle world dir (cos yaw, sin yaw)."""
        return (0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0)

    def front_dir() -> tuple[float, float, float]:
        """World unit vector out the alcove's open front (shelf-local -y)."""
        _sp, syaw = shelf_pose()
        return (math.sin(syaw), -math.cos(syaw), syaw)

    def drop_ball(body, x: float, y: float, z: float, settle_steps: int = 90) -> None:
        place_body(body, x, y, z, settle_steps=settle_steps)

    def green_xy() -> tuple[float, float]:
        g = (scene.green_pad_w() - scene.env_origins)[0]
        return float(g[0]), float(g[1])

    def gray_xy() -> tuple[float, float]:
        g = (scene.gray_pad_w() - scene.env_origins)[0]
        return float(g[0]), float(g[1])

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    report("reset")
    step(60)
    report("show")
    fin0 = all(torch.isfinite(getattr(scene, nm).data.root_state_w).all()
               for nm in ("shelf", "pads", "pan", "bowl", "egg", "tomato"))
    bowl = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
    egg = (scene.egg.data.root_pos_w - scene.env_origins)[0]
    tom = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
    in_bowl = (float((egg[:2] - bowl[:2]).norm()) < c.bowl_base_r
               and float((tom[:2] - bowl[:2]).norm()) < c.bowl_base_r)
    pan_flat = float(scene._pan_up_z()[0]) > 0.95
    check("settle: states finite, balls in the bowl, pan flat right-side-up, at rest",
          bool(fin0) and in_bowl and pan_flat and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, sides = [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        p = (scene.pan.data.root_pos_w - scene.env_origins)[0]
        pyaw = yaw_of(scene.pan.data.root_quat_w[0])
        b = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        e = (scene.egg.data.root_pos_w - scene.env_origins)[0]
        sp, syaw = shelf_pose()
        # which side is green: sign of green offset along the SHELF's local +x
        gx, gy = green_xy()
        side = math.copysign(1.0, (gx - float(sp[0])) * math.cos(syaw)
                             + (gy - float(sp[1])) * math.sin(syaw))
        sides.append(side)
        reads.append((float(p[0]), float(p[1]), pyaw, float(b[0]), float(b[1]),
                      float(e[0]), float(e[1]), float(sp[0]), float(sp[1]), syaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pan_x, pan_y, pan_yaw, bowl_x, bowl_y, "
          f"egg_x, egg_y, shelf_x, shelf_y, shelf_yaw):\n{arr}", flush=True)
    print(f"[smoke] green side per seed: {sides}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: pan xy + yaw, bowl xy, egg xy vary across seeded resets",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2
          and spread[3] > 0.005 and spread[4] > 0.005
          and spread[5] > 0.005 and spread[6] > 0.005)
    check("randomization: shelf xy + yaw vary AND the green side flips across seeds",
          spread[7] > 0.005 and spread[8] > 0.005 and spread[9] > 0.03
          and (+1.0 in sides) and (-1.0 in sides))

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy ===========================================
    # The seed's literal plan: carry the PAN (flat, right-side-up) under the shelf and
    # leave it there; the egg is never touched. The rubric owes it ~nothing.
    env.reset(seed=41)
    step(10)
    sp, syaw = shelf_pose()
    place_body(scene.pan, float(sp[0]), float(sp[1]), 0.020,
               quat=(math.cos((syaw - math.pi / 2) / 2), 0.0, 0.0,
                     math.sin((syaw - math.pi / 2) / 2)),
               settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: pan carried right-side-up under the shelf, egg untouched — "
          "score <= 0.15, no success", s <= 0.15 and not ok)

    # =========================== 7. wrong object ============================================
    # The red TOMATO seated on the green pad and covered by the upturned pan: at most the
    # flip-prep credit sticks; seat/cover reference the EGG only.
    env.reset(seed=51)
    step(10)
    gx, gy = green_xy()
    drop_ball(scene.tomato, gx, gy, c.egg_seat_z + 0.015)
    _sp, syaw = shelf_pose()
    place_body(scene.pan, gx, gy, c.rim_rest_z + 0.020,
               quat=inv_quat(syaw - math.pi / 2), settle_steps=120)
    report("wrong-object")
    s, ok = judge()
    tom_rel = (scene.tomato.data.root_pos_w - scene.green_pad_w())[0, :2].norm()
    check("wrong object: TOMATO seated + covered on the green pad — score <= 0.25, "
          "no success, no seat/cover latch",
          float(tom_rel) < c.egg_xy_tol and bool(scene.pan_dome()[0])
          and s <= 0.25 and not ok
          and not bool(scene.seat_latch[0]) and not bool(scene.cover_latch[0]))

    # =========================== 8. wrong place =============================================
    # The EGG seated + covered on the GRAY decoy pad: the seat is green-pad-relative.
    env.reset(seed=61)
    step(10)
    dx, dy = gray_xy()
    drop_ball(scene.egg, dx, dy, c.egg_seat_z + 0.015)
    _sp, syaw = shelf_pose()
    place_body(scene.pan, dx, dy, c.rim_rest_z + 0.020,
               quat=inv_quat(syaw - math.pi / 2), settle_steps=120)
    report("wrong-place")
    s, ok = judge()
    egg_rel_gray = (scene.egg.data.root_pos_w - scene.gray_pad_w())[0, :2].norm()
    check("wrong place: egg seated + covered on the GRAY pad — score <= 0.30, no "
          "success, no seat/cover latch",
          float(egg_rel_gray) < c.egg_xy_tol and bool(scene.pan_dome()[0])
          and s <= 0.30 and not ok
          and not bool(scene.seat_latch[0]) and not bool(scene.cover_latch[0]))

    # =========================== 9. near-miss: seated but uncovered =========================
    # Egg correctly seated; the inverted pan parked rim-down 0.20 m out the front —
    # flip-prep latches but cover does not: strictly partial credit.
    env.reset(seed=71)
    step(10)
    gx, gy = green_xy()
    drop_ball(scene.egg, gx, gy, c.egg_seat_z + 0.015)
    fx, fy, syaw = front_dir()
    place_body(scene.pan, gx + 0.20 * fx, gy + 0.20 * fy, c.rim_rest_z + 0.020,
               quat=inv_quat(syaw - math.pi / 2), settle_steps=120)
    report("near-miss")
    s_near, ok = judge()
    check("near-miss: egg seated, inverted pan parked beside — score in (0.40, 0.60), "
          "NOT success, cover not latched",
          bool(scene.egg_on_pad()[0]) and bool(scene.pan_dome()[0])
          and 0.40 <= s_near <= 0.60 and not ok and not bool(scene.cover_latch[0]))

    # =========================== 10. tilted rim-perch =======================================
    # Pan dropped so its rim lands ON an off-axis egg (egg under the wall ring): it
    # either perches tilted/high (rest-height window rejects) or slides off AWAY from
    # the egg (enclosure rejects). Never a covered state.
    env.reset(seed=81)
    step(10)
    gx, gy = green_xy()
    drop_ball(scene.egg, gx, gy, c.egg_seat_z + 0.015)
    fx, fy, syaw = front_dir()
    place_body(scene.pan, gx + c.wall_rm * fx, gy + c.wall_rm * fy, 0.095,
               quat=inv_quat(syaw - math.pi / 2), settle_steps=150)
    report("rim-perch")
    s, ok = judge()
    check("tilted rim-perch: pan dropped rim-onto the off-axis egg — never covered, "
          "no success, no cover latch",
          not ok and not bool(scene.cover_latch[0]) and not bool(scene.covered()[0]))

    # =========================== 11. egg inside the right-side-up pan =======================
    # The lazy ending: pan un-flipped on the green pad with the egg lying INSIDE it.
    # The egg rides ~13 mm above the seat window and the pan fails the inversion gate.
    env.reset(seed=91)
    step(10)
    gx, gy = green_xy()
    _sp, syaw = shelf_pose()
    place_body(scene.pan, gx, gy, 0.020,
               quat=(math.cos((syaw - math.pi / 2) / 2), 0.0, 0.0,
                     math.sin((syaw - math.pi / 2) / 2)),
               settle_steps=60)
    pan_now = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    drop_ball(scene.egg, float(pan_now[0]), float(pan_now[1]),
              float(pan_now[2]) + c.pan_base_t / 2 + c.egg_r + 0.015)
    report("in-upright-pan")
    s, ok = judge()
    check("right-side-up pan: egg lying inside the un-flipped pan on the pad — seat z "
          "window + inversion reject, score <= 0.25, no success",
          not bool(scene.egg_on_pad()[0]) and not bool(scene.pan_dome()[0])
          and s <= 0.25 and not ok
          and not bool(scene.seat_latch[0]) and not bool(scene.cover_latch[0]))

    # =========================== 12. latched credit survives removal ========================
    env.reset(seed=101)
    step(10)
    gx, gy = green_xy()
    drop_ball(scene.egg, gx, gy, c.egg_seat_z + 0.015)
    s_in, _ = judge()
    bowl = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
    drop_ball(scene.egg, float(bowl[0]), float(bowl[1]) + 0.01, 0.05, settle_steps=60)
    report("egg-removed")
    s_out, ok = judge()
    check("latched credit: seating the egg then carrying it back to the bowl leaves "
          f"the latched score unchanged ({s_in:.3f} -> {s_out:.3f})",
          s_in >= 0.40 and abs(s_out - s_in) < 1e-3 and not ok)

    # =========================== 13. egg balanced ON TOP of the dome ========================
    # Upturned pan correctly rim-down on the pad, egg balanced on the pan's upturned
    # bottom directly over the pad centre: covered()'s xy test alone would pass — the
    # egg SEAT z window is what rejects the state.
    env.reset(seed=111)
    step(10)
    gx, gy = green_xy()
    _sp, syaw = shelf_pose()
    place_body(scene.pan, gx, gy, c.rim_rest_z + 0.020,
               quat=inv_quat(syaw - math.pi / 2), settle_steps=90)
    drop_ball(scene.egg, gx, gy, c.rim_rest_z + c.pan_base_t / 2 + c.egg_r + 0.012,
              settle_steps=90)
    report("egg-on-dome")
    s, ok = judge()
    check("egg on top of the dome: pan rim-down but egg riding on its upturned bottom "
          "— seat z window rejects, no success, no seat/cover latch",
          bool(scene.pan_dome()[0]) and not bool(scene.egg_on_pad()[0])
          and not ok and not bool(scene.seat_latch[0])
          and not bool(scene.cover_latch[0]) and s <= 0.30)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(torch.isfinite(getattr(scene, nm).data.root_state_w).all()
              for nm in ("shelf", "pads", "pan", "bowl", "egg", "tomato"))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pan_dome_egg")
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
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
