"""Smoke / rubric-REJECTION battery for PanUnderLowShelfScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i4`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the contact-dynamics push — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite and still, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: pan spawn xy+yaw, shelf
                            xy+yaw all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's lift-and-lower plan: the pan released from
                            above the goal footprint lands ON the roof (the alcove is
                            covered); no insertion credit, no success, score ~0.2 max
                            (approach only);
  7.  wrong object        — the white bowl settled fully inside the alcove (it fits):
                            identity matters, score ~0, no success;
  8.  near-miss           — pan settled under the roof but 17 mm short of the
                            fully-in band: NOT success, score < 0.9;
  9.  straddle            — pan settled half in the doorway: partial credit strictly
                            inside (0, 0.9), NOT success;
  10. monotonicity        — the deeper probe latched strictly more insertion credit
                            than the shallower one;
  11. latched credit      — pulling the pan back OUT after the near-miss leaves the
                            latched score unchanged (credit does not evaporate);
  12. rejection audit     — success() was never True at ANY judged point in this
                            battery (rejection-only contract);
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i4.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pan_lowshelf_slide")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.00, -1.05, 0.80)) + o),
                                tuple(np.array((0.05, 0.00, 0.04)) + o),
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

    def pan_local():
        return scene._pan_local()[0]

    def shelf_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
        q = scene.shelf.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        loc = pan_local()
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pan_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) frac_latch={float(scene.frac_latch[0]):.3f} "
              f"appr_latch={float(scene.approach_latch[0]):.3f} score={s:.3f} "
              f"success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_body(body, x: float, y: float, z: float, yaw: float = 0.0,
                   settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the i17/i53 zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_pan_local(x_l: float, y_l: float, settle_steps: int = 60) -> None:
        """Place the pan flat at shelf-local (x_l, y_l), handle pointing out the front."""
        sp, syaw = shelf_pose()
        cy, sy = math.cos(syaw), math.sin(syaw)
        wx = float(sp[0]) + cy * x_l - sy * y_l
        wy = float(sp[1]) + sy * x_l + cy * y_l
        place_body(scene.pan, wx, wy, c.pan_h / 2 + 0.002, yaw=syaw - math.pi / 2,
                   settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.pan.data.root_state_w
    base_z = float((scene.pan.data.root_pos_w - scene.env_origins)[0, 2]) - c.pan_h / 2
    check("settle: pan state finite, at rest flat on the table",
          bool(torch.isfinite(st0).all()) and bool(scene.settled()[0]) and abs(base_z) < 0.01)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        p = (scene.pan.data.root_pos_w - scene.env_origins)[0]
        pq = scene.pan.data.root_quat_w[0]
        pyaw = 2.0 * math.atan2(float(pq[3]), float(pq[0]))
        sp, syaw = shelf_pose()
        reads.append((float(p[0]), float(p[1]), pyaw, float(sp[0]), float(sp[1]), syaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pan_x, pan_y, pan_yaw, shelf_x, shelf_y, "
          f"shelf_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: pan spawn xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: shelf xy + yaw vary across seeded resets (readback)",
          spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.03)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: lift-and-lower is blocked ================
    # The seed's plan — carry the pan through the air and lower it into the goal region —
    # meets the roof: released flat above the alcove's footprint, the pan lands ON TOP.
    torch.manual_seed(41)
    env.reset()
    step(10)
    sp, syaw = shelf_pose()
    place_body(scene.pan, float(sp[0]), float(sp[1]), 0.12, yaw=syaw - math.pi / 2,
               settle_steps=120)
    report("seed-strategy")
    loc = pan_local()
    s, ok = judge()
    check("seed strategy (lower from above): pan lands ON the roof — no insertion credit, "
          "no success, score <= 0.25",
          float(loc[2]) > c.h_gap and not ok
          and float(scene.frac_latch[0]) <= 0.05 and s <= 0.25)

    # =========================== 7. wrong object ============================================
    torch.manual_seed(51)
    env.reset()
    step(10)
    sp, syaw = shelf_pose()
    place_body(scene.bowl, float(sp[0]), float(sp[1]), c.bowl_h / 2 + 0.002,
               settle_steps=60)
    report("wrong-object")
    s, ok = judge()
    bowl_rel = (scene.bowl.data.root_pos_w - scene.shelf.data.root_pos_w)[0, :2].norm()
    check("wrong object: the bowl settled fully inside the alcove scores ~0, no success",
          float(bowl_rel) < 0.03 and s <= 0.05 and not ok)

    # =========================== 8. near-miss (just outside the depth band) =================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_pan_local(0.0, -(c.alcove_d / 2 - c.pan_r + c.edge_tol) - 0.017)  # 17 mm short
    report("near-miss")
    s_near, ok = judge()
    frac_near = float(scene.frac_latch[0])
    check("near-miss: pan settled 17 mm short of fully-in — NOT success, score < 0.9",
          not bool(scene.fully_in()[0]) and not ok and 0.1 < s_near < 0.9)

    # =========================== 9. straddle (half in the doorway) ==========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_pan_local(0.0, -c.alcove_d / 2)  # disc centre on the front plane
    report("straddle")
    s_half, ok = judge()
    frac_half = float(scene.frac_latch[0])
    check("straddle: pan half inside the doorway — partial credit strictly in (0, 0.9), "
          "NOT success",
          not ok and 0.05 < s_half < 0.9 and not bool(scene.fully_in()[0]))

    # =========================== 10. monotonicity ===========================================
    check("monotonicity: deeper probe latched strictly more insertion credit "
          f"({frac_half:.3f} < {frac_near:.3f})", frac_half + 0.05 < frac_near)

    # =========================== 11. latched credit survives pull-out =======================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_pan_local(0.0, -(c.alcove_d / 2 - c.pan_r + c.edge_tol) - 0.017)
    s_in, _ = judge()
    place_pan_local(0.0, -(c.alcove_d / 2 + c.pan_r + 0.10))  # pull it back out front
    report("pulled-out")
    s_out, ok = judge()
    check("latched credit: pulling the pan back out leaves the latched score unchanged",
          abs(s_out - s_in) < 1e-3 and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.pan.data.root_state_w).all()
           and torch.isfinite(scene.bowl.data.root_state_w).all()
           and torch.isfinite(scene.shelf.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pan_lowshelf_slide")
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
