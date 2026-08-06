"""Smoke / rubric-REJECTION battery for HatchShelfScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i306`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the force-driven over-centre lid close + gravity
placement — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as
a settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: lid resting fully open past
                            vertical, bowls standing on the floor, score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: bowl spawn slots swap and
                            jitter; the lid's initial open angle varies;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (carry the bowl to the top of the cabinet
                            and release it) executed with the lid as found (open): the
                            bowl falls THROUGH the mouth into the cavity -> no success;
  7.  buried is not on    — lid then closed OVER the trapped bowl (follower-only
                            re-pose): lid closed + bowl inside the cabinet -> the bowl
                            is UNDER the top, not ON it -> no success;
  8.  wrong object        — fresh episode, lid closed, WHITE decoy bowl placed dead
                            centre on the lid -> no success;
  9.  wrong orientation   — BLACK bowl placed UPSIDE-DOWN on the closed lid -> the
                            upright clause rejects -> no success;
  10. near-miss placement — BLACK bowl upright ON the lid but parked at the lid's
                            front edge, outside the centered xy tolerance -> no success;
  11. latched credit      — the bowl removed back to the floor: the latched close/carry
                            credit does not evaporate (score unchanged), no success;
  12. open lid no shelf   — BLACK bowl released against the OPEN lid's raised face ->
                            slides/tumbles off; wherever it lands it is not on a closed
                            lid -> no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i306.smoke --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hatch_shelf")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    cx, cy = c.cab_pos

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.25, 1.00)) + o),
                                tuple(np.array((0.50, 0.00, 0.20)) + o),
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

    def report(tag: str) -> None:
        p_blk = (scene.bowl_black.data.root_pos_w - scene.env_origins)[0]
        p_wht = (scene.bowl_white.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | black=({float(p_blk[0]):+.3f},{float(p_blk[1]):+.3f},"
              f"{float(p_blk[2]):.3f}) white=({float(p_wht[0]):+.3f},"
              f"{float(p_wht[1]):+.3f},{float(p_wht[2]):.3f}) "
              f"lid={float(scene.lid_open_deg()[0]):+.1f}deg "
              f"closed={bool(scene.lid_closed()[0])} "
              f"on_blk={bool(scene._on_lid(scene.bowl_black)[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def lid_pose(open_deg: float) -> None:
        """Follower-only re-pose of the lid on its hinge arc (instrumentation)."""
        h = math.radians(-open_deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_x, cy, c.hinge_z
        st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
        st[:, 0:3] += env.iscene.env_origins
        scene.lid.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def blk_z() -> float:
        return float((scene.bowl_black.data.root_pos_w - scene.env_origins)[0, 2])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.bowl_black.data.root_state_w).all()
            and torch.isfinite(scene.bowl_white.data.root_state_w).all()
            and torch.isfinite(scene.lid.data.root_state_w).all())
    z_blk, z_wht = blk_z(), float((scene.bowl_white.data.root_pos_w
                                   - scene.env_origins)[0, 2])
    lid_open = float(scene.lid_open_deg()[0])
    still = bool(scene._bowl_still(scene.bowl_black)[0]
                 and scene._bowl_still(scene.bowl_white)[0])
    check("settle: states finite, lid resting fully open past vertical (> 95 deg), "
          "both bowls standing on the floor, everything still",
          bool(fin0) and lid_open > 95.0 and z_blk < 0.03 and z_wht < 0.03 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)  # readback before gravity moves the lid off its written angle
        bx_, by_ = obj_xy(scene.bowl_black)
        reads.append((bx_, by_, float(scene.lid_open_deg()[0]),
                      1.0 if by_ > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (black_x, black_y, lid_open_deg, "
          f"black_left):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: bowl spawn slots swap and xy jitter is real across seeded "
          "resets (readback)",
          0.0 < arr[:, 3].mean() < 1.0 and spread[0] > 0.015 and spread[1] > 0.05)
    check("randomization: the lid's initial open angle varies across seeds (readback "
          "spread > 3 deg, all within the open band)",
          spread[2] > 3.0 and arr[:, 2].min() > 95.0
          and arr[:, 2].max() < c.open_limit_deg + 3.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: bowl to the top, lid as found ============
    # The seed's plan verbatim: carry the black bowl to "the top of the cabinet" and
    # release it. With the lid standing open there is no top: the bowl falls through
    # the mouth into the cavity.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.bowl_black, cx, cy, 0.40)
    step(150)
    report("seed-strategy")
    s, ok = judge()
    in_cavity = (blk_z() < 0.18
                 and abs(obj_xy(scene.bowl_black)[0] - cx) < c.in_d / 2
                 and abs(obj_xy(scene.bowl_black)[1] - cy) < c.in_w / 2)
    check("seed strategy: bowl released over the open-lidded cabinet falls THROUGH "
          "the mouth into the cavity — no success, score <= 0.25",
          in_cavity and not ok and s <= 0.25)

    # =========================== 7. buried under the closed lid is not "on top" =============
    lid_pose(0.0)
    step(90)
    report("buried")
    s7, ok = judge()
    check("buried: lid closed OVER the trapped bowl — lid closed but the bowl is "
          "UNDER the top, not on it: no success, score <= 0.60",
          bool(scene.lid_closed()[0]) and blk_z() < 0.18 and not ok and s7 <= 0.60)

    # =========================== 8. wrong object: the white decoy ===========================
    torch.manual_seed(51)
    env.reset()
    step(10)
    lid_pose(0.0)
    step(30)
    place(scene.bowl_white, c.lid_center_x, cy, c.lid_top_z + 0.020)
    step(90)
    report("wrong-object")
    s, ok = judge()
    w_on = bool(scene._on_lid(scene.bowl_white)[0])
    check("wrong object: WHITE decoy bowl settled dead centre on the closed lid — "
          "no success, score <= 0.40 (the black bowl is untouched on the floor)",
          w_on and not ok and s <= 0.40)

    # =========================== 9. wrong orientation: upside-down ==========================
    torch.manual_seed(61)
    env.reset()
    step(10)
    lid_pose(0.0)
    step(30)
    place(scene.bowl_black, c.lid_center_x, cy, c.lid_top_z + c.bowl_h + 0.012,
          quat=(0.0, 1.0, 0.0, 0.0))  # 180 deg about x: rim down
    step(90)
    report("upside-down")
    s, ok = judge()
    q = scene.bowl_black.data.root_quat_w[0]
    r33 = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    on_xy = (abs(obj_xy(scene.bowl_black)[0] - c.lid_center_x) < c.on_xy_tol
             and abs(obj_xy(scene.bowl_black)[1] - cy) < c.on_xy_tol)
    check("wrong orientation: BLACK bowl upside-down on the closed lid, centered — "
          "the upright clause rejects: no success",
          on_xy and blk_z() > c.on_z_lo and r33 < 0.0 and not ok)

    # =========================== 10. near-miss: parked at the lid's edge ====================
    torch.manual_seed(71)
    env.reset()
    step(10)
    lid_pose(0.0)
    step(30)
    edge_x = c.hinge_x + 0.020  # bowl centre 2 cm inboard of the lid's front edge
    place(scene.bowl_black, edge_x, cy, c.lid_top_z + 0.020)
    step(90)
    report("edge-parked")
    s10, ok = judge()
    on_lid_z = blk_z() > c.on_z_lo
    off_center = abs(obj_xy(scene.bowl_black)[0] - c.lid_center_x) > c.on_xy_tol
    check("near-miss: BLACK bowl upright ON the closed lid but parked at the front "
          "edge, outside the centered tolerance — no success",
          on_lid_z and off_center and not ok)

    # =========================== 11. latched credit survives regression =====================
    place(scene.bowl_black, 0.30, 0.25, 0.003)
    step(60)
    report("regressed")
    s11, ok = judge()
    check("latched credit: bowl removed back to the floor — the latched close/carry "
          f"credit does not evaporate ({s10:.3f} -> {s11:.3f}), still no success",
          abs(s11 - s10) < 1e-3 and s11 >= c.w_closed - 0.01 and not ok)

    # =========================== 12. the open lid is not a shelf ============================
    torch.manual_seed(81)
    env.reset()
    step(10)
    # release the bowl against the OPEN lid's raised face: it cannot rest there
    a = math.radians(float(scene.lid_open_deg()[0]))
    face_x = c.hinge_x + 0.5 * c.lid_l * math.cos(a) - 0.05
    face_z = c.hinge_z + 0.5 * c.lid_l * math.sin(a) + 0.02
    place(scene.bowl_black, face_x, cy, face_z)
    step(180)
    report("open-lid-drop")
    s, ok = judge()
    check("open lid is not a shelf: bowl released against the standing lid's face "
          "slides/tumbles off — wherever it lands, no success",
          not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.bowl_black.data.root_state_w).all()
           and torch.isfinite(scene.bowl_white.data.root_state_w).all()
           and torch.isfinite(scene.lid.data.root_state_w).all()
           and torch.isfinite(scene.cabinet.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hatch_shelf")
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
