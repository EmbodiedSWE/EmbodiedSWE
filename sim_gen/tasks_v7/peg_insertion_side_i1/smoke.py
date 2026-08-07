"""Smoke / rubric-REJECTION battery for RamEjectScene (sim_gen task
`peg_insertion_side_i1`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — enter the bore and ram the ball out under contact
dynamics — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite; ball readback INSIDE the bore,
                            rod at rack grasp height; score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: rod spawn xy+yaw, assembly
                            xy+yaw, and the ball's bore depth all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's plan end state: rod inserted partway into the
                            hole, nothing else moved -> NOT success, score capped at
                            the small instrumental-insertion share;
  7.  near-miss push      — ball driven to just short of the muzzle (still supported on
                            the bore floor), settled -> NOT success, score < 0.9;
  8.  wrong-end/landing   — the yellow ball settled on the TABLE outside the basin (a
                            backwards ejection / bad landing) -> NOT success;
  9.  wrong object        — the RED decoy placed in the basin, settled -> identity
                            matters: NOT success, score ~0;
  10. monotonicity        — a deeper ball push latched strictly more transmission
                            credit than a shallower one (same seed, same s0);
  11. latched credit      — teleporting the ball back toward the entry after the
                            near-miss leaves the latched score unchanged;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i1.smoke --headless
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
    env = ENVS.get("simgen.ram_eject")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.70)) + o),
                                tuple(np.array((0.02, 0.02, 0.05)) + o),
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

    def asm_pose() -> tuple[torch.Tensor, float]:
        ap = (scene.assembly.data.root_pos_w - scene.env_origins)[0]
        q = scene.assembly.data.root_quat_w[0]
        return ap, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene.ball_local()[0]
        tip = scene.rod_tip_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) tip_x={float(tip[0]):+.3f} "
              f"insert_latch={float(scene.insert_latch[0]):.3f} "
              f"push_latch={float(scene.push_latch[0]):.3f} score={s:.3f} success={ok} "
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

    def place_local(body, x_l: float, y_l: float, z: float, quat=None,
                    settle_steps: int = 60) -> None:
        """Place a body at assembly-local (x_l, y_l) and world z."""
        ap, ayaw = asm_pose()
        ca, sa = math.cos(ayaw), math.sin(ayaw)
        wx = float(ap[0]) + ca * x_l - sa * y_l
        wy = float(ap[1]) + sa * x_l + ca * y_l
        if quat is None:
            quat = (1.0, 0.0, 0.0, 0.0)
        place_body(body, wx, wy, z, quat=quat, settle_steps=settle_steps)

    def rod_axis_quat(ayaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qy(90): rod local +z along assembly local +x."""
        cy2, sy2 = math.cos(ayaw / 2), math.sin(ayaw / 2)
        c45 = math.cos(math.pi / 4)
        return (cy2 * c45, -sy2 * c45, cy2 * c45, sy2 * c45)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    bl = scene.ball_local()[0]
    rod_z = float((scene.rod.data.root_pos_w - scene.env_origins)[0, 2])
    fin0 = bool(torch.isfinite(scene.rod.data.root_state_w).all()
                and torch.isfinite(scene.ball.data.root_state_w).all())
    check("settle: states finite; ball readback INSIDE the bore at its spawn depth; "
          "rod at rack grasp height",
          fin0 and c.ball_depth_range[0] - 0.01 <= float(bl[0]) <= c.ball_depth_range[1] + 0.01
          and abs(float(bl[1])) < c.bore_half and abs(float(bl[2]) - (c.bore_floor_z + c.ball_r)) < 0.006
          and abs(rod_z - c.rod_rest_z) < 0.008 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        r = (scene.rod.data.root_pos_w - scene.env_origins)[0]
        ax_w = scene._rod_axis_w()[0]
        rod_yaw = math.atan2(float(ax_w[1]), float(ax_w[0]))
        ap, ayaw = asm_pose()
        reads.append((float(r[0]), float(r[1]), rod_yaw, float(ap[0]), float(ap[1]), ayaw,
                      float(scene.s0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rod_x, rod_y, rod_yaw, asm_x, asm_y, asm_yaw, "
          f"ball_s0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rod spawn xy + axis yaw vary across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.2)
    check("randomization: assembly xy + yaw AND ball bore-depth vary across seeded "
          "resets (readback)",
          spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.03 and spread[6] > 0.005)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy: rod partway in, nothing else =============
    # The seed's whole plan ends at insertion depth: peg in the hole = done. Constructed
    # here: rod inserted partway into the bore (tip short of the ball), ball unmoved.
    # The rubric must refuse it: no success, credit capped at the small instrumental-
    # insertion share.
    torch.manual_seed(41)
    env.reset()
    step(10)
    ap, ayaw = asm_pose()
    tip_probe = 0.045  # inside the bore, short of the shallowest ball spawn (0.055+)
    place_local(scene.rod, tip_probe - c.rod_len / 2, 0.0, c.bore_center_z - 0.004,
                quat=rod_axis_quat(ayaw), settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    bl = scene.ball_local()[0]
    ball_moved = abs(float(bl[0]) - float(scene.s0[0]))
    check("seed strategy (rod partway into the hole, nothing else moved): NOT success, "
          "ball unmoved, score <= 0.45",
          not ok and ball_moved < 0.01 and float(scene.insert_latch[0]) > 0.10 and s <= 0.45)

    # =========================== 7. near-miss push (ball just short of the muzzle) ==========
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_local(scene.ball, c.tube_len - 0.012, 0.0, c.bore_floor_z + c.ball_r + 0.0005,
                settle_steps=90)
    report("near-miss")
    s_near, ok = judge()
    push_near = float(scene.push_latch[0])
    bl = scene.ball_local()[0]
    check("near-miss: ball just short of the muzzle, still supported in the bore — "
          "NOT success, score < 0.9",
          not ok and float(bl[0]) < c.tube_len and float(bl[2]) > c.bore_floor_z
          and push_near > 0.5 and s_near < 0.9)

    # =========================== 8. wrong-end / bad landing =================================
    # A backwards ejection (ramming from the muzzle side) or a bad landing leaves the
    # yellow ball on the TABLE outside the basin: real displacement, wrong place.
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_local(scene.ball, -0.10, 0.0, c.ball_r + 0.002, settle_steps=90)
    report("wrong-end")
    s, ok = judge()
    bl = scene.ball_local()[0]
    check("wrong-end ejection: yellow ball settled on the table outside the basin — "
          "NOT success, no containment credit",
          not ok and not bool(scene.ball_in_basin()[0]) and float(bl[2]) < 0.05 and s <= 0.20)

    # =========================== 9. wrong object ============================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_local(scene.decoy, (c.basin_in_x0 + c.basin_in_x1) / 2, 0.0,
                c.basin_floor_t + c.ball_r + 0.002, settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    dec_loc = scene._to_local(scene.decoy.data.root_pos_w)[0]
    check("wrong object: the RED decoy settled inside the basin scores ~0, no success "
          "(identity matters)",
          bool(scene._in_basin(dec_loc.unsqueeze(0))[0]) and s <= 0.05 and not ok)

    # =========================== 10. monotonicity ===========================================
    torch.manual_seed(51)  # same seed as the near-miss -> same s0
    env.reset()
    step(10)
    place_local(scene.ball, 0.125, 0.0, c.bore_floor_z + c.ball_r + 0.0005, settle_steps=90)
    report("shallow-push")
    judge()
    push_shallow = float(scene.push_latch[0])
    check("monotonicity: deeper ball push latched strictly more transmission credit "
          f"({push_shallow:.3f} < {push_near:.3f})", push_shallow + 0.05 < push_near)

    # =========================== 11. latched credit survives regression =====================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_local(scene.ball, c.tube_len - 0.012, 0.0, c.bore_floor_z + c.ball_r + 0.0005,
                settle_steps=90)
    s_in, _ = judge()
    place_local(scene.ball, 0.065, 0.0, c.bore_floor_z + c.ball_r + 0.0005, settle_steps=60)
    report("regressed")
    s_out, ok = judge()
    check("latched credit: moving the ball back toward the entry leaves the latched "
          "score unchanged",
          abs(s_out - s_in) < 1e-3 and not ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.rod.data.root_state_w).all()
           and torch.isfinite(scene.ball.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.assembly.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ram_eject")
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
