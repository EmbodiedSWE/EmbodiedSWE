"""Smoke / rubric-REJECTION battery for BowlLetterboxScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i212`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lift, reorient on edge, stage, force-push through
the slot — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
as a settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bowl FLAT on the floor in
                            front of the cabinet, bottle upright, all still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the bowl's LEFT/RIGHT side
                            flips (Bernoulli), the cabinet yaw varies, and per-slot
                            xy jitter is real (measured in the cabinet frame);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed-pose physical  — the seed task's carry pose (bowl FLAT) pressed at the
                            lower slot with the bounded push drive: the bowl ADVANCES,
                            JAMS against the front wall (130 mm dish vs 70 mm slot),
                            never enters — NOT success, insertion latch ~0 (the
                            reorientation requirement is load-bearing, physically);
  7.  top-cell decoy      — bowl teleport-settled INSIDE the upper compartment (the
                            dark-framed decoy slot's cell): wrong floor -> NOT
                            success, zero insertion credit (z-gate);
  8.  straddle near-miss  — on-edge bowl left resting HALF-THROUGH the lower slot
                            (sticking out): NOT success, insertion credit strictly
                            partial, score <= 0.85;
  9.  latched credit      — teleporting that straddling bowl back out to the floor
                            leaves the latched score unchanged, still no success;
  10. roof                — bowl lying flat ON TOP of the cabinet -> score ~0;
  11. wrong object        — the fat RED BOTTLE parked on the apron against the slot,
                            bowl untouched -> score ~0, no success;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_letterbox")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -0.95, 0.75)) + o),
                                tuple(np.array((0.45, 0.00, 0.20)) + o),
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

    def loc0() -> list:
        return [float(v) for v in scene.bowl_local()[0]]

    def axz() -> float:
        return float(scene.bowl_axis_z()[0])

    def cab_yaw_deg() -> float:
        q = scene.cabinet.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def report(tag: str) -> None:
        lx, ly, lz = loc0()
        s, ok = judge()
        print(f"[smoke] {tag:16s} | loc=({lx:+.3f},{ly:+.3f},{lz:.3f}) axz={axz():+.2f} "
              f"near={float(scene._near_max[0]):.3f} posed={bool(scene._posed[0])} "
              f"ins={float(scene._ins_max[0]):.3f} in0={bool(scene.bowl_in_cell(0)[0])} "
              f"in1={bool(scene.bowl_in_cell(1)[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, lz: float, quat_extra=None) -> None:
        """Teleport a body to a CABINET-frame pose (probe constructor)."""
        cp = scene.cabinet.data.root_pos_w[0]
        cq = scene.cabinet.data.root_quat_w[0]
        lp = torch.tensor([lx, ly, lz], device=device, dtype=torch.float32)
        pw = cp + quat_apply(cq.unsqueeze(0), lp.unsqueeze(0))[0]
        q = cq if quat_extra is None else quat_mul(cq.unsqueeze(0),
                                                   quat_extra.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pw
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)

    q_edge_local = quat_from_angle_axis(
        torch.tensor([math.pi / 2], device=device),
        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]  # on-edge, plane = cab x-z

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    step(90)
    report("reset-settle")
    fin0 = (torch.isfinite(scene.cabinet.data.root_state_w).all()
            and torch.isfinite(scene.bowl.data.root_state_w).all()
            and torch.isfinite(scene.bottle.data.root_state_w).all())
    lx, ly, lz = loc0()
    still = (bool(scene.bowl_still()[0])
             and float(scene.bottle.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, bowl FLAT on the floor in front of the cabinet, "
          "bottle upright, all still",
          bool(fin0) and lx < -0.25 and abs(axz()) > 0.9
          and abs(lz - c.bowl_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        blx, bly, _ = loc0()
        reads.append((blx, bly, cab_yaw_deg(), 1.0 if bly > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bowl_lx, bowl_ly, cab_yaw_deg, on_left):\n{arr}",
          flush=True)
    flags = arr[:, 3]
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    check("randomization: bowl side flips across seeded resets AND cabinet yaw varies "
          f"(spread {yaw_spread:.1f} deg > 3)",
          0.0 < flags.mean() < 1.0 and yaw_spread > 3.0
          and float(np.abs(arr[:, 2]).max()) <= c.yaw_max_deg + 0.5)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    check("randomization: per-slot spawn jitter is real (cabinet-frame readback spread "
          "> 4 mm within a side group)", jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-pose physical probe ================================
    # The seed task's carry pose — the bowl FLAT (as it would be lowered into a drawer)
    # — pushed at the lower slot with the same bounded drive the solve uses. The 130 mm
    # dish jams against the 70 mm slot's front wall and never enters: the reorientation
    # requirement is load-bearing PHYSICALLY, not just in the rubric. The probe asserts
    # the actuator really acted (the bowl advanced to wall contact) so it is not vacuous.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.bowl, -0.16, 0.0, c.floor_z + c.bowl_h / 2 + 0.004)
    step(30)
    x_start = loc0()[0]
    x_max = x_start
    for _ in range(420):
        x_now = loc0()[0]
        x_max = max(x_max, x_now)
        # gentle position servo toward the wall, capped 4 N (quasi-static: no vaulting)
        scene.push_drive[:] = max(0.0, min(4.0, 20.0 * (-0.02 - x_now)))
        env.step(no_action)
    scene.push_drive[:] = 0.0
    step(120)
    report("flat-jam")
    s6, ok = judge()
    lx6 = loc0()[0]
    check("seed-pose physical probe: FLAT bowl driven at the lower slot ADVANCES "
          f"(start {x_start:+.3f} -> max {x_max:+.3f}), jams at the front wall, never "
          "enters — NOT success, insertion latch ~0, score <= 0.5",
          x_max > x_start + 0.02 and x_max > -0.10 and x_max < 0.0 and lx6 < 0.0
          and abs(axz()) > 0.9 and not bool(scene.bowl_in_cell(0)[0])
          and float(scene._ins_max[0]) <= 0.2 and not ok and s6 <= 0.5)

    # =========================== 7. top-cell decoy ==========================================
    # The upper compartment behind the DARK-framed slot: same slot, wrong floor. The
    # bowl teleport-settled flat inside it earns no insertion credit (z-gate) and no
    # success.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_local(scene.bowl, 0.15, 0.0, c.cell1_z[0] + c.bowl_h / 2 + 0.004)
    step(150)
    report("top-cell")
    s7, ok = judge()
    check("top-cell decoy: bowl settled flat INSIDE the upper compartment — in the top "
          "cell, NOT the bottom cell, zero insertion credit, NOT success, score <= 0.85",
          bool(scene.bowl_in_cell(1)[0]) and not bool(scene.bowl_in_cell(0)[0])
          and float(scene._ins_max[0]) <= 0.01 and not ok and s7 <= 0.85)

    # =========================== 8. straddle near-miss ======================================
    # On-edge bowl left resting half-through the lower slot: partially inside, sticking
    # out the front. Insertion credit is strictly partial; success is out of reach.
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_local(scene.bowl, 0.02, 0.0, c.floor_z + c.bowl_out_r + 0.003,
                quat_extra=q_edge_local)
    step(180)
    report("straddle")
    s8, ok = judge()
    lx8 = loc0()[0]
    check("straddle near-miss: on-edge bowl resting HALF-THROUGH the lower slot "
          f"(centre x {lx8:+.3f} < containment cut {c.succ_x[0]:+.3f}) — NOT success, "
          "insertion credit strictly partial (0.1..0.9), score <= 0.85",
          -0.04 < lx8 < c.succ_x[0] and abs(axz()) < 0.35
          and not bool(scene.bowl_in_cell(0)[0])
          and 0.1 <= float(scene._ins_max[0]) <= 0.9 and not ok and s8 <= 0.85)

    # =========================== 9. latched credit survives regression ======================
    place_local(scene.bowl, -0.55, 0.0, c.bowl_h / 2 + 0.004)  # open floor, clear of both spawn slots
    step(60)
    report("regressed")
    s9, ok = judge()
    check("latched credit: teleporting the straddling bowl back out to the floor leaves "
          f"the latched score unchanged ({s8:.3f} -> {s9:.3f}), still no success",
          abs(s9 - s8) < 1e-3 and not ok)

    # =========================== 10. roof ===================================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.bowl, 0.15, 0.0, c.cell1_z[1] + c.t + c.bowl_h / 2 + 0.004)
    step(150)
    report("roof")
    s10, ok = judge()
    check("roof: bowl lying flat ON TOP of the cabinet — outside every compartment, "
          "score ~0 (<= 0.02), no success",
          not bool(scene.bowl_in_cell(0)[0]) and not bool(scene.bowl_in_cell(1)[0])
          and loc0()[2] > c.cell1_z[1] + c.t - 0.01 and not ok and s10 <= 0.02)

    # =========================== 11. wrong object ===========================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_local(scene.bottle, -0.13, 0.0, c.floor_z + c.bottle_h / 2 + 0.004)
    step(150)
    report("bottle-apron")
    s11, ok = judge()
    fin_b = torch.isfinite(scene.bottle.data.root_state_w).all()
    check("wrong object: the RED BOTTLE parked on the apron against the slot, bowl "
          "untouched — rubric unmoved: score ~0 (<= 0.02), no success",
          bool(fin_b) and not ok and s11 <= 0.02)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.cabinet.data.root_state_w).all()
           and torch.isfinite(scene.bowl.data.root_state_w).all()
           and torch.isfinite(scene.bottle.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bowl_letterbox")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
