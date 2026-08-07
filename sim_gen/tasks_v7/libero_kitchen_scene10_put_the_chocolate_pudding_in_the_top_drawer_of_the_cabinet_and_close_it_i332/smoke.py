"""Smoke / rubric-REJECTION battery for DrawbridgeVaultScene (sim_gen task
`libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i332`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — pudding pushed up the lowered drawbridge ramp and
through the doorway by contact forces, then the bridge torqued past vertical and
dropped shut by gravity — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bridge resting LOWERED on
                            the table (~-20 deg ramp), both boxes standing on the
                            table, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the brown/white slot
                            assignment flips; per-box xy jitter and free yaw vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the end state the seed's plan (grasp, hover over the
                            receptacle, release from above) produces here: pudding
                            dropped from above the chamber -> lands on the ROOF,
                            rejected (the chamber has no top opening);
  7.  doorway near-miss   — pudding THROUGH the doorway resting on the chamber
                            floor but centre just short of the deep-inside band ->
                            rejected (not deep enough);
  8.  inside-but-open     — pudding settled deep inside but the bridge still
                            lowered -> inside credit only, no success (the seed's
                            "place but never close" end state);
  9.  empty-close gating  — bridge dropped shut on its stop with NOTHING inside ->
                            close credit is GATED on the inside latch: score stays
                            ~0, no success;
  10. bistable near-miss  — bridge released at 85 deg (just below vertical) with
                            the pudding inside -> it falls back OPEN: a settled
                            just-short-of-closed state is physically unreachable,
                            and an under-rotated bridge is rejected;
  11. order violation     — the out-of-order plan: pudding left straddling the
                            doorway, then the solve's own capped closing torque
                            applied -> no success (the box obstructs the swing /
                            ends trapped short of deep-inside);
  12. wrong object        — WHITE decoy deep inside + bridge shut, pudding still
                            on the table -> rejected;
  13. latched credit      — inside credit earned, then the pudding yanked back out
                            to the table -> the latched score holds (credit does
                            not evaporate), still no success;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i332.smoke --headless
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
    env = ENVS.get("simgen.drawbridge_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    cy = c.keep_pos[1]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -1.05, 0.85)) + o),
                                tuple(np.array((0.52, 0.00, 0.15)) + o),
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p, d = loc(scene.pudding), loc(scene.decoy)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | pud=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) dec=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):.3f}) bridge={float(scene.bridge_deg()[0]):+.1f}deg "
              f"shut={bool(scene.bridge_shut()[0])} "
              f"in={bool(scene.box_inside(scene.pudding)[0])} "
              f"lat=[a{float(scene._appr_max[0]):.2f} i{int(scene._inside[0])} "
              f"c{float(scene._close_max[0]):.2f} s{int(scene._closed[0])}] "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        half = math.radians(yaw_deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 6] = math.cos(half), math.sin(half)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def pose_bridge(deg: float) -> None:
        """Pose-write the bridge (follower-only, about its unchanged hinge line)."""
        half = math.radians(deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_x, cy, c.hinge_z
        st[:, 3], st[:, 5] = math.cos(half), math.sin(half)
        st[:, 0:3] += scene.env_origins
        scene.bridge.write_root_state_to_sim(st, all_ids)

    def states_finite() -> bool:
        return bool(torch.isfinite(scene.pudding.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all()
                    and torch.isfinite(scene.bridge.data.root_state_w).all()
                    and torch.isfinite(scene.keep.data.root_state_w).all())

    def all_still() -> bool:
        return (float(scene.pudding.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed
                and float(scene.bridge.data.root_ang_vel_w[0].norm()) < c.settle_omega)

    floor_box_z = c.floor_z + c.edge / 2 + 0.002   # box resting on the chamber floor
    deep_x = (c.inside_x_lo + c.inside_x_hi) / 2   # deep-inside band centre

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    p, d = loc(scene.pudding), loc(scene.decoy)
    a0 = float(scene.bridge_deg()[0])
    check("settle: states finite, bridge resting lowered on the table (-24..-15 deg), "
          "both boxes standing on the table, everything still",
          states_finite() and -24.0 < a0 < -15.0
          and float(p[2]) < 0.05 and float(d[2]) < 0.05 and all_still())
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        p, d = loc(scene.pudding), loc(scene.decoy)
        q = scene.pudding.data.root_quat_w[0]
        yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        reads.append((float(p[0]), float(p[1]), yaw, float(d[0]), float(d[1]),
                      1.0 if float(p[1]) > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pud_x, pud_y, pud_yaw_deg, dec_x, dec_y, "
          f"pud_side_pos_y):\n{arr}", flush=True)
    check("randomization: brown/white slot assignment flips across seeded resets "
          "(readback)", 0.0 < arr[:, 5].mean() < 1.0)
    check("randomization: per-box xy jitter and free yaw vary across seeded resets "
          "(readback)",
          float(arr[:, 0].max() - arr[:, 0].min()) > 0.01
          and float(arr[:, 3].max() - arr[:, 3].min()) > 0.01
          and float(arr[:, 2].max() - arr[:, 2].min()) > 20.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: release from above =======================
    # The seed's plan — grasp, hover over the receptacle, release from above — dropped
    # over this chamber lands on the ROOF: the chamber has no top opening.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.pudding, c.keep_pos[0] + 0.02, cy, c.roof_top_z + c.edge / 2 + 0.10)
    step(120)
    report("seed-strategy")
    s, ok = judge()
    pz = float(loc(scene.pudding)[2])
    check("seed strategy: pudding dropped from above the chamber lands ON THE ROOF — "
          "not inside, no success, score <= 0.12",
          pz > c.roof_top_z - 0.02 and not bool(scene.box_inside(scene.pudding)[0])
          and not bool(scene._inside[0]) and not ok and s <= 0.12)

    # =========================== 7. doorway near-miss (not deep enough) =====================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place(scene.pudding, c.inside_x_lo - 0.030, cy, floor_box_z)
    step(90)
    report("shallow")
    s, ok = judge()
    p = loc(scene.pudding)
    check("doorway near-miss: pudding THROUGH the doorway resting on the chamber floor "
          "but centre just short of the deep-inside band — rejected, score <= 0.12",
          float(p[2]) > c.floor_z and float(p[0]) < c.inside_x_lo
          and not bool(scene.box_inside(scene.pudding)[0])
          and not bool(scene._inside[0]) and not ok and s <= 0.12)

    # =========================== 8. inside but the bridge never closed ======================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place(scene.pudding, deep_x, cy, floor_box_z)
    step(90)
    report("inside-open")
    s, ok = judge()
    check("inside-but-open: pudding settled deep inside but the bridge still lowered — "
          "inside credit only (0.30 <= score <= 0.45), no success",
          bool(scene.box_inside(scene.pudding)[0]) and bool(scene._inside[0])
          and float(scene.bridge_deg()[0]) < 0.0 and not ok and 0.30 <= s <= 0.45)

    # =========================== 9. closing an EMPTY vault earns nothing ====================
    torch.manual_seed(71)
    env.reset()
    step(10)
    pose_bridge(96.0)  # past vertical: gravity carries it onto the +98 stop
    step(120)
    report("empty-shut")
    s, ok = judge()
    check("empty-close gating: bridge settled SHUT on its stop with nothing inside — "
          "close credit gated on the inside latch: close_max stays 0, score <= 0.05, "
          "no success",
          bool(scene.bridge_shut()[0]) and float(scene._close_max[0]) < 1e-6
          and not ok and s <= 0.05)

    # =========================== 10. bistable: under-rotated bridge falls back open =========
    torch.manual_seed(81)
    env.reset()
    step(10)
    place(scene.pudding, deep_x, cy, floor_box_z)
    step(90)
    pose_bridge(85.0)  # just BELOW vertical: gravity must reject it back to lowered
    step(300)
    report("under-rotated")
    s, ok = judge()
    check("bistable near-miss: bridge released at 85 deg (below vertical) with the "
          "pudding inside — falls back OPEN (settled just-short-of-closed is "
          "physically unreachable), never shut, no success",
          float(scene.bridge_deg()[0]) < 45.0 and not bool(scene.bridge_shut()[0])
          and not bool(scene._closed[0]) and bool(scene.box_inside(scene.pudding)[0])
          and not ok)

    # =========================== 11. order violation: close onto a doorway box ==============
    # The out-of-order plan: the box is left straddling the doorway (protruding past
    # the hinge plane) and the solver tries to close anyway, with the solve's own
    # capped torque (0.8 N*m). Physics must refuse the combination: either the swing
    # stalls short of the closed band, or the plate wedges the box forward but leaves
    # it trapped short of deep-inside. Both ways: no success.
    torch.manual_seed(91)
    env.reset()
    step(10)
    place(scene.pudding, c.hinge_x + 0.015, cy, floor_box_z)
    step(60)
    report("doorway-box")
    for _ in range(700):
        a = float(scene.bridge_deg()[0])
        if a >= 93.0:
            break
        w_tgt = 1.2 if a < 70.0 else 0.5
        w = float(scene.bridge.data.root_ang_vel_w[0, 1])
        tq = max(0.0, min(0.8, 2.5 * (w_tgt - w)))
        t_w = torch.tensor([0.0, tq, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.bridge.set_external_force_and_torque(zero_wrench, t_w.contiguous(),
                                                   env_ids=all_ids, is_global=True)
        step(1)
    scene.bridge.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(240)
    report("order-violate")
    s, ok = judge()
    p = loc(scene.pudding)
    print(f"[smoke] order violation outcome: bridge={float(scene.bridge_deg()[0]):+.1f} "
          f"deg, pudding_x={float(p[0]):+.3f} (deep band starts {c.inside_x_lo:.3f})",
          flush=True)
    check("order violation: closing torque applied with the pudding straddling the "
          "doorway — no success (bridge stalled short of shut, or the box left "
          "trapped short of the deep-inside band)",
          not ok and not (bool(scene.bridge_shut()[0])
                          and bool(scene.box_inside(scene.pudding)[0])))

    # =========================== 12. wrong object inside ====================================
    torch.manual_seed(101)
    env.reset()
    step(10)
    place(scene.decoy, deep_x, cy, floor_box_z)
    step(90)
    pose_bridge(96.0)
    step(120)
    report("wrong-box")
    s, ok = judge()
    check("wrong object: WHITE decoy deep inside + bridge shut, BROWN pudding still on "
          "the table — rejected, score <= 0.12",
          bool(scene.box_inside(scene.decoy)[0]) and bool(scene.bridge_shut()[0])
          and not bool(scene.box_inside(scene.pudding)[0]) and not ok and s <= 0.12)

    # =========================== 13. latched credit survives regression =====================
    torch.manual_seed(111)
    env.reset()
    step(10)
    place(scene.pudding, deep_x, cy, floor_box_z)
    step(90)
    s_a, ok_a = judge()
    place(scene.pudding, 0.30, 0.25, c.edge / 2 + 0.002)  # yank it back out
    step(90)
    report("regressed")
    s_b, ok_b = judge()
    check("latched credit: inside credit earned then the pudding yanked back to the "
          f"table — score holds ({s_a:.3f} -> {s_b:.3f}), >= 0.30, still no success",
          bool(scene._inside[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.30
          and not ok_a and not ok_b)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", states_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drawbridge_vault")
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
