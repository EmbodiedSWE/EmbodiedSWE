"""Smoke / rubric-REJECTION battery for BallastRockerScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i176`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — ballast dropped into the socket, then the bowl
dropped onto the ballasted zone, both under gravity — is the acceptance evidence that
the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that. Probe order is chosen so no prefix of any episode satisfies the
goal.

  1-2. settle/no-NaN      — reset layout settles finite: empty tray resting on its
                            +12 deg stop, ballast and bowl standing on the floor,
                            score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: ballast/bowl spawn slots
                            swap and xy-jitter; free yaw varies;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (carry the bowl to the top and set it
                            down) executed as found (no ballast): the bowl's weight
                            out-torques the tray bias, the tray pivots to its -35 deg
                            stop and DUMPS the bowl off the raised end -> no success;
  7.  trap aftermath      — the emptied tray rocks back and re-settles on its rest
                            stop (the trap resets itself; still no success);
  8.  out-of-order        — ballast seated AFTER the dump: interlock established but
                            the bowl lies on the floor -> no success, score <= 0.50;
  9.  swapped roles       — fresh episode, the BALLAST dropped onto the green zone:
                            it tips the tray and is dumped itself -> no success;
  10. near-miss socket    — ballast resting ON the tray, socket side, but OUTSIDE the
                            socket's xy tolerance; the bowl then settles in the zone
                            (this off-spec stack happens to hold) -> the interlock
                            stage is unmet: no success, score <= 0.30;
  11. near-miss zone      — ballast properly seated, bowl upright ON the tray but
                            inboard of the cleat, outside the zone band -> no success;
  12. wrong orientation   — bowl UPSIDE-DOWN on the zone of the ballasted tray -> the
                            upright clause rejects -> no success;
  13. latched credit      — the bowl removed back to the floor: latched seat/carry
                            credit does not evaporate (score unchanged), no success;
  14. hover fly-through   — bowl held in the air above the zone (inside xy, above the
                            z band), judged WITHOUT stepping -> transport through the
                            air never scores: no success;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i176.smoke --headless
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
    env = ENVS.get("simgen.ballast_rocker")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    half = math.radians(c.rest_deg) / 2
    rest_quat = (math.cos(half), math.sin(half), 0.0, 0.0)
    hflip = math.radians(c.rest_deg + 180.0) / 2
    flip_quat = (math.cos(hflip), math.sin(hflip), 0.0, 0.0)  # upside-down at tray tilt

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.15, 1.05)) + o),
                                tuple(np.array((0.50, 0.00, 0.30)) + o),
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
        p_bal = (scene.ballast.data.root_pos_w - scene.env_origins)[0]
        p_bowl = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bal=({float(p_bal[0]):+.3f},{float(p_bal[1]):+.3f},"
              f"{float(p_bal[2]):.3f}) bowl=({float(p_bowl[0]):+.3f},"
              f"{float(p_bowl[1]):+.3f},{float(p_bowl[2]):.3f}) "
              f"tray={float(scene.tray_deg()[0]):+.1f}deg "
              f"insock={bool(scene.ballast_in_socket()[0])} "
              f"seated={bool(scene.ballast_seated()[0])} "
              f"inzone={bool(scene.bowl_in_zone()[0])} "
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

    def rest_place(body, y_local: float, z_local: float, quat=None) -> None:
        """Place a body at tray-frame (0, y_local, z_local) mapped at the REST angle."""
        x, y, z = c.rest_point(y_local, z_local)
        place(body, x, y, z, quat=quat or rest_quat)

    def settle(body, max_steps: int = 400, need: int = 30) -> None:
        quiet = 0
        for _ in range(max_steps):
            step(1)
            ok_now = bool(scene._still(body)[0]) and bool(scene._tray_still()[0])
            quiet = quiet + 1 if ok_now else 0
            if quiet >= need:
                break

    def seat_ballast() -> None:
        """The legitimate interlock move: hover above the socket, gravity drop, settle."""
        rest_place(scene.ballast, c.sock_y, 0.075)
        settle(scene.ballast)

    def obj_z(body) -> float:
        return float((body.data.root_pos_w - scene.env_origins)[0, 2])

    def bal_xy() -> tuple[float, float]:
        p = (scene.ballast.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.ballast.data.root_state_w).all()
            and torch.isfinite(scene.bowl.data.root_state_w).all()
            and torch.isfinite(scene.tray.data.root_state_w).all())
    still = bool(scene._still(scene.ballast)[0] and scene._still(scene.bowl)[0]
                 and scene._tray_still()[0])
    check("settle: states finite, empty tray resting on its +12 deg stop, ballast and "
          "bowl standing on the floor, everything still",
          bool(fin0) and bool(scene.tray_at_rest()[0]) and obj_z(scene.ballast) < 0.05
          and obj_z(scene.bowl) < 0.03 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)  # readback right after the reset writes
        bx_, by_ = bal_xy()
        q = scene.ballast.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        reads.append((bx_, by_, yaw, 1.0 if by_ > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bal_x, bal_y, bal_yaw, bal_left):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: ballast/bowl spawn-slot swap and xy jitter are real across "
          "seeded resets (readback: swap mean strictly inside (0,1), x spread > 1.5 cm)",
          0.0 < arr[:, 3].mean() < 1.0 and spread[0] > 0.015)
    check("randomization: free yaw varies across seeds (readback spread > 0.5 rad)",
          spread[2] > 0.5)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps (the tray already "
          "rests on its stop — no stage is free)", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: bowl straight to the top =================
    # The seed's plan verbatim: carry the black bowl to the cabinet top and set it
    # down — WITHOUT the ballast. The bowl out-torques the tray bias 2:1, the tray
    # pivots to its -35 deg stop (past the friction angle) and dumps the bowl.
    torch.manual_seed(41)
    env.reset()
    step(10)
    rest_place(scene.bowl, c.zone_y, 0.060)
    min_tray = 90.0
    for _ in range(360):
        step(1)
        min_tray = min(min_tray, float(scene.tray_deg()[0]))
        judge()  # rubric judged mid-dump: transit through the zone must earn nothing
    step(240)  # let the emptied tray rock back and everything settle
    report("seed-strategy")
    s6, ok = judge()
    check("seed strategy: bowl set down on the unballasted tray — the tray tips past "
          f"-20 deg (min {min_tray:.1f} deg) and DUMPS the bowl off the raised end: "
          "bowl off the tray, no success, score <= 0.25",
          min_tray <= -20.0 and obj_z(scene.bowl) < 0.30
          and not bool(scene.bowl_in_zone()[0]) and not ok and s6 <= 0.25)

    # =========================== 7. trap aftermath: the tray rocks back =====================
    check("trap aftermath: the emptied tray rocked back and re-settled on its rest "
          "stop (the trap resets itself; still no success)",
          bool(scene.tray_at_rest()[0]) and bool(scene._tray_still()[0]) and not ok)

    # =========================== 8. out-of-order end state ==================================
    # Same episode: NOW seat the ballast (legitimate interlock move). The interlock
    # alone — with the bowl lying on the floor — is only partial credit.
    seat_ballast()
    report("out-of-order")
    s8, ok = judge()
    check("out-of-order: ballast seated AFTER the dump — interlock established but "
          "the bowl lies on the floor: no success, score <= 0.50",
          bool(scene.ballast_seated()[0]) and obj_z(scene.bowl) < 0.10
          and not ok and s8 <= 0.50)

    # =========================== 9. swapped roles: ballast onto the zone ====================
    torch.manual_seed(51)
    env.reset()
    step(10)
    rest_place(scene.ballast, c.zone_y, 0.075)
    min_tray = 90.0
    for _ in range(360):
        step(1)
        min_tray = min(min_tray, float(scene.tray_deg()[0]))
        judge()
    step(240)
    report("swapped-roles")
    s9, ok = judge()
    check("swapped roles: the BALLAST dropped onto the green zone tips the tray "
          f"(min {min_tray:.1f} deg) and is dumped itself — never in the socket, "
          "no success",
          min_tray <= -20.0 and not bool(scene.ballast_in_socket()[0])
          and obj_z(scene.ballast) < 0.30 and not ok)

    # =========================== 10. near-miss: ballast outside the socket ==================
    # Ballast dropped onto the tray's socket SIDE but centred 10 cm from the socket
    # centre (outside sock_xy_tol 3 cm), on the bare plate inboard of the fences. Its
    # torque happens to hold the tray when the bowl lands — but the interlock stage
    # demands the ballast INSIDE the socket, so this off-spec stack earns no seat
    # credit and no success.
    torch.manual_seed(61)
    env.reset()
    step(10)
    rest_place(scene.ballast, -0.05, 0.075)
    settle(scene.ballast)
    report("bal-off-socket")
    rest_place(scene.bowl, c.zone_y, 0.060)
    settle(scene.bowl)
    report("off-spec-stack")
    s10, ok = judge()
    check("near-miss socket: ballast resting ON the tray but outside the socket's xy "
          "tolerance, bowl settled in the zone on this off-spec stack — the interlock "
          "stage is unmet: no success, score <= 0.30",
          not bool(scene.ballast_in_socket()[0]) and bool(scene.tray_at_rest()[0])
          and obj_z(scene.ballast) > 0.30 and not ok and s10 <= 0.30)

    # =========================== 11. near-miss: bowl inboard of the cleat ===================
    torch.manual_seed(71)
    env.reset()
    step(10)
    seat_ballast()
    rest_place(scene.bowl, 0.015, 0.060)
    settle(scene.bowl)
    report("bowl-inboard")
    s11, ok = judge()
    p_loc = scene._tray_local(scene.bowl)[0]
    check("near-miss zone: ballast seated, bowl upright ON the tray but inboard of "
          "the cleat, outside the zone band — no success",
          bool(scene.ballast_seated()[0]) and float(p_loc[2]) < 0.04
          and abs(float(p_loc[1]) - c.zone_y) > c.zone_y_tol
          and bool(scene._upright(scene.bowl)[0]) and not ok)

    # =========================== 12. wrong orientation: upside-down =========================
    rest_place(scene.bowl, c.zone_y, 0.080, quat=flip_quat)
    settle(scene.bowl)
    report("upside-down")
    s12, ok = judge()
    q = scene.bowl.data.root_quat_w[0]
    r33 = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    check("wrong orientation: bowl UPSIDE-DOWN on the zone of the ballasted tray — "
          "the upright clause rejects: no success",
          bool(scene.ballast_seated()[0]) and r33 < 0.0 and not ok)

    # =========================== 13. latched credit survives regression =====================
    place(scene.bowl, 0.30, 0.25, 0.003)
    step(60)
    report("regressed")
    s13, ok = judge()
    check("latched credit: bowl removed back to the floor — the latched seat/carry "
          f"credit does not evaporate ({s12:.3f} -> {s13:.3f}), still no success",
          abs(s13 - s12) < 1e-3 and s13 >= c.w_seat - 0.01 and not ok)

    # =========================== 14. hover fly-through never scores =========================
    rest_place(scene.bowl, c.zone_y, 0.060)
    s14, ok = judge()  # judged WITHOUT stepping: pure transport pose
    check("hover fly-through: bowl held in the air above the zone (inside xy, above "
          "the z band), judged without stepping — no success",
          not bool(scene.bowl_in_zone()[0]) and not ok)
    place(scene.bowl, 0.30, 0.25, 0.003)
    step(30)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.ballast.data.root_state_w).all()
           and torch.isfinite(scene.bowl.data.root_state_w).all()
           and torch.isfinite(scene.tray.data.root_state_w).all()
           and torch.isfinite(scene.cabinet.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_rocker")
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
