"""Smoke / rubric-REJECTION battery for SwingPumpScene (sim_gen task
`stack_cube_i382`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — resonant pumping through the access window, then a
controlled ballistic release past the check flap — is the acceptance evidence that the
rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1.   settle/no-NaN      — reset layout settles finite: flap CLOSED (angle readback),
                            pendulum near its random hang angle, bob EXPOSED, score ~0;
  2-3. randomization      — READBACK over 6 seeded resets: apparatus xy + yaw (circular
                            pairwise) and the initial pendulum angle all move;
  4.   null policy        — 400 idle steps: pendulum still in the small-swing band, flap
                            closed, no latches, score ~0;
  5.   exposure window    — the access restriction is REAL in readback: bob exposed at
                            the hang, NOT exposed at 80 deg (inside the plate sandwich);
  6.   SEED strategy      — the seed's whole plan is a static precise PLACEMENT of the
                            red cube: the pendulum is written directly into the pocket
                            at rest — physically in-pocket, still, flap closed, yet
                            REFUSED (trajectory latches unearned), score ~0;
  7.   one-way gate       — from that resting-on-the-blade state, the declared 2 N cap
                            pushed back toward the window for 3+ s: the bob verifiably
                            presses the blade (readback) but CANNOT re-cross — the
                            gravity ratchet holds;
  8.   fake transit       — pendulum written into the pocket WITH the flap written wide
                            open (staging the 'mid-pass' picture): no transit credit —
                            the pass latch demands a genuine continuous crossing;
  9.   amp-only swing     — a genuine free swing to ~80 deg peak: amplitude latch earns
                            EXACTLY the 0.30 stage, no pass latch, no success;
  10.  flap near-miss     — a genuine swing short of the pass energy: the bob verifiably
                            REACHES the blade and deflects it (actuation proof, angle
                            readback) but never crosses the pass line; flap re-closes,
                            score still 0.30;
  11.  pass then extract  — a genuine high swing DOES earn the transit latch (0.60);
                            the bob is immediately pulled back out (teleport) before it
                            settles: the latched credit survives regression, NOT success;
  12.  self-reclosing     — after the pass the shoved-open flap falls SHUT on its own
                            (readback) and the latched 0.60 survives the mechanism
                            resetting;
  13.  monotonicity       — the battery's earned stages are strictly increasing:
                            null ~0 < amplitude 0.30 < transit 0.60;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.stack_cube_i382.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_pump")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.35, 1.05)) + o),
                                tuple(np.array((0.0, 0.0, 0.45)) + o),
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

    def th_deg() -> float:
        return math.degrees(float(scene.theta()[0]))

    def flap_deg() -> float:
        return math.degrees(float(scene.flap_open()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | theta={th_deg():+7.1f}deg flap={flap_deg():+6.1f}deg "
              f"exposed={bool(scene.bob_exposed()[0])} amp={bool(scene.lat_amp[0])} "
              f"pass={bool(scene.lat_pass[0])} pocket={bool(scene.in_pocket()[0])} "
              f"still={bool(scene.pend_still()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def set_pend(theta_deg: float, w: float = 0.0) -> None:
        """Kinematic probe write of the pendulum, CONSISTENT with its hinge: root at the
        hinge point, orientation frame*Ry(-theta), swing rate about the frame -Y axis
        (instrumentation, not a solution — the latches refuse teleported histories)."""
        qf = scene.frame.data.root_quat_w
        th = torch.full((n,), math.radians(theta_deg), device=device)
        v = torch.tensor([0.0, 0.0, c.hinge_h], device=device).unsqueeze(0).expand(n, 3)
        qp = quat_mul(qf, scene_mod._qy(-th))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.frame.data.root_pos_w + quat_apply(qf, v)
        st[:, 3:7] = qp
        # root-state velocities are of the CoM: give the rigid-consistent pair
        # (omega about the hinge, v_com = omega x r_com) or the hinge constraint
        # projects the written spin away
        omega = quat_apply(qf, torch.tensor([0.0, -1.0, 0.0],
                                            device=device).unsqueeze(0).expand(n, 3) * w)
        r_com = quat_apply(qp, torch.tensor([0.0, 0.0, -c.pend_com_d],
                                            device=device).unsqueeze(0).expand(n, 3))
        st[:, 7:10] = torch.cross(omega, r_com, dim=1)
        st[:, 10:13] = omega
        scene.pendulum.write_root_state_to_sim(st, all_ids)

    def set_flap(phi_deg: float) -> None:
        """Kinematic probe write of the check flap about its own hinge."""
        qf = scene.frame.data.root_quat_w
        ph = torch.full((n,), math.radians(phi_deg), device=device)
        fh = torch.tensor(c.flap_hinge_local, device=device).unsqueeze(0).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.frame.data.root_pos_w + quat_apply(qf, fh)
        st[:, 3:7] = quat_mul(qf, scene_mod._qy(ph))
        scene.flap.write_root_state_to_sim(st, all_ids)

    def w_for_peak(peak_deg: float) -> float:
        """Hinge rate at the bottom for a free-swing peak of `peak_deg` (cfg constants)."""
        return math.sqrt(2.0 * c.mgd * (1.0 - math.cos(math.radians(peak_deg)))
                         / c.i_hinge)

    def push_pend(d: float, k: int) -> float:
        """Bounded push (|F| = the declared pump_force_max cap) along the frame's
        d*(+x), body-frame encoded each step. Returns the minimum theta reached (deg)."""
        th_min = th_deg()
        for _ in range(k):
            f_frame = torch.tensor([d * c.pump_force_max, 0.0, 0.0],
                                   device=device).unsqueeze(0).expand(n, 3)
            f_world = quat_apply(scene.frame.data.root_quat_w, f_frame)
            f_body = quat_apply_inverse(scene.pendulum.data.root_quat_w, f_world)
            scene.pendulum.set_external_force_and_torque(
                f_body.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids)
            step(1)
            th_min = min(th_min, th_deg())
        scene.pendulum.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                     env_ids=all_ids)
        return th_min

    def all_finite() -> bool:
        return bool(torch.isfinite(scene.frame.data.root_state_w).all()
                    and torch.isfinite(scene.pendulum.data.root_state_w).all()
                    and torch.isfinite(scene.flap.data.root_state_w).all())

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    s, ok = judge()
    check("settle: states finite; flap CLOSED (|angle| < 3 deg readback), pendulum "
          "within its small random start band, bob EXPOSED in the window, score ~0",
          all_finite() and abs(flap_deg()) < 3.0 and abs(th_deg()) < c.theta0_range_deg + 8.0
          and bool(scene.bob_exposed()[0]) and s <= 0.02 and not ok)

    # =========================== 2-3. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(3)
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        q = scene.frame.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0]) * float(q[3]), 1 - 2 * float(q[3]) ** 2)
        reads.append((float(fp[0]), float(fp[1]), yaw, math.radians(th_deg())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (frame_x, frame_y, yaw, theta0):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    yaw_pair = max(abs(math.atan2(math.sin(a - b), math.cos(a - b)))
                   for i, a in enumerate(arr[:, 2]) for b in arr[:i, 2])
    check("randomization: apparatus xy AND yaw (max pairwise circular distance) vary "
          "across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and yaw_pair > 0.5)
    check("randomization: initial pendulum angle varies across seeded resets (readback)",
          spread[3] > 0.08)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(400)
    report("null-policy")
    s, ok = judge()
    check("null policy: after 400 idle steps the pendulum is still in the small-swing "
          "band (|theta| < 30 deg), flap closed, no latches, score ~0",
          abs(th_deg()) < 30.0 and abs(flap_deg()) < 3.0 and not bool(scene.lat_amp[0])
          and not bool(scene.lat_pass[0]) and s <= 0.02 and not ok)
    s_null = s

    # =========================== 5. exposure window readback ================================
    env.reset(seed=41)
    step(30)
    exp_hang = bool(scene.bob_exposed()[0])
    z_hang = float(scene.bob_frame_local()[0][2])
    set_pend(80.0)
    step(2)
    exp_high = bool(scene.bob_exposed()[0])
    z_high = float(scene.bob_frame_local()[0][2])
    print(f"[smoke] exposure readback: hang z={z_hang:.3f} exposed={exp_hang}; "
          f"80deg z={z_high:.3f} exposed={exp_high} (plate edge {c.window_z:.3f})",
          flush=True)
    check("exposure window: bob EXPOSED at the hang (below the plate edge, readback) and "
          "NOT exposed at 80 deg inside the plate sandwich — the access restriction the "
          "pump must live with is real",
          exp_hang and z_hang < c.window_z and not exp_high and z_high > c.window_z)

    # =========================== 6. SEED strategy: static placement =========================
    # The seed's whole plan is one precise static placement of the red cube. Constructed
    # literally here: the pendulum written directly INTO the pocket at rest — it settles
    # resting on the closed blade, physically in-pocket and still, yet earns nothing.
    env.reset(seed=51)
    step(10)
    set_pend(118.0)
    step(300)
    report("seed-placement")
    s, ok = judge()
    check("seed strategy (static placement at the goal): pendulum written into the "
          "pocket settles resting on the blade — in_pocket AND still AND flap closed by "
          "readback, yet REFUSED (no trajectory latches), score ~0",
          bool(scene.in_pocket()[0]) and bool(scene.pend_still()[0])
          and bool(scene.flap_closed()[0]) and s <= 0.02 and not ok)

    # =========================== 7. one-way gate potency ====================================
    # From that resting-on-the-blade state, push the bob BACK toward the window at the
    # declared cap for 3+ s: the ratchet must hold (the blade wedges on its stop).
    th_before = th_deg()
    th_min = push_pend(-1.0, 400)
    step(60)
    report("extraction-try")
    s, ok = judge()
    check("one-way gate: from rest on the blade the declared 2 N cap pressed back toward "
          f"the window for 3.3 s moves the bob only within the pocket (min theta "
          f"{th_min:.1f} deg stays > 98) — the gravity ratchet physically holds; still "
          "no credit",
          th_before > 98.0 and th_min > 98.0 and th_deg() > 98.0 and s <= 0.02 and not ok)

    # =========================== 8. fake transit (flap written open) ========================
    # Stage the 'mid-pass' picture: pendulum in the pocket AND flap written wide open.
    # Judged before the staged state can evolve into real dynamics: no transit credit.
    env.reset(seed=61)
    step(10)
    set_flap(60.0)
    set_pend(124.0)  # deep enough in the pocket to clear the written-open blade's tip
    step(12)
    report("fake-transit")
    s, ok = judge()
    fake_ok = (bool(scene.in_pocket()[0]) and not bool(scene.lat_pass[0])
               and not bool(scene.lat_amp[0]) and s <= 0.02 and not ok)
    set_pend(0.0)  # kill the staged energy before it swings into genuine crossings
    step(120)
    check("fake transit: pendulum written into the pocket WITH the flap written wide "
          "open — the staged mid-pass picture earns NO latches (the pass latch demands "
          "a genuine continuous crossing), score ~0",
          fake_ok)

    # =========================== 9. amplitude-only swing ====================================
    # A genuine free swing (written bottom rate, continuous thereafter) to ~80 deg peak:
    # the amplitude latch must earn EXACTLY its 0.30 stage and nothing more.
    env.reset(seed=71)
    step(30)
    set_pend(0.0, w_for_peak(80.0))
    step(300)
    report("amp-only")
    s_amp, ok = judge()
    check("amplitude-only: a genuine swing to ~80 deg peak earns the amplitude latch at "
          "exactly the 0.30 stage — no pass latch, no success",
          bool(scene.lat_amp[0]) and not bool(scene.lat_pass[0])
          and abs(s_amp - c.stage_scores[0]) < 0.02 and not ok)

    # =========================== 10. flap near-miss (actuation proof) =======================
    # A genuine swing with LESS energy than the pass line demands: the bob verifiably
    # reaches the blade and deflects it, but the pass latch stays unearned.
    env.reset(seed=81)
    step(30)
    # Earn the amplitude stage first with a SLOW 60-deg crossing (an 80-deg-peak swing,
    # like the honest pump ladder early on) — the near-miss swing itself crosses 60 deg
    # faster than the continuity guard admits, by design.
    set_pend(0.0, w_for_peak(80.0))
    step(300)
    amp_pre = bool(scene.lat_amp[0])
    set_pend(0.0, w_for_peak(102.0))
    th_max, ph_max = -180.0, -180.0
    for _ in range(500):
        step(1)
        th_max = max(th_max, th_deg())
        ph_max = max(ph_max, flap_deg())
    step(120)
    report("flap-nearmiss")
    s, ok = judge()
    check("flap near-miss: a genuine sub-pass-energy swing verifiably REACHES the blade "
          f"and deflects it (max theta {th_max:.1f} deg in (90, {c.theta_pass_deg:.0f}), "
          f"max flap {ph_max:.1f} deg > 3) yet the pass latch stays unearned and the "
          f"flap re-closes ({flap_deg():+.1f} deg); score still the 0.30 stage",
          amp_pre and 90.0 < th_max < c.theta_pass_deg and ph_max > 3.0
          and abs(flap_deg()) < 8.0 and not bool(scene.lat_pass[0])
          and abs(s - c.stage_scores[0]) < 0.02 and not ok)

    # =========================== 11. genuine pass, then extract =============================
    # A genuine high swing DOES earn the transit latch — positive control for the 0.60
    # stage — and the bob is immediately pulled back out (teleport) before it can settle:
    # the latched credit survives regression, and success is never granted.
    env.reset(seed=91)
    step(30)
    set_pend(0.0, w_for_peak(128.0))
    got_pass = False
    for _ in range(400):
        step(1)
        if bool(scene.lat_pass[0]):
            got_pass = True
            break
    report("pass-latched")
    set_pend(0.0)  # yank the bob straight back out of the pocket
    step(150)
    report("pulled-back-out")
    s_pass, ok = judge()
    check("pass then extract: a genuine high swing earns the flap-transit latch (0.60 "
          "stage), and after the bob is yanked back out of the pocket the latched "
          "credit survives regression — but success is never granted",
          got_pass and abs(s_pass - c.stage_scores[1]) < 0.02 and not ok
          and not bool(scene.in_pocket()[0]))

    # =========================== 12. flap self-recloses =====================================
    step(200)
    report("flap-reclosed")
    s, ok = judge()
    check("self-reclosing: the flap the passing bob shoved open has fallen SHUT on its "
          f"own (readback {flap_deg():+.1f} deg, |.| < 8) and the latched 0.60 survives "
          "the mechanism resetting; still no success",
          abs(flap_deg()) < 8.0 and abs(s - s_pass) < 0.02 and not ok)

    # =========================== 13. monotonicity ===========================================
    check("monotonicity: the battery's earned stages are strictly increasing "
          f"(null {s_null:.2f} < amplitude {s_amp:.2f} < transit {s_pass:.2f}), and "
          "success (1.0) was never granted",
          s_null + 0.10 < s_amp and s_amp + 0.10 < s_pass)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.swing_pump")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
