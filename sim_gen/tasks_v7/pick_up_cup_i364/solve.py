"""Teleport solution for CooperCurveScene (sim_gen task `pick_up_cup_i364`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the tapered CUP from its
   randomized pen slot to the STAGING pose at the gallery mouth — lying on its rims,
   base rim toward the arena centre, at the apex-true radius (rho_stage). The write
   sets zero velocity and satisfies no gallery clause by itself (staging credit is
   0.10 of 1.0 and requires the settled pose, which gravity+contact produce after the
   drop). Before writing, the ACTUAL decoy axis segment is read back and a staging
   candidate is chosen that clears it (segment-segment distance > rim sum + 8 mm);
   if every candidate is blocked the decoy is first transported to a clear park spot
   deeper in the pen (also clearance-checked against the cup's actual pose). All
   destinations are open-air pen floor — exactly the carry a gripper performs.
2. LAUNCH (applied force, pen only): a velocity-servo wrench at the cup's CoM,
   horizontal-force-capped, pushes the cup tangentially through the staging zone.
   The force is CUT (an explicit zero-wrench write) once the cup reaches azimuth
   -76.5 deg — before the gallery roof at -75 deg. Everything after that boundary
   is hands-off: the taper alone steers 150 deg of coast. This mirrors the
   embodiment: a robot can reach the open-topped pen but nothing under the roof.
3. COAST + ARREST (pure physics): the cup rolls the gallery, crosses the three
   checkpoint sectors in order, enters the catch bay and is stopped by the
   restitution-0 arrest wall, then settles. No writes, no forces.
4. RETRY, not cheat: if a coast stalls short (all attempts start below the friction
   budget), the cup is transported BACK to staging and relaunched with an escalated
   speed/force cap — the successful trajectory is still 100 % contact-steered, and
   the latched credit never decreases. The DECOY is never pushed and ends out of
   the bay (restraint).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing), then holds
HANDS-OFF >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_up_cup_i364.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# Staging candidates (deg, ordered: max launch runway first) and radial offsets (m,
# all within the 0.045 staging radius tolerance).
STAGE_CAND_PHI = [-89.0, -87.5, -86.0, -84.5, -83.0, -81.5, -80.0, -78.5]
STAGE_CAND_RHO = [0.0, +0.020, -0.020]
# Decoy park candidates (deg): radial-yaw spots deep in the pen, wall-clear.
PARK_CAND_PHI = [-148.0, -145.0, -142.0]
# Launch ladder: (target coast speed m/s, horizontal force cap N). Even the top cap
# (3.4 N) keeps rolling traction: needed floor friction ~ 0.27*F = 0.92 N < mu*m*g
# = 1.96 N.
LAUNCH_LADDER = [(0.75, 1.6), (0.95, 2.2), (1.15, 2.8), (1.35, 3.4)]
KP = 8.0  # servo gain 1/s; KP*dt = 0.067 << 1 (wrench acts one substep late)


def _seg_min_dist(a, b) -> float:
    """Min distance between two 2D segments a, b = (cx, cy, ux, uy, h): sampled
    points of one projected onto the other (both directions)."""

    def pt_seg(px, py, cx, cy, ux, uy, h):
        t = max(-h, min(h, (px - cx) * ux + (py - cy) * uy))
        return math.hypot(px - (cx + t * ux), py - (cy + t * uy))

    best = math.inf
    for f in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        best = min(best, pt_seg(a[0] + f * a[4] * a[2], a[1] + f * a[4] * a[3], *b))
        best = min(best, pt_seg(b[0] + f * b[4] * b[2], b[1] + f * b[4] * b[3], *a))
    return best


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cooper_curve")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        rho, phi = scene._polar(scene.cup.data.root_pos_w)
        v = float(scene.cup.data.root_lin_vel_w.norm(dim=-1)[0])
        w = float(scene.cup.data.root_ang_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:16s} | cup rho={float(rho[0]):.3f} "
              f"phi={float(phi[0]):+7.2f} z={float(scene._rel_z(scene.cup)[0]):.3f} "
              f"v={v:.3f} w={w:.2f} staged={bool(scene._staged[0])} "
              f"cp1={bool(scene._cp1[0])} cp2={bool(scene._cp2[0])} "
              f"cp3={bool(scene._cp3[0])} bay={bool(scene._bay[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def body_seg(body):
        """Actual (cx, cy, ux, uy, h) axis segment of a roller, env-local xy."""
        p = (body.data.root_pos_w - scene.env_origins)[0]
        ax = scene._axis_w(body)[0]
        axn = math.hypot(float(ax[0]), float(ax[1]))
        ux, uy = ((float(ax[0]) / axn, float(ax[1]) / axn) if axn > 1e-6
                  else (1.0, 0.0))
        return (float(p[0]), float(p[1]), ux, uy, c.length / 2.0)

    def cand_seg(rho: float, phi_deg: float):
        ar = math.radians(phi_deg)
        return (rho * math.cos(ar), rho * math.sin(ar),
                math.cos(ar), math.sin(ar), c.length / 2.0)

    CLEAR = c.r_mouth + c.r_decoy + 0.008  # teleport destination clearance

    def transport(body, rho: float, phi_deg: float, tapered: bool) -> None:
        """TRANSPORT-only teleport to a rest pose (radial yaw), then settle."""
        rho_t = torch.full((n,), rho, device=device)
        phi_t = torch.full((n,), math.radians(phi_deg), device=device)
        st = scene._rest_state(rho_t, phi_t, phi_t.clone(), tapered=tapered)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(60)  # hands-off settle: the resting pose is made by gravity+contact

    def stage_cup() -> float:
        """Stage the cup at a decoy-clear candidate; returns the staging phi used."""
        dseg = body_seg(scene.decoy)
        for phi_deg in STAGE_CAND_PHI:
            for drho in STAGE_CAND_RHO:
                if _seg_min_dist(cand_seg(c.rho_stage() + drho, phi_deg), dseg) > CLEAR:
                    transport(scene.cup, c.rho_stage() + drho, phi_deg, tapered=True)
                    return phi_deg
        # every staging candidate blocked by the decoy: park the decoy first
        print("[solve] staging blocked by the decoy; parking it deeper in the pen",
              flush=True)
        cseg = body_seg(scene.cup)
        for park_phi in PARK_CAND_PHI:
            if _seg_min_dist(cand_seg(c.slot_rho, park_phi), cseg) > CLEAR:
                transport(scene.decoy, c.slot_rho, park_phi, tapered=False)
                break
        else:
            print("SIM_GEN_SOLVE: FAIL (no clear park/staging destination)", flush=True)
            os._exit(1)
        dseg = body_seg(scene.decoy)
        for phi_deg in STAGE_CAND_PHI:
            for drho in STAGE_CAND_RHO:
                if _seg_min_dist(cand_seg(c.rho_stage() + drho, phi_deg), dseg) > CLEAR:
                    transport(scene.cup, c.rho_stage() + drho, phi_deg, tapered=True)
                    return phi_deg
        print("SIM_GEN_SOLVE: FAIL (staging still blocked after parking)", flush=True)
        os._exit(1)
        return 0.0  # unreachable

    def launch(v_des: float, f_cap: float) -> None:
        """Velocity-servo push at the CoM, horizontal cap `f_cap`, CUT at the
        hand-off azimuth (-76.5 deg, before the roof at -75). Wrench is re-written
        every step; the cut is an explicit zero-wrench write."""
        zero = torch.zeros(n, 1, 3, device=device)
        cut_phi = c.stage_phi_hi - 0.5
        for _ in range(480):
            rho, phi = scene._polar(scene.cup.data.root_pos_w)
            if float(phi[0]) >= cut_phi:
                break
            pr = torch.deg2rad(phi)
            u_t = torch.stack([-torch.sin(pr), torch.cos(pr)], dim=-1)
            v_xy = scene.cup.data.root_lin_vel_w[:, :2]
            f_xy = c.roller_mass * KP * (v_des * u_t - v_xy)
            mag = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_xy = f_xy * (mag.clamp(max=f_cap) / mag)
            wrench = torch.zeros(n, 1, 3, device=device)
            wrench[:, 0, 0:2] = f_xy
            scene.cup.set_external_force_and_torque(wrench, zero, env_ids=all_ids,
                                                    is_global=True)
            env.step(no_action)
        scene.cup.set_external_force_and_torque(zero, zero, env_ids=all_ids,
                                               is_global=True)
        env.step(no_action)

    def coast(max_seconds: float = 16.0) -> str:
        """Hands-off coast; returns 'success' | 'stalled' | 'timeout'."""
        still = 0
        for i in range(int(max_seconds * 120 / 10)):
            step(10)
            if bool(scene.success()[0]):
                return "success"
            v = float(scene.cup.data.root_lin_vel_w.norm(dim=-1)[0])
            in_bay = bool(scene.in_bay(scene.cup)[0])
            still = still + 1 if v < 0.04 else 0
            if i % 12 == 11:
                report(f"coast t={i * 10 / 120:5.1f}s")
            if still >= 12 and not in_bay:
                return "stalled"  # came to rest short of the bay
            if still >= 48 and in_bay:
                # at rest in the bay but success not live: give a long settle for
                # residual spin, then decide
                step(240)
                return "success" if bool(scene.success()[0]) else "stalled"
        return "timeout"

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats on the pen floor
    rho_c, phi_c = scene._polar(scene.cup.data.root_pos_w)
    rho_d, phi_d = scene._polar(scene.decoy.data.root_pos_w)
    mass = float(scene.cup.root_physx_view.get_masses().flatten()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cup=(rho {float(rho_c[0]):.3f}, phi {float(phi_c[0]):+.1f}) "
          f"decoy=(rho {float(rho_d[0]):.3f}, phi {float(phi_d[0]):+.1f}) "
          f"cup_slot={int(scene.cup_slot[0])} cup_mass={mass:.3f} "
          f"rho_stage={c.rho_stage():.4f} d_apex={c.d_apex():.4f} "
          f"rest_pitch={math.degrees(c.rest_pitch()):.2f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.cup.data.root_pos_w).all(), "NaN/inf after settle"
    assert abs(mass - c.roller_mass) < 0.02, f"authored mass not applied: {mass}"
    assert bool(scene.lying()[0]), "cup must start lying on its rims"
    s0 = print_score("P0 reset+settle (both rollers scattered in the pen)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: stage the cup at the gallery mouth --------------------------
    stage_phi = stage_cup()
    report("staged")
    assert bool(scene._staged[0]), "staging latch did not set after the settle"
    assert not bool(scene.success()[0]), "cannot be success at staging"
    s1 = print_score(f"P1 cup staged at phi={stage_phi:.1f} deg, base toward centre")
    assert s1 >= s0 - 1e-6 and 0.095 <= s1 <= 0.11, f"P1 score {s1} (expect 0.10)"

    # ---------------- phases 2+3: launch, hands-off coast (retry ladder) -------------------
    done = False
    for attempt, (v_des, f_cap) in enumerate(LAUNCH_LADDER):
        print(f"[solve] launch attempt {attempt + 1}: v_des={v_des:.2f} m/s "
              f"cap={f_cap:.1f} N", flush=True)
        launch(v_des, f_cap)
        report("hand-off")
        res = coast()
        report(f"coast-end({res})")
        if res == "success":
            done = True
            break
        if attempt + 1 < len(LAUNCH_LADDER):
            print(f"[solve] attempt {attempt + 1} {res}; re-staging for a harder "
                  f"launch (latched credit is kept)", flush=True)
            stage_phi = stage_cup()
            report("re-staged")
    if not done:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (cup never settled in the bay)", flush=True)
        os._exit(1)
    s2 = print_score("P2+3 cup rolled the gallery end to end and rests in the bay")
    assert s2 >= s1 - 1e-6 and s2 >= 0.99, f"success score must be 1.0, got {s2}"
    assert bool(scene._cp1[0] and scene._cp2[0] and scene._cp3[0] and scene._bay[0]), \
        "full checkpoint chain must be latched"
    assert not bool(scene.in_bay(scene.decoy)[0]), "decoy must be out of the bay"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
    except BaseException as exc:  # noqa: BLE001 - die fast, Kit won't exit on its own
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
