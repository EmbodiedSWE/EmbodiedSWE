"""Wrench solution for DieRollScene (sim_gen task `pick_cube_i125`) —
the task's legitimacy certificate.

There are NO teleports at all: the die starts on the floor where it will be worked,
and every state change is produced by contact dynamics under applied external
wrenches (the stand-ins for a robot's pushing contacts):

1. PERCEIVE + PLAN: read the die's orientation (which colored face is up, where the
   local +z BLUE axis points) and the GREEN mat's position (the mats jitter and swap
   sides). Choose a quarter-roll sequence: blue sideways -> one roll about
   u = normalize(blue x z_hat) lifts blue up; blue DOWN -> two rolls about the SAME
   axis u = z_hat x dir(goal) (each roll advances the die one edge-length toward the
   goal and the second brings blue up). The planner is closed-loop: it re-reads the
   pose after every roll and re-plans, so a scuffed roll just costs an extra
   iteration.
2. ROLL (tip over an edge): a bang-bang torque about the horizontal roll axis
   (tau0 = 1.25 * m g e/2 ~ 0.25 N m, angular-rate cap 1.3 rad/s) pivots the die
   over its ground edge; the torque is RELEASED at ~50 deg (past the 45 deg balance
   point) and gravity slams the die onto the next face. Landing retains ~omega/4
   about the new edge (5 mJ << the 82 mJ barrier for a second roll), so each command
   produces exactly one quarter-roll. The angular rate is measured by finite
   difference of the root quaternion (ang-vel readback is unreliable under active
   external wrenches on these pods).
3. SLIDE (translate without reorienting): a bang-bang horizontal force at the CoM
   (3.6 N, inside the [mu m g = 3.09, m g = 4.41] N slide-without-tip window;
   velocity capped at 0.12 m/s, tapered near the goal) pushes the die onto the
   green mat center, then releases and lets friction stop it.
4. FRAME ROBUSTNESS: the pods' external-wrench frame handling varies (world /
   rotation-since-reset drag / body frame). An ExternalWrenchDriver (in scene.py)
   is calibrated ONCE by a push probe (along -x, away from the mats), and every
   roll/slide re-verifies the measured response direction and cycles the encode
   mode if it disagrees — no mode is hard-coded.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds through and after the hold.

Run (forge): python -u -m simgen_tasks.pick_cube_i125.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qconj = scene_mod._qmul, scene_mod._qconj

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

FACE_NAMES = ("+x/RED", "-x/ORANGE", "+y/YELLOW", "-y/WHITE", "+z/BLUE", "-z/PURPLE")
DT = 1.0 / 120.0

# --- roll controller constants (physics audited in scene.py / TASK.md) -------------------
TAU_STATIC = 0.45 * 9.81 * 0.045          # m g e/2 = 0.199 N m (edge-pivot breakaway)
TAU0 = 1.25 * TAU_STATIC                  # 0.248 N m starting bang-bang torque
TAU_MAX = 2.5 * TAU_STATIC                # escalation ceiling
W_CAP = 1.3                               # rad/s bang-bang rate cap during the tip
RELEASE_ANG = math.radians(50.0)          # release past the 45 deg balance point
# --- slide controller constants ----------------------------------------------------------
F_SLIDE = 3.6                             # N: mu*m*g = 3.09 < F < m*g = 4.41 (no tip)
F_SLIDE_MAX = 3.85
V_CAP = 0.12                              # m/s slide speed cap (far), tapered near goal
SLIDE_STOP = 0.020                        # m: cut the force here (zone_tol = 0.055)


def _rotvec(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3) rotation vector (axis * angle, radians)."""
    w = q[:, 0].clamp(-1.0, 1.0)
    v = q[:, 1:]
    vn = v.norm(dim=-1).clamp_min(1e-9)
    ang = 2.0 * torch.atan2(vn, w.abs())
    sgn = torch.where(w >= 0, 1.0, -1.0).unsqueeze(-1)
    return sgn * v / vn.unsqueeze(-1) * ang.unsqueeze(-1)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int = 1) -> None:
        for _ in range(k):
            env.step(no_action)

    def die_xy() -> torch.Tensor:
        return scene.die.data.root_pos_w[:, :2]

    def goal_xy() -> torch.Tensor:
        return scene.goal_mat.data.root_pos_w[:, :2]

    def report(tag: str) -> None:
        p = (scene.die.data.root_pos_w - scene.env_origins)[0]
        idx, best = scene.face_up()
        ez = scene._die_axes()[2]
        print(f"[solve] {tag:14s} | die=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) up={FACE_NAMES[int(idx[0])]}({float(best[0]):+.3f}) "
              f"blue_z={float(ez[0, 2]):+.3f} d_goal={float(scene.goal_dist()[0]):.3f} "
              f"rolled={bool(scene._rolled[0])} blue={bool(scene._blue[0])} "
              f"near={bool(scene._near[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_die(min_steps: int = 40, need: int = 10, timeout: int = 600) -> bool:
        """Hands-off settle: consecutive-streak stillness with a minimum step count
        (a velocity threshold alone fires at swing turning points)."""
        drv.clear()
        streak = 0
        for i in range(timeout):
            step()
            if i >= min_steps and bool(scene.settled()[0]):
                streak += 1
                if streak >= need:
                    return True
            else:
                streak = 0
        return False

    # ---------------- phase 0: reset, settle, baseline -----------------------------------
    step(180)  # seat on the ground
    idx0, best0 = scene.face_up()
    p0 = (scene.die.data.root_pos_w - scene.env_origins)[0]
    gp = (scene.goal_mat.data.root_pos_w - scene.env_origins)[0]
    dp = (scene.decoy_mat.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"die=({float(p0[0]):+.3f},{float(p0[1]):+.3f}) "
          f"up={FACE_NAMES[int(idx0[0])]}({float(best0[0]):+.3f}) "
          f"goal_mat=({float(gp[0]):+.3f},{float(gp[1]):+.3f}) "
          f"decoy_mat=({float(dp[0]):+.3f},{float(dp[1]):+.3f})", flush=True)
    report("reset")
    m_meas = float(scene.die.root_physx_view.get_masses()[0].sum())
    print(f"[solve] die mass readback: {m_meas:.3f} kg (authored {c.die_m})", flush=True)
    assert abs(m_meas - c.die_m) < 0.02, \
        f"die mass {m_meas} != authored {c.die_m} (MassAPI not applied -> slide window wrong)"
    assert torch.isfinite(scene.die.data.root_state_w).all(), "NaN/inf after settle"
    assert int(idx0[0]) != 4, "blue must never start up (scene contract)"
    assert float(best0[0]) > c.up_snap, "die must start flat on a face"
    s0 = print_score("P0 reset+settle (die flat, blue not up, far from the mats)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- wrench driver + one-time frame calibration -------------------------
    drv = scene_mod.ExternalWrenchDriver(scene.die, n, device)
    drv.snapshot_ref()
    # Push AWAY from the mats (-x) so the calibration shove cannot drift the die
    # toward the approach-latch radius.
    drv.calibrate(lambda: step(), force=-F_SLIDE, tag="(reset)")
    settle_die()
    report("calibrated")

    zero3 = torch.zeros(n, 3, device=device)

    # ---------------- roll primitive ------------------------------------------------------
    def do_roll(u_xy: tuple[float, float], tag: str) -> bool:
        """One quarter-roll about the world horizontal unit axis u (right-hand rule;
        the die translates along u x z_hat by one edge length). Bang-bang torque with
        an FD rate cap, release at RELEASE_ANG, hands-off landing. Verifies the
        response axis ~25 steps in and cycles the wrench encode mode on disagreement;
        escalates torque on stall. Returns True once the die has demonstrably tipped
        (geodesic >= 57 deg from the roll start) and settled flat."""
        un = math.hypot(u_xy[0], u_xy[1])
        u = torch.tensor([[u_xy[0] / un, u_xy[1] / un, 0.0]], device=device).expand(n, 3)
        tau = TAU0
        for attempt in range(7):
            q_start = scene.die.data.root_quat_w.clone()
            q_prev = q_start.clone()
            tipped = False
            verdict = "timeout"
            for i in range(700):
                q_now = scene.die.data.root_quat_w
                rv_step = _rotvec(_qmul(q_now, _qconj(q_prev)))
                q_prev = q_now.clone()
                w_u = float((rv_step[0] * u[0]).sum()) / DT   # FD rate about u
                rv_tot = _rotvec(_qmul(q_now, _qconj(q_start)))
                ang = float(rv_tot[0].norm())
                if i == 25:
                    if ang < 0.02:
                        verdict = "stall"
                        break
                    if float((rv_tot[0] * u[0]).sum()) < 0.5 * ang:
                        verdict = "misencode"
                        break
                if ang >= RELEASE_ANG:
                    tipped = True
                    verdict = "tipped"
                    break
                if w_u < W_CAP:
                    drv.apply(zero3, tau * u)
                else:
                    drv.clear()
                step()
            drv.clear()
            if verdict == "misencode":
                settle_die()
                print(f"[solve] {tag}: response axis disagrees -> encode mode "
                      f"'{drv.cycle()}' (attempt {attempt + 1})", flush=True)
                continue
            if verdict == "stall":
                if tau >= TAU_MAX - 1e-6:
                    print(f"[solve] {tag}: stalled at tau_max -> encode mode "
                          f"'{drv.cycle()}'", flush=True)
                else:
                    tau = min(tau * 1.35, TAU_MAX)
                    print(f"[solve] {tag}: no motion -> tau={tau:.3f} N m "
                          f"(attempt {attempt + 1})", flush=True)
                settle_die()
                continue
            if tipped:
                settle_die(min_steps=50)
                rv_end = _rotvec(_qmul(scene.die.data.root_quat_w, _qconj(q_start)))
                if float(rv_end[0].norm()) > 1.0 and float(scene.face_up()[1][0]) > c.up_snap:
                    report(tag)
                    return True
                print(f"[solve] {tag}: fell back after release -> tau up", flush=True)
                tau = min(tau * 1.35, TAU_MAX)
                continue
            # timeout: rate cap kept it slow but it never reached release
            tau = min(tau * 1.35, TAU_MAX)
            settle_die()
            print(f"[solve] {tag}: release not reached -> tau={tau:.3f}", flush=True)
        print(f"SIM_GEN_SOLVE: FAIL ({tag}: quarter-roll never completed)", flush=True)
        os._exit(1)

    # ---------------- slide primitive -----------------------------------------------------
    def do_slide(tag: str) -> bool:
        """Bang-bang CoM force toward the green mat center, velocity-capped, force
        inside the slide-without-tip window. Aborts (returns False, for a replan) if
        the die starts tipping. Frame robustness: no advance is treated as evidence
        of a MISENCODED force (a wrong drag/world/body mode after the rolls rotates
        the horizontal push into the floor — zero motion at ANY magnitude), so after
        escalating to the window ceiling the encode mode is cycled; only once every
        mode has stalled is the ceiling raised as a last resort (the tip limit is
        m g = 4.41 N and the rock-abort guards it)."""
        f_push, f_max = F_SLIDE, F_SLIDE_MAX
        stalls_at_max = 0
        p_mark = die_xy().clone()
        mark_i = 0
        p_prev = die_xy().clone()
        v_fd_along = 0.0
        for i in range(4800):
            d = goal_xy() - die_xy()
            dist = float(d[0].norm())
            if dist < SLIDE_STOP:
                break
            ez = scene._die_axes()[2]
            if float(ez[0, 2]) < 0.92:  # rocking/tipping: abort and replan
                drv.clear()
                settle_die()
                print(f"[solve] {tag}: die rocked during slide -> replan", flush=True)
                return False
            dirv = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            # Velocity READBACK is unreliable while an external wrench is active on
            # this build (phantom values would clear the force every step and fake a
            # stall): gate on a position finite difference instead.
            p_now = die_xy()
            v_fd = (p_now - p_prev)[0] / DT
            p_prev = p_now.clone()
            v_fd_along = float((v_fd * dirv[0]).sum())
            vcap = min(V_CAP, max(0.03, 1.2 * dist))
            if v_fd_along < vcap:
                f = torch.zeros(n, 3, device=device)
                f[:, :2] = f_push * dirv
                drv.apply(f, zero3)
            else:
                drv.clear()
            step()
            if i - mark_i >= 90:
                adv = float(((die_xy() - p_mark)[0] * dirv[0]).sum())
                v_rb = float((scene.die.data.root_lin_vel_w[0, :2] * dirv[0]).sum())
                print(f"[solve] {tag}: i={i} d={dist * 1000:.0f}mm adv={adv * 1000:+.1f}mm "
                      f"vfd={v_fd_along:+.3f} vrb={v_rb:+.3f} F={f_push:.2f} "
                      f"mode={drv.mode}", flush=True)
                if adv < -0.004:
                    drv.clear()
                    settle_die(min_steps=10, need=5, timeout=180)
                    print(f"[solve] {tag}: drift opposes push -> encode mode "
                          f"'{drv.cycle()}'", flush=True)
                    f_push = F_SLIDE
                elif adv < 0.002:
                    if f_push < f_max - 1e-6:
                        f_push = min(f_push * 1.12, f_max)
                        print(f"[solve] {tag}: slide stalled -> F={f_push:.2f} N",
                              flush=True)
                    else:
                        drv.clear()
                        settle_die(min_steps=10, need=5, timeout=180)
                        stalls_at_max += 1
                        f_push = F_SLIDE
                        print(f"[solve] {tag}: no advance at F_max -> encode mode "
                              f"'{drv.cycle()}' (stall #{stalls_at_max})", flush=True)
                        if stalls_at_max == 3:
                            f_max = 4.2  # all modes stalled: friction, not frame
                        elif stalls_at_max >= 7:
                            print(f"[solve] {tag}: giving up -> replan", flush=True)
                            return False
                p_mark = die_xy().clone()
                mark_i = i
        drv.clear()
        settle_die(min_steps=30)
        report(tag)
        return True

    # ---------------- closed-loop plan/act loop ------------------------------------------
    s_prev = s0
    p1_done = False
    p2_done = False
    slide_calibrated = False
    for it in range(14):
        settle_die(min_steps=10)
        if scene._rolled[0] and not p1_done:
            s1 = print_score("P1 first quarter-roll landed (up face changed)")
            assert s1 >= s_prev - 1e-6 and s1 >= 0.195, f"P1 score {s1} (expect >= 0.20)"
            s_prev, p1_done = s1, True
        if bool(scene.blue_up()[0]) and bool(scene.settled()[0]):
            if not p2_done:
                s2 = print_score("P2 blue face up (roll sequence complete)")
                assert s2 >= s_prev - 1e-6 and s2 >= 0.44, f"P2 score {s2} (expect >= 0.45)"
                s_prev, p2_done = s2, True
            if bool(scene.on_goal()[0]):
                break
            if not slide_calibrated:
                # The reset-time calibration cannot tell "drag" from "world" (they
                # coincide at the reset orientation). NOW the die has rolled, so the
                # modes are distinguishable: re-calibrate with a push AWAY from the
                # mats before trusting the slide force direction.
                drv.calibrate(lambda: step(), force=-F_SLIDE, tag="(pre-slide)")
                settle_die(min_steps=10, need=5, timeout=240)
                slide_calibrated = True
            if do_slide(f"slide[{it}]"):
                if bool(scene.blue_up()[0]) and bool(scene.on_goal()[0]):
                    break
            continue
        ez = scene._die_axes()[2][0]
        bz = float(ez[2])
        if bz < -0.5:
            # blue DOWN: roll toward the goal; the same axis next iteration brings
            # blue up while advancing one edge-length toward the mat.
            d = (goal_xy() - die_xy())[0]
            dn = float(d.norm())
            u = (-float(d[1]) / dn, float(d[0]) / dn)          # z_hat x dir(goal)
            do_roll(u, f"roll[{it}] (blue down, advance to goal)")
        else:
            # blue SIDEWAYS: one roll about (blue x z_hat) lifts blue up.
            bh = math.hypot(float(ez[0]), float(ez[1]))
            u = (float(ez[1]) / bh, -float(ez[0]) / bh)        # blue x z_hat
            do_roll(u, f"roll[{it}] (lift blue up)")
    else:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (plan loop exhausted)", flush=True)
        os._exit(1)

    # ---------------- phase 3: at rest, blue up, centered on the green mat ----------------
    settle_die(min_steps=30)
    report("final")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success at final settle)", flush=True)
        os._exit(1)
    s3 = print_score("P3 die at rest, blue up, centered on the GREEN mat")
    assert s3 >= s_prev - 1e-6 and s3 >= 0.999, f"P3 score {s3} (expect 1.0)"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    main()
