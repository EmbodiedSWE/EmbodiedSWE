"""Teleport solution for GriddleSpatulaScene (sim_gen task `meat_off_grill_i355`) —
the task's legitimacy certificate.

ONE teleport, transport only: the free spatula is transported from its spawn spot on
the ground to a hover pose on the griddle behind the patty (a pose the arm would
reach by carrying the tool). EVERY load-bearing interaction is contact dynamics,
executed through a 6-DOF external-wrench PD servo on the spatula (the wrench a hand
holding the grip post applies — bounded force, bounded torque):

1. WEDGE (contact dynamics): the blade is pressed down against the griddle and
   driven along the fixture's +x. The patty is shoved ahead until the far rim wall
   pins it, then the slick tapered ramp slides UNDER the pinned patty and the patty
   climbs onto the blade. The patty is never touched by anything but the tool.
2. LIFT + TRANSIT (contact dynamics): the loaded blade pitches slightly nose-up
   (the grip post is the rear backstop), rises over the rim, and is carried to the
   dish — the patty rides as a passive payload held by friction.
3. DISCHARGE (contact dynamics): over the dish the blade tilts nose-down past the
   patty/blade friction angle; the patty slides off the slick ramp and lands FLAT
   inside the dish. Gravity finishes the drop.
4. PARK: the tool is carried away and set down on the ground far from the dish;
   forces are cut; the verdict state is fully hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
from types import SimpleNamespace

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
DT = 1.0 / 120.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.griddle_spatula")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import (axis_angle_from_quat, quat_apply, quat_conjugate,
                                     quat_mul)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        gp = scene._local(scene.griddle, scene.patty)[0]
        sp = scene._local(scene.spatula, scene.patty)[0]
        pz = float(scene.patty.data.root_pos_w[0, 2])
        sz = float(scene.spatula.data.root_pos_w[0, 2])
        print(f"[solve] {tag:12s} | patty_griddle=({float(gp[0]):+.3f},{float(gp[1]):+.3f},"
              f"{float(gp[2]):+.3f}) patty_spat=({float(sp[0]):+.3f},{float(sp[1]):+.3f},"
              f"{float(sp[2]):+.3f}) patty_z={pz:.3f} spat_z={sz:.3f} "
              f"on_griddle={bool(scene._on_griddle()[0])} scooped={bool(scene._scooped()[0])} "
              f"in_dish={bool(scene._in_dish()[0])} tool_away={bool(scene._tool_away()[0])} "
              f"eng={float(scene._eng_max[0]):.2f} transit={bool(scene._transit_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline readback -----------------------------
    step(60)
    f_pos = scene.griddle.data.root_pos_w[0].clone()          # world (incl. origin)
    f_quat = scene.griddle.data.root_quat_w.clone()           # (1, 4)
    f_yaw = 2.0 * math.atan2(float(f_quat[0, 3]), float(f_quat[0, 0]))
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3)
    fwd = quat_apply(f_quat, ex)[0]                           # fixture +x in world
    patty_loc0 = scene._local(scene.griddle, scene.patty)[0].clone()
    dish_pos = scene.dish.data.root_pos_w[0].clone()
    side = float(scene._dish_side[0])
    m_read_p = float(scene.patty.root_physx_view.get_masses().sum())
    m_read_s = float(scene.spatula.root_physx_view.get_masses().sum())
    print(f"[solve] layout readback (seed {args.seed}): "
          f"fix=({float(f_pos[0]):+.3f},{float(f_pos[1]):+.3f}) yaw={math.degrees(f_yaw):+.1f}deg "
          f"patty_local=({float(patty_loc0[0]):+.3f},{float(patty_loc0[1]):+.3f}) "
          f"dish=({float(dish_pos[0]):+.3f},{float(dish_pos[1]):+.3f}) side={side:+.0f} "
          f"spat=({float(scene.spatula.data.root_pos_w[0, 0]):+.3f},"
          f"{float(scene.spatula.data.root_pos_w[0, 1]):+.3f}) "
          f"masses(patty={m_read_p:.3f},spatula={m_read_s:.3f})", flush=True)
    assert abs(m_read_p - c.patty_mass) < 0.03, "patty MassAPI mass not applied"
    assert abs(m_read_s - c.spat_mass) < 0.06, "spatula MassAPI mass not applied"
    report("reset")
    assert bool(scene._on_griddle()[0]), "patty not on the griddle at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- wrench-servo controller (the hand holding the grip post) --------------
    m_s, m_p = c.spat_mass, c.patty_mass
    KP, KD = 80.0, 14.0          # translational PD (per unit mass)
    KR, KW = 1.2, 0.10           # rotational PD (authored diag inertia 2e-3)
    H = SimpleNamespace(p_ref=scene.spatula.data.root_pos_w[0].clone(),
                        pitch=0.0, lead=0.10, yaw=0.0)

    def patty_rel_yaw() -> float:
        # patty yaw relative to the fixture, folded onto the square's 90-deg
        # symmetry (the rim undercut repeats on every face)
        pq = scene.patty.data.root_quat_w
        yaw_p = 2.0 * math.atan2(float(pq[0, 3]), float(pq[0, 0]))
        return (yaw_p - f_yaw + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0

    def loc2world(loc) -> torch.Tensor:
        v = torch.tensor([float(loc[0]), float(loc[1]), float(loc[2])],
                         device=device).view(1, 3)
        return f_pos + quat_apply(f_quat, v)[0]

    def rider_on_blade() -> bool:
        # physical support test (wider than the rubric's scoop window): the hand
        # keeps feeling the payload weight until the patty actually leaves the tool
        rel = quat_apply(quat_conjugate(scene.spatula.data.root_quat_w),
                         scene.patty.data.root_pos_w - scene.spatula.data.root_pos_w)[0]
        return (-0.07 < float(rel[0]) < 0.13 and abs(float(rel[1])) < 0.08
                and -0.03 < float(rel[2]) < 0.06)

    def drive() -> None:
        p = scene.spatula.data.root_pos_w[0]
        v = scene.spatula.data.root_lin_vel_w[0]
        err = (H.p_ref - p).clamp(-H.lead, H.lead)
        ff = (m_s + (m_p if rider_on_blade() else 0.0)) * G
        F = m_s * (KP * err - KD * v)
        F[2] += ff
        F = F.clamp(-15.0, 15.0)
        half = 0.5 * H.pitch
        q_pitch = torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0],
                               device=device).view(1, 4)
        hy = 0.5 * H.yaw
        q_yaw = torch.tensor([math.cos(hy), 0.0, 0.0, math.sin(hy)],
                             device=device).view(1, 4)
        q_ref = quat_mul(quat_mul(f_quat, q_yaw), q_pitch)
        q = scene.spatula.data.root_quat_w
        aa = axis_angle_from_quat(quat_mul(q_ref, quat_conjugate(q)))[0]
        w = scene.spatula.data.root_ang_vel_w[0]
        T = (KR * aa - KW * w).clamp(-0.8, 0.8)
        scene.spatula.set_external_force_and_torque(
            F.view(n, 1, 3).contiguous(), T.view(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)

    def move_to(target: torch.Tensor, pitch_t: float, *, speed: float = 0.05,
                prate: float = math.radians(25.0), hold: int = 0,
                max_steps: int = 1500, stop=None) -> bool:
        """Rate-limited carrot toward `target` + pitch ramp toward `pitch_t`."""
        held = 0
        for _ in range(max_steps):
            d = target - H.p_ref
            dn = float(d.norm())
            if dn > 1e-9:
                H.p_ref += d * (min(speed * DT, dn) / dn)
            dp = pitch_t - H.pitch
            H.pitch += max(-prate * DT, min(prate * DT, dp))
            drive()
            if stop is not None and stop():
                return True
            if dn < 5e-4 and abs(dp) < 2e-3:
                held += 1
                if held > hold:
                    return stop is None
        return False

    # ---------------- phase 1: TELEPORT-IN (transport only) then WEDGE ----------------------
    # Transport: spatula to a hover pose on the griddle behind the patty, level, yaw
    # aligned with the fixture. From here on, contact dynamics only.
    ramp_reach = c.blade_l / 2 + c.ramp_l * math.cos(math.radians(c.ramp_deg))
    start_x = float(patty_loc0[0]) - c.patty_side / 2 - 0.010 - ramp_reach
    entry = loc2world((start_x, float(patty_loc0[1]), c.slab_h + 0.010))
    # enter SQUARE to the patty's sampled yaw so the taper meets the undercut
    # mouth face-on (a corner-first meeting jams; the rim squares the patty later)
    H.yaw = patty_rel_yaw()
    hy0 = 0.5 * H.yaw
    q_e = quat_mul(f_quat, torch.tensor([math.cos(hy0), 0.0, 0.0, math.sin(hy0)],
                                        device=device).view(1, 4))
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = entry
    st[0, 3:7] = q_e[0]
    scene.spatula.write_root_state_to_sim(st, all_ids)
    step(25)  # settle onto the griddle top
    report("entry")
    assert bool(scene._on_griddle()[0]), "patty left the griddle during tool entry?!"

    # Wedge: light down-press (z_ref a little below the surface) + advance along
    # fixture +x at a few cm/s. The far rim wall is the backstop: the patty is
    # pinned there, the tapered tip enters its rim undercut, and the overhanging
    # rim rides up the slick ramp onto the blade.
    H.p_ref = loc2world((start_x, float(patty_loc0[1]), c.slab_h - 0.020))
    H.pitch = math.radians(3.0)  # slight nose-down: keep the taper tip on the plate
    wedge_end_x = 0.108          # ramp tip stops ~8 mm short of the +x rim wall
    streak, best_x, last_bump = 0, -1.0, 0
    scooped_ok = False
    dither = 0.0  # yaw-wiggle amplitude, armed only after the lead cap is maxed
    for i in range(3600):
        sx = float(scene._local(scene.griddle, scene.spatula)[0, 0])
        adv = quat_apply(f_quat, torch.tensor([0.045 * DT, 0.0, 0.0],
                                              device=device).view(1, 3))[0]
        if sx < wedge_end_x:
            H.p_ref += adv
        # track the patty's live yaw (rate-limited) so the blade face stays
        # square to the undercut mouth while the rim squares the patty
        tgt = max(-0.26, min(0.26, patty_rel_yaw())) \
            + dither * math.sin(2.0 * math.pi * 1.2 * i * DT)
        H.yaw += max(-math.radians(45.0) * DT,
                     min(math.radians(45.0) * DT, tgt - H.yaw))
        drive()
        streak = streak + 1 if bool(scene._scooped()[0]) else 0
        if streak >= 40 and sx >= wedge_end_x - 0.010:
            scooped_ok = True
            break
        if sx > best_x + 0.002:
            best_x, last_bump = sx, i
        elif i - last_bump > 240:  # stalled: press/pull harder, then wiggle
            if H.lead < 0.22:
                H.lead = min(H.lead + 0.05, 0.22)
            else:
                dither = math.radians(3.0)
            last_bump = i
            print(f"[solve] wedge stall at spat_x={sx:+.3f} -> lead={H.lead:.2f} "
                  f"dither={math.degrees(dither):.0f}deg", flush=True)
        if i % 200 == 199:
            report(f"wedge{i + 1}")
    report("wedged")
    assert scooped_ok, "the blade never scooped the patty (wedge failed)"
    s1 = print_score("P1 patty scooped onto the blade")
    assert s1 >= s0 - 1e-6, "score decreased across the wedge"

    # ---------------- phase 2: LIFT over the rim + TRANSIT to the dish ----------------------
    H.lead = 0.10
    H.yaw = 0.0  # patty is squared against the rim by now; go fixture-square
    cur = scene.spatula.data.root_pos_w[0].clone()
    H.p_ref = cur.clone()
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    lift = cur.clone()
    lift[2] = 0.20
    lift -= fwd * 0.015  # ease the tip off the rim wall while rising
    ok = move_to(lift, math.radians(-6.0), speed=0.05, hold=30, max_steps=900)
    report("lifted")
    assert bool(scene._scooped()[0]), "patty fell off the blade during the lift"

    # nose at ~-0.03 from the dish centre at 28 deg tilt: the patty's trailing edge
    # exits just inside the near wall and the slow slide parks it around the centre
    over = dish_pos + up * 0.20 - fwd * 0.105
    ok = move_to(over, math.radians(-6.0), speed=0.06, hold=30, max_steps=1400)
    report("transit")
    assert bool(scene._scooped()[0]), "patty fell off the blade during the carry"
    assert bool(scene._transit_ever[0]), "transit latch never fired"
    s2 = print_score("P2 loaded blade carried over the dish")
    assert s2 >= s1 - 1e-6, "score decreased across the carry"

    # ---------------- phase 3: DISCHARGE (tilt-pour) then PARK the tool ---------------------
    low = over.clone()
    low[2] = 0.055  # nose ends ~3 mm above the dish floor at full tilt: the patty
    move_to(low, 0.0, speed=0.04, hold=20, max_steps=900)  # has no room to up-end

    # PRESSED-CONTACT POUR: drop the z-reference deep so the lead-clamped PD holds
    # a bounded downward press (~m_s*KP*lead ~ 1.8 N) that pins the nose to the
    # dish floor for the whole pour. The payload feedforward cannot track the
    # weight handover (patty weight moves to the floor while ff still credits it,
    # surplus/(m_s*KP) ~ 5 cm of float); a pressed contact absorbs that error in
    # the floor reaction instead of blade height.
    H.p_ref[2] = -0.02

    def discharged() -> bool:
        return (not bool(scene._scooped()[0])
                and float(scene.patty.data.root_pos_w[0, 2]) < 0.06)

    # gentle pour: a few degrees past the ~21.8 deg patty/blade friction angle, so
    # the patty exits slowly and cannot trip over the dish's far wall
    done = False
    for i in range(900):
        H.pitch = min(H.pitch + math.radians(20.0) * DT, math.radians(28.0))
        drive()
        if discharged():
            done = True
            break
    if not done:
        # BRIDGE stall: the patty's leading edge anchors on the dish floor
        # (mu 0.55) while its rear still rests on the blade — tilt alone cannot
        # break the bridge. Steepen the pour well past every pair friction
        # angle, still pressed (the nose stays pinned, so no room to tumble).
        print("[solve] bridged at 28 deg -> steepen to 40 deg", flush=True)
        for i in range(400):
            H.pitch = min(H.pitch + math.radians(20.0) * DT, math.radians(40.0))
            drive()
            if discharged():
                done = True
                break
    if not done:
        # last resort: pull the sheet out — withdraw the pressed blade backward;
        # the floor friction anchors the patty while the slick blade slides out
        # from under its rear (the deep-z press stays on the reference)
        print("[solve] still bridged -> pull the blade out from under", flush=True)
        back0 = H.p_ref.clone() - fwd * 0.06
        move_to(back0, H.pitch, speed=0.02, hold=10, max_steps=700,
                stop=discharged)
        done = discharged()
    for _ in range(60):  # keep DRIVING (hold pose) while the patty finishes sliding
        drive()          # off — a frozen stale wrench would drift the blade upward
    report("discharged")
    assert done, "the patty never slid off the blade over the dish"

    # withdraw the blade BACKWARD from under the patty's trailing edge (pull the
    # sheet out) — same pressed height, same tilt, so the nose slides out without
    # lifting or flicking the patty
    back = H.p_ref.clone() - fwd * 0.04  # ref-based: keep the deep-z press active
    move_to(back, H.pitch, speed=0.03, hold=20, max_steps=500)

    # park: rise, level, carry the tool away, set it down, cut forces
    rise = scene.spatula.data.root_pos_w[0].clone()
    rise[2] = 0.16
    H.p_ref = scene.spatula.data.root_pos_w[0].clone()
    move_to(rise, 0.0, speed=0.06, prate=math.radians(40.0), hold=10, max_steps=600)
    park_hi = loc2world((c.park_local[0], c.park_local[1], 0.16))
    move_to(park_hi, 0.0, speed=0.08, hold=10, max_steps=1200)
    park_lo = loc2world((c.park_local[0], c.park_local[1], 0.012))
    move_to(park_lo, 0.0, speed=0.04, hold=30, max_steps=700)
    scene.spatula.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    report("parked")

    # ---------------- phase 4: hands-off success + persistence ------------------------------
    streak = 0
    for _ in range(900):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 30:
            break
    report("success-wait")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success after parking)", flush=True)
        os._exit(1)
    s4 = print_score("P4 success: patty flat in the dish, tool away")
    assert s4 >= s2 - 1e-6, "score decreased at success"

    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except BaseException:  # noqa: BLE001 — die loudly instead of idling to the watchdog
        import traceback
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
