"""Teleport solution for DecantReturnScene (sim_gen task
`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i417`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (the "carry" a gripper would do): the bottle+marble
assembly is teleported to a hover pose over the tray, and the empty bottle is later
teleported to a drop point above the socket. Every load-bearing interaction is real
contact dynamics under applied wrenches:

1. HELD POUR (6-DOF CoM wrench servo, contact dynamics): the bottle is held at a
   mouth-anchored reference over the tray (p_des = p_aim - R_des * r_mouth) and rolled
   about the rig's +x axis through an escalating tilt ladder (115/135/155/170 deg,
   ~26 deg/s). Interior friction is slick (mu 0.15), so past ~100 deg the marble
   slides down the cavity wall, crosses the 45-deg funnel, and discharges through the
   40 mm mouth — a real pour, which fires the scene's continuity-guarded decant
   latch. The wrench: force kp 50 / kd 10 / cap 12 N + live gravity feed-forward
   (marble weight included only while it is still inside); torque kq 0.8 / kdw 0.02
   (kdw*dt/I < 1 — the bottle's inertia is ~2.7e-4 kg m^2) / cap 1.0 N m + the
   marble-offset gravity moment.
   FORCE-FRAME CONVENTION: some pods rotate an applied external wrench by the body's
   rotation since its reference orientation (applied = R_now * R_ref^T * arg). The
   wrench is therefore PRE-ENCODED through a probed mode (mode 1 multiplies by
   R_ref * R_now^T), the mode is probed at the hover engage (early-bail 0.06 m so the
   marble never leaves during a wrong-mode attempt) and again with a 35-deg tilt
   probe, and an IN-POUR DIVERGENCE GUARD (0.10 m, sized to trip far before the
   discharge angle) catches the residual case, toggles the mode, re-teleports the
   still-loaded assembly to the hover and retries once.
2. AIMED LANDING (gravity + contact): the marble exits mostly along -y_rig, so the
   mouth is aimed 30 mm on the +y side of the tray center; the 55 mm walls catch the
   roll-out. The pose is held until the marble rests in the tray.
3. SEAT THE BOTTLE (gravity drop, contact dynamics): the bottle is rolled back to
   upright while held, then teleported to 20 mm above the socket plate (yaw aligned
   with the rig — the 70 mm square base only enters the 82 mm square interior
   un-rotated) and DROPPED. The drop-in registration against plate and lip is what
   seats it; up to 3 measured-offset re-drops.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i417.solve --headless [--seed N]
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
import sys
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # forge fallback: run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.decant_return")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    env.reset(seed=args.seed)  # seed AFTER build
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    bottle, marble = scene.bottle, scene.marble
    m_b, m_m = c.bottle_mass, c.marble_mass
    r_mouth = torch.tensor([0.0, 0.0, c.bottle_zmid], device=device).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    NEG = torch.tensor([1.0, -1.0, -1.0, -1.0], device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rig_xyz(body) -> tuple[float, float, float]:
        p = scene._rig_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        bx, by, bz = rig_xyz(bottle)
        mx, my, mz = rig_xyz(marble)
        print(f"[solve] {tag:14s} | bottle=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"marble=({mx:+.3f},{my:+.3f},{mz:+.3f}) "
              f"in_bottle={bool(scene.marble_in_bottle()[0])} "
              f"in_tray={bool(scene.marble_in_tray()[0])} "
              f"seated={bool(scene.bottle_seated()[0])} "
              f"latch=[l {int(scene._lifted[0])} h {int(scene._hover[0])} "
              f"d {int(scene._decant[0])} t {int(scene._tray_ever[0])} "
              f"s {int(scene._seated[0])}] "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def clamp_norm(v: torch.Tensor, cap: float) -> torch.Tensor:
        nn = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return v * (nn.clamp(max=cap) / nn)

    # ----- 6-DOF held-pose wrench servo ----------------------------------------------------
    # Force-frame convention: some pods rotate an applied external wrench by the
    # body's rotation since a STALE reference orientation (applied = R_now*R_ref^T*arg;
    # R_ref is typically the spawn pose — here an ARBITRARY yaw, the bottle spawns
    # yaw-free, so the drag is a yaw error up to 180 deg even at the upright hover).
    # mode 0 passes world vectors through; mode 1 pre-encodes with R_ref*R_now^T,
    # where q_ref is MEASURED by kick_probe (never assumed equal to a readback).
    state = {"mode": 0, "q_ref": None}

    def encode(vec_w: torch.Tensor) -> torch.Tensor:
        if state["mode"] == 0 or state["q_ref"] is None:
            return vec_w
        q_now = bottle.data.root_quat_w
        drag = scene_mod._qmul(state["q_ref"], q_now * NEG)
        return quat_apply(drag, vec_w)

    def z_aim(theta: float) -> float:
        # mouth height above the ground: high while upright (bottle hangs below the
        # mouth, must clear the tray walls), lower as it rolls (shorter drop)
        return 0.135 + 0.075 * (1.0 + math.cos(theta)) / 2.0

    def pose_des(theta: float) -> tuple[torch.Tensor, torch.Tensor]:
        q_rig = scene.rig.data.root_quat_w
        axis = quat_apply(q_rig, ex)
        h = theta / 2.0
        q_roll = torch.zeros(n, 4, device=device)
        q_roll[:, 0] = math.cos(h)
        q_roll[:, 1:4] = axis * math.sin(h)
        q_des = scene_mod._qmul(q_roll, q_rig)
        aim = torch.tensor([c.tray_x, c.tray_y + 0.030, z_aim(theta)],
                           device=device).expand(n, 3)
        p_aim = quat_apply(q_rig, aim) + scene.rig.data.root_pos_w
        p_des = p_aim - quat_apply(q_des, r_mouth)
        return p_des, q_des

    def servo_step(theta: float) -> tuple[float, float]:
        """One held-pose wrench step toward pose_des(theta); returns the (position,
        rotation) tracking errors BEFORE the step (the divergence-guard signals —
        the ROTATION loop diverges ~sqrt(kq/I) ~ 50 rad/s, far faster than the
        position loop, so guarding position alone lets a wrong frame mode spin the
        bottle and centrifuge the marble out before any position error shows)."""
        p_des, q_des = pose_des(theta)
        pb, vb = bottle.data.root_pos_w, bottle.data.root_lin_vel_w
        qb, wb = bottle.data.root_quat_w, bottle.data.root_ang_vel_w
        inside = scene.marble_in_bottle().float().unsqueeze(-1)
        err = float((p_des - pb).norm(dim=-1)[0])

        F = 50.0 * (p_des - pb) - 10.0 * vb
        F = F + ez * ((m_b + m_m * inside) * G)
        F = clamp_norm(F, 12.0)

        q_err = scene_mod._qmul(q_des, qb * NEG)
        q_err = torch.where(q_err[:, :1] < 0, -q_err, q_err)
        xyz = q_err[:, 1:4]
        sn = xyz.norm(dim=-1, keepdim=True)
        rotvec = xyz / sn.clamp_min(1e-9) * (2.0 * torch.atan2(sn, q_err[:, :1]))
        err_r = float(rotvec.norm(dim=-1)[0])
        tau_ff = torch.cross(marble.data.root_pos_w - pb, ez * (m_m * G), dim=-1) * inside
        T = 0.8 * rotvec - 0.02 * wb + tau_ff
        T = clamp_norm(T, 1.0)

        bottle.set_external_force_and_torque(
            encode(F).view(n, 1, 3), encode(T).view(n, 1, 3),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        return err, err_r

    def guard(errs: tuple[float, float], lim_p: float, lim_r: float,
              lim_v: float, lim_w: float) -> bool:
        ep, er = errs
        return (ep > lim_p or er > lim_r
                or float(bottle.data.root_lin_vel_w[0].norm()) > lim_v
                or float(bottle.data.root_ang_vel_w[0].norm()) > lim_w)

    # ----- transport teleports --------------------------------------------------------------
    def hover_teleport() -> None:
        """Transport the bottle (and, if inside, the marble — relative pose preserved)
        to the upright hover reference over the tray. Zero velocities."""
        clear_force(bottle)
        inside = bool(scene.marble_in_bottle()[0])
        mloc = scene._bottle_local(marble.data.root_pos_w).clone() if inside else None
        p_des, q_des = pose_des(0.0)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3], st[:, 3:7] = p_des, q_des
        bottle.write_root_state_to_sim(st, all_ids)
        if inside:
            stm = torch.zeros(n, 13, device=device)
            stm[:, 0:3] = p_des + quat_apply(q_des, mloc)
            stm[:, 3] = 1.0
            marble.write_root_state_to_sim(stm, all_ids)
        env.iscene.update(0.0)

    def reinsert_marble() -> None:
        """Recovery from a PROBE ACCIDENT only: a diverging frame probe spun the
        bottle and threw the marble out. Put the marble back inside the (already
        re-teleported, upright) bottle and UN-EARN every latch the accident granted
        — the accidental exit is not a pour; keeping that credit would be a lie."""
        pb, qb = bottle.data.root_pos_w, bottle.data.root_quat_w
        mloc = torch.tensor(
            [0.0, 0.0, c.floor_t + c.marble_r + 0.004 - c.bottle_zmid],
            device=device).expand(n, 3)
        stm = torch.zeros(n, 13, device=device)
        stm[:, 0:3] = pb + quat_apply(qb, mloc)
        stm[:, 3] = 1.0
        marble.write_root_state_to_sim(stm, all_ids)
        env.iscene.update(0.0)
        scene._decant[:] = False
        scene._tray_ever[:] = False
        scene._seated[:] = False
        print("[solve] recovery: marble re-inserted, accidental latch credit "
              "cleared", flush=True)

    def yaw_of(q) -> float:
        return math.atan2(2.0 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                          1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2))

    def kick_probe() -> None:
        """Measure the pod's wrench frame directly, without ever risking the marble.
        From rest at the upright hover apply weight-cancelling ff (vertical — a pure
        yaw drag cannot corrupt it, the bottle is upright and R_ref is an upright
        spawn pose) plus a small horizontal kick along world +x, torque-free, and
        read the direction the bottle actually travels: that angle IS the drag yaw
        of R_now*R_ref^T. Build q_ref = Rz(yaw_now - psi) so mode 1 cancels the
        drag exactly. A well-behaved pod measures psi ~ 0 and starts in mode 0;
        the ambiguity left at psi ~ 0 (no drag vs R_ref ~ hover yaw — identical
        here, invisible upright) is resolved by tilt_probe."""
        p0 = bottle.data.root_pos_w.clone()
        inside = scene.marble_in_bottle().float().unsqueeze(-1)
        ff = ez * ((m_b + m_m * inside) * G)
        for _ in range(20):
            bottle.set_external_force_and_torque(
                (ff + ex * 0.6).view(n, 1, 3), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_force(bottle)
        d = (bottle.data.root_pos_w - p0)[0]
        psi = math.atan2(float(d[1]), float(d[0]))
        yaw_ref = yaw_of(bottle.data.root_quat_w[0]) - psi
        q_ref = torch.zeros(n, 4, device=device)
        q_ref[:, 0] = math.cos(yaw_ref / 2.0)
        q_ref[:, 3] = math.sin(yaw_ref / 2.0)
        state["q_ref"] = q_ref
        state["mode"] = 0 if abs(psi) < math.radians(10.0) else 1
        print(f"[solve] force-frame kick probe: drag yaw {math.degrees(psi):+.1f} deg "
              f"(|d|={float(d.norm()):.3f} m) -> start mode {state['mode']}",
              flush=True)
        hover_teleport()

    def recover_and_toggle(tag: str) -> None:
        clear_force(bottle)
        print(f"[solve] {tag} diverged (mode {state['mode']} -> "
              f"{state['mode'] ^ 1}); recovering", flush=True)
        state["mode"] ^= 1
        hover_teleport()
        if not bool(scene.marble_in_bottle()[0]):
            reinsert_marble()

    def engage_hover() -> bool:
        """(Re)engage the held pose at the upright hover. Tight bails — 0.04 m,
        0.35 rad, 0.8 m/s, 6 rad/s — abort a wrong mode within a few steps, long
        before the marble could be centrifuged out. Toggle-and-retry, recoverable."""
        for attempt in range(3):
            ok = True
            for i in range(60):
                if guard(servo_step(0.0), 0.04, 0.35, 0.8, 6.0):
                    ok = False
                    break
            if ok and servo_step(0.0)[0] < 0.03:
                return True
            recover_and_toggle("hover hold")
        return False

    def tilt_probe() -> bool:
        """Roll the held bottle to 35 deg and back — far below the discharge angle,
        the marble provably stays in the cavity — to expose a wrong force-frame mode
        that is invisible at the upright hover. Toggle and retry once."""
        for attempt in range(2):
            ok = True
            for i in range(180):
                th = math.radians(35.0) * min(1.0, i / 90.0)
                if guard(servo_step(th), 0.10, 0.5, 1.2, 8.0):
                    ok = False
                    break
            if ok:
                for i in range(120):
                    th = math.radians(35.0) * max(0.0, 1.0 - i / 90.0)
                    servo_step(th)
                return bool(scene.marble_in_bottle()[0])
            recover_and_toggle("tilt probe")
            if not engage_hover():
                return False
        return False

    # ----- the pour --------------------------------------------------------------------------
    DTH = math.radians(26.0) / 120.0      # tilt ramp rate per step
    RUNGS = [math.radians(a) for a in (115.0, 135.0, 155.0, 170.0)]

    def run_pour() -> tuple[str, float]:
        """Ramp through the tilt ladder, dwell at each rung, hold through the landing.
        Returns (status, theta): 'landed' | 'diverged' | 'stuck' | 'lost'."""
        theta = 0.0
        discharged = False
        for rung in RUNGS:
            while theta < rung and not discharged:
                theta = min(theta + DTH, rung)
                if guard(servo_step(theta), 0.10, 0.6, 1.5, 10.0):
                    return "diverged", theta
                discharged = not bool(scene.marble_in_bottle()[0])
            if not discharged:
                for _ in range(150):  # dwell 1.25 s at the rung
                    if guard(servo_step(theta), 0.10, 0.6, 1.5, 10.0):
                        return "diverged", theta
                    if not bool(scene.marble_in_bottle()[0]):
                        discharged = True
                        break
            if discharged:
                print(f"[solve] pour: discharge at theta={math.degrees(theta):.1f} deg",
                      flush=True)
                break
        if not discharged:
            return "stuck", theta
        # hold the pose while the marble lands and settles in the tray
        settled_run = 0
        for _ in range(720):
            if guard(servo_step(theta), 0.10, 0.6, 1.5, 10.0):
                return "diverged", theta
            ok = bool(scene.marble_in_tray()[0]) and bool(scene._settled(marble)[0])
            settled_run = settled_run + 1 if ok else 0
            if settled_run >= 30:
                return "landed", theta
        # marble discharged but never came to rest inside the tray
        return ("landed", theta) if bool(scene.marble_in_tray()[0]) else ("lost", theta)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rig.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    bx, by, bz = rig_xyz(bottle)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"bottle_rig=({bx:+.3f},{by:+.3f},{bz:+.3f})", flush=True)
    report("reset")
    for body in (bottle, marble):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    # custom-spawner MassAPI actually applied (cfg schemas are NOT auto-applied)
    mb = float(bottle.root_physx_view.get_masses().reshape(-1)[0])
    mm = float(marble.root_physx_view.get_masses().reshape(-1)[0])
    print(f"[solve] masses: bottle={mb:.4f} kg marble={mm:.4f} kg", flush=True)
    assert abs(mb - m_b) < 0.02 and abs(mm - m_m) < 0.005, "authored masses missing"
    assert abs(bx - c.bottle_start[0]) < c.bottle_jitter + 0.02, "bottle start x"
    assert abs(by - c.bottle_start[1]) < c.bottle_jitter + 0.02, "bottle start y"
    assert bool(scene.marble_in_bottle()[0]), "marble must start inside the bottle"
    s0 = print_score("P0 reset+settle (marble sealed in the bottle)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: transport to hover, engage + probe the held pose -------------
    hover_teleport()
    kick_probe()
    if not engage_hover():
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (could not hold the hover in either frame mode)",
              flush=True)
        os._exit(1)
    if not tilt_probe():
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (tilt probe failed / marble lost)", flush=True)
        os._exit(1)
    report("hover")
    assert bool(scene.marble_in_bottle()[0]), "marble must still be inside at hover"
    assert bool(scene._lifted[0]) and bool(scene._hover[0]), "lift/hover latches"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 bottle held hovering over the tray (frame mode "
                     f"{state['mode']})")
    assert s1 >= 0.19, f"P1 score {s1} (expect 0.20)"

    # ---------------- phase 2: the aimed pour ----------------------------------------------
    status, theta = run_pour()
    if status == "diverged":
        print(f"[solve] pour: DIVERGENCE at theta={math.degrees(theta):.1f} deg — "
              f"toggling frame mode and retrying from the hover", flush=True)
        clear_force(bottle)
        if bool(scene.marble_in_tray()[0]):
            status = "landed"  # the marble made it out and in despite the drag
        else:
            recover_and_toggle("pour")  # re-inserts + un-earns credit if thrown
            if engage_hover():
                status, theta = run_pour()
    if status != "landed":
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL (pour {status})", flush=True)
        os._exit(1)
    report("poured")
    assert bool(scene._decant[0]), "decant latch must have fired"
    assert bool(scene._tray_ever[0]), "marble-in-tray latch must have fired"
    s2 = print_score("P2 marble poured out of the mouth, at rest in the tray")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: re-erect, transport to the socket, drop-in -------------------
    DTH2 = math.radians(40.0) / 120.0
    while theta > 0.0:
        theta = max(theta - DTH2, 0.0)
        servo_step(theta)  # empty bottle: divergence no longer load-bearing
    for _ in range(30):
        servo_step(0.0)

    def socket_drop(dx: float, dy: float, dz: float) -> None:
        clear_force(bottle)
        q_rig = scene.rig.data.root_quat_w
        loc = torch.tensor([c.sock_x + dx, -c.tray_y + dy,
                            c.sock_plate + c.bottle_zmid + dz], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = quat_apply(q_rig, loc) + scene.rig.data.root_pos_w
        st[:, 3:7] = q_rig  # yaw-aligned: the square base only enters un-rotated
        bottle.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)
        step(150)

    seated = False
    dx = dy = 0.0
    for attempt in range(4):
        socket_drop(dx, dy, 0.020 if attempt == 0 else 0.012)
        seated = bool(scene.bottle_seated()[0]) and bool(scene._settled(bottle)[0])
        bxx, byy, bzz = rig_xyz(bottle)
        print(f"[solve] seat attempt {attempt}: bottle=({bxx:+.3f},{byy:+.3f},{bzz:+.3f}) "
              f"seated={seated}", flush=True)
        if seated:
            break
        # measured-offset correction from where the base actually rests
        up = scene.bottle_up()
        base = scene._rig_local(bottle.data.root_pos_w - up * c.bottle_zmid)[0]
        dx += 0.6 * (c.sock_x - float(base[0]))
        dy += 0.6 * (-c.tray_y - float(base[1]))
    if not seated:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (bottle never seated in the socket)", flush=True)
        os._exit(1)
    report("seated")
    s3 = print_score("P3 empty bottle dropped-in and seated in the socket")
    assert s3 >= s2 - 1e-6 and s3 >= 0.69, f"P3 score {s3} (expect >= 0.70)"

    # ---------------- phase 4: judged state, live ------------------------------------------
    step(60)
    report("judge")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success at the judged state)", flush=True)
        os._exit(1)
    s4 = print_score("P4 success live (marble in tray, bottle seated, decant observed)")
    assert s4 >= 0.99, f"P4 score {s4} (expect 1.0)"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        print(f"[solve] EXCEPTION: {exc!r}", flush=True)
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
