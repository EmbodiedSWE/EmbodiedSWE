"""Teleport solution for CarafePourScene (sim_gen task
`living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i316`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): ONE root-state write carries the red-can CARAFE — with the
   can still seated inside its bore, relative pose preserved, zero velocity — from
   its ground slot to a hover pose beside/above the tray. This is what a grasp-and-
   carry delivers. At the write the can is still captive in the bore and inside NO
   rubric credit volume (asserted; the score is unchanged).
2. HOLD (applied wrench): a 6-DOF overdamped servo wrench at the carafe's CoM — the
   force/torque a Franka wrist applies through a grasp on the 70 mm flats (force
   capped 12 N, torque capped 1.0 N m) — holds the hover. FORCE-FRAME GUARD: some
   pods rotate an applied "global" wrench by the body's rotation since reset; the
   hold PROBES the convention at runtime and toggles `encode_force` mode if the
   carafe accelerates away.
3. POUR (applied wrench + gravity + contact): the same servo rolls the carafe about
   the tray's y axis while a MOUTH-ANCHORED trajectory lowers the mouth over the
   tray interior. Past ~97 deg the slick bore lets the can slide out under gravity
   — the can's exit, fall and landing in the tray are pure ballistics and contact,
   never written. If the can has not exited at 105 deg the reference escalates
   (120, 135, 150 deg) until it does — outcome-driven, servo-sag tolerant.
4. RETRACT + SET-DOWN (applied wrench): the servo rights the carafe, carries it
   back over free ground and lowers it onto its vacated spawn slot, then releases.
   Every success clause (can in tray, in no carafe, no carafe in the tray, carafes
   grounded, settled) is then a live, settled contact outcome.
5. IDENTITY: only the red-can carafe is ever touched. The corn can never leaves its
   carafe.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i316.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, _qinv, _qmul, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, _qmul, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81


def rotvec_to(q_des: torch.Tensor, q_now: torch.Tensor) -> torch.Tensor:
    """Hemisphere-fixed quaternion error q_des * q_now^-1 as a rotation vector."""
    qe = _qmul(q_des, _qinv(q_now))
    qe = torch.where(qe[:, :1] < 0, -qe, qe)
    ang = 2.0 * torch.acos(qe[:, 0].clamp(-1.0, 1.0))
    ax = qe[:, 1:]
    axn = ax.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return ax / axn * ang.unsqueeze(-1)


def q_aa(axis: torch.Tensor, ang: float) -> torch.Tensor:
    """Quaternion for `ang` radians about the (unit, world) `axis` rows, (N,3)->(N,4)."""
    q = torch.zeros(axis.shape[0], 4, device=axis.device)
    q[:, 0] = math.cos(ang / 2.0)
    q[:, 1:] = axis * math.sin(ang / 2.0)
    return q


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carafe_pour")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    red = scene.alphabet

    def report(tag: str) -> None:
        rp = red.data.root_pos_w[0] - scene.env_origins[0]
        tl = scene._tray_local(red.data.root_pos_w)[0]
        jz = [float(j.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
              for j in scene.jars]
        print(f"[solve] {tag:14s} | can_w=({float(rp[0]):+.3f},{float(rp[1]):+.3f},"
              f"{float(rp[2]):+.3f}) can_tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):+.3f}) in_jar={bool(scene.in_any_jar(red.data.root_pos_w)[0])} "
              f"in_tray={bool(scene.in_tray(red.data.root_pos_w)[0])} "
              f"jz=({jz[0]:.3f},{jz[1]:.3f}) dec={bool(scene._decanted[0])} "
              f"del={bool(scene._delivered[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # cans seat in the bores, carafes/tray seat on the ground
    # custom spawn funcs ignore cfg mass schemas — assert the authored masses took
    jm = float(scene.jar_a.root_physx_view.get_masses().sum())
    tm = float(scene.tray.root_physx_view.get_masses().sum())
    am = float(red.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: carafe={jm:.3f} tray={tm:.3f} can={am:.3f} kg",
          flush=True)
    assert 0.18 < jm < 0.35, f"carafe mass wrong: {jm}"
    assert 0.95 < tm < 1.50, f"tray mass wrong: {tm}"
    assert 0.22 < am < 0.40, f"can mass wrong: {am}"
    m_jar, m_can = jm, am

    red_jar_i = int(scene.red_jar[0])
    jar = scene.jars[red_jar_i]
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    tq = scene.tray.data.root_quat_w[0]
    tyaw = math.degrees(2.0 * math.atan2(float(tq[3]), float(tq[0])))
    japos = (scene.jar_a.data.root_pos_w - scene.env_origins)[0]
    jbpos = (scene.jar_b.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) yaw={tyaw:+.1f}deg "
          f"jar_a=({float(japos[0]):+.3f},{float(japos[1]):+.3f}) "
          f"jar_b=({float(jbpos[0]):+.3f},{float(jbpos[1]):+.3f}) "
          f"red_jar={red_jar_i}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.in_jar(jar, red.data.root_pos_w)[0]), \
        "red can must start captive in its carafe"
    assert bool(scene.in_jar(scene.jars[1 - red_jar_i],
                             scene.corn.data.root_pos_w)[0]), \
        "corn can must start captive in the other carafe"
    assert not bool(scene.in_tray(red.data.root_pos_w)[0]), \
        "red can must start outside the tray"
    s0 = print_score("P0 reset+settle (cans captive in the carafes)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # record the red carafe's vacated slot for the set-down
    slot_w = jar.data.root_pos_w[0].clone()

    # ---------------- 6-DOF carry/pour servo -------------------------------------------------
    kp, kd = 50.0, 10.0
    kq, kdw = 0.8, 0.08
    state = {"mode": 0, "q_ref": jar.data.root_quat_w.clone()}
    com_local = torch.tensor([[0.0, 0.0, 0.050]], device=device).expand(n, 3)

    def hold_step(p_des: torch.Tensor, q_des: torch.Tensor) -> None:
        """One servo step: CoM force (position PD + live gravity ff for carafe +
        captive can, capped 12 N) + CoM torque (orientation PD + can-offset gravity
        moment ff, capped 1.0 N m) — the wrench a wrist applies through a grasp."""
        p = jar.data.root_pos_w
        v = jar.data.root_lin_vel_w
        q = jar.data.root_quat_w
        w = jar.data.root_ang_vel_w
        in_j = scene.in_jar(jar, red.data.root_pos_w).float()
        f = kp * (p_des - p) - kd * v
        f[:, 2] += (m_jar + m_can * in_j) * G
        fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f = f * (fn.clamp(max=12.0) / fn)
        tq_c = kq * rotvec_to(q_des, q) - kdw * w
        # can-weight moment about the carafe CoM while the can rides inside
        com_w = p + _qapply(q, com_local)
        r = red.data.root_pos_w - com_w
        f_can = torch.zeros(n, 3, device=device)
        f_can[:, 2] = -m_can * G
        tq_c = tq_c + torch.cross(r, f_can, dim=-1) * in_j.unsqueeze(-1)
        tn = tq_c.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tq_c = tq_c * (tn.clamp(max=1.0) / tn)
        fe = encode_force(state["mode"], state["q_ref"], q, f).view(n, 1, 3)
        te = encode_force(state["mode"], state["q_ref"], q, tq_c).view(n, 1, 3)
        jar.set_external_force_and_torque(fe, te, env_ids=all_ids, is_global=True)
        env.step(no_action)

    def release_wrench() -> None:
        jar.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 1: TRANSPORT the carafe (+ captive can) to a tray-side hover ----
    tray_p = scene.tray.data.root_pos_w.clone()
    tray_q = scene.tray.data.root_quat_w.clone()
    y_axis_w = quat_apply(tray_q, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))

    def tray_to_w(loc) -> torch.Tensor:
        v = torch.tensor([loc], device=device, dtype=torch.float).expand(n, 3)
        return tray_p + _qapply(tray_q, v)

    hover_p = tray_to_w((-0.070, 0.0, 0.230))
    hover_q = tray_q.clone()  # upright, tray yaw

    def teleport_to_hover() -> None:
        """One rigid transport write: carafe -> hover, can carried with it (relative
        pose preserved), both zero velocity."""
        jp, jq = jar.data.root_pos_w.clone(), jar.data.root_quat_w.clone()
        rel_p = _qapply(_qinv(jq), red.data.root_pos_w - jp)
        rel_q = _qmul(_qinv(jq), red.data.root_quat_w)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hover_p
        st[:, 3:7] = hover_q
        jar.write_root_state_to_sim(st, all_ids)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hover_p + _qapply(hover_q, rel_p)
        st[:, 3:7] = _qmul(hover_q, rel_q)
        red.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    teleport_to_hover()
    # the write itself grants nothing: can captive, outside every credit volume
    assert bool(scene.in_jar(jar, red.data.root_pos_w)[0]), \
        "can must still be captive after the transport write"
    assert not bool(scene.in_tray(red.data.root_pos_w)[0]), \
        "hover must be OUTSIDE the tray interior volume"
    assert not bool(scene._decanted[0]) and not bool(scene._delivered[0]), \
        "transport must not latch any credit"

    # engage the hold; PROBE the force-frame convention (retry once with the mode
    # toggled if the carafe accelerates away from the hover)
    ok_hold = False
    for attempt in range(2):
        state["q_ref"] = jar.data.root_quat_w.clone()
        for _ in range(50):
            hold_step(hover_p, hover_q)
        err = float((jar.data.root_pos_w - hover_p)[0].norm())
        if err < 0.06:
            ok_hold = True
            break
        print(f"[solve] hover hold diverged (err {err:.3f} m); force-frame mode "
              f"{state['mode']} -> {state['mode'] ^ 1}", flush=True)
        state["mode"] ^= 1
        teleport_to_hover()
    if not ok_hold:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (could not hold the hover)", flush=True)
        os._exit(1)

    # TILT PROBE: the upright hold cannot distinguish the two force-frame
    # conventions (the drag R_now R_ref^T is identity until the body rotates).
    # Roll the held carafe to 35 deg — far below the ~97 deg discharge angle, the
    # can provably stays in the bore — and watch tracking: a wrong mode rotates
    # the ~5.4 N gravity feedforward by the tilt (≈3 N lateral at 35 deg) and the
    # position error blows up within ~0.5 s. Toggle the mode and retry if so.
    tray_p_now = scene.tray.data.root_pos_w.clone()
    y_axis_probe = quat_apply(scene.tray.data.root_quat_w,
                              torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    ok_tilt = False
    for attempt in range(2):
        tilt_ok = True
        for i in range(180):
            th = math.radians(35.0) * min(1.0, i / 90.0)
            q_des = _qmul(q_aa(y_axis_probe, th), hover_q)
            hold_step(hover_p, q_des)
            if float((jar.data.root_pos_w - hover_p)[0].norm()) > 0.10:
                tilt_ok = False
                break
        if tilt_ok:
            # roll back upright and restabilize
            for i in range(120):
                th = math.radians(35.0) * max(0.0, 1.0 - i / 90.0)
                q_des = _qmul(q_aa(y_axis_probe, th), hover_q)
                hold_step(hover_p, q_des)
            ok_tilt = True
            break
        print(f"[solve] tilt probe diverged — force-frame mode "
              f"{state['mode']} -> {state['mode'] ^ 1}", flush=True)
        state["mode"] ^= 1
        if not bool(scene.in_jar(jar, red.data.root_pos_w)[0]):
            report("FAIL-state")
            print("SIM_GEN_SOLVE: FAIL (can ejected during the tilt probe)", flush=True)
            os._exit(1)
        teleport_to_hover()
        state["q_ref"] = jar.data.root_quat_w.clone()
        for _ in range(50):
            hold_step(hover_p, hover_q)
    if not ok_tilt:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (tilt probe failed in both force-frame modes)",
              flush=True)
        os._exit(1)
    assert bool(scene.in_jar(jar, red.data.root_pos_w)[0]), \
        "can must survive the 35 deg tilt probe captive"
    report("hover")
    assert bool(scene.in_jar(jar, red.data.root_pos_w)[0]), "can left the bore at hover?"
    s1 = print_score("P1 carafe transported to the tray-side hover (can still captive)")
    assert s1 <= s0 + 1e-6, "transport must not add credit"
    assert not bool(scene.success()[0]), "cannot be success with the can captive"

    # ---------------- phase 2: POUR — mouth-anchored roll about the tray y axis -------------
    m0 = (-0.070, 0.0, 0.388)  # mouth start (= hover mouth), tray frame
    m1 = (0.0, 0.0, 0.135)     # mouth end: low over the tray interior
    rim = torch.tensor([[0.0, 0.0, c.rim_z]], device=device).expand(n, 3)

    def pour_ref(t: float, theta: float):
        mt = tuple(m0[k] + (m1[k] - m0[k]) * min(1.0, t) for k in range(3))
        q_des = _qmul(q_aa(y_axis_w, theta), hover_q)
        p_des = tray_to_w(mt) - _qapply(q_des, rim)
        return p_des, q_des

    def run_pour() -> tuple[bool, bool, float]:
        """Escalating pour. Returns (exited, diverged, theta_end). `diverged` fires
        when the servo position error blows past 0.15 m — the signature of a WRONG
        force-frame mode, which is invisible at the upright hover (the frame drag
        R_ref R_now^T is identity until the body rotates) and only bites mid-roll."""
        exited = False
        theta_now = 0.0
        for theta_max_deg in (105.0, 120.0, 135.0, 150.0):
            theta_max = math.radians(theta_max_deg)
            ramp = max(2, int((theta_max - theta_now) / math.radians(105.0) * 480))
            for i in range(ramp):
                t_frac = i / (ramp - 1)
                th = theta_now + (theta_max - theta_now) * t_frac
                # anchor progress tied to overall tilt progress (0..1 at 105 deg)
                p_des, q_des = pour_ref(th / math.radians(105.0), th)
                hold_step(p_des, q_des)
                # wrong-mode drag crosses 0.10 m tracking error by ~70-90 deg,
                # safely BEFORE the ~97 deg discharge; correct-mode error ~0.03
                if float((jar.data.root_pos_w - p_des)[0].norm()) > 0.10:
                    return False, True, th
                if not bool(scene.in_any_jar(red.data.root_pos_w)[0]):
                    return True, False, th
            theta_now = theta_max
            # hold at this tilt and give the can time to slide
            p_des, q_des = pour_ref(1.0, theta_max)
            for _ in range(240):
                hold_step(p_des, q_des)
                if float((jar.data.root_pos_w - p_des)[0].norm()) > 0.10:
                    return False, True, theta_now
                if not bool(scene.in_any_jar(red.data.root_pos_w)[0]):
                    return True, False, theta_now
            print(f"[solve] can has not exited at {theta_max_deg:.0f} deg; "
                  f"escalating", flush=True)
        return exited, False, theta_now

    exited, diverged, theta_now = run_pour()
    if diverged:
        print(f"[solve] pour diverged at theta~{math.degrees(theta_now):.0f} deg — "
              f"suspect force-frame drag; mode {state['mode']} -> "
              f"{state['mode'] ^ 1}, restarting the pour", flush=True)
        state["mode"] ^= 1
        teleport_to_hover()
        state["q_ref"] = jar.data.root_quat_w.clone()
        for _ in range(50):
            hold_step(hover_p, hover_q)
        exited, diverged, theta_now = run_pour()
    if not exited:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (can never poured out"
              + (", pour diverged" if diverged else "") + ")", flush=True)
        os._exit(1)
    print(f"[solve] can exited the bore at theta~{math.degrees(theta_now):.0f} deg",
          flush=True)
    # keep holding the carafe still while the can lands and settles in the tray
    p_des, q_des = pour_ref(1.0, theta_now)
    landed = False
    for j in range(480):
        hold_step(p_des, q_des)
        if bool(scene._delivered[0]) \
                and float(red.data.root_lin_vel_w[0].norm()) < 0.05:
            landed = True
            break
        if j % 120 == 119:
            report(f"land-{j + 1}")
    report("poured")
    if not (landed or bool(scene._delivered[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (poured can did not settle in the tray)", flush=True)
        os._exit(1)
    assert bool(scene._decanted[0]), "decant latch must be set"
    s2 = print_score("P2 can poured out, at rest in the tray (carafe still held aloft)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.60 - 1e-4, f"P2 score {s2} (expect ~0.60)"
    assert not bool(scene.success()[0]), \
        "cannot be success while the carafe is held aloft"

    # ---------------- phase 3: RETRACT — right the carafe, set it down on its slot ----------
    # (a) reverse the pour: roll upright while the mouth retraces its path
    steps_a = 300
    for i in range(steps_a):
        t_frac = 1.0 - i / (steps_a - 1)
        th = theta_now * t_frac
        p_des, q_des = pour_ref(th / math.radians(105.0), th)
        hold_step(p_des, q_des)
    # (b) translate to a hover above the vacated slot
    start = jar.data.root_pos_w.clone()
    goal = start.clone()
    goal[:, 0] = slot_w[0]
    goal[:, 1] = slot_w[1]
    goal[:, 2] = scene.env_origins[:, 2] + 0.250
    steps_b = 300
    for i in range(steps_b):
        t_frac = (i + 1) / steps_b
        hold_step(start + (goal - start) * t_frac, hover_q)
    # (c) descend and set down
    low = goal.clone()
    low[:, 2] = scene.env_origins[:, 2] + 0.008
    steps_c = 300
    for i in range(steps_c):
        t_frac = (i + 1) / steps_c
        hold_step(goal + (low - goal) * t_frac, hover_q)
    release_wrench()
    step(240)  # hands-off settle
    report("set-down")
    if not bool(scene.success()[0]):
        # one more settle margin — slow creep on set-down
        step(240)
        report("set-down+")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (end state is not success)", flush=True)
        os._exit(1)
    assert bool(scene.jars_grounded()[0]), "both carafes must be grounded"
    s3 = print_score("P3 carafe righted and set down on its slot — success live")
    assert s3 >= 0.999, f"P3 score {s3} (expect 1.0)"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
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
    try:
        main()
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
