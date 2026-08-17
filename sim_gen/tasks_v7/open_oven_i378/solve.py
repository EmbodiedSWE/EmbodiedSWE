"""solve — TELEPORT-contract solution for SpitRoastScene (open_oven_i378).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY: exactly one root-pose
write, parking the spit rod in FREE AIR collinear with the block's bore axis, 280 mm
short of the bore, at bore height, zero velocity (the arm's carry-and-present). Every
load-bearing interaction runs through live contact dynamics via the scene's external
wrench buffers (`spit_force`/`spit_torque` — the stand-in for the Franka's grasp wrench
on the knob; the ROAST buffers are never touched, the block is fully passive):

  THREAD   a lag-clamped carrot walks the rod's bare tip down the bore axis at 50 mm/s;
           the 12 mm rod must actually pass through the 26 mm square bore, pushing the
           block against the cradle's backstop, until the tip protrudes past both faces
           and the rod is centred in the block (knob side out — the 32 mm knob cannot
           pass, so the threading direction is physically forced);
  LIFT     raise the loaded spit; the block is picked up BY THE ROD (its bore top rides
           the rod surface) — the 0.7 kg load feedforward gates on the block actually
           rising, and a gravity-moment feedforward cancels the hanging load's pitch;
  TRAVERSE carry to above the rack, slewing the rod axis to the rack's x axis at a
           rate-limited 0.5 rad/s (carry height 0.30 m clears the funnel tops 0.267 m;
           the hanging block passes BETWEEN the notch assemblies);
  SEAT     descend at 30 mm/s; the 45-deg funnel plates gather the rod ends into the two
           15 mm slots; when both ends read seated the wrench ramps to zero over 30
           steps — gravity keeps the spit in its seats, the meat hanging between the
           posts. Nothing is pinned, no velocity is written, no rubric state is touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (never decreasing — asserted).
After success() first holds, keeps simulating >= 3.5 more simulated seconds with ALL
wrench buffers zero (asserted); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer as
backstop.

Run (forge): python -u -m simgen_tasks.open_oven_i378.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.open_oven_i378 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# --- grasp-wrench servo gains (the Franka-authority stand-in) ------------------------------------
# Force: PD + gravity FF on a 0.4 kg rod (1.1 kg loaded). KD*dt/m = 18/120/0.4 = 0.37 < 1
# (one-substep wrench delay), zeta = 1.5 unloaded / 0.9 loaded. Cap 30 N ~ a gentle
# single-arm carry (the loaded spit weighs 10.8 N).
KP, KD, F_CAP = 90.0, 18.0, 30.0
# Attitude: PD on the rod axis + hanging-load gravity-moment FF. I_transverse = 6.4e-3:
# zeta ~ 1.0, residual carry tilt < 2 deg — under the mu=0.15 slide angle (8.5 deg), so
# the block does NOT slide along the rod. Cap 1.2 N*m ~ two-finger wrist authority.
KA, KW, T_CAP = 3.0, 0.28, 1.2
LEASH = 0.03          # carrot never leads the rod by more than this (m)
V_THREAD = 0.05       # carrot rates (m/s)
V_LIFT = 0.08
V_MOVE = 0.12
V_SEAT = 0.03
W_SLEW = 0.5          # axis slew rate (rad/s)
CARRY_Z = 0.30        # carry height (asserted above the funnel tops)
G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spit_roast")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    dt = env.dt
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)
    assert CARRY_Z > c.funnel_top + c.rod_r + 0.02, "carry height must clear the funnels"

    from isaaclab.utils.math import quat_apply

    def rod_state():
        p = scene.spit.data.root_pos_w[0] - scene.env_origins[0]
        v = scene.spit.data.root_lin_vel_w[0]
        w = scene.spit.data.root_ang_vel_w[0]
        d = quat_apply(scene.spit.data.root_quat_w[0:1], ex.unsqueeze(0))[0]
        return p, v, w, d

    def roast_state():
        p = scene.roast.data.root_pos_w[0] - scene.env_origins[0]
        a = quat_apply(scene.roast.data.root_quat_w[0:1], ex.unsqueeze(0))[0]
        return p, a

    def carrying() -> bool:
        p_r, _ = roast_state()
        return bool(scene.threaded()[0]) and float(p_r[2]) > c.roast_rest_z + 0.005

    def servo_step(p_ref: torch.Tensor, d_ref: torch.Tensor, scale: float = 1.0,
                   load_ff: bool = False) -> None:
        """One physics step under the grasp-wrench servo (the ONLY writer of the spit
        buffers). `scale` ramps the whole wrench for the release. `load_ff` adds the
        block's weight to the gravity feedforward (the P-term alone cannot lift the
        0.7 kg block, so the lift phases must REQUEST the load — gating it on the block
        already rising is chicken-and-egg; observed stall at exactly the 7 mm bore
        clearance)."""
        p, v, w, d = rod_state()
        carry = carrying()
        m_ff = c.rod_mass + (c.roast_mass if (carry or (load_ff and bool(scene.threaded()[0])))
                             else 0.0)
        err = (p_ref - p).clamp(-LEASH, LEASH)
        f = KP * err - KD * v + m_ff * G * ez
        fn = float(f.norm())
        if fn > F_CAP:
            f = f * (F_CAP / fn)
        e = torch.linalg.cross(d, d_ref)
        w_t = w - (w @ d) * d
        tau = KA * e - KW * w_t
        if carry:
            # cancel the hanging block's pitch moment about the rod CoM (at +com_x)
            p_r, _ = roast_state()
            s_off = float((p_r - p) @ d) - c.rod_com_x
            tau = tau + torch.linalg.cross(s_off * d, c.roast_mass * G * ez)
        tn = float(tau.norm())
        if tn > T_CAP:
            tau = tau * (T_CAP / tn)
        scene.spit_force[0] = scale * f
        scene.spit_torque[0] = scale * tau
        env.step(no_action)

    def hands_off(k: int) -> None:
        scene.spit_force[0] = 0.0
        scene.spit_torque[0] = 0.0
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        p, v, _, d = rod_state()
        p_r, _ = roast_state()
        print(f"[solve] {tag:12s} rod=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
              f"axis=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}) roast_z={p_r[2]:.3f} "
              f"ent={int(scene.entered()[0])} thr={int(scene.threaded()[0])} "
              f"seat=({int(scene.seated(0)[0])},{int(scene.seated(1)[0])}) "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ================= PHASE 0: reset + settle ====================================================
    env.reset(seed=args.seed)
    hands_off(90)
    report("reset")
    p_r0, a0 = roast_state()
    assert abs(float(p_r0[2]) - c.roast_rest_z) < 0.01, "block must rest on the cradle"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: TRANSPORT (the one teleport — free air, zero velocity) ===========
    a = a0.clone()
    a[2] = 0.0
    a = a / a.norm()  # bore axis, horizontal projection (block sits flat on the cradle)
    park = p_r0 + 0.28 * a
    park[2] = p_r0[2]
    yaw = math.atan2(float(a[1]), float(a[0]))
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = scene.env_origins[0] + park
    st[0, 3] = math.cos(yaw / 2.0)
    st[0, 6] = math.sin(yaw / 2.0)
    scene.spit.write_root_state_to_sim(st, torch.tensor([0], device=device))
    for _ in range(60):  # stabilize the hover under the servo
        servo_step(park, a)
    report("parked")
    phase_score("parked")  # still ~0.000

    # ================= PHASE 2: THREAD (contact insertion through the bore) ======================
    s_ref = 0.28
    done = False
    for i in range(2400):
        p_r, a_r = roast_state()
        a_r[2] = 0.0
        a_r = a_r / a_r.norm()
        if float(a_r @ a) < 0.0:
            a_r = -a_r
        a = a_r  # track the block's live bore axis
        p, _, _, _ = rod_state()
        s_now = float((p - p_r) @ a)
        s_ref = max(0.005, min(s_ref - V_THREAD * dt, s_now + 0.005))
        s_ref = max(s_ref, s_now - LEASH)  # lag clamp: never drag through a jam
        servo_step(p_r + s_ref * a, a)
        if bool(scene.threaded()[0]) and s_now <= 0.010:
            done = True
            break
        if i % 300 == 299:
            report(f"thread{i}")
    report("threaded")
    if not done:
        print("[solve] PHASE 2 FAILED: rod not threaded through the bore", flush=True)
        verdict(False)
    phase_score("threaded")  # 0.400

    # ================= PHASE 3: LIFT (the block is picked up BY the rod) =========================
    p, _, _, _ = rod_state()
    hold_xy = p.clone()
    z_ref = float(p[2])
    done = False
    for _ in range(1200):
        z_ref = min(CARRY_Z, z_ref + V_LIFT * dt)
        p_ref = hold_xy.clone()
        p_ref[2] = z_ref
        servo_step(p_ref, a, load_ff=True)
        p, v, _, _ = rod_state()
        z_ref = max(z_ref, float(p[2]) - LEASH)
        if float(p[2]) > CARRY_Z - 0.01 and carrying():
            done = True
            break
    report("lifted")
    if not done:
        print("[solve] PHASE 3 FAILED: loaded spit not lifted", flush=True)
        verdict(False)
    phase_score("lifted")  # 0.650

    # ================= PHASE 4: TRAVERSE to above the rack, axis -> rack x =======================
    tgt_sign = 1.0 if float(a[0]) >= 0.0 else -1.0
    yaw_ref = math.atan2(float(a[1]), float(a[0]))
    yaw_tgt = 0.0 if tgt_sign > 0 else math.pi
    goal = torch.tensor([c.rack_x, 0.0, CARRY_Z], device=device)
    p_ref = rod_state()[0].clone()
    done = False
    for _ in range(1800):
        dyaw = (yaw_tgt - yaw_ref + math.pi) % (2.0 * math.pi) - math.pi
        yaw_ref += max(-W_SLEW * dt, min(W_SLEW * dt, dyaw))
        d_ref = torch.tensor([math.cos(yaw_ref), math.sin(yaw_ref), 0.0], device=device)
        to_go = goal - p_ref
        dist = float(to_go.norm())
        if dist > 1e-6:
            p_ref = p_ref + to_go * min(1.0, V_MOVE * dt / dist)
        servo_step(p_ref, d_ref, load_ff=True)
        p, v, _, d = rod_state()
        p_ref = p + (p_ref - p).clamp(-LEASH, LEASH)
        if (dist < 0.01 and abs(dyaw) < 0.02
                and float((p - goal).norm()) < 0.02
                and float(torch.linalg.cross(d, d_ref).norm()) < 0.05
                and float(v.norm()) < 0.08):
            done = True
            break
    report("over-rack")
    if not done:
        print("[solve] PHASE 4 FAILED: not stabilized over the rack", flush=True)
        verdict(False)
    if not carrying():
        print("[solve] PHASE 4 FAILED: block lost during traverse", flush=True)
        verdict(False)
    phase_score("over-rack")  # 0.650 (latched)

    # ================= PHASE 5: SEAT (funnels gather the ends; ramped release) ===================
    d_ref = torch.tensor([math.cos(yaw_tgt), math.sin(yaw_tgt), 0.0], device=device)
    z_ref = CARRY_Z
    done = False
    for _ in range(1800):
        z_ref = max(c.z_seat - 0.004, z_ref - V_SEAT * dt)
        p_ref = torch.tensor([c.rack_x, 0.0, z_ref], device=device)
        servo_step(p_ref, d_ref, load_ff=True)
        p, _, _, _ = rod_state()
        z_ref = max(z_ref, float(p[2]) - LEASH)
        if (bool(scene.seated(0)[0]) and bool(scene.seated(1)[0])
                and abs(float(p[2]) - c.z_seat) < 0.004):
            done = True
            break
    report("seated")
    if not done:
        print("[solve] PHASE 5 FAILED: rod ends not seated in the slots", flush=True)
        verdict(False)
    for i in range(30):  # ramp the grasp wrench to zero — gravity owns the seat
        servo_step(torch.tensor([c.rack_x, 0.0, c.z_seat], device=device), d_ref,
                   scale=1.0 - (i + 1) / 30.0, load_ff=True)
    scene.spit_force[0] = 0.0
    scene.spit_torque[0] = 0.0

    # ================= PHASE 6: settle to success =================================================
    ok = False
    for _ in range(72):  # up to 6 s
        hands_off(10)
        if bool(scene.success()[0]):
            ok = True
            break
    report("released")
    if not ok:
        print("[solve] PHASE 6 FAILED: success() not reached hands-off", flush=True)
        verdict(False)
    phase_score("success")  # 1.000

    # ================= PHASE 7: persistence (>= 3.5 simulated seconds, hands off) ================
    assert float(scene.spit_force.abs().max()) == 0.0
    assert float(scene.spit_torque.abs().max()) == 0.0
    assert float(scene.roast_force.abs().max()) == 0.0
    assert float(scene.roast_torque.abs().max()) == 0.0
    persist = int(round(3.5 / dt))
    hands_off(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist} steps ({persist * dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
