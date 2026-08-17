"""Teleport solution for SpindleRollScene (sim_gen task
`put_toilet_roll_on_stand_i188`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact dynamics:
  1. LOAD (transport + gravity): the roll is teleported to a hover 20 mm above the
     cradle's V, axis along the groove, face 2 mm from the backstop — then RELEASED.
     Gravity drops it into the V; the V centers it. Nothing is teleported into contact.
  2. THREAD (contact dynamics): the spindle is teleported to the groove axis with its
     leading ball 6 mm OUTSIDE the roll's near face (rubric reads un-threaded), then a
     regulated push — axial velocity-capped force + clamped lateral/vertical support
     PD + weak attitude PD (a rigid grasp, as a gripper would hold the rod) — drives
     it through the 40 mm core. The 13 mm radial passage tolerance is enforced by
     COLLISION (core walls, backstop notch), never by the wrench: the support gains
     are too weak to pull the rod through a wall. Once inside, the support target
     drops 9 mm so the leading ball rides the core bottom and exits through the
     backstop notch at its physical riding height. When the rod is axially centered
     the wrench is CUT: the rod falls 6 mm and rests on the core bottom — the settled
     threaded state is pure gravity + contact.
  3. CARRY (transport): rod and roll teleported TOGETHER, preserving their exact
     relative pose, to a hover 200 mm above the rack, rod axis along the slot line.
     The rod is held by a regulated wrench (gravity feed-forward that ramps up as the
     roll's weight transfers + PD + attitude PD); the roll is NEVER held — it falls
     26 mm relative to the rod and forms the hang (rod on core top) mid-air, by
     gravity and rod-core contact alone.
  4. MOUNT (contact dynamics): a velocity-regulated descent (-0.08 m/s) lowers the
     assembly; the funnels and slots guide the shaft. At 12 mm above the seats ALL
     wrenches are cut — the last drop into both seats, the ball-end axial retention,
     and the final hang are pure gravity + contact. success() reads the assembled,
     mounted, suspended, settled state. Nothing is welded, pinned, or held at the end.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it persists.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i188.solve --headless [--seed N]
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
    import scene as scene_mod

CAP_Z = scene_mod.CAP_Z
CAP_R = scene_mod.CAP_R
CORE_R = scene_mod.CORE_R
ROD_TIP = scene_mod.ROD_TIP
ROLL_HALF_W = scene_mod.ROLL_HALF_W
CRADLE_REST_Z = scene_mod.CRADLE_REST_Z
SEAT_Z = scene_mod.SEAT_Z

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spindle_roll")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    m_rod, m_roll, g = c.rod_mass, c.roll_mass, 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor, tau_w: torch.Tensor | None = None) -> None:
        """Apply a WORLD force/torque (n,3) to `body`, expressed in its CURRENT link
        frame (`is_global=True` silently drops torques on this stack — transform
        manually, the house convention). Re-set every step while pushing."""
        q = body.data.root_link_quat_w
        if tau_w is None:
            tau_w = torch.zeros(n, 3, device=device)
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            quat_apply_inverse(q, tau_w).unsqueeze(1),
            env_ids=all_ids)

    zero3 = torch.zeros(n, 3, device=device)

    def cut(body) -> None:
        wrench(body, zero3, zero3)

    def place(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def axis(body) -> torch.Tensor:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device).expand(n, 4)  # local +z -> local +x

    def report(tag: str) -> None:
        r, s = rel(scene.roll), rel(scene.spindle)
        radial, align, plo, phi = scene.thread_metrics()
        print(f"[solve] {tag:12s} | roll=({float(r[0]):+.3f},{float(r[1]):+.3f},"
              f"{float(r[2]):.3f}) rod=({float(s[0]):+.3f},{float(s[1]):+.3f},"
              f"{float(s[2]):.3f}) rad={float(radial[0]):.4f}"
              f" prot=({float(plo[0]):+.3f},{float(phi[0]):+.3f})"
              f" thr={bool(scene.threaded()[0])} seat={bool(scene.seated()[0])}"
              f" hang={bool(scene.hanging()[0])} set={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}"
              f" | rodv={float(scene.spindle.data.root_lin_vel_w[0].norm()):.4f}"
              f" rodw={float(scene.spindle.data.root_ang_vel_w[0].norm()):.4f}"
              f" rollv={float(scene.roll.data.root_lin_vel_w[0].norm()):.4f}"
              f" rollw={float(scene.roll.data.root_ang_vel_w[0].norm()):.4f}"
              f" still={int(scene.still_count[0])}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cr, rk = rel(scene.cradle), rel(scene.rack)
    qc0 = scene.cradle.data.root_quat_w[0]
    qr0 = scene.rack.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(qc0[3]), float(qc0[0])))
    ryaw = math.degrees(2.0 * math.atan2(float(qr0[3]), float(qr0[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cradle=({float(cr[0]):+.3f},{float(cr[1]):+.3f}) yaw={cyaw:+.1f} "
          f"rack=({float(rk[0]):+.3f},{float(rk[1]):+.3f}) yaw={ryaw:+.1f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: LOAD the cradle (transport + gravity) -----------------------
    # Hover 20 mm above the V rest line, axis along the groove, face 2 mm from the
    # backstop — released clear of all contact; gravity seats it.
    q_c = scene.cradle.data.root_quat_w
    p_c = scene.cradle.data.root_pos_w
    q_roll_t = quat_mul(q_c, qy90)
    loc = torch.tensor([-0.002, 0.0, CRADLE_REST_Z + 0.020], device=device).expand(n, 3)
    place(scene.roll, p_c + quat_apply(q_c, loc), q_roll_t)
    step(90)
    report("cradled")
    assert bool(scene.roll_in_cradle()[0]), "roll did not seat in the cradle V"
    s1 = print_score("P1 roll loaded into the cradle (transport + gravity drop)")
    assert s1 >= 0.20 - 1e-6 and s1 >= s0 - 1e-6

    # ---------------- phase 2: THREAD the spindle (contact dynamics) -----------------------
    # Start pose: on the groove axis, leading ball 6 mm outside the roll's near face.
    ex_c = quat_apply(q_c, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    p_roll_c = quat_apply_inverse(q_c, scene.roll.data.root_pos_w - p_c)
    x_face = p_roll_c[:, 0] - ROLL_HALF_W
    start = torch.stack([x_face - ROD_TIP - 0.006, p_roll_c[:, 1], p_roll_c[:, 2]], dim=-1)
    place(scene.spindle, p_c + quat_apply(q_c, start), quat_mul(q_c, qy90))
    step(2)
    assert not bool(scene.threaded()[0]), "rubric reads threaded before the push"
    report("pre-thread")

    centered = False
    for i in range(900):
        p_roll_c = quat_apply_inverse(q_c, scene.roll.data.root_pos_w - p_c)
        p_rod_c = quat_apply_inverse(q_c, scene.spindle.data.root_pos_w - p_c)
        if bool((p_rod_c[:, 0] - p_roll_c[:, 0]).abs()[0] < 0.006):
            centered = True
            break
        v_c = quat_apply_inverse(q_c, scene.spindle.data.root_lin_vel_w)
        ball_x = p_rod_c[:, 0] + CAP_Z  # leading ball center (local +z along +x_c)
        inside = ball_x > p_roll_c[:, 0] - ROLL_HALF_W + 0.016
        z_tgt = p_roll_c[:, 2] - torch.where(inside, 0.009, 0.0)
        f_c = torch.zeros(n, 3, device=device)
        f_c[:, 0] = (4.0 * (0.10 - v_c[:, 0])).clamp(-1.0, 1.2)
        f_c[:, 1] = (5.0 * (p_roll_c[:, 1] - p_rod_c[:, 1]) - 1.2 * v_c[:, 1]).clamp(-1.0, 1.0)
        f_c[:, 2] = (m_rod * g + 5.0 * (z_tgt - p_rod_c[:, 2])
                     - 1.2 * v_c[:, 2]).clamp(0.0, 2 * m_rod * g)
        a = axis(scene.spindle)
        w = scene.spindle.data.root_ang_vel_w
        # damp only the PERPENDICULAR omega: about the rod's own axis I ~ 3e-6, so
        # any usable damping gain violates the wrench-delay bound (D*dt/I >> 1) and
        # bang-bangs the clamp; the engine's angular damping owns the axial spin.
        w_perp = w - (w * a).sum(-1, keepdim=True) * a
        tau = (0.010 * torch.cross(a, ex_c, dim=-1) - 0.005 * w_perp).clamp(-0.010, 0.010)
        wrench(scene.spindle, quat_apply(q_c, f_c), tau)
        env.step(no_action)
    cut(scene.spindle)
    assert centered, "the push never centered the spindle in the core"
    step(40)  # rod falls off the wrench onto the core bottom; everything settles
    report("threaded")
    assert bool(scene.threaded()[0]), "spindle not threaded after the push"
    s2 = print_score("P2 spindle pushed through the core (contact dynamics)")
    assert s2 >= 0.45 - 1e-6 and s2 >= s1 - 1e-6

    # ---------------- phase 3: CARRY to the rack (transport, relative pose kept) -----------
    q_r = scene.rack.data.root_quat_w
    p_r = scene.rack.data.root_pos_w
    ex_r = quat_apply(q_r, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    q_rod = scene.spindle.data.root_quat_w
    p_rod = scene.spindle.data.root_pos_w
    q_rod_t = quat_mul(q_r, qy90)
    hover = torch.tensor([0.0, 0.0, 0.200], device=device).expand(n, 3)
    p_rod_t = p_r + quat_apply(q_r, hover)
    q_ch = quat_mul(q_rod_t, quat_conjugate(q_rod))  # the rigid transport transform
    p_roll_t = p_rod_t + quat_apply(q_ch, scene.roll.data.root_pos_w - p_rod)
    q_roll_t2 = quat_mul(q_ch, scene.roll.data.root_quat_w)
    place(scene.spindle, p_rod_t, q_rod_t)
    place(scene.roll, p_roll_t, q_roll_t2)
    # Hold the ROD ONLY (regulated); the roll falls 26 mm and hangs on it by contact.
    for _ in range(180):
        p = scene.spindle.data.root_pos_w
        v = scene.spindle.data.root_lin_vel_w
        sigma = ((p[:, 2] - scene.roll.data.root_pos_w[:, 2]) / 0.010).clamp(0.0, 1.0)
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (m_rod * g + sigma * m_roll * g + 6.0 * (p_rod_t[:, 2] - p[:, 2])
                   - 1.5 * v[:, 2]).clamp(0.0, 4.0)
        f[:, 0:2] = (5.0 * (p_rod_t - p)[:, 0:2] - 1.5 * v[:, 0:2]).clamp(-1.5, 1.5)
        a = axis(scene.spindle)
        w = scene.spindle.data.root_ang_vel_w
        w_perp = w - (w * a).sum(-1, keepdim=True) * a
        tau = (0.020 * torch.cross(a, ex_r, dim=-1) - 0.008 * w_perp).clamp(-0.020, 0.020)
        wrench(scene.spindle, f, tau)
        env.step(no_action)
    report("hover")
    assert bool(scene.threaded()[0]), "assembly came apart during the carry"
    assert not bool(scene.seated()[0]), "seated at hover?!"
    assert float(scene.roll.data.root_pos_w[0, 2]) < float(
        scene.spindle.data.root_pos_w[0, 2]), "the hang did not form at hover"
    s3 = print_score("P3 assembly carried to hover; hang formed mid-air (transport)")
    assert s3 >= 0.60 - 1e-6 and s3 >= s2 - 1e-6

    # ---------------- phase 4: MOUNT (contact dynamics, wrench cut before the seats) -------
    landed = False
    for i in range(600):
        p_c2 = quat_apply_inverse(q_r, scene.spindle.data.root_pos_w - p_r)
        if bool(p_c2[0, 2] < SEAT_Z + 0.012):
            landed = True
            break
        v_c = quat_apply_inverse(q_r, scene.spindle.data.root_lin_vel_w)
        sigma = ((scene.spindle.data.root_pos_w[:, 2]
                  - scene.roll.data.root_pos_w[:, 2]) / 0.010).clamp(0.0, 1.0)
        f_c = torch.zeros(n, 3, device=device)
        f_c[:, 2] = ((m_rod + sigma * m_roll) * g
                     + 6.0 * (-0.08 - v_c[:, 2])).clamp(0.0, 4.0)
        f_c[:, 0] = (5.0 * (0.0 - p_c2[:, 0]) - 1.5 * v_c[:, 0]).clamp(-1.5, 1.5)
        f_c[:, 1] = (5.0 * (0.0 - p_c2[:, 1]) - 1.5 * v_c[:, 1]).clamp(-1.5, 1.5)
        a = axis(scene.spindle)
        w = scene.spindle.data.root_ang_vel_w
        w_perp = w - (w * a).sum(-1, keepdim=True) * a
        tau = (0.020 * torch.cross(a, ex_r, dim=-1) - 0.008 * w_perp).clamp(-0.020, 0.020)
        wrench(scene.spindle, quat_apply(q_r, f_c), tau)
        env.step(no_action)
    cut(scene.spindle)
    assert landed, "descent never reached the seat approach height"
    # Hands off: the last 12 mm into both seats is pure gravity + contact.
    for j in range(32):  # up to 8 s of settling
        if bool(scene.success()[0]):
            break
        step(30)
        if j % 4 == 3:
            print(f"[solve] settle chunk {j}: "
                  f"rodv={float(scene.spindle.data.root_lin_vel_w[0].norm()):.4f} "
                  f"rodw={float(scene.spindle.data.root_ang_vel_w[0].norm()):.4f} "
                  f"rollv={float(scene.roll.data.root_lin_vel_w[0].norm()):.4f} "
                  f"rollw={float(scene.roll.data.root_ang_vel_w[0].norm()):.4f} "
                  f"still={int(scene.still_count[0])}", flush=True)
    if bool(scene.success()[0]):
        step(240)  # 2 s more hands-off margin before the strict persistence window
    report("mounted")
    s4 = print_score("P4 assembly dropped into both seats (gravity + contact)")
    assert s4 >= s3 - 1e-6, "score decreased across mounting"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after mount+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"thr={bool(scene.threaded()[0])} seat={bool(scene.seated()[0])} "
                      f"hang={bool(scene.hanging()[0])} set={bool(scene.settled()[0])} "
                      f"rodv={float(scene.spindle.data.root_lin_vel_w[0].norm()):.4f} "
                      f"rollv={float(scene.roll.data.root_lin_vel_w[0].norm()):.4f} "
                      f"rollw={float(scene.roll.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
