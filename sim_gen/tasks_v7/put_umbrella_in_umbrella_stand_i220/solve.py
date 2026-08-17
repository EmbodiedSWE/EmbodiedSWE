"""Teleport solution for UmbrellaUnrackCradleScene (sim_gen task
`put_umbrella_in_umbrella_stand_i220`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Both load-bearing interactions go through
contact dynamics:
  1. EXTRACT (dynamics): the umbrella starts SEATED tip-down in a deep socket. A
     velocity-regulated vertical force (gravity feed-forward + PD to +0.25 m/s, force
     at the CoM only — no torque, no orientation pinning) slides it UP the socket,
     guided by wall contact, until its lower end clears the mouth plane. It is never
     teleported out of the socket.
  2. CARRY (transport): once fully in free air, the umbrella is teleported to a hover
     pose above the cradle — horizontal, axis along the cradle groove, shaft 50 mm
     above the two V-notches. Nothing touches.
  3. LAY (dynamics): a velocity-regulated descent (-0.04 m/s, force at the CoM only)
     lowers the shaft to ~1 mm above nested in both V-notches; a gentle position-servo
     PRESS closes the last air gap and forms both contacts with the wrench still
     carrying the weight; then a STAGED QUASI-STATIC LOAD TRANSFER (support ramped
     0.60 -> 0.35 -> 0.15 of weight with pure velocity damping, no position target)
     hands the weight to the notches without a set-down jolt; the final nesting +
     settling is pure contact + gravity. The umbrella is never teleported into
     contact, never welded, never held.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i220.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.umbrella_unrack_cradle")().build(num_envs=args.num_envs,
                                                            device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor) -> None:
        """Apply a WORLD force (n,3) at the CoM, expressed in the CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        u = rel(scene.umbrella)
        d = scene.seat_dists()[0]
        print(f"[solve] {tag:14s} | umb=({float(u[0]):+.3f},{float(u[1]):+.3f},"
              f"{float(u[2]):.3f}) low_end={float(scene.low_end_height()[0]):.3f}"
              f" seat_d=({float(d[0]):.4f},{float(d[1]):.4f})"
              f" seated={bool(scene.seated()[0])} horiz={bool(scene.horizontal()[0])}"
              f" set={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    u = rel(scene.umbrella)
    ca = rel(scene.cane)
    sp = rel(scene.stand)
    cp = rel(scene.cradle)
    side = float(scene.umb_side[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) "
          f"cradle=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) "
          f"umb=({float(u[0]):+.3f},{float(u[1]):+.3f},{float(u[2]):.3f}) "
          f"cane=({float(ca[0]):+.3f},{float(ca[1]):+.3f}) side={side:+.0f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: EXTRACT (contact dynamics) ----------------------------------
    # Velocity-regulated pull straight up the socket axis; the walls guide the
    # canopy. The wrench never exceeds 2x weight and applies no torque.
    q_s = scene.stand.data.root_quat_w
    p_s = scene.stand.data.root_pos_w
    sock_loc = torch.tensor([0.0, side * float(scene.cfg.sock_y), 0.0],
                            device=device).expand(n, 3)
    sock_xy = (p_s + quat_apply(q_s, sock_loc))[:, 0:2]
    mg = c.umb_mass * 9.81
    clear_exit = float(scene.cfg.mouth_z) + c.clear_margin + 0.02
    pulled = False
    for i in range(900):
        if float(scene.low_end_height()[0]) > clear_exit:
            pulled = True
            break
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (mg + 6.0 * (0.25 - v[:, 2])).clamp(0.0, 2.0 * mg)
        f[:, 0:2] = 4.0 * (sock_xy - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-2.0, 2.0)
        wrench(scene.umbrella, f)
        env.step(no_action)
    report("extracted")
    assert pulled, "the pull never brought the umbrella clear of the socket"
    # brief regulated zero-velocity hold so the readbacks are calm
    hold_xy = scene.umbrella.data.root_pos_w[:, 0:2].clone()
    hold_z = scene.umbrella.data.root_pos_w[:, 2].clone()
    for _ in range(20):
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (mg + 8.0 * (hold_z - scene.umbrella.data.root_pos_w[:, 2])
                   - 4.0 * v[:, 2]).clamp(0.0, 2.0 * mg)
        f[:, 0:2] = 4.0 * (hold_xy - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-2.0, 2.0)
        wrench(scene.umbrella, f)
        env.step(no_action)
    s1 = print_score("P1 extracted from the socket (contact dynamics)")
    assert s1 >= s0 - 1e-6
    assert s1 >= 0.30 - 1e-6, "extraction latch did not fire"

    # ---------------- phase 2: CARRY (transport only) --------------------------------------
    # Teleport to a hover above the cradle: horizontal, axis along the groove, the
    # shaft 50 mm above the two V seats. Nothing touches.
    q_c = scene.cradle.data.root_quat_w
    p_c = scene.cradle.data.root_pos_w
    psi = 2.0 * torch.atan2(q_c[:, 3], q_c[:, 0])
    half = psi / 2
    zc = torch.zeros_like(half)
    q_yaw = torch.stack([torch.cos(half), zc, zc, torch.sin(half)], dim=-1)
    # q_u = qz(psi) * qx(-90 deg): body +z (tip->handle) -> cradle local +y
    s45 = math.sin(math.pi / 4)
    qx_n = torch.tensor([s45, -s45, 0.0, 0.0], device=device).expand(n, 4)
    aw, ax, ay, az = q_yaw.unbind(-1)
    bw, bx, by, bz = qx_n.unbind(-1)
    q_u = torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)
    hover = 0.050
    org_loc = torch.tensor(
        [0.0, float(scene.cfg.y_org),
         float(scene.cfg.seat_z) + float(scene.cfg.r_rest) + hover],
        device=device).expand(n, 3)
    pos_u = p_c + quat_apply(q_c, org_loc)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pos_u
    st[:, 3:7] = q_u
    scene.umbrella.write_root_state_to_sim(st, all_ids)
    xy_hold = pos_u[:, 0:2].clone()
    z_hold = pos_u[:, 2].clone()
    for _ in range(10):
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (mg + 8.0 * (z_hold - scene.umbrella.data.root_pos_w[:, 2])
                   - 4.0 * v[:, 2]).clamp(0.0, 2.0 * mg)
        f[:, 0:2] = 4.0 * (xy_hold - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-2.0, 2.0)
        wrench(scene.umbrella, f)
        env.step(no_action)
    report("hover")
    s2 = print_score("P2 carried to hover above the cradle (transport)")
    assert s2 >= s1 - 1e-6
    d_hover = scene.seat_dists()[0]
    assert float(d_hover.max()) > c.seat_tol, \
        "hover must start with the shaft OUTSIDE the nested-seat tolerance"

    # ---------------- phase 3: LAY (contact dynamics) --------------------------------------
    # Slow velocity-regulated descent until the shaft reads ~1 mm above nested in both
    # V-notches, then a STAGED LOAD TRANSFER: support force ramped down with pure
    # velocity damping (no position target — the V geometry centers, friction kills the
    # axial creep). An instant wrench cut at speed jolts the shaft down the open groove
    # (observed live: it slid axially and pitched an end onto the floor).
    landed = False
    for i in range(700):
        d = scene.seat_dists()[0]
        if float(d.max()) < 0.0140:
            landed = True
            break
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (mg + 6.0 * (-0.04 - v[:, 2])).clamp(0.0, 2.0 * mg)
        f[:, 0:2] = 4.0 * (xy_hold - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-2.0, 2.0)
        wrench(scene.umbrella, f)
        env.step(no_action)
    report("touchdown")
    assert landed, "the descent never brought the shaft to the notches"
    # press INTO the seats with the wrench still carrying ~the weight: a z position
    # servo aimed 6 mm below the current height closes the last ~1 mm air gap at
    # regulated speed and forms BOTH contacts before any load moves
    z_tgt = scene.umbrella.data.root_pos_w[:, 2] - 0.006
    for _ in range(40):
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (mg + 15.0 * (z_tgt - scene.umbrella.data.root_pos_w[:, 2])
                   - 6.0 * v[:, 2]).clamp(0.0, 2.0 * mg)
        f[:, 0:2] = 4.0 * (xy_hold - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-2.0, 2.0)
        wrench(scene.umbrella, f)
        env.step(no_action)
    report("pressed")
    # staged QUASI-STATIC load transfer (gap closed, velocities ~0): the support
    # force ramps down with pure velocity damping — no position target, the V
    # geometry centers, friction holds the groove direction
    for frac in (0.60, 0.35, 0.15):
        for _ in range(25):
            v = scene.umbrella.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = (frac * mg - 6.0 * v[:, 2]).clamp(0.0, 2.0 * mg)
            f[:, 0:2] = (-3.0 * v[:, 0:2]).clamp(-2.0, 2.0)
            wrench(scene.umbrella, f)
            env.step(no_action)
        report(f"transfer {frac:.2f}")
    wrench(scene.umbrella, zero3.expand(n, 3))
    report("laid")

    # hands off: contact + gravity settle the rest
    for k in range(24):  # up to 6 s
        if bool(scene.success()[0]):
            break
        step(30)
        if k % 4 == 3:
            report(f"settling+{(k + 1) * 30}")
    if bool(scene.success()[0]):
        step(120)  # 1 s extra hands-off calm before the strict persistence window
    report("settled")
    s3 = print_score("P3 laid to rest in both V-notches (contact dynamics)")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after extract+carry+lay+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold_ok, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold_ok = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                u = scene.umbrella
                d = scene.seat_dists()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"seated={bool(scene.seated()[0])} "
                      f"horiz={bool(scene.horizontal()[0])} "
                      f"seat_d=({float(d[0]):.4f},{float(d[1]):.4f}) "
                      f"lin={float(u.data.root_lin_vel_w[0].norm()):.4f} "
                      f"ang={float(u.data.root_ang_vel_w[0].norm()):.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
