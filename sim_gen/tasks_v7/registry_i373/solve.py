"""Teleport solution for TotemTunnelScene (sim_gen task `registry_i373`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TOPPLE (contact): the totem is CARRIED (one root-state write, zero velocity,
   free space above the pen floor) to a launch spot on the tunnel axis, standing,
   faces squared with the pen. Then a horizontal world-frame force of ~0.78 N is
   applied at its centre of mass, aimed through the tunnel. 0.78 N is above the
   tipping threshold m*g*a/h = 0.54 N and below the static-friction slide
   threshold mu*m*g = 0.88 N, so the totem PIVOTS over its base edge and falls
   flat — a physically honest knock-down, never a written orientation.
2. THREAD (contact): with the totem lying aligned, a regulated wrench servo
   (feedforward ~0.9 N just above sliding friction 0.74 N, plus a velocity loop,
   a lateral centring term and a small yaw-alignment torque, all converted to the
   body frame every step) slides it lengthwise through the 66 x 56 mm bore under
   the lintel. The order-chained transit latches (entered -> mid-bore -> through)
   fire from real poses the sliding body passes through; the mid-bore pose (body
   spanning the whole roofed aperture) is unreachable any other way.
3. ERECT (transport + gravity): the totem, now outside, is carried (teleport =
   free-space transport, reorientation included — the carry a gripper performs
   with a wrist rotation) to an UPRIGHT hover 10 mm above the goal disc and
   released. It falls, seats on the disc, and settles under gravity + contact.
   Every success clause (upright, base at disc height, centred, settled) is
   produced by contact, never written.

If a stage leaves the totem geometrically off (a bounce mis-aligned it), it is
picked up again and the stage is re-run — a retry, not a cheat: the final
configuration is still 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it
persists.

Run (forge): python -u -m simgen_tasks.registry_i373.solve --headless [--seed N]
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

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.totem_tunnel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    yaw0 = scene.pen_yaw  # (n,) — pen frame is fixed for the episode
    zeros = torch.zeros_like(yaw0)
    ex_w = torch.stack([torch.cos(yaw0), torch.sin(yaw0), zeros], dim=-1)   # tunnel axis
    ey_w = torch.stack([-torch.sin(yaw0), torch.cos(yaw0), zeros], dim=-1)  # lateral
    qpen = _qz(yaw0)

    def loc_axis():
        loc = scene._pen_local(scene.totem.data.root_pos_w)
        ax = scene._axis_w()
        a_x = (ax * ex_w).sum(-1)   # long-axis alignment with the tunnel direction
        a_y = (ax * ey_w).sum(-1)
        return loc, ax, a_x, a_y

    def report(tag: str) -> None:
        loc, ax, a_x, _a_y = loc_axis()
        v = float(scene.totem.data.root_lin_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | loc=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},"
              f"{float(loc[0, 2]):+.3f}) axz={float(ax[0, 2]):+.2f} "
              f"ax_x={float(a_x[0]):+.2f} v={v:.3f} "
              f"lying={bool(scene._lying[0])} ent={bool(scene._entered[0])} "
              f"mid={bool(scene._mid[0])} thr={bool(scene._through[0])} "
              f"near={bool(scene._near[0])} prog={float(scene._prog[0]):.2f} "
              f"onpad={bool(scene.on_pad_upright()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wrench(f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        q = scene.totem.data.root_quat_w
        fb = quat_apply_inverse(q, f_world)
        tb = quat_apply_inverse(q, t_world)
        scene.totem.set_external_force_and_torque(
            fb.unsqueeze(1), tb.unsqueeze(1), env_ids=all_ids)

    def wrench_off() -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.totem.set_external_force_and_torque(z, z, env_ids=all_ids)

    def carry(local_x: float, local_y: float, z_w: float, quat_w: torch.Tensor) -> None:
        """TRANSPORT: one root-state write to a free-space pose (zero velocity)."""
        cy, sy = torch.cos(yaw0), torch.sin(yaw0)
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.pen_pos_w[:, 0] + cy * local_x - sy * local_y
        st[:, 1] = scene.pen_pos_w[:, 1] + sy * local_x + cy * local_y
        st[:, 2] = scene.pen_pos_w[:, 2] + z_w
        st[:, 3:7] = quat_w
        scene.totem.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    loc, ax, a_x, _ = loc_axis()
    pp = (scene.pen_pos_w - scene.env_origins)[0]
    pad_l = scene._pen_local(scene.pad_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pen=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"yaw={math.degrees(float(yaw0[0])):+.1f}deg "
          f"totem_local=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f}) "
          f"pad_local=({float(pad_l[0]):+.3f},{float(pad_l[1]):+.3f})", flush=True)
    report("reset")
    assert torch.isfinite(scene.totem.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.upright()[0]), "totem must start standing"
    assert bool((loc[0, :2].abs() < c.interior_half).all()), "totem must start inside the pen"
    s0 = print_score("P0 reset+settle (totem standing inside the pen)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: topple (contact) --------------------------------------------
    tip_f = 0.78  # N: above m*g*a/h = 0.54, below mu_s*m*g = 0.88 -> pivots, no slide
    lying_ok = False
    for attempt in range(3):
        # carry standing to the launch spot on the tunnel axis, faces squared
        carry(-0.008, 0.0, c.totem_h / 2 + 0.003, qpen)
        step(30)
        for _ in range(600):  # push at the CoM until it is clearly going over
            wrench(ex_w * tip_f, torch.zeros(n, 3, device=device))
            env.step(no_action)
            if float(scene._axis_w()[0, 2]) < 0.5:
                break
        wrench_off()
        step(180)  # hands-off fall + settle
        loc, ax, a_x, _ = loc_axis()
        lying_ok = (bool(scene.lying()[0]) and float(a_x[0]) > 0.90
                    and abs(float(loc[0, 1])) < 0.035 and float(loc[0, 0]) < 0.125)
        if lying_ok:
            break
        print(f"[solve] topple attempt {attempt + 1} off "
              f"(axz={float(ax[0, 2]):+.2f} ax_x={float(a_x[0]):+.2f}); retrying", flush=True)
    if not lying_ok:
        report("topple-FAIL")
        print("SIM_GEN_SOLVE: FAIL (topple never landed aligned)", flush=True)
        os._exit(1)
    report("toppled")
    assert bool(scene._lying[0]), "lying latch did not set"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 totem toppled flat, long axis through the tunnel")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect lying=0.15)"

    # ---------------- phases 2+3: thread the bore (contact servo) --------------------------
    v_des = 0.06   # m/s along the tunnel axis
    ff = 0.90      # N feedforward, just above sliding friction mu*m*g = 0.74 N
    for i in range(2400):
        loc = scene._pen_local(scene.totem.data.root_pos_w)
        if float(loc[0, 0]) >= c.through_x + 0.015:
            break
        v = scene.totem.data.root_lin_vel_w
        v_ax = (v * ex_w).sum(-1)
        v_lat = (v * ey_w).sum(-1)
        f_ax = (ff + 3.0 * (v_des - v_ax)).clamp(0.0, 1.6)
        f_lat = (-6.0 * loc[:, 1] - 2.0 * v_lat).clamp(-0.35, 0.35)
        f_w = f_ax.unsqueeze(-1) * ex_w + f_lat.unsqueeze(-1) * ey_w
        ax = scene._axis_w()
        sgn = torch.sign((ax * ex_w).sum(-1)).clamp(min=-1.0)
        psi = torch.atan2(sgn * (ax * ey_w).sum(-1), (sgn * ax * ex_w).sum(-1).abs())
        wz = scene.totem.data.root_ang_vel_w[:, 2]
        t_z = (-0.15 * psi - 0.02 * wz).clamp(-0.025, 0.025)
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = t_z
        wrench(f_w, t_w)
        env.step(no_action)
        if i % 200 == 0:
            report(f"thread-{i}")
    wrench_off()
    step(120)
    report("threaded")
    loc, ax, a_x, _ = loc_axis()
    assert bool(scene._entered[0]), "entered latch did not fire"
    assert bool(scene._mid[0]), "mid-bore latch did not fire"
    assert bool(scene._through[0]), "through latch did not fire"
    assert float(loc[0, 0]) >= c.through_x, \
        f"totem must be clear of the wall, x_l={float(loc[0, 0]):.3f}"
    assert not bool(scene.success()[0]), "cannot be success while lying outside"
    s2 = print_score("P2+P3 totem slid lengthwise through the bore (entered->mid->through)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 4: erect on the goal disc (transport + gravity) ----------------
    pad_l = scene._pen_local(scene.pad_pos_w)
    seated = False
    for attempt in range(3):
        carry(float(pad_l[0, 0]), float(pad_l[0, 1]),
              c.pad_t + c.totem_h / 2 + 0.010, qpen)
        for i in range(240):
            env.step(no_action)
            if i > 20 and float(scene.totem.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_lin:
                break
        step(60)
        if bool(scene.on_pad_upright()[0]) and bool(scene.success()[0]):
            seated = True
            break
        print(f"[solve] erect attempt {attempt + 1}: not seated; re-dropping", flush=True)
        report("erect-retry")
    if not seated:
        report("erect-FAIL")
        print("SIM_GEN_SOLVE: FAIL (totem never seated upright on the disc)", flush=True)
        os._exit(1)
    report("erected")
    s4 = print_score("P4 totem standing upright on the goal disc")
    assert s4 >= s2 - 1e-6, "score decreased across the erection"

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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
