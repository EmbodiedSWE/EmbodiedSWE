"""Teleport solution for PyramidFreightScene (sim_gen task `stack_pyramid_i218`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. BUILD ON THE CART (contact statics): each cube is TELEPORTED to the pose a
   gripper would release it in — square to the cart, 3-4 mm above its rest
   height — and RELEASED. Red and green drop into the two tray cells (the lips
   seat them); blue drops onto the ACTUAL settled base pair's midpoint (read
   back in the cart body frame) and bridges the seam. Every configuration the
   `seated`/`built` latches credit is a settled contact state produced by
   gravity, never authored directly. The tray bounds the seam slack to <= 4 mm
   (tray 84 mm vs 2 x 40 mm cubes), so the bridging drop cannot wedge.
2. DELIVERY PUSH (applied force, the signature interaction): the loaded cart is
   pushed along the rails by a horizontal force servo — the shove a gripper
   applies to the black post. F = M*(mu*g + kp*(v_des - v_along)) along the
   door direction, capped at 2.0 N; a soft lateral hold and a weak yaw-
   uprighting torque model the grip on the post. v_des = 0.08 m/s cruise,
   0.03 m/s on final approach, so the back-stop jolt slides the free-riding
   blue cube < 1 mm (v^2 / (2 mu g)). The pyramid arriving intact is real
   stability-limited transport: a hard shove throws the blue cube off (smoke's
   shove certificate). The push force cap keeps peak accel ~5 m/s^2, well
   under the ~8.8 m/s^2 blue-slip threshold — gentle BY CONSTRUCTION.
3. ARRIVAL (contact): the cart stalls against the depot's back stop; forces
   are cleared and everything settles hands-off before success is judged.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.stack_pyramid_i218.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 — registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pyramid_freight")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

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
        loc = scene.cubes_sled_local()[0]
        sx = scene.sled_fix_local()[0]
        print(f"[solve] {tag:12s} | sled depot=({float(sx[0]):+.3f},{float(sx[1]):+.3f}) "
              f"red=({float(loc[0, 0]):+.3f},{float(loc[0, 1]):+.3f},{float(loc[0, 2]):+.3f}) "
              f"grn=({float(loc[1, 0]):+.3f},{float(loc[1, 1]):+.3f},{float(loc[1, 2]):+.3f}) "
              f"blu=({float(loc[2, 0]):+.3f},{float(loc[2, 1]):+.3f},{float(loc[2, 2]):+.3f}) "
              f"seated={bool(scene._seated[0])} built={bool(scene._built[0])} "
              f"deliv={bool(scene._deliv[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.sled.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def sled_local_to_world(x_loc: torch.Tensor, y_loc: torch.Tensor,
                            z_loc: torch.Tensor) -> torch.Tensor:
        """Root state (zero vel) for a cube at cart-local (x,y,z), square to the
        cart — exactly the release pose of a gripper placing into the moving-frame
        tray."""
        st = torch.zeros(n, 13, device=device)
        lp = torch.stack([x_loc, y_loc, z_loc], dim=-1)
        st[:, 0:3] = scene.sled.data.root_pos_w \
            + quat_apply(scene.sled.data.root_quat_w, lp)
        st[:, 3:7] = scene.sled.data.root_quat_w
        return st

    # ---------------- phase 0: reset, settle, baseline -----------------------------------------
    step(180)
    fx = scene.fix_xy[0] - scene.env_origins[0, 0:2]
    sx0 = scene.sled_fix_local()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"depot=({float(fx[0]):+.3f},{float(fx[1]):+.3f}) "
          f"yaw={math.degrees(float(scene.fix_yaw[0])):+.1f}deg "
          f"sled depot-x={float(sx0[0]):+.3f} "
          f"perm={scene.slot_perm[0].tolist()}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(sx0[0]) > 0.25, f"cart must start out on the rails, depot-x {float(sx0[0]):.3f}"
    s0 = print_score("P0 reset+settle (cubes scattered, cart at the loading end)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: seat red + green in the tray cells ------------------------------
    # Release poses: square to the cart, cell centres tray_cx +/- cell_dx, 4 mm
    # above rest height; the lips seat them as they drop.
    z_rel = torch.full((n,), c.deck_top + c.cube_s / 2 + 0.004, device=device)
    y0 = torch.zeros(n, device=device)
    scene.cubes["cube_red"].write_root_state_to_sim(
        sled_local_to_world(torch.full((n,), c.tray_cx + c.cell_dx, device=device),
                            y0, z_rel), all_ids)
    step(60)
    scene.cubes["cube_green"].write_root_state_to_sim(
        sled_local_to_world(torch.full((n,), c.tray_cx - c.cell_dx, device=device),
                            y0, z_rel), all_ids)
    step(90)
    report("seated")
    seated = scene.base_seated()[0]
    assert bool(seated[0]) and bool(seated[1]), \
        f"red+green must settle seated in the tray, seated={seated.tolist()}"
    assert bool(scene._seated[0]), "seated latch must be set by the settled bases"
    s1 = print_score("P1 red+green seated in the tray (settled)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_seated - 1e-6, \
        f"P1 score {s1} (expect >= {c.w_seated})"

    # ---------------- phase 2: bridge blue over the ACTUAL settled base pair -------------------
    loc = scene.cubes_sled_local()
    mid = (loc[:, 0, :] + loc[:, 1, :]) / 2
    base_top = mid[:, 2] + c.cube_s / 2
    scene.cubes["cube_blue"].write_root_state_to_sim(
        sled_local_to_world(mid[:, 0], mid[:, 1],
                            base_top + c.cube_s / 2 + 0.003), all_ids)
    step(90)
    report("built")
    assert bool(scene.pyramid_ok()[0]), "blue must settle bridging BOTH base cubes"
    assert bool(scene._built[0]), "built latch must be set by the settled pyramid"
    assert not bool(scene.delivered_ok()[0]), \
        "cart must not count as delivered while still out on the rails"
    s2 = print_score("P2 pyramid built on the cart (settled)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_seated + c.w_built - 1e-6, \
        f"P2 score {s2} (expect >= {c.w_seated + c.w_built})"

    # ---------------- phase 3: push the loaded cart into the depot -----------------------------
    # Horizontal force servo on the cart — the hand on the black post. Gentle BY
    # CONSTRUCTION: cap 2.0 N on a 0.40 kg load = 5 m/s^2 peak, under the
    # 8.8 m/s^2 blue-slip threshold; final approach at 0.03 m/s so the back-stop
    # jolt slides blue < 1 mm.
    m_tot = c.sled_mass + 3 * c.cube_mass
    g = 9.81
    door_w = torch.stack([-torch.cos(scene.fix_yaw), -torch.sin(scene.fix_yaw),
                          torch.zeros_like(scene.fix_yaw)], dim=-1)
    lat_w = torch.stack([torch.sin(scene.fix_yaw), -torch.cos(scene.fix_yaw),
                         torch.zeros_like(scene.fix_yaw)], dim=-1)  # depot -y axis... sign below
    e_x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    arrived = False
    stall_ct = 0
    for i in range(2400):
        x_dep = float(scene.sled_fix_local()[0, 0])
        v_des = 0.08 if x_dep > 0.16 else 0.03
        vel = scene.sled.data.root_lin_vel_w
        v_along = (vel * door_w).sum(dim=-1)
        # push force: friction feedforward + P on the along-track speed
        f_mag = (m_tot * (0.12 * g + 8.0 * (v_des - v_along))).clamp(0.0, 2.0)
        # soft lateral hold toward the channel centreline (depot y = 0).
        # lat_w = depot -y in world, so the restoring force -k*y_dep (depot
        # frame) is +k*y_dep along lat_w; v_lat = vel . lat_w = -v_y_dep, so
        # depot-frame damping -d*v_y_dep is -d*v_lat along lat_w.
        y_dep = scene.sled_fix_local()[:, 1]
        v_lat = (vel * lat_w).sum(dim=-1)
        f_lat = (m_tot * (20.0 * y_dep - 4.0 * v_lat)).clamp(-0.3, 0.3)
        f = f_mag.unsqueeze(-1) * door_w + f_lat.unsqueeze(-1) * lat_w
        # weak yaw hold: keep the cart front on the door direction (the grip
        # torque of the hand on the post). cross(front, door)_z > 0 means the
        # door is CCW of the front, so the restoring torque is +k*cross_z.
        front_w = quat_apply(scene.sled.data.root_quat_w, e_x)
        cross_z = front_w[:, 0] * door_w[:, 1] - front_w[:, 1] * door_w[:, 0]
        tau_z = 0.02 * cross_z - 0.005 * scene.sled.data.root_ang_vel_w[:, 2]
        tau = torch.zeros(n, 3, device=device)
        tau[:, 2] = tau_z
        scene.sled.set_external_force_and_torque(
            f.unsqueeze(1), tau.unsqueeze(1), env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i % 120 == 0:
            print(f"[solve] push step {i:4d}: depot-x {x_dep:+.3f} "
                  f"v_along {float(v_along[0]):+.3f} "
                  f"pyramid={bool(scene.pyramid_ok()[0])}", flush=True)
        x_dep = float(scene.sled_fix_local()[0, 0])
        v_now = float(scene.sled.data.root_lin_vel_w[0].norm())
        if x_dep < 0.12 and v_now < 0.005:
            stall_ct += 1
        else:
            stall_ct = 0
        if x_dep < 0.085 or stall_ct >= 30:
            arrived = True
            print(f"[solve] cart arrived at step {i}: depot-x {x_dep:+.3f} "
                  f"(stall streak {stall_ct})", flush=True)
            break
    clear_force()
    if not arrived:
        report("PUSH-FAIL")
        print("SIM_GEN_SOLVE: FAIL (cart never reached the depot pocket)", flush=True)
        os._exit(1)
    step(240)  # hands-off settle
    report("delivered")
    assert bool(scene.pyramid_ok()[0]), "the pyramid must survive the delivery push"
    assert bool(scene.delivered_ok()[0]), "the cart must rest inside the depot pocket"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded cart delivered into the depot (settled)")
    assert s3 >= s2 - 1e-6, "score decreased across the delivery"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -----------------
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
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
