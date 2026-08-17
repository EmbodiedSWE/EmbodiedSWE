"""Teleport solution for HutchFeedScene (sim_gen task `put_banana_i132`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN (contact-respecting kinematic lift + transport teleport): the door is
   pose-HELD each step (the kinematic-hold emulation of a knob grasp) and raised slowly,
   straight up, out of its open-topped channel — if the slab clashed with the rails the
   hold would fight real contacts. Once clear of the channel it is TELEPORTED (pure
   free-space transport) to a parking spot on the ground, dropped the last 2 cm, and
   left to settle lying flat. Nothing about this satisfies the goal: opening alone is
   worth 0.15 latched credit and success() still requires a CLOSED door at the end.
2. FEED (contact dynamics — the core interaction; no teleport can produce it without
   bypassing the task): the block is teleported (transport) to a staging point on the
   doorway axis, OUTSIDE the hutch — a state a hand could trivially set up — and is then
   PUSHED along the ground with bounded horizontal forces (velocity-regulated CoM push,
   <= 0.9 N) through the 9 cm doorway until fully inside. The passage under the lintel
   and between the jambs happens entirely through friction + contacts; the block is
   never pose-written past the aperture. A runtime probe toggles the force-frame
   encoding if the pod rotates applied wrenches by the body's rotation since reset.
3. CLOSE (contact-respecting kinematic lower + gravity seat): the door is carried
   (teleport transport) to a hover above the channel, pose-hold lowered INTO the channel
   to 18 mm above its seat — asserted NOT yet door_closed_now (outside the 12 mm height
   tolerance) — then RELEASED: gravity drops it the last stretch and it beds down
   between wall, rails and stops under contact. success() first turns True here, judged
   on settled poses.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_banana_i132.solve --headless [--seed N]
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

DT = 1.0 / 120.0
G_DT = 9.81 * DT  # gravity-compensated kinematic hold: write vz=+g*dt so net vz ~ 0


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hutch_feed")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.block.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def report(tag: str) -> None:
        b = (scene.block.data.root_pos_w - scene.env_origins)[0]
        d = (scene.door.data.root_pos_w - scene.env_origins)[0]
        bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | block=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) block_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"door=({float(d[0]):+.3f},{float(d[1]):+.3f},{float(d[2]):.3f}) "
              f"blocking={bool(scene.door_blocking()[0])} "
              f"closed={bool(scene.door_closed_now()[0])} "
              f"inside={bool(scene.block_inside_now()[0])} "
              f"opened={bool(scene._opened[0])} app={float(scene._app_max[0]):.3f} "
              f"in={bool(scene._inside[0])} closed_after={bool(scene._closed_after[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_door(pos: torch.Tensor, quat: torch.Tensor) -> None:
        """One kinematic-hold write: pose imposed, vz=+g*dt cancels the gravity kick."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        st[:, 9] = G_DT
        scene.door.write_root_state_to_sim(st, all_ids)

    def local_to_world(loc) -> torch.Tensor:
        """Hutch-local point -> world (env origins included)."""
        p = torch.tensor(loc, device=device).expand(n, 3)
        return scene.hutch.data.root_pos_w + quat_apply(scene.hutch.data.root_quat_w, p)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)
    h = (scene.hutch.data.root_pos_w - scene.env_origins)[0]
    hq = scene.hutch.data.root_quat_w[0]
    yaw = 2.0 * math.atan2(float(hq[3]), float(hq[0]))
    b0 = (scene.block.data.root_pos_w - scene.env_origins)[0]
    d0 = (scene.door.data.root_pos_w - scene.env_origins)[0]
    masses = scene.door.root_physx_view.get_masses()
    print(f"[solve] layout readback (seed {args.seed}): hutch=({float(h[0]):+.3f},"
          f"{float(h[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"door=({float(d0[0]):+.3f},{float(d0[1]):+.3f},{float(d0[2]):.3f}) "
          f"block=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"door_mass={float(masses.reshape(-1)[0]):.4f}", flush=True)
    report("reset")
    assert 0.02 < float(masses.reshape(-1)[0]) < 0.15, "door MassAPI not applied"
    assert bool(scene.door_closed_now()[0]), "door did not settle seated/closed"
    assert not bool(scene.block_inside_now()[0]), "block must start outside"
    s0 = print_score("P0 reset+settle (door closed, block outside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: OPEN — kinematic lift out of the channel --------------------
    # Pose-hold the door (knob-grasp emulation) and raise it straight up at 0.07 m/s
    # until its bottom clears the rail tops; the channel is open-topped, so the lift is
    # contact-free — but it goes through the hold, not a jump, so any clash would fight
    # real contacts. Then one transport teleport parks it lying flat, clear of the
    # doorway, dropped the last 2 cm under gravity.
    seat_pos = scene.door.data.root_pos_w.clone()
    seat_quat = scene.door.data.root_quat_w.clone()
    lift = 0.10
    steps_up = int(lift / 0.07 / DT)
    for i in range(steps_up):
        p = seat_pos.clone()
        p[:, 2] += lift * (i + 1) / steps_up
        hold_door(p, seat_quat)
        env.step(no_action)
    report("lifted")
    assert not bool(scene.door_blocking()[0]), "lifted door still counts as blocking"
    # transport: park lying flat at hutch-local (0.30, 0.30), knob sideways
    q_y90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                         device=device).expand(n, 4)
    park = local_to_world((0.30, 0.30, 0.0))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = park[:, :2]
    st[:, 2] = c.door_t / 2 + 0.02
    st[:, 3:7] = quat_mul(seat_quat, q_y90)
    scene.door.write_root_state_to_sim(st, all_ids)
    step(90)
    report("parked")
    assert bool(scene._opened[0]), "opened latch did not set"
    assert not bool(scene.door_blocking()[0]), "parked door still blocks the doorway"
    s1 = print_score("P1 door lifted out of the channel and parked clear")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_open - 1e-6, f"P1 score {s1}"

    # ---------------- phase 2: FEED — push the block through the doorway -------------------
    # Transport teleport: stage the block on the doorway axis, OUTSIDE the hutch, square
    # to the doorway (a state a hand trivially sets up; satisfies no interior clause).
    stage_w = local_to_world((0.26, 0.0, 0.0))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = stage_w[:, :2]
    st[:, 2] = c.block_rest_z + 0.002
    st[:, 3:7] = scene.hutch.data.root_quat_w
    scene.block.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    assert not bool(scene.block_inside_now()[0]), "staged block must still be outside"

    # Velocity-regulated horizontal CoM push toward the hutch centre; the doorway
    # passage happens through friction + contacts. Runtime force-frame probe: some pods
    # rotate applied wrenches by the body's rotation since reset — if measured progress
    # goes the wrong way, toggle the encoding.
    q_ref = scene.block.data.root_quat_w.clone()
    mode = 0

    def encode(f_world: torch.Tensor) -> torch.Tensor:
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_inv(scene.block.data.root_quat_w)),
                          f_world)

    target_xy = scene.hutch.data.root_pos_w[:, :2].clone()
    floor_f = 0.45  # stiction floor (mu*m*g ~ 0.24 N)
    win_i, win_dist = 0, float((target_xy[0] - scene.block.data.root_pos_w[0, :2]).norm())
    pushed_in = False
    for i in range(2400):
        loc = scene._hutch_local(scene.block.data.root_pos_w)[0]
        if float(loc[0]) < 0.03 and abs(float(loc[1])) < 0.06:
            pushed_in = True
            clear_forces()
            break
        d_vec = target_xy[0] - scene.block.data.root_pos_w[0, :2]
        dist = float(d_vec.norm())
        u = d_vec / max(dist, 1e-6)
        v_des = u * (0.12 if dist > 0.10 else 0.07)
        v = scene.block.data.root_lin_vel_w[0, :2]
        f_xy = 2.0 * (v_des - v)
        f_along = float((f_xy * u).sum())
        if float(v.norm()) < 0.02 and f_along < floor_f:
            f_xy = f_xy + u * (floor_f - f_along)  # break stiction
        fn = float(f_xy.norm())
        if fn > 0.9:
            f_xy = f_xy * (0.9 / fn)
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :2] = f_xy
        scene.block.set_external_force_and_torque(
            encode(f_world).view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i - win_i >= 60:  # progress probe
            if dist > win_dist + 0.008:
                mode = 1 - mode
                print(f"[solve] push: moving away (dist {win_dist:.3f} -> {dist:.3f}); "
                      f"force-frame mode -> {mode}", flush=True)
            elif dist > win_dist - 0.004:
                floor_f = min(floor_f + 0.15, 0.9)
                print(f"[solve] push: stalled at dist {dist:.3f}; stiction floor -> "
                      f"{floor_f:.2f} N", flush=True)
            win_i, win_dist = i, dist
    clear_forces()
    step(60)  # free settle
    report("fed")
    assert pushed_in, "push loop timed out before the block was inside"
    assert bool(scene.block_inside_now()[0]), "block did not settle inside the hutch"
    s2 = print_score("P2 block pushed through the doorway (contact dynamics), settled inside")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_open + c.w_in - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: CLOSE — kinematic lower + gravity seat ----------------------
    # Transport the door to a hover above the channel, pose-hold lower it INTO the
    # channel to 18 mm above the seat (asserted NOT yet closed — outside the 12 mm
    # height tolerance), then release: gravity beds it down between wall, rails and
    # stops through real contact.
    hover = local_to_world((c.door_seat_lx, 0.0, 0.0))
    hover[:, 2] = c.door_seat_lz + 0.10
    hq_all = scene.hutch.data.root_quat_w.clone()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = hover
    st[:, 3:7] = hq_all
    st[:, 9] = G_DT
    scene.door.write_root_state_to_sim(st, all_ids)
    lower = 0.10 - 0.018
    steps_dn = int(lower / 0.06 / DT)
    for i in range(steps_dn):
        p = hover.clone()
        p[:, 2] -= lower * (i + 1) / steps_dn
        hold_door(p, hq_all)
        env.step(no_action)
    report("pre-release")
    assert not bool(scene.door_closed_now()[0]), \
        "held door already satisfies the closed clause (release band too low)"
    # release: one write with zero velocity at the held pose, then hands off
    p = hover.clone()
    p[:, 2] -= lower
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = p
    st[:, 3:7] = hq_all
    scene.door.write_root_state_to_sim(st, all_ids)
    step(150)  # 1.25 s: drop 18 mm, bed down in the channel, settle
    report("closed")
    s3 = print_score("P3 door released into the channel, gravity-seated")
    assert s3 >= s2 - 1e-6, "score decreased across the close"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after door close)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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
    main()
