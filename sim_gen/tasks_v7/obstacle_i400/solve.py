"""Teleport solution for BreachDoorwayScene (sim_gen task `obstacle_i400`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY; every load-bearing interaction goes through
CONTACT DYNAMICS:
  P1-P3 — DEMOLITION, top brick first (2, 1, 0). Each brick is slid out of the
  doorway slot by a horizontal velocity-servo force along the fixture-local -y axis
  (is_global — the body-frame default drags with the body). The force is CUT once
  the brick centre is 60 mm clear of the wall's outer face; the brick tips off its
  support and lands on the floor in front under gravity. After it settles, the
  ALREADY-FREE brick is teleported to a parking spot far to the side (transport
  only — extraction itself was pure contact physics).
  P4 — the cargo cube is teleported from its scatter spawn to a STAGING point on
  the doorway axis, 160 mm in front of the wall, on open floor (transport only).
  From there it is PUSHED through the doorway bore with a forward velocity-servo
  force plus a lateral PD that holds the doorway centreline; the force is cut once
  the centre is 105 mm past the outer face (12 mm beyond the success threshold) and
  the cube coasts to rest inside the roofed court. The bore-transit latch is set by
  the push itself — the cargo is never teleported into, past, or over the wall.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene
latches its credit), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.obstacle_i400.solve --headless [--seed N]
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
    env = ENVS.get("simgen.breach_doorway")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def fix_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return fp, yaw

    def to_world(local_xy: tuple) -> tuple[float, float]:
        fp, yaw = fix_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + local_xy[0] * cy - local_xy[1] * sy,
                float(fp[1]) + local_xy[0] * sy + local_xy[1] * cy)

    def report(tag: str) -> None:
        cl = scene._fix_local(scene.cargo.data.root_pos_w)[0]
        bl = [scene._fix_local(b.data.root_pos_w)[0] for b in scene.bricks]
        bs = " ".join(f"b{k}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})"
                      for k, p in enumerate(bl))
        print(f"[solve] {tag:14s} | cargo_local=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) {bs} out={float(scene.bricks_out_now()[0]):.0f} "
              f"out_latch={float(scene.out_latch[0]):.0f} "
              f"enter={bool(scene.cargo_in_bore()[0])} "
              f"enter_latch={float(scene.enter_latch[0]):.0f} "
              f"through={bool(scene.cargo_through()[0])} "
              f"in_court={bool(scene.cargo_in_court()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fyaw = fix_pose()
    cy, sy = math.cos(fyaw), math.sin(fyaw)
    cargo0 = scene._fix_local(scene.cargo.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): fixture=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg "
          f"cargo_local=({float(cargo0[0]):+.3f},{float(cargo0[1]):+.3f}) "
          f"bricks_x=" + ",".join(
              f"{float(scene._fix_local(b.data.root_pos_w)[0][0]):+.4f}"
              for b in scene.bricks), flush=True)
    report("reset")
    s_prev = print_score("P0 reset+settle")

    # Fixture is kinematic — its frame is constant for the whole episode.
    out_dir = torch.tensor([sy, -cy, 0.0], device=device)  # fixture-local -y in world
    fwd = torch.tensor([-sy, cy, 0.0], device=device)  # fixture-local +y in world
    lat = torch.tensor([cy, sy, 0.0], device=device)  # fixture-local +x in world
    m_brick = c.brick_mass

    # ---------------- phases 1-3: demolition, top brick first ------------------------------
    for slot, k in enumerate((2, 1, 0), start=1):
        brick = scene.bricks[k]
        pulled = False
        for i in range(600):
            bl = scene._fix_local(brick.data.root_pos_w)[0]
            if float(bl[1]) < -0.06:
                pulled = True
                break
            if i % 150 == 149:
                print(f"[solve] pull b{k} i={i} y={float(bl[1]):+.3f}", flush=True)
            v_out = float(torch.dot(brick.data.root_lin_vel_w[0], out_dir))
            gain = 35.0 + 15.0 * (i // 150)  # escalate past static friction if bound
            cap = 3.0 + 1.0 * (i // 150)
            f_mag = max(0.0, min(cap, m_brick * gain * (0.25 - v_out)))
            f_world = f_mag * out_dir
            brick.set_external_force_and_torque(
                f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_wrench(brick)
        assert pulled, f"brick{k} never came free of the doorway"
        step(100)  # tip off the stack, land, settle

        # Transport only: the brick is already free — park it far to the side.
        wx, wy = to_world((-0.62, -0.30 - 0.15 * (slot - 1)))
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.brick_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        brick.write_root_state_to_sim(st, all_ids)
        step(30)
        report(f"brick{k} out")
        s = print_score(f"P{slot} brick{k} extracted + parked")
        assert s >= s_prev - 1e-6, f"score decreased across brick{k} extraction"
        s_prev = s

    # ---------------- phase 4: cargo staging (teleport) + threaded push --------------------
    # Staging: on the doorway axis, 160 mm in front of the outer face — open floor,
    # outside the plug region, aligned with the bore.
    wx, wy = to_world((0.0, -0.16))
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.cargo_size / 2 + 0.002
    st[:, 3], st[:, 6] = math.cos(fyaw / 2), math.sin(fyaw / 2)  # face the bore
    st[:, 0:3] += scene.env_origins
    scene.cargo.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    s = print_score("P4a cargo transport to staging")
    assert s >= s_prev - 1e-6, "score decreased across cargo transport"
    s_prev = s

    # Push controller: friction feed-forward (~mu*m*g) + velocity servo. The bias is
    # what breaks static friction (a pure servo saturates at m*G*v_des, below the
    # cube-ground breakaway); the 0.90 N cap stays under the mg ~ 0.98 N CoM-push
    # tipping bound so the cube slides instead of tumbling.
    m_cargo, v_des = c.cargo_mass, 0.12
    through = False
    for i in range(900):
        cl = scene._fix_local(scene.cargo.data.root_pos_w)[0]
        if float(cl[1]) > 0.105:  # 12 mm past the success threshold
            through = True
            break
        v_w = scene.cargo.data.root_lin_vel_w[0]
        v_fwd = float(torch.dot(v_w, fwd))
        v_lat = float(torch.dot(v_w, lat))
        e_lat = -float(cl[0])
        f_fwd = max(0.0, min(0.90, 0.55 + m_cargo * 40.0 * (v_des - v_fwd)))
        f_lat = max(-0.6, min(0.6, m_cargo * (50.0 * e_lat - 10.0 * v_lat)))
        if i % 150 == 149:
            print(f"[solve] push i={i} y={float(cl[1]):+.3f} v={v_fwd:+.3f} "
                  f"f={f_fwd:.2f}", flush=True)
        f_world = f_fwd * fwd + f_lat * lat
        scene.cargo.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
    clear_wrench(scene.cargo)
    if not through:
        report("push-stalled")
    assert through, "cargo never made it through the doorway bore"
    step(120)  # coast to rest inside the court
    report("post-push")
    s = print_score("P4 cargo pushed through the bore + settled")
    assert s >= s_prev - 1e-6, "score decreased across the push"
    s_prev = s
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the push settled)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 — die NOW, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
