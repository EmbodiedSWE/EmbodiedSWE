"""Teleport solution for CliffSweepScene (sim_gen task `approach_grasp_bowl_i133`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, and every teleport of the scoop ends in FREE
SPACE: the scoop is carried to a hover pose ~5 cm above its drop spot (wall bottoms
clear of every ball top), oriented mouth-toward-cliff, with velocities zeroed — then
GRAVITY lowers it over the balls through real contact. The judged objects (the balls)
are NEVER teleported. Everything load-bearing happens through CONTACT DYNAMICS:
  - CAGE: the dropped dome's walls surround the ball pack on the stage floor;
  - PLOW: a velocity-servoed horizontal force on the scoop (applied at wall-bottom
    height via the equivalent COM force + r x F counter-torque, plus a yaw-hold
    torque keeping the mouth toward the cliff) slides the cage across the stage,
    the back wall pushing the rolling pack;
  - RELEASE: the scoop BRAKES short of the edge; the balls coast on out through the
    90-degree mouth, roll over the red-striped cliff edge, and gravity drops them
    into the basin — a curling-style delivery; nothing is ever lifted or carried
    (the spoil latch at `lift_z` stays quiet, and smoke proves a carry trips it);
  - CLEANUP: stragglers left on the stage are re-covered (hover-drop with clamped
    xy so the descent path is clear of rails and cliff) and re-plowed, up to 6 times;
  - PARK: the scoop is hover-dropped back onto the open stage and left at rest.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i133.solve --headless [--seed N]
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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cliff_sweep")().build(num_envs=args.num_envs, device=device)
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

    side = float(scene.side[0])  # +1: cliff on +x, -1: cliff on -x
    psi_des = 0.0 if side > 0 else math.pi  # mouth (+x body) toward the cliff

    def balls_xy() -> torch.Tensor:
        """(3,2) ball centres, env-local."""
        return scene._ball_pos()[0, :, :2]

    def balls_z() -> torch.Tensor:
        return scene._ball_pos()[0, :, 2]

    def scoop_p() -> torch.Tensor:
        """(3,) scoop origin, env-local."""
        return scene._scoop_pos()[0]

    def scoop_yaw() -> float:
        q = scene.scoop.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def in_basin_now() -> torch.Tensor:
        """(3,) bool."""
        return scene._in_basin(scene._ball_pos())[0]

    def clear_wrench() -> None:
        scene.scoop.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        bp = scene._ball_pos()[0]
        inb = in_basin_now()
        sp = scoop_p()
        print(f"[solve] {tag:14s} | balls_mx=[" + ",".join(
            f"{float(bp[i, 0]) * side:+.3f}" for i in range(3)) + "] balls_z=[" + ",".join(
            f"{float(bp[i, 2]):.3f}" for i in range(3)) + "] in_basin=[" + ",".join(
            "T" if bool(inb[i]) else "f" for i in range(3)) + "] "
            f"scoop=({float(sp[0]) * side:+.3f},{float(sp[1]):+.3f},{float(sp[2]):.3f}) "
            f"captured={float(scene.captured[0]):.0f} "
            f"delivered={float(scene.delivered[0].sum()):.0f} "
            f"spoiled={bool(scene.spoiled[0])} parked={bool(scene.scoop_parked()[0])} "
            f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
            flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- transport + contact primitives ---------------------------------------------------
    hover_z = c.stage_h + 2 * c.ball_r + c.scoop_wall_h / 2 + 0.010  # wall bottoms clear balls

    def hover_drop(wx: float, wy: float) -> None:
        """TRANSPORT: one teleport to a free-space hover (mouth toward the cliff,
        velocities zero), then hands-off — gravity lowers the dome over whatever is
        below through real contact."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = wx
        st[:, 1] = wy
        st[:, 2] = hover_z
        st[:, 3] = math.cos(psi_des / 2)
        st[:, 6] = math.sin(psi_des / 2)
        st[:, 0:3] += scene.env_origins
        scene.scoop.write_root_state_to_sim(st, all_ids)
        step(100)  # gravity drop + settle (~0.83 s)

    # Cover-centre clamps: cap edge clear of the side rails, the end rail, and the cliff.
    cover_y_max = c.half_y - c.cap_r - 0.006          # ~0.166
    cover_mx_min = -(c.end_rail_x - c.rail_t / 2 - c.cap_r - 0.006)  # ~-0.274
    cover_mx_max = 0.285                              # front cap edge ~0.373 < cliff 0.38

    def cover(target_mx: float, target_y: float) -> None:
        mx = min(max(target_mx, cover_mx_min), cover_mx_max)
        wy = min(max(target_y, -cover_y_max), cover_y_max)
        hover_drop(side * mx, wy)

    def mec_center(pts: torch.Tensor) -> torch.Tensor:
        """(2,) approx. min-enclosing-circle centre of (k,2) points (iterative pull
        toward the farthest point) — minimizes the worst ball offset under the cage."""
        ctr = pts.mean(dim=0)
        for _ in range(40):
            d = (pts - ctr).norm(dim=-1)
            j = int(d.argmax())
            ctr = ctr + 0.25 * (pts[j] - ctr)
        return ctr

    def plow(stop_mx: float, hold_y: float, v_slow: float = 0.14, v_fast: float = 0.34,
             switch_mx: float = 0.12, max_steps: int = 1600) -> None:
        """CONTACT: velocity-servo the scoop along mirrored +x to `stop_mx`, holding y
        and yaw; force is applied at wall-bottom height (COM force + r x F torque with
        r=(0,0,-wall_h/2)) so the dome slides flat instead of pitching; then BRAKE hard
        and let the caged balls coast out through the mouth and over the cliff."""
        K, m = 60.0, c.scoop_mass  # K*dt = 0.5 < 1 (wrench acts one substep late)
        for _ in range(max_steps):
            sp = scoop_p()
            mx = float(sp[0]) * side
            if mx > stop_mx:
                break
            v_mag = v_slow if mx < switch_mx else v_fast
            v = scene.scoop.data.root_lin_vel_w[0, :2]
            vy_des = min(max(2.0 * (hold_y - float(sp[1])), -0.08), 0.08)
            v_des = torch.tensor([side * v_mag, vy_des], device=device)
            f_xy = m * K * (v_des - v)
            f_xy[0] += side * 1.2  # friction feedforward along the plow direction
            f_xy = f_xy.clamp(-6.0, 6.0)
            r_z = -c.scoop_wall_h / 2  # push acts at the wall bottoms, not the COM
            tq_xy = (r_z * -float(f_xy[1]), r_z * float(f_xy[0]))
            wz = float(scene.scoop.data.root_ang_vel_w[0, 2])
            tz = min(max(0.03 * _wrap(psi_des - scoop_yaw()) - 0.006 * wz, -0.06), 0.06)
            f = torch.tensor([float(f_xy[0]), float(f_xy[1]), 0.0], device=device)
            tq = torch.tensor([tq_xy[0], tq_xy[1], tz], device=device)
            scene.scoop.set_external_force_and_torque(
                f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        # Brake: kill the scoop's velocity; the pack coasts on out of the mouth.
        for _ in range(300):
            v = scene.scoop.data.root_lin_vel_w[0, :2]
            if float(v.norm()) < 0.03:
                break
            f_xy = (m * 80.0 * -v).clamp(-8.0, 8.0)
            r_z = -c.scoop_wall_h / 2
            f = torch.tensor([float(f_xy[0]), float(f_xy[1]), 0.0], device=device)
            tq = torch.tensor([r_z * -float(f_xy[1]), r_z * float(f_xy[0]), 0.0],
                              device=device)
            scene.scoop.set_external_force_and_torque(
                f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_wrench()
        step(200)  # hands-off: balls fly, drop, and start settling (~1.7 s)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    bxy0 = balls_xy()
    sp0 = scoop_p()
    print(f"[solve] layout readback (seed {args.seed}): side={side:+.0f} balls=[" + " ".join(
        f"({float(bxy0[i, 0]):+.3f},{float(bxy0[i, 1]):+.3f})" for i in range(3)) + "] "
        f"scoop=({float(sp0[0]):+.3f},{float(sp0[1]):+.3f}) yaw={scoop_yaw():+.2f}",
        flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: cover the pack (transport hover + gravity drop) --------------
    ctr = mec_center(balls_xy())
    worst = float((balls_xy() - ctr).norm(dim=-1).max())
    print(f"[solve] cover centre=({float(ctr[0]):+.3f},{float(ctr[1]):+.3f}) "
          f"worst_offset={worst:.3f} (cage inner {c.inner_r:.3f})", flush=True)
    cover(float(ctr[0]) * side, float(ctr[1]))
    d = (balls_xy() - scoop_p()[:2]).norm(dim=-1)
    print(f"[solve] caged dists: [" + ",".join(f"{float(v):.3f}" for v in d)
          + f"] capture_r={c.capture_r:.3f}", flush=True)
    report("covered")
    s1 = print_score("P1 cover (transport + drop)")
    assert s1 >= s0 - 1e-6, "score decreased across cover"

    # ---------------- phase 2: plow the pack over the cliff ---------------------------------
    plow(stop_mx=0.285, hold_y=float(scoop_p()[1]))
    report("plow-1")
    s2 = print_score("P2 first plow + release")
    assert s2 >= s1 - 1e-6, "score decreased across the plow"

    # ---------------- phase 3: cleanup — re-cover and re-plow stragglers --------------------
    for attempt in range(6):
        step(120)  # let everything finish rolling before judging stragglers
        inb = in_basin_now()
        if bool(inb.all()):
            break
        stragglers = [i for i in range(3) if not bool(inb[i])]
        bxy = balls_xy()
        bz = balls_z()
        # Target the straggler farthest from the cliff (plow sweeps the rest forward too).
        i = min(stragglers, key=lambda j: float(bxy[j, 0]) * side)
        print(f"[solve] cleanup {attempt}: stragglers={stragglers} target ball {i} at "
              f"(mx={float(bxy[i, 0]) * side:+.3f},y={float(bxy[i, 1]):+.3f},"
              f"z={float(bz[i]):.3f})", flush=True)
        cover(float(bxy[i, 0]) * side + 0.04, float(bxy[i, 1]))
        report(f"re-covered-{attempt}")
        plow(stop_mx=0.285, hold_y=float(scoop_p()[1]), v_slow=0.12, v_fast=0.30)
        report(f"re-plow-{attempt}")
    s3 = print_score("P3 cleanup plows")
    assert s3 >= s2 - 1e-6, "score decreased across cleanup"
    if not bool(in_basin_now().all()):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (balls still on the stage after cleanup)", flush=True)
        os._exit(1)

    # ---------------- phase 4: park the scoop back on the stage -----------------------------
    cover(-0.20, 0.0)  # hover-drop onto the open closed-end floor
    step(120)
    report("parked")
    s4 = print_score("P4 park")
    assert s4 >= s3 - 1e-6, "score decreased across parking"
    if not bool(scene.success()[0]):
        step(240)  # basin balls may still be creeping — give them 2 more seconds
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after park+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.4 s")
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
    main()
