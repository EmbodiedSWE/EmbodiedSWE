"""Teleport solution for SwingArrestScene (sim_gen task `draw_svg_i49`) — the
task's legitimacy certificate.

Teleportation is TRANSPORT ONLY: an arrestor block is teleported to an empty spot
of bench (never into contact with anything), and later teleported away again.
Everything load-bearing happens through CONTACT DYNAMICS: the swinging bob strikes
the block's post face, the inelastic impact plus bench friction under the slab
soak up the swing energy, the pendulum settles resting near plumb, and after the
block is taken away the tiny residual swing is already below the stillness gates
and shrinks further under the rod's angular damping. The only applied force is a hold-down on the block DURING the
strike (the robot's hand steadying the block against impact recoil) — it is
removed before anything is judged clean, and the final state is fully HANDS-OFF.

Plan (oracle side), per pendulum, in any order:
  1. read the pendulum's live phase; wait until the bob is far out on one side S;
  2. teleport an arrestor block onto the bench on side -S, its strike face a few
     millimetres past the bob's plumb position, aligned to the gantry's live yaw;
     press the block down (hold-down force);
  3. the bob swings back through plumb and strikes the post: restitution is zero
     and the block+bench friction is high, so bob and block stop together within
     millimetres — the bob comes to rest leaning on the post ~1 degree past plumb;
  4. release the hold-down, teleport the block to a parking spot far outside the
     exclusion radius; the bob's ~1 degree residual swing sits below the
     stillness gates by design and keeps shrinking under the rod damping;
  5. re-place and retry if a strike goes wrong (readback on the arrest latch).
Then wait for success() (both plumb + sustained still + blocks clear + gantries
home) and hold >= 3.3 simulated seconds hands-off, printing SIM_GEN_SCORE at each
phase boundary (non-decreasing: the scene's credit is latched) and
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds at the end.

Run (forge): python -u -m simgen_tasks.draw_svg_i49.solve --headless [--seed N]
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

PARK_XY = ((-0.45, -0.35), (0.45, -0.35))  # block parking spots (env-local, on bench)
HOLD_N = 45.0                              # hold-down force on the block during a strike


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_arrest")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(kk: int) -> None:
        for _ in range(kk):
            env.step(no_action)

    def report(tag: str) -> None:
        a = scene.angle()[0]
        w = scene.rod_avel()[0]
        print(f"[solve] {tag:16s} | th=({math.degrees(float(a[0])):+7.2f},"
              f"{math.degrees(float(a[1])):+7.2f}) deg  w=({float(w[0]):5.2f},"
              f"{float(w[1]):5.2f})  arrest={scene.arrest_latch[0].tolist()} "
              f"clean={scene.clean_latch[0].tolist()} "
              f"clear={scene.blocks_clear()[0].tolist()} "
              f"home={bool(scene.gantries_home()[0])} still={bool(scene.still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero_f = torch.zeros(n, 1, 3, device=device)

    def hold_down(j: int, newtons: float) -> None:
        """Press block j down (or release with 0). Pure -z force: immune to the
        pod's wrench-rotates-with-body quirk for a yaw-only body."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = -newtons
        scene.blocks[j].set_external_force_and_torque(f, zero_f, env_ids=all_ids)

    def teleport_block(j: int, xy_world: torch.Tensor, quat: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = xy_world
        st[:, 2] = scene.env_origins[:, 2] + c.bench_top + 0.003
        st[:, 3:7] = quat
        scene.blocks[j].write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: verify the pendulums really swing ----------------------------
    # (the scene's whole premise: stored kinetic energy at reset). Track amplitude
    # over ~1.5 periods; also a first look at the decay rate.
    amp = torch.zeros(n, 2, device=device)
    for _ in range(180):
        step(1)
        amp = torch.maximum(amp, scene.angle().abs())
    th0 = [math.degrees(float(scene.theta0[0, i])) for i in range(2)]
    print(f"[solve] readback (seed {args.seed}): release angles "
          f"({th0[0]:+.1f}, {th0[1]:+.1f}) deg, observed swing amplitudes "
          f"({math.degrees(float(amp[0, 0])):.1f}, {math.degrees(float(amp[0, 1])):.1f}) deg",
          flush=True)
    for i in range(2):
        assert float(amp[0, i]) > math.radians(25.0), \
            f"pendulum {i} is not swinging — scene premise broken"
    report("reset")
    s_prev = print_score("P0 reset (both pendulums swinging)")

    # ---------------- per pendulum: place block, strike, extract ----------------------------
    face_off = c.bob_size / 2 + c.strike_gap + c.slab_y / 2  # gantry-local |y| of block centre
    ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
    pi_t = torch.full((n,), math.pi, device=device)

    def place(i: int, j: int) -> None:
        """Wait for bob i far out on one side, then stand block j at the strike
        gap on the opposite side (aligned to the gantry's live pose)."""
        for _ in range(900):
            th = scene.angle()[:, i]
            if abs(float(th[0])) > math.radians(18.0):
                break
            step(1)
        else:
            print(f"SIM_GEN_SOLVE: FAIL (pendulum {i} never swung past 18 deg)", flush=True)
            os._exit(1)
        s_side = torch.sign(th)                       # bob side
        u = -s_side                                   # block side
        g = scene.gantries[i]
        gq = g.data.root_quat_w
        ydir = quat_apply(gq, ey)
        xy = g.data.root_pos_w[:, :2] + (u * face_off).unsqueeze(-1) * ydir[:, :2]
        # block front face (local -y) must face the gantry: flip yaw on the -y side
        quat = torch.where((u > 0).unsqueeze(-1), gq, scene_mod._qmul(gq, scene_mod._qz(pi_t)))
        teleport_block(j, xy, quat)
        hold_down(j, HOLD_N)
        print(f"[solve] pendulum {i}: block {j} placed on side "
              f"{'+y' if float(u[0]) > 0 else '-y'} (bob at "
              f"{math.degrees(float(th[0])):+.1f} deg)", flush=True)

    def arrest(i: int, j: int) -> None:
        nonlocal s_prev
        for attempt in range(3):
            place(i, j)
            done = False
            for _ in range(24):                       # <= 6 s for strike + settle
                step(30)
                if float(scene.arrest_latch[0, i]) > 0.0:
                    done = True
                    break
            hold_down(j, 0.0)
            if done:
                break
            print(f"[solve] pendulum {i}: no arrest after placement "
                  f"(attempt {attempt}), re-placing", flush=True)
        else:
            report("FAIL-arrest")
            print(f"SIM_GEN_SOLVE: FAIL (pendulum {i} would not arrest)", flush=True)
            os._exit(1)
        report(f"arrest-{i}")
        s_now = print_score(f"P{2 * i + 1} pendulum {i} arrested")
        assert s_now >= s_prev - 1e-6, "score decreased across arrest"
        s_prev = s_now

    def extract(i: int, j: int) -> None:
        nonlocal s_prev
        park = torch.tensor(PARK_XY[j], device=device).expand(n, 2)
        xy = scene.env_origins[:, :2] + park
        teleport_block(j, xy, scene_mod._qz(torch.zeros(n, device=device)))
        # hands-off: the ~1 deg residual swing is below the stillness gates
        for _ in range(20):
            step(30)
            if float(scene.clean_latch[0, i]) > 0.0:
                break
        report(f"extract-{i}")
        s_now = print_score(f"P{2 * i + 2} pendulum {i} clean (block parked)")
        assert s_now >= s_prev - 1e-6, "score decreased across extraction"
        s_prev = s_now
        if float(scene.clean_latch[0, i]) <= 0.0:
            print(f"SIM_GEN_SOLVE: FAIL (pendulum {i} never judged clean)", flush=True)
            os._exit(1)

    for i in range(2):        # long first, then short — but no order is required
        arrest(i, i)
        extract(i, i)

    # ---------------- final: wait for full success (everything still at once) ---------------
    for _ in range(30):
        if bool(scene.success()[0]):
            break
        step(30)
    report("all-still")
    s_prev = print_score("P5 all still + clear")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success after both arrests)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) ---------------
    hold = True
    for _ in range(10):       # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
