"""Teleport solution for LetterboxBinScene (sim_gen task `sweep_to_dustpan_i156`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY: each present RED cube is carried (pose write through free
space) to a hover just above the staging LEDGE in front of the slot and dropped there.
The load-bearing interaction — getting a cube INTO the sealed bin — is executed through
contact dynamics, every time:

1. PRESS THROUGH THE FLAP (applied horizontal force + hinge/contact physics): the cube
   is pushed at its CoM straight at the slot (the push a fingertip or closed gripper
   produces), velocity-regulated to <= 0.08 m/s, force capped at 1.8 N. The cube slides
   along the ledge runway, presses the yellow one-way flap, the flap swings inward on
   its gravity hinge, the cube crosses the sill and tumbles into the bin, and the flap
   falls shut behind it. The cube is NEVER teleported inside, over the sill, or past
   the flap: every entry goes through the flap under contact. The force-frame mode
   (some pods rotate applied wrenches by the body's rotation since reset) is PROBED
   from measured progress and toggled if the cube regresses — never assumed. A stiction
   floor escalates on measured stall (ledge friction is a physical unknown).
2. IDENTITY: only the PRESENT red cubes are posted (present count is sampled per
   episode and read back); the two blue keep cubes are never touched — leaving them
   outside is part of the goal.
3. NO teleport ever enters the bin: hover targets sit outside the front wall plane,
   above the ledge, in free air.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.sweep_to_dustpan_i156.solve --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_bin")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                     f_world: torch.Tensor) -> torch.Tensor:
        """Pre-encode a desired WORLD force. mode 0: pass through. mode 1: premultiply
        by R_ref * R_now^T for pods that rotate wrenches by rotation-since-reset."""
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_world)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def loc_of(body) -> torch.Tensor:
        return scene._bin_local(body.data.root_pos_w)[0]

    def flap_deg() -> float:
        return float(scene.flap_open_deg()[0])

    def report(tag: str) -> None:
        ins = scene._inserted[0].tolist()
        pres = scene.present_red[0].tolist()
        print(f"[solve] {tag:14s} | flap={flap_deg():+6.1f}deg present={pres} "
              f"inserted={ins} appr={bool(scene._appr[0])} "
              f"contam={bool(scene._contam[0])} "
              f"n_red_in={int(scene.red_inside()[0].sum())} "
              f"n_blue_in={int(scene.blue_inside()[0].sum())} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)
        locs = ", ".join(
            f"{nm}=({float(l[0]):+.3f},{float(l[1]):+.3f},{float(l[2]):+.3f})"
            for nm, l in ((nm, loc_of(b)) for nm, b in scene.reds.items()))
        print(f"[solve]   red loc (bin frame): {locs}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    bx, by = c.bin_pos
    origin = scene.env_origins[0]
    ledge_x = bx - c.half_x - c.ledge_len / 2          # ledge runway centre (world x)
    hover_z = c.sill_z + c.cube_size / 2 + 0.020       # drop height above the ledge

    def place_on_ledge(body, jitter: float) -> bool:
        """TRANSPORT teleport: carry the cube to a hover above the ledge, drop, settle.
        Returns True when it rests on the ledge runway."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = ledge_x + jitter
        st[:, 1] = by
        st[:, 2] = hover_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)
        step(70)
        loc = loc_of(body)
        v = float(body.data.root_lin_vel_w[0].norm())
        on = (-c.half_x - c.ledge_len - 0.005 < float(loc[0]) < -c.half_x + c.wall_t
              and abs(float(loc[1])) < 0.05
              and c.sill_z + 0.005 < float(loc[2]) < c.sill_z + 0.030 and v < 0.10)
        print(f"[solve]   placed at loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) v={v:.3f} on_ledge={on}", flush=True)
        return on

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(180)
    present = scene.present_red[0].tolist()
    k = int(scene.present_red[0].sum())
    print(f"[solve] layout readback (seed {args.seed}): present_red={present} (k={k}) "
          f"flap={flap_deg():+.2f}deg", flush=True)
    for name, body in list(scene.reds.items()) + list(scene.blues.items()):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {name}: ({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(flap_deg()) < 3.0, f"flap must hang closed at reset, got {flap_deg():.1f}"
    assert k >= 1, "at least one red cube must be present"
    assert int(scene.red_inside()[0].sum()) == 0 and int(scene.blue_inside()[0].sum()) == 0
    s_prev = print_score("P0 reset+settle (bin sealed, debris scattered)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"

    # Force-frame reference orientations (readback at reset), one per red cube.
    q_ref = {name: b.data.root_quat_w.clone() for name, b in scene.reds.items()}

    # ---------------- per-cube: carry to the ledge, press through the flap ------------------
    posted = 0
    for i, (name, body) in enumerate(scene.reds.items()):
        if not present[i]:
            continue
        placed = False
        for attempt in range(4):
            if place_on_ledge(body, jitter=0.000 if attempt == 0 else -0.008 * attempt):
                placed = True
                break
        if not placed:
            print("SIM_GEN_SOLVE: FAIL (could not stage the cube on the ledge)", flush=True)
            os._exit(1)

        # press through the flap: +x velocity servo, stiction floor, tipping-safe cap
        mode, floor_f = 0, 0.5
        win_i, win_x = 0, float(loc_of(body)[0])
        max_flap = 0.0
        entered = False
        for j in range(1800):
            loc = loc_of(body)
            x_loc, z_loc = float(loc[0]), float(loc[2])
            if bool(scene.red_inside()[0, i]) or (x_loc > -c.half_x + c.wall_t
                                                  and z_loc < c.sill_z - 0.010):
                entered = True
                break
            if z_loc < 0.09 and x_loc < -c.half_x - 0.010:
                print(f"[solve]   {name} fell off the ledge; restaging", flush=True)
                clear_forces(body)
                if not place_on_ledge(body, jitter=-0.010):
                    print("SIM_GEN_SOLVE: FAIL (restage failed)", flush=True)
                    os._exit(1)
                win_i, win_x = j, float(loc_of(body)[0])
                continue
            v = body.data.root_lin_vel_w[0, :2]
            v_des = torch.tensor([0.08 if x_loc < -c.half_x - 0.005 else 0.05, 0.0],
                                 device=device)
            f_xy = 6.0 * (v_des - v)
            f_x = float(f_xy[0])
            if float(v.norm()) < 0.02 and f_x < floor_f:
                f_xy[0] += floor_f - f_x  # break stiction
            fn = float(f_xy.norm())
            if fn > 1.8:
                f_xy = f_xy * (1.8 / fn)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = f_xy
            f_arg = encode_force(mode, q_ref[name], body.data.root_quat_w, f_world)
            body.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_wrench,
                                               env_ids=all_ids, is_global=True)
            env.step(no_action)
            max_flap = max(max_flap, flap_deg())
            # progress probe: wrong force-frame mode pushes the cube AWAY
            if j - win_i >= 45:
                x_now = float(loc_of(body)[0])
                if x_now < win_x - 0.008:
                    mode = 1 - mode
                    print(f"[solve]   {name}: moving away (x {win_x:+.3f} -> "
                          f"{x_now:+.3f}); force-frame mode -> {mode}", flush=True)
                elif x_now < win_x + 0.003:
                    floor_f = min(floor_f + 0.25, 1.6)
                    print(f"[solve]   {name}: stalled at x={x_now:+.3f} "
                          f"flap={flap_deg():+.1f}deg; stiction floor -> {floor_f:.2f} N",
                          flush=True)
                win_i, win_x = j, x_now
        clear_forces(body)
        step(150)  # cube settles on the bin floor, flap swings shut, hands-off
        report(f"posted {name}")
        if not entered:
            print(f"SIM_GEN_SOLVE: FAIL ({name} never entered the bin)", flush=True)
            os._exit(1)
        assert max_flap > 15.0, \
            f"the flap must have been pressed open (max {max_flap:.1f} deg) — no bypass"
        assert bool(scene._inserted[0, i]), f"insert latch must have fired for {name}"
        posted += 1
        s = print_score(f"P{posted} {name} pressed through the flap ({posted}/{k})")
        assert s >= s_prev - 1e-6, f"score decreased: {s_prev} -> {s}"
        s_prev = s

    # ---------------- final: flap shut, judge, persist --------------------------------------
    step(120)
    report("final")
    assert abs(flap_deg()) < c.flap_closed_deg, \
        f"flap must hang closed, got {flap_deg():.1f} deg"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after all reds posted)", flush=True)
        os._exit(1)
    s_f = print_score(f"P{posted + 1} all {k} present reds inside, blues out, flap shut")
    assert s_f >= 0.999, f"final score {s_f} (expect success=1.0)"

    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_p = print_score("persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_p >= s_f - 1e-6
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
