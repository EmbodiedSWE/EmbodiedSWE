"""Teleport solution for CrownSocketScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i246`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. PLUG EXTRACTION (contact dynamics — the core interaction; teleporting the plug out
   of the well would bypass the task and is never done): a regulated vertical force
   (velocity-servoed at 0.25 m/s, gravity feedforward, capped at 12 N) pulls the
   seated plug straight up. The plug's body slides up through the 4 mm/side well
   clearance, past the rim, into free air — the whole 7.5 cm escape travel happens
   under an applied force against gravity and wall contacts. The solve ASSERTS the
   plug started seated (socket occupied) and rose >= 5 cm UNDER THE FORCE before any
   pose write touches it.
2. TRANSPORT (teleport x2, free space only): (a) ONE pose write moves the extracted
   plug — at that instant hovering in free air above the rim — down to an empty patch
   of floor away from the cabinet and the bowl (discard). (b) ONE pose write moves
   the black bowl from its floor slot to a hover over the vacated socket, bowl bottom
   1 cm ABOVE the rim — touching nothing, above the seated z band, so neither
   teleport lands inside any scoring band (asserted).
3. SEATING (contact dynamics): the bowl is released from the hover and falls through
   the rim aperture under gravity; the drop, any wall contact, and settling on the
   well floor are pure physics. The socket physically admits the bowl ONLY because
   the plug is out — with the plug seated there is no room (the smoke test proves
   the converse).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: vacated and
seated are persistent physical states), then holds HANDS-OFF for >= 3 simulated
seconds after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i246.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

PLUG_DISCARD = (0.10, -0.45)  # empty floor: clear of the cabinet and both bowl slots


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crown_socket")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        lp, lb = scene._local(scene.plug)[0], scene._local(scene.bowl)[0]
        print(f"[solve] {tag:12s} | plug_loc=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):.3f}) bowl_loc=({float(lb[0]):+.3f},{float(lb[1]):+.3f},"
              f"{float(lb[2]):.3f}) occ={bool(scene.plug_occupies()[0])} "
              f"vac={bool(scene.vacated()[0])} seated={bool(scene.bowl_seated()[0])} "
              f"still={bool(scene.bowl_still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cab = scene._cab_xy[0]
    yaw_deg = float(torch.rad2deg(scene._cab_yaw[0]))
    lb0 = scene._local(scene.bowl)[0]
    bowl_w0 = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): cab=({float(cab[0]):+.3f},"
          f"{float(cab[1]):+.3f}) yaw={yaw_deg:+.1f}deg "
          f"bowl=({float(bowl_w0[0]):+.3f},{float(bowl_w0[1]):+.3f},{float(bowl_w0[2]):.3f})",
          flush=True)
    report("reset")
    assert bool(scene.plug_occupies()[0]), "plug did not settle seated in the socket"
    assert float(bowl_w0[2]) < 0.05, "bowl did not settle standing on the floor"
    assert not bool(scene.bowl_seated()[0]), "bowl must not read seated at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "score not ~0 at reset (null policy must earn nothing)"

    # ---------------- phase 1: EXTRACT THE PLUG (contact dynamics) --------------------------
    # Regulated vertical pull at the plug: gravity feedforward + velocity servo at
    # 0.25 m/s, capped at 12 N. The body slides up through the well's 4 mm/side
    # clearance and past the rim entirely under the applied force.
    z_start = float(scene._local(scene.plug)[0, 2])
    mg = c.plug_mass * 9.81
    pulled = 0
    for _ in range(900):
        lz = float(scene._local(scene.plug)[0, 2])
        if lz > 0.51:  # bottom 3.5 cm above the rim: fully clear, well outside occ band
            break
        vz = float(scene.plug.data.root_lin_vel_w[0, 2])
        fz = max(0.0, min(12.0, mg + 10.0 * (0.25 - vz)))
        f_w = torch.tensor([0.0, 0.0, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.plug.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        pulled += 1
    z_end = float(scene._local(scene.plug)[0, 2])
    print(f"[solve] extraction: pulled {pulled} steps, plug bottom {z_start:.3f} -> "
          f"{z_end:.3f} m", flush=True)
    assert z_end - z_start >= 0.05, \
        "plug did not rise >= 5 cm under the applied force (extraction not demonstrated)"
    assert not bool(scene.plug_occupies()[0]), "plug still occupies the socket after the pull"
    # Cut the force, then discard: ONE pose write from free air above the rim to empty
    # floor (transport only — the escape travel already happened under force).
    scene.plug.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = PLUG_DISCARD[0], PLUG_DISCARD[1]
    st[:, 2] = 0.01
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.plug.write_root_state_to_sim(st, all_ids)
    step(60)
    report("vacated")
    plug_w = (scene.plug.data.root_pos_w - scene.env_origins)[0]
    assert float(plug_w[2]) < 0.05, "discarded plug did not settle on the floor"
    assert bool(scene.vacated()[0]), "socket does not read vacated after the discard"
    s1 = print_score("P1 plug extracted under force and discarded")
    assert s1 >= 0.35, "vacated credit missing"
    assert s1 >= s0 - 1e-6, "score decreased across the extraction"

    # ---------------- phase 2: TRANSPORT the bowl (one free-space teleport) -----------------
    # Endpoint: centred over the vacated socket, root z = 0.510 -> bowl bottom 0.485,
    # 1 cm ABOVE the rim (0.475) — free space, and above the seated z band (<= 0.478),
    # so the teleport satisfies nothing.
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = scene._cab_xy[:, 0], scene._cab_xy[:, 1]
    st[:, 2] = 0.510
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.bowl.write_root_state_to_sim(st, all_ids)
    report("hover")
    assert not bool(scene.bowl_seated()[0]) and not bool(scene.success()[0]), \
        "hovering above the rim must not read seated (teleport is transport only)"
    s2 = print_score("P2 transport: bowl hovering above the vacated socket")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: SEATING (gravity drop + settle, pure physics) ----------------
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = bool(scene.bowl_seated()[0]) and bool(scene.bowl_still()[0])
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("seated")
    assert bool(scene.bowl_seated()[0]), \
        "bowl did not settle seated on the well floor after the drop"
    s3 = print_score("P3 bowl dropped in and seated (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the seating"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
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
