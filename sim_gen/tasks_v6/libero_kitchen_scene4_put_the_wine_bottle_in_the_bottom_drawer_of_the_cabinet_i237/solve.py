"""Teleport solution for TiltBinStowScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i237`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN THE BIN (contact dynamics — never teleported): a bounded torque about the real
   hinge (the applied-wrench emulation of the arm pulling the red handle bar) swings
   the bin against gravity from shut to past its centre-of-mass crossover; the drive is
   then REMOVED and the bin coasts onto its 48 deg limit stop, where GRAVITY ALONE
   holds it open. The bin's pose is never written after reset — its whole trajectory is
   joint physics.
2. TRANSPORT (teleport): one pose write carries the green wine bottle from its floor
   slot to a hover pose just ABOVE the open bin's mouth plane (verifiably OUTSIDE the
   containment volume — asserted), axis horizontal along the mouth's long axis. Hovering
   satisfies no rubric clause beyond the mouth-approach ramp (approach credit is exactly
   what carrying an object toward a goal earns; containment and closing credit remain 0).
3. DEPOSIT (contact dynamics): the bottle is RELEASED — it free-falls through the open
   mouth, strikes the tilted bin floor, slides down into the floor/front-wall V under
   real friction and settles INSIDE. No pose write touches the bottle from release to
   verdict.
4. CLOSE LOADED (contact dynamics): a reverse hinge torque (the arm pushing the bin
   face) drives the loaded bin back past crossover, the drive is removed, and gravity +
   the viscous hinge damping seat it SHUT on the lower limit with the bottle riding
   inside through real contacts. success() first turns True here, judged on settled
   physical state (containment in the shut bin, decoy excluded, everything at rest).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    env = ENVS.get("simgen.tilt_bin_stow")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg() -> float:
        return math.degrees(float(scene.bin_angle()[0]))

    def report(tag: str) -> None:
        w = (scene.wine.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | angle={deg():6.1f} deg "
              f"wine=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):.3f}) "
              f"in_bin={bool(scene._inside_bin(scene.wine)[0])} "
              f"decoy_in={bool(scene._inside_bin(scene.decoy)[0])} "
              f"opened={bool(scene._opened[0])} app={float(scene._app_max[0]):.3f} "
              f"dep={bool(scene._in[0])} close={float(scene._close_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drive_to(theta_t_deg: float, stop, max_steps: int, kp: float = 3.0,
                 kd: float = 0.35, cap: float = 2.0) -> bool:
        """Torque-servo the hinge toward theta_t (proportional drive on the applied
        wrench — the bin's pose is pure physics), until `stop()` or step budget."""
        theta_t = math.radians(theta_t_deg)
        for _ in range(max_steps):
            theta = float(scene.bin_angle()[0])
            omega = float(scene.bin.data.root_ang_vel_w[0, 1])
            tau = max(-cap, min(cap, kp * (theta_t - theta) - kd * omega))
            scene.bin_drive[:] = tau
            env.step(no_action)
            if stop():
                scene.bin_drive[:] = 0.0
                return True
        scene.bin_drive[:] = 0.0
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # the randomly-ajar bin falls shut under gravity here
    wine0 = (scene.wine.data.root_pos_w - scene.env_origins)[0]
    decoy0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"wine=({float(wine0[0]):+.3f},{float(wine0[1]):+.3f}) "
          f"decoy=({float(decoy0[0]):+.3f},{float(decoy0[1]):+.3f}) "
          f"bin_angle={deg():.1f} deg", flush=True)
    report("reset")
    assert deg() <= c.closed_tol_deg, "bin did not fall shut at reset"
    assert not bool(scene._inside_bin(scene.wine)[0]), "wine bottle spawned inside the bin"
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: OPEN through the hinge (contact dynamics) --------------------
    ok = drive_to(c.open_limit_deg - 2.0,
                  lambda: float(scene.bin_angle()[0]) >= math.radians(44.0), 600)
    assert ok, "hinge drive failed to swing the bin open"
    step(90)  # drive off: coast onto the limit stop, gravity holds it open
    report("opened")
    assert deg() >= 40.0, f"bin did not REST open (angle {deg():.1f} deg after release)"
    assert bool(scene._opened[0]), "opened latch did not fire"
    s1 = print_score("P1 bin swung open by hinge torque, resting on its stop")
    assert s1 >= s0 - 1e-6, "score decreased across opening"

    # ---------------- phase 2: TRANSPORT (teleport, provably outside) -----------------------
    # One pose write: wine bottle to 35 mm above the open mouth plane, axis along the
    # mouth's long axis (bin-local +y). The containment box tops out 4 mm above the
    # mouth plane, so this hover state is OUTSIDE the bin — asserted below.
    from isaaclab.utils.math import quat_apply, quat_mul

    hover_loc = torch.tensor([c.mouth_local[0], 0.0, c.bin_t + c.bin_in_h + 0.035],
                             device=device).expand(n, 3)
    hover_pos = scene.bin.data.root_pos_w + quat_apply(scene.bin.data.root_quat_w, hover_loc)
    q_rel = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                         device=device).expand(n, 4)  # bottle +z -> local +y
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = hover_pos
    st[:, 3:7] = quat_mul(scene.bin.data.root_quat_w, q_rel)
    scene.wine.write_root_state_to_sim(st, all_ids)
    report("transported")
    assert not bool(scene._inside_bin(scene.wine)[0]), \
        "hover pose is already inside the containment volume (teleport must stay outside)"
    s2 = print_score("P2 transport to a hover above the open mouth (still outside)")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: DEPOSIT through the mouth (gravity + contact) ----------------
    settled = False
    for i in range(420):
        env.step(no_action)
        if (i > 30 and bool(scene._inside_bin(scene.wine)[0])
                and float(scene.wine.data.root_lin_vel_w[0].norm()) < 0.03):
            settled = True
            break
    report("deposited")
    assert settled and bool(scene._inside_bin(scene.wine)[0]), \
        "wine bottle did not settle inside the open bin"
    s3 = print_score("P3 gravity deposit through the open mouth")
    assert s3 >= s2 - 1e-6, "score decreased across the deposit"

    # ---------------- phase 4: CLOSE the loaded bin (contact dynamics) ----------------------
    ok = drive_to(-3.0, lambda: float(scene.bin_angle()[0]) <= math.radians(3.0), 600,
                  kp=2.0, kd=0.30)
    assert ok, "hinge drive failed to push the loaded bin shut"
    # Drive off: seat on the lower limit, then WAIT for genuine stillness (the lying
    # bottle is a roller — it can rock inside the shut bin for a while).
    quiet = 0
    for _ in range(900):
        env.step(no_action)
        still = (float(scene.wine.data.root_lin_vel_w[0].norm()) < 0.03
                 and float(scene.bin.data.root_ang_vel_w[0].norm()) < 0.10)
        quiet = quiet + 1 if still else 0
        if quiet >= 30:
            break
    report("shut")
    assert deg() <= c.closed_tol_deg, f"bin did not rest shut ({deg():.1f} deg)"
    assert bool(scene._inside_bin(scene.wine)[0]), "wine bottle escaped during closing"
    s4 = print_score("P4 loaded bin pushed shut, settled")
    assert s4 >= s3 - 1e-6, "score decreased across closing"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after closing)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
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
    main()
