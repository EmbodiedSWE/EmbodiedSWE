"""Teleport solution for CellarRollStowScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i250`)
— the task's legitimacy certificate.

Every teleport here is TRANSPORT ONLY; each load-bearing interaction runs through
contact dynamics:

1. UNBAR (transport): the red bar rests freely in upward-open U-notches — lifting it
   out is unobstructed carrying, so one pose write parks it on the open floor, well
   clear of both bays. Nothing is skipped: the notches have no latch, spring, or cover.
2. LAY THE BOTTLE (transport): one pose write carries the green bottle from its random
   spawn to a lying pose (axis across the ramp, along world y) hovering 4 mm above the
   TARGET ramp, verifiably OUTSIDE the bay containment box (asserted). This is exactly
   the reorient-and-place a gripper would do; no containment credit is granted by it.
3. ROLL-IN DEPOSIT (contact dynamics): the bottle is RELEASED — gravity rolls it down
   the 7.4 deg ramp, through the letterbox slot, onto the inward-pitched bay floor,
   where it parks against the back wall and settles. No pose write touches the bottle
   from release to verdict.
4. RE-BAR (transport then contact dynamics): one pose write hovers the bar level in the
   notch gap ABOVE the seat window (asserted not-yet-seated), then it is RELEASED —
   gravity drops it ~15 mm onto the post tops between the prongs, where it seats and
   stills. success() first turns True here, judged on settled physical state.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cellar_roll_stow")().build(num_envs=args.num_envs, device=device)
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

    # Mass readback sanity (custom spawners: MassAPI must actually have taken).
    mw = float(scene.wine.root_physx_view.get_masses()[0].sum())
    mb = float(scene.bar.root_physx_view.get_masses()[0].sum())
    print(f"[solve] masses: wine={mw:.3f} kg bar={mb:.3f} kg", flush=True)
    assert abs(mw - c.wine_mass) < 0.02, f"wine mass wrong ({mw})"
    assert abs(mb - c.bar_mass) < 0.02, f"bar mass wrong ({mb})"

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        w = rel(scene.wine)
        b = rel(scene.bar)
        print(f"[solve] {tag:12s} | wine=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):.3f}) "
              f"bar=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"side={float(scene.target_sign[0]):+.0f} "
              f"in_bay={bool(scene.bottle_in_bay()[0])} lying={bool(scene.bottle_lying()[0])} "
              f"seated={bool(scene.bar_seated()[0])} unbar={bool(scene._unbarred[0])} "
              f"app={float(scene._app_max[0]):.3f} stow={bool(scene._stowed[0])} "
              f"rebar={float(scene._rebar_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_until(cond, budget: int, streak_need: int = 40) -> bool:
        """Step until `cond()` has held for `streak_need` consecutive steps."""
        streak = 0
        for _ in range(budget):
            env.step(no_action)
            streak = streak + 1 if cond() else 0
            if streak >= streak_need:
                return True
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # everything settles: bar onto its notch seat, bottle onto its base
    sign = float(scene.target_sign[0])
    wine0 = rel(scene.wine)
    bar0 = rel(scene.bar)
    print(f"[solve] layout readback (seed {args.seed}): target side {sign:+.0f} "
          f"wine=({float(wine0[0]):+.3f},{float(wine0[1]):+.3f}) "
          f"bar_y={float(bar0[1]):+.4f}", flush=True)
    report("reset")
    assert bool(scene.bar_seated()[0]), "bar did not start seated in the target notches"
    assert not bool(scene.bottle_in_bay()[0]), "bottle spawned inside the bay"
    assert not bool(scene.bottle_lying()[0]), "bottle did not spawn standing"
    assert not bool(scene._unbarred[0]), "unbar latch fired at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score not ~0 ({s0})"

    # ---------------- phase 1: UNBAR (transport — lift the free bar out, park it) -----------
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = 0.08
    st[:, 1] = 0.35 * torch.as_tensor(scene.target_sign, device=device)  # park on target side, clear of both bays
    st[:, 2] = c.bar_size[2] / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.bar.write_root_state_to_sim(st, all_ids)
    step(60)  # the parked bar settles flat on the ground
    report("unbarred")
    assert bool(scene._unbarred[0]), "unbar latch did not fire"
    assert not bool(scene.bar_seated()[0]), "bar still reads seated after parking"
    s1 = print_score("P1 bar lifted out of its notches and parked aside")
    assert s1 >= s0 - 1e-6, "score decreased across unbar"

    # ---------------- phase 2: LAY THE BOTTLE (transport, provably outside) -----------------
    # One pose write: bottle lying (local +z -> world +y) hovering 4 mm above the
    # TARGET ramp at x=0.315 — outside the bay containment box (asserted).
    th = math.atan2(c.ramp_z0 - c.ramp_z1, c.ramp_x1 - c.ramp_x0)
    lay_x = 0.315
    ramp_z = c.ramp_z0 - (lay_x - c.ramp_x0) * math.tan(th)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = lay_x
    st[:, 1] = c.bay_cy * torch.as_tensor(scene.target_sign, device=device)
    st[:, 2] = ramp_z + c.body_r + 0.004
    st[:, 3] = math.cos(math.pi / 4)  # rotate +90 deg about x: local +z -> world -y
    st[:, 4] = math.sin(math.pi / 4)
    st[:, 0:3] += scene.env_origins
    scene.wine.write_root_state_to_sim(st, all_ids)
    report("laid")
    assert bool(scene.bottle_lying()[0]), "laid pose is not lying"
    assert not bool(scene.bottle_in_bay()[0]), \
        "laid pose is already inside the containment box (teleport must stay outside)"
    s2 = print_score("P2 bottle laid on the target ramp (still outside the bay)")
    assert s2 >= s1 - 1e-6, "score decreased across the lay"

    # ---------------- phase 3: ROLL-IN DEPOSIT (gravity + contact) --------------------------
    def rolled_home() -> bool:
        return (bool(scene.bottle_in_bay()[0]) and bool(scene.bottle_lying()[0])
                and float(scene.wine.data.root_lin_vel_w[0].norm()) < 0.03)

    def rolled_home_traced() -> bool:
        nonlocal trace_i
        trace_i += 1
        if trace_i % 120 == 0:
            w = rel(scene.wine)
            print(f"[solve]   roll t={trace_i:4d} x={float(w[0]):+.3f} z={float(w[2]):.3f} "
                  f"|v|={float(scene.wine.data.root_lin_vel_w[0].norm()):.4f} "
                  f"|w|={float(scene.wine.data.root_ang_vel_w[0].norm()):.3f}", flush=True)
        return rolled_home()

    trace_i = 0
    ok = settle_until(rolled_home_traced, budget=900, streak_need=40)
    report("rolled-in")
    assert ok and bool(scene.bottle_in_bay()[0]), \
        "bottle did not roll through the slot and settle inside the target bay"
    assert bool(scene._stowed[0]), "stowed latch did not fire"
    s3 = print_score("P3 gravity roll through the letterbox slot, parked at the back wall")
    assert s3 >= s2 - 1e-6, "score decreased across the roll-in"

    # ---------------- phase 4: RE-BAR (transport to a hover, then gravity drop-seat) --------
    # Hover the bar level in the notch gap ABOVE the seat window (seat_z_hi=0.064; the
    # prong gap is 34 mm wide vs the 20 mm bar, so the hover is collision-free), then
    # release: gravity drops it ~15 mm onto the post tops where it seats.
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = c.post_x
    st[:, 1] = c.bay_cy * torch.as_tensor(scene.target_sign, device=device)
    st[:, 2] = 0.070
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.bar.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.bar_seated()[0]), \
        "hover pose is already inside the seat window (drop-seat must run through gravity)"

    def reseated() -> bool:
        return (bool(scene.bar_seated()[0])
                and float(scene.bar.data.root_lin_vel_w[0].norm()) < 0.03
                and float(scene.wine.data.root_lin_vel_w[0].norm()) < 0.03)

    ok = settle_until(reseated, budget=600, streak_need=40)
    report("rebarred")
    assert ok and bool(scene.bar_seated()[0]), "bar did not drop-seat back into its notches"
    s4 = print_score("P4 bar drop-seated back into the target notches")
    assert s4 >= s3 - 1e-6, "score decreased across the re-bar"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after re-barring)", flush=True)
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
