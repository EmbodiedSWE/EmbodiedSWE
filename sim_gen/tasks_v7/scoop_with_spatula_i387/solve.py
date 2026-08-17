"""Teleport solution for QuarryScene (sim_gen task `scoop_with_spatula_i387`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. UNCOVERING (the occlusion structure is real): the prize starts buried — the
   capping stone rests ON the prize by contact (verified at P0: `covered()` is True
   and the cap's weight is carried by the prize). Each stone is removed TOP-DOWN in
   current-height order, exactly the accessibility order a gripper faces.
2. TRANSPORT (teleport, one stone at a time): a single root-state write carries a
   stone from the pile to a free-space HOVER above an empty drop slot in the EAST
   half of the pit (zero velocity). The write satisfies no rubric clause by itself
   — every latch requires the stone CALM and the containment invariant live.
3. RELOCATION (gravity + contact, hands-off): from the hover the stone FALLS onto
   the pit floor and settles inside the walls. Nothing is ever written into a
   settled pose.
4. EXTRACTION + ENTHRONEMENT: only after `covered()` reads False is the prize
   itself carried — a hover a few centimetres ABOVE the open pedestal pocket, then
   a hands-off DROP into the pocket where it seats against the pocket floor and
   walls by contact. The prize is never written into the pocket.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.scoop_with_spatula_i387.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

assert scene_mod is not None

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.quarry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

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
        tl = scene._token_local()[0]
        print(f"[solve] {tag:12s} | token=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):+.3f}) covered={bool(scene.covered()[0])} "
              f"all_in_pit={bool(scene.all_in_pit()[0])} "
              f"seated={bool(scene.token_seated()[0])} "
              f"clear={bool(scene._clear[0])} expose={bool(scene._expose[0])} "
              f"lift={bool(scene._lift[0])} near={bool(scene._near[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 480, tail: int = 60) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(tail)  # extra hands-off settle

    def relocate_stone(name: str, slot_xy, tag: str) -> None:
        """TRANSPORT the named stone to a free-space hover above its drop slot in
        the east half (zero velocity), then hands-off fall + settle. Up to 3
        attempts (a bounce is re-carried — the final pose is still contact-made)."""
        sx, sy = slot_xy
        for attempt in range(3):
            st = torch.zeros(n, 13, device=device)
            st[:, 0] = sx
            st[:, 1] = sy
            st[:, 2] = c.wall_h + c.rubble_size / 2 + 0.045
            st[:, 3] = 1.0
            st[:, 0:3] += scene.env_origins
            scene.rubble[name].write_root_state_to_sim(st, all_ids)
            settle()
            loc, vel = scene._rubble_tensors()
            idx = scene.RUBBLE_NAMES.index(name)
            in_pit = bool(scene.rubble_in_pit()[0, idx])
            d = float((loc[0, idx, :2] - scene.token_spawn[0]).norm())
            if in_pit and d > c.clear_r and float(vel[0, idx]) < c.settle_speed:
                return
            print(f"[solve] {tag}: stone not settled at its slot after attempt "
                  f"{attempt + 1} (in_pit={in_pit} d={d:.3f}); re-carrying", flush=True)
        report(f"{tag}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL ({tag} never settled in the east half)", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(240)  # the pile seats: cap onto the prize, ring onto the floor
    spawn = scene.token_spawn[0].tolist()
    print(f"[solve] layout readback (seed {args.seed}): token_spawn="
          f"({spawn[0]:+.3f},{spawn[1]:+.3f}) pile_slot={scene.pile_slot[0].tolist()}",
          flush=True)
    report("reset")
    tl = scene._token_local()[0]
    assert bool(scene.covered()[0]), "the prize must start BURIED (covered)"
    assert bool(scene.all_in_pit()[0]), "all stones must start inside the pit"
    assert abs(float(tl[2]) - c.token_size / 2) < 0.012, \
        f"prize must rest on the pit floor, z={float(tl[2]):.3f}"
    assert float(tl[0]) < c.pit_center[0], "prize must start in the west half"
    loc, _v = scene._rubble_tensors()
    assert torch.isfinite(loc).all(), "NaN/inf in stone states after settle"
    s0 = print_score("P0 reset+settle (prize buried under the pile, east half empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: clear the overburden, TOP-DOWN, within the pit --------------
    # Height order = accessibility order (what a gripper faces). Each stone goes to
    # its own empty drop slot in the east half — the containment invariant is kept
    # throughout, which is exactly what the latches require.
    loc, _v = scene._rubble_tensors()
    order = sorted(range(6), key=lambda i: -float(loc[0, i, 2]))
    print(f"[solve] clearing order (top-down): {[scene.RUBBLE_NAMES[i] for i in order]}",
          flush=True)
    for k, i in enumerate(order):
        relocate_stone(scene.RUBBLE_NAMES[i], c.clear_slots[k], f"stone-{k}")
        if k == 2:
            report("mid-clear")
            s_mid = print_score("P1a three stones relocated (containment kept)")
            assert s_mid >= s0 - 1e-6, "score decreased during clearing"
    settle(600, 90)
    report("cleared")
    assert not bool(scene.covered()[0]), "prize must be UNCOVERED after clearing"
    assert bool(scene.all_in_pit()[0]), "all stones must still be inside the pit"
    assert bool(scene._clear[0]) and bool(scene._expose[0]), \
        "clear+expose latches must have fired"
    s1 = print_score("P1 overburden cleared to the east half; prize exposed")
    assert s1 >= 0.34, f"P1 score {s1} (expect clear+expose=0.35)"

    # ---------------- phase 2: extract the prize and DROP it into the pocket ---------------
    seat_z = c.ped_h + c.token_size / 2
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = c.ped_pos[0]
        st[:, 1] = c.ped_pos[1]
        st[:, 2] = seat_z + 0.055  # hover ABOVE the open pocket; gravity does the seat
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.token.write_root_state_to_sim(st, all_ids)
        settle()
        if bool(scene.token_seated()[0]):
            break
        print(f"[solve] prize not seated after attempt {attempt + 1}; re-carrying",
              flush=True)
    report("enthroned")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the prize drop)", flush=True)
        os._exit(1)
    s2 = print_score("P2 prize dropped into the pedestal pocket: success")
    assert s2 >= s1 - 1e-6, "score decreased across the enthronement"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s (prize seated, stones contained)")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
