"""Teleport solution for WeighServeScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i189) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): each move is ONE root-state write that carries a single bowl
   across free space — from its row slot to ~20 mm ABOVE a balance pan, or (after the
   measurement) from the pan to ~40 mm above the plate. The write satisfies no rubric
   clause by itself: at the drop point the bowl is airborne.
2. WEIGHING (gravity + contact): the two candidate bowls FALL into the rimmed pans and
   the beam — a dynamic body on a revolute pivot with its CoM below the pivot — tilts
   or levels purely under gravity and contact. The scene latches the comparison only
   when exactly one bowl rests in each pan, the third is clear, everything is settled
   and the verdict is decisive. Nothing about the beam is ever written after reset.
3. INFERENCE (readback of a physical observable): the solution reads the BEAM TILT and
   the pan occupancy — never the hidden mass assignment — and branches: tilted ->
   the bowl in the LOWER pan is the ballasted one; level -> the left-out bowl is.
   (An oracle cross-check against the true masses is asserted AFTER the decision, as
   certificate evidence that the physical inference is sound — the decision itself is
   made from the tilt readback alone.)
4. SERVE (gravity + contact): the inferred bowl is teleported 40 mm above the plate and
   DROPPED; the resting pose success() judges is produced by the landing dynamics.
   The spoil monitors stay clear because the serve happens only after the latched
   weighing and only with the inferred (ballasted) bowl.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage credit is
latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weigh_serve")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        in_p, in_n, _ob = scene.pan_occupancy()
        t = math.degrees(float(scene.tilt()[0]))
        print(f"[solve] {tag:18s} | tilt={t:+6.2f}deg in_p={in_p[0].tolist()} "
              f"in_n={in_n[0].tolist()} weighed={bool(scene._s_weigh[0])} "
              f"spoiled={bool(scene._spoiled[0])} served={bool(scene.served()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def debug_pose(tag: str) -> None:
        o = scene.env_origins[0]
        bpd = scene.beam.data.root_pos_w[0] - o
        bqd = scene.beam.data.root_quat_w[0]
        _ip, _in_, ob = scene.pan_occupancy()
        print(f"[dbg] {tag}: beam_pos=({float(bpd[0]):+.4f},{float(bpd[1]):+.4f},"
              f"{float(bpd[2]):+.4f}) beam_quat=({float(bqd[0]):+.4f},{float(bqd[1]):+.4f},"
              f"{float(bqd[2]):+.4f},{float(bqd[3]):+.4f}) on_beam={ob[0].tolist()}",
              flush=True)
        for i in range(3):
            p = scene.bowls[i].data.root_pos_w[0] - o
            v = float(scene.bowls[i].data.root_lin_vel_w[0].norm())
            print(f"[dbg]   bowl{i}: pos=({float(p[0]):+.4f},{float(p[1]):+.4f},"
                  f"{float(p[2]):+.4f}) |v|={v:.4f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def bowl_write(i: int, pos_w: torch.Tensor) -> None:
        """Teleport bowl i (upright, zero velocity) to a world point — transport only."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = 1.0
        scene.bowls[i].write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # bowls settle at their slots
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    slot_of = scene.slot_of[0].tolist()
    side = int(scene.plate_side[0])
    pp = scene.plate.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[solve] layout readback (seed {args.seed}): masses={masses} "
          f"slot_of_body={slot_of} plate=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"side={side:+d}", flush=True)
    assert abs(max(masses) - c.mass_heavy) < 0.02, "mass authoring readback failed"
    report("reset")
    debug_pose("after settle")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene._spoiled[0]), "fresh reset must not spoil"
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: load the two candidate bowls onto the pans ------------------
    # Choose candidates BY POSITION (the bowls in row slots 0 and 1 — a rule any solver
    # could follow), never by body identity. Slot of each body comes from position
    # readback, exactly what a camera would provide.
    bows = torch.stack([b.data.root_pos_w[0] for b in scene.bowls], dim=0) \
        - scene.env_origins[0]
    ys = torch.tensor(c.slot_ys, device=device)
    slot_by_pos = [int((ys - bows[i, 1]).abs().argmin()) for i in range(3)]
    print(f"[solve] slot-by-position readback: {slot_by_pos}", flush=True)
    cand_n = slot_by_pos.index(0)   # slot 0 bowl -> the -y pan
    cand_p = slot_by_pos.index(1)   # slot 1 bowl -> the +y pan
    left_out = slot_by_pos.index(2)

    def pan_target(sgn: float) -> torch.Tensor:
        """A point just above the (possibly tilted) pan's rim, from the LIVE beam pose."""
        bp = scene.beam.data.root_pos_w
        bq = scene.beam.data.root_quat_w
        off = torch.tensor([[0.0, sgn * c.pan_dy, c.pan_z + c.rim_h + 0.006]],
                           device=device).expand(n, 3)
        return bp + quat_apply(bq, off)

    # SEQUENTIAL loading: drop one bowl, let the beam finish its (damped) swing, then
    # drop the other into the now-tilted pan. Simultaneous drops make the beam whip and
    # catapult the light bowl out.
    bowl_write(cand_n, pan_target(-1.0))
    step(300)  # ~2.5 s: land + creep to the stop + settle
    report("first bowl in")
    in_p, in_n, _ob = scene.pan_occupancy()
    assert bool(in_n[0, cand_n]), "first bowl missed the -y pan"
    bowl_write(cand_p, pan_target(+1.0))
    for _ in range(900):
        env.step(no_action)
        if bool(scene._s_weigh[0]):
            break
    step(30)  # extra settle, hands-off
    report("weighing")
    debug_pose("after weighing wait")
    in_p, in_n, _ob = scene.pan_occupancy()
    assert bool(in_p[0, cand_p]), "second bowl missed the +y pan"
    assert bool(scene._s_weigh[0]), "the comparison weighing never latched"
    assert not bool(scene._spoiled[0]), "weighing must not spoil"
    s1 = print_score("P1 comparison weighing latched")
    assert s1 >= 0.34, f"expected >= 0.35 after weighing, got {s1}"
    assert s1 >= s0 - 1e-6, "score decreased"

    # ---------------- phase 2: infer the ballasted bowl from the TILT readback -------------
    t = float(scene.tilt()[0])
    in_p, in_n, _ob = scene.pan_occupancy()
    if t <= -math.radians(c.tilt_min_deg):
        heavy = int(in_p[0].float().argmax())      # +y pan is DOWN
        verdict = f"tilt {math.degrees(t):+.1f}deg -> +y pan down"
    elif t >= math.radians(c.tilt_min_deg):
        heavy = int(in_n[0].float().argmax())      # -y pan is DOWN
        verdict = f"tilt {math.degrees(t):+.1f}deg -> -y pan down"
    else:
        heavy = left_out                            # level -> the left-out bowl
        verdict = f"tilt {math.degrees(t):+.1f}deg -> LEVEL, left-out bowl"
    print(f"[solve] verdict: {verdict} -> ballasted bowl = body {heavy} "
          f"(slot {slot_by_pos[heavy]})", flush=True)
    # certificate cross-check (AFTER the decision): the physical inference is sound
    assert abs(masses[heavy] - c.mass_heavy) < 0.02, \
        f"inference from tilt named body {heavy} but masses are {masses}"

    # ---------------- phase 3: serve the inferred bowl on the plate ------------------------
    pp_w = scene.plate.data.root_pos_w
    tgt = pp_w.clone()
    tgt[:, 2] += c.plate_h / 2 + 0.040
    bowl_write(heavy, tgt)
    for _ in range(600):
        env.step(no_action)
        if bool(scene.served()[0]):
            break
    step(30)  # extra settle, hands-off
    report("served")
    assert not bool(scene._spoiled[0]), "legal serve must not spoil"
    s2 = print_score("P3 ballasted bowl served")
    assert s2 >= s1 - 1e-6, "score decreased"

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after serve)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_end = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_end >= s2 - 1e-6
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
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc!r})", flush=True)
        os._exit(1)
