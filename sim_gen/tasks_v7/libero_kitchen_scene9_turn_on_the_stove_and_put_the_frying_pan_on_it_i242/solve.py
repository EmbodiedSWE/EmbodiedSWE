"""Teleport solution for BallastStoveScene (sim_gen task
`libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i242`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. BALLAST (gravity + contact, the "turn on"): one root-state write carries the
   IRON INGOT to a free-space HOVER above the ballast tray — inside the wall
   funnel but ABOVE the judged z band (asserted: nothing is satisfied at the
   hover). The ingot then FALLS into the tray, and its weight — pure gravity
   acting through tray-floor contact — out-torques the beam's bias and rotates
   the rocker from its +16 deg trivet-down stop to the LEVEL stop. Every rubric
   fact (ingot resting in the tray band, beam at level) is produced by physics;
   no force, torque, or joint command is ever applied and the beam is never
   written after reset.
2. TRANSPORT (teleport, the pan carry): one root-state write carries the pan to
   a HOVER 20 mm above the now-level trivet, upright, handle south, zero
   velocity — exactly the carry a gripper performs. The hover satisfies
   NOTHING (pan_on_trivet needs the base within 10 mm of its seat; asserted
   False at the hover).
3. SEATING (gravity + contact, hands-off): the pan falls, lands between the
   trivet's rim-fence lips and settles flat. If a bounce leaves it perched off
   the seat, it is picked up again (fresh transport to the same free-space
   hover) and re-dropped — a retry, not a cheat: the final configuration is
   still 100 % contact-made.

The success it certifies is LIVE (a deadman): beam level AND ingot in the tray
AND pan seated, all settled — held throughout a >= 3 s hands-off persistence
window, during which gravity alone keeps the ballast working.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's partial credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if
success() held at every persistence checkpoint.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i242.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qz = scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_stove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def yaw_of(q) -> float:
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def pitch() -> float:
        return float(scene.beam_pitch_deg()[0])

    def report(tag: str) -> None:
        vmax = max(float(scene.ingot.data.root_lin_vel_w.norm(dim=-1)[0]),
                   float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]))
        wb = float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | pitch={pitch():+.2f}deg "
              f"vmax={vmax:.3f} w_beam={wb:.3f} "
              f"level={bool(scene.beam_level()[0])} "
              f"tray={bool(scene.ingot_in_tray()[0])} "
              f"pan={bool(scene.pan_on_trivet()[0])} "
              f"latches=(t={bool(scene._tray_l[0])},l={bool(scene._level_l[0])},"
              f"p={bool(scene._pan_l[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def beam_local_to_world(local_xyz) -> torch.Tensor:
        """A beam-frame point in world coords, using the LIVE beam pose."""
        from isaaclab.utils.math import quat_apply

        loc = torch.tensor(local_xyz, device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # beam sinks onto its tilt stop; blocks and pan seat on the deck
    ip = (scene.ingot.data.root_pos_w - scene.env_origins)[0]
    bp = (scene.block.data.root_pos_w - scene.env_origins)[0]
    pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): pitch={pitch():+.2f}deg "
          f"swap={int(scene.swap[0])} "
          f"ingot=({float(ip[0]):+.3f},{float(ip[1]):+.3f}) "
          f"decoy=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"pan_yaw={yaw_of(scene.pan.data.root_quat_w[0]):+.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.beam.data.root_pos_w).all() \
        and torch.isfinite(scene.ingot.data.root_pos_w).all() \
        and torch.isfinite(scene.pan.data.root_pos_w).all(), "NaN/inf after settle"
    # The pitch readback proves the joint sign convention: gravity bias holds
    # the beam pressed on its trivet-down stop at ~ +tilt_deg.
    assert c.tilt_deg - 1.2 <= pitch() <= c.tilt_deg + 1.2, \
        f"beam must rest on its tilt stop (~{c.tilt_deg} deg), got {pitch():+.2f}"
    assert abs(abs(float(ip[0])) - c.slot_x) <= c.slot_jitter + 0.01, \
        "ingot must start in an apron slot"
    assert abs(abs(float(bp[0])) - c.slot_x) <= c.slot_jitter + 0.01, \
        "decoy must start in an apron slot"
    assert float(ip[0]) * float(bp[0]) < 0, "ingot and decoy in OPPOSITE slots"
    from isaaclab.utils.math import quat_apply

    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    assert float(quat_apply(scene.pan.data.root_quat_w, ez)[0, 2]) > 0.95, \
        "pan must start upright on the apron"
    s0 = print_score("P0 reset+settle (beam on tilt stop, gas OFF, items on apron)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: BALLAST — drop the ingot; gravity levels the beam -----------
    # Transport hover: beam-frame point over the tray, INSIDE the wall funnel
    # but ABOVE the judged z band (p.z - plate_top = 0.055 > tray_z_hi), so the
    # write itself satisfies nothing; orientation matched to the tilted beam so
    # the drop lands flat on the tray floor.
    hover_local = (-c.arm, 0.0, c.plate_top + 0.055)
    pos = beam_local_to_world(hover_local)
    write_pose(scene.ingot, pos, scene.beam.data.root_quat_w.clone())
    env.step(no_action)  # refresh readbacks (falls < 1 mm in one substep)
    assert not bool(scene.ingot_in_tray()[0]), \
        "the ingot hover itself must NOT satisfy ingot_in_tray()"
    assert not bool(scene.beam_level()[0]), "beam still tilted at the hover"
    # Hands-off: the ingot falls into the tray; its weight rotates the rocker.
    for i in range(600):
        env.step(no_action)
        if i % 60 == 0:
            print(f"[solve] ballast i={i:3d} pitch={pitch():+.2f}deg "
                  f"w={float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0]):.3f}",
                  flush=True)
        if i > 60 and bool(scene.beam_level()[0]) and bool(scene.settled()[0]):
            break
    step(90)  # extra hands-off settle: latches read the calm end state
    report("ballast")
    assert bool(scene.beam_level()[0]), \
        f"ingot ballast must level the beam (pitch={pitch():+.2f}deg)"
    assert bool(scene.ingot_in_tray()[0]), "ingot must rest in the tray"
    assert bool(scene._tray_l[0]) and bool(scene._level_l[0]), "P1 latches unset"
    assert not bool(scene.success()[0]), "no success before the pan is seated"
    s1 = print_score("P1 ballast placed: beam level, gas ON")
    assert s1 >= s0 - 1e-6 and s1 >= 0.39, f"P1 score {s1} (expect 0.40)"

    # ---------------- phase 2: pan onto the trivet (transport hover + gravity seat) --------
    def pan_hover_pose():
        """Hover 20 mm above the pan's seat on the (now level) trivet, upright,
        handle SOUTH (over the low rim lip, clear by construction); target read
        back from the live beam pose. pan_on_trivet() is False at the hover
        (z band is 10 mm)."""
        pos = beam_local_to_world((c.arm, 0.0, c.pan_seat_z + 0.020))
        quat = _qz(torch.full((n,), -math.pi / 2, device=device))
        return pos, quat

    pos, quat = pan_hover_pose()
    write_pose(scene.pan, pos, quat)
    env.step(no_action)  # refresh (falls < 1 mm in one substep)
    assert not bool(scene.pan_on_trivet()[0]), \
        "the hover itself must NOT satisfy pan_on_trivet()"
    for attempt in range(3):
        for i in range(240):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(60)  # extra hands-off settle
        if bool(scene.pan_on_trivet()[0]):
            break
        print(f"[solve] pan not seated after attempt {attempt + 1}; re-dropping",
              flush=True)
        report("pan-retry")
        pos, quat = pan_hover_pose()
        write_pose(scene.pan, pos, quat)
    else:
        if not bool(scene.pan_on_trivet()[0]):
            report("pan-FAIL")
            print("SIM_GEN_SOLVE: FAIL (pan never seated on the trivet)", flush=True)
            os._exit(1)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the seat)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pan seated flat on the level, lit trivet")
    assert s2 >= s1 - 1e-6, "score decreased across the seat"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    # The deadman holds itself: gravity through the ballast keeps the beam on
    # its level stop with the pan riding the trivet the whole window.
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
