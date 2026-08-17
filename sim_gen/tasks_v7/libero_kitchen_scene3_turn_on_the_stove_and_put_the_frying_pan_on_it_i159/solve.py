"""Teleport solution for PiezoStoveScene (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i159`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. IGNITION (gravity + contact, the "turn on"): one root-state write carries
   the pan to a free-space pose HANDLE-DOWN above the igniter's guard shaft
   (handle axis on the shaft axis, tip 15 mm above the striker cap, dish well
   above the mouth) — exactly the carry a gripper performs before dipping the
   handle. Then HANDS OFF: the pan falls, the handle tip lands on the striker
   cap, and the PAN'S OWN WEIGHT drives the spring-loaded striker down its
   prismatic travel past the click depth. The ignition latch is set by the
   scene's substep press-streak counter reading REAL striker displacement made
   by contact — nothing is written to the striker, ever. The pan is then
   carried away (transport) and the return spring pops the striker back up
   (asserted: depth ~0 again, lit stays latched).
2. TRANSPORT (teleport, the pan carry): one root-state write carries the pan
   to a free-space HOVER 20 mm above the cook plate, upright, handle south,
   zero velocity. The hover satisfies NOTHING (pan_on_burner needs the bottom
   within 10 mm of the plate top; asserted False at the hover).
3. SEATING (gravity + contact, hands-off): the pan falls onto the plate and
   settles flat. Every rubric fact (bottom at plate-top height, centre inside
   the radial window, upright) is produced by ballistics and contact.

If the drop-in fails to click (a bounce walked the handle off the cap) or a
seat lands off-window, the pan is picked up again (fresh transport to the same
free-space pose) and re-dropped — a retry, not a cheat: the final
configuration is still 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i159.solve --headless [--seed N]
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
import traceback

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

_qz, _qy = scene_mod._qz, scene_mod._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.piezo_stove")().build(num_envs=args.num_envs, device=device)
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

    def depth() -> float:
        return float(scene.striker_depth()[0])

    def report(tag: str) -> None:
        pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | depth={depth() * 1000:+.1f}mm "
              f"streak={int(scene._streak[0])} "
              f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):.3f}) "
              f"v_pan={float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
              f"on_burner={bool(scene.pan_on_burner()[0])} "
              f"latches=(lift={bool(scene._lift[0])},lit={bool(scene._lit[0])},"
              f"seat={bool(scene._seat[0])}) "
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

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # everything seats (striker hangs on its spring, pan/kettle on slots)
    pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    kp = (scene.kettle.data.root_pos_w - scene.env_origins)[0]
    bp = (scene.burner.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"burner=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"pan_yaw={yaw_of(scene.pan.data.root_quat_w[0]):+.1f}deg "
          f"kettle=({float(kp[0]):+.3f},{float(kp[1]):+.3f}) "
          f"pan_east={bool(scene.pan_east[0])} depth={depth() * 1000:+.2f}mm",
          flush=True)
    report("reset")
    assert torch.isfinite(scene.pan.data.root_pos_w).all() \
        and torch.isfinite(scene.striker.data.root_pos_w).all(), "NaN/inf after settle"
    # masses readback (custom spawners: MassAPI must have won over density)
    m_pan = float(scene.pan.root_physx_view.get_masses().sum())
    m_str = float(scene.striker.root_physx_view.get_masses().sum())
    assert abs(m_pan - c.pan_mass) < 0.02, f"pan mass readback {m_pan}"
    assert abs(m_str - c.striker_mass) < 0.005, f"striker mass readback {m_str}"
    assert depth() < 0.002, f"striker must rest at the TOP of its travel ({depth()})"
    assert not bool(scene._lit[0]), "burner must start OFF"
    assert float(scene._up_w(scene.pan)[0, 2]) > 0.95, "pan must start upright"
    assert abs(abs(float(pp[0])) - c.slot_x) < c.slot_x_jitter + 0.02, \
        "pan must start on a spawn slot"
    s0 = print_score("P0 reset+settle (burner off, striker up, pan on its slot)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: IGNITION — dip the handle, weight clicks the striker --------
    tower = scene.tower.data.root_pos_w[0]  # world (env origin included)

    def press_pose():
        """Handle-down over the shaft: qy(+90) sends local +x (the handle) to
        world -z; the handle axis is offset +handle_z in world x after the
        rotation, so the origin compensates. Tip lands 15 mm above the cap."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = tower[0] - c.handle_z
        pos[:, 1] = tower[1]
        cap_top = scene.striker.data.root_pos_w[0, 2] + c.cap_t / 2
        pos[:, 2] = cap_top + 0.015 + c.handle_tip_reach
        return pos, _qy(torch.full((n,), math.pi / 2, device=device))

    lit = False
    for attempt in range(3):
        pos, quat = press_pose()
        write_pose(scene.pan, pos, quat)
        env.step(no_action)
        assert depth() < c.press_depth, "the carry pose itself must not press"
        max_depth = 0.0
        for i in range(360):
            env.step(no_action)
            max_depth = max(max_depth, depth())
            if bool(scene._lit[0]):
                lit = True
                break
        print(f"[solve] press attempt {attempt + 1}: lit={lit} "
              f"max_depth={max_depth * 1000:.1f}mm", flush=True)
        if lit:
            break
        report("press-retry")
    if not lit:
        report("press-FAIL")
        print("SIM_GEN_SOLVE: FAIL (drop-in never clicked the striker)", flush=True)
        os._exit(1)
    # carry the pan away; the spring must return the striker on its own
    lift = torch.zeros(n, 3, device=device)
    lift[:, 0], lift[:, 1] = tower[0] - 0.20, tower[1] - 0.25
    lift[:, 2] = scene.env_origins[0, 2] + c.deck_h + 0.25
    write_pose(scene.pan, lift, _qz(torch.full((n,), -math.pi / 2, device=device)))
    step(60)
    report("ignited")
    assert bool(scene._lit[0]), "lit latch must survive the release (it is latched)"
    assert depth() < 0.002, f"spring must return the striker ({depth() * 1000:.1f}mm)"
    assert not bool(scene.success()[0]), "no success before the pan is on the plate"
    s1 = print_score("P1 igniter clicked by the pan's weight; striker sprung back")
    assert s1 >= s0 - 1e-6 and s1 >= 0.39, f"P1 score {s1} (expect 0.40 = lift+lit)"

    # ---------------- phase 2: pan onto the plate (transport hover + gravity seat) ---------
    def pan_hover_pose():
        """Hover 20 mm above the plate top, upright, handle SOUTH; target xy
        read back from the burner body itself. pan_on_burner() is False at the
        hover (z band is 10 mm)."""
        pos = scene.burner.data.root_pos_w.clone()
        pos[:, 2] = pos[:, 2] + c.plinth_h + c.plate_h + c.pan_bottom_dz + 0.020
        quat = _qz(torch.full((n,), -math.pi / 2, device=device))
        return pos, quat

    pos, quat = pan_hover_pose()
    write_pose(scene.pan, pos, quat)
    env.step(no_action)  # refresh (falls < 1 mm in one substep)
    assert not bool(scene.pan_on_burner()[0]), \
        "the hover itself must NOT satisfy pan_on_burner()"
    for attempt in range(3):
        for i in range(240):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(60)  # extra hands-off settle
        if bool(scene.pan_on_burner()[0]):
            break
        print(f"[solve] pan not seated after attempt {attempt + 1}; re-dropping",
              flush=True)
        report("pan-retry")
        pos, quat = pan_hover_pose()
        write_pose(scene.pan, pos, quat)
    else:
        if not bool(scene.pan_on_burner()[0]):
            report("pan-FAIL")
            print("SIM_GEN_SOLVE: FAIL (pan never seated on the plate)", flush=True)
            os._exit(1)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the seat)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pan seated on the lit burner")
    assert s2 >= s1 - 1e-6, "score decreased across the seat"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
