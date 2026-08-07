"""Teleport solution for BurnerSnuffScene (sim_gen task
`libero_kitchen_scene8_turn_off_the_stove_i13`) — the task's legitimacy
certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one body at a time): a single root-state write carries a
   body to a free-space HOVER with zero velocity — exactly the carry a gripper
   performs. Three writes total:
     (a) kettle straight UP off the post to an open-air hover (this is the lift a
         gripper performs; the `clear` latch reads the physical fact that the
         kettle bottom is far above the post top);
     (b) kettle to a hover 12 mm above the GREEN trivet (the side of which is
         randomized — the target xy is read back from the trivet body itself,
         i.e. the color-identity binding);
     (c) cup, reoriented MOUTH-DOWN (the in-hand 180-degree flip), to a hover
         with its rim 12 mm ABOVE the post top — the post is NOT yet inside the
         cup at the hover, and `capped()` is asserted False there.
2. SEATING (gravity + contact, hands-off): from each hover the body FALLS and
   lands on the real trivet / slides down over the real post until its rim rests
   on the real hob plate. Every rubric fact (kettle upright at trivet-top height,
   rim in the hob-level band, cup axis on the post axis) is produced by
   ballistics and contact, never written.
3. ORDER: kettle first, cup second — physically inherent: while the kettle
   stands on the post the cup cannot reach the rim band (98 mm body vs 94 mm
   interior; smoke check proves the naive cup-first drop fails).

If a settle leaves a stage geometrically off (a bounce drifted the cup off the
post axis), the body is picked up again (fresh transport to the same free-space
hover) and re-dropped — a retry, not a cheat: the final configuration is still
100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene8_turn_off_the_stove_i13.solve --headless [--seed N]
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

_qmul, _qz, _qx = scene_mod._qmul, scene_mod._qz, scene_mod._qx

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.burner_snuff")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        kp = scene.kettle.data.root_pos_w[0]
        up = scene._up_w(scene.cup)[0]
        vmax = max(float(scene.kettle.data.root_lin_vel_w.norm(dim=-1)[0]),
                   float(scene.cup.data.root_lin_vel_w.norm(dim=-1)[0]))
        print(f"[solve] {tag:12s} | kettle_z={float(kp[2]):.3f} "
              f"cup_upz={float(up[2]):+.2f} vmax={vmax:.3f} "
              f"on_burner={bool(scene.kettle_on_burner()[0])} "
              f"on_trivet={bool(scene.kettle_on_trivet()[0])} "
              f"capped={bool(scene.capped()[0])} "
              f"latches=(c={bool(scene._clear[0])},k={bool(scene._kettle[0])},"
              f"f={bool(scene._flip[0])},x={bool(scene._cap[0])}) "
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

    def settle(body, max_steps: int = 240) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i > 20 and float(body.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
                break
        step(60)  # extra hands-off settle

    def kettle_trivet_pose():
        """Hover 12 mm above the GREEN trivet — target xy read from the trivet
        body itself (the color-identity binding; its side is randomized)."""
        pos = scene.trivet.data.root_pos_w.clone()
        pos[:, 2] = pos[:, 2] + c.trivet_t / 2 + c.kettle_h / 2 + 0.012
        # handle pointed at world -x (open air over the deck front; nothing there
        # stands taller than the deck, and the handle rides high on the body)
        quat = _qz(torch.full((n,), math.pi, device=device))
        return pos, quat

    def cup_cap_pose():
        """MOUTH-DOWN hover centred on the burner axis, rim 12 mm ABOVE the post
        top: the post is not yet inside the cup — free space, capped() False."""
        pos = scene.burner.data.root_pos_w.clone()
        pos[:, 2] = scene._post_top_z() + 0.012 + c.cup_h / 2
        quat = _qmul(_qz(torch.zeros(n, device=device)),
                     _qx(torch.full((n,), math.pi, device=device)))
        return pos, quat

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats (kettle on the post, cup on the deck)
    bp = (scene.burner.data.root_pos_w - scene.env_origins)[0]
    tp = (scene.trivet.data.root_pos_w - scene.env_origins)[0]
    pp = (scene.plate.data.root_pos_w - scene.env_origins)[0]
    up_ = (scene.cup.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"burner=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) "
          f"side={int(scene.side[0])} "
          f"trivet=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) "
          f"plate=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"kettle_yaw={yaw_of(scene.kettle.data.root_quat_w[0]):+.1f}deg "
          f"cup=({float(up_[0]):+.3f},{float(up_[1]):+.3f}) "
          f"cup_yaw={yaw_of(scene.cup.data.root_quat_w[0]):+.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.kettle.data.root_pos_w).all() \
        and torch.isfinite(scene.cup.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.kettle_on_burner()[0]), "kettle must start standing on the post"
    assert float(scene._up_w(scene.cup)[0, 2]) > 0.9, "cup must start mouth-up"
    s0 = print_score("P0 reset+settle (kettle on the lit burner, cup mouth-up)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: lift the kettle clear of the burner -------------------------
    # Straight up to open air: bottom ~80 mm above the post top (the gripper lift).
    kpos = scene.kettle.data.root_pos_w.clone()
    kpos[:, 2] = scene._post_top_z() + 0.080 + c.kettle_h / 2
    write_pose(scene.kettle, kpos, scene.kettle.data.root_quat_w.clone())
    step(2)  # refresh + latch (falls < 1 mm in 2 substeps)
    report("lift")
    assert bool(scene._clear[0]), "clear latch must fire at the lift hover"
    assert not bool(scene.kettle_on_burner()[0]), "kettle still reads on the burner"
    s1 = print_score("P1 kettle lifted clear of the burner post")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect clear=0.15)"

    # ---------------- phase 2: kettle onto the GREEN trivet --------------------------------
    for attempt in range(3):
        pos, quat = kettle_trivet_pose()
        write_pose(scene.kettle, pos, quat)
        settle(scene.kettle)
        if bool(scene.kettle_on_trivet()[0]):
            break
        print(f"[solve] kettle-on-trivet not met after attempt {attempt + 1}; "
              f"re-dropping", flush=True)
        report("kettle-retry")
    else:
        report("kettle-FAIL")
        print("SIM_GEN_SOLVE: FAIL (kettle never seated on the trivet)", flush=True)
        os._exit(1)
    report("kettle")
    assert bool(scene._kettle[0]), "kettle latch did not set"
    assert not bool(scene.success()[0]), "cannot be success with the burner uncapped"
    s2 = print_score("P2 kettle at rest upright on the green trivet")
    assert s2 >= s1 - 1e-6 and s2 >= 0.39, f"P2 score {s2} (expect 0.40)"

    # ---------------- phase 3: flip the cup mouth-down (hover over the post) ---------------
    pos, quat = cup_cap_pose()
    write_pose(scene.cup, pos, quat)
    step(2)  # refresh + flip latch (falls < 1 mm in 2 substeps)
    report("flip")
    assert bool(scene._flip[0]), "flip latch must fire at the mouth-down hover"
    assert not bool(scene.capped()[0]), \
        "the hover itself must NOT satisfy capped() (rim far above the band)"
    s3 = print_score("P3 cup flipped mouth-down, hovering above the post")
    assert s3 >= s2 - 1e-6 and s3 >= 0.54, f"P3 score {s3} (expect 0.55)"

    # ---------------- phase 4: gravity seats the cup over the post -------------------------
    # Hands-off from here: the cup falls, the post enters the interior, the rim
    # lands on the hob plate. Retries re-transport to the same free-space hover.
    settle(scene.cup)
    for attempt in range(3):
        if bool(scene.capped()[0]):
            break
        print(f"[solve] capped() not met after attempt {attempt + 1}; "
              f"re-dropping", flush=True)
        report("cap-retry")
        pos, quat = cup_cap_pose()
        write_pose(scene.cup, pos, quat)
        settle(scene.cup)
    else:
        if not bool(scene.capped()[0]):
            report("cap-FAIL")
            print("SIM_GEN_SOLVE: FAIL (cup never seated over the post)", flush=True)
            os._exit(1)
    report("capped")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the cap)", flush=True)
        os._exit(1)
    s4 = print_score("P4 cup seated: rim on the hob plate, post+flame enclosed")
    assert s4 >= s3 - 1e-6, "score decreased across the cap"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
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
