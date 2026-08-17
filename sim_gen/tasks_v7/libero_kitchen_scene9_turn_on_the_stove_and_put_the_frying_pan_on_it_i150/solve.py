"""Teleport solution for ShuntStoveScene (sim_gen task
`libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i150`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. SHUNT (applied force + contact, the "turn on"): a velocity-regulated
   horizontal force at the grate (clamped at 8 N — a fingertip push on its
   47 mm tab) drives it along the rail toward the well. Every rubric-relevant
   displacement is made by the CONTACT CASCADE: the grate's tab face collides
   with the cover's plate edge and shunts it ahead; the train stops when the
   COVER hits its joint-limit end stop, and by construction exactly there the
   grate is centred over the well (the self-aligning stop). The final grate
   position is produced by the hard stop, not by any positioning of ours: the
   push simply saturates against it. The grate is a prismatic slider that
   NEVER rotates, so the pod's wrench-frame quirk (wrenches rotated by the
   rotation-since-reset) is identically moot — rotation-since-reset stays I.
2. TRANSPORT (teleport, the pan carry): one root-state write carries the pan to
   a free-space HOVER 20 mm above the grate top, upright, handle south, zero
   velocity — exactly the carry a gripper performs. The hover satisfies
   NOTHING (pan_seated needs the bottom within 10 mm of the grate top;
   asserted False at the hover).
3. SEATING (gravity + contact, hands-off): the pan falls, lands between the
   grate's retainer bars and settles flat over the flames. Every rubric fact
   (pan bottom at grate-top height, centre inside the retainer window,
   upright) is produced by ballistics and contact, never written.

If a settle leaves the seat geometrically off (a bounce drifted the pan onto a
bar), the pan is picked up again (fresh transport to the same free-space hover)
and re-dropped — a retry, not a cheat: the final configuration is still 100 %
contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i150.solve --headless [--seed N]
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
    env = ENVS.get("simgen.shunt_stove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

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

    def x_c() -> float:
        return float(scene._rel_x(scene.cover)[0])

    def x_g() -> float:
        return float(scene._rel_x(scene.grate)[0])

    def report(tag: str) -> None:
        vmax = max(float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]),
                   float(scene.grate.data.root_lin_vel_w.norm(dim=-1)[0]),
                   float(scene.cover.data.root_lin_vel_w.norm(dim=-1)[0]))
        print(f"[solve] {tag:12s} | cover_x={x_c() * 1000:+.1f}mm "
              f"grate_x={x_g() * 1000:+.1f}mm "
              f"pan_z={float(scene.pan.data.root_pos_w[0, 2]):.3f} vmax={vmax:.3f} "
              f"clear={bool(scene.cover_clear()[0])} "
              f"at_well={bool(scene.grate_at_well()[0])} "
              f"seated={bool(scene.pan_seated()[0])} "
              f"latches=(e={bool(scene._exposed[0])},g={bool(scene._grate_l[0])},"
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

    def clear_forces() -> None:
        scene.grate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids, is_global=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # everything seats (sliders hang on their joints, pan on the apron)
    pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cover_x={x_c() * 1000:+.1f}mm grate_x={x_g() * 1000:+.1f}mm "
          f"side={int(scene.side[0])} "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"pan_yaw={yaw_of(scene.pan.data.root_quat_w[0]):+.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.pan.data.root_pos_w).all() \
        and torch.isfinite(scene.grate.data.root_pos_w).all(), "NaN/inf after settle"
    assert abs(x_c()) <= c.cover_jitter + 0.005, "cover must start sealing the well"
    assert abs(x_g()) >= c.bay_x - c.bay_jitter - 0.005, "grate must start in a bay"
    assert float(scene._up_w(scene.pan)[0, 2]) > 0.95, "pan must start upright"
    s0 = print_score("P0 reset+settle (well sealed, grate bayed, pan on the apron)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: SHUNT — push the grate; the stop aligns it ------------------
    # Velocity-regulated force at the grate (the fingertip push on its tab).
    # World-frame is exact here: the prismatic grate never rotates, so the
    # pod's rotation-since-reset wrench quirk is identically the identity.
    side = 1.0 if x_g() > 0 else -1.0
    v_des = -side * 0.15                      # 0.15 m/s toward the well
    gain, f_max = 20.0, 8.0                   # K*dt/m = 20*0.5/120/0.5 ~ 0.17 (stable)
    stall = 0
    pushed = False
    for i in range(900):
        v = float(scene.grate.data.root_lin_vel_w[0, 0])
        f = max(-f_max, min(f_max, c.grate_mass * gain * (v_des - v)))
        fw = torch.zeros(n, 1, 3, device=device)
        fw[:, 0, 0] = f
        scene.grate.set_external_force_and_torque(fw, zero_wrench,
                                                  env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i % 90 == 0:
            print(f"[solve] shunt i={i:3d} grate_x={x_g() * 1000:+.1f}mm "
                  f"cover_x={x_c() * 1000:+.1f}mm v={v:+.3f} f={f:+.2f}N", flush=True)
        if abs(x_g()) <= 0.006:               # train saturated on the self-align stop
            pushed = True
            break
        stall = stall + 1 if abs(v) < 0.005 and i > 60 else 0
        if stall > 90 and abs(x_g()) <= 0.02:  # parked hard against the stop
            pushed = True
            break
    clear_forces()
    step(60)  # hands-off settle: the latches read the calm end state
    report("shunt")
    assert pushed, f"shunt never reached the stop (grate_x={x_g() * 1000:+.1f}mm)"
    assert abs(x_g()) <= c.grate_x_tol, \
        f"grate not centred by the stop: {x_g() * 1000:+.1f}mm"
    assert abs(x_c()) >= c.exposed_min, \
        f"cover not shunted clear: {x_c() * 1000:+.1f}mm"
    assert bool(scene._exposed[0]) and bool(scene._grate_l[0]), "shunt latches unset"
    assert not bool(scene.success()[0]), "no success before the pan is seated"
    s1 = print_score("P1 shunt complete: cover off the well, grate centred by the stop")
    assert s1 >= s0 - 1e-6 and s1 >= 0.39, f"P1 score {s1} (expect 0.40)"

    # ---------------- phase 2: pan onto the grate (transport hover + gravity seat) ---------
    def pan_hover_pose():
        """Hover 20 mm above the grate top, upright, handle SOUTH (clear of the
        tabs); target xy read back from the grate body itself. pan_seated() is
        False at the hover (z band is 10 mm)."""
        pos = scene.grate.data.root_pos_w.clone()
        pos[:, 2] = pos[:, 2] + c.plate_t / 2 + c.pan_bottom_dz + 0.020
        quat = _qz(torch.full((n,), -math.pi / 2, device=device))
        return pos, quat

    pos, quat = pan_hover_pose()
    write_pose(scene.pan, pos, quat)
    env.step(no_action)  # refresh (falls < 1 mm in one substep)
    assert not bool(scene.pan_seated()[0]), \
        "the hover itself must NOT satisfy pan_seated()"
    for attempt in range(3):
        for i in range(240):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(60)  # extra hands-off settle
        if bool(scene.pan_seated()[0]):
            break
        print(f"[solve] pan not seated after attempt {attempt + 1}; re-dropping",
              flush=True)
        report("pan-retry")
        pos, quat = pan_hover_pose()
        write_pose(scene.pan, pos, quat)
    else:
        if not bool(scene.pan_seated()[0]):
            report("pan-FAIL")
            print("SIM_GEN_SOLVE: FAIL (pan never seated on the grate)", flush=True)
            os._exit(1)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the seat)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pan seated on the grate over the flames")
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
    main()
