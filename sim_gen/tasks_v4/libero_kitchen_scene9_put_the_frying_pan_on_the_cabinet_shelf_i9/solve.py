"""Teleport solution for PanCubbyScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i9`) — the task's
legitimacy certificate.

The task's two load-bearing interactions and how each is executed:

1. EXTRACTION (contact dynamics, no teleport): the pan starts INSIDE the low cubby, so
   there is no free space to transport it through — teleporting it out would bypass the
   task's core constraint. Instead a horizontal external force (world frame, velocity-
   regulated bang-bang, applied at the CoM) drags the pan across the cubby floor and out
   through the open mouth; friction, the roof and the side walls act on it the whole way.
   The force is cut only after the pan centre is fully clear of the cubby footprint and
   the pan coasts to rest on real friction.
2. TRANSPORT (teleport, the only pose write): one root-state write carries the now-FREE
   pan across open ground from where it coasted to a release pose 28 mm ABOVE the burner
   top — never seated, and deliberately OUTSIDE the 20 mm on-burner z-tolerance, so the
   freshly-teleported state does not satisfy success(). This bypasses no interaction:
   both endpoints are in the open.
3. SET-DOWN (contact dynamics): gravity drops the pan the last 28 mm onto the burner
   pedestal; it impacts, beds down and settles under physics. success() first turns True
   only here, judged on the settled pose (on-burner height, centring, uprightness,
   stillness).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i9.solve --headless [--seed N]
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
    env = ENVS.get("simgen.pan_cubby_retrieval")().build(num_envs=args.num_envs, device=device)
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

    def cubby_pose() -> tuple[torch.Tensor, float]:
        cp = (scene.cubby.data.root_pos_w - scene.env_origins)[0]
        q = scene.cubby.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return cp, yaw

    def report(tag: str) -> None:
        loc = scene._pan_local()[0]
        print(f"[solve] {tag:12s} | pan_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) pe_latch={float(scene._pe_max[0]):.3f} "
              f"ext={bool(scene._ext[0])} pp_latch={float(scene._pp_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.pan.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    cp, cyaw = cubby_pose()
    pan0 = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    bur0 = (scene.burner.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): cubby=({float(cp[0]):+.3f},"
          f"{float(cp[1]):+.3f}) yaw={math.degrees(cyaw):+.1f}deg "
          f"pan_spawn=({float(pan0[0]):+.3f},{float(pan0[1]):+.3f}) "
          f"burner=({float(bur0[0]):+.3f},{float(bur0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: EXTRACTION through contact dynamics -------------------------
    # World-frame horizontal force along the cubby's +x (out the mouth), applied at the
    # pan CoM: velocity-regulated bang-bang (push while axial speed < v_des, escalate on
    # stall). Friction below, the 29 mm-headroom roof above and the side walls act on the
    # pan the whole way. A small lateral correction keeps it centred while still inside.
    # The force is CUT only once the pan centre has fully cleared the cubby footprint
    # (loc_x > x_exit), then the pan coasts out on real friction. No pose write touches
    # the pan anywhere in this phase.
    cy, sy = math.cos(cyaw), math.sin(cyaw)
    push_dir = torch.tensor([cy, sy, 0.0], device=device)   # cubby local +x, in world
    lat_dir = torch.tensor([-sy, cy, 0.0], device=device)   # cubby local +y, in world
    f_push, v_des = 5.0, 0.10
    best_x, last_bump = -1.0, 0
    pushed_steps = 0
    for i in range(2400):
        loc = scene._pan_local()[0]
        x_l, y_l = float(loc[0]), float(loc[1])
        if x_l > c.x_exit + 0.01:  # centre fully clear of the footprint, with margin
            break
        v_axis = float((scene.pan.data.root_lin_vel_w[0] * push_dir).sum())
        f_axis = f_push if v_axis < v_des else 0.0
        f_lat = max(-1.2, min(1.2, -30.0 * y_l)) if x_l < c.cubby_d / 2 else 0.0
        f_w = (push_dir * f_axis + lat_dir * f_lat).view(1, 1, 3).expand(n, 1, 3)
        scene.pan.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        pushed_steps += 1
        if x_l > best_x + 0.005:
            best_x, last_bump = x_l, i
        elif i - last_bump > 240:  # stalled: push harder (friction was underestimated)
            f_push = min(f_push + 2.0, 12.0)
            last_bump = i
            print(f"[solve] drag stalled at x_local={x_l:+.3f}, raising force to "
                  f"{f_push:.1f} N", flush=True)
    clear_force()
    step(30)  # coast out and settle on real friction
    report("extracted")
    print(f"[solve] extraction complete after {pushed_steps} steps, force {f_push:.1f} N",
          flush=True)
    assert bool(scene._ext[0]), "pan did not clear the cubby footprint under the drag force"
    s1 = print_score("P1 contact-dynamics extraction")
    assert s1 >= s0 - 1e-6, "score decreased across extraction"

    # ---------------- phase 2: TRANSPORT (teleport across free space only) -----------------
    # The pan is now free, at rest in the open. One pose write carries it over open ground
    # to a release pose with its base 28 mm ABOVE the burner top — NOT seated, and outside
    # the 20 mm on-burner z-tolerance, so this state does not satisfy success(); the
    # actual set-down is left to gravity and contact in phase 3. Orientation is kept
    # exactly as the pan coasted to rest (transport does not reorient it); velocities are
    # zeroed.
    bur = scene.burner.data.root_pos_w[0]
    burner_top = float(bur[2]) + c.burner_h / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = bur[0]  # already world coords (env origin included)
    st[:, 1] = bur[1]
    st[:, 2] = burner_top + 0.028 + c.base_t / 2
    st[:, 3:7] = scene.pan.data.root_quat_w[0]
    scene.pan.write_root_state_to_sim(st, all_ids)
    report("transported")
    s2 = print_score("P2 transport to release pose above the burner")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: SET-DOWN through gravity + contact, settle, judge -----------
    step(150)  # 1.25 s: drop 15 mm, impact, bed down, settle
    report("set-down")
    s3 = print_score("P3 gravity set-down + settle")
    assert s3 >= s2 - 1e-6, "score decreased across set-down"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after set-down)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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
