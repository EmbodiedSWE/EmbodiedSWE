"""Teleport solution for LipHangRackScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i90`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. TRANSPORT (teleport): one pose write stages the bottle UPRIGHT at the open FRONT
   END of the GREEN-tagged slot — rack-local x at the mouth (11 cm from the tower
   stop), laterally centred on the green slot, lip underside 3 mm ABOVE the rail-top
   plane, zero velocity. This is the pose the arm reaches by carrying the grasped
   bottle to rail height: nothing is threaded yet (depth progress is 0 there by
   construction) and success is impossibly far (depth 0.110 vs the 0.045 stop band).
2. CATCH (contact dynamics): the bottle is released; it drops the 3 mm, the 50 mm lip
   lands across the two rail tops of the 34 mm gap, and the bottle HANGS — the first
   and only support it ever gets from the rack is this physical lip-on-rail contact.
3. SLIDE TO THE STOP (contact dynamics — never teleported): a velocity-limited
   horizontal world force pushes the hanging bottle backward along the slot, lip
   riding on the rail tops, plus (a) a small lateral centering term and (b) the
   compensating torque that re-expresses the CoM force as a push at NECK height
   (r x F), i.e. the wrench emulation of the arm nudging the bottle just under the
   lip — so the bottle stays upright instead of pendulum-tilting. The force is CUT
   once the readback depth enters the stop band; the bottle coasts the last
   millimetres, stops against the tower if it gets there, and settles hands-off into
   the final hang. From the force cut to the verdict nothing touches the bottle.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

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
    env = ENVS.get("simgen.lip_hang_rack")().build(num_envs=args.num_envs, device=device)
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

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        loc = scene._bottle_loc()[0]
        d = float(scene._depth()[0])
        up = float(scene._bottle_up()[0])
        thr = bool(scene._threaded(c.mouth_lat, c.mouth_z, c.mouth_max_deg)[0])
        print(f"[solve] {tag:12s} | loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) depth={d:+.3f} up={up:+.3f} threaded={thr} "
              f"lift={bool(scene._lift_ever[0])} mouth={bool(scene._mouth_ever[0])} "
              f"depth_max={float(scene._depth_max[0]):.3f} "
              f"settled={bool(scene._settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    r = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rack.data.root_quat_w[0]
    r_yaw = 2.0 * math.atan2(float(rq[3]), float(rq[0]))
    b0 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    green_v = float(scene._green_v[0])
    # readback: the GREEN tag body really sits at the green slot (identify by color)
    gt_loc = scene._rack_local(scene.green_tag.data.root_pos_w)[0]
    rt_loc = scene._rack_local(scene.red_tag.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(r[0]):+.3f},{float(r[1]):+.3f}) yaw={math.degrees(r_yaw):+.1f}deg "
          f"green_side={'+' if green_v > 0 else '-'} green_tag_v={float(gt_loc[1]):+.3f} "
          f"red_tag_v={float(rt_loc[1]):+.3f} "
          f"bottle=({float(b0[0]):+.3f},{float(b0[1]):+.3f},{float(b0[2]):.3f})",
          flush=True)
    report("reset")
    assert abs(float(gt_loc[1]) - green_v) < 0.005, "green tag not at the green slot"
    assert abs(float(rt_loc[1]) + green_v) < 0.005, "red tag not at the other slot"
    assert abs(float(b0[2]) - c.stand_z) < 0.01, "bottle did not settle standing"
    assert not bool(scene.success()[0]), "success at reset?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT to the mouth + CATCH (drop onto the rails) ---------
    # One pose write: upright at the open front end of the GREEN slot, lip underside
    # 3 mm above the rail tops, zero velocity. Then hands-off: it drops and the lip
    # catches across the rails — the hang itself is pure contact.
    r_pos = scene.rack.data.root_pos_w
    r_quat = scene.rack.data.root_quat_w
    stage_loc = torch.zeros(n, 3, device=device)
    stage_loc[:, 0] = c.face_x + c.d_mouth
    stage_loc[:, 1] = scene._green_v
    stage_loc[:, 2] = c.hang_z0 + 0.003
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = r_pos + quat_apply(r_quat, stage_loc)
    st[:, 3] = 1.0  # identity: upright (the bottle is axisymmetric, yaw is free)
    scene.bottle.write_root_state_to_sim(st, all_ids)
    # settle into the hang at the mouth
    for _ in range(300):
        env.step(no_action)
        if bool(scene._settled()[0]):
            break
    report("caught")
    loc = scene._bottle_loc()[0]
    assert abs(float(loc[2]) - c.hang_z0) < 0.012, \
        f"bottle is not hanging at lip height (z_loc={float(loc[2]):.3f})"
    assert bool(scene._threaded(c.mouth_lat, c.mouth_z, c.mouth_max_deg)[0]), \
        "bottle at the mouth does not read as threaded"
    assert not bool(scene.success()[0]), "staged at the mouth must NOT be success"
    s1 = print_score("P1 transported to the mouth, lip caught on the rails")
    assert s1 >= s0 - 1e-6, "score decreased across transport"
    assert s1 <= 0.35, f"staging alone scored too much ({s1})"

    # ---------------- phase 2: SLIDE TO THE STOP (contact dynamics) -------------------------
    # Velocity-limited world force toward the tower (rack -x), lateral centering, and
    # the compensating torque of a push applied at NECK height (10 cm above the CoM)
    # so the hanging bottle does not pendulum-tilt under the CoM force.
    push_dir = quat_apply(r_quat, torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3))
    lat_dir = quat_apply(r_quat, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    arm_loc = torch.tensor([0.0, 0.0, 0.10], device=device).expand(n, 3)
    f_push, v_des = 1.2, 0.10
    best_d, last_bump = 10.0, 0
    reached = False
    for i in range(1500):
        d = float(scene._depth()[0])
        if d <= 0.040:
            reached = True
            break
        loc = scene._bottle_loc()
        vel = scene.bottle.data.root_lin_vel_w
        u_vel = float((vel[0] * push_dir[0]).sum())  # speed toward the tower
        v_vel = float((vel[0] * lat_dir[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.0
        lat_err = float(loc[0, 1]) - float(scene._green_v[0])
        f_lat = max(-0.4, min(0.4, -6.0 * lat_err - 1.5 * v_vel))
        f_w = (push_dir * f_axis + lat_dir * f_lat).view(n, 1, 3).contiguous()
        arm_w = quat_apply(scene.bottle.data.root_quat_w, arm_loc)
        tau_w = torch.cross(arm_w, f_w.view(n, 3), dim=-1).view(n, 1, 3).contiguous()
        scene.bottle.set_external_force_and_torque(f_w, tau_w, env_ids=all_ids,
                                                   is_global=True)
        env.step(no_action)
        if d < best_d - 0.003:
            best_d, last_bump = d, i
        elif i - last_bump > 240:  # stalled on the rails: push a little harder
            f_push = min(f_push + 0.5, 3.0)
            last_bump = i
            print(f"[solve] stall at depth={d:+.3f} -> f_push={f_push:.1f} N", flush=True)
    scene.bottle.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("slid")
    assert reached, "slide never carried the lip into the stop band"

    # hands-off: coast the last millimetres, stop at the tower, settle into the hang
    quiet = 0
    for _ in range(900):
        env.step(no_action)
        if bool(scene.success()[0]):
            quiet += 1
        else:
            quiet = 0
        if quiet >= 30:
            break
    report("settled")
    s2 = print_score("P2 slid to the tower stop, settled hanging")
    assert s2 >= s1 - 1e-6, "score decreased across the slide"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
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
