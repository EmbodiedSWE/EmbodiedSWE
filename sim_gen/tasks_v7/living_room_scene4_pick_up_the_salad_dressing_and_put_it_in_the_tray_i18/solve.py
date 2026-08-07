"""Teleport solution for WedgeHopperScene (sim_gen task
`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, wedge only): a single root-state write carries the blue
   wedge from its floor spawn to a FREE-SPACE hover 2 mm above the runway channel,
   thin end toward the hopper, aligned with the rig heading READ from the scene.
   The write puts the wedge in open air behind the hopper; it lifts nothing and
   satisfies no rubric clause by itself. The ball and the shim are NEVER teleported
   (the ball is sealed inside the hopper — that is the point of the task).
2. PUSH (applied CoM force, contact dynamics): an escalating horizontal force along
   the rig's +x axis — the fingertip/palm push a Franka would deliver against the
   wedge's back block — drives the wedge along the low-friction runway, between the
   guide rails, into the 10 mm slot under the hopper's raised back edge. The
   inclined plane converts the push into LIFT UNDER LOAD: the hopper pitches about
   its cradled pivot bar, through level, past the drain tilt. Every newton is a
   contact-consistent external force at the wedge CoM, cleared as soon as the
   target tilt is reached.
3. SELF-LOCK + DRAIN (gravity + contact, hands-off): with the force cleared, the
   wedge top's high friction (mu_avg 0.70 > tan 23.3 deg = 0.43) holds the tilt.
   The amber ball rolls down the hopper floor, crosses the MOUTH window (latching
   the pathway), flies over the cradle, sheds down the deflector and settles inside
   the walled basin — all produced by gravity and rolling contact, never written.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18.solve --headless [--seed N]
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

_qmul, _qy = scene_mod._qmul, scene_mod._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_hopper")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rig_xyz(body) -> tuple[float, float, float]:
        p = scene._rig_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def tilt_deg() -> float:
        return math.degrees(float(scene.drain_tilt()[0]))

    def report(tag: str) -> None:
        wx, wy, wz = rig_xyz(scene.wedge)
        bx, by, bz = rig_xyz(scene.ball)
        print(f"[solve] {tag:14s} | tilt={tilt_deg():+.2f}deg "
              f"wedge=({wx:+.3f},{wy:+.3f},{wz:+.3f}) "
              f"ball=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"in_hop={bool(scene.ball_in_hopper()[0])} "
              f"in_basin={bool(scene.ball_in_basin()[0])} "
              f"latch=[stg {int(scene._staged[0])} p {int(scene._lift_part[0])} "
              f"f {int(scene._lift_full[0])} m {int(scene._via_mouth[0])} "
              f"b {int(scene._ever_basin[0])}] "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def rig_pose(x: float, y: float, z: float) -> torch.Tensor:
        """(N,13) root state at rig-frame (x, y, z), orientation = rig heading
        (wedge +x toward the hopper), zero velocity."""
        from isaaclab.utils.math import quat_apply

        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x, y, z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        st[:, 3:7] = scene.rig.data.root_quat_w
        return st

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_axis() -> torch.Tensor:
        """(N,3) world direction of the rig's +x axis (runway -> hopper -> basin)."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.rig.data.root_quat_w, ex)

    def wedge_speed() -> float:
        return float(scene.wedge.data.root_lin_vel_w.norm(dim=-1)[0])

    def stage_wedge() -> None:
        """TRANSPORT the wedge to a free-space hover 2 mm above the runway channel,
        tip toward the hopper, ~50 mm short of the back edge; hands-off settle."""
        scene.wedge.write_root_state_to_sim(
            rig_pose(-0.31, 0.0, c.runway_top + 0.002), all_ids)
        step(90)

    def push_wedge_to_tilt(target_deg: float, f_max: float = 14.0) -> bool:
        """Slow, speed-capped CoM push along the rig +x axis, monitored EVERY step:
        the force is cleared whenever the wedge moves faster than 0.10 m/s (no
        momentum ram), escalated only on a genuine stall, and dropped the moment
        the drain tilt reaches target_deg."""
        newtons = 1.0
        last_x = rig_xyz(scene.wedge)[0]
        stall_ref, pushing = last_x, False
        for i in range(2400):  # 20 s budget
            t = tilt_deg()
            wx, wy, wz = rig_xyz(scene.wedge)
            if t >= target_deg:
                clear_force(scene.wedge)
                print(f"[solve] target tilt reached: {t:+.2f} deg "
                      f"(wedge_x={wx:+.3f}); force cleared", flush=True)
                return True
            if wz < c.runway_top - 0.02 or wz > c.runway_top + 0.06 \
                    or abs(wy) > 0.09 or wx > -0.13:
                clear_force(scene.wedge)
                print(f"[solve] wedge left the working zone "
                      f"({wx:+.3f},{wy:+.3f},{wz:+.3f}) tilt={t:+.2f}", flush=True)
                return False
            if wedge_speed() > 0.10:
                if pushing:
                    clear_force(scene.wedge)
                    pushing = False
            else:
                f = push_axis().reshape(n, 1, 3) * newtons
                scene.wedge.set_external_force_and_torque(
                    f, zero_wrench, env_ids=all_ids, is_global=True)
                pushing = True
            env.step(no_action)
            if i % 45 == 44:
                # escalate only on a genuine stall (no advance in 45 steps)
                if wx - stall_ref < 0.0015 and wedge_speed() < 0.02:
                    newtons = min(newtons * 1.4, f_max)
                stall_ref = wx
            if i % 90 == 0:
                print(f"[solve] push step {i:4d}: F={newtons:5.2f} N "
                      f"tilt={t:+.2f}deg wedge_x={wx:+.3f} v={wedge_speed():.3f}",
                      flush=True)
        clear_force(scene.wedge)
        print(f"[solve] push budget exhausted at tilt {tilt_deg():+.2f} deg", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rig.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    spawns = []
    for nm, body in (("wedge", scene.wedge), ("shim", scene.shim)):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        spawns.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f})")
    bx, by, bz = rig_xyz(scene.ball)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"tilt0={tilt_deg():+.2f}deg ball_rig=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
          + " ".join(spawns), flush=True)
    report("reset")
    for body in (scene.hopper, scene.wedge, scene.shim, scene.ball):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.ball_in_hopper()[0]), "ball must start inside the hopper"
    assert -4.5 < tilt_deg() < -0.6, f"resting back-tilt off: {tilt_deg():+.2f} deg"
    s0 = print_score("P0 reset+settle (ball sealed in the back-tilted hopper)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT the wedge into the runway channel -----------------
    # Hover 2 mm above the runway, tip (+x) toward the hopper, ~50 mm short of the
    # hopper's back edge. This is the release pose of a pick-and-place; it touches
    # nothing and lifts nothing.
    stage_wedge()
    report("staged")
    assert bool(scene._staged[0]), "wedge staged latch did not set"
    assert bool(scene.ball_in_hopper()[0]), "staging must not disturb the ball"
    assert not bool(scene.success()[0]), "a staged wedge cannot be success"
    s1 = print_score("P1 wedge staged in the channel (transport only)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.09, f"P1 score {s1} (expect 0.10)"

    # ---------------- phase 2: PUSH the wedge home; hopper tips; ball drains ---------------
    target = c.lift_full_deg + 1.0  # comfortably past the full-lift stage
    tipped = False
    for attempt in range(3):
        if attempt:
            print(f"[solve] re-staging the wedge (attempt {attempt + 1})", flush=True)
            stage_wedge()
        if not push_wedge_to_tilt(target):
            continue
        step(60)  # hands-off: does the self-lock hold?
        if tilt_deg() >= c.lift_full_deg:
            tipped = True
            break
        print(f"[solve] self-lock slipped to {tilt_deg():+.2f} deg", flush=True)
    if not tipped:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (push never held the drain tilt)", flush=True)
        os._exit(1)
    # hands-off: self-lock holds the tilt; the ball rolls out of the mouth, over
    # the cradle, down the deflector, into the basin
    drained = False
    for i in range(30):  # up to 1200 steps = 10 s
        step(40)
        if bool(scene.success()[0]):  # in basin, via mouth, fully settled
            drained = True
            break
        if i == 10 and bool(scene.ball_in_hopper()[0]):
            # ball still inside: a bit more tilt (self-lock margin allows up to ~9 deg)
            more = min(tilt_deg() + 1.5, 9.0)
            print(f"[solve] ball still inside at tilt {tilt_deg():+.2f} deg; "
                  f"pushing to {more:+.2f} deg", flush=True)
            push_wedge_to_tilt(more)
    report("drained")
    if not drained:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (ball did not settle in the basin)", flush=True)
        os._exit(1)
    assert bool(scene._via_mouth[0]), "drain must cross the mouth window"
    s2 = print_score("P2 wedge driven home; ball drained through the mouth into the basin")
    assert s2 >= s1 - 1e-6 and s2 >= 0.74, f"P2 score {s2} (expect 0.75)"

    # ---------------- phase 3: judged state, live ------------------------------------------
    step(90)
    report("judge")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after drain)", flush=True)
        os._exit(1)
    s3 = print_score("P3 ball at rest inside the basin, tilt held by the self-locked wedge")
    assert s3 >= s2 - 1e-6, "score decreased at the judged state"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
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
