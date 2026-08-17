"""Teleport solution for RailHangerScene (sim_gen task
`libero_pick_salad_dressing_i286`) — the task's legitimacy certificate.

Teleport = TRANSPORT ONLY. The single root-state write carries the amber bottle
from its ground station to FREE AIR beside the open mouth of the rail slot
(rack-local x = +0.25, outside the flare tips at +0.178 — touching nothing).
Every load-bearing interaction afterwards is applied wrenches + contact:

1. THREAD (applied force): a PD/velocity servo — the wrench a Franka wrist
   applies through a barrel grasp — holds the bottle upright at insertion
   height (neck level with the rail slot, cap above the rail tops, body below
   the rail undersides) and drives it slowly along the slot axis through the
   flared mouth into the slot. The neck (dia 20) travels between the rail
   inner faces (gap 32); the flares recapture small lateral errors.
2. HOOK (contact): lift support is cut to 0.7*m*g — the bottle sinks until its
   cap flange (dia 52 > gap 32) lands on the rail tops and HANGS; the rails
   carry the remaining 0.3*m*g through slick cap-on-rail contact. `hooked`
   latches on this genuinely supported state while slow.
3. SLIDE (applied force + contact): the same servo (still at 0.7*m*g lift, cap
   riding the rails) drives the hanging bottle along the slot to the closed
   end until the bottle BODY stops against the full-height column face
   (root x = -0.130 + body_r = -0.102, confirmed by forge readback). The
   servo gain escalates if the slide stalls. `seated` latches at the stop.
4. RELEASE (gravity + contact): the wrench is zeroed while slow; the bottle
   hangs free — its whole weight on the cap-on-rails contact — and stays at
   the seat. success() turns True after the consecutive-still window.
5. IDENTITY / DECOYS: the RED bottle and the basket are never touched; the
   decoy-exclusion clause is asserted throughout.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_salad_dressing_i286.solve --headless [--seed N]
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
    from . import scene as _task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rail_hanger")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    G = 9.81
    MG = c.bottle_mass * G                    # 2.45 N
    EZ = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
    ZERO = torch.zeros(n, 1, 3, device=device)
    # Servo gains (m = 0.25 kg, dt = 1/120; the wrench acts one substep late, so
    # keep K*dt/m well under 1: kd_z*dt/m = 0.1, kx*dt/m = 0.1 at kx = 3).
    KP_Z, KD_Z = 12.0, 3.0
    KP_Y, KD_Y = 8.0, 2.0
    KQ, KW = 0.05, 0.010                      # uprighting PD (I ~ 4.6e-4 kg m^2)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rack_q() -> torch.Tensor:
        return scene.rack.data.root_quat_w

    def bot_loc() -> torch.Tensor:
        return scene._rack_local(scene.dressing.data.root_pos_w)

    def hands_off() -> None:
        scene.dressing.set_external_force_and_torque(ZERO, ZERO)

    def teleport_to_mouth(z_tgt: float) -> None:
        """TRANSPORT: free-air pose beside the open mouth, upright, zero velocity,
        rack-yaw-aligned. Rack-local x = +0.25 > flare tips (+0.178): no contact."""
        hands_off()
        loc = torch.tensor([[0.25, 0.0, z_tgt]], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rack.data.root_pos_w + quat_apply(rack_q(), loc)
        st[:, 3:7] = rack_q()
        scene.dressing.write_root_state_to_sim(st, all_ids)

    def servo_step(z_tgt: float | None, vx_des: float, lift_frac: float | None,
                   kx: float) -> None:
        """One substep of the grasp-wrench servo, computed in the rack frame:
        - z: PD to z_tgt with gravity feedforward (free-air hold), OR a constant
          lift_frac*m*g (cap riding the rails carries the rest);
        - y: PD to the slot centreline;
        - x: velocity servo toward vx_des along the slot axis;
        - orientation: uprighting PD torque (the grasp holds the bottle upright).
        Forces are clamped; conversion rack-local -> world -> BODY frame."""
        q_r = rack_q()
        loc = bot_loc()
        v_loc = quat_apply_inverse(q_r, scene.dressing.data.root_lin_vel_w)
        f_loc = torch.zeros(n, 3, device=device)
        if lift_frac is None:
            f_loc[:, 2] = (MG + KP_Z * (z_tgt - loc[:, 2])
                           - KD_Z * v_loc[:, 2]).clamp(0.0, 2.0 * MG)
        else:
            f_loc[:, 2] = lift_frac * MG
        f_loc[:, 1] = (KP_Y * (0.0 - loc[:, 1]) - KD_Y * v_loc[:, 1]).clamp(-2.0, 2.0)
        f_loc[:, 0] = (kx * (vx_des - v_loc[:, 0])).clamp(-3.0, 3.0)
        f_w = quat_apply(q_r, f_loc)
        q_b = scene.dressing.data.root_quat_w
        up = quat_apply(q_b, EZ)
        tau_w = KQ * torch.cross(up, EZ, dim=-1) - KW * scene.dressing.data.root_ang_vel_w
        scene.dressing.set_external_force_and_torque(
            quat_apply_inverse(q_b, f_w).reshape(n, 1, 3),
            quat_apply_inverse(q_b, tau_w).reshape(n, 1, 3))
        env.step(no_action)

    def report(tag: str) -> None:
        loc = bot_loc()[0]
        up = float(scene._up_z(scene.dressing)[0])
        print(f"[solve] {tag:14s} | "
              f"loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"up_z={up:+.3f} "
              f"hang={bool(scene.hanging(scene.dressing)[0])} "
              f"seat={bool(scene.seated(scene.dressing)[0])} "
              f"latch=[{int(scene._hooked[0])}{int(scene._seated[0])}] "
              f"decoy_off={bool(scene.decoy_off_rack()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # pieces settle at their scatter stations
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    rq = rack_q()[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    stn = int(scene.dressing_station[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"dressing at station {stn}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.hanging(scene.dressing)[0]), "dressing must start on the ground"
    assert not bool(scene.hanging(scene.decoy)[0]), "decoy must start on the ground"
    assert bool(scene.decoy_off_rack()[0]), "decoy must start off the rack"
    s0 = print_score("P0 reset+settle (bottles standing at their stations)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: thread the neck in through the open mouth -------------------
    # Insertion window: root z in (0.210, 0.230) — cap above the rail tops, body
    # below the rail undersides. Retries re-transport at small z offsets.
    threaded = False
    for attempt, z_tgt in enumerate((0.220, 0.215, 0.225)):
        teleport_to_mouth(z_tgt)
        for _ in range(700):
            servo_step(z_tgt, -0.10, None, 3.0)
            if float(bot_loc()[0, 0]) < 0.02:
                threaded = True
                break
        if threaded:
            if attempt > 0:
                print(f"[solve] thread: succeeded on retry {attempt} (z={z_tgt:.3f})",
                      flush=True)
            break
        loc = bot_loc()[0]
        print(f"[solve] thread attempt {attempt} (z={z_tgt:.3f}) timed out at "
              f"loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}); "
              f"re-transporting", flush=True)
    if not threaded:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (thread never entered the slot)", flush=True)
        os._exit(1)

    # HOOK: cut lift to 0.7*m*g and stop the traverse — the cap sinks onto the
    # rail tops and the bottle hangs (rails carry 0.3*m*g); slow -> hooked latches.
    for _ in range(150):
        servo_step(None, 0.0, 0.7, 3.0)
    report("threaded+hook")
    assert bool(scene.hanging(scene.dressing)[0]), "bottle must be hanging in the slot"
    assert bool(scene._hooked[0]), "hooked credit must have latched"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 neck threaded through the mouth; cap hanging on the rails")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_hook - 1e-6, f"P1 score {s1}"

    # ---------------- phase 2: slide along the slot to the closed-end seat -----------------
    kx = 3.0
    last_x = float(bot_loc()[0, 0])
    stall = 0
    at_seat = False
    for _ in range(900):
        servo_step(None, -0.10, 0.7, kx)
        x_now = float(bot_loc()[0, 0])
        if x_now <= -0.099:  # body-on-column stop is at -0.102
            at_seat = True
            break
        stall = stall + 1 if x_now > last_x - 0.0003 else 0
        last_x = x_now
        if stall >= 150:
            kx = min(kx * 1.6, 12.0)
            stall = 0
            print(f"[solve] slide stalled at x={x_now:+.3f}; escalating kx -> {kx:.1f}",
                  flush=True)
    if not at_seat:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (slide never reached the seat)", flush=True)
        os._exit(1)
    # press gently onto the stop while slow so `seated` latches on the pressed
    # state; keep this SHORT (40 < still_steps 60) so success cannot fire yet
    for _ in range(40):
        servo_step(None, -0.05, 0.7, 3.0)
    report("slid->seat")
    assert bool(scene.seated(scene.dressing)[0]), "bottle must hang at the closed end"
    assert bool(scene._seated[0]), "seated credit must have latched"
    assert not bool(scene.success()[0]), "no success while the still window runs"
    s2 = print_score("P2 slid along the slot; body at the column stop")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_hook + c.w_seat - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: release — the rails carry the whole weight ------------------
    hands_off()
    step(300)  # free hang; still counter fills (60 substeps)
    report("released")
    assert bool(scene.hanging(scene.dressing)[0]), "bottle must still hang after release"
    assert bool(scene.seated(scene.dressing)[0]), "bottle must stay at the seat"
    assert bool(scene.decoy_off_rack()[0]), "decoy must remain off the rack"
    if not bool(scene.success()[0]):
        step(180)  # give the still counter margin
        report("release-wait")
    assert bool(scene.success()[0]), "seated + settled + decoy-off must be success"
    s3 = print_score("P3 released; bottle hanging free at the seat")
    assert s3 >= s2 - 1e-6 and s3 >= 1.0 - 1e-6, f"P3 score {s3} (expect 1.0)"

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
    try:
        main()
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
