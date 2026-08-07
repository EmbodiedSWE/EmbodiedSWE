"""Teleport solution for MoatCausewayScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_the_plate_i19`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write ending in FREE SPACE:
  P1 — the BLUE channel plank is carried from its ground spawn to a hover pose
  ~30 mm above the open moat, aligned with the crossing axis (read from the scene:
  the layout yaw and gap are randomized), and released. The BRIDGE-LAYING itself
  happens through CONTACT DYNAMICS: the plank falls and settles with both ends
  resting on the two deck rims under gravity and real contact — exactly the release
  an arm performs when laying a plank across a gap.
Everything else is DRIVEN, not bypassed:
  P2 — a force governor applies a bounded horizontal force to the RED cube along the
  crossing axis (the force a hand pushing the 110 mm cube applies — it is wider than
  the 80 mm jaw, so pushing is the only embodied option), with a lateral centring
  term and a gentle yaw-keeping torque. The cube slides along the start deck, climbs
  onto the channel floor, CROSSES THE MOAT riding the bridge between the curb rails,
  descends onto the green deck and onto the yellow pad. The force is cut and
  everything settles on real contact. The cube is never teleported over the moat or
  onto the pad; the span latch, the crossing latch, the arrival latch and success()
  are all earned by physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still
holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_the_plate_i19.solve --headless [--seed N]
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
    env = ENVS.get("simgen.moat_causeway")().build(num_envs=args.num_envs, device=device)
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

    from isaaclab.utils.math import matrix_from_quat, quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def axes() -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame unit vectors of the crossing axis (+x) and lateral (+y)."""
        _, mq = scene.mid_frame()
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        return quat_apply(mq, ex)[0], quat_apply(mq, ey)[0]

    def clear_wrench() -> None:
        scene.cube.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # EMPIRICAL (probed on the pod): set_external_force_and_torque applies the given
    # wrench dragged by the body's rotation since its reference pose —
    # applied_world = (R_now @ R_ref^T) @ f_given. While the cube stays in its spawn
    # orientation this is the identity, but after the quarter-tumble over the channel
    # lip "forward" turned into "press into the floor" and "backward" into "lift"
    # (probes: raw u -> 0 mm; R^T u and R u -> +71 mm forward). Encode every command
    # with the inverse, M = R_ref @ R_now^T, so the drag cancels and the wrench lands
    # in the world frame as intended.
    q_ref_holder: list = [None]

    def encode() -> torch.Tensor:
        Rn = matrix_from_quat(scene.cube.data.root_quat_w[0:1])[0]
        if q_ref_holder[0] is None:
            return torch.eye(3, device=device)
        Rr = matrix_from_quat(q_ref_holder[0])[0]
        return Rr @ Rn.T

    def push(f_ax: float, f_lat: float, tq_z: float) -> None:
        u, v = axes()
        M = encode()
        fvec = M @ (u * f_ax + v * f_lat)
        tvec = M @ torch.tensor([0.0, 0.0, tq_z], device=device)
        fw = fvec.view(1, 1, 3).expand(n, 1, 3).contiguous()
        tw = tvec.view(1, 1, 3).expand(n, 1, 3).contiguous()
        scene.cube.set_external_force_and_torque(fw, tw, env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
        tl = scene.tgt_local(scene.cube.data.root_pos_w)[0]
        pl = scene.mid_local(scene.plank.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | gap={float(scene.gap()[0]) * 1000:.0f}mm "
              f"cube_mid=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"cube_tgt=({float(tl[0]):+.3f},{float(tl[1]):+.3f}) "
              f"plank_mid=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
              f"span={bool(scene.spanning()[0])} on_pad={bool(scene.on_pad()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"span_l={float(scene.span_latch[0]):.2f} "
              f"cross_l={float(scene.cross_latch[0]):.2f} "
              f"arr_l={float(scene.arrive_latch[0]):.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    mp, mq = scene.mid_frame()
    ctr = (mp - scene.env_origins)[0]
    yaw = float(scene._yaw_of(scene.target_ped)[0])
    ml0 = scene.mid_local(scene.cube.data.root_pos_w)[0]
    pl0 = scene.mid_local(scene.plank.data.root_pos_w)[0]
    dl0 = scene.mid_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): centre=({float(ctr[0]):+.3f},"
          f"{float(ctr[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"gap={float(scene.gap()[0]) * 1000:.0f}mm "
          f"cube_mid=({float(ml0[0]):+.3f},{float(ml0[1]):+.3f}) "
          f"plank_mid=({float(pl0[0]):+.3f},{float(pl0[1]):+.3f}) "
          f"decoy_mid=({float(dl0[0]):+.3f},{float(dl0[1]):+.3f})", flush=True)
    report("reset")
    q_ref_holder[0] = scene.cube.data.root_quat_w[0:1].clone()  # settled spawn pose
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: plank TRANSPORT (hover over the moat) + gravity lay ---------
    # Hover: ~30 mm above deck-top height, centred on the moat, aligned with the
    # crossing axis, laterally aligned to the cube's lane (clamped to keep both ends
    # on the decks) — then free fall + settle: the release an arm performs when
    # laying a plank across a gap. The plank origin is its UNDERSIDE centre.
    y_max = c.deck_w / 2 - c.plank_w / 2 - 0.01
    y_c = max(-y_max, min(y_max, float(ml0[1])))

    def lay_at(local_y: float, hover: float) -> None:
        mp_, mq_ = scene.mid_frame()
        off = torch.tensor([[0.0, local_y, hover]], device=device)
        w = mp_[0:1] + quat_apply(mq_[0:1], off)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = w
        st[:, 3:7] = mq_[0:1]
        scene.plank.write_root_state_to_sim(st, all_ids)
        step(150)  # fall + settle on the rims (1.25 s)

    lay_at(y_c, 0.030)
    if not bool(scene.spanning()[0]):  # rare: bounced askew — retry centred
        print("[solve] lay retry at moat-local y=0", flush=True)
        y_c = 0.0
        lay_at(0.0, 0.030)
    report("bridged")
    s1 = print_score("P1 plank transport + gravity lay across the moat")
    assert s1 >= s0 - 1e-6, "score decreased across the plank transport"
    if not bool(scene.spanning()[0]):
        print("SIM_GEN_SOLVE: FAIL (plank did not span the moat)", flush=True)
        os._exit(1)

    # ---------------- phase 2: PUSH the cube across the bridge -----------------------------
    # Force governor along the crossing axis: f = clamp(25 * (0.07 - v_along), +/-3 N)
    # — the force of a hand pushing a 0.2 kg cube on low-friction faces — plus a
    # QUASI-STATIC BOOST that ramps up only while the cube is stalled (climbing the
    # 12 mm channel-floor lip needs ~2.5 N of tip-over force; the ramp pops it at
    # just-above-threshold force, never violently). A lateral PD centres the cube on
    # the plank lane; a gentle torque keeps its yaw square (mod 90 deg). When the
    # target line is crossed the governor BRAKES the cube to rest instead of letting
    # it coast. Abort-fail if the cube ever leaves deck height (fell).
    yaw_l = float(scene._yaw_of(scene.target_ped)[0])
    v_tgt, cap, boost = 0.07, 3.0, 0.0
    stop_x = c.pad_cx - 0.01  # target-frame x where the push ends (inside the pad)
    rail_half = c.plank_len / 2 - c.curb_apron + c.cube_s / 2  # cube overlaps the rails
    fell = False
    done = False

    def sq_yaw_err() -> float:
        """Yaw error mod 90 deg, TUMBLE-SAFE: measured from the cube's most
        horizontal body axis (after a quarter-roll over the floor lip the root quat
        is no longer a pure-z rotation)."""
        R = matrix_from_quat(scene.cube.data.root_quat_w[0:1])[0]
        best_az, err = 2.0, 0.0
        for k in range(3):
            a = R[:, k]
            az = abs(float(a[2]))
            if az < best_az:
                best_az = az
                ang = math.atan2(float(a[1]), float(a[0]))
                err = (ang - yaw_l + math.pi / 4) % (math.pi / 2) - math.pi / 4
        return err

    def drive_step(v_want: float) -> None:
        nonlocal boost
        u, v = axes()
        ml_ = scene.mid_local(scene.cube.data.root_pos_w)[0]
        vel = scene.cube.data.root_lin_vel_w[0]
        v_ax = float(torch.dot(vel, u))
        z = float(ml_[2])
        wsp = float(scene.cube.data.root_ang_vel_w[0].norm())
        # Mounting the 12 mm floor lip is a quasi-static quarter-tumble: the CoM must
        # arc over the pivot (balance at z ~ 0.082; flat-on-deck 0.055, flat-on-floor
        # 0.067). While mid-tip AND RISING, hold a CONSTANT just-above-tip-over
        # force — a velocity governor sees the tip's own motion and drops its force,
        # rocking the cube back forever. On the FALLING side of the roll, cut to
        # near-zero so the cube lands under gravity alone (a powered landing slams
        # an edge into the floor hard enough to tunnel and interlock).
        vz = float(vel[2])
        tipping = 0.071 < z < 0.105
        if tipping:
            f_ax = 2.9 if vz > -0.02 else 0.4
        else:
            f_base = max(-cap, min(cap, 25.0 * (v_want - v_ax)))
            if v_ax < 0.02:
                boost = min(boost + 0.004, 2.0)  # ~0.5 N/s quasi-static ramp
            else:
                boost *= 0.95
            f_ax = min(f_base + boost, 4.2)
        # Lateral centring everywhere EXCEPT alongside the curb rails (side loads
        # there only wedge the cube); yaw squaring always, tiny near the rails.
        in_rails = abs(float(ml_[0])) < rail_half
        unsettled = tipping or wsp > 0.4
        f_lat = 0.0
        if not in_rails and not unsettled:
            v_lat = float(torch.dot(vel, v))
            e_y = y_c - float(ml_[1])
            f_lat = max(-0.8, min(0.8, 6.0 * e_y - 2.0 * v_lat))
        wz = float(scene.cube.data.root_ang_vel_w[0, 2])
        tq_cap = 0.05 if in_rails else 0.10
        tq = 0.0 if unsettled else max(-tq_cap, min(tq_cap,
                                                    -0.4 * sq_yaw_err() - 0.02 * wz))
        push(f_ax, f_lat, tq)
        env.step(no_action)

    def unwedge() -> None:
        """Stalled at max boost: back off along the axis, square the yaw unloaded,
        then let the main loop resume."""
        nonlocal boost
        print("[solve] push: stalled at max boost — backing off to re-square",
              flush=True)
        boost = 0.0
        for _ in range(100):
            u, _v = axes()
            v_ax = float(torch.dot(scene.cube.data.root_lin_vel_w[0], u))
            push(max(-1.5, min(1.5, 25.0 * (-0.04 - v_ax))), 0.0, 0.0)
            env.step(no_action)
        clear_wrench()
        for _ in range(150):
            wz = float(scene.cube.data.root_ang_vel_w[0, 2])
            tq = max(-0.06, min(0.06, -0.8 * sq_yaw_err() - 0.05 * wz))
            push(0.0, 0.0, tq)
            env.step(no_action)
        clear_wrench()

    def brake() -> None:
        for _ in range(120):
            u, _v = axes()
            v_ax = float(torch.dot(scene.cube.data.root_lin_vel_w[0], u))
            if abs(v_ax) < 0.02:
                break
            push(max(-3.0, min(3.0, 25.0 * (0.0 - v_ax))), 0.0, 0.0)
            env.step(no_action)
        clear_wrench()

    blast = False
    stall = 0
    best_x = -10.0
    for i in range(5400):
        tl = scene.tgt_local(scene.cube.data.root_pos_w)[0]
        ml = scene.mid_local(scene.cube.data.root_pos_w)[0]
        if float(tl[0]) >= stop_x:
            done = True
            break
        if float(ml[2]) < 0.020:  # below deck height anywhere = fell
            fell = True
            break
        spd = float(scene.cube.data.root_lin_vel_w[0].norm())
        wsp = float(scene.cube.data.root_ang_vel_w[0].norm())
        # Solver blast guard. A powered quarter-roll over the channel lip is a
        # real, healthy motion that briefly hits |v| ~ 1, |w| ~ 12-15 — that must
        # NOT be fatal. Hard-fail only for truly ballistic states (flying); for a
        # merely fast tumble still at deck height, drop the wrench and let the
        # cube settle, then resume the governor.
        if spd > 3.0 or float(ml[2]) > 0.30:
            print(f"[solve] BLAST at step {i}: mid=({float(ml[0]):+.3f},"
                  f"{float(ml[1]):+.3f},{float(ml[2]):+.3f}) |v|={spd:.2f} "
                  f"|w|={wsp:.1f} boost={boost:.2f}", flush=True)
            blast = True
            break
        if spd > 1.2 or wsp > 20.0:
            print(f"[solve] push: fast tumble at step {i} (|v|={spd:.2f} "
                  f"|w|={wsp:.1f}) — coasting to settle", flush=True)
            clear_wrench()
            boost = 0.0
            for _ in range(80):
                env.step(no_action)
                if float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.05:
                    break
            continue
        x_now = float(ml[0])
        if x_now > best_x + 0.005:  # progress-based stall detection
            best_x = x_now
            stall = 0
        else:
            stall += 1
        if stall > 700:
            unwedge()
            stall = 0
            continue
        drive_step(v_tgt)
        if i > 0 and i % 150 == 0:
            print(f"[solve] push: step {i} mid=({float(ml[0]):+.3f},"
                  f"{float(ml[1]):+.3f},{float(ml[2]):+.3f}) |v|={spd:.3f} "
                  f"|w|={wsp:.2f} boost={boost:.2f} "
                  f"yaw_err={math.degrees(sq_yaw_err()):+.1f}", flush=True)
    brake()
    step(90)
    report("pushed")
    if blast:
        print("SIM_GEN_SOLVE: FAIL (solver blast during the push)", flush=True)
        os._exit(1)
    if fell or float(scene.mid_local(scene.cube.data.root_pos_w)[0, 2]) < 0.020:
        print("SIM_GEN_SOLVE: FAIL (cube fell off the causeway)", flush=True)
        os._exit(1)
    if not done:
        print("SIM_GEN_SOLVE: FAIL (push budget exhausted)", flush=True)
        os._exit(1)

    # Nudge correction: if the cube settled short of the pad, small pulses forward.
    for _ in range(3):
        if bool(scene.on_pad()[0]) or not bool(scene.on_target_deck()[0]):
            break
        tl = scene.tgt_local(scene.cube.data.root_pos_w)[0]
        print(f"[solve] nudge: cube at tgt-x {float(tl[0]):+.3f}, pad needs "
              f">= {c.pad_cx - c.pad_l / 2:+.3f}", flush=True)
        boost = 0.0
        for _ in range(120):
            tl = scene.tgt_local(scene.cube.data.root_pos_w)[0]
            if float(tl[0]) >= stop_x:
                break
            drive_step(0.04)
        brake()
        step(60)
    report("placed")
    s2 = print_score("P2 cube pushed across the bridge onto the pad")
    assert s2 >= s1 - 1e-6, "score decreased across the push"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the push)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
