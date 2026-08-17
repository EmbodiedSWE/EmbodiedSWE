"""Teleport solution for TeaBallTransferScene (sim_gen task
`approach_grasp_ceramic_teapot_i209`) — the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. UNSEAL (contact dynamics): a velocity-servoed vertical force at the lid's CoM
   (the emulation of the arm pinching the 20 mm knob and lifting) raises the
   stopper straight out of the source canister's funnel mouth. The plug leaves its
   wedge seat through real contact; nothing is written "open". The scene's
   `_lid_open` latch fires from physics.
2. PARK THE LID (teleport, free-space transport only): one pose write stages the
   lid hovering over open floor away from both canisters; it FALLS and settles on
   the ground. It is never written into any goal-relevant state.
3. EXTRACT THE BALL (contact dynamics): a velocity-servoed vertical force at the
   ball's CoM (the arm's top-down in-pot grasp-and-lift) raises the ball out of
   the interior, through the 116 mm aperture and the funnel mouth, to clear air
   above the rim. The caging is real: this force could NOT have moved the ball
   out while the lid was seated (smoke proves that separately).
4. DEPOSIT (teleport transport + contact dynamics): the ball is staged hovering
   over the DESTINATION canister's mouth (free space), then RELEASED — it falls
   through the funnel, rattles, and settles inside under gravity. The `_ball_in`
   band is below the sill, so only a genuine settled containment counts.
5. SEAT THE LID (teleport transport + contact dynamics — the precision finish):
   the lid is staged hovering level over the destination mouth with a DELIBERATE
   6 mm lateral offset (inside the funnel's ~14 mm capture, off the perfect
   centre — the seat must tolerate a realistic place, not a magic one), then
   RELEASED. The plug drops into the collar, self-centres down the funnel flats,
   and wedges level at bearing depth. The scene's `_lid_seated_on(dest)` readback
   (axis offset + depth + tilt + stillness) must confirm it.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

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
    env = ENVS.get("simgen.tea_ball_transfer")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ls = scene._local(scene.lid, True)[0]
        ld = scene._local(scene.lid, False)[0]
        bd = scene._local(scene.ball, False)[0]
        print(f"[solve] {tag:12s} | lid_src=({float(ls[0]):+.3f},{float(ls[1]):+.3f},"
              f"{float(ls[2]):.3f}) lid_dst=({float(ld[0]):+.3f},{float(ld[1]):+.3f},"
              f"{float(ld[2]):.3f}) ball_dst=({float(bd[0]):+.3f},{float(bd[1]):+.3f},"
              f"{float(bd[2]):.3f}) opened={bool(scene._opened_ever[0])} "
              f"transferred={bool(scene._transferred_ever[0])} "
              f"seated_dst={bool(scene._lid_seated_on(False)[0])} "
              f"ball_in_dst={bool(scene._ball_in(False)[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def lift_vertical(body, mass: float, done, tag: str, max_iters: int = 900) -> bool:
        """Velocity-servoed vertical force at the body's CoM until done() is True.
        Runtime probe: if the pod's external-force API drags the wrench frame,
        measured height progress collapses — toggle the pre-encoding and continue.
        Returns True once done() fired; leaves the wrench CLEARED."""
        q_ref = body.data.root_quat_w.clone()
        encode_mode = 0  # 0: raw world force; 1: pre-rotate by R_ref * R_now^T

        def encode(f_w: torch.Tensor) -> torch.Tensor:
            if encode_mode == 0:
                return f_w
            q_now = body.data.root_quat_w
            return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_w)

        v_des, k_gain = 0.25, 20.0
        f_cap = 3.0 * mass * 9.81
        z0 = float(body.data.root_pos_w[0, 2])
        best_z, last_bump, stalls = z0, 0, 0
        ok = False
        for i in range(max_iters):
            if bool(done()):
                ok = True
                break
            vz = float(body.data.root_lin_vel_w[0, 2])
            f_up = mass * (9.81 + k_gain * (v_des - vz))
            f_up = max(0.0, min(f_up, f_cap))
            f_w = torch.tensor([0.0, 0.0, f_up], device=device).view(1, 1, 3).expand(
                n, 1, 3)
            body.set_external_force_and_torque(encode(f_w).contiguous(), zero_wrench,
                                               env_ids=all_ids, is_global=True)
            env.step(no_action)
            z = float(body.data.root_pos_w[0, 2])
            if z > best_z + 0.002:
                best_z, last_bump, stalls = z, i, 0
            elif i - last_bump > 120:  # stalled
                stalls += 1
                last_bump = i
                if stalls >= 2:
                    encode_mode ^= 1
                    f_cap = 3.0 * mass * 9.81
                    stalls = 0
                    print(f"[solve] {tag}: stall persists at z={z:.3f} -> toggle "
                          f"pre-encode to mode {encode_mode}", flush=True)
                else:
                    f_cap = min(f_cap * 1.5, 8.0 * mass * 9.81)
                    print(f"[solve] {tag}: stall at z={z:.3f} -> f_cap={f_cap:.2f} N",
                          flush=True)
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        return ok

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    src_pos, src_quat = scene._pot_state(True)
    dst_pos, dst_quat = scene._pot_state(False)
    sp = (src_pos - scene.env_origins)[0]
    dp = (dst_pos - scene.env_origins)[0]
    ls0 = scene._local(scene.lid, True)[0]
    bs0 = scene._local(scene.ball, True)[0]
    print(f"[solve] layout readback (seed {args.seed}): src_is_a="
          f"{bool(scene.src_is_a[0])} src=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) "
          f"dst=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) "
          f"lid_src_loc=({float(ls0[0]):+.3f},{float(ls0[1]):+.3f},{float(ls0[2]):.4f}) "
          f"ball_src_loc=({float(bs0[0]):+.3f},{float(bs0[1]):+.3f},{float(bs0[2]):.4f})",
          flush=True)
    report("reset")
    assert bool(scene._lid_seated_on(True)[0]), \
        f"lid did not settle seated on the source (lid_src_z={float(ls0[2]):.4f}, " \
        f"analytic seat {c.lid_seat_z:.4f})"
    assert bool(scene._ball_in(True)[0]), "ball did not spawn resting inside the source"
    assert not bool(scene._opened_ever[0]), "opened latch fired at reset?!"
    s0 = print_score("P0 reset+settle (sealed source, empty destination)")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: UNSEAL (contact dynamics) ------------------------------------
    # Vertical velocity-servo force lifts the stopper straight out of the funnel.
    lifted = lift_vertical(
        scene.lid, c.lid_mass,
        lambda: scene._local(scene.lid, True)[0, 2] > c.lid_seat_z + 0.12,
        "lid-lift")
    report("unsealed")
    assert lifted, "the lift force never raised the lid clear of the mouth"
    assert bool(scene._opened_ever[0]), "opening did not latch"
    s1 = print_score("P1 lid force-lifted clear of the source mouth (unsealed)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_open - 1e-6, "opening credit missing"

    # ---------------- phase 2: park the lid (teleport transport + gravity) ------------------
    # Free-space transport to a hover over open floor; it falls and rests there.
    park = torch.tensor([0.15, 0.0, 0.03], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.env_origins + park
    st[:, 3] = 1.0
    scene.lid.write_root_state_to_sim(st, all_ids)
    quiet = 0
    for _ in range(360):
        env.step(no_action)
        still = float(scene.lid.data.root_lin_vel_w[0].norm()) < c.settle_speed
        quiet = quiet + 1 if still else 0
        if quiet >= 20:
            break
    report("parked")
    assert quiet >= 20, "parked lid did not settle on the floor"
    s2 = print_score("P2 lid parked on open floor")
    assert s2 >= s1 - 1e-6, "score decreased across the park"

    # ---------------- phase 3: EXTRACT THE BALL (contact dynamics) --------------------------
    # Vertical velocity-servo force lifts the ball out through the (now open) mouth.
    raised = lift_vertical(
        scene.ball, c.ball_mass,
        lambda: scene._local(scene.ball, True)[0, 2] > c.rim_z + 0.08,
        "ball-lift")
    report("extracted")
    assert raised, "the lift force never raised the ball clear of the source"
    assert not bool(scene._ball_in(True)[0]), "ball still reads as inside the source"
    s3 = print_score("P3 ball force-lifted out of the source")
    assert s3 >= s2 - 1e-6, "score decreased across the extraction"

    # ---------------- phase 4: DEPOSIT (teleport hover + gravity drop) ----------------------
    # Stage the ball over the destination mouth (free space), release, let it fall
    # through the funnel and settle inside. Containment band is below the sill —
    # only a genuine settled rest counts.
    dst_pos, dst_quat = scene._pot_state(False)
    hover = torch.tensor([0.004, 0.0, c.rim_z + 0.06], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = dst_pos + quat_apply(dst_quat, hover)
    st[:, 3] = 1.0
    scene.ball.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene._ball_in(False)[0]), \
        "hover pose already reads as inside (teleport must not place the ball)"
    quiet = 0
    landed = False
    for i in range(720):
        env.step(no_action)
        if i % 120 == 0:
            report(f"drop+{i}")
        inside = bool(scene._ball_in(False)[0])
        still = bool(scene._ball_still()[0])
        quiet = quiet + 1 if (inside and still) else 0
        if quiet >= 30:
            landed = True
            break
    report("deposited")
    assert landed, "the ball did not settle inside the destination canister"
    assert bool(scene._transferred_ever[0]), "transfer did not latch"
    s4 = print_score("P4 ball dropped through the destination funnel and settled inside")
    assert s4 >= s3 - 1e-6 and s4 >= 0.40 - 1e-6, "transfer credit missing"

    # ---------------- phase 5: SEAT THE LID (teleport hover + contact wedge-seat) -----------
    # Stage the lid level over the destination mouth with a deliberate 6 mm lateral
    # offset (inside the funnel capture), release: the plug self-centres down the
    # collar flats and wedges at bearing depth. Never written seated.
    dst_pos, dst_quat = scene._pot_state(False)
    hover = torch.tensor([0.006, 0.0, c.lid_seat_z + 0.10], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = dst_pos + quat_apply(dst_quat, hover)
    st[:, 3:7] = dst_quat  # level
    scene.lid.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene._lid_seated_on(False, require_still=False)[0]), \
        "hover pose already reads as seated (teleport must not seat the lid)"
    quiet = 0
    seated = False
    for i in range(720):
        env.step(no_action)
        if i % 120 == 0:
            report(f"seat+{i}")
        quiet = quiet + 1 if bool(scene._lid_seated_on(False)[0]) else 0
        if quiet >= 30:
            seated = True
            break
    report("sealed")
    assert seated, "the lid did not settle seated in the destination mouth"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after sealing)", flush=True)
        os._exit(1)
    s5 = print_score("P5 lid dropped into the destination funnel and wedge-seated")
    assert s5 >= s4 - 1e-6 and s5 >= 1.0 - 1e-6, "success score missing"

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
