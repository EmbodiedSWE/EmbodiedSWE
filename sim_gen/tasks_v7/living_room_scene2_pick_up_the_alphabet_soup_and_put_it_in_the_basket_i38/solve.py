"""Teleport solution for SpringBayScene (sim_gen task
`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i38`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the blue can is teleported once, from its floor
spawn to a hover pose above the bay's channel (horizontal, axis along the channel,
back end 2.5 mm clear of the plunger face, front end over the orange lip), with zero
velocity — exactly what a pick-and-carry delivers — and RELEASED. Everything the
rubric reads happens through contact dynamics afterwards:

  drop    — gravity lands the can PROPPED: back rim on the channel floor, front end
            resting on the lip's inner top edge (the rest gap is 18 mm shorter than
            the can — this is the state a naive "drop it in the basket" ends in, and
            it scores only partial credit; the rubric's compression gate stays shut).
  press   — an applied wrench (a proxy for the gripper pressing the can axially,
            low on its back end: a bay -x force plus the pitch-down torque that
            moves the line of action to the back-bottom contact — a bare CoM push
            rears the propped can upright instead)
            pushes the can back into the spring plunger. The spring compresses, the
            propped front end slides backward across the slick lip top, tips over the
            inner edge, and pivots down the lip's inner face while the press
            continues to ~21 mm of compression, leaving the can flat on the channel
            floor with its front face behind the lip.
  release — forces cleared. The STORED SPRING ENERGY finishes the task: the plunger
            shoves the can forward until its front face seats against the back of the
            lip, leaving ~18 mm of standing compression. Nothing is touched again.

The press force is world-encoded through `encode_force` with a RUNTIME force-frame
probe (some pods rotate applied wrenches by the body's rotation since its reference
orientation — and this can rotates 90 degrees from its spawn before being pressed):
the press is attempted in mode 0 (raw world force); if the spring shows no
compression progress the state is rolled back to the propped snapshot and the press
retries pre-encoded against the reset readback, then against the press-start
readback.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0.10 carry -> 0.25 propped -> 0.55 pressed -> 1.0 seated), then holds HANDS-OFF
for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import L_CAN, PLG_TRAVEL, Z_F, _qapply, _qmul, _qy, encode_force  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import L_CAN, PLG_TRAVEL, Z_F, _qapply, _qmul, _qy, encode_force  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER_LOCAL = (0.038, 0.0, Z_F + 0.088)  # can-center release pose in the bay frame
PRESS_EXIT_COMP = 0.021  # press until 21 mm compression (seat needs 18, travel is 32)
PRESS_EXIT_Z = 0.040  # ... with the can center down at floor level (lying: 0.033)
PRESS_EXIT_AX = math.cos(math.radians(14.0))  # ... and the axis flat along the channel


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spring_bay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def comp() -> float:
        return float(scene.compression()[0])

    def blue_loc() -> torch.Tensor:
        return scene.bay_local(scene.blue.data.root_pos_w)[0]

    def clear_forces() -> None:
        scene.blue.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        loc = blue_loc()
        ax = scene.can_axis_local(scene.blue)[0]
        print(f"[solve] {tag:12s} | blue_bay=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]) - Z_F:+.3f}) ax_x={float(ax[0]):+.3f} "
              f"comp={comp() * 1000:5.1f}mm chan={bool(scene.in_channel_loose(scene.blue)[0])} "
              f"seated={bool(scene.seated_geom()[0])} "
              f"settled={bool(scene.settled(scene.blue)[0])} "
              f"plg_settled={bool(scene.plunger_settled()[0])} "
              f"decoy_clear={bool(scene.decoy_clear()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.blue)[0]) and bool(scene.plunger_settled()[0]):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    bay_p = (scene.bay.data.root_pos_w - scene.env_origins)[0]
    blue_p = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    red_p = (scene.red.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.bay.data.root_quat_w[0, 3]),
                           float(scene.bay.data.root_quat_w[0, 0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"bay=({float(bay_p[0]):+.3f},{float(bay_p[1]):+.3f}) yaw={math.degrees(yaw):+.0f}deg "
          f"blue=({float(blue_p[0]):+.3f},{float(blue_p[1]):+.3f}) "
          f"red=({float(red_p[0]):+.3f},{float(red_p[1]):+.3f}) "
          f"comp={comp() * 1000:.1f}mm", flush=True)
    report("reset")
    assert comp() < 0.004, f"plunger must rest near zero compression, got {comp() * 1000:.1f}mm"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (plunger at rest, cans on the floor)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientation candidates for `encode_force`.
    q_reset = scene.blue.data.root_quat_w.clone()  # upright, at reset settle

    # ---------------- phase 1: TRANSPORT ONLY — carry the can over the channel -------------
    def teleport_hover() -> None:
        """Place the blue can at rest above the channel: horizontal, axis along bay x
        (local +z -> bay +x via a 90 deg turn about y), back end clear of the plunger
        face, front end over the lip. Zero velocity, then hands off."""
        q_bay = scene.bay.data.root_quat_w
        local = torch.tensor(HOVER_LOCAL, device=device).expand(n, 3)
        half_pi = torch.full((n,), math.pi / 2, device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.bay.data.root_pos_w + _qapply(q_bay, local)
        st[:, 3:7] = _qmul(q_bay, _qy(half_pi))
        scene.blue.write_root_state_to_sim(st, all_ids)

    teleport_hover()
    report("carry")
    s1 = print_score("P1 carried over the bay (teleport transport, zero velocity)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"

    # ---------------- phase 2: hands-off drop -> PROPPED on the lip ------------------------
    for attempt in range(3):
        wait_settled(600)
        if bool(scene.in_channel_loose(scene.blue)[0]):
            break
        print(f"[solve] drop attempt {attempt}: can not in the channel — re-dropping",
              flush=True)
        teleport_hover()
    report("propped")
    assert bool(scene.in_channel_loose(scene.blue)[0]), "can failed to land in the channel"
    assert comp() < c.comp_engage, \
        f"gravity alone must not engage the spring (comp {comp() * 1000:.1f}mm)"
    assert not bool(scene.success()[0]), \
        "the casual drop must NOT be success (this is the seed's strategy, rejected)"
    s2 = print_score("P2 dropped -> propped on the lip (gravity + contact; the naive "
                     "end state, not success)")
    assert s2 >= s1 - 1e-6, "score decreased across the drop"

    # ---------------- phase 3: press axially into the spring plunger -----------------------
    snap = scene.get_state(all_ids)

    def press(mode: int, q_ref: torch.Tensor, gain: float, tag: str) -> bool:
        """Ramped press: bay -x force PLUS the pitch-down torque that makes the
        wrench equivalent to pushing at the can's back-bottom contact (a bare CoM
        push torques the propped can about that contact and rears it UPRIGHT —
        observed on the forge: ax_x -> 0.00, standing can, stall). tau_y ~
        F*(L/2*|ax_z| + R*|ax_x|) cancels the rear-up; gravity then lowers the front
        end as the gap opens. Returns True once the can lies flat with >=
        PRESS_EXIT_COMP of spring compression; aborts early (False) on no spring
        progress (wrong force-frame mode) or a vertical/ejected can."""
        comp0 = comp()
        for i in range(1500):
            F = min(5.0 + 11.0 * (i / 500.0), 16.0)
            if comp() > PLG_TRAVEL - 0.005:  # safety: never bottom out the joint
                F = min(F, 10.0)
            ax_v = scene.can_axis_local(scene.blue)[0]
            axx, axz = float(ax_v[0].abs()), float(ax_v[2].abs())
            f_bay = torch.tensor([-F, 0.0, 0.0], device=device).expand(n, 3)
            f_world = _qapply(scene.bay.data.root_quat_w, f_bay).clone()
            f_world[:, 2] -= 0.20 * F  # mild extra pitch-down + keeps contact loaded
            # Pitch-down torque, TAPERED to zero as the can flattens (axz -> 0): a
            # residual torque on a flat can pivots it over its FRONT rim instead
            # (observed on the forge: reared upright at 29 mm compression).
            tau = gain * F * (L_CAN / 2 * axz + 0.033 * axx * min(1.0, axz / 0.35))
            t_bay = torch.tensor([0.0, tau, 0.0], device=device).expand(n, 3)
            t_world = _qapply(scene.bay.data.root_quat_w, t_bay)
            q_now = scene.blue.data.root_quat_w
            f_arg = encode_force(mode, q_ref, q_now, f_world)
            t_arg = encode_force(mode, q_ref, q_now, t_world)
            scene.blue.set_external_force_and_torque(
                f_arg.view(n, 1, 3), t_arg.view(n, 1, 3), env_ids=all_ids,
                is_global=True)
            env.step(no_action)
            if i % 120 == 119:
                loc = blue_loc()
                print(f"[solve] {tag} @{i + 1}: F={F:4.1f}N tau={tau:4.2f}Nm "
                      f"comp={comp() * 1000:5.1f}mm z={float(loc[2]) - Z_F:+.3f} "
                      f"ax_x={axx:+.3f}", flush=True)
            # wrong-mode probe: no compression progress after 2 s
            if i == 239 and comp() < comp0 + 0.003:
                clear_forces()
                print(f"[solve] {tag}: no spring progress after 2 s "
                      f"(comp {comp0 * 1000:.1f} -> {comp() * 1000:.1f}mm) — wrong "
                      f"force-frame mode", flush=True)
                return False
            loc = blue_loc()
            # vertical trap: the can reared upright anyway — abort, roll back, retune
            if i >= 120 and axx < 0.35 and float(loc[2]) - Z_F > 0.046:
                clear_forces()
                print(f"[solve] {tag}: can reared upright (ax_x={axx:.2f}, "
                      f"z={float(loc[2]) - Z_F:+.3f}) — aborting this attempt",
                      flush=True)
                return False
            if i >= 239 and not bool(scene.in_channel_loose(scene.blue)[0]):
                clear_forces()
                print(f"[solve] {tag}: can left the channel mid-press", flush=True)
                return False
            if comp() >= PRESS_EXIT_COMP and axx >= PRESS_EXIT_AX \
                    and float(loc[2]) - Z_F <= PRESS_EXIT_Z:
                # hold briefly so the flat pose stabilizes under load, then done
                for _ in range(60):
                    q_now = scene.blue.data.root_quat_w
                    f_arg = encode_force(mode, q_ref, q_now, f_world)
                    scene.blue.set_external_force_and_torque(
                        f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids,
                        is_global=True)
                    env.step(no_action)
                clear_forces()
                return True
        clear_forces()
        print(f"[solve] {tag}: press budget exhausted (comp {comp() * 1000:.1f}mm)",
              flush=True)
        return False

    pressed = False
    for mode, q_ref, gain, tag in (
            (1, None, 1.0, "press[m1/press-ref g1.0]"),
            (0, q_reset, 1.0, "press[m0/world g1.0]"),
            (1, q_reset, 1.0, "press[m1/reset-ref g1.0]"),
            (1, None, 1.6, "press[m1/press-ref g1.6]")):
        if pressed:
            break
        if q_ref is None:  # press-start readback (captured fresh after any rollback)
            q_ref = scene.blue.data.root_quat_w.clone()
        pressed = press(mode, q_ref, gain, tag)
        if pressed:
            print(f"[solve] {tag}: can flat at {comp() * 1000:.1f}mm compression",
                  flush=True)
        else:
            scene.set_state(snap, all_ids)
            step(60)  # re-settle the rolled-back propped state
    assert pressed, "press failed in every force-frame configuration"
    report("pressed")
    s3 = print_score("P3 pressed flat behind the lip (spring loaded through contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the press"

    # ---------------- phase 4: release — the spring seats the can --------------------------
    wait_settled(600)
    report("released")
    for _ in range(12):  # up to 3 s extra hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s4 = print_score("P4 released — spring pins the can against the lip (stored "
                     "energy, hands off)")
    assert s4 >= s3 - 1e-6, "score decreased across the release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after press+release)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                lv = float(scene.blue.data.root_lin_vel_w[0].norm())
                pv = float(scene.plunger.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: comp={comp() * 1000:.1f}mm "
                      f"seated={bool(scene.seated_geom()[0])} blue_lin={lv:.4f} "
                      f"plg_lin={pv:.4f} decoy={bool(scene.decoy_clear()[0])}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s5 = print_score("P-persist persistence 3.3 s (spring still loaded)")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
