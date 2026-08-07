"""Teleport solution for CellarTowScene (sim_gen task `libero_pick_bbq_sauce_i43`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Two teleports of free bodies through free space,
each with zero velocity, each exactly what a pick-and-carry delivers:
  (a) the probe, from its floor spawn to a hover with its tip just ABOVE the roof slot
      over the sled's socket tube (touching nothing), and later from mid-air to a
      lay-down spot on the open floor after it has been dynamically pulled clear;
  (b) the bottle, from the cradle of the ALREADY-EXTRACTED sled (standing in the open —
      straight up is free space; under the roof this same lift is geometrically
      impossible and is never attempted) to a hover 8 mm above the goal pad.
Every load-bearing interaction happens through contact dynamics and applied wrenches
(a proxy for a hand on the T-handle):

  insert  — a gentle downward + xy-servo wrench guides the vertical probe down through
            the 24 mm roof slot into the 32 mm socket tube; slot edges and tube walls
            do the aligning by contact. Gravity supplies most of the descent.
  tow     — a horizontal servo wrench on the probe (capped ~5 N, plus a small
            downforce that keeps the tip seated and an attitude-hold torque standing
            in for the wrist) drags the peg along the slot; the SLED IS NEVER TOUCHED
            OR TELEPORTED — every newton it feels arrives through the peg-in-eye
            contact coupling, and the bottle rides its cradle by contact alone.
  release — hands off; the sled and cargo coast to rest in the open.
  retract — an upward wrench pulls the peg dynamically out of the tube (the coupling
            is UNMADE through the same contact it was made through); only the freed,
            airborne probe is then teleported aside.
  place   — the bottle is released 8 mm over the pad and lands by gravity.

All wrenches go through `encode_force` with a RUNTIME force-frame probe (some pods
rotate applied wrenches by the body's rotation since its reference orientation): the
insertion is attempted in mode 1 (pre-encoded against the hover readback); if the tip
makes no downward progress the state is rolled back and mode 0 is tried, and the
winning mode is locked in for the tow and retract.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the
scene: 0 -> 0.15 engaged -> 0.40 towed -> 0.55 extracted -> 0.70 retracted -> 1.0
placed), then holds HANDS-OFF for >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import (BOT_H, PAD_T, PROBE_L, ROOF_TOP, TUBE_TOP, TUBE_X, _qapply,
                        encode_force)
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (BOT_H, PAD_T, PROBE_L, ROOF_TOP, TUBE_TOP, TUBE_X, _qapply,
                       encode_force)
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

TIP_HOVER_Z = ROOF_TOP + 0.015  # tip release height (just above the roof, in the clear)
TOW_SPEED = 0.10  # towing waypoint speed (m/s)
TOW_EXIT_X = 0.47  # canopy-local sled x that ends the tow (clear_r is 0.44)
KP_XY, KD_XY = 60.0, 3.0  # probe xy servo (N/m, N s/m)
KQ, KDW = 0.25, 0.010  # attitude-hold torque gains (N m/rad, N m s/rad)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cellar_tow")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
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

    def clear_forces() -> None:
        scene.probe.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def tube_xy_w() -> torch.Tensor:
        """(N,2) world xy of the socket tube center (sled readback)."""
        loc = torch.tensor([TUBE_X, 0.0, 0.0], device=device).expand(n, 3)
        return (scene.sled.data.root_pos_w + _qapply(scene.sled.data.root_quat_w, loc))[:, :2]

    def tip_sled() -> torch.Tensor:
        return scene._local(scene.sled, scene.probe_tip_w())[0]

    def probe_axis() -> torch.Tensor:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return _qapply(scene.probe.data.root_quat_w, ez)

    def hold_torque() -> torch.Tensor:
        """(N,3) world attitude-hold torque driving the shaft to vertical."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        axis = probe_axis()
        tau = KQ * torch.cross(axis, ez, dim=-1) - KDW * scene.probe.data.root_ang_vel_w
        return tau.clamp(-0.5, 0.5)

    def apply_wrench(mode: int, q_ref: torch.Tensor, f_world: torch.Tensor,
                     t_world: torch.Tensor) -> None:
        q_now = scene.probe.data.root_quat_w
        f_arg = encode_force(mode, q_ref, q_now, f_world)
        t_arg = encode_force(mode, q_ref, q_now, t_world)
        scene.probe.set_external_force_and_torque(
            f_arg.view(n, 1, 3), t_arg.view(n, 1, 3), env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        ts = tip_sled()
        sx = float(scene.sled_x_local()[0])
        print(f"[solve] {tag:12s} | tip_sled=({float(ts[0]) - TUBE_X:+.3f},"
              f"{float(ts[1]):+.3f},{float(ts[2]):+.3f}) sled_x={sx:+.3f} "
              f"eng={bool(scene.engaged()[0])} clear={bool(scene.sled_clear()[0])} "
              f"cradled={bool(scene.bottle_in_cradle()[0])} "
              f"on_pad={bool(scene.bottle_on_pad()[0])} "
              f"dis={bool(scene.probe_disengaged()[0])} "
              f"decoy={bool(scene.decoy_ok()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(bodies, max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in bodies):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled([scene.probe, scene.bottle, scene.decoy], 360)
    can_p = (scene.canopy.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.canopy.data.root_quat_w[0, 3]),
                           float(scene.canopy.data.root_quat_w[0, 0]))
    pr_p = (scene.probe.data.root_pos_w - scene.env_origins)[0]
    pd_p = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"canopy=({float(can_p[0]):+.3f},{float(can_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg sled_x={float(scene.sled_x_local()[0]):+.3f} "
          f"probe=({float(pr_p[0]):+.3f},{float(pr_p[1]):+.3f}) "
          f"pad=({float(pd_p[0]):+.3f},{float(pd_p[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene.bottle_in_cradle()[0]), "bottle must start cradled on the sled"
    assert not bool(scene.sled_clear()[0]), "sled must start under the canopy"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (sled parked in the cellar, probe on the floor)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — hover the probe over the slot --------------
    def teleport_hover() -> None:
        """Stand the probe vertically with its tip TIP_HOVER_Z above the ground,
        directly over the socket tube (which the reset keeps under the roof slot).
        Touching nothing; zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = tube_xy_w()
        st[:, 2] = scene.env_origins[:, 2] + TIP_HOVER_Z + PROBE_L / 2
        st[:, 3] = 1.0  # identity: shaft along world z, tip down
        scene.probe.write_root_state_to_sim(st, all_ids)

    teleport_hover()
    report("hover")
    s1 = print_score("P1 probe carried over the roof slot (teleport transport, zero velocity)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"
    snap_hover = scene.get_state(all_ids)
    q_hover = scene.probe.data.root_quat_w.clone()

    # ---------------- phase 2: thread the peg down through the slot into the tube ----------
    def insert(mode: int, q_ref: torch.Tensor, tag: str) -> bool:
        """Guided descent: xy servo toward the tube center, mild downforce (gravity
        does most of it), attitude hold. Contact with slot edges / tube walls does the
        fine alignment. True once the tip sits deep in the bore."""
        z0 = float(scene.probe_tip_w()[0, 2])
        for i in range(1200):
            p = scene.probe.data.root_pos_w
            v = scene.probe.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, :2] = (KP_XY * (tube_xy_w() - p[:, :2]) - KD_XY * v[:, :2]).clamp(-3.0, 3.0)
            # gentle descent: light downforce, damped to ~0.17 m/s terminal velocity
            f[:, 2] = (-0.6 - 8.0 * v[:, 2]).clamp(-2.0, 2.5)
            apply_wrench(mode, q_ref, f, hold_torque())
            env.step(no_action)
            ts = tip_sled()
            if i % 120 == 119:
                print(f"[solve] {tag} @{i + 1}: tip_sled=({float(ts[0]) - TUBE_X:+.3f},"
                      f"{float(ts[1]):+.3f},{float(ts[2]):+.3f}) "
                      f"vert={bool(scene.probe_vertical()[0])}", flush=True)
            # wrong-mode probe: no downward progress after 2 s
            if i == 239 and float(scene.probe_tip_w()[0, 2]) > z0 - 0.010:
                clear_forces()
                print(f"[solve] {tag}: tip made no descent in 2 s — wrong force-frame "
                      f"mode or hung on the roof", flush=True)
                return False
            if bool(scene.engaged()[0]) and float(ts[2]) < 0.050:
                # hold briefly under light downforce so the seat stabilizes
                for _ in range(30):
                    fz = torch.zeros(n, 3, device=device)
                    fz[:, 2] = -0.4
                    apply_wrench(mode, q_ref, fz, hold_torque())
                    env.step(no_action)
                clear_forces()
                return True
        clear_forces()
        print(f"[solve] {tag}: insertion budget exhausted", flush=True)
        return False

    mode_locked = None
    for mode, tag in ((1, "insert[m1/hover-ref]"), (0, "insert[m0/world]")):
        if insert(mode, q_hover, tag):
            mode_locked = mode
            print(f"[solve] {tag}: engaged (mode {mode} locked in)", flush=True)
            break
        scene.set_state(snap_hover, all_ids)
        step(2)
    assert mode_locked is not None, "insertion failed in both force-frame modes"
    report("engaged")
    s2 = print_score("P2 peg threaded through the slot into the socket tube (contact-"
                     "guided descent)")
    assert s2 >= 0.15 - 1e-6 and s2 >= s1 - 1e-6, "engagement credit missing"

    # ---------------- phase 3: TOW — drag the coupled sled out along the slot --------------
    def tow(mode: int, q_ref: torch.Tensor, tag: str) -> bool:
        """Horizontal servo on the probe toward a waypoint sliding along the canopy's
        local +x at TOW_SPEED, plus downforce keeping the tip seated. The sled follows
        purely through the peg-in-eye contact. True once the sled stands clear."""
        q_can = scene.canopy.data.root_quat_w
        dir_x = _qapply(q_can, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[:, :2]
        dir_x = dir_x / dir_x.norm(dim=-1, keepdim=True)
        start = scene.probe.data.root_pos_w[:, :2].clone()
        lost = 0
        for i in range(2400):
            adv = min(TOW_SPEED * (i / 120.0), 0.90)
            target = start + dir_x * adv
            p = scene.probe.data.root_pos_w
            v = scene.probe.data.root_lin_vel_w
            f = torch.zeros(n, 3, device=device)
            f[:, :2] = (KP_XY * (target - p[:, :2]) - KD_XY * v[:, :2]).clamp(-5.0, 5.0)
            f[:, 2] = -1.2
            apply_wrench(mode, q_ref, f, hold_torque())
            env.step(no_action)
            scene.score()  # keep the tow-fraction latch current while engaged
            sx = float(scene.sled_x_local()[0])
            if i % 120 == 119:
                ts = tip_sled()
                print(f"[solve] {tag} @{i + 1}: sled_x={sx:+.3f} adv={adv:.2f} "
                      f"tip_z={float(ts[2]):+.3f} eng={bool(scene.engaged()[0])} "
                      f"cradled={bool(scene.bottle_in_cradle()[0])}", flush=True)
            lost = 0 if bool(scene.engaged()[0]) else lost + 1
            if lost > 90:
                clear_forces()
                print(f"[solve] {tag}: coupling lost for 90 steps at sled_x={sx:+.3f}",
                      flush=True)
                return False
            if not bool(scene.bottle_in_cradle()[0]):
                clear_forces()
                print(f"[solve] {tag}: cargo left the cradle mid-tow", flush=True)
                return False
            if bool(scene.sled_clear()[0]) and sx >= TOW_EXIT_X:
                clear_forces()
                return True
        clear_forces()
        print(f"[solve] {tag}: tow budget exhausted (sled_x="
              f"{float(scene.sled_x_local()[0]):+.3f})", flush=True)
        return False

    snap_eng = scene.get_state(all_ids)
    towed = tow(mode_locked, q_hover, f"tow[m{mode_locked}]")
    if not towed:  # one retry from the engaged snapshot with the other frame mode
        scene.set_state(snap_eng, all_ids)
        step(2)
        alt = 1 - mode_locked
        towed = tow(alt, q_hover, f"tow[m{alt}/retry]")
        if towed:
            mode_locked = alt
    assert towed, "tow failed in both force-frame modes"
    wait_settled([scene.sled, scene.bottle], 300)
    report("extracted")
    assert bool(scene.sled_clear()[0]), "sled not clear after tow"
    assert bool(scene.bottle_in_cradle()[0]), "cargo must still be cradled after tow"
    s3 = print_score("P3 sled towed out of the cellar through the peg coupling, cargo "
                     "intact (hands-off coast to rest)")
    assert s3 >= 0.55 - 1e-6 and s3 >= s2 - 1e-6, "extraction credit missing"

    # ---------------- phase 4: retract the probe (unmake the coupling), lay it aside -------
    def retract(mode: int, q_ref: torch.Tensor) -> bool:
        for i in range(600):
            f = torch.zeros(n, 3, device=device)
            f[:, 2] = 1.7  # ~2.2x weight: a steady upward pull
            apply_wrench(mode, q_ref, f, hold_torque())
            env.step(no_action)
            tip_z = float(scene.probe_tip_w()[0, 2] - scene.env_origins[0, 2])
            if tip_z > TUBE_TOP + 0.06:
                clear_forces()
                return True
        clear_forces()
        return False

    assert retract(mode_locked, q_hover), "probe failed to pull free of the tube"
    # the freed, airborne probe is now teleported aside (transport only) and laid down
    rest = torch.zeros(n, 13, device=device)
    rest[:, 0:2] = scene.canopy.data.root_pos_w[:, :2] \
        + torch.tensor([-0.10, 0.55], device=device).expand(n, 2)
    rest[:, 2] = scene.env_origins[:, 2] + 0.05
    rest[:, 3], rest[:, 4] = math.sqrt(0.5), math.sqrt(0.5)  # lying flat (roll 90)
    scene.probe.write_root_state_to_sim(rest, all_ids)
    wait_settled([scene.probe, scene.sled], 300)
    report("retracted")
    assert bool(scene.probe_disengaged()[0]), "probe must end disengaged"
    s4 = print_score("P4 coupling unmade — probe pulled free through contact and laid "
                     "aside (teleport transport of the freed body)")
    assert s4 >= 0.70 - 1e-6 and s4 >= s3 - 1e-6, "retraction credit missing"

    # ---------------- phase 5: TRANSPORT — lift the bottle from the OPEN cradle to the pad -
    # The sled stands in the open: straight up from the cradle is free space (under the
    # roof this lift is geometrically impossible — smoke proves the rubric rejects any
    # attempt to shortcut it).
    place = torch.zeros(n, 13, device=device)
    place[:, 0:2] = scene.pad.data.root_pos_w[:, :2]
    place[:, 2] = scene.pad.data.root_pos_w[:, 2] + PAD_T / 2 + BOT_H / 2 + 0.008
    place[:, 3] = 1.0
    scene.bottle.write_root_state_to_sim(place, all_ids)
    for _ in range(10):  # gravity landing + settle
        step(30)
        if bool(scene.success()[0]):
            break
    report("placed")
    s5 = print_score("P5 bottle released 8 mm over the pad — lands by gravity")
    assert s5 >= s4 - 1e-6, "score decreased across the placement"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"on_pad={bool(scene.bottle_on_pad()[0])} "
                      f"bot_set={bool(scene.settled(scene.bottle)[0])} "
                      f"sled_set={bool(scene.settled(scene.sled)[0])} "
                      f"probe_set={bool(scene.settled(scene.probe)[0])} "
                      f"dis={bool(scene.probe_disengaged()[0])} "
                      f"decoy={bool(scene.decoy_ok()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P-persist persistence 3.3 s (bottle standing on the pad)")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
