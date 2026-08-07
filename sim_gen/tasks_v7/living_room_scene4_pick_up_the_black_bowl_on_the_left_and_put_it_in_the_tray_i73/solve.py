"""Teleport solution for RollerFreightScene (sim_gen task `living_room_scene4_pick_up_
the_black_bowl_on_the_left_and_put_it_in_the_tray_i73`) — the task's legitimacy
certificate.

Teleportation handles TRANSPORT ONLY: each free roller is carried from its rack chock
to a spot 3 mm above the open channel floor and RELEASED with zero velocity (exactly
what a pick-and-carry of a 30 mm steel cylinder delivers); if the migrating bed ever
needs a re-feed, only a roller that has fully EXITED behind the slab (free on the open
floor, touching nothing but the floor) is carried forward the same way. The slab and
the bowl are NEVER teleported. Every load-bearing interaction happens through contact
dynamics and one applied force (a proxy for a fingertip on the slab's rear face at CoM
height):

  stage — rollers dropped across the channel settle onto the gritty floor by gravity.
  push  — a horizontal velocity-servo force on the slab, HARD-CAPPED at the scene's
          push budget (cfg.push_cap, ~40 N), aimed along the channel axis, no applied
          torque (a fingertip cannot twist the freight; the walls do the aligning).
          The slab launches off the slick plinth, its nose drops one roller diameter
          onto the staged bed, and from there the rollers carry it — the SAME capped
          push that the smoke battery proves CANNOT move the slab on the bare floor.
          Every roller motion is contact-driven: the bed migrates rearward at half
          slab speed as pure rolling kinematics.
  dock  — the push holds the slab against the yellow end stop; forces are cleared and
          the freight coasts to rest riding its bed, bowl upright aboard.

The push force goes through `encode_force` with a RUNTIME force-frame probe (some pods
rotate applied wrenches by the body's rotation since its reference orientation): the
launch is attempted in mode 1 (pre-encoded against the settle readback); if the slab
makes no forward progress on the SLICK plinth within 2 s the state is rolled back and
mode 0 is tried, and the winning mode is locked in.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the
scene: 0 -> 0.20 bed staged -> 0.45 riding -> 0.70 mid-channel -> 1.0 docked), then
holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and prints
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
    from .scene import (FLOOR_T, ROLL_R, SLAB_L, X_PE, X_STOP, _qapply, encode_force)
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (FLOOR_T, ROLL_R, SLAB_L, X_PE, X_STOP, _qapply, encode_force)
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

STAGE_X = (0.000, 0.100, 0.200)  # roller drop spots (fixture-local x; edge is at -0.05)
V_DES = 0.080  # push waypoint speed (m/s)
KP_V = 800.0  # velocity-servo gain (N per m/s of error)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roller_freight")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    cap = float(scene.cfg.push_cap) - 1.0  # stay strictly inside the budget
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
        scene.slab.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def chan_dir() -> torch.Tensor:
        """(N,3) world unit vector of the channel's +x (delivery direction)."""
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        d = _qapply(scene.fixture.data.root_quat_w, ex)
        d = d / d.norm(dim=-1, keepdim=True)
        return d

    def fix_local(pos_w: torch.Tensor) -> torch.Tensor:
        return scene._local(scene.fixture, pos_w)

    def apply_push(mode: int, q_ref: torch.Tensor, f_world: torch.Tensor) -> None:
        q_now = scene.slab.data.root_quat_w
        f_arg = encode_force(mode, q_ref, q_now, f_world)
        scene.slab.set_external_force_and_torque(
            f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        fx = float(scene.slab_front_x()[0])
        sl = scene.slab_local()[0]
        rl = [fix_local(r.data.root_pos_w)[0] for r in scene.rollers]
        rs = " ".join(f"({float(p[0]):+.3f},{float(p[1]):+.3f})" for p in rl)
        print(f"[solve] {tag:12s} | front_x={fx:+.3f} slab=({float(sl[0]):+.3f},"
              f"{float(sl[1]):+.3f},z{float(sl[2]):.3f}) rollers={rs} "
              f"staged={bool(scene.rollers_staged()[0])} "
              f"riding={bool(scene.slab_riding()[0])} "
              f"under={bool(scene.roller_under()[0])} "
              f"aboard={bool(scene.bowl_aboard()[0])} "
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
    wait_settled([scene.slab, scene.bowl, *scene.rollers], 360)
    f_p = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.fixture.data.root_quat_w[0, 3]),
                           float(scene.fixture.data.root_quat_w[0, 0]))
    bl = scene._local(scene.slab, scene.bowl.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"fixture=({float(f_p[0]):+.3f},{float(f_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.1f}deg front_x={float(scene.slab_front_x()[0]):+.3f} "
          f"bowl_slab=({float(bl[0]):+.3f},{float(bl[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene.bowl_aboard()[0]), "bowl must start aboard the slab"
    assert not bool(scene.roller_under()[0]), "no roller starts under the slab"
    assert not bool(scene.rollers_staged()[0]), "rollers start chocked in the rack"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (slab on the plinth, rollers in the rack)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — stage the roller bed -----------------------
    def carry_roller(idx: int, x_loc: float, verify: bool = True) -> None:
        """Carry roller `idx` from wherever it lies to 1.5 mm above the channel floor
        at fixture-local x, axis across the channel; release with zero velocity.
        With verify=True (pre-push staging, the roller is guaranteed free) check it
        parked where dropped (a free capsule can roll on landing) and re-carry up to
        3 attempts — a repeated pick-and-place of a free body stays transport-only.
        Refeeds during the push use verify=False: one drop, no re-carry (the roller
        may wedge under the advancing nose, where a re-grab would not be free)."""
        loc = torch.tensor([x_loc, 0.0, FLOOR_T + ROLL_R + 0.0015],
                           device=device).expand(n, 3)
        for attempt in range(3 if verify else 1):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = scene.fixture.data.root_pos_w + _qapply(
                scene.fixture.data.root_quat_w, loc)
            st[:, 3:7] = scene.fixture.data.root_quat_w  # authored axis Y = across
            scene.rollers[idx].write_root_state_to_sim(st, all_ids)
            step(30)  # gravity landing (1.5 mm fall + impact damping)
            if not verify:
                return
            rl = fix_local(scene.rollers[idx].data.root_pos_w)[0]
            if abs(float(rl[0]) - x_loc) <= 0.020 and abs(float(rl[1])) <= 0.025:
                return
            if attempt < 2:
                print(f"[solve] carry_roller {idx}: landed at ({float(rl[0]):+.3f},"
                      f"{float(rl[1]):+.3f}) vs target {x_loc:+.3f} — re-carrying",
                      flush=True)

    def roller_x(i: int) -> float:
        return float(fix_local(scene.rollers[i].data.root_pos_w)[0, 0])

    for i, x_loc in enumerate(STAGE_X):
        carry_roller(i, x_loc)
    # Bed audit: GPU capsule-on-box contact injects a constant ~0.04 m/s phantom
    # creep along the channel into a FREE resting capsule that no damping kills (a
    # solver limit cycle) — the bed never sits perfectly still, so audit POSITION
    # only: re-carry strays that left their slot band, then push IMMEDIATELY (the
    # longer the dawdle, the further the creep; mid-ride starvation is covered by
    # the refeed logic).
    for _round in range(2):
        off = [abs(roller_x(i) - STAGE_X[i]) for i in range(3)]
        if max(off) <= 0.030:
            break
        print(f"[solve] bed audit: off={[f'{o:.3f}' for o in off]} — re-carrying "
              f"strays", flush=True)
        for i in range(3):
            if off[i] > 0.030:
                carry_roller(i, STAGE_X[i])
    report("staged")
    assert bool(scene.rollers_staged()[0]), "roller bed failed to stage"
    s1 = print_score("P1 rollers carried from the rack and released across the channel "
                     "floor (teleport transport, zero velocity)")
    assert s1 >= 0.20 - 1e-6 and s1 >= s0 - 1e-6, "staging credit missing"
    snap_staged = scene.get_state(all_ids)
    q_ref = scene.slab.data.root_quat_w.clone()

    # ---------------- phase 2: PUSH — capped fingertip servo along the channel -------------
    def push(mode: int, tag: str, budget: int = 3000) -> bool:
        """Velocity-servo push (hard cap = the scene's budget) along the channel axis.
        Launches the slab off the slick plinth onto the staged bed and rolls it to the
        end stop. Wrong-frame probe: no progress on the SLICK plinth within 2 s.
        Re-feeds an EXITED roller (free on the open floor behind the slab) at most
        twice if the bed starves. True once the nose is pressed into the dock window."""
        x0 = float(scene.slab_front_x()[0])
        stall, refeeds = 0, 0
        for i in range(budget):
            d = chan_dir()
            v_along = (scene.slab.data.root_lin_vel_w * d).sum(-1)
            f_mag = (KP_V * (V_DES - v_along)).clamp(0.0, cap)
            apply_push(mode, q_ref, d * f_mag.unsqueeze(-1))
            env.step(no_action)
            scene.score()  # keep the latches current while riding
            fx = float(scene.slab_front_x()[0])
            if i % 120 == 119:
                sl = scene.slab_local()[0]
                print(f"[solve] {tag} @{i + 1}: front_x={fx:+.3f} "
                      f"v={float(v_along[0]):+.3f} F={float(f_mag[0]):.1f} "
                      f"z={float(sl[2]):.3f} riding={bool(scene.slab_riding()[0])} "
                      f"under={bool(scene.roller_under()[0])} "
                      f"aboard={bool(scene.bowl_aboard()[0])}", flush=True)
            # wrong-mode probe: the plinth is slick — no launch in 2 s means bad frame
            if i == 239 and fx < x0 + 0.010:
                clear_forces()
                print(f"[solve] {tag}: no launch off the slick plinth in 2 s — wrong "
                      f"force-frame mode", flush=True)
                return False
            if not bool(scene.bowl_aboard()[0]):
                clear_forces()
                print(f"[solve] {tag}: cargo bowl left the deck at front_x={fx:+.3f}",
                      flush=True)
                return False
            if fx >= X_STOP - 0.012 or (fx >= X_STOP - float(scene.cfg.dock_win) + 0.004
                                        and abs(float(v_along[0])) < 0.010 and i > 400):
                clear_forces()
                return True
            # bed starvation: creeping/stalled off-plinth with force at the cap
            stall = stall + 1 if (abs(float(v_along[0])) < 0.012
                                  and float(f_mag[0]) > cap - 0.5 and i > 300) else 0
            if stall > 360:
                # a roller is free iff IT has fully exited behind the slab rear
                # (per-roller test — touching nothing but the open floor)
                free = None
                rear_x = fx - SLAB_L  # slab rear in fixture x (aligned by the walls)
                for j, r in enumerate(scene.rollers):
                    rl = fix_local(r.data.root_pos_w)[0]
                    if float(rl[0]) < rear_x - 0.03:
                        free = j
                if free is None or refeeds >= 2:
                    clear_forces()
                    print(f"[solve] {tag}: stalled at front_x={fx:+.3f} with no free "
                          f"roller to re-feed", flush=True)
                    return False
                refeeds += 1
                stall = 0
                print(f"[solve] {tag}: re-feeding exited roller {free} ahead of the "
                      f"nose (transport of a free body)", flush=True)
                carry_roller(free, min(fx + 0.045, X_STOP - 0.035), verify=False)
        clear_forces()
        print(f"[solve] {tag}: push budget exhausted at front_x="
              f"{float(scene.slab_front_x()[0]):+.3f}", flush=True)
        return False

    mode_locked = None
    for mode in (1, 0):
        if push(mode, f"push[m{mode}]"):
            mode_locked = mode
            break
        scene.set_state(snap_staged, all_ids)
        step(2)
    assert mode_locked is not None, "push failed in both force-frame modes"
    report("docked")
    s2 = print_score("P2 slab launched off the plinth, rode the migrating roller bed "
                     "down the channel, pressed into the dock (capped fingertip push)")
    assert s2 >= 0.70 - 1e-6 and s2 >= s1 - 1e-6, "riding/half credit missing"

    # ---------------- phase 3: hands off — coast to rest at the dock -----------------------
    wait_settled([scene.slab, scene.bowl, *scene.rollers], 360)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after docking)", flush=True)
        os._exit(1)
    s3 = print_score("P3 freight at rest against the end stop, riding its bed, bowl "
                     "upright aboard")
    assert s3 >= 1.0 - 1e-6 and s3 >= s2 - 1e-6, "success credit missing"

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"front_x={float(scene.slab_front_x()[0]):+.3f} "
                      f"riding={bool(scene.slab_riding()[0])} "
                      f"under={bool(scene.roller_under()[0])} "
                      f"aligned={bool(scene.slab_aligned()[0])} "
                      f"aboard={bool(scene.bowl_aboard()[0])} "
                      f"slab_set={bool(scene.settled(scene.slab)[0])} "
                      f"bowl_set={bool(scene.settled(scene.bowl)[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s (freight parked at the dock)")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
