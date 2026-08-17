"""Teleport solution for BasculeKeepScene (sim_gen task `libero_pick_ketchup_i411`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY (each one a plain pick-and-carry to an open,
reachable pose, written with zero velocity):
  (a) block A: lifted out of the open-top ballast tray, set on the open ground depot;
  (b) block B: same, to the mirrored depot — the tray sits on the raised tail under
      open sky, both are jaw-sized 5 cm steel blocks;
  (c) the RED ketchup: stood upright onto the DEPLOYED deck near the hinge (open sky
      above the bridge — no roof exists over it), zero velocity.
No teleport ever enters the keep: it is roofed, and the scene config asserts that the
raised machine leaves no bottle-sized aperture anyway.

Every load-bearing interaction happens through gravity and contact dynamics:

  unballast — after the blocks are REMOVED the machine actuates ITSELF: the leaf's
              authored tip-heavy CoM swings the bridge down on its trunnion until the
              tongue rests on the doorway sill. No force is ever applied to the
              mechanism. (After removing only block A the solve explicitly waits and
              ASSERTS the bridge is still raised — one block alone holds it.)
  push      — the ketchup is pushed along the curb channel, over the tongue, through
              the doorway onto the keep floor by an emulated hand force: a
              velocity-capped horizontal force at the bottle plus the low-push-point
              torque tau = r x F (push point ~5 cm below the CoM) so the bottle
              slides without tipping. The wrench goes through `encode_force` with a
              RUNTIME force-frame probe (mode 1 pre-encoded against the push-start
              readback; if the bottle does not advance the state is rolled back and
              mode 0 is tried).
  release   — forces cleared; the bottle settles on the keep floor, the bridge stays
              deployed on its sill, and success() holds hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the
scene: 0 -> 0.10 first block out -> 0.20 tray empty -> 0.45 bridge deployed -> 0.70
crossed -> 1.0 success), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import (BLOCK, BOT_L, DECK_T, PIV_Z, THETA_UP, WALL_X1, _qapply,
                        encode_force)
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (BLOCK, BOT_L, DECK_T, PIV_Z, THETA_UP, WALL_X1, _qapply,
                       encode_force)
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DEPOT = {"a": (-0.42, 0.42), "b": (-0.42, -0.42)}  # rig-frame ground depots (clear of
#     the bottle slots at (-0.52, +/-0.20+/-0.05), the raised heel (~-0.20) and the keep)
PLACE_X = -0.020  # rig-frame x where the ketchup is stood on the deployed deck
PUSH_STOP_X = 0.300  # stop pushing once the bottle center passes this rig-frame x
V_DES = 0.080  # push speed target (m/s) — slow, quasi-static
KP_X, F_MAX = 12.0, 1.5  # velocity-servo gain / force cap (tip limit ~2 N at CoM)
R_PUSH = -0.050  # push point 5 cm below the bottle CoM (body frame)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bascule_keep")().build(num_envs=args.num_envs, device=device)
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
            scene.score()  # keep the latches current

    def clear_forces() -> None:
        scene.ketchup.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    def report(tag: str) -> None:
        kloc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | tilt={float(scene.leaf_tilt_deg()[0]):+.2f}deg "
              f"hinge={bool(scene.hinge_intact()[0])} "
              f"down={bool(scene.bridge_down()[0])} "
              f"tray_a={bool(scene.in_tray(scene.block_a)[0])} "
              f"tray_b={bool(scene.in_tray(scene.block_b)[0])} "
              f"k_rig=({float(kloc[0]):+.3f},{float(kloc[1]):+.3f},"
              f"{float(kloc[2]):+.3f}) "
              f"keep={bool(scene.in_keep(scene.ketchup)[0])} "
              f"mus_out={bool(scene.mustard_out()[0])} "
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

    def rig_place(loc_xyz) -> torch.Tensor:
        """World position of a rig-frame point (N,3)."""
        loc = torch.tensor(loc_xyz, device=device).expand(n, 3)
        return scene.rig.data.root_pos_w + _qapply(scene.rig.data.root_quat_w, loc)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled([scene.leaf, scene.block_a, scene.block_b, scene.ketchup,
                  scene.mustard], 360)
    rig_p = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.rig.data.root_quat_w[0, 3]),
                           float(scene.rig.data.root_quat_w[0, 0]))
    k_loc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
    m_loc = scene._rig_local(scene.mustard.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rig_p[0]):+.3f},{float(rig_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.1f}deg "
          f"tilt={float(scene.leaf_tilt_deg()[0]):+.2f}deg "
          f"ketchup=({float(k_loc[0]):+.3f},{float(k_loc[1]):+.3f}) "
          f"mustard=({float(m_loc[0]):+.3f},{float(m_loc[1]):+.3f})", flush=True)
    report("reset")
    tilt0 = float(scene.leaf_tilt_deg()[0])
    up_deg = math.degrees(THETA_UP)
    assert up_deg - 2.0 <= tilt0 <= up_deg + 1.0, \
        f"bridge must start raised on its heel, tilt={tilt0}"
    assert bool(scene.in_tray(scene.block_a)[0]), "block A must start in the tray"
    assert bool(scene.in_tray(scene.block_b)[0]), "block B must start in the tray"
    assert bool(scene.hinge_intact()[0]), "hinge must start intact"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (bridge raised on ballast, bottles on the ground)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — lift block A out to the depot --------------
    def teleport_block(body, key: str) -> None:
        """Lift one counterweight out of the open-top tray and set it on the open
        ground depot (plain pick-and-carry, zero velocity)."""
        dx, dy = DEPOT[key]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rig_place([dx, dy, BLOCK[2] / 2 + 0.003])
        st[:, 3:7] = scene.rig.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)

    teleport_block(scene.block_a, "a")
    wait_settled([scene.block_a, scene.leaf], 420)
    report("one-out")
    assert not bool(scene.in_tray(scene.block_a)[0]), "block A must be out of the tray"
    tilt1 = float(scene.leaf_tilt_deg()[0])
    assert tilt1 >= scene.cfg.raised_min_deg, \
        f"ONE remaining block must still hold the bridge raised, tilt={tilt1}"
    s1 = print_score("P1 block A lifted out (teleport transport) — bridge STILL "
                     "raised: one counterweight alone holds it")
    assert s1 >= 0.10 - 1e-6 and s1 >= s0 - 1e-6, "first-block credit missing"

    # ---------------- phase 2: unballast — gravity deploys the bridge on its own -----------
    teleport_block(scene.block_b, "b")
    step(10)
    assert bool(scene.tray_empty()[0]), "tray must be empty after both lifts"
    # hands off: the tip-heavy leaf swings itself down onto the sill
    for i in range(20):
        step(30)
        if bool(scene.bridge_down()[0]) and bool(scene.settled(scene.leaf)[0]):
            break
        if i % 4 == 3:
            print(f"[solve] deploying @+{(i + 1) * 30}: "
                  f"tilt={float(scene.leaf_tilt_deg()[0]):+.2f}deg", flush=True)
    report("deployed")
    assert bool(scene.bridge_down()[0]), \
        f"bridge must deploy by itself, tilt={float(scene.leaf_tilt_deg()[0]):+.2f}"
    assert bool(scene.hinge_intact()[0]), "hinge must survive the deploy"
    s2 = print_score("P2 block B lifted out — tray empty, gravity swung the bridge "
                     "level onto the doorway sill (no force ever applied to it)")
    assert s2 >= 0.45 - 1e-6 and s2 >= s1 - 1e-6, "deploy credit missing"

    # ---------------- phase 3: TRANSPORT ONLY — stand the ketchup on the deck --------------
    def teleport_ketchup_on_deck() -> None:
        """Stand the RED bottle upright on the deployed deck near the hinge (open sky
        above the bridge; a plain pick-and-carry). Zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = rig_place([PLACE_X, 0.0, PIV_Z + DECK_T / 2 + BOT_L / 2 + 0.004])
        st[:, 3:7] = scene.rig.data.root_quat_w
        scene.ketchup.write_root_state_to_sim(st, all_ids)

    teleport_ketchup_on_deck()
    wait_settled([scene.ketchup], 240)
    report("staged")
    kz = float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 2])
    assert kz > PIV_Z, f"ketchup must stand on the deck, z={kz}"
    assert not bool(scene.past_doorway(scene.ketchup)[0]), "not through yet"
    s3 = print_score("P3 ketchup stood on the deployed deck (teleport transport, "
                     "zero velocity)")
    assert s3 >= s2 - 1e-6

    # ---------------- phase 4: PUSH the bottle across the bridge through the doorway -------
    def push_wrench(mode: int, q_ref: torch.Tensor) -> None:
        """One step of the emulated hand: rig-frame +x velocity-capped force at the
        bottle + the low-push-point torque tau = r x F, world-encoded."""
        q_rig = scene.rig.data.root_quat_w
        q_bot = scene.ketchup.data.root_quat_w
        loc = scene._rig_local(scene.ketchup.data.root_pos_w)
        vel_r = _qapply_inv(q_rig, scene.ketchup.data.root_lin_vel_w)
        fx = (KP_X * (V_DES - vel_r[:, 0])).clamp(0.0, F_MAX)
        fy = (-10.0 * loc[:, 1] - 2.0 * vel_r[:, 1]).clamp(-0.5, 0.5)
        f_r = torch.stack([fx, fy, torch.zeros_like(fx)], dim=-1)
        f_w = _qapply(q_rig, f_r)
        r_w = _qapply(q_bot, torch.tensor([0.0, 0.0, R_PUSH],
                                          device=device).expand(n, 3))
        t_w = torch.cross(r_w, f_w, dim=-1)
        f_arg = encode_force(mode, q_ref, q_bot, f_w)
        t_arg = encode_force(mode, q_ref, q_bot, t_w)
        scene.ketchup.set_external_force_and_torque(
            f_arg.view(n, 1, 3), t_arg.view(n, 1, 3), env_ids=all_ids, is_global=True)

    def _qapply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        qi = q.clone()
        qi[:, 1:] = -qi[:, 1:]
        return _qapply(qi, v)

    def push_across(mode: int, q_ref: torch.Tensor, tag: str) -> bool:
        """Servo the bottle down the channel, over the tongue, through the doorway.
        True once its center passes PUSH_STOP_X inside the keep."""
        x_start = float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 0])
        for i in range(1500):
            push_wrench(mode, q_ref)
            env.step(no_action)
            scene.score()
            loc = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
            if i % 150 == 149:
                up = float(_qapply(scene.ketchup.data.root_quat_w,
                                   torch.tensor([0.0, 0.0, 1.0], device=device)
                                   .expand(n, 3))[0, 2])
                print(f"[solve] {tag} @{i + 1}: x={float(loc[0]):+.3f} "
                      f"y={float(loc[1]):+.3f} up={up:+.3f} "
                      f"crossed={bool(scene._crossed[0])}", flush=True)
            # wrong-mode probe: no advance in the first 240 steps
            if i == 239 and float(loc[0]) < x_start + 0.010:
                clear_forces()
                print(f"[solve] {tag}: bottle did not advance — wrong force-frame "
                      f"mode?", flush=True)
                return False
            # toppled? (axis off vertical) — treat as mode failure, roll back
            up = float(_qapply(scene.ketchup.data.root_quat_w,
                               torch.tensor([0.0, 0.0, 1.0], device=device)
                               .expand(n, 3))[0, 2])
            if up < 0.85:
                clear_forces()
                print(f"[solve] {tag}: bottle tipped (up={up:+.2f})", flush=True)
                return False
            if float(loc[0]) >= PUSH_STOP_X:
                clear_forces()
                return True
        clear_forces()
        print(f"[solve] {tag}: push budget exhausted "
              f"(x={float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 0]):+.3f})",
              flush=True)
        return False

    snap_staged = scene.get_state(all_ids)
    q_push = scene.ketchup.data.root_quat_w.clone()
    crossed = push_across(1, q_push, "push[m1/ref]")
    if not crossed:  # rollback, other force-frame mode
        scene.set_state(snap_staged, all_ids)
        step(2)
        crossed = push_across(0, q_push, "push[m0/world]")
    assert crossed, "push failed to cross the doorway in both force-frame modes"
    report("crossed")
    s4 = print_score("P4 ketchup pushed along the channel, over the tongue, through "
                     "the doorway onto the keep floor")
    assert s4 >= 0.70 - 1e-6 and s4 >= s3 - 1e-6, "crossing credit missing"

    # ---------------- phase 5: release — everything settles, success ----------------------
    wait_settled([scene.ketchup, scene.leaf], 600)
    report("settled")
    assert bool(scene.in_keep(scene.ketchup)[0]), "ketchup must rest inside the keep"
    assert bool(scene.bridge_down()[0]), "bridge must remain deployed"
    assert bool(scene.tray_empty()[0]), "tray must remain empty"
    s5 = print_score("P5 hands off — bottle at rest inside the keep, bridge deployed "
                     "on its sill, tray empty")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release)", flush=True)
        os._exit(1)
    assert s5 >= 1.0 - 1e-6, "success must score 1.0"

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        env.step(no_action)
        scene.score()
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"keep={bool(scene.in_keep(scene.ketchup)[0])} "
                      f"k_set={bool(scene.settled(scene.ketchup)[0])} "
                      f"down={bool(scene.bridge_down()[0])} "
                      f"l_set={bool(scene.settled(scene.leaf)[0])} "
                      f"tray_empty={bool(scene.tray_empty()[0])} "
                      f"mus_out={bool(scene.mustard_out()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P-persist persistence 3.3 s (bottle in the keep, bridge "
                     "deployed)")
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
