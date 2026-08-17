"""Teleport solution for BallastGateScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i426`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each carton is teleported exactly where a
one-armed pick-and-place would carry it — the JUICE from its floor spawn onto the
gate's ballast tray (P1) and later off the tray to the far floor (P3), the MILK from
its floor spawn to a held HOVER outside the open doorway (P2 start). Everything the
rubric reads happens through contact dynamics:

  sink  — the juice's dead weight (4.4 N) on the tray overwhelms the 2.4-4.2 N
          spring and the gate slides down its prismatic joint to the bottom stop,
          HANDS-FREE. Nothing pushes the gate; the placed ballast does it.
  carry — the milk crosses the doorway under an applied force that emulates the
          holding hand: exact-gravity feedforward + a z position servo, a gentle
          horizontal velocity servo (<= a few N), recomputed every step in the
          VAULT frame and applied in the CARTON BODY frame (frame-drag safe: the
          pod's global-wrench reference orientation is captured at the FIRST
          application and goes stale across resets). It passes OVER the sunken
          gate's top edge and UNDER the roof — the corridor the ballast opened.
  lay   — over the crib the carry lowers and releases; gravity seats the carton
          on the crib pad. Hands off.
  seal  — with the tray emptied (P3 transport), the spring alone drives the gate
          back up to its upper stop, re-sealing the vault behind the milk.

The retry ladder escalates the carry servo (faster / stiffer / higher caps) and
every retry rolls back to the pre-carry snapshot first, so no failed attempt's
debris pollutes the next.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0.25 gate sunk open by the ballast -> 0.65 milk resting in the crib -> 1.0
gate re-sealed, everything settled), then holds HANDS-OFF for >= 3.3 simulated
seconds after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only
if it still holds.

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
    from .scene import (  # noqa: F401
        BASE_H, CRIB_CX, CRIB_PAD_T, GATE_PLANE_X, GATE_TRAVEL, GATE_Z0, MILK_W,
        TRAY_CX, TRAY_CY, TRAY_CZ, TRAY_T, _qapply, _qinv, _qmul, _qx, _qy,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BASE_H, CRIB_CX, CRIB_PAD_T, GATE_PLANE_X, GATE_TRAVEL, GATE_Z0, MILK_W,
        TRAY_CX, TRAY_CY, TRAY_CZ, TRAY_T, _qapply, _qinv, _qmul, _qx, _qy,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

TRAY_TOP_CLOSED = GATE_Z0 + TRAY_CZ + TRAY_T / 2  # 0.535 (vault frame, gate closed)
BALLAST_X = GATE_PLANE_X + TRAY_CX  # ballast set-down, vault frame
BALLAST_Y = TRAY_CY
CARRY_Z = 0.390  # carry height: bottom 0.360 clears the 0.340 curb and 0.316 rim
CARRY_X0 = 0.460  # held hover outside the doorway (vault frame)
LOWER_Z = 0.310  # release height over the crib pad (top 0.266)
MILK_MG = 0.35 * 9.81
DT = 1.0 / 120.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_gate")().build(num_envs=args.num_envs, device=device)
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

    def gate_q() -> float:
        return float(scene.gate_q()[0])

    def milk_loc() -> torch.Tensor:
        return scene.vault_local(scene.milk.data.root_pos_w)[0]

    def clear_force() -> None:
        scene.milk.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        ml = milk_loc()
        ju = scene.vault_local(scene.juice.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | q={gate_q():+.4f} "
              f"milk_v=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"juice_v=({float(ju[0]):+.3f},{float(ju[1]):+.3f},{float(ju[2]):+.3f}) "
              f"in_crib={bool(scene.milk_in_crib()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"open={bool(scene.gate_open_deep()[0])} "
              f"decoy_out={bool(scene.decoy_out()[0])} "
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
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.juice)[0]) \
                    and bool(scene.settled(scene.gate)[0]):
                break

    def place(body, local_pos, quat_local) -> None:
        """TRANSPORT teleport: set a body at rest at a vault-frame pose (what a
        pick-and-carry of a free-standing 60 mm carton delivers)."""
        q_vault = scene.vault.data.root_quat_w
        local = torch.tensor(local_pos, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.vault.data.root_pos_w + _qapply(q_vault, local)
        st[:, 3:7] = _qmul(q_vault, quat_local)
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.vault.data.root_quat_w[0, 3]),
                           float(scene.vault.data.root_quat_w[0, 0]))
    ml = milk_loc()
    ju = scene.vault_local(scene.juice.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg "
          f"milk_v=({float(ml[0]):+.3f},{float(ml[1]):+.3f}) "
          f"juice_v=({float(ju[0]):+.3f},{float(ju[1]):+.3f}) "
          f"q={gate_q():+.4f}", flush=True)
    report("reset")
    assert gate_q() > -0.01, f"gate must spawn pressed shut, q={gate_q():+.4f}"
    assert not bool(scene.milk_in_crib()[0]), "milk must start outside the crib"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (gate sealed, cartons on the floor)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT the ballast onto the tray; the gate sinks ---------
    half_pi = torch.full((n,), math.pi / 2, device=device)
    opened = False
    for attempt in range(2):
        # lay the juice LYING (long axis along vault y) on the tray plate, 6 mm
        # settle drop — the end state of picking it up and laying it in the tray
        place(scene.juice,
              (BALLAST_X, BALLAST_Y, TRAY_TOP_CLOSED + MILK_W / 2 + 0.006),
              _qx(half_pi))
        for _ in range(30):  # up to 7.5 s: sink terminal speed ~0.14 m/s over 225 mm
            step(30)
            if bool(scene.gate_open_deep()[0]) and bool(scene.settled(scene.gate)[0]) \
                    and bool(scene.settled(scene.juice)[0]):
                break
        if bool(scene.gate_open_deep()[0]):
            opened = True
            break
        print(f"[solve] ballast attempt {attempt}: gate only at q={gate_q():+.4f} — "
              f"re-placing", flush=True)
    assert opened, f"ballast failed to sink the gate, q={gate_q():+.4f}"
    report("ballasted")
    s1 = print_score("P1 juice laid on the ballast tray (teleport transport) — its "
                     "weight sank the gate fully open, hands-free")
    assert s1 >= s0 - 1e-6, "score decreased across the ballast placement"
    assert s1 >= 0.25 - 1e-6, f"gate-open credit missing, got {s1}"

    # ---------------- phase 2: carry the milk through the held-open doorway ----------------
    snap = scene.get_state(all_ids)

    def carry(vx_des: float, kx: float, cap_x: float, tag: str,
              budget: int = 1500) -> bool:
        """Held carry: exact-gravity feedforward + z servo keeps the carton at
        carry height; a gentle velocity servo walks it along vault -x through the
        doorway; over the crib it lowers and releases. Forces are built in the
        VAULT frame each step and applied in the CARTON BODY frame."""
        q_vault = scene.vault.data.root_quat_w
        z_ref, mode = CARRY_Z, "in"
        for i in range(budget):
            loc = milk_loc()
            v_vault = _qapply(_qinv(q_vault), scene.milk.data.root_lin_vel_w)[0]
            x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
            vx, vy, vz = float(v_vault[0]), float(v_vault[1]), float(v_vault[2])
            if mode == "in" and x <= CRIB_CX + 0.030:
                mode = "lower"
                print(f"[solve] {tag}: over the crib @step {i}, lowering", flush=True)
            if mode == "lower":
                z_ref = max(z_ref - 0.15 * DT, LOWER_Z)
                if z <= LOWER_Z + 0.006:
                    clear_force()
                    print(f"[solve] {tag}: released @step {i} "
                          f"({x:+.3f},{y:+.3f},{z:+.3f})", flush=True)
                    return True
                fx = min(max(6.0 * (CRIB_CX - x) - 2.0 * vx, -1.5), 1.5)
            else:
                fx = min(max(kx * (-vx_des - vx), -cap_x), cap_x)
            fy = min(max(-6.0 * y - 2.0 * vy, -1.2), 1.2)
            fz = min(max(MILK_MG + 40.0 * (z_ref - z) - 8.0 * vz, 0.0), 2.0 * MILK_MG)
            f_vault = torch.tensor([fx, fy, fz], device=device).expand(n, 3)
            f_world = _qapply(q_vault, f_vault)
            f_body = _qapply(_qinv(scene.milk.data.root_quat_w), f_world)
            # mild angular damping in the body frame: the held carton doesn't spin
            w_body = _qapply(_qinv(scene.milk.data.root_quat_w),
                             scene.milk.data.root_ang_vel_w)
            t_body = (-0.02 * w_body).clamp(-0.05, 0.05)
            scene.milk.set_external_force_and_torque(f_body.view(n, 1, 3),
                                                     t_body.view(n, 1, 3),
                                                     env_ids=all_ids)
            env.step(no_action)
            if i % 120 == 119:
                lo = milk_loc()
                print(f"[solve] {tag} @{i + 1}: mode={mode} "
                      f"milk_v=({float(lo[0]):+.3f},{float(lo[1]):+.3f},"
                      f"{float(lo[2]):+.3f}) q={gate_q():+.4f}", flush=True)
        clear_force()
        print(f"[solve] {tag}: carry budget exhausted", flush=True)
        return False

    delivered = False
    for attempt, (vx_des, kx, cap_x) in enumerate(
            ((0.15, 15.0, 2.0), (0.22, 25.0, 3.0), (0.30, 40.0, 5.0))):
        tag = f"carry[a{attempt}/v{vx_des:.2f}/cap{cap_x:.1f}]"
        # held hover outside the doorway, lying long-axis along vault x — the pose
        # a hand holds it in before crossing the threshold
        place(scene.milk, (CARRY_X0, 0.0, CARRY_Z), _qy(half_pi))
        if carry(vx_des, kx, cap_x, tag):
            wait_settled(420)
            delivered = bool(scene.milk_in_crib()[0])
        if delivered:
            print(f"[solve] {tag}: milk laid in the crib", flush=True)
            break
        lo = milk_loc()
        print(f"[solve] {tag}: not in crib "
              f"({float(lo[0]):+.3f},{float(lo[1]):+.3f},{float(lo[2]):+.3f}) — "
              f"rolling back", flush=True)
        scene.set_state(snap, all_ids)
        step(60)
        assert bool(scene.gate_open_deep()[0]), "rollback lost the open gate"
    assert delivered, "doorway carry failed on every servo setting"
    report("delivered")
    s2 = print_score("P2 milk carried through the ballast-opened doorway and laid in "
                     "the crib (force-held transit over the sunken gate)")
    assert s2 >= s1 - 1e-6, "score decreased across the carry"
    assert s2 >= 0.65 - 1e-6, f"in-crib credit missing, got {s2}"

    # ---------------- phase 3: TRANSPORT the ballast away; the spring re-seals -------------
    # take the juice off the tray (it sits in the open, outside the vault footprint)
    # and stand it on the far floor, well clear of the vault
    place(scene.juice, (0.75, -0.45, scene_mod.MILK_H / 2 + 0.003),
          torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4))
    for _ in range(20):  # spring re-close: ~1 s + settle
        step(30)
        if bool(scene.gate_closed()[0]) and bool(scene.settled(scene.gate)[0]):
            break
    assert bool(scene.gate_closed()[0]), \
        f"spring failed to re-seal the gate, q={gate_q():+.4f}"
    assert bool(scene.milk_in_crib()[0]), "re-close disturbed the delivered milk"
    # Streak-gate the settle: a single instantaneous success() reading can land at a
    # velocity turning point — require it to HOLD for 60 consecutive steps (0.5 s)
    # before starting the persistence clock.
    streak = 0
    for _ in range(1200):
        step(1)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 60:
            break
    report("sealed")
    s3 = print_score("P3 ballast removed (teleport transport) — the spring re-sealed "
                     "the vault behind the milk")
    assert s3 >= s2 - 1e-6, "score decreased across the re-seal"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after re-seal)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) --------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                mv = float(scene.milk.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: "
                      f"in_crib={bool(scene.milk_in_crib()[0])} "
                      f"closed={bool(scene.gate_closed()[0])} "
                      f"milk_lin={mv:.4f} "
                      f"decoy={bool(scene.decoy_out()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s (hands off)")
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
