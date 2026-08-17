"""Teleport solution for PressLatchVaultScene (sim_gen task
`libero_pick_alphabet_soup_i430`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the BLUE press bar is teleported exactly where
a one-armed pick-and-place would carry it — to a held hover ABOVE the two white pin
caps (P1 start) and later to a parking spot on the far floor (P2). Everything the
rubric reads happens through contact dynamics:

  press — a downward force on the bar (the pressing hand, ~25 N) drives the bar
          onto BOTH caps at once; the two spring pins sink together, their collars
          clear the catch strips in the same substep (the transient simultaneity
          event the score latches), and the gate's own drive slides it fully open.
          Nothing ever pushes the gate; the bridged press merely unblocks it.
  ratchet — releasing the press lets both pins spring back up between the strips;
          the gate is held open by its drive and can never be fully re-sealed.
  push  — the RED can is PUSHED along the porch with a small horizontal force
          (<= 1.6 N, under the 1.8 N tip bound), a continuous velocity law plus a
          y-steering term, recomputed every substep in the VAULT frame and applied
          in the CAN BODY frame (per-step frame conversion: the can may spin).
          It crosses the 2 mm step-downs, the doorway sill, and coasts to rest
          fully inside the roofed vault. Gravity + friction seat it; hands off.

Retries roll back to a pre-phase snapshot (scene.get_state/set_state, which carries
the progress latches too), so no failed attempt's debris pollutes the next.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0.50 both-pin press + self-opened gate -> 0.80 can inside -> 1.0 settled
success), then holds HANDS-OFF for >= 3.3 simulated seconds after success() first
turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
        BAR_H, CAN_H, CAP_C, CAP_ZT, GATE_TRAVEL, PORCH_TOP, _qapply, _qinv, _qmul, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BAR_H, CAN_H, CAP_C, CAP_ZT, GATE_TRAVEL, PORCH_TOP, _qapply, _qinv, _qmul, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DT = 1.0 / 120.0
HOVER_Z = CAP_ZT + BAR_H / 2 + 0.008   # bar hover center: 8 mm above the cap tops
PARK = (-0.75, 0.0, BAR_H / 2 + 0.010)  # bar parking spot (vault frame, far floor)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.press_latch_vault")().build(num_envs=args.num_envs, device=device)
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

    def pins_q() -> tuple:
        return float(scene.pin_q(scene.pin_a)[0]), float(scene.pin_q(scene.pin_b)[0])

    def can_loc() -> torch.Tensor:
        return scene.vault_local(scene.can_red.data.root_pos_w)[0]

    def clear_forces() -> None:
        for body in (scene.bar, scene.can_red):
            body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        qa, qb = pins_q()
        cl = can_loc()
        print(f"[solve] {tag:12s} | gate={gate_q():+.4f} pins=({qa:+.4f},{qb:+.4f}) "
              f"can_v=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
              f"in={bool(scene.can_in_vault()[0])} "
              f"L=({bool(scene.l_unlock[0])},{bool(scene.l_open[0])},{bool(scene.l_enter[0])}) "
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
            if bool(scene.settled(scene.can_red)[0]) and bool(scene.settled(scene.bar)[0]) \
                    and bool(scene.settled(scene.gate)[0]):
                break

    def place(body, local_pos, quat_local=None) -> None:
        """TRANSPORT teleport: set a body at rest at a vault-frame pose (what a
        pick-and-carry of a free-standing part delivers)."""
        q_vault = scene.vault.data.root_quat_w
        if quat_local is None:
            quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        local = torch.tensor(local_pos, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.vault.data.root_pos_w + _qapply(q_vault, local)
        st[:, 3:7] = _qmul(q_vault, quat_local)
        body.write_root_state_to_sim(st, all_ids)

    def push_body(body, f_vault: torch.Tensor) -> None:
        """Apply a vault-frame force in the BODY frame of `body` (per-step conversion:
        global-wrench reference frames go stale, and pushed cans spin)."""
        f_world = _qapply(scene.vault.data.root_quat_w, f_vault)
        f_body = _qapply(_qinv(body.data.root_quat_w), f_world)
        body.set_external_force_and_torque(f_body.view(n, 1, 3), zero_wrench,
                                           env_ids=all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.vault.data.root_quat_w[0, 3]),
                           float(scene.vault.data.root_quat_w[0, 0]))
    cl = can_loc()
    qa, qb = pins_q()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) yaw={math.degrees(yaw):+.0f}deg "
          f"can_v=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
          f"gate={gate_q():+.4f} pins=({qa:+.4f},{qb:+.4f})", flush=True)
    report("reset")
    assert gate_q() < 0.010, f"gate must spawn sealed, q={gate_q():+.4f}"
    assert qa > -0.005 and qb > -0.005, f"pins must spawn up, ({qa:+.4f},{qb:+.4f})"
    assert not bool(scene.can_in_vault()[0]), "red can must start outside"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (gate sealed by both pins, can on the porch)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: bridge both caps with the bar, PRESS, gate self-opens -------
    snap = scene.get_state(all_ids)
    opened = False
    for press_force in (25.0, 35.0):
        # hover the bar across both caps (long axis along vault y — its spawn frame)
        place(scene.bar, (CAP_C, 0.0, HOVER_Z))
        step(10)  # free 8 mm drop onto the caps
        f_press = torch.tensor([0.0, 0.0, -press_force], device=device).expand(n, 3)
        streak = 0
        for _ in range(600):  # up to 5 s of pressing
            push_body(scene.bar, f_press)
            env.step(no_action)
            streak = streak + 1 if gate_q() >= 0.160 else 0
            if streak >= 30:
                break
        clear_forces()
        step(60)  # release: pins pop up between the strips; drive holds the gate
        report(f"press {press_force:.0f}N")
        if bool(scene.l_open[0]) and gate_q() >= 0.150:
            opened = True
            break
        print(f"[solve] press at {press_force:.0f} N failed — rollback and escalate",
              flush=True)
        scene.set_state(snap, all_ids)
        step(30)
    assert opened, "bridged press failed to open the gate on every force setting"
    qa, qb = pins_q()
    assert qa > -0.010 and qb > -0.010, "released pins must spring back up (ratchet)"
    s1 = print_score("P1 bar bridged both pins, simultaneous press, gate drove itself "
                     "fully open, pins ratcheted up")
    assert s1 >= s0 - 1e-6 and s1 >= 0.50 - 1e-6, f"open credit missing, got {s1}"

    # ---------------- phase 2: TRANSPORT the bar out of the porch lane ---------------------
    place(scene.bar, PARK)
    step(60)
    assert gate_q() >= 0.150, "parking the bar disturbed the open gate"
    s2 = print_score("P2 bar parked clear of the porch lane (gate stays open on its drive)")
    assert s2 >= s1 - 1e-6, "score decreased across the bar parking"

    # ---------------- phase 3: PUSH the red can through the doorway ------------------------
    snap2 = scene.get_state(all_ids)
    delivered = False
    for v_des, f_base, f_cap in ((0.10, 0.75, 1.6), (0.13, 0.95, 1.7)):
        for i in range(1200):  # up to 10 s of pushing
            loc = scene.vault_local(scene.can_red.data.root_pos_w)
            v_loc = _qapply(_qinv(scene.vault.data.root_quat_w),
                            scene.can_red.data.root_lin_vel_w)
            # continuous kinetic law (no speed-gated limit cycle), tip-safe cap
            fx = -(f_base + 3.0 * (v_des + v_loc[:, 0])).clamp(0.2, f_cap)
            fy = (-2.0 * loc[:, 1]).clamp(-0.3, 0.3)  # steer to the lane center
            f_vault = torch.stack([fx, fy, torch.zeros_like(fx)], dim=-1)
            push_body(scene.can_red, f_vault)
            env.step(no_action)
            if float(loc[0, 0]) < 0.055:
                break
        clear_forces()
        for _ in range(20):  # coast + settle inside
            step(30)
            if bool(scene.settled(scene.can_red)[0]):
                break
        report(f"push v={v_des:.2f}")
        if bool(scene.can_in_vault()[0]):
            delivered = True
            break
        print(f"[solve] push (v_des={v_des:.2f}) failed — rollback and escalate", flush=True)
        scene.set_state(snap2, all_ids)
        step(30)
    assert delivered, "porch push failed on every servo setting"
    s3 = print_score("P3 red can pushed along the porch, over the sill, to rest inside")
    assert s3 >= s2 - 1e-6 and s3 >= 0.80 - 1e-6, f"inside credit missing, got {s3}"

    # Streak-gate the settle: a single instantaneous success() can land at a velocity
    # turning point — require 60 consecutive true substeps before the persistence clock.
    streak = 0
    for _ in range(1200):
        step(1)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 60:
            break
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no settled success)", flush=True)
        os._exit(1)
    s4 = print_score("P4 can settled inside the roofed vault (success)")
    assert s4 >= s3 - 1e-6 and s4 >= 1.0 - 1e-6, f"success score must be 1.0, got {s4}"

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) --------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                cv = float(scene.can_red.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: "
                      f"in={bool(scene.can_in_vault()[0])} can_lin={cv:.4f} "
                      f"gate={gate_q():+.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s5 = print_score("P-persist persistence 3.3 s (hands off)")
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
