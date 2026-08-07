"""Teleport solution for KeystoneCascadeScene (sim_gen task `poke_cube_i83`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the red keystone is teleported from its floor
scatter slot to a standing pose just above the blue trigger pad (exactly what a
pick-carry-place delivers) with zero velocity, and RELEASED. It settles under
gravity; the scene's `armed` latch fires on its own once it stands still on the pad
with the whole run intact. The poke is an APPLIED WRENCH — a small torque on the
keystone, the honest stand-in for a fingertip push on its upper half — held only
until the keystone passes its balance angle (~9 deg). Everything the rubric reads
happens hands-off after that: the keystone falls through the tunnel mouth, strikes
the first domino's top corner, the run relays domino to domino, the last domino
fells the hammer, the hammer swings through the kiosk letterbox and its shaft sweeps
the ball off the perch onto the pit floor. Nothing inside the tunnel or kiosk is
ever teleported or wrenched.

The wrench frame on this stack is ambiguous (external wrenches have been observed
applied in a frozen body frame), so the poke torque is CALIBRATED: four candidate
encodings of "torque about the alley +y axis" are probed at low magnitude from a
saved state, the state is restored after each probe, and the encoding that actually
tips the keystone downstream is used — then escalated until the keystone passes its
balance angle.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the armed
latch never clears, toppled pieces stay toppled, the potted ball stays potted), then
holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and
prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.poke_cube_i83.solve --headless [--seed N]
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
    import scene as scene_mod

_PAD_X = scene_mod._PAD_X
_KH = scene_mod._KH

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.keystone_cascade")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    env.reset(seed=args.seed)  # seed AFTER build
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tilt_x() -> float:
        """Alley-x component of the keystone's up axis (>0: leaning downstream)."""
        q = scene.keystone.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        u = quat_apply(q, ez)
        return float((u * scene.alley_dir([1.0, 0.0, 0.0])).sum(dim=-1)[0])

    def clear_wrench() -> None:
        scene.keystone.set_external_force_and_torque(zero3, zero3)

    def encode(tau_w: torch.Tensor, mode: int, q_ref: torch.Tensor) -> torch.Tensor:
        """Candidate encodings of a desired WORLD torque for this stack's wrench API."""
        q_now = scene.keystone.data.root_quat_w
        if mode == 0:  # applied in the current body frame
            return quat_apply_inverse(q_now, tau_w)
        if mode == 1:  # applied in the world frame
            return tau_w
        if mode == 2:  # applied rotated by rotation-since-reference (pre-undo it)
            return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), tau_w)
        return quat_apply_inverse(q_ref, tau_w)  # applied in the frozen reference frame

    def apply_torque(mag: float, mode: int, q_ref: torch.Tensor, substeps: int,
                     stop_at: float | None = None) -> float:
        """Hold `mag` N*m about the alley +y axis for `substeps` (re-encoded each
        substep against the live quat); stop early past `stop_at` tilt. Returns tilt."""
        tau_w = scene.alley_dir([0.0, 1.0, 0.0]) * mag
        for _ in range(substeps):
            scene.keystone.set_external_force_and_torque(
                zero3, encode(tau_w, mode, q_ref).view(n, 1, 3))
            env.step(no_action)
            if stop_at is not None and tilt_x() > stop_at:
                break
        clear_wrench()
        return tilt_x()

    def report(tag: str) -> None:
        ups = " ".join(f"{float(scene.up_z(b)[0]):+.2f}" for b in scene.run)
        print(f"[solve] {tag:14s} | run_upz=[{ups}]", flush=True)
        print(f"[solve] {tag:14s} | armed={bool(scene.armed[0])} "
              f"key_upz={float(scene.up_z(scene.keystone)[0]):+.3f} tilt_x={tilt_x():+.3f} "
              f"run_frac={float(scene.run_toppled_frac()[0]):.2f} "
              f"ball_pit={bool(scene.ball_in_pit()[0])} "
              f"ball_v={float(scene.ball.data.root_lin_vel_w[0].norm()):.3f} | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    loc = scene.to_alley(scene.keystone.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): key_slot={int(scene.key_slot[0])} "
          f"keystone alley=({float(loc[0]):+.3f},{float(loc[1]):+.3f}) "
          f"anchor=({float(scene.anchor[0, 0]):+.3f},{float(scene.anchor[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.yaw[0])):+.1f} deg", flush=True)
    report("reset")
    assert bool(scene.run_standing()[0]), "run must stand intact after reset"
    assert not bool(scene.armed[0]), "latch must be clear after reset"
    s_prev = print_score("P0 reset+settle (run intact, keystone loose on the floor)")

    # ---------------- phase 1: place the keystone standing on the pad (transport) ----------
    pos = scene.env_origins.clone()
    pos[:, 0:2] += scene.anchor
    pos += scene.alley_dir([_PAD_X, 0.0, 0.0])
    pos[:, 2] = _KH / 2 + 0.004
    half = scene.yaw / 2
    q_stand = torch.stack([torch.cos(half), torch.zeros_like(half),
                           torch.zeros_like(half), torch.sin(half)], dim=-1)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pos
    st[:, 3:7] = q_stand
    scene.keystone.write_root_state_to_sim(st, all_ids)
    for _ in range(12):  # wait for the latch (it needs a settled streak)
        step(15)
        if bool(scene.armed[0]):
            break
    report("placed")
    assert bool(scene.armed[0]), "armed latch must fire after a settled pad placement"
    s_now = print_score("P1 keystone placed standing on the pad -> armed latch set")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    # ---------------- phase 2: calibrate the wrench frame, then poke ------------------------
    q_ref = scene.keystone.data.root_quat_w.clone()
    saved = scene.get_state(all_ids)
    best_mode, best_r = -1, 0.0
    for mode in range(4):
        r = apply_torque(0.020, mode, q_ref, 12)
        print(f"[solve] calib mode {mode}: tilt_x={r:+.4f}", flush=True)
        scene.set_state(saved, all_ids)
        step(3)
        if r > best_r:
            best_mode, best_r = mode, r
    assert best_mode >= 0 and best_r > 0.01, \
        f"no wrench encoding tipped the keystone downstream (best {best_r:+.4f})"
    print(f"[solve] calibrated wrench encoding: mode {best_mode} (tilt {best_r:+.4f})",
          flush=True)

    tipped = False
    for mag in (0.03, 0.06, 0.12, 0.24):
        r = apply_torque(mag, best_mode, q_ref, 45, stop_at=0.20)
        print(f"[solve] poke torque {mag:.2f} N*m -> tilt_x={r:+.4f}", flush=True)
        if r > 0.18:
            tipped = True
            break
        scene.set_state(saved, all_ids)  # un-rock before the stronger attempt
        step(3)
    assert tipped, "poke failed to take the keystone past its balance angle"
    report("poked")
    s_now = print_score("P2 keystone poked past its balance angle (wrench released)")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    # ---------------- phase 3: hands-off cascade -------------------------------------------
    for _ in range(40):  # up to 10 s simulated
        step(30)
        if bool(scene.success()[0]):
            break
        if float(scene.run_toppled_frac()[0]) >= 1.0 and bool(scene.ball_in_pit()[0]) \
                and bool(scene.ball_settled()[0]):
            break
    report("cascade")
    s_now = print_score("P3 cascade ran hands-off (run down, ball potted)")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    for _ in range(16):  # up to 4 s extra settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after cascade+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {i}: "
                      f"run_frac={float(scene.run_toppled_frac()[0]):.2f} "
                      f"ball_pit={bool(scene.ball_in_pit()[0])} "
                      f"ball_v={float(scene.ball.data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
