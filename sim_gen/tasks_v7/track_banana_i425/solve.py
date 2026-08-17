"""Teleport solution for GlazingBenchScene (sim_gen task `track_banana_i425`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, each move exactly what a pick-and-place
delivers: every present offcut pebble is lifted out of the recess and set down at a
ground depot; the teal pane is carried to a hover centered above the seat and
RELEASED to free-fall and settle flush under gravity; the keeper bar is set down on
the apron drop zone east of the housing. The lock itself is an APPLIED FORCE — a
small horizontal push on the bar (the honest stand-in for a fingertip slide),
velocity-capped and re-encoded every substep — driving the tongue under the housing
roof until the head jams on the housing face; then the force is released and the
rubric reads the settled state. Nothing is ever teleported into the recess or under
the roof.

The wrench frame on this stack is ambiguous (external wrenches have been observed
applied in a frozen body frame), so the push force is CALIBRATED: four candidate
encodings of "force along bench -x" are probed at low magnitude from a saved state,
the state is restored after each probe, and the encoding that actually moves the
bar toward the housing is used — then escalated until the slide completes.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: cleared
pebbles stay cleared, the seated pane stays seated, the jammed keeper stays put),
then holds HANDS-OFF for >= 3 simulated seconds after success() first turns True
and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.track_banana_i425.solve --headless [--seed N]
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

_TOP = scene_mod._TOP
_TON_T = scene_mod._TON_T
_TIP_OFF = scene_mod._TIP_OFF
_DROP_X = scene_mod._DROP_X
_TIP_LOCK = scene_mod._TIP_LOCK
_PANE_T = scene_mod._PANE_T
_PEB = scene_mod._PEB

_DEPOT = ((0.42, 0.30), (0.42, -0.30), (0.50, 0.30), (0.50, -0.30))

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.glazing_bench")().build(num_envs=args.num_envs, device=device)
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

    def bench_pos(body) -> torch.Tensor:
        return scene.to_bench(body.data.root_pos_w)

    def tip_x() -> float:
        loc = bench_pos(scene.keeper)
        proj = (scene.body_axis(scene.keeper, (1.0, 0.0, 0.0))
                * scene.bench_dir([1.0, 0.0, 0.0])).sum(dim=-1)
        return float((loc[:, 0] - _TIP_OFF * proj)[0])

    def q_yaw() -> torch.Tensor:
        half = scene.yaw / 2
        z = torch.zeros_like(half)
        return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)

    def place(body, local_xyz, quat) -> None:
        """Transport-only teleport: set-down/release pose with ZERO velocity."""
        pos = scene.env_origins.clone()
        lx = torch.tensor(local_xyz, device=device).expand(n, 3)
        cy, sy = torch.cos(scene.yaw), torch.sin(scene.yaw)
        pos[:, 0] += scene.anchor[:, 0] + lx[:, 0] * cy - lx[:, 1] * sy
        pos[:, 1] += scene.anchor[:, 1] + lx[:, 0] * sy + lx[:, 1] * cy
        pos[:, 2] = lx[:, 2]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)

    def clear_wrench() -> None:
        scene.keeper.set_external_force_and_torque(zero3, zero3)

    def encode(f_w: torch.Tensor, mode: int, q_ref: torch.Tensor) -> torch.Tensor:
        """Candidate encodings of a desired WORLD force for this stack's wrench API."""
        q_now = scene.keeper.data.root_quat_w
        if mode == 0:  # applied in the current body frame
            return quat_apply_inverse(q_now, f_w)
        if mode == 1:  # applied in the world frame
            return f_w
        if mode == 2:  # applied rotated by rotation-since-reference (pre-undo it)
            return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_w)
        return quat_apply_inverse(q_ref, f_w)  # applied in the frozen reference frame

    def push(mag: float, mode: int, q_ref: torch.Tensor, substeps: int,
             v_cap: float = 0.12, stop_at: float | None = None) -> float:
        """Hold `mag` N along bench -x for `substeps` (re-encoded each substep,
        force gated off above `v_cap`); stop early past `stop_at` tip-x. Returns
        the final tongue-tip bench-x."""
        f_w = scene.bench_dir([-1.0, 0.0, 0.0]) * mag
        for _ in range(substeps):
            v = float(scene.keeper.data.root_lin_vel_w[0].norm())
            f = encode(f_w, mode, q_ref).view(n, 1, 3) if v < v_cap else zero3
            scene.keeper.set_external_force_and_torque(f, zero3)
            env.step(no_action)
            if stop_at is not None and tip_x() <= stop_at:
                break
        clear_wrench()
        return tip_x()

    def report(tag: str) -> None:
        kloc = bench_pos(scene.keeper)[0]
        ploc = bench_pos(scene.pane)[0]
        print(f"[solve] {tag:12s} | out_frac={float(scene.pebs_out_frac()[0]):.2f} "
              f"seated={bool(scene.pane_seated()[0])} "
              f"pane=({float(ploc[0]):+.3f},{float(ploc[1]):+.3f},{float(ploc[2]):.4f}) "
              f"upz={float(scene.up_z(scene.pane)[0]):+.4f} | "
              f"locked={bool(scene.keeper_locked()[0])} tip_x={tip_x():+.4f} "
              f"keeper=({float(kloc[0]):+.3f},{float(kloc[1]):+.3f},{float(kloc[2]):.4f}) | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    n_present = int(scene.present[0].sum())
    print(f"[solve] layout readback (seed {args.seed}): pebbles={n_present} "
          f"pane_slot={int(scene.pane_slot[0])} "
          f"anchor=({float(scene.anchor[0, 0]):+.3f},{float(scene.anchor[0, 1]):+.3f}) "
          f"yaw={math.degrees(float(scene.yaw[0])):+.1f} deg", flush=True)
    report("reset")
    assert n_present >= 2, "at least two pebbles must be present"
    assert float(scene.pebs_out_frac()[0]) < 1e-6, "all present pebbles start in the seat"
    assert not bool(scene.pane_seated()[0]), "pane must start unseated"
    assert not bool(scene.keeper_locked()[0]), "keeper must start unlocked"
    s_prev = print_score("P0 reset+settle (seat fouled, parts on the ground)")

    # ---------------- phase 1: clear the pebbles (transport) -------------------------------
    for i in range(len(scene.pebs)):
        if not bool(scene.present[0, i]):
            continue
        place(scene.pebs[i], (_DEPOT[i][0], _DEPOT[i][1], _PEB / 2 + 0.003), q_yaw())
        step(30)
    step(90)
    report("cleared")
    assert float(scene.pebs_out_frac()[0]) >= 1.0 - 1e-6, "seat must be clear"
    s_now = print_score("P1 every offcut pebble carried out of the seat")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    # ---------------- phase 2: drop the pane flush into the seat ---------------------------
    seated = False
    for hover in (0.080, 0.066, 0.060):
        place(scene.pane, (0.0, 0.0, hover), q_yaw())
        step(240)
        seated = bool(scene.pane_seated()[0])
        print(f"[solve] pane drop from z={hover:.3f}: seated={seated} "
              f"z={float(bench_pos(scene.pane)[0, 2]):.4f} "
              f"upz={float(scene.up_z(scene.pane)[0]):+.4f}", flush=True)
        if seated:
            break
    report("seated")
    assert seated, "pane failed to settle flush in the seat"
    s_now = print_score("P2 teal pane released above the seat -> settled flush")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    # ---------------- phase 3: set the keeper on the apron, slide it home ------------------
    place(scene.keeper, (_DROP_X, 0.0, _TOP + _TON_T / 2 + 0.003), q_yaw())
    step(90)
    report("staged")
    assert bool(scene.settled(scene.keeper)[0]), "keeper must settle in the drop zone"
    t0 = tip_x()
    assert t0 > _TIP_LOCK + 0.05, f"keeper must start clear of the housing (tip {t0:+.3f})"

    # calibrate the force encoding (probe low, restore, compare displacement)
    q_ref = scene.keeper.data.root_quat_w.clone()
    saved = scene.get_state(all_ids)
    best_mode, best_d = -1, 0.0
    for mode in range(4):
        r = push(0.55, mode, q_ref, 25)
        d = t0 - r  # positive = moved toward the housing
        print(f"[solve] calib mode {mode}: tip_x={r:+.4f} (moved {d * 1000:+.1f} mm)",
              flush=True)
        scene.set_state(saved, all_ids)
        step(3)
        if d > best_d:
            best_mode, best_d = mode, d
    assert best_mode >= 0 and best_d > 0.004, \
        f"no force encoding slid the keeper toward the housing (best {best_d:+.4f})"
    print(f"[solve] calibrated force encoding: mode {best_mode} "
          f"({best_d * 1000:+.1f} mm)", flush=True)

    slid = False
    for mag in (0.7, 1.2, 2.2, 4.0):
        r = push(mag, best_mode, q_ref, 600, stop_at=_TIP_LOCK + 0.0015)
        print(f"[solve] push {mag:.1f} N -> tip_x={r:+.4f}", flush=True)
        if r <= _TIP_LOCK + 0.004:
            slid = True
            break
        # hold at the stop a moment longer with the same force to square up
        r = push(mag, best_mode, q_ref, 120)
        if r <= _TIP_LOCK + 0.004:
            slid = True
            break
    clear_wrench()
    step(120)
    report("locked")
    assert slid, f"keeper never reached the stop (tip {tip_x():+.4f})"
    assert bool(scene.keeper_locked()[0]), "keeper must judge locked after release"
    s_now = print_score("P3 keeper pushed home, head jammed on the housing (force off)")
    assert s_now >= s_prev - 1e-6
    s_prev = s_now

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after lock)", flush=True)
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
                      f"seated={bool(scene.pane_seated()[0])} "
                      f"locked={bool(scene.keeper_locked()[0])} "
                      f"out={float(scene.pebs_out_frac()[0]):.2f}", flush=True)
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
