"""Teleport solution for TrayPackScene (sim_gen task `native_tasks_i37`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one block at a time): a single root-state write carries a
   block across free space — the standing blue slab OUT of the tray onto the open
   ground (laid flat, exactly the unstack-regrasp a robot would do first), and later
   each block to a HOVER point ~3 cm above its intended pocket, level, with a small
   deliberate yaw/offset error, zero velocity. No write ever places a block in
   contact with the floor, the walls or another block; a hovering block satisfies no
   rubric clause.
2. SEATING (gravity + contact): from the hover the block FALLS onto the tray floor
   and settles by contact. Every drop is aimed COARSE (5-20 mm off its final pose).
3. FLUSHING (pushed contact sliding — the packing itself): horizontal CoM forces
   slide each landed block flush against the tray walls (and thereby against its
   already-seated neighbours' pockets), exactly the fingertip nudges a robot would
   use. The tray only admits all three blocks when each is squeezed to its wall, so
   these pushes are what MAKE the arrangement feasible — they are the task. Force
   direction is verified by a runtime probe (measured progress in the tray frame)
   and re-encoded if the backend's external-force frame drags with body rotation;
   magnitude escalates from ~1x sliding friction only as needed, and forces are
   cleared before any judgement.
4. ORDER (chosen for clearance, not required by the rubric): slab out -> red bar
   flush along the south wall -> blue slab flush into the north-west corner -> green
   brick dropped into the one remaining pocket and flushed north-east.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.native_tasks_i37.solve --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tray_pack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import matrix_from_quat, quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc_of(name: str) -> torch.Tensor:
        return scene._tray_local(scene.blocks[name].data.root_pos_w)[0]

    def report(tag: str) -> None:
        seat = scene.seated()[0]
        lat = scene._seat[0]
        locs = " ".join(
            f"{nm}=({float(loc_of(nm)[0]):+.3f},{float(loc_of(nm)[1]):+.3f},"
            f"{float(loc_of(nm)[2]):+.3f})" for nm in scene.BLOCK_NAMES)
        _p, _q, vel, _w = scene._block_tensors()
        print(f"[solve] {tag:12s} | {locs} seated={seat.tolist()} "
              f"latch={lat.tolist()} vmax={float(vel[0].max()):.3f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def tray_world(loc_xyz) -> torch.Tensor:
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.tray.data.root_pos_w + quat_apply(scene.tray.data.root_quat_w, loc)

    def hover_drop(name: str, loc_xy, yaw_deg: float) -> None:
        """TRANSPORT the block to a hover ~3 cm above the tray floor at the given
        tray-frame xy with a deliberate yaw error, level, zero velocity — then let
        GRAVITY seat it on the floor."""
        body = scene.blocks[name]
        hover_z = c.z_floor + 0.020 + 0.032
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tray_world((loc_xy[0], loc_xy[1], hover_z))
        st[:, 3:7] = _qmul(scene.tray.data.root_quat_w,
                           _qz(torch.full((n,), math.radians(yaw_deg), device=device)))
        body.write_root_state_to_sim(st, all_ids)
        step(90)  # free fall ~3 cm + settle, hands-off

    # ----- pushed contact sliding: flush a landed block toward a tray wall -------------------
    # The backend's external-force frame can drag with body rotation (pod-dependent).
    # Probe at runtime: if the measured tray-frame progress goes the wrong way,
    # toggle to the pre-encoded mode M = R_ref . R_now^T (memory: frame-drag quirk).
    R_ref: dict[str, torch.Tensor] = {}
    mode: dict[str, int] = {nm: 0 for nm in scene.BLOCK_NAMES}

    def push_flush(name: str, axis: int, sgn: float, target: float) -> None:
        """Push `name` along tray-frame axis (0=x, 1=y) * sgn until its centre
        coordinate passes `target` (flush against the wall / pocket). Bursts of
        20 steps, settle 20, escalating force, direction probe."""
        body = scene.blocks[name]
        e = torch.zeros(3, device=device)
        e[axis] = sgn
        force = 2.0
        for burst in range(40):
            coord = float(loc_of(name)[axis])
            if sgn * coord >= sgn * target:
                clear_force(body)
                step(30)
                print(f"[solve] {name} flush axis{axis} sgn{sgn:+.0f}: "
                      f"coord={coord:+.4f} target={target:+.4f} "
                      f"(burst {burst}, F={force:.1f} N, mode={mode[name]})",
                      flush=True)
                return
            dir_w = quat_apply(scene.tray.data.root_quat_w, e.expand(n, 3))
            f_cmd = dir_w * force
            if mode[name] == 1:
                m_now = matrix_from_quat(body.data.root_quat_w)
                m_enc = R_ref[name] @ m_now.transpose(-1, -2)
                f_cmd = (m_enc @ f_cmd.unsqueeze(-1)).squeeze(-1)
            for _ in range(20):
                body.set_external_force_and_torque(
                    f_cmd.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
                env.step(no_action)
            clear_force(body)
            step(20)
            delta = sgn * (float(loc_of(name)[axis]) - coord)
            if delta < -0.002:
                mode[name] ^= 1
                print(f"[solve] {name} moved AWAY under push (d={delta * 1000:+.1f} mm)"
                      f" — toggling force encoding to mode {mode[name]}", flush=True)
            elif delta < 0.0008:
                force = min(force * 1.5, 8.0)
        report(f"{name}-STUCK")
        print(f"SIM_GEN_SOLVE: FAIL ({name} never flushed on axis {axis})", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # blocks settle on ground / slab settles standing in the tray
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    tq = scene.tray.data.root_quat_w[0]
    tyaw = math.degrees(2.0 * math.atan2(float(tq[3]), float(tq[0])))
    sl = loc_of("slab")
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) yaw={tyaw:+.1f}deg "
          f"slab_local=({float(sl[0]):+.3f},{float(sl[1]):+.3f},{float(sl[2]):+.3f}) "
          f"bar_w=({float(scene.blocks['bar'].data.root_pos_w[0, 0] - scene.env_origins[0, 0]):+.3f},"
          f"{float(scene.blocks['bar'].data.root_pos_w[0, 1] - scene.env_origins[0, 1]):+.3f}) "
          f"brick_w=({float(scene.blocks['brick'].data.root_pos_w[0, 0] - scene.env_origins[0, 0]):+.3f},"
          f"{float(scene.blocks['brick'].data.root_pos_w[0, 1] - scene.env_origins[0, 1]):+.3f})",
          flush=True)
    report("reset")
    pos, _q, _v, _w = scene._block_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in block states after settle"
    assert float(sl[2]) > c.z_floor + 0.030, \
        f"slab must start standing on its side in the tray, loc_z={float(sl[2]):.3f}"
    assert not bool(scene.flat()[0, 1]), "slab must NOT start flat"
    assert not bool(scene.seated()[0].any()), "no block may start seated"
    # capture reset-frame rotations for the force-encoding probe
    for nm in scene.BLOCK_NAMES:
        R_ref[nm] = matrix_from_quat(scene.blocks[nm].data.root_quat_w).clone()
    s0 = print_score("P0 reset+settle (slab standing in the tray, bar+brick outside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: clear the blocker — slab OUT, laid flat ---------------------
    # Transport the standing slab out of the tray onto open ground (flat, level).
    # This is pure transport: it earns nothing (score stays 0) but empties the floor,
    # exactly as an arm would unstack the obstruction first.
    body = scene.blocks["slab"]
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = 0.66
    st[:, 1] = 0.34
    st[:, 2] = 0.020 + 0.003
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    body.write_root_state_to_sim(st, all_ids)
    step(90)
    report("slab-out")
    assert bool(scene.flat()[0, 1]), "slab should lie flat on the ground now"
    assert not bool(scene.seated()[0, 1]), "slab on the ground must not count"
    s1 = print_score("P1 blocker cleared: slab parked flat outside")
    assert s1 >= s0 - 1e-6, "score decreased in P1"

    # ---------------- phase 2: red bar — drop south, flush south-west ----------------------
    hover_drop("bar", (0.0, -0.030), 6.0)
    report("bar-drop")
    push_flush("bar", 1, -1.0, -0.0435)   # flush south wall
    push_flush("bar", 0, -1.0, -0.0035)   # flush west wall
    step(60)
    report("bar-flush")
    assert bool(scene.seated()[0, 0]), "bar must be seated after flushing"
    s2 = print_score("P2 red bar flush along the south wall")
    assert s2 >= max(s1, 0.25) - 1e-6, f"P2 score {s2} (expect >= 0.25)"

    # ---------------- phase 3: blue slab — drop NW, flush north-west -----------------------
    hover_drop("slab", (-0.012, 0.018), 4.0)
    report("slab-drop")
    push_flush("slab", 0, -1.0, -0.0235)  # flush west wall
    push_flush("slab", 1, 1.0, 0.0235)    # flush north wall
    step(60)
    report("slab-flush")
    assert bool(scene.seated()[0, 1]), "slab must be seated after flushing"
    assert bool(scene.seated()[0, 0]), "bar must remain seated"
    s3 = print_score("P3 blue slab packed into the north-west corner")
    assert s3 >= max(s2, 0.50) - 1e-6, f"P3 score {s3} (expect >= 0.50)"

    # ---------------- phase 4: green brick — drop into the last pocket, flush NE -----------
    hover_drop("brick", (0.038, 0.017), 94.0)
    report("brick-drop")
    push_flush("brick", 0, 1.0, 0.0415)   # flush east wall
    push_flush("brick", 1, 1.0, 0.0215)   # flush north wall
    step(90)
    report("brick-flush")
    assert bool(scene.seated()[0].all()), "all three blocks must be seated"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success with all three seated)", flush=True)
        os._exit(1)
    s4 = print_score("P4 green brick seats the last pocket: tray packed")
    assert s4 >= s3 - 1e-6, "score decreased across the final placement"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
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
    main()
