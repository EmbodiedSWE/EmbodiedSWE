"""Smoke / rubric-REJECTION battery for CellarTowScene (sim_gen task
`libero_pick_bbq_sauce_i43`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — thread the probe through the roof slot into the
sled's socket, tow the sled out, withdraw, place the bottle — is the acceptance
evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it, plus physics probes that
prove the mechanism is real (the peg-in-eye coupling genuinely drags the sled; the
cellar roof genuinely blocks cradle escape). One probe DOES construct the genuine end
state on purpose — the acceptance construct — and two flip pairs re-enter it; every
other judged point must stay success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; bottle cradled on the parked sled,
                            probe/decoy/pad on the floor; score ~0, no success;
  3-5.  randomization     — READBACK over 8 seeded resets: canopy xy + yaw vary; sled
                            park depth varies; probe / decoy / pad positions vary;
  6.    null policy       — 300 idle steps -> score ~0, no success;
  7.    coupling is real  — the probe teleported into the socket tube and driven with
                            a modest velocity along the slot DRAGS the sled (readback:
                            sled advances, cargo stays cradled) — the peg-in-eye link
                            transmits real force; still no success;
  8.    cellar blocks     — the cradled bottle kicked hard (up + toward the exit)
        escape              under the roof CANNOT leave the canopy: walls + ceiling
                            cage it (readback: still under the footprint, not on any
                            goal) — the tow-first ordering is physics, not fiat;
  9.    wrong object      — the seed's move with the wrong item: the BLUE decoy stood
                            on the goal pad -> no success (and the decoy clause is now
                            violated on top);
  10.   cellar bypass     — the BROWN bottle stood on the pad while the sled is STILL
                            parked in the cellar (a state physics cannot reach, built
                            by fiat): sled-clear gate rejects it -> no success;
  11.   acceptance        — from 10, the sled teleported out into the open (now the
                            full genuine end state): success TRUE;
  12-13. probe re-engaged — the probe stood back in the socket tube: success flips
                            FALSE (disengage clause); probe laid back on the floor:
                            success returns TRUE;
  14-15. decoy disturbed  — the decoy knocked over at its spot: success flips FALSE;
                            stood back up at its spawn: success returns TRUE;
  16.   near miss         — the bottle stood upright on the GROUND 11 cm off the pad
                            center: xy/z gates reject it -> no success;
  17.   settle gate       — the bottle back on the pad but kicked and judged
                            immediately: NOT success (must be at rest);
  18.   cargo still       — the loaded sled parked ON the goal pad (bottle cradled,
        aboard              never lifted): off-sled and height gates reject it;
  19.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct and the two flip-backs (11, 13, 15);
  20.   final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import (  # noqa: F401
        BOT_H, CAN_HX, CAN_HY, CRADLE_X, DECK_T, PAD_T, PROBE_L, TUBE_X, _qapply, _qx,
        _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BOT_H, CAN_HX, CAN_HY, CRADLE_X, DECK_T, PAD_T, PROBE_L, TUBE_X, _qapply, _qx,
        _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cellar_tow")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.05, 0.85)) + o),
                                tuple(np.array((0.00, 0.00, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        sx = float(scene.sled_x_local()[0])
        print(f"[smoke] {tag:18s} | sled_x={sx:+.3f} eng={bool(scene.engaged()[0])} "
              f"clear={bool(scene.sled_clear()[0])} "
              f"cradled={bool(scene.bottle_in_cradle()[0])} "
              f"on_pad={bool(scene.bottle_on_pad()[0])} "
              f"dis={bool(scene.probe_disengaged()[0])} "
              f"decoy={bool(scene.decoy_ok()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    lying = _qx(torch.full((n,), math.pi / 2, device=device))

    def canopy_pose(local) -> torch.Tensor:
        return scene.canopy.data.root_pos_w + _qapply(
            scene.canopy.data.root_quat_w, torch.tensor(local, device=device).expand(n, 3))

    def pad_top_pose(dx: float = 0.0, dy: float = 0.0, on_pad: bool = True) -> torch.Tensor:
        base = scene.pad.data.root_pos_w.clone()
        base[:, 0] += dx
        base[:, 1] += dy
        base[:, 2] = scene.env_origins[:, 2] \
            + ((PAD_T if on_pad else 0.0) + BOT_H / 2 + 0.003)
        return base

    def tube_center(tip_z: float) -> torch.Tensor:
        loc = torch.tensor([TUBE_X, 0.0, 0.0], device=device).expand(n, 3)
        p = scene.sled.data.root_pos_w + _qapply(scene.sled.data.root_quat_w, loc)
        p = p.clone()
        p[:, 2] = scene.env_origins[:, 2] + tip_z + PROBE_L / 2
        return p

    def probe_to_floor(settle_steps: int = 90) -> None:
        pos = canopy_pose((-0.10, 0.58, 0.05))
        teleport(scene.probe, pos, lying, settle_steps=settle_steps)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in
                   (scene.bottle, scene.decoy, scene.probe, scene.sled)):
                break

    def all_finite() -> bool:
        bodies = [scene.canopy, scene.sled, scene.probe, scene.bottle, scene.decoy,
                  scene.pad]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, bottle cradled on the parked sled, sled under "
          "the canopy, probe disengaged on the floor",
          all_finite() and bool(scene.bottle_in_cradle()[0])
          and not bool(scene.sled_clear()[0]) and bool(scene.probe_disengaged()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        cp = (scene.canopy.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.canopy.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        pr = (scene.probe.data.root_pos_w - scene.env_origins)[0]
        dc = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(cp[0]), float(cp[1]), yaw, float(scene.sled_x_local()[0]),
                      float(pr[0]), float(pr[1]), float(dc[0]), float(dc[1]),
                      float(pd[0]), float(pd[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (canopy_x, canopy_y, canopy_yaw, sled_x, "
          f"probe_x, probe_y, decoy_x, decoy_y, pad_x, pad_y):\n{arr.round(3)}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: canopy pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.10)
    check(f"randomization: sled park depth varies (readback: dx={spread[3]:.3f})",
          spread[3] > 0.03)
    check("randomization: probe / decoy / pad positions vary (readback: "
          f"dprobe={max(spread[4], spread[5]):.3f} ddecoy={max(spread[6], spread[7]):.3f} "
          f"dpad={max(spread[8], spread[9]):.3f})",
          max(spread[4], spread[5]) > 0.05 and max(spread[6], spread[7]) > 0.05
          and max(spread[8], spread[9]) > 0.05)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 7. the coupling is real ====================================
    # Probe teleported into the socket tube (through the roof slot), then driven with a
    # modest velocity along the slot: the peg-in-eye contact must DRAG the sled.
    sx0 = float(scene.sled_x_local()[0])
    teleport(scene.probe, tube_center(0.020), ident, settle_steps=8)
    dir_x = _qapply(scene.canopy.data.root_quat_w,
                    torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    vel = [float(dir_x[0]) * 0.35, float(dir_x[1]) * 0.35, 0.0]
    for _ in range(300):  # velocity-driven drawbar: instrumentation, not a solution
        teleport(scene.probe, scene.probe.data.root_pos_w,
                 scene.probe.data.root_quat_w, vel=vel, ang=[0.0, 0.0, 0.0],
                 settle_steps=1)
    sx1 = float(scene.sled_x_local()[0])
    s, ok = judge()
    report("coupling-drag", s, ok)
    check("mechanism (the coupling is real): the probe driven along the slot drags "
          f"the sled through the peg-in-eye contact (sled_x {sx0:+.3f} -> {sx1:+.3f}, "
          f"cargo stays cradled), no success",
          sx1 > sx0 + 0.02 and bool(scene.bottle_in_cradle()[0]) and not ok)

    # =========================== 8. the cellar blocks cradle escape =========================
    env.reset(seed=35)
    settle_all(300)
    exit_dir = _qapply(scene.canopy.data.root_quat_w,
                       torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    kick = [float(exit_dir[0]) * 0.5, float(exit_dir[1]) * 0.5, 0.7]
    teleport(scene.bottle, scene.bottle.data.root_pos_w,
             scene.bottle.data.root_quat_w, vel=kick, ang=[3.0, 0.0, 0.0],
             settle_steps=240)
    bl = scene._local(scene.canopy, scene.bottle.data.root_pos_w)[0]
    s, ok = judge()
    report("cellar-cage", s, ok)
    check("mechanism (the cellar is real): the cradled bottle kicked hard (0.9 m/s, "
          "up + toward the exit) cannot leave the canopy — walls and ceiling cage it "
          f"(canopy-frame x={float(bl[0]):+.3f}, y={float(bl[1]):+.3f}, inside the "
          f"footprint), not on any goal, no success",
          abs(float(bl[0])) < CAN_HX and abs(float(bl[1])) < CAN_HY
          and not bool(scene.bottle_on_pad()[0]) and not ok)

    # =========================== 9. wrong object on the goal ================================
    env.reset(seed=41)
    settle_all(300)
    teleport(scene.decoy, pad_top_pose(), ident, settle_steps=150)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object (the seed's move with the wrong item): the BLUE decoy stood "
          f"on the goal pad is rejected — no success, decoy clause violated on top "
          f"(score={s:.3f} <= 0.02)",
          not ok and s <= 0.02 and not bool(scene.decoy_ok()[0]))

    # =========================== 10. cellar bypass (fiat state) =============================
    env.reset(seed=51)
    settle_all(300)
    teleport(scene.bottle, pad_top_pose(), ident, settle_steps=150)
    s, ok = judge()
    report("cellar-bypass", s, ok)
    check("cellar bypass: the BROWN bottle stood on the pad while the sled is STILL "
          "parked in the cellar (a state physics cannot reach — probe 8 — built here "
          f"by fiat) is rejected by the sled-clear gate: no success (score={s:.3f})",
          not ok and bool(scene.bottle_on_pad()[0]) and not bool(scene.sled_clear()[0]))

    # =========================== 11. acceptance construct ===================================
    teleport(scene.sled, canopy_pose((0.50, 0.0, 0.002)),
             scene.canopy.data.root_quat_w, settle_steps=120)
    settle_all(300)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: from 10, the sled teleported out into the open — the "
          "full genuine end state (bottle on pad, sled clear, probe aside, decoy "
          f"untouched) -> success TRUE (score={s:.3f})", ok and s >= 0.99)

    # =========================== 12-13. probe re-engaged flips success ======================
    teleport(scene.probe, tube_center(0.020), ident, settle_steps=60)
    s, ok = judge()
    report("probe-in-tube", s, ok)
    still_placed = bool(scene.bottle_on_pad()[0]) and bool(scene.sled_clear()[0])
    check("out of order (probe left engaged): the probe stood back in the socket tube "
          "while everything else is the genuine end state flips success FALSE "
          "(disengage clause)", still_placed and not ok
          and not bool(scene.probe_disengaged()[0]))
    probe_to_floor(settle_steps=120)
    s, ok = judge_accept()
    report("probe-removed", s, ok)
    check("probe laid back on the floor -> success returns TRUE (12's rejection was "
          "the disengage clause and nothing else)", ok)

    # =========================== 14-15. decoy clause flips success ==========================
    dc_xy = scene._decoy_spawn.clone()
    dc_lying = torch.zeros(n, 3, device=device)
    dc_lying[:, 0:2] = dc_xy
    dc_lying[:, 2] = scene.env_origins[:, 2] + 0.026
    teleport(scene.decoy, dc_lying, lying, settle_steps=120)
    s, ok = judge()
    report("decoy-knocked", s, ok)
    check("decoy disturbed: the blue bottle knocked over at its own spot flips "
          "success FALSE (untouched clause)", not ok and not bool(scene.decoy_ok()[0]))
    dc_up = dc_lying.clone()
    dc_up[:, 2] = scene.env_origins[:, 2] + BOT_H / 2 + 0.002
    teleport(scene.decoy, dc_up, ident, settle_steps=120)
    s, ok = judge_accept()
    report("decoy-restored", s, ok)
    check("decoy stood back up at its spawn -> success returns TRUE (14's rejection "
          "was the decoy clause and nothing else)", ok)

    # =========================== 16. near miss ==============================================
    teleport(scene.bottle, pad_top_pose(dx=0.11, on_pad=False), ident, settle_steps=150)
    s, ok = judge()
    report("near-miss", s, ok)
    check("near miss: the bottle stood upright on the GROUND 11 cm off the pad center "
          "is rejected by the xy/z gates — no success",
          not ok and not bool(scene.bottle_on_pad()[0])
          and bool(scene.bottle_upright(scene.bottle)[0]))

    # =========================== 17. settle gate ============================================
    teleport(scene.bottle, pad_top_pose(), ident, settle_steps=150)
    assert bool(scene.success()[0]), "acceptance state must reconstruct before the kick"
    teleport(scene.bottle, scene.bottle.data.root_pos_w,
             scene.bottle.data.root_quat_w, vel=[0.0, 0.0, 0.4], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.bottle.data.root_lin_vel_w[0].norm())
    av = float(scene.bottle.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check("settle gate: the placed bottle kicked (lin={:.2f} m/s, ang={:.1f} rad/s) "
          "and judged immediately is NOT success (must be at rest)".format(lv, av),
          (lv > scene.cfg.settle_lin or av > scene.cfg.settle_ang) and not ok)

    # =========================== 18. cargo still aboard =====================================
    env.reset(seed=61)
    settle_all(300)
    pad_xy = scene.pad.data.root_pos_w.clone()
    pad_xy[:, 2] = scene.env_origins[:, 2] + PAD_T + 0.003
    b_loc = torch.tensor([CRADLE_X, 0.0, DECK_T + BOT_H / 2 + 0.004],
                         device=device).expand(n, 3)
    teleport(scene.sled, pad_xy, ident, settle_steps=0)
    teleport(scene.bottle, scene.sled.data.root_pos_w + _qapply(
        scene.sled.data.root_quat_w, b_loc), ident, settle_steps=180)
    s, ok = judge()
    report("cargo-aboard", s, ok)
    check("cargo still aboard: the loaded sled parked ON the goal pad (bottle still "
          "cradled, never lifted out) is rejected by the off-sled and height gates — "
          f"no success (score={s:.3f})",
          not ok and bool(scene.bottle_in_cradle()[0]))

    # =========================== 19-20. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct and the two flip-backs", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cellar_tow")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
