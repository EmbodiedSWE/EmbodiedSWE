"""Smoke / rubric-REJECTION battery for BurrowRamScene (sim_gen task
`lift_peg_upright_i262`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-servo the rod through the bore so it pushes
the captive cube out the far mouth, then drop the freed cube onto the pedestal — is
the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes on forge
seeds 0 and 1). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome as a settled state and asserts the rubric REJECTS it. No probe in
this battery ever reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN     — reset settles finite, cube captive in the bore, score ~0,
                          no success;
  2a-b. randomization   — READBACK over 12 seeded resets: burrow xy+yaw, cube depth,
                          pedestal and rod positions all vary; the pedestal lands on
                          BOTH sides of the tunnel axis;
  3.  null policy       — 300 idle steps -> score ~0, no success (the cube never
                          moves on its own);
  4.  seed strategy     — maniskill/lift_peg_upright's whole goal (the red rod stood
                          UPRIGHT, free, settled, correct height) earns ~0 here: the
                          cube is still captive, no success;
  5.  near-miss freed   — cube shifted to the mouth, sticking half out but center
                          still over the footprint: shift credit only (~0.25), NOT
                          freed, no success;
  6.  freed-only        — cube at rest on the open floor, fully clear: stage credit
                          ~0.60, no success (not on the pedestal);
  7.  near-miss place   — cube ON the pedestal top but ~36 mm off-axis (> ped_xy_tol,
                          still statically supported): not placed, no success;
  8.  wrong object      — the ROD laid to rest on the pedestal top, cube captive:
                          score ~0, no success (only the cube is judged);
  9.  stacking cheat    — cube resting ON the rod lying across the pedestal top:
                          right xy, wrong height (z gap ~ rod thickness) -> not
                          placed, no success;
  10. wrong place       — cube resting on the burrow ROOF: inside the footprint, so
                          neither freed nor placed -> score ~0, no success;
  11. rejection audit   — success() was never True at ANY judged point;
  12. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.lift_peg_upright_i262.smoke --headless
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
# RTX recipe: kit mis-decodes the pod driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.burrow_ram")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.10, 0.85)) + o),
                                tuple(np.array((0.00, 0.00, 0.06)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def report(tag: str) -> None:
        s, ok = judge()
        cl = scene.cube_local()[0]
        print(f"[smoke] {tag:18s} | cube_local=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) captive={bool(scene.cube_captive()[0])} "
              f"freed={bool(scene.cube_freed()[0])} placed={bool(scene.cube_placed()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, settle_steps: int = 45) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def burrow_frame(local) -> torch.Tensor:
        return scene.burrow.data.root_pos_w + _qapply(
            scene.burrow.data.root_quat_w,
            torch.tensor(local, device=device, dtype=torch.float32).expand(n, 3))

    def ped_frame(local) -> torch.Tensor:
        return scene.pedestal.data.root_pos_w + _qapply(
            scene.pedestal.data.root_quat_w,
            torch.tensor(local, device=device, dtype=torch.float32).expand(n, 3))

    def rod_upright() -> bool:
        ax = _qapply(scene.rod.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        return float(ax[0, 2].abs()) > math.cos(math.radians(15.0))

    def all_finite() -> bool:
        bodies = [scene.burrow, scene.pedestal, scene.cube, scene.rod]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def cube_lx() -> float:
        return float(scene.cube_local()[0, 0])

    # =========================== 1. settle / no-NaN ======================================
    env.reset(seed=5)
    step(240)
    report("reset")
    s, ok = judge()
    check("settle: all states finite, cube starts CAPTIVE in the bore "
          f"(cube_lx={cube_lx():+.4f}), score ~0, no success",
          all_finite() and bool(scene.cube_captive()[0])
          and bool(scene.cube_settled()[0]) and s <= 0.02 and not ok)

    # =========================== 2. randomization is real ================================
    reads, sides = [], []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(2)
        bp = (scene.burrow.data.root_pos_w - scene.env_origins)[0]
        q = scene.burrow.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        pp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
        rp = (scene.rod.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(bp[0]), float(bp[1]), yaw, cube_lx(),
                      float(pp[0]), float(pp[1]), float(rp[0]), float(rp[1])))
        ped_ly = float(scene._burrow_local(scene.pedestal.data.root_pos_w)[0, 1])
        sides.append(ped_ly > 0.0)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bx, by, yaw, cube_lx, px, py, rx, ry):\n"
          f"{arr.round(3)}\n[smoke] pedestal on +y side: {sides}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: burrow pose, cube depth, pedestal and rod positions all "
          f"vary across seeded resets (readback: dbx={spread[0]:.3f} dby={spread[1]:.3f} "
          f"dyaw={spread[2]:.2f} dcube_lx={spread[3]:.3f} dped={spread[4]:.3f} "
          f"drod={spread[6]:.3f})",
          spread[0] > 0.03 and spread[1] > 0.03 and spread[2] > 1.0
          and spread[3] > 0.012 and max(spread[4], spread[5]) > 0.15
          and max(spread[6], spread[7]) > 0.15)
    check("randomization: the pedestal lands on BOTH sides of the tunnel axis "
          f"(+y {sides.count(True)}/12, -y {sides.count(False)}/12)",
          sides.count(True) >= 2 and sides.count(False) >= 2)

    # =========================== 3. null policy ==========================================
    env.reset(seed=7)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps (the captive "
          "cube never moves on its own)",
          s <= 0.02 and not ok and bool(scene.cube_captive()[0]))

    # =========================== 4. seed strategy (rod stood upright, free) ==============
    env.reset(seed=9)
    step(60)
    # maniskill/lift_peg_upright's SUCCESS state: the peg (here: the rod) stood
    # upright on the open ground, correct height, settled. Here: worthless.
    q_up = torch.tensor([math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0],
                        device=device).expand(n, 4)
    pos = burrow_frame([0.45, 0.0, 0.0])
    pos = pos.clone()
    pos[:, 2] = scene.env_origins[:, 2] + c.rod_l / 2 + 0.003
    teleport(scene.rod, pos, q_up, settle_steps=240)
    rod_z = float((scene.rod.data.root_pos_w - scene.env_origins)[0, 2])
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (the seed task's goal: rod stood UPRIGHT free on the ground, "
          f"standing={rod_upright()}, center z={rod_z:.3f} ~ {c.rod_l / 2:.3f}): the "
          "cube is still captive — score ~0, no success",
          rod_upright() and abs(rod_z - c.rod_l / 2) < 0.02
          and bool(scene.cube_captive()[0]) and s <= 0.02 and not ok)

    # =========================== 5. near-miss freed (half out of the mouth) ==============
    env.reset(seed=11)
    step(60)
    sign = 1.0 if cube_lx() >= 0.0 else -1.0
    teleport(scene.cube, burrow_frame([sign * 0.100, 0.0, c.cube_s / 2 + 0.003]),
             scene.burrow.data.root_quat_w, settle_steps=90)
    report("near-miss-freed")
    s, ok = judge()
    check("near-miss: cube shifted to the mouth, sticking half out but center over "
          f"the footprint (|cube_lx|={abs(cube_lx()):.3f} < freed_x {c.freed_x:.3f}): "
          f"shift credit only (score={s:.2f} ~ 0.25), NOT freed, no success",
          bool(scene.cube_shifted()[0]) and not bool(scene.cube_freed()[0])
          and 0.20 <= s <= 0.30 and not ok)

    # =========================== 6. freed-only (on the open floor) =======================
    teleport(scene.cube, burrow_frame([sign * 0.200, 0.0, c.cube_s / 2 + 0.003]),
             scene.burrow.data.root_quat_w, settle_steps=90)
    report("freed-only")
    s, ok = judge()
    check("freed-only: cube at rest on the open floor, fully clear of the footprint "
          f"(|cube_lx|={abs(cube_lx()):.3f} >= {c.freed_x:.3f}): stage credit "
          f"(score={s:.2f} ~ 0.60) but NO success (not on the pedestal)",
          bool(scene.cube_freed()[0]) and 0.55 <= s <= 0.62 and not ok)

    # =========================== 7. near-miss placement (off-center on the pedestal) =====
    teleport(scene.cube, ped_frame([0.036, 0.0, c.ped_h / 2 + c.cube_s / 2 + 0.003]),
             scene.pedestal.data.root_quat_w, settle_steps=120)
    d_xy = float((scene.cube.data.root_pos_w[0, :2]
                  - scene.pedestal.data.root_pos_w[0, :2]).norm())
    report("near-miss-place")
    s, ok = judge()
    check("near-miss placement: cube ON the pedestal top but off-axis "
          f"(|xy|={d_xy:.3f} > tol {c.ped_xy_tol:.3f}, still supported): not placed, "
          "no success",
          d_xy > c.ped_xy_tol and not bool(scene.cube_placed()[0])
          and not ok and s <= 0.62)

    # =========================== 8. wrong object (rod on the pedestal) ===================
    env.reset(seed=13)
    step(60)
    teleport(scene.rod, ped_frame([0.0, 0.0, c.ped_h / 2 + c.rod_w / 2 + 0.003]),
             scene.pedestal.data.root_quat_w, settle_steps=150)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the ROD laid to rest on the pedestal top while the cube is "
          "still captive: score ~0, no success (only the cube is judged)",
          bool(scene.cube_captive()[0]) and s <= 0.02 and not ok)

    # =========================== 9. stacking cheat (cube on rod on pedestal) =============
    teleport(scene.cube, ped_frame([0.0, 0.0, c.ped_h / 2 + c.rod_w + c.cube_s / 2 + 0.006]),
             scene.pedestal.data.root_quat_w, settle_steps=150)
    dz = float((scene.cube.data.root_pos_w - scene.pedestal.data.root_pos_w)[0, 2])
    report("stacking-cheat")
    s, ok = judge()
    check("stacking cheat: cube resting ON the rod lying across the pedestal — right "
          f"xy, wrong height (dz={dz:.3f} vs required {c.ped_h / 2 + c.cube_s / 2:.3f}"
          f"+/-{c.ped_z_tol:.3f}): not placed, no success",
          dz > c.ped_h / 2 + c.cube_s / 2 + c.ped_z_tol + 0.01
          and not bool(scene.cube_placed()[0]) and not ok)

    # =========================== 10. wrong place (cube on the burrow roof) ===============
    env.reset(seed=17)
    step(60)
    roof_top = c.tun_h + 0.024  # _ROOF_T
    teleport(scene.cube, burrow_frame([0.0, 0.0, roof_top + c.cube_s / 2 + 0.003]),
             scene.burrow.data.root_quat_w, settle_steps=150)
    report("cube-on-roof")
    s, ok = judge()
    check("wrong place: cube resting on the burrow ROOF (inside the footprint, "
          f"|cube_lx|={abs(cube_lx()):.3f} < freed_x): neither freed nor placed — "
          f"score ~0 (={s:.2f}), no success",
          not bool(scene.cube_freed()[0]) and not bool(scene.cube_placed()[0])
          and s <= 0.02 and not ok)

    # =========================== 11-12. audit + no-NaN ===================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict ==========================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.burrow_ram")
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
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
