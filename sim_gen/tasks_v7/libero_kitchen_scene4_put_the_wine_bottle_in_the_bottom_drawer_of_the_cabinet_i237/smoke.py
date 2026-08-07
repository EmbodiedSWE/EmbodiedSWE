"""Smoke / rubric-REJECTION battery for TiltBinStowScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i237`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — hinge-torque open, gravity deposit, hinge-torque
close — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bin shut, both bottles
                            standing at their floor slots, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: wine/decoy slot assignment
                            flips (Bernoulli side swap) and per-slot xy jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's end state (bottle resting inside an OPEN
                            receptacle): wine deposited in the bin resting on its open
                            stop -> NOT success, score < 0.9 (the shut clause is
                            load-bearing);
  7.  on-top cheat        — wine bottle parked on the cabinet's top panel -> score ~0,
                            no success;
  8.  against-face        — wine bottle standing against the shut bin's front face
                            (as close as the mechanism allows without opening) ->
                            score ~0, no success;
  9.  near-miss close     — wine inside but the bin held ajar just OUTSIDE the closed
                            tolerance (15 deg vs 8) -> NOT success (bottle teleported
                            away before release so the empty bin falls shut without
                            ever reaching success);
  10. wrong object        — the WHITE decoy inside the shut bin, wine outside ->
                            score ~0, no success (color identification is
                            load-bearing);
  11. decoy exclusion     — wine AND decoy both inside the shut settled bin: every
                            other gate passes, the decoy clause alone rejects -> NOT
                            success, score <= 0.85;
  12. latched credit      — teleporting the wine back OUT to the floor afterwards
                            leaves the latched score unchanged (credit does not
                            evaporate), still no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_bin_stow")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.00, 0.90)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def deg() -> float:
        return math.degrees(float(scene.bin_angle()[0]))

    def report(tag: str) -> None:
        w = (scene.wine.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | angle={deg():6.1f} deg "
              f"wine=({float(w[0]):+.3f},{float(w[1]):+.3f},{float(w[2]):.3f}) "
              f"wine_in={bool(scene._inside_bin(scene.wine)[0])} "
              f"decoy_in={bool(scene._inside_bin(scene.decoy)[0])} "
              f"opened={bool(scene._opened[0])} app={float(scene._app_max[0]):.3f} "
              f"dep={bool(scene._in[0])} close={float(scene._close_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def pose_bin(theta_deg: float) -> None:
        """Follower-only re-pose of the bin about its unchanged hinge (probe
        constructor / the same joint-coordinate write reset uses)."""
        h = math.radians(theta_deg) / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_pos
        st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
        st[:, 0:3] += env.iscene.env_origins
        scene.bin.write_root_state_to_sim(st, all_ids)

    def place_in_bin(body, loc_x: float, loc_y: float, loc_z: float,
                     lying: bool = True) -> None:
        """Teleport `body` to a bin-local point of the bin's CURRENT pose (probe
        constructor: builds the inside-the-bin relation through walls)."""
        from isaaclab.utils.math import quat_apply, quat_mul

        loc = torch.tensor([loc_x, loc_y, loc_z], device=device).expand(n, 3)
        pos = scene.bin.data.root_pos_w + quat_apply(scene.bin.data.root_quat_w, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        if lying:
            q_rel = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0],
                                 device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(scene.bin.data.root_quat_w, q_rel)
        else:
            st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)  # random ajar angle falls shut here
    report("show")
    fin0 = (torch.isfinite(scene.bin.data.root_state_w).all()
            and torch.isfinite(scene.wine.data.root_state_w).all()
            and torch.isfinite(scene.decoy.data.root_state_w).all())
    wz = float((scene.wine.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.wine.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.bin.data.root_ang_vel_w[0].norm()) < c.settle_omega)
    check("settle: states finite, bin fell shut, wine standing at its slot, all still",
          bool(fin0) and deg() <= c.closed_tol_deg and abs(wz - c.wine_body_h / 2) < 0.01
          and still and not bool(scene._inside_bin(scene.wine)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        wx, wy = obj_xy(scene.wine)
        dx, dy = obj_xy(scene.decoy)
        reads.append((wx, wy, dx, dy, 1.0 if wy > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (wine_x, wine_y, decoy_x, decoy_y, "
          f"wine_on_left):\n{arr}", flush=True)
    flags = arr[:, 4]
    check("randomization: wine/decoy slot assignment flips across seeded resets AND the "
          "two bottles always take opposite slots (readback)",
          0.0 < flags.mean() < 1.0
          and all((r[1] > 0) != (r[3] > 0) for r in reads))
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    check("randomization: per-slot spawn jitter is real (readback spread > 4 mm within "
          "a slot group)", jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: bottle in the OPEN receptacle ============
    # The seed's end state — the bottle resting inside an OPEN receptacle. The bin rests
    # on its 48 deg stop (bistable), the wine bottle is dropped in through the mouth and
    # settles inside. The shut clause is load-bearing: NOT success.
    torch.manual_seed(41)
    env.reset()
    step(30)
    pose_bin(c.open_limit_deg)
    step(30)  # rests on the stop under gravity
    place_in_bin(scene.wine, c.mouth_local[0], 0.0, c.bin_t + c.bin_in_h + 0.035)
    step(240)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: wine settled INSIDE the bin but the bin rests OPEN on its "
          "stop — NOT success, score < 0.9 (shut clause is load-bearing)",
          bool(scene._inside_bin(scene.wine)[0]) and deg() >= 40.0
          and not ok and s < 0.9)

    # =========================== 7. on-top cheat ============================================
    torch.manual_seed(51)
    env.reset()
    step(30)
    top_z = c.shell_in_h + c.shell_t + c.wine_body_h / 2 + 0.002
    place(scene.wine, c.hinge_pos[0] - 0.08, 0.0, top_z)
    step(90)
    report("on-top")
    s, ok = judge()
    check("on-top cheat: wine parked on the cabinet's top panel — score ~0 (<= 0.05), "
          "no success", not bool(scene._inside_bin(scene.wine)[0]) and not ok and s <= 0.05)

    # =========================== 8. against the shut face ===================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    place(scene.wine, c.hinge_pos[0] + c.handle_out + c.wine_body_r + 0.004, 0.0,
          c.wine_body_h / 2 + 0.002)
    step(90)
    report("against-face")
    s, ok = judge()
    check("against-face: wine standing against the shut bin's face (closest approach "
          "without opening) — score ~0 (<= 0.05), no success",
          not bool(scene._inside_bin(scene.wine)[0]) and not ok and s <= 0.05)

    # =========================== 9. near-miss: ajar just outside tolerance ==================
    # Wine inside, bin HELD at 15 deg (tolerance is 8): every containment/stillness gate
    # can pass but the angle gate rejects. The bin is pose-HELD on its joint arc each
    # step; the wine is teleported far away BEFORE release so the empty bin falls shut
    # without the battery ever crossing success.
    torch.manual_seed(71)
    env.reset()
    step(30)
    ajar = c.closed_tol_deg + 7.0
    pose_bin(ajar)
    place_in_bin(scene.wine, -c.bin_t - c.bin_in_d / 2, 0.0, c.bin_t + c.wine_body_r + 0.002)
    for _ in range(90):
        pose_bin(ajar)  # hold on the joint arc while the bottle settles inside
        env.step(no_action)
    report("near-miss-ajar")
    s, ok = judge()
    near_ok = (bool(scene._inside_bin(scene.wine)[0]) and abs(deg() - ajar) < 1.0
               and not ok and s < 0.9)
    place(scene.wine, 0.15, 0.30, c.wine_body_h / 2 + 0.002)  # away BEFORE release
    step(90)  # empty bin falls shut
    check("near-miss: wine inside but the bin ajar at 15 deg (tol 8) — angle gate "
          "rejects, NOT success", near_ok)

    # =========================== 10. wrong object: decoy stowed =============================
    torch.manual_seed(81)
    env.reset()
    step(30)
    place_in_bin(scene.decoy, -c.bin_t - c.bin_in_d / 2, 0.0,
                 c.bin_t + c.decoy_body_r + 0.002)
    step(90)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: WHITE decoy inside the shut bin, wine outside — score ~0 "
          "(<= 0.05), no success (color identification is load-bearing)",
          bool(scene._inside_bin(scene.decoy)[0]) and deg() <= c.closed_tol_deg
          and not ok and s <= 0.05)

    # =========================== 11. decoy-exclusion clause =================================
    # Same episode: ALSO stow the wine. Wine inside + bin shut + settled — every other
    # gate passes; the decoy clause alone must reject.
    # side-by-side lying bottles at the interior extremes: wine against the back wall,
    # decoy against the front wall (centres 61 mm apart vs 58 mm radius sum)
    place_in_bin(scene.decoy, -c.bin_t - 0.001 - c.decoy_body_r, 0.0,
                 c.bin_t + c.decoy_body_r + 0.002)
    place_in_bin(scene.wine, -c.bin_t - c.bin_in_d + c.wine_body_r, 0.0,
                 c.bin_t + c.wine_body_r + 0.002)
    step(150)
    report("both-stowed")
    s11, ok = judge()
    wine_still = float(scene.wine.data.root_lin_vel_w[0].norm()) < c.settle_speed
    check("decoy exclusion: wine AND decoy both settled inside the shut bin — all other "
          "gates pass, the decoy clause alone rejects: NOT success, score <= 0.85",
          bool(scene._inside_bin(scene.wine)[0]) and bool(scene._inside_bin(scene.decoy)[0])
          and deg() <= c.closed_tol_deg and wine_still and not ok and s11 <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    place(scene.wine, 0.42, 0.24, c.wine_body_h / 2 + 0.002)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: teleporting the wine back OUT of the bin leaves the latched "
          f"score unchanged ({s11:.3f} -> {s12:.3f}), still no success",
          abs(s12 - s11) < 1e-3 and not bool(scene._inside_bin(scene.wine)[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.bin.data.root_state_w).all()
           and torch.isfinite(scene.wine.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.shell.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_bin_stow")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
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
    main()
