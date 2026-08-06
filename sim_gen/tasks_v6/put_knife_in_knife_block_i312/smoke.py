"""Smoke / rubric-REJECTION battery for BladeGuardScene (sim_gen task
`put_knife_in_knife_block_i312`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the servoed contact threading of the sleeve down
over the fixed blade — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: sleeve and decoy standing on
                            the table, vise fixture in place, score ~0 at rest;
  3-4. randomization      — READBACK over 8 seeded resets: sleeve/decoy spawn slots
                            swap and jitter; the vise's xy and yaw vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  beside the blade    — the seed-analog end state ("bring the object to the
                            block" without threading): sleeve standing ON the vise
                            shoulders BESIDE the blade, seat-height but the blade
                            outside the channel -> no success;
  7.  yaw mismatch        — sleeve dropped over the tip rotated 90 deg: the 4.5 cm
                            blade width cannot pass the 2.2 cm channel dimension; the
                            sleeve balances on the tip or tumbles off -> never seated,
                            no success (the load-bearing orientation tolerance);
  8.  wrong object        — the solid ORANGE decoy dropped yaw-matched over the tip:
                            no channel, it can only rest on the tip or fall ->
                            no success, sleeve untouched, score ~0;
  9.  wrong place         — sleeve lying on its side on the table next to the vise
                            ("guard left at the block") -> no success;
  10. latched credit      — a genuine partial insertion (tip 2.5 cm into the channel)
                            latches lift/entry/depth credit; the sleeve is then
                            removed back to the table: the latched credit does not
                            evaporate (score unchanged), no success;
  11. hover is not seated — sleeve held still, blade deep in the channel but 2 cm
                            ABOVE the seat: the seat band rejects a held hover ->
                            no success at any step of the hold;
  12. rejection audit     — success() was never True at ANY judged point;
  13. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_knife_in_knife_block_i312.smoke --headless
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
    env = ENVS.get("simgen.blade_guard")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.50, -1.20, 0.95)) + o),
                                tuple(np.array((0.48, 0.00, 0.16)) + o),
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
        p_slv = (scene.sleeve.data.root_pos_w - scene.env_origins)[0]
        p_dcy = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
        tip = scene._tip_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | sleeve=({float(p_slv[0]):+.3f},{float(p_slv[1]):+.3f},"
              f"{float(p_slv[2]):.3f}) decoy=({float(p_dcy[0]):+.3f},"
              f"{float(p_dcy[1]):+.3f},{float(p_dcy[2]):.3f}) "
              f"tip_z_local={float(tip[2]):+.3f} in_ch={bool(scene.tip_in_channel()[0])} "
              f"yaw_err={float(scene.yaw_err_deg()[0]):.1f} "
              f"seated={bool(scene.seated_config()[0])} "
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

    def vise_frame() -> tuple[float, float, float]:
        return (float(scene._vise_xy[0, 0]), float(scene._vise_xy[0, 1]),
                float(scene._vise_yaw[0]))

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def slv_z() -> float:
        return float((scene.sleeve.data.root_pos_w - scene.env_origins)[0, 2])

    def slv_xy() -> tuple[float, float]:
        p = (scene.sleeve.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.sleeve.data.root_state_w).all()
            and torch.isfinite(scene.decoy.data.root_state_w).all()
            and torch.isfinite(scene.vise.data.root_state_w).all())
    z_slv = slv_z()
    z_dcy = float((scene.decoy.data.root_pos_w - scene.env_origins)[0, 2])
    still = bool(scene._still()[0])
    check("settle: states finite, sleeve and decoy standing on the table, still",
          bool(fin0) and z_slv < 0.03 and z_dcy < 0.03 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        sx, sy = slv_xy()
        vx, vy, vyaw = vise_frame()
        reads.append((sx, sy, 1.0 if sy > 0 else 0.0, vx, vy, math.degrees(vyaw)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (sleeve_x, sleeve_y, sleeve_left, vise_x, "
          f"vise_y, vise_yaw_deg):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: sleeve/decoy spawn slots swap and xy jitter is real across "
          "seeded resets (readback)",
          0.0 < arr[:, 2].mean() < 1.0 and spread[0] > 0.015 and spread[1] > 0.05)
    check("randomization: the vise's xy and yaw vary across seeds (readback: yaw "
          "spread > 10 deg, xy spread > 1 cm, all within the declared band)",
          spread[5] > 10.0 and (spread[3] > 0.01 or spread[4] > 0.01)
          and np.abs(arr[:, 5]).max() <= c.vise_yaw_deg + 1.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. on the shoulders BESIDE the blade =======================
    # The seed-analog wrong outcome: the object brought TO the block but never
    # threaded. Seat-height alone must not count: blade outside the channel.
    torch.manual_seed(41)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    off = 0.035
    bx = vx + off * math.cos(vyaw)
    by = vy + off * math.sin(vyaw)
    place(scene.sleeve, bx, by, c.shoulder_z + 0.003, quat=yaw_quat(vyaw))
    step(120)
    report("beside-blade")
    s, ok = judge()
    at_seat_h = abs(slv_z() - c.shoulder_z) < 0.02
    check("beside the blade: sleeve standing ON the vise shoulders at seat height but "
          "with the blade OUTSIDE the channel — no success, score <= 0.45",
          at_seat_h and not bool(scene.tip_in_channel()[0]) and not ok and s <= 0.45)

    # =========================== 7. yaw mismatch cannot thread ==============================
    torch.manual_seed(51)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    place(scene.sleeve, vx, vy, c.tip_z + 0.005, quat=yaw_quat(vyaw + math.pi / 2))
    step(180)
    report("yaw-mismatch")
    s, ok = judge()
    check("yaw mismatch: sleeve dropped over the tip rotated 90 deg cannot thread "
          "(4.5 cm blade width vs 2.2 cm channel) — never seated, no success",
          not bool(scene.seated_config()[0]) and not ok)

    # =========================== 8. wrong object: the solid decoy ===========================
    torch.manual_seed(61)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    slv_before = slv_xy()
    place(scene.decoy, vx, vy, c.tip_z + 0.005, quat=yaw_quat(vyaw))
    step(180)
    report("wrong-object")
    s, ok = judge()
    slv_after = slv_xy()
    untouched = (abs(slv_after[0] - slv_before[0]) < 0.01
                 and abs(slv_after[1] - slv_before[1]) < 0.01)
    check("wrong object: the solid ORANGE decoy dropped over the tip can only rest on "
          "it or fall off — no success, sleeve untouched on the table, score ~0",
          untouched and not ok and s <= 0.02)

    # =========================== 9. wrong place: left lying by the vise =====================
    torch.manual_seed(71)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    place(scene.sleeve, vx - 0.18, vy, c.out_y / 2 + 0.005,
          quat=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0))
    step(120)
    report("left-lying")
    s, ok = judge()
    check("wrong place: sleeve lying on its side on the table next to the vise — "
          "no success",
          slv_z() < 0.06 and not ok)

    # =========================== 10. latched credit survives regression =====================
    torch.manual_seed(81)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    # genuine partial insertion: tip 2.5 cm into the channel (transient, falling)
    place(scene.sleeve, vx, vy, c.tip_z - 0.025, quat=yaw_quat(vyaw))
    step(2)
    report("partial-entry")
    s10a, ok_a = judge()
    # remove the sleeve back to the table
    place(scene.sleeve, 0.30, 0.25, 0.002, quat=yaw_quat(0.0))
    step(60)
    report("regressed")
    s10b, ok_b = judge()
    check("latched credit: a partial insertion latches lift+entry+depth credit and "
          f"removal back to the table does not evaporate it ({s10a:.3f} -> {s10b:.3f}), "
          "never success",
          s10a >= c.w_lift + c.w_entry - 0.01 and abs(s10b - s10a) < 1e-3
          and not ok_a and not ok_b)

    # =========================== 11. a held hover is not seated =============================
    torch.manual_seed(91)
    env.reset()
    step(10)
    vx, vy, vyaw = vise_frame()
    hover_ok = True
    for _ in range(30):
        place(scene.sleeve, vx, vy, c.shoulder_z + 0.020, quat=yaw_quat(vyaw))
        env.step(no_action)
        s, ok = judge()
        hover_ok = hover_ok and not ok and not bool(scene.seated_config()[0])
    report("held-hover")
    check("hover is not seated: sleeve held still 2 cm above the seat with the blade "
          "deep in the channel — the seat band rejects it at every step of the hold",
          hover_ok)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.sleeve.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.vise.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.blade_guard")
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
