"""Smoke / rubric-REJECTION battery for ButterCellarScene (sim_gen task
`libero_kitchen_scene10_..._i307`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — butter dropped onto the tray, the gravity-ingot
press through the one-way pawl, the ingot removal and the spring-vs-pawl latch — is
the acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled
state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: platform at its top stop,
                            pawl horizontal, all three blocks outside on the floor,
                            everything still, score ~0 at rest;
  3-4.  randomization     — READBACK over 8 seeded resets: the object->slot
                            permutation varies; per-slot xy jitter and spawn yaw vary;
  5.   null policy        — 240 idle steps -> score ~0, no success;
  6.   seed strategy      — the end state the seed's plan (place the item inside the
                            receptacle, receptacle restored "shut") produces here:
                            butter settled on the tray with the platform still UP at
                            its rest stop -> rejected (the press cycle is the task);
  7.   empty press        — platform latched under the pawl with NO butter aboard ->
                            rejected, score <= 0.6;
  8.   one-way ratchet    — from the empty-latched state, 240 more idle steps: the
                            spring never pushes the platform past the pawl (the latch
                            band is a genuine trap) -> platform stays, still no
                            success;
  9.   under-press        — platform released just ABOVE the pawl: the spring returns
                            it all the way to the top stop (the not-past-the-ratchet
                            near miss physically self-rejects) -> no success;
  10.  wrong object       — WHITE lard latched on the tray, butter outside ->
                            rejected (color identification is load-bearing);
  11.  butter on the cap  — butter balanced in the plunger socket of a latched
                            platform (inside the silo column, but not ON the tray) ->
                            rejected;
  12.  ingot not removed  — butter on the tray, platform pressed to the bottom stop
                            by the ingot still sitting in the socket -> the
                            ingot-clear clause rejects;
  13.  latched credit     — from state 12, the butter yanked back out: the latched
                            score holds (credit does not evaporate), still no success;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i307.smoke --headless
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
    env = ENVS.get("simgen.butter_cellar")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.20, -1.10, 0.80)) + o),
                                tuple(np.array((0.56, 0.00, 0.25)) + o),
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

    def pawl_deg() -> float:
        return math.degrees(float(scene.pawl_angle()[0]))

    def plat_z() -> float:
        return float(scene.plat_z()[0])

    def report(tag: str) -> None:
        b = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        g = (scene.ingot.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | butter=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) ingot=({float(g[0]):+.3f},{float(g[1]):+.3f},"
              f"{float(g[2]):.3f}) plat_z={plat_z():.3f} pawl={pawl_deg():+.1f}deg "
              f"on_tray={bool(scene.on_tray(scene.butter)[0])} "
              f"lard_in={bool(scene.in_bore(scene.lard)[0])} "
              f"ingot_clear={bool(scene.ingot_clear()[0])} "
              f"loaded={bool(scene._loaded[0])} latched={bool(scene._latched[0])} "
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

    def write_platform(z: float) -> None:
        """Follower-only re-pose of the platform along its unchanged vertical slide
        (probe constructor; the cellar and the pawl are never moved)."""
        place(scene.platform, c.cellar_pos[0], c.cellar_pos[1], z)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    cx, cy = c.cellar_pos
    lp_x, lp_y = cx + c.load_xy[0], cy + c.load_xy[1]
    bh = c.butter_size[2] / 2

    def tray_load_z(pz: float) -> float:
        return pz + c.tray_top_dz + bh + 0.002

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.butter.data.root_state_w).all()
            and torch.isfinite(scene.lard.data.root_state_w).all()
            and torch.isfinite(scene.ingot.data.root_state_w).all()
            and torch.isfinite(scene.platform.data.root_state_w).all()
            and torch.isfinite(scene.pawl.data.root_state_w).all())
    still = (float(scene.butter.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.platform.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, platform at its top stop, pawl horizontal, all three "
          "blocks outside on the floor, everything still",
          bool(fin0) and abs(plat_z() - c.z_top) < 0.010 and abs(pawl_deg()) <= 3.0
          and still and not bool(scene.on_tray(scene.butter)[0])
          and not bool(scene.in_bore(scene.butter)[0])
          and not bool(scene.in_bore(scene.lard)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    slots = np.array(c.spawn_slots)
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        row = []
        for body in (scene.butter, scene.lard, scene.ingot):
            px, py = obj_xy(body)
            d = np.hypot(slots[:, 0] - px, slots[:, 1] - py)
            row += [px, py, int(d.argmin())]
        q = scene.butter.data.root_quat_w[0]
        row.append(math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))
        reads.append(row)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bx, by, b_slot, lx, ly, l_slot, gx, gy, "
          f"g_slot, b_yaw_deg):\n{arr}", flush=True)
    sigs = {tuple(r[[2, 5, 8]].astype(int)) for r in arr}
    check("randomization: the object->slot permutation varies across seeded resets "
          f"(readback: {len(sigs)} distinct assignments)", len(sigs) >= 2)
    jit = 0.0
    for col, slot_col in ((0, 2), (3, 5), (6, 8)):
        for s_id in range(3):
            grp = arr[arr[:, slot_col] == s_id]
            if len(grp) >= 2:
                jit = max(jit, float((grp[:, col:col + 2].max(axis=0)
                                      - grp[:, col:col + 2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 9].max() - arr[:, 9].min())
    check("randomization: per-slot xy jitter and spawn yaw vary (readback)",
          jit > 0.004 and yaw_spread > 20.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: place inside, receptacle left "shut" ====
    # The seed's plan — put the item into the receptacle and restore the receptacle to
    # its resting configuration — maps here to butter on the tray with the platform
    # still UP at its spring rest stop. Constructed settled: rejected (the one-way
    # press cycle, not placement, is the task).
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.butter, lp_x, lp_y, tray_load_z(plat_z()) + 0.02)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: butter settled on the tray but the platform never pressed "
          "(still at the top stop) — no success, score <= 0.40",
          bool(scene.on_tray(scene.butter)[0]) and abs(plat_z() - c.z_top) < 0.012
          and not ok and s <= 0.40)

    # =========================== 7. empty press =============================================
    # Platform latched under the pawl with no butter aboard: the probe writes the
    # platform (follower-only) just below the pawl; the scene's own spring then seats
    # it up against the pawl — a genuine latched state, but an empty one.
    torch.manual_seed(51)
    env.reset()
    step(10)
    write_platform(0.130)
    step(120)
    report("empty-press")
    s, ok = judge()
    check("empty press: platform latched under the pawl with NO butter aboard — no "
          "success, score <= 0.65",
          c.latch_z_lo < plat_z() < c.latch_z_hi and abs(pawl_deg()) <= c.pawl_closed_deg
          and not bool(scene.on_tray(scene.butter)[0]) and not ok and s <= 0.65)

    # =========================== 8. one-way ratchet holds ===================================
    step(240)
    report("ratchet-holds")
    s, ok = judge()
    check("one-way ratchet: 240 more idle steps against the spring — the platform never "
          f"escapes past the pawl (plat_z={plat_z():.3f} < {c.latch_z_hi}), still no "
          "success", plat_z() < c.latch_z_hi and not ok)

    # =========================== 9. under-press returns to the top ==========================
    # Released just ABOVE the pawl: the spring returns the platform all the way up —
    # the not-past-the-ratchet near miss physically self-rejects.
    torch.manual_seed(61)
    env.reset()
    step(10)
    write_platform(0.170)
    step(180)
    report("under-press")
    s, ok = judge()
    check("under-press: platform released just above the pawl springs back to the top "
          f"stop (plat_z={plat_z():.3f}) — the latch band is unreachable without "
          "passing the ratchet, no success",
          abs(plat_z() - c.z_top) < 0.012 and not ok and s <= 0.4)

    # =========================== 10. wrong object: the white lard ===========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    write_platform(0.130)
    place(scene.lard, lp_x, lp_y, tray_load_z(0.130) + 0.01)
    step(150)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: WHITE lard latched on the tray, butter outside — no success, "
          "score <= 0.6 (color identification is load-bearing)",
          bool(scene.on_tray(scene.lard)[0]) and bool(scene.in_bore(scene.lard)[0])
          and not bool(scene.on_tray(scene.butter)[0]) and not ok and s <= 0.65)

    # =========================== 11. butter balanced on the socket cap ======================
    torch.manual_seed(81)
    env.reset()
    step(10)
    write_platform(0.130)
    sock_z = 0.130 + c.socket_floor_dz
    # NOTE: the 58 mm butter does not fit the 50 mm socket — written above the lip, it
    # settles BRIDGING the lip rails (never intersecting them at the write instant).
    place(scene.butter, cx + c.post_xy[0], cy + c.post_xy[1],
          sock_z + c.lip_h + bh + 0.004)
    step(150)
    report("butter-on-cap")
    s, ok = judge()
    bz = float((scene.butter.data.root_pos_w - scene.env_origins)[0, 2])
    check("butter on the cap: butter perched in the plunger socket of a latched "
          f"platform (z={bz:.3f}, inside the silo column but not ON the tray) — no "
          "success", bz > 0.30 and not bool(scene.on_tray(scene.butter)[0]) and not ok)

    # =========================== 12. ingot left in the socket ===============================
    # Butter on the tray AND platform at full depth, but the press weight never
    # removed: the ingot-clear clause rejects. (One ingot-removal short of success —
    # the ingot stays put so the battery never constructs it.)
    torch.manual_seed(91)
    env.reset()
    step(10)
    write_platform(0.090)
    place(scene.butter, lp_x, lp_y, tray_load_z(0.090) + 0.005)
    place(scene.ingot, cx + c.post_xy[0], cy + c.post_xy[1],
          0.090 + c.socket_floor_dz + c.ingot_size[2] / 2 + 0.004)
    step(150)
    report("ingot-in-socket")
    s, ok = judge()
    check("ingot not removed: butter on the tray, platform held at the bottom stop by "
          "the ingot still in the socket — the ingot-clear clause rejects, no success, "
          "score <= 0.85",
          bool(scene.on_tray(scene.butter)[0]) and plat_z() < 0.105
          and not bool(scene.ingot_clear()[0]) and not ok and s <= 0.85)

    # =========================== 13. latched credit survives regression =====================
    place(scene.butter, 0.30, 0.30, bh + 0.002)  # yank the butter back out
    step(60)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    report("regressed2")
    s_b, ok_b = judge()
    check("latched credit: loaded/depth/latch latches earned then the butter yanked "
          f"back outside — score holds ({s_a:.3f} -> {s_b:.3f}), still no success",
          bool(scene._loaded[0]) and bool(scene._latched[0]) and abs(s_a - s_b) < 1e-3
          and s_a >= 0.5 and not ok_a and not ok_b)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.butter.data.root_state_w).all()
           and torch.isfinite(scene.lard.data.root_state_w).all()
           and torch.isfinite(scene.ingot.data.root_state_w).all()
           and torch.isfinite(scene.platform.data.root_state_w).all()
           and torch.isfinite(scene.pawl.data.root_state_w).all()
           and torch.isfinite(scene.cellar.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.butter_cellar")
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
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
