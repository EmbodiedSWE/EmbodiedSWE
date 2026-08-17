"""Smoke battery for FrostScrapeScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — per chip, a fingertip-scale
lateral push through contact dynamics that tips the chip off its narrow ledge, then a
passive slide-and-capture into the trough; the Franka strategy is TASK.md's embodiment
argument). Probe drives here are the same fingertip-scale numbers as solve.py (shared
numbers = shared honesty). Wrong OUTCOMES are constructed as settled states (teleport,
settle, evaluate) — teleports here build the states the rubric must REJECT, never
credit.

One linear run, 11 named checks:
  1. settle    — clean reset: finite state everywhere, every present chip physically
                 racked at its sampled slot (world-pose readback vs the cfg formula),
                 score ~0, no success;
  2. random    — chip count, row height and slot jitter vary across 8 seeds; chips
                 racked at the sampled slots on every draw (READBACK);
  3. null      — 2 s of nothing: score < 0.05, no success;
  4. negative  — the SEED's plan (wipe the glass): pressing every chip straight INTO
                 the pane at 2 N for 1.5 s dislodges nothing and earns ~0 — wiping
                 contact does not do this task;
  5. negative  — near miss inside the load-bearing tolerance: a real 8 mm sideways
                 push (chip CoM still over its ledge) leaves the chip racked, ~0;
  6. negative  — dislodged but MISSED: chip settled flat on the table in FRONT of the
                 trough wall -> dislodge partial credit only, not contained, no
                 success;
  7. negative  — wrong place: chip settled on the table BESIDE the trough, past the
                 end cap (inside the trough's x-band, outside its y-band) -> rejected;
  8. negative  — wrong side: chip settled on the table BEHIND the pane -> rejected;
  9. exactness — the full correct strategy (push each chip off through contact, let
                 the trough catch it) -> success() and score == 1.0, still true 1 s
                 later;
 10. latch     — plucking one captured chip back out of the trough revokes success
                 (success is live state); the latched share of earned credit remains;
 11. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.wipe_window_i168.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.wipe_window_i168 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1500.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale push as solve.py (shared numbers = shared honesty).
KV = 2.5
V_DES0 = 0.12
V_DES_MAX = 0.25
F_CAP = 1.0
DROP_RELEASE = 0.03
CLEAR_DY = 0.055
FALL_CONFIRM = 0.15
PRESS_F = 2.0  # N — the seed-style wiping press, straight into the glass
PARTIAL_DY = 0.008  # m — sideways push that leaves the chip CoM over its 32 mm ledge

PROBE_SEED = 5


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.frost_scrape")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.85, 1.15)) + o),
                                tuple(np.array((0.45, 0.0, 0.55)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p = scene.chip_pos()[0]
        pres = scene.present[0]
        drops = [f"{float(scene.start_z[0, i] - p[i, 2]):+.3f}" if bool(pres[i]) else "  --  "
                 for i in range(4)]
        print(f"[smoke] {tag:14s} | drops={drops} "
              f"cont={[int(bool(v)) for v in scene.contained_now()[0]]} "
              f"latch(d)={[int(bool(v)) for v in scene.dislodge_latch[0]]} "
              f"latch(c)={[int(bool(v)) for v in scene.contained_latch[0]]} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def rack_pose_err() -> float:
        """Max |chip pos - the cfg rack formula| over present chips (readback, m)."""
        zl_row = scene.row_s[:1] - c.pane_size[2] / 2
        err = 0.0
        for i in range(4):
            if not bool(scene.present[0, i]):
                continue
            want = scene._pane_to_world(
                torch.full((1,), c.chip_xl, device=device),
                scene.slot_y[:1, i], zl_row + c.chip_zl_off)[0]
            got = scene.chip_pos()[0, i]
            err = max(err, float((got - want).norm()))
        return err

    def teleport_chip(i: int, pos_env: tuple, quat: tuple = (1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor(pos_env, device=device) + scene.env_origins[0]
        st[0, 3:7] = torch.tensor(quat, device=device)
        scene.chips[i].write_root_state_to_sim(st, torch.tensor([0], device=device))

    def push_off(i: int) -> bool:
        """solve.py's push: lateral velocity cascade until the chip tips off its ledge."""
        v_des = V_DES0
        for _attempt in range(3):
            y0 = float(scene.chip_pos()[0, i, 1])
            z_rack = float(scene.start_z[0, i])
            for _k in range(900):
                p = scene.chip_pos()[0, i]
                v = scene.chips[i].data.root_lin_vel_w[0]
                if z_rack - float(p[2]) > DROP_RELEASE or float(p[1]) - y0 < -CLEAR_DY:
                    break
                f = KV * (-v_des - float(v[1]))
                scene.drive_f[0, i, 1] = max(-F_CAP, min(F_CAP, f))
                step(1)
            scene.drive_f[0, i, :] = 0.0
            for _ in range(48):
                if float(z_rack - scene.chip_pos()[0, i, 2]) > FALL_CONFIRM:
                    return True
                step(10)
            v_des = min(V_DES_MAX, v_des * 1.4)
        return False

    def wait_capture(i: int) -> bool:
        for _ in range(120):
            if bool(scene.contained_now()[0, i]) \
                    and float(scene.chip_speed()[0, i]) < 0.10:
                break
            step(10)
        step(60)
        return bool(scene.contained_latch[0, i])

    press_dir = (math.cos(c.theta), 0.0, -math.sin(c.theta))  # straight INTO the glass

    # ========================= 1. settle / clean-slate + rack readback ========================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(60)
    report("reset")
    n_pres = int(scene.present[0].sum())
    rerr = rack_pose_err()
    print(f"[smoke] present={scene.present[0].tolist()} row_s={float(scene.row_s[0]):.3f} "
          f"rack readback err={rerr * 1000:.1f}mm", flush=True)
    check("settle: clean reset (finite, chips racked at their sampled slots, score ~0)",
          finite_all() and n_pres >= 2 and rerr < 0.010
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    draws = []
    ok_rack = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        kk = int(scene.present[0].sum())
        rr = round(float(scene.row_s[0]), 3)
        ys = tuple(round(float(scene.slot_y[0, i]), 4) for i in range(4))
        ok_rack &= rack_pose_err() < 0.012
        draws.append((kk, rr, ys))
    print(f"[smoke] draws (k, row_s, slot_y): {draws}", flush=True)
    check("randomization is real (chip count, row height, slot jitter vary; racked "
          "poses match by readback)",
          len({d[0] for d in draws}) >= 2 and len({d[1] for d in draws}) >= 5
          and len({d[2] for d in draws}) >= 5 and ok_rack)

    # ========================= 3. null policy =================================================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative: the SEED's plan (wipe the glass) =================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    for _k in range(180):  # 1.5 s wiping press on EVERY present chip
        for i in range(4):
            if bool(scene.present[0, i]):
                scene.drive_f[0, i, 0] = PRESS_F * press_dir[0]
                scene.drive_f[0, i, 2] = PRESS_F * press_dir[2]
        step(1)
    scene.drive_f[0] = 0.0
    step(90)
    report("wipe-press")
    max_drop = max(float(scene.start_z[0, i] - scene.chip_pos()[0, i, 2])
                   for i in range(4) if bool(scene.present[0, i]))
    check("negative (seed strategy): pressing chips INTO the glass dislodges nothing",
          max_drop < 0.02 and finite_all()
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: near-miss partial push ============================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    tgt = [i for i in range(4) if bool(scene.present[0, i])][0]
    y0 = float(scene.chip_pos()[0, tgt, 1])
    for _k in range(600):
        p1 = float(scene.chip_pos()[0, tgt, 1])
        if p1 - y0 < -PARTIAL_DY:
            break
        v = float(scene.chips[tgt].data.root_lin_vel_w[0, 1])
        scene.drive_f[0, tgt, 1] = max(-F_CAP, min(F_CAP, KV * (-0.05 - v)))
        step(1)
    scene.drive_f[0, tgt, :] = 0.0
    step(120)
    report("partial-push")
    drop = float(scene.start_z[0, tgt] - scene.chip_pos()[0, tgt, 2])
    check("negative (near miss): an 8 mm sideways push leaves the chip racked, ~0",
          drop < 0.02 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]) and finite_all())

    # ========================= 6. negative: dislodged but MISSED (front) ======================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    kk = int(scene.present[0].sum())
    tgt = [i for i in range(4) if bool(scene.present[0, i])][0]
    flat = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # thickness up
    teleport_chip(tgt, (c.pane_base_x - 0.25, 0.0, c.table_top + 0.02), flat)
    step(120)
    report("miss-front")
    s6 = float(scene.score()[0])
    check("negative (missed the trough): chip flat on the table in FRONT of the wall — "
          "dislodge partial credit only, not contained, no success",
          bool(scene.dislodge_latch[0, tgt]) and not bool(scene.contained_now()[0, tgt])
          and not bool(scene.contained_latch[0, tgt])
          and 0.01 < s6 < c.w_dislodge / kk + 0.03 and not bool(scene.success()[0]))

    # ========================= 7. negative: wrong place (beside, past the cap) ================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    tgt = [i for i in range(4) if bool(scene.present[0, i])][0]
    teleport_chip(tgt, (c.pane_base_x - 0.06, 0.42, c.table_top + 0.02), flat)
    step(120)
    report("miss-side")
    check("negative (wrong place): chip on the table BESIDE the trough, past the end "
          "cap — not contained, no success",
          not bool(scene.contained_now()[0, tgt])
          and not bool(scene.contained_latch[0, tgt])
          and not bool(scene.success()[0]) and finite_all())

    # ========================= 8. negative: wrong side (behind the pane) ======================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    tgt = [i for i in range(4) if bool(scene.present[0, i])][0]
    teleport_chip(tgt, (c.pane_base_x + 0.22, 0.0, c.table_top + 0.02), flat)
    step(120)
    report("miss-behind")
    check("negative (wrong side): chip on the table BEHIND the pane — not contained, "
          "no success",
          not bool(scene.contained_now()[0, tgt])
          and not bool(scene.contained_latch[0, tgt])
          and not bool(scene.success()[0]) and finite_all())

    # ========================= 9. exactness: the correct strategy =============================
    torch.manual_seed(PROBE_SEED)
    env.reset()
    step(30)
    order = sorted([i for i in range(4) if bool(scene.present[0, i])],
                   key=lambda i: float(scene.slot_y[0, i]))
    all_in = True
    for i in order:
        all_in &= push_off(i) and wait_capture(i)
    good = False
    for _ in range(48):
        if bool(scene.success()[0]):
            good = True
            break
        step(10)
    report("goal-state")
    s_goal = float(scene.score()[0])
    step(120)  # ...and it persists hands-off
    check("exactness: every chip pushed off through contact and caught by the trough -> "
          "success() and score == 1.0, stable",
          all_in and good and abs(s_goal - 1.0) < 1e-3
          and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 10. achievement latch ==========================================
    out_chip = order[0]
    teleport_chip(out_chip, (c.pane_base_x - 0.25, 0.15, c.table_top + 0.02), flat)
    step(120)
    report("revoked")
    sr = float(scene.score()[0])
    check("achievement latch: plucking a chip back out revokes success; latched credit "
          "remains",
          not bool(scene.success()[0])
          and abs(sr - (c.w_dislodge + c.w_contained)) < 0.05)

    # ========================= 11. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.frost_scrape")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
