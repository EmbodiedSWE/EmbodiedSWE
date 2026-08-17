"""Smoke battery for PuddingCarouselScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — align the drum with a lever-scale
axle torque, slide the pudding box through the window with a fingertip-scale push,
rotate the loaded drum to the sealed stop; the Franka strategy is TASK.md's embodiment
argument). Teleported states here are rubric INSTRUMENTATION: construct an outcome as
a settled state under the scene's live drum plant, then assert the rubric's verdict.

One linear run, 12 named checks:
  1. settle     — clean reset: finite state, drum at its authored start angle
                  (readback), both boxes at their floor slots, score ~0;
  2. random     — drum angle / slot assignment / box xy draws differ across seeds
                  (READBACK);
  3. null       — 2 s of nothing: score < 0.05, no success;
  4. negative A — the SEED's plan (drop the box into the receptacle from above): the
                  box lands on the drum ROOF and never enters the bay; a subsequent
                  seal attempt yields no credit — top access is physically killed;
  5. order A    — with the drum MISALIGNED (its authored >= 48 deg start) the same
                  fingertip push that loads in solve.py cannot get the box into the
                  bay, and the shove does not accidentally align the drum;
  6. order B    — with the drum SEALED first, the drum's flank wall fills the window
                  and the pushed box stops at it, far outside the bay (loading after
                  sealing is impossible — order forced both ways);
  7. negative B — near miss: loaded, but the drum servo-LANDED at ~80 deg (~zero
                  rate, short of the 87 deg sealed band) — rejected, partial credit;
  8. negative C — near miss: push CUT early so the box settles protruding past the
                  disc rim (centre outside `bay_x_lo`) — in_bay False, no load credit;
  9. negative D — the WHITE DECOY sealed inside instead of the pudding — drum fully
                  sealed yet rejected, credit ~= the align latch only;
 10. exactness  — full gentle construction (solve's own phases) -> success() and
                  score == 1.0 (accept side of every boundary above);
 11. latch      — rotating the drum back open revokes success; score falls to the
                  latched 0.75 (earned credit does not evaporate; the live 0.25 does);
 12. frames     — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i118.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
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
    from simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i118 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale drives as solve.py (shared numbers = shared honesty).
TAU_MAX = 0.8  # N*m about the axle (lever push)
KV = 2.0
OMEGA_CAP = 0.45
K_APPROACH = 1.5
CUT_ALIGN = math.radians(2.0)
CUT_SEAL = math.radians(89.0)
HOLD_TAU = -0.15
PUSH_MAX = 2.5  # N on the 100 g box (fingertip slide)
KP_V = 30.0
V_PUSH = 0.12
KY = 2.0
DEEP_X = -0.050


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pudding_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    ax, ay = c.axis_xy

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.65, 0.72)) + o),
                                tuple(np.array((0.42, 0.00, 0.10)) + o),
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
        th = math.degrees(float(scene.drum_angle()[0]))
        loc = scene.box_local()[0].tolist()
        print(f"[smoke] {tag:14s} | drum={th:6.1f}deg "
              f"box_local=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"in_bay={bool(scene.in_bay()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def servo_drum(target_rad: float, cut_rad: float, direction: float,
                   hold: float = 0.0, budget: int = 3000) -> bool:
        """solve.py's axle velocity-servo: drive toward `target_rad`, cut the drive
        once past `cut_rad` in `direction` (+1 seals, -1 aligns), leave `hold` on."""
        for _ in range(budget):
            th = float(scene.drum_angle()[0])
            if (direction > 0 and th >= cut_rad) or (direction < 0 and th <= cut_rad):
                scene.drum_drive[0] = hold
                return True
            w = float(scene.drum_rate()[0])
            w_des = direction * min(OMEGA_CAP, K_APPROACH * abs(th - target_rad))
            scene.drum_drive[0] = max(-TAU_MAX, min(TAU_MAX, KV * (w_des - w)))
            step(1)
        scene.drum_drive[0] = hold
        return False

    def land_drum(target_rad: float, budget: int = 3000) -> bool:
        """Servo-LAND the drum at `target_rad` with ~zero rate, then release. With
        viscous-only axle friction the parked drum stays put."""
        for _ in range(budget):
            th = float(scene.drum_angle()[0])
            w = float(scene.drum_rate()[0])
            if abs(th - target_rad) <= math.radians(0.5) and abs(w) <= 0.02:
                scene.drum_drive[0] = 0.0
                return True
            w_des = max(-OMEGA_CAP, min(OMEGA_CAP, K_APPROACH * (target_rad - th)))
            scene.drum_drive[0] = max(-TAU_MAX, min(TAU_MAX, KV * (w_des - w)))
            step(1)
        scene.drum_drive[0] = 0.0
        return False

    def teleport_box_world(body, pos: tuple, mark: bool = False) -> None:
        """Transport-only: box to a free-space world point, zero velocity, yaw 0."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(pos, device=device)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        step(2)
        if mark:
            scene.mark_box_ref()  # wrench-frame reference for the world push

    def push_toward_window(cut_local_x: float | None, budget: int,
                           hold: float = 0.0) -> bool:
        """solve.py's fingertip push: world +x velocity-servo with world-y centering
        on the window axis. Cut when the box centre passes `cut_local_x` in the DRUM
        frame (None = run the whole budget, for the blocked-window probes). Returns
        True iff the cut threshold was reached."""
        scene.drum_drive[0] = hold
        reached = False
        for _ in range(budget):
            if cut_local_x is not None and float(scene.box_local()[0, 0]) >= cut_local_x:
                reached = True
                break
            p = scene.pudding.data.root_pos_w[0] - scene.env_origins[0]
            v = scene.pudding.data.root_lin_vel_w[0]
            fx = KP_V * (V_PUSH - float(v[0]))
            fy = KP_V * (KY * (ay - float(p[1])) - float(v[1]))
            scene.box_drive[0, 0] = max(-PUSH_MAX, min(PUSH_MAX, fx))
            scene.box_drive[0, 1] = max(-PUSH_MAX, min(PUSH_MAX, fy))
            step(1)
        scene.box_drive[0] = 0.0
        scene.drum_drive[0] = 0.0
        return reached

    def load_gently() -> bool:
        """solve's PHASE 1+2: align the drum, hover the box onto the sill, push it in."""
        ok_align = servo_drum(0.0, CUT_ALIGN, -1.0) and settle_until(
            lambda: bool(scene.aligned()[0]) and abs(float(scene.drum_rate()[0])) < 0.05,
            max_steps=360)
        teleport_box_world(scene.pudding,
                           (c.sill_center[0], ay, c.box_rest_z + 0.003), mark=True)
        step(20)
        ok_push = push_toward_window(DEEP_X, budget=1800, hold=HOLD_TAU)
        ok_seat = settle_until(
            lambda: bool((scene.in_bay()
                          & (scene.pudding.data.root_lin_vel_w.norm(dim=-1)
                             < c.settle_box))[0]))
        return ok_align and ok_push and ok_seat

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("settled")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.drum, scene.pudding, scene.decoy)) \
        and bool(torch.isfinite(scene.score()).all())
    drum_ok = abs(float(scene.drum_angle()[0]) - float(scene.theta0[0])) < math.radians(2.0) \
        and c.theta0_min_deg - 1.0 <= math.degrees(float(scene.theta0[0])) \
        <= c.theta0_max_deg + 1.0
    slots_ok = True
    for body in (scene.pudding, scene.decoy):
        p = (body.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        near = min(math.hypot(p[0] - sx, p[1] - sy) for sx, sy in (c.slot_a, c.slot_b))
        slots_ok = slots_ok and near < c.box_jitter * 1.5 + 0.02 \
            and abs(p[2] - c.box_half) < 0.01
    check("settle: clean reset (finite, drum at authored angle, boxes at slots, score ~0)",
          finite and drum_ok and slots_ok and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization readback ======================================
    draws = []
    for seed in (11, 12, 13, 14):
        torch.manual_seed(seed)
        env.reset()
        step(10)
        pud = (scene.pudding.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).tolist()
        slot = "A" if pud[1] > 0 else "B"  # which floor slot got the pudding
        draws.append((round(math.degrees(float(scene.theta0[0])), 1), slot,
                      round(pud[0], 3), round(pud[1], 3)))
    print(f"[smoke] draws (theta0, pudding slot, pudding xy): {draws}", flush=True)
    check("randomization is real (drum angle / slot swap / box xy differ across seeds)",
          len({str(d) for d in draws}) >= 3
          and len({d[0] for d in draws}) >= 3
          and len({d[1] for d in draws}) == 2)

    # ========================= 3. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the SEED's plan (drop from above) ===============
    # The seed's move is put-the-box-in-from-above, then close. Here the receptacle is
    # roofed: the dropped box lands ON the drum roof and never enters; sealing after
    # earns nothing. Judged on the settled aftermath.
    torch.manual_seed(31)
    env.reset()
    step(30)
    teleport_box_world(scene.pudding, (ax + 0.04, ay + 0.04, 0.30))
    step(300)
    z_roof = float((scene.pudding.data.root_pos_w[0] - scene.env_origins[0])[2])
    on_roof = abs(z_roof - 0.1725) < 0.02  # roof top 0.145 + box half 0.0275
    report("roof-drop")
    servo_drum(math.radians(c.drum_limit_deg), CUT_SEAL, +1.0, budget=2000)
    step(360)
    report("roof-sealed")
    check("negative (seed strategy): dropped box lands on the ROOF, never enters — rejected",
          on_roof and float(scene.load_latch[0]) == 0.0 and not bool(scene.in_bay()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 5. order A: misaligned drum blocks the window ==================
    # The drum starts >= 48 deg from aligned: the window sees the curved rim / wall
    # edges. The SAME push that loads in solve.py must fail, and must not rotate the
    # drum into alignment as a side effect (the wedge geometry turns it AWAY).
    torch.manual_seed(41)
    env.reset()
    step(30)
    th_start = math.degrees(float(scene.drum_angle()[0]))
    teleport_box_world(scene.pudding,
                       (c.sill_center[0], ay, c.box_rest_z + 0.003), mark=True)
    step(20)
    push_toward_window(None, budget=600)  # 5 s of pushing at a blocked window
    step(120)
    th_end = math.degrees(float(scene.drum_angle()[0]))
    report("misaligned-push")
    check("order A: misaligned drum — the loading push cannot enter, drum never aligns",
          not bool(scene.in_bay()[0]) and float(scene.load_latch[0]) == 0.0
          and th_end > c.aligned_tol_deg + 2.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))
    print(f"[smoke]   drum {th_start:.1f} -> {th_end:.1f} deg under the wedge push",
          flush=True)

    # ========================= 6. order B: sealed drum blocks the window ======================
    # Seal FIRST (empty), then push: the drum flank fills the window; the box cannot
    # even reach the disc rim. Loading after sealing is impossible for the arm too.
    torch.manual_seed(51)
    env.reset()
    step(30)
    ok_seal = servo_drum(math.radians(c.drum_limit_deg), CUT_SEAL, +1.0)
    settle_until(lambda: bool(scene.sealed()[0])
                 and abs(float(scene.drum_rate()[0])) < 0.05, max_steps=360)
    teleport_box_world(scene.pudding,
                       (c.sill_center[0], ay, c.box_rest_z + 0.003), mark=True)
    step(20)
    push_toward_window(None, budget=600)
    step(120)
    x_end = float((scene.pudding.data.root_pos_w[0] - scene.env_origins[0])[0])
    th_end = math.degrees(float(scene.drum_angle()[0]))
    report("sealed-push")
    check("order B: sealed drum — the flank blocks the window, the bay stays inaccessible",
          ok_seal and th_end > 80.0 and x_end < 0.378
          and not bool(scene.in_bay()[0]) and float(scene.load_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 7. negative B: near miss (under-rotation) ======================
    # Loaded correctly, but the drum is LANDED at ~80 deg with ~zero rate — short of
    # the 87 deg sealed band. Parked (viscous-only friction), so it stays. Partial
    # credit, no success.
    torch.manual_seed(61)
    env.reset()
    step(30)
    ok_load = load_gently()
    ok_land = land_drum(math.radians(80.0))
    step(240)
    th_park = math.degrees(float(scene.drum_angle()[0]))
    report("under-rotated")
    check("negative (near miss): loaded but drum parked short of the sealed band — rejected",
          ok_load and ok_land and 75.0 < th_park < 84.0
          and bool(scene.in_bay()[0]) and not bool(scene.sealed()[0])
          and not bool(scene.success()[0])
          and 0.60 < float(scene.score()[0]) < 0.90)

    # ========================= 8. negative C: near miss (protruding load) =====================
    # Push CUT early: the box settles with its centre outside `bay_x_lo`, hanging out
    # past the disc rim over the sill — geometrically NOT contained, no load credit.
    torch.manual_seed(71)
    env.reset()
    step(30)
    ok_align = servo_drum(0.0, CUT_ALIGN, -1.0) and settle_until(
        lambda: bool(scene.aligned()[0]) and abs(float(scene.drum_rate()[0])) < 0.05,
        max_steps=360)
    teleport_box_world(scene.pudding,
                       (c.sill_center[0], ay, c.box_rest_z + 0.003), mark=True)
    step(20)
    ok_cut = push_toward_window(-0.098, budget=1800, hold=HOLD_TAU)
    step(240)
    loc_x = float(scene.box_local()[0, 0])
    report("protruding")
    check("negative (near miss): box protruding past the rim — in_bay False, no load credit",
          ok_align and ok_cut and loc_x < c.bay_x_lo
          and not bool(scene.in_bay()[0]) and float(scene.load_latch[0]) == 0.0
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.10) < 0.02)

    # ========================= 9. negative D: wrong object (decoy sealed) =====================
    # The white decoy fully loaded and the vault fully sealed: the drum state mimics
    # the goal exactly, but the PUDDING is still on the floor — rejected, align-latch
    # credit only. (Decoy placed by transport-only teleport into the open bay's free
    # interior while aligned; the seal carry is live contact dynamics.)
    torch.manual_seed(81)
    env.reset()
    step(30)
    ok_align = servo_drum(0.0, CUT_ALIGN, -1.0) and settle_until(
        lambda: bool(scene.aligned()[0]) and abs(float(scene.drum_rate()[0])) < 0.05,
        max_steps=360)
    teleport_box_world(scene.decoy, (ax - 0.05, ay, c.box_rest_z + 0.002))
    step(120)
    ok_seal = servo_drum(math.radians(c.drum_limit_deg), CUT_SEAL, +1.0)
    settle_until(lambda: bool(scene.sealed()[0])
                 and abs(float(scene.drum_rate()[0])) < 0.05, max_steps=600)
    dec_loc = (scene.decoy.data.root_pos_w[0] - scene.drum.data.root_pos_w[0]).norm()
    report("decoy-sealed")
    check("negative (wrong object): decoy sealed inside — rejected, align credit only",
          ok_align and ok_seal and bool(scene.sealed()[0]) and float(dec_loc) < 0.12
          and not bool(scene.in_bay()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.12)

    # ========================= 10. exactness: success == score 1.0 ============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    ok_load = load_gently()
    ok_seal = servo_drum(math.radians(c.drum_limit_deg), CUT_SEAL, +1.0)
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("goal-state")
    check("exactness: full gentle construction -> success() and score == 1.0",
          ok_load and ok_seal and ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 11. achievement latch ==========================================
    reopened = servo_drum(0.0, math.radians(40.0), -1.0, budget=1500)
    step(240)
    report("reopened")
    check("achievement latch: reopening revokes success; latched 0.75 remains",
          reopened and not bool(scene.sealed()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.75) < 0.02)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pudding_carousel")
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
