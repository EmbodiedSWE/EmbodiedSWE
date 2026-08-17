"""Smoke / rubric-REJECTION battery for ShutterGarageScene (sim_gen task
`plug_charger_i230`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — open the shutter, push the brick through the
doorway onto the plate, close the shutter, all under a floating-hand force
controller — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
and asserts the rubric REJECTS it; no probe in this battery ever reaches success(),
and a final audit check asserts exactly that.

  1.  settle/no-NaN     — reset layout settles finite: shutter verified CLOSED over
                          the doorway and brick verified OUTSIDE by dock-frame READBACK;
  2.  SEED strategy     — the seed task's whole plan (align prongs, insert, episode
                          over) constructed literally: doorway opened, brick docked
                          prongs-on-plate, shutter LEFT OPEN -> NOT success, score
                          capped at 0.70 (the re-close is a judged, mandatory step);
  3-4. randomization    — READBACK over 8 seeded resets: garage xy + yaw vary, and the
                          brick's dock-frame spawn xy + relative yaw vary;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  blocked entry     — CONTACT probe, no teleport past anything: 2 simulated
                          seconds of a 3 N floor-level push drives the brick INTO the
                          CLOSED shutter; the probe must actually advance to the slab
                          (non-vacuous) and be stopped there: prong tip never crosses
                          the facade plane, doorway-transit credit never latches, the
                          shutter stays closed -> opening the shutter first is
                          physically mandatory;
  7.  door-open only    — shutter fully open, brick untouched: travel credit only
                          (score ~0.20), NOT success;
  8.  near-miss transit — door open, brick in the bay but tip short of the seat band:
                          partial credit only, docked never latches, NOT success;
  9.  sideways parking  — brick fully inside with the shutter closed but yawed ~90 deg
                          (prongs at the side wall): alignment fails -> NOT docked,
                          NOT success (being sealed inside is not the goal — docking is);
  10. backwards dock    — brick pushed to the plate BACK-FIRST (prongs facing the
                          doorway), shutter closed: prong-axis readback < 0 -> NOT
                          docked, NOT success (orientation matters, as in the seed);
  11. roof percher      — brick settled on the garage ROOF: outside the doorway lane
                          (readback), no transit credit, NOT success, score ~0;
  12. latched door credit — opening travel credit survives the shutter being closed
                          again (credit never evaporates, success stays gone);
  13. monotonicity      — a deeper doorway transit latches strictly more credit;
  14. latched entry credit — transit credit survives pulling the brick back OUT of
                          the bay to the apron;
  15. rejection audit   — success() was never True at ANY judged point;
  16. final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.plug_charger_i230.smoke --headless
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_garage")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.62, 0.48)) + o),
                                tuple(np.array((0.0, 0.03, 0.04)) + o),
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

    def garage_pose() -> tuple[torch.Tensor, float]:
        gp = (scene.garage.data.root_pos_w - scene.env_origins)[0]
        q = scene.garage.data.root_quat_w[0]
        return gp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        sd = scene.shutter_dock()[0]
        bd = scene.brick_dock()[0]
        td = scene.tip_dock()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | shutter_x={float(sd[0]):+.4f} "
              f"brick_dock=({float(bd[0]):+.3f},{float(bd[1]):+.3f},{float(bd[2]):.3f}) "
              f"tip_y={float(td[1]):+.4f} axis_in={float(scene.prong_axis_in()[0]):+.2f} "
              f"in_rails={bool(scene.shutter_in_rails()[0])} "
              f"closed={bool(scene.shutter_closed()[0])} "
              f"docked={bool(scene.docked()[0])} "
              f"latches=({float(scene.door_latch[0]):.2f},{float(scene.entry_latch[0]):.2f},"
              f"{float(scene.dock_latch[0]):.0f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def write_pose(body, x: float, y: float, z: float,
                   quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Kinematic probe write (instrumentation, not a solution); no stepping."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def dock_to_world(lx: float, ly: float) -> tuple[float, float, float]:
        gp, gy = garage_pose()
        ca, sa = math.cos(gy), math.sin(gy)
        return (float(gp[0]) + ca * lx - sa * ly,
                float(gp[1]) + sa * lx + ca * ly, gy)

    def place_brick(lx: float, ly: float, yaw_off: float = 0.0,
                    settle_steps: int = 30) -> None:
        """Probe write of the brick at garage(dock)-local (lx, ly) on the floor with
        yaw = garage yaw + yaw_off, then REAL physics steps (the zero-step trap)."""
        wx, wy, gy = dock_to_world(lx, ly)
        write_pose(scene.brick, wx, wy, c.brick_h / 2 + 0.0015, quat=yaw_quat(gy + yaw_off))
        step(settle_steps)

    def place_shutter(lx: float, settle_steps: int = 30) -> None:
        """Probe write of the shutter at rail position garage-local x = lx."""
        wx, wy, gy = dock_to_world(lx, c.door_yc)
        write_pose(scene.shutter, wx, wy, c.door_zc + 0.0005, quat=yaw_quat(gy))
        step(settle_steps)

    dock_ly = c.plate_y - c.tip_half - 0.001  # brick origin y of a just-touching dock

    # =========================== 1. settle + reset readback =================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    sd = scene.shutter_dock()[0]
    td = scene.tip_dock()[0]
    fin0 = bool(torch.isfinite(scene.garage.data.root_state_w).all()
                and torch.isfinite(scene.shutter.data.root_state_w).all()
                and torch.isfinite(scene.brick.data.root_state_w).all())
    check("settle: states finite; shutter verified CLOSED over the doorway and brick "
          "verified OUTSIDE the garage by dock-frame readback; everything settled",
          fin0 and bool(scene.settled()[0]) and bool(scene.shutter_closed()[0])
          and abs(float(sd[0])) < 0.006 and float(td[1]) < -0.06)

    # =========================== 2. SEED strategy ===========================================
    # The seed's whole plan, constructed literally: get the doorway open, put the
    # charger prongs-on-plate, and STOP — the episode is over in the seed the moment
    # the plug is seated. Here the shutter is still open, so success is withheld and
    # the score caps at 0.70: the re-close is a judged, mandatory step.
    place_shutter(c.x_door_clear + 0.003, settle_steps=20)
    place_brick(0.0, dock_ly, settle_steps=40)
    report("seed-strategy")
    s, ok = judge()
    check("SEED strategy: doorway open + brick docked prongs-on-plate + shutter LEFT "
          "OPEN (insert-and-stop, the seed's terminal state) — docked latches but "
          "NOT success, score <= 0.705",
          bool(scene.docked()[0]) and not bool(scene.shutter_closed()[0])
          and s <= 0.705 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd_i in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd_i)
        step(5)
        gp, gy = garage_pose()
        bd = scene.brick_dock()[0]
        q = scene.brick.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        rel = math.atan2(math.sin(byaw - gy), math.cos(byaw - gy))
        reads.append((float(gp[0]), float(gp[1]), gy, float(bd[0]), float(bd[1]), rel))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (garage_x, garage_y, garage_yaw, "
          f"brick_lx, brick_ly, brick_rel_yaw):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: garage xy + yaw vary across 8 seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.2)
    check("randomization: brick dock-frame spawn xy + relative yaw vary across 8 "
          "seeded resets (readback)",
          spread[3] > 0.02 and spread[4] > 0.01 and spread[5] > 0.5)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. blocked entry (contact, the shutter guards) =============
    # Stage the brick on the apron prongs-in (transport instrumentation), then push it
    # at the CLOSED shutter with a floor-level 3 N force for 2 simulated seconds. The
    # probe must be NON-VACUOUS (the brick really advances to the slab) and must be
    # STOPPED there: the prong tip never crosses the facade plane, transit credit
    # never latches, and the shutter stays closed.
    env.reset(seed=41)
    step(30)
    place_brick(0.0, -0.13, settle_steps=10)  # tip ~33 mm short of the shutter slab
    tip_start = float(scene.tip_dock()[0, 1])
    _, in_w_gy = garage_pose()
    fpush = torch.zeros(n, 1, 3, device=device)
    fpush[:, 0, 0] = 3.0 * -math.sin(in_w_gy)
    fpush[:, 0, 1] = 3.0 * math.cos(in_w_gy)
    scene.brick.set_external_force_and_torque(fpush, zero_wrench, env_ids=all_ids,
                                              is_global=True)
    tip_max = -1.0
    for _ in range(240):
        step(1)
        tip_max = max(tip_max, float(scene.tip_dock()[0, 1]))
    scene.brick.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)
    report("blocked-entry")
    s, ok = judge()
    check("blocked entry: 2 s of a 3 N floor push at the CLOSED shutter — the brick "
          f"really advanced (tip {tip_start:+.3f} -> max {tip_max:+.3f}, non-vacuous) "
          "but the tip never crossed the facade plane, transit credit never latched, "
          "shutter still closed, score <= 0.02, no success",
          tip_max > tip_start + 0.010 and tip_max < -0.005
          and float(scene.entry_latch[0]) == 0.0 and bool(scene.shutter_closed()[0])
          and s <= 0.02 and not ok)

    # =========================== 7. door open only ==========================================
    env.reset(seed=51)
    step(10)
    place_shutter(c.x_door_clear + 0.003, settle_steps=30)
    report("door-open-only")
    s, ok = judge()
    check("door-open only: shutter fully open, brick untouched — travel credit only "
          "(0.18 <= score <= 0.22), NOT success",
          0.18 <= s <= 0.22 and not ok)

    # =========================== 8. near-miss transit (short of the seat) ===================
    env.reset(seed=61)
    step(10)
    place_shutter(c.x_door_clear + 0.003, settle_steps=20)
    place_brick(0.0, 0.095 - c.tip_half, settle_steps=40)  # tip at y ~= 0.095 < seat band
    report("short-of-seat")
    s, ok = judge()
    td = scene.tip_dock()[0]
    check("near-miss transit: door open, brick in the bay but tip short of the seat "
          "band (readback) — partial credit only (0.40 <= score <= 0.62), docked "
          "never latches, NOT success",
          float(td[1]) < c.y_seat_min - 0.004 and float(scene.dock_latch[0]) == 0.0
          and 0.40 <= s <= 0.62 and not ok)

    # =========================== 9. sideways parking, sealed inside =========================
    env.reset(seed=71)
    step(10)
    place_brick(0.0, 0.052, yaw_off=math.pi / 2, settle_steps=40)  # prongs at a side wall
    report("sideways-inside")
    s, ok = judge()
    check("sideways parking: brick fully inside with the shutter CLOSED but yawed "
          "~90 deg (prong-axis readback ~0) — NOT docked, NOT success, score <= 0.30",
          bool(scene.shutter_closed()[0]) and abs(float(scene.prong_axis_in()[0])) < 0.5
          and not bool(scene.docked()[0]) and s <= 0.30 and not ok)

    # =========================== 10. backwards dock =========================================
    env.reset(seed=81)
    step(10)
    place_brick(0.0, 0.068, yaw_off=math.pi, settle_steps=40)  # back face at the plate
    report("backwards-dock")
    s, ok = judge()
    check("backwards dock: brick parked against the plate BACK-FIRST (prongs facing "
          "the doorway, axis readback < -0.8), shutter closed — NOT docked, NOT "
          "success, score <= 0.15",
          float(scene.prong_axis_in()[0]) < -0.8 and not bool(scene.docked()[0])
          and bool(scene.shutter_closed()[0]) and s <= 0.15 and not ok)

    # =========================== 11. roof percher ===========================================
    env.reset(seed=91)
    step(10)
    wx, wy, gy = dock_to_world(0.0, 0.060)
    write_pose(scene.brick, wx, wy, c.facade_h + c.roof_t + c.brick_h / 2 + 0.002,
               quat=yaw_quat(gy))
    step(60)
    report("roof-percher")
    s, ok = judge()
    bd = scene.brick_dock()[0]
    check("roof percher: brick settled on the garage ROOF (dock-frame height "
          "readback above the lane) — no transit credit, NOT docked, NOT success, "
          "score <= 0.02",
          float(bd[2]) > c.facade_h and float(scene.entry_latch[0]) == 0.0
          and not bool(scene.docked()[0]) and s <= 0.02 and not ok)

    # =========================== 12. latched door credit survives re-close ==================
    env.reset(seed=101)
    step(10)
    place_shutter(c.x_door_clear + 0.003, settle_steps=20)
    s_open, _ = judge()
    place_shutter(0.0, settle_steps=20)
    report("door-reclosed")
    s_back, ok = judge()
    check("latched door credit: closing the shutter again leaves the latched travel "
          f"credit unchanged ({s_open:.3f} -> {s_back:.3f}), still no success "
          "(brick never entered)",
          s_open >= 0.18 and abs(s_back - s_open) < 0.02 and not ok)

    # =========================== 13-14. transit monotonicity + latch =======================
    env.reset(seed=111)
    step(10)
    place_shutter(c.x_door_clear + 0.003, settle_steps=20)  # a real transit has the door open
    place_brick(0.0, 0.35 * c.y_seat_min - c.tip_half, settle_steps=8)
    a_part = float(scene.entry_latch[0])
    place_brick(0.0, 0.80 * c.y_seat_min - c.tip_half, settle_steps=8)
    a_deep = float(scene.entry_latch[0])
    judge()
    check("monotonicity: a deeper doorway transit latches strictly more credit "
          f"({a_part:.3f} < {a_deep:.3f})", a_part + 0.10 < a_deep)
    s_deep, _ = judge()
    place_brick(0.0, -0.10, settle_steps=20)  # pull it back OUT to the apron
    report("pulled-back-out")
    s_out, ok = judge()
    check("latched transit credit: pulling the brick back OUT of the bay leaves the "
          f"latched credit unchanged ({s_deep:.3f} -> {s_out:.3f}), still no success",
          s_deep >= 0.40 and abs(s_out - s_deep) < 0.02 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.garage.data.root_state_w).all()
           and torch.isfinite(scene.shutter.data.root_state_w).all()
           and torch.isfinite(scene.brick.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shutter_garage")
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
