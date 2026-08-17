"""Smoke / rubric-REJECTION battery for GantryStampScene (sim_gen task
`draw_svg_i388`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-servo the two axes and spring-press
each red pad — is the acceptance evidence). Teleports here are instrumentation
that PLACE the machine (a transport the axis-servo delivers in solve) so a
press probe can be run at a chosen spot; every press itself is a real applied
force through the real spring and real contact. Rejection probes construct
wrong or partial outcomes and assert the rubric REJECTS them; a few probes
construct the genuine end state on purpose (the acceptance run and its
restore-flips) — every other judged point must stay success()=False and a
final audit asserts exactly that.

  1-2. settle/no-NaN      — reset settles finite; the return spring HOLDS the
                            stylus tip parked ~19 mm above the pads; score ~0;
  3-5. randomization      — READBACK over 8 seeded resets: frame xy+yaw vary;
                            the pad layout varies; the initial bridge/carriage
                            configuration varies;
  6.  null policy         — 400 idle steps -> score ~0, no success, tip does
                            not drift down (the spring holds against gravity);
  7.  seed strategy fails — the seed task's strategy (sweep the marker along a
                            path ACROSS every target) transplanted here: the
                            tip carried over all five pads at traverse height
                            scores 0 — nothing latches without a full press;
  8.  spring is real      — a full 6 N press over the EMPTY slot drives the tip
                            its full stroke down to the bare board (through
                            stamp depth, no pad there -> no latch), and on
                            release the spring retracts it back past retract_z;
  9.  near miss (xy)      — a full-depth press 30 mm from a red pad's centre
                            (tip ON the pad's edge, outside stamp_r): rejected;
  10. near miss (depth)   — a 1.2 N partial press dead-centre over a red pad
                            descends several mm but stalls on the spring above
                            stamp depth: rejected;
  11. real stamp          — the same spot pressed at 6 N latches: score 0.25,
                            still no success (2 reds remain);
  12. latches persist     — parking the machine far away afterwards does not
                            erase earned score (latched rubric);
  13. wrong object        — a full press on a GRAY pad latches the FOUL;
  14. foul irreversible   — the remaining reds then stamped honestly: all 3
                            red latched, yet success stays False and the score
                            stays capped at 0.75 — the gray press can never be
                            undone;
  15. acceptance          — fresh seed, three honest 6 N presses on the three
                            red pads, release, hands off -> success TRUE, 1.0;
  16. retract clause      — the stylus held pressed down in the accepted state:
                            success flips FALSE; released -> TRUE again;
  17. settle gate         — the bridge kicked in the accepted state and judged
                            immediately: NOT success (must be at rest); settled
                            -> TRUE again;
  18. rejection audit     — success() was never True at any judged point EXCEPT
                            the constructed acceptance probes;
  19. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=16)
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
        BRIDGE_Z, CAR_Z, N_RED, SLOT_X, SLOT_Y, STYLUS_Z0, TIP_OFF, _qapply,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BRIDGE_Z, CAR_Z, N_RED, SLOT_X, SLOT_Y, STYLUS_Z0, TIP_OFF, _qapply,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

PRESS_F = 6.0  # the solve's full press (N)
PARTIAL_F = 1.2  # spring stalls this press ~7 mm above stamp depth


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gantry_stamp")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.00, -0.85, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.15)) + o),
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

    def tipz() -> float:
        return float(scene.machine_coords()[2][0])

    def report(tag: str, s: float, ok: bool) -> None:
        bx, cy, tz = scene.machine_coords()
        print(f"[smoke] {tag:18s} | bridge={float(bx[0]) * 1000:+7.1f}mm "
              f"car={float(cy[0]) * 1000:+7.1f}mm tip_z={float(tz[0]) * 1000:6.1f}mm "
              f"red={int(scene.stamped_red()[0])}/3 foul={bool(scene.fouled()[0])} "
              f"retracted={bool(scene.stylus_retracted()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_machine(bx: float, cy: float, settle_steps: int = 40) -> None:
        """Instrumented transport: write the WHOLE machine chain coherently at
        the given machine coordinates, stylus retracted, in the frame's pose
        (a placement the solve's axis-servo delivers with real pushes)."""
        fp = scene.frame.data.root_pos_w
        q = scene.frame.data.root_quat_w
        for body, local in ((scene.bridge, (bx, 0.0, BRIDGE_Z)),
                            (scene.carriage, (bx, cy, CAR_Z)),
                            (scene.stylus, (bx + TIP_OFF, cy, STYLUS_Z0))):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = fp + _qapply(q, torch.tensor(local, device=device).expand(n, 3))
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def press(force_n: float, steps: int) -> float:
        """Real physics press: a constant downward force on the stylus head
        against its return spring. Returns the deepest tip height reached."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = -force_n
        min_tz = 1e9
        for _ in range(steps):
            try:
                scene.stylus.set_external_force_and_torque(
                    f, zero_wrench, env_ids=all_ids, is_global=True)
            except TypeError:  # older API: world-frame is the default
                scene.stylus.set_external_force_and_torque(f, zero_wrench,
                                                           env_ids=all_ids)
            step(1)
            min_tz = min(min_tz, tipz())
        scene.stylus.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                   env_ids=all_ids)
        return min_tz

    def stamp_at(bx: float, cy: float) -> float:
        """Place, full-press, release, wait for the spring to retract."""
        place_machine(bx, cy)
        min_tz = press(PRESS_F, 90)
        step(90)
        return min_tz

    def pad_xy(i: int) -> tuple[float, float]:
        lay = scene.pad_layout()[0]
        return float(lay[i, 0]), float(lay[i, 1])

    def free_slot_xy() -> tuple[float, float]:
        """Frame-local centre of the one slot the 5 pads did NOT occupy."""
        lay = scene.pad_layout()[0]  # (5,2)
        slots = torch.tensor([(sx, sy) for sy in SLOT_Y for sx in SLOT_X],
                             device=device, dtype=lay.dtype)
        d = (slots.unsqueeze(0) - lay.unsqueeze(1)).norm(dim=-1)  # (5,6)
        occupied = set(d.argmin(dim=1).tolist())
        free = [i for i in range(6) if i not in occupied]
        assert len(free) == 1, f"slot bookkeeping broken: {occupied}"
        return float(slots[free[0], 0]), float(slots[free[0], 1])

    def all_finite() -> bool:
        bodies = [scene.frame, scene.bridge, scene.carriage, scene.stylus] + scene.pads
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(150)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite; the return spring HOLDS the stylus parked "
          f"(tip_z={tipz() * 1000:.1f}mm >= retract {c.retract_z * 1000:.0f}mm, "
          f"pads top {c.pad_top * 1000:.1f}mm)",
          all_finite() and bool(scene.stylus_retracted()[0])
          and tipz() > c.pad_top + 0.012)
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.01 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        yaw = 2.0 * math.atan2(float(scene.frame.data.root_quat_w[0, 3]),
                               float(scene.frame.data.root_quat_w[0, 0]))
        r0x, r0y = pad_xy(0)
        bx, cy, _tz = scene.machine_coords()
        reads.append((float(fp[0]), float(fp[1]), yaw, r0x, r0y,
                      float(bx[0]), float(cy[0])))
    arr = np.array(reads)
    print("[smoke] randomization readback (frame_x, frame_y, yaw, red0_x, red0_y, "
          f"bridge0, car0):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: frame pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.15)
    check("randomization: the pad layout varies (readback: red0 frame-local moves "
          f"dx={spread[3] * 1000:.0f}mm dy={spread[4] * 1000:.0f}mm)",
          max(spread[3], spread[4]) > 0.05)
    check("randomization: the initial bridge/carriage configuration varies "
          f"(readback: dbridge={spread[5] * 1000:.0f}mm dcar={spread[6] * 1000:.0f}mm)",
          spread[5] > 0.03 and spread[6] > 0.03)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(60)
    tz0 = tipz()
    step(400)
    s, ok = judge()
    report("null-policy", s, ok)
    check("null policy: score ~0, no success, and the tip does not drift down after "
          f"400 idle steps (score={s:.3f}, drift={(tz0 - tipz()) * 1000:.2f}mm)",
          s <= 0.01 and not ok and abs(tz0 - tipz()) < 0.004)

    # =========================== 7. the seed's strategy fails ===============================
    # draw_svg's plan — sweep the marker along a path across every target — done
    # here with the captive tool: carry the tip over ALL five pads at traverse
    # height. Nothing may latch.
    min_traverse_tz = 1e9
    for i in range(5):
        px, py = pad_xy(i)
        place_machine(px - TIP_OFF, py, settle_steps=25)
        min_traverse_tz = min(min_traverse_tz, tipz())
    s, ok = judge()
    report("seed-traverse", s, ok)
    check("seed strategy fails: the tip swept OVER all five pads at traverse height "
          f"(lowest tip_z={min_traverse_tz * 1000:.1f}mm, stamp depth needs "
          f"<={(c.pad_top + c.stamp_dz) * 1000:.1f}mm) latches nothing: score 0",
          s <= 0.01 and not ok and int(scene.stamp_latch[0].sum()) == 0
          and min_traverse_tz > c.pad_top + c.stamp_dz + 0.010)

    # =========================== 8. the spring mechanism is real ============================
    fx, fy = free_slot_xy()
    place_machine(fx - TIP_OFF, fy)  # tip over the free slot centre
    min_tz = press(PRESS_F, 90)
    tz_pressed = tipz()
    step(100)  # release: the spring must retract the stylus
    s, ok = judge()
    report("empty-press", s, ok)
    check("spring is real: a full 6 N press over the EMPTY slot drives the tip its "
          f"full stroke to the bare board (min tip_z={min_tz * 1000:.1f}mm, through "
          f"stamp depth {(c.pad_top + c.stamp_dz) * 1000:.1f}mm — no pad there, no "
          f"latch), and on release the spring parks it back "
          f"(tip_z {tz_pressed * 1000:.1f} -> {tipz() * 1000:.1f}mm)",
          min_tz < c.pad_top - 0.002 and int(scene.stamp_latch[0].sum()) == 0
          and not ok and bool(scene.stylus_retracted()[0]))

    # =========================== 9. near miss: right press, wrong spot ======================
    r0x, r0y = pad_xy(0)
    off = 0.030  # 30 mm: the tip lands ON the pad's edge but outside stamp_r
    place_machine(r0x + off - TIP_OFF, r0y)
    min_tz = press(PRESS_F, 90)
    step(90)
    s, ok = judge()
    report("near-miss-xy", s, ok)
    check(f"near miss (xy): a full-depth press {off * 1000:.0f}mm from the red pad "
          f"centre (outside stamp_r={c.stamp_r * 1000:.0f}mm; tip did descend to "
          f"{min_tz * 1000:.1f}mm) latches nothing",
          min_tz < c.pad_top + c.stamp_dz + 0.003
          and int(scene.stamp_latch[0].sum()) == 0 and not ok)

    # =========================== 10. near miss: right spot, partial press ===================
    place_machine(r0x - TIP_OFF, r0y)
    min_tz = press(PARTIAL_F, 120)
    step(90)
    s, ok = judge()
    report("partial-press", s, ok)
    check(f"near miss (depth): a {PARTIAL_F:.1f} N partial press dead-centre over "
          f"the red pad descends to {min_tz * 1000:.1f}mm — real travel "
          f"({(STYLUS_Z0 - 0.13 - min_tz) * 1000:.1f}mm down) but the spring stalls "
          f"it above stamp depth {(c.pad_top + c.stamp_dz) * 1000:.1f}mm: no latch",
          min_tz > c.pad_top + c.stamp_dz + 0.001 and min_tz < 0.052
          and int(scene.stamp_latch[0].sum()) == 0 and not ok)

    # =========================== 11. a real stamp latches ===================================
    min_tz = press(PRESS_F, 90)
    step(90)
    s, ok = judge()
    report("stamp-red0", s, ok)
    check("real stamp: the same spot pressed at 6 N reaches the pad face "
          f"(min tip_z={min_tz * 1000:.1f}mm) and LATCHES: score {s:.3f} = 0.25, "
          "still no success (2 reds remain)",
          bool(scene.stamp_latch[0, 0]) and abs(s - 0.25) < 1e-3 and not ok
          and not bool(scene.fouled()[0]))

    # =========================== 12. latches persist ========================================
    place_machine(-0.19, 0.0, settle_steps=80)  # park the machine far away
    s2, ok = judge()
    report("latch-persist", s2, ok)
    check("latches persist: parking the machine far away does not erase earned "
          f"score ({s:.3f} -> {s2:.3f})", s2 >= s - 1e-6 and not ok)

    # =========================== 13. wrong object: gray press fouls =========================
    g0x, g0y = pad_xy(N_RED)  # gray0
    stamp_at(g0x - TIP_OFF, g0y)
    s, ok = judge()
    report("gray-press", s, ok)
    check("wrong object: a full press on a GRAY pad latches the FOUL "
          f"(fouled={bool(scene.fouled()[0])}, score={s:.3f})",
          bool(scene.fouled()[0]) and not ok)

    # =========================== 14. the foul is irreversible ===============================
    for i in range(1, N_RED):  # stamp the remaining reds honestly
        px, py = pad_xy(i)
        stamp_at(px - TIP_OFF, py)
    step(120)
    s, ok = judge()
    report("foul-then-finish", s, ok)
    check("foul irreversible: all 3 red pads now honestly stamped "
          f"({int(scene.stamped_red()[0])}/3), stylus parked, machine still — yet "
          f"success stays False and the score stays capped at {s:.3f} <= 0.75",
          int(scene.stamped_red()[0]) == N_RED and not ok and s <= 0.75 + 1e-6
          and bool(scene.stylus_retracted()[0]))

    # =========================== 15. acceptance construct ===================================
    env.reset(seed=61)
    step(120)
    for i in range(N_RED):
        px, py = pad_xy(i)
        stamp_at(px - TIP_OFF, py)
    step(150)
    s, ok = judge_accept()
    report("accept-run", s, ok)
    check("acceptance: three honest 6 N presses on the three red pads, released, "
          f"hands off -> success TRUE, score={s:.3f} >= 0.99", ok and s >= 0.99)

    # =========================== 16. retract clause flips success ===========================
    press(PRESS_F, 50)  # hold the stylus down (over the last red pad: no foul)...
    f = torch.zeros(n, 1, 3, device=device)
    f[:, 0, 2] = -PRESS_F
    try:
        scene.stylus.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
    except TypeError:
        scene.stylus.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids)
    step(20)  # ...and JUDGE while pressed at depth
    tz_dn = tipz()
    s, ok = judge()
    report("held-down", s, ok)
    check("retract clause: the stylus held pressed down "
          f"(tip_z={tz_dn * 1000:.1f}mm < retract {c.retract_z * 1000:.0f}mm) flips "
          f"success FALSE while all latches still hold (red={int(scene.stamped_red()[0])})",
          not ok and tz_dn < c.retract_z and int(scene.stamped_red()[0]) == N_RED)
    scene.stylus.set_external_force_and_torque(zero_wrench, zero_wrench,
                                               env_ids=all_ids)
    step(150)
    s, ok = judge_accept()
    report("released", s, ok)
    check("retract clause: released — the spring re-parks the stylus and success is "
          f"TRUE again (16's rejection was the retract clause and nothing else)", ok)

    # =========================== 17. settle gate ============================================
    st = scene.bridge.data.root_state_w.clone()
    kick = _qapply(scene.frame.data.root_quat_w,
                   torch.tensor([[0.35, 0.0, 0.0]], device=device).expand(n, 3))
    st[:, 7:10] = kick
    scene.bridge.write_root_state_to_sim(st, all_ids)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.bridge.data.root_lin_vel_w[0].norm())
    s, ok = judge()
    gate_ok = lv > c.settle_lin and not ok
    report("settle-gate", s, ok)
    check(f"settle gate: the bridge kicked ({lv:.2f} m/s) in the accepted state and "
          "judged immediately is NOT success (must be at rest)", gate_ok)
    step(200)
    s, ok = judge_accept()
    report("re-settled", s, ok)
    check("settle gate: the machine back at rest -> success TRUE again", ok)

    # =========================== 18-19. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gantry_stamp")
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
