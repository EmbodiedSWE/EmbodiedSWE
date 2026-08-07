"""Smoke / rubric-REJECTION battery for SpringBayScene (sim_gen task
`living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i38`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the can in, press it into the spring plunger,
release — is the acceptance evidence). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it, plus physics probes that prove the mechanism is real (the spring
genuinely restores, and genuinely finishes the seating from stored energy alone).
Two probes DO construct the genuine end state on purpose — the acceptance construct
and the decoy-removed flip — every other judged point must stay success()=False and
a final audit asserts exactly that.

  1-2. settle/no-NaN      — reset settles finite; plunger at REST (~0 compression),
                            cans upright on the floor; score ~0, no success;
  3-4. randomization      — READBACK over 8 seeded resets: bay xy + free yaw vary;
                            both can positions vary;
  5.  null policy         — 300 idle steps -> score ~0, no success (cans spawn
                            beyond the near-credit radius);
  6.  carried, not placed — the can held (posed) at the hover release pose above the
                            channel: near credit only, no success;
  7.  seed strategy       — the seed's move (drop the object into the open
                            container): released over the channel it settles PROPPED
                            on the lip, spring NOT engaged (gravity's ~4 mm is below
                            the engage gate) -> partial credit only, no success;
  8.  upright in channel  — the can STOOD upright inside the channel gap: geometry
                            windows reject it (too high, axis across the channel),
                            no compression, no success;
  9.  perched on walls    — the can laid ACROSS the side-wall tops above the channel
                            finds no resting state up there: it rolls off the bay on
                            its own (readback: ends on the ground), and at no point
                            is anything engaged — no channel/spring credit, no
                            success;
  10. spring restores     — the plunger teleported to 25 mm compression with NO can
                            returns to ~0 on its own (the drive is a real spring);
  11. acceptance construct— plunger held retracted, can laid flat in the opened gap,
                            RELEASED: the stored spring energy alone shoves the can
                            forward and seats it against the lip at ~18 mm standing
                            compression -> success TRUE (the rubric accepts the
                            genuine end state; the spring does real work);
  12. decoy on the bay    — the red can laid across the bay walls while the blue can
                            is genuinely seated: success flips FALSE (decoy clause);
  13. decoy removed       — red can back on the floor: success returns TRUE (12's
                            rejection was the decoy clause and nothing else);
  14. settle gate         — the seated can given a velocity kick and judged
                            immediately: NOT success (must be at rest); destroyed;
  15. wrong object        — the RED can seated by the same construct (blue far away):
                            rejected twice over (blue not seated, decoy on the bay);
  16. rejection audit     — success() was never True at any judged point EXCEPT the
                            two constructed acceptance probes (11 and 13);
  17. final no-NaN        — all task-object states finite at the end.

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
        H_LIP, L_CAN, PLG_FACE_X, PLG_T, R_CAN, W_CH, WALL_H, Z_F,
        _qapply, _qmul, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        H_LIP, L_CAN, PLG_FACE_X, PLG_T, R_CAN, W_CH, WALL_H, Z_F,
        _qapply, _qmul, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER_LOCAL = (0.038, 0.0, Z_F + 0.088)  # solve's release pose (bay frame)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spring_bay")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.10, -0.90, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
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

    def comp() -> float:
        return float(scene.compression()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        loc = scene.bay_local(scene.blue.data.root_pos_w)[0]
        ax = scene.can_axis_local(scene.blue)[0]
        print(f"[smoke] {tag:18s} | blue_bay=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]) - Z_F:+.3f}) ax_x={float(ax[0]):+.3f} "
              f"comp={comp() * 1000:5.1f}mm "
              f"chan={bool(scene.in_channel_loose(scene.blue)[0])} "
              f"seated={bool(scene.seated_geom()[0])} "
              f"decoy_clear={bool(scene.decoy_clear()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def bay_pose(local, extra_quat=None):
        q_bay = scene.bay.data.root_quat_w
        pos = scene.bay.data.root_pos_w + _qapply(
            q_bay, torch.tensor(local, device=device).expand(n, 3))
        q = q_bay if extra_quat is None else _qmul(q_bay, extra_quat)
        return pos, q

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

    def teleport_floor(body, x: float, y: float, settle_steps: int = 45) -> None:
        pos = torch.tensor([x, y, L_CAN / 2 + 0.003], device=device).expand(n, 3) \
            + scene.env_origins
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        teleport(body, pos, quat, settle_steps=settle_steps)

    half_pi = torch.full((n,), math.pi / 2, device=device)
    q_along_x = _qy(half_pi)  # can local +z (its axis) -> bay +x
    q_along_y = _qmul(_qz(half_pi), _qy(half_pi))  # can axis -> bay +y

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.blue)[0]) and bool(scene.settled(scene.red)[0]) \
                    and bool(scene.plunger_settled()[0]):
                break

    def seat_by_spring(body, settle_steps: int = 360) -> None:
        """ACCEPTANCE construct: hold the plunger retracted (25 mm), lay `body` flat
        in the opened gap, release — the spring alone shoves it forward and seats it
        against the lip. No force is ever applied to the can."""
        plg_pos, plg_q = bay_pose((PLG_FACE_X - PLG_T / 2 - 0.025, 0.0, Z_F + 0.030))
        teleport(scene.plunger, plg_pos, plg_q, settle_steps=0)
        pos, q = bay_pose((0.012, 0.0, Z_F + R_CAN + 0.002), q_along_x)
        teleport(body, pos, q, settle_steps=0)
        step(settle_steps)

    def all_finite() -> bool:
        bodies = [scene.bay, scene.plunger, scene.blue, scene.red]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    cans_up = all(abs(float(b.data.root_pos_w[0, 2]) - L_CAN / 2) < 0.012
                  for b in (scene.blue, scene.red))
    check("settle: all states finite, plunger at rest "
          f"(comp={comp() * 1000:.1f}mm < 4mm), cans upright on the floor",
          all_finite() and comp() < 0.004 and cans_up)
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        bp = (scene.bay.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.bay.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        bl = (scene.blue.data.root_pos_w - scene.env_origins)[0]
        rd = (scene.red.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(bp[0]), float(bp[1]), yaw, float(bl[0]), float(bl[1]),
                      float(rd[0]), float(rd[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (bay_x, bay_y, bay_yaw, blue_x, blue_y, "
          f"red_x, red_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: bay pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: both can positions vary (readback: "
          f"dblue={max(spread[3], spread[4]):.3f} dred={max(spread[5], spread[6]):.3f})",
          max(spread[3], spread[4]) > 0.05 and max(spread[5], spread[6]) > 0.05)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 6. carried, not placed =====================================
    env.reset(seed=41)
    settle_all(300)
    pos, q = bay_pose(HOVER_LOCAL, q_along_x)
    teleport(scene.blue, pos, q, settle_steps=2)  # judge while held aloft
    s, ok = judge()
    report("carried", s, ok)
    check("carried, not placed: the can POSED at the hover release pose earns near "
          f"credit only (score={s:.3f} <= 0.12), no success", s <= 0.12 and not ok)

    # =========================== 7. seed strategy (drop it in) ==============================
    # The seed's move — pick up the can and RELEASE it over the open container. Here
    # the rest gap is shorter than the can: it settles PROPPED on the lip, the spring
    # stays disengaged, and the compression gate stays shut.
    settle_all(600)
    s, ok = judge()
    report("seed-strategy", s, ok)
    loc = scene.bay_local(scene.blue.data.root_pos_w)[0]
    check("seed strategy (drop into the container): the can settles PROPPED on the "
          f"lip (comp={comp() * 1000:.1f}mm < engage {c.comp_engage * 1000:.0f}mm, "
          f"z={float(loc[2]) - Z_F:+.3f}) — partial credit only "
          f"(score={s:.3f} <= 0.27), no success",
          comp() < c.comp_engage and s <= 0.27 and not ok
          and bool(scene.in_channel_loose(scene.blue)[0]))

    # =========================== 8. upright in the channel ==================================
    env.reset(seed=51)
    settle_all(300)
    pos, q = bay_pose((0.025, 0.0, Z_F + L_CAN / 2 + 0.003))
    teleport(scene.blue, pos, q, settle_steps=180)
    s, ok = judge()
    report("upright-in-chan", s, ok)
    ax_x = float(scene.can_axis_local(scene.blue)[0, 0].abs())
    check("upright in the channel: the can STOOD in the gap is rejected (axis across "
          f"the channel ax_x={ax_x:.2f}, center too high, comp={comp() * 1000:.1f}mm) "
          f"— no success (score={s:.3f})",
          not ok and comp() < c.comp_engage and not bool(scene.seated_geom()[0]))

    # =========================== 9. perched across the wall tops ============================
    env.reset(seed=61)
    settle_all(300)
    pos, q = bay_pose((0.0, 0.0, Z_F + WALL_H + R_CAN + 0.002), q_along_y)
    teleport(scene.blue, pos, q, settle_steps=240)
    s, ok = judge()
    report("perched", s, ok)
    loc = scene.bay_local(scene.blue.data.root_pos_w)[0]
    check("perched across the wall tops: a can laid crosswise on the side-wall tops "
          "finds no resting state up there — it rolls off the bay on its own "
          f"(bay-frame z={float(loc[2]) - Z_F:+.3f}), never enters the channel, never "
          f"engages the spring — no channel/spring credit, no success (score={s:.3f})",
          not ok and s <= 0.12 and comp() < c.comp_engage
          and not bool(scene.in_channel_loose(scene.blue)[0]))

    # =========================== 10. the spring is real =====================================
    env.reset(seed=71)
    settle_all(300)
    plg_pos, plg_q = bay_pose((PLG_FACE_X - PLG_T / 2 - 0.025, 0.0, Z_F + 0.030))
    teleport(scene.plunger, plg_pos, plg_q, settle_steps=0)
    step(2)
    comp_posed = comp()
    step(180)
    s, ok = judge()
    report("spring-restore", s, ok)
    check("mechanism (the spring is real): the plunger teleported to "
          f"{comp_posed * 1000:.0f}mm compression with NO can returns to rest on its "
          f"own (settled comp={comp() * 1000:.1f}mm < 4mm)",
          comp_posed > 0.020 and comp() < 0.004 and not ok)

    # =========================== 11. acceptance construct (spring seats it) =================
    seat_by_spring(scene.blue)
    settle_all(480)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: plunger held retracted, can laid flat in the opened "
          "gap, RELEASED — the stored spring energy alone seats it against the lip "
          f"(comp={comp() * 1000:.1f}mm >= {c.comp_min * 1000:.0f}mm) -> success TRUE",
          ok and comp() >= c.comp_min and s >= 0.99)

    # =========================== 12-13. decoy clause flips success ==========================
    pos, q = bay_pose((0.0, 0.0, Z_F + WALL_H + R_CAN + 0.002), q_along_y)
    teleport(scene.red, pos, q, settle_steps=240)
    s, ok = judge()
    report("decoy-on-bay", s, ok)
    still_seated = bool(scene.seated_geom()[0]) and comp() >= c.comp_min
    check("decoy on the bay: the red can laid across the bay while the blue can is "
          f"STILL genuinely seated (comp={comp() * 1000:.1f}mm) flips success FALSE "
          "(decoy clause)", still_seated and not ok
          and not bool(scene.decoy_clear()[0]))
    teleport_floor(scene.red, 0.85, 0.55, settle_steps=120)
    s, ok = judge_accept()
    report("decoy-removed", s, ok)
    check("decoy removed: red can back on the floor -> success returns TRUE (12's "
          "rejection was the decoy clause and nothing else)", ok)

    # =========================== 14. settle gate ============================================
    # Kick the genuinely-seated can and judge IMMEDIATELY: statically correct
    # geometry + compression, but not at rest -> not success. Destroyed after.
    pos = scene.blue.data.root_pos_w
    q = scene.blue.data.root_quat_w
    teleport(scene.blue, pos, q, vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.blue.data.root_lin_vel_w[0].norm())
    av = float(scene.blue.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    gate_ok = (lv > c.settle_lin or av > c.settle_ang) and not ok
    teleport_floor(scene.blue, -0.85, 0.55, settle_steps=60)  # destroy the construct
    report("settle-gate", s, ok)
    check("settle gate: the seated can kicked (lin={:.2f} m/s, ang={:.1f} rad/s) and "
          "judged immediately is NOT success (must be at rest); state destroyed"
          .format(lv, av), gate_ok)

    # =========================== 15. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    seat_by_spring(scene.red)
    settle_all(480)
    s, ok = judge()
    report("wrong-object", s, ok)
    red_loc = scene.bay_local(scene.red.data.root_pos_w)[0]
    check("wrong object: the RED can seated by the same spring construct "
          f"(red_x={float(red_loc[0]):+.3f}, comp={comp() * 1000:.1f}mm) is rejected "
          f"twice over — blue not seated AND the decoy is on the bay "
          f"(score={s:.3f} <= 0.02)", not ok and s <= 0.02
          and not bool(scene.decoy_clear()[0]))

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "two constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spring_bay")
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
