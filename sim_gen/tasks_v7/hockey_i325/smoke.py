"""Smoke / rubric-REJECTION battery for GateGoalScene (sim_gen task `hockey_i325`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gate board force-slid up out of its channel, white
ball force-rolled through the mouth — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome as a settled state and asserts the rubric REJECTS it; no
probe in this battery ever reaches success(), and a final audit check asserts exactly
that.

  1-2. settle/no-NaN     — reset layout settles finite: board seated in the gate
                           (barring the mouth), balls at their slots, all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: white/black slot assignment
                           flips (Bernoulli side swap), per-slot xy jitter and goal yaw
                           spread are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (strike the ball
                           at the goal): the white ball is FIRED at the sealed mouth,
                           bounces/stops against the closed gate, the captive board
                           stays seated -> score ~0, no success (the gate is the
                           interlock; no shot opens it);
  7.  drop-in (roof)     — white ball released over the box rests ON THE ROOF ->
                           score ~0, no success (the roof is the second interlock);
  8.  drop-in (slot)     — gate out, white ball dropped over the open 50 mm channel
                           slot: it cannot fit through next to the mouth plane and ends
                           OUTSIDE -> NOT success (floor-level entry is the only way);
  9.  near-miss mouth    — gate out, white ball settled centred just OUTSIDE the mouth
                           plane -> NOT success, score <= 0.35;
  10. near-miss line     — gate out, white ball settled just past the mouth plane but
                           SHORT of the inside line (u = +0.02 < 0.045) -> NOT success
                           (the inside line is the load-bearing tolerance);
  11. wrong object       — board seated, BLACK ball constructed inside the chamber,
                           white untouched -> score ~0, no success (color
                           identification is load-bearing);
  12. exclusion clause   — gate out, white AND black both settled inside: every white
                           gate passes, the black clause alone rejects -> NOT success,
                           score <= 0.75;
  13. latched credit     — teleporting the white ball back OUT afterwards leaves the
                           latched score unchanged (credit does not evaporate), still
                           no success;
  14. beside-wall        — white ball settled against the chamber's OUTSIDE wall at an
                           in-range depth (u in band, v outside) -> NOT success (goal-
                           frame math, not world coordinates, judges containment);
  15. rejection audit    — success() was never True at ANY judged point;
  16. final no-NaN       — all task-object states finite at the end.

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
    env = ENVS.get("simgen.gate_goal")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -0.95, 0.80)) + o),
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

    def report(tag: str) -> None:
        wl = scene._goal_local(scene.white)[0]
        bl = scene._goal_local(scene.board)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | white_loc=({float(wl[0]):+.3f},{float(wl[1]):+.3f},"
              f"{float(wl[2]):.3f}) board_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) in_gate={bool(scene._board_in_gate()[0])} "
              f"gate_out={bool(scene._gate_out[0])} app={float(scene._app_max[0]):.3f} "
              f"inside={bool(scene._inside(scene.white)[0])} "
              f"black_in={bool(scene._inside(scene.black)[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, u: float, v: float, z: float, vel_u: float = 0.0) -> None:
        """Teleport `body` to a goal-local point of the goal's CURRENT pose (probe
        constructor: builds inside/on-top relations directly, walls notwithstanding),
        optionally with an initial velocity along the goal axis (the seed's shot)."""
        g_pos = scene.goal.data.root_pos_w
        g_quat = scene.goal.data.root_quat_w
        loc = torch.tensor([u, v, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = g_pos + quat_apply(g_quat, loc)
        st[:, 3] = 1.0
        if vel_u:
            axis = quat_apply(g_quat, torch.tensor([1.0, 0.0, 0.0],
                                                   device=device).expand(n, 3))
            st[:, 7:10] = axis * vel_u
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def clear_gate() -> None:
        """Probe constructor: park the board flat at the free patch (what solve.py
        earns through the channel, constructed here to build post-extraction states)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = c.park_spot[0], c.park_spot[1]
        st[:, 2] = c.board_t / 2 + 0.004
        h = math.pi / 4
        st[:, 3], st[:, 5] = math.cos(h), math.sin(h)
        st[:, 0:3] += scene.env_origins
        scene.board.write_root_state_to_sim(st, all_ids)

    def obj_xy(body) -> tuple[float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1])

    def goal_yaw() -> float:
        q = scene.goal.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.goal.data.root_state_w).all()
                    and torch.isfinite(scene.board.data.root_state_w).all()
                    and torch.isfinite(scene.white.data.root_state_w).all()
                    and torch.isfinite(scene.black.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    wz = float((scene.white.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.white.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.board.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, board seated IN the gate (mouth barred), white ball "
          "at its slot, all still",
          finite_all() and bool(scene._board_in_gate()[0])
          and abs(wz - c.ball_r) < 0.012 and still
          and not bool(scene._inside(scene.white)[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        wx, wy = obj_xy(scene.white)
        bx, by = obj_xy(scene.black)
        reads.append((wx, wy, bx, by, 1.0 if wy > 0 else 0.0, goal_yaw()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (white_x, white_y, black_x, black_y, "
          f"white_on_left, goal_yaw_deg):\n{arr}", flush=True)
    flags = arr[:, 4]
    check("randomization: white/black slot assignment flips across seeded resets AND "
          "the two balls always take opposite slots (readback)",
          0.0 < flags.mean() < 1.0
          and all((r[1] > 0) != (r[3] > 0) for r in reads))
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 5].max() - arr[:, 5].min())
    check("randomization: per-slot spawn jitter (> 4 mm) and goal yaw spread (> 2 deg) "
          "are real (readback)", jit > 0.004 and yaw_spread > 2.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: shoot the ball at the goal ===============
    # The seed's plan — propel the ball at the goal mouth — executed here: the white
    # ball is FIRED along the goal axis at the SEALED mouth. It stops against the
    # closed gate; the captive board stays seated; nothing scores.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.white, -0.30, 0.0, c.ball_r + 0.002, vel_u=1.0)
    step(240)
    report("seed-strategy")
    wl = scene._goal_local(scene.white)[0]
    s, ok = judge()
    check("seed strategy: white ball FIRED at the sealed mouth bounces off the closed "
          "gate and ends OUTSIDE, board still seated — score ~0 (<= 0.02), no success "
          "(no shot opens the gate)",
          bool(scene._board_in_gate()[0]) and float(wl[0]) < 0.0
          and not bool(scene._gate_out[0]) and not ok and s <= 0.02)

    # =========================== 7. drop-in attempt: on the roof ============================
    torch.manual_seed(51)
    env.reset()
    step(30)
    roof_top = c.roof_z + c.roof_t
    place_local(scene.white, 0.08, 0.0, roof_top + c.ball_r + 0.004)
    step(180)
    report("on-roof")
    wl = scene._goal_local(scene.white)[0]
    s, ok = judge()
    check("drop-in (roof): white ball released over the box rests ON THE ROOF (or "
          "rolls off outside) — never inside, score ~0 (<= 0.02), no success",
          not bool(scene._inside(scene.white)[0]) and not bool(scene._gate_out[0])
          and not ok and s <= 0.02)

    # =========================== 8. drop-in attempt: the open channel slot ==================
    # Gate out; the ball dropped over the open 50 mm slot next to the mouth plane
    # cannot fit through into the chamber — it ends up OUTSIDE the inside line.
    torch.manual_seed(61)
    env.reset()
    step(30)
    clear_gate()
    step(20)
    place_local(scene.white, -0.025, 0.0, 0.20)
    step(240)
    report("slot-drop")
    wl = scene._goal_local(scene.white)[0]
    s, ok = judge()
    check("drop-in (slot): gate out, white ball dropped over the open 50 mm channel "
          "slot lands OUTSIDE the inside line (the slot is narrower than the ball) — "
          "NOT success",
          float(wl[0]) < c.in_u_min and not bool(scene._inside(scene.white)[0])
          and not ok)

    # =========================== 9. near-miss: settled just outside the mouth ===============
    torch.manual_seed(71)
    env.reset()
    step(30)
    clear_gate()
    step(20)
    place_local(scene.white, -0.05, 0.0, c.ball_r + 0.002)
    step(120)
    report("near-miss-mouth")
    s, ok = judge()
    check("near-miss mouth: gate out, white ball settled centred just OUTSIDE the "
          "mouth plane — NOT success, score <= 0.35",
          not bool(scene._inside(scene.white)[0]) and not ok and s <= 0.35)

    # =========================== 10. near-miss: short of the inside line ====================
    torch.manual_seed(81)
    env.reset()
    step(30)
    clear_gate()
    step(20)
    place_local(scene.white, 0.02, 0.0, c.ball_r + 0.002)
    step(120)
    report("near-miss-line")
    wl = scene._goal_local(scene.white)[0]
    s, ok = judge()
    check("near-miss line: white ball settled past the mouth plane but SHORT of the "
          "inside line (u < 0.045) — NOT success, score <= 0.35 (the inside line is "
          "the load-bearing tolerance)",
          0.0 < float(wl[0]) < c.in_u_min and not bool(scene._inside(scene.white)[0])
          and not ok and s <= 0.35)

    # =========================== 11. wrong object: black ball inside ========================
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_local(scene.black, 0.10, 0.0, c.ball_r + 0.004)
    step(150)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: BLACK ball inside the chamber, white untouched, board still "
          "seated — score ~0 (<= 0.02), no success (color identification is "
          "load-bearing)",
          bool(scene._inside(scene.black)[0]) and not ok and s <= 0.02)

    # =========================== 12. exclusion clause =======================================
    # Gate out, white AND black both settled inside: every white gate passes; the
    # black clause alone must reject.
    torch.manual_seed(101)
    env.reset()
    step(30)
    clear_gate()
    step(20)
    place_local(scene.black, 0.10, 0.045, c.ball_r + 0.004)
    place_local(scene.white, 0.10, -0.035, c.ball_r + 0.004)
    step(240)
    report("both-inside")
    s12, ok = judge()
    w_still = float(scene.white.data.root_lin_vel_w[0].norm()) < c.settle_speed
    check("exclusion clause: white AND black both settled inside — all white gates "
          "pass, the black clause alone rejects: NOT success, score <= 0.75",
          bool(scene._inside(scene.white)[0]) and bool(scene._inside(scene.black)[0])
          and w_still and not ok and s12 <= 0.75)

    # =========================== 13. latched credit survives regression =====================
    place_world(scene.white, c.slot_a[0], c.slot_a[1], c.ball_r + 0.002)
    step(60)
    report("regressed")
    s13, ok = judge()
    check("latched credit: teleporting the white ball back OUT of the chamber leaves "
          f"the latched score unchanged ({s12:.3f} -> {s13:.3f}), still no success",
          abs(s13 - s12) < 1e-3 and not bool(scene._inside(scene.white)[0]) and not ok)

    # =========================== 14. beside-wall (frame math) ===============================
    torch.manual_seed(111)
    env.reset()
    step(30)
    clear_gate()
    step(20)
    place_local(scene.white, 0.10, 0.125, c.ball_r + 0.002)
    step(120)
    report("beside-wall")
    wl = scene._goal_local(scene.white)[0]
    s, ok = judge()
    check("beside-wall: white ball settled against the chamber's OUTSIDE wall at an "
          "in-range depth — v gate rejects: NOT success, score <= 0.45",
          abs(float(wl[1])) > c.in_w / 2 and not bool(scene._inside(scene.white)[0])
          and not ok and s <= 0.45)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gate_goal")
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
