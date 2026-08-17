"""Smoke / rubric-REJECTION battery for RockerLockScene (sim_gen task
`libero_pick_ketchup_i104`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lay the ketchup into the rocker's tray, press the
paddle, let the machine ferry it through the window, release to re-seal — is the
acceptance evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it, plus
physics probes that prove the mechanism is real (the paddle press genuinely opens the
gate and the bias genuinely re-closes it; the closed gate genuinely stops a bottle
slammed into it). One construct DOES build the genuine end state on purpose — the
acceptance construct — and three flip pairs leave and re-enter it; every other judged
point must stay success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; gate closed, axle seated, bottles
                            standing on the floor; score ~0, no success;
  3-5.  randomization     — READBACK over 8 seeded resets: rig xy + yaw vary; the
                            bottles' floor slots genuinely SWAP (both orders occur,
                            always opposite); per-bottle jitter varies;
  6.    null policy       — 300 idle steps -> score ~0, no success;
  7.    press is real     — a ~32 N paddle wrench (the solve's press) pitches the
                            beam onto its press stop (tilt readback crosses the open
                            threshold); released, the mass bias re-parks it (gate
                            closed again). The machine works both ways; no success;
  8.    closed gate       — the ketchup laid in the tray and SLAMMED up the ramp at
        blocks the ramp     the closed gate (1.6 m/s) reaches the tip zone (readback)
                            but NEVER crosses into the bin: every closed-gate
                            aperture undercuts the bottle. No success;
  9.    seed strategy     — the seed's move (carry the bottle over the receptacle and
        fails               drop): released over the bin it lands ON THE ROOF — the
                            bin is sealed; not inside, score ~0, no success;
  10.   wrong object      — the YELLOW mustard placed inside the bin (by fiat,
                            through the wall) with the ketchup left outside: the
                            identification + exclusion clauses reject it;
  11.   gate left open    — ketchup inside the bin but the beam HELD pressed
                            (kinematic hold, the "walked away holding the paddle"
                            state): gate-closed clause rejects -> no success;
  12.   acceptance        — the hold released: the bias re-parks the beam WITH the
                            cargo sealed inside (readback: tilt back over the closed
                            threshold) -> the genuine end state, success TRUE;
  13-14. mustard flips    — mustard added into the bin: success flips FALSE
                            (exclusion clause); state restored via set_state:
                            success returns TRUE;
  15-16. machine flips    — the beam lifted off its journal and laid on the open
                            floor: success flips FALSE (axle-seated + gate clauses);
                            restored via set_state: success returns TRUE;
  17.   settle gate       — the sealed ketchup kicked and judged while moving: NOT
                            success (must be at rest); resettled -> success again;
  18.   near miss (tray)  — ketchup loaded in the tray, gate closed, never
                            delivered: partial credit only, no success;
  19.   near miss (wall)  — ketchup standing on the floor against the bin's outer
                            side wall: outside the interior box, no success;
  20.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct, the flip-backs and the resettle
                            (12, 14, 16, 17b);
  21.   final no-NaN      — all task-object states finite at the end.

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
        BIN_HY, BIN_ROOF, BIN_X0, BOT_L, BOT_R, FLOOR_TOP, TILT_PRESS, TILT_REST,
        WALL_T, Z_PIV, _qapply, _qmul, _qx, _qy, encode_force,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BIN_HY, BIN_ROOF, BIN_X0, BOT_L, BOT_R, FLOOR_TOP, TILT_PRESS, TILT_REST,
        WALL_T, Z_PIV, _qapply, _qmul, _qx, _qy, encode_force,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

PADDLE_GRIP = (0.100, 0.1175, 0.060)  # beam-frame lever arm of the paddle press point
LOAD_X = -0.25  # beam-frame x of the loading bay


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_lock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.95)) + o),
                                tuple(np.array((-0.05, 0.00, 0.12)) + o),
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

    def report(tag: str, s: float, ok: bool) -> None:
        print(f"[smoke] {tag:18s} | tilt={float(scene.beam_tilt_deg()[0]):+.2f}deg "
              f"seated={bool(scene.axle_seated()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"tray={bool(scene.in_tray(scene.ketchup)[0])} "
              f"bin={bool(scene.in_bin(scene.ketchup)[0])} "
              f"mus_out={bool(scene.mustard_out()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

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

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    def rig_pose(local) -> torch.Tensor:
        return scene.rig.data.root_pos_w + _qapply(
            scene.rig.data.root_quat_w, torch.tensor(local, device=device).expand(n, 3))

    def lying_in_rig() -> torch.Tensor:
        """Cylinder axis across the rig frame (along rig-local y)."""
        return _qmul(scene.rig.data.root_quat_w,
                     _qx(torch.full((n,), math.pi / 2, device=device)))

    def load_tray(settle_steps: int = 90) -> None:
        """Lay the ketchup across the tray channel at the loading bay (solve's P1)."""
        loc = torch.tensor([LOAD_X, 0.0, FLOOR_TOP + BOT_R + 0.006],
                           device=device).expand(n, 3)
        q_beam = scene.beam.data.root_quat_w
        pos = scene.beam.data.root_pos_w + _qapply(q_beam, loc)
        quat = _qmul(q_beam, _qx(torch.full((n,), math.pi / 2, device=device)))
        teleport(scene.ketchup, pos, quat, settle_steps=settle_steps)

    def press_wrench(force_n: float, q_ref: torch.Tensor) -> None:
        """One step of the solve's emulated paddle press (mode-1 world encoding)."""
        q_now = scene.beam.data.root_quat_w
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = -force_n
        r_paddle = torch.tensor(PADDLE_GRIP, device=device).expand(n, 3)
        t_w = torch.cross(_qapply(q_now, r_paddle), f_w, dim=-1)
        f_arg = encode_force(1, q_ref, q_now, f_w)
        t_arg = encode_force(1, q_ref, q_now, t_w)
        scene.beam.set_external_force_and_torque(
            f_arg.view(n, 1, 3), t_arg.view(n, 1, 3), env_ids=all_ids, is_global=True)

    def clear_forces() -> None:
        scene.beam.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in
                   (scene.beam, scene.ketchup, scene.mustard)):
                break

    def all_finite() -> bool:
        bodies = [scene.rig, scene.beam, scene.ketchup, scene.mustard]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, gate closed, axle seated, both bottles far "
          "outside the bin on the open floor",
          all_finite() and bool(scene.gate_closed()[0])
          and bool(scene.axle_seated()[0])
          and float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 0]) < -0.3
          and float(scene._rig_local(scene.mustard.data.root_pos_w)[0, 0]) < -0.3)
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.rig.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        kl = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
        ml = scene._rig_local(scene.mustard.data.root_pos_w)[0]
        reads.append((float(rp[0]), float(rp[1]), yaw, float(kl[0]), float(kl[1]),
                      float(ml[0]), float(ml[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (rig_x, rig_y, rig_yaw, ket_x, ket_y, "
          f"mus_x, mus_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rig pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.10)
    ket_pos = arr[:, 4] > 0
    mus_pos = arr[:, 6] > 0
    check("randomization: the bottles' floor slots genuinely swap (readback: ketchup "
          f"lands on +y in {int(ket_pos.sum())}/8 seeds, both orders occur, and the "
          "mustard always takes the opposite slot)",
          0 < int(ket_pos.sum()) < 8 and bool(np.all(ket_pos != mus_pos)))
    check("randomization: per-bottle jitter varies within the slots (readback: "
          f"dket_x={spread[3]:.3f} dmus_x={spread[5]:.3f})",
          spread[3] > 0.02 and spread[5] > 0.02)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 7. the press mechanism is real =============================
    env.reset(seed=35)
    settle_all(300)
    tilt0 = float(scene.beam_tilt_deg()[0])
    q_ref = scene.beam.data.root_quat_w.clone()
    min_tilt = tilt0
    for i in range(600):
        press_wrench(32.0, q_ref)
        step(1)
        min_tilt = min(min_tilt, float(scene.beam_tilt_deg()[0]))
        if min_tilt <= -(TILT_PRESS - 0.5):
            break
    clear_forces()
    opened = min_tilt <= -(TILT_PRESS - 0.5)
    settle_all(480)
    tilt_back = float(scene.beam_tilt_deg()[0])
    s, ok = judge()
    report("press-cycle", s, ok)
    check("mechanism (the press is real): a ~32 N paddle wrench pitches the beam "
          f"onto its press stop (tilt {tilt0:+.1f} -> {min_tilt:+.1f} deg, open "
          f"threshold {-(TILT_PRESS - 0.5):+.1f}) and on release the mass bias "
          f"re-parks it (tilt back to {tilt_back:+.1f} deg, gate closed), axle still "
          "seated, no success",
          opened and bool(scene.gate_closed()[0]) and bool(scene.axle_seated()[0])
          and not ok)

    # =========================== 8. the closed gate blocks the ramp =========================
    load_tray(settle_steps=90)
    q_beam = scene.beam.data.root_quat_w
    kick_v = _qapply(q_beam, torch.tensor([1.6, 0.0, 0.0], device=device).expand(n, 3))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.ketchup.data.root_pos_w
    st[:, 3:7] = scene.ketchup.data.root_quat_w
    st[:, 7:10] = kick_v
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    max_beam_x, max_rig_x = -1.0, -1.0
    for _ in range(300):
        step(1)
        max_beam_x = max(max_beam_x,
                         float(scene._beam_local(scene.ketchup.data.root_pos_w)[0, 0]))
        max_rig_x = max(max_rig_x,
                        float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 0]))
    s, ok = judge()
    report("gate-slam", s, ok)
    check("mechanism (the seal is real): the ketchup slammed 1.6 m/s up the tray at "
          f"the CLOSED gate reaches the tip zone (max beam-frame x {max_beam_x:+.3f}) "
          f"but never crosses the window (max rig-frame x {max_rig_x:+.3f} < bin "
          f"{BIN_X0:+.3f}) — every closed-gate aperture undercuts the bottle; no "
          "success", max_beam_x > 0.05 and max_rig_x < BIN_X0
          and not bool(scene.in_bin(scene.ketchup)[0]) and not ok)

    # =========================== 9. the seed strategy fails =================================
    env.reset(seed=41)
    settle_all(300)
    drop = rig_pose((0.31, 0.0, BIN_ROOF + 0.012 + BOT_L / 2 + 0.08))
    teleport(scene.ketchup, drop, ident, settle_steps=300)
    kz = float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 2])
    s, ok = judge()
    report("roof-drop", s, ok)
    check("seed strategy fails: the bottle carried over the receptacle and dropped "
          f"(the seed's whole plan) lands ON the sealed bin's roof (rig-frame "
          f"z={kz:+.3f}), not inside — score ~0, no success (score={s:.3f})",
          not bool(scene.in_bin(scene.ketchup)[0]) and s <= 0.02 and not ok)

    # =========================== 10. wrong object ===========================================
    teleport(scene.mustard, rig_pose((0.30, 0.0, 0.012 + BOT_R + 0.004)),
             lying_in_rig(), settle_steps=150)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object (identification): the YELLOW mustard resting inside the bin "
          "(placed through the wall by fiat) with the RED ketchup outside is "
          f"rejected — exclusion clause violated, no success (score={s:.3f})",
          not ok and not bool(scene.mustard_out()[0]) and s <= 0.02)

    # =========================== 11. gate left open (kinematic hold) ========================
    env.reset(seed=51)
    settle_all(300)
    teleport(scene.ketchup, rig_pose((0.30, 0.0, 0.012 + BOT_R + 0.004)),
             lying_in_rig(), settle_steps=120)
    piv = rig_pose((0.0, 0.0, Z_PIV))
    q_press = _qmul(scene.rig.data.root_quat_w,
                    _qy(torch.full((n,), math.radians(TILT_PRESS - 0.3), device=device)))
    for _ in range(150):  # kinematic hold: the "still pressing the paddle" state
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = piv
        st[:, 3:7] = q_press
        scene.beam.write_root_state_to_sim(st, all_ids)
        step(1)
    s, ok = judge()
    report("gate-held-open", s, ok)
    tilt_held = float(scene.beam_tilt_deg()[0])
    check("out of order (gate left open): the ketchup already inside the bin but the "
          f"beam HELD pressed (tilt {tilt_held:+.1f} deg — the solver walked away "
          "without releasing the paddle) is rejected by the gate-closed clause: no "
          "success", not ok and bool(scene.in_bin(scene.ketchup)[0])
          and not bool(scene.gate_closed()[0]))

    # =========================== 12. acceptance construct ===================================
    settle_all(480)
    tilt_free = float(scene.beam_tilt_deg()[0])
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: the hold released — the mass bias re-parks the beam "
          f"WITH the cargo sealed inside (tilt {tilt_held:+.1f} -> {tilt_free:+.1f} "
          f"deg, gate closed) -> the genuine end state, success TRUE (score={s:.3f})",
          ok and s >= 0.99 and tilt_free > tilt_held + 5.0)
    snap = scene.get_state(all_ids)

    # =========================== 13-14. mustard clause flips success ========================
    teleport(scene.mustard, rig_pose((0.24, 0.0, 0.012 + BOT_R + 0.004)),
             lying_in_rig(), settle_steps=120)
    s, ok = judge()
    report("mustard-in", s, ok)
    check("exclusion clause: the mustard added into the bin beside the sealed "
          "ketchup flips success FALSE", not ok and not bool(scene.mustard_out()[0]))
    scene.set_state(snap, all_ids)
    step(30)
    s, ok = judge_accept()
    report("mustard-restored", s, ok)
    check("state restored via set_state -> success returns TRUE (13's rejection was "
          "the exclusion clause and nothing else)", ok)

    # =========================== 15-16. machine clause flips success ========================
    teleport(scene.beam, rig_pose((-0.20, 0.60, 0.10)),
             scene.rig.data.root_quat_w, settle_steps=360)
    s, ok = judge()
    report("beam-removed", s, ok)
    check("machine intact clause: the beam lifted off its journal and dumped on the "
          "open floor (cargo still sealed... but the machine wrecked) flips success "
          "FALSE (axle-seated clause)", not ok and not bool(scene.axle_seated()[0]))
    scene.set_state(snap, all_ids)
    step(30)
    s, ok = judge_accept()
    report("beam-restored", s, ok)
    check("state restored via set_state -> success returns TRUE (15's rejection was "
          "the machine clause and nothing else)", ok)

    # =========================== 17. settle gate ============================================
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.ketchup.data.root_pos_w
    st[:, 3:7] = scene.ketchup.data.root_quat_w
    st[:, 7:10] = torch.tensor([0.0, 0.0, 0.35], device=device)
    st[:, 10:13] = torch.tensor([0.0, 0.0, 6.0], device=device)
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.ketchup.data.root_lin_vel_w[0].norm())
    av = float(scene.ketchup.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    moving = lv > scene.cfg.settle_lin or av > scene.cfg.settle_ang
    settle_all(300)
    s2, ok2 = judge_accept()
    report("re-settled", s2, ok2)
    check("settle gate: the sealed bottle kicked (lin={:.2f} m/s, ang={:.1f} rad/s) "
          "and judged immediately is NOT success (must be at rest); once resettled "
          "inside, success returns".format(lv, av), moving and not ok and ok2)

    # =========================== 18-19. near misses =========================================
    env.reset(seed=61)
    settle_all(300)
    load_tray(settle_steps=120)
    s, ok = judge()
    report("tray-only", s, ok)
    check("near miss (stopped short): the ketchup loaded in the tray, gate closed, "
          f"never delivered — partial credit only (score={s:.3f}), no success",
          not ok and bool(scene.in_tray(scene.ketchup)[0]) and s <= 0.15 + 1e-4)
    teleport(scene.ketchup,
             rig_pose((0.30, BIN_HY + WALL_T + BOT_R + 0.012, BOT_L / 2 + 0.003)),
             scene.rig.data.root_quat_w, settle_steps=150)
    s, ok = judge()
    report("beside-wall", s, ok)
    check("near miss (right place, wrong side of the wall): the ketchup standing on "
          "the floor against the bin's outer side wall is outside the interior box — "
          "no success", not ok and not bool(scene.in_bin(scene.ketchup)[0]))

    # =========================== 20-21. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct, the flip-backs and the resettle",
          not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rocker_lock")
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
