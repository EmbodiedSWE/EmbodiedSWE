"""Smoke / rubric-REJECTION battery for SteleWalkScene (sim_gen task
`living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i419`)
— NullRobot, teleported/wrenched probe states, RECORDED.

This is NOT a solution (solve.py — WALK the left stele down the causeway with a
rock-and-swivel gait, zero teleports — is the acceptance evidence). Every teleport
and wrench here is instrumentation that CONSTRUCTS a wrong, partial, or forbidden
outcome and asserts the rubric answers correctly. One explicitly-marked ACCEPTANCE
construct (target teleported standing in the socket) proves the predicate is
satisfiable; it is excluded from the rejection audit. Every other judged point must
never reach success(), and a final audit asserts exactly that.

   1. settle/no-NaN       — reset layout settles finite, both steles standing at the berth;
   2. layout readback     — target on the LEFT (fixture +y), decoy RIGHT, latches all
                            False, score ~0, no success;
   3. null policy         — 240 idle steps -> score ~0, no success;
   4. hover calibration   — the target held 20 mm clear of the deck (< the 32 mm carry
                            line) for 20 substeps over the causeway: NO foul — kerb-top
                            (20 mm) and threshold (12 mm) rests can never read as carried;
   5. SEED strategy fouls — the seed's whole plan is "pick it up and carry it to the
                            receptacle": the target is lifted 80 mm clear and carried
                            down the causeway -> the carried foul fires mid-band;
   6. carry is forfeit    — the SAME fouled episode then has the target placed PERFECTLY
                            standing in the socket (in-socket readback True): success
                            stays False and the score is 0, forever — carrying forfeits;
   7. slide jam           — the other continuous strategy: a hard low push (20 N at
                            ~2 cm height, velocity-capped) slides the flat target down
                            the causeway; its leading base edge JAMS on the 20 mm kerb 1
                            face (readback: pinned short, kerb-1 latch never fires,
                            score <= 0.10 + dep only, no foul, no success);
   8. tumble is forfeit   — a large pitch torque rolls the target end-over-end past the
                            30-deg line -> topple foul, score 0, no success;
   9. wrong object        — the DECOY teleported standing in the socket (readback True):
                            success False, score ~0 — positional identity matters;
  10. near-miss           — the target perched on the socket threshold, rear base corners
                            outside the interior: settled, upright-ish, but NOT all-corners
                            -in -> no success, score capped at the 0.65 walk credit;
  11. latched credit      — the target walked-by-fiat past the departure line (0.10
                            latches), then returned to the berth: score STAYS 0.10 —
                            latches never regress, and returning earns nothing more;
  12. ACCEPTANCE          — (non-audited) the target standing centred in the socket,
                            decoy at the berth, settled: success True, score 1.0 — the
                            predicate is satisfiable exactly at the declared goal;
  13. randomization       — READBACK over 3 seeded resets (max-pairwise): causeway xy
                            and yaw and the target's world position all actually move;
  14. swap coverage       — over 10 seeded resets the LEFT body is sometimes A and
                            sometimes B (fair identity draw);
  15. rejection audit     — success() was never True at ANY audited judged point;
  16. final no-NaN        — all task-object states finite at the end.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stele_walk")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.00, -1.10, 0.85)) + o),
                                tuple(np.array((0.25, 0.00, 0.10)) + o),
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

    def tick() -> None:
        nonlocal step_i
        env.step(no_action)
        if (annot is not None and step_i % args.record_every == 0
                and len(frames) < args.max_frames):
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                frames.append(arr[..., :3].astype(np.uint8).copy())
        step_i += 1

    def step(k: int) -> None:
        for _ in range(k):
            tick()

    ever_success = [False]

    def judge(audit: bool = True) -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        if audit:
            ever_success[0] = ever_success[0] or ok
        return s, ok

    def tgt_body():
        return scene.stele_a if bool(scene.tgt_is_a[0]) else scene.stele_b

    def decoy_body():
        return scene.stele_b if bool(scene.tgt_is_a[0]) else scene.stele_a

    def report(tag: str, audit: bool = True) -> tuple[float, bool]:
        s, ok = judge(audit)
        tf = scene.target_fix()[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(scene.target_tilt_cos()[0])))))
        print(f"[smoke] {tag:18s} | tgt_fix=({float(tf[0]):+.3f},{float(tf[1]):+.3f}) "
              f"tilt={tilt:4.1f} clr={float(scene.min_corner_clear()[0]) * 1000:5.1f}mm "
              f"in_sock={bool(scene.target_in_socket()[0])} "
              f"decoy_in={bool(scene.decoy_in_socket()[0])} "
              f"fouled={bool(scene.fouled[0])} air={int(scene.air_ctr[0])} "
              f"dep={int(scene.dep_latch[0])} k1={int(scene.k1_latch[0])} "
              f"k2={int(scene.k2_latch[0])} ap={int(scene.apron_latch[0])} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)
        return s, ok

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def rz(yaw: float) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0], q[:, 3] = math.cos(yaw / 2), math.sin(yaw / 2)
        return q

    def place_fix(body, x: float, y: float, z: float, yaw: float = 0.0,
                  settle_steps: int = 30) -> None:
        """Kinematic probe placement in the CAUSEWAY FIXTURE frame (instrumentation,
        not a solution) + REAL physics steps before judging (the zero-step trap)."""
        q_fix = scene.cway.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.cway.data.root_pos_w + quat_apply(q_fix, loc)
        st[:, 3:7] = quat_mul(q_fix, rz(yaw))
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def hover_carry(body, x0: float, x1: float, clear: float, k: int) -> None:
        """Hold the stele kinematically at `clear` base-corner clearance and sweep its
        fixture-x from x0 to x1 over k substeps (pose re-written EVERY substep — a
        gripper carry, the seed's strategy, expressed as pure transport)."""
        for i in range(k):
            xi = x0 + (x1 - x0) * i / max(1, k - 1)
            q_fix = scene.cway.data.root_quat_w
            loc = torch.tensor([xi, 0.02, c.stele_h / 2 + clear], device=device).expand(n, 3)
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = scene.cway.data.root_pos_w + quat_apply(q_fix, loc)
            st[:, 3:7] = q_fix
            body.write_root_state_to_sim(st, all_ids)
            tick()

    z_stand = c.stele_h / 2 + 0.003

    # =========================== 1-2. settle + layout readback ==============================
    env.reset(seed=0)
    step(60)
    fin0 = (torch.isfinite(scene.cway.data.root_state_w).all()
            and torch.isfinite(scene.stele_a.data.root_state_w).all()
            and torch.isfinite(scene.stele_b.data.root_state_w).all())
    up_a = float(scene.tilt_cos(scene.stele_a.data.root_quat_w)[0])
    up_b = float(scene.tilt_cos(scene.stele_b.data.root_quat_w)[0])
    s0, ok0 = report("reset+settle")
    check("settle/no-NaN: reset layout finite, both steles standing upright at the berth",
          bool(fin0) and up_a > c.cos_upright and up_b > c.cos_upright)
    tf0 = scene.target_fix()[0]
    dfix0 = scene._fix_local(scene.decoy_pos())[0]
    latches0 = (bool(scene.dep_latch[0]) or bool(scene.k1_latch[0])
                or bool(scene.k2_latch[0]) or bool(scene.apron_latch[0])
                or bool(scene.fouled[0]))
    check("layout readback: target spawned LEFT (fixture +y), decoy RIGHT, no latch set, "
          f"score ~0 (tgt_y={float(tf0[1]):+.3f}, decoy_y={float(dfix0[1]):+.3f})",
          float(tf0[1]) > 0.04 and float(dfix0[1]) < -0.04 and not latches0
          and s0 < 0.001 and not ok0)

    # =========================== 3. null policy =============================================
    step(240)
    s_null, ok_null = report("null policy")
    check("null policy: 240 idle steps -> score ~0, no success",
          s_null < 0.001 and not ok_null)

    snap = scene.get_state(all_ids)
    tgt = tgt_body()

    # =========================== 4. hover calibration (below the carry line) ================
    # 20 mm clearance < air_h (32 mm): legitimate step-over heights never foul.
    hover_carry(tgt, 0.10, 0.16, 0.020, 20)
    report("low hover 20mm")
    check("hover calibration: 20 mm base-corner clearance over the causeway for 20 substeps "
          "does NOT foul (air counter never armed)",
          (not bool(scene.fouled[0])) and int(scene.air_ctr[0]) == 0)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 5-6. SEED strategy: carry = forfeit ========================
    hover_carry(tgt, 0.08, 0.45, 0.080, 48)
    report("seed carry")
    check("SEED strategy: lifting the target 80 mm clear and carrying it down the causeway "
          "fires the CARRIED foul mid-band", bool(scene.fouled[0]))
    place_fix(tgt, c.sock_x, 0.0, z_stand, settle_steps=60)
    s_carry, ok_carry = report("carry->socket")
    check("carry is forfeit: the fouled episode's target then placed PERFECTLY standing in "
          "the socket (in-socket readback True) is still score 0, success False",
          bool(scene.target_in_socket()[0]) and not ok_carry and s_carry < 0.001)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 7. slide jam on kerb 1 =====================================
    # A hard LOW push (20 N at ~2 cm height via compensating torque — is_global CoM force
    # + tau = r x F, r = 10 cm below CoM) overcomes friction WITHOUT tipping (tip needs
    # ~65 N at that height) and slides the flat stele at the kerb: the base edge jams.
    r_low = torch.tensor([0.0, 0.0, -0.10], device=device).expand(n, 3)
    for _ in range(420):
        q_fix = scene.cway.data.root_quat_w
        v_fwd = float(quat_apply_inverse(q_fix, tgt.data.root_lin_vel_w)[0, 0])
        mag = 20.0 if v_fwd < 0.12 else 0.0
        f_w = quat_apply(q_fix, torch.tensor([mag, 0.0, 0.0], device=device).expand(n, 3))
        tau = torch.cross(r_low, f_w, dim=-1)
        tgt.set_external_force_and_torque(f_w.view(n, 1, 3), tau.view(n, 1, 3),
                                         env_ids=all_ids, is_global=True)
        tick()
    tgt.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids,
                                      is_global=True)
    step(40)
    s_slide, ok_slide = report("slide jam")
    lead_x = float(scene.target_corners_fix()[0, :, 0].max())
    k1_face = c.kerb_x[0] - c.kerb_w / 2
    check("slide jam: a flat 20 N low push slides the stele but its base edge PINS on the "
          f"kerb-1 face (lead corner {lead_x:+.3f} <= {k1_face + 0.012:+.3f}), kerb-1 latch "
          "never fires, no foul, no success, score <= departure credit",
          bool(scene.dep_latch[0]) and lead_x <= k1_face + 0.012
          and not bool(scene.k1_latch[0]) and not bool(scene.fouled[0])
          and not ok_slide and s_slide <= 0.11)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 8. tumble = forfeit ========================================
    # 6 N*m of pitch torque (>> the 1.29 N*m edge-topple hump) rolls it end over end.
    tau_roll = quat_apply(scene.cway.data.root_quat_w,
                          torch.tensor([0.0, -6.0, 0.0], device=device).expand(n, 3))
    for _ in range(150):
        tgt.set_external_force_and_torque(zero_wrench,
                                          tau_roll.view(n, 1, 3),
                                          env_ids=all_ids, is_global=True)
        tick()
        if bool(scene.fouled[0]):
            break
    tgt.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids,
                                      is_global=True)
    step(60)
    s_roll, ok_roll = report("tumble")
    check("tumble is forfeit: rolling the stele end-over-end crosses the 30-deg line -> "
          "topple foul, score 0, no success",
          bool(scene.fouled[0]) and s_roll < 0.001 and not ok_roll)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 9. wrong object ============================================
    place_fix(decoy_body(), c.sock_x, 0.0, z_stand, settle_steps=40)
    s_dec, ok_dec = report("decoy in socket")
    check("wrong object: the DECOY standing in the socket (readback True) is not success "
          "and scores ~0 — only the stele that spawned on the LEFT counts",
          bool(scene.decoy_in_socket()[0]) and not ok_dec and s_dec < 0.001)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 10. near-miss on the threshold =============================
    # Perched straddling the 12 mm threshold: settles tilted-forward with the rear base
    # corners OUTSIDE the interior -> all-corners-in fails; walk credit caps at 0.65.
    place_fix(tgt, c.sock_x - c.sock_in / 2 + 0.035, 0.0,
              z_stand + c.thresh_h, settle_steps=80)
    s_nm, ok_nm = report("threshold perch")
    corners_in = bool(scene.target_in_socket()[0])
    check("near-miss: the target perched over the socket threshold, rear corners outside "
          "the interior -> in-socket False, no success, score <= the 0.65 walk-credit cap",
          (not corners_in) and (not ok_nm) and s_nm <= 0.651)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 11. latched credit / no regression =========================
    place_fix(tgt, 0.10, 0.06, z_stand, settle_steps=30)
    s_dep, _ = report("past departure")
    place_fix(tgt, 0.0, c.berth_dy, z_stand, settle_steps=30)
    s_back, ok_back = report("returned to berth")
    check("latched credit: crossing the departure line latches 0.10 and RETURNING to the "
          f"berth keeps it ({s_dep:.3f} -> {s_back:.3f}), nothing more, no success",
          s_dep >= 0.099 and abs(s_back - s_dep) < 1e-6 and s_back <= 0.11 and not ok_back)
    scene.set_state(snap, all_ids)
    step(5)

    # =========================== 12. ACCEPTANCE construct (non-audited) =====================
    place_fix(tgt, c.sock_x, 0.0, z_stand, settle_steps=60)
    s_acc, ok_acc = report("ACCEPTANCE", audit=False)
    check("acceptance construct (non-audited): the target standing centred in the socket, "
          "decoy at the berth, settled -> success True, score 1.0",
          ok_acc and s_acc > 0.999)

    # =========================== 13. randomization readback =================================
    rng_rows = []
    for sd in (11, 22, 33):
        env.reset(seed=sd)
        step(10)
        cw = (scene.cway.data.root_pos_w - scene.env_origins)[0]
        qf = scene.cway.data.root_quat_w[0]
        cyaw = math.degrees(2.0 * math.atan2(float(qf[3]), float(qf[0])))
        tp = (scene.target()[0] - scene.env_origins)[0]
        rng_rows.append((float(cw[0]), float(cw[1]), cyaw, float(tp[0]), float(tp[1])))
        print(f"[smoke] rng seed {sd}: cway=({rng_rows[-1][0]:+.3f},{rng_rows[-1][1]:+.3f}) "
              f"yaw={cyaw:+.1f}deg tgt=({rng_rows[-1][3]:+.3f},{rng_rows[-1][4]:+.3f})",
              flush=True)

    def spread(i: int) -> float:
        vals = [r[i] for r in rng_rows]
        return max(abs(a - b) for a in vals for b in vals)

    check("randomization real (3-seed max-pairwise readback): causeway xy "
          f"({spread(0) * 1000:.0f}/{spread(1) * 1000:.0f} mm), yaw ({spread(2):.1f} deg) "
          f"and target world xy ({spread(3) * 1000:.0f}/{spread(4) * 1000:.0f} mm) all move",
          max(spread(0), spread(1)) > 0.010 and spread(2) > 3.0
          and max(spread(3), spread(4)) > 0.010)

    # =========================== 14. identity swap coverage =================================
    seen = set()
    for sd in range(10):
        env.reset(seed=1000 + sd)
        seen.add(bool(scene.tgt_is_a[0]))
    check(f"swap coverage: over 10 seeded resets the LEFT stele is sometimes body A and "
          f"sometimes body B (saw {sorted(seen)})", seen == {False, True})

    # =========================== 15-16. audit + no-NaN ======================================
    step(10)
    check("rejection audit: success() was never True at any audited judged point",
          not ever_success[0])
    fin = (torch.isfinite(scene.cway.data.root_state_w).all()
           and torch.isfinite(scene.stele_a.data.root_state_w).all()
           and torch.isfinite(scene.stele_b.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stele_walk")
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
