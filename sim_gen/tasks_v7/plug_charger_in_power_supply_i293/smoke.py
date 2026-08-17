"""Smoke / rubric-REJECTION battery for TotePourDockScene (sim_gen task
`plug_charger_in_power_supply_i293`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — carry the loaded tote, pour past vertical under a
wrench hold, park the tote, nudge the bank to the mat centre — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1.  settle/no-NaN       — reset layout settles finite: bank verified FLAT on the
                            tote floor by tote-frame READBACK;
  2.  SEED strategy       — the seed task's whole plan (get the object seated inside
                            a fixture by direct grasp-and-push) terminates in exactly
                            this task's RESET state (object seated inside a fixture):
                            score ~0, no success — and the direct grasp does not even
                            exist here (mouth < palm, depth > fingers, cfg-asserted);
  3-4. randomization      — READBACK over 8 seeded resets: tote xy + yaw and mat
                            xy + yaw all vary; bank in-tote x + relative yaw and the
                            mat AZIMUTH around the tote all vary;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  sub-vertical hold   — CONTACT probe: the loaded tote is wrench-held airborne at
                            85 deg tilt (past "knocked far over", short of the
                            friction release angle ~105 deg) for 2 simulated seconds:
                            the bank slides ONTO the pour-side wall (non-vacuous,
                            readback) but NEVER leaves — freed never latches, score
                            <= 0.20 -> pouring genuinely requires going past vertical;
  7.  near-miss placement — bank flat on the floor 35 mm OUTSIDE the placed band:
                            freed credit only (0.30 <= score <= 0.40), NOT placed;
  8.  on-edge on the mat  — bank balanced ON EDGE at the mat centre: height above the
                            flat-rest band + flatness fail -> NOT placed, NOT success;
  9.  tote dumped on mat  — bank PERFECTLY placed, but the tote left standing ON the
                            mat: clearance fails -> NOT success, score <= 0.85 (the
                            "pour and drop the tote where you stand" shortcut);
  10. knocked-over tote   — the loaded tote lying on its SIDE on the floor (90 deg):
                            the bank rests on the interior wall and stays INSIDE —
                            freed never latches, score <= 0.20 -> tipping the tote
                            over on the ground is NOT a pour;
  11. hover dropper       — bank in mid-air above the mat centre (transient judged
                            probe): height band -> NOT placed, NOT success;
  12. latched credit      — freed credit survives teleporting the bank BACK inside
                            the tote (credit never evaporates, success stays gone);
  13. monotonicity        — a deeper airborne tilt (85 vs 45 deg, bank inside)
                            latches strictly more tip credit;
  14. empty-tote gate     — tilting the EMPTIED tote to 90 deg latches ZERO tip
                            credit (tip progress is gated on the bank being inside);
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.plug_charger_in_power_supply_i293.smoke --headless
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

G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tote_pour_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import (axis_angle_from_quat, quat_apply,
                                     quat_apply_inverse, quat_inv, quat_mul)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.75)) + o),
                                tuple(np.array((0.0, 0.0, 0.08)) + o),
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

    def tote_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return ((scene.tote.data.root_pos_w - scene.env_origins)[0],
                scene.tote.data.root_quat_w[0])

    def mat_pose() -> tuple[torch.Tensor, torch.Tensor]:
        return ((scene.mat.data.root_pos_w - scene.env_origins)[0],
                scene.mat.data.root_quat_w[0])

    def report(tag: str) -> None:
        bl = scene._tote_local(scene.bank.data.root_pos_w)[0]
        ml = scene._mat_local(scene.bank.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bank_tote_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) bank_mat_local=({float(ml[0]):+.3f},{float(ml[1]):+.3f},"
              f"{float(ml[2]):.3f}) tilt={math.degrees(float(scene.tote_tilt()[0])):.1f}deg "
              f"in_tote={bool(scene.bank_in_tote(c.freed_expand)[0])} "
              f"placed={bool(scene.bank_placed()[0])} clear={bool(scene.tote_clear()[0])} "
              f"latches=({float(scene.tip_latch[0]):.2f},{float(scene.freed_latch[0]):.0f},"
              f"{float(scene.place_latch[0]):.0f}) "
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

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Probe write + REAL physics steps before judging (the zero-step trap)."""
        write_pose(body, x, y, z, quat)
        step(settle_steps)

    def write_assembly(x: float, y: float, z: float, tilt_deg: float,
                       axis=(0.0, 1.0, 0.0)) -> torch.Tensor:
        """Write tote+bank as ONE rigid assembly (relative pose read back and
        preserved) at a tilted pose; returns the tote quat used."""
        r_rel = scene._tote_local(scene.bank.data.root_pos_w)[0].clone()
        q_rel = quat_mul(quat_inv(scene.tote.data.root_quat_w[0:1]),
                         scene.bank.data.root_quat_w[0:1])[0]
        half = math.radians(tilt_deg) / 2
        ax = torch.tensor(axis, device=device)
        q_t = torch.cat([torch.tensor([math.cos(half)], device=device),
                         math.sin(half) * ax])
        write_pose(scene.tote, x, y, z, quat=tuple(float(v) for v in q_t))
        p_b = torch.tensor([x, y, z], device=device) \
            + quat_apply(q_t.unsqueeze(0), r_rel.unsqueeze(0))[0]
        q_b = quat_mul(q_t.unsqueeze(0), q_rel.unsqueeze(0))[0]
        write_pose(scene.bank, float(p_b[0]), float(p_b[1]), float(p_b[2]),
                   quat=tuple(float(v) for v in q_b))
        return q_t

    def hold_tote(p_hold: torch.Tensor, q_hold: torch.Tensor, steps: int,
                  m_eff: float) -> list[float]:
        """Wrench-hold the tote at a fixed airborne pose (gravity feedforward +
        pose PD, same floating-hand proxy as solve.py); returns tilt samples."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        tilts = []
        for i in range(steps):
            p, q = tote_pose()
            v = scene.tote.data.root_lin_vel_w[0]
            w = scene.tote.data.root_ang_vel_w[0]
            f = m_eff * G * ez + 20.0 * (p_hold - p) - 6.0 * v
            fn = float(f.norm())
            if fn > 18.0:
                f = f * (18.0 / fn)
            rot = axis_angle_from_quat(quat_mul(q_hold.unsqueeze(0),
                                                quat_inv(q.unsqueeze(0))))[0]
            tq = 1.2 * rot - 0.05 * w
            tn = float(tq.norm())
            if tn > 0.6:
                tq = tq * (0.6 / tn)
            # frame law on these pods: DEFAULT call = live body frame; is_global=True
            # drags by a stale reset reference (see solve.apply_wrench). Pre-encode.
            q_now = scene.tote.data.root_quat_w[0:1]
            f_b = quat_apply_inverse(q_now, f.view(1, 3))
            t_b = quat_apply_inverse(q_now, tq.view(1, 3))
            scene.tote.set_external_force_and_torque(
                f_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                t_b.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids)
            step(1)
            if i % 40 == 0:
                tilts.append(math.degrees(float(scene.tote_tilt()[0])))
        return tilts

    # =========================== 1-2. settle / SEED strategy ================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    bl = scene._tote_local(scene.bank.data.root_pos_w)[0]
    fin0 = bool(torch.isfinite(scene.tote.data.root_state_w).all()
                and torch.isfinite(scene.bank.data.root_state_w).all()
                and torch.isfinite(scene.mat.data.root_state_w).all())
    check("settle: states finite; bank verified FLAT on the tote floor by tote-frame "
          "readback; everything settled",
          fin0 and bool(scene.settled()[0])
          and abs(float(bl[0])) < 0.010 and abs(float(bl[1])) < 0.008
          and abs(float(bl[2]) - (c.floor_t + c.bank_lz / 2)) < 0.005
          and bool(scene.bank_in_tote(c.freed_expand)[0]))
    s, ok = judge()
    check("SEED strategy: the seed's terminal state (object seated inside a fixture "
          "by direct grasp-and-push) IS this task's reset state — score ~0 "
          "(<= 0.02), no success; the direct grasp does not even exist here",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        tp, tq = tote_pose()
        mp, mq = mat_pose()
        tyaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
        myaw = 2.0 * math.atan2(float(mq[3]), float(mq[0]))
        bq = scene.bank.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(bq[3]), float(bq[0]))
        rel = math.atan2(math.sin(byaw - tyaw), math.cos(byaw - tyaw))
        blx = float(scene._tote_local(scene.bank.data.root_pos_w)[0, 0])
        az = math.atan2(float(mp[1] - tp[1]), float(mp[0] - tp[0]))
        reads.append((float(tp[0]), float(tp[1]), tyaw, float(mp[0]), float(mp[1]), myaw,
                      blx, rel, math.cos(az), math.sin(az)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (tote_x, tote_y, tote_yaw, mat_x, mat_y, "
          f"mat_yaw, bank_local_x, bank_rel_yaw, cos_az, sin_az):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: tote xy + yaw and mat xy + yaw all vary across 8 seeded "
          "resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.5
          and spread[3] > 0.05 and spread[4] > 0.05 and spread[5] > 0.5)
    check("randomization: bank in-tote x + relative yaw and the mat AZIMUTH around "
          "the tote all vary across seeded resets (readback)",
          spread[6] > 0.003 and spread[7] > 0.04 and max(spread[8], spread[9]) > 0.8)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. sub-vertical hold does NOT pour (contact) ===============
    # The loaded tote is wrench-held airborne at 85 deg — well past "knocked over",
    # short of the ~105 deg friction release angle. The bank must slide onto the
    # pour-side wall (real contact, non-vacuous readback) yet NEVER leave the tote.
    env.reset(seed=41)
    step(30)
    q_t = write_assembly(0.0, 0.0, 0.25, 85.0)
    p_hold = torch.tensor([0.0, 0.0, 0.25], device=device)
    tilts = hold_tote(p_hold, q_t, 240, c.tote_mass + c.bank_mass)
    bl = scene._tote_local(scene.bank.data.root_pos_w)[0]
    report("hold-85deg")
    s, ok = judge()
    print(f"[smoke] hold tilts sampled: {[f'{t:.1f}' for t in tilts]}", flush=True)
    scene.tote.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    check("sub-vertical hold: 2 s at ~85 deg — the bank slid onto the pour-side "
          "wall (readback, non-vacuous) but stayed INSIDE: freed never latched, "
          "score <= 0.20, no success -> the pour genuinely needs to pass vertical",
          min(tilts[1:]) > 75.0 and max(tilts) < 95.0
          and float(bl[0]) > 0.010 and bool(scene.bank_in_tote(c.freed_expand)[0])
          and float(scene.freed_latch[0]) == 0.0 and s <= 0.20 + 1e-4 and not ok)

    # =========================== 7. near-miss placement =====================================
    env.reset(seed=51)
    step(10)
    mp, mq = mat_pose()
    off = c.place_xy + 0.035
    off_w = quat_apply(mq.unsqueeze(0),
                       torch.tensor([[off, 0.0, 0.0]], device=device))[0]
    place_body(scene.bank, float(mp[0] + off_w[0]), float(mp[1] + off_w[1]),
               c.bank_lz / 2 + 0.002,
               quat=yaw_quat(2.0 * math.atan2(float(mq[3]), float(mq[0]))),
               settle_steps=60)
    report("near-miss-xy")
    s, ok = judge()
    ml = scene._mat_local(scene.bank.data.root_pos_w)[0]
    check("near-miss placement: bank flat on the floor 35 mm OUTSIDE the placed band "
          "(readback) — freed credit only (0.30 <= score <= 0.40), NOT placed, NOT "
          "success",
          max(abs(float(ml[0])), abs(float(ml[1]))) > c.place_xy
          and not bool(scene.bank_placed()[0]) and 0.30 <= s <= 0.40 and not ok)

    # =========================== 8. on-edge on the mat ======================================
    env.reset(seed=61)
    step(10)
    mp, mq = mat_pose()
    c45 = math.cos(math.pi / 4)
    place_body(scene.bank, float(mp[0]), float(mp[1]), c.bank_ly / 2 + 0.002,
               quat=(c45, c45, 0.0, 0.0),  # rolled 90 deg: thin axis horizontal
               settle_steps=60)
    report("on-edge")
    s, ok = judge()
    ml = scene._mat_local(scene.bank.data.root_pos_w)[0]
    check("on-edge: bank balanced on its long edge at the mat centre — origin above "
          "the flat-rest band / flatness fails (readback) => NOT placed, NOT "
          "success, score <= 0.40",
          (float(ml[2]) > c.place_z_hi or not bool(scene.bank_flat()[0]))
          and not bool(scene.bank_placed()[0]) and s <= 0.40 and not ok)

    # =========================== 9. tote dumped on the mat ==================================
    # Order matters (no prefix may satisfy the goal): FIRST stand the tote on the
    # mat (violating clearance), THEN place the bank perfectly.
    env.reset(seed=71)
    step(10)
    mp, mq = mat_pose()
    myaw = 2.0 * math.atan2(float(mq[3]), float(mq[0]))
    viol = quat_apply(mq.unsqueeze(0),
                      torch.tensor([[0.14, 0.10, 0.0]], device=device))[0]
    place_body(scene.tote, float(mp[0] + viol[0]), float(mp[1] + viol[1]), 0.002,
               quat=yaw_quat(myaw), settle_steps=30)
    place_body(scene.bank, float(mp[0]), float(mp[1]), c.bank_lz / 2 + 0.010,
               quat=yaw_quat(myaw + 0.4), settle_steps=90)
    report("tote-on-mat")
    s, ok = judge()
    tl = scene._mat_local(scene.tote.data.root_pos_w)[0]
    check("tote dumped on mat: bank PERFECTLY placed (readback) but the tote left "
          "standing ON the mat — clearance fails, NOT success, score <= 0.85",
          bool(scene.bank_placed()[0])
          and max(abs(float(tl[0])), abs(float(tl[1]))) < c.clear_cheb
          and not bool(scene.tote_clear()[0]) and s <= 0.85 + 1e-4 and not ok)

    # =========================== 10. knocked-over tote is not a pour ========================
    env.reset(seed=81)
    step(10)
    write_assembly(-0.05, 0.05, c.outer_y / 2 + 0.002, 90.0, axis=(1.0, 0.0, 0.0))
    step(120)
    report("side-lying")
    s, ok = judge()
    tilt = math.degrees(float(scene.tote_tilt()[0]))
    check("knocked-over tote: the loaded tote lying on its SIDE on the floor "
          "(tilt ~90 deg readback) — the bank rests on the interior wall and stays "
          "INSIDE: freed never latched, score <= 0.20, no success -> ground "
          "tipping is NOT a pour",
          85.0 < tilt < 95.0 and bool(scene.bank_in_tote(c.freed_expand)[0])
          and float(scene.freed_latch[0]) == 0.0 and s <= 0.20 + 1e-4 and not ok)

    # =========================== 11. hover dropper (transient probe) ========================
    env.reset(seed=91)
    step(10)
    mp, mq = mat_pose()
    place_body(scene.bank, float(mp[0]), float(mp[1]), 0.060,
               quat=(1.0, 0.0, 0.0, 0.0), settle_steps=3)  # transient judged probe
    report("hover")
    s, ok = judge()
    ml = scene._mat_local(scene.bank.data.root_pos_w)[0]
    check("hover dropper: bank in mid-air above the mat centre — origin above the "
          "flat-rest band (readback) => NOT placed, NOT success",
          float(ml[2]) > c.place_z_hi and not bool(scene.bank_placed()[0]) and not ok)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=101)
    step(10)
    mp, mq = mat_pose()
    out_w = quat_apply(mq.unsqueeze(0),
                       torch.tensor([[0.14, 0.0, 0.0]], device=device))[0]
    place_body(scene.bank, float(mp[0] + out_w[0]), float(mp[1] + out_w[1]),
               c.bank_lz / 2 + 0.002, settle_steps=30)
    s_out, _ = judge()
    tp, tq = tote_pose()
    tyaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
    in_pos = torch.tensor([0.0, 0.0, c.floor_t + c.bank_lz / 2 + 0.0015], device=device)
    p_in = tp + quat_apply(tq.unsqueeze(0), in_pos.unsqueeze(0))[0]
    place_body(scene.bank, float(p_in[0]), float(p_in[1]), float(p_in[2]),
               quat=yaw_quat(tyaw), settle_steps=30)
    report("re-caged")
    s_back, ok = judge()
    check("latched credit: freed credit survives teleporting the bank BACK inside "
          f"the tote ({s_out:.3f} -> {s_back:.3f}, bank verified back inside), "
          "still no success",
          s_out >= 0.30 and abs(s_back - s_out) < 0.02
          and bool(scene.bank_in_tote(c.freed_expand)[0]) and not ok)

    # =========================== 13. tip monotonicity =======================================
    env.reset(seed=111)
    step(10)
    write_assembly(0.0, 0.0, 0.30, 45.0)
    step(2)
    a_part = float(scene.tip_latch[0])
    write_assembly(0.0, 0.0, 0.30, 85.0)
    step(2)
    a_deep = float(scene.tip_latch[0])
    judge()
    check("monotonicity: a deeper airborne tilt with the bank inside latches "
          f"strictly more tip credit ({a_part:.3f} < {a_deep:.3f})",
          a_part >= 0.30 and a_part + 0.20 < a_deep)

    # =========================== 14. empty-tote gate ========================================
    env.reset(seed=121)
    step(10)
    write_pose(scene.bank, 0.40, -0.40, c.bank_lz / 2 + 0.002)  # bank far away, on the floor
    step(20)  # freed latches (the bank is out) — but tip must stay gated
    write_assembly_tilt = write_assembly  # noqa: F841  (bank far: only the tote write matters)
    half = math.radians(90.0) / 2
    write_pose(scene.tote, 0.0, 0.0, 0.30,
               quat=(math.cos(half), 0.0, math.sin(half), 0.0))
    step(3)
    report("empty-tilt")
    s, ok = judge()
    tilt = math.degrees(float(scene.tote_tilt()[0]))
    check("empty-tote gate: tilting the EMPTIED tote to ~90 deg (readback) latches "
          "ZERO tip credit — tip progress is gated on the bank being inside; "
          "score <= 0.40, no success",
          tilt > 80.0 and float(scene.tip_latch[0]) == 0.0 and s <= 0.40 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.tote.data.root_state_w).all()
           and torch.isfinite(scene.bank.data.root_state_w).all()
           and torch.isfinite(scene.mat.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tote_pour_dock")
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
