"""Smoke / rubric-REJECTION battery for MugDispenserScene (sim_gen task
`put_mug_i228`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage the mug under the silo outlet, force-pull
the gate through its channel, gravity drop, hands-off settle — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and
a final audit check asserts exactly that.

  1-2.  settle/no-NaN  — reset layout settles finite: rig inside its jitter box,
                         gate seated at the closed pose, balls stacked in the bore,
                         mug upright on the ground at rest; score ~0, no success;
  3-4.  randomization  — READBACK over 6 seeded resets: rig xy + yaw vary; mug xy
                         + free yaw vary, the mug's spawn side FLIPS across seeds,
                         and the mug never spawns near the rig;
  5.    null policy    — 240 idle steps -> score ~0, no success;
  6.    seed strategy  — the seed's outcome (carry the mug and set it down on a
                         target spot) executed on open ground: earns NOTHING;
  7.    staged-only    — mug staged perfectly under the silo outlet, settled: only
                         the staged latch fires (score ~0.15), no success — the
                         balls are still sealed in the silo;
  8.    partial open   — the gate force-pulled just past first crack (readback:
                         it really moved) but short of `open_x`: the bore gap is
                         narrower than a ball, the balls stay sealed, the opened
                         latch does NOT fire, score unchanged;
  9.    spill          — the gate force-pulled fully open with the mug elsewhere:
                         opened latch fires (0.25) but every ball ends OUTSIDE the
                         mug (readback: they really fell), no success;
  10.   near miss      — two balls settled IN the mug, one beside it on the ground,
                         a genuinely settled state (stillness-streak readback):
                         not success, only 2 ball latches (~0.30);
  11.   tipped mug     — mug on its SIDE with all three balls resting in the cavity
                         (balls_in geometrically true, readback): not upright =>
                         no success, ball latches only (<= 0.46);
  12.   fly-through    — the three balls teleported DIRECTLY into the resting mug
                         (the exact success geometry, zero velocity) and judged
                         WITHOUT stepping -> NOT success (pose-jump guard);
  13.   fly-through B  — a few real frames after the teleport (streak still fresh),
                         then one ball yanked back out: success never fired at any
                         point, latched ball credit persists (~0.45), no success;
  14.   latch survives — staging credit earned on the base plate does not evaporate
                         when the mug is returned to the ground (~0.15, no success);
  15.   rejection audit— success() was never True at ANY judged point;
  16.   final no-NaN   — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_mug_i228.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_dispenser")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.30, 1.05)) + o),
                                tuple(np.array((0.30, 0.00, 0.22)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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
        s, ok = judge()
        bz = [float((b.data.root_pos_w - scene.env_origins)[0, 2])
              for b in scene.balls]
        print(f"[smoke] {tag:14s} | mug_z={float(scene._mug_z()[0]):.3f} "
              f"gate_x={float(scene.gate_open_x()[0]):.3f} "
              f"staged={bool(scene.staged_now()[0])} "
              f"balls_in={int(scene.balls_in()[0].sum())} "
              f"ball_z={['%.3f' % v for v in bz]} still={int(scene._still[0])} "
              f"latch(st/op/in)={int(scene._staged[0])}/{int(scene._opened[0])}/"
              f"{int(scene._ball_in_ever[0].sum())} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_state(obj, pos_w: torch.Tensor, quat: torch.Tensor,
                    settle_steps: int = 0) -> None:
        """One root-state write at a WORLD pose (already env-origin absolute)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w.unsqueeze(0)
        st[:, 3:7] = quat.unsqueeze(0)
        obj.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def rig_frame():
        rp = scene.rig.data.root_pos_w[0]
        rq = scene.rig.data.root_quat_w[0]
        ryaw = 2.0 * math.atan2(float(rq[3]), float(rq[0]))
        return rp, rq, ryaw

    def stage_mug(settle: int = 60) -> None:
        """Solve-P1 staged pose: mug upright on the base plate under the bore."""
        rp, rq, ryaw = rig_frame()
        off = torch.tensor([0.0, 0.0, c.base_t + c.body_h / 2 + 0.002],
                           device=device)
        pos = rp + quat_apply(rq.unsqueeze(0), off.unsqueeze(0))[0]
        myaw = ryaw + math.pi / 2
        q = torch.tensor([math.cos(myaw / 2), 0.0, 0.0, math.sin(myaw / 2)],
                         device=device)
        write_state(scene.mug, pos, q, settle_steps=settle)

    def pull_gate(target_x: float, f0: float, f_cap: float, v_des: float,
                  max_steps: int) -> None:
        """The solve's velocity-regulated bang-bang pull (probes must not walk
        latches with constant force); breaks on gate-x readback."""
        _rp, rq, _ryaw = rig_frame()
        d = quat_apply(rq.unsqueeze(0),
                       torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        d[2] = 0.0
        d = d / d.norm()
        f_mag, best, last_bump = f0, float(scene.gate_open_x()[0]), 0
        for i in range(max_steps):
            gx = float(scene.gate_open_x()[0])
            if gx >= target_x:
                break
            v_along = float((scene.gate.data.root_lin_vel_w[0] * d).sum())
            f_des = d * (f_mag if v_along < v_des else 0.0)
            scene.gate.set_external_force_and_torque(
                f_des.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if gx > best + 0.004:
                best, last_bump = gx, i
            elif i - last_bump > 240:
                f_mag = min(f_mag + 1.5, f_cap)
                last_bump = i
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def balls_to_mug(which=(0, 1, 2)) -> None:
        """Teleport the listed balls to resting positions INSIDE the mug (two on
        the floor, one nestled on top), in the mug's live body frame."""
        mp = scene.mug.data.root_pos_w[0]
        mq = scene.mug.data.root_quat_w[0]
        floor_rest = -(c.body_h / 2) + c.floor_t + c.ball_r  # -0.027 local
        locs = [(0.0115, 0.0, floor_rest + 0.002),
                (-0.0115, 0.0, floor_rest + 0.002),
                (0.0, 0.0, floor_rest + c.ball_r * math.sqrt(3.0) + 0.002)]
        for k in which:
            loc = torch.tensor(locs[k], device=device)
            pos = mp + quat_apply(mq.unsqueeze(0), loc.unsqueeze(0))[0]
            write_state(scene.balls[k], pos,
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

    def settle_until_still(max_chunks: int = 20, chunk: int = 30) -> None:
        for _ in range(max_chunks):
            step(chunk)
            if int(scene._still[0]) >= c.still_steps:
                break

    def min_ball_z() -> float:
        return min(float((b.data.root_pos_w - scene.env_origins)[0, 2])
                   for b in scene.balls)

    def max_ball_z() -> float:
        return max(float((b.data.root_pos_w - scene.env_origins)[0, 2])
                   for b in scene.balls)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.rig.data.root_state_w).all()
            and torch.isfinite(scene.gate.data.root_state_w).all()
            and torch.isfinite(scene.mug.data.root_state_w).all()
            and all(torch.isfinite(b.data.root_state_w).all()
                    for b in scene.balls))
    rp0 = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    mp0 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    still0 = (float(scene.mug.data.root_lin_vel_w[0].norm()) < c.settle_lin
              and max(float(b.data.root_lin_vel_w[0].norm())
                      for b in scene.balls) < c.settle_lin)
    check("settle: states finite, rig inside its jitter box, gate seated at the "
          "closed pose, balls stacked in the bore, mug upright on the ground, "
          "all at rest (readback)",
          bool(fin0)
          and abs(float(rp0[0]) - c.rig_pos[0]) < c.rig_jitter + 0.005
          and abs(float(rp0[1]) - c.rig_pos[1]) < c.rig_jitter + 0.005
          and abs(float(scene.gate_open_x()[0]) - c.gate_closed_x) < 0.006
          and min_ball_z() > 0.16 and max_ball_z() < 0.30
          and abs(float(mp0[2]) - c.body_h / 2) < 0.012
          and bool(scene.mug_up()[0]) and still0)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        rp1 = (scene.rig.data.root_pos_w - scene.env_origins)[0]
        rq1 = scene.rig.data.root_quat_w[0]
        ryaw1 = 2.0 * math.atan2(float(rq1[3]), float(rq1[0]))
        mp1 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        mq1 = scene.mug.data.root_quat_w[0]
        myaw1 = 2.0 * math.atan2(float(mq1[3]), float(mq1[0]))
        reads.append((float(rp1[0]), float(rp1[1]), ryaw1, float(mp1[0]),
                      float(mp1[1]), myaw1,
                      float((mp1[:2] - rp1[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (rig_x, rig_y, rig_yaw, mug_x, mug_y, "
          f"mug_yaw, sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rig xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.008 and spread[1] > 0.008 and spread[2] > 0.08)
    check("randomization: mug xy + free yaw vary, the mug's spawn side FLIPS "
          "across seeds, and the mug never spawns within 15 cm of the rig",
          spread[3] > 0.02 and spread[5] > 0.3
          and float(arr[:, 4].min()) < -0.05 and float(arr[:, 4].max()) > 0.05
          and float(arr[:, 6].min()) >= 0.15)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: carry & set down =========================
    # The seed's whole plan — carry the mug and set it down upright on a target
    # spot — executed on open ground. With no marked target and the balls still
    # sealed in the silo, it counts NOTHING.
    torch.manual_seed(41)
    env.reset()
    step(10)
    pos = torch.tensor([0.15, 0.0, c.body_h / 2 + 0.002], device=device) \
        + scene.env_origins[0]
    write_state(scene.mug, pos, ident, settle_steps=120)
    report("set-down")
    s, ok = judge()
    check("seed strategy: mug carried and set down upright on open ground (the "
          "seed's put-on-target outcome) earns nothing — no success, score <= 0.02",
          bool(scene.mug_up()[0]) and float(scene._mug_z()[0]) < 0.10
          and not ok and s <= 0.02)

    # =========================== 7. staged-only =============================================
    torch.manual_seed(51)
    env.reset()
    step(10)
    stage_mug(settle=90)
    report("staged-only")
    s, ok = judge()
    check("staged-only: mug staged upright under the silo outlet (readback) fires "
          "ONLY the staged latch — score ~0.15, no success, balls still sealed "
          "in the silo",
          bool(scene.staged_now()[0]) and bool(scene._staged[0])
          and min_ball_z() > 0.16 and not ok and 0.14 <= s <= 0.16)

    # =========================== 8. partial open ============================================
    # Pull the gate just past first crack: the bore gap (gate_x - 0.020) stays
    # narrower than a ball diameter, so nothing may drop and the opened latch (at
    # open_x) must NOT fire. The gate-x readback proves the probe actually moved
    # the gate (not a vacuous pull).
    torch.manual_seed(61)
    env.reset()
    step(10)
    gx0 = float(scene.gate_open_x()[0])
    pull_gate(target_x=0.031, f0=1.2, f_cap=5.0, v_des=0.03, max_steps=1200)
    step(45)
    report("partial-open")
    gx1 = float(scene.gate_open_x()[0])
    s, ok = judge()
    check("partial open: gate really moved (readback) but stopped short of "
          "`open_x` — bore gap < ball diameter, balls stay sealed, opened latch "
          "NOT fired, score unchanged (<= 0.02)",
          gx1 > gx0 + 0.010 and gx1 < c.open_x - 0.004
          and not bool(scene._opened[0]) and min_ball_z() > 0.16
          and not ok and s <= 0.02)

    # =========================== 9. spill: opened with the mug elsewhere ====================
    torch.manual_seed(71)
    env.reset()
    step(10)
    pull_gate(target_x=0.058, f0=1.5, f_cap=8.0, v_des=0.05, max_steps=2400)
    step(360)  # balls fall onto the base plate / ground and scatter
    report("spill")
    s, ok = judge()
    check("spill: gate pulled fully open with the mug elsewhere — opened latch "
          "fires (score ~0.25) but every ball ends OUTSIDE the mug (readback: "
          "they really fell out of the silo), no success",
          float(scene.gate_open_x()[0]) > c.open_x and bool(scene._opened[0])
          and max_ball_z() < 0.16 and int(scene.balls_in()[0].sum()) == 0
          and not ok and 0.24 <= s <= 0.27)

    # =========================== 10. near miss: 2 in, 1 out =================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    balls_to_mug(which=(0, 1))
    mp = scene.mug.data.root_pos_w[0]
    out_pos = mp.clone()
    out_pos[0] -= 0.11  # beside the mug, clear of the handle sweep and the rig
    out_pos[2] = scene.env_origins[0, 2] + c.ball_r + 0.001
    write_state(scene.balls[2], out_pos, ident)
    settle_until_still()
    report("near-miss")
    s, ok = judge()
    check("near miss: two balls settled IN the mug, one beside it on the ground — "
          "a genuinely settled state (stillness readback) is NOT success; only "
          "the two ball latches count (score ~0.30)",
          int(scene.balls_in()[0].sum()) == 2
          and int(scene._still[0]) >= c.still_steps
          and not ok and 0.29 <= s <= 0.32)

    # =========================== 11. tipped mug with the balls inside =======================
    # Mug on its SIDE (rolled 90 deg about world x, mouth toward -y), the three
    # balls resting along the lowest interior line of the cavity. balls_in() is
    # geometrically TRUE (readback) — the upright gate alone must reject it.
    torch.manual_seed(91)
    env.reset()
    step(10)
    q_roll = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
                          device=device)  # local y->z, z->-y
    mp = torch.tensor([0.12, -0.02, c.body_r + 0.002], device=device) \
        + scene.env_origins[0]
    write_state(scene.mug, mp, q_roll, settle_steps=30)
    mq = scene.mug.data.root_quat_w[0]
    mp2 = scene.mug.data.root_pos_w[0]
    for k, zk in enumerate((-0.028, -0.005, 0.018)):
        loc = torch.tensor([0.0, -0.020, zk], device=device)  # local -y = down
        pos = mp2 + quat_apply(mq.unsqueeze(0), loc.unsqueeze(0))[0]
        write_state(scene.balls[k], pos, ident)
    settle_until_still()
    report("tipped")
    s, ok = judge()
    check("tipped mug: all three balls resting in the CAVITY of a mug lying on "
          "its side (balls_in geometrically true, readback) — not upright, so "
          "NOT success; ball latches only (score <= 0.46)",
          not bool(scene.mug_up()[0]) and int(scene.balls_in()[0].sum()) == 3
          and int(scene._still[0]) >= c.still_steps
          and not ok and s <= 0.46)

    # =========================== 12-13. fly-through / teleport guard ========================
    # The exact success geometry teleported in cold: balls resting in the resting,
    # upright mug, zero velocity — judged WITHOUT stepping it must NOT be success
    # (pose-jump guard); then a few real frames (streak still fresh) and one ball
    # yanked out: latched credit persists, success never.
    torch.manual_seed(101)
    env.reset()
    step(60)  # build an (irrelevant) stillness streak first
    balls_to_mug(which=(0, 1, 2))
    in0 = int(scene.balls_in()[0].sum())
    s0, ok0 = judge()
    report("fly-through")
    check("fly-through: the three balls teleported DIRECTLY into the resting "
          "upright mug (zero velocity) judged WITHOUT stepping is NOT success "
          "(pose-jump guard)",
          in0 == 3 and bool(scene.mug_up()[0]) and not ok0)
    mid_ok = False
    for _ in range(12):  # real frames; streak (reset by the jump) stays << still_steps
        step(1)
        mid_ok = mid_ok or bool(scene.success()[0])
    ever_success[0] = ever_success[0] or mid_ok
    out_pos = torch.tensor([-0.12, 0.05, c.ball_r + 0.001], device=device) \
        + scene.env_origins[0]
    write_state(scene.balls[0], out_pos, ident, settle_steps=90)
    report("yank-out")
    s, ok = judge()
    check("fly-through B: a few real frames after the teleport, then one ball "
          "yanked back out — success never fired at any point, latched ball "
          "credit persists (~0.45), no success",
          not mid_ok and not ok and 0.40 <= s <= 0.50
          and int(scene._ball_in_ever[0].sum()) == 3)

    # =========================== 14. staged latch survives ==================================
    torch.manual_seed(111)
    env.reset()
    step(10)
    stage_mug(settle=60)
    staged_mid = bool(scene._staged[0])
    ground = torch.tensor([0.10, -0.18, c.body_h / 2 + 0.002], device=device) \
        + scene.env_origins[0]
    write_state(scene.mug, ground, ident, settle_steps=90)
    report("stage-return")
    s, ok = judge()
    check("latch survives: staging credit earned on the base plate does not "
          "evaporate when the mug returns to the ground (score ~0.15, no "
          "success)",
          staged_mid and not bool(scene.staged_now()[0])
          and 0.14 <= s <= 0.16 and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = (torch.isfinite(scene.rig.data.root_state_w).all()
           and torch.isfinite(scene.gate.data.root_state_w).all()
           and torch.isfinite(scene.mug.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all()
                   for b in scene.balls))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.mug_dispenser")
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
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
