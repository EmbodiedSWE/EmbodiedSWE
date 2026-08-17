"""Smoke / rubric-REJECTION battery for RollMagazineScene (sim_gen task
`put_toilet_roll_on_stand_i304`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage each roll on the apron, push it under the
roof edge with a regulated force, let gravity roll it down the covered ramp into the
bay — is the acceptance evidence that the rubric ACCEPTS a correct outcome; it passes
on seeds 0/1/2). Every teleport here is instrumentation that CONSTRUCTS a wrong (or
partial) outcome and asserts the rubric REJECTS it. No probe in this battery ever
reaches success(), and a final audit asserts exactly that.

  1-2.  settle/no-NaN   — reset layout settles finite: both rolls lying on the open
                          floor uphill of the apron, magazine near its jittered
                          nominal; score ~0, no success;
  3-4.  randomization   — READBACK over 6 seeded resets: magazine position + heading
                          and both roll ground poses + yaws all vary;
  5.    null policy     — 240 idle steps -> score ~0, no success;
  6.    seed-strategy   — the seed family's outcome ("put the roll ON the stand")
                          reconstructed: the roll set down ON the magazine rolls down
                          the sloped ROOF and perches against the stop wall ABOVE the
                          bay — plan-view over the goal, rejected by the surface band
                          (never inside), score ~0;
  7.    apron-left      — a roll staged on the apron and abandoned latches only the
                          0.10 staging credit: staged is not loaded;
  8.    crooked feed    — PHYSICAL probe: a roll staged AXIS-ALONG the channel and
                          pushed with the solve's own regulated servo slides past the
                          crest and JAMS on the ramp (kinetic friction beats the 18
                          degree slope for a non-rolling orientation): inside but
                          never parked, aligned() False -> no success;
  9.    stack on apron  — a second roll placed ON TOP of a staged roll reads ~105 mm
                          above the apron — outside the surface band, not staged;
  10.   no back door    — PHYSICAL probe: a roll driven straight at the stop wall
                          from OUTSIDE the magazine (the probe asserts it really
                          moved) is blocked by the sealed low end and never reads
                          inside: the mouth is the only opening;
  11.   one-roll-only   — roll A alone parked+aligned in the bay is NOT success (both
                          rolls are the payload); only its 0.20 loaded credit latches;
  12.   settle gate     — the TRUE goal geometry with an injected uphill launch on
                          roll B is NOT success while anything moves (sustained pose-
                          stillness is load-bearing);
  13.   latched credit  — teleporting both rolls out of the goal state keeps the
                          latched 0.40 while parked() drops to False — credit never
                          evaporates, an abandoned state never becomes success;
  14.   rejection audit — success() was never True at ANY judged point;
  15.   final no-NaN    — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i304.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

OUT_FLAT = scene_mod.OUT_FLAT
OUT_CORNER = scene_mod.OUT_CORNER
THETA = scene_mod.THETA
TAN_TH = scene_mod.TAN_TH
Z_AP = scene_mod.Z_AP
X_AP0 = scene_mod.X_AP0
X_CREST = scene_mod.X_CREST
X_WALL = scene_mod.X_WALL
X_BAY = scene_mod.X_BAY
CLR = scene_mod.CLR
WALL_T_G = scene_mod.WALL_T_G


def srz(x: float) -> float:
    """Channel surface height at garage-local x (python-float twin of scene.surf_z)."""
    return Z_AP if x < X_CREST else Z_AP - (x - X_CREST) * TAN_TH


REST_H = OUT_FLAT / math.cos(THETA)  # roll center height above the ramp surface, vertical

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roll_magazine")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.60, -1.15, 1.10)) + o),
                                tuple(np.array((0.42, 0.00, 0.12)) + o),
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

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def g_frame() -> tuple[torch.Tensor, torch.Tensor, float]:
        """(env-rel pos (3,), quat (4,), yaw) of the magazine."""
        p = rel(scene.magazine)
        q = scene.magazine.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return p, q, yaw

    def local(loc) -> torch.Tensor:
        """Env-rel world position of a garage-local point."""
        p, q, _ = g_frame()
        return p + quat_apply(q.view(1, 4), torch.tensor([loc], device=device))[0]

    def locg(body) -> torch.Tensor:
        """Garage-local position (3,) of a roll."""
        q_g = scene.magazine.data.root_quat_w
        return quat_apply_inverse(q_g, body.data.root_pos_w
                                  - scene.magazine.data.root_pos_w)[0]

    QY90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device)  # tube local +z -> local +x
    QZ90 = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
                        device=device)

    def q_across() -> torch.Tensor:
        """Quat: tube axis ACROSS the channel (garage +y)."""
        _p, q, _ = g_frame()
        return quat_mul(q.view(1, 4), quat_mul(QZ90.view(1, 4), QY90.view(1, 4)))[0]

    def q_along() -> torch.Tensor:
        """Quat: tube axis ALONG the channel (garage +x) — the crooked orientation."""
        _p, q, _ = g_frame()
        return quat_mul(q.view(1, 4), QY90.view(1, 4))[0]

    def report(tag: str) -> None:
        s, ok = judge()
        a, b = locg(scene.rolls[0]), locg(scene.rolls[1])
        st, ins = scene.staged(), scene.inside()
        pk, al = scene.parked(), scene.aligned()
        print(f"[smoke] {tag:16s} | "
              f"A=({float(a[0]):+.3f},{float(a[1]):+.3f},{float(a[2]):.3f}) "
              f"B=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"staged=({bool(st[0, 0])},{bool(st[0, 1])}) "
              f"inside=({bool(ins[0, 0])},{bool(ins[0, 1])}) "
              f"parked=({bool(pk[0, 0])},{bool(pk[0, 1])}) "
              f"aligned=({bool(al[0, 0])},{bool(al[0, 1])}) "
              f"set={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, xyz, quat=None, vel=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = (xyz if torch.is_tensor(xyz)
                      else torch.tensor([float(v) for v in xyz], device=device))
        if quat is None:
            st[:, 3] = 1.0
        elif torch.is_tensor(quat):
            st[:, 3:7] = quat
        else:
            st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        if vel is not None:
            st[:, 7:10] = (vel if torch.is_tensor(vel)
                           else torch.tensor([float(v) for v in vel], device=device))
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(body, f_w: torch.Tensor) -> None:
        """WORLD force at the CoM, expressed in the body's link frame (the house
        convention — is_global drops torques on this stack)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def cutf(body) -> None:
        wrench(body, torch.zeros(n, 3, device=device))

    def layout_sane(tag: str) -> bool:
        """Reset honesty: magazine near its jittered nominal, both rolls lying flat
        on the open floor uphill of the apron edge."""
        gp, _q, gyaw = g_frame()
        jg = c.garage_jitter + 0.006
        ok = (abs(float(gp[0]) - c.garage_pos[0]) < jg
              and abs(float(gp[1]) - c.garage_pos[1]) < jg
              and abs(gyaw) < math.radians(c.garage_yaw_deg) + 0.05)
        for k, sp in ((0, c.spawn_a), (1, c.spawn_b)):
            p = locg(scene.rolls[k])
            ji = c.roll_jitter + 0.008
            ok = ok and (abs(float(p[0]) - sp[0]) < ji and abs(float(p[1]) - sp[1]) < ji
                         and float(p[2]) < 0.06
                         and float(p[0]) < X_AP0 - OUT_CORNER)
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}", flush=True)
        return ok

    bodies = (scene.magazine, scene.rolls[0], scene.rolls[1])

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; both rolls lying on the open floor uphill of "
          "the apron, magazine near its jittered nominal", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        gp, _qg, gyaw = g_frame()
        a, b = locg(scene.rolls[0]), locg(scene.rolls[1])
        # roll yaw readback: heading of the tube axis in the world xy plane
        ez = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        ax_a = quat_apply(scene.rolls[0].data.root_quat_w[:1], ez)[0]
        yaw_a = math.atan2(abs(float(ax_a[1])), abs(float(ax_a[0])))  # folded to [0, pi/2]
        reads.append((float(gp[0]), float(gp[1]), gyaw, float(a[0]), float(a[1]),
                      float(b[0]), float(b[1]), yaw_a))
    arr = np.array(reads)
    print("[smoke] randomization readback (mag x,y,yaw | A x,y | B x,y | A axis yaw):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: magazine pose varies (spreads "
          f"x={spread[0]:.3f} y={spread[1]:.3f} yaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.25)
    check("randomization: roll spawn poses and yaws vary (A spread "
          f"({spread[3]:.3f},{spread[4]:.3f}), B spread ({spread[5]:.3f},"
          f"{spread[6]:.3f}), axis-yaw spread {spread[7]:.2f} rad), every reset sane",
          spread[3] > 0.02 and spread[4] > 0.02 and spread[5] > 0.02
          and spread[6] > 0.02 and spread[7] > 0.3 and sane)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed-strategy: roll put ON the stand ====================
    # rlbench/put_toilet_roll_on_stand's OUTCOME is the roll placed directly ON the
    # stand. Reconstruct its analog: the roll set down ON the magazine (its top —
    # the roof — is the only "on" surface over the goal). It rolls down the sloped
    # roof and perches against the stop wall directly ABOVE the bay: plan-view over
    # the goal, ~130 mm above the ramp surface -> never inside, score ~0.
    env.reset(seed=41)
    step(30)
    x0r = 0.020
    place(scene.rolls[0],
          local((x0r, 0.0, srz(x0r) + CLR + 0.014 + OUT_CORNER + 0.003)),
          quat=q_across(), settle_steps=150)
    report("roof-perch")
    a = locg(scene.rolls[0])
    dz = float(a[2]) - srz(float(a[0]))
    s, ok = judge()
    check("seed-strategy analog: the roll set ON the magazine rolls down the roof and "
          f"perches over the bay (local x={float(a[0]) * 1000:.0f} mm > bay line "
          f"{X_BAY * 1000:.0f} mm, {dz * 1000:.0f} mm above the surface, band "
          f"<={c.band_hi * 1000:.0f} mm) — never inside, score ~0, no success",
          float(a[0]) > X_BAY and dz > c.band_hi + 0.030
          and not bool(scene.inside()[0, 0]) and s <= 0.02 and not ok)

    # =========================== 7. apron-left: staged is not loaded ========================
    env.reset(seed=51)
    step(30)
    place(scene.rolls[0], local((-0.190, 0.0, Z_AP + OUT_CORNER + 0.015)),
          quat=q_across(), settle_steps=60)
    report("apron-left")
    s, ok = judge()
    check("apron-left: a roll staged on the apron and abandoned latches ONLY the "
          f"0.10 staging credit (score={s:.2f}) — no success",
          bool(scene.staged()[0, 0]) and abs(s - 0.10) < 0.005 and not ok)

    # =========================== 8. crooked feed: axis-along roll jams ======================
    # PHYSICAL probe with the solve's own regulated push: staged AXIS-ALONG (the
    # crooked orientation the instruction warns about), the roll cannot roll — past
    # the crest it SLIDES and kinetic friction (mu_k=0.5 > tan 18deg=0.32) jams it
    # on the upper ramp, far short of the bay.
    env.reset(seed=61)
    step(30)
    place(scene.rolls[0], local((-0.190, 0.0, Z_AP + OUT_CORNER + 0.015)),
          quat=q_along(), settle_steps=60)
    x_start = float(locg(scene.rolls[0])[0])
    q_g = scene.magazine.data.root_quat_w
    crossed = False
    for _i in range(600):
        p_l = quat_apply_inverse(q_g, scene.rolls[0].data.root_pos_w
                                 - scene.magazine.data.root_pos_w)
        if bool(p_l[0, 0] > X_CREST + 0.030):
            crossed = True
            break
        v_l = quat_apply_inverse(q_g, scene.rolls[0].data.root_lin_vel_w)
        f_l = torch.zeros(n, 3, device=device)
        f_l[:, 0] = (3.0 * (0.30 - v_l[:, 0])).clamp(-0.6, 1.2)
        f_l[:, 1] = (4.0 * (0.0 - p_l[:, 1]) - 1.0 * v_l[:, 1]).clamp(-0.6, 0.6)
        wrench(scene.rolls[0], quat_apply(q_g, f_l))
        env.step(no_action)
    cutf(scene.rolls[0])
    step(180)
    report("crooked-feed")
    a = locg(scene.rolls[0])
    s, ok = judge()
    check("crooked feed: the axis-along roll pushed with the solve's own servo "
          f"(moved {(float(a[0]) - x_start) * 1000:.0f} mm, past the crest) JAMS on "
          f"the ramp at local x={float(a[0]) * 1000:.0f} mm — inside but far short "
          f"of the bay ({X_BAY * 1000:.0f} mm), aligned() False, no success",
          crossed and bool(scene.inside()[0, 0])
          and float(a[0]) < X_BAY - 0.030
          and not bool(scene.aligned()[0, 0])
          and not bool(scene.parked()[0, 0]) and s <= 0.301 and not ok)

    # =========================== 9. stack on apron: the band rejects it =====================
    env.reset(seed=71)
    step(30)
    place(scene.rolls[0], local((-0.190, 0.0, Z_AP + OUT_CORNER + 0.015)),
          quat=q_across(), settle_steps=60)
    a_now = locg(scene.rolls[0])
    place(scene.rolls[1],
          local((float(a_now[0]), 0.0, Z_AP + 3 * OUT_FLAT + 0.004)),
          quat=q_across(), settle_steps=6)
    report("stacked")
    b = locg(scene.rolls[1])
    dz_b = float(b[2]) - Z_AP
    s, ok = judge()
    check("stack on apron: a roll ON TOP of the staged roll reads "
          f"{dz_b * 1000:.0f} mm above the apron (band <={c.band_hi * 1000:.0f} mm) "
          "— not staged, nothing latches for it, no success",
          dz_b > c.band_hi + 0.030 and not bool(scene.staged()[0, 1])
          and s <= 0.11 and not ok)

    # =========================== 10. no back door: the low end is sealed ====================
    env.reset(seed=81)
    step(30)
    x_out0 = X_WALL + WALL_T_G + OUT_FLAT + 0.060
    place(scene.rolls[0], local((x_out0, 0.0, OUT_FLAT + 0.002)),
          quat=q_across(), settle_steps=10)
    q_g = scene.magazine.data.root_quat_w
    for _i in range(240):
        v_l = quat_apply_inverse(q_g, scene.rolls[0].data.root_lin_vel_w)
        p_l = quat_apply_inverse(q_g, scene.rolls[0].data.root_pos_w
                                 - scene.magazine.data.root_pos_w)
        f_l = torch.zeros(n, 3, device=device)
        f_l[:, 0] = (3.0 * (-0.15 - v_l[:, 0])).clamp(-1.0, 0.6)
        f_l[:, 1] = (4.0 * (0.0 - p_l[:, 1]) - 1.0 * v_l[:, 1]).clamp(-0.6, 0.6)
        wrench(scene.rolls[0], quat_apply(q_g, f_l))
        env.step(no_action)
    cutf(scene.rolls[0])
    step(30)
    report("back-door")
    a = locg(scene.rolls[0])
    s, ok = judge()
    check("no back door: the roll driven at the stop wall from OUTSIDE moved "
          f"{(x_out0 - float(a[0])) * 1000:.0f} mm (non-vacuous) but is blocked at "
          f"local x={float(a[0]) * 1000:.0f} mm > wall {X_WALL * 1000:.0f} mm — "
          "never inside, score ~0",
          (x_out0 - float(a[0])) > 0.030 and float(a[0]) > X_WALL
          and not bool(scene.inside()[0, 0]) and s <= 0.02 and not ok)

    # =========================== 11. one-roll-only is not success ===========================
    env.reset(seed=91)
    step(30)
    xa = X_WALL - OUT_CORNER - 0.004
    place(scene.rolls[0], local((xa, 0.0, srz(xa) + REST_H + 0.002)),
          quat=q_across(), settle_steps=80)
    report("one-roll")
    s_one, ok = judge()
    check("one-roll-only: roll A alone parked+aligned in the bay is NOT success "
          f"(both rolls are the payload); only its loaded credit latches (score={s_one:.2f})",
          bool(scene.parked()[0, 0]) and bool(scene.aligned()[0, 0])
          and abs(s_one - 0.20) < 0.005 and not ok)

    # =========================== 12-13. settle gate + latched credit ========================
    # The TRUE goal geometry, but roll B arrives with an injected UPHILL launch: not
    # success while anything moves. Then both rolls are removed BEFORE the stillness
    # window can fill (this battery must never reach success) — the latched credit
    # survives, the state checks drop.
    _pg, qg1, _y1 = g_frame()
    launch = quat_apply(qg1.view(1, 4),
                        torch.tensor([[-0.30, 0.0, 0.0]], device=device))[0]
    xb = float(locg(scene.rolls[0])[0]) - 2 * OUT_FLAT * math.cos(THETA) - 0.002
    place(scene.rolls[1], local((xb, 0.0, srz(xb) + REST_H + 0.002)),
          quat=q_across(), vel=tuple(float(v) for v in launch), settle_steps=0)
    gate_ok, v_peak = False, 0.0
    for _ in range(8):
        step(1)
        s, ok = judge()
        v_b = float(scene.rolls[1].data.root_lin_vel_w[0].norm())
        v_peak = max(v_peak, v_b)
        if (bool(scene.parked()[0, 0]) and bool(scene.aligned()[0, 0])
                and v_b > 0.05 and not bool(scene.settled()[0]) and not ok):
            gate_ok = True
    report("goal-moving")
    s_before, ok = judge()
    check("settle gate: the goal geometry with roll B still moving (peak "
          f"{v_peak:.2f} m/s) is NOT success — sustained stillness is load-bearing",
          gate_ok and not ok and not bool(scene.settled()[0]))
    place(scene.rolls[0], (-0.20, 0.45, OUT_FLAT + 0.002), quat=QY90, settle_steps=0)
    place(scene.rolls[1], (-0.20, -0.45, OUT_FLAT + 0.002), quat=QY90, settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting both rolls out keeps the latched score "
          f"({s_before:.2f} -> {s_after:.2f}, loaded 0.20+0.20) while parked() drops "
          "— and an abandoned goal state never succeeds",
          abs(s_after - 0.40) < 0.005 and s_after >= s_before - 1e-4
          and not bool(scene.parked()[0].any()) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roll_magazine")
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
