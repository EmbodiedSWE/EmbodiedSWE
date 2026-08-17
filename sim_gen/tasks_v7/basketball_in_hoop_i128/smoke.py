"""Smoke / rubric-REJECTION battery for CraterRunScene (sim_gen task
`basketball_in_hoop_i128`) — NullRobot, teleported probe states + force probes, RECORDED.

This is NOT a solution (solve.py — force-servo the ball up incline A, around the turn
pocket, up incline B, over the lip — is the acceptance evidence that the rubric
ACCEPTS a correct outcome; it passes on seeds 0/1/2). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it — plus applied-force probes that prove the containment
geometry is physically real. No probe in this battery ever reaches success(), and a
final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: active ramp at the
                            workspace, twin parked in the depot, ball resting in the
                            foot bay; score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: BOTH chiralities occur,
                            structure yaw/xy jitter is physically posed and varies;
                            the ball start varies inside the foot-bay window and
                            tracks its structure;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy N/A   — rlbench/basketball_in_hoop's verb is grasp-carry-drop.
                            That strategy family does not exist here BY CONSTRUCTED
                            PREMISE: the ball (90 mm) is wider than the Franka jaw
                            span (80 mm) — readback-asserted from the live cfg, so
                            no grasp end-state can be constructed; documented N/A;
  7.  gravity adversary   — ball released mid-incline ROLLS BACK to the foot bay
                            (canonical x drops >= 0.15 m): sustained conveyance is
                            physically required, and only the touched-stage latch
                            (0.15) is credited;
  8.  landing near-miss   — ball settled against the lip on the LANDING side (the
                            "almost there" outcome) -> rejected by both x and z
                            windows, only the landing latch (0.15), no success;
  9.  latched credit      — teleporting that ball off to the floor leaves the
                            latched score unchanged while in_basin stays False;
  10. lip-top perch       — ball balanced ON the red lip top reads above the basin
                            height window -> rejected, removed before it topples;
  11. rim perch           — ball balanced on the crater rim wall top -> rejected,
                            removed before it topples;
  12. floor ball          — ball knocked off onto the ground beside the basin ->
                            below the height window and outside xy, score ~0;
  13. settle gate         — ball INSIDE the basin geometrically but still moving
                            (spun + shoved) is NOT success (velocity gates are
                            real); removed before it can settle;
  14. part-way + cap      — all four stage latches constructed, ball parked back in
                            the pocket: score == 0.60 cap (float32 + eps), NOT
                            success — full credit short of success is impossible
                            without the basin;
  15. wrong chirality     — the mirror-image position of the basin (where the basin
                            would be if the ramp had the OTHER handedness) is empty
                            space: a ball delivered there falls to the floor and is
                            rejected — a policy that memorizes one handedness fails;
  16. wall retention      — regulated quasi-static sideways push (<= 3 N) presses
                            the ball against the pocket wall: it MOVES >= 50 mm
                            (non-vacuous) then stalls in real contact, never climbs
                            over (canonical z stays at deck level), no success;
  17. rejection audit     — success() was never True at ANY judged point;
  18. final no-NaN        — all task-object states finite at the end;
  19. camera              — >= 20 rgb frames captured -> frames.npz.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i128.smoke --headless
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
    env = ENVS.get("simgen.crater_run")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    r = c.ball_r

    def legA(x: float) -> float:  # incline-A floor height at canonical x
        return (c.deck_h + c.pocket_h) / 2 + c.grade * x

    def legB(x: float) -> float:  # incline-B floor height at canonical x
        return (c.pocket_h + c.land_h) / 2 - c.grade * x

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 1.00)) + o),
                                tuple(np.array((-0.05, 0.05, 0.12)) + o),
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
        p = scene.ball_canon()[0]
        print(f"[smoke] {tag:16s} | mirror={float(scene.mirror[0]):+.0f}"
              f" canon=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
              f" latches=({float(scene.climbA_latch[0]):.0f},"
              f"{float(scene.turn_latch[0]):.0f},{float(scene.climbB_latch[0]):.0f},"
              f"{float(scene.land_latch[0]):.0f})"
              f" in_basin={bool(scene.in_basin()[0])}"
              f" score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_canon(canon_xyz, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement at a CANONICAL-frame point (instrumentation,
        not a solution) + REAL physics steps before judging (the zero-step trap).
        The canonical->world transform reads the live structure pose, so probes
        track chirality/yaw/jitter automatically."""
        canon = torch.tensor(canon_xyz, device=device).view(1, 3).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canon_to_world(canon.clone())
        st[:, 3] = 1.0
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_rel(xyz, settle_steps: int = 45) -> None:
        """Env-relative world placement (for parking the ball on the open floor)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.tensor(xyz, device=device)
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def wrench(f3: torch.Tensor) -> None:
        """World force on the ball expressed in its CURRENT link frame (the house
        convention) — re-encoded by the caller every step (the ball tumbles)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = scene.ball.data.root_link_quat_w
        scene.ball.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device), env_ids=all_ids)

    def layout_sane(tag: str) -> bool:
        """Reset honesty: active structure at the workspace (within jitter), the
        chirality twin parked in the depot, ball resting in the foot bay."""
        active = scene.ramp_r if float(scene.mirror[0]) > 0 else scene.ramp_l
        parked = scene.ramp_l if float(scene.mirror[0]) > 0 else scene.ramp_r
        ap = (active.data.root_pos_w - scene.env_origins)[0]
        pp = (parked.data.root_pos_w - scene.env_origins)[0]
        p = scene.ball_canon()[0]
        ok = (float(ap[0]) ** 2 + float(ap[1]) ** 2 <= (c.pos_jitter + 0.01) ** 2
              and abs(float(pp[0]) - c.park_xy[0]) < 0.05
              and abs(float(pp[1]) - c.park_xy[1]) < 0.05
              and c.ball_x0 - 0.03 <= float(p[0]) <= c.ball_x1 + 0.03
              and abs(float(p[1])) <= c.ball_y_jitter + 0.03
              and abs(float(p[2]) - (c.deck_h + r)) < 0.02
              and not bool(scene.in_basin()[0]))
        if not ok:
            print(f"[smoke] layout sanity VIOLATION at {tag}: ap={ap} pp={pp} p={p}",
                  flush=True)
        return ok

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.ramp_r, scene.ramp_l, scene.ball)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("settle: all states finite; active ramp at the workspace, twin parked in "
          "the depot, ball resting in the foot bay", fin and layout_sane("reset"))
    s, ok = judge()
    check("settle: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    sane = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(30)
        sane = sane and layout_sane(f"seed {sd}")
        active = scene.ramp_r if float(scene.mirror[0]) > 0 else scene.ramp_l
        q = active.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        ap = (active.data.root_pos_w - scene.env_origins)[0]
        p = scene.ball_canon()[0]
        reads.append((float(scene.mirror[0]), yaw, float(ap[0]), float(ap[1]),
                      float(p[0]), float(p[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (mirror, yaw, x, y, ball_cx, ball_cy):\n"
          f"{np.round(arr, 4)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    chir = {float(x) for x in arr[:, 0]}
    check("randomization: BOTH chiralities occur and the structure yaw/xy jitter is "
          f"physically posed and varies (readback: {len(chir)} chiralities, yaw "
          f"spread {math.degrees(spread[1]):.1f} deg, xy spread "
          f"({spread[2]:.3f},{spread[3]:.3f}))",
          len(chir) == 2 and spread[1] > 0.03
          and (spread[2] > 0.008 or spread[3] > 0.008) and sane)
    check("randomization: ball start varies inside the foot-bay window and tracks "
          f"its structure (readback spread x={spread[4]:.3f}, y={spread[5]:.3f})",
          spread[4] > 0.01 and spread[5] > 0.005)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy N/A (by constructed premise) ==============
    # rlbench/basketball_in_hoop's verb is grasp-carry-drop. No grasp end-state can be
    # constructed here: the ball is wider than the Franka parallel-jaw span. Readback
    # from the LIVE cfg (also asserted at cfg construction) — documented N/A.
    check("seed strategy (grasp-carry-drop) N/A by premise: ball diameter "
          f"{2 * r * 1000:.0f} mm > jaw span {c.jaw_span * 1000:.0f} mm + 8 mm "
          "margin — the ball cannot be grasped, only pushed",
          2 * r > c.jaw_span + 0.008)

    # =========================== 7. gravity is a real adversary =============================
    env.reset(seed=41)
    step(30)
    place_canon((0.10, 0.0, legA(0.10) + r + 0.002), settle_steps=240)
    report("rollback")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("gravity adversary: ball released mid-incline rolls BACK down to the foot "
          f"bay (canonical x {float(p[0]):+.3f} < -0.05, fell >= 0.15 m of x) — "
          "sustained conveyance is required; only the touched-stage latch (0.15) "
          "is credited",
          float(p[0]) < -0.05 and 0.10 - float(p[0]) >= 0.15
          and 0.15 - 1e-6 <= s <= 0.15 + 1e-5 and not ok)

    # =========================== 8-9. landing near-miss + latched credit ====================
    env.reset(seed=51)
    step(30)
    place_canon((-0.29, c.laneB_y, c.land_h + r + 0.002), settle_steps=90)
    report("landing-miss")
    p = scene.ball_canon()[0]
    s_before, ok = judge()
    check("landing near-miss: ball settled against the lip on the LANDING side "
          f"(canonical x={float(p[0]):+.3f}, z={float(p[2]):.3f}) is rejected by "
          "both the x and z windows — only the landing latch (0.15), no success",
          not bool(scene.in_basin()[0])
          and abs(float(p[0]) - c.basin_cx) > c.basin_xy_tol
          and float(p[2]) > c.basin_z_hi
          and 0.15 - 1e-6 <= s_before <= 0.15 + 1e-5 and not ok)
    place_rel((-1.2, -1.2, r + 0.002), settle_steps=40)
    report("moved-away")
    s_after, ok = judge()
    check("latched credit: teleporting the ball off to the floor leaves the latched "
          f"score unchanged ({s_before:.2f} -> {s_after:.2f}) while in_basin stays "
          "False",
          abs(s_after - s_before) < 1e-3 and not bool(scene.in_basin()[0])
          and not ok)

    # =========================== 10. lip-top perch ==========================================
    env.reset(seed=61)
    step(30)
    lip_x = c.basin_cx + c.basin_half_x + 0.01  # lip box center
    place_canon((lip_x, c.laneB_y, c.lip_top + r + 0.001), settle_steps=4)
    p = scene.ball_canon()[0]
    report("lip-perch")
    _s, ok = judge()
    perch_ok = (float(p[2]) > c.basin_z_hi and not bool(scene.in_basin()[0])
                and not ok)
    place_rel((-1.2, 1.2, r + 0.002), settle_steps=20)  # remove before it topples
    check("lip-top perch: ball balanced ON the red lip top reads canonical "
          f"z={float(p[2]) * 1000:.0f} mm > {c.basin_z_hi * 1000:.0f} mm -> "
          "rejected by the height window", perch_ok)

    # =========================== 11. rim perch ==============================================
    env.reset(seed=71)
    step(30)
    rim_x = c.basin_cx - c.basin_half_x - c.wall_t / 2  # far rim wall center
    place_canon((rim_x, c.laneB_y, c.rim_top + r + 0.001), settle_steps=4)
    p = scene.ball_canon()[0]
    report("rim-perch")
    _s, ok = judge()
    perch_ok = (float(p[2]) > c.basin_z_hi and not bool(scene.in_basin()[0])
                and not ok)
    place_rel((-1.2, 1.2, r + 0.002), settle_steps=20)  # remove before it topples
    check("rim perch: ball balanced on the crater rim wall top reads canonical "
          f"z={float(p[2]) * 1000:.0f} mm > {c.basin_z_hi * 1000:.0f} mm -> "
          "rejected by the height window", perch_ok)

    # =========================== 12. knocked off onto the floor =============================
    env.reset(seed=81)
    step(30)
    place_canon((c.basin_cx - 0.17, c.laneB_y, r + 0.002), settle_steps=60)
    report("floor-ball")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("floor ball: ball on the ground beside the basin reads canonical "
          f"z={float(p[2]) * 1000:.0f} mm < {c.basin_z_lo * 1000:.0f} mm -> below "
          "the height window and outside xy, score ~0",
          float(p[2]) < c.basin_z_lo and not bool(scene.in_basin()[0])
          and s <= 0.02 and not ok)

    # =========================== 13. settle gate ============================================
    env.reset(seed=91)
    step(30)
    place_canon((c.basin_cx, c.laneB_y, c.basin_floor_h + r + 0.002),
                vel=(0.25, 0.0, 0.0), ang=(0.0, 0.0, 9.0), settle_steps=2)
    v_now = float(scene.ball.data.root_lin_vel_w[0].norm())
    w_now = float(scene.ball.data.root_ang_vel_w[0].norm())
    report("settle-gate")
    _s, ok = judge()
    gate_ok = (bool(scene.in_basin()[0]) and (v_now > c.settle_lin
                                              or w_now > c.settle_ang) and not ok)
    # remove it BEFORE it can settle in the basin (this battery must never succeed)
    place_rel((-1.2, -1.2, r + 0.002), settle_steps=30)
    check("settle gate: ball INSIDE the basin geometrically but moving "
          f"(|v|={v_now:.2f} m/s, |w|={w_now:.1f} rad/s) is NOT success "
          "(velocity gates are real); removed before it can settle", gate_ok)

    # =========================== 14. part-way + the 0.60 cap ================================
    env.reset(seed=101)
    step(30)
    place_canon((0.05, 0.0, legA(0.05) + r + 0.002), settle_steps=25)  # climbA latch
    place_canon((0.28, 0.08, c.pocket_h + r + 0.002), settle_steps=25)  # turn latch
    place_canon((-0.05, c.laneB_y, legB(-0.05) + r + 0.002),
                settle_steps=25)  # climbB latch
    place_canon((-0.28, c.laneB_y, c.land_h + r + 0.002),
                settle_steps=25)  # landing latch
    place_canon((0.30, 0.08, c.pocket_h + r + 0.002), settle_steps=60)  # park: pocket
    report("part-way")
    s, ok = judge()
    lat = (float(scene.climbA_latch[0]), float(scene.turn_latch[0]),
           float(scene.climbB_latch[0]), float(scene.land_latch[0]))
    check("part-way + cap: all four stage latches constructed "
          f"(latches={lat}) with the ball parked back in the pocket -> score == "
          f"0.60 cap ({s:.4f}, float32 + eps), NOT success — full credit short of "
          "success is impossible without the basin",
          lat == (1.0, 1.0, 1.0, 1.0) and 0.595 <= s <= 0.60 + 1e-5 and not ok)

    # =========================== 15. wrong chirality ========================================
    # The mirror-image position of the basin — where the basin WOULD be if the ramp
    # had the other handedness — is empty space beside the structure for either
    # chirality (canonical y -> -y). A ball delivered there falls to the floor.
    env.reset(seed=111)
    step(30)
    place_canon((c.basin_cx, -c.laneB_y, c.basin_floor_h + r + 0.002),
                settle_steps=90)
    report("wrong-chir")
    p = scene.ball_canon()[0]
    s, ok = judge()
    check("wrong chirality: the mirror-image basin position is empty space — the "
          f"ball falls to the floor (canonical z={float(p[2]) * 1000:.0f} mm, "
          f"y={float(p[1]):+.3f}) and is rejected; a policy that memorizes one "
          "handedness fails",
          float(p[2]) < c.basin_z_lo and not bool(scene.in_basin()[0])
          and s <= 0.02 and not ok)

    # =========================== 16. wall retention (force probe, non-vacuous) ==============
    env.reset(seed=121)
    step(30)
    place_canon((0.285, 0.02, c.pocket_h + r + 0.002), settle_steps=30)
    p0 = scene.ball_canon()[0]
    y0 = float(p0[1])
    ymax, zmax = y0, float(p0[2])
    f3 = torch.zeros(3, device=device)
    tgt = torch.tensor([[0.285, 0.60, c.pocket_h + r]], device=device).expand(n, 3)
    for _ in range(360):
        d_w = scene.canon_to_world(tgt.clone()) - scene.ball.data.root_pos_w
        d_w[:, 2] = 0.0
        d_w = d_w / d_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        v_w = scene.ball.data.root_lin_vel_w.clone()
        v_w[:, 2] = 0.0
        f = 4.0 * (0.08 * d_w - v_w)  # quasi-static: the static-vs-dynamic trap
        f_norm = f.norm(dim=-1, keepdim=True)
        f = f * (f_norm.clamp(max=3.0) / f_norm.clamp_min(1e-9))
        wrench(f[0])
        step(1)
        p = scene.ball_canon()[0]
        ymax, zmax = max(ymax, float(p[1])), max(zmax, float(p[2]))
    wrench(zero3)
    step(45)
    report("wall-push")
    p = scene.ball_canon()[0]
    s, ok = judge()
    wall_inner = c.laneB_y + c.lane_half  # pocket +y wall inner face (canonical)
    check("wall retention: the regulated quasi-static push (<= 3 N ~ 1.2x ball "
          f"weight) moves the ball {(ymax - y0) * 1000:.0f} mm (>= 50, non-vacuous) "
          f"then stalls it against the pocket wall by real contact — center max y "
          f"{ymax:.3f} < wall face {wall_inner:.3f}, canonical z stays at deck "
          f"level (max {zmax:.3f}), never over the wall, no success",
          ymax - y0 >= 0.050 and ymax < wall_inner - 0.005
          and zmax < c.pocket_h + r + 0.05 and s <= 0.15 + 1e-5 and not ok)

    # =========================== 17-18. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== 19. camera + save + verdict ================================
    check(f"camera: >= 20 rgb frames captured ({len(frames)})", len(frames) >= 20)
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.crater_run")
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
