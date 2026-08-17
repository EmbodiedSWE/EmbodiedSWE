"""Smoke battery for CaromCourtScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — aim the vane with a P-servo,
release only at rest, lift the gate, let physics carry the ball; the Franka strategy
is TASK.md's embodiment argument). Drives here are the same fingertip-scale numbers
as solve.py (shared numbers = shared honesty); wrong outcomes are CONSTRUCTED through
the live plant or by placing the ball in settled wrong places — never by faking state.

One linear run, 13 named checks:
  1. settle    — clean reset: finite state everywhere, vane physically AT its sampled
                 start angle, gate closed, ball parked behind it, score ~0;
  2. readback  — the beacon physically stands ON the sampled pointer ray (its azimuth
                 IS theta_star), the vane reads theta0, and the bay back wall stands at
                 the exit point M recomputed INDEPENDENTLY by the float carom formula;
  3. random    — theta_star (via beacon-azimuth READBACK), theta0 (via vane READBACK)
                 and the bay pose vary across 6 seeds and stay inside their bands;
  4. null      — 2.5 s of nothing: ball still parked behind the closed gate, score
                 < 0.05, no success;
  5. negative  — the SEED's plan (carry it over and drop it): the ball dropped from
                 above the bay and above the court lands ON the roofs — both volumes
                 are sealed from the top, it never reaches the goal or the court;
  6. negative  — out of order (release FIRST): the unaimed ball strands at the far
                 wall (never within approach_d of M), the gate re-closes, and even
                 aiming correctly AFTERWARD yields partial credit only, no success;
  7. negative  — wrong bearing: vane rested 20 deg off the beacon ray — aim never
                 latches, the carom misses the doorway, ball never in the bay, ~0;
  8. negative  — fast sweep: bang-bang torque sweeps the pointer THROUGH the beacon
                 ray 3x at speed (peak in-band |w| >> slow_gate) — nothing latches;
  9. negative  — near misses placed settled: just short of the band (doorway plane),
                 in the court mouth, and ON the bay roof (xy in band, wrong z) — all
                 rejected by the box/height gates;
 10. negative  — the bay is sealed: the ball pushed at the back wall from behind and
                 at the side wall from the side (solve-scale force) never enters;
 11. exactness — full correct strategy (aim at rest -> lift gate -> hands off) ->
                 success() and score == 1.0, still true 2 s later;
 12. latch     — teleporting the ball out of the bay revokes success (live state);
                 the latched 0.70 of earned credit remains;
 13. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i413.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.basketball_in_hoop_i413 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Same fingertip-scale drives as solve.py (shared numbers = shared honesty).
KP = 0.6  # N*m/rad vane aim servo
TAU_MAX = 0.45  # N*m vane torque cap
ERR_DONE = math.radians(1.5)
W_DONE = 0.05  # rad/s
GATE_HOLD = 1.5  # N up on the gate
PUSH_F = 1.5  # N — bay-sealed probe force (10x the ball's rolling resistance)

SEED_PICK = 5  # fixed probe seed (no preconditions — every draw is legal)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carom_court")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)
    origin0 = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = scene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -1.10, 1.30)) + o),
                                tuple(np.array((-0.05, -0.05, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p = scene.ball_local()[0]
        b = scene.bay_coords()[0]
        print(f"[smoke] {tag:14s} | vane={math.degrees(float(scene.vane_yaw()[0])):+6.1f}deg "
              f"err={math.degrees(float(scene.aim_err()[0])):5.1f}deg "
              f"gate={float(scene.gate_lift()[0]) * 1000:5.1f}mm "
              f"ball=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"bay=({float(b[0]):+.3f},{float(b[1]):+.3f}) "
              f"latches=({float(scene.aim_latch[0]):.0f},{float(scene.launch_latch[0]):.0f},"
              f"{float(scene.approach_latch[0]):.0f}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    # --- geometry helpers -------------------------------------------------------------------
    def bay_uv() -> tuple:
        th = float(scene.theta_star[0])
        return (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))

    def bay_to_world(bx: float, by: float) -> tuple:
        u, v = bay_uv()
        mx, my = (float(x) for x in scene.exit_m[0])
        return mx + bx * u[0] + by * v[0], my + bx * u[1] + by * v[1]

    def put_ball(px: float, py: float, pz: float) -> None:
        """Place the ball at rest (env-local) — used ONLY to construct settled wrong
        outcomes / revocation probes, never to produce success."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = px
        st[0, 1] = py
        st[0, 2] = pz
        st[0, 3] = 1.0
        st[0, 0:3] += origin0
        scene.ball.write_root_state_to_sim(st, ids0)

    def in_box() -> bool:
        """Geometric bay-box occupancy (success minus the settle gate)."""
        b = scene.bay_coords()[0]
        p = scene.ball_local()[0]
        return (c.goal_x[0] <= float(b[0]) <= c.goal_x[1]) \
            and abs(float(b[1])) <= c.goal_y and float(p[2]) <= c.goal_z

    def in_court() -> bool:
        p = scene.ball_local()[0]
        return float(p[:2].norm()) < c.launch_r and -0.05 < float(p[2]) < 0.115

    # --- drive helpers (solve.py's controllers) ----------------------------------------------
    def servo_vane(tgt: float, budget: int = 2400) -> bool:
        """P-servo (joint damper is the D term); releases only close AND slow."""
        for _ in range(budget):
            yaw = float(scene.vane_yaw()[0])
            w = float(scene.vane.data.root_ang_vel_w[0, 2])
            e = wrap(tgt - yaw)
            if abs(e) < ERR_DONE and abs(w) < W_DONE:
                scene.drive_vane_t[0] = 0.0
                return True
            scene.drive_vane_t[0] = max(-TAU_MAX, min(TAU_MAX, KP * e))
            step(1)
        scene.drive_vane_t[0] = 0.0
        return False

    def release_gate(budget: int = 600) -> bool:
        """Hold the gate up until the ball is inside the court, then let it fall."""
        scene.drive_gate_f[0] = GATE_HOLD
        entered = False
        for _ in range(budget):
            step(1)
            p = scene.ball_local()[0]
            if float(p[:2].norm()) < 0.26 and float(p[2]) < 0.115:
                entered = True
                break
        scene.drive_gate_f[0] = 0.0
        return entered

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(60)
    report("reset")
    vane_err0 = abs(wrap(float(scene.vane_yaw()[0]) - float(scene.theta0[0])))
    park = scene.ball_local()[0] - torch.tensor(c.ball_start, device=device)
    check("settle: clean reset (finite, vane AT theta0, gate closed, ball parked, ~0)",
          finite_all() and vane_err0 < math.radians(2.5)
          and float(scene.gate_lift()[0]) < 0.003 and float(park.norm()) < 0.04
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. pose readback vs independent formula ========================
    th_s = float(scene.theta_star[0])
    b_pos = scene.beacon.data.root_pos_w[0] - origin0
    az_beacon = math.atan2(float(b_pos[1]), float(b_pos[0]))
    r_beacon = float(b_pos[:2].norm())
    m_ind, _az = scene_mod._exit_point(az_beacon, c.y_line, c.carom_off, c.r_in)
    u, _v = bay_uv()
    back_want = (m_ind[0] + 0.176 * u[0], m_ind[1] + 0.176 * u[1])
    back_got = scene.bay["bay_back"].data.root_pos_w[0] - origin0
    back_err = math.hypot(float(back_got[0]) - back_want[0], float(back_got[1]) - back_want[1])
    m_err = math.hypot(float(scene.exit_m[0, 0]) - m_ind[0],
                       float(scene.exit_m[0, 1]) - m_ind[1])
    print(f"[smoke] theta*={math.degrees(th_s):+.1f}deg beacon az="
          f"{math.degrees(az_beacon):+.1f}deg r={r_beacon:.3f} "
          f"M_ind=({m_ind[0]:+.3f},{m_ind[1]:+.3f}) m_err={m_err * 1000:.2f}mm "
          f"back_err={back_err * 1000:.2f}mm", flush=True)
    check("readback: beacon azimuth IS theta*, bay stands at the independent carom point M",
          abs(wrap(az_beacon - th_s)) < 1e-3 and abs(r_beacon - c.beacon_range) < 1e-3
          and m_err < 0.002 and back_err < 0.002)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    ok_band = True
    for seed in (11, 12, 13, 14, 15, 16):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        bp = scene.beacon.data.root_pos_w[0] - origin0
        ts = math.degrees(math.atan2(float(bp[1]), float(bp[0])))  # READBACK theta*
        t0 = math.degrees(float(scene.vane_yaw()[0]))  # READBACK theta0
        bk = scene.bay["bay_back"].data.root_pos_w[0] - origin0
        draws.append((round(ts, 1), round(t0, 1),
                      round(float(bk[0]), 2), round(float(bk[1]), 2)))
        ok_band &= c.theta_star_deg[0] - 0.5 <= ts <= c.theta_star_deg[1] + 0.5
        ok_band &= c.theta0_deg[0] - 0.5 <= t0 <= c.theta0_deg[1] + 0.5
        ok_band &= t0 - ts >= 55.5  # start always far from the aim band
    print(f"[smoke] draws (theta*, theta0, back_x, back_y): {draws}", flush=True)
    check("randomization is real (bearing + start via READBACK vary; bay moves; bands hold)",
          len({d[0] for d in draws}) >= 5 and len({d[1] for d in draws}) >= 5
          and len({(d[2], d[3]) for d in draws}) >= 5 and ok_band)

    # ========================= 4. null policy =================================================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(300)  # 2.5 s of nothing
    report("null")
    park = scene.ball_local()[0] - torch.tensor(c.ball_start, device=device)
    check("null policy: ball parked behind the closed gate, score < 0.05, no success",
          float(park.norm()) < 0.04 and float(scene.gate_lift()[0]) < 0.003
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: the SEED's plan (drop from above) =================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    bad = False
    # Above the bay's goal band, clear of the beacon (which stands over the bay at
    # bay-frame ~(0.08, +0.04) for this seed — a drop through it deflects the ball).
    bx, by = bay_to_world(0.14, -0.04)
    put_ball(bx, by, 0.30)
    for _ in range(120):
        step(1)
        bad |= in_box() or bool(scene.success()[0])
    zi_bay = float(scene.ball_local()[0, 2])  # rested ON the bay roof, 1 s after landing
    for _ in range(180):
        step(1)
        bad |= in_box() or bool(scene.success()[0])
    # Above the court roof, clear of the shaft slot (|y| < 0.013) and of the pointer
    # bar (parked at theta0 ~ +39 deg for this seed; azimuth here is ~124 deg).
    put_ball(-0.10, 0.15, 0.30)
    for _ in range(120):
        step(1)
        bad |= in_box() or in_court() or bool(scene.success()[0])
    zi_court = float(scene.ball_local()[0, 2])  # rested ON the court roof
    for _ in range(180):
        step(1)
        bad |= in_box() or in_court() or bool(scene.success()[0])
    report("drop-probe")
    check("negative (seed strategy): ball dropped from above lands ON both roofs — "
          "never in the bay or the court",
          not bad and zi_bay > c.goal_z + 0.02 and zi_court > 0.115
          and finite_all() and float(scene.score()[0]) < 0.05)

    # ========================= 6. negative: out of order (release first) ======================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    entered = release_gate()  # gate lifted with the vane still at theta0 — unaimed
    d_min = float("inf")
    for _ in range(1200):
        step(1)
        d_min = min(d_min, float((scene.ball_local()[0, :2] - scene.exit_m[0]).norm()))
        if float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.04:
            break
    step(120)
    gate_closed = float(scene.gate_lift()[0]) < 0.003
    stranded_r = float(scene.ball_local()[0, :2].norm())
    report("stranded")
    aimed_late = servo_vane(float(scene.theta_star[0]))  # correct aim, too late
    step(240)
    report("late-aim")
    check("negative (out of order): unaimed ball strands beyond the blade sweep, gate "
          "re-closes; aiming afterward earns partial credit only, no success",
          entered and gate_closed and d_min > c.approach_d
          and stranded_r > c.blade_size[0] + 0.02 and aimed_late
          and float(scene.aim_latch[0]) == 1.0 and float(scene.approach_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.50)

    # ========================= 7. negative: wrong bearing =====================================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    ok_wrong = servo_vane(float(scene.theta_star[0]) + math.radians(20.0))
    step(60)
    latched_wrong = float(scene.aim_latch[0])
    entered = release_gate()
    bad = False
    for _ in range(1200):
        step(1)
        bad |= in_box() or bool(scene.success()[0])
    report("wrong-aim")
    check("negative (wrong bearing): vane rested 20 deg off the beacon ray — aim never "
          "latches, carom misses the doorway, ~0 credit",
          ok_wrong and latched_wrong == 0.0 and entered and not bad
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 8. negative: fast sweep latches nothing ========================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    ts = float(scene.theta_star[0])
    tol = math.radians(c.aim_tol_deg)
    peak = 0.0
    crossings = 0
    for dest in (ts - math.radians(12.0), ts + math.radians(12.0),
                 ts - math.radians(12.0), ts + math.radians(12.0)):
        for _ in range(600):
            yaw = float(scene.vane_yaw()[0])
            w = float(scene.vane.data.root_ang_vel_w[0, 2])
            if abs(wrap(yaw - ts)) < tol:
                peak = max(peak, abs(w))
            e = wrap(dest - yaw)
            if abs(e) < math.radians(2.0):
                crossings += 1
                break
            scene.drive_vane_t[0] = math.copysign(TAU_MAX, e)
            step(1)
        scene.drive_vane_t[0] = 0.0
    step(120)
    report("fast-sweep")
    err_now = float(scene.aim_err()[0])
    check("negative (fast sweep): pointer swept THROUGH the beacon ray at speed — "
          "slow-gate keeps the aim latch at zero",
          crossings == 4 and peak > 2.0 * c.slow_gate and err_now > 1.2 * tol
          and float(scene.aim_latch[0]) == 0.0 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 9. negative: settled near misses ===============================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    results = []
    for tag, (bxf, byf, z) in (("short-of-band", (0.0, 0.0, 0.031)),
                               ("in-the-mouth", (-0.06, 0.0, 0.031)),
                               ("on-bay-roof", (0.10, 0.0, 0.161))):
        px, py = bay_to_world(bxf, byf)
        put_ball(px, py, z)
        step(240)
        b = scene.bay_coords()[0]
        results.append((tag, bool(scene.success()[0]), round(float(b[0]), 3),
                        round(float(scene.ball_local()[0, 2]), 3)))
        report(tag)
    check("negative (near miss): short of the band / in the mouth / on the bay roof — "
          "all settled, none succeed",
          all(not s for _t, s, _bx, _z in results) and finite_all()
          and float(scene.score()[0]) < 0.05)

    # ========================= 10. negative: the bay is sealed ================================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(30)
    u, v = bay_uv()
    bad = False
    px, py = bay_to_world(0.24, 0.0)  # behind the back wall, pushing inward (-x')
    put_ball(px, py, 0.031)
    scene.drive_ball_f[0, 0] = -PUSH_F * u[0]
    scene.drive_ball_f[0, 1] = -PUSH_F * u[1]
    for _ in range(360):
        step(1)
        bad |= in_box() or bool(scene.success()[0])
    scene.drive_ball_f[0] = 0.0
    report("push-back")
    px, py = bay_to_world(0.10, 0.15)  # beside the left wall, pushing inward (-y')
    put_ball(px, py, 0.031)
    scene.drive_ball_f[0, 0] = -PUSH_F * v[0]
    scene.drive_ball_f[0, 1] = -PUSH_F * v[1]
    for _ in range(360):
        step(1)
        bad |= in_box() or bool(scene.success()[0])
    scene.drive_ball_f[0] = 0.0
    step(60)
    report("push-side")
    check("negative (sealed bay): solve-scale pushes at the back and side walls never "
          "put the ball inside",
          not bad and finite_all() and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 11. exactness: full correct strategy ===========================
    torch.manual_seed(SEED_PICK)
    env.reset()
    step(60)
    ok_aim = servo_vane(float(scene.theta_star[0]))
    step(40)  # slow-gate latch matures
    aim_ok = float(scene.aim_latch[0]) == 1.0
    entered = release_gate()
    settled = False
    for _ in range(1200):
        step(1)
        if bool(scene.success()[0]):
            settled = True
            break
    step(240)  # ...and it persists hands-off
    report("goal-state")
    check("exactness: aim at rest + lift gate + hands off -> success() and score == 1.0, "
          "stable 2 s later",
          ok_aim and aim_ok and entered and settled and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 12. achievement latch ==========================================
    put_ball(0.45, 0.45, 0.031)  # out of the bay, onto open ground (revocation probe)
    step(60)
    report("revoked")
    check("achievement latch: ball removed from the bay revokes success; the latched "
          "0.70 of credit remains",
          not bool(scene.success()[0]) and 0.65 < float(scene.score()[0]) < 0.75)

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carom_court")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad_names = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad_names}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    t = threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
