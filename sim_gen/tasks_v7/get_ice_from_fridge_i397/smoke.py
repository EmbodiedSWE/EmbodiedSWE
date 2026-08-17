"""Smoke battery for IceCaddyDockScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — carry the caddy to the dock, let
the probe post open the spring valve under the caddy's weight, drain, re-seal, park;
the Franka strategy is TASK.md's embodiment argument). Teleported states here are
rubric INSTRUMENTATION: construct an outcome as a settled state under the scene's
live spring/contact plant, then assert the rubric's verdict on it.

One linear run, 13 named checks:
  1. settle    — clean reset: finite, caddy standing sealed at its sampled pose, every
                 present ball INSIDE it (caddy-frame readback), score ~0;
  2. random    — dock side/positions, caddy pose+yaw, ball count differ across 3 seeds
                 (max-pairwise, READBACK of the physical bodies, count by position);
  3. null      — 2 s of nothing: caddy stays put, score < 0.02, no success;
  4. sealed    — the anti-pour interlock: the assembly held INVERTED and shaken for
                 2.5 s releases NOTHING (spring preload > plug weight, roof solid) —
                 no ball escapes, the valve stays shut;
  5. decoy     — seated on the look-alike basin with no probe: the valve never opens,
                 nothing drains, no dock-open credit (the SEED-analog "just press it
                 at the machine" plan finds nothing to press);
  6. force-open— instrumentation force opens the valve AWAY from the dock (the moved-
                 actuator proof): balls may spill on the ground — ZERO credit, and the
                 spring re-seals on release;
  7. wrong bin — every ball dumped in the DECOY basin, caddy parked: rejected, ~0;
  8. partial   — one ball left inside, rest delivered, caddy parked: no success,
                 score = delivered fraction only;
  9. dock drain— the REAL physics: dropped onto the dock, the probe opens the valve
                 and every ball drains (this is solve's core, proven here) — but the
                 caddy LEFT ON THE DOCK caps at 0.70, no success;
 10. tipped    — all delivered but the caddy lying on its side on the pad: rejected;
 11. exactness — constructed full goal state -> success() and score == 1.0;
 12. revoke    — a ball plucked back OUT of the basin revokes success: score falls to
                 the latched 0.70, not 1.0;
 13. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.get_ice_from_fridge_i397.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
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
    from simgen_tasks.get_ice_from_fridge_i397 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# Same hand-scale drives as solve.py (shared numbers = shared honesty).
F_HOLD = 2.5
HOVER = 0.035


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ice_caddy_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    ids = torch.arange(1, device=device)
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.55, 0.80)) + o),
                                tuple(np.array((0.30, 0.00, 0.12)) + o),
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

    def ext_mm() -> float:
        return float(scene.plug_ext()[0]) * 1000.0

    def delivered() -> int:
        return int((scene.ball_latch[0] * scene.present[0].float()).sum().round())

    def report(tag: str) -> None:
        p = scene.caddy_pos()[0].tolist()
        print(f"[smoke] {tag:12s} | caddy=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"ext={ext_mm():5.1f}mm delivered={delivered()}/{int(scene.n_balls[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        """Poll `pred` while stepping. NEVER tests before stepping: a teleport-write
        leaves zero velocities, so a stillness predicate would pass vacuously on the
        un-simulated hover pose (and basin balls would never get a post_step to latch)."""
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    # ----- shared constructors (rubric instrumentation) ------------------------------------
    def transport(xy, z: float, quat=(1.0, 0.0, 0.0, 0.0), ball_idx=None) -> None:
        """Relocate the caddy (+plug, + the selected present balls) preserving all
        CURRENT caddy-frame offsets, zero velocity — the linkage moves as one."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        cp = scene.caddy.data.root_pos_w[0]
        cq = scene.caddy.data.root_quat_w[0:1]
        qn = torch.tensor([quat], device=device, dtype=torch.float32)

        def off(body) -> torch.Tensor:
            r = quat_apply_inverse(cq, (body.data.root_pos_w[0] - cp).unsqueeze(0))
            return quat_apply(qn, r)[0]

        target = torch.tensor([xy[0], xy[1], z], device=device) + scene.env_origins[0]
        plug_off = off(scene.plug)
        ball_off = [off(b) for b in scene.balls]
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = target
        st[0, 3:7] = qn[0]
        scene.caddy.write_root_state_to_sim(st, ids)
        ps = st.clone()
        ps[0, 0:3] = target + plug_off
        scene.plug.write_root_state_to_sim(ps, ids)
        if ball_idx is None:
            return
        for i, b in enumerate(scene.balls):
            if not bool(scene.present[0, i]) or i not in ball_idx:
                continue
            bs = torch.zeros(1, 13, device=device)
            bs[0, 0:3] = target + ball_off[i]
            bs[0, 3] = 1.0
            b.write_root_state_to_sim(bs, ids)

    def place_ball(i: int, xy, z: float) -> None:
        bs = torch.zeros(1, 13, device=device)
        bs[0, 0] = xy[0]
        bs[0, 1] = xy[1]
        bs[0, 2] = z
        bs[0, 3] = 1.0
        bs[0, 0:3] += scene.env_origins[0]
        scene.balls[i].write_root_state_to_sim(bs, ids)

    basin_spots = ((0.035, 0.035), (-0.035, 0.035), (0.035, -0.035), (-0.035, -0.035))

    def fill_basin(center, skip=()) -> None:
        """Teleport the present balls (minus `skip`) onto a basin floor, spread around
        the (possible) probe."""
        for i in range(c.n_balls_max):
            if not bool(scene.present[0, i]) or i in skip:
                continue
            sx, sy = basin_spots[i]
            place_ball(i, (center[0] + sx, center[1] + sy), 0.020)

    def dock_xy():
        return scene.dock_xy[0].tolist()

    def decoy_xy():
        return scene.decoy_xy[0].tolist()

    def pad_xy():
        return scene.pad_xy[0].tolist()

    def balls_in_caddy_frame() -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        cq = scene.caddy.data.root_quat_w[0:1]
        cp = scene.caddy.data.root_pos_w[0]
        return torch.stack([quat_apply_inverse(cq, (b.data.root_pos_w[0] - cp)
                                               .unsqueeze(0))[0] for b in scene.balls])

    def seat_on(center, hold_steps: int) -> None:
        """Drop the whole assembly onto a basin from the standard hover and hold it
        down with solve's steady force while it seats (live probe-vs-spring contact)."""
        transport(center, c.seated_root_z + HOVER, ball_idx=set(range(c.n_balls_max)))
        step(3)
        for _ in range(hold_steps):
            scene.caddy_force_w[0, 2] = -F_HOLD
            step(1)
        scene.caddy_force_w[0] = 0.0

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("reset")
    bodies = [scene.caddy, scene.plug, scene.dock, scene.decoy, scene.pad] + scene.balls
    finite = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies) \
        and bool(torch.isfinite(scene.score()).all())
    bf = balls_in_caddy_frame()
    inside = True
    for i in range(c.n_balls_max):
        if bool(scene.present[0, i]):
            inside &= bool((bf[i, 0:2].abs() < 0.075).all()) \
                and -0.108 < float(bf[i, 2]) < -0.020
    z_ok = abs(float(scene.caddy_pos()[0, 2]) - c.stand_root_z) < 0.005
    check("clean reset: finite, caddy standing sealed, every present ball inside it, "
          "score ~0",
          finite and inside and z_ok and ext_mm() < 4.0 and bool(scene.upright()[0])
          and float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 2. randomization by readback ===================================
    feats = []
    sides = []
    for s in (20, 21, 22):
        torch.manual_seed(s)
        env.reset()
        step(30)
        dk = (scene.dock.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        dc = (scene.decoy.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        pd = (scene.pad.data.root_pos_w[0] - scene.env_origins[0]).tolist()
        cdy = scene.caddy_pos()[0].tolist()
        q = scene.caddy.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        # ball count by PHYSICAL readback: present balls sit within 0.12 of the caddy.
        bp = scene.ball_pos()[0]
        cnt = int(((bp[:, 0:2] - torch.tensor(cdy[0:2], device=device)).norm(dim=-1)
                   < 0.12).sum())
        intent = dock_xy()
        match = abs(dk[0] - intent[0]) < 1e-3 and abs(dk[1] - intent[1]) < 1e-3
        feats.append([dk[0], dk[1], dc[1], pd[0], pd[1], cdy[0], cdy[1],
                      0.05 * math.cos(yaw), 0.05 * math.sin(yaw), 0.01 * cnt])
        sides.append((dk[1] > 0, dc[1] > 0, match, cnt == int(scene.n_balls[0])))
        print(f"[smoke] seed {s}: dock=({dk[0]:.3f},{dk[1]:.3f}) pad=({pd[0]:.3f},"
              f"{pd[1]:.3f}) caddy=({cdy[0]:.3f},{cdy[1]:.3f}) yaw={math.degrees(yaw):.0f} "
              f"n={cnt}", flush=True)
    d_max = max(math.dist(a, b) for a in feats for b in feats)
    mirrored = all(sd != sc_ for sd, sc_, _m, _n in sides)
    ok_read = all(m and nq for _sd, _sc, m, nq in sides)
    check("randomization: fixture layout / caddy pose / ball count differ across seeds "
          "(physical readback; decoy mirrors the dock)",
          d_max > 0.02 and mirrored and ok_read)

    # ========================= 3. null policy =================================================
    torch.manual_seed(4)
    env.reset()
    step(30)
    p0 = scene.caddy_pos()[0, 0:2].clone()
    step(240)  # 2 s of nothing
    report("null")
    moved = float((scene.caddy_pos()[0, 0:2] - p0).norm())
    check("null policy: caddy stays put, sealed, score < 0.02, no success",
          moved < 0.01 and ext_mm() < 4.0 and float(scene.score()[0]) < 0.02
          and not bool(scene.success()[0]))

    # ========================= 4. sealed under inversion + shake ==============================
    torch.manual_seed(5)
    env.reset()
    step(30)
    transport((0.65, 0.0), 0.45, quat=(0.0, 1.0, 0.0, 0.0),
              ball_idx=set(range(c.n_balls_max)))
    # The caddy is IN A HAND the whole time: hold its pose from the instant of the
    # flip (a free `step()` here would let the assembly free-fall, and re-teleporting
    # it up 30 cm would slam the plug through its joint limit — a probe artifact).
    hold0 = torch.zeros(1, 13, device=device)
    hold0[0, 0] = 0.65
    hold0[0, 2] = 0.45
    hold0[0, 4] = 1.0
    hold0[0, 0:3] += scene.env_origins[0]
    for _ in range(30):  # balls fall onto the (now-floor) roof inner face
        scene.caddy.write_root_state_to_sim(hold0, ids)
        step(1)
    ext_peak = 0.0
    ball_far = 0.0
    ball_zmin = 9.9
    w = 2.0 * math.pi * 2.5  # 2.5 Hz, 30 mm amplitude: a vigorous hand shake
    for t_i in range(300):  # 2.5 s
        t = t_i * env.dt
        st = torch.zeros(1, 13, device=device)
        st[0, 0] = 0.65 + 0.03 * math.sin(w * t)
        st[0, 2] = 0.45
        st[0, 4] = 1.0  # quat (0,1,0,0): inverted
        st[0, 7] = 0.03 * w * math.cos(w * t)
        st[0, 0:3] += scene.env_origins[0]
        scene.caddy.write_root_state_to_sim(st, ids)
        step(1)
        ext_peak = max(ext_peak, abs(float(scene.plug_ext()[0])))
        bp = scene.ball_pos()[0]
        cp = scene.caddy_pos()[0]
        for i in range(c.n_balls_max):
            if bool(scene.present[0, i]):
                ball_far = max(ball_far, float((bp[i] - cp).norm()))
                ball_zmin = min(ball_zmin, float(bp[i, 2]))
    # Hold the caddy still (inverted, in-hand) for 1.5 s: the spring must re-seal any
    # transient flutter the rattling balls hammered into the plug — hands off the plug.
    hold = torch.zeros(1, 13, device=device)
    hold[0, 0:3] = scene.caddy.data.root_pos_w[0]
    hold[0, 4] = 1.0
    for _ in range(180):
        scene.caddy.write_root_state_to_sim(hold, ids)
        step(1)
        bp = scene.ball_pos()[0]
        cp = scene.caddy_pos()[0]
        for i in range(c.n_balls_max):
            if bool(scene.present[0, i]):
                ball_far = max(ball_far, float((bp[i] - cp).norm()))
                ball_zmin = min(ball_zmin, float(bp[i, 2]))
    report("inverted")
    print(f"[smoke]   inverted diag: ext_peak={ext_peak * 1000:.1f}mm (ball-pass needs "
          f">=22.0) resealed={ext_mm():.1f}mm ball_far={ball_far:.3f} "
          f"ball_zmin={ball_zmin:.3f}", flush=True)
    check("sealed: inverted + shaken 2.5 s, the valve never opens a ball-passable gap "
          "(needs ext >= 22 mm), NO ball escapes, and the spring re-seals in-hand",
          ext_peak < 0.016 and ext_mm() < 6.0 and ball_far < 0.20 and ball_zmin > 0.15
          and delivered() == 0 and float(scene.dock_open_latch[0]) == 0.0)

    # ========================= 5. decoy: seats, but nothing opens =============================
    torch.manual_seed(6)
    env.reset()
    step(30)
    seat_on(decoy_xy(), hold_steps=120)
    ext_decoy_peak = 0.0
    for _ in range(360):  # 3 s seated hands-off on the decoy
        step(1)
        ext_decoy_peak = max(ext_decoy_peak, float(scene.plug_ext()[0]))
    report("decoy")
    z_seated = abs(float(scene.caddy_pos()[0, 2]) - c.seated_root_z) < 0.008
    check("decoy basin (no probe): the caddy seats but the valve NEVER opens, nothing "
          "drains, no dock-open credit — score is the lift latch alone",
          z_seated and ext_decoy_peak < 0.006 and delivered() == 0
          and float(scene.dock_open_latch[0]) == 0.0
          and float(scene.score()[0]) <= 0.13 and not bool(scene.success()[0]))

    # ========================= 6. force-opened AWAY from the dock: zero credit ================
    torch.manual_seed(7)
    env.reset()
    step(30)
    # Empty the caddy first (balls to the depot): this probes the VALVE mechanism alone.
    # (With balls aboard, a ball wedges between the rising head rim and the funnel and
    # honestly limits the stroke — a separate effect, not what this check is about.)
    for i in range(c.n_balls_max):
        if bool(scene.present[0, i]):
            place_ball(i, (c.depot[0], c.depot[1] + 0.08 * i), c.ball_r + 0.001)
    step(60)
    ext_forced = 0.0
    for _ in range(300):  # 2.5 s of a 4 N upward force on the plug (instrumentation)
        scene.plug_force_w[0, 2] = 4.0
        step(1)
        ext_forced = max(ext_forced, float(scene.plug_ext()[0]))
    scene.plug_force_w[0] = 0.0
    step(180)
    report("force-open")
    print(f"[smoke]   force-open diag: ext_forced={ext_forced * 1000:.1f}mm "
          f"resealed={ext_mm():.1f}mm", flush=True)
    check("valve forced open AWAY from the dock: it DOES open (actuator moved) — "
          "but earns zero credit, and the spring re-seals on release",
          ext_forced > 0.015 and float(scene.dock_open_latch[0]) == 0.0
          and delivered() == 0 and ext_mm() < 6.0
          and float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 7. wrong basin: balls dumped in the decoy ======================
    torch.manual_seed(8)
    env.reset()
    step(30)
    fill_basin(decoy_xy())
    transport(pad_xy(), c.park_root_z + 0.020)
    ok7 = settle_until(lambda: bool(scene.settled()[0]), max_steps=360)
    report("wrong-basin")
    check("every ball dumped in the DECOY basin, caddy parked: rejected, score ~0",
          ok7 and delivered() == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.02)

    # ========================= 8. partial delivery: one ball left inside ======================
    torch.manual_seed(9)
    env.reset()
    step(30)
    nb = int(scene.n_balls[0])
    fill_basin(dock_xy(), skip=(0,))  # ball 0 stays inside the caddy ...
    transport(pad_xy(), c.park_root_z + 0.020, ball_idx={0})  # ... and rides to the pad
    ok8 = settle_until(lambda: bool(scene.settled()[0]), max_steps=360)
    report("partial")
    s8 = float(scene.score()[0])
    want8 = 0.40 * (nb - 1) / nb
    check("one ball left inside the parked caddy: no success, score = delivered "
          "fraction only",
          ok8 and delivered() == nb - 1 and not bool(scene.success()[0])
          and abs(s8 - want8) < 0.02)

    # ========================= 9. real dock drain, but caddy LEFT ON the dock =================
    torch.manual_seed(10)
    env.reset()
    step(30)
    nb = int(scene.n_balls[0])
    seat_on(dock_xy(), hold_steps=180)
    opened = float(scene.plug_ext()[0]) > c.open_ext
    drained = False
    for i in range(1440):  # 12 s hands-off dwell: gravity does the feeding
        step(1)
        if delivered() >= nb:
            drained = True
            break
    step(60)
    report("dock-drain")
    s9 = float(scene.score()[0])
    check("REAL dock: the probe opens the valve on seating and every ball drains by "
          "gravity — but the caddy left on the dock caps at 0.70, no success",
          opened and drained and not bool(scene.success()[0]) and abs(s9 - 0.70) < 0.02)

    # ========================= 10. tipped on the pad ==========================================
    torch.manual_seed(11)
    env.reset()
    step(30)
    fill_basin(dock_xy())
    # Caddy on its SIDE over the pad (90 deg about y), dropped to find its rest.
    transport(pad_xy(), 0.14, quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))
    ok10 = settle_until(lambda: bool(scene.settled()[0]), max_steps=600)
    report("tipped")
    check("all delivered but the caddy lying on its side on the pad: not parked, "
          "no success",
          ok10 and not bool(scene.upright()[0]) and not bool(scene.parked()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.99)

    # ========================= 11. exactness: constructed goal state ==========================
    torch.manual_seed(12)
    env.reset()
    step(30)
    fill_basin(dock_xy())
    transport(pad_xy(), c.park_root_z + 0.020)
    ok11 = settle_until(lambda: bool(scene.success()[0]), max_steps=480)
    report("goal-state")
    check("exactness: constructed goal state -> success() and score == 1.0",
          ok11 and abs(float(scene.score()[0]) - 1.0) < 2e-3)

    # ========================= 12. revoke: pluck a ball back out ==============================
    victim = next(i for i in range(c.n_balls_max) if bool(scene.present[0, i]))
    place_ball(victim, (0.65, -0.30), c.ball_r + 0.001)  # onto open floor
    step(120)
    report("revoked")
    s12 = float(scene.score()[0])
    check("achievement latch: removing one ball from the basin revokes success — "
          "score falls to the latched 0.70, not 1.0",
          not bool(scene.success()[0]) and abs(s12 - 0.70) < 0.02)

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ice_caddy_dock")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[smoke] CRASH: {type(e).__name__}: {e}", flush=True)
        os._exit(1)
