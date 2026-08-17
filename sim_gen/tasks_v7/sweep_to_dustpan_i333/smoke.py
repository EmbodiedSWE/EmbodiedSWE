"""Smoke / rubric-REJECTION battery for RiddleTrayScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts crate-drops + a torque-servo tilt
discharge and the latched credit is monotone along it). This battery proves the rubric
REJECTS wrong outcomes and that the physical claims the task rests on are load-bearing:
the hopper is sealed against everything except the tray's slot, the slot passes marbles
and refuses cubes, ground-level transport (the seed strategy) earns nothing, the rest
recline retains marbles, empty tilting earns no tilt credit, wrong destinations earn
nothing, and a tray held up blocks success. Every probe is CONSTRUCTED (teleport, real
physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; tray reclined on its rest stop;
                           all present pieces on the bed; nothing binned/crated; score
                           0, no success;
   2. randomization      — two seeded resets: piece world xy, cube yaws, and the crate
                           pose differ (world readback);
   3. count coverage     — over 25 resets kb covers {2,3,4} and kc covers {1,2,3};
                           every piece index is present and parked at least once, and
                           the depot world-readback agrees with the present mask;
   4. null-policy        — 400 idle steps: the tray does not drift off its stop,
                           nothing enters the hopper, score ~0, no success;
   5. SEED STRATEGY      — the seed's plan (transport debris along the GROUND into the
                           receptacle): a marble dragged quasi-statically along the
                           ground straight at the hopper travels freely under the tray
                           and ends pressed against the hopper's ground-level front
                           wall — never inside, no bin latch, score ~0;
   6. SLOT SELECTIVITY   — tray pinned at full tilt by a paddle-torque: a cube pushed
                           down the bed at 1.2x its weight visibly slides but is
                           REFUSED by the 28 mm slot header and stays on the bed; a
                           marble released at the same station rolls out through the
                           slot into the hopper (both sides of the riddle are live);
   7. RECLINE RETENTION  — a marble placed just behind the slot on the resting tray
                           rolls BACKWARD (away from the slot) to the back wall; the
                           null tray is strictly retentive;
   8. EMPTY-TILT GATE    — full tilt with no marble aboard: the tilt latch stays dark
                           (tilt credit requires a genuine discharge attempt);
   9. WRONG DESTINATIONS — a cube constructed in the HOPPER + a marble constructed in
                           the CRATE: no latch fires, score ~0, no success;
  10. near-misses        — a marble dropped over the 20 mm sky slot cannot enter the
                           hopper; a cube on the ground beside the crate is not crated;
  11. TRAY-HELD-UP GATE  — everything binned+crated but the tray still pressed at full
                           tilt: success False, score capped 0.80; released onto the
                           stop: the rest conjunct (the only blocker) flips it True;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.sweep_to_dustpan_i333.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _clear(body) -> None:
    zero = torch.zeros(_ENV.num_envs, 1, 3, device=_ENV.device)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.riddle_tray")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    origins = env.iscene.env_origins

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.90, 0.70)) + o),
                                tuple(np.array((0.20, 0.10, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def angle() -> float:
        _refresh()
        return float(scene.tray_angle_deg()[0])

    def pb() -> list[bool]:
        return scene.present_ball[0].tolist()

    def pc() -> list[bool]:
        return scene.present_cube[0].tolist()

    def ball(i: int):
        return scene.balls[scene.BALL_NAMES[i]]

    def cube(i: int):
        return scene.cubes[scene.CUBE_NAMES[i]]

    def first_ball() -> int:
        return pb().index(True)

    def first_cube() -> int:
        return pc().index(True)

    def body_world(body, loc) -> torch.Tensor:
        """Body-local point -> world (uses the body's live pose)."""
        _refresh()
        p = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return body.data.root_pos_w + quat_apply(body.data.root_quat_w, p)

    def ground_world(x, y, h) -> torch.Tensor:
        p = torch.tensor([x, y, h], device=device, dtype=torch.float).expand(n, 3)
        return p + origins

    def report(tag: str) -> None:
        _refresh()
        print(f"[smoke] {tag:18s} | tray={angle():+6.2f}deg pb={pb()} pc={pc()} "
              f"n_bin={int(scene.balls_in_hopper()[0].sum())} "
              f"n_crate={int(scene.cubes_in_crate()[0].sum())} "
              f"ball_l={scene._ball_l[0].tolist()} cube_l={scene._cube_l[0].tolist()} "
              f"tilt_l={bool(scene._tilt[0])} rest={bool(scene.tray_at_rest()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    zero = torch.zeros(n, 1, 3, device=device)
    TILT_HI = c.tilt_max_deg - 0.5

    def _servo_step(theta_des_deg: float) -> None:
        """One step of the paddle-press torque servo (gravity ff + PD about the world-y
        hinge axis, capped at 3.5 N*m — same actuator model as solve.py). A ramped
        servo, never a torque slam: a slam would fling the pieces riding the bed."""
        th = math.radians(float(scene.tray_angle_deg()[0]))
        wy = float(scene.tray.data.root_ang_vel_w[0, 1])
        tau = 1.2 + 12.0 * (math.radians(theta_des_deg) - th) - 0.5 * wy
        tau = max(-3.5, min(3.5, tau))
        t = torch.zeros(n, 1, 3, device=device)
        t[0, 0, 1] = tau
        scene.tray.set_external_force_and_torque(zero, t, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)

    def tilt_up(steps: int = 96) -> None:
        for i in range(steps):
            _servo_step(c.rest_deg + (TILT_HI - c.rest_deg) * (i + 1) / steps)

    def hold_tilt(steps: int) -> None:
        for _ in range(steps):
            _servo_step(TILT_HI)

    def release_tray(settle: int = 240) -> None:
        for i in range(90):
            _servo_step(TILT_HI + (c.rest_deg - TILT_HI) * (i + 1) / 90)
        _clear(scene.tray)
        _step(settle)

    def park_balls(skip: int = -1) -> None:
        """Teleport present marbles (except `skip`) to ground parking off-stage."""
        for i in range(c.n_balls):
            if i == skip or not pb()[i]:
                continue
            _write_body(ball(i), ground_world(1.10 + 0.08 * i, 1.10, c.ball_r + 0.003))
        _step(20)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    report("settle")
    _REC["on"] = False
    kb1 = sum(pb())
    aboard1 = int((scene.balls_on_tray()[0] & scene.present_ball[0]).sum())
    check("settle/no-NaN: seeded reset settles finite, tray reclined on its rest stop "
          f"({angle():+.2f} deg), all {kb1} present marbles on the bed, nothing "
          "binned/crated, score 0, no success",
          bool(scene._finite()[0]) and c.rest_deg - 1.5 < angle() < c.rest_max_deg
          and aboard1 == kb1 and int(scene.balls_in_hopper()[0].sum()) == 0
          and int(scene.cubes_in_crate()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        pos = torch.stack([b.data.root_pos_w[0, :2].clone()
                           for b in list(scene.balls.values()) + list(scene.cubes.values())])
        yaws = [math.degrees(2.0 * math.atan2(float(b.data.root_quat_w[0, 3]),
                                              float(b.data.root_quat_w[0, 0])))
                for b in scene.cubes.values()]
        cr = scene.crate.data.root_pos_w[0, :2].clone()
        cy = math.degrees(2.0 * math.atan2(float(scene.crate.data.root_quat_w[0, 3]),
                                           float(scene.crate.data.root_quat_w[0, 0])))
        return pos, yaws, cr, cy, (tuple(pb()), tuple(pc()))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_pos, a_yaw, a_cr, a_cy, a_pat = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_pos, b_yaw, b_cr, b_cy, b_pat = readback()
    d_pos = float((a_pos - b_pos).norm(dim=1).max())
    d_yaw = max(abs((a - b + 180.0) % 360.0 - 180.0) for a, b in zip(a_yaw, b_yaw))
    d_cr = float((a_cr - b_cr).norm())
    d_cy = abs((a_cy - b_cy + 180.0) % 360.0 - 180.0)
    print(f"[smoke] randomization deltas: max piece xy {d_pos * 1000:.1f}mm, max cube "
          f"yaw {d_yaw:.1f}deg, crate {d_cr * 1000:.1f}mm / {d_cy:.1f}deg, "
          f"patterns {a_pat} vs {b_pat}", flush=True)
    check("randomization-is-real: piece positions, cube yaws and the crate pose all "
          "differ across seeds (world readback)",
          d_pos > 0.030 and d_yaw > 5.0 and (d_cr > 0.005 or d_cy > 1.0))

    # ================= 3. count coverage ==========================================================
    kbs: set[int] = set()
    kcs: set[int] = set()
    seen_p = [False] * (c.n_balls + c.n_cubes)
    seen_a = [False] * (c.n_balls + c.n_cubes)
    for s in range(25):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        kbs.add(sum(pb()))
        kcs.add(sum(pc()))
        for i, p in enumerate(pb() + pc()):
            seen_p[i] |= p
            seen_a[i] |= not p
        # parked pieces really are in the depot (world readback, not the mask)
        for i, p in enumerate(pb()):
            x = float((ball(i).data.root_pos_w - origins)[0, 0])
            assert (x > 1.0) == (not p), f"present mask vs world pos mismatch (ball_{i})"
        for i, p in enumerate(pc()):
            x = float((cube(i).data.root_pos_w - origins)[0, 0])
            assert (x > 1.0) == (not p), f"present mask vs world pos mismatch (cube_{i})"
    print(f"[smoke] over 25 resets: kb {sorted(kbs)}, kc {sorted(kcs)}, present "
          f"coverage {seen_p}, absent coverage {seen_a}", flush=True)
    check("count coverage: kb covers {2,3,4}, kc covers {1,2,3}, every piece index is "
          "present and parked at least once (depot readback agrees with the mask)",
          kbs == {2, 3, 4} and kcs == {1, 2, 3} and all(seen_p) and all(seen_a))

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    a0 = angle()
    _step(400)
    report("null-policy")
    check("null-policy-fails: 400 idle steps, tray drift "
          f"{abs(angle() - a0):.2f} deg, nothing enters the hopper, score ~0, no success",
          abs(angle() - a0) < 0.5 and int(scene.balls_in_hopper()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: ground transport at the receptacle =======================
    # The seed's whole plan: sweep debris along the GROUND into the receptacle mouth.
    # Construct it: a present marble on the ground in front of the dock, dragged
    # quasi-statically (3x weight, speed-capped — sweeping, not hurling) straight at
    # the hopper. The hopper's front wall reaches the ground and its only mouth is
    # 85 mm up behind the tray's slot: the marble travels freely under the tray and
    # ends pressed against the blank wall at ground level — never inside.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i5 = first_ball()
    print(f"[smoke] seed-strategy: dragging ball_{i5} along the ground", flush=True)
    _REC["on"] = True
    _write_body(ball(i5), ground_world(-0.15, 0.0, c.ball_r + 0.003))
    _step(60)
    # drag with peak-approach tracking (a rolling ball rebounds off the wall on its
    # own spin after force-off, so the refusal is measured AT the wall, not at rest)
    f5 = torch.tensor((3.0 * c.ball_mass * 9.81, 0.0, 0.0),
                      device=device).view(1, 1, 3).expand(n, 1, 3).contiguous()
    peak_x, peak_z = -1e9, 0.0
    for _ in range(480):
        v = float(ball(i5).data.root_lin_vel_w[0, :2].norm())
        ball(i5).set_external_force_and_torque(f5 if v < 0.35 else zero, zero,
                                               env_ids=_all_ids(), is_global=True)
        _step(1)
        lx = scene._local(scene.dock, ball(i5).data.root_pos_w)[0]
        if float(lx[0]) > peak_x:
            peak_x, peak_z = float(lx[0]), float(lx[2])
    _clear(ball(i5))
    _step(120)
    report("seed-strategy")
    l5 = scene._local(scene.dock, ball(i5).data.root_pos_w)[0]
    wall_face = c.wall_x - c.wall_t / 2
    print(f"[smoke] ground drag: peak approach dock-frame x={peak_x * 1000:.0f}mm at "
          f"z={peak_z * 1000:.0f}mm (wall face {wall_face * 1000:.0f}mm, mouth sill z "
          f"{c.wall_top_z * 1000:.0f}mm); after force-off it rolls back to "
          f"x={float(l5[0]) * 1000:.0f}mm", flush=True)
    check("negative (SEED strategy): a quasi-static ground drag straight at the hopper "
          "travels freely and genuinely reaches the hopper's blank ground-level front "
          "wall, which refuses it (peak approach at the wall face, far below the mouth "
          "sill) — never inside at any step (latch dark), score ~0, no success",
          peak_x > wall_face - c.ball_r - 0.010 and peak_x < wall_face + 0.005
          and peak_z < c.wall_top_z - 0.02
          and int(scene.balls_in_hopper()[0].sum()) == 0
          and not bool(scene._ball_l[0, i5])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. SLOT SELECTIVITY: cube refused, marble passes ===========================
    # Tray pinned at full tilt by a constant paddle torque. The marbles are parked off
    # the tray first so the tilt latch stays dark for this probe (no marble aboard).
    # (a) a cube pushed down-bed at 1.2x weight visibly slides yet is refused by the
    #     28 mm header (cube 32 mm) and stays on the bed;
    # (b) a marble released at the same station rolls out through the slot on its own
    #     and lands in the hopper.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i6b, i6c = first_ball(), first_cube()
    park_balls()
    # park the OTHER cubes so the probe lane is clear
    for i in range(c.n_cubes):
        if i != i6c and pc()[i]:
            _write_body(cube(i), ground_world(1.10 + 0.08 * i, 1.30, c.cube_size / 2 + 0.003))
    _step(20)
    _REC["on"] = True
    tilt_up()
    hold_tilt(60)
    a6 = angle()
    assert a6 > c.tilt_max_deg - 2.0, f"servo should press the tray to full tilt, got {a6}"
    # (a) cube: place on the tilted bed, push toward the slot
    _write_body(cube(i6c), body_world(scene.tray, (-0.30, -0.06, c.cube_size / 2 + 0.004)))
    hold_tilt(60)
    p6c = cube(i6c).data.root_pos_w[0].clone()
    fmag = 1.2 * c.cube_mass * 9.81
    fdir = (math.cos(math.radians(a6)) * fmag, 0.0, -math.sin(math.radians(a6)) * fmag)
    f6 = torch.tensor(fdir, device=device).view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(360):
        v = float(cube(i6c).data.root_lin_vel_w[0].norm())
        cube(i6c).set_external_force_and_torque(f6 if v < 0.12 else zero, zero,
                                                env_ids=_all_ids(), is_global=True)
        _servo_step(TILT_HI)
    _clear(cube(i6c))
    hold_tilt(90)
    lc6 = scene._local(scene.tray, cube(i6c).data.root_pos_w)[0]
    movedc6 = float((cube(i6c).data.root_pos_w[0] - p6c).norm())
    on_bed = (float(lc6[0]) > -(c.floor_len - c.fence_t)) and (float(lc6[0]) < 0.0) \
        and (-0.005 < float(lc6[2]) < 0.05)
    print(f"[smoke] cube push at full tilt: moved {movedc6 * 1000:.0f}mm, ends at "
          f"tray-frame x={float(lc6[0]) * 1000:.0f}mm z={float(lc6[2]) * 1000:.0f}mm "
          f"(on bed: {on_bed})", flush=True)
    cube_refused = movedc6 > 0.120 and on_bed and float(lc6[0]) > -0.10
    # (b) marble: same station, no push needed — rolls out through the slot
    _write_body(ball(i6b), body_world(scene.tray, (-0.30, 0.08, c.ball_r + 0.004)))
    binned6 = False
    for _ in range(300):
        _servo_step(TILT_HI)
        if bool(scene.balls_in_hopper()[0, i6b]):
            binned6 = True
            break
    hold_tilt(60)
    report("slot-selectivity")
    release_tray()
    check("SLOT SELECTIVITY: at full tilt a pushed cube visibly slides but the 28 mm "
          "slot refuses it (still on the bed, pressed at the header) while a marble "
          "released at the same station rolls through into the hopper",
          cube_refused and binned6 and bool(scene.balls_in_hopper()[0, i6b]))
    _REC["on"] = False

    # ================= 7. RECLINE RETENTION: the null tray drains backward ========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i7 = first_ball()
    _REC["on"] = True
    # just behind the slot, under the canopy (marble d 22 mm < canopy clearance 36 mm)
    _write_body(ball(i7), body_world(scene.tray, (-0.06, 0.0, c.ball_r + 0.002)))
    p7 = ball(i7).data.root_pos_w[0].clone()
    _step(360)
    report("recline-retention")
    l7 = scene._local(scene.tray, ball(i7).data.root_pos_w)[0]
    moved7 = float((ball(i7).data.root_pos_w[0] - p7).norm())
    print(f"[smoke] recline: marble placed at x=-60mm rolled to tray-frame "
          f"x={float(l7[0]) * 1000:.0f}mm (moved {moved7 * 1000:.0f}mm)", flush=True)
    check("RECLINE RETENTION: a marble placed just behind the slot on the resting tray "
          "rolls BACKWARD to the back wall — never out through the slot, no bin latch",
          moved7 > 0.100 and float(l7[0]) < -0.25
          and bool(scene.balls_on_tray()[0, i7])
          and int(scene.balls_in_hopper()[0].sum()) == 0
          and not bool(scene._ball_l[0, i7]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. EMPTY-TILT GATE: tilt credit needs a load ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_balls()  # no marble aboard
    _REC["on"] = True
    tilt_up()
    hold_tilt(200)
    a8 = angle()
    tilt_dark = not bool(scene._tilt[0])
    release_tray()
    report("empty-tilt")
    check("EMPTY-TILT GATE: the tray pressed to full tilt "
          f"({a8:+.1f} deg) with no marble aboard leaves the tilt latch dark — no "
          "tilt credit for riddling an empty tray, score ~0",
          a8 > c.tilt_credit_deg + 1.0 and tilt_dark and not bool(scene._tilt[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. WRONG DESTINATIONS: swapped receptacles earn nothing ====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i9b, i9c = first_ball(), first_cube()
    _REC["on"] = True
    # cube constructed INSIDE the sealed hopper (probe-only ability), marble in the crate
    _write_body(cube(i9c), body_world(scene.dock, (0.13, 0.0, -0.09)))
    _write_body(ball(i9b), body_world(scene.crate, (0.0, 0.0, 0.03)))
    _step(150)
    report("wrong-dest")
    lc9 = scene._local(scene.dock, cube(i9c).data.root_pos_w)[0]
    lb9 = scene._local(scene.crate, ball(i9b).data.root_pos_w)[0]
    print(f"[smoke] wrong destinations: cube at dock-frame z={float(lc9[2]) * 1000:.0f}mm "
          f"(in hopper box), marble at crate-frame z={float(lb9[2]) * 1000:.0f}mm "
          f"(in crate box)", flush=True)
    check("WRONG DESTINATIONS: a cube constructed in the hopper and a marble settled "
          "in the crate fire no latch and earn nothing — no bin credit, no crate "
          "credit, score ~0, no success",
          float(lc9[2]) < c.hop_z_max and abs(float(lb9[0])) < c.crate_xy
          and not bool(scene._ball_l[0, i9b]) and not bool(scene._cube_l[0, i9c])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 10. near-misses: sky slot + beside the crate ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    i10b, i10c = first_ball(), first_cube()
    _REC["on"] = True
    # (a) marble dropped from above the 20 mm sky slot (marble d 22 mm) — cannot enter
    _write_body(ball(i10b), ground_world(0.31, 0.0, 0.45))
    _step(240)
    ball_out = int(scene.balls_in_hopper()[0].sum()) == 0 \
        and not bool(scene._ball_l[0, i10b])
    lb10 = scene._local(scene.dock, ball(i10b).data.root_pos_w)[0]
    # (b) cube on the ground beside the crate (touching the wall) — not crated
    _write_body(cube(i10c), body_world(scene.crate, (0.13, 0.0, c.cube_size / 2 + 0.003)))
    _step(120)
    report("near-miss")
    lc10 = scene._local(scene.crate, cube(i10c).data.root_pos_w)[0]
    print(f"[smoke] near-misses: dropped marble ends at dock-frame "
          f"x={float(lb10[0]) * 1000:.0f}mm z={float(lb10[2]) * 1000:.0f}mm (not in "
          f"hopper: {ball_out}); cube beside crate at crate-frame "
          f"x={float(lc10[0]) * 1000:.0f}mm (bound {c.crate_xy * 1000:.0f}mm)", flush=True)
    check("near-misses: a marble dropped over the 20 mm sky slot never enters the "
          "sealed hopper, and a cube on the ground beside the crate is not crated — "
          "no latches, no success",
          ball_out and float(lc10[0]) > c.crate_xy
          and not bool(scene._cube_l[0, i10c]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. TRAY-HELD-UP GATE: rest conjunct is live ===============================
    # Construct the full sorted state but keep the paddle pressed: every present cube
    # crated (teleport-drop), every present marble discharged through the slot by the
    # pinned-tilt tray (contact physics). Held: success False, score capped 0.80.
    # Released onto the stop: the rest conjunct — the only blocker — flips it True.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    kb11, kc11 = sum(pb()), sum(pc())
    _REC["on"] = True
    slots11 = ((-0.045, -0.045), (0.045, -0.045), (0.0, 0.045))
    j = 0
    for i in range(c.n_cubes):
        if not pc()[i]:
            continue
        tgt = body_world(scene.crate, (slots11[j][0], slots11[j][1], 0.0))
        tgt[:, 2] = origins[:, 2] + 0.10
        _write_body(cube(i), tgt)
        _step(80)
        j += 1
    assert int((scene.cubes_in_crate()[0] & scene.present_cube[0]).sum()) == kc11, \
        "check-11 construct: cubes must settle in the crate"
    tilt_up()
    held = 0
    while held < 1200:
        _servo_step(TILT_HI)
        held += 1
        if bool(((scene._ball_l[0] | ~scene.present_ball[0])).all()):
            break
    hold_tilt(90)
    report("held-up")
    a11 = angle()
    all_binned11 = int((scene.balls_in_hopper()[0] & scene.present_ball[0]).sum()) == kb11
    held_blocked = not bool(scene.success()[0]) and not bool(scene.tray_at_rest()[0])
    s_held = float(scene.score()[0])
    release_tray(300)
    report("released")
    check("TRAY-HELD-UP GATE: with everything binned+crated but the tray still pressed "
          f"at {a11:+.1f} deg, success is False and the score is capped at 0.80; "
          "released onto its rest stop, the rest conjunct flips it True (score 1.0)",
          all_binned11 and held_blocked and s_held <= c.cap + 1e-4
          and bool(scene.tray_at_rest()[0]) and bool(scene.success()[0])
          and float(scene.score()[0]) >= 0.999)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.riddle_tray")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

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
    except Exception:  # noqa: BLE001 - Kit teardown hangs; die loudly instead
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
