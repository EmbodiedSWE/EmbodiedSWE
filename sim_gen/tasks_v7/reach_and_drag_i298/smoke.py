"""Smoke / rubric-REJECTION battery for DropCourierScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome, with monotone
latched credit, along a real contact-driven trajectory). This battery proves the rubric
REJECTS wrong outcomes and that the physical claims the task rests on are load-bearing:
the drop over the walls is the ONLY way into the tray, the guide wall keeps ground cargo
out of the channel, a drive-by under the drop line is not alignment, and a misdrop is
refused. Every probe is CONSTRUCTED (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite: cargo on the ledge, tray in the
                          channel; score 0, no success;
   2. randomization     — two seeded resets: READBACK rig xy, rig yaw, cargo slot and
                          tray start all differ;
   3. bay-end shuffle   — over 14 resets the green post stands at BOTH channel ends
                          (world-pose readback, consistent with `bay_sign`);
   4. null-policy       — 300 idle steps: tray does not drift, cargo stays on the
                          ledge, score ~0, no success;
   5. SEED STRATEGY     — the seed's whole plan (drag the cube across the open ground
                          onto the target): cargo CONSTRUCTED on the open ground, then
                          quasi-static 3x-weight drag aimed at the tray — it travels
                          freely but the guide wall (top above the cube's centre)
                          refuses entry into the channel; never contained, score ~0;
   6. GROUND LOAD SHOVE — walls are load-bearing: cargo on the channel floor beside
                          the tray, quasi-static 3x-weight shove at the tray — the
                          cube shoves the TRAY around instead of entering it (wall top
                          62 mm > cube centre 45 mm); never contained, no success;
   7. WRONG ORDER       — tray parked at the bay FIRST, then the cargo sent off the
                          drop edge: it falls onto the channel floor, not into the
                          tray — a misdrop earns nothing (score ~0, no success);
   8. FLYBY             — the tray swept UNDER the drop line at speed without
                          stopping: the `aligned` latch must NOT fire (`tray_still`
                          clause);
   9. near-miss park    — loaded tray constructed 60 mm short of the bay band:
                          contained but not parked — no success, score <= 0.4501;
  10. flipped tray      — tray upside-down with the cargo resting ON its upturned
                          floor: supported by the tray yet not contained (tray-body
                          frame) and not upright — no credit, no success;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.reach_and_drag_i298.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tl = scene._rig_local(scene.tray.data.root_pos_w)[0]
    cl = scene._rig_local(scene.cargo.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
          f"{float(tl[2]):+.3f}) cargo=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
          f"{float(cl[2]):+.3f}) on_ledge={bool(scene.cargo_on_ledge()[0])} "
          f"contained={bool(scene.contained()[0])} parked={bool(scene.parked()[0])} "
          f"aligned_l={bool(scene._aligned[0])} loaded_l={bool(scene._loaded[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_courier")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.85, 0.75)) + o),
                                tuple(np.array((0.45, 0.00, 0.10)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def rig_world(loc_xyz) -> torch.Tensor:
        """Rig-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)

    def tray_y() -> float:
        _refresh()
        return float(scene._rig_local(scene.tray.data.root_pos_w)[0, 1])

    def cargo_loc() -> torch.Tensor:
        _refresh()
        return scene._rig_local(scene.cargo.data.root_pos_w)[0]

    def rig_q() -> torch.Tensor:
        return scene.rig.data.root_quat_w

    def qs_drag(body, target_w: torch.Tensor, mass: float, steps: int,
                v_cap: float = 0.35) -> None:
        """Quasi-static 3x-weight drag toward a world point: horizontal force applied
        only while slow (dragging, not hurling — a slam could vault a wall that the
        static geometry honestly refuses)."""
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            _refresh()
            u = target_w - body.data.root_pos_w[0]
            u[2] = 0.0
            nrm = float(u.norm())
            if nrm < 0.01:
                break
            u = u / max(nrm, 1e-9)
            v = float(body.data.root_lin_vel_w[0, :2].norm())
            f = (3.0 * mass * 9.81 * u).view(1, 1, 3).expand(n, 1, 3).contiguous()
            body.set_external_force_and_torque(f if v < v_cap else zero, zero,
                                               env_ids=_all_ids(), is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    cl = cargo_loc()
    check("settle/no-NaN: seeded reset settles finite — cargo on the ledge, tray in "
          "the channel; score 0, no success",
          bool(scene._finite()[0]) and bool(scene.cargo_on_ledge()[0])
          and bool(scene.tray_in_channel()[0]) and abs(float(cl[2]) - (c.ledge_h + 0.045)) < 0.02
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.rig.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.rig.data.root_quat_w[0]),
                float(cargo_loc()[1]),
                tray_y())

    obs = []
    for s2 in (101, 202, 303):
        torch.manual_seed(s2)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_xy = max(float((obs[i][0] - obs[j][0]).norm()) for i, j in pairs)
    d_yaw = max(abs(obs[i][1] - obs[j][1]) for i, j in pairs)
    d_cy = max(abs(obs[i][2] - obs[j][2]) for i, j in pairs)
    d_ty = max(abs(obs[i][3] - obs[j][3]) for i, j in pairs)
    print(f"[smoke] randomization max pairwise deltas over 3 seeds: "
          f"rig_xy={d_xy * 1000:.1f}mm rig_yaw={d_yaw:.1f}deg "
          f"cargo_y={d_cy * 1000:.1f}mm tray_y={d_ty * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rig xy, rig yaw, cargo slot and tray start "
          "readback all differ across 3 seeds (max pairwise)",
          d_xy > 0.003 and d_yaw > 1.0 and d_cy > 0.010 and d_ty > 0.010)

    # ================= 3. bay-end shuffle (green post readback) ===================================
    signs_seen: set[int] = set()
    consistent = True
    for s in range(14):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        py = float(scene._rig_local(scene.post.data.root_pos_w)[0, 1])
        sgn = 1 if py > 0 else -1
        signs_seen.add(sgn)
        consistent = consistent and abs(py - float(scene.bay_sign[0]) * c.post_y) < 0.01
    print(f"[smoke] over 14 resets the green post stood at ends {sorted(signs_seen)} "
          f"(readback consistent with bay_sign: {consistent})", flush=True)
    check("bay-end shuffle: the green post stands at BOTH channel ends over 14 "
          "resets (world-pose readback, consistent with bay_sign)",
          signs_seen == {-1, 1} and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ty0 = tray_y()
    _step(300)
    _report("null-policy")
    drift = abs(tray_y() - ty0)
    check("null-policy-fails: 300 idle steps — tray holds its start "
          f"(drift {drift * 1000:.1f} mm), cargo on the ledge, score ~0, no success",
          drift < 0.005 and bool(scene.cargo_on_ledge()[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: ground-drag onto the target ==============================
    # The seed's plan: drag the cube across the open ground onto the target. Construct
    # it: cargo on the OPEN ground outside the rig, then a quasi-static 3x-weight drag
    # aimed straight at the tray (the goal container). It travels freely across the
    # ground, then the guide wall (top 55 mm, above the 45 mm cube centre) refuses
    # channel entry — and even in the channel the tray walls (62 mm) do the same.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ty5 = tray_y()
    p = rig_world((-0.32, ty5, c.cube_size / 2 + 0.003))
    _write_body(scene.cargo, p, rig_q())
    _step(60)
    p0 = scene.cargo.data.root_pos_w[0].clone()
    qs_drag(scene.cargo, scene.tray.data.root_pos_w[0].clone(), c.cube_mass, 480)
    _step(120)
    _report("seed-strategy")
    moved5 = float((scene.cargo.data.root_pos_w[0] - p0).norm())
    cl5 = cargo_loc()
    print(f"[smoke] ground drag: travelled {moved5 * 1000:.0f}mm, ends at rig-frame "
          f"x={float(cl5[0]) * 1000:.0f}mm (wall outer face "
          f"{(c.wall_inner_x - 0.015) * 1000:.0f}mm), z={float(cl5[2]) * 1000:.0f}mm",
          flush=True)
    check("negative (SEED strategy): a quasi-static ground drag aimed at the tray "
          "travels freely but is refused at the guide wall — cargo never in the "
          "channel, never contained, score ~0, no success",
          moved5 > 0.080 and float(cl5[0]) < c.wall_inner_x - 0.030
          and not bool(scene.contained()[0]) and not bool(scene._loaded[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. GROUND LOAD SHOVE: tray walls are load-bearing ==========================
    # Cargo constructed ON the channel floor beside the tray, then a quasi-static
    # 3x-weight shove straight at the tray. The wall top (62 mm) is above the cube's
    # centre (45 mm): the cube shoves the whole TRAY down the channel instead of
    # climbing in. This is exactly why a misdrop is unrecoverable.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ty6 = tray_y()
    sgn6 = -1.0 if ty6 > 0 else 1.0     # approach from the side with more room
    p = rig_world((c.tray_x, ty6 + sgn6 * 0.145, c.cube_size / 2 + 0.003))
    _write_body(scene.cargo, p, rig_q())
    _step(60)
    p0 = scene.cargo.data.root_pos_w[0].clone()
    t0 = tray_y()
    qs_drag(scene.cargo, scene.tray.data.root_pos_w[0].clone(), c.cube_mass, 600)
    _step(150)
    _report("ground-shove")
    moved6 = float((scene.cargo.data.root_pos_w[0] - p0).norm())
    tray_moved = abs(tray_y() - t0)
    cl6 = cargo_loc()
    print(f"[smoke] ground shove: cube moved {moved6 * 1000:.0f}mm, shoved the tray "
          f"{tray_moved * 1000:.0f}mm; cube z={float(cl6[2]) * 1000:.0f}mm", flush=True)
    check("negative (GROUND LOAD SHOVE): a quasi-static 3x-weight shove at the tray "
          "visibly acts (cube and tray both move) but the cube never enters — "
          "never contained, no success",
          moved6 > 0.030 and tray_moved > 0.015 and float(cl6[2]) < 0.075
          and not bool(scene.contained()[0]) and not bool(scene._loaded[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. WRONG ORDER: park first, then drop ======================================
    # A cargo sent off the edge with the tray parked at the bay lands on the channel
    # floor. The forced order is align-then-drop; doing it backwards earns nothing.
    s7 = None
    for s in range(700, 712):
        torch.manual_seed(s)
        env.reset()
        _step(30)
        cy = float(cargo_loc()[1])
        bay = float(scene.bay_sign[0]) * c.bay_center_y
        if abs(cy - bay) > 0.20:
            s7 = s
            break
    assert s7 is not None, "no seed with the cargo slot far from the bay"
    _REC["on"] = True
    cy7 = float(cargo_loc()[1])
    bay7 = float(scene.bay_sign[0]) * c.bay_center_y
    print(f"[smoke] wrong-order seed {s7}: cargo_y={cy7 * 1000:+.0f}mm "
          f"bay_y={bay7 * 1000:+.0f}mm", flush=True)
    # park the (empty) tray at the bay by construction
    _write_body(scene.tray, rig_world((c.tray_x, bay7, 0.002)), rig_q())
    _step(60)
    parked7 = bool(scene.parked()[0])
    # send the cargo over the drop edge at its own slot (free fall from the edge)
    _write_body(scene.cargo, rig_world((c.ledge_face_x - 0.048, cy7,
                                        c.ledge_h + c.cube_size / 2 + 0.010)), rig_q())
    _step(240)
    _report("wrong-order")
    cl7 = cargo_loc()
    check("negative (WRONG ORDER): with the tray parked at the bay first, the "
          "dropped cargo lands on the channel floor — not contained, score ~0, "
          "no success (a misdrop earns nothing)",
          parked7 and float(cl7[2]) < 0.075 and float(cl7[0]) < c.ledge_face_x
          and not bool(scene.contained()[0]) and not bool(scene._loaded[0])
          and not bool(scene._aligned[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. FLYBY: a drive-by under the drop line is not alignment ==================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    err0 = float(scene.align_err()[0])
    sgn8 = 1.0 if err0 > 0 else -1.0    # tray must travel toward the cargo slot
    ey_w = quat_apply(rig_q(), torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    zero = torch.zeros(n, 1, 3, device=device)
    fmax = torch.zeros(n, 1, 3, device=device)
    fmax[0, 0, :3] = ey_w * sgn8 * 7.0
    min_abs_err, v_at_cross = 1.0, 0.0
    for i in range(600):
        _refresh()
        e = float(scene.align_err()[0])
        v = float(scene.tray.data.root_lin_vel_w[0, :3].norm())
        if abs(e) < min_abs_err:
            min_abs_err, v_at_cross = abs(e), v
        if sgn8 * e < -0.10:            # 100 mm past the window: brake
            break
        scene.tray.set_external_force_and_torque(
            fmax if v < 0.30 else zero, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    for _ in range(240):                # brake to rest, far past the window
        _refresh()
        vy = float((scene.tray.data.root_lin_vel_w[0, :2] * ey_w[:2]).sum())
        if abs(vy) < 0.03:
            break
        br = torch.zeros(n, 1, 3, device=device)
        br[0, 0, :3] = -ey_w * math.copysign(5.0, vy)
        scene.tray.set_external_force_and_torque(br, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
    scene.tray.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    _report("flyby")
    e8 = float(scene.align_err()[0])
    print(f"[smoke] flyby: min|err|={min_abs_err * 1000:.1f}mm at "
          f"|v|={v_at_cross:.2f} m/s (tray_still {c.tray_still}), "
          f"final err={e8 * 1000:+.0f}mm", flush=True)
    check("FLYBY: sweeping the tray under the drop line at speed and stopping far "
          "past it does NOT latch `aligned`; cargo still on the ledge, score ~0",
          min_abs_err < c.align_tol and v_at_cross > 2 * c.tray_still
          and abs(e8) > c.align_tol and not bool(scene._aligned[0])
          and bool(scene.cargo_on_ledge()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 9. near-miss: loaded but parked short ======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    bay9 = float(scene.bay_sign[0]) * c.bay_center_y
    y9 = bay9 - math.copysign(0.060, bay9)   # 60 mm short of the bay
    _write_body(scene.tray, rig_world((c.tray_x, y9, 0.002)), rig_q())
    _step(30)
    _write_body(scene.cargo, rig_world((c.tray_x, y9,
                                        c.tray_floor_t + c.cube_size / 2 + 0.006)), rig_q())
    _step(150)
    _report("near-miss-park")
    be9 = float(scene.bay_err()[0])
    print(f"[smoke] near-miss: contained={bool(scene.contained()[0])} "
          f"bay_err={be9 * 1000:.0f}mm (park_tol {c.park_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (park short): cargo contained in the tray 60 mm short of the "
          "bay — loaded credit only, no success, score <= 0.4501",
          bool(scene.contained()[0]) and 0.035 < be9 < 0.100
          and not bool(scene.parked()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.4501)
    _REC["on"] = False

    # ================= 10. flipped tray: containment is body-frame honest =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ty10 = tray_y()
    q_flip = quat_mul(rig_q(), torch.tensor([0.0, 1.0, 0.0, 0.0],
                                            device=device).expand(n, 4))
    _write_body(scene.tray, rig_world((c.tray_x, ty10, c.tray_wall_top + 0.002)), q_flip)
    _step(30)
    _write_body(scene.cargo, rig_world((c.tray_x, ty10,
                                        c.tray_wall_top + c.cube_size / 2 + 0.006)), rig_q())
    _step(150)
    _report("flipped-tray")
    cl10 = cargo_loc()
    print(f"[smoke] flipped tray: cargo rests at z={float(cl10[2]) * 1000:.0f}mm "
          f"(on the upturned floor), upright={bool(scene.tray_upright()[0])}", flush=True)
    check("negative (flipped tray): cargo resting ON the upside-down tray is "
          "supported by it yet NOT contained (tray-body frame) and the tray is not "
          "upright — no credit, no success",
          float(cl10[2]) > 0.085 and not bool(scene.tray_upright()[0])
          and not bool(scene.contained()[0]) and not bool(scene._loaded[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drop_courier")
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
    except Exception:  # noqa: BLE001 - Kit teardown hangs; die loudly instead
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
