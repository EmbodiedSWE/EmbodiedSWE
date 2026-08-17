"""Smoke / rubric-REJECTION battery for CarouselDispatchScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real contact-driven trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the physical claims the task rests on — the
carousel is the only path in, the plinth blocks the ground, the roof blocks the air,
a flyby is not alignment — are load-bearing. Every probe is CONSTRUCTED (teleport,
real physics steps, judge) — instrumentation, never a solution: no probe here reaches
success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; platter seated + centred on its
                           bearing, cargo in the bay; score 0, no success;
   2. randomization      — two seeded resets: READBACK pedestal xy, pedestal yaw and
                           bay->red start error all differ;
   3. dock shuffle       — over 15 resets the RED garage occupies ALL THREE azimuth
                           slots (readback from the dock's world pose, not internals);
   4. null-policy        — 300 idle steps: the platter does not drift (start error
                           unchanged), cargo stays aboard, score ~0, no success;
   5. MISALIGNED SHOVE   — mechanism is load-bearing: with the bay >= 60 deg away, the
                           cargo is shoved straight toward the red garage at 3x its
                           weight for 1.5 s — the force visibly acts (the cube rides,
                           the platter is dragged around), but the bay retains the
                           cube: it never gets inside any garage; no success;
   6. SEED STRATEGY      — the seed's plan (drag the cube along the ground to the
                           target): cargo CONSTRUCTED on the ground at the red mouth
                           axis, then shoved at the mouth at 3x weight for 1.5 s — it
                           advances to the plinth but cannot climb the 66 mm blank
                           face; ends at ground level, outside, score ~0, no success;
   7. FLYBY              — the platter is spun FAST (>= 1.0 rad/s) through the aligned
                           window and braked far past it: the `aligned` latch must NOT
                           fire (`spin_still` clause), cargo still aboard;
   8. wrong garage       — cargo settled fully inside the GREEN garage — no success,
                           no delivered credit;
   9. near-miss half-in  — cargo resting in the red mouth with its centre 15 mm short
                           of the full-inside plane (rear face proud of the mouth) —
                           no success, score <= 0.6001;
  10. drop-in blocked    — cargo dropped from above the red garage lands ON THE ROOF
                           and stays there — never inside, no success;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.reach_and_drag_i111.smoke --headless
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    cl = scene._dock_local("red", scene.cargo.data.root_pos_w)[0]
    err = math.degrees(float(scene.align_err_rad()[0]))
    print(f"[smoke] {tag:18s} | err={err:+7.1f}deg "
          f"cargo_red_loc=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
          f"in_bay={bool(scene.cargo_in_bay()[0])} "
          f"aligned={bool(scene._aligned[0])} delivered={bool(scene._delivered[0])} "
          f"resting={bool(scene.resting_in_red()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cargo_carousel")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.75, 0.70)) + o),
                                tuple(np.array((0.40, 0.00, 0.07)) + o),
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

    def err_deg() -> float:
        _refresh()
        return math.degrees(float(scene.align_err_rad()[0]))

    def dock_world(name: str, loc_xyz) -> torch.Tensor:
        """Dock-local point -> world (per-env)."""
        _refresh()
        d = scene.docks[name]
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return d.data.root_pos_w + quat_apply(d.data.root_quat_w, loc)

    def cargo_red_x() -> float:
        _refresh()
        return float(scene._dock_local("red", scene.cargo.data.root_pos_w)[0, 0])

    def reset_with_err(min_err_deg: float, seed0: int) -> int:
        """Seeded reset whose bay->red start error exceeds `min_err_deg`."""
        for s in range(seed0, seed0 + 12):
            torch.manual_seed(s)
            env.reset()
            _step(60)
            if abs(err_deg()) > min_err_deg:
                return s
        raise AssertionError(f"no seed in [{seed0},{seed0 + 12}) with err > {min_err_deg}")

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    pz = float((scene.platter.data.root_pos_w - scene.env_origins)[0, 2])
    off = float((scene.platter.data.root_pos_w[0, :2]
                 - scene.pedestal.data.root_pos_w[0, :2]).norm())
    print(f"[smoke] platter rest: z={pz:.4f} (nominal {c.z_platter:.4f}) "
          f"centre_off={off * 1000:.1f}mm", flush=True)
    check("settle/no-NaN: layout settles finite; platter seated and centred on its "
          "bearing, cargo in the bay; score 0, no success",
          bool(scene._finite()[0]) and abs(pz - c.z_platter) < 0.006 and off < 0.012
          and bool(scene.cargo_in_bay()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.pedestal.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pedestal.data.root_quat_w[0]),
                err_deg())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_xy, a_yaw, a_err = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_xy, b_yaw, b_err = readback()
    d_xy = float((a_xy - b_xy).norm())
    d_yaw = abs(a_yaw - b_yaw)
    d_err = abs(a_err - b_err)
    print(f"[smoke] randomization deltas: base_xy={d_xy * 1000:.1f}mm "
          f"base_yaw={d_yaw:.1f}deg start_err={d_err:.1f}deg", flush=True)
    check("randomization-is-real: pedestal xy, pedestal yaw and bay->red start error "
          "readback differ across seeds",
          d_xy > 0.003 and d_yaw > 1.0 and d_err > 5.0)

    # ================= 3. dock color shuffle (readback from world poses) ==========================
    slots_seen: set[int] = set()
    slot_az = list(c.slot_az_deg)
    for s in range(15):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        bq = scene.pedestal.data.root_quat_w[0]
        byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
        d = (scene.docks["red"].data.root_pos_w[0, :2]
             - scene.pedestal.data.root_pos_w[0, :2])
        az = (math.degrees(math.atan2(float(d[1]), float(d[0]))) - byaw) % 360.0
        slot = min(range(3), key=lambda i: min(abs(az - slot_az[i]),
                                               360.0 - abs(az - slot_az[i])))
        slots_seen.add(slot)
    print(f"[smoke] over 15 resets the RED garage occupied slots {sorted(slots_seen)} "
          f"(azimuths {[slot_az[i] for i in sorted(slots_seen)]})", flush=True)
    check("dock shuffle: the RED garage occupies ALL THREE azimuth slots over 15 "
          "resets (world-pose readback)", slots_seen == {0, 1, 2})

    # ================= 4. null policy fails (and the bearing holds) ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    e0 = err_deg()
    _step(300)
    _report("null-policy")
    e1 = err_deg()
    check("null-policy-fails: 300 idle steps, the platter holds its angle "
          f"(drift {abs(e1 - e0):.2f} deg), cargo aboard, score ~0, no success",
          abs(e1 - e0) < 2.0 and bool(scene.cargo_in_bay()[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. MISALIGNED SHOVE: the carousel is the only way in =======================
    # With the bay >= 60 deg away from the red garage, shove the cargo straight toward
    # the red garage at 3x its weight for 1.5 s. The cube must MOVE (the probe is not
    # vacuous) yet end outside every garage: rails/backstop resist, the deck edge drops
    # it to the ground, and the 66 mm plinth face is taller than the cube's CoM.
    s5 = reset_with_err(60.0, 400)
    print(f"[smoke] misaligned-shove seed {s5}: start_err={err_deg():+.1f}deg", flush=True)
    _REC["on"] = True
    e5_0 = err_deg()
    p0 = scene.cargo.data.root_pos_w[0].clone()
    u = (scene.docks["red"].data.root_pos_w[0, :3] - scene.cargo.data.root_pos_w[0, :3])
    u[2] = 0.0
    u = u / u.norm().clamp(min=1e-9)
    f = 3.0 * c.cargo_mass * 9.81 * u
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(scene.cargo, f, 20)
    _step(120)
    _report("misaligned-shove")
    moved = float((scene.cargo.data.root_pos_w[0] - p0).norm())
    in_any = any(bool(((scene._dock_local(nm, scene.cargo.data.root_pos_w)[0, 0]
                        > c.inside_x_min)
                       & (scene._dock_local(nm, scene.cargo.data.root_pos_w)[0, 0]
                          < c.dock_x_back)
                       & (scene._dock_local(nm, scene.cargo.data.root_pos_w)[0, 1].abs()
                          < c.dock_y_half)
                       & (scene._dock_local(nm, scene.cargo.data.root_pos_w)[0, 2]
                          > c.plinth_top - 0.005)
                       & (scene._dock_local(nm, scene.cargo.data.root_pos_w)[0, 2]
                          < c.roof_z)))
                 for nm in ("red", "green", "blue"))
    e5_1 = err_deg()
    print(f"[smoke] shove displaced the cargo {moved * 1000:.0f}mm, dragged the "
          f"platter {abs(e5_1 - e5_0):.1f}deg; inside any garage: {in_any}", flush=True)
    check("MISALIGNED SHOVE: 3x-weight shove toward the red garage with the bay >= "
          "60 deg away visibly acts (cube and platter move) but never gets the cube "
          "inside ANY garage; no success",
          (moved > 0.020 or abs(e5_1 - e5_0) > 5.0) and not in_any
          and not bool(scene._delivered[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. SEED STRATEGY: ground-drag to the target ================================
    # The seed's whole plan is to DRAG the cube along the ground to the target. Here the
    # ground lane in front of the mouth is a 10 mm deck-to-plinth gap — a 90 mm cube
    # cannot even STAND there. Build the drag: cube on OPEN ground between two garages,
    # then a quasi-static 3x-weight drag (speed-capped: dragging, not hurling) aimed at
    # the point in front of the red mouth. It travels freely across the ground, then
    # ends pressed against blank plinth/deck faces BELOW floor level — never inside.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    b_xy = scene.pedestal.data.root_pos_w[0, :2]
    d_xy = scene.docks["red"].data.root_pos_w[0, :2] - b_xy
    th = math.atan2(float(d_xy[1]), float(d_xy[0])) + math.radians(45.0)
    p = torch.zeros(n, 3, device=device)
    p[:, 0] = b_xy[0] + 0.32 * math.cos(th)
    p[:, 1] = b_xy[1] + 0.32 * math.sin(th)
    p[:, 2] = env.iscene.env_origins[:, 2] + c.cargo_size / 2 + 0.003
    _write_body(scene.cargo, p, scene.docks["red"].data.root_quat_w)
    _step(60)
    p0 = scene.cargo.data.root_pos_w[0].clone()
    target = dock_world("red", (c.dock_mouth_x - 0.020, 0.0, 0.0))[0]
    u6 = target - scene.cargo.data.root_pos_w[0]
    u6[2] = 0.0
    u6 = u6 / u6.norm().clamp(min=1e-9)
    f6 = (3.0 * c.cargo_mass * 9.81 * u6).view(1, 1, 3).expand(n, 1, 3).contiguous()
    zero6 = torch.zeros(n, 1, 3, device=device)
    for _ in range(420):  # quasi-static drag: force only while slow (<= 0.35 m/s)
        v = float(scene.cargo.data.root_lin_vel_w[0, :2].norm())
        scene.cargo.set_external_force_and_torque(
            f6 if v < 0.35 else zero6, zero6, env_ids=_all_ids(), is_global=True)
        _step(1)
    scene.cargo.set_external_force_and_torque(zero6, zero6, env_ids=_all_ids())
    _step(120)
    _report("seed-strategy")
    moved6 = float((scene.cargo.data.root_pos_w[0] - p0).norm())
    x1 = cargo_red_x()
    z1 = float(scene._dock_local("red", scene.cargo.data.root_pos_w)[0, 2])
    print(f"[smoke] ground drag: travelled {moved6 * 1000:.0f}mm, ends at red-frame "
          f"x={x1 * 1000:.0f}mm z={z1 * 1000:.0f}mm (floor rest would be "
          f"{(c.plinth_top + c.cargo_size / 2) * 1000:.0f}mm)", flush=True)
    check("negative (SEED strategy): a quasi-static ground drag toward the red garage "
          "travels freely but ends blocked below floor level at blank plinth/deck "
          "faces — never inside, score ~0, no success",
          moved6 > 0.050 and z1 < 0.095 and not bool(scene.in_red_zone()[0])
          and not bool(scene._delivered[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. FLYBY: fast rotation through the window is not alignment ================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    e_start = err_deg()
    sgn = 1.0 if e_start > 0 else -1.0
    zero = torch.zeros(n, 1, 3, device=device)
    tq = torch.zeros(n, 1, 3, device=device)
    min_abs_err, crossed, w_at_cross = 180.0, False, 0.0
    for i in range(900):
        e = err_deg()
        w = float(scene.platter.data.root_ang_vel_w[0, 2])
        if abs(e) < min_abs_err:
            min_abs_err, w_at_cross = abs(e), w
        if not crossed and abs(e) < c.align_tol_deg:
            crossed = True
        if crossed and abs(e) > 25.0:
            break
        tq[:, 0, 2] = sgn * (2.5 if abs(w) < 1.2 else 0.0)  # spin FAST, hold ~1.2 rad/s
        scene.platter.set_external_force_and_torque(zero, tq, env_ids=_all_ids(),
                                                    is_global=True)
        _step(1)
    # brake to rest far from the window
    for _ in range(240):
        w = float(scene.platter.data.root_ang_vel_w[0, 2])
        if abs(w) < 0.05:
            break
        tq[:, 0, 2] = -2.0 * (1.0 if w > 0 else -1.0)
        scene.platter.set_external_force_and_torque(zero, tq, env_ids=_all_ids(),
                                                    is_global=True)
        _step(1)
    scene.platter.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    _report("flyby")
    print(f"[smoke] flyby: window crossed={crossed} min|err|={min_abs_err:.1f}deg at "
          f"|w|={abs(w_at_cross):.2f} rad/s (spin_still {c.spin_still}), "
          f"final err={err_deg():+.1f}deg", flush=True)
    check("FLYBY: sweeping the bay through the aligned window at >= 1 rad/s and "
          "stopping far past it does NOT latch `aligned`; cargo still aboard",
          crossed and abs(w_at_cross) > 2 * c.spin_still
          and abs(err_deg()) > c.align_tol_deg and not bool(scene._aligned[0])
          and bool(scene.cargo_in_bay()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 8. negative: wrong garage ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = dock_world("green", (0.020, 0.0, c.plinth_top + c.cargo_size / 2 + 0.006))
    _write_body(scene.cargo, p, scene.docks["green"].data.root_quat_w)
    _step(120)
    _report("wrong-garage")
    gl = scene._dock_local("green", scene.cargo.data.root_pos_w)[0]
    check("negative (wrong garage): cargo settled fully inside the GREEN garage — "
          "no delivered credit, no success",
          float(gl[0]) > c.inside_x_min and float(gl[2]) < c.rest_z_max
          and not bool(scene.in_red_zone()[0]) and not bool(scene._delivered[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 9. near-miss: half-in at the red mouth =====================================
    # Centre 15 mm SHORT of the full-inside plane: rear face proud of the mouth. On the
    # garage floor, in the mouth, at rest — only the full-inside clause fails.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = dock_world("red", (c.inside_x_min - 0.015, 0.0,
                           c.plinth_top + c.cargo_size / 2 + 0.006))
    _write_body(scene.cargo, p, scene.docks["red"].data.root_quat_w)
    _step(120)
    _report("half-in")
    x = cargo_red_x()
    print(f"[smoke] half-in: cargo centre x_loc={x * 1000:.0f}mm "
          f"(full-inside plane {c.inside_x_min * 1000:.0f}mm)", flush=True)
    check("near-miss (half-in): cargo at rest on the red garage floor with its rear "
          "face proud of the mouth — full-inside clause refuses; no success, "
          "score <= 0.6001",
          x < c.inside_x_min - 0.005 and not bool(scene.in_red_zone()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.6001)
    _REC["on"] = False

    # ================= 10. drop-in blocked: the roof is load-bearing ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p = dock_world("red", (0.020, 0.0, c.roof_z + 0.200))
    _write_body(scene.cargo, p, scene.docks["red"].data.root_quat_w)
    _step(180)
    _report("drop-in")
    zl = float(scene._dock_local("red", scene.cargo.data.root_pos_w)[0, 2])
    print(f"[smoke] drop: cargo rests at z_loc={zl * 1000:.0f}mm "
          f"(roof top ~{(c.roof_z + 0.010) * 1000:.0f}mm)", flush=True)
    check("negative (drop-in blocked): cargo dropped from above the red garage lands "
          "ON the roof and stays out — no delivered credit, no success",
          zl > c.roof_z and not bool(scene.in_red_zone()[0])
          and not bool(scene._delivered[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cargo_carousel")
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
