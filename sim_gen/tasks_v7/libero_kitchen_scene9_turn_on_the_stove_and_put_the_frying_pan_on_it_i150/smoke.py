"""Smoke / rubric-REJECTION battery for ShuntStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the claims the task rests on — the
self-aligning shunt stop, the fall-in hazard of the open well, the retainer
windows and the settle gate — are load-bearing. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success() (audited at every step).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; cover sealing the well, grate
                          in a side bay, pan upright on the apron; score 0, no success;
   2. randomization     — two seeded resets: READBACK cover x, grate x, pan xy and
                          pan yaw all differ;
   3. bay side          — over 10 resets the grate bay appears on BOTH sides;
   4. null-policy       — 240 idle steps: score ~0, no success, nothing drifts
                          (sliders hold their rail poses — no phantom creep);
   5. SEED-NAIVE        — pan parked on the CLOSED COVER over the sealed well (the
                          seed-family move: put the pan "on the stove" without turning
                          anything on) -> nothing is judged, score ~0, no success;
   6. HAZARD            — cover slid clear, pan set over the OPEN well WITHOUT the
                          grate: the dish is narrower than the mouth — it tips into
                          the fire pit (tilted/sunk, never seated) -> no success,
                          score <= exposed credit only: the bridge-first analogue is
                          physics, not decree;
   7. pan-in-bay        — pan correctly seated ON the grate while the grate is still
                          parked in its BAY (positive control for the grate-relative
                          seat clause) -> seat latch only, no success, score <= 0.151;
   8. near-miss grate   — cover clear, grate stopped 35 mm short of the well axis
                          (outside the 15 mm tolerance), pan genuinely seated on it ->
                          grate_at_well refuses, no success;
   9. rim-bar perch     — cover clear, grate at the well, pan perched half-over a rim
                          bar (a REAL rest pose: ~10 deg, inside the z band) -> the xy
                          retainer window alone refuses, no success;
  10. settle gate       — the exact success pose written WITH 0.4 m/s velocity ->
                          success() refuses the moving state (dismantled immediately);
  11. MECHANISM         — the solve's fingertip push applied to the GRATE only, from a
                          fresh reset: the cover is displaced to its end stop purely by
                          tab-edge CONTACT and the grate ends centred by the stop; the
                          actuator provably moved (non-vacuous probe) -> score 0.40,
                          no success (no pan on the grate);
  12. rejection audit   — success() was never True at ANY step of the battery;
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i150.smoke --headless
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
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"hit": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["hit"] = _AUDIT["hit"] or bool(env.scene.success().any())
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


def _write_state(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                 vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | cover_x={float(scene._rel_x(scene.cover)[0]) * 1000:+.1f}mm "
          f"grate_x={float(scene._rel_x(scene.grate)[0]) * 1000:+.1f}mm "
          f"pan_z={float(scene.pan.data.root_pos_w[0, 2]):.3f} "
          f"seated={bool(scene.pan_seated()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shunt_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 0.80)) + o),
                                tuple(np.array((0.00, 0.00, 0.12)) + o),
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

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def x_c() -> float:
        _refresh()
        return float(scene._rel_x(scene.cover)[0])

    def x_g() -> float:
        _refresh()
        return float(scene._rel_x(scene.grate)[0])

    def bench_xy() -> torch.Tensor:
        _refresh()
        return scene.bench.data.root_pos_w[:, :2].clone()

    oz = float(env.iscene.env_origins[0, 2])
    z_slider = oz + c.slider_z
    z_pan_on_grate = oz + c.grate_top + c.pan_bottom_dz  # exact seat height
    yaw_s = _qz(torch.full((n,), -math.pi / 2, device=device))  # handle SOUTH

    def side_sign() -> float:
        return 1.0 if x_g() > 0 else -1.0

    def put(body, xy_w: torch.Tensor, z_center: float, quat=None,
            drop: float = 0.004, vel_x: float = 0.0) -> None:
        """CONSTRUCT: write the body just above its rest pose (probe
        instrumentation; the caller settles when the arrangement is complete)."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0:2] = xy_w
        pos[:, 2] = z_center + drop
        _write_state(body, pos, quat, vel_x)

    def slide_to(body, x_rel: float) -> None:
        """CONSTRUCT: teleport a rail slider ALONG its joint axis (within
        limits, on the constraint manifold), then let it settle calm."""
        xy = bench_xy()
        xy[:, 0] += x_rel
        put(body, xy, z_slider, None, drop=0.0)
        _step(40)

    def pan_onto_grate(dx: float = 0.0, dy: float = 0.0, drop: float = 0.004) -> None:
        _refresh()
        xy = scene.grate.data.root_pos_w[:, :2].clone()
        xy[:, 0] += dx
        xy[:, 1] += dy
        put(scene.pan, xy, z_pan_on_grate + 0.004, yaw_s, drop=drop)
        _step(120)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.pan.data.root_pos_w).all()
               and torch.isfinite(scene.grate.data.root_pos_w).all()
               and torch.isfinite(scene.cover.data.root_pos_w).all())
    check("settle/no-NaN: cover sealing the well, grate in a bay, pan upright on "
          "the apron; score 0, no success",
          fin and abs(x_c()) <= c.cover_jitter + 0.005
          and abs(x_g()) >= c.bay_x - c.bay_jitter - 0.005
          and float(scene._up_w(scene.pan)[0, 2]) > 0.95
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (x_c(), x_g(), scene.pan.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pan.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_c, a_g, a_p, a_y = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_c, b_g, b_p, b_y = readback()
    d_cov, d_gr = abs(a_c - b_c), abs(a_g - b_g)
    d_pan, d_yw = float((a_p - b_p).norm()), dyaw(a_y, b_y)
    print(f"[smoke] randomization deltas: cover_x={d_cov * 1000:.1f}mm "
          f"grate_x={d_gr * 1000:.1f}mm pan_xy={d_pan * 1000:.1f}mm "
          f"pan_yaw={d_yw:.1f}deg", flush=True)
    check("randomization-is-real: cover x, grate x, pan xy, pan yaw readback differ",
          d_cov > 0.001 and d_gr > 0.001 and d_pan > 0.005 and d_yw > 3.0)

    # ================= 3. grate bay side permutation ==============================================
    sides = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        gx = x_g()
        assert (gx > 0) == (int(scene.side[0]) > 0), "side flag vs grate pose mismatch"
        sides.append(1 if gx > 0 else -1)
    print(f"[smoke] grate bay sides over 10 resets: {sides}", flush=True)
    check("bay side: the grate bay appears on BOTH sides over 10 resets",
          (1 in sides) and (-1 in sides))

    # ================= 4. null policy fails (and nothing creeps) ==================================
    torch.manual_seed(100)
    env.reset()
    _step(10)
    c0, g0 = x_c(), x_g()
    _step(230)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, sliders hold "
          "their rail poses",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and abs(x_c() - c0) < 0.01 and abs(x_g() - g0) < 0.01)

    # ================= 5. SEED-NAIVE: pan parked on the closed cover ==============================
    # The seed-family move — put the pan "on the stove" without turning anything
    # on. The well is sealed; the pan ends up sitting on the cover plate.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    xy = scene.cover.data.root_pos_w[:, :2].clone()
    xy[:, 1] += 0.055  # beside the knob, still fully on the plate
    put(scene.pan, xy, z_slider + c.plate_t / 2 + c.pan_bottom_dz + 0.004, yaw_s)
    _step(120)
    _report("seed-naive")
    check("SEED-NAIVE: pan parked on the CLOSED cover over the sealed well — "
          "nothing is judged: no success, score ~0, pan still up at counter level",
          not bool(scene.pan_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01
          and float(scene.pan.data.root_pos_w[0, 2]) > oz + c.deck_h)
    _REC["on"] = False

    # ================= 6. HAZARD: the open well swallows an ungrated pan =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    e = side_sign()
    slide_to(scene.cover, -e * 0.190)  # cover clear (away from the grate's bay)
    _step(30)                          # calm -> the exposed latch fires (0.20)
    put(scene.pan, bench_xy(), oz + c.deck_h + c.pan_bottom_dz + 0.040, yaw_s)
    _step(180)
    _report("hazard")
    up_z = float(scene._up_w(scene.pan)[0, 2])
    pz = float(scene.pan.data.root_pos_w[0, 2])
    print(f"[smoke] hazard: pan z={pz - oz:.3f} (deck {c.deck_h:.3f}) up_z={up_z:+.2f}",
          flush=True)
    check("HAZARD: pan set over the OPEN well without the grate tips into the fire "
          "pit (sunk/tilted, never seated) — no success, only the exposed credit",
          (pz < oz + 0.140 or up_z < 0.90)
          and not bool(scene.pan_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.201)
    _REC["on"] = False

    # ================= 7. pan seated on the grate IN ITS BAY ======================================
    # Positive control for the grate-relative seat clause: the seat latch fires
    # wherever the grate is, but a bayed grate is no stove -> no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pan_onto_grate()
    _report("pan-in-bay")
    check("pan-in-bay: pan correctly seated ON the grate parked in its bay — seat "
          "latch only, grate not at the well: no success, score <= 0.151",
          bool(scene.pan_seated()[0]) and not bool(scene.grate_at_well()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.151)

    # ================= 8. near-miss: grate 35 mm short of the well axis ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    e = side_sign()
    slide_to(scene.cover, -e * 0.199)
    slide_to(scene.grate, e * 0.035)   # outside the 15 mm tolerance
    pan_onto_grate()
    _report("near-miss")
    gx = abs(x_g())
    check("near-miss (grate): pan genuinely seated on a grate stopped 35 mm short "
          "of the well axis — grate_at_well refuses: no success, score <= 0.351",
          bool(scene.pan_seated()[0]) and gx > c.grate_x_tol + 0.005
          and not bool(scene.grate_at_well()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.351)

    # ================= 9. rim-bar perch: the retainer window is load-bearing ======================
    # A REAL rest pose: the pan's north edge rides the rim bar (~10 deg, inside
    # the 12 deg cone and the z band at mid-plate) — ONLY the xy window refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    e = side_sign()
    slide_to(scene.cover, -e * 0.199)
    slide_to(scene.grate, e * 0.002)
    pan_onto_grate(dy=0.055)
    _report("bar-perch")
    _refresh()
    dyy = float((scene.pan.data.root_pos_w[0, 1] - scene.grate.data.root_pos_w[0, 1]))
    print(f"[smoke] bar-perch: pan dy={dyy * 1000:+.1f}mm (tol "
          f"{c.pan_xy_tol * 1000:.0f}mm) up_z={float(scene._up_w(scene.pan)[0, 2]):+.3f}",
          flush=True)
    check("rim-bar perch: pan resting half-over a rim bar at the well — the xy "
          "retainer window refuses: not seated, no success, score <= 0.401",
          abs(dyy) > c.pan_xy_tol and not bool(scene.pan_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.401)

    # ================= 10. settle gate: the success pose in motion is refused =====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    e = side_sign()
    slide_to(scene.cover, -e * 0.199)
    slide_to(scene.grate, e * 0.002)
    _refresh()
    xy = scene.grate.data.root_pos_w[:, :2].clone()
    put(scene.pan, xy, z_pan_on_grate + 0.002, yaw_s, drop=0.0, vel_x=0.40)
    _step(1)
    _report("settle-gate")
    moving_refused = bool(scene.pan_seated()[0]) and not bool(scene.success()[0]) \
        and float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]) > c.settle_speed
    # dismantle IMMEDIATELY (before the slide damps into a genuine success)
    xy = bench_xy()
    xy[:, 1] += c.pan_y_nom
    put(scene.pan, xy, oz + c.pan_spawn_z, yaw_s)
    _step(40)
    check("settle gate: the exact success pose moving at 0.4 m/s is refused "
          "(geometry alone is not success; dismantled before it could calm)",
          moving_refused and not bool(scene.success()[0]))

    # ================= 11. MECHANISM: the shunt really is contact-made ============================
    # The solve's fingertip push, applied to the GRATE ONLY from a fresh reset:
    # the cover must end at its stop having been moved purely by tab-edge
    # contact, and the grate must end centred BY the stop. The probe asserts the
    # actuator itself moved (no vacuous pass) and that no force ever touched the
    # cover or the pan.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    g_start, c_start = x_g(), x_c()
    e = side_sign()
    all_ids = _all_ids()
    zero_w = torch.zeros(n, 1, 3, device=device)
    v_des = -e * 0.15
    stall = 0
    no_action = torch.empty(0, device=device)
    for i in range(900):
        v = float(scene.grate.data.root_lin_vel_w[0, 0])
        f = max(-8.0, min(8.0, c.grate_mass * 20.0 * (v_des - v)))
        fw = torch.zeros(n, 1, 3, device=device)
        fw[:, 0, 0] = f
        scene.grate.set_external_force_and_torque(fw, zero_w, env_ids=all_ids,
                                                  is_global=True)
        env.step(no_action)
        _AUDIT["hit"] = _AUDIT["hit"] or bool(scene.success().any())
        if abs(x_g()) <= 0.006:
            break
        stall = stall + 1 if abs(v) < 0.005 and i > 60 else 0
        if stall > 90 and abs(x_g()) <= 0.02:
            break
    scene.grate.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                              is_global=True)
    _step(60)
    _report("mechanism")
    print(f"[smoke] mechanism: grate {g_start * 1000:+.1f} -> {x_g() * 1000:+.1f}mm, "
          f"cover {c_start * 1000:+.1f} -> {x_c() * 1000:+.1f}mm (contact only)",
          flush=True)
    check("MECHANISM: pushing the grate alone shunts the cover to its stop by "
          "contact and the stop centres the grate (actuator provably moved) — "
          "score 0.40, still no success",
          abs(x_g() - g_start) > 0.15 and abs(x_c()) >= c.exposed_min
          and abs(x_g()) <= c.grate_x_tol
          and 0.39 <= float(scene.score()[0]) <= 0.401
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shunt_stove")
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
    main()
