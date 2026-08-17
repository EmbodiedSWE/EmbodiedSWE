"""Smoke / rubric-REJECTION battery for ShuttleHatchScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct push-in / chimney-drop /
pull-out episode and the latched credit is monotone along it, two seeds). This battery
proves the rubric REJECTS wrong outcomes, and that the geometry claims the task rests
on — the serve window is sealed above the tray, dropping out of order lands BEHIND the
tray and blocks the load stop, the shuttle is captive behind the stop bar — are
physics, not fiat. Every probe is CONSTRUCTED as a settled state (teleport, real
physics steps, judge) — instrumentation, never a solution. Smoke constructs may write
bodies inside rubric volumes (that is the point of a rubric probe); only solve.py is
barred from doing so.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; tray seated at serve, bottles
                           on the ground; score ~0, no success;
   2. randomization      — two seeded resets: READBACK station yaw, station xy, tray
                           start x and the ketchup bottle's pose all differ;
   3. slot shuffle       — over 10 resets the ketchup bottle occupies >= 2 different
                           ground slots;
   4. null-policy        — 240 idle steps: tray stays put (< 8 mm), score ~0;
   5. SEED STRATEGY      — the seed's plan ("carry the bottle over the tray and lower
                           it in from above"): the bottle ends ON THE CLOSED ROOF —
                           not in the tray, no success, score ~0;
   6. window blocked     — the bottle FLUNG at the serve window (0.7 m/s, through the
                           12 mm-over-the-walls gap region) bounces off the tray's
                           front wall: it demonstrably travels inward yet never enters
                           the tray (non-vacuous: min station-x is asserted);
   7. ORDER violation    — bottle dropped down the chimney with the tray still at
                           serve: lands on the bare plate BEHIND the tray; a 5 N /
                           2.5 s push then moves the tray >= 15 mm but the stranded
                           bottle physically blocks it from ever reaching the load
                           gate; nothing scores;
   8. serve near-miss    — tray parked 22 mm short of the serve gate WITH the bottle
                           inside -> in_tray credit only (0.25), no success;
   9. abandoned at load  — bottle in the tray but the tray left at the back stop ->
                           latched credit capped at 0.40, no success;
  10. wrong object       — MUSTARD in the tray at serve, ketchup on the ground -> no
                           success, no credit;
  11. captive shuttle    — from the load stop an 8 N yank pulls the tray forward: it
                           reaches the serve zone (the pull-out is feasible) but the
                           stop bar holds it — it never leaves the station, never
                           climbs the bar;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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

_qmul = task_scene._qmul

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel_w is not None:
        st[:, 7:10] = lin_vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tl = scene.tray_loc()[0]
    kl = scene._tray_local(scene.ketchup.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | tray_loc=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
          f"{float(tl[2]):+.3f}) ketchup_tray=({float(kl[0]):+.3f},{float(kl[1]):+.3f},"
          f"{float(kl[2]):+.3f}) in={bool(scene.ketchup_in_tray()[0])} "
          f"at_load={bool(scene.tray_at_load()[0])} "
          f"at_serve={bool(scene.tray_at_serve()[0])} "
          f"loaded={bool(scene._loaded[0])} in_latch={bool(scene._in_tray[0])} "
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
    env = ENVS.get("simgen.shuttle_hatch")().build(num_envs=args.num_envs,
                                                  device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.78, 0.62)) + o),
                                tuple(np.array((0.30, 0.00, 0.14)) + o),
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

    def station_world(loc_xyz) -> torch.Tensor:
        """Station-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w + quat_apply(
            scene.station.data.root_quat_w, loc)

    def station_quat() -> torch.Tensor:
        _refresh()
        return scene.station.data.root_quat_w

    def tray_world(loc_xyz) -> torch.Tensor:
        """Tray-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.tray.data.root_pos_w + quat_apply(scene.tray.data.root_quat_w, loc)

    def x_dir_w() -> torch.Tensor:
        """World unit vector along station-local +x (out of the serve window)."""
        _refresh()
        return quat_apply(scene.station.data.root_quat_w,
                          torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]

    def tray_x() -> float:
        _refresh()
        return float(scene.tray_loc()[0, 0])

    def bottle_in_tray_region(body) -> bool:
        return bool(scene.in_tray(body.data.root_pos_w)[0])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; tray seated at serve, ketchup "
          "outside the tray; score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.tray_at_serve()[0])
          and not bool(scene.ketchup_in_tray()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.station.data.root_quat_w[0]),
                scene.station.data.root_pos_w[0, :2].clone(),
                tray_x(),
                scene.ketchup.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.ketchup.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_sp, a_tx, a_kp, a_ky = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_sp, b_tx, b_kp, b_ky = readback()
    d_yaw, d_sp = dyaw(a_yaw, b_yaw), float((a_sp - b_sp).norm())
    d_tx = abs(a_tx - b_tx)
    d_kp, d_ky = float((a_kp - b_kp).norm()), dyaw(a_ky, b_ky)
    print(f"[smoke] randomization deltas: station_yaw={d_yaw:.1f}deg "
          f"station_xy={d_sp * 1000:.1f}mm tray_x={d_tx * 1000:.1f}mm "
          f"ketchup_xy={d_kp * 1000:.1f}mm ketchup_yaw={d_ky:.1f}deg", flush=True)
    check("randomization-is-real: station yaw, station xy and the ketchup bottle's "
          "pose readback differ across seeds",
          d_yaw > 2.0 and d_sp > 0.003 and d_kp > 0.005 and d_ky > 2.0)

    # ================= 3. bottle slot shuffle =====================================================
    slots_seen = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slots_seen.add(int(scene.ketchup_slot[0]))
    print(f"[smoke] over 10 resets: ketchup slots {sorted(slots_seen)}", flush=True)
    check("slot shuffle: the ketchup bottle occupies >= 2 different ground slots "
          "over 10 resets", len(slots_seen) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(20)
    x0 = tray_x()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, tray stays put (< 8 mm), score ~0, "
          "no success",
          abs(tray_x() - x0) < 0.008 and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "carry the bottle over the tray and lower it in from
    # above". Here the tray sits under the station ROOF: the plan's end state is the
    # bottle resting on the closed roof. Must not be success and must score ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ketchup,
                station_world((0.04, 0.0, 0.216 + c.bottle_size[2] / 2 + 0.008)),
                station_quat())
    _step(180)
    _report("seed-strategy")
    kz = float(scene._station_local(scene.ketchup.data.root_pos_w)[0, 2])
    print(f"[smoke] seed strategy: bottle ends at station z={kz * 1000:.0f}mm "
          f"(roof top 216mm)", flush=True)
    check("negative (SEED strategy): bottle lowered-from-above over the tray ends "
          "ON THE CLOSED ROOF — not in the tray, no success, score ~0",
          not bool(scene.ketchup_in_tray()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 6. the serve window is sealed ==============================================
    # Fling the bottle AT the window (0.7 m/s inward, mid-window height, offset from
    # the handle bar): it must demonstrably fly inward (min station-x asserted — the
    # probe moved, not vacuous) and bounce off the tray's front wall, never entering
    # the tray. The 12 mm roof-over-wall gap is far below the bottle's 36 mm minimum
    # dimension.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ketchup, station_world((0.26, 0.06, 0.125)), station_quat(),
                lin_vel_w=-0.7 * x_dir_w())
    min_x = 1.0
    for _ in range(120):
        _step(1)
        min_x = min(min_x,
                    float(scene._station_local(scene.ketchup.data.root_pos_w)[0, 0]))
    _step(120)
    _report("window-blocked")
    print(f"[smoke] window fling: min station-x reached {min_x * 1000:.0f}mm "
          f"(tray front wall face at ~133mm)", flush=True)
    check("window blocked: the flung bottle travels inward (min station-x < 210mm) "
          "yet never enters the tray — no success, no credit",
          min_x < 0.210 and not bool(scene.ketchup_in_tray()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 7. ORDER violation: drop before pushing the tray in =======================
    # Chimney drop with the tray still at serve: the hole has NO overlap with the
    # tray's interior (hole max x -0.068 < tray interior min x -0.033), so the
    # bottle lands on the bare plate BEHIND the tray. A 5 N / 2.5 s push then drives
    # the tray inward >= 15 mm (the push is real) but the stranded bottle (>= 36 mm
    # thick) blocks it from ever reaching the load gate at -0.092. Nothing scores.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ketchup,
                station_world((c.x_load, 0.0, c.chimney_top + c.bottle_size[2] / 2 + 0.008)),
                station_quat())
    _step(180)
    _report("order-drop")
    dropped_ok = not bool(scene.ketchup_in_tray()[0])
    x0 = tray_x()
    x_min = x0
    zero = torch.zeros(n, 1, 3, device=device)
    f_in = (-5.0 * x_dir_w()).view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(300):
        scene.tray.set_external_force_and_torque(f_in, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
        x_min = min(x_min, tray_x())
    scene.tray.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("order-jam")
    print(f"[smoke] order violation: tray pushed from x={x0 * 1000:.0f}mm to min "
          f"{x_min * 1000:.0f}mm (load gate {c.load_x_gate * 1000:.0f}mm)", flush=True)
    check("ORDER violation: bottle dropped before the push lands BEHIND the tray; "
          "the 5 N push moves the tray >= 15 mm but the stranded bottle blocks the "
          "load gate — not in tray, no credit, no success",
          dropped_ok and (x0 - x_min) >= 0.015 and x_min > c.load_x_gate
          and not bool(scene.ketchup_in_tray()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. near-miss: tray short of the serve gate =================================
    # Bottle in the tray but the tray parked 22 mm short of the serve gate: in_tray
    # credit only, no success. (Smoke may write into rubric volumes — this is a
    # rubric probe, not a solution.)
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.tray, station_world((0.008, 0.0, c.plate_top + 0.002)),
                station_quat())
    _write_body(scene.ketchup, tray_world((0.0, 0.0, 0.095)), station_quat())
    _step(150)
    _report("serve-near-miss")
    check("near-miss (short of serve): bottle in the tray, tray 22 mm short of the "
          "serve gate — in_tray credit only (0.25), no success",
          bool(scene.ketchup_in_tray()[0]) and not bool(scene.tray_at_serve()[0])
          and abs(float(scene.score()[0]) - c.w_in) < 0.01
          and not bool(scene.success()[0]))

    # ================= 9. near-miss: loaded and abandoned at the back stop ========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.tray, station_world((-0.110, 0.0, c.plate_top + 0.002)),
                station_quat())
    _step(60)  # tray settles at the load stop -> loaded latch
    _write_body(scene.ketchup, tray_world((0.0, 0.0, 0.095)), station_quat())
    _step(150)
    _report("abandoned-load")
    s9 = float(scene.score()[0])
    check("near-miss (abandoned at load): bottle in the tray at the back stop — "
          "latched credit capped at 0.40, no success",
          bool(scene.ketchup_in_tray()[0]) and bool(scene._loaded[0])
          and 0.39 <= s9 <= 0.401  # float32 clamp boundary: allow +eps
          and not bool(scene.success()[0]))

    # ================= 10. negative: wrong object =================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.mustard, tray_world((0.0, 0.0, 0.095)), station_quat())
    _step(150)
    _report("wrong-object")
    check("negative (wrong object): MUSTARD in the tray at serve, ketchup on the "
          "ground — no success, no credit",
          bottle_in_tray_region(scene.mustard)
          and not bool(scene.ketchup_in_tray()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 11. the shuttle is captive (and the pull-out is feasible) ==================
    # From the load stop an 8 N yank (>> any needed pull) drags the tray forward: it
    # must REACH the serve zone (the pull-out the solve and a Franka perform is
    # real) and must STOP at the bar — never past it, never climbing it (the roof
    # passes 12 mm over the walls: the tray is boxed in).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.tray, station_world((-0.110, 0.0, c.plate_top + 0.002)),
                station_quat())
    _step(60)
    x_max = tray_x()
    f_out = (8.0 * x_dir_w()).view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(180):
        scene.tray.set_external_force_and_torque(f_out, zero, env_ids=_all_ids(),
                                                 is_global=True)
        _step(1)
        x_max = max(x_max, tray_x())
    scene.tray.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    _report("captive-pull")
    print(f"[smoke] captive pull: max tray x {x_max * 1000:.0f}mm "
          f"(physical stop at {c.x_serve * 1000:.0f}mm), final x {tray_x() * 1000:.0f}mm",
          flush=True)
    check("captive shuttle: an 8 N yank from the load stop reaches the serve zone "
          "(>= 30mm) but the stop bar holds — max x <= 85mm, tray seated at serve, "
          "no success without a bottle",
          x_max >= 0.030 and x_max <= 0.085 and bool(scene.tray_at_serve()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shuttle_hatch")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
