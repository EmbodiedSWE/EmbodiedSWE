"""Smoke / rubric-REJECTION battery for WedgeHopperScene — NullRobot, teleported probes.

solve.py is the acceptance proof (wedge staged by transport, driven home by contact
force, self-lock holds, the ball drains through the mouth into the basin, score 1.0).
This battery proves the rubric REJECTS wrong outcomes and that the physical claims the
task rests on are load-bearing: the seed's pick-and-carry plan is dead (a delivered
ball is refused for skipping the mouth), the thin shim really lifts nothing, and the
back-tilt containment really re-swallows a ball that only reaches the mouth. Every
probe is CONSTRUCTED as a state (teleport, real physics steps, judge) —
instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; ball inside the hopper, hopper
                          back-tilted ~-2 deg, wedge + shim flat on the ground outside
                          the machine; score 0, no success;
   2. randomization     — two seeded resets: READBACK rig yaw, rig xy, wedge ground
                          spawn and ball world position all differ (the push axis and
                          the tool must be found by looking);
   3. null-policy       — 240 idle steps: ball stays parked against the back wall
                          (the ~2 deg back-tilt is the containment), score ~0;
   4. SEED STRATEGY     — the seed's plan is grasp-carry-release into the container.
                          Its end state CONSTRUCTED: ball teleported into the basin,
                          settled — a REAL contained equilibrium, REJECTED: the drain
                          never crossed the mouth (score <= 0.21, no success). The
                          pathway latch is load-bearing;
   5. wedge staged only — wedge transported into the channel and left: exactly the
                          staging credit (~0.10), no lift, no success;
   6. SHIM decoy        — the 9 mm shim staged and DRIVEN fully under the hopper's
                          back edge with the same push the wedge gets: it vanishes
                          into the 10 mm slot and lifts NOTHING — tilt unchanged,
                          ball stays inside, no lift credit, no success;
   7. mouth-only        — ball placed in the mouth doorway of the resting hopper:
                          the back-tilt re-swallows it (it rolls back inside); the
                          pathway latch alone is worth <= 0.16 and is NOT success;
   8. near-miss outside — ball settled on the ground just outside the basin wall:
                          settled, near the target, REJECTED (containment window);
   9. rejection audit   — success() never fired at any judged step during the
                          negative probes (checks 4-8);
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18.smoke --headless
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

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"on": False, "fired": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUDIT["on"]:
            _AUDIT["fired"] |= bool(env.scene.success()[0])
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


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_hopper")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.75)) + o),
                                tuple(np.array((-0.10, 0.00, 0.10)) + o),
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

    def rig_xyz(body) -> tuple[float, float, float]:
        _refresh()
        p = scene._rig_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def tilt_deg() -> float:
        _refresh()
        return math.degrees(float(scene.drain_tilt()[0]))

    def rig_pose(x: float, y: float, z: float, rig_quat: bool = True) -> torch.Tensor:
        """(N,13) root state at rig-frame (x, y, z), zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x, y, z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        if rig_quat:
            st[:, 3:7] = scene.rig.data.root_quat_w
        else:
            st[:, 3] = 1.0
        return st

    def hopper_pose(x: float, y: float, z: float) -> torch.Tensor:
        """(N,13) root state at hopper-frame (x, y, z), identity quat, zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x, y, z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.hopper.data.root_pos_w \
            + quat_apply(scene.hopper.data.root_quat_w, loc)
        st[:, 3] = 1.0
        return st

    def on_ground_outside(body) -> bool:
        """Body flat on the ground (z < 25 mm) and outside the machine's channel."""
        x, y, z = rig_xyz(body)
        return z < 0.025 and not (-0.72 < x < 0.28 and abs(y) < 0.18)

    def report(tag: str) -> None:
        _refresh()
        wx, wy, wz = rig_xyz(scene.wedge)
        sx, sy, sz = rig_xyz(scene.shim)
        bx, by, bz = rig_xyz(scene.ball)
        print(f"[smoke] {tag:16s} | tilt={tilt_deg():+.2f}deg "
              f"wedge=({wx:+.3f},{wy:+.3f},{wz:+.3f}) "
              f"shim=({sx:+.3f},{sy:+.3f},{sz:+.3f}) "
              f"ball=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"in_hop={bool(scene.ball_in_hopper()[0])} "
              f"in_basin={bool(scene.ball_in_basin()[0])} "
              f"latch=[stg {int(scene._staged[0])} p {int(scene._lift_part[0])} "
              f"f {int(scene._lift_full[0])} m {int(scene._via_mouth[0])} "
              f"b {int(scene._ever_basin[0])}] "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
              flush=True)

    def push_along_channel(body, newtons: float, bouts: int, x_stop: float) -> None:
        """Bouted CoM force along the rig +x axis (the same push the wedge gets in
        solve.py), cleared between bouts, stopped once the body passes x_stop."""
        zero = torch.zeros(n, 1, 3, device=device)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        for _ in range(bouts):
            if rig_xyz(body)[0] > x_stop:
                break
            f = quat_apply(scene.rig.data.root_quat_w, ex).reshape(n, 1, 3) * newtons
            body.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                               is_global=True)
            _step(30)
            body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
            _step(15)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    report("settle")
    _REC["on"] = False
    bodies = [scene.hopper, scene.wedge, scene.shim, scene.ball]
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all()) for b in bodies)
    check("settle/no-NaN: seeded reset settles finite; ball inside the back-tilted "
          "hopper (~-2 deg), wedge and shim flat on the ground outside the machine; "
          "score 0, no success",
          finite and bool(scene.ball_in_hopper()[0])
          and -4.5 < tilt_deg() < -0.6
          and on_ground_outside(scene.wedge) and on_ground_outside(scene.shim)
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.rig.data.root_quat_w[0]),
                scene.rig.data.root_pos_w[0, :2].clone(),
                scene.wedge.data.root_pos_w[0, :2].clone(),
                scene.ball.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_rp, a_wp, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_rp, b_wp, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_rp = float((a_rp - b_rp).norm())
    d_wp = float((a_wp - b_wp).norm())
    d_bp = float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: rig_yaw={d_yawv:.1f}deg "
          f"rig_xy={d_rp * 1000:.1f}mm wedge_xy={d_wp * 1000:.1f}mm "
          f"ball_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rig yaw, rig xy, wedge ground spawn and ball "
          "world position readback all differ between seeded resets",
          d_yawv > 4.0 and d_rp > 0.003 and d_wp > 0.03 and d_bp > 0.008)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    check("null-policy-fails: 240 idle steps — the ~2 deg back-tilt keeps the ball "
          "parked against the back wall; score ~0, no success",
          bool(scene.ball_in_hopper()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. SEED STRATEGY: carry the ball to the container ==========================
    # The seed's plan (grasp the target, carry it, release it over the container),
    # CONSTRUCTED at its end state: the ball teleported to a hover over the basin and
    # dropped. It settles INSIDE the basin — genuinely contained, genuinely at rest —
    # and is REJECTED because the drain never crossed the mouth.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.ball.write_root_state_to_sim(rig_pose(0.115, 0.0, 0.055, rig_quat=False),
                                       _all_ids())
    _step(300)
    report("seed-strategy")
    _REC["on"] = False
    lv = float(scene.ball.data.root_lin_vel_w.norm(dim=-1)[0])
    check("negative (SEED strategy / bypass): ball delivered into the basin and "
          "settled — real containment WITHOUT the mouth pathway — success refused, "
          "score <= 0.21",
          bool(scene.ball_in_basin()[0]) and lv < c.settle_lin
          and not bool(scene._via_mouth[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.21)

    # ================= 5. wedge staged only =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    scene.wedge.write_root_state_to_sim(
        rig_pose(-0.31, 0.0, c.runway_top + 0.002), _all_ids())
    _step(120)
    report("staged-only")
    check("wedge-staged-only: wedge transported into the channel and left — exactly "
          "the staging credit (~0.10), tilt unchanged, no success",
          bool(scene._staged[0]) and tilt_deg() < -0.6
          and 0.09 <= float(scene.score()[0]) <= 0.11
          and not bool(scene.success()[0]))

    # ================= 6. SHIM decoy driven fully under ===========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.shim.write_root_state_to_sim(
        rig_pose(-0.31, 0.0, c.runway_top + 0.002), _all_ids())
    _step(90)
    push_along_channel(scene.shim, newtons=3.0, bouts=16, x_stop=-0.17)
    _step(180)
    sx, _sy, _sz = rig_xyz(scene.shim)
    report("shim-decoy")
    _REC["on"] = False
    check("negative (SHIM decoy): the 9 mm shim driven fully under the hopper's "
          "back edge (same push as the wedge) lifts NOTHING — tilt unchanged, ball "
          "still inside, no lift latch, score 0, no success",
          sx > -0.20 and tilt_deg() < -0.6 and bool(scene.ball_in_hopper()[0])
          and not bool(scene._lift_part[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 7. mouth-only: back-tilt re-swallows the ball ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.ball.write_root_state_to_sim(
        hopper_pose(-0.005, 0.0, c.ball_r + 0.003), _all_ids())
    _step(240)
    report("mouth-only")
    _REC["on"] = False
    check("near-miss (mouth-only): ball placed in the mouth doorway of the resting "
          "hopper rolls BACK inside (the back-tilt containment); the pathway latch "
          "alone is <= 0.16 and is not success",
          bool(scene._via_mouth[0]) and bool(scene.ball_in_hopper()[0])
          and float(scene.score()[0]) <= 0.16 and not bool(scene.success()[0]))

    # ================= 8. near-miss just outside the basin ========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    scene.ball.write_root_state_to_sim(rig_pose(0.30, 0.12, c.ball_r + 0.002,
                                                rig_quat=False), _all_ids())
    _step(240)
    bx, by, bz = rig_xyz(scene.ball)
    lv = float(scene.ball.data.root_lin_vel_w.norm(dim=-1)[0])
    report("outside-basin")
    check("near-miss (outside): ball settled on the ground just past the basin wall "
          "— at rest, centimetres from the target, rejected by the containment "
          "window; score 0, no success",
          lv < c.settle_lin and not bool(scene.ball_in_basin()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 9. rejection audit =========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-8)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wedge_hopper")
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
