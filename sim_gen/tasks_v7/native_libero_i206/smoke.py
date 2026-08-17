"""Smoke / rubric-REJECTION battery for TiltMazeScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real tilt-steered trajectory and
the latched credit is monotone along it). This battery proves the rubric REJECTS wrong
outcomes, and that the physical claims the task rests on — the roof seals the ball in,
the press tilts the tray, the pocket lip retains at full tilt, the keel returns the
tray to level — are load-bearing. Every probe is CONSTRUCTED (teleport = transport,
real physics steps, judge) — instrumentation, never a solution: success() is monitored
at EVERY step and must never turn True anywhere in the battery (the audit is itself a
check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; tray mass + CoM READBACK match
                           the authored keel values; ball in the NORTH lane; score 0,
                           no success;
   2. randomization      — two seeded resets: READBACK stand xy, assembly yaw and the
                           ball's tray-frame y all differ (ball x always rolls to the
                           west wall under the heel, so x is not asserted);
   3. null-policy        — 300 idle steps: ball stays in the north lane, score ~0,
                           no success;
   4. SEED STRATEGY      — the seed family's plan is grasp-and-place; the ball is
                           sealed under the cage roof, so the best transport can do is
                           set the ball ON TOP of the roof directly above the pocket:
                           it rests far above the sink plane, then rolls off the tray
                           entirely — never in the pocket, score ~0 (with the cavity
                           latch gate, the off-roof fall latches nothing);
   5. tilted pocket      — press probe: a body-frame hinge torque drives the tray to
                           its stop (the ACTUATOR the task offers, direction probed);
                           the yellow ball placed sunk in the pocket while pressed is
                           RETAINED by the lip at full tilt, but `level` refuses:
                           no success while the tray is held tilted;
   6. keel return        — ball back at the start, torque released: the tray returns
                           from the stop to within level_tol on its own and calms —
                           the self-levelling the rubric relies on is real;
   7. wrong object       — the WHITE spare ball sunk in the pocket instead (yellow
                           still at the start): no success, score ~0 — in_pocket
                           judges the YELLOW ball and spare_home is violated;
   8. restraint          — yellow ball correctly SUNK, tray level and settled, but the
                           spare ball displaced from its cradle: every other clause
                           holds and success still refuses — the distractor clause is
                           load-bearing (score <= 0.75);
   9. latched credit     — ball teleported into the bay latches bay(+crossed) credit,
                           score ~0.50, no success;
  10. regression         — ball put back at the start: the latched credit persists
                           unchanged, still no success (credit is for reaching, not
                           for camping);
  11. settle gate        — ball sunk in the pocket but still moving (~0.8 m/s): the
                           settle clause refuses while anything moves (spare parked
                           away first so the probe can never ripen into success);
  12. rejection audit    — success() was never True at any step of this battery;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.native_libero_i206.smoke --headless
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

_qapply, _qinv = task_scene._qapply, task_scene._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    loc = scene.ball_local()[0]
    print(f"[smoke] {tag:16s} | tilt={math.degrees(float(scene.tilt()[0])):+6.2f}deg "
          f"ball_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
          f"pocket={bool(scene.in_pocket()[0])} level={bool(scene.level()[0])} "
          f"settled={bool(scene.settled()[0])} spare={bool(scene.spare_home()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_maze")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    r = c.ball_r
    px, py = c.pocket_center
    start_loc = (-0.13, c.lane_n_y)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.02, -0.80, 0.60)) + o),
                                tuple(np.array((0.45, -0.08, 0.10)) + o),
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

    def tray_pt(loc_xyz) -> torch.Tensor:
        """Tray-body-frame point -> world (per-env), using the LIVE tray pose."""
        _refresh()
        loc = torch.tensor([float(v) for v in loc_xyz], device=device).expand(n, 3)
        return scene.tray.data.root_pos_w + _qapply(scene.tray.data.root_quat_w, loc)

    def spare_local_z() -> float:
        _refresh()
        return float(scene._tray_local(scene.spare.data.root_pos_w)[0, 2])

    def tilt_deg() -> float:
        _refresh()
        return math.degrees(float(scene.tilt()[0]))

    zero3 = torch.zeros(n, 1, 3, device=device)

    def press(tau: float, steps: int) -> None:
        """Body-frame hinge torque about the tray's own y (hinge) axis — the paddle
        press, expressed as the wrench a fingertip would transmit. Drag-immune."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 1] = tau
        scene.tray.set_external_force_and_torque(zero3, t, env_ids=_all_ids())
        _step(steps)

    def release() -> None:
        scene.tray.set_external_force_and_torque(zero3, zero3, env_ids=_all_ids())

    def park_spare_away() -> None:
        """Transport the spare out of its cradle onto the open floor (stand-frame
        offset from the cradle, clear of the base slab and the tray sweep)."""
        _refresh()
        off = _qapply(scene.stand.data.root_quat_w,
                      torch.tensor([0.14, 0.0, 0.0], device=device).expand(n, 3))
        p = scene.cradle.data.root_pos_w + off
        p = p.clone()
        p[:, 2] = scene.env_origins[:, 2] + r + 0.002
        _write_body(scene.spare, p)
        _step(20)

    # ================= 1. settle / no-NaN / authored-mass readback ================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    mass = float(torch.as_tensor(scene.tray.root_physx_view.get_masses()).reshape(-1)[0])
    com = torch.as_tensor(scene.tray.root_physx_view.get_coms()).reshape(-1)
    com_z = float(com[2])
    loc = scene.ball_local()[0]
    print(f"[smoke] tray mass readback={mass:.3f}kg (authored {c.tray_mass}) "
          f"com_z={com_z:+.4f}m (authored {c.tray_com_z})", flush=True)
    check("settle/no-NaN: layout settles finite; tray mass+CoM readback match the "
          "authored keel; ball in the NORTH lane; score 0, no success",
          bool(scene._finite()[0]) and abs(mass - c.tray_mass) < 0.06
          and abs(com_z - c.tray_com_z) < 0.02
          and float(loc[1]) > c.div_hy + 0.005 and abs(float(loc[2]) - r) < 0.010
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.stand.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.stand.data.root_quat_w[0]),
                float(scene.ball_local()[0, 1]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_sp, a_sy, a_by = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_sp, b_sy, b_by = readback()
    d_sp = float((a_sp - b_sp).norm())
    d_sy = dyaw(a_sy, b_sy)
    d_by = abs(a_by - b_by)
    print(f"[smoke] randomization deltas: stand_xy={d_sp * 1000:.1f}mm "
          f"yaw={d_sy:.1f}deg ball_y={d_by * 1000:.1f}mm "
          f"(ball_y readbacks {a_by * 1000:+.1f} / {b_by * 1000:+.1f}mm)", flush=True)
    check("randomization-is-real: stand xy, assembly yaw and ball tray-frame y "
          "readback all differ across seeds",
          d_sp > 0.003 and d_sy > 3.0 and d_by > 0.002)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(300)
    _report("null-policy")
    loc = scene.ball_local()[0]
    check("null-policy-fails: 300 idle steps, ball still in the north lane, "
          "score ~0, no success",
          float(loc[1]) > c.div_hy + 0.005 and not bool(scene.in_pocket()[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED FAMILY'S OWN STRATEGY ================================
    # The seed grasps the object and places it at the goal region. The ball is sealed
    # under the cage roof, so the closest transport can come is setting the ball ON
    # TOP of the roof directly above the pocket: it rests ~50 mm above the sink plane
    # and then simply rolls off the tray. Never in the pocket, never any credit.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ball, tray_pt((px, -0.045, c.roof_hi + r + 0.004)))
    _step(30)
    loc = scene.ball_local()[0]
    z_roof = float(loc[2])
    print(f"[smoke] roof-parked ball tray-frame z={z_roof * 1000:.1f}mm "
          f"(sink plane {c.sink_z * 1000:.0f}mm)", flush=True)
    roof_high = z_roof > 0.045 and not bool(scene.in_pocket()[0])
    _step(300)
    _report("seed-strategy")
    check("negative (SEED strategy): ball set on TOP of the cage roof above the "
          "pocket — rests far above the sink plane, then rolls off the tray: "
          "never in the pocket, no success, score ~0",
          roof_high and not bool(scene.in_pocket()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 5. tilted pocket: press to the stop, lip retains, level refuses ============
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    tau = -0.25                      # red (pocket) end down; direction probed below
    press(tau, 60)
    if tilt_deg() > 2.0:             # pod applied the body torque with the other sense
        print(f"[smoke] press probe: tilt={tilt_deg():+.2f}deg — flipping torque sign",
              flush=True)
        tau = -tau
    press(tau, 150)
    for _ in range(3):               # escalate if friction holds it short of the stop
        if tilt_deg() <= -8.0:
            break
        tau *= 1.6
        print(f"[smoke] press probe: tilt={tilt_deg():+.2f}deg — torque -> {tau:+.2f} N·m",
              flush=True)
        press(tau, 120)
    pressed = tilt_deg()
    _write_body(scene.ball, tray_pt((px, py, 0.010)))    # sunk, while held at the stop
    _step(90)
    _report("tilted-pocket")
    check("tilted pocket: hinge torque drives the tray to its stop (actuator real, "
          "tilt <= -8 deg), the lip RETAINS the sunk ball at full tilt, and `level` "
          "refuses success",
          pressed <= -8.0 and tilt_deg() <= -8.0 and bool(scene.in_pocket()[0])
          and not bool(scene.level()[0]) and not bool(scene.success()[0]))

    # ================= 6. keel return =============================================================
    _write_body(scene.ball, tray_pt((start_loc[0], start_loc[1], r + 0.006)))
    _step(10)
    release()
    _step(360)
    _report("keel-return")
    check("keel return: torque released from the stop, the tray swings back within "
          "level_tol on its own and calms — the self-levelling is real",
          pressed <= -8.0 and abs(tilt_deg()) < c.level_tol_deg
          and float(scene.tray.data.root_ang_vel_w[0].norm()) < c.tray_calm
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. negative: wrong object ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.spare, tray_pt((px, py, 0.010)))
    _step(120)
    _report("wrong-object")
    print(f"[smoke] spare tray-frame z={spare_local_z() * 1000:.1f}mm "
          f"(sunk < {c.sink_z * 1000:.0f}mm)", flush=True)
    check("negative (wrong object): the WHITE spare ball sunk in the pocket instead "
          "— spare_home violated, yellow not in the pocket: no success, score ~0",
          spare_local_z() < c.sink_z and not bool(scene.spare_home()[0])
          and not bool(scene.in_pocket()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 8. restraint: spare out of its cradle ======================================
    # Spare parked away FIRST (audit safety), then the yellow ball genuinely sunk with
    # the tray level and settled: every other success clause holds, only the
    # distractor clause refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    park_spare_away()
    _write_body(scene.ball, tray_pt((px, py, 0.010)))
    _step(150)
    _report("restraint")
    check("restraint: yellow ball SUNK, tray level, all settled — but the spare is "
          "out of its cradle: success refuses on the distractor clause alone, "
          "score <= 0.75",
          bool(scene.in_pocket()[0]) and bool(scene.level()[0])
          and bool(scene.settled()[0]) and not bool(scene.spare_home()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 9+10. latched credit + regression ==========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ball, tray_pt((0.15, -0.030, r + 0.004)))   # bay, south of the baffle
    _step(120)
    _report("bay-latch")
    s_latched = float(scene.score()[0])
    check("latched credit: ball transported into the BAY latches bay(+crossed) "
          "credit (score ~0.50), no success",
          s_latched >= c.w_bay - 1e-6 and s_latched <= 0.55
          and not bool(scene.success()[0]))
    _write_body(scene.ball, tray_pt((start_loc[0], start_loc[1], r + 0.004)))
    _step(90)
    _report("regression")
    check("regression: ball back at the start — the latched credit persists "
          "unchanged, still no success",
          abs(float(scene.score()[0]) - s_latched) < 1e-4
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. settle gate ============================================================
    # Full success geometry (ball sunk, tray level) but the ball still MOVING. The
    # spare is parked away first so the probe can never ripen into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    park_spare_away()
    vel_w = _qapply(scene.tray.data.root_quat_w,
                    torch.tensor([0.8, 0.0, 0.0], device=device).expand(n, 3))
    _write_body(scene.ball, tray_pt((px, py, 0.004)), lin_vel=vel_w)
    gate_ok = True
    for _ in range(2):
        _step(1)
        v = float(scene.ball.data.root_lin_vel_w[0].norm())
        gate_ok = gate_ok and v > c.settle_speed and not bool(scene.settled()[0]) \
            and not bool(scene.success()[0])
    _step(90)
    _report("settle-gate")
    check("settle gate: ball sunk in the pocket but still moving (~0.8 m/s) — "
          "success refuses while anything moves",
          gate_ok and not bool(scene.success()[0]))

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_maze")
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
