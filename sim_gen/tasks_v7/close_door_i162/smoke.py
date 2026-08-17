"""Smoke / rubric-REJECTION battery for BarredGateScene — NullRobot, teleported probes.

solve.py is the acceptance proof (torque-close the door, force-drop the lock rod
through eyelet + staple). This battery proves the rubric REJECTS wrong outcomes and
that the LOCK — the task's strategic differentiator from the seed — is physically
load-bearing in both directions: an unbarred door reopens under a real torque, a
barred door arrests one. Every probe is CONSTRUCTED as a settled state (teleport,
real physics steps, judge); constructed partial states may earn latched partial
credit but none may reach success() unless the full lock is genuinely built.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; door holds its random angle
                           on the damped hinge, rods upright in the caddy; score ~0,
                           no success;
   2. randomization      — two seeded resets: READBACK frame yaw, frame xy, door
                           initial angle and caddy position all differ;
   3. well swap          — across seeds BOTH well assignments occur, and the stored
                           flag matches the caddy-local rod position readback;
   4. null-policy        — 240 idle steps: nothing moves, score ~0, no success;
   5. SEED STRATEGY      — the seed's whole plan ("push the door shut", nothing
                           else): door CONSTRUCTED closed (hinge-consistent write),
                           rods untouched -> closed but UNBARRED: no success, and a
                           1.5 N.m opening torque swings it back past 20 deg — the
                           physics that voids close-only;
   6. wrong order A      — rod hung in the door's eyelet BEFORE closing: the solve's
                           own closing servo ARRESTS (hanging shaft strikes the
                           staple pedestal), door never reaches the 3 deg tolerance;
   7. wrong order B      — rod standing in the staple BEFORE closing: the closing
                           eyelet block strikes the exposed shaft — arrest again;
   8. decoy near-miss    — door closed, RED short stub fully dropped into the
                           eyelet: its tip cannot reach the staple; no success, and
                           the door still reopens under the same 1.5 N.m torque;
   9. near-miss ajar     — long rod standing seated in the STAPLE but the door held
                           ajar at ~8 deg: `seated` reads true yet the shaft does
                           not thread the eyelet — no success, score <= 0.70;
  10. lock-reality       — full lock constructed (door on stop, rod seated through
                           both channels): a 2 N.m opening torque CANNOT open the
                           door past the closed tolerance — the bar, not a score
                           artifact, holds the gate;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_door_i162.smoke --headless
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

_qmul, _qz = task_scene._qmul, task_scene._qz

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
    ang = float(scene.open_angle_deg()[0])
    tip = scene._frame_local(scene.rod.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | angle={ang:+6.2f}deg "
          f"tip=({float(tip[0]):+.3f},{float(tip[1]):+.3f},{float(tip[2]):+.3f}) "
          f"seated={bool(scene.rod_seated()[0])} thread={bool(scene.rod_threading_eyelet()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.barred_gate")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.85, 0.80)) + o),
                                tuple(np.array((0.42, 0.05, 0.28)) + o),
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

    def ang() -> float:
        _refresh()
        return float(scene.open_angle_deg()[0])

    zero = torch.zeros(n, 1, 3, device=device)

    def door_torque(tau: float) -> None:
        """Constant torque about the vertical hinge axis (+ closes, - opens)."""
        _refresh()
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        t_b = quat_apply_inverse(scene.door.data.root_quat_w, t_w)
        scene.door.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))

    def servo_close(steps: int = 600) -> float:
        """The solve's own closing servo (same gains/clamp): PD torque toward the
        closed stop. Returns the final angle; wrench zeroed afterwards."""
        no_action = torch.empty(0, device=device)
        for _ in range(steps):
            a_rad = math.radians(ang() + 2.0)
            wz = float(scene.door.data.root_ang_vel_w[0, 2])
            tau = max(-4.0, min(4.0, 8.0 * a_rad - 2.0 * wz))
            door_torque(tau)
            rec = _REC["on"] and _REC["annot"] is not None
            env.step(no_action, render=rec)
            if rec and _REC["i"] % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(_REC["annot"].get_data())
                if arr.size:
                    _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
            _REC["i"] += 1
        scene.door.set_external_force_and_torque(zero, zero)
        _step(60)
        return ang()

    def hinge_world() -> torch.Tensor:
        _refresh()
        h = torch.tensor([0.0, c.hinge_y, 0.0], device=device).expand(n, 3)
        return scene.frame.data.root_pos_w + quat_apply(scene.frame.data.root_quat_w, h)

    def set_door_angle(deg: float) -> None:
        """Teleport the door to opening angle `deg` — the door origin sits ON the
        hinge axis, so this is a hinge-consistent pure pose write (zero velocity)."""
        q = _qmul(scene.frame.data.root_quat_w,
                  _qz(torch.full((n,), -math.radians(deg), device=device)))
        _write_body(scene.door, hinge_world(), q)

    def put_rod_door_local(body, loc_xyz) -> None:
        """Teleport a rod upright at a door-local point (tip at loc_xyz)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.door.data.root_pos_w + quat_apply(scene.door.data.root_quat_w, loc)
        _write_body(body, pos, scene.door.data.root_quat_w)

    def put_rod_frame_local(body, loc_xyz) -> None:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.frame.data.root_pos_w + quat_apply(scene.frame.data.root_quat_w, loc)
        _write_body(body, pos, scene.frame.data.root_quat_w)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    a0 = float(scene.a0[0])
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    rod_z = float(scene.rod.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    up_z = float(quat_apply(scene.rod.data.root_quat_w,
                            torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0, 2])
    check("settle/no-NaN: layout settles finite; door holds its random angle on the "
          "damped hinge, lock rod upright in the caddy; score ~0, no success",
          bool(scene._finite()[0]) and abs(ang() - a0) < 5.0
          and 0.02 < rod_z < 0.10 and up_z > 0.95
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        fyaw = yaw_of(scene.frame.data.root_quat_w[0])
        fp = scene.frame.data.root_pos_w[0, :2].clone()
        cp = scene.caddy.data.root_pos_w[0, :2].clone()
        return fyaw, fp, float(scene.a0[0]), cp

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_fp, a_a0, a_cp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_fp, b_a0, b_cp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_fp = float((a_fp - b_fp).norm())
    d_a0 = abs(a_a0 - b_a0)
    d_cp = float((a_cp - b_cp).norm())
    print(f"[smoke] randomization deltas: frame_yaw={d_yawv:.1f}deg "
          f"frame_xy={d_fp * 1000:.1f}mm door_a0={d_a0:.1f}deg "
          f"caddy_xy={d_cp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: frame yaw, frame xy, door initial angle and caddy "
          "position readback all differ across seeds",
          d_yawv > 2.0 and d_fp > 0.003 and d_a0 > 2.0 and d_cp > 0.02)

    # ================= 3. randomization: the well swap occurs and reads back ======================
    seen = {True: 0, False: 0}
    consistent = True
    for sd in range(300, 310):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        flag = bool(scene.rod_in_well_p[0])
        loc = quat_apply_inverse(scene.caddy.data.root_quat_w,
                                 scene.rod.data.root_pos_w - scene.caddy.data.root_pos_w)[0]
        consistent &= (float(loc[0]) > 0) == flag and abs(abs(float(loc[0])) - c.caddy_well_dx) < 0.02
        seen[flag] += 1
        if seen[True] and seen[False] and sd >= 303:
            break
    print(f"[smoke] well swap counts over seeds: +x={seen[True]} -x={seen[False]} "
          f"consistent={consistent}", flush=True)
    check("randomization (well swap): both rod/decoy well assignments occur across "
          "seeds and the flag matches the caddy-local readback",
          seen[True] > 0 and seen[False] > 0 and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the door keeps its angle, rods stay "
          "in the caddy, score ~0, no success",
          abs(ang() - a0) < 5.0 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Close the door" — push the hinged panel shut, release, done. CONSTRUCT its
    # end state: door on its stop (hinge-consistent write), rods untouched in the
    # caddy. Closed-but-unbarred must NOT succeed, and a modest real opening torque
    # must swing the gate back open — the physics that demands the lock rod.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_door_angle(0.0)
    _step(180)
    _report("seed-closed")
    closed_unbarred_ok = ang() <= c.closed_max_deg and not bool(scene.success()[0]) \
        and float(scene.score()[0]) <= 0.32
    door_torque(-1.5)
    _step(360)
    door_torque(0.0)
    _step(60)
    _report("seed-reopened")
    _REC["on"] = False
    check("negative (SEED strategy): door pushed fully shut with NO bar is not "
          "success, and a 1.5 N.m pull swings it back open past 20 deg",
          closed_unbarred_ok and ang() > 20.0 and not bool(scene.success()[0]))

    # ================= 6. negative: WRONG ORDER A (rod in the eyelet first) =======================
    # Hang the lock rod in the OPEN door's eyelet (head in the funnel V, shaft
    # protruding below), then run the solve's own closing servo: the hanging shaft
    # strikes the staple pedestal and the door must ARREST outside the tolerance.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_rod_door_local(scene.rod, (c.ch_x, c.ey_y, c.eyelet_z1 - c.rod_len + 0.002))
    _step(150)
    tip_d = quat_apply_inverse(scene.door.data.root_quat_w,
                               scene.rod.data.root_pos_w - scene.door.data.root_pos_w)[0]
    hang_ok = abs(float(tip_d[0]) - c.ch_x) < 0.02 and abs(float(tip_d[1]) - c.ey_y) < 0.02 \
        and 0.25 < float(tip_d[2]) < 0.32
    print(f"[smoke] hanging rod tip (door-local): ({float(tip_d[0]):+.3f},"
          f"{float(tip_d[1]):+.3f},{float(tip_d[2]):+.3f}) hang_ok={hang_ok}", flush=True)
    a_end = servo_close(600)
    _report("orderA-arrest")
    _REC["on"] = False
    check("negative (wrong order A): rod hung in the eyelet BEFORE closing — the "
          "closing servo arrests on the staple pedestal, door never reaches the "
          "closed tolerance, no success",
          hang_ok and a_end > c.closed_max_deg + 0.4 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 7. negative: WRONG ORDER B (rod in the staple first) =======================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_rod_frame_local(scene.rod, (c.ch_x, c.ch_y, c.staple_z0 + 0.003))
    _step(150)
    tip_f = scene._frame_local(scene.rod.data.root_pos_w)[0]
    stand_ok = abs(float(tip_f[0]) - c.ch_x) < 0.02 and abs(float(tip_f[1]) - c.ch_y) < 0.02 \
        and float(tip_f[2]) < c.staple_z0 + 0.02
    a_end = servo_close(600)
    _report("orderB-arrest")
    check("negative (wrong order B): rod standing in the staple BEFORE closing — "
          "the closing eyelet block strikes the exposed shaft, the door arrests, "
          "no success",
          stand_ok and a_end > c.closed_max_deg + 0.4 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 8. near-miss: the DECOY cannot lock ========================================
    # Door properly closed, RED stub dropped fully into the eyelet: its tip hangs
    # 35+ mm short of the staple. No success — and the same 1.5 N.m pull still
    # reopens the door (the stub rides along, arresting nothing).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_door_angle(0.0)
    _step(120)
    put_rod_door_local(scene.decoy, (c.ch_x, c.ey_y, c.eyelet_z1 - c.decoy_len + 0.002))
    _step(180)
    _report("decoy-dropped")
    decoy_no_success = not bool(scene.success()[0]) and ang() <= c.closed_max_deg
    door_torque(-1.5)
    _step(360)
    door_torque(0.0)
    _step(60)
    _report("decoy-reopened")
    _REC["on"] = False
    check("near-miss (decoy): short red stub fully dropped into the eyelet locks "
          "nothing — no success, and the door still reopens past 20 deg",
          decoy_no_success and ang() > 20.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 9. near-miss: staple filled but the door AJAR ==============================
    # `seated` alone must not be success: rod standing seated in the staple with the
    # door held ajar at 8 deg does not thread the eyelet — the lock is not built.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_door_angle(8.0)
    _step(60)
    put_rod_frame_local(scene.rod, (c.ch_x, c.ch_y, c.staple_z0 + 0.003))
    _step(150)
    _report("ajar-staple")
    check("near-miss (ajar): rod seated in the staple but the door ajar at ~8 deg "
          "— shaft does not thread the eyelet, no success, score <= 0.70",
          bool(scene.rod_seated()[0]) and not bool(scene.rod_threading_eyelet()[0])
          and ang() > c.closed_max_deg + 1.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 10. lock-reality: the bar physically holds the gate ========================
    # Construct the full lock, then try to force the gate open with MORE torque
    # than the reopen probes used: the shaft jams between eyelet and staple walls
    # and the door must stay within the closed tolerance. This is what "locked"
    # means — the rubric's success state is mechanically load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_door_angle(0.0)
    _step(120)
    put_rod_frame_local(scene.rod, (c.ch_x, c.ch_y, c.staple_z0 + 0.003))
    _step(180)
    _report("lock-built")
    lock_built = bool(scene.rod_seated()[0]) and bool(scene.rod_threading_eyelet()[0]) \
        and ang() <= c.closed_max_deg
    max_a = 0.0
    no_action = torch.empty(0, device=device)
    door_torque(-2.0)
    for i in range(300):
        env.step(no_action, render=False)
        if i % 10 == 0:
            max_a = max(max_a, ang())
    max_a = max(max_a, ang())
    door_torque(0.0)
    _step(90)
    _report("lock-held")
    _REC["on"] = False
    print(f"[smoke] barred door under 2 N.m opening pull: max angle {max_a:.2f} deg "
          f"(tolerance {c.closed_max_deg:.1f})", flush=True)
    check("lock-reality: with the rod seated through BOTH channels a 2 N.m opening "
          "pull cannot open the door past the closed tolerance — the bar is "
          "load-bearing",
          lock_built and max_a <= c.closed_max_deg + 0.3
          and bool(scene.rod_seated()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.barred_gate")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
