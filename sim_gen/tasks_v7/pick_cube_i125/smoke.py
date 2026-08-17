"""Smoke / rubric-REJECTION battery for DieRollScene — NullRobot, constructed probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — sliding can NEVER
reorient the die, tipping has a real torque threshold — are load-bearing. Every probe
is CONSTRUCTED as a settled state (teleport, real physics steps, judge) or driven by
real wrenches — instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN       — seeded reset settles finite; die flat on a face, BLUE not
                            up, far from both mats; score ~0, no success;
   2. randomization       — two seeded resets: READBACK die xy, die orientation and
                            green-mat xy all differ;
   3. coverage            — over 10 resets: >= 3 distinct start faces appear and the
                            green mat occupies BOTH y sides (the swap is real);
   4. null-policy         — 240 idle steps: die stays put, score ~0, no success;
   5. SLIDE CANNOT ORIENT — a real CoM push in the slide window (3.6 N, velocity
                            capped) drags the die >= 0.10 m across the floor: the up
                            face NEVER changes and blue never rises ("reorientation
                            is reachable only through rolls" is physics, not fiat);
   6. torque tip gate     — 0.6x the edge-pivot breakaway torque held 300 steps does
                            NOT tip the die (up-dot stays ~1); 1.8x DOES tip it onto
                            a new face (the roll is a thresholded physical act, and
                            the sub-threshold half is non-vacuous);
   7. SEED-analog carry   — the die teleport-carried (orientation preserved — the
                            seed's whole skill, a pure position change) to the GREEN
                            mat center: blue still not up -> no success, score <= 0.16;
   8. wrong-face place    — die constructed centered on the green mat with PURPLE up
                            (blue DOWN): looks placed, orientation wrong -> no success;
   9. near-miss position  — die blue-up, settled 75 mm from the mat center (outside
                            zone_tol = 55 mm): all latches earned, score pinned at the
                            0.60 cap (float32 0.60000002), success refused;
  10. latch persistence   — from 9's state the die is carried far away and settled:
                            the latched 0.60 survives (monotone credit), still no
                            success;
  11. decoy mat           — die blue-up centered on the DARK-GRAY decoy: no approach
                            credit, no success, score <= 0.46;
  12. settle gate         — die written blue-up dead-center on the green mat but
                            MOVING (0.45 m/s / 3 rad/s): judged immediately, success
                            must refuse (then the die is removed before it can settle);
  13. frames.npz          — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_cube_i125.smoke --headless
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

_qmul, _qconj = task_scene._qmul, task_scene._qconj

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

FACE_NAMES = ("+x/RED", "-x/ORANGE", "+y/YELLOW", "-y/WHITE", "+z/BLUE", "-z/PURPLE")
TAU_STATIC = 0.45 * 9.81 * 0.045  # m g e/2: edge-pivot breakaway torque (audited)

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    idx, best = scene.face_up()
    ez = scene._die_axes()[2]
    print(f"[smoke] {tag:18s} | up={FACE_NAMES[int(idx[0])]}({float(best[0]):+.3f}) "
          f"blue_z={float(ez[0, 2]):+.3f} d_goal={float(scene.goal_dist()[0]):.3f} "
          f"rolled={bool(scene._rolled[0])} blue={bool(scene._blue[0])} "
          f"near={bool(scene._near[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -0.90, 0.75)) + o),
                                tuple(np.array((0.42, 0.00, 0.05)) + o),
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

    def put_die(xy_env, quat_row=None, z: float | None = None,
                lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> None:
        """CONSTRUCT: write the die root state (env-frame xy, resting height by
        default) — probe instrumentation; the caller settles and judges."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(xy_env[0])
        st[:, 1] = float(xy_env[1])
        st[:, 2] = (c.die_e / 2 + 0.003) if z is None else z
        if quat_row is None:
            st[:, 3] = 1.0  # identity: BLUE up
        else:
            st[:, 3:7] = torch.tensor([list(quat_row)], device=device)
        st[:, 7:10] = torch.tensor([list(lin_vel)], device=device)
        st[:, 10:13] = torch.tensor([list(ang_vel)], device=device)
        st[:, 0:3] += scene.env_origins
        scene.die.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def die_env_xy() -> torch.Tensor:
        return (scene.die.data.root_pos_w - scene.env_origins)[0, :2].clone()

    def goal_env_xy() -> tuple[float, float]:
        g = (scene.goal_mat.data.root_pos_w - scene.env_origins)[0]
        return float(g[0]), float(g[1])

    def decoy_env_xy() -> tuple[float, float]:
        g = (scene.decoy_mat.data.root_pos_w - scene.env_origins)[0]
        return float(g[0]), float(g[1])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    idx0, best0 = scene.face_up()
    check("settle/no-NaN: seeded reset settles finite, die flat on a non-blue face, "
          "far from the mats; score ~0, no success",
          bool(torch.isfinite(scene.die.data.root_state_w).all())
          and float(best0[0]) > c.up_snap and int(idx0[0]) != 4
          and float(scene.goal_dist()[0]) > c.approach_r + 0.03
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (die_env_xy(), scene.die.data.root_quat_w[0].clone(),
                torch.tensor(goal_env_xy()))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_xy, a_q, a_g = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_xy, b_q, b_g = readback()
    q_rel = _qmul(a_q.unsqueeze(0), _qconj(b_q.unsqueeze(0)))[0]
    d_ang = math.degrees(2.0 * math.atan2(float(q_rel[1:].norm()),
                                          abs(float(q_rel[0]))))
    d_xy, d_g = float((a_xy - b_xy).norm()), float((a_g - b_g).norm())
    print(f"[smoke] randomization deltas: die_xy={d_xy * 1000:.1f}mm "
          f"die_orient={d_ang:.1f}deg goal_mat_xy={d_g * 1000:.1f}mm", flush=True)
    check("randomization-is-real: die xy, die orientation, green-mat xy readback differ",
          d_xy > 0.003 and d_ang > 3.0 and d_g > 0.003)

    # ================= 3. start-face + mat-side coverage ==========================================
    faces, sides = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        faces.add(int(scene.init_face[0]))
        sides.add(1 if goal_env_xy()[1] > 0 else -1)
    print(f"[smoke] over 10 resets: start faces {sorted(faces)} "
          f"({[FACE_NAMES[f] for f in sorted(faces)]}), goal-mat sides {sorted(sides)}",
          flush=True)
    check("coverage: >= 3 distinct start faces and the green mat occupies BOTH y "
          "sides over 10 resets (4 is BLUE and must never appear)",
          len(faces) >= 3 and 4 not in faces and sides == {-1, 1})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p_before = die_env_xy()
    _step(240)
    _report("null-policy")
    moved = float((die_env_xy() - p_before).norm())
    check("null-policy-fails: 240 idle steps, die stays put (< 10 mm), score ~0, "
          "no success",
          moved < 0.010 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 5. SLIDE CANNOT ORIENT (the identity claim is physics) =====================
    # A real CoM push inside the slide window (3.6 N in [mu m g = 3.09, m g = 4.41]),
    # velocity-capped at 0.25 m/s, dragging the die >= 0.10 m AWAY from the mats: the
    # up face must never change and blue must never rise. This is the claim that makes
    # rolls mandatory. Non-vacuity: the displacement itself.
    torch.manual_seed(100)
    env.reset()
    _step(90)
    drv = task_scene.ExternalWrenchDriver(scene.die, n, device)
    drv.snapshot_ref()
    drv.calibrate(_step, force=-3.6, tag="(smoke)")
    _step(60)
    _refresh()
    idx_a, _ = scene.face_up()
    p_a = die_env_xy()
    zero3 = torch.zeros(n, 3, device=device)
    push = torch.tensor([[-3.6, 0.0, 0.0]], device=device).expand(n, 3)
    min_updot, max_bluez = 1.0, -1.0
    _REC["on"] = True
    p_prev5 = die_env_xy().clone()
    for _i in range(900):
        _refresh()
        _idx, best = scene.face_up()
        ez = scene._die_axes()[2]
        min_updot = min(min_updot, float(best[0]))
        max_bluez = max(max_bluez, float(ez[0, 2]))
        disp = float((die_env_xy() - p_a).norm())
        if disp >= 0.12:
            break
        # velocity readback is unreliable under an active wrench: FD-position gate
        p_now5 = die_env_xy()
        v_fd5 = float((p_now5 - p_prev5).norm()) * 120.0
        p_prev5 = p_now5.clone()
        if v_fd5 < 0.25:
            drv.apply(push, zero3)
        else:
            drv.clear()
        _step()
    drv.clear()
    _step(90)
    _REC["on"] = False
    _report("slide-probe")
    idx_b, best_b = scene.face_up()
    disp = float((die_env_xy() - p_a).norm())
    print(f"[smoke] slide probe: dragged {disp * 1000:.0f}mm, min_updot={min_updot:.3f} "
          f"max_blue_z={max_bluez:.3f} face {FACE_NAMES[int(idx_a[0])]} -> "
          f"{FACE_NAMES[int(idx_b[0])]}", flush=True)
    check("SLIDE CANNOT ORIENT: a slide-window CoM push drags the die >= 0.10 m yet "
          "the up face never changes and blue never rises",
          disp >= 0.10 and int(idx_b[0]) == int(idx_a[0]) and min_updot > 0.90
          and max_bluez < 0.90 and float(best_b[0]) > c.up_snap
          and not bool(scene.success()[0]))

    # ================= 6. torque tip gate (threshold is real, both halves) ========================
    # Held sub-threshold torque (0.6x m g e/2) about a horizontal axis must NOT tip
    # the die; 1.8x must tip it onto a NEW face. The strong half is the non-vacuity
    # witness for the weak half (same axis, same encode, same die).
    torch.manual_seed(100)
    env.reset()
    _step(90)
    drv.snapshot_ref()
    _refresh()
    idx_a, _ = scene.face_up()
    face_a = int(idx_a[0])

    def face_dot(i: int) -> float:
        """Up-dot of a SPECIFIC face direction (face_up()'s best-dot never drops
        below ~0.707 mid-tip because the best face swaps at 45 deg)."""
        ex, ey, ez = scene._die_axes()
        d = torch.stack([ex[:, 2], -ex[:, 2], ey[:, 2], -ey[:, 2],
                         ez[:, 2], -ez[:, 2]], dim=1)
        return float(d[0, i])

    # torque about -y: the tip direction is -x, AWAY from the mats (no latch grazing)
    tq = torch.tensor([[0.0, -0.6 * TAU_STATIC, 0.0]], device=device).expand(n, 3)
    min_updot = 1.0
    for _i in range(300):
        drv.apply(zero3, tq)
        _step()
        _refresh()
        min_updot = min(min_updot, face_dot(face_a))
    drv.clear()
    _step(60)
    _refresh()
    idx_w, _ = scene.face_up()
    weak_held = int(idx_w[0]) == face_a and min_updot > 0.97
    print(f"[smoke] weak torque (0.6x): min_updot={min_updot:.3f} "
          f"face {FACE_NAMES[face_a]} -> {FACE_NAMES[int(idx_w[0])]}", flush=True)
    _REC["on"] = True
    tq = torch.tensor([[0.0, -1.8 * TAU_STATIC, 0.0]], device=device).expand(n, 3)
    tipped = False
    for _i in range(600):
        _refresh()
        if face_dot(face_a) < 0.6:  # ~53 deg: past the balance point — cut and coast
            tipped = True
            break
        drv.apply(zero3, tq)
        _step()
    drv.clear()
    _step(150)
    _REC["on"] = False
    _report("torque-gate")
    idx_s, best_s = scene.face_up()
    strong_tipped = tipped and int(idx_s[0]) != int(idx_a[0]) and float(best_s[0]) > c.up_snap
    print(f"[smoke] strong torque (1.8x): tipped={tipped} "
          f"face {FACE_NAMES[int(idx_a[0])]} -> {FACE_NAMES[int(idx_s[0])]}", flush=True)
    check("torque tip gate: 0.6x breakaway torque held 300 steps does not tip the "
          "die; 1.8x tips it onto a new settled face",
          weak_held and strong_tipped)

    # ================= 7. negative: the SEED's naive analog =======================================
    # The seed's whole skill is a pure POSITION change of an object whose orientation
    # is irrelevant (grasp + lift 0.1 m). CONSTRUCT its analog: the die carried to the
    # green mat center with its reset orientation preserved. Placed perfectly — but
    # blue is not up, so only the approach latch pays.
    torch.manual_seed(100)
    env.reset()
    _step(90)
    _refresh()
    q_keep = scene.die.data.root_quat_w[0].tolist()
    put_die(goal_env_xy(), q_keep)
    _step(150)
    _report("seed-carry")
    check("negative (SEED analog): die carried to the mat center orientation-"
          "preserved — blue not up, no success, score <= 0.16 (approach only)",
          not bool(scene.blue_up()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.16)

    # ================= 8. negative: wrong face placed on the mat ==================================
    # PURPLE up (blue DOWN), dead-center on the green mat: "an object neatly placed on
    # the target" — the seed-shaped outcome — with the orientation clause wrong.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put_die(goal_env_xy(), task_scene.START_QUATS[4])  # -z up: PURPLE up, blue down
    _step(150)
    _report("wrong-face")
    check("negative (wrong face): die centered on the green mat with PURPLE up (blue "
          "down) — no success, score <= 0.36",
          bool(scene.on_goal()[0]) and not bool(scene.blue_up()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.36)

    # ================= 9. near-miss: blue up, 75 mm off the mat center ============================
    # Everything right except position: 75 mm > zone_tol = 55 mm. All three latches
    # legitimately earned -> the score must pin at the 0.60 cap (float32 0.60000002)
    # and success must refuse on the position clause.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    gx, gy = goal_env_xy()
    put_die((gx + 0.075, gy))  # identity quat: BLUE up
    _step(150)
    _report("near-miss-pos")
    s9 = float(scene.score()[0])
    check("near-miss (position): blue-up die settled 75 mm from the mat center — "
          "latches pay to the 0.60 cap but success refuses",
          bool(scene.blue_up()[0]) and not bool(scene.on_goal()[0])
          and not bool(scene.success()[0]) and 0.599 <= s9 <= 0.601)

    # ================= 10. latch persistence (monotone credit) ====================================
    # Same episode: carry the die far away and let it settle. The latched 0.60 must
    # survive the retreat; success stays False.
    put_die((gx - 0.45, gy - 0.30))
    _step(120)
    _report("latch-persist")
    s10 = float(scene.score()[0])
    check("latch persistence: after retreating far from the mat the latched 0.60 "
          "survives, still no success",
          0.599 <= s10 <= 0.601 and not bool(scene.success()[0]))

    # ================= 11. negative: the decoy mat pays nothing ===================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put_die(decoy_env_xy())  # BLUE up, dead-center on the DECOY
    _step(150)
    _report("decoy")
    _REC["on"] = False
    check("negative (decoy): blue-up die centered on the DARK-GRAY decoy mat — no "
          "approach credit, no success, score <= 0.46",
          not bool(scene.on_goal()[0]) and not bool(scene._near[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.46)

    # ================= 12. settle gate: moving state is not success ===============================
    # Blue up, dead-center on the green mat — but MOVING. Judged immediately: the
    # settle clause must refuse. The die is removed before it can settle into a real
    # success (no probe in this battery reaches success()).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put_die(goal_env_xy(), lin_vel=(0.45, 0.0, 0.0), ang_vel=(0.0, 0.0, 3.0))
    ok_now = not bool(scene.success()[0])
    _step(2)
    ok_2 = not bool(scene.success()[0])
    _report("moving-on-goal")
    put_die((gx - 0.45, gy - 0.30))  # remove before it settles into success
    _step(60)
    check("settle gate: blue-up on the mat center but moving (0.45 m/s, 3 rad/s) is "
          "refused at the judged instants",
          ok_now and ok_2 and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.die_roll")
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
