"""Smoke / rubric-REJECTION battery for DieTumbleScene — NullRobot, teleported probes.

solve.py is the acceptance proof (real tumbles reach the goal and the latched credit is
monotone along that trajectory). This battery proves the rubric REJECTS wrong outcomes,
that the randomization is real, and that the physical claim the task rests on — the
raised rim cannot be crossed by sliding, so entry forces one more quarter-turn — is
load-bearing physics, not fiat. Every probe is CONSTRUCTED as a settled state
(teleport, real physics steps, judge); no probe here reaches success().

Checks:
   1. settle/baseline   — seeded reset settles finite; die at ground rest outside the
                          tray, red not up; score 0, no success;
   2. randomization     — tray xy and die start xy READBACK differ across seeds; die
                          starts on the -x side at a distance inside the sampled band;
   3. orientation cover — over 10 seeded resets the red face starts in >= 4 distinct
                          world directions and NEVER up — the tumble plan must be
                          perceived per episode, not memorized;
   4. null-policy       — 360 idle steps: score ~0, no success;
   5. SEED STRATEGY     — the seed's plan verbatim ("transport the piece to the goal
                          and set it down"): the die dropped INTO the tray with its
                          reset orientation kept — arrived latches, but red is not up:
                          no success, score <= 0.41;
   6. red-up-outside    — die set red-face-up on open floor: oriented latches, but it
                          is not in the tray: no success, score <= 0.41;
   7. near-miss wall    — die red-up settled flush against the tray's OUTER wall
                          (2 mm shy): |tray-local x| ~ 0.140 >> in_tol 0.055 — the
                          containment window refuses, no success;
   8. RIM certificate   — a quasi-static capped push (2.2 N, > the ~1.6 N slide
                          threshold, < the ~5 N step-climb threshold) really slides
                          the die >= 40 mm across the floor INTO the rim, where it
                          stalls OUTSIDE: never in_tray, never climbs — sliding
                          cannot enter the tray, entry must tumble (and therefore
                          quarter-turns the die);
   9. motion gate       — the die dropped into the tray red-up and SPINNING (5 rad/s):
                          success() refuses on every step the readback shows motion —
                          only settled rest counts; the probe state is discarded by a
                          reset before it can settle into the goal;
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.stack_pyramid_i255.smoke --headless
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

import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401 — registers the scene
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


def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_tumble")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.75, 0.60)) + o),
                                tuple(np.array((-0.10, 0.00, 0.03)) + o),
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

    def settle(max_steps: int = 600, need: int = 12) -> bool:
        streak = 0
        for _ in range(max_steps):
            _step(1)
            streak = streak + 1 if bool(scene.still()[0]) else 0
            if streak >= need:
                return True
        return False

    def report(tag: str) -> None:
        loc = scene.die_tray_local()[0]
        r = scene.red_world()[0]
        print(f"[smoke] {tag:16s} | die tray-local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) red_w=({float(r[0]):+.2f},{float(r[1]):+.2f},"
              f"{float(r[2]):+.2f}) in_tray={bool(scene.in_tray()[0])} "
              f"red_up={bool(scene.red_up()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(_REC['frames'])}", flush=True)

    def red_dir() -> tuple[int, int, int]:
        r = scene.red_world()[0]
        ax = int(torch.argmax(r.abs()))
        v = [0, 0, 0]
        v[ax] = 1 if float(r[ax]) > 0 else -1
        return tuple(v)  # type: ignore[return-value]

    def put_die(x_w: float, y_w: float, z_env: float, quat=None,
                ang_z: float = 0.0) -> None:
        """Author a die root state: position, optional orientation (wxyz, default:
        keep the current readback), optional spin about z."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x_w
        st[:, 1] = y_w
        st[:, 2] = scene.env_origins[:, 2] + z_env
        if quat is None:
            st[:, 3:7] = scene.die.data.root_quat_w.clone()
        else:
            st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        st[:, 12] = ang_z
        scene.die.write_root_state_to_sim(st, _all_ids())
        _refresh()

    z_ground = c.die_s / 2 + 0.0015                 # floor rest + tiny clearance
    z_plate = c.plate_t + c.die_s / 2 + 0.0015      # slab rest + tiny clearance
    x_wall_out = c.inner / 2 + c.wall_t + c.die_s / 2 + 0.002  # flush-outside |x| offset

    # ================= 1. settle / baseline =======================================================
    env.reset(seed=42)
    _REC["on"] = True
    _step(180)
    report("settle")
    _REC["on"] = False
    loc = scene.die_tray_local()[0]
    check("settle/baseline: layout settles finite; die at ground rest outside the tray, "
          "red not up; score 0, no success",
          bool(scene._finite()[0])
          and abs(float(loc[2]) - c.die_s / 2) < 0.005
          and not bool(scene.in_tray()[0]) and not bool(scene.red_up()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def layout():
        _refresh()
        loc = scene.die_tray_local()[0]
        return (scene.tray_xy[0].clone(), float(loc[0]), float(loc[1]))

    env.reset(seed=7)
    _step(5)
    t_a, dx_a, dy_a = layout()
    r_a = (dx_a ** 2 + dy_a ** 2) ** 0.5
    env.reset(seed=8)
    _step(5)
    t_b, dx_b, dy_b = layout()
    r_b = (dx_b ** 2 + dy_b ** 2) ** 0.5
    d_tray = float((t_a - t_b).norm())
    d_die = ((dx_a - dx_b) ** 2 + (dy_a - dy_b) ** 2) ** 0.5
    print(f"[smoke] randomization deltas: tray_xy={d_tray * 1000:.1f}mm "
          f"die_rel_xy={d_die * 1000:.1f}mm dist_a={r_a:.3f} dist_b={r_b:.3f}", flush=True)
    check("randomization-is-real: tray xy and die start (tray-relative) readback differ "
          "across seeds; die starts on the -x side inside the sampled distance band",
          d_tray > 0.005 and d_die > 0.010
          and dx_a < -0.1 and dx_b < -0.1
          and c.die_r_min - 0.02 < r_a < c.die_r_max + 0.02
          and c.die_r_min - 0.02 < r_b < c.die_r_max + 0.02)

    # ================= 3. orientation coverage ====================================================
    dirs = set()
    never_up = True
    for s in range(10):
        env.reset(seed=100 + s)
        _step(5)
        d = red_dir()
        dirs.add(d)
        never_up = never_up and d != (0, 0, 1) and not bool(scene.red_up()[0])
    print(f"[smoke] over 10 resets: red starts in {len(dirs)} distinct directions "
          f"{sorted(dirs)}", flush=True)
    check("orientation coverage: red starts in >= 4 distinct world directions over 10 "
          "resets and NEVER up — the tumble plan must be perceived per episode",
          len(dirs) >= 4 and never_up)

    # ================= 4. null policy fails =======================================================
    env.reset(seed=42)
    _step(360)
    report("null-policy")
    check("null-policy-fails: 360 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed (stack_pyramid) is pure pick-and-place: transport the piece to the goal
    # and set it down — orientation is never considered. Verbatim here: the die set
    # down INSIDE the tray with its reset orientation kept. arrived latches; red is
    # not up (the start set guarantees it): no success.
    env.reset(seed=42)
    _step(30)
    _REC["on"] = True
    put_die(float(scene.tray_xy[0, 0]), float(scene.tray_xy[0, 1]), z_plate + 0.003)
    _step(240)
    report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): die set down inside the tray with its start "
          "orientation kept — arrived latches but red is not up: no success, "
          "score <= 0.41",
          bool(scene.in_tray()[0]) and not bool(scene.red_up()[0])
          and bool(scene._arrived[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.41)

    # ================= 6. red up but OUTSIDE the tray =============================================
    env.reset(seed=42)
    _step(30)
    put_die(float(scene.tray_xy[0, 0]) - 0.30, float(scene.tray_xy[0, 1]) + 0.10,
            z_ground, quat=(1.0, 0.0, 0.0, 0.0))
    _step(240)
    report("red-up-outside")
    check("red-up-outside: die red-face-up on open floor — oriented latches but the "
          "die is not in the tray: no success, score <= 0.41",
          bool(scene.red_up()[0]) and not bool(scene.in_tray()[0])
          and bool(scene._oriented[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.41)

    # ================= 7. near-miss: flush against the OUTER wall =================================
    env.reset(seed=42)
    _step(30)
    put_die(float(scene.tray_xy[0, 0]) - x_wall_out, float(scene.tray_xy[0, 1]),
            z_ground, quat=(1.0, 0.0, 0.0, 0.0))
    assert settle(), "near-miss probe must settle"
    _step(60)
    report("near-miss-wall")
    loc = scene.die_tray_local()[0]
    check("near-miss wall: die red-up settled flush against the tray's OUTER wall "
          "(|tray-local x| ~ 0.140 >> in_tol 0.055) — containment refuses, no success",
          bool(scene.red_up()[0]) and abs(float(loc[0])) > 0.12
          and abs(float(loc[2]) - c.die_s / 2) < 0.005
          and not bool(scene.in_tray()[0]) and not bool(scene.success()[0]))

    # ================= 8. RIM certificate (sliding cannot enter) ==================================
    # Quasi-static velocity-servoed push toward +x, hard cap 2.2 N: above the ~1.6 N
    # ground-slide threshold, far below the ~5 N quasi-static step-climb threshold
    # and the mg = 2.9 N mid-height tip threshold. The die must really slide (>= 40
    # mm — the probe actuates, not vacuous), then stall at the rim OUTSIDE: never
    # in_tray, never climbing. Wrench frame: per-step body-frame pre-encode
    # (quat_apply_inverse) — this IsaacLab drags "world" wrenches by the rotation
    # since first application.
    from isaaclab.utils.math import quat_apply_inverse

    env.reset(seed=42)
    _step(30)
    put_die(float(scene.tray_xy[0, 0]) - 0.28, float(scene.tray_xy[0, 1]), z_ground)
    assert settle(), "rim probe must settle before the push"
    _REC["on"] = True
    x0 = float(scene.die_tray_local()[0, 0])
    zero3 = torch.zeros(n, 1, 3, device=device)
    entered_ever = False
    z_max = 0.0
    x_prev, stall = x0, 0
    f_last = 0.0
    for i in range(900):
        vx = float(scene.die.data.root_lin_vel_w[0, 0])
        f_mag = max(0.0, min(2.2, 1.6 + 1.8 * (0.05 - vx)))
        f_last = f_mag
        f_w = torch.tensor([f_mag, 0.0, 0.0], device=device).expand(n, 3)
        f_b = quat_apply_inverse(scene.die.data.root_quat_w, f_w)
        scene.die.set_external_force_and_torque(f_b.unsqueeze(1), zero3,
                                                env_ids=_all_ids())
        _step(1)
        loc = scene.die_tray_local()[0]
        entered_ever = entered_ever or bool(scene.in_tray()[0])
        z_max = max(z_max, float(loc[2]))
        if i % 30 == 29:
            x_now = float(loc[0])
            stall = stall + 1 if abs(x_now - x_prev) < 0.001 else 0
            x_prev = x_now
            if stall >= 4 and i > 240:  # 4 windows = 1 s without progress at max push
                break
    scene.die.set_external_force_and_torque(zero3, zero3, env_ids=_all_ids())
    _step(120)
    report("rim-certificate")
    _REC["on"] = False
    loc = scene.die_tray_local()[0]
    slid = float(loc[0]) - x0
    print(f"[smoke] rim probe: slid {slid * 1000:.1f}mm, stalled at tray-local "
          f"x={float(loc[0]):+.3f} (flush {-x_wall_out:+.3f}), z_max={z_max:.3f}, "
          f"last F={f_last:.2f} N, entered_ever={entered_ever}", flush=True)
    check("RIM certificate: a 2.2 N-capped quasi-static push really slides the die "
          ">= 40 mm into the rim, where it stalls OUTSIDE (never in_tray, never "
          "climbs) — sliding cannot enter, entry must tumble",
          slid > 0.04 and not entered_ever
          and abs(float(loc[0])) > 0.12
          and z_max < c.die_s / 2 + 0.006
          and not bool(scene.in_tray()[0]) and not bool(scene.success()[0]))

    # ================= 9. spin gate (only settled rest counts) ====================================
    env.reset(seed=42)
    _step(30)
    _REC["on"] = True
    put_die(float(scene.tray_xy[0, 0]), float(scene.tray_xy[0, 1]), z_plate + 0.008,
            quat=(1.0, 0.0, 0.0, 0.0), ang_z=5.0)
    w0 = float(scene.die.data.root_ang_vel_w[0].norm())
    moving_steps, refused = 0, True
    for _ in range(6):
        _step(1)
        w = float(scene.die.data.root_ang_vel_w[0].norm())
        v = float(scene.die.data.root_lin_vel_w[0].norm())
        if w > 0.5 or v > c.settle_lin:
            moving_steps += 1
            refused = refused and not bool(scene.success()[0])
    report("motion-gate")
    _REC["on"] = False
    print(f"[smoke] motion probe: w after write={w0:.2f} rad/s, "
          f"moving steps={moving_steps}/6", flush=True)
    env.reset(seed=42)  # discard the probe state before it can settle into the goal
    check("motion gate: die red-up over the tray slab but MOVING (spin + drop; "
          ">= 2 steps with |w| > 0.5 or |v| > settle_lin readback) — success() "
          "refuses on every moving step: only settled rest counts",
          moving_steps >= 2 and refused)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.die_tumble")
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
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
