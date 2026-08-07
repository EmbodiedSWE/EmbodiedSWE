"""Smoke / rubric-REJECTION battery for CheckerCryptScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claims the task rests on — the seated lid is
flush (nothing to pinch) and rim-locked (nowhere to slide), so the crate cannot be
opened without the pry — are physically load-bearing. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
   1. settle/flush       — seeded reset settles finite; lid top FLUSH with the rim top
                           (< 2 mm proud: no graspable edge), lid flat, all three kings
                           inside the sealed crate; score 0, no success;
   2. randomization      — two seeded resets: READBACK crate yaw, crate xy, pad xy,
                           bar xy all differ;
   3. slot permutation   — over 10 resets the RED king occupies every one of the three
                           interior anchors at least once;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEAL INTERLOCK     — the seated lid is shoved horizontally at 3x its own weight
                           for 1.5 s toward the rim gap and again sideways: it never
                           leaves the pocket AND never tilts past the pry gate — the
                           "cannot slide it off, prying is the only way" claim, and the
                           honesty of the 3 deg pried-latch, are physics, not fiat;
   6. SEED STRATEGY      — the seed's plan ("place the piece on its square"): the red
                           king CONSTRUCTED standing on the pad while the crate is
                           still SEALED -> NO success (the lid clause refuses);
   7. lid near-miss      — lid off but on the ground right beside the crate wall
                           (0.19 m < the 0.24 m clear gate), red on the pad -> no
                           success, score capped;
   8. wrong object       — lid properly clear, a WHITE king stood on the pad, the red
                           left in the crate -> no success;
   9. restraint          — lid clear, red on the pad, but one WHITE king also taken
                           out onto the ground -> no success;
  10. pad near-miss      — lid clear, red king upright but 70 mm from the pad centre
                           (outside the 45 mm tolerance) -> no success;
  11. on-side            — lid clear, red king ON the pad but lying on its SIDE ->
                           the uprightness clause refuses, no success;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_checkers_i39.smoke --headless
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
    d, z, loc = scene._lid_stats()
    print(f"[smoke] {tag:18s} | lid d={float(d[0]):.3f} z={float(z[0]):.3f} "
          f"tilt={float(scene.lid_tilt_deg()[0]):.2f}deg "
          f"in_crate={scene.in_crate()[0].tolist()} "
          f"pried={bool(scene._pried[0])} opened={bool(scene._opened[0])} "
          f"out={bool(scene._out[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


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
    env = ENVS.get("simgen.checker_crypt")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -0.80, 0.80)) + o),
                                tuple(np.array((0.38, -0.02, 0.08)) + o),
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

    def crate_world(loc_xyz) -> torch.Tensor:
        """Crate-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.crate.data.root_pos_w + quat_apply(scene.crate.data.root_quat_w, loc)

    def ground_z(dz: float) -> torch.Tensor:
        return env.iscene.env_origins[:, 2] + dz

    def remove_lid_clear() -> None:
        """CONSTRUCT the lid resting on open ground, well clear of the crate."""
        p = crate_world((0.32, 0.10, 0.0))
        p[:, 2] = ground_z(0.05)
        _write_body(scene.lid, p, None)
        _step(90)

    def red_to_pad(offset_xy=(0.0, 0.0), on_side: bool = False) -> None:
        p = scene.pad.data.root_pos_w.clone()
        p[:, 0] += offset_xy[0]
        p[:, 1] += offset_xy[1]
        if on_side:
            p[:, 2] = ground_z(c.pad_t + c.puck_r + 0.004)
            s = math.sqrt(0.5)
            q = torch.tensor([s, s, 0.0, 0.0], device=device).expand(n, 4)  # 90 deg roll
        else:
            p[:, 2] = ground_z(0.06)
            q = None
        _write_body(scene.pucks["red_0"], p, q)
        _step(90)

    # ================= 1. settle / flush (the no-grasp certificate) ===============================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    _d0, _z0, lid_loc = scene._lid_stats()
    lid_top = float(lid_loc[0, 2]) + c.lid_t / 2
    rim_top = c.z_seat + c.rim_h
    pos, _q, _v = scene._puck_tensors()
    print(f"[smoke] flushness: lid top {lid_top * 1000:.1f}mm vs rim top "
          f"{rim_top * 1000:.1f}mm (proud {(lid_top - rim_top) * 1000:+.1f}mm), "
          f"tilt {float(scene.lid_tilt_deg()[0]):.2f}deg", flush=True)
    check("settle/flush: layout settles finite; lid top flush with the rim (< 2 mm "
          "proud — no pinchable edge), lid flat, three kings sealed inside; score 0, "
          "no success",
          bool(torch.isfinite(pos).all())
          and abs(lid_top - rim_top) < 0.002
          and float(scene.lid_tilt_deg()[0]) < 1.5
          and bool(scene.in_crate()[0].all())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.crate.data.root_quat_w[0]),
                scene.crate.data.root_pos_w[0, :2].clone(),
                scene.pad.data.root_pos_w[0, :2].clone(),
                scene.bar.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_pp, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_pp, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_cp = float((a_cp - b_cp).norm())
    d_pp = float((a_pp - b_pp).norm())
    d_bp = float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: crate_yaw={d_yawv:.1f}deg "
          f"crate_xy={d_cp * 1000:.1f}mm pad_xy={d_pp * 1000:.1f}mm "
          f"bar_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: crate yaw, crate xy, pad xy, bar xy readback differ",
          d_yawv > 2.0 and d_cp > 0.003 and d_pp > 0.003 and d_bp > 0.003)

    # ================= 3. red-anchor permutation coverage =========================================
    red_anchors = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        red_anchors.add(int(scene.slot_of[0, 0]))
    print(f"[smoke] over 10 resets: red king anchor indices {sorted(red_anchors)}",
          flush=True)
    check("slot permutation: the red king occupies all three interior anchors over "
          "10 resets — where the red sits must be perceived, not memorized",
          red_anchors == {0, 1, 2})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEAL INTERLOCK: the seated lid cannot be slid or jostled off ============
    # Shove the seated lid horizontally at 3x its own weight for 0.75 s toward the
    # rim GAP (the weakest side), then 0.75 s sideways. The rim must hold it in the
    # pocket, and the shove must never tilt it past the 3 deg pry gate — otherwise
    # "prying is the only way in" (and the pried-latch itself) would be fiat.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    mag = 3.0 * c.lid_mass * 9.81
    max_off, max_tilt = 0.0, 0.0
    for direction in ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        dir_w = quat_apply(scene.crate.data.root_quat_w,
                           torch.tensor(direction, device=device).expand(n, 3))[0]
        for _ in range(9):  # 9 x 10 = 90 steps = 0.75 s per direction
            _push(scene.lid, mag * dir_w, 10)
            _refresh()
            _d, _z, loc = scene._lid_stats()
            max_off = max(max_off, float(loc[0, 0:2].abs().max()))
            max_tilt = max(max_tilt, float(scene.lid_tilt_deg()[0]))
    _step(60)
    _report("seal-interlock")
    _d, _z, loc = scene._lid_stats()
    print(f"[smoke] seal shove: max lid offset {max_off * 1000:.1f}mm, max tilt "
          f"{max_tilt:.2f}deg (pry gate {c.pry_tilt_deg:.1f}deg)", flush=True)
    check("SEAL INTERLOCK: 3x-weight horizontal shoves (toward the rim gap, then "
          "sideways) never free the seated lid and never tilt it past the pry gate",
          max_off < 0.03 and max_tilt < c.pry_tilt_deg
          and float(loc[0, 2]) > 0.07 and not bool(scene._opened[0])
          and not bool(scene._pried[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # "Place the piece on its square" — the only skill the seed's plan exercises.
    # CONSTRUCT the red king standing perfectly on the pad while the crate is still
    # SEALED (a state no policy could reach, which is the point): the lid clause
    # must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    red_to_pad()
    _report("seed-strategy")
    check("negative (SEED strategy): red king standing on the pad but the crate "
          "still sealed — lid clause refuses, no success, score <= 0.30",
          bool(scene.red_on_pad()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.30)
    _REC["on"] = False

    # ================= 7. near-miss: lid off but NOT clear ========================================
    # Lid flat on the ground right beside the crate wall (0.19 m < the 0.24 m gate),
    # red on the pad, whites in: "off" is not "clear".
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p = crate_world((-0.19, 0.0, 0.0))
    p[:, 2] = ground_z(c.lid_t / 2 + 0.002)
    _write_body(scene.lid, p, None)
    _step(60)
    red_to_pad()
    _report("lid-near-miss")
    _d, _z, _loc = scene._lid_stats()
    check("near-miss (lid): lid on the ground 0.19 m out — beside the crate but "
          "inside the 0.24 m clear gate — red on pad, no success, score <= 0.75",
          float(_d[0]) < c.lid_clear_r and bool(scene.red_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 8. negative: wrong object on the pad =======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    remove_lid_clear()
    p = scene.pad.data.root_pos_w.clone()
    p[:, 2] = ground_z(0.06)
    _write_body(scene.pucks["white_0"], p, None)
    _step(90)
    _report("wrong-object")
    check("negative (wrong object): lid properly clear but a WHITE king stands on "
          "the pad and the red stays in the crate — no success",
          bool(scene.lid_clear()[0]) and bool(scene.in_crate()[0, 0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 9. negative: restraint violated ============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    remove_lid_clear()
    red_to_pad()
    p = crate_world((0.0, -0.30, 0.0))
    p[:, 2] = ground_z(c.puck_h / 2 + 0.002)
    _write_body(scene.pucks["white_1"], p, None)
    _step(90)
    _report("restraint")
    check("negative (restraint): lid clear and red on the pad, but a white king was "
          "also taken out onto the ground — no success, score <= 0.75",
          bool(scene.lid_clear()[0]) and bool(scene.red_on_pad()[0])
          and not bool(scene.in_crate()[0, 2])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 10. near-miss: red outside the pad tolerance ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    remove_lid_clear()
    red_to_pad(offset_xy=(0.070, 0.0))
    _report("pad-near-miss")
    rd = float((scene.pucks["red_0"].data.root_pos_w[0, 0:2]
                - scene.pad.data.root_pos_w[0, 0:2]).norm())
    print(f"[smoke] red-to-pad distance {rd * 1000:.1f}mm (tol "
          f"{c.pad_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (pad): lid clear, red king upright ~70 mm from the pad centre "
          "(tol 45 mm) — no success, score <= 0.75",
          rd > c.pad_tol and not bool(scene.red_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 11. negative: red on the pad but on its SIDE ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    remove_lid_clear()
    red_to_pad(on_side=True)
    _report("on-side")
    ax_z = float(scene._z_axis_w(scene.pucks["red_0"].data.root_quat_w)[0, 2])
    print(f"[smoke] on-side: red axis z-component {ax_z:+.3f} (upright gate "
          f"{math.cos(math.radians(c.upright_max_deg)):.3f})", flush=True)
    check("negative (on-side): lid clear, red king lying on its side at the pad — "
          "the uprightness clause refuses, no success",
          bool(scene.lid_clear()[0]) and not bool(scene.red_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.checker_crypt")
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
