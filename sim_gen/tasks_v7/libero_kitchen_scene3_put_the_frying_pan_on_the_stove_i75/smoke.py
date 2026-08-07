"""Smoke / rubric-REJECTION battery for PanHookScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real thread-and-hang and the
latched credit is monotone along it). This battery proves the rubric REJECTS wrong
outcomes and that the claims the task rests on — the loop-through-peg geometry, the
free-suspension clause, and the fact that the SEED task's goal state (pan on the
burner) is this task's zero — are load-bearing. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success().

Checks:
  1. settle/no-NaN    — seeded reset settles finite; the pan lies FLAT on the lit
                        burner (the seed task's GOAL state) and scores 0, no success;
  2. randomization    — two seeded resets: READBACK burner xy, pan yaw, hook y and
                        hook z all differ;
  3. hook spread      — over 10 resets the hook's y and z each span > 30 mm and
                        every sample lies inside the configured ranges;
  4. null-policy      — 240 idle steps: score ~0, no success (the seed's solved
                        state is worth nothing here);
  5. deck set-down    — pan teleported off the burner and settled FLAT on the deck
                        (a set-down, the seed's move): clear fires but nothing else —
                        no success, score <= 0.16;
  6. wall lean        — pan settled LEANING handle-up against the wall below the
                        hook: clear + vertical latches fire but the pan touches the
                        deck and no peg is through the loop — no success,
                        score <= 0.31;
  7. cavity hang      — pan hung on the peg by its BODY: the peg is inserted into
                        the open skillet cavity and the pan released so a rim wall
                        catches on the peg. It hangs — but threaded() is False (the
                        peg is nowhere near the handle loop): no success,
                        score <= 0.31 ("hooked by anything other than the loop does
                        not count" is geometry, not fiat);
  8. grounded thread  — the hook is rewritten LOW on the wall (out of the
                        randomized band; instrumentation) and a genuinely threaded
                        hang constructed on it: the pan hangs by the loop, settled,
                        handle-up — but its lowest point is inside the 30 mm deck
                        clearance band, so suspended() refuses: no success, score
                        stays at the 0.60 cap;
  9. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_put_the_frying_pan_on_the_stove_i75.smoke --headless
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
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects
# RTX -> the annotator returns EMPTY frames. Disable the check.
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

_qmul, _qz, _qy, _q_hang = (task_scene._qmul, task_scene._qz, task_scene._qy,
                            task_scene._q_hang)

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ----------------------------------------------------------
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
    print(f"[smoke] {tag:16s} | on_burner={bool(scene.pan_on_burner()[0])} "
          f"clear={bool(scene.clear_of_burner()[0])} "
          f"threaded={bool(scene.threaded()[0])} "
          f"susp={bool(scene.suspended()[0])} "
          f"handle_upz={float(scene._pan_axis_x()[0, 2]):+.2f} "
          f"pan_min_z={float(scene._pan_min_z()[0]):+.3f} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pan_hook")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.75)) + o),
                                tuple(np.array((0.22, 0.00, 0.26)) + o),
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

    oz = float(env.iscene.env_origins[0, 2])
    oxy = env.iscene.env_origins[:, :2]

    def hook_yz():
        _refresh()
        h = scene.hook.data.root_pos_w[0] - env.iscene.env_origins[0]
        return float(h[1]), float(h[2])

    def q_base():
        """Vertical attitude: local x (handle) -> world up, local z (aperture
        axis) -> world +x (no peg pitch)."""
        s = math.sqrt(0.5)
        return torch.tensor([0.0, s, 0.0, s], device=device).expand(n, 4).clone()

    # ================= 1. settle / no-NaN (start = the SEED task's goal) ========================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.pan.data.root_pos_w).all()
               and torch.isfinite(scene.pan.data.root_lin_vel_w).all())
    check("settle/no-NaN: pan flat on the lit burner (the seed's GOAL state), "
          "score 0, no success",
          fin and bool(scene.pan_on_burner()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        hy, hz = hook_yz()
        return (scene.burner.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pan.data.root_quat_w[0]), hy, hz)

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_b, a_y, a_hy, a_hz = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_b, b_y, b_hy, b_hz = readback()
    d_b = float((a_b - b_b).norm())
    d_y = dyaw(a_y, b_y)
    print(f"[smoke] randomization deltas: burner_xy={d_b * 1000:.1f}mm "
          f"pan_yaw={d_y:.1f}deg hook_dy={abs(a_hy - b_hy) * 1000:.1f}mm "
          f"hook_dz={abs(a_hz - b_hz) * 1000:.1f}mm", flush=True)
    check("randomization-is-real: burner xy, pan yaw, hook y, hook z readback differ",
          d_b > 0.003 and d_y > 5.0
          and abs(a_hy - b_hy) > 0.005 and abs(a_hz - b_hz) > 0.005)

    # ================= 3. hook placement spread + in-range ======================================
    ys, zs = [], []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        hy, hz = hook_yz()
        ys.append(hy)
        zs.append(hz)
    in_rng = all(c.hook_y_range[0] - 1e-3 <= y <= c.hook_y_range[1] + 1e-3 for y in ys) \
        and all(c.hook_z_range[0] - 1e-3 <= z <= c.hook_z_range[1] + 1e-3 for z in zs)
    print(f"[smoke] hook y span={(max(ys) - min(ys)) * 1000:.0f}mm "
          f"z span={(max(zs) - min(zs)) * 1000:.0f}mm over 10 resets", flush=True)
    check("hook spread: y and z each span > 30 mm over 10 resets, all in range",
          in_rng and (max(ys) - min(ys)) > 0.03 and (max(zs) - min(zs)) > 0.03)

    # ================= 4. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps on the lit burner, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. deck set-down (the seed's MOVE, wrong here) ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = 0.55
    pos[:, 1] = -0.18
    pos[:, 2] = oz + c.deck_top + 0.004
    pos[:, :2] += oxy
    _write_body(scene.pan, pos, None)
    _step(150)
    _report("deck-set-down")
    check("negative (set-down): pan settled FLAT on the bare deck — clear fires, "
          "nothing else: no success, score <= 0.16",
          bool(scene.clear_of_burner()[0]) and not bool(scene.threaded()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)

    # ================= 6. leaning against the wall ==============================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    hy, _hz = hook_yz()
    lean_y = -0.20 if hy > 0 else 0.20  # keep clear of the randomized hook
    th = math.radians(-15.0)  # tip the top toward the wall (-x)
    q_lean = _qmul(_qy(torch.full((n,), th, device=device)), q_base())
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = 0.055
    pos[:, 1] = lean_y
    pos[:, 2] = oz + c.deck_top + 0.004
    pos[:, :2] += oxy
    _write_body(scene.pan, pos, q_lean)
    _step(240)
    _report("wall-lean")
    _REC["on"] = False
    up_z = float(scene._pan_axis_x()[0, 2])
    check("negative (lean): pan leaning handle-up against the wall — vertical "
          "latch fires but it touches the deck and no peg is through the loop: "
          "no success, score <= 0.31",
          up_z > 0.7 and not bool(scene.threaded()[0])
          and not bool(scene.suspended()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.31)

    # ================= 7. cavity hang: hooked by the BODY, not the loop =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # cavity mouth toward the wall (local +z -> world -x), handle up
    q_cav = _qmul(_qz(torch.full((n,), math.pi, device=device)), _q_hang(n, device))
    # put the peg's near-tip point just inside the cavity, close to the upper rim
    # wall, so the release drops the rim wall onto the peg
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415

    p_peg = scene.peg_point_w(c.hook_plate_t + c.peg_len - 0.030)
    anchor = torch.tensor([0.045, 0.0, 0.027], device=device).expand(n, 3)
    pos = p_peg - quat_apply(q_cav, anchor)
    _write_body(scene.pan, pos, q_cav)
    _step(300)
    _report("cavity-hang")
    _REC["on"] = False
    hung_high = float(scene._pan_min_z()[0]) > c.deck_top + 0.005
    check("negative (cavity hang): pan hung on the peg by its BODY (rim wall "
          "caught on the peg, pan airborne) — threaded() is False: no success, "
          "score <= 0.31",
          hung_high and not bool(scene.threaded()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.31)

    # ================= 8. grounded thread: suspension is load-bearing ===========================
    # INSTRUMENTATION: rewrite the (kinematic) hook LOW on the wall, then construct
    # a genuinely threaded hang on it. The pan hangs by the loop, settled and
    # handle-up — but its lowest point sits inside the 30 mm deck-clearance band.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    low_z = 0.340
    hpos = torch.zeros(n, 3, device=device)
    hpos[:, 2] = oz + low_z
    hpos[:, :2] += oxy
    pitch = torch.full((n,), math.radians(-c.hook_pitch_deg), device=device)
    _write_body(scene.hook, hpos, _qy(pitch))
    # hang the loop at mid-peg and release
    qh = _q_hang(n, device)
    h0 = torch.tensor(c.hole_local, device=device).expand(n, 3)
    p_mid = scene.peg_point_w(c.hook_plate_t + 0.060)
    _write_body(scene.pan, p_mid - quat_apply(qh, h0), qh)
    _step(360)
    _report("grounded-thread")
    _REC["on"] = False
    _refresh()
    min_z = float(scene._pan_min_z()[0])
    in_band = c.deck_top < min_z < c.deck_top + c.suspend_clear
    print(f"[smoke] grounded thread: pan_min_z-deck_top={ (min_z - c.deck_top) * 1000:+.1f}mm "
          f"(clearance bound {c.suspend_clear * 1000:.0f}mm)", flush=True)
    check("near-miss (grounded thread): pan REALLY hanging by the loop on a low "
          "hook, handle up — lowest point inside the 30 mm clearance band, "
          "suspended() refuses: no success, score <= 0.60",
          bool(scene.threaded()[0]) and in_band
          # 0.605: the 0.60 cap is stored in float32 (0.60000002...), so an
          # exact <= 0.60 comparison in float64 would fail on the cap itself.
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.605)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pan_hook")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
