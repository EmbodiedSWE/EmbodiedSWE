"""Smoke battery for RatchetRampScene (sim_gen task
`libero_kitchen_scene1_open_top_drawer_i325`) — REJECTION-ONLY: every check either
verifies basic health/randomization or CONSTRUCTS a settled wrong outcome and
asserts the rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — flaps hang closed at their hinges, balls in their bays,
                        score ~0, no success.
 2. randomization     — bay permutation and ball xy jitter vary across resets
                        (readback).
 3. null policy       — 240 idle steps -> score ~0, no success.
 4. SEED strategy     — the seed's whole skill (swing the articulated part) done
                        for real: flap 1 torqued open past 45 deg (flaps_closed
                        goes False at the peak — non-vacuous), released, falls
                        shut again. Latches nothing, score ~0, no success.
 5. ratchet retention — a ball placed between the flaps rolls back downhill and
                        is HELD by flap 1 (the one-way pawl, for real); only the
                        p1 latch fires (score 0.10), no success.
 6. top-entry holds   — a ball dropped through the open mouth stretch lands
                        upstream of flap 1 and rolls back out of the mouth: no
                        latch, no credit.
 7. roof holds        — a ball dropped from above the basin lands ON the roof,
                        never inside: no bin latch, no credit.
 8. decoy poisons     — the RED decoy constructed in the basin (with one cargo):
                        the decoy latches nothing (score counts only the cargo),
                        success refused. The basin holds at most two balls and
                        extraction is impossible (check 11), so a delivered decoy
                        is a permanent dead end.
 9. missing delivery  — one cargo in the basin, the other still in its bay:
                        partial 0.35 only, no success.
10. latched credit    — the delivered cargo removed to open ground: the 0.35
                        latch survives, success does not.
11. extraction holds  — a sustained full-strength (2.5 N cap, velocity-capped so
                        the probe stays quasi-static) downhill press pins a basin
                        ball against the retaining step (it travels to the step —
                        non-vacuous) but cannot lift it out: still in the basin
                        after release.
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — >10 frames recorded, frames.npz written to the CWD.

NOTE: no check constructs BOTH cargo balls in the basin — any such state, once
settled, IS success (there is no residual disqualifier: the decoy cannot fit in
beside them and the flaps re-close by gravity), and the battery's audit demands
success() never fire. The full-success trajectory is solve.py's job.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i325.smoke
             --headless
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
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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


def _write_body(body, pos_w: torch.Tensor) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = 1.0
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    loc = scene._ball_loc()[0]
    ib = scene.balls_in_basin()[0]
    ang = scene.flap_angle()[0]
    print(f"[smoke] {tag:16s} | "
          + " ".join(f"{nm}=({float(loc[i, 0]):+.3f},{float(loc[i, 1]):+.3f},"
                     f"{float(loc[i, 2]):+.3f}){'B' if bool(ib[i]) else ''}"
                     for i, nm in enumerate(scene.BALLS))
          + f" flaps=({math.degrees(float(ang[0])):+.1f},"
          f"{math.degrees(float(ang[1])):+.1f})deg "
          f"closed={bool(scene.flaps_closed()[0])} "
          f"L=p1{[int(v) for v in scene._p1[0]]}p2{[int(v) for v in scene._p2[0]]}"
          f"b{[int(v) for v in scene._bin[0]]} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ratchet_ramp")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.95, 0.60)) + o),
                                tuple(np.array((0.25, 0.00, 0.08)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def floor_top(x: float) -> float:
        return c.ramp_z_lo + (x - c.ramp_x_lo) * c.tan_t

    def put_local(ball: str, x: float, y: float, z: float) -> None:
        pos = scene.env_origins + torch.tensor([x, y, z], device=device).expand(n, 3)
        _write_body(scene.balls[ball], pos)

    def put_in_basin(ball: str, x: float) -> None:
        """CONSTRUCT: write the ball just above the basin floor at station x."""
        put_local(ball, x, 0.0, c.basin_floor_top + c.ball_r + 0.004)

    zero = torch.zeros(n, 1, 3, device=device)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(120)
    _report("reset")
    loc1 = scene._ball_loc()[0]
    in_bays = all(float(loc1[i, 0]) < c.wall_x0 - c.ball_r
                  and abs(float(loc1[i, 2]) - c.ball_r) < 0.01 for i in range(3))
    check("settle/no-NaN: flaps hang closed at their hinges, balls on the ground in "
          "their bays, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.flaps_closed()[0]) and in_bays
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 2. randomization readback ==================================================
    perms, xys = [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(6)
        _refresh()
        perms.append(tuple(scene.bay_perm[0].tolist()))
        xys.append(scene._ball_loc()[0, :, :2].reshape(-1).tolist())
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: perms={perms} ({len(set(perms))} distinct) "
          f"xystd={xystd:.4f}", flush=True)
    check("randomization: bay permutation varies (>=3 distinct in 8 resets) and "
          "ball positions jitter (mean per-coordinate std > 8 mm)",
          len(set(perms)) >= 3 and xystd > 0.008)

    # ================= 3. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 4. SEED strategy (swing the articulated part, for real) ====================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    flap = scene.flaps["flap_1"]
    tq = torch.zeros(n, 1, 3, device=device)
    tq[:, 0, 1] = -0.02        # N*m about the hinge axis (body y == world y always)
    peak = 0.0
    open_seen = False
    for _ in range(150):
        flap.set_external_force_and_torque(zero, tq)
        _step(1)
        a = float(scene.flap_angle()[0, 0])
        peak = min(peak, a)
        open_seen = open_seen or not bool(scene.flaps_closed()[0])
    flap.set_external_force_and_torque(zero, zero)
    _step(360)                 # gravity return
    _report("seed-skill")
    check("SEED strategy: flap 1 torqued open past 45 deg for real (flaps_closed "
          "went False at the peak), falls shut on release — latches nothing, "
          "score ~0, no success",
          peak < -math.radians(45) and open_seen and bool(scene.flaps_closed()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 5. ratchet retention =======================================================
    torch.manual_seed(51)
    env.reset()
    _step(90)
    x5 = 0.250                 # between the flaps, past the p1 threshold band
    put_local("cargo_0", x5, 0.0, floor_top(x5) + c.ball_r + 0.003)
    _step(300)
    _report("ratchet")
    xf = float(scene._ball_loc()[0, 0, 0])
    s5 = float(scene.score()[0])
    check("ratchet retention: a ball placed between the flaps rolls back downhill "
          "(it moved) and is HELD by flap 1 — only the p1 latch fires (0.10), "
          "no success",
          xf < x5 - 0.010 and 0.175 < xf < 0.235
          and bool(scene._p1[0, 0]) and not bool(scene._p2[0, 0])
          and abs(s5 - c.w_p1) < 1e-3 and not bool(scene.success()[0]))

    # ================= 6. top-entry holds =========================================================
    torch.manual_seed(61)
    env.reset()
    _step(90)
    put_local("cargo_1", 0.100, 0.0, 0.220)   # above the open mouth stretch
    _step(300)
    _report("top-entry")
    x6 = float(scene._ball_loc()[0, 1, 0])
    check("top-entry holds: a ball dropped through the open mouth stretch lands "
          "upstream of flap 1 and rolls back out of the mouth — no latch, no credit",
          x6 < 0.05 and not bool(scene._p1[0, 1])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 7. roof holds ==============================================================
    put_local("cargo_0", 0.530, 0.0, c.wall_top + c.roof_t + c.ball_r + 0.050)
    _step(150)
    _report("roof-drop")
    z7 = float(scene._ball_loc()[0, 0, 2])
    check("roof holds: a ball dropped from above the basin lands ON the roof, "
          "never inside — no bin latch, no credit",
          z7 > c.wall_top and not bool(scene.balls_in_basin()[0, 0])
          and not bool(scene._bin[0, 0]) and not bool(scene.success()[0]))

    # ================= 8. decoy poisons the basin =================================================
    torch.manual_seed(71)
    env.reset()
    _step(90)
    put_in_basin("decoy_0", 0.575)     # decoy FIRST: no prefix may satisfy success
    _step(60)
    put_in_basin("cargo_0", 0.512)
    _step(150)
    _report("decoy-in")
    s8 = float(scene.score()[0])
    check("decoy poisons: the RED decoy settled in the basin (with one cargo) — "
          "the decoy latches nothing, score counts only the cargo (0.35), "
          "success refused",
          bool(scene.balls_in_basin()[0, 2]) and bool(scene.balls_in_basin()[0, 0])
          and 0.34 <= s8 <= 0.36 and not bool(scene._p1[0, 1])
          and not bool(scene.success()[0]))

    # ================= 9. missing delivery ========================================================
    torch.manual_seed(81)
    env.reset()
    _step(90)
    put_in_basin("cargo_1", 0.575)
    _step(150)
    _report("missing-one")
    s9 = float(scene.score()[0])
    check("missing delivery: one cargo settled in the basin, the other still in "
          "its bay — partial 0.35 only, no success",
          bool(scene.balls_in_basin()[0, 1]) and 0.34 <= s9 <= 0.36
          and not bool(scene.success()[0]))

    # ================= 10. latched credit =========================================================
    ground = scene.env_origins + torch.tensor([0.9, 0.9, c.ball_r],
                                              device=device).expand(n, 3)
    _write_body(scene.balls["cargo_1"], ground)
    _step(90)
    _report("latch")
    s10 = float(scene.score()[0])
    check("latched credit: the delivered cargo removed to open ground — the 0.35 "
          "latch survives, success does not",
          not bool(scene.balls_in_basin()[0, 1]) and 0.34 <= s10 <= 0.36
          and not bool(scene.success()[0]))

    # ================= 11. extraction holds =======================================================
    torch.manual_seed(91)
    env.reset()
    _step(90)
    put_in_basin("cargo_0", 0.575)
    _step(60)
    body = scene.balls["cargo_0"]
    # Quasi-static press (a constant 2.5 N shove would ram the ball at ~1.6 m/s and
    # VAULT the step ballistically — a static retention claim must be probed
    # statically): velocity-capped creep to the step, then the full 2.5 N press.
    min_x = 1.0
    for _ in range(480):
        vx = body.data.root_lin_vel_w[:, 0]
        fx = (6.0 * (-0.06 - vx)).clamp(-2.5, 0.0)
        fw = torch.stack([fx, torch.zeros_like(fx), torch.zeros_like(fx)], dim=-1)
        fb = quat_apply_inverse(body.data.root_quat_w, fw)
        body.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
        _step(1)
        min_x = min(min_x, float(scene._ball_loc()[0, 0, 0]))
    body.set_external_force_and_torque(zero, zero)
    _step(120)
    _report("extract")
    check("extraction holds: a sustained full-strength (2.5 N cap) downhill press "
          "pins the basin ball against the retaining step (it travelled to the "
          "step — non-vacuous) but cannot lift it out; still in the basin after "
          "release",
          min_x < 0.510 and min_x > 0.465
          and bool(scene.balls_in_basin()[0, 0]) and not bool(scene.success()[0]))

    # ================= 12-14. audit, no-NaN, video, verdict =======================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ratchet_ramp")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
