"""Smoke / rubric-REJECTION battery for MilkShuntYardScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct push-routed trajectory
with monotone latched credit). This battery proves the rubric REJECTS wrong outcomes
and that the physical claims the task rests on are load-bearing: the APERTURE-TRANSIT
CREDENTIAL (only a fall through the green hole counts — hand placement into the basket,
even from as high as fits under the deck, never earns it), the TERMINAL trap, the
blocker-in-basket hazard, and the siding's refusal of the crate. Probes are constructed
states / instrumented pushes — never a solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; crate seated in the bay,
                           blocker at the junction, basket under the GREEN aperture,
                           score 0, no success;
   2. randomization      — 3 seeded resets: max-pairwise yard xy / yard yaw / crate
                           yaw readback deltas all real (GPU 2-seed collision guard);
   3. side coverage      — across 10 resets both goal sides are drawn and the yard
                           yaw readback shows a real spread;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — crate teleported into the cell just under the aperture
                           plane (the highest a "hand" could lower it) and dropped
                           into the basket: geometric containment + outcome_ok are
                           REAL, but the CoM never crosses the aperture plane moving
                           down -> no credential, no success, score 0;
   6. TRAP terminality   — crate dropped through the RED aperture: `_trapped` latches,
                           the crate rests in the enclosed trap cell, no goal
                           credential, no success;
   7. blocker hazard     — blocker dropped through the GREEN aperture into the basket,
                           then the crate legitimately falls through after it: the
                           crate's credential fires, but the blocker occupies the
                           basket -> success False, score capped;
   8. siding refusal     — physical probe: blocker parked aside, the CRATE is pushed
                           straight north at the siding mouth with 2x solve force: it
                           engages the mouth (anti-vacuity readback) and jams far
                           short of the shunt line (100 mm crate vs 94 mm pocket);
   9. near-miss gates    — crate hovering above the basket interior ceiling rejected
                           by in_basket; settled containment accepted by outcome_ok;
                           basket tilted 25 deg rejected by outcome_ok;
  10. latch persistence  — commit credit (0.20) latches on the goal half and survives
                           the crate being carried back to the bay; still no success;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_milk_i359.smoke --headless
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


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    _step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
    if pred():
        return True
    waited = poll
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _yard_yaw() -> float:
    q = _ENV.scene.yard.data.root_quat_w[0]
    return 2.0 * math.atan2(float(q[3]), float(q[0]))


def _write_local(body, lx: float, ly: float, lz: float, lyaw: float = 0.0) -> None:
    """Teleport `body` (env 0 broadcast) to a yard-local pose, zero velocity."""
    env = _ENV
    scene = env.scene
    yp = scene.yard.data.root_pos_w[0]
    yaw = _yard_yaw()
    cy, sy = math.cos(yaw), math.sin(yaw)
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = float(yp[0]) + cy * lx - sy * ly
    st[:, 1] = float(yp[1]) + sy * lx + cy * ly
    st[:, 2] = float(scene.env_origins[0, 2]) + lz
    st[:, 3] = math.cos((yaw + lyaw) / 2)
    st[:, 6] = math.sin((yaw + lyaw) / 2)
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene._to_yard(scene.crate.data.root_pos_w)[0]
    b = scene._to_yard(scene.blocker.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | crate=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
          f"blocker=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
          f"shunt={bool(scene._shunted[0])} commit={bool(scene._committed[0])} "
          f"deliv={bool(scene._delivered[0])} trap={bool(scene._trapped[0])} "
          f"in_basket={bool(scene.in_basket(scene.crate)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push_crate_north(force: float, vmax: float, iters: int) -> float:
    """Velocity-regulated +y (yard frame) CoM push on the crate, x P-steered to 0.
    Returns the max yard-local y reached (anti-vacuity readback)."""
    scene = _ENV.scene
    yaw = _yard_yaw()
    cy, sy = math.cos(yaw), math.sin(yaw)
    y_max = -1.0
    for _ in range(iters):
        _refresh()
        p = scene._to_yard(scene.crate.data.root_pos_w)[0]
        y_max = max(y_max, float(p[1]))
        dx = max(-0.5, min(0.5, 25.0 * (0.0 - float(p[0]))))
        nrm = math.hypot(dx, 1.0)
        wx = (cy * dx - sy * 1.0) / nrm
        wy = (sy * dx + cy * 1.0) / nrm
        v = scene.crate.data.root_lin_vel_w[0]
        along = float(v[0]) * wx + float(v[1]) * wy
        f = 0.0 if along > vmax else force
        scene.crate_force[0, 0] = f * wx
        scene.crate_force[0, 1] = f * wy
        _step(2)
    scene.crate_force[:] = 0.0
    _refresh()
    return y_max


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.milk_shunt_yard")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.15)) + o),
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

    def crate_local():
        _refresh()
        return scene._to_yard(scene.crate.data.root_pos_w)[0]

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    states = torch.cat([b.data.root_state_w for b in
                        (scene.yard, scene.crate, scene.blocker, scene.basket,
                         scene.green_tab, scene.red_tab)], dim=-1)
    p = crate_local()
    b = scene._to_yard(scene.blocker.data.root_pos_w)[0]
    bk = scene._to_yard(scene.basket.data.root_pos_w)[0]
    gt = scene._to_yard(scene.green_tab.data.root_pos_w)[0]
    gs = float(scene.goal_side[0])
    check("settle/no-NaN: layout settles finite; crate in the bay, blocker at the "
          "junction, basket under the GREEN aperture, score 0, no success",
          bool(torch.isfinite(states).all())
          and abs(float(p[1]) - c.crate_spawn_y) < 0.03 and float(p[2]) > c.deck_z
          and abs(float(b[1]) - c.blocker_spawn_y) < 0.03 and float(b[2]) > c.deck_z
          and abs(float(bk[0]) * gs - c.hole_cx) < 0.02
          and float(gt[0]) * gs > 0.0
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ====================
    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        _refresh()
        cq = scene.crate.data.root_quat_w[0]
        obs.append((scene.yard.data.root_pos_w[0, :2].clone(),
                    math.degrees(_yard_yaw()),
                    math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))))
    d_xy = max(float((a[0] - bb[0]).norm())
               for i, a in enumerate(obs) for bb in obs[i + 1:])
    d_yaw = max(abs(a[1] - bb[1]) for i, a in enumerate(obs) for bb in obs[i + 1:])
    d_cy = max(abs(a[2] - bb[2]) for i, a in enumerate(obs) for bb in obs[i + 1:])
    print(f"[smoke] randomization max-pairwise deltas: yard_xy={d_xy * 1000:.1f}mm "
          f"yard_yaw={d_yaw:.2f}deg crate_yaw={d_cy:.2f}deg", flush=True)
    check("randomization-is-real: yard xy / yard yaw / crate yaw readbacks differ "
          "across 3 seeds",
          d_xy > 0.005 and d_yaw > 1.0 and d_cy > 0.5)

    # ================= 3. goal-side coverage ======================================================
    sides, yaws = set(), []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add(float(scene.goal_side[0]) > 0)
        yaws.append(math.degrees(_yard_yaw()))
    yaw_spread = max(yaws) - min(yaws)
    print(f"[smoke] over 10 resets: sides={sides} yard_yaw_spread={yaw_spread:.2f}deg",
          flush=True)
    check("side coverage: both goal sides drawn over 10 resets and a real yard-yaw "
          "spread",
          sides == {True, False} and yaw_spread > 3.0)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: hand placement into the basket ===========================
    # The crate is lowered (teleport, zero velocity) to the HIGHEST pose a hand could
    # reach inside the cell — CoM just under the aperture plane band — and dropped into
    # the basket. Containment and outcome_ok are genuinely satisfied, but the CoM never
    # crosses the aperture plane moving down, so the credential never fires.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    gs = float(scene.goal_side[0])
    _write_local(scene.crate, gs * c.hole_cx, c.hole_cy, 0.145)
    ok_settle = _settle_until(
        lambda: bool(scene.in_basket(scene.crate)[0]) and bool(scene.settled(scene.crate)[0]),
        max_steps=360)
    _report("seed-strategy")
    check("negative (SEED strategy): crate hand-lowered into the cell and dropped into "
          "the basket IS contained (outcome_ok True) but earns no aperture credential "
          "-> no success, score 0",
          ok_settle and bool(scene.outcome_ok()[0]) and not bool(scene._delivered[0])
          and float(scene.score()[0]) <= 1e-4 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. TRAP terminality ========================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    gs = float(scene.goal_side[0])
    _write_local(scene.crate, -gs * c.hole_cx, c.hole_cy, c.deck_z + 0.10)
    ok_settle = _settle_until(
        lambda: bool(scene.in_trap()[0]) and bool(scene.settled(scene.crate)[0]),
        max_steps=360)
    _report("trap-drop")
    check("TRAP terminality: crate falls through the RED aperture -> `_trapped` "
          "latches, crate rests in the enclosed trap cell, no goal credential, no "
          "success",
          ok_settle and bool(scene._trapped[0]) and not bool(scene._delivered[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. blocker-in-basket hazard ================================================
    # Ordered so NO prefix satisfies the goal: the blocker occupies the basket FIRST,
    # then the crate falls through the green aperture with a genuine credential.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    gs = float(scene.goal_side[0])
    _write_local(scene.blocker, gs * c.hole_cx, c.hole_cy, c.deck_z + 0.10)
    ok_b = _settle_until(
        lambda: bool(scene.in_basket(scene.blocker)[0]) and bool(scene.settled(scene.blocker)[0]),
        max_steps=360)
    _write_local(scene.crate, gs * c.hole_cx, c.hole_cy, c.deck_z + 0.12)
    _settle_until(lambda: bool(scene.settled(scene.crate)[0]), max_steps=360)
    _report("blocker-hazard")
    check("blocker hazard: blocker dropped into the basket first, crate credentialed "
          "through the green aperture after it -> delivery latched but the blocker "
          "occupies the basket: success False, score capped",
          ok_b and bool(scene._delivered[0]) and bool(scene.in_basket(scene.blocker)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.65 + 1e-4)
    _REC["on"] = False

    # ================= 8. siding refuses the crate (physical probe) ===============================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    # park the blocker on the ground, far off the yard
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = scene.env_origins[0, 0] - 0.30
    st[:, 1] = scene.env_origins[0, 1] + 0.40
    st[:, 2] = c.blocker_s / 2 + 0.004
    st[:, 3] = 1.0
    scene.blocker.write_root_state_to_sim(st, _all_ids())
    _refresh()
    # seat the crate at the junction and shove it straight north at the siding mouth
    _write_local(scene.crate, 0.0, c.blocker_spawn_y, c.deck_z + c.crate_s / 2 + 0.002)
    _step(20)
    y0 = float(crate_local()[1])
    y_max = _push_crate_north(force=6.0, vmax=0.08, iters=400)
    _report("siding-refusal")
    print(f"[smoke] siding refusal: y0={y0:+.3f} y_max={y_max:+.3f} "
          f"(shunt line {c.shunt_y_min:+.3f})", flush=True)
    check("siding refusal: crate shoved north at the siding with 2x solve force "
          "engages the mouth (moved >= 15 mm, readback) but jams far short of the "
          "shunt line (100 mm crate vs 94 mm pocket)",
          y_max > y0 + 0.015 and y_max < c.shunt_y_min - 0.02
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. near-miss gates =========================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    gs = float(scene.goal_side[0])
    # (a) hovering above the basket interior ceiling: rejected by in_basket
    bkw = scene.basket.data.root_pos_w[0].clone()
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0] = bkw[0]
    st[:, 1] = bkw[1]
    st[:, 2] = bkw[2] + 0.130
    st[:, 3] = 1.0
    scene.crate.write_root_state_to_sim(st, _all_ids())
    _step(1)
    _refresh()
    hover_rej = not bool(scene.in_basket(scene.crate)[0])
    # (b) let it settle in: outcome_ok True (containment gates are satisfiable)
    ok_in = _settle_until(lambda: bool(scene.outcome_ok()[0]), max_steps=360)
    # (c) tilt the basket 25 deg (> 15 deg gate): outcome_ok False instantly
    q = torch.zeros(env.num_envs, 4, device=env.device)
    half = math.radians(25.0) / 2
    q[:, 0], q[:, 1] = math.cos(half), math.sin(half)
    st = torch.zeros(env.num_envs, 13, device=env.device)
    st[:, 0:3] = scene.basket.data.root_pos_w[0]
    st[:, 3:7] = q
    scene.basket.write_root_state_to_sim(st, _all_ids())
    _step(1)
    _refresh()
    tilt_rej = not bool(scene.outcome_ok()[0])
    print(f"[smoke] near-misses: hover rej={hover_rej} settled ok={ok_in} "
          f"tilt rej={tilt_rej}", flush=True)
    check("near-miss gates: crate hovering above the basket ceiling rejected; settled "
          "containment accepted by outcome_ok; 25 deg tilted basket rejected",
          hover_rej and ok_in and tilt_rej)

    # ================= 10. latch persistence + regression =========================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    gs = float(scene.goal_side[0])
    _write_local(scene.crate, gs * 0.115, -0.020, c.deck_z + c.crate_s / 2 + 0.002)
    _step(10)
    _refresh()
    committed = bool(scene._committed[0])
    s_commit = float(scene.score()[0])
    # regress BEYOND the latched metric: carry the crate all the way back to the bay
    _write_local(scene.crate, 0.0, c.crate_spawn_y, c.deck_z + c.crate_s / 2 + 0.002)
    _step(60)
    _report("regressed")
    check("latch persistence: commit credit (0.20) latches on the goal half and "
          "survives the crate being carried back to the bay; committed_now False "
          "again, still no success",
          committed and abs(s_commit - 0.20) < 1e-4
          and not bool(scene.committed_now()[0])
          and abs(float(scene.score()[0]) - 0.20) < 1e-4
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.milk_shunt_yard")
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

    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
