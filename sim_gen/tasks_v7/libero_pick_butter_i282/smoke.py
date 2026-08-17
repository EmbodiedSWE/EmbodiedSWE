"""Smoke / rubric-REJECTION battery for ButterDispenserScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct trajectory with monotone
latched credit). This battery proves the rubric REJECTS wrong outcomes and that the two
physical claims the task rests on are load-bearing: the stage-before-dispense ORDER
(an early dispense grounds the block; hand repair can never earn `_delivered`) and the
EXACTLY-ONE retention (the front wall holds every block but the bottom one, even under
an aggressive shove). Probes are constructed settled states — instrumentation, never a
solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; all blocks in the tower, blade
                           home, basket off the mat; score 0, no success;
   2. randomization      — two seeded resets: READBACK basket xy / yaw and block yaw
                           all differ;
   3. count+side coverage— across 10 resets both block counts {2,3} and both basket
                           spawn sides are drawn (readback);
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — stage the basket, then hand-drop a block INTO it from above
                           (the seed's grasp-carry-release): geometric containment is
                           real, but no outlet credential, no delivery -> score stays
                           at the staging credit, no success;
   6. ORDER part 1       — dispense with the basket UNSTAGED (real blade push, full
                           stroke): block lands on the bare ground, `_dispensed`
                           latches but `_delivered` does NOT; score <= actuated credit;
   7. ORDER part 2       — repair attempt after 6: stage the basket, hand-drop the
                           grounded block in. Credentialed containment, still no
                           `_delivered` -> success False forever (early dispense is
                           irreversible in the rubric, matching the physical claim);
   8. EXACTLY-ONE        — 3-block episode, unstaged, AGGRESSIVE push (2x force, 3x
                           speed cap) through the full stroke + retract: exactly one
                           block leaves the tower, the rest stay magazined;
   9. credential honesty — a block dropped from above the outlet lip bounces off the
                           front wall / lip and can NEVER fire the outlet-transit
                           credential (the wall top is 140 mm above the slot window);
  10. staged near-misses — settled 55 mm off the mat center rejected; tilted 25 deg
                           rejected; hovering 60 mm above the mat rejected;
  11. latch persistence  — staging credit (0.20) survives the basket being knocked far
                           off the mat afterwards; still no success;
  12. overfill           — stage, hand-preload a block into the basket (no credential
                           -> no prefix of this probe is ever success), then dispense
                           legitimately: delivery latches but a second block occupies
                           the basket -> exactly-one fails, success False, score 0.70;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_butter_i282.smoke --headless
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    ib = scene.in_basket()[0] & scene.present[0]
    print(f"[smoke] {tag:20s} | disp={float(scene.blade_disp()[0]) * 1000:6.1f}mm "
          f"staged={bool(scene._staged[0])} act={bool(scene._actuated[0])} "
          f"deliv={bool(scene._delivered[0])} disp_lat={scene._dispensed[0].tolist()} "
          f"in_basket={int(ib.sum())} in_tower="
          f"{int((scene.in_tower()[0] & scene.present[0]).sum())} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _stage_basket() -> bool:
    """Legit staging: free-space teleport above the mat, real gravity set-down."""
    scene = _ENV.scene
    c = scene.cfg
    cxy = scene._catch_xy()[0]
    p = torch.zeros(_ENV.num_envs, 3, device=_ENV.device)
    p[:, 0] = cxy[0]
    p[:, 1] = cxy[1]
    p[:, 2] = scene.env_origins[0, 2] + c.mat_t + 0.020
    _write_body(scene.basket, p)
    return _settle_until(lambda: bool(scene.staged_now()[0]), max_steps=360)


def _drive_blade(target_disp: float, force: float, vmax: float, iters: int) -> float:
    """Velocity-regulated x-force on the blade until it passes target_disp; returns
    the max displacement reached (anti-vacuity readback for physical probes)."""
    scene = _ENV.scene
    sgn = -1.0 if force < 0 else 1.0
    d_max = -1.0
    for _ in range(iters):
        _refresh()
        disp = float(scene.blade_disp()[0])
        d_max = max(d_max, disp)
        if (sgn < 0 and disp >= target_disp) or (sgn > 0 and disp <= target_disp):
            break
        vx = float(scene.blade.data.root_lin_vel_w[0, 0])
        moving = (vx < -vmax) if sgn < 0 else (vx > vmax)
        scene.blade_force[:] = 0.0 if moving else force
        _step(2)
    scene.blade_force[:] = 0.0
    return d_max


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_dispenser")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.75)) + o),
                                tuple(np.array((0.20, 0.00, 0.20)) + o),
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

    def top_idx() -> int:
        return int(scene.present[0].sum()) - 1

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    states = torch.cat([b.data.root_state_w for b in
                        (scene.tower, scene.blade, scene.basket, scene.mat, *scene.blocks)],
                       dim=-1)
    n_pres = int(scene.present[0].sum())
    in_twr = int((scene.in_tower()[0] & scene.present[0]).sum())
    check("settle/no-NaN: layout settles finite; every present block in the tower, "
          "blade home, score 0, no success",
          bool(torch.isfinite(states).all()) and in_twr == n_pres
          and abs(float(scene.blade_disp()[0])) < 0.01
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.basket.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.basket.data.root_quat_w[0]),
                yaw_of(scene.blocks[0].data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_bp, a_by, a_ky = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_bp, b_by, b_ky = readback()
    d_bp = float((a_bp - b_bp).norm())
    d_by, d_ky = dyaw(a_by, b_by), dyaw(a_ky, b_ky)
    print(f"[smoke] randomization deltas: basket_xy={d_bp * 1000:.1f}mm "
          f"basket_yaw={d_by:.1f}deg block0_yaw={d_ky:.2f}deg", flush=True)
    check("randomization-is-real: basket xy and yaw readbacks differ",
          d_bp > 0.01 and d_by > 3.0)

    # ================= 3. count + side + block-yaw coverage =======================================
    counts, sides, kyaws = set(), set(), []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        counts.add(int(scene.present[0].sum()))
        sides.add(bool(scene.basket.data.root_pos_w[0, 1]
                       - scene.env_origins[0, 1] > 0))
        kyaws.append(yaw_of(scene.blocks[0].data.root_quat_w[0]))
    ky_spread = max(kyaws) - min(kyaws)
    print(f"[smoke] over 10 resets: counts={sorted(counts)} sides={sides} "
          f"block0_yaw_spread={ky_spread:.2f}deg", flush=True)
    check("count+side coverage: both block counts {2,3}, both basket sides, and a real "
          "block-yaw spread drawn over 10 resets",
          counts == {2, 3} and sides == {True, False} and ky_spread > 0.5)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: hand-drop into the staged basket =========================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    ok_stage = _stage_basket()
    ti = top_idx()
    bp = scene.basket.data.root_pos_w[0].clone()
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = bp[0]
    p[:, 1] = bp[1]
    p[:, 2] = bp[2] + c.basket_wall_h + c.block_h / 2 + 0.03
    _write_body(scene.blocks[ti], p)
    _settle_until(lambda: bool(scene.blocks_settled()[0, ti]), max_steps=240)
    _report("seed-strategy")
    ib = scene.in_basket()[0] & scene.present[0]
    check("negative (SEED strategy): block hand-dropped into the staged basket IS "
          "geometrically contained, but no outlet credential and no delivery -> "
          "score stays at the staging credit, no success",
          ok_stage and int(ib.sum()) == 1 and not bool(scene._dispensed[0].any())
          and not bool(scene._delivered[0]) and float(scene.score()[0]) <= 0.20 + 1e-4
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. ORDER part 1: unstaged dispense forfeits the block ======================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    d_max = _drive_blade(0.97 * c.stroke, force=-10.0, vmax=0.20, iters=500)
    _settle_until(lambda: bool(scene.blocks_settled()[0, 0]), max_steps=300)
    _drive_blade(0.005, force=8.0, vmax=0.12, iters=500)
    _step(30)
    _report("unstaged-dispense")
    _refresh()
    b0z = float(scene.blocks[0].data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"[smoke] unstaged dispense: d_max={d_max * 1000:.1f}mm block0_z={b0z:.3f}",
          flush=True)
    check("ORDER 1 (unstaged dispense): full real stroke (readback), block0 credentialed "
          "out the outlet but lands on bare ground -> delivered False, score <= actuated "
          "credit, no success",
          d_max >= 0.9 * c.stroke and bool(scene._dispensed[0, 0])
          and not bool(scene._delivered[0]) and b0z < 0.08
          and float(scene.score()[0]) <= 0.35 + 1e-4 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. ORDER part 2: late staging + hand repair still fails ====================
    # (continues from 6 — the grounded block is moved aside, the basket is staged, and
    # the credentialed block is hand-dropped in: everything a repair could earn, except
    # `_delivered`, which only latches DURING a staged transit.)
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = scene.env_origins[0, 0] - 0.30
    p[:, 1] = scene.env_origins[0, 1] + 0.40
    p[:, 2] = c.block_h / 2 + 0.004
    _write_body(scene.blocks[0], p)
    _step(30)
    ok_stage = _stage_basket()
    bp = scene.basket.data.root_pos_w[0].clone()
    p[:, 0] = bp[0]
    p[:, 1] = bp[1]
    p[:, 2] = bp[2] + c.basket_wall_h + c.block_h / 2 + 0.03
    _write_body(scene.blocks[0], p)
    _settle_until(lambda: bool(scene.blocks_settled()[0, 0]), max_steps=240)
    _report("late-stage-repair")
    ib = scene.in_basket()[0] & scene.present[0]
    check("ORDER 2 (late repair): staged afterwards + credentialed block hand-dropped "
          "in -> containment + credential BOTH hold, but delivered stays False: "
          "success False, score <= 0.35 (early dispense is irreversible)",
          ok_stage and int(ib.sum()) == 1 and bool((ib & scene._dispensed[0]).any())
          and not bool(scene._delivered[0]) and float(scene.score()[0]) <= 0.35 + 1e-4
          and not bool(scene.success()[0]))

    # ================= 8. EXACTLY-ONE: aggressive push frees only the bottom block ================
    found = None
    for s in range(12):
        torch.manual_seed(800 + s)
        env.reset()
        _refresh()
        if int(scene.present[0].sum()) == 3:
            found = s
            break
    _step(40)
    _REC["on"] = True
    d_max = _drive_blade(0.97 * c.stroke, force=-20.0, vmax=0.60, iters=500)
    _settle_until(lambda: bool(scene.blocks_settled()[0].all()), max_steps=300)
    _drive_blade(0.005, force=8.0, vmax=0.12, iters=500)
    _step(60)
    _report("aggressive-push")
    _refresh()
    in_twr = int((scene.in_tower()[0] & scene.present[0]).sum())
    print(f"[smoke] aggressive push (seed {800 + (found or 0)}): d_max={d_max * 1000:.1f}mm "
          f"in_tower={in_twr}/3 dispensed={scene._dispensed[0].tolist()}", flush=True)
    check("EXACTLY-ONE: 3-block episode, 2x-force 3x-speed full stroke (readback) -> "
          "only the bottom block leaves; both riding blocks retained by the front wall",
          found is not None and d_max >= 0.9 * c.stroke and in_twr == 2
          and bool(scene._dispensed[0, 0]) and not bool(scene._dispensed[0, 1:].any())
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. credential honesty: over-wall drop can't fake a transit =================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    twr = scene._tower_xy()[0]
    ti = top_idx()
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = twr[0] + c.lip_x
    p[:, 1] = twr[1]
    p[:, 2] = c.top_z + 0.06
    _write_body(scene.blocks[ti], p)
    _settle_until(lambda: bool(scene.blocks_settled()[0, ti]), max_steps=300)
    _report("over-wall-drop")
    check("credential honesty: block dropped from above the outlet lip deflects off the "
          "front wall and NEVER fires the outlet-transit credential",
          not bool(scene._dispensed[0, ti]) and not bool(scene._delivered[0])
          and not bool(scene.success()[0]))

    # ================= 10. staged-gate near-misses ================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    cxy = scene._catch_xy()[0]
    # (a) settled, 55 mm off the mat center (gate 30 mm)
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = cxy[0]
    p[:, 1] = cxy[1] + c.catch_tol + 0.025
    p[:, 2] = scene.env_origins[0, 2] + c.mat_t + 0.020
    _write_body(scene.basket, p)
    _settle_until(lambda: bool(scene.basket.data.root_lin_vel_w[0].norm() < 0.02),
                  max_steps=240)
    off_rej = not bool(scene.staged_now()[0])
    # (b) tilted 25 deg at the mat center (instantaneous gate probe)
    q = torch.zeros(env.num_envs, 4, device=env.device)
    half = math.radians(25.0) / 2
    q[:, 0], q[:, 1] = math.cos(half), math.sin(half)
    p[:, 1] = cxy[1]
    p[:, 2] = scene.env_origins[0, 2] + c.mat_t + 0.060
    _write_body(scene.basket, p, q)
    _step(1)
    _refresh()
    tilt_rej = not bool(scene.staged_now()[0])
    # (c) hovering 60 mm above the mat, upright (instantaneous gate probe)
    _write_body(scene.basket, p)
    _step(1)
    _refresh()
    hover_rej = not bool(scene.staged_now()[0])
    print(f"[smoke] staged near-misses: off-center rej={off_rej} tilt rej={tilt_rej} "
          f"hover rej={hover_rej}", flush=True)
    check("staged-gate near-misses: 55 mm off-center settled, 25 deg tilted, and "
          "60 mm hover all rejected",
          off_rej and tilt_rej and hover_rej)

    # ================= 11. latch persistence ======================================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    ok_stage = _stage_basket()
    s_staged = float(scene.score()[0])
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = scene.env_origins[0, 0] + c.basket_spawn[0]
    p[:, 1] = scene.env_origins[0, 1] - c.basket_spawn[1]
    p[:, 2] = 0.02
    _write_body(scene.basket, p)
    _step(90)
    _report("knocked-off")
    check("latch persistence: staging credit (0.20) survives the basket being knocked "
          "off the mat; staged_now False again, still no success",
          ok_stage and abs(s_staged - 0.20) < 1e-4 and not bool(scene.staged_now()[0])
          and abs(float(scene.score()[0]) - 0.20) < 1e-4
          and not bool(scene.success()[0]))

    # ================= 12. overfilled basket rejected =============================================
    # Ordered so NO prefix satisfies the goal: the hand-added block goes in FIRST (no
    # credential -> never success), THEN a legit staged dispense adds the second block.
    # Final state: delivery latched and one block credentialed, but the basket holds a
    # second block -> the exactly-one clause rejects it.
    found2 = None
    for s in range(12):
        torch.manual_seed(900 + s)
        env.reset()
        _refresh()
        if int(scene.present[0].sum()) == 2:
            found2 = s
            break
    _step(40)
    _REC["on"] = True
    ok_stage = _stage_basket()
    bp = scene.basket.data.root_pos_w[0].clone()
    p = torch.zeros(env.num_envs, 3, device=env.device)
    p[:, 0] = bp[0]
    p[:, 1] = bp[1]
    p[:, 2] = bp[2] + c.basket_wall_h + c.block_h / 2 + 0.03
    _write_body(scene.blocks[1], p)  # hand-preload the TOP block into the basket
    _settle_until(lambda: bool(scene.blocks_settled()[0, 1]), max_steps=240)
    pre_success = bool(scene.success()[0])
    d_max = _drive_blade(0.97 * c.stroke, force=-10.0, vmax=0.20, iters=500)
    _settle_until(lambda: bool(scene.blocks_settled()[0].all()), max_steps=360)
    _drive_blade(0.005, force=8.0, vmax=0.12, iters=500)
    _step(30)
    _report("overfilled")
    _refresh()
    ib = scene.in_basket()[0] & scene.present[0]
    print(f"[smoke] overfill (seed {900 + (found2 or 0)}): d_max={d_max * 1000:.1f}mm "
          f"in_basket={int(ib.sum())} dispensed={scene._dispensed[0].tolist()} "
          f"pre_success={pre_success}", flush=True)
    check("overfill: basket pre-loaded by hand, then a legit staged dispense (readback) "
          "-> delivery latched but a second block occupies the basket: exactly-one "
          "fails, success False, score capped at 0.70",
          found2 is not None and ok_stage and not pre_success
          and d_max >= 0.9 * c.stroke and bool(scene._dispensed[0, 0])
          and bool(scene._delivered[0]) and int(ib.sum()) >= 1
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.70 + 1e-4)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.butter_dispenser")
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
