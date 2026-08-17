"""Smoke / rubric-REJECTION battery for DeadmanStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the claims the task rests on — the
dead-man's LIVE (non-latching) gas state, the mass-based selection between the
ballast and the identical-shape foam decoy, the streak gate against momentary
presses, the spring return, the placement windows and the settle gate — are
load-bearing. Every probe is CONSTRUCTED (teleport, real physics steps, judge)
— instrumentation, never a solution: no probe here reaches success() (audited
at every step).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; gas SHUT, pedal at the top of
                          its travel, pan/ballast/decoy each on its dealt slot
                          (permutation readback), pan upright; score 0, no success;
   2. randomization     — two seeded resets: READBACK burner xy, pan xy, pan yaw,
                          ballast xy and decoy xy all differ;
   3. slot permutation  — over 10 resets the pan AND the ballast each visit >= 2
                          distinct slots, and every object's pose readback matches
                          its obj_slot flag;
   4. null-policy       — 240 idle steps: score ~0, no success, not lit, the pedal
                          holds the top of its travel (no spring creep);
   5. SEED-NAIVE        — pan seated perfectly on the COLD plate (the seed-family
                          move: put the pan "on the stove" without turning anything
                          on) -> seat latch only, no success, score <= 0.151;
   6. DECOY-IS-TOO-LIGHT— the full wrong-object plan: the foam decoy parked squarely
                          ON the pedal (it really rests there — z readback) and the
                          pan seated on the plate. The spring's preload beats the
                          decoy: pedal depth audited every substep never reaches the
                          gas threshold, never lit -> no success, score <= 0.151;
   7. DEAD-MAN          — the signature check: ballast parked (gas genuinely held,
                          lit), then the ballast REMOVED — the spring shuts the
                          valve: lit() dies within a second (NOTHING latches), the
                          gas *credit* stays, score stays at the partial cap; then
                          the pan seated on the plate anyway -> STILL no success
                          (the flame is out), score <= 0.5502;
   8. PAN-AS-BALLAST    — cross-arrangement: the PAN parked on the pedal. Its 500 g
                          really opens the gas (depth readback past the threshold,
                          lit) — but the pan cannot also be on the burner: no
                          success; and removing it kills the flame again;
   9. near-miss xy      — gas held by the ballast, pan genuinely resting 65 mm off
                          the plate centre (outside the 45 mm radial window) ->
                          refused, no success;
  10. inverted pan      — gas held, pan seated UPSIDE-DOWN on the plate centre ->
                          the upright/z clauses refuse, no success;
  11. settle gate       — gas held, the exact success pose written WITH 0.4 m/s
                          velocity -> success() refuses the moving state
                          (dismantled immediately);
  12. shallow dip       — the pedal written 8 mm down its travel (on the joint
                          manifold, under the 15 mm threshold) and released: never
                          lit AND the spring provably returns it to the top;
  13. PRESS-AND-RELEASE — the "hold it down yourself and let go" exploit: the pedal
                          written to FULL travel (past the threshold — depth
                          readback proves it) with nothing on it -> the spring
                          pops it back up in a couple of substeps, the 30-substep
                          streak never completes, NEVER lit;
  14. MECHANISM         — the solve's move from a fresh reset: ballast carried high
                          (carry latch) and released over the pedal — its weight
                          drives the pedal past the threshold BY CONTACT (depth
                          readback; the pedal is never written) and HOLDS it: lit
                          turns on and stays -> score 0.40, no success (no pan);
  15. rejection audit   — success() was never True at ANY step of the battery;
  16. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i278.smoke --headless
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
import traceback  # noqa: E402

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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"hit": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["hit"] = _AUDIT["hit"] or bool(env.scene.success().any())
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


def _write_state(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                 vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    pp = scene.pan.data.root_pos_w[0]
    print(f"[smoke] {tag:18s} | depth={float(scene.pedal_depth()[0]) * 1000:+.1f}mm "
          f"streak={int(scene._streak[0])} lit={bool(scene.lit()[0])} "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):.3f}) "
          f"on_burner={bool(scene.pan_on_burner()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.deadman_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.95, 0.85)) + o),
                                tuple(np.array((0.05, 0.00, 0.22)) + o),
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

    def depth() -> float:
        _refresh()
        return float(scene.pedal_depth()[0])

    oz = float(env.iscene.env_origins[0, 2])
    yaw_s = _qz(torch.full((n,), -math.pi / 2, device=device))  # handle SOUTH
    slot_x = [-c.slot_dx, 0.0, c.slot_dx]
    ped_top = oz + c.pedal_top_rest_w  # world z of the pedal top face at rest
    ped_xy = torch.tensor([[c.pedal_x, c.pedal_y]], device=device).expand(n, 2) \
        + env.iscene.env_origins[:, :2]
    _HOME: dict[str, torch.Tensor] = {}

    def fresh(seed: int = 100) -> None:
        """Deterministic reset + settle; snapshot every movable's rest state so
        probes can send bodies HOME (their own dealt slot — guaranteed clear by
        the cfg slot asserts; avoids probe-debris blocking the next probe)."""
        torch.manual_seed(seed)
        env.reset()
        _step(30)
        _refresh()
        for name in ("pan", "ballast", "decoy"):
            _HOME[name] = getattr(scene, name).data.root_state_w[:].clone()

    def send_home(name: str) -> None:
        getattr(scene, name).write_root_state_to_sim(_HOME[name], _all_ids())
        _refresh()

    def burner_xy() -> torch.Tensor:
        _refresh()
        return scene.burner.data.root_pos_w[:, :2].clone()

    def plate_top() -> float:
        _refresh()
        return float(scene.burner.data.root_pos_w[0, 2]) + c.plate_top_dz

    def put(body, xy_w: torch.Tensor, z_center: float, quat=None,
            drop: float = 0.004, vel_x: float = 0.0) -> None:
        """CONSTRUCT: write the body just above its rest pose (probe
        instrumentation; the caller settles when the arrangement is complete)."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0:2] = xy_w
        pos[:, 2] = z_center + drop
        _write_state(body, pos, quat, vel_x)

    def pan_onto_plate(dx: float = 0.0, dy: float = 0.0, quat=None,
                       z_extra: float = 0.0, drop: float = 0.004) -> None:
        xy = burner_xy()
        xy[:, 0] += dx
        xy[:, 1] += dy
        put(scene.pan, xy, plate_top() + c.pan_bottom_dz + z_extra,
            yaw_s if quat is None else quat, drop=drop)
        _step(120)

    def park_on_pedal(body, half_h: float) -> None:
        """Level hover, body bottom 10 mm above the pedal top, dead centre."""
        put(body, ped_xy, ped_top + half_h, drop=0.010)

    def make_gas(tag: str) -> tuple[bool, float]:
        """PHYSICAL gas-open (the solve's move): carry the ballast through a
        high waypoint (latches the carry credit deterministically) then release
        it over the pedal; its weight must press the pedal past the threshold
        BY CONTACT and hold it (the pedal is never written). Returns
        (lit, max observed depth) — the depth proves the pedal really moved."""
        max_d, lit = 0.0, False
        for _attempt in range(2):
            wp = torch.zeros(n, 3, device=device)
            wp[:, 0:2] = ped_xy
            wp[:, 1] -= 0.10
            wp[:, 2] = oz + c.deck_h + 0.25
            _write_state(scene.ballast, wp)
            _step(1)
            park_on_pedal(scene.ballast, c.block_h / 2)
            for _ in range(240):
                _step(1)
                max_d = max(max_d, depth())
                if bool(scene.lit()[0]):
                    lit = True
                    break
            if lit:
                break
        print(f"[smoke] {tag}: make_gas -> lit={lit} max_depth={max_d * 1000:.1f}mm "
              f"depth_now={depth() * 1000:.1f}mm", flush=True)
        return lit, max_d

    def obj_on_slot(name: str, j: int) -> bool:
        _refresh()
        x = float(getattr(scene, name).data.root_pos_w[0, 0] - scene.env_origins[0, 0])
        want = slot_x[int(scene.obj_slot[0, j])]
        return abs(x - want) < c.slot_x_jitter + 0.02

    # ================= 1. settle / no-NaN =========================================================
    fresh(100)
    _REC["on"] = True
    _step(30)
    _report("settle")
    _REC["on"] = False
    _refresh()
    fin = bool(torch.isfinite(scene.pan.data.root_pos_w).all()
               and torch.isfinite(scene.pedal.data.root_pos_w).all()
               and torch.isfinite(scene.ballast.data.root_pos_w).all()
               and torch.isfinite(scene.decoy.data.root_pos_w).all())
    check("settle/no-NaN: gas SHUT, pedal at the top of its travel, pan/ballast/decoy "
          "each on its dealt slot, pan upright; score 0, no success",
          fin and depth() < 0.002 and not bool(scene.lit()[0])
          and float(scene._up_w(scene.pan)[0, 2]) > 0.95
          and obj_on_slot("pan", 0) and obj_on_slot("ballast", 1)
          and obj_on_slot("decoy", 2)
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.burner.data.root_pos_w[0, :2].clone(),
                scene.pan.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pan.data.root_quat_w[0]),
                scene.ballast.data.root_pos_w[0, :2].clone(),
                scene.decoy.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_b, a_p, a_y, a_bl, a_d = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_b, b_p, b_y, b_bl, b_d = readback()
    d_bur = float((a_b - b_b).norm())
    d_pan, d_yw = float((a_p - b_p).norm()), dyaw(a_y, b_y)
    d_bal, d_dec = float((a_bl - b_bl).norm()), float((a_d - b_d).norm())
    print(f"[smoke] randomization deltas: burner_xy={d_bur * 1000:.1f}mm "
          f"pan_xy={d_pan * 1000:.1f}mm pan_yaw={d_yw:.1f}deg "
          f"ballast_xy={d_bal * 1000:.1f}mm decoy_xy={d_dec * 1000:.1f}mm", flush=True)
    check("randomization-is-real: burner xy, pan xy, pan yaw, ballast xy, decoy xy "
          "readback differ across seeds",
          d_bur > 0.005 and d_pan > 0.005 and d_yw > 3.0
          and d_bal > 0.005 and d_dec > 0.005)

    # ================= 3. slot permutation ========================================================
    pan_slots, bal_slots, match = [], [], True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        match = match and obj_on_slot("pan", 0) and obj_on_slot("ballast", 1) \
            and obj_on_slot("decoy", 2)
        pan_slots.append(int(scene.obj_slot[0, 0]))
        bal_slots.append(int(scene.obj_slot[0, 1]))
    print(f"[smoke] pan slots over 10 resets: {pan_slots}; ballast slots: {bal_slots}",
          flush=True)
    check("slot permutation: pan AND ballast each visit >= 2 distinct slots over 10 "
          "resets, and every pose readback matches its obj_slot flag",
          match and len(set(pan_slots)) >= 2 and len(set(bal_slots)) >= 2)

    # ================= 4. null policy fails (and nothing creeps) ==================================
    fresh(100)
    d0 = depth()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, not lit, the "
          "pedal holds the top of its travel (no spring creep)",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and not bool(scene.lit()[0]) and depth() < 0.002 and abs(depth() - d0) < 0.002)

    # ================= 5. SEED-NAIVE: pan on the COLD plate =======================================
    # The seed-family move — put the pan "on the stove" without turning anything
    # on. The placement is PERFECT; only the gas is shut.
    fresh(100)
    _REC["on"] = True
    pan_onto_plate()
    _report("seed-naive")
    check("SEED-NAIVE: pan seated perfectly on the COLD plate — seat latch only, "
          "gas shut: no success, score <= 0.151",
          bool(scene.pan_on_burner()[0]) and not bool(scene.lit()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.151)
    _REC["on"] = False

    # ================= 6. DECOY-IS-TOO-LIGHT ======================================================
    # The full wrong-object plan, best case: the foam decoy parked SQUARELY on
    # the pedal and the pan seated on the plate. Non-vacuous: the decoy really
    # rests ON the pedal (z readback) and the pedal depth is audited every
    # substep — the spring's preload alone beats pedal+decoy.
    fresh(100)
    _REC["on"] = True
    park_on_pedal(scene.decoy, c.block_h / 2)
    max_d = 0.0
    for _ in range(180):
        _step(1)
        max_d = max(max_d, depth())
    pan_onto_plate()
    for _ in range(60):
        _step(1)
        max_d = max(max_d, depth())
    _report("decoy-park")
    dz = float(scene.decoy.data.root_pos_w[0, 2])
    rest = ped_top + c.block_h / 2  # pedal NOT depressed under the decoy
    print(f"[smoke] decoy: z={dz:.3f} (rest-on-pedal {rest:.3f}) "
          f"max_depth={max_d * 1000:.2f}mm", flush=True)
    check("DECOY-IS-TOO-LIGHT: foam decoy parked on the pedal + pan on the plate — "
          "the pedal (audited every substep) never reaches the gas threshold, never "
          "lit: no success, score <= 0.151",
          abs(dz - rest) < 0.010 and max_d < c.gas_on_depth
          and not bool(scene.lit()[0]) and bool(scene.pan_on_burner()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.151)
    _REC["on"] = False

    # ================= 7. DEAD-MAN: remove the ballast and the flame dies =========================
    # The signature of the task. Gas genuinely held (ballast parked, lit), then
    # the ballast is REMOVED: the spring shuts the valve — lit() dies within a
    # second (NOTHING latches), while the gas *credit* latch stays. Seating the
    # pan afterwards STILL is not success: the flame is out.
    fresh(100)
    _REC["on"] = True
    lit7, _ = make_gas("dead-man")
    _step(60)
    held = bool(scene.lit()[0]) and depth() >= c.gas_on_depth
    send_home("ballast")
    _step(120)  # a second of hands-off: the spring must shut the valve
    _report("dead-man-released")
    died = not bool(scene.lit()[0]) and depth() < 0.002 and int(scene._streak[0]) == 0
    credit_kept = bool(scene._gas[0]) and 0.39 <= float(scene.score()[0]) <= 0.401
    pan_onto_plate()
    _report("dead-man-pan-late")
    check("DEAD-MAN: gas held by the ballast, then the ballast removed — lit() dies "
          "(the spring shuts the valve; nothing latches), the gas CREDIT stays; "
          "seating the pan afterwards is STILL not success, score <= 0.5502",
          lit7 and held and died and credit_kept
          and bool(scene.pan_on_burner()[0]) and not bool(scene.lit()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.5502)
    _REC["on"] = False

    # ================= 8. PAN-AS-BALLAST: the cross-arrangement ===================================
    # The pan's 500 g really CAN hold the gas open (mass, not identity, is the
    # mechanism — depth readback proves the press) — but then the pan is on the
    # pedal, not the burner: no success. Removing it kills the flame again.
    fresh(100)
    park_on_pedal(scene.pan, c.pan_bottom_dz)
    lit8, max_d8 = False, 0.0
    for _ in range(240):
        _step(1)
        max_d8 = max(max_d8, depth())
        if bool(scene.lit()[0]):
            lit8 = True
            break
    _step(60)
    _report("pan-as-ballast")
    held8 = bool(scene.lit()[0]) and depth() >= c.gas_on_depth
    on_burner8 = bool(scene.pan_on_burner()[0])
    no_succ8 = not bool(scene.success()[0])
    send_home("pan")
    _step(120)
    print(f"[smoke] pan-as-ballast: max_depth={max_d8 * 1000:.1f}mm; after removal "
          f"lit={bool(scene.lit()[0])} depth={depth() * 1000:.1f}mm", flush=True)
    check("PAN-AS-BALLAST: the pan parked on the pedal really opens the gas (depth "
          "readback past the threshold, lit) but is then not on the burner — no "
          "success; removing it shuts the valve again",
          lit8 and max_d8 >= c.gas_on_depth and held8 and not on_burner8 and no_succ8
          and not bool(scene.lit()[0]) and depth() < 0.002)

    # ================= 9. near-miss xy ============================================================
    fresh(100)
    lit9, _ = make_gas("near-miss")
    pan_onto_plate(dy=-0.065)
    _report("near-miss")
    _refresh()
    d_xy = float((scene.pan.data.root_pos_w[0, :2]
                  - scene.burner.data.root_pos_w[0, :2]).norm())
    print(f"[smoke] near-miss: pan radial offset {d_xy * 1000:.1f}mm "
          f"(tol {c.pan_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss xy: gas held live, pan genuinely resting 65 mm off the plate "
          "centre — outside the radial window: not seated, no success, score <= 0.401",
          lit9 and d_xy > c.pan_xy_tol + 0.005 and not bool(scene.pan_on_burner()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.401)

    # ================= 10. inverted pan ===========================================================
    fresh(100)
    lit10, _ = make_gas("inverted")
    q_flip = torch.zeros(n, 4, device=device)
    q_flip[:, 1] = 1.0  # qx(pi): dish mouth down
    pan_onto_plate(quat=q_flip, z_extra=c.pan_wall_h + c.pan_base_t)
    _report("inverted")
    up_z = float(scene._up_w(scene.pan)[0, 2])
    check("inverted pan: gas held live, pan UPSIDE-DOWN on the plate centre — the "
          "upright/z clauses refuse: no success, score <= 0.401",
          lit10 and up_z < -0.90 and not bool(scene.pan_on_burner()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.401)

    # ================= 11. settle gate: the success pose in motion is refused =====================
    fresh(100)
    lit11, _ = make_gas("settle-gate")
    put(scene.pan, burner_xy(), plate_top() + c.pan_bottom_dz + 0.002, yaw_s,
        drop=0.0, vel_x=0.40)
    _step(1)
    _report("settle-gate")
    moving_refused = bool(scene.pan_on_burner()[0]) and not bool(scene.success()[0]) \
        and float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]) > c.settle_speed
    # dismantle IMMEDIATELY (before the slide damps into a genuine success)
    send_home("pan")
    _step(30)
    check("settle gate: the exact success pose moving at 0.4 m/s is refused "
          "(geometry alone is not success; dismantled before it could calm)",
          lit11 and moving_refused and not bool(scene.success()[0]))

    # ================= 12. shallow dip + spring return ============================================
    # Pedal written 8 mm down its travel (ON the joint manifold, within the
    # limits) and released: 8 mm < the 15 mm threshold, so it must NOT open the
    # gas — and the spring must provably return it to the top.
    fresh(100)
    spos = torch.zeros(n, 3, device=device)
    spos[:, 0:2] = ped_xy
    spos[:, 2] = oz + c.pedal_rest_w - 0.008
    _write_state(scene.pedal, spos)
    d_wr = depth()
    _step(60)
    _report("shallow-dip")
    check("shallow dip: pedal released 8 mm down (under the 15 mm threshold) never "
          "opens the gas and the spring returns it to the top",
          0.006 < d_wr < 0.010 and not bool(scene.lit()[0]) and depth() < 0.002
          and float(scene.score()[0]) <= 0.01)

    # ================= 13. PRESS-AND-RELEASE: the streak gate =====================================
    # "Hold the pedal down yourself and let go": the pedal written to FULL
    # travel (depth readback proves it crossed the threshold) with NOTHING on
    # it. The spring pops it back above the threshold within a couple of
    # substeps, so the 30-substep dead-man streak never completes: NEVER lit.
    fresh(100)
    spos = torch.zeros(n, 3, device=device)
    spos[:, 0:2] = ped_xy
    spos[:, 2] = oz + c.pedal_rest_w - c.travel
    _write_state(scene.pedal, spos)
    d_wr = depth()
    max_streak = 0
    ever_lit = False
    for _ in range(120):
        _step(1)
        max_streak = max(max_streak, int(scene._streak[0]))
        ever_lit = ever_lit or bool(scene.lit()[0])
    _report("press-release")
    print(f"[smoke] press-and-release: written depth={d_wr * 1000:.1f}mm "
          f"max_streak={max_streak} (gate {c.gas_streak})", flush=True)
    check("PRESS-AND-RELEASE: pedal written to FULL travel (past the threshold) with "
          "nothing on it — the spring returns it before the 30-substep streak "
          "completes: never lit, score ~0",
          d_wr > c.gas_on_depth and max_streak < c.gas_streak and not ever_lit
          and depth() < 0.002 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 14. MECHANISM: the gas-open really is contact-made =========================
    # The solve's move from a FRESH reset, fully instrumented: the ballast is
    # carried high (carry latch) and released over the pedal. Its weight must
    # drive the pedal past the threshold purely by contact (the pedal is never
    # written) and HOLD it there: lit turns on and STAYS.
    fresh(100)
    _REC["on"] = True
    lit14, max_d14 = make_gas("mechanism")
    _step(60)
    _report("mechanism")
    check("MECHANISM: the released ballast's weight presses the pedal past the "
          "threshold by contact (depth readback) and HOLDS it — lit on and stable, "
          "carry+gas credit = score 0.40, still no success",
          lit14 and max_d14 > c.gas_on_depth + 0.002
          and bool(scene.lit()[0]) and depth() >= c.gas_on_depth
          and bool(scene._carry[0]) and bool(scene._gas[0])
          and 0.39 <= float(scene.score()[0]) <= 0.401
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.deadman_stove")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
