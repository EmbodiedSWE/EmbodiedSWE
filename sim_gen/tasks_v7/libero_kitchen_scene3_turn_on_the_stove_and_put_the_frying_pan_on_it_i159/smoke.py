"""Smoke / rubric-REJECTION battery for PiezoStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes, and that the claims the task rests on — the
recessed striker's reach denial, the kettle's uselessness, the click
threshold, the spring return, the placement windows and the settle gate — are
load-bearing. Every probe is CONSTRUCTED (teleport, real physics steps, judge)
— instrumentation, never a solution: no probe here reaches success() (audited
at every step).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; burner OFF, striker at the top
                          of its travel, pan upright on a slot, kettle on the other;
                          score 0, no success;
   2. randomization     — two seeded resets: READBACK burner xy, pan xy, pan yaw and
                          kettle xy all differ;
   3. slot coin         — over 10 resets the pan spawns on BOTH slots (flag matches
                          the pose readback);
   4. null-policy       — 240 idle steps: score ~0, no success, not lit, the striker
                          holds the top of its travel (no spring creep);
   5. SEED-NAIVE        — pan seated perfectly on the COLD plate (the seed-family
                          move: put the pan "on the stove" without turning anything
                          on) -> seat latch only, no success, score <= 0.151;
   6. FAT-OBJECT        — the kettle offered to the igniter in its BEST pose (lid
                          knob down, over the shaft): it settles perched on the
                          mouth — the knob dangles far above the striker, the
                          striker never moves past the click -> not lit, score ~0;
   7. shallow press     — the striker written 5 mm down its travel (on the joint
                          manifold, under the 8 mm click) and released: never
                          ignites AND the spring provably returns it to the top;
   8. wrong-object      — burner LIT (by the pan, physically), then the KETTLE
                          seated on the plate instead of the pan -> no success,
                          score <= 0.401;
   9. near-miss xy      — burner lit, pan genuinely seated but 55 mm off the plate
                          centre (outside the 40 mm radial window) -> refused;
  10. inverted pan      — burner lit, pan seated UPSIDE-DOWN on the plate centre ->
                          the upright/z clauses refuse, no success;
  11. settle gate       — burner lit, the exact success pose written WITH 0.4 m/s
                          velocity -> success() refuses the moving state
                          (dismantled immediately);
  12. MECHANISM         — the solve's ignition from a fresh reset: pan teleported
                          handle-down over the shaft and RELEASED — the pan's own
                          weight drives the striker past the click by CONTACT (the
                          actuator provably moved; non-vacuous) -> lit, score 0.40,
                          no success (no pan on the plate);
  13. rejection audit   — success() was never True at ANY step of the battery;
  14. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i159.smoke --headless
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

_qz, _qy = task_scene._qz, task_scene._qy

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
    print(f"[smoke] {tag:18s} | depth={float(scene.striker_depth()[0]) * 1000:+.1f}mm "
          f"lit={bool(scene._lit[0])} "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):.3f}) "
          f"on_burner={bool(scene.pan_on_burner()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.piezo_stove")().build(num_envs=args.num_envs, device=device)
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
        return float(scene.striker_depth()[0])

    oz = float(env.iscene.env_origins[0, 2])
    yaw_s = _qz(torch.full((n,), -math.pi / 2, device=device))  # handle SOUTH
    park_xy = torch.tensor([[0.32, -0.24]], device=device).expand(n, 2)

    def tower_w() -> torch.Tensor:
        _refresh()
        return scene.tower.data.root_pos_w[0]

    def burner_xy() -> torch.Tensor:
        _refresh()
        return scene.burner.data.root_pos_w[:, :2].clone()

    def plate_top() -> float:
        _refresh()
        return float(scene.burner.data.root_pos_w[0, 2]) + c.plinth_h + c.plate_h

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

    def park_pan() -> None:
        """Carry the pan to the free front-right corner (clear of the tower, the
        kettle's slot band and the burner at any jitter — cfg margins)."""
        put(scene.pan, park_xy, oz + c.pan_spawn_z, yaw_s, drop=0.010)
        _step(30)

    def press_pose():
        """The solve's carry pose: handle-down over the shaft, tip 15 mm above
        the cap (qy(+90) sends local +x to world -z; the handle axis sits
        +handle_z in world x after the rotation, compensated at the origin)."""
        t = tower_w()
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = t[0] - c.handle_z
        pos[:, 1] = t[1]
        cap_top = float(scene.striker.data.root_pos_w[0, 2]) + c.cap_t / 2
        pos[:, 2] = cap_top + 0.015 + c.handle_tip_reach
        return pos, _qy(torch.full((n,), math.pi / 2, device=device))

    def make_lit(tag: str) -> tuple[bool, float]:
        """PHYSICAL ignition (the solve's move): drop the pan handle-down into
        the shaft, let its weight click the striker, then park the pan away.
        Returns (lit, max observed press depth) — the depth proves the striker
        really moved (non-vacuous)."""
        max_d, lit = 0.0, False
        for _attempt in range(2):
            pos, quat = press_pose()
            _write_state(scene.pan, pos, quat)
            for _ in range(360):
                _step(1)
                max_d = max(max_d, depth())
                if bool(scene._lit[0]):
                    lit = True
                    break
            if lit:
                break
        park_pan()
        print(f"[smoke] {tag}: make_lit -> lit={lit} max_depth={max_d * 1000:.1f}mm "
              f"depth_now={depth() * 1000:.1f}mm", flush=True)
        return lit, max_d

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    _refresh()
    px = float(scene.pan.data.root_pos_w[0, 0] - scene.env_origins[0, 0])
    kx = float(scene.kettle.data.root_pos_w[0, 0] - scene.env_origins[0, 0])
    fin = bool(torch.isfinite(scene.pan.data.root_pos_w).all()
               and torch.isfinite(scene.striker.data.root_pos_w).all()
               and torch.isfinite(scene.kettle.data.root_pos_w).all())
    check("settle/no-NaN: burner OFF, striker at the top of its travel, pan upright "
          "on a slot, kettle on the other; score 0, no success",
          fin and depth() < 0.002 and not bool(scene._lit[0])
          and float(scene._up_w(scene.pan)[0, 2]) > 0.95
          and abs(abs(px) - c.slot_x) < c.slot_x_jitter + 0.02
          and px * kx < 0.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.burner.data.root_pos_w[0, :2].clone(),
                scene.pan.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pan.data.root_quat_w[0]),
                scene.kettle.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_b, a_p, a_y, a_k = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_b, b_p, b_y, b_k = readback()
    d_bur = float((a_b - b_b).norm())
    d_pan, d_yw, d_ket = float((a_p - b_p).norm()), dyaw(a_y, b_y), float((a_k - b_k).norm())
    print(f"[smoke] randomization deltas: burner_xy={d_bur * 1000:.1f}mm "
          f"pan_xy={d_pan * 1000:.1f}mm pan_yaw={d_yw:.1f}deg "
          f"kettle_xy={d_ket * 1000:.1f}mm", flush=True)
    check("randomization-is-real: burner xy, pan xy, pan yaw, kettle xy readback differ",
          d_bur > 0.005 and d_pan > 0.005 and d_yw > 3.0 and d_ket > 0.005)

    # ================= 3. slot coin ===============================================================
    sides = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        px = float(scene.pan.data.root_pos_w[0, 0] - scene.env_origins[0, 0])
        assert (px > 0) == bool(scene.pan_east[0]), "slot flag vs pan pose mismatch"
        sides.append(1 if px > 0 else -1)
    print(f"[smoke] pan slot sides over 10 resets: {sides}", flush=True)
    check("slot coin: the pan spawns on BOTH slots over 10 resets (flag matches pose)",
          (1 in sides) and (-1 in sides))

    # ================= 4. null policy fails (and nothing creeps) ==================================
    torch.manual_seed(100)
    env.reset()
    _step(10)
    d0 = depth()
    _step(230)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, not lit, the "
          "striker holds the top of its travel (no spring creep)",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and not bool(scene._lit[0]) and depth() < 0.002 and abs(depth() - d0) < 0.002)

    # ================= 5. SEED-NAIVE: pan on the COLD plate =======================================
    # The seed-family move — put the pan "on the stove" without turning anything
    # on. The placement is PERFECT; only the burner is off.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pan_onto_plate()
    _report("seed-naive")
    check("SEED-NAIVE: pan seated perfectly on the COLD plate — seat latch only, "
          "burner off: no success, score <= 0.151",
          bool(scene.pan_on_burner()[0]) and not bool(scene._lit[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.151)
    _REC["on"] = False

    # ================= 6. FAT-OBJECT: the kettle cannot press the striker =========================
    # The kettle's BEST case: lid knob straight down the shaft axis. The body is
    # wider than the shaft, so it perches on the mouth; the 24 mm knob dangles
    # ~62 mm above the striker. Non-vacuous: the kettle really settles ON the
    # mouth (height readback) and the striker depth is audited every substep.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    t = tower_w()
    kxy = torch.stack([t[0].expand(n), t[1].expand(n)], dim=-1)
    q_inv = torch.zeros(n, 4, device=device)
    q_inv[:, 1] = 1.0  # qx(pi): knob down
    put(scene.kettle, kxy, float(t[2]) + c.tower_h + c.knob_h + c.kettle_h / 2,
        q_inv, drop=0.006)
    max_d = 0.0
    for _ in range(180):
        _step(1)
        max_d = max(max_d, depth())
    _report("fat-object")
    kz = float(scene.kettle.data.root_pos_w[0, 2]) - oz
    # the 24 mm knob slips INTO the 48 mm shaft; the 70 mm body face rests on
    # the mouth rim (its knob dangles ~62 mm above the striker cap)
    perch = c.deck_h + c.tower_h + c.kettle_h / 2
    print(f"[smoke] fat-object: kettle z={kz:.3f} (perch {perch:.3f}) "
          f"max_depth={max_d * 1000:.2f}mm", flush=True)
    check("FAT-OBJECT: the kettle knob-down on the shaft perches on the mouth — the "
          "knob never reaches the striker (max depth ~0), not lit, score ~0",
          abs(kz - perch) < 0.010 and max_d < c.press_depth
          and not bool(scene._lit[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. shallow press + spring return ===========================================
    # Striker written 5 mm down its travel (ON the joint manifold, within the
    # limits) and released: 5 mm < the 8 mm click, so it must NOT ignite — and
    # the spring must provably return it to the top of the travel.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    t = tower_w()
    spos = torch.zeros(n, 3, device=device)
    spos[:, 0], spos[:, 1] = t[0], t[1]
    spos[:, 2] = oz + c.striker_rest_w - 0.005
    _write_state(scene.striker, spos)
    d_wr = depth()
    _step(60)
    _report("shallow-press")
    check("shallow press: striker released 5 mm down (under the 8 mm click) never "
          "ignites and the spring returns it to the top",
          0.003 < d_wr < 0.007 and not bool(scene._lit[0]) and depth() < 0.002
          and float(scene.score()[0]) <= 0.01)

    # ================= 8. wrong object on the plate ===============================================
    # Burner genuinely lit (the pan's press), then the KETTLE seated on the
    # plate instead of the pan.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    lit8, _ = make_lit("wrong-object")
    put(scene.kettle, burner_xy(), plate_top() + c.kettle_h / 2, None, drop=0.004)
    _step(120)
    _report("wrong-object")
    kz = float(scene.kettle.data.root_pos_w[0, 2]) - oz
    check("wrong-object: burner lit but the KETTLE seated on the plate — the pan "
          "clauses refuse: no success, score <= 0.401",
          lit8 and abs(kz - (c.plinth_h + c.plate_h + c.kettle_h / 2 + c.deck_h)) < 0.010
          and not bool(scene.pan_on_burner()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.401)
    _REC["on"] = False

    # ================= 9. near-miss xy ============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lit9, _ = make_lit("near-miss")
    pan_onto_plate(dy=-0.055)
    _report("near-miss")
    _refresh()
    d_xy = float((scene.pan.data.root_pos_w[0, :2]
                  - scene.burner.data.root_pos_w[0, :2]).norm())
    print(f"[smoke] near-miss: pan radial offset {d_xy * 1000:.1f}mm "
          f"(tol {c.pan_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss xy: burner lit, pan genuinely resting 55 mm off the plate "
          "centre — outside the radial window: not seated, no success, score <= 0.401",
          lit9 and d_xy > c.pan_xy_tol + 0.005 and not bool(scene.pan_on_burner()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.401)

    # ================= 10. inverted pan ===========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lit10, _ = make_lit("inverted")
    q_flip = torch.zeros(n, 4, device=device)
    q_flip[:, 1] = 1.0  # qx(pi): dish mouth down
    pan_onto_plate(quat=q_flip, z_extra=c.pan_wall_h + c.pan_base_t)
    _report("inverted")
    up_z = float(scene._up_w(scene.pan)[0, 2])
    check("inverted pan: burner lit, pan UPSIDE-DOWN on the plate centre — the "
          "upright/z clauses refuse: no success, score <= 0.401",
          lit10 and up_z < -0.90 and not bool(scene.pan_on_burner()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.401)

    # ================= 11. settle gate: the success pose in motion is refused =====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lit11, _ = make_lit("settle-gate")
    put(scene.pan, burner_xy(), plate_top() + c.pan_bottom_dz + 0.002, yaw_s,
        drop=0.0, vel_x=0.40)
    _step(1)
    _report("settle-gate")
    moving_refused = bool(scene.pan_on_burner()[0]) and not bool(scene.success()[0]) \
        and float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]) > c.settle_speed
    # dismantle IMMEDIATELY (before the slide damps into a genuine success)
    park_pan()
    check("settle gate: the exact success pose moving at 0.4 m/s is refused "
          "(geometry alone is not success; dismantled before it could calm)",
          lit11 and moving_refused and not bool(scene.success()[0]))

    # ================= 12. MECHANISM: ignition really is contact-made =============================
    # The solve's move from a FRESH reset, fully instrumented: teleport the pan
    # handle-down over the shaft and release. The pan's weight must drive the
    # striker past the click purely by handle-tip contact (the striker is never
    # written), and the spring must return it after the pan is carried away.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    lit12, max_d = make_lit("mechanism")
    _report("mechanism")
    check("MECHANISM: the released pan's weight clicks the striker by contact "
          "(actuator provably moved past the click depth), the spring returns it, "
          "lit latches — score 0.40, still no success",
          lit12 and max_d > c.press_depth + 0.002 and depth() < 0.002
          and bool(scene._lit[0])
          and 0.39 <= float(scene.score()[0]) <= 0.401
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 13. rejection audit ========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.piezo_stove")
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
