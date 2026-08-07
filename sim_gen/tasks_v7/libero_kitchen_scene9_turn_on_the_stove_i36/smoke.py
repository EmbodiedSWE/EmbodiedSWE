"""Smoke / rubric-REJECTION battery for GapFerryScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the chasm swallows an
unbridged crossing, the seated plank carries the rolling ball, seating is a real state
gravity produces — are load-bearing. Every probe is CONSTRUCTED (teleport, real
physics steps, judge) — instrumentation, never a solution: success() is monitored at
EVERY step and must never turn True anywhere in the battery (the audit is itself a
check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; plank racked (not bridged),
                           ball in the start dock; score 0, no success;
   2. randomization      — two seeded resets: READBACK assembly xy+yaw, plank xy+yaw
                           and ball xy all differ;
   3. rack side swap     — over 10 resets the rack (with the plank) spawns on BOTH
                           sides of the docks;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed family's naive transfer (actuate toward the goal,
                           build nothing): the ball is simply pushed toward the target
                           with NO bridge -> it falls into the chasm, unrecoverable,
                           score ~0 (the gap is load-bearing);
   6. bridge-only        — plank gravity-seated across the chasm, ball untouched ->
                           bridged, score 0.30, NO success (a bridge is not delivery);
   7. MECHANISM: ferry   — the ball ROLLS from the start deck onto the seated bridge
                           and is physically SUPPORTED over the void (on the plank,
                           over the chasm, at rolling height) — probe dismantled (ball
                           retrieved) before it could arrive;
   8. diving-board       — plank laid from the start deck toward the gap, far end
                           unsupported (cannot reach the far ledge from there) -> not
                           bridged;
   9. rails-down seat    — plank dropped over the chasm UPSIDE-DOWN -> not bridged
                           (attitude/height clauses);
  10. far-end park       — ball parked ON the seated plank's far end, over the far
                           ledge: close, but the delivery x window rejects it;
  11. ledge rest         — ball resting on the far dock's bare LEDGE (no bridge) ->
                           the height window rejects it (deck rest only);
  12. wrong terminus     — ball on the FLOOR beside the target dock -> rejected;
  13. settle gate        — the exact success position but the ball still moving at
                           ~0.45 m/s -> success refuses while anything moves (probe
                           dismantled before it can settle into a real success);
  14. latched credit     — the plank teleported off a made bridge: live bridged()
                           False, the latched 0.30 survives, still no success;
  15. rejection audit    — success() was never True at any step of this battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i36.smoke --headless
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

_qz, _qy, _qmul, _qapply = (task_scene._qz, task_scene._qy, task_scene._qmul,
                            task_scene._qapply)

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    pp = (scene.plank.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:16s} | ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
          f"{float(bp[2]):+.3f}) plank_z={float(pp[2]):+.3f} "
          f"up_z={float(scene.plank_up_z()[0]):+.3f} "
          f"bridged={bool(scene.bridged()[0])} on_plank={bool(scene.on_plank()[0])} "
          f"in_far={bool(scene.in_far_dock()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gap_ferry")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    seat_z = c.deck_h - c.rebate_drop + c.plank_t / 2

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.70, 0.60)) + o),
                                tuple(np.array((0.40, 0.00, 0.10)) + o),
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

    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def start_axis() -> torch.Tensor:
        """(3,) world unit direction of the start dock's +x (into the chasm)."""
        _refresh()
        u = _qapply(scene.start_dock.data.root_quat_w, ex)[0].clone()
        u[2] = 0.0
        return u / u.norm()

    def crossing_yaw() -> torch.Tensor:
        u = start_axis()
        return torch.atan2(u[1], u[0]).view(1)

    def dock_pt(dock, loc_xyz, z: float) -> torch.Tensor:
        """Dock-frame xy point at world height z (relative to env origin)."""
        _refresh()
        loc = torch.tensor([loc_xyz[0], loc_xyz[1], 0.0], device=device).expand(n, 3)
        p = dock.data.root_pos_w + _qapply(dock.data.root_quat_w, loc)
        p = p.clone()
        p[:, 2] = env.iscene.env_origins[:, 2] + z
        return p

    def chasm_centre(z: float) -> torch.Tensor:
        _refresh()
        p = scene.start_dock.data.root_pos_w + start_axis().expand(n, 3) * (c.gap / 2)
        p = p.clone()
        p[:, 2] = env.iscene.env_origins[:, 2] + z
        return p

    def seat_plank() -> None:
        """Gravity-seat the plank: free hover ~22 mm above seat height, aligned, drop."""
        _write_body(scene.plank, chasm_centre(seat_z + 0.022),
                    _qz(crossing_yaw().to(device)))
        _step(180)

    def park_ball_start() -> None:
        """Retrieve the ball to the start deck (probe dismantling, transport only)."""
        _write_body(scene.ball,
                    dock_pt(scene.start_dock, (-0.12, 0.0), c.deck_h + c.ball_r + 0.003),
                    None)

    def ball_x_start() -> float:
        _refresh()
        return float(scene._dock_local(scene.start_dock,
                                       scene.ball.data.root_pos_w)[0, 0])

    def ball_x_far() -> float:
        _refresh()
        return float(scene._dock_local(scene.far_dock,
                                       scene.ball.data.root_pos_w)[0, 0])

    def ball_z() -> float:
        _refresh()
        return float(scene.ball.data.root_pos_w[0, 2] - scene.env_origins[0, 2])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; plank racked (not bridged), ball in "
          "the start dock; score 0, no success",
          bool(scene._finite()[0]) and not bool(scene.bridged()[0])
          and not bool(scene.in_far_dock()[0]) and ball_x_start() < -c.rebate_d
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.start_dock.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.start_dock.data.root_quat_w[0]),
                scene.plank.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.plank.data.root_quat_w[0]),
                scene.ball.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_dp, a_dy, a_pp, a_py, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_dp, b_dy, b_pp, b_py, b_bp = readback()
    d_dp, d_pp = float((a_dp - b_dp).norm()), float((a_pp - b_pp).norm())
    d_bp = float((a_bp - b_bp).norm())
    d_dy, d_py = dyaw(a_dy, b_dy), dyaw(a_py, b_py)
    print(f"[smoke] randomization deltas: dock_xy={d_dp * 1000:.1f}mm "
          f"dock_yaw={d_dy:.1f}deg plank_xy={d_pp * 1000:.1f}mm "
          f"plank_yaw={d_py:.1f}deg ball_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: assembly xy+yaw, plank xy+yaw and ball xy readback "
          "differ across seeds",
          d_dp > 0.003 and d_dy > 1.0 and d_pp > 0.003 and d_py > 1.0 and d_bp > 0.003)

    # ================= 3. rack side swap ==========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+y" if float(scene.rack_side[0]) > 0 else "-y")
    print(f"[smoke] over 10 resets: rack sides {sorted(sides)}", flush=True)
    check("rack side swap: the rack (with the plank) spawns on BOTH sides of the "
          "docks over 10 resets", sides == {"+y", "-y"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED FAMILY'S OWN STRATEGY ================================
    # The seed actuates a mechanism the scene built (turn the knob). The naive
    # transfer here is to actuate straight toward the goal and BUILD NOTHING: push
    # the ball at the target dock with no bridge. The chasm takes it — the gap the
    # task is named for is physically load-bearing, and the naive plan scores ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    v = torch.zeros(n, 3, device=device)
    v[0] = start_axis() * 0.35
    _write_body(scene.ball, scene.ball.data.root_pos_w.clone(), None, lin_vel=v)
    _step(240)
    _report("seed-strategy")
    print(f"[smoke] naive push: ball z={ball_z() * 1000:.0f}mm "
          f"(chasm floor rest {(c.ball_r) * 1000:.0f}mm)", flush=True)
    check("negative (SEED strategy): the ball pushed toward the target with NO "
          "bridge falls into the chasm, unrecoverable — no success, score ~0",
          bool(scene.ball_in_chasm()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 6. bridge-only =============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    seat_plank()
    _report("bridge-only")
    check("bridge-only: gravity-seated plank bridges the chasm (score 0.30) but the "
          "ball is still in the start dock — NO success",
          bool(scene.bridged()[0]) and bool(scene.settled()[0])
          and abs(float(scene.score()[0]) - c.w_bridge) < 0.01
          and not bool(scene.success()[0]))

    # ================= 7. MECHANISM: the bridge carries the rolling ball ==========================
    # Continue on the made bridge: the ball rolls from the start deck onto the plank
    # and is physically SUPPORTED over the void — the mechanism the task is built on.
    # The probe is dismantled (ball retrieved) before it could arrive at the far dock.
    _refresh()
    ploc_y = float(scene._dock_local(scene.start_dock,
                                     scene.plank.data.root_pos_w)[0, 1])
    ploc_y = max(-0.03, min(0.03, ploc_y))
    v = torch.zeros(n, 3, device=device)
    v[0] = start_axis() * 0.40
    _write_body(scene.ball,
                dock_pt(scene.start_dock, (-(c.ball_r + 0.030), ploc_y),
                        c.deck_h + c.ball_r + 0.002),
                None, lin_vel=v)
    supported = False
    mid_z = 0.0
    for _i in range(240):
        _step(1)
        x = ball_x_start()
        if 0.040 < x < 0.100 and bool(scene.on_plank()[0]) and ball_z() > 0.150:
            supported = True
            mid_z = ball_z()
        if x > 0.100 or ball_z() < 0.10:
            break
    park_ball_start()  # dismantle before it can arrive
    _step(30)
    _report("mech-ferry")
    print(f"[smoke] ferry: ball supported over the void at z={mid_z * 1000:.0f}mm "
          f"(chasm floor {(c.ball_r) * 1000:.0f}mm, deck {c.deck_h * 1000:.0f}mm)",
          flush=True)
    check("MECHANISM (ferry): the ball rolled onto the seated bridge is SUPPORTED "
          "over the void (on the plank, over the chasm, rolling height) — probe "
          "dismantled before arrival",
          supported and bool(scene.bridged()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. diving-board plank ======================================================
    # From the start deck the plank cannot reach the far ledge (deck-to-ledge span
    # exceeds it): laid with its far end toward the gap it is a diving board, however
    # it settles — never a bridge.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.plank,
                dock_pt(scene.start_dock, (-0.050, 0.0), c.deck_h + c.plank_t / 2 + 0.002),
                _qz(crossing_yaw().to(device)))
    _step(180)
    _report("diving-board")
    check("diving-board: plank laid from the start deck toward the gap (far end "
          "unsupported, cannot reach the far ledge) — not bridged, no success",
          not bool(scene.bridged()[0]) and not bool(scene.success()[0]))

    # ================= 9. rails-down seat =========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    q_flip = _qmul(_qz(crossing_yaw().to(device)),
                   _qy(torch.full((1,), math.pi, device=device)))
    _write_body(scene.plank, chasm_centre(seat_z + 0.030), q_flip)
    _step(180)
    _report("rails-down")
    print(f"[smoke] rails-down: up_z={float(scene.plank_up_z()[0]):+.3f} "
          f"plank_z={(float(scene.plank.data.root_pos_w[0, 2] - scene.env_origins[0, 2])) * 1000:.1f}mm "
          f"(seat {seat_z * 1000:.1f}±{c.seat_z_tol * 1000:.0f}mm)", flush=True)
    check("rails-down: the plank dropped over the chasm UPSIDE-DOWN — attitude and "
          "height clauses refuse: not bridged, no success",
          not bool(scene.bridged()[0]) and not bool(scene.success()[0]))

    # ================= 10. far-end park ===========================================================
    # A correct bridge, and the ball parked ON its far end — over the far ledge, one
    # ball-width from delivery. The x window refuses: not delivered until it is fully
    # ON the deck, past the ledge.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_plank()
    assert bool(scene.bridged()[0]), "probe setup: bridge must be made"
    _refresh()
    axis = scene._plank_axis()[0].clone()
    axis[2] = 0.0
    axis = axis / axis.norm()
    s = 1.0 if float((axis * start_axis()).sum()) > 0 else -1.0
    p = scene.plank.data.root_pos_w + (s * axis).expand(n, 3) * 0.085
    p = p.clone()
    p[:, 2] = env.iscene.env_origins[:, 2] + c.deck_h + c.ball_r + 0.002
    _write_body(scene.ball, p, None)
    _step(90)
    _report("far-end-park")
    print(f"[smoke] far-end park: far-frame x={ball_x_far() * 1000:.0f}mm "
          f"(window {c.dock_x_lo * 1000:.0f}..{c.dock_x_hi * 1000:.0f}mm)", flush=True)
    check("far-end park: ball settled on the plank's far end over the far ledge — "
          "the delivery x window rejects it: no success, score <= 0.70",
          bool(scene.bridged()[0]) and bool(scene.on_plank()[0])
          and ball_x_far() > c.dock_x_hi and not bool(scene.in_far_dock()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.70)

    # ================= 14 (staged here): latched credit ===========================================
    # Continue from 10: carry the plank away (teleport = a perfect lift). The live
    # bridge is gone — the ball drops onto the bare ledge — but the latched `bridged`
    # credit survives.
    _refresh()
    rack_p = scene.rack.data.root_pos_w.clone()
    rack_p[:, 2] = env.iscene.env_origins[:, 2] + c.rack_h + c.plank_t / 2 + 0.003
    _write_body(scene.plank, rack_p, _qz(crossing_yaw().to(device)))
    _step(120)
    _report("unbridge")
    latched_ok = (not bool(scene.bridged()[0])
                  and float(scene.score()[0]) >= c.w_bridge - 1e-6
                  and not bool(scene.success()[0]))

    # ================= 11. ledge rest =============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.ball,
                dock_pt(scene.far_dock, (-0.015, 0.0),
                        c.deck_h - c.rebate_drop + c.ball_r + 0.003), None)
    _step(90)
    _report("ledge-rest")
    print(f"[smoke] ledge rest: ball z={ball_z() * 1000:.0f}mm "
          f"(success window {c.ball_z_lo * 1000:.0f}..{c.ball_z_hi * 1000:.0f}mm), "
          f"far-frame x={ball_x_far() * 1000:.0f}mm", flush=True)
    check("ledge rest: ball resting on the far dock's bare LEDGE — below the deck "
          "height window (and short of the x window): no success, score ~0",
          ball_z() < c.ball_z_lo and not bool(scene.in_far_dock()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 12. wrong terminus: floor beside the dock ==================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.ball,
                dock_pt(scene.far_dock, (-0.10, 0.25), c.ball_r + 0.003), None)
    _step(90)
    _report("floor-beside")
    check("wrong terminus: ball on the FLOOR beside the target dock — no window "
          "holds: no success, score ~0",
          not bool(scene.in_far_dock()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 13. settle gate ============================================================
    # The exact success position — ball centred on the far deck — but still moving at
    # ~0.45 m/s. success() must refuse while anything moves. The probe is dismantled
    # (ball retrieved) before friction could settle it into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    v_dir = _qapply(scene.far_dock.data.root_quat_w,
                    torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    v = torch.zeros(n, 3, device=device)
    v[0] = v_dir * 0.45
    _write_body(scene.ball,
                dock_pt(scene.far_dock, (-0.11, 0.0), c.deck_h + c.ball_r + 0.002),
                None, lin_vel=v)
    moving_ok = True
    for _ in range(5):
        _step(1)
        vb = float(scene.ball.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and vb > 0.1 and bool(scene.in_far_dock()[0]) \
            and not bool(scene.settled()[0]) and not bool(scene.success()[0])
    _report("settle-gate")
    park_ball_start()  # dismantle before it can settle
    _step(30)
    check("settle gate: the exact success position still moving at ~0.45 m/s is "
          "refused while anything moves", moving_ok)

    # ================= 14. latched credit (verdict from the staged probe) =========================
    check("latched credit: after the plank is carried off a made bridge, live "
          "bridged() is False but the latched 0.30 survives (still no success)",
          latched_ok)

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gap_ferry")
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
    except BaseException:  # noqa: BLE001 - Kit teardown hangs; die loudly and fast
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
