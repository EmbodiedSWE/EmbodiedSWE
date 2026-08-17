"""Smoke / rubric-REJECTION battery for MoveClockScene — NullRobot, teleported probes.

solve.py is the acceptance proof (released contact seating, then a real hinge press
through the gravity-bistable rocker). This battery proves the rubric REJECTS wrong
outcomes and that the physical claims the task rests on are load-bearing: the
rocker really is a held toggle (a weak press cannot flip it), an ILLEGAL punch
(clock pressed before the move) forfeits the episode even though the FINAL STATE is
pixel-identical to the success state, and the seat/upright clauses refuse
near-misses. Every probe is CONSTRUCTED as a settled state (teleport / bounded
press torque, real physics steps, judge) — instrumentation, never a solution: no
probe here reaches success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite; queen upright on the half
                          opposite the promotion square, rocker held on its start
                          stop; score 0, no success;
   2. randomization     — 3 seeded resets: READBACK board yaw, board xy, the
                          promotion-square spot and the queen spot all differ (max
                          pairwise), and the frame body sits where the scene's
                          frame_xy target says;
   3. side coverage     — over 10 resets both rocker start sides occur, and the
                          rocker angle readback follows the drawn side every time;
   4. null-policy       — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY     — the seed's plan (placement only, no clock, no order):
                          queen dropped perfectly into the promotion square, clock
                          never touched -> seat credit only (0.45), no success;
   6. ILLEGAL PUNCH     — the solve's own press executed BEFORE the move (beam
                          verifiably flips: live actuator), THEN the queen seated
                          perfectly: the end state equals the success end state
                          (seated + flipped + settled) yet the foul latch holds —
                          no success, no flip credit, score <= 0.45 forever;
   7. near-miss seat    — queen standing upright 8.5 cm OFF the square's centre
                          (outside the >= 5.1 cm out-of-frame bound): no seat
                          latch, score ~0;
   8. near-miss pose    — queen LYING DOWN at the square's centre: inside the
                          square but not standing -> no seat latch, score ~0;
   9. latches-vs-live   — legal full solve, then the queen knocked off the square:
                          both latches hold (score == 0.75 cap) yet success
                          (judged live) refuses;
  10. TOGGLE HOLDS      — a weak press (0.004 N*m, ~0.4x the gravity holding
                          torque; same actuator path check 6 proved live) for 2.5 s
                          does NOT move the beam off its stop: no foul, score 0 —
                          the bistable mechanism is physics, not fiat;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_chess_i317.smoke --headless
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

_qmul, _qconj, _qx = task_scene._qmul, task_scene._qconj, task_scene._qx

# Global watchdog (daemon): if anything wedges, die loudly before the forge timeout.
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
    ql = scene.board_local(scene.queen.data.root_pos_w)
    ang = math.degrees(float(scene.rocker_angle()[0]))
    print(f"[smoke] {tag:16s} | queen_board=({float(ql[0, 0]):+.3f},{float(ql[0, 1]):+.3f},"
          f"{float(ql[0, 2]):+.3f}) rocker={ang:+.1f}deg "
          f"seat={bool(scene._l_seat[0])} flip={bool(scene._l_flip[0])} "
          f"foul={bool(scene._foul[0])} seated={bool(scene.seated()[0])} "
          f"flipped={bool(scene.flipped()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.move_clock")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.80, 0.75)) + o),
                                tuple(np.array((0.18, 0.0, 0.12)) + o),
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

    def place_board(body, x, y, z, quat_extra: torch.Tensor | None = None) -> None:
        """Teleport `body` to board-local (x, y, z) with the board's heading
        (optionally composed with an extra local rotation)."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = x
        loc[:, 1] = y
        loc[:, 2] = z
        q = scene.board.data.root_quat_w
        if quat_extra is not None:
            q = _qmul(q, quat_extra)
        _write_body(body, scene.board_to_world(loc), q)

    def press(tau: float, max_steps: int, stop_fn=None) -> bool:
        """The solve's own press: body-frame hinge torque toward the far side with
        the bang-bang ~0.8 rad/s speed governor; wrench cleared after. Returns True
        if `stop_fn` fired (the actuator provably moved the beam)."""
        zero = torch.zeros(n, 1, 3, device=device)
        torque = torch.zeros(n, 1, 3, device=device)
        w_cap = 0.8
        fired = False
        for _ in range(max_steps):
            if stop_fn is not None and stop_fn():
                fired = True
                break
            w_h = scene.rocker.data.root_ang_vel_w[:, 0]
            prog = (-scene.start_side) * w_h
            on = (prog < w_cap).float()
            torque[:, 0, 0] = (-scene.start_side) * tau * on
            scene.rocker.set_external_force_and_torque(zero, torque, env_ids=_all_ids())
            _step(1)
        scene.rocker.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        return fired

    def seat_queen() -> None:
        """The solve's own P1 move: release the queen 20 mm above the square's
        centre; the seating is a real contact settle."""
        _refresh()
        place_board(scene.queen, float(scene.frame_xy[0, 0]), float(scene.frame_xy[0, 1]),
                    c.board_h + 0.020)
        _step(300)

    z_stand = c.board_h + 0.0015  # base bottom on the playing surface

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    q0 = scene.board_local(scene.queen.data.root_pos_w)[0]
    fy = float(scene.frame_xy[0, 1])
    s0 = float(scene.start_side[0])
    check("settle/no-NaN: seeded reset settles finite; queen upright on the half "
          "opposite the promotion square, rocker held on its start stop; score 0, "
          "no success",
          bool(torch.isfinite(scene.queen.data.root_state_w).all())
          and bool(torch.isfinite(scene.rocker.data.root_state_w).all())
          and bool(scene.upright(scene.queen)[0])
          and abs(float(q0[2]) - c.board_h) < 0.006
          and fy * float(q0[1]) < 0.0
          and s0 * float(scene.rocker_angle()[0]) > math.radians(c.flip_deg)
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        fl = scene.board_local(scene.frame.data.root_pos_w)[0]
        ql = scene.board_local(scene.queen.data.root_pos_w)[0]
        ferr = float((fl[:2] - scene.frame_xy[0]).norm())
        return (yaw_of(scene.board.data.root_quat_w[0]),
                scene.board.data.root_pos_w[0, :2].clone(),
                fl[:2].clone(), ql[:2].clone(), ferr)

    rb = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        rb.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_yawv = max(dyaw(rb[i][0], rb[j][0]) for i, j in pairs)
    d_b = max(float((rb[i][1] - rb[j][1]).norm()) for i, j in pairs)
    d_f = max(float((rb[i][2] - rb[j][2]).norm()) for i, j in pairs)
    d_q = max(float((rb[i][3] - rb[j][3]).norm()) for i, j in pairs)
    f_err = max(r[4] for r in rb)
    print(f"[smoke] randomization max pairwise deltas (3 seeds): "
          f"board_yaw={d_yawv:.1f}deg board_xy={d_b * 1000:.1f}mm "
          f"frame_xy={d_f * 1000:.1f}mm queen_xy={d_q * 1000:.1f}mm "
          f"frame-vs-target readback err={f_err * 1000:.1f}mm", flush=True)
    check("randomization-is-real: board yaw, board xy, the promotion-square spot and "
          "the queen spot readback all differ across 3 seeds (max pairwise), and the "
          "frame body sits where the scene's frame_xy target says",
          d_yawv > 2.0 and d_b > 0.003 and d_f > 0.010 and d_q > 0.010
          and f_err < 0.005)

    # ================= 3. rocker start-side coverage ==============================================
    sides, follow_ok = set(), True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _step(15)
        _refresh()
        side = float(scene.start_side[0])
        sides.add(side)
        follow_ok = follow_ok and side * float(scene.rocker_angle()[0]) \
            > math.radians(c.flip_deg)
    print(f"[smoke] over 10 resets: start sides {sorted(sides)} angle-follows={follow_ok}",
          flush=True)
    check("side coverage: both rocker start sides drawn over 10 resets; the rocker "
          "angle readback follows the drawn side every time",
          sides == {1.0, -1.0} and follow_ok)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is pure placement — no clock exists there, no order matters.
    # CONSTRUCT that outcome here: the queen dropped perfectly into the promotion
    # square (the solve's own release), the clock never touched. Placement alone
    # earns exactly the seat credit and nothing more.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_queen()
    _report("seed-strategy")
    check("negative (SEED strategy): queen seated perfectly in the promotion square, "
          "clock untouched — seat credit only (0.45), beam still on its start side, "
          "no success",
          bool(scene.seated()[0]) and bool(scene._l_seat[0])
          and bool(scene.on_start_side()[0]) and not bool(scene._l_flip[0])
          and not bool(scene.success()[0])
          and 0.44 <= float(scene.score()[0]) <= 0.4501)

    # ================= 6. ILLEGAL PUNCH: order is an event, not a state ===========================
    # The solve's own press executed BEFORE the move (live actuator: the beam must
    # provably flip), then the queen seated perfectly. The FINAL STATE is identical
    # to the success state — seated, flipped, settled — but the foul latched at
    # press time: no success, the flip credit is forfeit forever, score stays at
    # the seat credit. This is the arbiter's call: the end state alone cannot
    # distinguish a legal from an illegal punch; the event history can.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    moved = press(0.030, 900, stop_fn=lambda: bool(scene.flipped()[0]))
    _step(180)
    _report("illegal-punch")
    foul_after_press = bool(scene._foul[0]) and bool(scene.flipped()[0]) and moved
    seat_queen()
    _step(120)
    _report("late-move")
    _REC["on"] = False
    check("ILLEGAL PUNCH: clock pressed before the move (beam verifiably flipped), "
          "queen then seated perfectly — end state equals the success state, yet "
          "the foul holds: no success, no flip credit, score <= 0.45",
          foul_after_press
          and bool(scene.seated()[0]) and bool(scene.flipped()[0])
          and bool(scene.settled()[0]) and bool(scene._foul[0])
          and not bool(scene._l_flip[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.4501)

    # ================= 7. near-miss seat: outside the square ======================================
    # Queen standing upright 8.5 cm from the square's centre — a clean stand on the
    # board, clear of the frame walls, but outside the >= 5.1 cm out-of-frame bound.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    fx0, fy0 = float(scene.frame_xy[0, 0]), float(scene.frame_xy[0, 1])
    off_y = fy0 - math.copysign(0.085, fy0)  # 8.5 cm toward the board centre
    place_board(scene.queen, fx0, off_y, z_stand)
    _step(150)
    _report("near-miss-seat")
    check("near-miss seat: queen standing upright 8.5 cm OFF the square's centre — "
          "no seat latch, score ~0, no success",
          bool(scene.upright(scene.queen)[0]) and not bool(scene.seated()[0])
          and not bool(scene._l_seat[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 8. near-miss pose: lying in the square =====================================
    # Queen LYING DOWN over the square's centre: inside the tolerance in xy but not
    # standing — the upright clause is load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    lie = _qx(torch.full((n,), math.pi / 2, device=device))
    place_board(scene.queen, float(scene.frame_xy[0, 0]), float(scene.frame_xy[0, 1]),
                c.board_h + 0.035, quat_extra=lie)
    _step(240)
    _report("lying-queen")
    check("near-miss pose: queen LYING DOWN at the square's centre — not standing, "
          "no seat latch, score ~0, no success",
          not bool(scene.upright(scene.queen)[0]) and not bool(scene.seated()[0])
          and not bool(scene._l_seat[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 9. latches vs live success =================================================
    # Full LEGAL solve (seat, then press), then the queen knocked off the square:
    # both latches hold — the score sits exactly at the 0.75 cap — but success is
    # judged live and refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    seat_queen()
    ok_seat = bool(scene._l_seat[0])
    press(0.030, 900, stop_fn=lambda: bool(scene.flipped()[0]))
    _step(240)
    was_success = bool(scene.success()[0])
    _refresh()
    fx0, fy0 = float(scene.frame_xy[0, 0]), float(scene.frame_xy[0, 1])
    place_board(scene.queen, fx0, fy0 - math.copysign(0.10, fy0), z_stand)
    _step(150)
    _report("knocked-off")
    check("latches-vs-live: legal solve reached success, then the queen knocked off "
          "the square — seat+flip latches hold (score == 0.75 cap) yet live success "
          "refuses",
          ok_seat and was_success
          and bool(scene._l_seat[0]) and bool(scene._l_flip[0])
          and not bool(scene.seated()[0]) and not bool(scene.success()[0])
          and 0.74 <= float(scene.score()[0]) <= 0.7501)

    # ================= 10. the toggle HOLDS: weak press cannot flip it ============================
    # Same actuator path check 6 proved live, but 0.004 N*m — under half the
    # 9.5e-3 N*m gravity holding torque. 300 steps (2.5 s) of pressing must leave
    # the beam pressed on its start stop: the bistability is physical.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    s0 = float(scene.start_side[0])
    press(0.004, 300, stop_fn=None)
    _step(120)
    _report("weak-press")
    check("TOGGLE HOLDS: a 0.004 N*m press (~0.4x the gravity holding torque) for "
          "2.5 s leaves the beam on its start stop — no foul, score 0",
          s0 * float(scene.rocker_angle()[0]) > math.radians(c.flip_deg)
          and not bool(scene._foul[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.move_clock")
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
