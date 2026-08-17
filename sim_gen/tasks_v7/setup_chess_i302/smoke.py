"""Smoke / rubric-REJECTION battery for CaptureArenaScene — NullRobot, teleported probes.

solve.py is the acceptance proof (contact push -> gravity chute -> real containment,
then a released contact seating on the dais). This battery proves the rubric REJECTS
wrong outcomes and that the physical claims the task rests on are load-bearing:
the rim walls really retain a pushed piece everywhere but the one opening, a king
left OUTSIDE the box is not captured, and the occupied dais physically refuses the
white king until the capture is done. Every probe is CONSTRUCTED as a settled state
(teleport / bounded quasi-static force, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; black king standing ON the
                           dais (throne occupied), white upright on the board;
                           score 0, no success;
   2. randomization      — two seeded resets: READBACK platform yaw, platform xy,
                           the DAIS spot and the white start spot all differ, and
                           the dais readback matches the scene's throne target;
   3. gap-side coverage  — over 10 resets both gap sides occur; the chute readback
                           sits at the ACTIVE gap and the blocker plugs the MIRROR
                           gap every time;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. RIM RETENTION      — physical probe with a live actuator: the same
                           quasi-static velocity-servo push the solve uses, aimed
                           at the BLOCKED side, moves the black king >= 5 cm yet
                           the rim/blocker retain it ON the board — no ejection
                           latch, score ~0 ("only way off is the opening" is
                           physics, not fiat);
   6. SEED STRATEGY      — the seed's plan (independent pick-and-place, no
                           capture): white king placed perfectly upright on a
                           plain square while the black king still holds the
                           throne -> no success, score ~0;
   7. near-miss capture  — black king lying on the FLOOR beside the box (off the
                           board, but not contained), white properly enthroned on
                           the vacant dais -> captured False, no success, score
                           capped (a captured king left outside the box fails);
   8. near-miss throne   — black properly captured (dropped into the box), white
                           standing 8 cm OFF the dais: every latch fires (score
                           exactly the 0.75 cap) yet success refuses;
   9. ORDER IS OCCUPANCY — the solve's own P3 release over the throne, executed
                           BEFORE the capture: white dropped above the occupied
                           dais topples off the black king's crown — physically
                           refused, no enthronement, no success;
  10. swapped pieces     — WHITE king dropped into the capture box, black king
                           still on the throne: right box, wrong identity -> no
                           success, score ~0;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_chess_i302.smoke --headless
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

_qmul, _qconj, _qz = task_scene._qmul, task_scene._qconj, task_scene._qz

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
    b = scene.board_local(scene.black.data.root_pos_w)
    w = scene.board_local(scene.white.data.root_pos_w)
    ch = scene.chute_local(scene.black.data.root_pos_w)
    print(f"[smoke] {tag:16s} | black_board=({float(b[0, 0]):+.3f},{float(b[0, 1]):+.3f},"
          f"{float(b[0, 2]):+.3f}) black_chute=({float(ch[0, 0]):+.3f},"
          f"{float(ch[0, 1]):+.3f},{float(ch[0, 2]):+.3f}) "
          f"white=({float(w[0, 0]):+.3f},{float(w[0, 1]):+.3f},{float(w[0, 2]):+.3f}) "
          f"l1={bool(scene._l1[0])} l2={bool(scene._l2[0])} l3={bool(scene._l3[0])} "
          f"captured={bool(scene.black_captured()[0])} "
          f"enthroned={bool(scene.white_enthroned()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.capture_arena")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.20, 0.02, 0.85)) + o),
                                tuple(np.array((0.0, 0.0, 0.10)) + o),
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

    def place_board(body, x, y, z, yaw: float = 0.0) -> None:
        """Teleport `body` to board-local (x, y, z) with the platform's heading."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = x
        loc[:, 1] = y
        loc[:, 2] = z
        q = scene.platform.data.root_quat_w
        if yaw:
            q = _qmul(q, _qz(torch.full((n,), yaw, device=device)))
        _write_body(body, scene.board_to_world(loc), q)

    def place_chute(body, x, y, z, tilt_x: float = 0.0) -> None:
        """Teleport `body` to chute-local (x, y, z); optional roll about chute-x
        (tilt_x = +/-pi/2 lays the piece down along the chute's y axis)."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = x
        loc[:, 1] = y
        loc[:, 2] = z
        pos_w = scene.chute.data.root_pos_w \
            + quat_apply(scene.chute.data.root_quat_w, loc)
        q = scene.chute.data.root_quat_w
        if tilt_x:
            hx = torch.zeros(n, 4, device=device)
            hx[:, 0] = math.cos(tilt_x / 2)
            hx[:, 1] = math.sin(tilt_x / 2)
            q = _qmul(q, hx)
        _write_body(body, pos_w, q)

    def servo_to(body, tgt_fn, steps: int, v_max: float = 0.10) -> None:
        """The solve's quasi-static velocity-servoed CoM push (bounded force, v
        capped so nothing is slammed over a wall); board-local target, fixed step
        budget, force cleared after."""
        m_kg = float(c.piece_mass)
        zero = torch.zeros(n, 1, 3, device=device)
        q_ref = body.data.root_quat_w.clone()
        gain, f_cap = 30.0, 2.2
        for _ in range(steps):
            tgt_w = scene.board_to_world(tgt_fn())
            err = tgt_w - body.data.root_pos_w
            err[:, 2] = 0.0
            dist = err.norm(dim=-1)
            v = body.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            dirv = err / dist.clamp(min=1e-6).unsqueeze(-1)
            v_des = dirv * torch.minimum(
                torch.full_like(dist, v_max), 2.5 * dist).unsqueeze(-1)
            f_w = m_kg * gain * (v_des - v) + 0.35 * dirv
            fmag = f_w.norm(dim=-1, keepdim=True)
            f_w = f_w * (f_cap / fmag.clamp(min=1e-9)).clamp(max=1.0)
            f_enc = quat_apply(_qmul(q_ref, _qconj(body.data.root_quat_w)), f_w)
            body.set_external_force_and_torque(
                f_enc.unsqueeze(1), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    z_stand = c.board_h + 0.0015          # base bottom on the playing surface
    z_drop_dais = c.board_h + c.dais_h + 0.025  # the solve's P3 release height

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    b0 = scene.board_local(scene.black.data.root_pos_w)[0]
    check("settle/no-NaN: seeded reset settles finite; black king standing ON the dais "
          "(throne occupied), white upright on the board; score 0, no success",
          bool(torch.isfinite(scene.black.data.root_state_w).all())
          and bool(torch.isfinite(scene.white.data.root_state_w).all())
          and float((b0[:2] - scene.throne[0]).norm()) < 0.03
          and float(b0[2]) > c.board_h + c.dais_h - 0.004
          and bool(scene.upright(scene.black)[0]) and bool(scene.upright(scene.white)[0])
          and bool(scene.on_board(scene.white)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        w = scene.board_local(scene.white.data.root_pos_w)
        d = scene.board_local(scene.dais.data.root_pos_w)
        return (yaw_of(scene.platform.data.root_quat_w[0]),
                scene.platform.data.root_pos_w[0, :2].clone(),
                d[0, :2].clone(), float(w[0, 0]),
                float((d[0, :2] - scene.throne[0]).norm()))

    rb = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        rb.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_yawv = max(dyaw(rb[i][0], rb[j][0]) for i, j in pairs)
    d_p = max(float((rb[i][1] - rb[j][1]).norm()) for i, j in pairs)
    d_d = max(float((rb[i][2] - rb[j][2]).norm()) for i, j in pairs)
    d_wx = max(abs(rb[i][3] - rb[j][3]) for i, j in pairs)
    dt_err = max(r[4] for r in rb)
    print(f"[smoke] randomization max pairwise deltas (3 seeds): "
          f"platform_yaw={d_yawv:.1f}deg platform_xy={d_p * 1000:.1f}mm "
          f"dais_xy={d_d * 1000:.1f}mm white_x={d_wx * 1000:.1f}mm "
          f"dais-vs-throne readback err={dt_err * 1000:.1f}mm", flush=True)
    check("randomization-is-real: platform yaw, platform xy, the dais spot and the "
          "white start spot readback all differ across 3 seeds (max pairwise), and "
          "the dais body sits where the scene's throne target says",
          d_yawv > 2.0 and d_p > 0.003 and d_d > 0.003
          and d_wx > 0.003 and dt_err < 0.005)

    # ================= 3. gap-side coverage: chute + blocker follow the draw ======================
    sides, follow_ok = set(), True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        side = float(scene.side[0])
        sides.add(side)
        ch_y = float(scene.board_local(scene.chute.data.root_pos_w)[0, 1])
        bk_y = float(scene.board_local(scene.blocker.data.root_pos_w)[0, 1])
        follow_ok = follow_ok and (side * ch_y > 0.15) and (-side * bk_y > 0.15)
    print(f"[smoke] over 10 resets: gap sides {sorted(sides)} chute/blocker follow={follow_ok}",
          flush=True)
    check("gap-side coverage: both gap sides drawn over 10 resets; chute readback at "
          "the active gap and blocker plugging the mirror gap every time",
          sides == {1.0, -1.0} and follow_ok)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. RIM RETENTION: live push at the blocked side ============================
    # The solve's own quasi-static servo (a real, proven actuator — it is what
    # ejects the king in solve.py), aimed at the BLOCKED gap. The king must
    # provably travel (>= 5 cm) yet the rim/blocker must retain it ON the board:
    # no ejection latch, score ~0. White is parked out of the push lane first.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    side = float(scene.side[0])
    place_board(scene.white, 0.15, side * 0.12, z_stand)
    _step(30)
    _REC["on"] = True
    y_start = float(scene.board_local(scene.black.data.root_pos_w)[0, 1])

    def tgt_blocked() -> torch.Tensor:
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = scene.throne[:, 0]
        loc[:, 1] = -scene.side * 0.24
        loc[:, 2] = c.board_h
        return loc

    servo_to(scene.black, tgt_blocked, steps=780)
    _step(90)
    _report("rim-retention")
    _REC["on"] = False
    bb = scene.board_local(scene.black.data.root_pos_w)[0]
    travelled = abs(float(bb[1]) - y_start)
    print(f"[smoke] rim retention: black y {y_start:+.3f} -> {float(bb[1]):+.3f} "
          f"(travelled {travelled * 1000:.0f}mm, board edge at "
          f"{c.board_xy / 2 - c.rim_t:+.3f})", flush=True)
    check("RIM RETENTION: the solve's quasi-static push aimed at the BLOCKED side "
          "moves the black king >= 5 cm (live actuator) yet the rim retains it ON "
          "the board — no ejection latch, score ~0",
          travelled >= 0.05 and bool(scene.on_board(scene.black)[0])
          and not bool(scene._l1[0]) and not bool(scene.black_ejected()[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # The seed treats every move as an independent pick-and-place onto a marked
    # square, with no notion of capture. CONSTRUCT that outcome: the white king
    # placed perfectly upright on a plain square while the black king still holds
    # the throne. Beautiful placement, zero progress.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    side = float(scene.side[0])
    place_board(scene.white, 0.15, -side * 0.10, z_stand)
    _step(150)
    _report("seed-strategy")
    check("negative (SEED strategy): white king placed perfectly on a plain square, "
          "black king untouched on the throne — no success, score ~0",
          bool(scene.upright(scene.white)[0]) and bool(scene.on_board(scene.white)[0])
          and not bool(scene.white_enthroned()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 7. near-miss capture: beside the box is not IN the box ======================
    # Black king constructed lying on the FLOOR beside the capture box (it left the
    # board, so the ejection latch may fire) and the white king properly enthroned
    # on the now-vacant dais. Capture must read False and success must refuse:
    # "a captured king left outside the box fails".
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place_chute(scene.black, 0.20, 0.30, -0.09, tilt_x=math.pi / 2)
    _step(150)
    place_board(scene.white, float(scene.throne[0, 0]) + 0.003,
                float(scene.throne[0, 1]), z_drop_dais)
    _step(240)
    _report("beside-box")
    _REC["on"] = False
    ch_b = scene.chute_local(scene.black.data.root_pos_w)[0]
    check("near-miss capture: black king on the floor BESIDE the box (outside its "
          "walls), white properly enthroned on the vacant dais — captured False, "
          "no success, score <= 0.75 cap",
          float(ch_b[0]) > c.bin_x_half and float(ch_b[2]) < -0.05
          and not bool(scene.black_captured()[0])
          and bool(scene.white_enthroned()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7501)

    # ================= 8. near-miss throne: every latch, no success ================================
    # Black properly captured (dropped into the box — a real contact landing), the
    # white king standing upright 8 cm OFF the dais: l1+l2+l3 all latch, the score
    # sits exactly at the 0.75 cap, success still refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    place_chute(scene.black, 0.0, 0.28, -0.05)
    _step(180)
    place_board(scene.white, float(scene.throne[0, 0]) + 0.08,
                float(scene.throne[0, 1]), z_stand)
    _step(120)
    _report("off-throne")
    check("near-miss throne: black captured for real, white upright 8 cm off the "
          "dais — all three latches fire (score == 0.75 cap) yet no success",
          bool(scene.black_captured()[0]) and bool(scene._l1[0]) and bool(scene._l2[0])
          and bool(scene._l3[0]) and not bool(scene.white_enthroned()[0])
          and not bool(scene.success()[0])
          and 0.74 <= float(scene.score()[0]) <= 0.7501)

    # ================= 9. ORDER IS OCCUPANCY: the throne refuses two ==============================
    # The solve's own P3 move executed BEFORE the capture: release the white king
    # above the dais while the black king still stands on it. It can only land on
    # the black king's crown and topple off — two 44 mm bases cannot both centre
    # within the 25 mm tolerance, so enthronement while occupied is impossible.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place_board(scene.white, float(scene.throne[0, 0]),
                float(scene.throne[0, 1]), c.board_h + c.dais_h + 0.135)
    _step(300)
    _report("occupied-drop")
    _REC["on"] = False
    wz = float(scene.board_local(scene.white.data.root_pos_w)[0, 2])
    check("ORDER IS OCCUPANCY: the solve's release over the throne executed BEFORE "
          "the capture — the white king topples off the occupying black king, no "
          "enthronement, no success, score ~0",
          wz < c.board_h + 0.05  # it really fell (construct is live)
          and not bool(scene.white_enthroned()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 10. negative: swapped pieces ===============================================
    # The WHITE king dropped into the capture box, black king still on the throne:
    # right box, wrong identity. The rubric tracks the pieces, not "a piece".
    torch.manual_seed(100)
    env.reset()
    _step(30)
    place_chute(scene.white, 0.0, 0.28, -0.05)
    _step(180)
    _report("swapped")
    ch_w = scene.chute_local(scene.white.data.root_pos_w)[0]
    check("negative (swapped pieces): WHITE king inside the capture box, black king "
          "still on the throne — no success, score ~0",
          float(ch_w[2]) < c.capture_z_max and abs(float(ch_w[0])) < c.bin_x_half
          and not bool(scene.black_captured()[0]) and not bool(scene._l1[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.capture_arena")
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
