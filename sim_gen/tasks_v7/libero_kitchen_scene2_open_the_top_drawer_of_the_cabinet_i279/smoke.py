"""Smoke / rubric-REJECTION battery for GateHopperScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes — the seed-style plan (slide the gate open where it stands),
level carries that never open the gate, a delivery aimed at bare ground, near-miss
end states around the bin — and that the mechanism claims the task rests on (the
gate is gravity-actuated and stays shut while level) are true. Every probe is
CONSTRUCTED (pose-consistent teleport, real physics steps, judge) — instrumentation,
never a solution. success() is monitored at EVERY step and must never turn True
anywhere in the rejection part of the battery (the audit is itself a check); the one
constructed-success sanity probe runs AFTER the audit closes.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; gate flush, ball sealed
                          inside, hopper upright far from the bin; score 0;
   2. randomization     — two seeded resets: READBACK hopper xy+yaw, bin xy+yaw and
                          the ball's in-cavity xy all differ;
   3. null-policy       — 240 idle steps: score ~0, no success;
   4. SEED-PLAN         — the seed task's outcome (the sliding member simply pulled
                          open where the box stands): plate teleported to full
                          extension with the hopper ON THE GROUND -> the ball drops
                          10 mm onto the ground UNDER the hopper, nothing is aloft,
                          no latch arms, score ~0, no success;
   5. MECHANISM (level) — hopper+contents carried aloft and held LEVEL 2 s: the
                          gate stays shut and the ball stays sealed (the "responds
                          only to tilt" claim of describe()); carry credit only;
   6. wrong place       — the full tilt maneuver executed over BARE GROUND: the
                          gate opens, the ball falls out — onto the ground. Lift +
                          gate credit but no drop, score <= 0.40, no success;
   7. near-miss beside  — the ball resting on the ground BESIDE the bin: not
                          inside, no credit, no success;
   8. toppled park      — ball IN the bin but the hopper knocked over on its side
                          near the bin: drop credit only, upright clause refuses;
   9. too-close park    — ball IN the bin, hopper upright and settled but centred
                          only 0.15 m from the bin (< clear_min, no contact):
                          clearance clause refuses, score <= drop credit;
  10. rejection audit   — success() was never True at any step of checks 1-9;
  11. success sanity    — ball centred in the bin, hopper left parked at its spawn:
                          settles into success() True and score 1.0 (audit closed);
  12. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i279.smoke --headless
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

_qz, _qy, _qmul, _qinv, _qapply = (task_scene._qz, task_scene._qy, task_scene._qmul,
                                   task_scene._qinv, task_scene._qapply)

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    ext = float(scene.gate_ext()[0])
    bw = scene.ball.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[smoke] {tag:18s} | ext={ext * 1000:6.1f}mm "
          f"ball_w=({float(bw[0]):+.3f},{float(bw[1]):+.3f},{float(bw[2]):+.3f}) "
          f"in_hopper={bool(scene.ball_in_hopper()[0])} "
          f"in_bin={bool(scene.ball_in_bin()[0])} "
          f"up={bool(scene.hopper_up()[0])} parked={bool(scene.hopper_parked()[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gate_hopper")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.70, 0.55)) + o),
                                tuple(np.array((0.30, 0.00, 0.06)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def rigid_teleport(tp: torch.Tensor, tq: torch.Tensor, bodies) -> None:
        """Carry `bodies` rigidly: the HOPPER frame lands at (tp, tq) with every
        relative pose preserved, zero velocity."""
        hp = scene.hopper.data.root_pos_w.clone()
        hq = scene.hopper.data.root_quat_w.clone()
        for b in bodies:
            loc = _qapply(_qinv(hq), b.data.root_pos_w - hp)
            _write_body(b, tp + _qapply(tq, loc), _qmul(tq, _qmul(_qinv(hq), b.data.root_quat_w)))

    def hold_hopper(tp: torch.Tensor, tq: torch.Tensor, steps: int) -> None:
        """Kinematic hold of the HOPPER only (per-step zero-vel rewrite); plate and
        ball stay free."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tp
        st[:, 3:7] = tq
        for _ in range(steps):
            scene.hopper.write_root_state_to_sim(st, _all_ids())
            _step(1)

    def write_plate_local(ext: float) -> None:
        """Pose-consistent plate write: on the rails at gate extension `ext`,
        composed with the CURRENT hopper frame."""
        _refresh()
        hp = scene.hopper.data.root_pos_w.clone()
        hq = scene.hopper.data.root_quat_w.clone()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.plate_closed_x + ext
        loc[:, 2] = c.rail_h + c.plate_t / 2 + 0.0005
        _write_body(scene.plate, hp + _qapply(hq, loc), hq)

    def ball_to(pos_env: torch.Tensor) -> None:
        p = pos_env + scene.env_origins
        _write_body(scene.ball, p, None)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    fin = torch.isfinite(scene.hopper.data.root_pos_w).all() \
        and torch.isfinite(scene.plate.data.root_pos_w).all() \
        and torch.isfinite(scene.ball.data.root_pos_w).all()
    check("settle/no-NaN: layout settles finite; gate flush, ball sealed inside, "
          "hopper upright far from the bin; score 0, no success",
          bool(fin) and abs(float(scene.gate_ext()[0])) < 0.004
          and bool(scene.ball_in_hopper()[0]) and bool(scene.hopper_up()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        bl = scene.ball_local()[0]
        return (scene.hopper.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.hopper.data.root_quat_w[0]),
                scene.bin.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.bin.data.root_quat_w[0]),
                bl[:2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_hp, a_hy, a_bp, a_by, a_bl = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_hp, b_hy, b_bp, b_by, b_bl = readback()
    d_hp = float((a_hp - b_hp).norm())
    d_hy = dyaw(a_hy, b_hy)
    d_bp = float((a_bp - b_bp).norm())
    d_by = dyaw(a_by, b_by)
    d_bl = float((a_bl - b_bl).norm())
    print(f"[smoke] randomization deltas: hopper_xy={d_hp * 1000:.1f}mm "
          f"hopper_yaw={d_hy:.1f}deg bin_xy={d_bp * 1000:.1f}mm "
          f"bin_yaw={d_by:.1f}deg ball_local_xy={d_bl * 1000:.1f}mm", flush=True)
    check("randomization-is-real: hopper xy+yaw, bin xy+yaw and the ball's "
          "in-cavity xy readback differ between seeds",
          d_hp > 0.003 and d_hy > 2.0 and d_bp > 0.003 and d_by > 2.0 and d_bl > 0.002)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED plan (slide the gate open in place) ==================
    # The seed task's whole plan is to pull the sliding member open where the box
    # stands. Constructed here: the plate teleported to full extension while the
    # hopper stays ON THE GROUND. The ball drops 10 mm onto the ground UNDER the
    # hopper and is still walled in; nothing was ever aloft, so no latch arms —
    # the seed's skill earns exactly nothing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_plate_local(c.travel_max - 0.003)
    _step(240)
    _report("seed-plan")
    _REC["on"] = False
    bw = scene.ball.data.root_pos_w[0] - scene.env_origins[0]
    check("negative (SEED plan): gate slid fully open with the hopper on the "
          "ground — ball drops to the ground under the hopper, nothing aloft, "
          "score ~0, no success",
          float(scene.gate_ext()[0]) > 0.05 and float(bw[2]) < c.ball_r + 0.010
          and not bool(scene.ball_in_bin()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 5. MECHANISM: held LEVEL, the gate stays shut ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    hover = torch.zeros(n, 3, device=device)
    hover[:, 0], hover[:, 1], hover[:, 2] = 0.27, -0.30, 0.135
    hover += scene.env_origins
    q0 = _qz(torch.zeros(n, device=device))
    rigid_teleport(hover, q0, [scene.hopper, scene.plate, scene.ball])
    hold_hopper(hover, q0, 240)  # 2 s aloft, dead level
    _report("level-hold")
    check("MECHANISM (level carry): hopper held aloft LEVEL for 2 s — the gate "
          "stays shut and the ball stays sealed inside; carry credit only",
          float(scene.gate_ext()[0]) < 0.020 and bool(scene.ball_in_hopper()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_lift + 1e-5)

    # ================= 6. negative: the delivery aimed at BARE GROUND =============================
    # The FULL maneuver — carry, tilt, gravity opens the gate, ball falls — but
    # nowhere near the bin. The mechanism works (gate credit latches), the ball
    # ends on the ground: no drop credit, no success, score <= 0.40.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    rigid_teleport(hover, q0, [scene.hopper, scene.plate, scene.ball])
    hold_hopper(hover, q0, 30)
    tilt = math.radians(30.0)
    st6 = torch.zeros(n, 13, device=device)
    ball_out = torch.zeros(n, dtype=torch.bool, device=device)
    _REC["on"] = True
    for i in range(560):
        th = tilt * min(1.0, (i + 1) / 300)
        tq = _qmul(q0, _qy(torch.full((n,), th, device=device)))
        st6[:, 0:3] = hover
        st6[:, 3:7] = tq
        scene.hopper.write_root_state_to_sim(st6, _all_ids())
        _step(1)
        ball_out |= ~scene.ball_in_hopper()
        if bool(ball_out.all()) and i > 300:
            break
    hold_hopper(hover, _qmul(q0, _qy(torch.full((n,), tilt, device=device))), 90)
    _report("ground-dump")
    _REC["on"] = False
    bw6 = scene.ball.data.root_pos_w[0] - scene.env_origins[0]
    d_bin6 = float((scene.ball.data.root_pos_w[0, :2]
                    - scene.bin.data.root_pos_w[0, :2]).norm())
    check("negative (wrong place): the gate maneuver executed over bare ground — "
          "gate opened aloft (latch), ball on the ground far from the bin, "
          "score <= 0.40, no success",
          bool(ball_out[0]) and float(bw6[2]) < c.ball_r + 0.010 and d_bin6 > 0.20
          and float(scene.gate_ext()[0]) > c.gate_open_ext
          and not bool(scene.ball_in_bin()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_lift + c.w_gate + 1e-5)

    # ================= 7. near-miss: ball beside the bin ==========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    beside = torch.zeros(n, 3, device=device)
    beside[:, :2] = scene.bin.data.root_pos_w[:, :2] - scene.env_origins[:, :2]
    beside[:, 0] += c.bin_in / 2 + c.bin_t + c.ball_r + 0.015
    beside[:, 2] = c.ball_r + 0.002
    ball_to(beside)
    _step(120)
    _report("beside-bin")
    check("near-miss (beside): the ball resting on the ground against the bin's "
          "outer wall — not inside, no drop credit, no success",
          not bool(scene.ball_in_bin()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 8. near-miss: ball delivered but hopper toppled ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    bin_xy = scene.bin.data.root_pos_w[:, :2] - scene.env_origins[:, :2]
    inbin = torch.zeros(n, 3, device=device)
    inbin[:, :2] = bin_xy
    inbin[:, 2] = c.bin_floor_t + c.ball_r + 0.002
    ball_to(inbin)
    lay = torch.zeros(n, 3, device=device)
    lay[:, :2] = bin_xy
    lay[:, 0] += 0.25
    lay[:, 2] = 0.075
    lay += scene.env_origins
    q_side = _qy(torch.full((n,), math.pi / 2, device=device))  # nose straight down -> topples
    rigid_teleport(lay, q_side, [scene.hopper, scene.plate])
    _step(360)
    _report("toppled")
    check("near-miss (toppled): ball IN the bin but the hopper knocked over on "
          "its side — drop credit only, upright clause refuses, no success",
          bool(scene.ball_in_bin()[0]) and not bool(scene.hopper_up()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_drop + 1e-5)

    # ================= 9. near-miss: ball delivered but hopper parked too close ===================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    bin_xy = scene.bin.data.root_pos_w[:, :2] - scene.env_origins[:, :2]
    ball_to(inbin)
    near = torch.zeros(n, 3, device=device)
    near[:, :2] = bin_xy
    near[:, 0] -= 0.15  # < clear_min 0.18 but > no-touch distance 0.146 + 0.004
    near[:, 2] = 0.002
    near += scene.env_origins
    rigid_teleport(near, _qz(torch.zeros(n, device=device)), [scene.hopper, scene.plate])
    _step(240)
    _report("too-close")
    d9 = float((scene.hopper.data.root_pos_w[0, :2]
                - scene.bin.data.root_pos_w[0, :2]).norm())
    check("near-miss (too close): ball IN the bin, hopper upright and settled but "
          "centred only 0.15 m away (< clear_min, no contact) — clearance clause "
          "refuses, score <= drop credit",
          bool(scene.ball_in_bin()[0]) and bool(scene.hopper_up()[0])
          and d9 < c.clear_min and not bool(scene.hopper_parked()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_drop + 1e-5)

    # ================= 10. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of checks 1-9",
          _AUD["hits"] == 0)
    _AUD["on"] = False

    # ================= 11. constructed success sanity (audit closed) ==============================
    # Hopper and plate stay exactly where they spawned (upright, far beyond
    # clear_min); only the ball is placed centred in the bin. If the predicate is
    # sound this settles into success() True and score 1.0.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    bin_xy = scene.bin.data.root_pos_w[:, :2] - scene.env_origins[:, :2]
    inbin = torch.zeros(n, 3, device=device)
    inbin[:, :2] = bin_xy
    inbin[:, 2] = c.bin_floor_t + c.ball_r + 0.002
    _REC["on"] = True
    ball_to(inbin)
    _step(240)
    _report("success-sanity")
    _REC["on"] = False
    check("success sanity: ball centred in the bin, hopper parked at its spawn — "
          "success() True and score 1.0 (predicate satisfiable)",
          bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-5)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gate_hopper")
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
