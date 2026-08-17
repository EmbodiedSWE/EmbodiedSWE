"""Teleport solution for CastlingGalleryScene (sim_gen task `setup_chess_i121`) —
the task's legitimacy certificate.

There is NO transport teleport in this solve: both pieces start captive in the
channel network and every centimetre of every placement is CONTACT — a horizontal
velocity-servoed push force at each piece's (base-weighted, near-floor) centre of
mass, exactly the wrench a Franka fingertip pressing on the exposed shaft would
transmit. The servo is F = m*K*(v_des - v) + friction feedforward, gain sized so
K*dt << 1, force released only when the piece is both close and slow. Commanded
world forces are pre-encoded by R_ref * R_now^T to be robust to the known
rotation-since-reset frame drag of set_external_force_and_torque (the pieces
barely rotate, so this is a no-op if the API is clean).

Phases (strict order — the geometry enforces it; the servo just follows):
  P0  settle + layout readback; baseline score ~0.
  P1  push the ROOK back along the gallery to the junction, then sideways into
      the BEACON-marked pocket until its flange is fully clear of the gallery
      (l1 latches, 0.20).
  P2  keep pushing the rook to the pocket's end stop (l2, 0.45).
  P3  push the KING along the gallery, past the now-clear junction (l3, 0.75),
      up to the gold end stop -> success, 1.0.
  P4  >= 3.3 simulated seconds HANDS-OFF; print SIM_GEN_SOLVE: SUCCESS only if
      success() still holds live.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's partial credit is latched).

Run (forge): python -u -m simgen_tasks.setup_chess_i121.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qconj = scene_mod._qmul, scene_mod._qconj

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.castling_gallery")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        k, r = scene._piece_loc()
        kv = float(scene.king.data.root_lin_vel_w.norm(dim=-1)[0])
        rv = float(scene.rook.data.root_lin_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | king=({float(k[0, 0]):+.3f},{float(k[0, 1]):+.3f},"
              f"{float(k[0, 2]):+.3f}) rook=({float(r[0, 0]):+.3f},{float(r[0, 1]):+.3f},"
              f"{float(r[0, 2]):+.3f}) kv={kv:.3f} rv={rv:.3f} "
              f"l1={bool(scene._l1[0])} l2={bool(scene._l2[0])} l3={bool(scene._l3[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # frame-drag-robust force encoding: command R_ref * R_now^T * f_world
    q_ref = {}

    def encode(body, name: str, f_world: torch.Tensor) -> torch.Tensor:
        return quat_apply(_qmul(q_ref[name], _qconj(body.data.root_quat_w)), f_world)

    def servo(body, name: str, loc_fn, *, tol: float = 0.012,
              max_steps: int = 2400) -> bool:
        """Push `body` toward the plinth-local waypoint `loc_fn()` with a
        horizontal velocity-servoed CoM force; release only when close AND slow,
        then settle hands-off. Escalates gain (bounded: K*dt <= 0.75) and cap on
        stall. Returns True iff the waypoint was reached."""
        m_kg = float(c.piece_mass)
        gain, f_cap = 30.0, 2.0
        v_max = 0.10
        reached = False
        last_xy = body.data.root_pos_w[:, 0:2].clone()
        for i in range(max_steps):
            tgt_w = scene.local_to_world(loc_fn())
            err = tgt_w - body.data.root_pos_w
            err[:, 2] = 0.0
            dist = err.norm(dim=-1)
            v = body.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            if float(dist[0]) < tol and float(v.norm(dim=-1)[0]) < 0.02:
                reached = True
                break
            dirv = err / dist.clamp(min=1e-6).unsqueeze(-1)
            v_des = dirv * torch.minimum(
                torch.full_like(dist, v_max), 2.5 * dist).unsqueeze(-1)
            f_w = m_kg * gain * (v_des - v) + 0.45 * dirv  # friction feedforward
            fmag = f_w.norm(dim=-1, keepdim=True)
            f_w = f_w * (f_cap / fmag.clamp(min=1e-9)).clamp(max=1.0)
            body.set_external_force_and_torque(
                encode(body, name, f_w).unsqueeze(1), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 240 == 239:
                moved = float((body.data.root_pos_w[:, 0:2] - last_xy).norm(dim=-1)[0])
                if moved < 0.005:
                    gain = min(gain * 1.6, 90.0)  # K*dt stays <= 0.75
                    f_cap = min(f_cap * 1.4, 6.0)
                    print(f"[solve] {name} stalled at dist={float(dist[0]):.3f}; "
                          f"gain->{gain:.0f} cap->{f_cap:.1f}", flush=True)
                last_xy = body.data.root_pos_w[:, 0:2].clone()
        clear_force(body)
        step(90)  # hands-off settle
        return reached

    def wp(x_fn, y_fn):
        """Waypoint factory: plinth-local (x, y) per env (z is ignored by the servo)."""
        def loc_fn() -> torch.Tensor:
            loc = torch.zeros(n, 3, device=device)
            loc[:, 0] = x_fn()
            loc[:, 1] = y_fn()
            loc[:, 2] = c.plinth_h
            return loc
        return loc_fn

    zero = lambda: 0.0  # noqa: E731
    xj = c.junction_x

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(180)
    q_ref["king"] = scene.king.data.root_quat_w.clone()
    q_ref["rook"] = scene.rook.data.root_quat_w.clone()
    pp = (scene.plinth.data.root_pos_w - scene.env_origins)[0]
    pq = scene.plinth.data.root_quat_w[0]
    pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
    k0, r0 = scene._piece_loc()
    bl = scene._local(scene.beacon.data.root_pos_w)[0]
    s0side = float(scene.tgt[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"plinth=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) yaw={pyaw:+.1f}deg "
          f"target_side={'+y' if s0side > 0 else '-y'} "
          f"beacon_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
          f"king_x={float(k0[0, 0]):+.3f} rook_x={float(r0[0, 0]):+.3f}", flush=True)
    report("reset")
    assert torch.isfinite(scene.king.data.root_state_w).all(), "NaN in king state"
    assert torch.isfinite(scene.rook.data.root_state_w).all(), "NaN in rook state"
    assert bool(scene.in_channel(k0).all()), "king must start captive in the channel"
    assert bool(scene.in_channel(r0).all()), "rook must start captive in the channel"
    assert bool((scene.tgt[0] * bl[1]) > 0.15), "beacon must sit on the target side"
    assert float(r0[0, 0]) > float(k0[0, 0]) + 0.05, "rook must start between king and home"
    s0 = print_score("P0 reset+settle (both pieces captive in the gallery)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: rook -> junction -> into the beacon pocket ------------------
    ok = servo(scene.rook, "rook", wp(lambda: xj, zero), tol=0.005)
    report("rook@junct")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (rook never reached the junction)", flush=True)
        os._exit(1)
    ok = servo(scene.rook, "rook",
               wp(lambda: xj, lambda: scene.tgt * 0.085), tol=0.012)
    report("rook-garage")
    if not (ok and bool(scene._l1[0])):
        print("SIM_GEN_SOLVE: FAIL (rook did not garage into the target pocket)",
              flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success with the king in place"
    s1 = print_score("P1 rook garaged in the beacon pocket (flange clear of the gallery)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.19, f"P1 score {s1} (expect l1=0.20)"

    # ---------------- phase 2: rook to the pocket end stop ---------------------------------
    ok = servo(scene.rook, "rook",
               wp(lambda: xj, lambda: scene.tgt * 0.129), tol=0.012)
    report("rook-seated")
    if not (ok and bool(scene.rook_seated()[0])):
        print("SIM_GEN_SOLVE: FAIL (rook not seated at the pocket end)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success before the king moves"
    s2 = print_score("P2 rook seated at the target pocket's end cell")
    assert s2 >= s1 - 1e-6 and s2 >= 0.44, f"P2 score {s2} (expect l1+l2=0.45)"

    # ---------------- phase 3: king past the junction to the gold home cell ----------------
    ok = servo(scene.king, "king", wp(lambda: xj + 0.10, zero), tol=0.015)
    report("king-past")
    if not (ok and bool(scene._l3[0])):
        print("SIM_GEN_SOLVE: FAIL (king never cleared the junction)", flush=True)
        os._exit(1)
    s3a = print_score("P3a king past the junction (gallery clear)")
    assert s3a >= s2 - 1e-6 and s3a >= 0.74, f"P3a score {s3a} (expect 0.75)"

    ok = servo(scene.king, "king", wp(lambda: c.home_x + 0.002, zero), tol=0.012)
    report("king-home")
    if not (ok and bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success with the king at the gold wall)",
              flush=True)
        os._exit(1)
    s3 = print_score("P3 king at the gold home cell: castling complete")
    assert s3 >= s3a - 1e-6 and s3 >= 0.99, f"P3 score {s3} (expect success=1.0)"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
