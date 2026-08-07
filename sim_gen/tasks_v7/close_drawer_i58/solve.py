"""Teleport solution for ReturnDockScene (sim_gen task `close_drawer_i58`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating a body the solver is already holding in
free air). Every load-bearing interaction is contact dynamics or applied force:

1. PERCEPTION: dock pose, WHICH socket holds the pin, and the tray's opening are
   read back from the episode state — never hard-coded.
2. BOTTLE OUT (applied force): a PD force + gravity feedforward (a firm grasp,
   <= 8 N) lifts the bottle straight up along the dock's local up, out through the
   tray's open top, with a small righting torque standing in for the grasp's
   orientation constraint. Once the bottle is in free air well above the station it
   is TELEPORTED over the roof pad and RELEASED 2 cm up — gravity seats it; the
   grippy pad holds it at the station's tilt. The dock clause is judged on the
   settled contact outcome, not on the carry.
3. PIN OUT (applied force): the pin is extracted VERTICALLY from its socket by the
   same PD carry while the tray presses on it under its own downhill weight — the
   well walls, then the tray face, are real contacts the extraction must beat.
   Once in free air above the station it is teleported to open ground beside the
   dock and released.
4. CLOSURE (nobody's hands): with the path clear, gravity glides the tray down the
   polished runway, through the doorway, onto the rear-wall stop. The solver's
   hands are off for this entire phase — the judged "drawer closes" outcome is
   produced by the scene's own mechanism, never by the solver.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_drawer_i58.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.return_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | face_x={float(scene.face_x()[0]):+.4f} "
              f"closed={bool(scene.tray_closed()[0])} "
              f"pin_clear={bool(scene.pin_clear()[0])} "
              f"out={bool(scene.bottle_out()[0])} "
              f"docked={bool(scene.bottle_docked()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def dock_frame():
        return scene.dock.data.root_pos_w, scene.dock.data.root_quat_w

    def to_world(local: torch.Tensor) -> torch.Tensor:
        dp, dq = dock_frame()
        return dp + quat_apply(dq, local)

    zero = torch.zeros(n, 1, 3, device=device)

    def carry(body, mass: float, tgt_local, *, kp: float, kd: float, clamp: float,
              ku: float, kw: float, steps: int, done=None, label: str = "") -> None:
        """Applied-force carry: PD toward a dock-local target + gravity feedforward
        (what a firm grasp does, force-limited), plus a small righting torque that
        stands in for the grasp's orientation constraint. The wrench goes through
        THIS body only; nothing else is ever forced."""
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        for i in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            f_w = mass * 9.81 * ez + kp * (to_world(tgt) - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            _, dq = dock_frame()
            up = quat_apply(dq, ez)
            axis = quat_apply(q, ez)
            t_w = ku * torch.cross(axis, up, dim=-1) - kw * w
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._dock_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    def teleport(body, local, quat_dock_aligned: bool, world_z: float | None = None) -> None:
        """TRANSPORT ONLY: relocate a body the solver is holding in free air.
        Never used to create a judged contact outcome — every placement is
        RELEASED above its rest and gravity finishes the job."""
        dp, dq = dock_frame()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = to_world(torch.tensor(local, device=device).expand(n, 3))
        if world_z is not None:
            st[:, 2] = world_z
        if quat_dock_aligned:
            st[:, 3:7] = dq
        else:
            st[:, 3] = 1.0
        body.write_root_state_to_sim(st, torch.arange(n, device=device))

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # tray glides ~4 mm onto the pin; bottle seats on the tray floor
    dp0 = (scene.dock.data.root_pos_w - scene.env_origins)[0]
    dq0 = scene.dock.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(dq0[3]), float(dq0[0])))
    k = int(scene.socket_idx[0])
    x0 = float(scene.x0[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"dock=({float(dp0[0]):+.3f},{float(dp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"socket={k} (x={c.sockets[k]:+.3f}) opening x0={x0:+.3f} "
          f"face_x={float(scene.face_x()[0]):+.4f}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.tray_closed()[0]), "tray must start held open"
    assert not bool(scene.pin_clear()[0]), "pin must start in its socket"
    assert not bool(scene.bottle_out()[0]), "bottle must start inside the tray"
    assert abs(float(scene.face_x()[0]) - x0) < 0.012, \
        "tray must be resting at its socket's opening"
    s_prev = print_score("P0 reset+settle (tray held open on the pin, bottle aboard)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: bottle out of the tray -> docked on the roof pad ------------------
    # Lift straight up along the dock's local up, out through the open top. The
    # bottle_out clause is judged DURING this force-driven lift (no teleport yet).
    bloc = scene._dock_local(scene.bottle.data.root_pos_w)[0]
    carry(scene.bottle, c.bottle_mass,
          (float(bloc[0]), float(bloc[1]), 0.30),
          kp=80.0, kd=10.0, clamp=8.0, ku=0.15, kw=0.02, steps=360,
          done=lambda: bool(scene.bottle_out()[0])
          and float(scene._dock_local(scene.bottle.data.root_pos_w)[0, 2]) > 0.27,
          label="bottle lift")
    assert bool(scene._out[0]), "bottle_out must latch during the force lift"
    # Transport the held bottle over the pad; release 2 cm up; gravity docks it.
    px, py = c.pad_center
    teleport(scene.bottle, (px, py, c.pad_z1 + c.bottle_h / 2 + 0.020),
             quat_dock_aligned=True)
    step(240)   # free fall 2 cm, seat on the grippy pad, settle
    report("bottle-dock")
    assert bool(scene.bottle_docked()[0]), "bottle must be docked upright on the pad"
    assert bool(scene._dock_l[0]), "docked latch must fire once settled"
    s = print_score("P1 bottle lifted out (force) and docked on the roof pad")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert s >= c.w_out + c.w_dock - 1e-6, f"P1 score {s:.3f} below out+dock credit"
    s_prev = s

    # ---------------- phase 2: pin extracted (force, under the tray's press) ---------------------
    # The tray presses on the pin under its own downhill weight; the extraction is
    # a straight vertical pull against well-wall and tray-face contact.
    sx = float(c.sockets[k])
    carry(scene.pin, c.pin_mass,
          (sx, 0.0, 0.30),
          kp=200.0, kd=8.0, clamp=6.0, ku=0.05, kw=0.01, steps=360,
          done=lambda: float(scene._dock_local(scene.pin.data.root_pos_w)[0, 2]) > 0.27,
          label="pin extract")
    assert bool(scene._pin_l[0]), "pin_clear must latch once the pin leaves the corridor"
    # Transport the held pin to open ground beside the dock; release upright.
    teleport(scene.pin, (0.10, 0.30, 0.0), quat_dock_aligned=False,
             world_z=c.pin_h / 2 + 0.003)
    step(120)
    report("pin-out")
    assert bool(scene.pin_clear()[0]), "pin must rest clear of the runway corridor"
    s = print_score("P2 pin extracted from its socket and parked clear")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    assert s >= c.w_out + c.w_dock + c.w_pin - 1e-6, f"P2 score {s:.3f} below pin credit"
    s_prev = s

    # ---------------- phase 3: hands off — the station closes its own tray ----------------------
    closed_at = None
    for i in range(720):   # up to 6 s: glide ~0.22 m at ~0.11 m/s terminal + seating
        env.step(no_action)
        if bool(scene.success()[0]):
            closed_at = i
            break
    report("self-close")
    assert bool(scene.tray_closed()[0]), \
        f"tray failed to self-close: face_x={float(scene.face_x()[0]):+.4f}"
    assert closed_at is not None and bool(scene.success()[0]), \
        "success() must hold once the tray seats"
    s = print_score(f"P3 tray glided shut by itself ({closed_at} hands-off steps)")
    assert s >= s_prev - 1e-6, "score decreased across P3"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
