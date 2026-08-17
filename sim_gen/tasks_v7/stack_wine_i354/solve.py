"""Teleport solution for BottleSpringBayScene (sim_gen task `stack_wine_i354`) — the
task's legitimacy certificate.

The task is a spring-compression capture cycle, and every load-bearing part of it
happens through CONTACT DYNAMICS under an external-force servo (a stand-in for the
robot's grasp): teleports are used for TRANSPORT ONLY, always ending in free air
ABOVE the target bay, never inside the bay, never touching the cup, never seated.

Phases (target bay read from the scene's beacon):

  T — TRANSPORT: teleport to a hover in free air well above the bay, pitched 45 deg
      base-up with the neck toward the plunger cup, velocities zeroed.
  A — HOLD/PROBE: a CoM force + axis-torque PD holds the hover; a diverging position
      error flips the wrench-frame encoding (forge pods drag world wrenches by the
      body's rotation — and this bottle is held far from identity, so the encode is
      exercised hard and probed before anything load-bearing happens).
  B — DESCEND: slew the neck-tip reference straight down into the open-top bay at
      45 deg pitch (the level bottle does not fit the relaxed gap — the pitch is
      what makes entry geometrically possible). Ends with the tip at cup-mouth
      height just in front of the cup face.
  C — PRESS + LOWER: one coupled sweep theta 45 -> 0 deg with the tip reference
      driven along tip_x(theta) = X_KEEP - r*sin(theta) - L*cos(theta): the neck
      enters the cup, the tip presses the spring back (feed-forward on the expected
      spring force), and the base swings down PAST the orange lip (X_KEEP keeps
      every body point clear of the lip's x band while it descends), ending level
      at ~c_need+2mm compression with the base on the deck in front of the lip.
  D — RELEASE: only once slow, all wrenches are zeroed; the SPRING (not the servo)
      shoves the bottle +x so the base slides UNDER the lip and seats against the
      end wall at the seat compression. Verified by rubric readback
      (`scene.seated_now()`), with full retries from stage T on failure.

Gain audit (m = 0.45 kg, dt = 1/120 s, I_t ~ 1.7e-3 kg m^2):
  CoM PD  Kp 60 N/m, Kd 10 N s/m -> wn 11.5 rad/s (wn*dt 0.10), zeta 0.96,
          Kp*dt^2/m 0.009, Kd*dt/m 0.19
  axis PD K_R 1.2 N m, K_OM 0.05 -> K_R*dt^2/I 0.049, K_OM*dt/I 0.25
  spring feed-forward: f_ff_x = -k_spring * c_ref(theta) (residual left to the PD)
All discrete-stability ratios are well under 1; wrenches act one substep late, so
release happens only below 0.05 m/s.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched:
0.00 -> 0.10 -> 0.50 -> 1.00), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.stack_wine_i354.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bottle_spring_bay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    MG = c.mass * 9.81
    KP, KD = 60.0, 10.0
    K_R, K_OM, T_MAX = 1.2, 0.05, 0.6
    F_XY_MAX = 8.0
    TH0 = math.radians(45.0)  # entry pitch (level does not fit; ~19.3 deg is the bound)
    X_KEEP = 0.120  # base-face x kept at/below this during the sweep: the whole body
    # stays clear of the lip's x band [x_wall - lip_d, x_wall] = [0.126, 0.140]
    TIP_X_DESC = -0.078  # tip x during the descent (just in front of the cup face)
    HOVER_TIP_Z = 0.180

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def a_of(th: float) -> torch.Tensor:
        """Bottle base->neck axis for pitch th (base-up, neck toward -x)."""
        return torch.tensor([-math.cos(th), 0.0, -math.sin(th)], device=device)

    def quat_of(th: float) -> tuple[float, float, float, float]:
        """Rotation about +y mapping local +z onto a_of(th): angle -(90deg + th)."""
        phi = -(math.pi / 2 + th)
        return (math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0)

    def bstate():
        b = scene.bottle
        pos = b.data.root_pos_w[0]
        quat = b.data.root_quat_w[0]
        axis = quat_apply(quat.unsqueeze(0), ez.unsqueeze(0))[0]
        return pos, quat, axis, b.data.root_lin_vel_w[0], b.data.root_ang_vel_w[0]

    def tip_env() -> torch.Tensor:
        pos, _q, axis, _v, _w = bstate()
        return pos + axis * c.tip_off - scene.env_origins[0]

    def bay() -> int:
        return int(scene.target_bay[0])

    def y_t() -> float:
        return (-1.0, 1.0)[bay()] * c.bay_dy

    def comp() -> float:
        return float(scene.compression()[0, bay()])

    encode_body = [True]  # pre-encode wrenches into the body frame (probed in stage A)

    def apply_wrench(f_w: torch.Tensor, t_w: torch.Tensor, quat: torch.Tensor) -> None:
        if encode_body[0]:
            f = quat_apply_inverse(quat.unsqueeze(0), f_w.unsqueeze(0))[0]
            t = quat_apply_inverse(quat.unsqueeze(0), t_w.unsqueeze(0))[0]
        else:
            f, t = f_w, t_w
        scene.bottle.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            t.view(1, 1, 3).expand(n, 1, 3).contiguous(), env_ids=all_ids)

    def clear_wrench() -> None:
        zw = torch.zeros(n, 1, 3, device=device)
        scene.bottle.set_external_force_and_torque(zw, zw, env_ids=all_ids)

    def servo_substep(tip_ref: torch.Tensor, a_ref: torch.Tensor,
                      ff_x: float = 0.0) -> None:
        """One substep of the CoM-force + axis-torque servo. `tip_ref` is the neck-tip
        target in ENV coordinates; the CoM reference follows from the axis reference.
        `ff_x` is the axial (spring) feed-forward."""
        pos, quat, axis, v, w = bstate()
        com_ref = tip_ref - a_ref * c.tip_off + scene.env_origins[0]
        e = com_ref - pos
        f = KP * e - KD * v
        f[0] = f[0] + ff_x
        fn = float(f[:2].norm())
        if fn > F_XY_MAX:
            f[:2] *= F_XY_MAX / fn
        f[2] = (f[2] + MG).clamp(0.0, 2.5 * MG)
        t = K_R * torch.linalg.cross(axis, a_ref) - K_OM * w
        tn = float(t.norm())
        if tn > T_MAX:
            t *= T_MAX / tn
        apply_wrench(f, t, quat)
        env.step(no_action)

    def report(tag: str) -> None:
        cm = scene.compression()[0]
        print(f"[solve] {tag:16s} | seated={bool(scene.seated_now()[0])} "
              f"settled={bool(scene.settled()[0])} comp=({float(cm[0]):+.4f},"
              f"{float(cm[1]):+.4f}) max_seat_ctr={int(scene.max_seat_ctr[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback -------------------------------
    step(90)
    p0 = scene.bottle.data.root_pos_w[0] - scene.env_origins[0]
    by = scene.beacon.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[solve] layout readback (seed {args.seed}): target bay {bay()} "
          f"(beacon y={float(by[1]):+.3f}); bottle spawn at "
          f"({float(p0[0]):+.3f},{float(p0[1]):+.3f}); plunger comp "
          f"({float(scene.compression()[0, 0]):+.4f},{float(scene.compression()[0, 1]):+.4f})",
          flush=True)
    s_prev = print_score("P0 reset+settle")

    # ---------------- capture-cycle stages ----------------------------------------------------
    def hover_teleport() -> None:
        """TRANSPORT ONLY: free air above the bay, pitched TH0, zero velocities."""
        a = a_of(TH0)
        tip = torch.tensor([TIP_X_DESC, y_t(), HOVER_TIP_Z], device=device)
        com = tip - a * c.tip_off
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = com + scene.env_origins[0]
        qw, qx, qy, qz = quat_of(TH0)
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = qw, qx, qy, qz
        scene.bottle.write_root_state_to_sim(st, all_ids)

    def hold_probe() -> bool:
        """Stage A: hold the hover; flip the wrench encode mode if the error diverges."""
        a_ref = a_of(TH0)
        tip_ref = torch.tensor([TIP_X_DESC, y_t(), HOVER_TIP_Z], device=device)
        e0 = None
        for t in range(300):
            servo_substep(tip_ref, a_ref)
            pos, _q, axis, v, w = bstate()
            e = float((tip_ref - tip_env()).norm())
            if e0 is None:
                e0 = max(e, 1e-4)
            if t > 30 and e < 0.010 and float(v.norm()) < 0.06 \
                    and float((axis - a_ref).norm()) < 0.06 and float(w.norm()) < 0.8:
                return True
            if t > 150 and e > max(4 * e0, 0.10):
                encode_body[0] = not encode_body[0]
                print(f"[solve] wrench-frame probe: error diverged ({e0:.3f} -> {e:.3f}"
                      f" m) — flipping encode mode to "
                      f"{'body' if encode_body[0] else 'world'}", flush=True)
                return False
        e = float((tip_ref - tip_env()).norm())
        print(f"[solve] hold timeout, residual e={e:.4f} m", flush=True)
        return e < 0.015

    def descend() -> bool:
        """Stage B: slew the tip straight down into the bay at TH0 pitch."""
        a_ref = a_of(TH0)
        z1 = c.axis_h + 0.002
        for k in range(300):
            frac = min(1.0, k / 240.0)
            z_ref = HOVER_TIP_Z + (z1 - HOVER_TIP_Z) * frac
            servo_substep(torch.tensor([TIP_X_DESC, y_t(), z_ref], device=device), a_ref)
        t = tip_env()
        ok = abs(float(t[2]) - z1) < 0.012 and abs(float(t[1]) - y_t()) < 0.008
        if not ok:
            print(f"[solve] descend off target: tip=({float(t[0]):+.4f},"
                  f"{float(t[1]):+.4f},{float(t[2]):+.4f})", flush=True)
        return ok

    def press_and_lower() -> bool:
        """Stage C: coupled sweep theta TH0 -> 0 — neck into the cup, spring pressed,
        base swung down past the lip onto the deck."""
        for k in range(540):
            frac = min(1.0, k / 420.0)
            th = TH0 * (1.0 - frac)
            tip_x = X_KEEP - c.body_r * math.sin(th) - c.L * math.cos(th)
            c_ref = max(0.0, c.x_cb0 - tip_x)
            servo_substep(torch.tensor([tip_x, y_t(), c.axis_h], device=device),
                          a_of(th), ff_x=-c.spring_k * c_ref)
        # hold level at full press until slow
        tip_x = X_KEEP - c.L
        ff = -c.spring_k * max(0.0, c.x_cb0 - tip_x)
        for _ in range(240):
            pos, _q, axis, v, w = bstate()
            if float(v.norm()) < 0.05 and float(w.norm()) < 0.5:
                break
            servo_substep(torch.tensor([tip_x, y_t(), c.axis_h], device=device),
                          a_of(0.0), ff_x=ff)
        cm = comp()
        pos, _q, axis, _v, _w = bstate()
        base = pos + axis * c.base_off - scene.env_origins[0]
        ok = (cm >= c.c_need - 0.004 and float(axis[0]) < -0.95
              and float(base[0]) < c.x_wall - c.lip_d - 0.002
              and abs(float(base[2]) - c.axis_h) < 0.015)
        print(f"[solve] press verdict: comp={cm:+.4f} (need ~{c.c_need:+.4f}) "
              f"base=({float(base[0]):+.4f},{float(base[1]):+.4f},{float(base[2]):+.4f}) "
              f"axis_x={float(axis[0]):+.3f}", flush=True)
        return ok

    def release_and_seat() -> bool:
        """Stage D: zero all wrenches — the SPRING drives the bottle under the lip."""
        clear_wrench()
        step(240)
        for _ in range(6):  # allow up to +3 s of extra settling
            if bool(scene.seated_now()[0]) and bool(scene.settled()[0]):
                pos, _q, axis, _v, _w = bstate()
                base = pos + axis * c.base_off - scene.env_origins[0]
                print(f"[solve] seated: comp={comp():+.4f} base_x={float(base[0]):+.4f}"
                      f" (wall at {c.x_wall:+.4f})", flush=True)
                return True
            step(60)
        report("release verdict")
        return bool(scene.seated_now()[0])

    placed = False
    for attempt in range(5):
        print(f"[solve] capture cycle, attempt {attempt}", flush=True)
        clear_wrench()
        hover_teleport()
        if not hold_probe():
            continue
        if not descend():
            continue
        if not press_and_lower():
            clear_wrench()
            continue
        if release_and_seat():
            placed = True
            break
        clear_wrench()
    if not placed:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (never seated)", flush=True)
        os._exit(1)

    report("seated")
    s1 = print_score("P1 seated (spring-retained clamp)")
    assert s1 >= s_prev - 1e-6, "score decreased across the capture"

    # ---------------- final settle + judge ---------------------------------------------------
    step(120)
    report("final settle")
    s2 = print_score("P2 final settle")
    assert s2 >= s1 - 1e-6, "score decreased across the final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) ----------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except BaseException:  # noqa: BLE001 — die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
