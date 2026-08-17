"""Teleport solution for UmbrellaBayonetScene (sim_gen task
`put_umbrella_in_umbrella_stand_i367`) — the task's legitimacy certificate.

NO transport teleports are needed at all: every object already starts where the plan
needs it (the runner is captive on the mast, the pin is seated in the mast). ALL
load-bearing interactions run through contact dynamics via velocity-regulated
external wrenches, re-set every step and zeroed before judging:
  - the PIN is pulled straight out along +x with a horizontal force servo (world
    force rotated into the pin's body frame every step — the frame-drag trap; the
    static-breakaway feedforward defeats channel friction), then dropped;
  - the RUNNER is driven along its two real DOFs with a gravity-feedforward heave
    force + a yaw torque (the runner's quat is PURE YAW — its body z axis is the
    world z axis, so both act directly). Gains respect the one-substep wrench delay
    (heave KV*dt/m ~= 0.25, yaw KV*dt/Izz ~= 0.27, both << 1).

PLAN (read-only, from scene.describe()): the seated pin jams the track at
q_block = 0.0575; the shelf plate passes the collar at any yaw but the lug only
through the +x gap; parked means the lug RESTS on the shelf ring, twisted away from
the gap:
  P1 pull the RED-knob transit pin out +x until it is clear of the mast, drop it
     (score latch 0.30),
  P2 spin the runner so the YELLOW lug points at the gap heading (+x),
  P3 hoist the runner up the pole; the lug threads through the keyed gap and clears
     the shelf top (pass latch, score 0.60),
  P4 TWIST +90 deg above the shelf (the bayonet move),
  P5 lower until the lug seats ON the green ring, release — it hangs parked
     hands-off; ring down to success,
  P6 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i367.solve
             --headless [--seed N]
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import os
import threading

import torch

import robobench
from robobench.core import ENVS

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
# Pin pull servo (mass 0.06: KV*dt/m = 2/(120*0.06) ~= 0.28; the FF beats the static
# breakaway mu_s*m*g ~= 0.29 N of the pin grinding on the channel floor).
P_KP = 3.0
P_VCAP = 0.12
P_KV = 2.0
P_FF = 0.35
P_FMAX = 1.5
P_XTGT = 0.20   # aim past the clear distance; the break is a position readback
# Runner heave servo (mass 0.20: KV*dt/m = 6/(120*0.20) = 0.25).
R_KP = 3.0
R_VUP = 0.15
R_VDN = 0.06
R_KV = 6.0
R_FMIN, R_FMAX = 0.4, 6.0
# Runner yaw servo (authored Izz 2.5e-4: KV*dt/I = 0.008/(120*2.5e-4) ~= 0.27).
Y_K = 3.0
Y_WCAP = 1.2
Y_KV = 0.008   # KV*dt/I_z = 0.008/(120*2.5e-4) = 0.27 < 1 (wrench-delay bound)
Y_TMAX = 0.02


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.umbrella_bayonet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def q0() -> float:
        return float(scene.runner_q()[0])

    def yaw() -> torch.Tensor:
        qq = scene.runner.data.root_quat_w
        dx = 1.0 - 2.0 * (qq[:, 2] ** 2 + qq[:, 3] ** 2)
        dy = 2.0 * (qq[:, 1] * qq[:, 2] + qq[:, 0] * qq[:, 3])
        return torch.atan2(dy, dx)

    def yaw0() -> float:
        return float(yaw()[0])

    def pd0() -> float:
        return float(scene.pin_dist()[0])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={q0():+.4f} yaw={yaw0():+.3f}rad "
              f"hcos={float(scene.lug_heading_cos()[0]):+.3f} pin_d={pd0():+.4f} | "
              f"parked={bool(scene.parked_now()[0])} aligned={bool(scene.aligned_now()[0])} "
              f"latches p/p={float(scene.pin_latch[0]):.0f}/{float(scene.pass_latch[0]):.0f} "
              f"settled={bool(scene.settled()[0])} | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def runner_wrench_off() -> None:
        scene.runner.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def wrap(e: torch.Tensor) -> torch.Tensor:
        return (e + math.pi) % (2 * math.pi) - math.pi

    def runner_servo(q_tgt: float, yaw_tgt: float, v_cap: float, tag: str,
                     max_steps: int = 1200, q_tol: float = 0.006,
                     yaw_tol: float = 0.06, hold_streak: int = 25) -> None:
        """Drive BOTH runner DOFs: gravity-feedforward heave force + yaw torque,
        re-set EVERY step (runner quat is pure yaw -> body z == world z)."""
        f = torch.zeros(n, 1, 3, device=device)
        tq = torch.zeros(n, 1, 3, device=device)
        done, i = 0, 0
        for i in range(max_steps):
            q = scene.runner_q()
            vz = scene.runner.data.root_lin_vel_w[:, 2]
            wz = scene.runner.data.root_ang_vel_w[:, 2]
            v_des = (R_KP * (q_tgt - q)).clamp(-v_cap, v_cap)
            f[:, 0, 2] = (c.runner_mass * G + R_KV * (v_des - vz)).clamp(R_FMIN, R_FMAX)
            e = wrap(yaw_tgt - yaw())
            w_des = (Y_K * e).clamp(-Y_WCAP, Y_WCAP)
            tq[:, 0, 2] = (Y_KV * (w_des - wz)).clamp(-Y_TMAX, Y_TMAX)
            scene.runner.set_external_force_and_torque(f, tq, env_ids=all_ids)
            env.step(no_action)
            if (i + 1) % 150 == 0:
                rp = (scene.runner.data.root_pos_w - scene.env_origins)[0]
                rq = scene.runner.data.root_quat_w[0]
                print(f"[solve] {tag} tel @{i + 1}: q={q0():+.4f} yaw={yaw0():+.4f} "
                      f"wz={float(wz[0]):+.4f} tq={float(tq[0, 0, 2]):+.5f} "
                      f"fz={float(f[0, 0, 2]):+.3f} vz={float(vz[0]):+.4f} "
                      f"xy=({float(rp[0]):+.4f},{float(rp[1]):+.4f}) "
                      f"quat=({float(rq[0]):+.4f},{float(rq[1]):+.4f},"
                      f"{float(rq[2]):+.4f},{float(rq[3]):+.4f})", flush=True)
            ok = (abs(q0() - q_tgt) <= q_tol and abs(float(wrap(yaw() - yaw_tgt)[0])) <= yaw_tol
                  and abs(float(vz[0])) < 0.03 and abs(float(wz[0])) < 0.15)
            done = done + 1 if ok else 0
            if done >= hold_streak:
                break
        print(f"[solve] {tag}: runner at q={q0():+.4f} yaw={yaw0():+.3f} "
              f"(targets {q_tgt:+.4f}/{yaw_tgt:+.3f}, {i + 1} servo steps)", flush=True)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(90)
    print(f"[solve] readback (seed {args.seed}): q={q0():+.4f} (bottom stop), "
          f"yaw={yaw0():+.3f} rad (randomized), pin_d={pd0():+.4f} (seated), "
          f"q_park={c.q_park:.4f} q_block={c.q_block:.4f}", flush=True)
    p = (scene.pin.data.root_pos_w - scene.env_origins)[0]
    sp = (scene.spare.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback: pin=({float(p[0]):+.4f},{float(p[1]):+.4f},"
          f"{float(p[2]):+.4f}) spare=({float(sp[0]):+.3f},{float(sp[1]):+.3f},"
          f"{float(sp[2]):+.3f})", flush=True)
    assert abs(q0()) <= 0.010, "runner must start on the bottom stop"
    assert pd0() <= 0.02, "pin must start seated at the mast axis"
    assert abs(float(p[2]) - c.pin_rest_z) <= 0.004, "pin must rest on the channel floor"
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: pull the transit pin out ------------------------------------
    # HELD carry: a bare horizontal CoM pull self-locks once the pin's CoM passes the
    # channel-floor edge (tail corner presses the roof, N grows without bound near full
    # extraction — the classic drawer jam; first forge run stalled at pin_d=0.033).
    # A pinch grasp holds the pin level, so the honest force model is mg feedforward
    # + z/y PD AT THE CoM (zero pitch moment) while the x-servo slides it out.
    zr = c.pin_rest_z + 0.0005
    done, i = 0, 0
    for i in range(1500):
        p = scene.pin.data.root_pos_w - scene.env_origins
        v = scene.pin.data.root_lin_vel_w
        x, vx = p[:, 0], v[:, 0]
        v_des = (P_KP * (P_XTGT - x)).clamp(-P_VCAP, P_VCAP)
        fx = P_KV * (v_des - vx)
        fx = fx + torch.where(vx.abs() < 0.02, P_FF * v_des.sign(), torch.zeros_like(fx))
        f_world = torch.zeros(n, 3, device=device)
        f_world[:, 0] = fx.clamp(-P_FMAX, P_FMAX)
        # kp*dt/m = 4/(120*0.06) = 0.56 < 1 (one-substep wrench delay bound)
        f_world[:, 1] = (4.0 * (0.0 - p[:, 1]) - 0.4 * v[:, 1]).clamp(-0.5, 0.5)
        f_world[:, 2] = (c.pin_mass * G + 4.0 * (zr - p[:, 2]) - 0.4 * v[:, 2]).clamp(0.0, 1.5)
        f_body = quat_apply_inverse(scene.pin.data.root_quat_w, f_world)
        scene.pin.set_external_force_and_torque(f_body.unsqueeze(1), zero_w, env_ids=all_ids)
        env.step(no_action)
        if (i + 1) % 200 == 0:
            print(f"[solve] P1 telemetry @{i + 1}: pin_d={pd0():+.4f} "
                  f"fx={float(fx[0]):+.3f}", flush=True)
        done = done + 1 if pd0() >= c.pull_clear else 0  # break on POSITION readback
        if done >= 3:
            break
    scene.pin.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(90)  # the freed pin falls and settles — wrench OFF
    print(f"[solve] P1: pin pulled clear in {i + 1} servo steps, "
          f"pin_d={pd0():+.4f} — wrench OFF", flush=True)
    assert pd0() >= 0.12, "pin must be clear of the mast"
    assert float(scene.pin_latch[0]) > 0.5, "pin latch must have fired"
    report("P1-done")
    s1 = print_score("P1 transit pin extracted")
    assert s1 >= max(s0, 0.29), "pin extraction credit missing"

    # ---------------- phase 2: spin the lug to the gap heading (+x) ------------------------
    runner_servo(0.0, 0.0, R_VUP, "P2", max_steps=900)
    assert abs(float(wrap(yaw())[0])) <= 0.10, "lug must point at the gap"
    s2 = print_score("P2 lug aligned to the keyed gap")
    assert s2 >= s1 - 1e-6, "score decreased across P2"

    # ---------------- phase 3: hoist through the keyed gap ---------------------------------
    runner_servo(c.hoist_q, 0.0, R_VUP, "P3", max_steps=1500)
    assert q0() >= c.hoist_q - 0.010, "runner must reach the hoist height"
    assert float(scene.pass_latch[0]) > 0.5, "pass latch must have fired"
    s3 = print_score("P3 lug hoisted through the gap, above the shelf")
    assert s3 >= max(s2, 0.59), "shelf-pass credit missing"

    # ---------------- phase 4: the bayonet TWIST (+90 deg above the shelf) -----------------
    runner_servo(c.hoist_q, math.pi / 2, R_VUP, "P4", max_steps=900)
    assert abs(float(wrap(yaw() - math.pi / 2)[0])) <= 0.10, "lug must be twisted ~90 deg"
    s4 = print_score("P4 twisted above the shelf")
    assert s4 >= s3 - 1e-6, "score decreased across P4"

    # ---------------- phase 5: lower onto the ring, release --------------------------------
    # Aim 2 mm BELOW the physical seat: the servo presses the lug gently onto the
    # ring (net ~0.1 N down), then the wrench goes OFF and it rests parked.
    runner_servo(c.q_park - 0.002, math.pi / 2, R_VDN, "P5", max_steps=1200,
                 q_tol=0.004, hold_streak=20)
    runner_wrench_off()
    step(60)
    print(f"[solve] P5 released: q={q0():+.4f} (park {c.q_park:+.4f}), "
          f"yaw={yaw0():+.3f} — wrench OFF", flush=True)
    assert abs(q0() - c.q_park) <= c.park_tol, "lug must rest ON the ring, wrench off"
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P5-ringdown")
    s5 = print_score("P5 lug seated on the ring, parked hands-off")
    assert s5 >= s4 - 1e-6, "score decreased across P5"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after park + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                rv = float(scene.runner.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: q={q0():+.4f} "
                      f"runner_v={rv:.4f} parked={bool(scene.parked_now()[0])} "
                      f"aligned={bool(scene.aligned_now()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
