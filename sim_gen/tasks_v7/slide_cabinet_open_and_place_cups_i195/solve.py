"""Teleport solution for WedgeGateCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i195`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY (staging objects in the open, in front of the
cabinet): the cups are parked out of the work lane, the wedge is set down aimed at the
slit, each cup is set down at the mouth of its push lane. NOTHING is teleported into
the rubric state — the containment band lies behind the closed gate and the only way
a cup gets there in this solve is by SLIDING along the floor through the propped
doorway under contact dynamics. Every load-bearing interaction is physical:

  - the GATE is never touched directly (it has no handle and gravity returns it):
    it is jacked open by driving the ramp WEDGE into the floor slit with a
    velocity-regulated external force — the incline lifts the gate up its
    spawn-authored track, the flat plateau then holds it with ZERO applied force
    (asserted hands-off before any cup moves);
  - each CUP is pushed along the floor through the propped gap with a
    tip-safe force cap (cap 0.50 N < 0.62 N tipping threshold);
  - the wedge is pulled back OUT the same way; the gate falls closed ON ITS OWN.

Servo design: outer position loop -> v_des (capped), inner PI velocity loop -> world
force converted world->body EVERY step (`quat_apply_inverse` — bodies may yaw; the
wrench API is body-frame). The proportional gain respects the one-substep wrench
delay (KV*dt/m: wedge 8/(120*0.18)=0.37, cup 3/(120*0.07)=0.36, both < 1); the slow
INTEGRAL term supplies the break-away force a pure P-loop can never reach (the
gain-stall trap: stall force of a P-loop is only KV*V_CAP), and is dumped the moment
the target is near so it cannot run away when resistance drops.

PLAN (read-only, from scene.describe()):
  P0 settle + readbacks (gate closed on its stop, score ~0),
  P1 park the cups clear, stage the wedge aimed tip-first, drive it in until the
     plateau spans the gate footprint; force OFF; assert the gate SITS at prop
     height hands-free,
  P2 slide cup_a through the propped gap beside the wedge (floor push, x -0.010),
  P3 slide cup_b through beside it (x +0.070),
  P4 pull the wedge back out to the front; the gate falls closed by itself,
  P5 ring down until success() holds 120 consecutive steps,
  P6 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds after
the hands-off hold.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i195.solve --headless [--seed N]
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

DT = 1.0 / 120.0
KP_POS = 3.0        # outer position loop -> desired velocity (1/s)

# Wedge servo (mass 0.18: KV*DT/m = 8/21.6 = 0.37; stall force comes from KI).
WEDGE_V_CAP = 0.05
WEDGE_KV = 8.0
WEDGE_KI = 60.0     # N/(m/s)/s -> ~3 N/s escalation at full stall
WEDGE_F_MAX = 12.0  # >> ~5 N ramp-lift estimate, << anything destructive

# Cup servo (mass 0.07: KV*DT/m = 3/8.4 = 0.36; cap BELOW the 0.62 N tip threshold).
CUP_V_CAP = 0.05
CUP_KV = 3.0
CUP_KI = 25.0
CUP_F_MAX = 0.50

# Staging poses (env frame; all in the open, in front of the cabinet).
PARK_A = (-0.22, 0.32)      # cup parking, clear of wedge home + lane + staging row
PARK_B = (-0.32, 0.32)
WEDGE_START_Y = 0.170       # pre-push wedge root y (tip at +0.075, aimed at slit)
WEDGE_IN_Y = -0.075         # inserted: plateau [-0.030,+0.020] spans gate [-0.016,-0.002]
WEDGE_OUT_Y = 0.170         # extracted: >> clear_y_min 0.115
CUP_START_Y = 0.150         # push-lane mouth (clear of the inserted wedge tail +0.020)
CUP_IN_Y = -0.130           # push target, well inside the containment band
CUP_A_X = -0.010            # lane beside the wedge (wedge right edge -0.050)
CUP_B_X = 0.070


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wedge_gate_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def gate_j() -> float:
        return float(scene.gate_j()[0])

    def wedge_y() -> float:
        return float(scene.wedge_local()[0, 1])

    def report(tag: str) -> None:
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f}) in={bool(scene.cup_inside(nm)[0])}")
        print(f"[solve] {tag:12s} | gate_j={gate_j():+.4f} "
              f"closed={bool(scene.gate_closed()[0])} | wedge_y={wedge_y():+.3f} "
              f"clear={bool(scene.wedge_clear()[0])} | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 600) -> None:
        """Hands-off until the scene reads persistently still (or budget runs out)."""
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    def teleport(body, x: float, y: float, z: float, tag: str) -> None:
        """TRANSPORT: set-down at (x, y, z) env-frame, zero velocity, identity yaw."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = z
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        print(f"[solve] transport {tag} -> set down at ({x:+.3f},{y:+.3f},{z:+.3f})",
              flush=True)

    def servo_push(body, y_target: float, tag: str, *, v_cap: float, kv: float,
                   ki: float, f_max: float, tol: float = 0.005,
                   max_steps: int = 3000) -> None:
        """Slide `body` along world y to `y_target` with a velocity-regulated PI
        external force (world->body converted every step; the wrench is re-set every
        step). The integrator supplies break-away force and is DUMPED once near the
        target (no wind-up runaway when resistance drops). Zero wrench + ring down at
        the end."""
        acc = torch.zeros(n, device=device)
        done = 0
        i = 0
        for i in range(max_steps):
            y = (body.data.root_pos_w - scene.env_origins)[:, 1]
            v = body.data.root_lin_vel_w[:, 1]
            err = y_target - y
            v_des = (KP_POS * err).clamp(-v_cap, v_cap)
            near = err.abs() <= tol
            acc = (acc + ki * (v_des - v) * DT).clamp(-f_max, f_max)
            acc = torch.where(near, torch.zeros_like(acc), acc)
            fy = (acc + kv * (v_des - v)).clamp(-f_max, f_max)
            slow = v.abs() < 0.02
            fy = torch.where(near & slow, torch.zeros_like(fy), fy)
            f_world = torch.zeros(n, 3, device=device)
            f_world[:, 1] = fy
            f_body = quat_apply_inverse(body.data.root_quat_w, f_world)
            body.set_external_force_and_torque(f_body.unsqueeze(1), zero_w,
                                               env_ids=all_ids)
            env.step(no_action)
            if bool(near[0]) and bool(slow[0]):
                done += 1
                if done >= 12:
                    break
            else:
                done = 0
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(30)
        yf = float((body.data.root_pos_w - scene.env_origins)[0, 1])
        print(f"[solve] {tag}: pushed to y={yf:+.4f} (target {y_target:+.4f}, "
              f"{i + 1} servo steps)", flush=True)
        assert abs(yf - y_target) <= tol + 0.004, \
            f"{tag}: push stalled at y={yf:+.4f} (target {y_target:+.4f})"

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    j0 = gate_j()
    print(f"[solve] readback (seed {args.seed}): gate_j={j0:+.4f} (rest {c.gate_j_lo}), "
          f"wedge=({float(scene.wedge_local()[0, 0]):+.3f},{wedge_y():+.3f})", flush=True)
    assert abs(j0 - c.gate_j_lo) <= 0.004, "gate must settle closed on its lower stop"
    for nm in c.cup_names:
        p = scene._cup_local(nm)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: park cups, stage + drive in the wedge -----------------------
    # Park the cups clear of the wedge home, the staging row, and the work lane
    # (transport in the open — nowhere near the rubric band).
    teleport(scene.cups[c.cup_names[0]], *PARK_A, c.cup_h / 2 + 0.003, "cup_a->park")
    teleport(scene.cups[c.cup_names[1]], *PARK_B, c.cup_h / 2 + 0.003, "cup_b->park")
    # Stage the wedge in front of the slit lane, tip aimed at the doorway.
    teleport(scene.wedge, c.wedge_insert_x, WEDGE_START_Y, 0.0035, "wedge->pre-push")
    step(30)
    # Drive it in: the incline jacks the gate, the plateau ends up under the whole
    # gate footprint.
    servo_push(scene.wedge, WEDGE_IN_Y, "P1-insert", v_cap=WEDGE_V_CAP, kv=WEDGE_KV,
               ki=WEDGE_KI, f_max=WEDGE_F_MAX)
    settle(300)
    jp = gate_j()
    print(f"[solve] P1: gate propped at j={jp:+.4f} (deep_j_min {c.deep_j_min}), "
          f"wedge_y={wedge_y():+.4f} — ZERO applied force", flush=True)
    assert jp >= c.deep_j_min, "plateau must hold the gate at full prop height hands-free"
    # Hands-off hold: prove the prop is static (nothing regulated is holding it).
    step(240)
    assert gate_j() >= c.deep_j_min, "gate must STAY propped with zero applied force"
    report("P1-propped")
    s1 = print_score("P1 wedge in, gate propped hands-free")
    assert s1 >= s0 - 1e-6 and s1 >= 0.29, "P1 must earn both gate-lift latches"

    # ---------------- phase 2: slide cup_a through the propped gap -------------------------
    teleport(scene.cups[c.cup_names[0]], CUP_A_X, CUP_START_Y, c.cup_h / 2 + 0.003,
             "cup_a->lane")
    step(30)
    servo_push(scene.cups[c.cup_names[0]], CUP_IN_Y, "P2-slide-a", v_cap=CUP_V_CAP,
               kv=CUP_KV, ki=CUP_KI, f_max=CUP_F_MAX, tol=0.006)
    settle(300)
    assert bool(scene.cup_inside(c.cup_names[0])[0]), \
        "cup_a must stand upright inside after the floor slide"
    report("P2-cup_a")
    s2 = print_score("P2 cup_a slid inside through the gap")
    assert s2 >= s1 - 1e-6 and s2 >= 0.45, "cup_a containment credit missing"

    # ---------------- phase 3: slide cup_b through beside it -------------------------------
    teleport(scene.cups[c.cup_names[1]], CUP_B_X, CUP_START_Y, c.cup_h / 2 + 0.003,
             "cup_b->lane")
    step(30)
    servo_push(scene.cups[c.cup_names[1]], CUP_IN_Y, "P3-slide-b", v_cap=CUP_V_CAP,
               kv=CUP_KV, ki=CUP_KI, f_max=CUP_F_MAX, tol=0.006)
    settle(300)
    assert bool(scene.cup_inside(c.cup_names[1])[0]), \
        "cup_b must stand upright inside after the floor slide"
    assert bool(scene.both_inside()[0]), "both cups must now be inside"
    report("P3-cup_b")
    s3 = print_score("P3 both cups inside; gate still propped")
    assert s3 >= s2 - 1e-6 and s3 >= 0.62, "cup_b containment credit missing"

    # ---------------- phase 4: pull the wedge out; the gate falls closed -------------------
    servo_push(scene.wedge, WEDGE_OUT_Y, "P4-extract", v_cap=WEDGE_V_CAP + 0.01,
               kv=WEDGE_KV, ki=WEDGE_KI, f_max=WEDGE_F_MAX)
    settle(300)
    print(f"[solve] P4: wedge_y={wedge_y():+.4f} (clear_y_min {c.clear_y_min}), "
          f"gate fell to j={gate_j():+.4f}", flush=True)
    assert bool(scene.wedge_clear()[0]), "wedge must be fully clear in front"
    assert bool(scene.gate_closed()[0]), "gate must fall fully closed on its own"
    report("P4-closed")
    s4 = print_score("P4 wedge out, gate fell closed by itself")
    assert s4 >= s3 - 1e-6 and s4 >= 0.72, "restored-mechanism credit missing"

    # ---------------- phase 5: ring down to persistent success -----------------------------
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
    s5 = print_score("P5 rung down, success held 1 s")
    assert s5 >= s4 - 1e-6, "score decreased across P5"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: gate_j={gate_j():+.4f} "
                      f"both={bool(scene.both_inside()[0])} "
                      f"closed={bool(scene.gate_closed()[0])} "
                      f"clear={bool(scene.wedge_clear()[0])} "
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
