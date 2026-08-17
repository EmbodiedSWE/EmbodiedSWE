"""Teleport solution for FloodlightEjectScene (sim_gen task `light_bulb_out_i204`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact dynamics:
  1. ACQUIRE + ALIGN (transport): the push rod is teleported from the floor to free
     air behind the housing's rear service port, axis on the port axis — pure
     transport across open, reachable space. From here on it is held by a 6-DOF
     wrench servo (the virtual grasp): position spring + velocity damping + gravity
     feedforward, alignment torque + transverse-only angular damping (the axial
     inertia of a 6 mm rod is 1e-6 — damping that axis would be discretely unstable).
  2. INSERT (dynamics): the servo walks the rod tip through the stepped countersink,
     the guide tube and the port hole in the back wall. Contact with the port chain —
     not a teleport — does the final centering.
  3. PUSH (dynamics): the tip meets the bulb's rear pole and pushes it forward over
     the amber retention ridge. The ridge is a real quasi-static threshold
     (~0.43 N for the 50 g sphere); the servo's position lead is clamped so the
     contact force stays bounded (max ~1.2 N ridge-breaking force, cap 5 N).
  4. EJECT (gravity): past the ridge the shroud floor is a downhill ramp; the bulb
     rolls out of the mouth on its own and falls ballistically into the disposal
     bin. Nothing touches it.
  5. RELEASE (dynamics): the wrench is zeroed; the rod comes to rest supported
     inside the guide tube (its CoM lies within the supported span). Hands-off
     persistence follows.

Servo stability (external-wrench plant recipe; `enable_external_forces_every_iteration`
is set in the scene): kd*dt/m = 2.4/(120*0.06) = 0.33 < 1 and
ka*dt/It = 0.004/(120*4.5e-4) = 0.074 < 1 against the AUTHORED rod inertia; wrenches
act one substep late, so both stay well under 1.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.light_bulb_out_i204.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.floodlight_eject")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)
    dt = 1.0 / 120.0
    half = c.rod_len / 2
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    # Servo gains (see module docstring for the discrete-stability audit).
    KP, KD, F_CAP = 24.0, 2.4, 5.0
    KR, KA, T_CAP = 0.04, 0.004, 0.10
    LEAD = 0.05  # anti-windup: p_des never leads the plant by more (1.2 N max spring)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def hpose() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.housing.data.root_pos_w, scene.housing.data.root_quat_w

    def tip_local() -> torch.Tensor:
        """(N, 3): the WORKING (+z) rod tip in the housing frame."""
        a = quat_apply(scene.rod.data.root_quat_w, ez)
        return scene.housing_local(scene.rod.data.root_pos_w + a * half)

    def rod_wrench(p_des_local: torch.Tensor) -> None:
        """One servo tick: world-frame wrench toward `p_des_local` (housing frame,
        rod CENTER) with the rod axis servoed onto the housing +x axis, encoded into
        the rod's body frame (the house convention for applied wrenches)."""
        hp, hq = hpose()
        p_des_w = hp + quat_apply(hq, p_des_local)
        p = scene.rod.data.root_pos_w
        v = scene.rod.data.root_lin_vel_w
        f = KP * (p_des_w - p) - KD * v
        f[:, 2] += c.rod_mass * 9.81  # gravity feedforward (no sag)
        fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f = f * (fn.clamp(max=F_CAP) / fn)
        u = quat_apply(scene.rod.data.root_quat_w, ez)
        ex = torch.zeros(n, 3, device=device)
        ex[:, 0] = 1.0
        u_des = quat_apply(hq, ex)
        t = KR * torch.cross(u, u_des, dim=-1)
        w = scene.rod.data.root_ang_vel_w
        w_perp = w - (w * u).sum(-1, keepdim=True) * u
        t = t - KA * w_perp
        tn = t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        t = t * (tn.clamp(max=T_CAP) / tn)
        rq = scene.rod.data.root_link_quat_w
        scene.rod.set_external_force_and_torque(
            quat_apply_inverse(rq, f).unsqueeze(1),
            quat_apply_inverse(rq, t).unsqueeze(1), env_ids=all_ids)

    def release_rod() -> None:
        scene.rod.set_external_force_and_torque(zero_rows, zero_rows,
                                                env_ids=all_ids)

    def hold(p_des_local: torch.Tensor, k: int) -> None:
        for _ in range(k):
            rod_wrench(p_des_local)
            env.step(no_action)

    def des_local(ct: float) -> torch.Tensor:
        d = torch.zeros(n, 3, device=device)
        d[:, 0] = ct
        d[:, 2] = c.axis_z
        return d

    def push_to(tip_target: float, tag: str, max_steps: int,
                rate: float = 0.03) -> float:
        """Walk the rod tip (housing x) to `tip_target` at `rate` m/s with the lead
        clamp; returns the final tip x. Servo stays engaged afterwards at the
        target."""
        ct_des = float(tip_local()[0, 0]) - half  # start from where the plant is
        tx = float(tip_local()[0, 0])
        for i in range(max_steps):
            tx = float(tip_local()[0, 0])
            if tx >= tip_target:
                break
            ct_des = min(ct_des + rate * dt, tx - half + LEAD,
                         tip_target - half + 0.005)
            rod_wrench(des_local(ct_des))
            env.step(no_action)
            if (i + 1) % 240 == 0:
                print(f"[solve] push {tag}: tip_x={tx:+.3f} -> {tip_target:+.3f}",
                      flush=True)
        return tx

    def report(tag: str) -> None:
        t = tip_local()[0]
        b = scene.bulb_local()[0]
        print(f"[solve] {tag:14s} | tip=({float(t[0]):+.3f},{float(t[1]):+.3f},"
              f"{float(t[2]):+.3f})"
              f" bulb=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})"
              f" latches=({float(scene.probe_latch[0]):.0f},"
              f"{float(scene.unseat_latch[0]):.0f},{float(scene.eject_latch[0]):.0f})"
              f" in_bin={bool(scene.in_bin()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    b = scene.bulb_local()[0]
    r = (scene.rod.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"bulb_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
          f"rod=({float(r[0]):+.3f},{float(r[1]):+.3f},{float(r[2]):.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: acquire + align (transport), insert to mid-tube -------------
    # Teleport the rod to free air behind the port, axis on the port axis: center at
    # housing-local (-0.44, 0, axis_z) => working tip at -0.29, just outside the
    # countersink entry plane (-0.285). Pure transport; zero velocity.
    hp, hq = hpose()
    qy90 = torch.zeros(n, 4, device=device)
    qy90[:, 0] = math.cos(math.pi / 4)
    qy90[:, 2] = math.sin(math.pi / 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = hp + quat_apply(hq, des_local(-0.44))
    st[:, 3:7] = quat_mul(hq, qy90)
    scene.rod.write_root_state_to_sim(st, all_ids)
    hold(des_local(-0.44), 30)  # let the servo latch on in free air
    tx = push_to(-0.20, "insert", 1800)
    t = tip_local()[0]
    assert tx > -0.21, f"insertion stalled at tip_x={tx:+.3f}"
    assert abs(float(t[1])) < 0.010 and abs(float(t[2]) - c.axis_z) < 0.020, \
        f"tip off the port axis inside the tube: y={float(t[1]):+.3f} z={float(t[2]):+.3f}"
    report("mid-tube")
    s1 = print_score("P1 rod aligned + inserted to mid-tube (contact dynamics)")
    assert s1 >= s0 - 1e-6

    # ---------------- phase 2: tip through the port hole (probe credit) --------------------
    tx = push_to(-0.162, "probe", 900)
    report("through port")
    assert bool(scene.probe_latch[0] > 0.5), "probe latch did not fire inside the port"
    s2 = print_score("P2 rod tip through the service port")
    assert s2 >= s1 - 1e-6 and s2 >= 0.15 - 1e-6

    # ---------------- phase 3: push the bulb over the ridge (contact) ----------------------
    tx = push_to(-0.085, "ridge", 1800, rate=0.02)
    report("pushed")
    assert bool(scene.unseat_latch[0] > 0.5), \
        f"bulb not pushed past the ridge (tip_x={tx:+.3f})"
    s3 = print_score("P3 bulb pushed over the retention ridge")
    assert s3 >= s2 - 1e-6 and s3 >= 0.40 - 1e-6

    # ---------------- phase 4: gravity ejection into the bin -------------------------------
    tgt = des_local(-0.085 - half)
    ok = False
    for _ in range(720):  # up to 6 s: roll down the ramp, fly, land in the bin
        rod_wrench(tgt)
        env.step(no_action)
        if bool(scene.eject_latch[0] > 0.5) and bool(scene.in_bin()[0]):
            ok = True
            break
    report("ejected")
    assert bool(scene.eject_latch[0] > 0.5), "bulb never crossed the mouth transit window"
    assert ok, "bulb did not land inside the bin"
    s4 = print_score("P4 bulb ejected by gravity into the bin")
    assert s4 >= s3 - 1e-6 and s4 >= 0.65 - 1e-6

    # ---------------- phase 5: release the rod, settle to success --------------------------
    release_rod()
    for _ in range(48):  # up to 12 s (bin walls + rolling friction park the sphere)
        if bool(scene.success()[0]):
            break
        step(30)
    report("released")
    s5 = print_score("P5 rod released (rests in the tube), bulb settled")
    assert s5 >= s4 - 1e-6, "score decreased across release/settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after insert+push+eject+settle)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.5 simulated seconds, hands-off) -----------
    held, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            held = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                bl = scene.bin_local()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"in_bin={bool(scene.in_bin()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"bin_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                      f"{float(bl[2]):+.3f}) "
                      f"blin={float(scene.bulb.data.root_lin_vel_w[0].norm()):.4f} "
                      f"bang={float(scene.bulb.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s6 = print_score("P6 persistence 3.5 s")
    ok = held and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
