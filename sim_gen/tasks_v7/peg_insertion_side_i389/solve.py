"""Teleport solution for LineBoreScene (sim_gen task `peg_insertion_side_i389`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY; every load-bearing interaction is contact dynamics:

  1. SEAT (transport + contact): the silver coupler is teleported from its floor slot
     to a hover point centered over the pocket, 15 mm above the seat, then DROPPED —
     gravity + rail/pylon contacts key it into the pocket (the same funnel a Franka
     place uses). Never spawned seated below the drop.
  2. THREAD (transport + contact): the shaft is teleported to a bore-aligned pose
     with its tip ~6 mm OUTSIDE the near pylon's window — no gate is pre-entered —
     then a held-carry PD wrench (gravity feedforward + position servo + axis-
     alignment torque, the stand-in for the Franka grip) drags it through window ->
     seated coupler channel -> far window along a lag-clamped carrot. All three gate
     passages happen through contact; a misalignment would stall against real
     geometry (the carrot cannot outrun the shaft).
  3. RELEASE + PERSIST: the wrench is cleared, everything settles on the sills by
     gravity, and success() must hold hands-off for >= 3.5 simulated seconds.

Servo notes: `enable_external_forces_every_iteration` is set in the scene; gains
respect the one-substep wrench delay (kd*dt/m = 4/(120*0.15) = 0.22 << 1; angular
kdw*dt/I = 0.01/(120*3.7e-4) = 0.23 << 1). On stall the position GAIN escalates,
not the force cap. The house wrench convention: forces/torques are handed to
set_external_force_and_torque in the CURRENT body frame, re-encoded every step.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing — the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() holds through
the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i389.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.line_bore")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)
    x_out = c.gap / 2 + c.pyl_t
    half = c.shank / 2
    seat_z = c.deck[2] + c.block_hz / 2
    flush_cx = -x_out + half  # shaft center x when the head is flush (entry -x)

    env.reset(seed=args.seed)  # seed AFTER build (the EnvCfg.build reseed trap)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench() -> None:
        scene.shaft.set_external_force_and_torque(zero_rows, zero_rows,
                                                  env_ids=all_ids)

    def wrench(f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        """World wrench on the shaft, encoded in its CURRENT body frame (the house
        convention), re-computed every step."""
        from isaaclab.utils.math import quat_apply_inverse

        q = scene.shaft.data.root_link_quat_w
        scene.shaft.set_external_force_and_torque(
            quat_apply_inverse(q, f_world).unsqueeze(1),
            quat_apply_inverse(q, t_world).unsqueeze(1), env_ids=all_ids)

    def report(tag: str) -> None:
        m = scene.shaft_metrics()
        b = scene.to_canon(scene.coupler.data.root_pos_w)[0]
        p = m["p"][0]
        print(f"[solve] {tag:12s} | shaft=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) rem={float(m['remaining'][0]):.3f}"
              f" coupler=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})"
              f" seated={bool(scene.seated()[0])}"
              f" engaged={bool(m['engaged'][0])} span={bool(m['spanning'][0])}"
              f" flush={bool(m['flush'][0])} enc={bool(m['encircle'][0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place_canon(body, canon_xyz, quat_rel_fixture=None, settle_steps: int = 0) -> None:
        """Teleport (TRANSPORT through free space) a body to a fixture-canonical pose."""
        from isaaclab.utils.math import quat_mul

        canon = torch.tensor(canon_xyz, device=device).view(1, 3).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canon_to_world(canon.clone())
        fq = scene.fixture.data.root_quat_w
        if quat_rel_fixture is None:
            st[:, 3:7] = fq
        else:
            qr = torch.tensor(quat_rel_fixture, device=device).view(1, 4).expand(n, 4)
            st[:, 3:7] = quat_mul(fq, qr)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    ss = float(scene.slot_sign[0])
    print(f"[solve] layout readback (seed {args.seed}): coupler at "
          f"{'+y' if ss > 0 else '-y'} slot", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: seat the coupler (drop + contact keying) --------------------
    for attempt in range(4):
        place_canon(scene.coupler, (0.0, 0.0, seat_z + 0.015), settle_steps=150)
        if bool(scene.seated()[0]):
            break
        print(f"[solve] seat attempt {attempt} missed — re-dropping", flush=True)
    report("seat")
    assert bool(scene.seated()[0]), "coupler never seated in the pocket"
    s1 = print_score("P1 coupler dropped + keyed into the pocket")
    assert s1 >= 0.30 - 1e-6 and s1 >= s0 - 1e-6, "seat credit missing"

    # ---------------- phase 2: thread the shaft (held-carry PD through 3 gates) ------------
    # transport: bore-aligned, tip 6 mm OUTSIDE the near window — no gate pre-entered
    start_cx = -x_out - 0.006 - half
    place_canon(scene.shaft, (start_cx, 0.0, c.win_z))
    kp, kd = 60.0, 4.0
    kr, kdw = 0.4, 0.010
    f_cap = 8.0
    mg = c.shaft_mass * 9.81
    carrot_x = start_cx
    v_carrot = 0.030 / 120.0  # 30 mm/s
    target_cx = flush_cx + 0.004  # drive the head gently onto the face
    last_prog, last_check = -1.0, 0
    fx_axis = None
    for i in range(2400):
        p_c = scene.to_canon(scene.shaft.data.root_pos_w)
        m = scene.shaft_metrics()
        rem = float(m["remaining"][0])
        if rem <= 0.005 and bool(m["spanning"][0]):
            print(f"[solve] thread: flush reached @step {i} (rem={rem:.4f})", flush=True)
            break
        # lag-clamped carrot: never outruns the shaft by more than 12 mm
        if carrot_x - float(p_c[0, 0]) < 0.012:
            carrot_x = min(carrot_x + v_carrot, target_cx)
        carrot = torch.tensor([carrot_x, 0.0, c.win_z], device=device).view(1, 3)
        tgt_w = scene.canon_to_world(carrot.expand(n, 3).clone())
        f = kp * (tgt_w - scene.shaft.data.root_pos_w) \
            - kd * scene.shaft.data.root_lin_vel_w
        f[:, 2] += mg  # gravity feedforward (held carry)
        f_norm = f.norm(dim=-1, keepdim=True)
        f = f * (f_norm.clamp(max=f_cap) / f_norm.clamp_min(1e-9))
        # axis-alignment torque (cylinder is roll-symmetric: align body x only)
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        a_cur = quat_apply(scene.shaft.data.root_quat_w, ex)
        if fx_axis is None:
            fx_axis = quat_apply(scene.fixture.data.root_quat_w, ex)
        tq = kr * torch.cross(a_cur, fx_axis, dim=-1) \
            - kdw * scene.shaft.data.root_ang_vel_w
        wrench(f, tq)
        env.step(no_action)
        if i - last_check >= 240:  # 2 s: stalled? escalate the GAIN, not the cap
            prog = float(p_c[0, 0])
            if prog - last_prog < 0.005:
                kp = min(kp * 1.5, 240.0)
                print(f"[solve] thread stall @step {i} x={prog:+.3f} rem={rem:.3f}"
                      f" kp->{kp:.0f}", flush=True)
            last_prog, last_check = prog, i
    clear_wrench()
    report("thread")
    m = scene.shaft_metrics()
    assert bool(m["spanning"][0]) and float(m["remaining"][0]) <= c.flush_tol, \
        "shaft never threaded to flush"
    s2 = print_score("P2 shaft threaded through window/channel/window to flush")
    assert s2 >= s1 - 1e-6 and s2 >= 0.70 - 1e-6, "thread credit missing"

    # ---------------- phase 3: release, settle, judge --------------------------------------
    step(150)
    report("release")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seat+thread+release)", flush=True)
        os._exit(1)
    s3 = print_score("P3 released + settled: assembly complete")
    assert s3 >= s2 - 1e-6, "score decreased at release"

    # ---------------- phase 4: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                mm = scene.shaft_metrics()
                print(f"[solve] persist flicker @step {i}: "
                      f"seated={bool(scene.seated()[0])} "
                      f"inst={bool(scene.installed()[0])} "
                      f"enc={bool(mm['encircle'][0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"slin={float(scene.shaft.data.root_lin_vel_w[0].norm()):.4f} "
                      f"sang={float(scene.shaft.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.5 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
