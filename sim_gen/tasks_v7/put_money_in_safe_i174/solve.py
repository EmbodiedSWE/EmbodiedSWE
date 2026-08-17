"""Teleport solution for DepositVaultScene (sim_gen task `put_money_in_safe_i174`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. OPENING the hatch (torque servo, real pivot + contact dynamics): a speed-capped
   PD torque about the vertical pivot (`set_external_force_and_torque` on the
   turret body — pure z torque, what a hand orbiting the crank knob applies) turns
   the turret from its seal stop to the 170 deg receive stop. The joint limit
   arrests it; the open credit is produced by the pivot dynamics, never written.
2. TRANSPORT (teleport): one root-state write carries the brick from its stand
   through free air to a hover just OUTSIDE the deposit window, long side leading,
   zero velocity. Open sky — no contact interaction is bypassed and no rubric
   clause is satisfied by the write (the brick is outside the vault).
3. INSERTION (guided force push through the aperture): a PD force + gravity
   feedforward (a firm force-limited grasp/push) drives the brick through the
   window; it slides across the sill onto the transfer shelf and into the open
   pocket. The `loaded` latch is produced by contact geometry after release.
4. DELIVERY (the mechanism does the work): the same speed-capped torque servo
   turns the turret back toward the seal stop; the pocket walls sweep the brick
   across the shelf, past the half-disc edge it loses support and FALLS into the
   bin — pure contact dynamics; nothing ever touches the brick directly after
   insertion. The servo finishes on the seal stop.
5. HANDS-OFF: all wrenches are zeroed; success must hold through >= 3.3 simulated
   seconds untouched — the brick rests in the bin, the hatch stays sealed.

Order is forced: sealed hatch refuses the brick (smoke proves it with a real
push); the pocket only delivers by rotation; the seal returns after the sweep.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.put_money_in_safe_i174.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.deposit_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def ang() -> float:
        return float(scene.turret_angle_deg()[0])

    def report(tag: str) -> None:
        p = scene._vault_local(scene.brick.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | theta={ang():+7.2f}deg "
              f"brick=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"in_bin={bool(scene.brick_in_bin()[0])} "
              f"loaded={bool(scene.brick_loaded()[0])} "
              f"sealed={bool(scene.sealed()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def turret_torque(tau: float) -> None:
        """Constant torque about the vertical pivot (+ opens toward receive).
        Pure z — immune to the pod's wrench frame-drag quirk."""
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        t_b = quat_apply_inverse(scene.turret.data.root_quat_w, t_w)
        scene.turret.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))

    def crank_to(target_deg: float, *, kp: float = 2.5, kd: float = 0.5,
                 clamp: float = 1.2, w_max: float = 2.0, steps: int = 1200,
                 tol: float = 2.0, label: str = "") -> None:
        """Speed-capped PD torque servo about the pivot: what a hand on the crank
        knob does. Gains sized for the turret's ~0.02 kg m^2 pivot inertia and the
        1-substep wrench delay (kd*dt/I ~= 0.2); the speed cap keeps the sweep
        slow so a carried brick is pushed, not flung, and the stop cannot be
        vaulted (constant-torque coast overshoot memory)."""
        streak = 0
        for _ in range(steps):
            err = math.radians(target_deg - ang())
            wz = float(scene.turret.data.root_ang_vel_w[0, 2])
            if abs(wz) > w_max:
                tau = -kd * wz  # brake only: speed cap
            else:
                tau = max(-clamp, min(clamp, kp * err - kd * wz))
            turret_torque(tau)
            env.step(no_action)
            if abs(target_deg - ang()) < tol and abs(wz) < 0.2:
                streak += 1
                if streak >= 30:
                    break
            else:
                streak = 0
        print(f"[solve] crank_to {label}: theta={ang():+.2f}deg "
              f"(target {target_deg:+.1f})", flush=True)

    def push_brick(tgt_local, *, kp: float = 30.0, kd: float = 9.0,
                   clamp: float = 5.0, ff: float = 2.0, ku: float = 0.03,
                   kw: float = 0.004, steps: int = 720, done=None,
                   label: str = "") -> None:
        """Applied-force carry/push: PD toward a vault-local target + gravity
        feedforward + a friction-beating constant feedforward `ff` toward the
        target (PD alone gain-stalls where kp*err drops under the ~1.2 N sliding
        friction of the brick on the shelf; escalating kp instead would break the
        wrench-delay bound kp*dt/m < 1). kd sets the terminal creep speed
        (ff - friction)/kd ~ 0.09 m/s, so the release skid is ~1 mm. A small
        righting torque stands in for the push's orientation constraint (kw*dt/I
        stays under the wrench-delay bound for the smallest inertia axis)."""
        m_brick = float(scene.brick.root_physx_view.get_masses()[0].sum())
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        for _ in range(steps):
            q = scene.brick.data.root_quat_w
            p = scene.brick.data.root_pos_w
            v = scene.brick.data.root_lin_vel_w
            w = scene.brick.data.root_ang_vel_w
            tgt_w = scene.vault.data.root_pos_w \
                + quat_apply(scene.vault.data.root_quat_w, tgt)
            err = tgt_w - p
            dist = err.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = m_brick * 9.81 * ez + kp * err - kd * v \
                + ff * (err / dist) * (dist > 0.005)
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            # keep the brick's long axis on the vault-local +x heading (the
            # insertion heading) with a weak yaw torque + spin damping
            bx = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
            hx = quat_apply(scene.vault.data.root_quat_w,
                            torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3))
            t_w = ku * torch.cross(bx, hx, dim=-1) - kw * w
            t_b = quat_apply_inverse(q, t_w)
            scene.brick.set_external_force_and_torque(f_b.reshape(n, 1, 3),
                                                      t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        scene.brick.set_external_force_and_torque(zero, zero)
        loc = scene._vault_local(scene.brick.data.root_pos_w)[0]
        print(f"[solve] push {label}: reached vault-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # vault/stand/brick settle; the damped turret holds its seal angle
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    vq = scene.vault.data.root_quat_w[0]
    vyaw = math.degrees(2.0 * math.atan2(float(vq[3]), float(vq[0])))
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) yaw={vyaw:+.1f}deg "
          f"theta0={float(scene.theta0[0]):.2f}deg theta={ang():+.2f}deg "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f})", flush=True)
    # mass readbacks: per-child density / root MassAPI must have produced real
    # masses (custom spawners apply no cfg mass schemas — guard the regression)
    brick_m = float(scene.brick.root_physx_view.get_masses()[0].sum())
    turret_m = float(scene.turret.root_physx_view.get_masses()[0].sum())
    vault_m = float(scene.vault.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: brick={brick_m:.3f} kg turret={turret_m:.2f} kg "
          f"vault={vault_m:.1f} kg", flush=True)
    assert 0.10 < brick_m < 0.60, f"brick mass {brick_m} (density not applied?)"
    assert 1.0 < turret_m < 10.0, f"turret mass {turret_m} (density not applied?)"
    assert vault_m > 40.0, f"vault mass {vault_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(ang() - float(scene.theta0[0])) < 4.0, \
        f"turret must hold its spawned angle, theta={ang():.2f} vs theta0={float(scene.theta0[0]):.1f}"
    s0 = print_score("P0 reset+settle (hatch sealed, brick on its stand)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: crank the turret to the receive stop ------------------------
    crank_to(c.receive_deg, label="open to receive")
    turret_torque(0.15)  # light hold against the receive stop while loading
    step(60)
    report("hatch-open")
    assert ang() > c.receive_deg - 4.0, f"turret not at receive, theta={ang():.2f}"
    assert not bool(scene.success()[0]), "an open empty hatch must not be success"
    s1 = print_score("P1 turret on the receive stop (pocket at the window)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_open - 0.02, f"P1 score {s1}"

    # ---------------- phase 2a: TRANSPORT the brick to a hover outside the window ----------
    hover = torch.zeros(n, 3, device=device)
    hover[:] = torch.tensor([c.half_out + 0.11, 0.0,
                             c.shelf_z1 + c.brick_h / 2 + 0.004], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.vault.data.root_pos_w \
        + quat_apply(scene.vault.data.root_quat_w, hover)
    st[:, 3:7] = scene.vault.data.root_quat_w  # long axis on the insertion heading
    scene.brick.write_root_state_to_sim(st, all_ids)
    print("[solve] P2a: brick transported to free air outside the deposit window "
          "(outside the vault: nothing judged is satisfied by the write)", flush=True)
    assert not bool(scene.brick_in_bin()[0]), "hover must not read as banked"
    assert not bool(scene.brick_loaded()[0]), "hover must not read as loaded"

    # ---------------- phase 2b: force-push the brick through the window into the pocket ----
    push_brick((c.insert_com_x, 0.0, c.shelf_z1 + c.brick_h / 2 + 0.002),
               done=lambda: float(scene._vault_local(
                   scene.brick.data.root_pos_w)[0, 0]) < c.insert_com_x + 0.008,
               label="insert through window")
    step(90)  # release; the brick settles on the shelf inside the pocket
    report("loaded")
    assert bool(scene._loaded[0]), "loaded must latch after the force insertion"
    assert not bool(scene.brick_in_bin()[0]), "a loaded pocket is not banked"
    s2 = print_score("P2 brick pushed through the window into the pocket")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_open + c.w_load - 0.02, f"P2 score {s2}"

    # ---------------- phase 3: crank back — the sweep delivers, the seal returns -----------
    crank_to(0.0, w_max=1.2, steps=1600, label="sweep and reseal")
    turret_torque(0.0)  # hands fully off
    step(120)
    report("delivered")
    if not bool(scene.brick_in_bin()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (sweep did not deliver the brick to the bin)",
              flush=True)
        os._exit(1)
    assert bool(scene.sealed()[0]), f"turret must reseal, theta={ang():.2f}"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (banked but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 brick delivered to the bin, hatch resealed")
    assert s3 >= s2 - 1e-6 and s3 >= 0.999, f"P3 score {s3}"

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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
