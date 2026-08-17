"""Teleport solution for LampIsolatorScene (sim_gen task `lamp_off_i247`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. FETCHING the handle bar (applied force): a PD force + gravity feedforward (a
   firm force-limited pinch on the shaft) lifts the bar off its saddle cradle,
   then a gentle righting torque stands it ball-up. The `fetched` latch fires
   DURING this force lift.
2. TRANSPORT (teleport): one root-state write carries the held bar through free
   air to a hover 30 mm ABOVE the hub's funnel mouth, aligned with the bore
   axis, zero velocity. Open sky — no contact interaction is bypassed and no
   rubric clause is satisfied by the write (the tip is far outside the socket).
3. INSERTION (guided force descent + funnel + gravity): the same force-limited
   carry lowers the bar along the tilted bore axis; the funnel mouth and the
   socket walls steer the tip onto the backstop floor. `inserted` is produced by
   contact — a real blind peg-in-hole with 2 mm/side slop. The bore axis passes
   through the axle, so seating pushes exert no throw torque (nothing is scored
   by the insertion push itself).
4. THE THROW (velocity-servo force on the BAR; the hub is driven only through
   the bar-socket contact — the lever is the mechanism): a tangential PD force
   at the bar, plus gravity feedforward and a light seating press, swings the
   hub from its ON stop up through the gravity dead centre; past ~+100 deg the
   wrenches are ZEROED and gravity alone completes and holds the throw onto the
   OFF stop. The success threshold is crossed hands-off.
5. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched —
   the overcenter geometry alone keeps the switch thrown.

Order: fetch -> insert -> throw is forced physically (empty out-of-reach socket;
nothing that fits the slot but the bar can reach the hub; smoke proves it).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.lamp_off_i247.solve --headless [--seed N]
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
    env = ENVS.get("simgen.lamp_isolator")().build(num_envs=args.num_envs, device=device)
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

    def phi() -> float:
        return float(scene.hub_angle_deg()[0])

    def report(tag: str) -> None:
        tip = scene._hub_local(scene.bar.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | phi={phi():+7.2f}deg "
              f"tip_hub=({float(tip[0]):+.3f},{float(tip[1]):+.3f},{float(tip[2]):+.3f}) "
              f"in_socket={bool(scene.bar_tip_in_socket()[0])} "
              f"fetched={bool(scene._fetched[0])} "
              f"inserted={bool(scene._inserted[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)

    def bore_axis() -> torch.Tensor:
        return quat_apply(scene.hub.data.root_quat_w, ez)

    def carry(body, mass: float, frame_body, tgt_local, *, kp: float, kd: float,
              clamp: float, ku: float, kw: float, steps: int, axis_tgt=None,
              done=None, label: str = "") -> None:
        """Applied-force carry: PD toward a frame_body-local target + gravity
        feedforward (a firm force-limited grasp), plus a righting torque toward
        `axis_tgt` (a callable -> (n,3) world axis; default straight up) that
        stands in for the grasp's orientation constraint. Wrenches only THIS
        body. Gains respect the 1-substep wrench delay: kp*dt/m ~= 0.5 and
        kw*dt/I_transverse ~= 0.45 for the 0.105 kg bar; the bar's AXIAL spin is
        never damped (axial inertia ~3e-6 kg m^2 -> instant instability)."""
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        for _ in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            tgt_w = frame_body.data.root_pos_w \
                + quat_apply(frame_body.data.root_quat_w, tgt)
            f_w = mass * 9.81 * ez + kp * (tgt_w - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            axis = quat_apply(q, ez)
            a_tgt = ez if axis_tgt is None else axis_tgt()
            w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
            t_w = ku * torch.cross(axis, a_tgt, dim=-1) - kw * w_perp
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._hub_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached hub-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # hub settles onto its ON stop; the bar settles into its cradle
    pp = (scene.panel.data.root_pos_w - scene.env_origins)[0]
    pq = scene.panel.data.root_quat_w[0]
    pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
    bdir = "+x" if bool(scene.ball_dir_p[0]) else "-x"
    lside = "+" if bool(scene.lamp_side_p[0]) else "-"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"panel=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) yaw={pyaw:+.1f}deg "
          f"phi={phi():+.2f}deg ball_dir={bdir} lamp_side={lside}", flush=True)
    # mass readbacks: per-child density / root MassAPI must have produced real
    # masses (custom spawners apply no cfg mass schemas — guard against regression)
    bar_m = float(scene.bar.root_physx_view.get_masses()[0].sum())
    hub_m = float(scene.hub.root_physx_view.get_masses()[0].sum())
    panel_m = float(scene.panel.root_physx_view.get_masses()[0].sum())
    lamp_m = float(scene.lamp.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: bar={bar_m:.3f} kg hub={hub_m:.3f} kg "
          f"panel={panel_m:.1f} kg lamp={lamp_m:.2f} kg", flush=True)
    assert 0.05 < bar_m < 0.25, f"bar mass {bar_m} (density not applied?)"
    assert 0.04 < hub_m < 0.30, f"hub mass {hub_m} (density not applied?)"
    assert panel_m > 25.0, f"panel mass {panel_m} (MassAPI not applied?)"
    assert 2.0 < lamp_m < 8.0, f"lamp mass {lamp_m} (density not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(phi() - c.phi_on) < 2.0, \
        f"hub must rest gravity-held on its ON stop, phi={phi():.2f}"
    tip_z0 = float(scene.bar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    assert 0.03 < tip_z0 < 0.10, f"bar must rest cradled on the saddle, tip z={tip_z0:.3f}"
    s0 = print_score("P0 reset+settle (hub ON, bar cradled)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: force-lift the bar off its saddle ---------------------------
    bloc = scene._panel_local(scene.bar.data.root_pos_w)[0]
    # P1a: straight lift, orientation nearly free (tiny righting so the tip
    # cannot dig into the saddle while the CoM rises)
    carry(scene.bar, bar_m, scene.panel,
          (float(bloc[0]), float(bloc[1]), float(bloc[2]) + 0.22),
          kp=6.0, kd=2.0, clamp=4.0, ku=0.004, kw=0.02, steps=360,
          done=lambda: float(scene.bar.data.root_pos_w[0, 2]
                             - scene.env_origins[0, 2]) > 0.20,
          label="bar lift")
    # P1b: right the held bar ball-up in free air
    bloc = scene._panel_local(scene.bar.data.root_pos_w)[0]
    up_z = lambda: float(quat_apply(scene.bar.data.root_quat_w, ez)[0, 2])  # noqa: E731
    carry(scene.bar, bar_m, scene.panel,
          (float(bloc[0]), float(bloc[1]), 0.28),
          kp=6.0, kd=2.0, clamp=4.0, ku=0.03, kw=0.02, steps=600,
          done=lambda: up_z() > 0.92 and float(
              scene.bar.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) > 0.22,
          label="bar upright")
    assert bool(scene._fetched[0]), "fetched must latch during the force lift"
    s1 = print_score("P1 bar fetched from the saddle")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_fetch - 0.02, f"P1 score {s1}"

    # ---------------- phase 2: TRANSPORT to a hover above the funnel mouth -----------------
    hover = torch.zeros(n, 3, device=device)
    hover[:, 2] = c.fun_z1 + 0.030
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.hub.data.root_pos_w + quat_apply(scene.hub.data.root_quat_w, hover)
    st[:, 3:7] = scene.hub.data.root_quat_w  # tip-down, aligned with the bore axis
    scene.bar.write_root_state_to_sim(st, all_ids)
    print("[solve] P2: bar transported to free air 30 mm above the funnel mouth "
          "(airborne: nothing judged is satisfied by the write)", flush=True)
    assert not bool(scene.bar_tip_in_socket()[0]), "hover must not read as inserted"

    # ---------------- phase 3: guided force descent into the socket ------------------------
    tipz = lambda: float(scene._hub_local(scene.bar.data.root_pos_w)[0, 2])  # noqa: E731
    carry(scene.bar, bar_m, scene.hub,
          (0.0, 0.0, c.insert_tip_z), kp=6.0, kd=2.0, clamp=3.5, ku=0.03, kw=0.02,
          steps=900, axis_tgt=bore_axis,
          done=lambda: bool(scene.bar_tip_in_socket()[0]) and tipz() < 0.028,
          label="bar descent")
    step(60)  # hands off a moment: the seated bar rests in the socket
    report("bar-seated")
    assert bool(scene.bar_tip_in_socket()[0]), "bar tip must rest inside the socket"
    assert bool(scene._inserted[0]), "inserted must latch (3-step persistence)"
    assert abs(phi() - c.phi_on) < 8.0, \
        f"the insertion push must not throw the hub (bore axis through the axle), phi={phi():.2f}"
    assert not bool(scene.success()[0]), "inserted-but-not-thrown must NOT be success"
    s3 = print_score("P3 bar seated in the socket (switch still ON: no success)")
    assert s3 >= s1 - 1e-6 and s3 >= c.w_fetch + c.w_insert - 0.02, f"P3 score {s3}"

    # ---------------- phase 4: the throw (velocity-servo lever, then hands-off) ------------
    r_com = c.insert_tip_z + 0.1345           # bar CoM along the bore from the axle
    phi_tgt = math.radians(106.0)
    kv, kv_max, clampf = 3.0, 9.0, 5.0
    seat = 1.2
    last_ck, stall_ref = 0, phi()
    released = False
    for i in range(2000):
        ph = math.radians(phi())
        w_des = max(-1.6, min(1.6, 3.0 * (phi_tgt - ph)))
        b = bore_axis()
        y_p = quat_apply(scene.panel.data.root_quat_w, ey)
        t_hat = torch.cross(y_p, b, dim=-1)
        v_des = w_des * r_com * t_hat
        v = scene.bar.data.root_lin_vel_w
        w = scene.bar.data.root_ang_vel_w
        q = scene.bar.data.root_quat_w
        f_w = bar_m * 9.81 * ez + kv * (v_des - v) - seat * b
        f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f_w = f_w * (f_norm.clamp(max=clampf) / f_norm)
        axis = quat_apply(q, ez)
        w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
        t_w = 0.03 * torch.cross(axis, b, dim=-1) - 0.02 * w_perp
        scene.bar.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).reshape(n, 1, 3),
            quat_apply_inverse(q, t_w).reshape(n, 1, 3))
        env.step(no_action)
        p_now = phi()
        if p_now >= 102.0 and abs(float(scene.hub.data.root_ang_vel_w[0].norm())) < 2.0:
            released = True
            break
        if i - last_ck >= 100:  # stall watch: escalate the GAIN, not the cap
            if p_now - stall_ref < 1.0 and kv < kv_max:
                kv *= 1.4
                print(f"[solve] throw stall at phi={p_now:+.1f}deg -> kv={kv:.2f}",
                      flush=True)
            last_ck, stall_ref = i, p_now
    scene.bar.set_external_force_and_torque(zero, zero)
    print(f"[solve] throw released at phi={phi():+.2f}deg "
          f"(gravity completes the overcenter throw)", flush=True)
    assert released, f"throw servo never reached the release band, phi={phi():+.2f}"
    step(150)  # hands off: gravity carries the hub onto its OFF stop and holds it
    report("thrown")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (thrown but success not reached)", flush=True)
        os._exit(1)
    s4 = print_score("P4 switch thrown past dead centre onto the OFF stop")
    assert s4 >= s3 - 1e-6 and s4 >= 0.999, f"P4 score {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
