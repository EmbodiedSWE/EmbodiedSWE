"""solve — TELEPORT solution for GondolaWheelScene
(libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; every
load-bearing interaction goes through contact dynamics or applied wrenches:

  PHASE 1  LOAD — transport the black bowl from the staging table to just ABOVE the
           gondola tray basin at the bottom stop (release point computed from the LIVE
           gondola pose, deliberately OUTSIDE the geometric "aboard" band so no credit
           can latch at the teleport instant). Gravity drops it in; it lands and rests
           by contact.
  PHASE 2  CRANK — drive the wheel with a torque servo (gravity feedforward +
           proportional velocity term, hard-clamped) from the 30-deg loading stop up
           and over top-dead-center; the torque is CUT at 182 deg and gravity alone
           carries the wheel onto its 186-deg over-center park stop and holds it there.
           The pendulum gondola self-levels through the whole arc; the bowl rides by
           contact and friction only.
  PHASE 3  SERVE — push the bowl with a small regulated horizontal force off the
           tray's open front edge, across the 15-mm gap, through the parapet window;
           it drops 9 mm onto the gallery deck and rests by contact. Forces off.
  PHASE 4  hands off for >= 3 simulated seconds; success() must persist (rejects
           fly-through or precarious states) before the verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a daemon watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan lives in TASK.md as the
embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gondola_wheel")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def settle_until(pred, max_steps: int = 600, poll: int = 10, min_steps: int = 0) -> bool:
        if min_steps:
            step(min_steps)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def ang() -> float:
        return math.degrees(float(scene.wheel_angle()[0]))

    def report(tag: str) -> None:
        bl = scene._to_gondola_frame(scene.bowl.data.root_pos_w)[0]
        bw = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:14s} ang={ang():+7.2f} "
              f"bowl_loc=({bl[0]:+.3f},{bl[1]:+.3f},{bl[2]:+.3f}) "
              f"bowl_w=({bw[0]:+.3f},{bw[1]:+.3f},{bw[2]:+.3f}) "
              f"aboard={bool(scene.bowl_in_tray()[0])} parked={bool(scene.wheel_parked()[0])} "
              f"deck={bool(scene.bowl_on_deck()[0])} settled={bool(scene.settled()[0])} "
              f"L={int(scene._loaded_ever[0])}{int(scene._rode_ever[0])}"
              f"{int(scene._hoisted_ever[0])}{int(scene._delivered_ever[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        if ok:
            print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        else:
            print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def clear_wrenches() -> None:
        scene.disc.set_external_force_and_torque(zero3, zero3)
        scene.bowl.set_external_force_and_torque(zero3, zero3)

    def teleport(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """TRANSPORT: set a root pose with zero velocity. Never used to enter a
        scoring band — release points are chosen outside every credit region."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, torch.arange(1, device=device))

    # ================= reset + settle =============================================================
    env.reset(seed=args.seed)
    # wheel falls from the sampled start angle onto its 30-deg loading stop
    ok0 = settle_until(lambda: bool(scene.wheel_still()[0])
                       and ang() < c.travel_lo_deg + 1.5, max_steps=600, min_steps=60)
    report("reset")
    assert ok0, "wheel did not settle onto the loading stop"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: LOAD (drop into the tray by gravity) ==============================
    # Release the bowl ~2.4 cm above its tray rest height: gondola-local z = 0.056,
    # above the aboard band's z_hi (0.050), so the teleport itself latches nothing.
    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    drop_loc = torch.tensor([0.0, -0.01, 0.056], device=device)
    assert float(drop_loc[2]) > c.tray_z_hi, "release point must be outside the aboard band"
    pos = (scene.gondola.data.root_pos_w[0]
           + quat_apply(scene.gondola.data.root_quat_w[0:1], drop_loc.view(1, 3))[0])
    teleport(scene.bowl, [float(v) for v in pos])
    step(1)
    report("release-bowl")
    assert not bool(scene._loaded_ever[0]), "loaded latch must not fire at the teleport"
    ok1 = settle_until(lambda: bool(scene._loaded_ever[0]) and bool(scene.bowl_in_tray()[0])
                       and float(scene.bowl.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed,
                       max_steps=360, min_steps=20)
    report("loaded")
    if not ok1:
        print("[solve] PHASE 1 FAILED: bowl did not come to rest aboard the tray", flush=True)
        verdict(False)
    phase_score("phase1")  # 0.150

    # ================= PHASE 2: CRANK (torque servo, gravity finishes over-center) ================
    # tau = clamp(tau_ff + k*(w_des - w), lo, hi); tau_ff resists the hanging load
    # M*g*R*sin(theta) (M = gondola + bowl). Cut at 182 deg; gravity alone carries the
    # wheel onto the 186-deg over-center stop and HOLDS it (no torque at the end).
    m_load = c.gondola_mass + c.bowl_mass
    k_w = 0.6
    crank_steps = 0
    while ang() < 182.0 and crank_steps < 1500:
        th = float(scene.wheel_angle()[0])
        w = float(scene.disc.data.root_ang_vel_w[0, 1])
        w_des = 0.7 if th < math.radians(150.0) else 0.35
        tau_ff = m_load * 9.81 * c.arm_r * math.sin(th)
        tau = max(-0.8, min(2.6, tau_ff + k_w * (w_des - w)))
        t3 = torch.tensor([0.0, tau, 0.0], device=device).view(1, 1, 3)
        scene.disc.set_external_force_and_torque(zero3, t3)
        env.step(no_action)
        crank_steps += 1
    clear_wrenches()
    print(f"[solve] crank: {crank_steps} steps, cut at ang={ang():+.2f}", flush=True)
    ok2 = settle_until(lambda: bool(scene.wheel_parked()[0]) and bool(scene.wheel_still()[0])
                       and bool(scene._hoisted_ever[0]), max_steps=600, min_steps=30)
    report("parked")
    if not ok2:
        print("[solve] PHASE 2 FAILED: wheel did not park over-center with the bowl aboard",
              flush=True)
        verdict(False)
    phase_score("phase2")  # 0.550

    # ================= PHASE 3: SERVE (regulated push through the window) =========================
    # Small horizontal force on the bowl: velocity-regulated +y push (target 0.12 m/s)
    # plus a weak x-centering term toward the window center line. World-frame force is
    # rotated into the bowl's body frame (set_external_force_and_torque is body-frame).
    push_steps = 0
    while float((scene.bowl.data.root_pos_w - scene.env_origins)[0, 1]) < 0.30 \
            and push_steps < 900:
        bp = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        vy = float(scene.bowl.data.root_lin_vel_w[0, 1])
        fy = max(0.0, min(5.0, 1.9 + 6.0 * (0.12 - vy)))
        fx = max(-1.0, min(1.0, 4.0 * (0.02 - float(bp[0]))))
        fw = torch.tensor([fx, fy, 0.0], device=device).view(1, 3)
        fb = quat_apply_inverse(scene.bowl.data.root_quat_w[0:1], fw)
        scene.bowl.set_external_force_and_torque(fb.view(1, 1, 3), zero3)
        env.step(no_action)
        push_steps += 1
    clear_wrenches()
    print(f"[solve] push: {push_steps} steps", flush=True)
    ok3 = settle_until(lambda: bool(scene.success()[0]), max_steps=600, min_steps=30)
    report("served")
    if not ok3:
        print("[solve] PHASE 3 FAILED: bowl did not come to rest on the gallery deck",
              flush=True)
        verdict(False)
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3 simulated seconds, hands off) ===================
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
