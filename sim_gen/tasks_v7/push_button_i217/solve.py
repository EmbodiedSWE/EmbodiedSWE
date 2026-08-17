"""solve — TELEPORT solution for RecoilButtonScene (push_button_i217).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; every
load-bearing interaction goes through contact dynamics via the scene's world-frame
probe-force buffers (frame-encoded plant-side in scene.post_step):

  PHASE 1  transport: the cartridge TRIO (cartridge + plunger + pawl) is moved by ONE
           rigid transform to a staging pose 10 cm outside the dock seat, aligned with
           the dock axis, zero velocity. This is the pick-and-carry leg (single-arm
           handle grasp, see TASK.md). The approach latch (0.15) fires — honest
           transport credit; nothing else can latch (seat is 10 cm away, asserted).
  PHASE 2  dock by CONTACT: a capped velocity-servo force on the cartridge pushes it
           into the U-pocket until its back rests on the dock backwall. Force off,
           settle, verify seated_now().
  PHASE 3  press by CONTACT: a capped force on the plunger along the press axis
           (feedforward spring load + small velocity servo) drives the plunger to the
           hard stop; held there, the internal pawl drops into the stem groove under
           gravity. Force off, settle: the spring shoves the groove step against the
           pawl and the button stays latched at ~30 mm depth, hands-off.
  PHASE 4  wait for the sustained-success counter, all forces off.
  PHASE 5  keep simulating >= 3 s more, hands-off; success() must still hold.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (asserted non-decreasing) and
exactly `SIM_GEN_SOLVE: SUCCESS` only if success() holds after the persistence window.
Hard exit (os._exit) with a watchdog Timer backstop.

Run (forge): python -u -m simgen_tasks.push_button_i217.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.push_button_i217 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.recoil_button")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        p = scene.cart_in_dock()[0]
        print(f"[solve] {tag:12s} d={float(scene.depth()[0]) * 1000:6.1f}mm "
              f"pawl_dz={float(scene.pawl_dz()[0]) * 1000:6.1f}mm "
              f"dock=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
              f"seated={bool(scene.seated_now()[0])} intact={bool(scene.intact()[0])} "
              f"hold={int(scene._hold[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def push_off() -> None:
        for k in scene.push_w:
            scene.push_w[k].zero_()

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: transport the trio to the staging pose =========================
    # ONE rigid transform (yaw-align to the dock + move to 10 cm outside the seat),
    # applied identically to cartridge, plunger and pawl so their internal state
    # (spring compression ~0, pawl resting on the stem) is preserved. Zero velocity.
    stage_gap = 0.10
    out_w = scene.dock_out_w()[0]
    seat_w = scene.seat_pos_w()[0]
    p_cart = scene.cartridge.data.root_pos_w[0].clone()
    q_cart = scene.cartridge.data.root_quat_w[0].clone()
    q_stage = scene.anvil.data.root_quat_w[0].clone()
    p_stage = seat_w + out_w * stage_gap
    p_stage[2] = p_cart[2]  # keep the settled height: transport, not lift-into-place
    q_t = quat_mul(q_stage.view(1, 4), quat_inv(q_cart.view(1, 4)))[0]
    p_t = p_stage - quat_apply(q_t.view(1, 4), p_cart.view(1, 3))[0]
    for body in (scene.cartridge, scene.plunger, scene.pawl):
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = quat_apply(q_t.view(1, 4), body.data.root_pos_w[0].view(1, 3))[0] + p_t
        st[0, 3:7] = quat_mul(q_t.view(1, 4), body.data.root_quat_w[0].view(1, 4))[0]
        body.write_root_state_to_sim(st, ids)
    step(1)
    assert not bool(scene.seated_now()[0]), "staging pose must be outside the seat window"
    step(30)  # let the trio re-settle at the staging pose
    report("staged")
    phase_score("phase1")  # 0.150 (approach latch)

    # ================= PHASE 2: dock by contact ================================================
    # Velocity-servo push on the CARTRIDGE into the pocket (world-frame force through the
    # scene's frame-encoded probe buffer) + a small lateral centring term. Gains audited:
    # Kax*dt/m = 100/(120*0.90) = 0.93 < 1; Ky*dt/m = 0.28. Stall authority Kax*v_des =
    # 12 N well above the ~3.5 N ground-sliding threshold (combined-mu patch friction).
    kax, v_des, f_cap = 100.0, 0.12, 12.0
    ky, cy, f_lat_cap = 30.0, 8.0, 4.0
    ey_w = quat_apply(scene.anvil.data.root_quat_w,
                      torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
    docked = False
    stall = 0
    for i in range(800):
        out_w = scene.dock_out_w()[0]
        v_in = float((scene.cartridge.data.root_lin_vel_w[0] * (-out_w)).sum())
        f_ax = max(0.0, min(f_cap, kax * (v_des - v_in)))
        y = float(scene.cart_in_dock()[0, 1])
        v_y = float((scene.cartridge.data.root_lin_vel_w[0] * ey_w).sum())
        f_lat = max(-f_lat_cap, min(f_lat_cap, -ky * y - cy * v_y))
        scene.push_w["cartridge"][0] = -out_w * f_ax + ey_w * f_lat
        step(1)
        x = float(scene.cart_in_dock()[0, 0])
        if x <= G.SEAT_X + 0.004 and abs(v_in) < 0.01:
            stall += 1
            if stall >= 25:
                docked = True
                break
        else:
            stall = 0
    push_off()
    step(60)
    report("docked")
    if not (docked and bool(scene.seated_now()[0])):
        print("[solve] PHASE 2 FAILED: cartridge not seated in the dock", flush=True)
        verdict(False)
    phase_score("phase2")  # 0.450 (seated latch)

    # ================= PHASE 3: press by contact ===============================================
    # Capped force on the PLUNGER along the press axis: spring feedforward + a small
    # velocity servo (Kv*dt/m = 10/(120*0.12) = 0.69 < 1). Drive to the hard stop, hold
    # for the pawl drop, release, settle.
    kv, v_des_p, f_cap_p = 10.0, 0.05, 16.0
    at_stop = 0
    reached = False
    for i in range(700):
        axis_w = scene.cart_axis_w()[0]
        d = float(scene.depth()[0])
        v_press = float(((scene.plunger.data.root_lin_vel_w[0]
                          - scene.cartridge.data.root_lin_vel_w[0]) * (-axis_w)).sum())
        f = max(0.0, min(f_cap_p, c.spring_f0 + c.spring_k * d + kv * (v_des_p - v_press)))
        scene.push_w["plunger"][0] = -axis_w * f
        step(1)
        if float(scene.depth()[0]) >= G.STROKE - 0.0015:
            at_stop += 1
            if at_stop >= 10:
                reached = True
                break
        else:
            at_stop = 0
    if not reached:
        push_off()
        report("press-stall")
        print("[solve] PHASE 3 FAILED: plunger never reached the hard stop", flush=True)
        verdict(False)
    # hold at the stop (v_des 0) while the pawl falls the ~10 mm into the groove
    dropped = False
    for hold_i in range(3):
        for i in range(48):
            axis_w = scene.cart_axis_w()[0]
            d = float(scene.depth()[0])
            v_press = float(((scene.plunger.data.root_lin_vel_w[0]
                              - scene.cartridge.data.root_lin_vel_w[0]) * (-axis_w)).sum())
            f = max(0.0, min(f_cap_p, c.spring_f0 + c.spring_k * d + kv * (0.0 - v_press)))
            scene.push_w["plunger"][0] = -axis_w * f
            step(1)
        if float(scene.pawl_dz()[0]) <= -0.009:
            dropped = True
            break
    report("held")
    push_off()
    step(90)  # release: the spring sets the groove step against the dropped pawl
    report("released")
    if not dropped or float(scene.pawl_dz()[0]) > -0.009:
        print("[solve] PHASE 3 FAILED: pawl did not drop into the groove", flush=True)
        verdict(False)
    if float(scene.depth()[0]) < c.press_depth_ok + 0.002:
        print("[solve] PHASE 3 FAILED: plunger did not stay latched at depth", flush=True)
        verdict(False)
    phase_score("phase3")  # 0.900 (clicked latch) or 1.000 already

    # ================= PHASE 4: wait for the sustained-success counter =========================
    okd = False
    for _ in range(30):
        step(10)
        if bool(scene.success()[0]):
            okd = True
            break
    report("success")
    if not okd:
        print("[solve] PHASE 4 FAILED: success never sustained", flush=True)
        verdict(False)
    phase_score("phase4")  # 1.000

    # ================= PHASE 5: persistence (>= 3 simulated seconds, hands off) ================
    persist = int(round(3.5 / env.dt))  # 420 substeps at 1/120 s
    step(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist} steps ({persist * env.dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
