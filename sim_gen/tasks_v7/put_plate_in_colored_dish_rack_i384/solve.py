"""Teleport-transport solution for FoldRackScene (sim_gen task
`put_plate_in_colored_dish_rack_i384`) — the task's legitimacy certificate.

PLAN (read from scene.describe(): the fin comb starts FOLDED covering the deck — no
slot exists; the hinge is over-center bistable, gravity parks it OPEN once pushed past
~overcenter_deg; only then does the BLUE gap exist to receive the plate):
  P1 DEPLOY — drive the comb hinge with a torque plant (gravity feedforward +
     velocity servo on a finite-difference hinge rate) up to just past over-center,
     CUT the torque, and let gravity carry the comb the rest of the way and park it
     on the open stop (the bistable hand-off is demonstrated, not scripted). Wait for
     the deploy latch (24 consecutive slow steps).
  P2 INSERT — velocity-servoed vertical FORCE lift raises the flat plate off its
     pedestal stand (a real extraction through contact), free-air teleport to a hover
     centered over the BLUE gap (velocities zeroed), then RELEASE: gravity threads
     the plate edgewise down the 34 mm fin gap onto the deck, where the posts, spine
     and lip arrest it by contact. Wait for the seat latch.
  P3 ring-down until success() holds 120 consecutive steps.
  P4 hands-off persistence >= 3.3 simulated seconds.

Teleports are TRANSPORT ONLY: both endpoints of the one teleport are free-air poses
(velocities zeroed); every load-bearing interaction — the hinge drive, the lift off
the stand against gravity, threading the fin gap, seating on the deck — happens
through contact dynamics (applied wrenches / gravity + contacts).

Wrench-frame policy (the pod's external-wrench frame semantics are measurably
unstable): the DEFAULT encoding is the validated per-step body pre-encode
(`quat_apply_inverse(root_quat_w, w_world)` on the default `is_global=False` call);
for the hinge torque this is exactly (tau, 0, 0) about the comb's own x = the hinge
axis, which is drag-invariant by construction. Every forced phase still runs under a
progress/escape monitor with a per-body encoding toggle (fallback: raw world arg with
`is_global=True`) and a gravity-drop reseat recovery. The insertion applies no force
at all — a centered free release is encoding-proof.

Servo stability audit (external-wrench plant recipe): hinge inertia about the axis
I_h = Ixx + m d^2 = 4.0e-4 + 0.35*0.03645^2 ~= 8.7e-4 kg m^2; torque gain
kt = 0.04 N m s -> kt*dt/I_h ~= 0.38 < 1. Plate lift: kp = 6 N s/m ->
kp*dt/m ~= 0.17 < 1. Hinge rate by finite difference (ang-vel readback is phantom
under external wrenches).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i384.solve \
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

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

BLUE_GAP_SIGN = scene_mod.BLUE_GAP_SIGN
GRAV = 9.81

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fold_rack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    dt = 1.0 / 120.0
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)
    weight = c.plate_m * GRAV
    # comb gravity-torque model (matches the cfg's CoM computation)
    com_d = math.hypot(c.com_y, c.com_z)
    phi0 = math.atan2(c.com_z, c.com_y)
    mgd = c.comb_mass * GRAV * com_d

    env.reset(seed=args.seed)  # seed AFTER build (the EnvCfg.build reseed trap)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)
    print(f"[solve] overcenter={c.overcenter_deg:.1f} deg  mgd={mgd:.4f} N m",
          flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def theta() -> float:
        return float(scene.comb_angle()[0])

    def plate_loc() -> torch.Tensor:
        return scene._local(scene.plate, scene.chassis)[0]

    # ----- per-body wrench encoding ("enc" = validated per-step body pre-encode) ----------
    mode = {"comb": "enc", "plate": "enc"}

    def toggle(name: str) -> None:
        mode[name] = "raw" if mode[name] == "enc" else "enc"
        print(f"[solve] wrench mode[{name}] -> '{mode[name]}'", flush=True)

    def apply_hinge_torque(tau: float) -> None:
        """World torque tau about the comb's hinge axis (= comb body x)."""
        if mode["comb"] == "enc":
            t_arg = torch.zeros(n, 3, device=device)
            t_arg[0, 0] = tau  # body-frame x IS the hinge axis (drag-invariant)
            scene.comb.set_external_force_and_torque(
                zero3, t_arg.view(n, 1, 3), env_ids=all_ids)
        else:
            ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
            axis_w = quat_apply(scene.comb.data.root_quat_w, ex)
            scene.comb.set_external_force_and_torque(
                zero3, (tau * axis_w).view(n, 1, 3), env_ids=all_ids, is_global=True)

    def clear_comb() -> None:
        scene.comb.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)

    def apply_fz(fz: float) -> None:
        """World-vertical force on the plate, through its current encoding."""
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, 2] = fz
        if mode["plate"] == "enc":
            f_arg = quat_apply_inverse(scene.plate.data.root_quat_w, f_world)
            scene.plate.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero3, env_ids=all_ids)
        else:
            scene.plate.set_external_force_and_torque(
                f_world.view(n, 1, 3), zero3, env_ids=all_ids, is_global=True)

    def clear_plate() -> None:
        scene.plate.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)

    def report(tag: str) -> None:
        lp = plate_loc()
        print(f"[solve] {tag:12s} | theta={math.degrees(theta()):6.1f} deg "
              f"plate_ch=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):+.3f}) | deployed={bool(scene.deployed()[0])} "
              f"seated_b={bool(scene.seated('blue')[0])} "
              f"settled={bool(scene.settled()[0])} | "
              f"latches d={float(scene.deploy_latch[0]):.0f}/"
              f"s={float(scene.seat_latch[0]):.0f} | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 720) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    def wait_latch(get_val, tag: str, chunks: int = 20) -> bool:
        for _ in range(chunks):
            if float(get_val()[0]) > 0.5:
                print(f"[solve] {tag}: latch fired", flush=True)
                return True
            step(15)
        print(f"[solve] {tag}: latch NEVER fired", flush=True)
        return False

    # ---------------- phase 1 driver: erect the comb past over-center ---------------------
    def deploy_comb(tag: str) -> bool:
        """Torque plant: gravity feedforward + velocity servo on the FD hinge rate,
        drive to theta_cut just past over-center, CUT, gravity parks the comb on the
        open stop. Stalls escalate the torque cap; 3 fruitless windows or a fall-back
        toggles the encoding and retries."""
        th_cut = math.radians(c.overcenter_deg + 10.0)
        kt, w_max, k_th = 0.04, 1.6, 3.0
        for attempt in range(4):
            t = f"{tag}-a{attempt}"
            cap = 0.30
            th_prev = theta()
            win_i, win_th, stalls = 0, th_prev, 0
            ok_drive = False
            for i in range(720):
                th = theta()
                if th >= th_cut:
                    ok_drive = True
                    break
                w = (th - th_prev) / dt
                th_prev = th
                w_des = min(w_max, max(0.10, k_th * (th_cut + 0.15 - th)))
                tau_ff = mgd * math.cos(phi0 + th)
                tau = max(-cap, min(cap, tau_ff + kt * (w_des - w)))
                apply_hinge_torque(tau)
                env.step(no_action)
                if i - win_i >= 120:
                    if th < win_th + math.radians(2.0):
                        stalls += 1
                        if stalls >= 3:
                            print(f"[solve] {t}: hinge stalled HARD at "
                                  f"{math.degrees(th):.1f} deg — encoding "
                                  f"'{mode['comb']}' suspect", flush=True)
                            break
                        cap += 0.15
                        print(f"[solve] {t}: hinge stalled at "
                              f"{math.degrees(th):.1f} deg; cap -> {cap:.2f}",
                              flush=True)
                    win_i, win_th = i, th
            clear_comb()
            if ok_drive:
                print(f"[solve] {t}: past over-center at "
                      f"{math.degrees(theta()):.1f} deg — torque CUT, gravity "
                      f"takes over", flush=True)
                for _ in range(20):  # coast: gravity parks the comb on the stop
                    step(15)
                    if bool(scene.deployed()[0]) and \
                            float(scene.comb.data.root_ang_vel_w[0].norm()) < 1.0:
                        break
                th = math.degrees(theta())
                if bool(scene.deployed()[0]):
                    print(f"[solve] {t}: comb PARKED OPEN at {th:.1f} deg "
                          f"(mode '{mode['comb']}')", flush=True)
                    return True
                print(f"[solve] {t}: comb fell back to {th:.1f} deg after cut",
                      flush=True)
            toggle("comb")
            step(120)  # let the comb settle (gravity re-closes it if under-center)
        return False

    # ---------------- phase 2 helpers: lift, hover, gravity insertion ---------------------
    def lift_plate(clear_z: float, tag: str, guard_xy=None) -> bool:
        """Velocity-servoed vertical force until the plate's WORLD z >= clear_z.
        guard_xy (world (x, y)): while low, lateral escape beyond 60 mm of it means
        the force is not arriving vertical — toggle encoding and retry (the caller
        re-stages). Regression below the start also aborts."""
        for attempt in range(4):
            t = f"{tag}-a{attempt}"
            kp, extra_cap, v_des = 6.0, 4.0, 0.12
            z0 = float(scene.plate.data.root_pos_w[0, 2])
            win_i, win_z, bad, stalls = 0, z0, False, 0
            for i in range(900):
                p = scene.plate.data.root_pos_w[0]
                z = float(p[2])
                if z >= clear_z:
                    clear_plate()
                    print(f"[solve] {t}: plate clear at z={z:.3f} after {i} steps "
                          f"(mode '{mode['plate']}')", flush=True)
                    return True
                if z < z0 - 0.015:
                    bad = True
                elif guard_xy is not None and z < 0.20 and \
                        math.hypot(float(p[0]) - guard_xy[0],
                                   float(p[1]) - guard_xy[1]) > 0.060:
                    bad = True
                if bad:
                    print(f"[solve] {t}: plate ESCAPED/regressed at "
                          f"({float(p[0]):+.3f},{float(p[1]):+.3f},{z:+.3f}) — "
                          f"encoding '{mode['plate']}' wrong", flush=True)
                    break
                vz = float(scene.plate.data.root_lin_vel_w[0, 2])
                apply_fz(weight
                         + max(-0.8 * weight, min(extra_cap, kp * (v_des - vz))))
                env.step(no_action)
                if i - win_i >= 90:
                    if z < win_z + 0.004:
                        stalls += 1
                        if stalls >= 3:
                            bad = True
                            print(f"[solve] {t}: stalled HARD at z={z:.3f} — "
                                  f"encoding '{mode['plate']}' suspect", flush=True)
                            break
                        kp = min(kp * 1.6, 25.0)
                        extra_cap = min(extra_cap + 2.0, 10.0)
                        print(f"[solve] {t}: stalled at z={z:.3f}; kp -> {kp:.1f}, "
                              f"cap -> {extra_cap:.1f}", flush=True)
                    win_i, win_z = i, z
            clear_plate()
            toggle("plate")
            settle(240)  # let the plate come to rest wherever it fell
        return False

    def hover_release_blue(tag: str) -> bool:
        """Free-air teleport to a hover centered over the BLUE gap (on-edge seat
        orientation, velocities zeroed), then RELEASE: gravity threads the plate down
        the fin gap onto the deck. No applied force — encoding-proof."""
        blue_cx = BLUE_GAP_SIGN * c.slot_dx
        q_ch = scene.chassis.data.root_quat_w
        local = torch.tensor([blue_cx, 0.012, 0.245], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.chassis.data.root_pos_w + quat_apply(q_ch, local)
        st[:, 3:7] = scene._seat_quat(q_ch)
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(2)
        lp = plate_loc()
        print(f"[solve] {tag}: hovering at ({float(lp[0]):+.3f},"
              f"{float(lp[1]):+.3f},{float(lp[2]):+.3f}) over the BLUE gap",
              flush=True)
        for _ in range(10):
            step(15)
            lp = plate_loc()
            if float(lp[2]) <= 0.10 and \
                    float(scene.plate.data.root_lin_vel_w[0].norm()) < 0.10:
                break
        settle()
        ok = bool(scene.seated("blue")[0])
        lp = plate_loc()
        print(f"[solve] {tag}: drop -> seated_blue={ok} at ({float(lp[0]):+.3f},"
              f"{float(lp[1]):+.3f},{float(lp[2]):+.3f}) "
              f"theta={math.degrees(theta()):.1f} deg", flush=True)
        return ok

    # ---------------- phase 0: reset, settle, plan readback -------------------------------
    step(90)
    report("reset")
    assert bool(scene.folded()[0]), "comb must start FOLDED"
    assert not bool(scene.deployed()[0])
    pz = float(scene.plate.data.root_pos_w[0, 2])
    assert abs(pz - (c.stand_h + c.plate_t / 2)) < 0.02, \
        f"plate must start flat on the stand (z={pz:.3f})"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (folded rack verified)")
    assert s0 < 0.05, f"score must start ~0, got {s0}"

    # ---------------- phase 1: DEPLOY the comb (create the slots) --------------------------
    assert deploy_comb("P1-deploy"), "P1 comb deployment failed"
    assert wait_latch(lambda: scene.deploy_latch, "P1-latch"), \
        "deploy latch must fire"
    report("P1-deploy")
    s1 = print_score("P1 comb erected past over-center, gravity-parked OPEN")
    assert s1 >= s0 - 1e-6 and s1 >= 0.345, f"P1 score {s1} (expect 0.35)"

    # ---------------- phase 2: INSERT the plate into the BLUE gap --------------------------
    stand_xy = (float(scene.stand.data.root_pos_w[0, 0]),
                float(scene.stand.data.root_pos_w[0, 1]))
    assert lift_plate(0.30, "P2-lift", guard_xy=stand_xy), "P2 lift failed"
    seated = False
    for attempt in range(3):
        if hover_release_blue(f"P2-insert-a{attempt}"):
            seated = True
            break
        if not bool(scene.deployed()[0]):  # drop knocked the comb shut (unexpected)
            assert deploy_comb(f"P2-redeploy-a{attempt}"), "re-deploy failed"
        assert lift_plate(0.30, f"P2-relift-a{attempt}"), "P2 re-lift failed"
    assert seated, "P2 insertion failed"
    assert wait_latch(lambda: scene.seat_latch, "P2-latch"), "seat latch must fire"
    report("P2-insert")
    s2 = print_score("P2 plate seated on edge in the BLUE gap")
    assert s2 >= s1 - 1e-6 and s2 >= 0.545, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: ring-down until success holds 1 s ---------------------------
    consec = 0
    for _ in range(2400):
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-ringdown")
    s3 = print_score("P3 success ring-down")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after insert + ring-down)", flush=True)
        os._exit(1)
    assert s3 >= 0.99, f"P3 score {s3} (expect 1.0)"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {i}: "
                      f"seated={bool(scene.seated('blue')[0])} "
                      f"deployed={bool(scene.deployed()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
