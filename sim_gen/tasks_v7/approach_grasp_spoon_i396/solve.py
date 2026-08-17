"""Teleport solution for IdlerGearboxScene (sim_gen task `approach_grasp_spoon_i396`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. TRANSPORT THE IDLER (teleport): one pose write stages the toothed wheel HOVERING
   over the bearing peg — bore roughly over the axis with a deliberate small lateral
   offset (a realistic place, not a magic one), well above the seat. Free-space
   transport only: the idler is NOT written into the seated state.
2. INSTALL THE IDLER (contact dynamics — never teleported into place): the wheel is
   RELEASED and falls; the stepped peg taper funnels the octagonal bore down onto the
   boss. If it lands tooth-top-on-tooth-top on the crank or rack teeth (a real
   mechanical outcome), the solver WIGGLES THE CRANK (small alternating physical
   torque) so the teeth re-phase and the wheel drops into mesh; if a drop wedges, it
   re-hovers with a rotated tooth phase and tries again. The scene's own
   `idler_seated()` readback must confirm the seat.
3. DRIVE THE TRANSMISSION (contact dynamics — the heart of the task): a
   velocity-regulated external torque about the crank's OWN body z-axis (the hinge
   axis, immune to wrench-frame drag) turns the crank counter-clockwise. Every
   newton of rack thrust flows crank teeth -> idler teeth -> rack teeth: the rack
   advances ~13 cm and its nose pushes the untouchable cube along the roofed tunnel
   and off the lip. A direction probe flips the torque sign if the rack does not
   advance; a stall escalator raises the torque cap.
4. HANDS-OFF DELIVERY: torque cut once the rack reaches its stop; the ejected cube
   falls off the lip and settles inside the walled pocket on gravity alone.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.idler_gearbox_i396")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    env.reset(seed=args.seed)  # seed AFTER build (the EnvCfg.build reseed trap)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        il = scene._local(scene.idler)[0]
        cl = scene._local(scene.cube)[0]
        print(f"[solve] {tag:12s} | idler_loc=({float(il[0]):+.3f},"
              f"{float(il[1]):+.3f},{float(il[2]):.3f}) "
              f"rack_q={float(scene.rack_q()[0]):+.4f} "
              f"geared_adv={float(scene.geared_adv[0]):.4f} "
              f"cube_loc=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) seated={bool(scene.idler_seated()[0])} "
              f"in_pocket={bool(scene.cube_in_pocket()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def crank_torque(tau: float) -> None:
        """Torque about the crank's OWN body z axis (= the hinge axis: no
        wrench-frame drag possible along a revolute's own axis)."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.crank.set_external_force_and_torque(zero_w, t, env_ids=all_ids,
                                                  is_global=False)

    def crank_w(f_up: torch.Tensor) -> float:
        """Crank angular rate about the frame's up axis."""
        return float((scene.crank.data.root_ang_vel_w[0] * f_up[0]).sum())

    def idler_wrench(fz: float, tau: float) -> None:
        """Press (body -z force) + spin (body z torque) on the idler — the
        install-wiggle a hand would do; the wheel stays near-upright so body z is
        the peg axis."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = fz
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.idler.set_external_force_and_torque(f, t, env_ids=all_ids,
                                                  is_global=False)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    fq = scene.frame.data.root_quat_w[0]
    f_yaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
    il0 = scene._local(scene.idler)[0]
    bl0 = scene._local(scene.blank)[0]
    cl0 = scene._local(scene.cube)[0]
    th0 = float(scene.crank_angle()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"frame=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) "
          f"yaw={math.degrees(f_yaw):+.1f}deg crank0={math.degrees(th0):+.1f}deg "
          f"idler_loc=({float(il0[0]):+.3f},{float(il0[1]):+.3f}) "
          f"blank_loc=({float(bl0[0]):+.3f},{float(bl0[1]):+.3f}) "
          f"cube_x={float(cl0[0]):+.3f}", flush=True)
    report("reset")
    assert not bool(scene.idler_seated()[0]), "idler spawned seated?!"
    assert float(scene.rack_q()[0]) < 0.008, "rack did not spawn at its q=0 stop"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    f_pos = scene.frame.data.root_pos_w
    f_quat = scene.frame.data.root_quat_w
    f_up = quat_apply(f_quat, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))

    # ---------------- phases 1+2: transport (teleport) then INSTALL (drop) ------------------
    # Hover the idler over the peg (small deliberate lateral offset), release, let
    # the stepped taper funnel the bore down onto the boss. Tooth-top rests are
    # cleared by physically wiggling the crank; a wedged drop earns a re-hover with
    # a rotated tooth phase.
    seated = False
    for attempt in range(4):
        off = 0.003 if attempt == 0 else 0.002
        ang = attempt * 1.7
        hover = torch.tensor([off * math.cos(ang), off * math.sin(ang), 0.092],
                             device=device).expand(n, 3)
        phase = attempt * (math.pi / c.idler_teeth)  # rotate tooth phase per try
        hq = torch.tensor([math.cos(phase / 2), 0.0, 0.0, math.sin(phase / 2)],
                          device=device).expand(n, 4)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = f_pos + quat_apply(f_quat, hover)
        st[:, 3:7] = torch.nn.functional.normalize(
            torch.stack([
                f_quat[:, 0] * hq[:, 0] - f_quat[:, 3] * hq[:, 3],
                torch.zeros(n, device=device), torch.zeros(n, device=device),
                f_quat[:, 0] * hq[:, 3] + f_quat[:, 3] * hq[:, 0]], dim=-1), dim=-1)
        scene.idler.write_root_state_to_sim(st, all_ids)
        step(2)
        if attempt == 0:
            assert not bool(scene.idler_seated()[0]), \
                "hover pose already reads as seated (teleport must not seat)"
            s1 = print_score("P1 idler transported to hover over the peg (not seated)")
        # release: gravity + taper do the insertion; if the wheel HANGS on the peg
        # above the seat (bore wedge or tooth-top rest) an escalating unjam
        # presses it down while spinning it and wiggling the crank — the exact
        # wiggle a hand doing this install would make.
        hang = 0
        for i in range(600):
            env.step(no_action)
            if bool(scene.idler_seated()[0]):
                seated = True
                break
            il = scene._local(scene.idler)[0]
            hanging = (float(il[0:2].norm()) < 0.009
                       and 0.037 < float(il[2]) < 0.085)
            hang = hang + 1 if hanging else 0
            if hang >= 50:
                hang = 0
                sgn = 1.0 if (i // 120) % 2 == 0 else -1.0
                print(f"[solve] idler hanging on the peg at z={float(il[2]):.3f} "
                      f"(step {i}) -> press+spin+crank-wiggle unjam", flush=True)
                idler_wrench(-5.0, 0.05 * sgn)
                crank_torque(0.06 * sgn)
                step(30)
                idler_wrench(0.0, 0.0)
                crank_torque(0.0)
            elif i in (150, 300, 450):  # off-peg tooth-top rest: crank wiggle
                wig = 0.06 if (i // 150) % 2 == 0 else -0.06
                crank_torque(wig)
                step(25)
                crank_torque(0.0)
        if seated:
            break
        report(f"drop-retry{attempt}")
        print(f"[solve] drop attempt {attempt} did not seat -> re-hover with "
              f"rotated tooth phase", flush=True)
    idler_wrench(0.0, 0.0)
    crank_torque(0.0)
    report("seated")
    assert seated, "idler never seated on the peg"
    step(30)  # let it settle in the seat
    assert bool(scene.seated_ever[0] > 0.5), "seating did not latch"
    s2 = print_score("P2 idler dropped onto the peg and seated (gear train complete)")
    assert s2 >= 0.25 - 1e-6, "seating credit missing"

    # ---------------- phase 3: DRIVE THE TRANSMISSION (contact dynamics) --------------------
    # Velocity-regulated torque about the crank's own axis. Direction probe: if the
    # rack has not advanced after a fair engagement window while the crank turned,
    # flip the sign. Stall escalator raises the cap.
    tau, w_des = 0.15, 2.0
    sign = +1.0  # +z = counter-clockwise from above = rack +x by the gear chain
    q_start = float(scene.rack_q()[0])
    best_q, last_bump = q_start, 0
    probe_flip_done = False
    done = False
    for i in range(2600):
        q = float(scene.rack_q()[0])
        if q > 0.126:
            done = True
            break
        w = crank_w(f_up)
        crank_torque(sign * tau if sign * w < w_des else 0.0)
        env.step(no_action)
        if q > best_q + 0.002:
            best_q, last_bump = q, i
        if i - last_bump > 300:
            last_bump = i
            if not probe_flip_done and best_q < q_start + 0.004:
                sign = -sign
                probe_flip_done = True
                print(f"[solve] rack not advancing (q={q:+.4f}) with crank "
                      f"turning -> flip torque sign to {sign:+.0f}", flush=True)
            else:
                tau = min(tau * 1.8, 0.6)
                print(f"[solve] stall at q={q:+.4f} -> tau={tau:.2f} N*m",
                      flush=True)
        if i % 400 == 0:
            report(f"crank+{i}")
    crank_torque(0.0)
    report("cranked")
    assert done, "cranking never drove the rack to its forward stop"
    assert float(scene.geared_adv[0]) >= c.trans_min, "transmission never latched"
    s3 = print_score("P3 crank driven: rack at full stroke, cube pushed off the lip")
    assert s3 >= s2 - 1e-6, "score decreased across the drive"

    # ---------------- phase 4: HANDS-OFF DELIVERY (pure physics) ----------------------------
    quiet = 0
    landed = False
    for i in range(720):
        env.step(no_action)
        still = (float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_lin)
        quiet = quiet + 1 if (still and bool(scene.cube_in_pocket()[0])) else 0
        if quiet >= 40:
            landed = True
            break
        if i % 240 == 0:
            report(f"deliver+{i}")
    report("delivered")
    assert landed and bool(scene.cube_in_pocket()[0]), \
        "the cube did not settle inside the pocket"
    s4 = print_score("P4 cube settled inside the pocket")
    assert s4 >= s3 - 1e-6, "score decreased across the delivery"
    if not bool(scene.success()[0]):
        step(60)  # give the settle counter time if it was the only gate
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
