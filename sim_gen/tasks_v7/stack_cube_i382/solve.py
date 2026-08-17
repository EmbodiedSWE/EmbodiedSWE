"""Contact-dynamics solution for SwingPumpScene (sim_gen task `stack_cube_i382`)
— the task's legitimacy certificate.

NO teleports at all: the pendulum is a jointed body and is never written after reset.
The whole solve is one bounded contact interaction — a resonant PUMP:

  - Each step, read the pendulum angle from the scene's own hinge readback and finite-
    difference the swing rate (root_ang_vel_w is phantom-prone under wrenches). Compute
    the hinge energy E = 1/2 I w^2 + Mgd (1 - cos th) from the cfg's authored constants.
    Force (|F| <= pump_force_max = 2 N, asserted in cfg to be unable to statically hold
    the bob at the window edge or to reach the flap in one transit) is applied ONLY
    while the bob is EXPOSED below the enclosure plates — the sole place a robot could
    touch it. The push direction is read from the FRAME's own pose every step (yaw-
    randomization-proof) and encoded into the pendulum's BODY frame every step (immune
    to the is_global wrench rotation-drag quirk).
  - The release toward the flap is a DECISION, not an accident of the energy ladder:
    on + ascents (toward the flap) the controller BRAKES whenever E exceeds a hold
    level (+ peaks stay ~92 deg, below the 100 deg flap), while - descents PUMP the
    far-side amplitude up toward a servoed reference. At each - turning point the
    peak angle is read back POSE-EXACT (w = 0, no FD error); if it lands in the
    release band [128, 137.5] deg — enough coast energy to carry the bob deep past
    the flap-fall conflict zone, yet short of the pocket end wall — the controller
    ARMS and the bob coasts force-free through the flap. Off-band peaks correct the
    reference by the exact PE error and the cycle repeats.
  - After the armed coast crosses the pass line the force stays off: the blade falls
    shut behind the bob and the returning bob is arrested on the closed blade — the
    gravity ratchet finishes the job hands-off. If a capture attempt still decays
    without success, the same controller simply re-pumps to another armed pass.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.stack_cube_i382.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.swing_pump")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench() -> None:
        scene.pendulum.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                     env_ids=all_ids)

    def th_deg() -> float:
        return math.degrees(float(scene.theta()[0]))

    def flap_deg() -> float:
        return math.degrees(float(scene.flap_open()[0]))

    def report(tag: str) -> None:
        print(f"[solve] {tag:14s} | theta={th_deg():+7.1f}deg flap={flap_deg():+6.1f}deg "
              f"exposed={bool(scene.bob_exposed()[0])} amp={bool(scene.lat_amp[0])} "
              f"pass={bool(scene.lat_pass[0])} pocket={bool(scene.in_pocket()[0])} "
              f"still={bool(scene.pend_still()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    qf0 = scene.frame.data.root_quat_w[0]
    yaw = math.atan2(2 * (float(qf0[0]) * float(qf0[3])), 1 - 2 * float(qf0[3]) ** 2)
    print(f"[solve] layout readback (seed {args.seed}): frame=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg theta0={th_deg():+.1f}deg "
          f"flap={flap_deg():+.2f}deg "
          f"E_target={c.e_target:.3f}J E_flap={c.e_flap:.3f}J", flush=True)
    report("reset")
    if abs(flap_deg()) > 4.0:
        fail("check flap not closed at reset")
    if float(scene.score()[0]) > 1e-6:
        fail("nonzero score at reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phases 1+2: resonant pump with a CONTROLLED release -------------------
    dt = c.sim_dt
    th_pass = math.radians(c.theta_pass_deg)
    # + ascents are braked above e_hold (+ peak ~92 deg — below the flap), so the bob
    # can only reach the flap side during an ARMED, force-free coast.
    e_hold = c.mgd * (1.0 - math.cos(math.radians(92.0)))
    # Release band on the FAR side, judged at the - turning point where the pose-exact
    # readback has zero FD error: PE(128 deg) = 1.478 J >= e_target = 1.465 J (deep
    # past the blade-fall conflict zone even after ~2-3% coast losses), PE(137.5 deg)
    # = 1.60 J < PE(pocket end 139 deg) = 1.62 J (no end-wall slam-and-bounce).
    rel = {"lo": 128.0, "hi": 137.5, "target": 132.0,
           "e_ref": c.mgd * (1.0 - math.cos(math.radians(132.0)))}
    amp = {"s1": None, "s0": s0}

    def apply_push(d: float) -> None:
        f_frame = torch.tensor([d * c.pump_force_max, 0.0, 0.0],
                               device=device).unsqueeze(0).expand(n, 3)
        f_world = quat_apply(scene.frame.data.root_quat_w, f_frame)
        f_body = quat_apply_inverse(scene.pendulum.data.root_quat_w, f_world)
        scene.pendulum.set_external_force_and_torque(
            f_body.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids)

    def pump_and_release(max_steps: int, tag: str) -> bool:
        """Pump the far-side amplitude into the release band, then coast through the
        flap. Returns True when theta crosses the pass line during an armed coast."""
        e_des = c.mgd * (1.0 - math.cos(math.radians(rel["target"])))
        th_prev = float(scene.theta()[0])
        armed = False
        th_min = 0.0
        for i in range(max_steps):
            th = float(scene.theta()[0])
            w = (th - th_prev) / dt
            e = 0.5 * c.i_hinge * w * w + c.mgd * (1.0 - math.cos(th))
            if armed and th_prev < th_pass <= th:
                clear_wrench()
                print(f"[solve] {tag}: armed coast crossed the pass line at "
                      f"t={i * dt:5.2f}s", flush=True)
                return True
            th_prev = th
            if w < 0.0:
                th_min = min(th_min, th)
            if armed and w < -0.4 and th < math.radians(45.0):
                # armed coast fell short of the flap: losses bigger than modeled —
                # raise the release band and resume pumping
                armed = False
                rel["lo"] = min(rel["lo"] + 2.0, rel["hi"] - 1.0)
                rel["target"] = min(rel["target"] + 2.0, rel["hi"])
                e_des = c.mgd * (1.0 - math.cos(math.radians(rel["target"])))
                print(f"[solve] {tag}: armed coast fell short — release band raised to "
                      f"[{rel['lo']:.0f},{rel['hi']:.0f}] deg", flush=True)
            if not armed and w > 0.4 and th_min < -1.2:
                # a genuine - turning point just happened: judge it pose-exact
                peak_deg = -math.degrees(th_min)
                pe = c.mgd * (1.0 - math.cos(th_min))
                if rel["lo"] <= peak_deg <= rel["hi"]:
                    armed = True
                    print(f"[solve] {tag}: -peak {peak_deg:.1f} deg (PE {pe:.3f} J) in "
                          f"release band -> coasting through the flap", flush=True)
                else:
                    # clamp below PE(swing stop) = 1.63 J so corrections never command
                    # a stop-slamming amplitude
                    rel["e_ref"] = max(1.20, min(1.65, rel["e_ref"] + (e_des - pe)))
                    print(f"[solve] {tag}: -peak {peak_deg:.1f} deg off-band, "
                          f"e_ref -> {rel['e_ref']:.3f} J", flush=True)
                th_min = 0.0
            d = 0.0
            if not armed:
                if abs(w) < 0.4:
                    if abs(th) < 0.05 and e < 0.05:
                        d = 1.0  # dead-hang starter kick
                elif w < 0.0:
                    d = -1.0 if e < rel["e_ref"] else 0.0  # pump the far side
                else:
                    d = 1.0 if e < e_hold else -1.0  # pump low, BRAKE high
            if d != 0.0 and bool(scene.bob_exposed()[0]):
                apply_push(d)
            else:
                clear_wrench()
            env.step(no_action)
            if amp["s1"] is None and bool(scene.lat_amp[0]):
                report("amp-crossed")
                amp["s1"] = print_score("P1 amplitude 60 deg genuinely up-crossed")
                assert amp["s1"] >= amp["s0"] - 1e-6, \
                    "score decreased across the amplitude latch"
            if i % 480 == 479:
                print(f"[solve] {tag} t={(i + 1) * dt:5.1f}s "
                      f"theta={math.degrees(th):+7.1f}deg w={w:+6.2f}rad/s E={e:.3f}J "
                      f"score={float(scene.score()[0]):.3f}", flush=True)
        clear_wrench()
        return False

    passed = pump_and_release(12000, "pump")
    if amp["s1"] is None:
        fail("amplitude latch never earned")
    if not passed:
        fail("pump could not carry the bob past the flap")
    if not bool(scene.lat_pass[0]):
        fail("pass-line crossed but the flap-transit latch was refused")
    report("flap-passed")
    s1 = amp["s1"]
    s2 = print_score("P2 check flap genuinely transited")
    assert s2 >= s1 - 1e-6, "score decreased across the flap transit"

    # ---------------- phase 3: hands-off ratchet capture ------------------------------------
    # Force is already cut. The blade falls shut behind the bob; the returning bob is
    # arrested on the closed blade and settles in the pocket. Pure gravity from here.
    # If a capture attempt loses the re-close race (bob slips back out under the still-
    # falling blade), simply re-pump to another pass — the latches are monotonic.
    done = False
    for attempt in range(3):
        escaped = False
        for i in range(2400):
            env.step(no_action)
            if bool(scene.success()[0]):
                done = True
                break
            th = float(scene.theta()[0])
            if i < 60 and i % 12 == 11:
                print(f"[solve]   ratchet t={(i + 1) * dt:.2f}s "
                      f"theta={math.degrees(th):+7.1f}deg flap={flap_deg():+6.1f}deg",
                      flush=True)
            if th < math.radians(55.0):
                escaped = True
                break
            if i % 240 == 239:
                report(f"capture t={(i + 1) * dt:.1f}s")
        if done or not escaped:
            break
        print(f"[solve] capture attempt {attempt + 1} escaped the pocket — re-pumping "
              "with a higher release band", flush=True)
        rel["lo"] = min(rel["lo"] + 2.0, rel["hi"] - 1.0)
        rel["target"] = min(rel["target"] + 2.0, rel["hi"])
        if not pump_and_release(9000, f"re-pump {attempt + 1}"):
            fail("re-pump could not carry the bob past the flap again")
    if not done:
        fail("no success after the flap transit — bob did not settle captured")
    report("captured")
    s3 = print_score("P3 bob arrested in the pocket, flap re-closed")
    assert s3 >= s2 - 1e-6, "score decreased across the capture"

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
