"""solve — demonstration solution for TwistlockUnplugScene (lamp_off_i101).

Scene-level env (robot="null"). NOTHING is teleported: both plugs start seated (the
reset state) and every bit of progress flows through the live plant. The solver only
writes the scene's `drive_f` / `drive_t` buffers — the stand-in for the Franka's
wrist roll and straight pull on the grasped plug cap (TASK.md) — and post_step
applies them as wrenches along the plug's ONE free axis, so the drive never drifts
with plug rotation.

  PHASE 1 (twist): a cascaded velocity servo — outer rate command
  w_des = clamp(K_ANG*(alpha - twist), +-1 rad/s), inner tau = KW*(w_des - w),
  |tau| <= TAU_MAX = 0.02 N*m — comfortable fingertip wrist-roll authority. The
  ~7e-6 kg*m^2 plug makes any stiffness servo ring against the sampled loop; the
  rate cascade is discretely monotone (KW*dt/I ~ 0.35 < 1) and parks the key bar
  at the episode's sampled slot angle (either sign) without ringing. The bar never
  touches the plate during the twist (it sits 3 mm behind it) — alignment credit
  is earned by rotation alone.

  PHASE 2 (pull): a velocity servo f = KV*(v_des - v) with
  v_des = min(V_CAP, K_APP*(x_des - ext)), V_CAP = 0.06 m/s, |f| <= F_MAX = 6 N,
  while the twist servo keeps holding alignment. The bar passes through the slot —
  live clearance geometry, ~7 mm a side when aligned; pulled UNALIGNED it jams on
  the plate by real collision (smoke proves that). If progress stalls the GAIN
  escalates (memory: servo stalls are gain-limited, not cap-limited). The drive is
  released past ext_goal + margin; body damping parks the plug.

The FAN's plug (the decoy) is never touched — its keep-alive constraint is
satisfied by correct target selection, exactly what the cord-tracing rubric grades.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing). After
success() first holds, keeps simulating >= 3.5 more simulated seconds with both
drive buffers zero (asserted); only if success() still holds prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, watchdog Timer
as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.lamp_off_i101.solve --headless [--seed N]
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
    from simgen_tasks.lamp_off_i101 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Twist servo (fingertip wrist-roll authority). Plug inertia about x ~ 7e-6 kg*m^2
# is TINY: a stiffness servo rings against the sampled loop, so use a cascaded
# velocity servo instead — outer rate command w_des = clamp(K_ANG*err, +-W_CAP_TW),
# inner tau = KW*(w_des - w). Discretely monotone: KW*dt/I ~ 0.35 < 1.
K_ANG = 4.0  # 1/s outer angle->rate gain (err decays at ~4/s)
W_CAP_TW = 1.0  # rad/s max twist rate
KW = 3e-4  # N*m*s/rad inner rate gain
TAU_MAX = 0.02  # N*m hard cap (>> the ~4e-5 N*m damping torque it must beat)
ALIGN_DONE = math.radians(3.0)  # phase-1 exit: within 3 deg of the slot angle
W_DONE = 0.3  # rad/s: ...and slow

# Pull servo (straight axial pull on the cap). The wrench is applied with a ONE-STEP
# delay (post_step runs after sim.step), so the discrete gain must stay well under 1:
# KV*dt/m = 0.28 — smooth tracking, no bang-bang (KV=8 rang at +-0.35 m/s).
KV0 = 2.0  # N*s/m starting gain
KV_MAX = 3.5  # escalation ceiling (stays delay-stable)
V_CAP = 0.06  # m/s max extraction rate
K_APP = 3.0  # v_des = min(V_CAP, K_APP * (x_des - ext)) -> soft landing
F_MAX = 6.0  # N hard cap
V_DONE = 0.02  # m/s: release only a SLOW plug (never release mid-swing)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.twistlock_unplug")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    ls = None  # lamp socket idx, set after reset

    def report(tag: str) -> None:
        tw = math.degrees(float(scene.plug_twist()[0, ls]))
        ext = float(scene.lamp_ext()[0])
        dec = float(scene.decoy_ext()[0])
        print(f"[solve] {tag:12s} twist={tw:+6.1f}deg ext={ext * 1000:6.1f}mm "
              f"decoy={dec * 1000:5.1f}mm align_err="
              f"{math.degrees(float(scene.lamp_align_err()[0])):5.1f}deg "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

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

    def twist_tau(alpha: float) -> float:
        tw = float(scene.plug_twist()[0, ls])
        w = float(scene.plugs[ls].data.root_ang_vel_w[0, 0])
        w_des = max(-W_CAP_TW, min(W_CAP_TW, K_ANG * (alpha - tw)))
        return max(-TAU_MAX, min(TAU_MAX, KW * (w_des - w)))

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    ls = int(scene.lamp_socket[0])
    alpha = math.radians(float(scene.slot_deg[0, ls]))
    print(f"[solve] seed={args.seed} lamp_socket={ls} "
        f"slot=({float(scene.slot_deg[0, 0]):+.1f}, {float(scene.slot_deg[0, 1]):+.1f})deg "
        f"-> target twist {math.degrees(alpha):+.1f}deg", flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.drive_t.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: twist the LAMP plug to the slot angle ==========================
    aligned = False
    for i in range(1200):  # 10 s budget; ~1 s expected
        tw = float(scene.plug_twist()[0, ls])
        w = float(scene.plugs[ls].data.root_ang_vel_w[0, 0])
        if abs(alpha - tw) < ALIGN_DONE and abs(w) < W_DONE:
            aligned = True
            break
        scene.drive_t[0, ls] = twist_tau(alpha)
        step(1)
        if i and i % 240 == 0:
            report("twisting")
    if not aligned:
        report("twist-fail")
        print("[solve] PHASE 1 FAILED: twist servo did not converge", flush=True)
        verdict(False)
    report("aligned")
    phase_score("phase1")  # ~0.250 (align latch)

    # ================= PHASE 2: pull straight out, twist held ==================================
    x_des = c.ext_goal + 0.008  # 48 mm target; stop past goal with margin
    kv = KV0
    out = False
    last_ext, last_ck = float(scene.lamp_ext()[0]), 0
    for i in range(2400):  # 20 s budget; ~2 s expected
        ext = float(scene.lamp_ext()[0])
        v = float(scene.plugs[ls].data.root_lin_vel_w[0, 0])
        if ext >= c.ext_goal + 0.005 and abs(v) < V_DONE:
            out = True
            break
        v_des = min(V_CAP, K_APP * (x_des - ext))
        scene.drive_f[0, ls] = max(-F_MAX, min(F_MAX, kv * (v_des - v)))
        scene.drive_t[0, ls] = twist_tau(alpha)  # hold alignment through the slot
        step(1)
        if i - last_ck >= 240:  # stall watch: escalate GAIN, not the cap
            if ext - last_ext < 0.0005 and kv < KV_MAX:
                kv = min(KV_MAX, kv * 1.5)
                print(f"[solve] pull stalled at {ext * 1000:.1f}mm -> KV={kv:.1f}",
                      flush=True)
            last_ext, last_ck = ext, i
        if i and i % 240 == 0:
            report("pulling")
    scene.drive_f[0, ls] = 0.0  # RELEASE: body damping parks the plug
    scene.drive_t[0, ls] = 0.0
    if not out:
        report("pull-fail")
        print("[solve] PHASE 2 FAILED: pull servo did not extract the plug", flush=True)
        verdict(False)
    report("released")

    def settle_ok() -> bool:
        return bool(scene.success()[0])

    settled = False
    for _ in range(48):  # up to 4 s to ring down
        step(10)
        if settle_ok():
            settled = True
            break
    if not settled:
        report("settle-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after release", flush=True)
        verdict(False)
    report("unplugged")
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.drive_t.abs().max()) == 0.0, \
        "drives must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
