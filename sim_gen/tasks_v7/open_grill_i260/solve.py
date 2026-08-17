"""solve — demonstration solution for ServingShelfScene (open_grill_i260).

Scene-level env (robot="null"). Teleportation is TRANSPORT ONLY: the platter is
teleported from the cart top to free air ABOVE the seated shelf — and nothing
else. Every load-bearing interaction runs through contact dynamics:
  - RAISE: a velocity-cascaded hinge-torque servo (gravity feedforward + rate
    loop, the stand-in for the Franka's grip on the leaf's grasp bar) swings the
    target leaf from its hanging stop up past vertical onto the +95 deg keeper
    stop, then RELEASES — gravity alone holds it there (over-vertical rest);
  - BRACE: a small pivot-torque servo (the finger on the orange tab) swings the
    brace bar 90 deg out under the shelf; the same torque magnitude cannot do
    this while the leaf hangs (smoke proves the sweep jams on the hanging leaf);
  - LOWER: the hinge servo brings the leaf back down; the leaf SEATS by contact
    on the deployed brace bar (release only when level and slow — without the
    brace it would just fall through level, which smoke also proves);
  - SERVE: the platter FALLS ~37 mm from the release pose and settles on the
    shelf by contact (never spawned seated).
All hinge torques go through the scene's post_step plant (body-frame along the
body-local joint axis — drag-invariant on this stack); rates are the scene's
finite-difference readbacks (root_ang_vel_w is phantom under external wrenches).

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5
more simulated seconds with all drive buffers zero (asserted); only if success()
still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after
the verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Servo stability audit (dt = 1/120):
  leaf: I_hinge ~ 5.7e-3 kg m^2 -> KV_L*dt/I = 0.15*0.0083/5.7e-3 ~ 0.22 < 1
  brace: I_pivot ~ 4.7e-4 kg m^2 -> KV_B*dt/I = 0.012*0.0083/4.7e-4 ~ 0.21 < 1

Run (forge): python -u -m simgen_tasks.open_grill_i260.solve --headless [--seed N]
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
    from simgen_tasks.open_grill_i260 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

# Leaf hinge servo (velocity cascade + gravity feedforward).
KX_L = 3.0  # 1/s outer angle->rate gain
W_L = 1.5  # rad/s rate cap (a calm one-hand swing)
KV_L = 0.15  # N*m*s/rad inner rate gain
# Brace pivot servo.
KX_B = 3.0
W_B = 2.0
KV_B = 0.012


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.serving_shelf")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    side = 1.0
    col = 0

    def report(tag: str) -> None:
        p = scene.platter_local()[0]
        print(f"[solve] {tag:12s} leaf={math.degrees(float(scene.target_leaf_angle()[0])):+7.1f}deg "
              f"brace={math.degrees(float(scene.target_brace_angle()[0])):+6.1f}deg "
              f"decoy=({math.degrees(float(scene.decoy_leaf_angle()[0])):+6.1f}, "
              f"{math.degrees(float(scene.decoy_brace_angle()[0])):+5.1f})deg "
              f"plat=({float(p[0]) * 1000:+6.1f}, {float(p[1]) * 1000:+6.1f}, "
              f"{float(p[2]) * 1000:+6.1f})mm on_shelf={bool(scene.platter_on_shelf()[0])} "
              f"latch=(u={float(scene.up_latch[0]):.0f}, b={float(scene.brace_latch[0]):.0f}, "
              f"s={float(scene.seat_latch[0]):.0f}, p={float(scene.plat_latch[0]):.0f}) "
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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def leaf_servo(th_tgt: float, *, exit_lo: float | None = None,
                   exit_hi: float | None = None, budget: int = 2400) -> bool:
        """Velocity-cascaded hinge torque on the target leaf: gravity ff + rate
        loop toward `th_tgt`; exits when the angle crosses the given bound AND the
        FD rate is slow. Leaves the drive holding (caller releases)."""
        for i in range(budget):
            th = float(scene.target_leaf_angle()[0])
            w = float(scene.target_rate()[0])
            done = ((exit_lo is not None and th > exit_lo and abs(w) < 0.3)
                    or (exit_hi is not None and th < exit_hi and abs(w) < 0.3))
            if done:
                return True
            w_des = max(-W_L, min(W_L, KX_L * (th_tgt - th)))
            ff = c.leaf_mgr * math.cos(th)
            scene.leaf_tau[0, col] = ff + KV_L * (w_des - w)
            step(1)
            if i and i % 240 == 0:
                report("leaf-servo")
        return False

    def brace_servo(ph_tgt: float, budget: int = 1200) -> bool:
        for i in range(budget):
            ph = float(scene.target_brace_angle()[0])
            w = float(scene.brace_rate[0, col])
            if abs(ph_tgt - ph) < math.radians(3.0) and abs(w) < 0.2:
                scene.brace_tau[0, col] = 0.0
                return True
            w_des = max(-W_B, min(W_B, KX_B * (ph_tgt - ph)))
            scene.brace_tau[0, col] = KV_B * (w_des - w)
            step(1)
            if i and i % 240 == 0:
                report("brace-servo")
        scene.brace_tau[0, col] = 0.0
        return False

    # ================= reset + settle ============================================================
    env.reset(seed=args.seed)
    step(60)
    side = float(scene.side[0])
    col = 0 if side > 0 else 1
    m_leaf = float(scene.leaves[side].root_physx_view.get_masses()[0].sum())
    m_plat = float(scene.platter.root_physx_view.get_masses()[0].sum())
    print(f"[solve] seed={args.seed} serve side={'+y (left)' if side > 0 else '-y (right)'} "
          f"cart_pose0={[round(float(v), 3) for v in scene.cart_pose0[0, :3]]} "
          f"plat_start={[round(float(v), 3) for v in scene.plat_start[0]]} "
          f"m_leaf={m_leaf:.3f} m_plat={m_plat:.3f}", flush=True)
    assert abs(m_leaf - c.leaf_mass) < 0.05, "leaf MassAPI not applied by the custom spawner"
    assert abs(m_plat - c.platter_mass) < 0.05, "platter MassAPI not applied"
    report("reset")
    assert float(scene.leaf_tau.abs().max()) == 0.0 and \
        float(scene.brace_tau.abs().max()) == 0.0 and \
        float(scene.platter_f.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: raise the leaf onto the keeper stop ==============================
    if not leaf_servo(c.leaf_hi + math.radians(4.0), exit_lo=math.radians(91.0)):
        report("raise-fail")
        print("[solve] PHASE 1 FAILED: leaf did not reach the keeper", flush=True)
        verdict(False)
    scene.leaf_tau[0, col] = 0.0  # release — gravity holds the over-vertical rest
    step(90)
    report("raised")
    if math.degrees(float(scene.target_leaf_angle()[0])) < 88.0:
        print("[solve] PHASE 1 FAILED: leaf fell off the keeper after release", flush=True)
        verdict(False)
    assert float(scene.up_latch[0]) == 1.0, "up latch must be set after the raise"
    phase_score("phase1-raise")  # ~0.15

    # ================= PHASE 2: swing the brace out under the shelf ==============================
    if not brace_servo(math.radians(90.0)):
        report("brace-fail")
        print("[solve] PHASE 2 FAILED: brace did not deploy", flush=True)
        verdict(False)
    step(60)
    report("braced")
    assert float(scene.brace_latch[0]) == 1.0, "brace latch must be set after the swing"
    phase_score("phase2-brace")  # ~0.30

    # ================= PHASE 3: lower the leaf onto the brace ====================================
    if not leaf_servo(math.radians(-2.0), exit_hi=math.radians(1.5)):
        report("lower-fail")
        print("[solve] PHASE 3 FAILED: leaf did not come level", flush=True)
        verdict(False)
    scene.leaf_tau[0, col] = 0.0  # release — the brace carries it now
    step(120)
    report("seated")
    if not bool((scene.target_leaf_angle().abs() < c.leaf_level_tol)[0]):
        print("[solve] PHASE 3 FAILED: leaf not level after release", flush=True)
        verdict(False)
    assert float(scene.seat_latch[0]) == 1.0, "seat latch must be set after the lowering"
    phase_score("phase3-seat")  # ~0.55

    # ================= PHASE 4: transport + gravity serve ========================================
    # Teleport = TRANSPORT ONLY: place the platter in FREE AIR above the seated
    # shelf (clear of the grasp bar), zero velocity, and let physics drop it.
    leaf = scene.leaves[side]
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415
    drop = torch.zeros(1, 13, device=device)
    off = torch.tensor([[0.0, c.plat_seat_y, 0.05]], device=device)
    drop[0, 0:3] = leaf.data.root_pos_w[0] + quat_apply(leaf.data.root_quat_w, off)[0]
    drop[0, 3:7] = leaf.data.root_quat_w[0]
    scene.platter.write_root_state_to_sim(drop, torch.tensor([0], device=device))
    for _ in range(40):  # fall + settle, watch the latch mature
        step(10)
        if float(scene.plat_latch[0]) == 1.0 and bool(scene.settled()[0]):
            break
    report("served")
    if float(scene.plat_latch[0]) != 1.0 or not bool(scene.platter_on_shelf()[0]):
        print("[solve] PHASE 4 FAILED: platter did not settle on the shelf", flush=True)
        verdict(False)
    phase_score("phase4-serve")  # ~0.75

    # ================= success + persistence (>= 3.5 simulated seconds, hands off) ==============
    ok = False
    for _ in range(48):  # up to 4 s for the full structure predicate to hold
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    if not ok:
        report("settle-fail")
        print("[solve] FAILED: success() not reached after the serve", flush=True)
        verdict(False)
    phase_score("success")  # 1.000

    assert float(scene.leaf_tau.abs().max()) == 0.0 and \
        float(scene.brace_tau.abs().max()) == 0.0 and \
        float(scene.platter_f.abs().max()) == 0.0, "drives must be zero for persistence"
    persist = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist} steps ({persist * env.dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
