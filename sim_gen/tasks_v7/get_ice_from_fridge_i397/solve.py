"""solve — teleport-transport + contact-dynamics solution for IceCaddyDockScene
(get_ice_from_fridge_i397).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY: the caddy (with its
sealed plug and its settled ice balls, all preserved in the caddy's own frame) is
relocated through free space exactly as a carrying hand would. Every load-bearing
interaction runs through the live plant:

  PHASE 1 (seat): transport the assembly to a hover 35 mm above the real dock and
  DROP it. The dock's probe post meets the plug stem through the bottom port and
  pushes the valve open against its spring under the falling caddy's weight — the
  scene's own contact dynamics, no state is written to the plug's joint. A 2.5 N
  hold-down (a hand steadying the caddy, ~half its weight) keeps it seated.

  PHASE 2 (drain): dwell, hold-down on. Gravity feeds the balls down the slick
  30-degree funnel, under the lifted head rim, through the throat past the probe,
  into the basin. If a ball dawdles, a gentle world-frame xy wiggle (<= 1.5 N, 3 Hz)
  on the caddy shakes it loose. Waits until every present ball's delivery latch fires.

  PHASE 3 (park): pull the caddy straight up (8 N, ~1.5x its weight) until the stem
  clears the probe — the spring re-seals the valve in flight, unscripted — then
  transport the caddy+plug (balls STAY in the basin) to a hover over the green pad
  and drop it. Settle until success() is live.

  PHASE 4 (persistence): all probe buffers asserted zero, >= 3.5 simulated seconds
  hands-off; success() is current state (a valve that were still open, a tipped
  caddy, a ball that escaped the basin would revert it). Only then SUCCESS.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (asserted non-decreasing).
Hard exit (os._exit) after the verdict, watchdog Timer as backstop.

Run (forge): python -u -m simgen_tasks.get_ice_from_fridge_i397.solve --headless [--seed N]
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
    from simgen_tasks.get_ice_from_fridge_i397 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

# Hand-scale drives (the honesty argument): the hold-down is ~half the caddy's weight,
# the lift ~1.5x it — one-hand forces on a 0.5 kg caddy carried by its bridge handle.
F_HOLD = 2.5  # N, straight-down steady while seated / draining
F_LIFT = 8.0  # N, straight-up pull to unseat (weight 4.9 N + plug 0.4 N)
F_WIGGLE = 1.5  # N, xy shake amplitude if a ball dawdles
WIGGLE_HZ = 3.0
HOVER = 0.035  # m, transport hover above the landing height (free drop)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ice_caddy_dock")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.arange(1, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def delivered() -> int:
        return int((scene.ball_latch[0] * scene.present[0].float()).sum().round())

    def report(tag: str) -> None:
        p = scene.caddy_pos()[0].tolist()
        print(f"[solve] {tag:12s} caddy=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f}) "
              f"ext={float(scene.plug_ext()[0]) * 1000:5.1f}mm "
              f"delivered={delivered()}/{int(scene.n_balls[0])} "
              f"parked={bool(scene.parked()[0])} score={sc():.3f} "
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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def transport(xy, z: float, with_balls: bool) -> None:
        """TRANSPORT ONLY: relocate the caddy to (xy, z) at identity yaw, carrying the
        plug (and, when still inside, the balls) at their CURRENT caddy-frame offsets.
        Zero velocities; nothing else is written — the valve state is whatever the
        spring/contacts made it."""
        from isaaclab.utils.math import quat_apply_inverse

        cp = scene.caddy.data.root_pos_w[0]
        cq = scene.caddy.data.root_quat_w[0:1]

        def rel(body) -> torch.Tensor:
            return quat_apply_inverse(cq, (body.data.root_pos_w[0] - cp).unsqueeze(0))[0]

        target = torch.tensor([xy[0], xy[1], z], device=device) + scene.env_origins[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = target
        st[0, 3] = 1.0
        plug_off = rel(scene.plug)
        ball_off = [rel(b) for b in scene.balls]
        scene.caddy.write_root_state_to_sim(st, ids)
        ps = st.clone()
        ps[0, 0:3] = target + plug_off
        scene.plug.write_root_state_to_sim(ps, ids)
        if with_balls:
            for i, b in enumerate(scene.balls):
                if not bool(scene.present[0, i]):
                    continue
                bs = torch.zeros(1, 13, device=device)
                bs[0, 0:3] = target + ball_off[i]
                bs[0, 3] = 1.0
                b.write_root_state_to_sim(bs, ids)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    dock = scene.dock_xy[0].tolist()
    pad = scene.pad_xy[0].tolist()
    print(f"[solve] seed={args.seed} side={float(scene.side[0]):+.0f} "
          f"dock=({dock[0]:.3f},{dock[1]:.3f}) pad=({pad[0]:.3f},{pad[1]:.3f}) "
          f"n_balls={int(scene.n_balls[0])}", flush=True)
    report("reset")
    assert float(scene.caddy_force_w.abs().max()) == 0.0
    assert float(scene.plug_force_w.abs().max()) == 0.0
    assert float(scene.plug_ext()[0]) < 0.004, "plug must start sealed"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: transport to the dock, drop to seat ============================
    transport(dock, c.seated_root_z + HOVER, with_balls=True)
    step(3)  # begin the free fall (lift latch fires from height — it IS lifted)
    seated = False
    for i in range(360):  # 3 s budget for drop + probe-vs-spring seating
        scene.caddy_force_w[0, 2] = -F_HOLD
        step(1)
        if float(scene.plug_ext()[0]) >= 0.020 \
                and abs(float(scene.caddy_pos()[0, 2]) - c.seated_root_z) < 0.006:
            seated = True
            break
    if not seated:
        report("seat-fail")
        print("[solve] PHASE 1 FAILED: probe did not open the valve", flush=True)
        verdict(False)
    step(30)  # let the seat settle under the hold-down
    report("seated-open")
    phase_score("phase1")  # 0.300 (lift + dock-open latches)

    # ================= PHASE 2: drain (hold-down on; wiggle if a ball dawdles) =================
    nb = int(scene.n_balls[0])
    drained = False
    for i in range(1800):  # 15 s budget; ~2-4 s expected
        scene.caddy_force_w[0, 2] = -F_HOLD
        if i > 480:  # dawdler: add a gentle 3 Hz xy shake
            ph = 2.0 * math.pi * WIGGLE_HZ * (i * env.dt)
            scene.caddy_force_w[0, 0] = F_WIGGLE * math.sin(ph)
            scene.caddy_force_w[0, 1] = F_WIGGLE * math.cos(ph)
        step(1)
        if delivered() >= nb:
            drained = True
            break
        if i and i % 360 == 0:
            report("draining")
    scene.caddy_force_w[0] = 0.0
    if not drained:
        report("drain-fail")
        print("[solve] PHASE 2 FAILED: not every ball drained", flush=True)
        verdict(False)
    report("drained")
    phase_score("phase2")  # 0.700 (all balls latched delivered)

    # ================= PHASE 3: lift off (spring re-seals), park on the pad ====================
    unseated = False
    for i in range(480):  # 4 s budget
        scene.caddy_force_w[0, 2] = F_LIFT
        step(1)
        if float(scene.caddy_pos()[0, 2]) > c.seated_root_z + 0.055:
            unseated = True
            break
    scene.caddy_force_w[0] = 0.0
    if not unseated:
        report("lift-fail")
        print("[solve] PHASE 3 FAILED: could not lift off the dock", flush=True)
        verdict(False)
    step(10)  # free flight: the spring snaps the valve shut on its own
    ext_fly = float(scene.plug_ext()[0])
    print(f"[solve] airborne: valve re-sealed to ext={ext_fly * 1000:.1f}mm "
          f"(spring, hands-off)", flush=True)
    transport(pad, c.park_root_z + 0.030, with_balls=False)  # balls STAY in the basin
    ok_settle = False
    for _ in range(48):  # up to 4 s to land + settle into success
        step(10)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    if not ok_settle:
        report("park-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after parking", flush=True)
        verdict(False)
    report("parked")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.caddy_force_w.abs().max()) == 0.0, "caddy force must be zero"
    assert float(scene.caddy_torque_w.abs().max()) == 0.0, "caddy torque must be zero"
    assert float(scene.plug_force_w.abs().max()) == 0.0, "plug force must be zero"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[solve] CRASH: {type(e).__name__}: {e}", flush=True)
        os._exit(1)
