"""solve — teleport solution for CatwalkBridgeScene (pick_cube_i295).

Scene-level env (robot="null"). Teleports handle TRANSPORT ONLY (carrying a body across
free space — what the Franka's grasp+carry does; TASK.md); every load-bearing
interaction goes through contact dynamics:

  PHASE 1 (build the bridge): the plank is teleported to hover 1 cm ABOVE its spanning
    pose (the pose the arm reaches by inserting it level through the tunnel mouth — the
    peel maneuver in TASK.md) and RELEASED — it drops onto the two sills and seats
    under gravity + contact. Nothing ever writes a seated pose; if the seats were wrong
    the plank would tip into the trench (smoke proves the lengthwise shove does exactly
    that).
  PHASE 2 (cross the trench): the cube is teleported onto the plank's BACK seat, still
    outside the tunnel mouth (free space, arm-reachable), then PUSHED along the plank
    with a capped-force velocity servo on the scene's `push_f` buffer (the stand-in for
    the rod-tip push at cube mid-height; cap 0.40 N is under the mg = 0.49 N tipping
    bound, K*dt/m = 0.067 << 1 against the one-substep wrench delay). The whole
    crossing — riding the bridge over the open trench, dropping off the plank's far end
    onto the island floor — is contact dynamics; the trench is exactly where the cube
    goes if the bridge is absent or bad.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched credit
must not evaporate). After success() first holds, keeps simulating >= 3.5 more
simulated seconds with the push buffers zero (asserted); only if success() still holds
prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict,
watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.pick_cube_i295.solve --headless [--seed N]
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

import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.pick_cube_i295 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Push servo (rod-tip authority on a 50 g cube). K*dt/m = 0.4*(1/120)/0.05 = 0.067 << 1
# against the one-substep wrench delay. Cap 0.40 N stays under the mg = 0.49 N
# CoM-height tipping bound; the friction FEEDFORWARD carries the sliding load
# (mu*mg ~ 0.15 N) and escalates on stalls — never the cap.
V_DES = 0.08  # m/s crossing speed
KV = 0.4  # N*s/m rate gain
F_FF0 = 0.16  # N starting friction feedforward
F_FF_MAX = 0.32  # escalation ceiling (still under F_CAP)
F_CAP = 0.40  # N hard cap on the push force (< mg tipping bound)
KX = 3.0  # N/m lateral centering on the corridor axis
KDX = 0.6  # N*s/m lateral damping
F_LAT = 0.15  # N lateral clamp


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.catwalk_bridge")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def local(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        pp, cp = local(scene.plank), local(scene.cube)
        print(f"[solve] {tag:12s} plank=({float(pp[0]):+.3f},{float(pp[1]):+.3f},"
              f"{float(pp[2]):.3f}) cube=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
              f"{float(cp[2]):.3f}) bridge={bool(scene.bridge_ok()[0])} "
              f"on_island={bool(scene.cube_on_island()[0])} "
              f"latch=({int(scene.latch_bridge[0])},{int(scene.latch_transit[0])}) "
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

    def teleport(idx: int, x: float, y: float, z: float) -> None:
        """TRANSPORT ONLY: set a body's pose in free space, identity yaw, zero vel.
        Re-references the wrench-drag pre-encode for that body."""
        body = scene.bodies[idx]
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = x, y, z
        st[0, 3] = 1.0
        st[0, 0:3] += scene.env_origins[0]
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        scene._q_ref[0, idx] = st[0, 3:7]

    def push_cube_to(x_stop: float, budget: int = 2400) -> bool:
        """Capped-force velocity servo + friction feedforward along +x (world intent;
        post_step pre-encodes the wrench frame drag) with lateral centering on y=0.
        Prints the transit phase score the first time the latch fires."""
        ff = F_FF0
        last_x = float(local(scene.cube)[0])
        last_ck = 0
        transit_seen = bool(scene.latch_transit[0])
        for i in range(budget):
            p = local(scene.cube)
            v = scene.cube.data.root_lin_vel_w[0]
            if not transit_seen and bool(scene.latch_transit[0]):
                transit_seen = True
                report("transit")
                phase_score("transit")  # ~0.600 mid-crossing
            if float(p[0]) >= x_stop:
                scene.push_f[0, 1] = 0.0
                return True
            fx = max(-F_CAP, min(F_CAP, KV * (V_DES - float(v[0])) + ff))
            fy = max(-F_LAT, min(F_LAT, KX * (0.0 - float(p[1])) - KDX * float(v[1])))
            scene.push_f[0, 1, 0] = fx
            scene.push_f[0, 1, 1] = fy
            scene.push_f[0, 1, 2] = 0.0
            step(1)
            if i - last_ck >= 120:  # stall watch: escalate the FEEDFORWARD, not the cap
                x = float(local(scene.cube)[0])
                if x - last_x < 0.04 and ff < F_FF_MAX:  # < ~half V_DES average
                    ff = min(F_FF_MAX, ff * 1.4)
                    print(f"[solve] push slow at x={x:+.3f} -> FF={ff:.2f} N", flush=True)
                last_x, last_ck = x, i
        scene.push_f[0, 1] = 0.0
        return False

    def wait_for(fn, budget_steps: int, chunk: int = 5) -> bool:
        for _ in range(max(1, budget_steps // chunk)):
            if bool(fn()):
                return True
            step(chunk)
        return bool(fn())

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    print(f"[solve] seed={args.seed} swap={bool(scene.spawn_swap[0])} spawns(plank,cube,rod)="
          f"{[tuple(round(v, 3) for v in scene.spawn_xy[0, i].tolist()) for i in range(3)]}",
          flush=True)
    report("reset")
    assert float(scene.push_f.abs().max()) == 0.0
    assert not bool(scene.bridge_ok()[0]), "no bridge may exist at reset"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: build the bridge ===============================================
    # Transport: hover the plank 1 cm above its spanning pose (the arm's peel-insert
    # set-down point), release; gravity + the two sills do the seating.
    teleport(0, c.span_center_x, 0.0, c.plank_seat_z + 0.010)
    if not wait_for(lambda: scene.bridge_ok()[0], 360):
        report("bridge-fail")
        print("[solve] PHASE 1 FAILED: plank did not seat as a spanning bridge", flush=True)
        verdict(False)
    step(30)
    report("bridge")
    phase_score("phase1")  # ~0.250

    # ================= PHASE 2: push the cube across ===========================================
    # Transport: set the cube onto the plank's BACK seat, still outside the tunnel mouth
    # (arm-reachable free space), then PUSH it across under contact — bridge transit,
    # step-down at the plank's far end, out onto the island floor.
    teleport(1, -0.105, 0.0, c.cube_ride_z + 0.006)
    step(30)  # settle the set-down onto the plank seat
    if not push_cube_to(0.17):
        report("push-fail")
        print("[solve] PHASE 2 FAILED: cube did not reach the island interior", flush=True)
        verdict(False)
    step(60)  # slide out + friction stop on the island floor
    report("landed")
    if not wait_for(lambda: scene.success()[0], 480):
        report("settle-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after landing", flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    assert float(scene.push_f.abs().max()) == 0.0, "push buffers must be zero for persistence"
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
    except Exception:  # noqa: BLE001 — crash must exit NOW, not idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)
