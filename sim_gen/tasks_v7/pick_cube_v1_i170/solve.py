"""solve — teleport solution for WeightAirlockScene (pick_cube_v1_i170).

Scene-level env (robot="null"). Teleports handle TRANSPORT ONLY (carrying a cube
across free table space — what the Franka's grasp+carry does; TASK.md); every
load-bearing interaction goes through contact dynamics:

  PHASE 1 (weigh the plate): the blue weight cube is teleported to hover just above
    the yellow plate and RELEASED — its gravity, through real contact, drives the
    spring-loaded plate to the bottom stop and the interlock slides the gate open.
    Nothing writes the plate or gate poses, ever.
  PHASE 2 (push the cargo through): the red cube is teleported to the table in front
    of the doorway (free space, gate already open), then PUSHED through the doorway
    into the vault with a capped-force velocity servo on the scene's `push_f` buffer
    (the stand-in for a fingertip push; K*dt/m kept under 1 against the one-substep
    wrench delay, friction feedforward escalates on stalls, never the cap). The whole
    transit slides on the table through the open doorway under contact dynamics.
  PHASE 3 (remove the weight): the blue cube is teleported OFF the plate to a free
    parking spot (transport again); the spring raises the plate and the interlock
    drives the gate shut — all physics.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5
more simulated seconds with the push buffers zero (asserted); only if success()
still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the
verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.pick_cube_v1_i170.solve --headless [--seed N]
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

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.pick_cube_v1_i170 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Push servo (fingertip authority on a 60 g cube). K*dt/m = 6*(1/120)/0.06 = 0.83 < 1
# against the one-substep wrench delay. A constant friction FEEDFORWARD carries the
# sliding-friction load (the pure P servo equilibrates at a crawl otherwise); stalls
# escalate the FEEDFORWARD, never the gain past the bound, never the F_CAP clamp.
V_DES = 0.12  # m/s push speed through the doorway
KV = 6.0  # N*s/m rate gain (kept at the wrench-delay stability bound)
F_FF0 = 0.45  # N starting friction feedforward
F_FF_MAX = 1.5  # escalation ceiling (still under F_CAP)
F_CAP = 2.0  # N hard cap on the push force
KX = 2.0  # N/m lateral centering on the doorway axis
KDX = 1.0  # N*s/m lateral damping


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weight_airlock")().build(num_envs=1, device=device)
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
        rp, bp = local(scene.red), local(scene.blue)
        print(f"[solve] {tag:12s} red=({float(rp[0]):+.3f},{float(rp[1]):+.3f},"
              f"{float(rp[2]):.3f}) blue=({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
              f"{float(bp[2]):.3f}) gate={float(scene.gate_disp()[0]) * 1000:6.1f}mm "
              f"plate_z={float(scene.plate_z()[0]):.3f} "
              f"open={bool(scene.gate_open()[0])} closed={bool(scene.gate_closed()[0])} "
              f"inside={bool(scene.red_inside()[0])} "
              f"latch=({int(scene.latch_open[0])},{int(scene.latch_inside[0])}) "
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

    def teleport(body, x: float, y: float, z: float, cube_idx: int | None = None) -> None:
        """TRANSPORT ONLY: set a cube's pose in free space, identity yaw, zero vel.
        Re-references the wrench-drag pre-encode for that cube."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = x, y, z
        st[0, 3] = 1.0
        st[0, 0:3] += scene.env_origins[0]
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        if cube_idx is not None:
            scene._q_ref[0, cube_idx] = st[0, 3:7]

    def push_red_to(y_stop: float, budget: int = 1500) -> bool:
        """Capped-force velocity servo + friction feedforward along +y (world intent;
        post_step pre-encodes the wrench frame drag) with lateral centering on x=0."""
        ff = F_FF0
        last_y = float(local(scene.red)[1])
        last_ck = 0
        for i in range(budget):
            rp = local(scene.red)
            rv = scene.red.data.root_lin_vel_w[0]
            if float(rp[1]) >= y_stop:
                scene.push_f[0, 0] = 0.0
                return True
            fy = max(-F_CAP, min(F_CAP, KV * (V_DES - float(rv[1])) + ff))
            fx = max(-1.0, min(1.0, KX * (0.0 - float(rp[0])) - KDX * float(rv[0])))
            scene.push_f[0, 0, 0] = fx
            scene.push_f[0, 0, 1] = fy
            scene.push_f[0, 0, 2] = 0.0
            step(1)
            if i - last_ck >= 120:  # stall watch: escalate the FEEDFORWARD, not the cap
                y = float(local(scene.red)[1])
                if y - last_y < 0.06 and ff < F_FF_MAX:  # < ~half V_DES average
                    ff = min(F_FF_MAX, ff * 1.5)
                    print(f"[solve] push slow at y={y:+.3f} -> FF={ff:.2f} N", flush=True)
                last_y, last_ck = y, i
        scene.push_f[0, 0] = 0.0
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
    print(f"[solve] seed={args.seed} slots(red,blue,decoy)="
          f"{[tuple(round(v, 3) for v in scene.spawn_xy[0, i].tolist()) for i in range(3)]}",
          flush=True)
    report("reset")
    assert float(scene.push_f.abs().max()) == 0.0
    assert bool(scene.gate_closed()[0]) and bool(scene.plate_up()[0]), \
        "mechanism must rest closed/up at reset"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: blue weight onto the plate =====================================
    # Transport: hover the blue cube just above the plate, release; gravity + contact
    # do the weighing — the plate descends and the interlock slides the gate open.
    teleport(scene.blue, c.ped_center[0], c.ped_center[1],
             c.plate_rest_z + c.plate_size[2] / 2 + c.blue_size / 2 + 0.008, cube_idx=1)
    if not wait_for(lambda: scene.plate_depressed()[0] & scene.gate_open()[0], 480):
        report("weigh-fail")
        print("[solve] PHASE 1 FAILED: plate not depressed / gate not open", flush=True)
        verdict(False)
    step(60)  # let the gate settle at the open stop
    report("gate-open")
    phase_score("phase1")  # ~0.250

    # ================= PHASE 2: push the red cargo through the doorway =========================
    # Transport: stage the red cube on the table in front of the doorway (free space —
    # the gate is open, the doorway is clear), then PUSH it through under contact.
    teleport(scene.red, 0.0, -0.06, c.z0 + c.red_size / 2 + 0.003, cube_idx=0)
    step(30)  # settle the set-down
    if not push_red_to(0.125):
        report("push-fail")
        print("[solve] PHASE 2 FAILED: red cube did not reach the interior", flush=True)
        verdict(False)
    step(90)  # slide out + friction stop; the inside latch matures while slow
    report("red-inside")
    if not bool(scene.red_inside()[0]):
        print("[solve] PHASE 2 FAILED: red not inside after settling", flush=True)
        verdict(False)
    phase_score("phase2")  # ~0.600

    # ================= PHASE 3: remove the weight — the gate locks the cargo in ================
    teleport(scene.blue, -0.25, -0.28, c.z0 + c.blue_size / 2 + 0.003, cube_idx=1)
    if not wait_for(lambda: scene.gate_closed()[0] & scene.plate_up()[0], 480):
        report("close-fail")
        print("[solve] PHASE 3 FAILED: gate did not close after weight removal", flush=True)
        verdict(False)
    if not wait_for(lambda: scene.success()[0], 480):
        report("settle-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after lock-up", flush=True)
        verdict(False)
    report("locked")
    phase_score("phase3")  # 1.000

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
    main()
