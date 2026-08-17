"""solve — demonstration solution for DetergentDrawerScene (open_washing_machine_i252).

Scene-level env (robot="null"). Teleportation is TRANSPORT ONLY: the pod is
teleported from its table spawn to free air ABOVE the open target cell — and
nothing else. Every load-bearing interaction runs through contact dynamics:
  - OPEN: a velocity-servoed pull force on the drawer body (the stand-in for the
    Franka's grip on the handle plate, TASK.md) slides the drawer out along its
    live PrismaticJoint against damping, released short of the end stop;
  - DEPOSIT: the pod FALLS ~40 mm from the release pose into the blue-marked
    cell and settles by contact on the tray floor (never spawned seated);
  - SHUT: a velocity-servoed push force slides the drawer home with the pod
    riding inside, onto the joint's lower stop, and is released.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5
more simulated seconds with all drive buffers zero (asserted); only if success()
still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after
the verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.open_washing_machine_i252.solve --headless [--seed N]
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
    from simgen_tasks.open_washing_machine_i252 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Drawer slide servo (hand-scale numbers; drawer mass 0.30 kg, damping 5/s).
# Discrete stability: KV*dt/m = 15*(1/120)/0.30 ~ 0.42 < 1 against the
# one-substep wrench delay.
KX = 3.0  # 1/s outer position->rate gain
V_OPEN = 0.15  # m/s rate cap while pulling out
V_SHUT = 0.10  # m/s rate cap while pushing home (end stop ahead — arrive gently)
KV = 15.0  # N*s/m inner rate gain
F_MAX = 6.0  # N hard cap (an easy one-hand pull on the 56 mm handle plate)
G_DONE = 0.005  # m: position band for servo exit...
V_DONE = 0.02  # m/s: ...and slow (never release a moving drawer)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.detergent_drawer")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        g = float(scene.opening()[0])
        p = scene.pod_local()[0]
        print(f"[solve] {tag:12s} opening={g * 1000:6.1f}mm "
              f"pod_local=({float(p[0]) * 1000:+6.1f}, {float(p[1]) * 1000:+6.1f}, "
              f"{float(p[2]) * 1000:+6.1f})mm in_cell={bool(scene.in_cell()[0])} "
              f"shut={bool(scene.shut_now()[0])} "
              f"latch=(o={float(scene.opened_latch[0]):.0f}, d={float(scene.deposit_latch[0]):.0f}) "
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

    def slide_to(g_tgt: float, v_cap: float, budget: int = 1800) -> bool:
        """Velocity-cascade force servo on the drawer's pull axis: slide the drawer
        to opening `g_tgt` through the live joint + damping plant; release only
        when close AND slow."""
        for i in range(budget):
            g = float(scene.opening()[0])
            v = float(scene.drawer.data.root_lin_vel_w[0, 0])
            e = g_tgt - g
            if abs(e) < G_DONE and abs(v) < V_DONE:
                scene.drive_f[0] = 0.0
                return True
            v_des = max(-v_cap, min(v_cap, KX * e))
            scene.drive_f[0] = max(-F_MAX, min(F_MAX, KV * (v_des - v)))
            step(1)
            if i and i % 240 == 0:
                report("sliding")
        scene.drive_f[0] = 0.0
        return False

    # ================= reset + settle ============================================================
    env.reset(seed=args.seed)
    step(60)
    side = float(scene.side[0])
    print(f"[solve] seed={args.seed} target side={'+y (left as seen)' if side > 0 else '-y'} "
          f"crack0={float(scene.crack0[0]) * 1000:.1f}mm "
          f"pod_start={[round(float(v), 3) for v in scene.pod_start[0]]}", flush=True)
    report("reset")
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.pod_f.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: pull the drawer open =============================================
    g_open = c.stroke - 0.008  # just short of the outer stop (no slam)
    if not slide_to(g_open, V_OPEN):
        report("open-fail")
        print("[solve] PHASE 1 FAILED: drawer did not reach the open target", flush=True)
        verdict(False)
    step(60)
    report("opened")
    assert float(scene.opened_latch[0]) == 1.0, "opened latch must be set after the pull"
    phase_score("phase1-open")  # ~0.20

    # ================= PHASE 2: transport + gravity deposit ======================================
    # Teleport = TRANSPORT ONLY: place the pod in FREE AIR above the exposed target
    # cell (outside the hood's footprint), zero velocity, and let physics drop it in.
    origin = scene.env_origins[0]
    drawer_x = float(scene.drawer.data.root_pos_w[0, 0])
    drop = torch.zeros(1, 13, device=device)
    drop[0, 0] = drawer_x + 0.015  # cell-local +15 mm: inside the exposed span
    drop[0, 1] = float(origin[1]) + side * c.cell_y_c
    drop[0, 2] = c.rim_z + 0.031  # pod bottom ~20 mm above the rim, clear of the hood
    drop[0, 3] = 1.0
    drop[0, 0] += 0.0  # x is already world (drawer_x includes the env origin)
    self_check = drop[0, 0] - float(origin[0])
    assert self_check > c.front_face_x + 0.02, "drop point must be clear of the hood footprint"
    scene.pod.write_root_state_to_sim(drop, torch.tensor([0], device=device))
    for _ in range(40):  # fall + settle, watch the latch mature
        step(10)
        if float(scene.deposit_latch[0]) == 1.0 and bool(scene.settled()[0]):
            break
    report("deposited")
    if float(scene.deposit_latch[0]) != 1.0 or not bool(scene.in_cell()[0]):
        print("[solve] PHASE 2 FAILED: pod did not settle in the target cell", flush=True)
        verdict(False)
    phase_score("phase2-deposit")  # ~0.65

    # ================= PHASE 3: push the drawer shut =============================================
    if not slide_to(0.0, V_SHUT):
        # the joint's lower stop can hold the servo a hair outside G_DONE: accept shut_now
        if not bool(scene.shut_now()[0]):
            report("shut-fail")
            print("[solve] PHASE 3 FAILED: drawer did not come home", flush=True)
            verdict(False)
        scene.drive_f[0] = 0.0
    step(90)
    report("shut")
    ok = False
    for _ in range(48):  # up to 4 s for success (shut + in-cell + settled) to hold
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    if not ok:
        report("settle-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after the push", flush=True)
        verdict(False)
    phase_score("phase3-shut")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ================
    assert float(scene.drive_f.abs().max()) == 0.0 and float(scene.pod_f.abs().max()) == 0.0, \
        "drives must be zero for persistence"
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
