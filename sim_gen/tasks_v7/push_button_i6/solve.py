"""solve — TELEPORT solution for WeighbridgeScene (push_button_i6).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT ONLY; the
load-bearing interaction — the spring compression that IS this task — goes through
contact dynamics:

  PHASE 1  transport the heavy load cube from its floor scatter slot to a hover pose
           centred over the LIVE plate axis, with the cube's bottom face 28 mm ABOVE
           the plate top — deliberately OUTSIDE the on-plate scoring band
           (on_plate_z_tol = 20 mm), so nothing can latch at the teleport instant
           (asserted in code). The lift latch (0.2) fires because the cube is high in
           free space, which is honest transport credit.
  PHASE 2  hands off. Gravity drops the cube the last 28 mm; the impact, the spring
           compression to the joint stop, and the sustained 90-substep press under the
           resting 0.50 kg are all pure contact dynamics — nothing is pinned, no
           velocity is clamped, no force is applied to any task object. success()
           requires live plate depth >= 28 mm, settled on-plate mass >= the spring's
           285 g trigger, plate still, sustained.
  PHASE 3  hands off for >= 3 simulated seconds more; success() must persist (success
           here is LIVE state — a bounce-off or slide-off would revert it) before the
           verdict is printed.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (the printed
sequence never decreases — asserted), and exactly `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds after the persistence window. Hard exit (os._exit) after the
verdict, with a watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (measured and previously
EXECUTED by a real arm in this exact scene: top-down pinch of the 55 mm cube, carry to
the plate axis, press-down release) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.push_button_i6.solve --headless
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
    from simgen_tasks.push_button_i6 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weighbridge")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    load_edge = c.blocks[0][1]  # the 55 mm load cube, manifest slot 0

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def settle_until(pred, max_steps: int = 900, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        lp = (scene.blocks["load"].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:16s} depth={float(scene.plate_depth()[0]) * 1000:5.1f}mm "
              f"m_on={float(scene.loaded_mass()[0]):.3f}kg "
              f"load=({lp[0]:+.3f},{lp[1]:+.3f},{lp[2]:+.3f}) "
              f"hold={int(scene._hold[0])} lifted={bool(scene._lifted[0])} "
              f"placed={bool(scene._placed[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        if ok:
            print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        else:
            print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def teleport(body, pos_w, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """TRANSPORT: set a root pose with zero velocity. Never used to enter a scoring
        band — the release point is chosen outside the on-plate band (asserted below)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos_w], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, ids)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: transport the load cube over the plate ========================
    # Hover pose: centred on the LIVE plate axis, cube bottom 28 mm above the LIVE plate
    # top — outside the on-plate band (|bottom - plate_top| < on_plate_z_tol = 20 mm), so
    # the teleport instant latches no placement credit. Identity yaw: flat-on-flat drop.
    release_gap = 0.028
    assert release_gap > c.on_plate_z_tol, "release point must be outside the on-plate band"
    plate = scene.plate.data.root_pos_w[0]
    plate_top = float(plate[2]) + c.plate_size[2] / 2
    hover = [float(plate[0]), float(plate[1]), plate_top + release_gap + load_edge / 2]
    teleport(scene.blocks["load"], hover)
    step(1)
    report("hover")
    assert not bool(scene._placed[0]), "placed latch must not fire at the teleport"
    assert bool(scene._lifted[0]), "lift latch should fire while hovering in free space"
    assert float(scene.loaded_mass()[0]) < 1e-6, "no mass may count as on-plate yet"
    phase_score("phase1")  # 0.200 (lifted latch only)

    # ================= PHASE 2: hands off — drop, spring compression, sustained press ==========
    # Gravity closes the last 28 mm; the spring bottoms out under the resting 0.50 kg
    # (trigger 285 g) and the 90-substep sustained-press counter fills. Pure contact.
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("pressed")
    if not ok:
        print("[solve] PHASE 2 FAILED: plate not held at depth by settled resting mass",
              flush=True)
        verdict(False)
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3 simulated seconds, hands off) ================
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
