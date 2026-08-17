"""solve — TELEPORT solution for BallastVaultScene (libero_pick_orange_juice_i129).

Scene-level env (robot="null"). Teleportation handles TRANSPORT ONLY — every
load-bearing interaction runs through the live contact/hinge dynamics, and the lid
is NEVER touched (`lid_drive` stays zero throughout, asserted):

  PHASE 1 (load): the IRON block is teleported from its staging slot to a hover pose
  ~60 mm above the yellow lever cup, along the cup's tilted normal, orientation
  matched to the cup floor, zero velocity — then RELEASED. Gravity does the rest:
  the block drops into the cup by real contact, and its weight (0.45 kg at a
  ~0.15 m lever arm, ~5x the lid's closing imbalance) swings the lid open to the
  65-degree stop through the hinge joint. Exactly what a Franka does after grasping
  the knob: carry through free space, hover, open the gripper.

  PHASE 2 (insert): the CARTON is teleported to a hover pose above the OPEN half of
  the vault mouth (vault-local y = -0.045, z = 0.33, yaw-aligned), zero velocity,
  and released. It free-falls ~25 cm through the aperture and seats on the vault
  floor by contact — the timed-aperture insertion itself is entirely live physics.

  PHASE 3 (unload): the IRON block is teleported OUT of the cup back to the carton's
  vacated staging slot (transport a gripper performs: grasp knob, lift along the cup
  axis, carry, set down). The lid then swings SHUT on its own — closing imbalance
  vs. viscous hinge damping, parked on the joint's 0-degree stop. No torque is ever
  written; the seal is the plant's own equilibrium.

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (the
printed sequence never decreases — asserted). After success() first holds, keeps
simulating >= 3 more simulated seconds hands-off; only if success() still holds (it
is live state — a lid that creeps open or a block that slumps back would revert it)
prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict,
with a watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.libero_pick_orange_juice_i129.solve --headless [--seed N]
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
    from simgen_tasks.libero_pick_orange_juice_i129 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

HOVER_CUP = 0.060   # m above the cup floor, along the cup's tilted normal
DROP_Y = -0.045     # vault-local y of the carton drop line (open half of the mouth)
DROP_Z = 0.33       # vault-local z of the carton hover (fall ~25 cm; no tunneling)


def _qx(ang: float, device) -> torch.Tensor:
    q = torch.zeros(1, 4, device=device)
    q[0, 0], q[0, 1] = math.cos(ang / 2), math.sin(ang / 2)
    return q


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_vault")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def deg() -> float:
        return math.degrees(float(scene.lid_angle()[0]))

    def report(tag: str) -> None:
        print(f"[solve] {tag:10s} lid={deg():6.1f}deg rate={float(scene.lid_rate()[0]):+.3f} "
              f"iron_in_cup={bool(scene.in_cup('iron')[0])} "
              f"carton_inside={bool(scene.carton_inside()[0])} "
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

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def teleport(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(1, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    slots = scene.slot_of[0].tolist()  # item order: carton, iron, foam
    print(f"[solve] seed={args.seed} slots(carton,iron,foam)={slots}", flush=True)
    report("reset")
    assert float(scene.lid_drive.abs().max()) == 0.0
    assert deg() < 5.0, "lid must rest closed after reset"
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: load the iron block (transport + gravity drop) =================
    # Hover along the cup's tilted normal, puck matched flat to the cup floor; the drop,
    # the seating, and the lid swinging open are all live contact + hinge dynamics.
    tilt = math.radians(c.cup_tilt_deg)
    hover_local = torch.tensor(
        [[0.0, c.cup_y - HOVER_CUP * math.sin(tilt), c.cup_z + HOVER_CUP * math.cos(tilt)]],
        device=device)
    hover_w = scene.lid.data.root_pos_w + quat_apply(scene.lid.data.root_quat_w, hover_local)
    q_iron = scene_mod._qmul(scene.lid.data.root_quat_w, _qx(tilt, device))
    teleport(scene.items["iron"], hover_w, q_iron)
    step(5)
    report("dropped")
    if not settle_until(lambda: bool(scene.in_cup("iron")[0])
                        and deg() > 58.0 and abs(float(scene.lid_rate()[0])) < 0.2,
                        max_steps=720):
        report("load-fail")
        print("[solve] PHASE 1 FAILED: iron did not open + hold the lid", flush=True)
        verdict(False)
    report("held-open")
    phase_score("phase1")  # 0.350 (approach + open latches)

    # ================= PHASE 2: drop the carton through the open aperture ======================
    drop_local = torch.tensor([[0.0, DROP_Y, DROP_Z]], device=device)
    drop_w = scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, drop_local)
    teleport(scene.items["carton"], drop_w, scene.vault.data.root_quat_w)
    step(5)
    report("released")
    if not settle_until(lambda: bool(scene.carton_inside()[0])
                        and float(scene.items["carton"].data.root_lin_vel_w.norm()) < 0.05,
                        max_steps=600):
        report("insert-fail")
        print("[solve] PHASE 2 FAILED: carton did not come to rest inside", flush=True)
        verdict(False)
    if not bool(scene.in_cup("iron")[0]):
        report("iron-lost")
        print("[solve] PHASE 2 FAILED: iron block left the cup during the insert", flush=True)
        verdict(False)
    report("inserted")
    phase_score("phase2")  # 0.650 (insert latch)

    # ================= PHASE 3: unload the iron block; the lid seals ITSELF ====================
    # Transport only: lift the block out of the cup and set it down on the carton's
    # vacated staging slot. The close is pure plant — imbalance vs. viscous damping.
    origin = scene.env_origins[0]
    slot_xy = torch.tensor(
        [c.table_pos[0] + c.slot_x[slots[0]], c.table_pos[1]], device=device)
    park = torch.zeros(1, 3, device=device)
    park[0, 0:2] = slot_xy
    park[0, 2] = c.table_size[2] + c.puck_h / 2 + 0.002
    park[0, :] += origin
    teleport(scene.items["iron"], park, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
    step(5)
    report("unloaded")
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=900):
        report("seal-fail")
        print("[solve] PHASE 3 FAILED: lid did not seal / success() not reached", flush=True)
        verdict(False)
    report("sealed")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3 simulated seconds, hands off) ================
    assert float(scene.lid_drive.abs().max()) == 0.0, "solve must never touch the lid"
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
