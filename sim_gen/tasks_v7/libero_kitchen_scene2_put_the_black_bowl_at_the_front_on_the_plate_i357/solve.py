"""Teleport solution for RackTrayServeScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i357) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. EXTRACTION (applied force + contact): the tray starts CAPTIVE — on edge in the
   rack's 16 mm channel with a roof rail ~4 mm above its top edge, so no teleport
   shortcut exists that the rubric would credit for "getting it out" honestly; the
   extraction stage is earned by a bang-bang horizontal force along the rack's +x
   (the channel direction, from live rack-yaw readback), velocity-capped at
   0.12 m/s, escalating x1.5 on a 120-step stall. The wrench emulates a LOW PUSH
   POINT (a fingertip on the tray's lower half): the world force is paired with the
   torque of that same force applied 80 mm below the CoM, both converted to the
   tray's body frame per step. The tray slides out of the mouth onto the low-fenced
   apron entirely under contact dynamics.
2. TRANSPORT (teleport): one root-state write carries the extracted tray across
   free space to a pose 30 mm ABOVE the stand's well, flat, yaw matched to the
   stand's (randomized) heading. The write satisfies no rubric clause: the tray is
   airborne, and `tray_seated` demands the well-floor resting height.
3. SEATING (gravity + contact): the tray FALLS into the 190 mm well (10 mm slack
   per side) and lands flat on the well floor. `tray_seated` is produced entirely
   by the landing dynamics.
4. SERVING (teleport + gravity + contact): the FRONT bowl — chosen BY POSITION
   (the body nearest the front row slot, exactly what a camera would provide, then
   cross-checked against the scene's latched target identity) — is teleported to
   25 mm above the seated tray's center and dropped; it lands upright on the tray.
   success() judges the settled hands-off state (seated tray + target bowl on it +
   decoys clear + rest).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rack_tray_serve")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        t = int(scene.target[0])
        lx = float(scene.tray_local_x()[0]) * 1000
        seat = bool(scene.tray_seated()[0])
        ot = bool(scene.on_tray()[0, t])
        print(f"[solve] {tag:18s} | tgt={t} tray_x={lx:+7.1f}mm out={bool(scene.tray_out()[0])} "
              f"seated={seat} on_tray={ot} clear={bool(scene.decoys_clear()[0])} "
              f"still_t={int(scene._still_tray[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def zero_wrench() -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.tray.set_external_force_and_torque(z, z)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything settles at spawn
    tray_mass = float(scene.tray.root_physx_view.get_masses().reshape(-1)[0])
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    o = scene.env_origins[0]
    rack_p = scene.rack.data.root_pos_w[0] - o
    rack_yaw = float(scene._yaw_of(scene.rack.data.root_quat_w)[0])
    stand_p = scene.stand.data.root_pos_w[0] - o
    stand_yaw = float(scene._yaw_of(scene.stand.data.root_quat_w)[0])
    print(f"[solve] layout readback (seed {args.seed}): tray_mass={tray_mass:.3f} "
          f"bowl_masses={masses} rack=({float(rack_p[0]):+.3f},{float(rack_p[1]):+.3f}) "
          f"rack_yaw={math.degrees(rack_yaw):+.1f}deg "
          f"stand=({float(stand_p[0]):+.3f},{float(stand_p[1]):+.3f}) "
          f"stand_yaw={math.degrees(stand_yaw):+.1f}deg "
          f"stow_x={float(scene._stow_x[0]) * 1000:+.1f}mm "
          f"slot_of_body={scene.slot_of[0].tolist()}", flush=True)
    assert abs(tray_mass - c.tray_mass) < 0.02, "tray mass authoring readback failed"
    assert all(abs(mv - c.bowl_mass) < 0.02 for mv in masses), "bowl mass readback failed"
    lx0 = float(scene.tray_local_x()[0])
    assert abs(lx0 - float(scene._stow_x[0])) < 0.015, \
        f"tray not at its stow depth after settle: {lx0:+.3f} vs {float(scene._stow_x[0]):+.3f}"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # target BY POSITION: the body nearest the FRONT row slot (largest x) — a rule any
    # solver could follow from a camera. Cross-check the scene's latched identity after.
    bows = torch.stack([b.data.root_pos_w[0] for b in scene.bowls], dim=0) - o
    xs = torch.tensor(c.slot_xs, device=device)
    slot_by_pos = [int((xs - bows[i, 0]).abs().argmin()) for i in range(3)]
    tgt = slot_by_pos.index(0)
    print(f"[solve] slot-by-position readback: {slot_by_pos} -> front body {tgt}",
          flush=True)
    assert tgt == int(scene.target[0]), \
        f"position readback named body {tgt}, scene latched {int(scene.target[0])}"

    # ---------------- phase 1: slide the tray out of the rack (applied force) --------------
    # Bang-bang force along the rack's +x (channel direction), velocity-capped;
    # the paired torque emulates the force acting 80 mm BELOW the CoM (a low push
    # point on the on-edge board), so escalated forces cannot pitch it over.
    # Friction estimate: mu ~0.12 -> ~0.18 N; start well above, escalate on stall.
    ex = torch.tensor([math.cos(rack_yaw), math.sin(rack_yaw), 0.0], device=device)
    r_low = torch.tensor([0.0, 0.0, -0.080], device=device)
    fmag, fmax = 0.9, 4.0
    v_cap = 0.12
    best = -math.inf
    stall = 0
    forces = torch.zeros(n, 1, 3, device=device)
    torques = torch.zeros(n, 1, 3, device=device)
    for k in range(2000):
        lx = float(scene.tray_local_x()[0])
        if lx > 0.105:
            break
        if lx > best + 0.0005:
            best = lx
            stall = 0
        else:
            stall += 1
        if stall >= 120:
            fmag = min(fmag * 1.5, fmax)
            stall = 0
            print(f"[solve] extraction stall at {lx * 1000:+.1f}mm -> F={fmag:.2f}N",
                  flush=True)
        v_along = float((scene.tray.data.root_lin_vel_w[0] * ex).sum())
        if v_along < v_cap:
            f_w = fmag * ex
            t_w = torch.linalg.cross(r_low, f_w)
            q = scene.tray.data.root_quat_w
            forces[:, 0, :] = quat_apply_inverse(q, f_w.expand(n, 3))
            torques[:, 0, :] = quat_apply_inverse(q, t_w.expand(n, 3))
        else:
            forces[:, 0, :] = 0.0
            torques[:, 0, :] = 0.0
        scene.tray.set_external_force_and_torque(forces, torques)
        env.step(no_action)
        if k % 150 == 0:
            report(f"slide k={k}")
    zero_wrench()
    step(60)  # coast + settle, hands-off
    report("extracted")
    assert bool(scene.tray_out()[0]), \
        f"tray did not clear the rack: local_x={float(scene.tray_local_x()[0]) * 1000:+.1f}mm"
    s1 = print_score("P1 tray slid out of the rack")
    assert s1 >= c.w_out - 1e-4, f"out-of-rack stage credit missing, got {s1}"
    assert s1 >= s0 - 1e-6, "score decreased"

    # ---------------- phase 2: transport + gravity-seat the tray in the well ---------------
    # Entry pose from LIVE stand readback: centred on the well axis, flat, yaw =
    # stand yaw (the square tray needs alignment mod 90 deg), 30 mm above the seat.
    # The drop through the 10 mm-per-side slack does the seating.
    seat_rel_z = c.base_h + c.tray_thick / 2
    entry = scene.stand.data.root_pos_w.clone()
    entry[:, 2] += seat_rel_z + 0.030
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = entry
    st[:, 3] = math.cos(stand_yaw / 2)
    st[:, 6] = math.sin(stand_yaw / 2)
    scene.tray.write_root_state_to_sim(st, all_ids)
    for _ in range(240):
        env.step(no_action)
        if bool(scene.tray_seated()[0]) and int(scene._still_tray[0]) >= c.still_steps:
            break
    step(30)  # extra settle, hands-off
    report("seated")
    assert bool(scene.tray_seated()[0]), "tray did not seat flat on the well floor"
    s2 = print_score("P2 tray seated in the well")
    assert s2 >= c.w_seated - 1e-4, f"seated stage credit missing, got {s2}"
    assert s2 >= s1 - 1e-6, "score decreased"

    # ---------------- phase 3: serve the front bowl on the tray ----------------------------
    # Teleport the target bowl (bottom-center origin) 25 mm above the seated tray's
    # center, upright; gravity does the placement.
    drop = scene.tray.data.root_pos_w.clone()
    drop[:, 2] += c.tray_thick / 2 + 0.025
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = drop
    st[:, 3] = 1.0
    scene.bowls[tgt].write_root_state_to_sim(st, all_ids)
    for _ in range(300):
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    step(30)  # extra settle, hands-off
    report("served")
    s3 = print_score("P3 front bowl on the tray")
    assert s3 >= s2 - 1e-6, "score decreased"

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after serving)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_end = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_end >= s3 - 1e-6
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
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc!r})", flush=True)
        os._exit(1)
