"""Teleport solution for BayonetDockScene (sim_gen task
libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate_i296) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TARGET SELECTION (readback): the front canister is picked BY POSITION — the body
   whose x is nearest the front row slot — exactly what a camera would provide. The
   scene's latched target identity is asserted AFTERWARDS as a certificate cross-check.
2. TRANSPORT (teleport): two root-state writes carry the target across free space —
   to a hover point above the dock, then to the ALIGNED ENTRY POSE: centred on the
   dock axis, yaw matched to the dock's (randomized) heading so the two lugs face the
   two flange slots, bottom ~28 mm above the collar floor with the lugs still fully
   ABOVE the flange ring. The writes satisfy no rubric clause: the canister is
   airborne at both points.
3. INSERTION (gravity + contact): from the entry pose the canister FALLS, the lugs
   pass through the two open slot gaps of the catch flange, and the canister lands
   on the collar floor. inserted() is produced entirely by the landing dynamics.
4. TWIST (applied torque + contact): a bang-bang body-z torque (starting 2x the
   friction estimate, escalating x1.6 on a 90-step stall, angular speed capped)
   rotates the seated canister counterclockwise; the lugs travel UNDER the flange
   until the relative twist passes the 52-degree cut (or the lug presses the red
   stop peg). The torque is then ZEROED and the canister settles; success() judges
   the settled, hands-off state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage credit
is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success() first
turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_dock")().build(num_envs=args.num_envs, device=device)
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
        d_xy, dz = scene._rel_dock()
        tw = math.degrees(float(scene.twist()[0, t]))
        print(f"[solve] {tag:18s} | tgt={t} d_xy={float(d_xy[0, t]) * 1000:6.1f}mm "
              f"dz={float(dz[0, t]) * 1000:+6.1f}mm twist={tw:+7.2f}deg "
              f"ins={bool(scene.inserted()[0, t])} still={int(scene._still[0, t])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def bowl_write(i: int, pos_w: torch.Tensor, yaw: float = 0.0) -> None:
        """Teleport canister i (upright at `yaw`, zero velocity) — transport only."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.bowls[i].write_root_state_to_sim(st, all_ids)

    def zero_wrench(i: int) -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.bowls[i].set_external_force_and_torque(z, z)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # canisters settle at their slots
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    o = scene.env_origins[0]
    dock_p = scene.dock.data.root_pos_w[0] - o
    dock_yaw = float(scene._yaw_of(scene.dock.data.root_quat_w)[0])
    print(f"[solve] layout readback (seed {args.seed}): masses={masses} "
          f"dock=({float(dock_p[0]):+.3f},{float(dock_p[1]):+.3f}) "
          f"dock_yaw={math.degrees(dock_yaw):+.1f}deg "
          f"slot_of_body={scene.slot_of[0].tolist()}", flush=True)
    assert all(abs(mv - c.mass) < 0.02 for mv in masses), "mass authoring readback failed"
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

    # ---------------- phase 1: lift the front canister -------------------------------------
    hover = scene.dock.data.root_pos_w.clone()
    hover[:, 2] += 0.20
    bowl_write(tgt, hover, yaw=dock_yaw)
    step(2)  # latch s_lift (airborne, no rubric clause satisfied beyond the lift stage)
    report("hover")
    s1 = print_score("P1 target lifted")
    assert s1 >= c.w_lift - 1e-4, f"lift stage credit missing, got {s1}"
    assert s1 >= s0 - 1e-6, "score decreased"

    # ---------------- phase 2: aligned entry + gravity drop through the slots --------------
    # Entry pose from LIVE dock readback: centred on the axis, yaw = dock yaw (lugs face
    # the slot gaps), bottom 28 mm above the collar floor — the lugs start fully ABOVE
    # the flange ring (lug bottom at local 0.048+? no: 0.040+0.020=0.060 > flange top
    # 0.058) and must pass THROUGH the two open gaps on the way down.
    entry = scene.dock.data.root_pos_w.clone()
    entry[:, 2] += c.base_h + 0.028
    bowl_write(tgt, entry, yaw=dock_yaw)
    for _ in range(240):
        env.step(no_action)
        if bool(scene.inserted()[0, tgt]) and bool(scene.at_rest()[0, tgt]):
            break
    step(30)  # extra settle, hands-off
    report("dropped in")
    assert bool(scene.inserted()[0, tgt]), "canister did not seat inside the collar"
    tw0 = math.degrees(float(scene.twist()[0, tgt]))
    assert abs(tw0) < 15.0, f"landed twist should be near-aligned, got {tw0:.1f}deg"
    s2 = print_score("P2 inserted through the slots")
    assert s2 >= c.w_insert - 1e-4, f"insert stage credit missing, got {s2}"
    assert s2 >= s1 - 1e-6, "score decreased"

    # ---------------- phase 3: counterclockwise twist under the flange ---------------------
    # Bang-bang body-z torque: apply tau while below the speed cap, coast above it;
    # escalate tau x1.6 after a 90-step stall (friction estimate ~0.015 N*m); cut at
    # 52 degrees (hard stop at ~56 catches any coast) or when pressed on the stop.
    tau, tau_max = 0.03, 0.12
    w_cap = 1.2
    best = -math.inf
    stall = 0
    cut_deg = 52.0
    forces = torch.zeros(n, 1, 3, device=device)
    torques = torch.zeros(n, 1, 3, device=device)
    for k in range(1500):
        tw = float(scene.twist()[0, tgt])
        deg = math.degrees(tw)
        if deg >= cut_deg:
            break
        if deg > best + 0.5:
            best = deg
            stall = 0
        else:
            stall += 1
        if stall >= 90:
            if deg >= 46.0:
                break  # pressed against the stop peg past the success gate
            tau = min(tau * 1.6, tau_max)
            stall = 0
            print(f"[solve] twist stall at {deg:+.1f}deg -> tau={tau:.3f}", flush=True)
        wz = float(scene.bowls[tgt].data.root_ang_vel_w[0, 2])
        torques[:, 0, 2] = tau if wz < w_cap else 0.0
        scene.bowls[tgt].set_external_force_and_torque(forces, torques)
        env.step(no_action)
        if k % 120 == 0:
            report(f"twist k={k}")
    zero_wrench(tgt)
    for _ in range(360):
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    step(30)  # extra settle, hands-off
    report("twisted+settled")
    s3 = print_score("P3 locked under the flange")
    assert s3 >= s2 - 1e-6, "score decreased"

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after twist)", flush=True)
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
