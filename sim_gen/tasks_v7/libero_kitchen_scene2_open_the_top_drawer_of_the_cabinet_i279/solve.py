"""Teleport solution for GateHopperScene (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i279`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport = transport only): one rigid root-state write per body carries
   the hopper AND its contents — gate plate and ball, relative poses preserved, gate
   still CLOSED — from the spawn to a level HOVER above the blue bin (base plane well
   above the rim, nose backed off so the floor opening will open over the bin mouth).
   The write satisfies nothing: the ball is still sealed inside, the gate is still
   flush, `success()` is False; only the `lift` latch arms (hopper aloft with the
   ball inside — exactly what a carry earns).
2. GATE ACTUATION (gravity through the mechanism): the hover hold pitches the hopper
   NOSE-DOWN along its own axis in a slow kinematic ramp (per-step root-state writes
   of the HOPPER ONLY — the exact hold of a hand on the roof handle). Gravity does
   the rest: the slick captive plate slides out through its slot until the rear
   stop-tab catches, the floor opening vacates the ball's footprint, and the ball
   FALLS out of the bottom into the bin. The plate and ball are never written during
   this phase — the gate opens or the run fails.
3. DELIVERY (ballistic + contact): the ball's drop into the bin is pure free fall
   and restitution-0 contact with the bin floor/walls. The landing point is aimed by
   the hover pose only; nothing repositions the ball.
4. PARK (transport + release): the hopper (and its extended plate, rigidly) is
   teleported to open ground far beyond `clear_min`, nose pointing AWAY from the
   bin, 2 mm hover, then RELEASED — the final rest pose is produced by gravity and
   ground contact, and the last 3+ seconds are fully hands-off.
5. ORDER (physically inherent): tilting before lifting dumps the ball onto the
   ground under the hopper (unrecoverable by the gate route); the hover must be over
   the bin BEFORE the tilt because the gate cannot be re-loaded once the ball is out.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i279.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, _qinv, _qmul, _qy, _qz
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, _qmul, _qy, _qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- aim parameters (tuned on the forge from printed landing readbacks) ------------------------
HOVER_BACKOFF = 0.030   # hover: hopper centre this far behind the bin centre, against the nose
                        # (measured exit point is ~22 mm nose-ward of the hover centre, so this
                        # lands the ball ~8 mm short of the bin centre — dead in the mouth)
HOVER_Z = 0.135         # hover: hopper base-plane height (m) — plate tip clears the rim when tilted
TILT_DEG = 30.0         # nose-down pitch that opens the gate (slide angle is ~6.3 deg)
TILT_RAMP_STEPS = 300   # 0 -> TILT_DEG over 2.5 s (gentle: the plate+ball ride the rotation)
TILT_HOLD_STEPS = 480   # max extra hold at full tilt waiting for the drop
LEVEL_RAMP_STEPS = 150  # TILT_DEG -> 0 after the drop
PARK_DIST = 0.30        # park: hopper centre this far from the bin centre (> clear_min = 0.18)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gate_hopper")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ext = float(scene.gate_ext()[0])
        bl = scene.ball_local()[0]
        print(f"[solve] {tag:12s} | ext={ext * 1000:6.1f}mm "
              f"ball_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"in_hopper={bool(scene.ball_in_hopper()[0])} "
              f"in_bin={bool(scene.ball_in_bin()[0])} "
              f"parked={bool(scene.hopper_parked()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def rigid_teleport(tp: torch.Tensor, tq: torch.Tensor, bodies) -> None:
        """One root-state write per body: carry `bodies` rigidly so that the HOPPER
        frame lands at (tp, tq) with every relative pose preserved. Zero velocity."""
        hp = scene.hopper.data.root_pos_w.clone()
        hq = scene.hopper.data.root_quat_w.clone()
        for b in bodies:
            loc = _qapply(_qinv(hq), b.data.root_pos_w - hp)
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = tp + _qapply(tq, loc)
            st[:, 3:7] = _qmul(tq, _qmul(_qinv(hq), b.data.root_quat_w))
            b.write_root_state_to_sim(st, all_ids)

    def hold_hopper(tp: torch.Tensor, tq: torch.Tensor, steps: int) -> None:
        """Kinematic hold: rewrite the HOPPER root state (zero vel) every step; the
        plate and ball stay free and respond through contacts only."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tp
        st[:, 3:7] = tq
        for _ in range(steps):
            scene.hopper.write_root_state_to_sim(st, all_ids)
            env.step(no_action)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(150)
    origins = scene.env_origins
    hp0 = (scene.hopper.data.root_pos_w - origins).clone()
    qh0 = scene.hopper.data.root_quat_w[0]
    yaw0 = float(torch.rad2deg(2.0 * torch.atan2(qh0[3], qh0[0])))
    bin_p = (scene.bin.data.root_pos_w - origins).clone()
    bl0 = scene.ball_local()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"hopper=({float(hp0[0, 0]):+.3f},{float(hp0[0, 1]):+.3f}) yaw={yaw0:+.1f}deg "
          f"bin=({float(bin_p[0, 0]):+.3f},{float(bin_p[0, 1]):+.3f}) "
          f"ball_loc=({float(bl0[0]):+.3f},{float(bl0[1]):+.3f},{float(bl0[2]):+.3f})",
          flush=True)
    mh = float(scene.hopper.root_physx_view.get_masses()[0, 0])
    mp = float(scene.plate.root_physx_view.get_masses()[0, 0])
    mb = float(scene.ball.root_physx_view.get_masses()[0, 0])
    print(f"[solve] mass readback: hopper={mh:.3f} plate={mp:.3f} ball={mb:.3f}", flush=True)
    assert abs(mh - c.hopper_mass) < 0.02, "hopper MassAPI must stick (custom spawner)"
    assert abs(mp - c.plate_mass) < 0.005, "plate MassAPI must stick (custom spawner)"
    assert abs(mb - c.ball_mass) < 0.005, "ball mass must stick"
    report("reset")
    assert bool(scene.ball_in_hopper().all()), "ball must start sealed inside the hopper"
    assert float(scene.gate_ext().abs().max()) < 0.005, "gate must start flush (closed)"
    s0 = print_score("P0 reset+settle (gate flush, ball sealed, hopper far from the bin)")
    assert not bool(scene.success().any()), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: carry the closed hopper to a hover over the bin -------------
    # Direction u: from the bin centre toward the hopper's spawn (guaranteed open
    # ground). Hover: hopper centre backed off from the bin centre along +u, NOSE
    # (local +x, the direction the gate slides and the ball exits) pointing -u, i.e.
    # TOWARD the bin — when tilted nose-down the ball leaves the cavity at its front
    # (~+40 mm along the nose from the centre) and falls near the bin centre.
    u = hp0[:, :2] - bin_p[:, :2]
    u = u / u.norm(dim=-1, keepdim=True)
    q_yaw = _qz(torch.atan2(-u[:, 1], -u[:, 0]))  # nose points -u (toward the bin)
    hover_p = torch.zeros(n, 3, device=device)
    hover_p[:, :2] = bin_p[:, :2] + u * HOVER_BACKOFF
    hover_p[:, 2] = HOVER_Z
    hover_p += origins
    rigid_teleport(hover_p, q_yaw, [scene.hopper, scene.plate, scene.ball])
    hold_hopper(hover_p, q_yaw, 60)  # half a second level: contents settle in the carry
    report("hover")
    assert bool(scene.ball_in_hopper().all()), "carry must not spill the ball"
    assert float(scene.gate_ext().abs().max()) < 0.01, "gate must stay shut while level"
    s1 = print_score("P1 carried aloft over the bin (gate still shut)")
    assert s1 >= 0.14, "lift latch should have armed"

    # ---------------- phase 2: tilt nose-down; gravity opens the gate; ball drops ----------
    # Kinematic hold of the HOPPER ONLY: position fixed at the hover, pitch ramped
    # about its own (yawed) y axis. The plate slides out by gravity; the ball rolls
    # to the front sill and falls the moment the retreating plate uncovers it.
    tilt = math.radians(TILT_DEG)
    dropped = torch.zeros(n, dtype=torch.bool, device=device)
    st = torch.zeros(n, 13, device=device)
    for i in range(TILT_RAMP_STEPS + TILT_HOLD_STEPS):
        th = tilt * min(1.0, (i + 1) / TILT_RAMP_STEPS)
        tq = _qmul(q_yaw, _qy(torch.full((n,), th, device=device)))
        st[:, 0:3] = hover_p
        st[:, 3:7] = tq
        scene.hopper.write_root_state_to_sim(st, all_ids)
        env.step(no_action)
        dropped |= scene.ball_in_bin()
        if i % 60 == 0:
            ext = float(scene.gate_ext()[0])
            bw = scene.ball.data.root_pos_w[0] - origins[0]
            print(f"[solve] tilt {math.degrees(th):5.1f}deg ext={ext * 1000:6.1f}mm "
                  f"ball_w=({float(bw[0]):+.3f},{float(bw[1]):+.3f},{float(bw[2]):+.3f}) "
                  f"in_bin={bool(scene.ball_in_bin()[0])}", flush=True)
        if bool(dropped.all()):
            break
    # landing readback (the aim-tuning signal)
    bw = scene.ball.data.root_pos_w - origins
    off = bw[:, :2] - bin_p[:, :2]
    print(f"[solve] landing: ball_w=({float(bw[0, 0]):+.3f},{float(bw[0, 1]):+.3f},"
          f"{float(bw[0, 2]):+.3f}) offset_from_bin=({float(off[0, 0]) * 1000:+.0f},"
          f"{float(off[0, 1]) * 1000:+.0f})mm in_bin={bool(scene.ball_in_bin()[0])}",
          flush=True)
    assert bool(dropped.all()), "the ball must have dropped into the bin"
    # keep holding at full tilt while the ball comes to rest in the bin
    hold_hopper(hover_p, _qmul(q_yaw, _qy(torch.full((n,), tilt, device=device))), 60)
    # level back out (slow kinematic ramp, still holding position)
    for i in range(LEVEL_RAMP_STEPS):
        th = tilt * (1.0 - (i + 1) / LEVEL_RAMP_STEPS)
        tq = _qmul(q_yaw, _qy(torch.full((n,), th, device=device)))
        st[:, 0:3] = hover_p
        st[:, 3:7] = tq
        scene.hopper.write_root_state_to_sim(st, all_ids)
        env.step(no_action)
    report("delivered")
    assert bool(scene.ball_in_bin().all()), "ball must rest inside the bin"
    s2 = print_score("P2 gate gravity-opened, ball delivered into the bin")
    assert s2 >= 0.69, "gate + drop latches should have armed"

    # ---------------- phase 3: park the hopper on open ground, release ---------------------
    # Park along +u, nose flipped to point AWAY from the bin so the extended plate
    # overhangs open ground; 2 mm hover, then hands off — gravity seats it.
    q_park = _qz(torch.atan2(u[:, 1], u[:, 0]))
    park_p = torch.zeros(n, 3, device=device)
    park_p[:, :2] = bin_p[:, :2] + u * PARK_DIST
    park_p[:, 2] = 0.002
    park_p += origins
    rigid_teleport(park_p, q_park, [scene.hopper, scene.plate])  # ball stays in the bin
    hold_hopper(park_p, q_park, 10)  # kill residuals, then HANDS OFF
    step(300)  # release: gravity seats the hopper; everything settles
    report("parked")
    assert bool(scene.hopper_parked().all()), "hopper must be parked upright, clear of the bin"
    assert bool(scene.success().all()), "success must hold after the park settle"
    s3 = print_score("P3 hopper parked; full success")
    assert s3 >= 0.999, "success must score 1.0"

    # ---------------- phase 4: hands-off persistence ----------------------------------------
    step(400)  # 3.33 simulated seconds, no writes of any kind
    report("persist")
    if bool(scene.success().all()) and float(scene.score().min()) >= 0.999:
        print_score("P4 hands-off persistence (>=3.3 s)")
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        rc = 0
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
        rc = 1
    threading.Timer(10.0, lambda: os._exit(rc)).start()
    app.close()
    os._exit(rc)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(4)
