"""Teleport solution for FlapChutePantryScene (sim_gen task
`living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray_i33`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the CREAM carton from
   its ground start slot to the free-space point 12 mm above its rest height on the
   APRON platform, square to the doorway, zero velocity. The path is free air (the
   apron top is open above), so the transport bypasses no contact interaction, and
   the write satisfies no rubric clause by itself: the carton is airborne over the
   apron.
2. STAGING (gravity + contact): the carton FALLS the last 12 mm and settles on the
   apron. The `staged` credit (at rest on the apron) is produced by contact, never
   written.
3. PUSH THROUGH THE FLAP (applied force + contact): a horizontal force at the
   carton's CoM, directed pantry-local -x (into the doorway), slides it along the
   slick apron into the ORANGE FLAP. The flap yields inward and rides over the
   carton (`breach` credit: flap deflected > 20 deg with the carton in the doorway
   — pure contact). This is the exact push a fingertip on the carton's back face
   would perform; the fingertip stand-in never needs to pass more than ~15 mm
   beyond the front wall's outer face. The force is released the moment the
   carton's CoM crosses the sill inner edge.
   FORCE-FRAME GUARD: some pods rotate an applied "global" wrench by the body's
   rotation since reset. The push PROBES the frame convention at runtime: every 45
   steps it checks progress along pantry -x and toggles the `encode_force` mode if
   the carton moved the wrong way (see scene.encode_force).
4. DROP + CHUTE + SELF-CLOSING FLAP (gravity + contact, hands-off): past the sill
   the carton tips, drops onto the slick internal chute, slides down and to the
   back wall — clear of the flap's swing arc — and the flap falls shut behind it
   under its own weight. Every success clause (inside + deep, flap closed, settled)
   is produced by ballistics and contact.
5. ORDER / IDENTITY: only the CREAM carton is touched. The BROWN decoy is never
   moved; success requires it to still be outside.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray_i33.solve --headless [--seed N]
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
    from .scene import encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.flap_chute_pantry")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def cream_loc() -> torch.Tensor:
        return scene._pantry_local(scene.cream.data.root_pos_w)[0]

    def report(tag: str) -> None:
        cl = cream_loc()
        bl = scene._pantry_local(scene.brown.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | cream_loc=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) brown_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) flap={float(scene.flap_deg()[0]):+.1f}deg "
              f"staged={bool(scene._staged[0])} breach={bool(scene._breach[0])} "
              f"inside={bool(scene._inside[0])} deep={bool(scene.cream_deep()[0])} "
              f"closed={bool(scene.flap_closed()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # flap settles hanging closed, cartons settle on the ground
    pp = (scene.pantry.data.root_pos_w - scene.env_origins)[0]
    pq = scene.pantry.data.root_quat_w[0]
    pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
    slot_side = "+y" if float(scene.cream_slot[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pantry=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) yaw={pyaw:+.1f}deg "
          f"flap={float(scene.flap_deg()[0]):+.1f}deg cream_slot={slot_side}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.flap_closed()[0]), \
        f"flap must hang closed after settle, got {float(scene.flap_deg()[0]):+.1f} deg"
    assert not bool(scene.cream_inside()[0]), "cream carton must start outside"
    assert not bool(scene.brown_inside()[0]), "brown carton must start outside"
    assert not bool(scene.on_apron(scene.cream.data.root_pos_w)[0]), \
        "cream carton must start on the ground, not the apron"
    s0 = print_score("P0 reset+settle (flap closed, cartons on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: cream carton -> apron (transport, gravity seats it) ---------
    pan_q = scene.pantry.data.root_quat_w
    pan_p = scene.pantry.data.root_pos_w
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = 0.280  # apron, ~85 mm short of the doorway
    loc[:, 2] = c.sill_z + c.carton_size[2] / 2 + 0.012  # 12 mm free fall
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pan_p + quat_apply(pan_q, loc)
    st[:, 3:7] = pan_q  # square to the doorway
    scene.cream.write_root_state_to_sim(st, all_ids)
    step(150)  # fall + settle, hands-off
    report("cream->apron")
    assert bool(scene.on_apron(scene.cream.data.root_pos_w)[0]), \
        "cream carton must rest on the apron"
    assert bool(scene._staged[0]), "staged latch must be set"
    assert not bool(scene.success()[0]), "cannot be success while outside"
    s1 = print_score("P1 cream carton staged on the apron")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_staged - 1e-6, \
        f"P1 score {s1} (expect staged={c.w_staged})"

    # ---------------- phase 2: push through the flap; gravity + chute do the rest ----------
    fdir = quat_apply(pan_q, torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3))
    state = {"mode": 0, "q_ref": scene.cream.data.root_quat_w.clone()}

    def push_until(release_x: float, release_z: float, fmag0: float,
                   max_steps: int) -> bool:
        """Bang-bang CoM push along pantry -x until the carton's CoM passes
        `release_x` or its height dips under `release_z` (tip started). Probes the
        force-frame convention every 45 steps; escalates on stall. Returns True on
        release, False on step budget exhausted. Always clears the force."""
        fmag = fmag0
        probe_x = float(cream_loc()[0])
        released = False
        for i in range(max_steps):
            cl = cream_loc()
            if float(cl[0]) < release_x or float(cl[2]) < release_z:
                released = True
                break
            v = float(scene.cream.data.root_lin_vel_w[0].norm())
            mag = fmag if v < 0.06 else (0.4 * fmag if v < 0.12 else 0.0)
            if mag > 0.0:
                f = encode_force(state["mode"], state["q_ref"],
                                 scene.cream.data.root_quat_w,
                                 mag * fdir).view(n, 1, 3)
                scene.cream.set_external_force_and_torque(f, zero_wrench,
                                                          env_ids=all_ids,
                                                          is_global=True)
            else:
                scene.cream.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                          env_ids=all_ids)
            env.step(no_action)
            if i % 45 == 44:
                x_now = float(cream_loc()[0])
                if x_now > probe_x + 0.004:
                    state["mode"] ^= 1  # frame drag: the push moved it the wrong way
                    state["q_ref"] = scene.cream.data.root_quat_w.clone()
                    print(f"[solve] push moved the carton AWAY (x {probe_x:+.3f} -> "
                          f"{x_now:+.3f}); force-frame mode -> {state['mode']}",
                          flush=True)
                elif x_now > probe_x - 0.002:
                    fmag = min(fmag + 0.4, 3.0)
                    print(f"[solve] push stalled at x_loc={x_now:+.3f} "
                          f"(flap={float(scene.flap_deg()[0]):+.1f}deg); "
                          f"force -> {fmag:.1f} N", flush=True)
                probe_x = x_now
        scene.cream.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)
        return released

    # Release deep enough that the flap's weight moment cannot hold the carton from
    # tipping: at CoM x 0.100 the gravity torque about the sill inner edge (0.130) is
    # ~3x what the 40 g flap can resist (measured: released at 0.123 it wedged at
    # 0.118 with the flap at 58 deg).
    dropped = push_until(0.100, 0.068, 0.6, 1500)
    if not dropped:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (carton never crossed the sill)", flush=True)
        os._exit(1)
    print(f"[solve] carton over the sill (x_loc={float(cream_loc()[0]):+.3f}, "
          f"z_loc={float(cream_loc()[2]):+.3f}); hands off — chute + flap take over",
          flush=True)

    # hands-off follow-through: tumble in, slide down the chute, flap falls shut.
    # Fallback: if the carton still straddles the doorway, push again, deeper.
    ok_p2 = False
    for j in range(900):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j in (240, 540) and float(cream_loc()[0]) > 0.05 \
                and float(cream_loc()[2]) > 0.066:
            print(f"[solve] carton straddling the doorway at "
                  f"x_loc={float(cream_loc()[0]):+.3f}; pushing deeper", flush=True)
            push_until(0.080, 0.062, 1.2, 300)
        if j % 180 == 179:
            report(f"follow-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("chute-drop")
    if not (ok_p2 or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the drop)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pushed through the flap; carton inside, flap closed")
    assert s2 >= s1 - 1e-6, "score decreased across the push"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
