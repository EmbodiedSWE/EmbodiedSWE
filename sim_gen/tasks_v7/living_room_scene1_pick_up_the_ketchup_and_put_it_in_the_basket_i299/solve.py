"""Teleport solution for LeakyBasketSealScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i299`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. SEAL THE LEAK (transport teleport + contact dynamics): one pose write STAGES the
   WIDE lid level in free air on the stand's axis inside the cavity, ~8 cm above the
   seat, nothing touching (the stand's xy/yaw are randomized, so the staging point is
   computed in the stand's CURRENT readback frame). The actual sealing — the drop,
   the funnel SELF-CENTRING the disc, and the disc SEATING flat over the throat — is
   pure contact physics. The lid is never written into a seated pose: if the funnel
   failed to carry it (as it provably fails to carry the narrow decoy) it would fall
   through the throat onto the floor below.
2. LOAD THE KETCHUP (transport teleport + contact dynamics): one pose write stages
   the red bottle upright just above the rim, slightly off the stand axis so it does
   not balance on the grip knob, zero velocity. The drop, the impact ON the seated
   lid, the topple/lean against the cavity wall and the ring-down to a settled state
   resting ON the seal are contact physics — the seal is load-bearing: without it
   the same drop discharges the bottle onto the floor under the stand (smoke #6).

From the second release to the verdict nothing touches any body. success() demands
the live physical state: wide lid seated over the throat, ketchup resting contained
above it, brown bottle and narrow lid out, everything still — a state only the
funnel-carried seal can sustain.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.leaky_basket_seal")().build(num_envs=args.num_envs,
                                                       device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def vels(tag: str) -> None:
        pv = float(scene.plug.data.root_lin_vel_w[0].norm())
        pw = float(scene.plug.data.root_ang_vel_w[0].norm())
        kv = float(scene.ketchup.data.root_lin_vel_w[0].norm())
        kw = float(scene.ketchup.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | plug v={pv:.4f} w={pw:.4f} "
              f"ketchup v={kv:.4f} w={kw:.4f}", flush=True)

    def report(tag: str) -> None:
        p_loc = scene._stand_local(scene.plug.data.root_pos_w)[0]
        k_loc = scene._stand_local(scene.ketchup.data.root_pos_w)[0]
        vels(tag)
        print(f"[solve] {tag:12s} | plug_loc=({float(p_loc[0]):+.3f},"
              f"{float(p_loc[1]):+.3f},{float(p_loc[2]):+.3f}) "
              f"k_loc_z={float(k_loc[2]):+.3f} "
              f"sealed={bool(scene.sealed()[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"decoy_in={bool(scene.decoy_in_cavity()[0])} "
              f"sealed_ever={bool(scene._sealed_ever[0])} "
              f"in_ever={bool(scene._in_ever[0])} "
              f"ret_ever={bool(scene._ret_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def stage_on_axis(body, local_z: float, quat=None, xy_off=(0.0, 0.0)) -> None:
        """One pose write: body at stand-local (xy_off, local_z), zero velocity —
        transport only; every staged pose is free air (asserted by the caller)."""
        s_pos = scene.stand.data.root_pos_w
        s_quat = scene.stand.data.root_quat_w
        loc = torch.tensor([xy_off[0], xy_off[1], local_z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
        st[:, 3:7] = s_quat if quat is None else quat
        body.write_root_state_to_sim(st)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sq = scene.stand.data.root_quat_w[0]
    s_yaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    s0p = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    p0 = (scene.plug.data.root_pos_w - scene.env_origins)[0]
    d0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    k0 = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    q0 = (scene.bbq.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(s0p[0]):+.3f},{float(s0p[1]):+.3f}) yaw={s_yaw:+.1f}deg "
          f"plug=({float(p0[0]):+.3f},{float(p0[1]):+.3f}) "
          f"decoy=({float(d0[0]):+.3f},{float(d0[1]):+.3f}) "
          f"ketchup=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"bbq=({float(q0[0]):+.3f},{float(q0[1]):+.3f}) "
          f"plug_left={bool(scene._plug_left[0])}", flush=True)
    report("reset")
    assert not bool(scene.sealed()[0]), "plug spawned sealed?!"
    assert not bool(scene.contained(scene.ketchup)[0]), "ketchup spawned contained?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: SEAL THE LEAK (stage on the axis; seat is physics) -----------
    # One pose write: wide lid level, on the stand axis, ~8 cm above the seat — inside
    # the cavity but touching nothing. Hands off: the funnel centres it and it seats
    # over the throat. A lid the funnel could not carry would end up on the floor.
    stage_on_axis(scene.plug, c.z_throat + 0.105)
    p_loc = scene._stand_local(scene.plug.data.root_pos_w)[0]
    assert not bool(scene.sealed()[0]), "staging pose already counts as sealed?!"
    assert float(p_loc[2]) > c.z_throat + c.seat_z_hi + 0.03, "staged too low"

    seated = False
    quiet = 0
    for i in range(720):
        env.step(no_action)
        ok = bool(scene.sealed()[0]) and bool(scene._still_body(scene.plug)[0])
        quiet = quiet + 1 if ok else 0
        if quiet >= 30:
            seated = True
            break
    report("sealed")
    assert seated, "wide lid did not seat over the throat / did not settle"
    assert bool(scene._sealed_ever[0]), "sealed latch did not set"
    s1 = print_score("P1 wide lid dropped, funnel-centred, seated over the throat")
    assert s1 >= s0 - 1e-6, "score decreased across the seal"
    assert s1 >= 0.29, f"seal credit missing, score {s1}"

    # ---------------- phase 2: LOAD THE KETCHUP (stage above the rim; drop) -----------------
    # One pose write in the stand's CURRENT frame: bottle upright, bottom ~8 mm above
    # the rim, a small lateral offset so it does not balance on the grip knob. The
    # landing on the SEATED lid and the lean-and-settle are contact physics.
    stage_on_axis(scene.ketchup, c.z_rim + 0.008 + c.body_h / 2,
                  quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4),
                  xy_off=(0.014, 0.010))
    assert not bool(scene.contained(scene.ketchup)[0]), \
        "staging pose is already contained (teleport must stay above the rim)"

    quiet = 0
    done = False
    for i in range(1200):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            done = True
            break
        if (i + 1) % 240 == 0:
            vels(f"P2 t+{(i + 1) / 120.0:.0f}s")
    report("loaded")
    assert bool(scene.sealed()[0]), "the drop knocked the lid off its seat"
    assert bool(scene.contained(scene.ketchup)[0]), \
        "ketchup did not come to rest contained on the seal"
    assert done, "loaded basket did not settle to success"
    s2 = print_score("P2 ketchup dropped onto the seated lid, settled contained")
    assert s2 >= s1 - 1e-6, "score decreased across the load"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after load)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
