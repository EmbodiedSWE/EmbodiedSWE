"""Teleport solution for PileDriverScene (sim_gen task `libero_pick_milk_i51`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — exactly what a pick-and-carry delivers:
the slug is teleported to a hover pose (zero velocity) and RELEASED; every joule
that moves a post is delivered by GRAVITY through a real collision:

  drop loop — hover the slug ~14 cm above the RED post head (yaw-aligned with the
              rig), release, and let it FALL. The impact's force spike breaks the
              friction clamp's ~200 N static grip for a few milliseconds and the
              post sinks ~4-10 mm; the clamp re-latches the new depth as the
              impulse decays (verified: proud height only moves during impacts,
              never between them). Re-pick the slug from wherever it bounced to
              (another pure transport) and repeat. Near flush the hover drops to
              ~8 cm for gentler finishing blows; the joint's hard stop at the
              flush band's lower edge plus the slug face out-gauging the deck
              aperture make overdrive physically impossible.
  park      — carry the slug back to the yellow holster (teleport to a hover just
              above the tray, zero velocity), drop it in, hands off.

NO external forces are applied to anything, ever — the posts are moved by
collision impulses alone, which is the point of the task.

Prints `SIM_GEN_SCORE <score>` at each phase boundary and after every blow
(non-decreasing by construction: the rubric is latched), then holds HANDS-OFF for
>= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import HOL_T, L_POST, SLUG_H, _qapply, _qz  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import HOL_T, L_POST, SLUG_H, _qapply, _qz  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DROP_H = 0.14  # slug bottom above the post head at release (full-power blow)
DROP_H_FINE = 0.08  # ... within FINE_BAND of flush (finishing blows)
FINE_BAND = 0.030  # switch to finishing blows once proud < this
MAX_DROPS = 40


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pile_driver")().build(num_envs=args.num_envs, device=device)
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

    def proud_r() -> float:
        return float(scene.proud(scene.post_red)[0])

    def proud_w() -> float:
        return float(scene.proud(scene.post_white)[0])

    def report(tag: str) -> None:
        sp = scene.slug.data.root_pos_w[0] - scene.env_origins[0]
        print(f"[solve] {tag:12s} | proud_red={proud_r() * 1000:+6.1f}mm "
              f"proud_white={proud_w() * 1000:+6.1f}mm "
              f"slug=({float(sp[0]):+.3f},{float(sp[1]):+.3f},{float(sp[2]):+.3f}) "
              f"flush={bool(scene.red_flush()[0])} white_ok={bool(scene.white_ok()[0])} "
              f"parked={bool(scene.slug_parked()[0])} "
              f"settled={bool(scene.all_settled()[0])}"
              f"(slug{int(scene.settled(scene.slug)[0])}"
              f"red{int(scene.settled(scene.post_red)[0])}"
              f"wht{int(scene.settled(scene.post_white)[0])}"
              f"rig{int(scene.settled(scene.rig)[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(max_steps: int = 720, min_steps: int = 60) -> None:
        done = 0
        for _ in range(max_steps // 20):
            step(20)
            done += 20
            if done >= min_steps and bool(scene.settled(scene.slug)[0]) \
                    and bool(scene.settled(scene.post_red)[0]):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(480, min_steps=120)
    rig_p = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    hol_p = (scene.holster.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.rig.data.root_quat_w[0, 3]),
                           float(scene.rig.data.root_quat_w[0, 0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rig_p[0]):+.3f},{float(rig_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg "
          f"proud_red={proud_r() * 1000:.1f}mm proud_white={proud_w() * 1000:.1f}mm "
          f"holster=({float(hol_p[0]):+.3f},{float(hol_p[1]):+.3f})", flush=True)
    report("reset")
    assert proud_r() > c.flush_hi + 0.030, "red post must start far above flush"
    assert bool(scene.white_ok()[0]), "white post must rest at its start height"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (posts held by their clamps, slug holstered)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — carry the slug over the red post -----------
    def teleport_hover(drop_h: float) -> None:
        """Place the slug at rest above the red post's head, yaw-aligned with the
        rig so its face gauges the deck aperture squarely. Zero velocity, hands off."""
        head = scene.post_red.data.root_pos_w.clone()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = head[:, 0:2]
        st[:, 2] = head[:, 2] + L_POST / 2 + drop_h + SLUG_H / 2
        st[:, 3:7] = scene.rig.data.root_quat_w
        scene.slug.write_root_state_to_sim(st, all_ids)

    teleport_hover(DROP_H)
    step(2)  # let post_step latch ever_near before judging
    report("carry")
    s1 = print_score("P1 slug carried over the rig (teleport transport, zero velocity)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"
    assert s1 >= 0.07, f"near-latch credit missing, got {s1}"

    # ---------------- phase 2: gravity-impact drop loop ------------------------------------
    s_prev = s1
    p_prev = proud_r()
    stalled = 0
    for blow in range(1, MAX_DROPS + 1):
        wait_settled(720, min_steps=80)  # free fall ~0.17 s, then impact + ring-down
        p_now = proud_r()
        s_now = print_score(f"P2 blow {blow} (gravity impact; hands off since release)")
        print(f"[solve] blow {blow:2d}: proud_red {p_prev * 1000:+6.1f} -> "
              f"{p_now * 1000:+6.1f}mm (advance {(p_prev - p_now) * 1000:5.1f}mm) "
              f"white={proud_w() * 1000:+6.1f}mm", flush=True)
        assert s_now >= s_prev - 1e-6, "score decreased across a blow"
        assert bool(scene.white_ok()[0]), \
            f"white post disturbed at blow {blow} — task lost (irreversible)"
        s_prev = s_now
        if bool(scene.red_flush()[0]):
            break
        stalled = stalled + 1 if p_prev - p_now < 0.0015 else 0
        assert stalled < 6, f"no advance in {stalled} consecutive blows — wedged"
        p_prev = p_now
        drop_h = DROP_H_FINE if p_now < FINE_BAND else DROP_H
        teleport_hover(drop_h)  # re-pick from wherever it bounced: transport only
    report("driven")
    assert bool(scene.red_flush()[0]), \
        f"red post not flush after {MAX_DROPS} blows (proud {proud_r() * 1000:.1f}mm)"
    assert bool(scene.white_ok()[0]), "white post disturbed"
    s2 = print_score("P2 red post flush with the deck (impacts only; flush latched)")
    assert s2 >= 0.70, f"flush stage credit missing, got {s2}"

    # ---------------- phase 3: park the slug back in the holster ---------------------------
    def teleport_park() -> None:
        """Hover the slug just above the holster tray, yaw-aligned with it, zero
        velocity; gravity stands it on the plate."""
        hol = scene.holster.data.root_pos_w.clone()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = hol[:, 0:2]
        st[:, 2] = hol[:, 2] + HOL_T + SLUG_H / 2 + 0.015
        st[:, 3:7] = scene.holster.data.root_quat_w
        scene.slug.write_root_state_to_sim(st, all_ids)

    for attempt in range(3):
        teleport_park()
        wait_settled(480, min_steps=60)
        if bool(scene.slug_parked()[0]):
            break
        print(f"[solve] park attempt {attempt}: not seated — retrying", flush=True)
    report("parked")
    assert bool(scene.slug_parked()[0]), "slug failed to seat in the holster"
    s3 = print_score("P3 slug parked back in the holster (transport + gravity)")
    assert s3 >= s2 - 1e-6, "score decreased across the park"

    # ---------------- phase 4: success + persistence ---------------------------------------
    for _ in range(12):  # up to 1 s extra hands-off settling
        if bool(scene.success()[0]):
            break
        step(20)
    report("settled")
    s4 = print_score("P4 goal state: red flush, white untouched, slug holstered, still")
    assert s4 >= s3 - 1e-6, "score decreased across final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after drive+park)", flush=True)
        os._exit(1)

    hold, flickers = True, 0
    for i in range(780):  # 780 steps = 3.25 s at 240 Hz, hands off
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"proud_red={proud_r() * 1000:+.1f}mm "
                      f"white_ok={bool(scene.white_ok()[0])} "
                      f"parked={bool(scene.slug_parked()[0])} "
                      f"settled={bool(scene.all_settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/780 steps", flush=True)
    report("persist")
    s5 = print_score("P-persist persistence 3.25 s (clamps hold, nothing touched)")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
