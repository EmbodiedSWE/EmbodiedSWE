"""solve — force-eviction + teleport-drop solution for LetterboxCabinetScene
(libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY; every credit-earning
interaction is applied external force + contact dynamics:

  PHASE E1 EVICT (back slab) — pulsed ~6 N force on the FRONT slab (slug_0) along the
           fixture's -x axis, speed-capped at 0.08 m/s. The front slab pushes the BACK
           slab (slug_1) ahead of it — a contact chain — through the discard-chute
           opening (per-slab passage latch) and over the plinth edge; gravity tips it
           onto the 30.8 deg slide (pair mu ~0.30 << tan 30.8 deg) and it lands in the
           catch bin. Force cut at back-slab x < -0.215; settle.            -> 0.250
  PHASE E2 EVICT (front slab) — same pulsed push on the front slab itself until ITS
           center passes x < -0.215 (through the chute band, over the edge). Cut,
           gravity-slide, both slabs at rest inside the bin.                -> 0.500
  PHASE D  DROP — identify the MIDDLE bowl (scene.target_idx) and teleport it from
           the staging row to fixture (0.03, 0, 0.44): above the roof port, ABOVE the
           port passage band (z 0.32..0.40), zero velocity. Assert the score is STILL
           0.500 — the teleport earned nothing. Then hands off: the bowl free-falls
           through the port (passage latch fires in the band), lands upright on the
           cleared seat, settles: success().                                -> 1.000
  PERSIST  >= 3.5 simulated seconds fully hands-off; success() must still hold
           before `SIM_GEN_SOLVE: SUCCESS` is printed.

The pod's external-force API may interpret wrenches in the body's current frame
(pod/version dependent), so drive() probes force encoding at runtime from measured
progress and toggles between raw-world and quat_apply_inverse(q_now, f).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (latched credit — the printed
sequence never decreases) and exactly `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the persistence window. Hard exit (os._exit) with a watchdog Timer.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
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
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_cabinet")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def clear_forces() -> None:
        for b in scene.slugs + scene.bowls:
            b.set_external_force_and_torque(zero3, zero3)

    def push(body, f_world: torch.Tensor) -> None:
        clear_forces()
        body.set_external_force_and_torque(f_world.view(1, 1, 3), zero3)

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

    def report(tag: str) -> None:
        tf = scene.target_fix()[0]
        sp = scene._slug_pos_fix()[0]
        print(f"[solve] {tag:10s} target_idx={int(scene.target_idx[0])} "
              f"tgt=({tf[0]:+.3f},{tf[1]:+.3f},{tf[2]:+.3f}) "
              f"slugA=({sp[0, 0]:+.3f},{sp[0, 2]:+.3f}) "
              f"slugB=({sp[1, 0]:+.3f},{sp[1, 2]:+.3f}) "
              f"chute={scene._chute_ever[0].tolist()} "
              f"binned={scene._binned_ever[0].tolist()} "
              f"ported={bool(scene._ported_ever[0])} "
              f"seated={bool(scene._seated_ever[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

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

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    assert not bool(scene._chute_ever[0].any() | scene._binned_ever[0].any()
                    | scene._ported_ever[0] | scene._seated_ever[0]), \
        "no latch may fire at reset"
    assert bool(scene.slugs_in_comp()[0].all()), "both slabs must start in the seat"
    phase_score("reset")  # ~0.000

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402,F401

    q_fix = scene.fixture.data.root_quat_w[0:1]
    p_fix = scene.fixture.data.root_pos_w[0:1]

    def fix_to_world(loc) -> torch.Tensor:
        loc_t = torch.tensor([loc], device=device, dtype=torch.float32)
        return (p_fix + quat_apply(q_fix, loc_t))[0]

    # eviction direction: fixture -x (toward the chute), world frame
    evict_ax = quat_apply(q_fix, torch.tensor([[-1.0, 0.0, 0.0]], device=device))[0]

    def slug_x(i: int) -> float:
        return float(scene._slug_pos_fix()[0, i, 0])

    def v_along(body, axis: torch.Tensor) -> float:
        return float((body.data.root_lin_vel_w[0] * axis).sum())

    def drive(body, axis: torch.Tensor, fmag: float, vmax: float, done,
              max_steps: int, tag: str) -> bool:
        """Pulsed push along world `axis` until `done()`. Probes force encoding from
        measured progress: if headway stalls, toggle between raw-world and
        quat_apply_inverse(q_now, f) (pod may rotate wrenches by the body's frame)."""
        mode = 0
        last_probe = float((body.data.root_pos_w[0] * axis).sum())
        for i in range(max_steps):
            if done():
                clear_forces()
                return True
            f = fmag if v_along(body, axis) < vmax else 0.0
            fw = f * axis
            if mode == 1:
                fw = quat_apply_inverse(body.data.root_quat_w[0:1], fw.view(1, 3))[0]
            push(body, fw)
            step(1)
            if (i + 1) % 40 == 0:
                cur = float((body.data.root_pos_w[0] * axis).sum())
                if cur - last_probe < 0.001:
                    mode ^= 1
                    print(f"[solve] {tag}: progress stalled, force mode -> {mode}",
                          flush=True)
                last_probe = cur
        clear_forces()
        return done()

    slug_a, slug_b = scene.slugs[0], scene.slugs[1]

    # ================= PHASE E1: contact-chain the BACK slab out through the chute =============
    # ~6 N pulsed on the front slab (chain friction ~2.8 N resists; speed cap keeps it
    # quasi-static, no tipping: 6 N * 0.0225 m < mg * 0.065 m restoring). The back slab
    # is shunted through the chute band (x -0.22..-0.145), over the plinth edge
    # (x = -0.19), tips onto the slide, and gravity delivers it to the bin.
    drive(slug_a, evict_ax, 6.0, 0.08, lambda: slug_x(1) < -0.215, 3600, "evict_B")
    ok1 = settle_until(lambda: bool(scene._binned_ever[0, 1])
                       and bool(scene.slugs_in_bin()[0, 1])
                       and bool(scene.settled()[0]), max_steps=1200)
    report("evict_B")
    if not ok1:
        print("[solve] PHASE E1 FAILED: back slab did not settle in the bin", flush=True)
        verdict(False)
    assert bool(scene._chute_ever[0, 1]), "back slab chute latch must have fired"
    assert not bool(scene._chute_ever[0, 0]), \
        "front slab must not have entered the chute band yet"
    phase_score("evict_B")  # 0.250

    # ================= PHASE E2: push the FRONT slab out the same way ==========================
    drive(slug_a, evict_ax, 4.0, 0.08, lambda: slug_x(0) < -0.215, 3600, "evict_A")
    ok2 = settle_until(lambda: bool(scene._binned_ever[0].all())
                       and bool(scene.slugs_in_bin()[0].all())
                       and bool(scene.settled()[0]), max_steps=1200)
    report("evict_A")
    if not ok2:
        print("[solve] PHASE E2 FAILED: front slab did not settle in the bin", flush=True)
        verdict(False)
    assert not bool(scene.slugs_in_comp()[0].any()), "seat must be clear of slabs"
    phase_score("evict_A")  # 0.500

    # ================= PHASE D: teleport the MIDDLE bowl above the port, then drop =============
    # Transport only: (0.03, 0, 0.44) is ABOVE the port passage band (z 0.32..0.40) and
    # inside the port's airspace (26 mm radial margin); the teleport must not move the
    # score. Hands off, the bowl free-falls ~15 cm through the band (the passage latch
    # fires mid-fall) onto the cleared seat.
    tgt = scene.bowls[int(scene.target_idx[0])]
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = fix_to_world((0.03, 0.0, 0.44))
    st[0, 3:7] = q_fix  # upright, aligned with the fixture
    tgt.write_root_state_to_sim(st, torch.tensor([0], device=device))
    assert abs(sc() - last_score) < 1e-6, \
        f"teleport transport moved the score: {last_score} -> {sc():.3f}"
    ok3 = settle_until(lambda: bool(scene.success()[0]), max_steps=900)
    report("drop")
    if not ok3:
        print("[solve] PHASE D FAILED: bowl did not seat through the port", flush=True)
        verdict(False)
    assert bool(scene._ported_ever[0]), "port passage latch must have fired mid-fall"
    phase_score("drop")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
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
