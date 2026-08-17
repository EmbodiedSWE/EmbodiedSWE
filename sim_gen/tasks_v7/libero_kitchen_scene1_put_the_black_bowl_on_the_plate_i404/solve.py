"""solve — TELEPORT solution for CaliberVaultScene
(libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404).

Scene-level env (robot="null").  Teleportation is used for TRANSPORT only — each marble
is teleported to a HOVER point at the mouth of its own admission machine and released;
every scoring displacement afterwards happens through contact dynamics:

  PHASE A  RAIL RUN (big marble): teleport the big marble to a hover just above the
           rails in the OPEN loading bay (rig-frame x = drop_x, outside the vault,
           under open sky) and release.  It settles into the rail groove, and because
           the diverging gap lowers its center covertly, it self-accelerates down the
           run, passes under the canopy, clears the bulkhead, and FALLS THROUGH the
           widening gap into the roofed vault — all under gravity and contact alone.
  PHASE B  LANE PUSH (small marble): teleport the small marble to a hover over the
           START of the open-top runway lane (outside the scoring band) and release.
           Then push it down the lane with a small per-step external CONTACT-SCALE
           force (bang-bang, velocity-capped, world direction re-expressed in the
           marble's body frame every step because the rolling body's frame spins).
           It rolls through the side port and drops over the sill into the vault; the
           force is cut the moment its center passes the wall plane.
  RETRY    each phase re-releases from the hover (transport from a non-scoring spot)
           if its marble somehow fails to arrive; the push escalates its force cap on
           a stall (velocity-servo memory: escalate, don't wiggle).
  PHASE C  hands off for >= 3.5 simulated seconds; success() must persist.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window.  Hard exit (os._exit)
after the verdict, with a daemon watchdog Timer as backstop.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404.solve --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.caliber_vault")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

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

    def loc(body) -> torch.Tensor:
        return scene.local(body)[0]

    def report(tag: str) -> None:
        lb, ls = loc(scene.big), loc(scene.small)
        print(f"[solve] {tag:12s} big_loc=({lb[0]:+.3f},{lb[1]:+.3f},{lb[2]:+.3f}) "
              f"small_loc=({ls[0]:+.3f},{ls[1]:+.3f},{ls[2]:+.3f}) "
              f"big_vault={bool(scene.in_vault(scene.big)[0])} "
              f"small_vault={bool(scene.in_vault(scene.small)[0])} "
              f"settled={bool(scene.settled()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def rig_to_world(p_local) -> torch.Tensor:
        p = torch.tensor([float(v) for v in p_local], device=device).unsqueeze(0)
        return (scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, p))[0]

    def tp(body, p_local) -> None:
        """TRANSPORT a marble to a rig-frame point (always a non-scoring hover)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = rig_to_world(p_local)
        st[0, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def clear_force() -> None:
        z = torch.zeros(1, 1, 3, device=device)
        scene.small.set_external_force_and_torque(z, z, env_ids=all_ids)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")

    for name, body, want in (("rig", scene.rig, c.rig_mass),
                             ("big", scene.big, c.big_mass),
                             ("small", scene.small, c.small_mass)):
        actual = float(body.root_physx_view.get_masses().reshape(-1)[0])
        assert abs(actual - want) < 1e-3, f"{name} mass readback {actual:.4f} != {want:.4f}"
        print(f"[solve] mass readback {name}: {actual:.3f} kg", flush=True)

    # staging sanity: marbles on the open floor in their swapped slots, nothing scored
    for body, r in ((scene.big, c.big_r), (scene.small, c.small_r)):
        lz = float(loc(body)[2])
        assert abs(lz - (r + 0.0)) < 0.02, f"marble not resting on the floor (local z {lz})"
    assert not bool(scene.success()[0]), "reset state must not satisfy the goal"
    assert sc() <= 0.005, "score must start at ~0"
    slot = "A" if bool(scene.big_in_a[0]) else "B"
    print(f"[solve] staging: big marble in slot {slot}", flush=True)
    phase_score("reset")  # 0.000

    # ================= PHASE A: rail run (big marble) ==========================================
    hover_big = (c.drop_x, 0.0, c.rail_z + c.ride_h0 + c.big_r + 0.012)
    for attempt in range(3):
        tp(scene.big, hover_big)
        step(30)  # falls ~12 mm onto the rails, seats into the groove
        ok = settle_until(
            lambda: bool(scene.in_vault(scene.big)[0]) and bool(scene.settled()[0]),
            max_steps=1500)
        report(f"rail-run{attempt + 1}")
        if ok:
            break
        print(f"[solve] rail run attempt {attempt + 1} did not deliver; re-releasing",
              flush=True)
    assert bool(scene.in_vault(scene.big)[0]), "big marble failed to arrive in the vault"
    assert bool(scene._ride_ever[0]) and bool(scene._canopy_ever[0]), (
        "delivery must have happened via the rail ride (latches missing)")
    phase_score("big-delivered")  # 0.500

    # ================= PHASE B: lane push (small marble) =======================================
    lane_cx = (c.port_x0 + c.port_x1) / 2
    hover_small = (lane_cx, 0.270, c.lane_floor_top + c.small_r + 0.012)
    f_mag = 0.06  # N — contact-scale nudge on a 30 g marble
    v_cap = 0.30  # m/s
    delivered = False
    for attempt in range(4):
        tp(scene.small, hover_small)
        step(30)  # settles onto the lane floor
        stall_ctr = 0
        for i in range(1500):
            p = loc(scene.small)
            if float(p[1]) < c.side_half_w_in - 0.003:  # center past the wall plane
                break
            dir_w = quat_apply(scene.rig.data.root_quat_w,
                               torch.tensor([[0.0, -1.0, 0.0]], device=device))[0]
            v_along = float(torch.dot(scene.small.data.root_lin_vel_w[0], dir_w))
            push = f_mag if v_along < v_cap else 0.0
            f_body = quat_apply_inverse(scene.small.data.root_quat_w,
                                        (push * dir_w).unsqueeze(0))
            zt = torch.zeros(1, 1, 3, device=device)
            scene.small.set_external_force_and_torque(f_body.unsqueeze(1), zt,
                                                      env_ids=all_ids)
            env.step(no_action)
            if abs(v_along) < 0.01 and float(p[1]) > c.lane_y0 + 0.01:
                stall_ctr += 1
                if stall_ctr > 240:  # 2 s stalled mid-lane: escalate the force
                    f_mag = min(f_mag * 1.5, 0.30)
                    stall_ctr = 0
                    print(f"[solve] lane stall — escalating push to {f_mag:.3f} N",
                          flush=True)
            else:
                stall_ctr = 0
        clear_force()
        step(2)
        ok = settle_until(
            lambda: bool(scene.in_vault(scene.small)[0]) and bool(scene.settled()[0]),
            max_steps=600)
        report(f"lane-push{attempt + 1}")
        if ok:
            delivered = True
            break
        f_mag = min(f_mag * 1.5, 0.30)
        print(f"[solve] lane push attempt {attempt + 1} did not deliver; retrying at "
              f"{f_mag:.3f} N", flush=True)
    assert delivered, "small marble failed to arrive in the vault"
    assert bool(scene._lane_ever[0]), "delivery must have happened via the lane (latch missing)"
    phase_score("small-delivered")

    # ================= success + persistence ===================================================
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("both-in")
    if not ok:
        print("[solve] FAILED: goal state not reached after both deliveries", flush=True)
        verdict(False)
    phase_score("success")  # 1.000

    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL", flush=True)
        os._exit(2)
