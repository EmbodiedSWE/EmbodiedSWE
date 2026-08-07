"""Teleport solution for CheckerCryptScene (sim_gen task `setup_checkers_i39`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. PRY (lever, pure contact): the bar is TELEPORTED into the pry pose — blade
   resting on the notch sill, tip under the lid, handle sticking out horizontally —
   exactly where a gripper would have slid it. That write satisfies no rubric
   clause. Then a downward force at the bar's CoM (the fingertip press on the
   handle; the CoM is handle-side of the sill fulcrum, so a plain down-force is a
   press) pivots the bar over the sill edge and the amplified tip force POPS the
   lid edge up past the rim. The lid's tilt — the `pried` latch and the graspable
   proud edge — is produced entirely by lever contact mechanics; if the pod's
   external-force frame drag defeats the compensated press, the solve re-seats the
   bar and retries with the raw encoding (both attempts are the same fingertip
   press, just encoded for the pod's wrench convention).
2. LID REMOVAL (grasp hand-off): ONLY while the pry physically holds the lid edge
   proud of the rim — verified by readback of a contact-produced tilt — is the lid
   teleported: straight up out of the pocket, then to open ground, where it is
   RELEASED and falls and settles under gravity. This mirrors pinching the popped
   edge and carrying the lid away; the "lid on the ground, clear of the crate"
   facts the rubric checks are produced by the drop and settle.
3. RETRIEVAL (transport + gravity): the RED king is teleported up out of the open
   crate and released a few cm above the green pad; it lands, uprights itself on
   contact, and settles. Both WHITE kings are never touched (the restraint).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.setup_checkers_i39.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul = scene_mod._qmul

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import matrix_from_quat, quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.checker_crypt")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        lid_d, lid_z, lid_loc = scene._lid_stats()
        pos, _q, vel = scene._puck_tensors()
        red_loc = scene._crate_local(pos)[0, 0]
        print(f"[solve] {tag:12s} | lid d={float(lid_d[0]):.3f} z={float(lid_z[0]):.3f} "
              f"tilt={float(scene.lid_tilt_deg()[0]):.2f}deg "
              f"red_loc=({float(red_loc[0]):+.3f},{float(red_loc[1]):+.3f},"
              f"{float(red_loc[2]):+.3f}) "
              f"in_crate={scene.in_crate()[0].tolist()} "
              f"pried={bool(scene._pried[0])} opened={bool(scene._opened[0])} "
              f"out={bool(scene._out[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.bar.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def crate_pt_w(loc_xyz) -> torch.Tensor:
        off = torch.tensor(loc_xyz, device=device, dtype=torch.float32).expand(n, 3)
        return scene.crate.data.root_pos_w + quat_apply(scene.crate.data.root_quat_w, off)

    # ---------------- phase 0: reset, settle, baseline -----------------------------------------
    step(180)
    cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    cq = scene.crate.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
    pp = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    bp = (scene.bar.data.root_pos_w - scene.env_origins)[0]
    slot_of = scene.slot_of[0].tolist()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"crate=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"pad=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"bar=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) slot_of={slot_of}",
          flush=True)
    report("reset")
    # flushness certificate: the seated lid offers nothing to grasp
    _d, _z, lid_loc = scene._lid_stats()
    lid_top = float(lid_loc[0, 2]) + c.lid_t / 2
    rim_top = c.z_seat + c.rim_h
    tilt0 = float(scene.lid_tilt_deg()[0])
    assert abs(lid_top - rim_top) < 0.004, \
        f"lid must sit flush with the rim: lid top {lid_top:.4f} vs rim {rim_top:.4f}"
    assert tilt0 < 1.5, f"seated lid should be flat, tilt={tilt0:.2f}deg"
    assert bool(scene.in_crate()[0].all()), "all three kings must start inside the crate"
    pos, _q, _v = scene._puck_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in king states after settle"
    s0 = print_score("P0 reset+settle (crate sealed, kings hidden)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: pry — lever the lid past the rim (contact only) -----------------
    # Pry pose: bar +x along crate +x, blade on the sill, tip 50 mm past the outer
    # wall face (inside, under the lid), handle out over open ground.
    tip_reach = 0.050                       # tip end beyond the fulcrum (outer wall face)
    fulcrum_x = -(c.inner_half + c.wall_t)  # -0.090
    bar_ctr_x = fulcrum_x + tip_reach - c.bar_len / 2

    def seat_bar() -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = crate_pt_w((bar_ctr_x, 0.0, c.notch_sill + c.bar_t / 2 + 0.001))
        st[:, 3:7] = scene.crate.data.root_quat_w
        scene.bar.write_root_state_to_sim(st, all_ids)
        step(30)  # gravity settles the bar: tip rises to touch the lid underside

    def press(mode: int) -> bool:
        """Fingertip press: escalating down-force at the bar CoM until the lid tilt
        clears the pry gate. mode 0 = raw world force; mode 1 = pre-encoded with
        M = R_ref @ R_now^T (the pod wrench-frame drag, see close_box_i26)."""
        r_ref = matrix_from_quat(scene.bar.data.root_quat_w)
        best = 0.0
        for mag_i in range(6):                    # 1.5 N .. 9 N
            mag = 1.5 * (mag_i + 1)
            for _ in range(90):
                f_des = torch.zeros(n, 3, device=device)
                f_des[:, 2] = -mag
                if mode == 1:
                    m_enc = r_ref @ matrix_from_quat(scene.bar.data.root_quat_w).transpose(1, 2)
                    f_des = (m_enc @ f_des.unsqueeze(-1)).squeeze(-1)
                scene.bar.set_external_force_and_torque(
                    f_des.unsqueeze(1), zero_wrench, env_ids=all_ids, is_global=True)
                env.step(no_action)
                tilt = float(scene.lid_tilt_deg()[0])
                best = max(best, tilt)
                if tilt > c.pry_tilt_deg + 0.7:   # popped, with margin over the gate
                    return True
            print(f"[solve] press mode={mode} mag={mag:.1f}N: lid tilt so far "
                  f"{best:.2f}deg", flush=True)
        return False

    seat_bar()
    popped = press(0)
    if not popped:
        print("[solve] raw press failed; re-seating the bar and retrying with "
              "frame-drag pre-encoding", flush=True)
        clear_force()
        seat_bar()
        popped = press(1)
    if not popped:
        report("PRY-FAIL")
        print("SIM_GEN_SOLVE: FAIL (lever never popped the lid)", flush=True)
        os._exit(1)
    # Certificate readback: the lid edge is held proud by the tip, right now.
    tilt = float(scene.lid_tilt_deg()[0])
    edge_rise = c.lid_size * math.sin(math.radians(tilt))
    print(f"[solve] lever pop held: lid tilt {tilt:.2f}deg "
          f"(near-edge rise ~{edge_rise * 1000:.1f} mm, rim is {c.rim_h * 1000:.0f} mm)"
          , flush=True)
    assert bool(scene._pried[0]), "pried latch must be set by the held pop"
    s1 = print_score("P1 lid levered past the rim (held by the bar tip)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_pry - 1e-6, f"P1 score {s1} (expect >= {c.w_pry})"

    # ---------------- phase 2: lid removal — grasp hand-off while held -------------------------
    # Straight up out of the pocket (the pinch-and-lift), while the tip holds the
    # edge proud; then carry to open ground and RELEASE — gravity settles it.
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = crate_pt_w((0.0, 0.0, 0.25))
    st[:, 3:7] = scene.crate.data.root_quat_w
    scene.lid.write_root_state_to_sim(st, all_ids)
    step(2)
    clear_force()
    # drop spot: around the crate at 0.30 m, maximizing clearance from pad and bar
    crate_xy = scene.crate.data.root_pos_w[0, 0:2]
    pad_xy = scene.pad.data.root_pos_w[0, 0:2]
    bar_xy = scene.bar.data.root_pos_w[0, 0:2]
    best_spot, best_d = None, -1.0
    for k in range(8):
        ang = 2 * math.pi * k / 8
        cand = crate_xy + 0.30 * torch.tensor(
            [math.cos(ang), math.sin(ang)], device=device)
        dmin = min(float((cand - pad_xy).norm()), float((cand - bar_xy).norm()))
        if dmin > best_d:
            best_d, best_spot = dmin, cand
    print(f"[solve] lid drop spot ({float(best_spot[0]):+.3f},"
          f"{float(best_spot[1]):+.3f}), clearance {best_d:.3f} m", flush=True)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = best_spot.unsqueeze(0)
    st[:, 2] = scene.env_origins[:, 2] + 0.05
    st[:, 3] = 1.0
    scene.lid.write_root_state_to_sim(st, all_ids)
    step(180)  # fall + settle, hands-off
    report("lid-removed")
    assert bool(scene.lid_clear()[0]), "lid must rest on the ground clear of the crate"
    assert bool(scene._opened[0]), "opened latch must be set"
    assert bool(scene.in_crate()[0].all()), "kings must still all be inside"
    assert not bool(scene.success()[0]), "cannot be success with the red king inside"
    s2 = print_score("P2 lid off and on the ground, crate open")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_pry + c.w_open - 1e-6, \
        f"P2 score {s2} (expect >= {c.w_pry + c.w_open})"

    # ---------------- phase 3: retrieve the RED king to the pad --------------------------------
    red = scene.pucks["red_0"]
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = crate_pt_w((0.0, 0.0, 0.20))   # up and out over the open crate
    st[:, 3] = 1.0
    red.write_root_state_to_sim(st, all_ids)
    step(2)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = scene.pad.data.root_pos_w[:, 0:2]
    st[:, 2] = scene.env_origins[:, 2] + 0.06   # release a few cm above the pad
    st[:, 3] = 1.0
    red.write_root_state_to_sim(st, all_ids)
    step(180)  # fall + settle, hands-off
    report("red-on-pad")
    assert bool(scene.red_on_pad()[0]), "red king must stand on the pad"
    assert bool(scene.in_crate()[0, 1]) and bool(scene.in_crate()[0, 2]), \
        "both white kings must remain in the crate"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the red king placement)", flush=True)
        os._exit(1)
    s3 = print_score("P3 red king upright on the pad, whites retained")
    assert s3 >= s2 - 1e-6, "score decreased across the retrieval"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -----------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    main()
