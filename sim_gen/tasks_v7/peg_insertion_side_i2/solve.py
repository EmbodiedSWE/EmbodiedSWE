"""Teleport solution for HaspPinScene (sim_gen task `peg_insertion_side_i2`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in two writes, each ending in FREE SPACE:
  P1 — the bar is carried from its table spawn to a hover pose 8 mm ABOVE the plate
  top, eye over the bore, yaw matched to the stand. Seating then happens through
  CONTACT DYNAMICS: the bar falls the last 8 mm and comes to rest ON the plate under
  gravity and real contact (exactly the release an arm performs at set-down).
  P2 — the pin is carried to a vertical hover with its tip 10 mm ABOVE the seated
  bar's eye — entirely outside both holes. The fastening then goes through CONTACT
  DYNAMICS: a floating-hand force controller (velocity-regulated descent at
  ~0.12 m/s, lateral PD toward the bore axis, upright-steadying torque — the forces a
  hand lowering a pin applies) lowers the pin so its shaft threads the bar's eye and
  the plate bore under real contact, until the wide head lands on the eye and carries
  the pin. Forces are then cut and everything settles on real contact. The pin is
  never teleported into either hole; the bar is never teleported onto the plate.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i2.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hasp_pin_link")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def stand_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        q = scene.stand.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        bl = scene._stand_local(scene.bar.data.root_pos_w)[0]
        _, tip_w, head_w = scene._pin_ends_w(scene.pin)
        tip_z = float((tip_w - scene.env_origins)[0, 2])
        head_z = float((head_w - scene.env_origins)[0, 2])
        print(f"[solve] {tag:12s} | bar_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) seated={bool(scene.bar_seated()[0])} "
              f"pin_tip_z={tip_z:.3f} pin_head_z={head_z:.3f} "
              f"linked={bool(scene.pin_linked()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.pin.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp, syaw = stand_pose()
    bar0 = (scene.bar.data.root_pos_w - scene.env_origins)[0]
    pin0 = (scene.pin.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): stand=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"bar_spawn=({float(bar0[0]):+.3f},{float(bar0[1]):+.3f},{float(bar0[2]):.3f}) "
          f"pin_spawn=({float(pin0[0]):+.3f},{float(pin0[1]):+.3f},{float(pin0[2]):.3f}) "
          f"d0={float(scene.d0[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: bar TRANSPORT (teleport to hover) + contact seat ------------
    # Hover: eye centre on the bore axis, bar yaw = stand yaw (handle over the open
    # side of the plate), underside 8 mm ABOVE the plate top — free space; the seating
    # itself is the 8 mm gravity drop onto real plate contact.
    cy2, sy2 = math.cos(syaw / 2), math.sin(syaw / 2)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(sp[0])
    st[:, 1] = float(sp[1])
    st[:, 2] = c.plate_top + 0.008 + c.bar_t / 2
    st[:, 3] = cy2
    st[:, 6] = sy2
    st[:, 0:3] += scene.env_origins
    scene.bar.write_root_state_to_sim(st, all_ids)
    step(120)  # drop + settle (1 s)
    report("bar-seated")
    s1 = print_score("P1 bar transport + gravity seat")
    assert s1 >= s0 - 1e-6, "score decreased across bar transport"
    if not bool(scene.bar_seated()[0]):
        print("SIM_GEN_SOLVE: FAIL (bar did not seat)", flush=True)
        os._exit(1)

    # ---------------- phase 2: pin TRANSPORT (hover above the eye) + contact thread --------
    # Hover: vertical, tip 10 mm ABOVE the seated eye top — outside both holes.
    sp, _ = stand_pose()  # re-read (bar drop cannot move the kinematic stand, but be exact)
    tip_hover = c.eye_top_seated + 0.010
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(sp[0])
    st[:, 1] = float(sp[1])
    st[:, 2] = tip_hover + c.pin_half
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.pin.write_root_state_to_sim(st, all_ids)
    step(1)
    report("pin-hover")

    # Floating-hand descent: velocity-regulated vertical support (descend ~0.12 m/s),
    # lateral PD toward the bore axis, upright-steadying torque. The shaft threads the
    # eye, then the plate bore, under real contact; the loop ends when the HEAD lands
    # on the eye and carries the pin.
    m_pin, g = c.pin_mass, 9.81
    v_des, extra_down = -0.12, 0.0
    tgt = torch.tensor([float(sp[0]), float(sp[1])], device=device)
    best_head_z, last_drop = 1.0, 0
    seated_head = False
    for i in range(2400):
        axis, tip_w, head_w = scene._pin_ends_w(scene.pin)
        head_z = float((head_w - scene.env_origins)[0, 2])
        tip_z = float((tip_w - scene.env_origins)[0, 2])
        v_w = scene.pin.data.root_lin_vel_w[0]
        w_w = scene.pin.data.root_ang_vel_w[0]
        if head_z <= c.eye_top_seated + 0.0015 and abs(float(v_w[2])) < 0.02:
            seated_head = True
            break
        # vertical: track the descent velocity (support force, never a downward pin)
        fz = m_pin * (g + 25.0 * (v_des - float(v_w[2]))) - extra_down
        fz = max(0.0, min(3.0 * m_pin * g, fz))
        # lateral PD toward the bore axis
        p_xy = (scene.pin.data.root_pos_w - scene.env_origins)[0, :2]
        e_xy = tgt - p_xy
        f_xy = m_pin * (60.0 * e_xy - 15.0 * v_w[:2])
        f_xy = f_xy.clamp(-0.6, 0.6)
        f_world = torch.tensor([float(f_xy[0]), float(f_xy[1]), fz], device=device)
        # upright-steadying torque
        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        tq = 0.02 * torch.linalg.cross(axis[0], ez) - 0.004 * w_w
        tq = tq.clamp(-0.04, 0.04)
        scene.pin.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if head_z < best_head_z - 0.002:
            best_head_z, last_drop = head_z, i
        elif i - last_drop > 300:  # stalled: lean on it a little harder
            extra_down = min(extra_down + 0.3, 1.5)
            last_drop = i
            print(f"[solve] descent stalled at head_z={head_z:.3f} tip_z={tip_z:.3f}, "
                  f"extra_down={extra_down:.1f} N", flush=True)
    clear_wrench()
    step(240)  # 2 s: head seats fully, everything settles on real contact
    report("pin-linked")
    print(f"[solve] descent loop done (seated_head={seated_head})", flush=True)
    s2 = print_score("P2 pin transport + contact threading + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the pin threading"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after thread+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) -----
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
    main()
