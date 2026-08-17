"""Teleport solution for FeedLineCullScene (sim_gen task `place_shape_in_shape_sorter_i180`)
— the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. CULL (teleport = transport, then gravity): one root-state write carries the red
   reject cube from the queue straight up and over to a HOVER above open ground,
   clear of the machine, zero velocity. The write satisfies nothing (at hover
   height red_clear() is false — its z clause requires ground level); the cube
   then FALLS and settles on the ground by contact dynamics. This is exactly the
   embodied pick-and-set-down the culling step means.
2. EXTRACTION (force servo, real mechanism): a velocity-servo axial force on the
   pin (`set_external_force_and_torque` — what a hand pinching the yellow knob
   applies) draws the pin out of both wall holes toward its knob side, with
   gravity feedforward so the near-weightless shaft slides in the hole slack, a
   weak alignment torque standing in for the grip's orientation constraint, and
   a speed cap. Gains respect the one-substep wrench delay (kv*dt/m ~ 0.27).
   Once the far tip has cleared the +y wall the servo brakes the axial motion,
   lets the pin down under 10% net gravity, and releases; the pin lands on the
   ground beside the chute. Nothing else is touched.
3. DELIVERY (the mechanism does the work): hands off. With the pin gone, the
   queue slides down the slick 16 deg chute (tan 16 >> mu ~ 0.125 by authored
   materials), launches off the lip and the walled catch bin collects every blue
   cube — pure contact dynamics; no body is ever pushed toward the goal.
4. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched —
   blues resting in the bin, red resting clear on the ground, pin out.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.place_shape_in_shape_sorter_i180.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.feed_line_cull")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def pin_y() -> float:
        return float(scene.pin_local()[0, 1])

    def banked_str() -> str:
        bk = scene.blues_banked()[0]
        pres = scene.present[0]
        return "".join("B" if (pres[j] and bk[j]) else ("." if pres[j] else "_")
                       for j in range(c.max_blue))

    def report(tag: str) -> None:
        rp = scene._rig_local(scene.red.data.root_pos_w)[0]
        pl = scene.pin_local()[0]
        print(f"[solve] {tag:14s} | pin=({float(pl[0]):+.3f},{float(pl[1]):+.3f},"
              f"{float(pl[2]):+.3f}) red=({float(rp[0]):+.3f},{float(rp[1]):+.3f},"
              f"{float(rp[2]):+.3f}) banked=[{banked_str()}] "
              f"red_clear={bool(scene.red_clear()[0])} "
              f"pin_out={bool(scene.pin_out()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # queue closes its spawn gaps and presses the seated pin
    rq = scene.rig.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    rp0 = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rp0[0]):+.3f},{float(rp0[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"n_blue={int(scene.n_blue[0])} red_slot={int(scene.red_slot[0])} "
          f"pin_y={pin_y():+.4f}", flush=True)
    pin_m = float(scene.pin.root_physx_view.get_masses()[0].sum())
    cube_m = float(scene.red.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: pin={pin_m:.3f} kg cube={cube_m:.3f} kg", flush=True)
    assert 0.10 < pin_m < 0.50, f"pin mass {pin_m} (density not applied?)"
    assert 0.05 < cube_m < 0.20, f"cube mass {cube_m} (density not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(pin_y()) < 0.006, f"pin must hold its seat, y={pin_y():+.4f}"
    red_loc = scene._rig_local(scene.red.data.root_pos_w)[0]
    assert float(red_loc[2]) > 0.20, "red reject must start ON the chute"
    for j in range(c.max_blue):
        if bool(scene.present[0, j]):
            bl = scene._rig_local(scene.blues[j].data.root_pos_w)[0]
            assert float(bl[2]) > 0.20, f"blue{j} must start ON the chute"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (queue held by the seated pin)")
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: CULL the reject (transport to a hover, then drop) -----------
    hover = torch.zeros(n, 3, device=device)
    hover[:] = torch.tensor([0.40, 0.32, 0.10], device=device)  # rig-local, open ground
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, hover)
    st[:, 3:7] = scene.rig.data.root_quat_w
    scene.red.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.red_clear()[0]), \
        "hover must not read as culled (the write satisfies nothing)"
    print("[solve] P1: red reject transported to a hover over open ground "
          "(0.10 m up: red_clear is false at the write; the drop settles it)", flush=True)
    step(90)  # falls ~0.08 m, settles on the ground
    report("culled")
    assert bool(scene.red_clear()[0]), "red must settle clear on the ground"
    assert bool(scene._red_latch[0]), "red_clear latch must be set"
    s1 = print_score("P1 reject culled off the line")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_red - 0.01, f"P1 score {s1}"

    # ---------------- phase 2: EXTRACT the release pin (axial force servo) ------------------
    def apply_pin(f_ax: float, support: float) -> None:
        """World wrench on the pin: axial (rig +y) force + gravity feedforward +
        weak axis-alignment torque; converted to the pin body frame (the pin
        barely rotates, so the pod's frame-drag quirk is moot)."""
        q = scene.pin.data.root_quat_w
        ey_w = quat_apply(scene.rig.data.root_quat_w, ey)
        f_w = f_ax * ey_w + support * pin_m * 9.81 * ez
        f_b = quat_apply_inverse(q, f_w)
        py_w = quat_apply(q, ey)  # pin's own axis in world
        w = scene.pin.data.root_ang_vel_w
        t_w = 0.02 * torch.cross(py_w, ey_w, dim=-1) - 0.002 * w
        t_b = quat_apply_inverse(q, t_w)
        scene.pin.set_external_force_and_torque(f_b.reshape(n, 1, 3),
                                                t_b.reshape(n, 1, 3))

    v_des, kv, f_hi = 0.06, 6.0, 8.0
    pulled = False
    for i in range(900):
        ey_w = quat_apply(scene.rig.data.root_quat_w, ey)
        v_ax = float((scene.pin.data.root_lin_vel_w * ey_w).sum(dim=-1)[0])
        f_ax = max(-2.0, min(f_hi, kv * (v_des - v_ax) + 0.5))
        apply_pin(f_ax, 1.0)
        env.step(no_action)
        # pull well past pin_out_y (0.118): at 0.175 the -y shaft tip (0.111) has
        # 24 mm of air over the bin side wall's outer face (0.087), so the pin
        # can be lowered to the ground beside the bin without perching on it
        if pin_y() >= 0.175:
            pulled = True
            break
    print(f"[solve] pin pull: y={pin_y():+.4f} after {i + 1} steps", flush=True)
    assert pulled, f"pin did not extract, y={pin_y():+.4f}"
    # let the freed pin down gently (brake axial, 10% net weight), then release
    for _ in range(240):
        ey_w = quat_apply(scene.rig.data.root_quat_w, ey)
        v_ax = float((scene.pin.data.root_lin_vel_w * ey_w).sum(dim=-1)[0])
        apply_pin(max(-2.0, min(2.0, kv * (0.0 - v_ax))), 0.90)
        env.step(no_action)
        if float(scene.pin.data.root_pos_w[0, 2] - scene.env_origins[0, 2]) < 0.035:
            break
    scene.pin.set_external_force_and_torque(zero, zero)
    step(90)
    report("pin-out")
    assert bool(scene.pin_out()[0]), f"pin must be fully out, y={pin_y():+.4f}"
    pz = float(scene.pin.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    assert pz < 0.06, f"pin should rest on the ground, z={pz:.3f}"
    s2 = print_score("P2 release pin fully extracted (mechanism stroke done)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_red + c.w_pin - 0.02, f"P2 score {s2}"

    # ---------------- phase 3: DELIVERY — gravity runs the line, hands off ------------------
    banked_all = False
    for k in range(16):  # up to 16 x 60 = 960 steps = 8 s
        step(60)
        bk = (scene._bank_latch[0] | ~scene.present[0]).all() \
            and (scene.blues_banked()[0] | ~scene.present[0]).all()
        if bool(bk) and bool(scene.settled()[0]):
            banked_all = True
            break
    report("delivered")
    for j in range(c.max_blue):
        if bool(scene.present[0, j]):
            bl = scene._rig_local(scene.blues[j].data.root_pos_w)[0]
            bv = float(scene.blues[j].data.root_lin_vel_w[0].norm())
            print(f"[solve] blue{j}: local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                  f"{float(bl[2]):+.3f}) |v|={bv:.3f} "
                  f"banked={bool(scene.blues_banked()[0, j])} "
                  f"latch={bool(scene._bank_latch[0, j])}", flush=True)
    if not banked_all:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (queue did not deliver every blue to the bin)",
              flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (banked but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 queue delivered — every blue cube banked")
    assert s3 >= s2 - 1e-6 and s3 >= 0.999, f"P3 score {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
