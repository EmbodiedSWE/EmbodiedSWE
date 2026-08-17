"""Teleport solution for LiddedShoeBoxScene (sim_gen task `put_shoes_in_box_i143`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. OPEN (transport + gravity): one root-state write carries the seated lid to a
   free-space hover over a dynamically-chosen CLEAR spot on the floor (away from the
   box and both shoes); hands-off, it falls and lands on its plug face. The mouth is
   now open — nothing was ever "popped" through the closed lid (smoke presses a shoe
   onto the seated lid with 3x its weight and it never enters: the order is enforced
   by geometry, and this solve honors it by taking the lid off FIRST).
2. PACK (transport + gravity + contact): each shoe is teleported to a hover just
   above the OPEN mouth — heading read from the box's randomized yaw, shoe B flipped
   180 degrees (antiparallel heel-to-toe nesting, the only packing that fits the
   132 mm interior) — then falls ~60 mm onto the box floor. Shoe A is then butted
   flush against its wall with small escalating CoM force pulses (a fingertip push;
   the force frame is PROBED from measured displacement per the pod's frame-drag
   quirk, and every pulse is cleared). This widens shoe B's drop corridor to >= 3 mm
   on each side.
3. CLOSE (transport + gravity): the lid is teleported to a centered hover with the
   plug underside 2 mm above the rim, box-matched heading, and RELEASED: the plug
   funnels into the mouth (4 mm per-side clearance) and the plate settles flush on
   the rim. Seating is a physical outcome — over any mis-packed shoe the plug
   cannot enter and the lid rides proud (smoke constructs exactly that).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still
holds.

Run (forge): python -u -m simgen_tasks.put_shoes_in_box_i143.solve --headless [--seed N]
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

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shoe_box_lid")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build; print the description so the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def box_local(body) -> torch.Tensor:
        return quat_apply_inverse(scene.box.data.root_quat_w,
                                  body.data.root_pos_w - scene.box.data.root_pos_w)

    def report(tag: str) -> None:
        s = scene._status()
        ll = s["lid_loc"][0]
        parts = []
        for nm in scene.SHOE_NAMES:
            p = box_local(scene.shoes[nm])[0]
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})")
        print(f"[solve] {tag:12s} | lid_loc=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
              f"{float(ll[2]):+.3f}) up_z={float(s['lid_up_z'][0]):+.3f} "
              + " ".join(parts)
              + f" inside={s['inside'][0].tolist()} seated={bool(s['seated'][0])} "
              f"clear={bool(s['clear'][0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        return sc

    def settle(body, tag: str, max_steps: int = 360, min_steps: int = 40) -> bool:
        for i in range(max_steps):
            env.step(no_action)
            lv = float(body.data.root_lin_vel_w.norm(dim=-1)[0])
            av = float(body.data.root_ang_vel_w.norm(dim=-1)[0])
            if i > min_steps and lv < 0.02 and av < 0.2:
                return True
        print(f"[solve] {tag}: settle timeout (lv={lv:.3f} av={av:.3f})", flush=True)
        return False

    def box_pose(x_loc: float, y_loc: float, z_loc: float, yaw_rel: float) -> torch.Tensor:
        """(N,13) root state at box-local coords with box-relative yaw, zero velocity."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x_loc, y_loc, z_loc
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.box.data.root_pos_w + quat_apply(scene.box.data.root_quat_w, loc)
        st[:, 3:7] = _qmul(scene.box.data.root_quat_w,
                           _qz(torch.full((n,), yaw_rel, device=device)))
        return st

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # -- force-frame probe state (pod-dependent rotation drag; see module docstring) --
    force_mode = {"m": 0}  # 0: raw world vector; 1: body-encoded (quat_apply_inverse)

    def apply_world_force(body, f_world: torch.Tensor, steps: int) -> None:
        f = f_world if force_mode["m"] == 0 else quat_apply_inverse(
            body.data.root_quat_w, f_world)
        body.set_external_force_and_torque(f.reshape(n, 1, 3), zero_wrench,
                                           env_ids=all_ids, is_global=True)
        step(steps)
        clear_force(body)

    def push_shoe_to_wall(name: str, y_sign: float, y_target: float) -> None:
        """Butt the shoe flush against its box wall (box-local +/-y) with small
        escalating CoM force pulses; frame probed from measured displacement."""
        body = scene.shoes[name]
        dir_loc = torch.zeros(n, 3, device=device)
        dir_loc[:, 1] = y_sign
        newtons = 0.8
        for attempt in range(10):
            y0 = float(box_local(body)[0, 1])
            if (y_sign < 0 and y0 <= y_target) or (y_sign > 0 and y0 >= y_target):
                break
            f_world = quat_apply(scene.box.data.root_quat_w, dir_loc) * newtons
            apply_world_force(body, f_world, 40)
            step(30)
            y1 = float(box_local(body)[0, 1])
            moved = (y1 - y0) * y_sign
            print(f"[solve] {name} push #{attempt} mode={force_mode['m']} "
                  f"F={newtons:.2f}N y {y0:+.4f}->{y1:+.4f}", flush=True)
            if moved < -0.001:  # moved AWAY from the wall: frame is wrong, flip it
                force_mode["m"] ^= 1
                print(f"[solve] force frame flipped -> mode {force_mode['m']}", flush=True)
                continue
            if moved < 0.0005:  # stiction: escalate gently
                newtons = min(newtons * 1.5, 4.0)
        settle(body, name)

    def drop_shoe(name: str, y_loc: float, yaw_rel: float) -> None:
        """TRANSPORT to a free-space hover just above the open mouth, then hands-off:
        the shoe falls onto the box floor. Retries with a tiny x jog if it lands
        tilted/perched (not fully inside)."""
        body = scene.shoes[name]
        idx = scene.SHOE_NAMES.index(name)
        hover_z = c.in_h + c.sole_t / 2 + 0.005
        for attempt in range(5):
            body.write_root_state_to_sim(
                box_pose(0.004 * attempt - 0.008 * (attempt // 2), y_loc, hover_z,
                         yaw_rel), all_ids)
            settle(body, name)
            s = scene._status()
            if bool(s["inside"][0, idx]):
                return
            p = box_local(body)[0]
            print(f"[solve] {name} drop attempt {attempt} not inside "
                  f"(loc={float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}); "
                  f"retrying", flush=True)
        print(f"SIM_GEN_SOLVE: FAIL ({name} would not land flat inside)", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    bp = (scene.box.data.root_pos_w - scene.env_origins)[0]
    bq = scene.box.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    spawns = []
    for nm in scene.SHOE_NAMES:
        p = (scene.shoes[nm].data.root_pos_w - scene.env_origins)[0]
        spawns.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f})")
    print(f"[solve] layout readback (seed {args.seed}): "
          f"box=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) yaw={byaw:+.1f}deg "
          + " ".join(spawns), flush=True)
    report("reset")
    for body in (scene.lid, *scene.shoes.values()):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    ll = scene._status()["lid_loc"][0]
    assert abs(float(ll[2]) - c.seat_z) < 0.004, "lid must start seated"
    s0 = print_score("P0 reset+settle (box closed, shoes on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: take the lid OFF (open the container) -----------------------
    # Choose a clear floor spot: away from the box and both shoes.
    spot = None
    box_xy = scene.box.data.root_pos_w[0, 0:2] - scene.env_origins[0, 0:2]
    shoe_xy = [scene.shoes[nm].data.root_pos_w[0, 0:2] - scene.env_origins[0, 0:2]
               for nm in scene.SHOE_NAMES]
    for r in (0.45, 0.55, 0.62):
        for k in range(16):
            ang = 2 * math.pi * k / 16
            cand = torch.tensor([r * math.cos(ang), r * math.sin(ang)], device=device)
            if (cand - box_xy).norm() < 0.35:
                continue
            if any((cand - sxy).norm() < 0.22 for sxy in shoe_xy):
                continue
            spot = cand
            break
        if spot is not None:
            break
    assert spot is not None, "no clear floor spot for the lid"
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = spot[0]
    st[:, 1] = spot[1]
    st[:, 2] = 0.05  # free-space hover; falls onto its plug face
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.lid.write_root_state_to_sim(st, all_ids)
    settle(scene.lid, "lid")
    report("lid-off")
    assert bool(scene._opened[0]), "lid-clear latch did not set"
    assert not bool(scene.success()[0]), "open box alone cannot be success"
    s1 = print_score("P1 lid taken off and set aside (mouth open)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect 0.15)"

    # ---------------- phase 2: first shoe in, butted flush against its wall ----------------
    drop_shoe("shoe_a", -0.026, 0.0)
    push_shoe_to_wall("shoe_a", -1.0, -0.0285)
    step(60)
    report("shoe_a")
    assert bool(scene._packed[0, 0]), "shoe_a packed latch did not set"
    assert not bool(scene.success()[0]), "one shoe cannot be success"
    s2 = print_score("P2 shoe A flat inside, flush against its wall")
    assert s2 >= s1 - 1e-6 and s2 >= 0.39, f"P2 score {s2} (expect 0.40)"

    # ---------------- phase 3: second shoe in, ANTIPARALLEL (flipped 180 deg) --------------
    drop_shoe("shoe_b", +0.027, math.pi)
    push_shoe_to_wall("shoe_b", +1.0, +0.0285)
    step(60)
    report("shoe_b")
    assert bool(scene._packed[0, 1]), "shoe_b packed latch did not set"
    assert not bool(scene.success()[0]), "shoes packed but box open cannot be success"
    s3 = print_score("P3 shoe B nested heel-to-toe (antiparallel)")
    assert s3 >= s2 - 1e-6 and s3 >= 0.64, f"P3 score {s3} (expect 0.65)"

    # ---------------- phase 4: put the lid back and seat it flush --------------------------
    seated = False
    for attempt in range(6):
        jx = (0.0015 * (attempt % 3 - 1)) if attempt else 0.0
        jy = (0.0015 * (attempt // 3)) if attempt else 0.0
        # plug underside 2 mm above the rim: origin z = in_h + plug_t + plate_t/2 + 0.002
        hover = c.in_h + c.plug_t + c.plate_t / 2 + 0.002
        scene.lid.write_root_state_to_sim(box_pose(jx, jy, hover, 0.0), all_ids)
        settle(scene.lid, "lid-seat")
        step(60)
        if bool(scene._status()["seated"][0]):
            seated = True
            break
        ll = scene._status()["lid_loc"][0]
        print(f"[solve] lid seat attempt {attempt} failed "
              f"(loc={float(ll[0]):+.4f},{float(ll[1]):+.4f},{float(ll[2]):+.4f}); "
              f"re-dropping", flush=True)
    report("lid-on")
    if not (seated and bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (lid did not seat / no success)", flush=True)
        os._exit(1)
    s4 = print_score("P4 lid seated flush — box closed over the packed pair")
    assert s4 >= s3 - 1e-6 and s4 >= 0.99, f"P4 score {s4} (expect 1.0)"

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 40 = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
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
    main()
