"""Teleport solution for NookCabinetScene (sim_gen task `open_cabinet_i398`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. DRAG (contact dynamics): the cabinet is pulled backwards out of the wall nook
   with a bounded WORLD-frame force through the scene's `cab_f` wrench slot
   (post_step velocity-gates it at `cab_vmax`) until real swing clearance exists
   in front of the face. This is the carry-bar drag a gripper performs: ~8 N
   against a ~4.9 N friction breakaway, far below the ~20 N tipping bound.
2. SWING (contact dynamics): the door is driven open with a small PD torque about
   its free vertical hinge (`door_tau` slot, capped 0.08 N m, omega-capped by the
   scene), then released; angular damping parks it past `open_min_deg`.
3. EXTRACT (contact dynamics): the cube is pushed out through the now-exposed
   mouth with a bounded force (`cube_f` slot), rolling over the sill onto the
   floor; released slow.
4. TRANSPORT (teleport, the only root-state write after reset): the cube — now
   out in the open where a gripper could trivially top-grasp it — is carried to a
   free-air HOVER 30 mm above the delivery mat (asserted clear of everything) and
   FALLS in; the resting placement the rubric scores is made by gravity+contact.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF (wrench slots asserted zero)
for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.open_cabinet_i398.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

DRAG_F = 6.0      # N, cabinet drag (breakaway ~4.9 N; tip bound ~20 N; gentle so
#                   the free door is not flung open by inertia during the drag)
DOOR_KP = 0.30    # N m / rad
DOOR_KD = 0.05    # N m s
DOOR_CAP = 0.08   # N m
PUSH_F = 0.6      # N, cube push (breakaway ~0.19 N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.nook_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

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

    def cab_xy():
        return (scene.cabinet.data.root_pos_w - scene.env_origins)[0, :2]

    def cube_local_x() -> float:
        return float(scene._cab_local(scene.cube.data.root_pos_w)[0, 0])

    def report(tag: str) -> None:
        cx = cab_xy()
        print(f"[solve] {tag:12s} | cab=({float(cx[0]):+.3f},{float(cx[1]):+.3f}) "
              f"clear={float(scene.front_clearance()[0]):+.3f} "
              f"door={math.degrees(float(scene.door_angle()[0])):+.1f}deg "
              f"cube_lx={cube_local_x():+.3f} inside={bool(scene.cube_inside()[0])} "
              f"on_mat={bool(scene.cube_on_mat()[0])} upright={bool(scene.cab_upright()[0])} "
              f"latch(m/c/o/x)={int(scene._moved[0])}{int(scene._clear[0])}"
              f"{int(scene._open[0])}{int(scene._out[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything seats
    cx = cab_xy()
    mc = scene.mat_center[0]
    cu = (scene.cube.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cab=({float(cx[0]):+.3f},{float(cx[1]):+.3f}) "
          f"spawn_gap={c.wall_x - float(cx[0]) - c.nose_x:+.4f} "
          f"cube=({float(cu[0]):+.3f},{float(cu[1]):+.3f},{float(cu[2]):+.3f}) "
          f"mat=({float(mc[0]):+.3f},{float(mc[1]):+.3f})", flush=True)
    report("reset")
    for body in (scene.cabinet, scene.door, scene.cube):
        assert torch.isfinite(body.data.root_state_w).all(), "NaN/inf after settle"
    assert bool(scene.cube_inside()[0]), "cube must start inside the chamber"
    d0 = math.degrees(float(scene.door_angle()[0]))
    assert abs(d0) < 8.0, f"door must start (near) shut, got {d0:.1f} deg"
    assert float(scene.front_clearance()[0]) < 0.10, "cabinet must start nooked"
    s0 = print_score("P0 reset+settle (cabinet nooked, cube sealed inside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: drag the cabinet out of the nook ----------------------------
    # Pull backwards along the cabinet's own -x (readback per step), bounded force,
    # scene-side velocity cap. Stop when the face has real swing room.
    target_clear = c.clear_min + 0.07
    for i in range(1200):
        if float(scene.front_clearance()[0]) >= target_clear:
            break
        fwd = quat_apply(scene.cabinet.data.root_quat_w,
                         torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        f = -DRAG_F * fwd
        f[:, 2] = 0.0
        scene.cab_f[:] = f
        env.step(no_action)
    scene.cab_f[:] = 0.0
    step(120)  # hands-off: cabinet stops (friction+damping), latches judge slow
    report("dragged")
    if float(scene.front_clearance()[0]) < c.clear_min + 0.03:
        fail("drag never reached swing clearance")
    if not bool(scene.cab_upright()[0]):
        fail("cabinet tipped during the drag")
    assert bool(scene.cube_inside()[0]), "cube must still be inside after the drag"
    ang1 = math.degrees(float(scene.door_angle()[0]))
    if ang1 > c.open_min_deg - 10.0:
        fail(f"drag flung the door to {ang1:.1f} deg — the swing must stay deliberate")
    s1 = print_score("P1 cabinet dragged out of the nook (swing clearance made)")
    assert s0 - 1e-6 <= s1 and 0.29 <= s1 <= 0.31, \
        f"P1 score {s1} (expect exactly moved+clear=0.30; open must NOT latch here)"
    assert not bool(scene.success()[0])

    # ---------------- phase 2: swing the door open ------------------------------------------
    theta_ref = math.radians(110.0)
    for i in range(900):
        th = float(scene.door_angle()[0])
        w = float(scene.door.data.root_ang_vel_w[0, 2])
        if th > math.radians(95.0) and abs(w) < 0.25:
            break
        tau = DOOR_KP * (theta_ref - th) - DOOR_KD * w
        scene.door_tau[:] = max(-DOOR_CAP, min(DOOR_CAP, tau))
        env.step(no_action)
    scene.door_tau[:] = 0.0
    step(120)  # hands-off: damping parks the free hinge
    report("door-open")
    ang = math.degrees(float(scene.door_angle()[0]))
    if ang < c.open_min_deg + 10.0:
        fail(f"door parked at only {ang:.1f} deg")
    s2 = print_score("P2 door swung past open_min on its free hinge, parked by damping")
    assert s2 >= s1 - 1e-6 and s2 >= 0.49, f"P2 score {s2} (expect 0.50)"
    assert not bool(scene.success()[0])

    # ---------------- phase 3: push the cube out through the mouth --------------------------
    exit_lx = c.inner_d / 2 + 0.10  # well past the padded inside() front bound
    for i in range(900):
        if cube_local_x() >= exit_lx and not bool(scene.cube_inside()[0]):
            break
        fwd = quat_apply(scene.cabinet.data.root_quat_w,
                         torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        f = PUSH_F * fwd
        f[:, 2] = 0.0
        scene.cube_f[:] = f
        env.step(no_action)
    scene.cube_f[:] = 0.0
    step(120)  # hands-off: cube comes to rest outside; out-latch judges slow
    report("cube-out")
    if bool(scene.cube_inside()[0]):
        fail("cube never left the chamber")
    s3 = print_score("P3 cube pushed out over the sill onto the open floor")
    assert s3 >= s2 - 1e-6 and s3 >= 0.74, f"P3 score {s3} (expect 0.75)"
    assert not bool(scene.success()[0])

    # ---------------- phase 4: carry the cube to the mat (transport + gravity) --------------
    ok_mat = False
    for attempt in range(3):
        hover = torch.zeros(n, 3, device=device)
        hover[:, :2] = scene.mat_center
        hover[:, 2] = c.mat_t + c.cube_s / 2 + 0.030
        hover += scene.env_origins
        # Free-air assert: hover clear of the cabinet/door/wall FOOTPRINTS
        # (point-to-box distance in each body's frame; margin > cube half-diag).
        from isaaclab.utils.math import quat_apply_inverse

        def box_gap(body, cx: float, cy: float, hx: float, hy: float) -> float:
            p = quat_apply_inverse(body.data.root_quat_w[:1],
                                   hover[:1] - body.data.root_pos_w[:1])[0]
            dx = max(abs(float(p[0]) - cx) - hx, 0.0)
            dy = max(abs(float(p[1]) - cy) - hy, 0.0)
            return math.hypot(dx, dy)

        g_cab = box_gap(scene.cabinet, -c.panel_t / 2, 0.0,
                        (c.inner_d + c.panel_t) / 2, c.half_w)
        g_door = box_gap(scene.door, (c.knob_len - c.door_t) / 2, -c.door_w / 2,
                         c.door_t / 2 + c.knob_len, c.door_w / 2)
        assert g_cab > 0.06 and g_door > 0.06, \
            f"hover not free: cab_gap={g_cab:.3f} door_gap={g_door:.3f}"
        assert float(hover[0, 0] - scene.env_origins[0, 0]) < c.wall_x - 0.10
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hover
        st[:, 3] = 1.0
        scene.cube.write_root_state_to_sim(st, all_ids)
        for i in range(240):
            env.step(no_action)
            if i > 20 and float(scene.cube.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
                break
        step(60)
        if bool(scene.cube_on_mat()[0]):
            ok_mat = True
            break
        print(f"[solve] mat drop attempt {attempt + 1} missed; re-carrying", flush=True)
        report("mat-retry")
    report("delivered")
    if not ok_mat or not bool(scene.success()[0]):
        fail("no success after the mat delivery")
    s4 = print_score("P4 cube resting centred on the mat: success")
    assert s4 >= s3 - 1e-6 and s4 >= 0.99, f"P4 score {s4} (expect 1.0)"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    assert float(scene.cab_f.abs().max()) == 0.0 and float(scene.cube_f.abs().max()) == 0.0 \
        and float(scene.door_tau.abs().max()) == 0.0, "wrench slots must be zero"
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s hands-off")
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
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 — die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(1)
