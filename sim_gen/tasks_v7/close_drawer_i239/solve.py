"""Teleport solution for DrawerRerailScene (sim_gen task `close_drawer_i239`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating and reorienting a body the solver holds
in free air). Every load-bearing interaction is contact dynamics under applied
force:

1. PERCEPTION: cabinet pose, WHICH bay the beacon marks, and the drawer's fallen
   pose are read back from the episode state — never hard-coded.
2. PRESENT (transport): the fallen drawer is picked off the ground and carried —
   teleported through free space, reorienting on the way — to an upright hover
   directly in front of the MARKED bay's mouth, handle out, fully OUTSIDE the
   cabinet. A short PD hold (gravity feedforward, force-limited: what a firm
   grasp does) stabilises the hover; the hold latch is judged here.
3. INSERT + SEAT (applied force, the load-bearing interaction): with most of the
   grasp's weight support released, a horizontal PD force drives the drawer
   through the bay mouth — the 8 mm side clearance and 34 mm headroom are real
   contact constraints — and slides it along the bay floor (riding the floor
   under ~15% of its weight, real friction) until its rear face presses the back
   wall: the hard stop IS the flush pose. Never teleported inside; never spawned
   seated.
4. RELEASE: forces zeroed, the drawer drops the last millimetres and settles;
   success() is judged on the settled contact outcome.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_drawer_i239.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_rerail")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
        up_dot, face_dot = scene._axes()
        loc = scene._cab_local(scene.drawer.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | front_x={float(scene.front_x()[0]):+.4f} "
              f"loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"up={float(up_dot[0]):+.2f} face={float(face_dot[0]):+.2f} "
              f"seated={bool(scene.seated()[0])} settled={bool(scene.settled()[0])} "
              f"hold={int(scene._hold[0])} ins={float(scene._ins[0]):.2f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def cab_frame():
        return scene.cabinet.data.root_pos_w, scene.cabinet.data.root_quat_w

    def to_world(local: torch.Tensor) -> torch.Tensor:
        cp, cq = cab_frame()
        return cp + quat_apply(cq, local)

    zero = torch.zeros(n, 1, 3, device=device)

    def carry(tgt_local, *, kp: float, kd: float, clamp: float, ff: float,
              kr: float, kw: float, steps: int, done=None, label: str = "") -> None:
        """Applied-force carry of the DRAWER: PD toward a cabinet-local target +
        scaled gravity feedforward (a firm force-limited grasp), plus an
        orientation PD (righting + facing + damping) standing in for the grasp's
        orientation constraint. The wrench goes through the drawer only."""
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        body = scene.drawer
        for _i in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            f_w = ff * c.drawer_mass * 9.81 * ez + kp * (to_world(tgt) - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            _, cq = cab_frame()
            cz = quat_apply(cq, ez)
            cx = quat_apply(cq, ex)
            dz = quat_apply(q, ez)
            dx = quat_apply(q, ex)
            t_w = kr * (torch.cross(dz, cz, dim=-1) + torch.cross(dx, cx, dim=-1)) - kw * w
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._cab_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    def teleport_drawer(local, cab_aligned: bool = True) -> None:
        """TRANSPORT ONLY: relocate + reorient the held drawer through free space.
        The target pose is always fully OUTSIDE the cabinet — the judged insertion
        is executed by applied force through the bay mouth, never by this."""
        _, cq = cab_frame()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = to_world(torch.tensor(local, device=device).expand(n, 3))
        st[:, 3:7] = cq if cab_aligned else torch.tensor([1.0, 0, 0, 0], device=device)
        scene.drawer.write_root_state_to_sim(st, torch.arange(n, device=device))

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # the fallen drawer settles on its side
    cq0 = scene.cabinet.data.root_quat_w[0]
    cp0 = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
    yaw = math.degrees(2.0 * math.atan2(float(cq0[3]), float(cq0[0])))
    k = int(scene.target_bay[0])
    floor_z = c.bay_z[k]
    rest_z = floor_z + c.d_h / 2
    beacon_loc = scene._cab_local(scene.beacon.data.root_pos_w)[0]
    drw_loc = scene._cab_local(scene.drawer.data.root_pos_w)[0]
    up0, _face0 = scene._axes()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cab=({float(cp0[0]):+.3f},{float(cp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"marked_bay={k} floor_z={floor_z:.3f} "
          f"beacon_z={float(beacon_loc[2]):.3f} "
          f"drawer=({float(drw_loc[0]):+.3f},{float(drw_loc[1]):+.3f},"
          f"{float(drw_loc[2]):+.3f}) up_dot={float(up0[0]):+.2f}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(float(beacon_loc[2]) - (c.bay_ceil[k] + c.shelf_t / 2)) < 0.005, \
        "beacon must be mounted over the marked bay"
    assert abs(float(up0[0])) < 0.30, "drawer must start lying on its side"
    assert float(drw_loc[2]) < c.d_w / 2 + 0.02, "drawer must start on the ground"
    assert not bool(scene.seated()[0]), "drawer must not start seated"
    s_prev = print_score("P0 reset+settle (drawer fallen on its side on the apron)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: present at the marked bay's mouth ---------------------------------
    # Transport: pick the fallen drawer up and carry it (reorienting in free air)
    # to an upright hover in front of the MARKED bay — handle out, fully outside.
    hover = (0.035 + c.d_l / 2, 0.0, rest_z + 0.005)
    teleport_drawer(hover, cab_aligned=True)
    carry(hover, kp=40.0, kd=8.0, clamp=6.0, ff=1.0, kr=0.35, kw=0.06,
          steps=90, label="present-hold")
    report("present")
    assert bool(scene._hold[0]), "hold latch must fire at the marked bay's mouth"
    assert float(scene.front_x()[0]) > c.d_l - 0.01, "drawer must still be fully outside"
    s = print_score("P1 drawer retrieved and presented upright at the marked bay's mouth")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert s >= c.w_hold - 1e-6, f"P1 score {s:.3f} below hold credit"
    s_prev = s

    # ---------------- phase 2: force insertion through the mouth to the back stop ----------------
    # The grasp lowers most of its weight support (ff 0.15): the drawer RIDES THE
    # BAY FLOOR under real friction while a horizontal PD drives it through the
    # mouth and presses it home against the back wall — the hard stop is the
    # flush pose. This whole traversal is contact dynamics; no teleport.
    # 2a — traversal: gentle gain rides the bay floor through the mouth and stalls
    # where kp*err == sliding friction (~10 mm short of the back wall). A high
    # gain here is WRONG: the in-flight sag (kp*dz == 0.85*mg) drops the bottom
    # below the bay floor and crashes the drawer into the plinth face.
    carry((c.seat_x - c.d_l / 2 + 0.002, 0.0, rest_z),
          kp=40.0, kd=8.0, clamp=6.0, ff=0.15, kr=0.35, kw=0.06,
          steps=720,
          done=lambda: float(scene.front_x()[0]) < 0.005
          and float(scene.drawer.data.root_lin_vel_w.norm(dim=-1)[0]) < 0.05,
          label="insert-traverse")
    # 2b — press home: from ~10 mm out the error is small, so a stiff gain aimed
    # 4 mm PAST the back wall presses through friction to the hard stop without
    # any slam; the residual error at the stop keeps the drawer pinned.
    carry((c.seat_x - c.d_l / 2 - 0.004, 0.0, rest_z),
          kp=120.0, kd=8.0, clamp=6.0, ff=0.15, kr=0.35, kw=0.06,
          steps=300,
          done=lambda: float(scene.front_x()[0]) < c.seat_x + 0.004
          and float(scene.drawer.data.root_lin_vel_w.norm(dim=-1)[0]) < 0.05,
          label="press-seat")
    report("inserted")
    assert float(scene.front_x()[0]) < c.front_tol, \
        f"drawer failed to reach flush: front_x={float(scene.front_x()[0]):+.4f}"
    assert float(scene._ins[0]) > 0.98, "insertion latch must reach ~1 at the stop"

    # ---------------- phase 3: release — judge the settled contact outcome -----------------------
    step(240)   # hands off: drop the last mm to the bay floor and settle
    report("released")
    assert bool(scene.seated()[0]), "drawer must rest seated after release"
    assert bool(scene.success()[0]), "success() must hold on the settled state"
    s = print_score("P2+P3 drawer driven home by force and released — seated flush")
    assert s >= s_prev - 1e-6, "score decreased across P2/P3"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
