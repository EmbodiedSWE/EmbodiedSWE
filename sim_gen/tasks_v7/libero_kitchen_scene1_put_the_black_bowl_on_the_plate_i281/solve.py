"""solve — TELEPORT solution for BerryPourScene
(libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281).

Scene-level env (robot="null"). Teleportation is used for TRANSPORT of the BOWL only
(a per-step teleport-hold plays the role of the hand that carries and tilts it); the
berries are NEVER teleported into a scoring state — every berry that ends on the plate
got there by rolling out of the tilted bowl and falling under gravity:

  PHASE 1  LIFT: teleport-hold the bowl smoothly up off the floor (transport; the
           berries ride inside on real contact — the rising floor of the cup carries
           them, nothing is attached).
  PHASE 2  CARRY: move the held bowl over the plate at safe height.
  PHASE 3  POUR (the load-bearing interaction): tilt the held bowl past horizontal
           (128 deg) around its pouring lip, the lip held just above the plate. The
           berries roll over the rim and drop onto the plate purely under gravity and
           contact; a small lateral shake dislodges stragglers. Nothing is spawned on
           the plate, nothing is pinned there.
  PHASE 4  RETURN: un-tilt, carry the emptied bowl back, lower it to just above the
           floor and RELEASE it (it falls the last ~3 mm and settles free).
  RETRY    if a berry missed the plate or stayed in the bowl, the stray is transported
           back INTO the bowl (asserted un-scored at release) and the pour cycle
           repeats — plate landings only ever happen through the pour.
  PHASE 5  hands off for >= 3.5 simulated seconds; success() must persist before the
           verdict.

Prints the scene readouts, `SIM_GEN_SCORE <score>` at each phase boundary (latched
credit — the printed sequence never decreases), and exactly `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds after the persistence window. Hard exit (os._exit) after
the verdict, with a daemon watchdog Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (rim pinch-grasp + wrist-roll
pour) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281.solve --headless
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

THETA_MAX = math.radians(128.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.berry_pour")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def settle_until(pred, max_steps: int = 600, poll: int = 10) -> bool:
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

    def n_pres() -> int:
        return int(scene.n_present()[0])

    def n_counted() -> int:
        return int(scene.counted()[0].sum())

    def n_in_bowl() -> int:
        return int((scene.in_bowl() & scene.present)[0].sum())

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} present={n_pres()} on_plate={n_counted()} "
              f"in_bowl={n_in_bowl()} bowl_clear={bool(scene.bowl_clear()[0])} "
              f"upright={bool(scene.bowl_upright()[0])} settled={bool(scene.settled()[0])} "
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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def hold_bowl(pos, quat) -> None:
        """Per-step teleport-hold of the bowl (the 'hand'): pose written, velocity
        zero. The berries are free bodies — they follow only through real contact."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos], device=device)
        st[0, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        scene.bowl.write_root_state_to_sim(st, all_ids)
        env.step(no_action)

    def teleport_berry(i: int, pos) -> None:
        """TRANSPORT a stray free berry across free space (never into a scoring
        state — asserted by the caller)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor([float(v) for v in pos], device=device)
        st[0, 3] = 1.0
        scene.berries[i].write_root_state_to_sim(st, all_ids)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")

    for name, body, want in (("plate", scene.plate, c.plate_mass),
                             ("bowl", scene.bowl, c.bowl_mass),
                             ("berry_0", scene.berries[0], c.berry_mass)):
        actual = float(body.root_physx_view.get_masses().reshape(-1)[0])
        assert abs(actual - want) < 1e-3, f"{name} mass readback {actual:.4f} != {want:.4f}"
        print(f"[solve] mass readback {name}: {actual:.3f} kg", flush=True)

    assert n_in_bowl() == n_pres(), "all present berries must start inside the bowl"
    assert bool(scene.bowl_clear()[0]), "bowl must start clear of the plate"
    assert not bool(scene.success()[0]), "reset state must not satisfy the goal"
    assert sc() <= 0.005, "score must start at ~0"
    phase_score("reset")  # 0.000

    origin = scene.env_origins[0]
    home = (scene.bowl.data.root_pos_w[0] - origin).clone()  # bowl rest pose (env frame)
    plate_c = (scene.plate.data.root_pos_w[0] - origin).clone()
    plate_top = float(scene.plate_top_z()[0] - origin[2])

    # pour geometry: u = horizontal unit bowl->plate, a = tilt axis (z x u)
    u = (plate_c[:2] - home[:2])
    u = u / u.norm()
    ux, uy = float(u[0]), float(u[1])
    ax, ay = -uy, ux  # a = z x u
    R = c.bowl_outer_r
    hw = c.bowl_wall_h / 2

    def pour_pose(theta: float, shake: float = 0.0):
        """Root pose holding the bowl tilted by theta about axis a, its pouring lip at
        L = plate_center - 0.02*u, lip height easing from 0.14 down to 0.065 above the
        plate top as the tilt passes 60 deg (body stays clear of the plate/berries)."""
        td = math.degrees(theta)
        h_lip = 0.14 - 0.075 * min(max((td - 60.0) / 50.0, 0.0), 1.0)
        lx = plate_c[0] - 0.02 * ux + shake * ax
        ly = plate_c[1] - 0.02 * uy + shake * ay
        lz = plate_top + h_lip
        co, si = math.cos(theta), math.sin(theta)
        off_u = R * co + hw * si
        off_z = hw * co - R * si
        pos = (float(origin[0]) + lx - off_u * ux,
               float(origin[1]) + ly - off_u * uy,
               float(origin[2]) + lz - off_z)
        h = theta / 2
        quat = (math.cos(h), math.sin(h) * ax, math.sin(h) * ay, 0.0)
        return pos, quat

    def yaw_of(quat_t) -> float:
        w, x, y, z = (float(v) for v in quat_t)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def pour_cycle(tag: str) -> None:
        """One full held-bowl cycle: lift, carry over the plate, tilt-pour with shake,
        un-tilt, carry home, lower, release."""
        b0 = (scene.bowl.data.root_pos_w[0] - origin).clone()
        yaw0 = yaw_of(scene.bowl.data.root_quat_w[0])
        # LIFT: straight up to carry height, easing yaw to 0
        for j in range(120):
            s = (j + 1) / 120
            yw = yaw0 * (1 - s)
            hold_bowl((float(origin[0] + b0[0]), float(origin[1] + b0[1]),
                       float(origin[2]) + float(b0[2]) + s * (0.20 - float(b0[2]))),
                      (math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)))
        if tag == "cycle1":
            report("lifted")
            phase_score("lift")  # 0.100
        # CARRY: to the pre-pour pose (theta = 0) over the plate
        p1 = (float(origin[0] + b0[0]), float(origin[1] + b0[1]), float(origin[2]) + 0.20)
        p2, _q = pour_pose(0.0)
        for j in range(120):
            s = (j + 1) / 120
            hold_bowl(tuple(p1[k] + s * (p2[k] - p1[k]) for k in range(3)),
                      (1.0, 0.0, 0.0, 0.0))
        # POUR: quasi-static tilt to THETA_MAX
        for j in range(300):
            s = (j + 1) / 300
            pos, quat = pour_pose(THETA_MAX * s)
            hold_bowl(pos, quat)
        # SHAKE: hold max tilt, small lateral shake along the tilt axis for stragglers
        for j in range(480):
            t = j * env.dt
            pos, quat = pour_pose(THETA_MAX, shake=0.006 * math.sin(2 * math.pi * 3.0 * t))
            hold_bowl(pos, quat)
            if j % 20 == 19 and n_in_bowl() == 0:
                break
        for _ in range(30):  # let the last berry land while still holding
            pos, quat = pour_pose(THETA_MAX)
            hold_bowl(pos, quat)
        report("poured")
        # UNTILT
        for j in range(150):
            s = 1.0 - (j + 1) / 150
            pos, quat = pour_pose(THETA_MAX * s)
            hold_bowl(pos, quat)
        # HOME: back over the bowl's spot, then lower and release
        p3, _q = pour_pose(0.0)
        p4 = (float(origin[0] + home[0]), float(origin[1] + home[1]), float(origin[2]) + 0.15)
        for j in range(90):
            s = (j + 1) / 90
            hold_bowl(tuple(p3[k] + s * (p4[k] - p3[k]) for k in range(3)),
                      (1.0, 0.0, 0.0, 0.0))
        z_rel = float(origin[2]) + c.bowl_rest_z + 0.003
        for j in range(60):
            s = (j + 1) / 60
            hold_bowl((p4[0], p4[1], p4[2] + s * (z_rel - p4[2])), (1.0, 0.0, 0.0, 0.0))
        step(60)  # release: free fall the last mm and settle
        settle_until(lambda: bool(scene.settled()[0]), max_steps=600)

    # ================= pour cycles (with stray-retry) ==========================================
    max_cycles = 3
    for cyc in range(max_cycles):
        pour_cycle("cycle1" if cyc == 0 else f"cycle{cyc + 1}")
        report(f"cycle{cyc + 1}")
        if bool(scene.all_delivered()[0]):
            break
        if cyc == max_cycles - 1:
            print("[solve] FAILED: berries still missing after all pour cycles", flush=True)
            verdict(False)
        # strays (on the floor / on the rim): transport back INTO the bowl, un-scored
        stray = (scene.present & ~scene.counted() & ~scene.in_bowl())[0]
        bowl_p = scene.bowl.data.root_pos_w[0]
        drop = 0
        for i in range(c.berry_max):
            if bool(stray[i]):
                teleport_berry(i, (float(bowl_p[0]), float(bowl_p[1]),
                                   float(bowl_p[2]) + hw + c.berry_r + 0.02 + 0.03 * drop))
                step(2)
                assert not bool(scene.counted()[0, i]), "stray release must be un-scored"
                drop += 1
                step(30)
        if drop:
            print(f"[solve] retry: {drop} stray berr{'y' if drop == 1 else 'ies'} "
                  f"returned to the bowl for re-pour", flush=True)
            settle_until(lambda: bool(scene.settled()[0]), max_steps=300)

    phase_score("poured")  # 0.10 + 0.55 * frac (frac = 1 when all delivered)

    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("set-down")
    if not ok:
        print("[solve] FAILED: goal state not reached after set-down", flush=True)
        verdict(False)
    phase_score("success")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
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
