"""Teleport solution for BurrowRamScene (sim_gen task `lift_peg_upright_i262`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY:
  - the rod is teleported once from its scatter spot to a rest pose on the ground,
    aligned with the bore, tip ~30 mm outside the entry mouth (exactly what a
    pick-and-carry delivers), and
  - the freed cube is teleported once from where the ram left it (at rest on the
    open floor) to 4 mm above the pedestal top, and DROPPED.

The load-bearing interaction — moving the captive cube — happens entirely through
contact dynamics: a per-step external force on the ROD (velocity servo on a
finite-difference rate; the velocity readback is phantom under external wrenches)
slides it into the bore; the rod's tip pushes the cube ahead of it down the tunnel
until the cube exits the far mouth and is fully clear of the footprint. The force is
then cut and everything settles before the cube is picked up. The cube is NEVER
teleported while captive, and no force is ever applied to the cube itself.

Entry mouth choice: the mouth farther from the cube's depth offset (shorter stroke,
cube exits the nearer mouth). Stall handling: if neither rod nor cube progresses,
the force cap escalates (the extraction-jam pattern); a frame-mode probe flips the
force encoding if the rod ever moves against the command (the stale-wrench-frame
guard).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
stage credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.lift_peg_upright_i262.solve --headless [--seed N]
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
    import scene as scene_mod

_qapply = scene_mod._qapply
_qinv = scene_mod._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

_DT = 1.0 / 120.0
_V_DES = 0.12  # ram speed (m/s): quasi-static enough that nothing vaults
_K_V = 2.0  # velocity-servo gain (K*dt/m = 2*0.0083/0.06 ~ 0.28 << 1: stable)
_F_BIAS0 = 0.8  # friction feedforward (rod+cube sliding resistance ~0.6-0.9 N)
_F_MAX0 = 3.0  # starting force cap
_F_MAX_LIM = 10.0
_V_GUARD = 0.30  # hard speed guard: cut the push if the FD rate ever exceeds this


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.burrow_ram")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def burrow_x_w() -> torch.Tensor:
        """(3,) the bore axis (burrow local +x) in world."""
        q = scene.burrow.data.root_quat_w
        return _qapply(q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]

    def cube_lx() -> float:
        return float(scene.cube_local()[0, 0])

    def rod_lx() -> float:
        return float(scene._burrow_local(scene.rod.data.root_pos_w)[0, 0])

    def clear_wrench() -> None:
        scene.rod.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        cl = scene.cube_local()[0]
        lv = float(scene.cube.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:14s} | cube_local=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) captive={bool(scene.cube_captive()[0])} "
              f"freed={bool(scene.cube_freed()[0])} placed={bool(scene.cube_placed()[0])} "
              f"cube_lin={lv:.3f} | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline ----------------------------------
    step(240)
    bp = (scene.burrow.data.root_pos_w - scene.env_origins)[0]
    q = scene.burrow.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
    pp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
    rp = (scene.rod.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): burrow=({float(bp[0]):+.3f},"
          f"{float(bp[1]):+.3f}) yaw={yaw:+.1f} deg | cube_lx={cube_lx():+.4f} | "
          f"pedestal=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) | "
          f"rod=({float(rp[0]):+.3f},{float(rp[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene.cube_captive()[0]), "cube must start captive in the bore"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s_prev = print_score("P0 reset+settle")
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: ram the cube out of the bore -----------------------------
    x0 = cube_lx()
    push_dir = -1.0 if x0 < 0.0 else 1.0  # push toward the mouth the cube is nearer to
    target_lx = push_dir * (c.freed_x + 0.018)  # stop with ~18 mm clearance past freed
    cube_ly = float(scene.cube_local()[0, 1])

    # TRANSPORT ONLY: set the rod on the ground, aligned with the bore, tip ~30 mm
    # outside the ENTRY mouth (the side opposite the push direction), zero velocity.
    tip_lx = -push_dir * (c.tun_l / 2 + 0.030)
    ctr_lx = tip_lx - push_dir * c.rod_l / 2
    bq = scene.burrow.data.root_quat_w
    local = torch.tensor([ctr_lx, cube_ly, c.rod_w / 2 + 0.002], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.burrow.data.root_pos_w + _qapply(bq, local)
    st[:, 3:7] = bq  # rod long axis along the bore
    scene.rod.write_root_state_to_sim(st, all_ids)
    step(30)  # drop the 2 mm, settle
    print(f"[solve] rod staged: push_dir={push_dir:+.0f} rod_lx={rod_lx():+.3f} "
          f"cube_lx={cube_lx():+.4f} target_lx={target_lx:+.3f}", flush=True)

    mode = 0  # 0: world-frame force; 1: body-frame pre-encode (probed)
    f_max = _F_MAX0
    f_bias = _F_BIAS0  # friction feedforward — the P-term alone tops out at K*V_DES=0.24 N
    rod_prev = rod_lx()
    anchor_i, anchor_rod, anchor_cube = 0, rod_prev, cube_lx()
    freed_now = False
    for i in range(2400):
        xc = cube_lx()
        if push_dir * xc >= push_dir * target_lx:
            freed_now = True
            break
        xr = rod_lx()
        v_fd = (xr - rod_prev) / _DT * push_dir
        rod_prev = xr
        if v_fd > _V_GUARD:  # hard guard: never let an escalated bias run the rod away
            f_mag = 0.0
        else:
            f_mag = max(0.0, min(f_max, f_bias + _K_V * (_V_DES - v_fd)))
        f_world = torch.zeros(n, 3, device=device)
        f_world[0] = burrow_x_w() * (push_dir * f_mag)
        if mode == 1:
            f_world = _qapply(_qinv(scene.rod.data.root_quat_w), f_world)
        scene.rod.set_external_force_and_torque(f_world.view(n, 1, 3), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i - anchor_i >= 90:  # progress audit every 0.75 s
            rod_prog = (rod_lx() - anchor_rod) * push_dir
            cube_prog = (cube_lx() - anchor_cube) * push_dir
            if rod_prog < -0.025:
                mode = 1 - mode
                print(f"[solve] ram: rod moving against command (prog {rod_prog:+.4f});"
                      f" force-frame mode -> {mode}", flush=True)
            elif rod_prog < 0.002 and cube_prog < 0.002:
                f_max = min(f_max * 1.6, _F_MAX_LIM)
                f_bias = min(f_bias * 1.6, f_max)
                print(f"[solve] ram stalled (rod {rod_prog:+.4f}, cube {cube_prog:+.4f})"
                      f" — bias -> {f_bias:.2f} N, cap -> {f_max:.1f} N", flush=True)
            elif rod_prog > 0.020 and f_bias > _F_BIAS0:
                f_bias = _F_BIAS0  # moving again: drop back to the gentle push
                print(f"[solve] ram unstuck (rod {rod_prog:+.4f}) — bias reset to "
                      f"{f_bias:.2f} N", flush=True)
            anchor_i, anchor_rod, anchor_cube = i, rod_lx(), cube_lx()
    clear_wrench()
    step(180)  # 1.5 s hands-off settle
    report("P1-rammed")
    print(f"[solve] ram finished: cube_lx={cube_lx():+.4f} (freed_x {c.freed_x:.3f}), "
          f"rod_lx={rod_lx():+.3f}, loop_broke={freed_now}", flush=True)
    assert bool(scene.cube_freed()[0]), \
        f"cube not freed after the ram (cube_lx={cube_lx():+.4f})"
    assert bool(scene.cube_settled()[0]), "cube did not settle after the ram"
    s_now = print_score("P1 cube rammed out of the bore, clear of the footprint")
    assert s_now >= s_prev - 1e-6, "score decreased across the ram"
    assert s_now >= 0.55, f"freed cube should score ~0.60, got {s_now}"
    s_prev = s_now

    # ---------------- phase 2: place the freed cube on the pedestal ---------------------
    placed = False
    for attempt in range(3):
        # TRANSPORT ONLY: the cube is a free object at rest on the open floor; carry
        # it to 4 mm above the pedestal top, zero velocity, and DROP it.
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.pedestal.data.root_pos_w.clone()
        st[:, 2] += c.ped_h / 2 + c.cube_s / 2 + 0.004
        st[:, 3:7] = scene.pedestal.data.root_quat_w
        scene.cube.write_root_state_to_sim(st, all_ids)
        step(240)  # 2 s: drop 4 mm, seat, ring out
        if bool(scene.cube_placed()[0]) and bool(scene.cube_settled()[0]):
            placed = True
            break
        print(f"[solve] cube did not seat on the pedestal (attempt {attempt}) — "
              f"re-dropping", flush=True)
    assert placed, "cube failed to seat on the pedestal top"
    report("P2-placed")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        os._exit(1)
    s_now = print_score("P2 cube resting on the pedestal top")
    assert s_now >= s_prev - 1e-6, "score decreased across the placement"
    s_prev = s_now

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                d = (scene.cube.data.root_pos_w - scene.pedestal.data.root_pos_w)[0]
                lv = float(scene.cube.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: d_ped=({float(d[0]):+.4f},"
                      f"{float(d[1]):+.4f},{float(d[2]):+.4f}) cube_lin={lv:.4f} "
                      f"placed={bool(scene.cube_placed()[0])} "
                      f"settled={bool(scene.cube_settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
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
