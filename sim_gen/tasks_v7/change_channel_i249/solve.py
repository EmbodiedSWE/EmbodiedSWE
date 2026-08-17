"""Teleport solution for ChannelConsoleScene (sim_gen task `change_channel_i249`) —
the task's legitimacy certificate.

Teleports handle TRANSPORT ONLY (carrying the already-extracted pin from above the
track to the ground depot). Every load-bearing interaction is contact dynamics:

  1. PIN EXTRACTION (dynamics): a vertical velocity-servo force lifts the lock pin
     out of its 36 mm socket, up through the floor hole and the lip slot — a guided
     contact extraction the whole way (the stand-in for the Franka gripping the pin
     head and pulling straight up). Only once the shaft bottom has CLEARED the lip
     slot (readback) is the free pin teleported to the ground depot: pure transport
     across free space, after the interaction is complete.
  2. SLIDE (dynamics): a horizontal velocity-servo force pushes the captive slider
     along the track — across the now-empty interlock column, over the floor-hole
     seam — into the color-matched target band, then releases; friction and the
     channel finish the job. The success state is never spawned.

Servo notes: `enable_external_forces_every_iteration` is set in the scene; gains
respect the one-substep wrench delay (pin: kv*dt/m = 3/(120*0.10) = 0.25 << 1;
slider: 12/(120*0.15) = 0.67 < 1). Stall authority comes from a stick-slip
feed-forward that BUILDS while stuck and decays to a small bias once moving (the
servo's static term kv*v_des alone is below breakaway friction, and the wrench-delay
bound forbids raising the gain instead). All targets are read from the scene per
episode (console yaw/xy, start side, target band) through canonical-frame readbacks,
so one script serves every seed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still
holds.

Run (forge): python -u -m simgen_tasks.change_channel_i249.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.channel_console")().build(num_envs=args.num_envs,
                                                     device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_world: torch.Tensor) -> None:
        """World-frame force encoded in the body's CURRENT link frame (the house
        wrench convention), re-computed by the caller every step."""
        body.set_external_force_and_torque(
            quat_apply_inverse(body.data.root_link_quat_w, f_world).unsqueeze(1),
            zero_rows, env_ids=all_ids)

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def report(tag: str) -> None:
        b = scene.block_canon()[0]
        pr, pb = scene.pin_canon()
        print(f"[solve] {tag:12s} | block=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) pin_bot_z={float(pb[0, 2]):.3f}"
              f" x_target={float(scene.x_target[0]):+.3f}"
              f" latches=({float(scene.pin_clear[0]):.0f},"
              f"{float(scene.crossed[0]):.0f},{float(scene.appr_max[0]):.2f})"
              f" in_band={bool(scene.in_band()[0])}"
              f" in_track={bool(scene.in_track()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    b0 = scene.block_canon()[0]
    m_blk = float(scene.block.root_physx_view.get_masses()[0])
    m_pin = float(scene.pin.root_physx_view.get_masses()[0])
    print(f"[solve] layout readback (seed {args.seed}): side="
          f"{float(scene.side[0]):+.0f} target_idx={int(scene.target_idx[0])} "
          f"({c.chan_names[int(scene.target_idx[0])]}) "
          f"x_target={float(scene.x_target[0]):+.3f} "
          f"block_x0={float(b0[0]):+.3f} masses(block,pin)=({m_blk:.3f},{m_pin:.3f})",
          flush=True)
    assert abs(m_blk - c.block_mass) < 0.02 and abs(m_pin - c.pin_mass) < 0.02, \
        "authored masses missing (custom-spawner schema trap)"
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: pin extraction (contact dynamics) ---------------------------
    # Vertical velocity servo on the pin: F = m*g*ez + kv*(v_des*ez - v), capped.
    g = 9.81
    kv, v_des, cap = 3.0, 0.10, 4.0
    lifted = False
    last_z, last_check = -1.0, 0
    for i in range(2400):
        _pr, pb = scene.pin_canon()
        bot_z = float(pb[0, 2])
        if bot_z > scene.cfg.lip_top + 0.020:  # shaft fully clear of the slot
            lifted = True
            print(f"[solve] pin clear of the slot @step {i} (bot_z={bot_z:.3f})",
                  flush=True)
            break
        v = scene.pin.data.root_lin_vel_w
        f = -kv * v
        f[:, 2] = m_pin * g + kv * (v_des - v[:, 2])
        f_norm = f.norm(dim=-1, keepdim=True)
        f = f * (f_norm.clamp(max=cap) / f_norm.clamp_min(1e-9))
        wrench(scene.pin, f)
        env.step(no_action)
        if i - last_check >= 240:  # 2 s without progress -> escalate GAIN
            if bot_z - last_z < 0.01:
                kv = min(kv * 1.6, 10.0)
                v_des = min(v_des * 1.3, 0.25)
                print(f"[solve] extraction stall (bot_z={bot_z:.3f}), kv->{kv:.1f} "
                      f"v_des->{v_des:.2f}", flush=True)
            last_z, last_check = bot_z, i
    clear(scene.pin)
    assert lifted, "pin never cleared the lip slot"
    assert float(scene.pin_clear[0]) == 1.0, "pin_clear latch did not fire"

    # TRANSPORT ONLY: the pin is free above the track — carry it to the ground depot.
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = 0.9
    st[:, 1] = -0.9
    st[:, 2] = 0.10
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.pin.write_root_state_to_sim(st, all_ids)
    step(90)  # let it land and settle
    report("pin-out")
    s1 = print_score("P1 pin extracted through the socket + slot, parked")
    assert s1 >= s0 - 1e-6 and s1 >= 0.20 - 1e-6, "pin-clear credit missing"

    # ---------------- phase 2: slide the selector (contact dynamics) -----------------------
    # Stick-slip breaker: the servo's static authority (kv*v_des) is below breakaway
    # friction and below the floor-seam edge catch at the interlock column, and the
    # wrench-delay bound caps kv (kv*dt/m < 1 -> kv <~ 18 at m=0.15). So a feed-
    # forward push BUILDS (+0.05 N/substep) while the block is stuck and DECAYS to a
    # small bias once it moves — force ramps are smooth, the servo gain stays stable,
    # and the traverse remains regulated near v_des.
    kv2, v_des2, cap2 = 12.0, 0.06, 6.0
    ff, ff_bias, ff_max = 0.0, 1.0, 6.0
    reached = False
    last_d, last_check = float("inf"), 0
    for i in range(4800):
        b = scene.block_canon()
        d = float((b[0, 0] - scene.x_target[0]).abs())
        if d <= 0.004:
            reached = True
            print(f"[solve] band center reached @step {i}", flush=True)
            break
        dirx = torch.sign(scene.x_target - b[:, 0])
        v_canon = quat_apply_inverse(scene.console.data.root_quat_w,
                                     scene.block.data.root_lin_vel_w)
        moving = float((v_canon[:, 0] * dirx)[0]) > 0.02
        ff = max(ff - 0.10, ff_bias) if moving else min(ff + 0.05, ff_max)
        f_canon = torch.zeros(n, 3, device=device)
        f_canon[:, 0] = ff * dirx + kv2 * (v_des2 * dirx - v_canon[:, 0])
        f_world = quat_apply(scene.console.data.root_quat_w, f_canon)
        f_norm = f_world.norm(dim=-1, keepdim=True)
        f_world = f_world * (f_norm.clamp(max=cap2) / f_norm.clamp_min(1e-9))
        wrench(scene.block, f_world)
        env.step(no_action)
        if i - last_check >= 240:  # progress telemetry every 2 s
            if last_d - d < 0.005:
                print(f"[solve] slide stall (d={d:.3f}, ff={ff:.2f} N)", flush=True)
            last_d, last_check = d, i
    clear(scene.block)
    assert reached, "slider never reached the target band center"
    for _ in range(8):  # up to 2 s hands-off settling inside the band
        if bool(scene.success()[0]):
            break
        step(30)
    report("slid")
    s2 = print_score("P2 selector slid into the color-matched band, released")
    assert s2 >= s1 - 1e-6, "score decreased across the slide"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after extract+slide)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                b = scene.block_canon()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"in_band={bool(scene.in_band()[0])} "
                      f"in_track={bool(scene.in_track()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"block=({float(b[0]):+.3f},{float(b[1]):+.3f},"
                      f"{float(b[2]):.3f}) "
                      f"v={float(scene.block.data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s3 = print_score("P3 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
