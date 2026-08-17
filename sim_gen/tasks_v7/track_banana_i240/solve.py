"""Teleport solution for BananaKilnScene (sim_gen task `track_banana_i240`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, ONCE): one pose write lays the banana flat into the OPEN-TOP
   feed channel, broadside in front of the pusher head — exactly the pick-and-place
   a real arm performs over open, reachable floor. This earns only the "fed" stage
   (0.25); the tunnel and the chamber cannot be reached this way (they are roofed —
   the whole point of the task), and no teleport ever touches them.
2. RAM FEED (contact): a velocity-servoed planar force (|F| <= 12 N) on the rammer
   drives its head down the channel along the kiln's live +x axis; the head
   BULLDOZES the banana through the low tunnel and out past the sill onto the
   chamber floor. The banana is never forced directly — everything past the mouth
   happens through pusher-face contact, floor friction and the tunnel walls. The
   force frame is PROBED from measured progress and the encoding toggled if the
   rammer consistently loses ground (this stack's wrench frame is pod-dependent).
3. WITHDRAW (contact): the same servo, reversed, pulls the rammer back until its
   head is clear of the tunnel (the knob-side pull a hand would do).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it persists.

Run (forge): python -u -m simgen_tasks.track_banana_i240.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.banana_kiln")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    from isaaclab.utils.math import quat_apply

    def kaxis() -> torch.Tensor:
        """World-frame unit vector of the kiln's live +x (feed) axis, xy-projected."""
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        u = quat_apply(scene.kiln.data.root_quat_w, ex)[0, 0:2]
        return u / u.norm().clamp_min(1e-6)

    def report(tag: str) -> None:
        b = scene.banana_local()[0]
        r = scene.rammer_local()[0]
        print(f"[solve] {tag:12s} | ban_local=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) ram_lx={float(r[0]):+.3f} "
              f"fed={bool(scene._fed_ever[0])} tun={bool(scene._tun_ever[0])} "
              f"cham={bool(scene._cham_ever[0])} in_ch={bool(scene.in_channel()[0])} "
              f"in_tun={bool(scene.in_tunnel()[0])} in_cham={bool(scene.inside_chamber()[0])} "
              f"retr={bool(scene.retracted()[0])} still={bool(scene._still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- rammer drive: force-frame probe + velocity servo along the kiln feed axis ------------
    mode = [0]  # 0 = raw world force (is_global=True), 1 = pre-rotated by q_ref * q_now^-1
    q_ref = [None]

    def qinv(q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[:, 1:] = -out[:, 1:]
        return out

    def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)

    def push_rammer(f_world: torch.Tensor) -> None:
        if mode[0] == 1:
            q_drag = qmul(q_ref[0], qinv(scene.rammer.data.root_quat_w))
            f_world = quat_apply(q_drag, f_world)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, :] = f_world
        scene.rammer.set_external_force_and_torque(f.contiguous(), zero3,
                                                   env_ids=all_ids, is_global=True)

    def drive_rammer(target_lx: float, tag: str, max_steps: int = 2200,
                     vmax: float = 0.12, cap: float = 12.0) -> None:
        """Velocity-servoed planar drive of the rammer to a kiln-local x depth, with
        a progress-based force-frame probe (toggle encoding if losing ground) and
        gain escalation on friction stall (raise GAIN, not the cap)."""
        q_ref[0] = scene.rammer.data.root_quat_w.clone()
        window_gain, window_abs, window_n = 0.0, 0.0, 0
        kp = 60.0
        last = float(scene.rammer_local()[0, 0])
        for _ in range(max_steps):
            lx = float(scene.rammer_local()[0, 0])
            e = target_lx - lx
            v = scene.rammer.data.root_lin_vel_w[0, 0:2]
            if abs(e) < 0.008 and float(v.norm()) < 0.04:
                break
            u = kaxis()
            sgn = 1.0 if e > 0 else -1.0
            v_des = u * (sgn * min(vmax, 0.6 * abs(e) + 0.02))
            f_xy = (kp * (v_des - v)).clamp(-cap, cap)
            fw = torch.zeros(n, 3, device=device)
            fw[0, 0:2] = f_xy
            push_rammer(fw)
            env.step(no_action)
            lx2 = float(scene.rammer_local()[0, 0])
            window_gain += (lx2 - last) * sgn
            window_abs += abs(lx2 - last)
            last = lx2
            window_n += 1
            if window_n >= 60:
                if window_gain < -0.005:
                    mode[0] ^= 1
                    print(f"[solve] drive[{tag}]: frame probe TOGGLED mode -> {mode[0]}",
                          flush=True)
                elif window_abs < 0.003 and abs(e) > 0.008:
                    kp = min(kp * 1.8, 900.0)
                    print(f"[solve] drive[{tag}]: stall at e={e * 1000:+.0f} mm, "
                          f"gain -> {kp:.0f}", flush=True)
                window_gain, window_abs, window_n = 0.0, 0.0, 0
        scene.rammer.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)
        for _ in range(120):
            env.step(no_action)
            if float(scene.rammer.data.root_lin_vel_w[0, 0:2].norm()) < 0.03:
                break
        lx = float(scene.rammer_local()[0, 0])
        print(f"[solve] drive[{tag}]: settled at ram_lx={lx:+.3f} "
              f"(target {target_lx:+.3f}, mode {mode[0]})", flush=True)

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(60)
    kx, ky = float(scene._kiln_xy[0, 0]), float(scene._kiln_xy[0, 1])
    kyaw = float(torch.rad2deg(scene._kiln_yaw[0]))
    print(f"[solve] layout readback (seed {args.seed}): kiln=({kx:+.3f},{ky:+.3f}) "
          f"yaw={kyaw:+.1f}deg", flush=True)
    report("reset")
    assert bool(scene.retracted()[0]), "rammer did not start withdrawn"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "score not ~0 at reset"

    # ---------------- phase 1: TRANSPORT the banana into the open channel (teleport) ------------
    # Lay it broadside (long axis across the channel) in front of the pusher face,
    # well short of the mouth — an open-top, hand-reachable placement.
    ram_lx0 = float(scene.rammer_local()[0, 0])
    stage_lx = max(ram_lx0 + c.head_l / 2 + 0.06, -0.17)
    lpt = torch.tensor([stage_lx, 0.0, c.ban_rest_z + 0.004], device=device).expand(n, 3)
    kq = scene.kiln.data.root_quat_w
    q90 = torch.zeros(n, 4, device=device)
    q90[:, 0] = 0.7071068
    q90[:, 3] = 0.7071068
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.kiln.data.root_pos_w + quat_apply(kq, lpt)
    st[:, 3:7] = qmul(kq, q90)
    scene.banana.write_root_state_to_sim(st, all_ids)
    step(60)
    report("fed")
    assert bool(scene.in_channel()[0]), "banana did not settle inside the feed channel"
    assert not bool(scene._tun_ever[0]) and not bool(scene._cham_ever[0]), \
        "feeding the channel must not touch the covered stages"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 transport: banana laid into the open feed channel")
    assert s1 >= 0.24, "channel stage did not latch"

    # ---------------- phase 2: RAM the banana through the tunnel into the chamber (contact) -----
    # Face reaches the sill minus a hair; the banana's broadside half-depth carries
    # its center past the sill. The head's REAR stays inside the tunnel-wall span,
    # so it remains yaw-guided and can always be pulled straight back out.
    face_need = c.sill_x - 0.005
    drive_rammer(face_need - c.head_l / 2, "ram-in")
    report("rammed")
    b = scene.banana_local()[0]
    assert float(b[0]) > c.sill_x + 0.005, \
        f"banana not past the sill (lx={float(b[0]):+.3f})"
    assert float(b[2]) < c.rest_z_max, "banana not on the chamber floor"
    assert bool(scene._tun_ever[0]) and bool(scene._cham_ever[0]), \
        "tunnel/chamber stages did not latch during the ram feed"
    s2 = print_score("P2 ram feed: banana bulldozed through the tunnel into the chamber")
    assert s2 >= 0.74, "chamber stage did not latch credit"

    # ---------------- phase 3: WITHDRAW the rammer (contact) ------------------------------------
    drive_rammer(c.retract_x - 0.09, "withdraw", vmax=0.15)
    report("withdrawn")
    assert bool(scene.retracted()[0]), "rammer head not clear of the tunnel"
    s3 = print_score("P3 rammer withdrawn clear of the tunnel")
    assert s3 >= s2 - 1e-6

    # ---------------- phase 4: hands-off persistence (>= 3.3 simulated seconds) -----------------
    for _ in range(240):
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    report("pre-persist")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (no success before persistence)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - Kit teardown hangs; die loudly and fast
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        threading.Timer(10.0, lambda: os._exit(2)).start()
        os._exit(2)
