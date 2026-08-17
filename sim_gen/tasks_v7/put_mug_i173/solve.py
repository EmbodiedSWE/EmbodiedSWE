"""Teleport solution for MugHookScene (sim_gen task `put_mug_i173`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the mug): one root-state write carries
   the mug from its upright ground spawn across open air to a HOVER pose near the red
   peg's tip: upright, handle-window plane perpendicular to the peg, the peg's axis
   already passing through the OPEN window (free space — the window clears the peg by
   ~10 mm on every side) with the peg line 27 mm BELOW the catch line. Both endpoints
   are contact-free; the reorientation is what a wrist rotation does. The frozen
   written state can NOT satisfy success(): the stillness streak resets on the pose
   jump, and the very next step the unsupported mug is falling.
2. CATCH + HANG (contact dynamics): released, the mug free-falls ~27 mm until the
   handle's top bar lands on the peg — a real dynamic catch — then pendulum-swings
   about the peg, the peg sliding along the bar into the loop's corner pocket, and
   the swing decays through contact friction. Success is only reachable through this
   settling; nothing supports the mug but the peg through its handle window.
3. SLIDE HOME (contact dynamics, no teleport): before the swing has settled, a
   horizontal external force at the mug's CoM (velocity-regulated bang-bang, runtime
   frame-encode probe, stall escalation) drags the HANGING mug along the tilted peg
   from near the tip toward the root — the handle loop sliding on the peg under load —
   then releases. The mug swings out and settles hanging deep on the hook.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_mug_i173.solve --headless [--seed N]
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
import traceback

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_hook")().build(num_envs=args.num_envs, device=device)
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

    from isaaclab.utils.math import matrix_from_quat, quat_apply

    def peg_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(root_w, tip_w, axis_unit) of the red peg, env 0, origin-relative frame
        irrelevant (differences only)."""
        root_w, tip_w = scene._peg_world("red")
        ax = tip_w[0] - root_w[0]
        return root_w[0], tip_w[0], ax / ax.norm()

    def d_root() -> float:
        """Window-centre distance from the peg ROOT, projected on the peg axis."""
        root_w, _tip, ax = peg_frame()
        return float(((scene._window_center_w()[0] - root_w) * ax).sum())

    def report(tag: str) -> None:
        z = float(scene._mug_z()[0])
        v = float(scene.mug.data.root_lin_vel_w[0].norm())
        w = float(scene.mug.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | mug_z={z:.3f} |v|={v:.3f} |w|={w:.3f} "
              f"threaded={bool(scene._threaded_now('red')[0])} d_root={d_root():.3f} "
              f"still={int(scene._still[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}"
        last_score[0] = s
        return s

    def clear_force() -> None:
        scene.mug.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    m_read = float(scene.mug.root_physx_view.get_masses().reshape(-1)[0])
    assert abs(m_read - c.mug_mass) < 0.02, \
        f"authored mug mass not applied (read {m_read})"
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    pp = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): stand=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"mug=({float(mp[0]):+.3f},{float(mp[1]):+.3f},{float(mp[2]):.3f}) "
          f"pad=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) mug_mass={m_read:.3f}",
          flush=True)
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    print_score("P0 reset+settle")
    R_ref = matrix_from_quat(scene.mug.data.root_quat_w)[0].clone()  # drag-encode ref

    # ---------------- phase 1: TRANSPORT (teleport across free air only) --------------------
    # One pose write: mug upright, local +y along the peg toward its root, positioned
    # so the peg axis crosses the OPEN handle window 20 mm from the tip, 27 mm below
    # the catch line (near the TIP: the dynamic catch itself ratchets the loop down
    # the tilted peg, so the forced slide phase must start with real distance left).
    # Everything about this pose is contact-free; the write leaves zero velocity and
    # the stillness streak invalidated (pose jump), so it can never judge as success.
    root_w, tip_w, ax = peg_frame()
    u = ax.clone()
    u[2] = 0.0
    u = u / u.norm()  # peg horizontal direction, root -> tip
    psi = math.atan2(float(u[0]), -float(u[1]))  # R(psi)·ey = -u  (local +y -> root)
    cross_w = tip_w - ax * 0.020
    qz = torch.tensor([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)], device=device)
    off = torch.tensor([c.wc[0], 0.0, -0.008], device=device)
    mug_pos = cross_w - quat_apply(qz.unsqueeze(0), off.unsqueeze(0))[0]
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = mug_pos.unsqueeze(0)
    st[:, 3:7] = qz.unsqueeze(0)
    scene.mug.write_root_state_to_sim(st, all_ids)
    assert bool(scene._threaded_now("red")[0]), \
        "hover pose must already thread the open window (free space)"
    assert not bool(scene.success()[0]), \
        "freshly teleported hover must NOT judge as success (streak guard)"
    report("hover")
    print_score("P1 transport to threaded hover off the peg tip")

    # ---------------- phase 2: CATCH through gravity + contact ------------------------------
    # Free-fall ~27 mm, top bar lands on the peg, pendulum swing begins. Run 1.5 s —
    # deliberately LESS than the settle streak needs, so the slide phase acts on a
    # still-live mug and success() first turns True only after the final release.
    step(180)
    assert bool(scene._threaded_now("red")[0]), "mug fell off the peg at the catch"
    assert float(scene._mug_z()[0]) > c.suspend_z_min, "mug not suspended after catch"
    report("caught")
    print_score("P2 gravity catch onto the peg (hanging, still swinging)")

    # ---------------- phase 3: SLIDE HOME along the peg (external force, contact) -----------
    # Drag the hanging mug along the tilted peg toward the root: horizontal
    # velocity-regulated bang-bang force at the CoM, runtime frame-encode probe
    # (memory: pod-dependent wrench rotation drag), stall escalation. The loop slides
    # on the peg under load the whole way.
    target_d = 0.058
    d_start = d_root()
    if d_start <= target_d + 0.006:
        # The catch dynamics already ratcheted the loop home along the tilted peg;
        # nothing left to slide (contact transport already happened in P2).
        print(f"[solve] slide skipped: catch already delivered the loop to "
              f"d_root={d_start:.3f} (target {target_d:.3f})", flush=True)
    elif not bool(scene.success()[0]):
        R_now = lambda: matrix_from_quat(scene.mug.data.root_quat_w)[0]  # noqa: E731
        mode = ["raw"]

        def encode(f_des: torch.Tensor) -> torch.Tensor:
            if mode[0] == "raw":
                return f_des
            return (R_ref @ R_now().T) @ f_des

        push_dir = -u  # toward the post/root
        f_mag, v_des = 1.5, 0.06
        best, last_bump, probe_at = d_start, 0, 240
        for i in range(1500):
            d = d_root()
            if d <= target_d:
                break
            v_along = float((scene.mug.data.root_lin_vel_w[0] * push_dir).sum())
            f_des = push_dir * (f_mag if v_along < v_des else 0.0)
            f_w = encode(f_des).view(1, 1, 3).expand(n, 1, 3)
            scene.mug.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            if d < best - 0.004:
                best, last_bump = d, i
            elif i - last_bump > 240:  # stalled: push harder
                f_mag = min(f_mag + 1.0, 5.0)
                last_bump = i
                print(f"[solve] slide stalled at d_root={d:.3f}, raising force to "
                      f"{f_mag:.1f} N", flush=True)
            if i == probe_at and best > d_start - 0.004 and mode[0] == "raw":
                mode[0] = "drag"  # frame-drag probe: no progress -> flip encoding
                print("[solve] slide encode probe: no progress in raw mode, "
                      "switching to drag pre-encode", flush=True)
        clear_force()
        assert bool(scene._threaded_now("red")[0]), "mug came off during the slide"
        assert d_root() < d_start - 0.010, \
            f"slide made no progress ({d_start:.3f} -> {d_root():.3f})"
    report("slid")
    print_score("P3 force-slide along the peg toward the root")

    # ---------------- phase 4: release-settle to success ------------------------------------
    ok_settle = False
    for _ in range(40):  # up to 10 s
        step(30)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    report("settled")
    if not ok_settle:
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        os._exit(1)
    print_score("P4 hands-off swing decay to settled hang")

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s5 >= 1.0 - 1e-6
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
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
