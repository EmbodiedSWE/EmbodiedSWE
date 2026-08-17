"""Teleport solution for RubbishChuteScene (sim_gen task `put_rubbish_in_bin_i147`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the ONLY pose write on the paper ball): one root-state write
   carries the white paper ball from its ground spawn across open air to a release
   pose above the chute's open trough — 118 mm above the pitched floor, OUTSIDE the
   trough's judged z band (0..100 mm), so the freshly-teleported state earns no load
   credit. Both endpoints are in free space.
2. LOAD (contact dynamics): gravity drops the ball into the trough; it lands on the
   16-degree floor, rolls downhill and comes to rest against the CLOSED guillotine
   gate. The `loaded` latch is earned by this physical rest, not by the teleport.
3. RELEASE (contact dynamics, no teleport ever touches the gate): the gate is captive
   in its guide slots and recloses under gravity — operating it IS the mechanism the
   task is about. It is driven by a vertical external force at its CoM: gravity
   feedforward (m*g from the runtime mass READBACK) + PD on the measured slide-lift
   (Kp*dt/m = 0.42 << 1 for the one-substep wrench delay). A purely VERTICAL command
   is invariant under every force-frame encoding seen on the forge pods (world frame:
   exact gravity cancellation; body/rotation-drag frames: z maps through qz(yaw)*qy(theta)
   onto the slide axis itself), so no encoding probe is needed — a progress probe
   still guards it. The gate is HELD up while the ball rolls under it, down the
   covered tunnel, through the bin window (`passed` latch earned mid-flight).
4. RESTORE (contact dynamics): the force is cleared once the ball is irrecoverably
   downhill of the gate; the gate free-falls shut onto its seat — the gravity-return
   behavior the task advertises. success() first turns True only here, judged on the
   settled physical state (ball inside the sealed bin, tomato out, gate reseated,
   everything at rest). The success state is never spawned.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_rubbish_in_bin_i147.solve --headless [--seed N]
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
    env = ENVS.get("simgen.rubbish_chute")().build(num_envs=args.num_envs, device=device)
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

    def struct_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.structure.data.root_pos_w - scene.env_origins)[0]
        q = scene.structure.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        r = scene._local(scene.rubbish)[0]
        print(f"[solve] {tag:12s} | rubbish_local=({float(r[0]):+.3f},{float(r[1]):+.3f},"
              f"{float(r[2]):.3f}) gate_lift={float(scene.gate_lift()[0]):+.4f} "
              f"loaded={bool(scene._loaded[0])} gate_max={float(scene._gate_max[0]):.3f} "
              f"passed={bool(scene._passed[0])} shut={bool(scene.gate_shut()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    sp, yaw = struct_pose()
    r0 = (scene.rubbish.data.root_pos_w - scene.env_origins)[0]
    t0 = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): structure=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"rubbish=({float(r0[0]):+.3f},{float(r0[1]):+.3f}) "
          f"tomato=({float(t0[0]):+.3f},{float(t0[1]):+.3f}) "
          f"gate_lift={float(scene.gate_lift()[0]):+.4f}", flush=True)
    # Custom spawners ignore cfg schemas — prove the authored gate mass really landed.
    m_gate = float(scene.gate.root_physx_view.get_masses().sum())
    print(f"[solve] gate mass readback: {m_gate:.4f} kg", flush=True)
    assert abs(m_gate - c.gate_mass) < 0.05, "gate MassAPI mass did not land"
    assert float(scene.gate_lift()[0]) < 0.005, "gate must start seated shut"
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free air only) --------------------
    # One pose write carries the paper ball to 118 mm above the trough floor at local
    # x=0.10 — geometrically OUTSIDE the trough's judged z band (0..100 mm), asserted
    # against the scene's own predicate, so the teleport itself earns nothing.
    zf = c.chute_h0 - 0.10 * math.tan(math.radians(c.theta_deg))
    drop_local = torch.tensor([[0.10, 0.0, zf + 0.118]], device=device)
    assert not bool(scene._in_trough(drop_local)[0]), \
        "release pose must be outside the judged trough band"
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = sp[0] + math.cos(yaw) * 0.10
    st[:, 1] = sp[1] + math.sin(yaw) * 0.10
    st[:, 2] = zf + 0.118
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.rubbish.write_root_state_to_sim(st, all_ids)
    report("transported")
    s1 = print_score("P1 transport to release pose above the open trough")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: LOAD through gravity + contact --------------------------------
    # The ball falls into the trough, rolls down the 16-degree floor and beds against
    # the CLOSED gate. The `loaded` latch is earned here, by physics.
    step(240)  # 2 s: drop, roll downhill, settle against the gate
    report("loaded")
    assert bool(scene._loaded[0]), "ball did not come to rest in the trough"
    r_loc = scene._local(scene.rubbish)[0]
    assert float(r_loc[0]) > 0.15, "ball did not roll down to the gate"
    assert float(scene.rubbish.data.root_lin_vel_w[0].norm()) < 0.10, \
        "ball not settled against the gate"
    s2 = print_score("P2 gravity load into the trough")
    assert s2 >= s1 - 1e-6, "score decreased across load"

    # ---------------- phase 3: RELEASE — lift and HOLD the gate through contact dynamics ----
    # Vertical force at the gate CoM: gravity feedforward + PD on measured slide-lift.
    # Held until the ball is irrecoverably downhill of the gate (past the covered
    # tunnel mouth), then phase 4 releases it. No pose write ever touches the gate.
    from isaaclab.utils.math import quat_apply

    n_world = quat_apply(scene.structure.data.root_quat_w[0],
                         torch.tensor(c.n_local, device=device))
    target = c.lift_ref + 0.006  # clears the ball; well below the 74.5 mm caps
    ff = m_gate * 9.81
    kp, kd, f_max = 8.0, 2.5, 10.0  # kp*dt/m = 0.22 << 1 (one-substep wrench delay)
    wrench = torch.zeros(n, 1, 3, device=device)
    released_at = None
    lifted_logged = False
    for i in range(1500):
        lift = float(scene.gate_lift()[0])
        v_along = float((scene.gate.data.root_lin_vel_w[0] * n_world).sum())
        f = ff + kp * (target - lift) - kd * v_along
        wrench[:, 0, 2] = max(0.0, min(f, f_max))
        scene.gate.set_external_force_and_torque(wrench, zero_wrench, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)
        if i == 60:  # progress probe: the servo must actually be lifting the gate
            assert float(scene.gate_lift()[0]) > 0.005, \
                "gate servo made no progress — force encoding or friction problem"
        if not lifted_logged and float(scene._gate_max[0]) > 0.99:
            lifted_logged = True
            report("gate-held")
            print_score("P3a gate lifted to full passing height (held)")
        r_loc = scene._local(scene.rubbish)[0]
        if bool(scene._passed[0]) and float(r_loc[0]) > c.gate_x + 0.12:
            released_at = i
            break
    assert released_at is not None, "ball never passed under the held gate"
    report("passed")
    assert float(scene._gate_max[0]) > 0.99, "gate never reached full lift"
    assert bool(scene._passed[0]), "passed latch not earned"
    s3 = print_score("P3 ball passed under the held gate into the tunnel")
    assert s3 >= s2 - 1e-6, "score decreased across the pass"

    # ---------------- phase 4: RESTORE — release; gravity recloses the gate ------------------
    clear_force()
    step(240)  # 2 s: ball finishes the tunnel + window flight and beds down in the
    # bin; the gate free-falls ~66 mm back onto its seat
    for _ in range(6):  # extra settle if anything is still rocking
        if bool(scene.success()[0]):
            break
        step(60)
    report("released")
    assert bool(scene.gate_shut()[0]), "gate did not reclose under gravity"
    s4 = print_score("P4 gate released — gravity reseats it, ball settled in the bin")
    assert s4 >= s3 - 1e-6, "score decreased across release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 — die fast and loudly, never hang in teardown
        print(f"[solve] EXCEPTION: {e!r}", flush=True)
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
