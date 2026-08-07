"""Teleport solution for MugRackScene (sim_gen task `track_ceramic_teapot_i22`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: per target mug, one pose write carries the mug
from the floor to a staged hover pose on its color hook's axis — handle up, handle
plane perpendicular to the hook, aperture center 15 mm OUTSIDE the retaining knob
(exactly the pose an arm reaches after picking the mug and aligning its handle; the
hook is NOT through the handle, so no threading gate is satisfied — only the honest
lift latch fires). The LOAD-BEARING interaction then goes through CONTACT DYNAMICS:
  THREAD + HANDOVER — a floating-hand PD force controller (gravity-compensating
  position hold + orientation hold) slides the mug along the hook axis so the rod and
  its knob pass through the handle aperture under real clearances, then the hold is
  RAMPED OUT over ~2 s so the mug's weight transfers onto the hook through handle-rod
  contact; forces cut, the mug settles hanging (it may lean on the rack — allowed).
The mug is never teleported onto the hook; threading and weight transfer are physical.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.track_ceramic_teapot_i22.solve --headless [--seed N]
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


# ----- scalar quaternion helpers (w, x, y, z) --------------------------------------------------
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qz(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def qx(a):
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def qy(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def orient_err(q_cur, q_des):
    """Axis * angle of the rotation taking q_cur to q_des (world frame), as floats."""
    w, x, y, z = q_cur
    qe = qmul(q_des, (w, -x, -y, -z))
    if qe[0] < 0.0:
        qe = tuple(-v for v in qe)
    s = math.sqrt(max(1e-12, 1.0 - qe[0] * qe[0]))
    ang = 2.0 * math.acos(min(1.0, qe[0]))
    if s < 1e-6:
        return (0.0, 0.0, 0.0)
    k = ang / s
    return (qe[1] * k, qe[2] * k, qe[3] * k)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_rack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bits = []
        for name, hook in (("mug_red", 0), ("mug_blue", 1)):
            p = (scene.mugs[name].data.root_pos_w - scene.env_origins)[0]
            bits.append(f"{name}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f}) "
                        f"thr={bool(scene.threaded(name, hook)[0])} "
                        f"hang={bool(scene.hanging(name, hook)[0])}")
        print(f"[solve] {tag:14s} | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(name: str) -> None:
        scene.mugs[name].set_external_force_and_torque(zero_wrench, zero_wrench,
                                                       env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    rq = scene.rack.data.root_quat_w[0]
    ryaw = 2.0 * math.atan2(float(rq[3]), float(rq[0]))
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): rack=({float(rp[0]):+.3f},"
          f"{float(rp[1]):+.3f}) yaw={math.degrees(ryaw):+.1f}deg", flush=True)
    for hook, tag in ((0, "RED"), (1, "BLUE")):
        r_w, e_w = scene.hook_seg_w(hook)
        r = (r_w - scene.env_origins)[0]
        e = (e_w - scene.env_origins)[0]
        print(f"[solve]   hook {tag}: root=({float(r[0]):+.3f},{float(r[1]):+.3f},"
              f"{float(r[2]):.3f}) end=({float(e[0]):+.3f},{float(e[1]):+.3f},"
              f"{float(e[2]):.3f})", flush=True)
    for name in c.mug_names:
        p = (scene.mugs[name].data.root_pos_w - scene.env_origins)[0]
        q = scene.mugs[name].data.root_quat_w[0]
        myaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        print(f"[solve]   {name}: pos=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) yaw={math.degrees(myaw):+.1f}deg", flush=True)
    report("reset")
    s_prev = print_score("P0 reset+settle")
    assert s_prev < 0.05, "score should start ~0"

    # Desired orientation for a mug staged on / traveling along a hook: local +y along
    # the hook axis, handle (local +x) up, i.e. q = qz(rack_yaw - pi/2) * qx(tilt) * qy(-pi/2).
    tilt = math.radians(c.hook_tilt_deg)
    q_des = qmul(qz(ryaw - math.pi / 2), qmul(qx(tilt), qy(-math.pi / 2)))
    qd_t = torch.tensor(q_des, device=device)
    ap_local = torch.tensor([c.ap_center_x, 0.0, 0.0], device=device)
    ap_off = quat_apply(qd_t.unsqueeze(0), ap_local.unsqueeze(0))[0]  # world offset

    d_hat = scene._hook_dir_w()[0]
    m_mug, g = c.mug_mass, 9.81
    kp, kd = 140.0, 24.0
    kr, kw = 0.04, 0.010

    def hold_step(name: str, a_target: torch.Tensor, alpha: float) -> None:
        """One controller step: PD position hold of the APERTURE CENTER at `a_target`
        plus orientation hold at q_des, everything scaled by `alpha` (handover ramp)."""
        mug = scene.mugs[name]
        p = mug.data.root_pos_w[0]
        v = mug.data.root_lin_vel_w[0]
        w_vel = mug.data.root_ang_vel_w[0]
        q = mug.data.root_quat_w[0]
        p_des = a_target - ap_off
        f = m_mug * (kp * (p_des - p) - kd * v)
        f[2] += m_mug * g
        f = (alpha * f).clamp(-4.0, 4.0)
        e = orient_err((float(q[0]), float(q[1]), float(q[2]), float(q[3])), q_des)
        tq = alpha * (kr * torch.tensor(e, device=device) - kw * w_vel)
        tq = tq.clamp(-0.05, 0.05)
        mug.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)

    def hang_mug(name: str, hook: int, tag: str) -> None:
        nonlocal s_prev
        root_w = scene.hook_root_w(hook)[0]
        # ---- TRANSPORT (teleport, free space only): aperture center 15 mm OUTSIDE the
        # knob on the hook axis, handle up, hook axis normal to the handle plane. The
        # hook is not through the handle; nothing but the lift latch can fire.
        a0 = root_w + d_hat * (c.hook_seg_len + 0.015)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = a0 - ap_off
        st[:, 3:7] = qd_t
        scene.mugs[name].write_root_state_to_sim(st, all_ids)
        for _ in range(30):  # stabilize the hold at the staged pose
            hold_step(name, a0, 1.0)
        report(f"{tag} staged")
        s = print_score(f"P1 {tag}: transport to hover outside the knob")
        assert s >= s_prev - 1e-6, "score decreased across transport"
        assert not bool(scene.threaded(name, hook)[0]), "staging must NOT thread the hook"
        s_prev = s

        # ---- THREAD (contact dynamics): slide the aperture center along the hook axis
        # from outside the knob to near the root; rod + knob pass through the handle.
        a1 = root_w + d_hat * 0.035
        travel = 330
        for i in range(travel):
            frac = min(1.0, i / 300.0)
            hold_step(name, a0 + (a1 - a0) * frac, 1.0)
        report(f"{tag} threaded")
        if not bool(scene.threaded(name, hook)[0]):
            report(f"{tag} FAIL-thread")
            print("SIM_GEN_SOLVE: FAIL (travel did not thread the hook)", flush=True)
            threading.Timer(10.0, lambda: os._exit(1)).start()
            os._exit(1)

        # ---- HANDOVER (contact dynamics): ramp the hold out over ~2 s so the mug's
        # weight transfers to the hook through handle-rod contact, then settle.
        for i in range(240):
            hold_step(name, a1, 1.0 - (i + 1) / 240.0)
        clear_wrench(name)
        step(480)  # 4 s free settle: the mug finds its hanging equilibrium
        report(f"{tag} hung")
        if not bool(scene.hanging(name, hook)[0]):
            report(f"{tag} FAIL-hang")
            print("SIM_GEN_SOLVE: FAIL (mug not hanging after handover)", flush=True)
            threading.Timer(10.0, lambda: os._exit(1)).start()
            os._exit(1)
        s = print_score(f"P2 {tag}: thread + handover + settle")
        assert s >= s_prev - 1e-6, "score decreased across thread/handover"
        s_prev = s

    # ---------------- phases 1+2: RED mug onto the RED hook --------------------------------
    hang_mug("mug_red", 0, "RED")
    # ---------------- phases 1+2 again: BLUE mug onto the BLUE hook ------------------------
    hang_mug("mug_blue", 1, "BLUE")

    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after both mugs hung)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s_prev - 1e-6
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
