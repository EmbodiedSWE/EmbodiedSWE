"""Teleport solution for BananaLineScene (sim_gen task `track_banana_i79`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, ONCE, while EMPTY): one pose write moves the empty crate
   across open floor to a 3 mm hover under the first YELLOW banana's clamp — touching
   nothing, satisfying no gate (the crate under a clamp scores nothing). The bananas,
   levers and gantry are NEVER teleported outside reset; no teleport ever touches a
   scoring region.
2. CLAMP ACTUATION (contact/mechanism): each release is a ramped pair of opposed
   z-torques on that station's two levers through the scene's `lever_drive` buffer —
   the wrench equivalent of squeezing the tail paddles — working AGAINST the clamp
   spring until the pads part past the knob; the banana then falls by GRAVITY into
   the crate and the drive is zeroed so the spring re-closes the clamp. No force is
   ever applied to a banana.
3. CRATE DRAG (contact): a velocity-servoed planar force (|F| <= 8 N, zero z)
   slides the loaded crate along the floor — station to station, then to the depot.
   The force frame is PROBED from measured progress and the encoding toggled if the
   crate moves away (this stack's wrench frame drag is pod-dependent).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.track_banana_i79.solve --headless [--seed N]
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
    env = ENVS.get("simgen.banana_line")().build(num_envs=args.num_envs, device=device)
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        bans = [loc(b) for b in scene.bananas]
        pc = loc(scene.crate)
        ang = torch.rad2deg(scene.lever_angles()[0])
        inc = scene.in_crate()[0]
        hng = scene.hanging()[0]
        print(f"[solve] {tag:12s} | crate=({float(pc[0]):+.3f},{float(pc[1]):+.3f}) "
              + " ".join(f"b{j}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})"
                         for j, p in enumerate(bans))
              + f" ang=[{','.join(f'{float(a):+.1f}' for a in ang)}]"
              f" in_crate={[bool(v) for v in inc]} hang={[bool(v) for v in hng]}"
              f" bruised={bool(scene.bruised()[0])} deliv={bool(scene._delivered[0])}"
              f" success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- crate drag: force-frame probe + velocity servo ---------------------------------------
    from isaaclab.utils.math import quat_apply

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

    def push_crate(f_world: torch.Tensor) -> None:
        if mode[0] == 1:
            q_drag = qmul(q_ref[0], qinv(scene.crate.data.root_quat_w))
            f_world = quat_apply(q_drag, f_world)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, :] = f_world
        scene.crate.set_external_force_and_torque(f.contiguous(), zero3,
                                                  env_ids=all_ids, is_global=True)

    def drag_to(target_xy: torch.Tensor, tol: float, tag: str, max_steps: int = 1400,
                hard: bool = True) -> None:
        """Velocity-servoed planar drag of the crate to target (env-local xy), with a
        progress-based force-frame probe (toggle encoding if moving away)."""
        q_ref[0] = scene.crate.data.root_quat_w.clone()
        window_gain, window_abs, window_n = 0.0, 0.0, 0
        kp = 60.0   # escalated on friction stall (raise GAIN, not the cap: the cap
        # bounds tipping torque; the gain determines the stall force at small error)
        last = loc(scene.crate)[0:2].clone()
        for i in range(max_steps):
            p = loc(scene.crate)[0:2]
            err = target_xy - p
            d = float(err.norm())
            v = scene.crate.data.root_lin_vel_w[0, 0:2]
            if d < tol and float(v.norm()) < 0.05:
                break
            u = err / max(d, 1e-6)
            v_des = u * min(0.15, 0.6 * d + 0.03)
            f_xy = (kp * (v_des - v)).clamp(-8.0, 8.0)
            f = torch.zeros(n, 3, device=device)
            f[0, 0:2] = f_xy
            push_crate(f)
            env.step(no_action)
            # progress probe: if we are consistently losing ground at speed, the force
            # frame is dragged -> toggle the encoding; if we are simply NOT MOVING with
            # error outstanding, the servo has stalled under breakaway friction ->
            # escalate the gain so the stall force reaches the cap
            p2 = loc(scene.crate)[0:2]
            step_vec = p2 - last
            window_gain += float(torch.dot(step_vec, u))
            window_abs += float(step_vec.norm())
            last = p2.clone()
            window_n += 1
            if window_n >= 60:
                if window_gain < -0.005:
                    mode[0] ^= 1
                    print(f"[solve] drag[{tag}]: frame probe TOGGLED mode -> {mode[0]}",
                          flush=True)
                elif window_abs < 0.003 and d > tol:
                    kp = min(kp * 1.8, 900.0)
                    print(f"[solve] drag[{tag}]: stall at d={d * 1000:.0f} mm, "
                          f"gain -> {kp:.0f}", flush=True)
                window_gain, window_abs, window_n = 0.0, 0.0, 0
        scene.crate.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)
        for _ in range(90):
            env.step(no_action)
            if float(scene.crate.data.root_lin_vel_w[0, 0:2].norm()) < 0.03:
                break
        p = loc(scene.crate)[0:2]
        d = float((target_xy - p).norm())
        print(f"[solve] drag[{tag}]: settled {d * 1000:.0f} mm from target "
              f"(mode {mode[0]})", flush=True)
        if hard and not d < tol + 0.015:
            report(f"drag-fail[{tag}]")
            raise AssertionError(f"drag[{tag}] missed the target by {d:.3f} m")

    # ---- routing around the gantry base slab (gantry-local frame) --------------------------
    # Inflated slab keep-out: side edges by crate half-diagonal (~0.146) + margin; the
    # front edge only lightly (face-on brushes slide off; the wedge failure mode is the
    # slab END corner). The depot lies BEHIND the gantry, so loaded runs must round an end.
    def _g_local(p_xy):
        gy = float(scene._gantry_yaw[0])
        c_, s_ = math.cos(gy), math.sin(gy)
        dx = float(p_xy[0] - scene._gantry_xy[0, 0])
        dy = float(p_xy[1] - scene._gantry_xy[0, 1])
        return (c_ * dx + s_ * dy, -s_ * dx + c_ * dy)

    def _g_world(lx, ly):
        gy = float(scene._gantry_yaw[0])
        c_, s_ = math.cos(gy), math.sin(gy)
        return torch.tensor([float(scene._gantry_xy[0, 0]) + c_ * lx - s_ * ly,
                             float(scene._gantry_xy[0, 1]) + s_ * lx + c_ * ly],
                            device=device)

    def _blocked(p, q) -> bool:
        for t in range(49):
            a = t / 48.0
            x = p[0] + a * (q[0] - p[0])
            y = p[1] + a * (q[1] - p[1])
            if abs(x) <= 0.54 and -0.28 <= y <= 0.105:
                return True
        return False

    def route_to(target_xy: torch.Tensor, tol: float, tag: str) -> None:
        lp = _g_local(loc(scene.crate)[0:2])
        lt = _g_local(target_xy)
        if not _blocked(lp, lt):
            drag_to(target_xy, tol, tag)
            return
        corners = [(-0.58, 0.20), (0.58, 0.20), (-0.58, -0.32), (0.58, -0.32)]
        nodes = [lp, lt] + corners
        nn = len(nodes)
        adj = [[] for _ in range(nn)]
        for a in range(nn):
            for b in range(a + 1, nn):
                if not _blocked(nodes[a], nodes[b]):
                    adj[a].append(b)
                    adj[b].append(a)
        prev = {0: None}
        queue = [0]
        while queue:
            u = queue.pop(0)
            if u == 1:
                break
            for v_ in adj[u]:
                if v_ not in prev:
                    prev[v_] = u
                    queue.append(v_)
        assert 1 in prev, f"route[{tag}]: no clear path around the gantry"
        path = []
        u = 1
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        print(f"[solve] route[{tag}]: {len(path) - 2} waypoint(s)", flush=True)
        for idx in path[1:-1]:
            drag_to(_g_world(*nodes[idx]), 0.045, f"{tag}-wp", hard=False)
        drag_to(target_xy, tol, tag)

    def open_clamp(station: int, jban: int, tag: str) -> None:
        """Squeeze-analog: ramped opposed z-torques on the station's levers until the
        banana falls; then release the drive and let the spring re-close."""
        kl, kr = 2 * station, 2 * station + 1
        fell = False
        # up to 4 squeeze cycles: ramp open, hold; if the knob tilt-perches on a pad
        # edge instead of dropping, release (spring snap re-seats it) and re-squeeze
        for cyc in range(4):
            for i in range(30 + 150):
                ramp = min(1.0, (i + 1) / 30.0)
                scene.lever_drive[:, kl] = +2.2 * ramp
                scene.lever_drive[:, kr] = -2.2 * ramp
                env.step(no_action)
                if float(loc(scene.bananas[jban])[2]) < 0.16:
                    fell = True
                    break
            b = loc(scene.bananas[jban])
            vz = float(scene.bananas[jban].data.root_lin_vel_w[0, 2])
            ang = torch.rad2deg(scene.lever_angles()[0])
            print(f"[solve] open[{tag}] cyc{cyc}: fell={fell} "
                  f"b=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
                  f"vz={vz:+.2f} ang=({float(ang[kl]):+.1f},{float(ang[kr]):+.1f}) deg",
                  flush=True)
            if fell:
                break
            scene.lever_drive[:, kl] = 0.0
            scene.lever_drive[:, kr] = 0.0
            for _ in range(25):        # spring snaps shut; knocks a perched knob loose
                env.step(no_action)
                if float(loc(scene.bananas[jban])[2]) < 0.16:
                    fell = True
                    break
            if fell:
                break
        scene.lever_drive[:, :] = 0.0
        assert fell, f"open[{tag}]: banana never fell (clamp did not release)"
        # settle: banana comes to rest in the crate, spring re-closes the clamp
        for _ in range(300):
            env.step(no_action)
            if bool(scene.in_crate()[0, jban]) and bool(scene._still()[0]):
                break
        report(f"caught-{tag}")
        assert bool(scene.in_crate()[0, jban]), \
            f"open[{tag}]: banana {jban} did not settle inside the crate"
        assert not bool(scene.bruised()[0]), f"open[{tag}]: a banana was bruised"

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(60)
    st = [int(v) for v in scene._ban_station[0]]
    gx, gy_ = float(scene._gantry_xy[0, 0]), float(scene._gantry_xy[0, 1])
    gyaw = math.degrees(float(scene._gantry_yaw[0]))
    dx, dy = float(scene._depot_xy[0, 0]), float(scene._depot_xy[0, 1])
    pc = loc(scene.crate)
    print(f"[solve] layout readback (seed {args.seed}): gantry=({gx:+.3f},{gy_:+.3f}) "
          f"yaw={gyaw:+.1f}deg stations(y0,y1,green)={st} crate=({float(pc[0]):+.3f},"
          f"{float(pc[1]):+.3f}) depot=({dx:+.3f},{dy:+.3f})", flush=True)
    report("reset")
    hng = scene.hanging()[0]
    assert all(bool(v) for v in hng), "bananas did not settle hanging on their clamps"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "score not ~0 at reset"

    # ---------------- phase 1: TRANSPORT the empty crate (one free-space teleport) --------------
    hp = scene.hang_points()[0]  # (3, 2) live hang points, env-local
    tgt0 = hp[0]
    stt = torch.zeros(n, 13, device=device)
    stt[:, 0:2] = tgt0
    stt[:, 2] = 0.003
    gq = scene.gantry.data.root_quat_w[0]
    stt[:, 3:7] = gq  # align crate long axis with the beam
    stt[:, 0:3] += scene.env_origins
    scene.crate.write_root_state_to_sim(stt, all_ids)
    step(30)
    report("staged-A")
    assert not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02, \
        "staging the empty crate must satisfy no gate"
    s1 = print_score("P1 transport: empty crate staged under the first yellow clamp")
    assert s1 >= s0 - 1e-6

    # ---------------- phase 2: SQUEEZE clamp A -> banana falls into the crate -------------------
    open_clamp(st[0], 0, "A")
    s2 = print_score("P2 first yellow released into the crate (mechanism + gravity)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.24, "first catch did not latch credit"

    # ---------------- phase 3: DRAG the loaded crate under clamp B (contact) --------------------
    hp = scene.hang_points()[0]
    route_to(hp[1].clone(), 0.020, "to-B")
    report("staged-B")
    s3 = print_score("P3 crate dragged under the second yellow clamp")
    assert s3 >= s2 - 1e-6

    # ---------------- phase 4: SQUEEZE clamp B ---------------------------------------------------
    open_clamp(st[1], 1, "B")
    s4 = print_score("P4 second yellow released into the crate")
    assert s4 >= s3 - 1e-6 and s4 >= 0.49, "second catch did not latch credit"

    # ---------------- phase 5: DRAG the loaded crate to the depot -------------------------------
    route_to(scene._depot_xy[0].clone(), 0.030, "to-depot")
    report("delivered")
    s5 = print_score("P5 loaded crate delivered to the depot")
    assert s5 >= s4 - 1e-6

    # ---------------- phase 6: release + persistence (>= 3.3 simulated seconds) -----------------
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
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
