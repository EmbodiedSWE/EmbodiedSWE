"""Solution for RailRingScene (sim_gen task `put_toilet_roll_on_stand_i342`) — the
task's legitimacy certificate.

This solve uses NO transport teleports at all: the ring is topologically captive on
the rail (the red ball cap seals the start end), so there is nothing to teleport —
every meter of the delivery is contact dynamics:

  1. TRAVERSE (contact dynamics): a pursuit controller pushes the hanging ring along
     the rail with a velocity-regulated force along the local course tangent (a rigid
     grasp sliding it, as a gripper would) plus a weak attitude PD that keeps the
     ring's axis on the course tangent through both 45-degree corner chamfers. The
     16 mm radial passage tolerance is enforced by COLLISION (the rail in the bore),
     never by the wrench — the gains are too weak to pull the ring through the rail.
     A stall detector (no arc-length gain for 3.3 s) fires a short reverse pulse and
     retries.
  2. COMMIT (gravity): once the ring's station passes the crest (the downturn entry),
     the wrench is CUT. The slide down the 45-degree chamfer, the pivot onto the
     vertical end post, the free drop down the post, and the landing flat on the
     plate around the post are pure gravity + contact. If the ring wedges above the
     post (belt-and-braces), the pursuit wrench re-engages briefly and cuts again.
  3. SETTLE: hands-off until success() (delivered AND sustained stillness), plus 2 s
     extra margin, then a >= 3.3 simulated-second persistence window with success()
     checked every substep. Nothing is welded, pinned, or held at the end.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), and `SIM_GEN_SOLVE: SUCCESS` only if success persists.

Run (forge): python -u -m simgen_tasks.put_toilet_roll_on_stand_i342.solve --headless [--seed N]
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

S_AT = scene_mod.S_AT
S_TOTAL = scene_mod.S_TOTAL
LAND_Z = scene_mod.LAND_Z
POST_XY = scene_mod.POST_XY
path_s_dist = scene_mod.path_s_dist
path_tangent = scene_mod.path_tangent

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rail_ring")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor, tau_w: torch.Tensor | None = None) -> None:
        """Apply a WORLD force/torque (n,3) to `body`, expressed in its CURRENT link
        frame (`is_global=True` silently drops torques on this stack — transform
        manually, the house convention). Re-set every step while pushing."""
        q = body.data.root_link_quat_w
        if tau_w is None:
            tau_w = torch.zeros(n, 3, device=device)
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            quat_apply_inverse(q, tau_w).unsqueeze(1),
            env_ids=all_ids)

    zero3 = torch.zeros(n, 3, device=device)

    def cut(body) -> None:
        wrench(body, zero3, zero3)

    def axis(body) -> torch.Tensor:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def station() -> tuple[float, float, float]:
        """(s, dist, z_local) of the ring on the course (env 0)."""
        p, _a = scene.ring_in_rail()
        s, d = path_s_dist(p)
        return float(s[0]), float(d[0]), float(p[0, 2])

    def report(tag: str) -> None:
        p, a = scene.ring_in_rail()
        s, d = path_s_dist(p)
        post = torch.tensor([POST_XY[0], POST_XY[1]], device=device)
        dpost = float((p[0, 0:2] - post).norm())
        print(f"[solve] {tag:12s} | ring_loc=({float(p[0, 0]):+.3f},"
              f"{float(p[0, 1]):+.3f},{float(p[0, 2]):.3f})"
              f" s={float(s[0]):.3f}/{S_TOTAL:.3f} dist={float(d[0]):.4f}"
              f" dpost={dpost:.3f} tilt_cos={float(a[0, 2].abs()):.3f}"
              f" del={bool(scene.delivered()[0])} set={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}"
              f" | v={float(scene.ring.data.root_lin_vel_w[0].norm()):.4f}"
              f" w={float(scene.ring.data.root_ang_vel_w[0].norm()):.4f}"
              f" still={int(scene.still_count[0])}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def pursuit_wrench(v_des: float) -> None:
        """One substep of the traversal controller: velocity-regulated push along the
        course tangent + weak attitude PD onto the tangent (perpendicular-omega
        damping only — the wrench-delay bound forbids damping the axial spin)."""
        q_r = scene.rail.data.root_quat_w
        p, _a = scene.ring_in_rail()
        s, _d = path_s_dist(p)
        t_loc = path_tangent(s + 0.020, device)
        t_w = quat_apply(q_r, t_loc)
        v = scene.ring.data.root_lin_vel_w
        v_along = (v * t_w).sum(-1, keepdim=True)
        fmag = (3.0 * (v_des - v_along)).clamp(-0.5, 0.8)
        a_w = axis(scene.ring)
        sgn = torch.sign((a_w * t_w).sum(-1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        a_eff = a_w * sgn
        w = scene.ring.data.root_ang_vel_w
        w_perp = w - (w * a_w).sum(-1, keepdim=True) * a_w
        tau = (0.006 * torch.cross(a_eff, t_w, dim=-1) - 0.002 * w_perp).clamp(-0.008, 0.008)
        wrench(scene.ring, t_w * fmag, tau)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    rp = (scene.rail.data.root_pos_w - scene.env_origins)[0]
    q0 = scene.rail.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(q0[3]), float(q0[0])))
    s_now, d_now, _z = station()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rail=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f} "
          f"ring s0={float(scene.s0[0]):.3f} s_settled={s_now:.3f} dist={d_now:.4f}",
          flush=True)
    report("reset")
    assert d_now < c.prog_gate_dist, "ring not hanging on the rail after reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"
    assert not bool(scene.delivered()[0]), "delivered at reset?!"

    # ---------------- phases 1-3: TRAVERSE to the crest (contact dynamics) -----------------
    crest = S_AT[3] + 0.012  # just past the downturn entry — committed to gravity
    marks = [(S_AT[1], "P1 first run traversed (contact push)"),
             (S_AT[2], "P2 corner chamfers negotiated (contact push)"),
             (crest, "P3 crest committed — wrench cut (gravity takes over)")]
    mark_i = 0
    s_best, last_gain, retries = s_now, 0, 0
    prev = 0.0
    i = 0
    while i < 12000:
        pursuit_wrench(0.12)
        env.step(no_action)
        i += 1
        s_now, d_now, _z = station()
        if s_now > s_best + 1e-4:
            s_best, last_gain = s_now, i
        while mark_i < len(marks) and s_now >= marks[mark_i][0]:
            report(f"mark{mark_i + 1}")
            sc = print_score(marks[mark_i][1])
            assert sc >= prev - 1e-6, "score decreased across a phase boundary"
            prev = sc
            mark_i += 1
        if mark_i >= len(marks):
            break
        if i - last_gain > 400:  # stalled 3.3 s: short reverse pulse, then retry
            retries += 1
            print(f"[solve] stall at s={s_now:.3f} (retry {retries})", flush=True)
            assert retries <= 8, "traversal wedged: too many stall retries"
            q_r = scene.rail.data.root_quat_w
            for j in range(60):
                p, _a = scene.ring_in_rail()
                s_t, _d = path_s_dist(p)
                t_w = quat_apply(q_r, path_tangent(s_t, device))
                wig = 0.004 if (j // 15) % 2 == 0 else -0.004
                tau = torch.zeros(n, 3, device=device)
                tau[:, 2] = wig
                wrench(scene.ring, -0.35 * t_w, tau)
                env.step(no_action)
            last_gain = i
    cut(scene.ring)
    assert mark_i >= len(marks), "traversal never reached the crest"

    # ---------------- phase 4: gravity delivery down the post + settle ---------------------
    # Hands off past the crest: chamfer slide -> post drop -> flat landing. If the
    # ring wedges above the post, briefly re-engage the pursuit wrench and cut again.
    reengages = 0
    down = False
    calm = 0
    for i in range(1800):
        env.step(no_action)
        s_now, d_now, z_loc = station()
        if z_loc < 0.055:
            down = True
            break
        v_now = float(scene.ring.data.root_lin_vel_w[0].norm())
        calm = calm + 1 if v_now < 0.03 else 0
        if i > 240 and calm > 60:
            reengages += 1
            print(f"[solve] descent wedged at z={z_loc:.3f} s={s_now:.3f} "
                  f"(re-engage {reengages})", flush=True)
            assert reengages <= 6, "descent wedged: too many re-engagements"
            for _ in range(120):
                pursuit_wrench(0.10)
                env.step(no_action)
            cut(scene.ring)
            calm = 0
    assert down, "the ring never descended to the post foot"
    report("descended")

    for j in range(32):  # up to 8 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
        if j % 4 == 3:
            p, _a = scene.ring_in_rail()
            print(f"[solve] settle chunk {j}: z={float(p[0, 2]):.4f} "
                  f"v={float(scene.ring.data.root_lin_vel_w[0].norm()):.4f} "
                  f"w={float(scene.ring.data.root_ang_vel_w[0].norm()):.4f} "
                  f"still={int(scene.still_count[0])}", flush=True)
    if bool(scene.success()[0]):
        step(240)  # 2 s more hands-off margin before the strict persistence window
    report("landed")
    s4 = print_score("P4 delivered flat on the plate around the post (gravity + contact)")
    assert s4 >= prev - 1e-6, "score decreased across delivery"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"del={bool(scene.delivered()[0])} set={bool(scene.settled()[0])} "
                      f"v={float(scene.ring.data.root_lin_vel_w[0].norm()):.4f} "
                      f"w={float(scene.ring.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
