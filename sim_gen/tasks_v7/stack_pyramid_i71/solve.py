"""Teleport solution for FalseworkTentScene (sim_gen task `stack_pyramid_i71`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. FALSEWORK LEANS (contact statics): each panel is TELEPORTED into the pose a
   gripper would release it in — tilted ~25 deg, base on the pad 3 mm up, face
   2 mm off the pillar's flat side — and RELEASED. Gravity closes both gaps: the
   base drops onto the pad and the panel tips onto the column (its CoM is inside
   the base-to-face support interval, worked out from the geometry), so the
   settled lean that the `lean`/`pair` latches credit is held entirely by
   panel-pillar-pad contact, not by the write. A lone panel in that pose with no
   pillar would fall — smoke #6 shows the collapse.
2. EXTRACTION (applied force, the falsework moment): the pillar is lifted by a
   VERTICAL force servo at its CoM — the straight-up pull a gripper applies to
   the yellow handle. F_z = m*(g + kp*(v_des - v_z)) capped at [0, 2.5 m g],
   plus a small horizontal position hold and an uprighting-torque orientation
   hold (together, the grip stiffness of the grasp on the handle bar). A
   vertical pull is yaw-invariant, so the pod's wrench-frame drag
   (rotation-since-reset) has nothing to bite on. The pull is two-speed: slow
   while the panels bear on the faces, fast through the release band so the
   column bottom outruns the released panels' rising top edges; the panels then
   fall inward onto each other; the combine-"min" 0.12 friction on the pillar
   faces means the slide up is nearly free. The tent that `free`/success credit
   is the settled result of that collapse — pure contact dynamics, never
   authored.
3. PARK (transport + gravity): once clear (root z > 0.30), forces are zeroed and
   the held pillar is carried (teleport-transport) to 3 cm above the magenta
   pad and RELEASED; it lands and settles under gravity.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.stack_pyramid_i71.solve --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qz, _qx = scene_mod._qmul, scene_mod._qz, scene_mod._qx

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.falsework_tent")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

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
        tilt = scene.panel_tilt_deg()[0]
        top, _bot = scene._ends()
        gap = float((top[0, 0] - top[0, 1]).norm())
        pz = scene.pillar.data.root_pos_w[0] - scene.env_origins[0]
        print(f"[solve] {tag:12s} | tilt=({float(tilt[0]):.1f},{float(tilt[1]):.1f})deg "
              f"apex_gap={gap:.3f} pillar=({float(pz[0]):+.3f},{float(pz[1]):+.3f},"
              f"{float(pz[2]):+.3f}) "
              f"lean={bool(scene._lean[0])} pair={bool(scene._pair[0])} "
              f"free={bool(scene._free[0])} tent={bool(scene.tent_standing()[0])} "
              f"parked={bool(scene.pillar_parked()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.pillar.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                   env_ids=all_ids)

    def pad_pose(y_loc: float, z_loc: float, alpha: float) -> torch.Tensor:
        """Root state for a body at pad-local (0, y_loc, z_loc), tipped by
        qx(alpha) in the pad frame, carried to world by the pad's yaw."""
        psi = scene.build_yaw
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.build_xy[:, 0] - torch.sin(psi) * y_loc
        st[:, 1] = scene.build_xy[:, 1] + torch.cos(psi) * y_loc
        st[:, 2] = scene.env_origins[:, 2] + z_loc
        st[:, 3:7] = _qmul(_qz(psi), _qx(torch.full((n,), alpha, device=device)))
        return st

    # ---------------- phase 0: reset, settle, baseline -----------------------------------------
    step(180)
    bx = scene.build_xy[0] - scene.env_origins[0, 0:2]
    px = scene.park_xy[0] - scene.env_origins[0, 0:2]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"build=({float(bx[0]):+.3f},{float(bx[1]):+.3f}) "
          f"yaw={math.degrees(float(scene.build_yaw[0])):+.1f}deg "
          f"park=({float(px[0]):+.3f},{float(px[1]):+.3f}) "
          f"swapped={bool(scene.slot_swapped[0])}", flush=True)
    report("reset")
    tilt0 = scene.panel_tilt_deg()[0]
    assert float(tilt0.min()) > 80.0, f"panels must start flat, tilts {tilt0.tolist()}"
    pz0 = float(scene.pillar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    assert pz0 < 0.05, f"pillar must start standing on the pad, root z {pz0:.3f}"
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    s0 = print_score("P0 reset+settle (panels flat, pillar standing)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- lean-pose geometry (pad-local, worked out from cfg) ----------------------
    # Contact plane of a pillar face: col_y/2 + 2 mm authoring gap. A panel released
    # at tilt theta with its base 3 mm above the pad tips ONTO the face (CoM inside
    # the support interval) and settles into a stable lean. Theta is chosen NEAR the
    # final tent angle so that when the thin column slides out, each panel has only a
    # few degrees to fall before catching the other.
    theta = math.radians(25.0)
    face = c.col_y / 2 + 0.002
    base_z = c.pad_t + 0.003

    def lean_state(i: int, side: float) -> torch.Tensor:
        """Panel i leaning on the pillar face at side=+1 (+y) or side=-1 (-y)."""
        length = c.panel_l[i]
        top_y = side * (face + (c.panel_t / 2) * math.cos(theta))
        base_y = top_y + side * length * math.sin(theta)
        ctr_y = (top_y + base_y) / 2
        ctr_z = base_z + (length / 2) * math.cos(theta)
        # qx(+a) maps local z -> (0, -sin a, cos a): side=+1 needs +theta to lean
        # INWARD (top toward the face at smaller y).
        return pad_pose(ctr_y, ctr_z, side * theta)

    # ---------------- phase 1: lean panel_a on the +y face (falsework skill) -------------------
    scene.panels["panel_a"].write_root_state_to_sim(lean_state(0, +1.0), all_ids)
    step(90)
    report("lean-A")
    ta = float(scene.panel_tilt_deg()[0, 0])
    assert c.tilt_lo_deg < ta < c.tilt_hi_deg, \
        f"panel_a should hold a settled lean on the pillar, tilt {ta:.1f}deg"
    assert bool(scene._lean[0]), "lean latch must be set by the settled lean"
    s1 = print_score("P1 panel_a leaning on the pillar (settled)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_lean - 1e-6, f"P1 score {s1} (expect >= {c.w_lean})"

    # ---------------- phase 2: lean panel_b on the -y face (opposed pair) ----------------------
    scene.panels["panel_b"].write_root_state_to_sim(lean_state(1, -1.0), all_ids)
    step(90)
    report("lean-B")
    tb = float(scene.panel_tilt_deg()[0, 1])
    assert c.tilt_lo_deg < tb < c.tilt_hi_deg, \
        f"panel_b should hold a settled lean on the pillar, tilt {tb:.1f}deg"
    assert bool(scene.opposed()[0]), "the two leans must be opposed"
    assert bool(scene._pair[0]), "pair latch must be set by the opposed settled leans"
    assert not bool(scene.tent_standing()[0]), \
        "tent must NOT count while the pillar is still inside it"
    s2 = print_score("P2 both panels leaning on the pillar (opposed pair)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_lean + c.w_pair - 1e-6, \
        f"P2 score {s2} (expect >= {c.w_lean + c.w_pair})"

    # ---------------- phase 3: extract the falsework (vertical force servo) --------------------
    # Straight-up pull on the handle: F_z servos the ascent to ~0.25 m/s; a soft
    # horizontal position hold is the grip stiffness. Panels release sequentially as
    # the column bottom passes their contact heights and collapse inward.
    from isaaclab.utils.math import quat_apply

    m_p = c.pillar_mass
    g = 9.81
    xy0 = scene.pillar.data.root_pos_w[:, 0:2].clone()
    e_z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    lifted = False
    for i in range(1200):
        z_rel = float(scene.pillar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        # Slow while the panels bear on the faces, then FAST through the release
        # band: a released panel tips inward and its top edge RISES (top height
        # = L cos(tilt) with tilt shrinking), so the column bottom must outrun
        # the rising tops or they smack into it and the pair scatters (observed:
        # collapse at z 0.147-0.186 with a 0.08 m/s crawl). Panel tops sit at
        # ~0.144 / 0.162; switch to the fast pull just below the first one.
        v_des = 0.08 if z_rel < 0.135 else 0.55
        vel = scene.pillar.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (m_p * (g + 8.0 * (v_des - vel[:, 2]))).clamp(0.0, 2.5 * m_p * g)
        f[:, 0:2] = (m_p * (25.0 * (xy0 - scene.pillar.data.root_pos_w[:, 0:2])
                            - 6.0 * vel[:, 0:2])).clamp(-0.5 * m_p * g, 0.5 * m_p * g)
        # Orientation hold — the rigid grasp on the handle: once airborne the pillar
        # has no rotational stiffness of its own and the (asymmetric) panel normal
        # forces would spin it. Uprighting torque + angular damping model the
        # gripper's grip; without them the falsework wanders and knocks the leans
        # down long before the release heights.
        u = quat_apply(scene.pillar.data.root_quat_w, e_z)
        tau = 1.5 * torch.cross(u, e_z, dim=-1) \
            - 0.08 * scene.pillar.data.root_ang_vel_w
        scene.pillar.set_external_force_and_torque(
            f.unsqueeze(1), tau.unsqueeze(1), env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i % 60 == 0:
            t = scene.panel_tilt_deg()[0]
            print(f"[solve] lift step {i:4d}: root z {z_rel:.3f} "
                  f"tilt=({float(t[0]):.1f},{float(t[1]):.1f})deg", flush=True)
        z_rel = float(scene.pillar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        if z_rel > 0.30:
            lifted = True
            print(f"[solve] pillar clear at step {i}: root z {z_rel:.3f}", flush=True)
            break
    if not lifted:
        clear_force()
        report("LIFT-FAIL")
        print("SIM_GEN_SOLVE: FAIL (vertical extraction never cleared the tent)",
              flush=True)
        os._exit(1)
    # Carry (transport-only) to 1 cm above the magenta pad and RELEASE (gentle set-
    # down: the thin column lands flat and stays up).
    clear_force()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = scene.park_xy
    st[:, 2] = scene.env_origins[:, 2] + c.pad_t + 0.01
    st[:, 3] = 1.0
    scene.pillar.write_root_state_to_sim(st, all_ids)
    step(360)  # fall + tent collapse fully settles, hands-off
    report("extracted")
    assert bool(scene.tent_standing()[0]), \
        "the panels must stand as a free tent after the falsework is removed"
    assert bool(scene._free[0]), "free latch must be set by the settled tent"
    assert bool(scene.pillar_parked()[0]), "pillar must rest on the park pad"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after extraction + park)", flush=True)
        os._exit(1)
    s3 = print_score("P3 falsework out, tent free-standing, pillar parked")
    assert s3 >= s2 - 1e-6, "score decreased across the extraction"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -----------------
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
    except Exception as exc:  # noqa: BLE001 — fail fast, don't hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
