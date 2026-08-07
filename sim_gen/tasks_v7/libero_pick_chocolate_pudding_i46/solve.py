"""Teleport solution for PuddingSiftScene (sim_gen task `libero_pick_chocolate_pudding_i46`)
— the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always starting and ending in FREE SPACE;
every load-bearing interaction goes through CONTACT DYNAMICS:
  P1 — CARRY + POUR: external forces (the floating-hand analog of a handle-bar
  grasp) lift the LOADED bin off the floor and servo it to a hover over the
  hopper's grate; the sealed contents ride inside through real contact. A torque
  servo then tilts the bin port-down over the grate: the pudding ball and the
  pearls slide over the sill and out through the port under gravity — nothing is
  ever teleported out of the sealed bin. The falling batch lands on the grate,
  where hole geometry does the sorting: pearls thread the 26 mm holes, the 42 mm
  pudding ball is retained on top. (Physics-API note: on this stack external
  wrenches are applied in a rotating body frame, so the desired WORLD wrench is
  pre-rotated by q_ref * q_now^-1 each step, q_ref re-anchored by a no-op root
  rewrite just before forces start.)
  P2 — PARK: the same force-carry moves the emptied bin to open floor on its own
  side, lowers it, and releases; the bin settles upright under gravity.
  P3 — SIFT-ASSIST: any pearl that came to rest ON a grate bar (or bounced to the
  floor) is teleported to free space a few mm ABOVE the nearest unoccupied hole
  and DROPPED — the fingertip-nudge analog. The aperture passage itself is pure
  gravity + contact; a pearl is never teleported into the basin, and a pearl still
  inside the sealed bin is never teleported (that would bypass the port and is a
  hard FAIL here). If the pudding ball bounced off the grate it is re-dropped over
  the grate centre the same way — the grate, not the teleport, retains it.
  P4 — DELIVER: the retained pudding ball is teleported from its exposed seat on
  the grate to free space above the open dish and dropped in; it settles by
  contact.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_chocolate_pudding_i46.solve --headless [--seed N]
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
    env = ENVS.get("simgen.pudding_sift")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def quat_rotate_single(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return quat_apply(q.view(1, 4), v.view(1, 3)).view(3)

    def quat_conj(q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[1:] = -out[1:]
        return out

    def quat_mul_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_mul

        return quat_mul(a.view(1, 4), b.view(1, 4)).view(4)

    def pearls_in_basin0() -> torch.Tensor:
        return scene.pearls_in_basin()[0]

    def pearls_in_bin0() -> torch.Tensor:
        return scene.pearls_in_bin()[0]

    def present0() -> torch.Tensor:
        return scene.present[0]

    def contents_out() -> bool:
        return (not bool(scene.pud_in_bin()[0])
                and not bool((pearls_in_bin0() & present0()).any()))

    def report(tag: str) -> None:
        in_basin = int((pearls_in_basin0() & present0()).sum())
        in_bin = int((pearls_in_bin0() & present0()).sum())
        npres = int(present0().sum())
        bl = (scene.bin.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | pearls {in_basin}/{npres} in basin, {in_bin} in bin "
              f"| pud_in_bin={bool(scene.pud_in_bin()[0])} "
              f"on_grate={bool(scene.pud_on_grate()[0])} "
              f"in_dish={bool(scene.pud_in_dish()[0])} "
              f"| bin=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"parked={bool(scene.bin_parked()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.bin.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    def hopper_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.hopper.data.root_pos_w - scene.env_origins)[0]
        q = scene.hopper.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def hopper_to_world(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        fp, yaw = hopper_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy,
                float(fp[1]) + lx * sy + ly * cy,
                float(fp[2]) + lz)

    def teleport(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp, hyaw = hopper_pose()
    side = float(scene.side[0])
    binp = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    qb = scene.bin.data.root_quat_w[0]
    byaw = 2.0 * math.atan2(float(qb[3]), float(qb[0]))
    dishp = (scene.dish.data.root_pos_w - scene.env_origins)[0]
    npres = int(present0().sum())
    print(f"[solve] layout readback (seed {args.seed}): hopper=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg side={side:+.0f} "
          f"bin=({float(binp[0]):+.3f},{float(binp[1]):+.3f}) "
          f"bin_yaw={math.degrees(byaw):+.1f}deg "
          f"dish=({float(dishp[0]):+.3f},{float(dishp[1]):+.3f}) "
          f"pearls_present={npres} in_bin={int((pearls_in_bin0() & present0()).sum())}",
          flush=True)
    if not bool(scene.pud_in_bin()[0]) or int((pearls_in_bin0() & present0()).sum()) != npres:
        fail("contents did not settle sealed inside the bin")
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- force-carry machinery -------------------------------------------------
    # External forces act on the BIN only (the floating-hand analog of a handle-bar
    # grasp); the contents ride via real contact. World wrench pre-rotated by
    # q_ref * q_now^-1 (rotating-body-frame wrench API).
    kp, kd = 240.0, 20.0   # per-kg linear gains
    kr, krd = 0.5, 0.05    # up-axis torque servo
    # Pour tilt must go PAST vertical: below ~90 deg gravity presses the contents
    # into the interior floor and the 28 mm sill retains them; past ~90 deg they
    # slide along the port wall to the roof-side band of the opening and fall out.
    # At 120 deg tilt + 24 deg lateral rock the bin's lowest corner is ~0.13 below
    # its root, so hover at z = 0.36 keeps >= 13 mm above the hopper's lip (0.220).
    hover_z = 0.36
    grate_c = (float(hp[0]), float(hp[1]))

    # Re-anchor the wrench reference frame with a no-op root rewrite (pose kept,
    # velocity zeroed), then capture q_ref.
    st = scene.bin.data.root_state_w.clone()
    st[:, 7:13] = 0.0
    scene.bin.write_root_state_to_sim(st, all_ids)
    q_ref = scene.bin.data.root_quat_w[0].clone()

    def port_dir() -> torch.Tensor:
        """Horizontal heading of the bin's +x (port) axis. Only valid while the bin
        is near-upright — past ~90 deg tilt the projection flips sign, so the pour
        phase FREEZES this direction instead of recomputing it."""
        xw = quat_rotate_single(scene.bin.data.root_quat_w[0], ex)
        p = torch.tensor([float(xw[0]), float(xw[1]), 0.0], device=device)
        nn = float(p.norm())
        return p / nn if nn > 1e-3 else ex.clone()

    def hover_pt(p: torch.Tensor) -> torch.Tensor:
        # The opening's exit corner sits ~0.065-0.09 ahead of the root along the
        # port heading while tilted past vertical: aim the root short of centre.
        return torch.tensor([grate_c[0] - 0.075 * float(p[0]),
                             grate_c[1] - 0.075 * float(p[1]), hover_z], device=device)

    def drive(steps: int, carrot_fn, u_t_fn=None, break_fn=None) -> int:
        """PD force-carry of the bin with an up-axis attitude servo. Returns the
        break step (or -1 if the loop ran out)."""
        for i in range(steps):
            pos = (scene.bin.data.root_pos_w - scene.env_origins)[0]
            vel = scene.bin.data.root_lin_vel_w[0]
            w = scene.bin.data.root_ang_vel_w[0]
            q_now = scene.bin.data.root_quat_w[0]
            m_tot = c.bin_mass \
                + (c.pud_mass if bool(scene.pud_in_bin()[0]) else 0.0) \
                + c.pearl_mass * int((pearls_in_bin0() & present0()).sum())
            carrot = carrot_fn(i)
            f = (kp * (carrot - pos) - kd * vel) * m_tot
            f[2] += m_tot * 9.81
            f = f.clamp(-12.0, 12.0)
            u_t = u_t_fn(i) if u_t_fn is not None else ez
            u = quat_rotate_single(q_now, ez)
            tau = kr * torch.linalg.cross(u, u_t) - krd * w
            tau = tau.clamp(-1.0, 1.0)
            q_corr = quat_mul_single(q_ref, quat_conj(q_now))
            f_a = quat_rotate_single(q_corr, f)
            t_a = quat_rotate_single(q_corr, tau)
            scene.bin.set_external_force_and_torque(
                f_a.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                t_a.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if break_fn is not None and break_fn(i):
                return i
        return -1

    def bin_tilt_deg() -> float:
        u = quat_rotate_single(scene.bin.data.root_quat_w[0], ez)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u[2])))))

    def lerp_carrot(a: torch.Tensor, b: torch.Tensor, steps: int):
        def fn(i: int) -> torch.Tensor:
            s = min(1.0, i / max(steps, 1))
            return a + s * (b - a)
        return fn

    # ---------------- phase 1: CARRY the loaded bin + POUR through the port ----------------
    # Rise vertically first (nothing above the bin), then translate at hover height
    # (clears the hopper lip by >= 80 mm), then tilt port-down over the grate.
    start = (scene.bin.data.root_pos_w - scene.env_origins)[0].clone()
    up_pt = torch.tensor([float(start[0]), float(start[1]), hover_z], device=device)
    drive(220, lerp_carrot(start, up_pt, 180))
    hov0 = hover_pt(port_dir())
    drive(320, lerp_carrot(up_pt, hov0, 260))
    # FREEZE the port heading and hover point for the whole pour (the heading
    # readback is meaningless past 90 deg tilt).
    p_pour = port_dir()
    yp = torch.linalg.cross(ez, p_pour)
    hov = hover_pt(p_pour)
    print(f"[solve] bin at hover over the grate (port heading "
          f"{math.degrees(math.atan2(float(p_pour[1]), float(p_pour[0]))):+.1f}deg); "
          f"pouring", flush=True)

    dt = 1.0 / 120.0
    th_a = 112.0   # phase A: past-vertical pour through the port slot
    th_b = 142.0   # phase B: near-inverted — stragglers slide the roof into the notch

    def pour_u_t(i: int) -> torch.Tensor:
        if i < 480:
            th, phi = th_a * i / 480.0, 0.0
        elif i < 1100:
            t = (i - 480) * dt
            th = th_a + 8.0 * math.sin(2.0 * math.pi * 1.0 * t)
            # lateral rock: tan(24 deg) = 0.45 beats the bin's mu of 0.22, so
            # contents wedged behind a pillar slide back across the opening
            phi = 24.0 * math.sin(2.0 * math.pi * 0.5 * t)
        else:
            # phase B: steady heading (no rotational rock — the up-axis servo has
            # no yaw authority, so the bin free-yaws and rotational rock never
            # produces bin-local y gravity; the shake below does the work).
            th = min(th_b, th_a + (th_b - th_a) * (i - 1100) / 240.0)
            phi = 0.0
        th, phi = math.radians(th), math.radians(phi)
        u_t = math.cos(th) * ez + math.sin(th) * (math.cos(phi) * p_pour
                                                  + math.sin(phi) * yp)
        return u_t / u_t.norm()

    def pour_carrot(i: int) -> torch.Tensor:
        # extra height for phase B: at 142 deg + rock the lowest corner is ~0.16
        # below the root
        z = hover_z if i < 1100 else min(0.40, hover_z + 0.04 * (i - 1100) / 100.0)
        base = torch.tensor([float(hov[0]), float(hov[1]), z], device=device)
        if i >= 1360:
            # translational y-shake: pearls jammed in the corner column beside the
            # port pillar move rigidly with the bin under any rotation (frozen
            # bin-local telemetry), but a lateral inertial shake of ~6 m/s^2
            # beats the ~3 m/s^2 friction hold and slides them the ~14 mm along
            # the roof underside into the open notch span, where they drop out.
            t = (i - 1360) * dt
            base = base + (0.035 * math.sin(2.0 * math.pi * 2.5 * t)) * yp
        return base

    out_streak = [0]

    def pour_done(i: int) -> bool:
        out_streak[0] = out_streak[0] + 1 if contents_out() else 0
        if i % 200 == 0:
            locs = []
            for j in range(c.n_pearls):
                if bool(present0()[j]) and bool(pearls_in_bin0()[j]):
                    bl = scene._local(scene.bin, scene.pearls[j].data.root_pos_w)[0]
                    locs.append(f"p{j}=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                                f"{float(bl[2]):+.3f})")
            print(f"[solve] pour step {i}: tilt={bin_tilt_deg():.1f}deg "
                  f"pud_in_bin={bool(scene.pud_in_bin()[0])} "
                  f"in_bin_local[{' '.join(locs)}]", flush=True)
        return i > 500 and out_streak[0] >= 30

    broke = drive(2000, pour_carrot, pour_u_t, pour_done)
    print(f"[solve] pour loop ended (break step {broke}, tilt now "
          f"{bin_tilt_deg():.1f}deg)", flush=True)
    # Ramp back upright while holding the hover.
    th_end = bin_tilt_deg()

    def unpour_u_t(i: int) -> torch.Tensor:
        th = math.radians(th_end * max(0.0, 1.0 - i / 200.0))
        u_t = math.cos(th) * ez + math.sin(th) * p_pour
        return u_t / u_t.norm()

    drive(280, lambda i: hov, unpour_u_t)
    if not contents_out():
        fail("pour did not empty the bin through the port")
    report("poured")
    s1 = print_score("P1 carry + pour through the port")
    assert s1 >= s0 - 1e-6, "score decreased across the pour phase"

    # ---------------- phase 2: PARK the emptied bin on open floor --------------------------
    park = torch.tensor([0.16, -0.40 * side, hover_z], device=device)
    hov_now = (scene.bin.data.root_pos_w - scene.env_origins)[0].clone()
    drive(300, lerp_carrot(hov_now, park, 240))
    low = torch.tensor([float(park[0]), float(park[1]), 0.015], device=device)
    drive(260, lerp_carrot(park, low, 220))
    clear_wrench()
    step(180)
    if not bool(scene.bin_parked()[0]):
        fail("bin did not park upright on the floor")
    report("parked")
    s2 = print_score("P2 bin parked")
    assert s2 >= s1 - 1e-6, "score decreased across the park phase"

    # ---------------- phase 3: SIFT-ASSIST — drop stragglers through free holes ------------
    # A pearl resting on a grate bar (or bounced to the floor) is teleported to free
    # space just above the nearest unoccupied hole and dropped: the hole passage is
    # gravity + contact. Pearls are NEVER teleported into the basin, and a pearl
    # still inside the sealed bin would be a bypass — hard FAIL instead.
    holes = [(hx, hy) for hx in c.hole_centers for hy in c.hole_centers]
    for rnd in range(10):
        stray = (present0() & ~pearls_in_basin0()).nonzero().flatten().tolist()
        pud_ok = (bool(scene.pud_on_grate()[0]) or bool(scene.pud_in_dish()[0]))
        if not stray and pud_ok:
            break
        if any(bool(pearls_in_bin0()[i]) for i in stray):
            fail("a pearl is still sealed inside the bin (no teleport bypass)")
        if bool(scene.pud_in_bin()[0]):
            fail("the pudding ball is still sealed inside the bin (no teleport bypass)")
        # Pudding recovery first: re-drop over the grate centre; the grate retains it.
        if not pud_ok:
            x, y, z = hopper_to_world(0.0, 0.0, c.grate_z1 + c.pud_r + 0.020)
            teleport(scene.pudding, x, y, z)
            step(60)
        pud_loc = scene._local(scene.hopper, scene.pudding.data.root_pos_w)[0]
        px, py = float(pud_loc[0]), float(pud_loc[1])
        used: list[tuple[float, float]] = []
        for i in stray:
            body = scene.pearls[i]
            loc = scene._local(scene.hopper, body.data.root_pos_w)[0]
            lx, ly = float(loc[0]), float(loc[1])
            free = [h for h in holes
                    if max(abs(h[0] - px), abs(h[1] - py)) > 0.032 and h not in used]
            hx, hy = min(free, key=lambda h: (h[0] - lx) ** 2 + (h[1] - ly) ** 2)
            used.append((hx, hy))
            x, y, z = hopper_to_world(hx, hy, c.grate_z1 + c.pearl_r + 0.020)
            teleport(body, x, y, z)
        step(90)
        print(f"[solve] sift round {rnd}: dropped {len(stray)} stragglers over free holes",
              flush=True)
    if not bool((pearls_in_basin0() | ~present0()).all()):
        fail("not every present pearl made it through the grate")
    if not (bool(scene.pud_on_grate()[0]) or bool(scene.pud_in_dish()[0])):
        fail("the pudding ball was not retained on the grate")
    report("sifted")
    s3 = print_score("P3 all pearls through the grate")
    assert s3 >= s2 - 1e-6, "score decreased across the sift phase"

    # ---------------- phase 4: DELIVER the retained pudding ball to the dish ---------------
    dp = (scene.dish.data.root_pos_w - scene.env_origins)[0]
    teleport(scene.pudding, float(dp[0]), float(dp[1]),
             float(dp[2]) + c.dish_floor_t + c.pud_r + 0.030)
    step(180)
    for _ in range(10):  # allow extra settling for the success gate
        if bool(scene.success()[0]):
            break
        step(60)
    if not bool(scene.success()[0]):
        fail("no success after the pudding ball was placed in the dish")
    report("delivered")
    s4 = print_score("P4 pudding delivered")
    assert s4 >= s3 - 1e-6, "score decreased across the delivery phase"

    # ---------------- phase 5: persistence (>= 3.4 simulated seconds, hands off) -----------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.4 s")
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
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - Kit teardown hangs on a bare traceback
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(2)
