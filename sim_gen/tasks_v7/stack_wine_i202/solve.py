"""Teleport solution for BottleHangRailScene (sim_gen task `stack_wine_i202`) — the
task's legitimacy certificate.

The task is a three-stage keyhole maneuver per carafe, and every load-bearing part of
it happens through CONTACT DYNAMICS under an external-force servo (a stand-in for the
robot's grasp): teleports are used for TRANSPORT ONLY, always ending in free space
BELOW the plate, never inside the keyhole and never in a hanging pose.

Per carafe (assigned keyhole = its own index; any bijection is accepted by the rubric):

  T  — TRANSPORT: teleport to a hover in free air ~28 mm below the plate, directly
       under the assigned port, upright (IDENTITY orientation), velocities zeroed.
  A  — CONVERGE: a force/torque servo (force at the root = CoM, so no spurious torque)
       holds the hover and pulls the collar onto the port's centreline until the xy
       error is < 3.5 mm and the carafe is slow and plumb.
  B  — INSERT: velocity-servo rise (~0.10 m/s, tapered) carries the collar UP THROUGH
       the port — real geometry: if the alignment were off, the collar would butt the
       plate from below (a stall detector backs off and retries).
  C  — SLIDE: with the collar hovering ~2.5 mm above the plate top (the neck captive
       in the slot), a capped horizontal pull walks the neck down the narrow channel
       until it stalls against the channel's far end (along ~ 27 mm, inside the hang
       band). The channel walls, not the servo, provide the lateral guidance.
  D  — RELEASE: only once slow (wrenches act one substep late), all forces are zeroed;
       the carafe DROPS the last 2.5 mm and hangs by collar-on-plate contact, then
       settles as a damped pendulum. The hang is verified by rubric readback
       (`scene.hang_cells()`), with full re-tries from stage T on failure.

Wrench-frame discipline (forge pods drag world wrenches by the body's rotation since
reset): the carafe is teleported to IDENTITY orientation and held plumb by an
uprighting PD torque, so R stays ~ I and every frame convention coincides; commands
are additionally pre-encoded into the CURRENT body frame (`quat_apply_inverse`) for
the default body-frame call, and stage A doubles as a progress probe — a diverging xy
error flips the encoding mode.

Gain audit (m = 0.30 kg, dt = 1/120 s, Ixx ~ 4.3e-4, Izz ~ 1.1e-4):
  xy PD  Kp 15 N/m, Kd 4 N·s/m  -> wn 7.1 rad/s (wn·dt 0.06), zeta 0.94, Kd·dt/m 0.11
  z  PD  Kp 40 N/m, Kd 6 N·s/m  -> wn 11.5 rad/s (wn·dt 0.10), zeta 0.87
  z  vel Kv 8 N·s/m             -> Kv·dt/m 0.22
  upright k_up 0.035 N·m, k_w 0.008 -> wn 9.1 rad/s, zeta ~ 1.0, k_w·dt/I <= 0.6
All discrete-stability ratios are well under 1.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched: 0.00 -> 0.25 -> 0.50 -> 0.75 -> 1.00), then holds HANDS-OFF for
>= 3.3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.stack_wine_i202.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bottle_hang_rail")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    MG = c.mass * 9.81
    KP_XY, KD_XY = 15.0, 4.0
    KP_Z, KD_Z = 40.0, 6.0
    KV_Z = 8.0
    K_UP, K_OM, T_MAX = 0.035, 0.008, 0.08
    Z_HOVER = c.seat_z_loc + 0.0025  # collar-centre rail-local z while sliding
    Z_UNDER = -c.plate_t / 2 - 0.028  # stage T/A hover: 28 mm below the plate bottom

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rail_pq() -> tuple[torch.Tensor, torch.Tensor]:
        return scene.rail.data.root_pos_w[0], scene.rail.data.root_quat_w[0]

    def rail_world(p_loc: torch.Tensor) -> torch.Tensor:
        rp, rq = rail_pq()
        return rp + quat_apply(rq.unsqueeze(0), p_loc.unsqueeze(0))[0]

    def bstate(i: int):
        """pos, quat, up-axis, collar centre (world), lin vel, ang vel of carafe i."""
        b = scene.bottles[i]
        pos = b.data.root_pos_w[0]
        quat = b.data.root_quat_w[0]
        axis = quat_apply(quat.unsqueeze(0), ez.unsqueeze(0))[0]
        return pos, quat, axis, pos + axis * c.collar_c, \
            b.data.root_lin_vel_w[0], b.data.root_ang_vel_w[0]

    def collar_local(i: int) -> torch.Tensor:
        rp, rq = rail_pq()
        collar = bstate(i)[3]
        return quat_apply_inverse(rq.unsqueeze(0), (collar - rp).unsqueeze(0))[0]

    encode_body = [True]  # pre-encode wrenches into the body frame (probed in stage A)

    def apply_wrench(i: int, f_w: torch.Tensor, t_w: torch.Tensor, quat: torch.Tensor) -> None:
        if encode_body[0]:
            f = quat_apply_inverse(quat.unsqueeze(0), f_w.unsqueeze(0))[0]
            t = quat_apply_inverse(quat.unsqueeze(0), t_w.unsqueeze(0))[0]
        else:
            f, t = f_w, t_w
        scene.bottles[i].set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            t.view(1, 1, 3).expand(n, 1, 3).contiguous(), env_ids=all_ids)

    def clear_wrench(i: int) -> None:
        zw = torch.zeros(n, 1, 3, device=device)
        scene.bottles[i].set_external_force_and_torque(zw, zw, env_ids=all_ids)

    def servo_substep(i: int, tgt_w: torch.Tensor, *, vz_des: float | None = None,
                      fxy_max: float = 1.2) -> None:
        """One substep of the CoM force + uprighting torque servo toward a collar
        target (world). Position-PD in z unless a rise velocity is commanded."""
        _pos, quat, axis, collar, v, w = bstate(i)
        e = tgt_w - collar
        f = torch.zeros(3, device=device)
        f[0] = KP_XY * e[0] - KD_XY * v[0]
        f[1] = KP_XY * e[1] - KD_XY * v[1]
        fn = float(f[:2].norm())
        if fn > fxy_max:
            f[:2] *= fxy_max / fn
        if vz_des is not None:
            f[2] = MG + KV_Z * (vz_des - v[2])
        else:
            f[2] = MG + KP_Z * e[2] - KD_Z * v[2]
        f[2] = f[2].clamp(0.2 * MG, 2.4 * MG)
        t = K_UP * torch.linalg.cross(axis, ez) - K_OM * w
        tn = float(t.norm())
        if tn > T_MAX:
            t *= T_MAX / tn
        apply_wrench(i, f, t, quat)
        env.step(no_action)

    def report(tag: str) -> None:
        hung, cell = scene.hang_cells()
        print(f"[solve] {tag:18s} | hung={[bool(v) for v in hung[0]]} "
              f"cell={[int(v) for v in cell[0]]} "
              f"settled={[bool(v) for v in scene.settled()[0]]} "
              f"max_hang_ctr={[int(v) for v in scene.max_hang_ctr[0]]} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback -------------------------------
    step(90)
    rp, rq = rail_pq()
    yaw = 2.0 * float(torch.atan2(rq[3], rq[0]))
    print(f"[solve] layout readback (seed {args.seed}): rail at "
          f"({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={yaw * 57.2958:+.1f} deg; "
          "ports at " + " ".join(
              f"cell{k}=({float(rail_world(torch.tensor([c.cells_x[k], c.port_cy, 0.0], device=device))[0]):+.3f},"
              f"{float(rail_world(torch.tensor([c.cells_x[k], c.port_cy, 0.0], device=device))[1]):+.3f})"
              for k in range(3)), flush=True)
    print("[solve] carafes: " + " ".join(
        f"carafe_{i}@({float(bstate(i)[0][0]):+.3f},{float(bstate(i)[0][1]):+.3f})"
        for i in range(3)), flush=True)
    s_prev = print_score("P0 reset+settle")

    # ---------------- per-carafe keyhole maneuver --------------------------------------------
    def hover_teleport(i: int, k: int) -> None:
        """TRANSPORT ONLY: free air below the plate, under port k, upright, zero vel."""
        tgt = rail_world(torch.tensor([c.cells_x[k], c.port_cy, Z_UNDER], device=device))
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = tgt[0]
        st[:, 1] = tgt[1]
        st[:, 2] = tgt[2] - c.collar_c  # root so that the COLLAR sits at the target
        st[:, 3] = 1.0  # identity orientation — keeps every wrench-frame mode identical
        scene.bottles[i].write_root_state_to_sim(st, all_ids)

    def converge_under_port(i: int, k: int) -> bool:
        tgt_loc = torch.tensor([c.cells_x[k], c.port_cy, Z_UNDER], device=device)
        e0 = None
        for t in range(420):
            tgt = rail_world(tgt_loc)
            servo_substep(i, tgt)
            _pos, _q, axis, collar, v, w = bstate(i)
            exy = float((tgt - collar)[:2].norm())
            if e0 is None:
                e0 = max(exy, 1e-4)
            if t > 30 and exy < 0.0035 and float(v.norm()) < 0.06 \
                    and float(axis[2]) > 0.995 and float(w.norm()) < 1.0:
                return True
            if t > 200 and exy > max(4 * e0, 0.10):
                encode_body[0] = not encode_body[0]
                print(f"[solve] wrench-frame probe: xy error diverged ({e0:.3f} -> "
                      f"{exy:.3f} m) — flipping encode mode to "
                      f"{'body' if encode_body[0] else 'world'}", flush=True)
                return False
        exy = float((rail_world(tgt_loc) - bstate(i)[3])[:2].norm())
        print(f"[solve] converge timeout, residual xy={exy:.4f} m", flush=True)
        return exy < 0.005

    def rise_through_port(i: int, k: int) -> bool:
        tgt_loc = torch.tensor([c.cells_x[k], c.port_cy, Z_HOVER], device=device)
        best, stall = -1.0, 0
        for _t in range(480):
            zl = float(collar_local(i)[2])
            if zl >= Z_HOVER - 0.001:
                return True
            vz = min(0.10, max(0.02, 6.0 * (Z_HOVER + 0.002 - zl)))  # tapered rise
            servo_substep(i, rail_world(tgt_loc), vz_des=vz)
            if zl > best + 0.001:
                best, stall = zl, 0
            else:
                stall += 1
            if stall > 120:
                print(f"[solve] rise stalled at collar z_loc={zl:+.4f} — backing off",
                      flush=True)
                return False
        print("[solve] rise timeout", flush=True)
        return False

    def slide_to_end(i: int, k: int) -> bool:
        # target past the reachable end -> capped pull stalls the neck on the channel end
        tgt_loc = torch.tensor([c.cells_x[k], c.chan_y0 + 0.032, Z_HOVER], device=device)
        best, stall = -1.0, 0
        along = -1.0
        for _t in range(900):
            along = float(collar_local(i)[1]) - c.chan_y0
            v = bstate(i)[4]
            if along >= 0.024 and float(v.norm()) < 0.04:
                return True
            if along >= c.hang_y_min + 0.004 and stall > 240 and float(v.norm()) < 0.05:
                return True  # deep in the band, hard-stalled — accept
            servo_substep(i, rail_world(tgt_loc), fxy_max=0.9)
            if along > best + 0.0008:
                best, stall = along, 0
            else:
                stall += 1
        print(f"[solve] slide timeout at along={along:+.4f} m", flush=True)
        return along >= c.hang_y_min + 0.004

    def release_and_settle(i: int, k: int) -> bool:
        # freeze in place to kill residual swing, then release ONLY when slow
        loc = collar_local(i)
        hold_loc = torch.tensor([c.cells_x[k], float(loc[1]), Z_HOVER], device=device)
        for _ in range(60):
            servo_substep(i, rail_world(hold_loc), fxy_max=0.6)
        for _ in range(180):
            if float(bstate(i)[4].norm()) < 0.05:
                break
            servo_substep(i, rail_world(hold_loc), fxy_max=0.6)
        clear_wrench(i)  # the carafe DROPS the last ~2.5 mm and hangs by contact
        step(300)
        for _ in range(6):  # allow up to +3 s of pendulum decay
            hung, cell = scene.hang_cells()
            if bool(hung[0, i]) and int(cell[0, i]) == k and bool(scene.settled()[0, i]):
                loc = collar_local(i)
                print(f"[solve] carafe_{i} hangs in keyhole {k}: collar loc="
                      f"({float(loc[0]):+.4f},{float(loc[1]):+.4f},{float(loc[2]):+.4f}) "
                      f"along={float(loc[1]) - c.chan_y0:+.4f}", flush=True)
                return True
            step(60)
        hung, cell = scene.hang_cells()
        loc = collar_local(i)
        print(f"[solve] release verdict: hung={bool(hung[0, i])} cell={int(cell[0, i])} "
              f"collar loc=({float(loc[0]):+.4f},{float(loc[1]):+.4f},"
              f"{float(loc[2]):+.4f})", flush=True)
        return bool(hung[0, i]) and int(cell[0, i]) == k

    for i in range(3):
        placed = False
        for attempt in range(4):
            print(f"[solve] carafe_{i} -> keyhole {i}, attempt {attempt}", flush=True)
            clear_wrench(i)
            hover_teleport(i, i)
            if not converge_under_port(i, i):
                continue
            if not rise_through_port(i, i):
                continue
            if not slide_to_end(i, i):
                continue
            if release_and_settle(i, i):
                placed = True
                break
            clear_wrench(i)
        if not placed:
            report("FAIL-state")
            print(f"SIM_GEN_SOLVE: FAIL (carafe_{i} never hung)", flush=True)
            os._exit(1)
        report(f"hung carafe_{i}")
        s_now = print_score(f"P1 carafe_{i} hung in keyhole {i}")
        assert s_now >= s_prev - 1e-6, "score decreased across a hang"
        s_prev = s_now

    # ---------------- final settle + judge ---------------------------------------------------
    step(120)
    report("all hung")
    s2 = print_score("P2 all hung + settle")
    assert s2 >= s_prev - 1e-6, "score decreased across the final settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after hanging)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) ----------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except BaseException:  # noqa: BLE001 — die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
