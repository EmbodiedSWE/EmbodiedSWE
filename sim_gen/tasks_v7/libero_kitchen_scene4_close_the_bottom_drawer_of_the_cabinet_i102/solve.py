"""Teleport solution for DrawerRefitScene (sim_gen task
`libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102`) — the task's
legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating bodies the solver is already holding in
free air). Every load-bearing interaction is contact dynamics or applied force:

1. PERCEPTION: the sideboard's pose/yaw, WHICH bay carries the blue tag chip, and
   the ground poses of the drawer and the block are read back from the episode
   state — never hard-coded.
2. LOAD (applied force + gravity): a PD force + gravity feedforward (a firm grasp,
   force-limited) lifts the green block into free air; the held block is
   TELEPORTED over the grounded drawer's open top and RELEASED — gravity drops it
   into the basin, and the block-in-basin latch is judged on that settled contact
   outcome.
3. TRANSPORT (applied force, then teleport): the same PD carry lifts the LOADED
   drawer into free air (the block rides inside on real contact); the held
   assembly is teleported — preserving the block's drawer-relative pose — to a
   hover pose at the TAGGED bay's mouth, aligned with the sideboard, and PD-held
   there (the grasp stabilizing before the slide).
4. INSERTION (applied force): a slow force-driven waypoint slide pushes the box
   through the face opening along the bay floor — jambs as lateral guides, the
   bay floor as the vertical guide — until the oversized face plate lands FLUSH
   on the jamb fronts (the real mechanical stop; the PD presses it home). All
   insertion credit and success() are judged on this contact-driven slide.
5. RELEASE + persistence: forces cleared, everything settles; success() must hold
   hands-off >= 3 simulated seconds before `SIM_GEN_SOLVE: SUCCESS` is printed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches partial credit; success is judged live).

Force-frame convention: on the forge pods the DEFAULT
`set_external_force_and_torque` call applies wrenches in the body's CURRENT
frame, so every commanded world wrench is pre-encoded with
`quat_apply_inverse(q_now, .)` — the exact pattern validated on nine forge seeds
by the sibling `close_drawer_i58` solve.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_refit")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
        print(f"[solve] {tag:12s} | gap={float(scene.plate_gap()[0]):+.4f} "
              f"gates={bool(scene._gates()[0])} "
              f"seated={bool(scene.drawer_seated()[0])} "
              f"cube_in={bool(scene.cube_in_drawer()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def board_frame():
        return scene.board.data.root_pos_w, scene.board.data.root_quat_w

    def b2w(local) -> torch.Tensor:
        bp, bq = board_frame()
        loc = torch.zeros(n, 3, device=device)
        loc[:] = torch.tensor(local, device=device)
        return bp + quat_apply(bq, loc)

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    def drive(body, mass: float, tgt_fn, *, kp: float, kd: float, clamp: float,
              ko: float, kw: float, clampt: float, steps: int,
              board_aligned: bool = False, done=None, label: str = "") -> None:
        """Applied-force carry/slide: PD toward a (possibly moving) world target +
        gravity feedforward — what a firm force-limited grasp does — plus a
        righting/alignment torque standing in for the grasp's orientation
        constraint. The wrench goes through THIS body only; anything riding inside
        it is carried by real contact."""
        for i in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            f_w = mass * 9.81 * ez + kp * (tgt_fn(i) - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            if board_aligned:
                _, bq = board_frame()
                tx, tz = quat_apply(bq, ex), quat_apply(bq, ez)
            else:
                tx, tz = None, ez
            dz = quat_apply(q, ez)
            t_w = ko * torch.cross(dz, tz, dim=-1) - kw * w
            if tx is not None:
                dx = quat_apply(q, ex)
                t_w = t_w + ko * torch.cross(dx, tx, dim=-1)
            t_norm = t_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            t_w = t_w * (t_norm.clamp(max=clampt) / t_norm)
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._board_local(body.data.root_pos_w)[0]
        print(f"[solve] drive {label}: reached board-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    all_ids = torch.arange(n, device=device)

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)   # drawer and block seat on the ground
    bp0 = (scene.board.data.root_pos_w - scene.env_origins)[0]
    bq0 = scene.board.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(bq0[3]), float(bq0[0])))
    k = int(scene.target_bay[0])
    zk = float(c.bay_floors[k])
    dloc = scene._board_local(scene.drawer.data.root_pos_w)[0]
    cloc = scene._board_local(scene.cube.data.root_pos_w)[0]
    tag_loc = scene._board_local(scene.tag.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"board=({float(bp0[0]):+.3f},{float(bp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"target_bay={k} (floor z={zk:.3f}) tag_z={float(tag_loc[2]):.3f} "
          f"drawer=({float(dloc[0]):+.3f},{float(dloc[1]):+.3f}) "
          f"cube=({float(cloc[0]):+.3f},{float(cloc[1]):+.3f})", flush=True)
    assert abs(float(tag_loc[2]) - (zk + c.open_h + c.tag_dz)) < 0.01, \
        "tag chip must sit above the TARGET bay's opening"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.cube_in_drawer()[0]), "block must start outside the drawer"
    assert not bool(scene.drawer_seated()[0]), "drawer must start removed"
    s_prev = print_score("P0 reset+settle (drawer and block on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: LOAD — block into the grounded drawer's basin ---------------------
    # Force-lift the block straight up into free air (a firm grasp, <= 4 N).
    cw = scene.cube.data.root_pos_w.clone()
    lift_tgt = cw.clone()
    lift_tgt[:, 2] = 0.20
    drive(scene.cube, c.cube_mass, lambda i: lift_tgt,
          kp=40.0, kd=4.0, clamp=4.0, ko=0.02, kw=0.005, clampt=0.10, steps=240,
          done=lambda: float(scene.cube.data.root_pos_w[0, 2]) > 0.18,
          label="block lift")
    assert float(scene.cube.data.root_pos_w[0, 2]) > 0.12, "block must be in free air"
    # Transport the held block over the drawer's open top; release; gravity loads it.
    dp = scene.drawer.data.root_pos_w
    dq = scene.drawer.data.root_quat_w
    drop = torch.zeros(n, 3, device=device)
    drop[:, 2] = c.box_h + 0.045
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = dp + quat_apply(dq, drop)
    st[:, 3:7] = dq
    scene.cube.write_root_state_to_sim(st, all_ids)
    step(240)   # free fall into the basin, settle
    report("load")
    assert bool(scene.cube_in_drawer()[0]), "block must rest inside the basin"
    assert bool(scene._cube_l[0]), "block-in-basin latch must fire once settled"
    s = print_score("P1 block loaded into the drawer basin (force lift + gravity drop)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert s >= c.w_cube - 1e-6, f"P1 score {s:.3f} below block credit"
    s_prev = s

    # ---------------- phase 2: TRANSPORT — loaded drawer to the tagged bay's mouth ---------------
    # Force-lift the loaded drawer into free air; the block rides on real contact.
    m_asm = c.drawer_mass + c.cube_mass
    dw = scene.drawer.data.root_pos_w.clone()
    lift_tgt = dw.clone()
    lift_tgt[:, 2] = 0.16
    drive(scene.drawer, m_asm, lambda i: lift_tgt,
          kp=60.0, kd=10.0, clamp=12.0, ko=0.30, kw=0.06, clampt=0.60, steps=300,
          done=lambda: float(scene.drawer.data.root_pos_w[0, 2]) > 0.15,
          label="drawer lift")
    assert float(scene.drawer.data.root_pos_w[0, 2]) > 0.10, "drawer must be in free air"
    assert bool(scene.cube_in_drawer()[0]), "block must still ride in the basin"
    # Teleport the held ASSEMBLY (drawer + block, preserving the block's
    # drawer-relative pose) to the hover pose at the tagged bay's mouth.
    dq = scene.drawer.data.root_quat_w
    off = quat_apply_inverse(dq, scene.cube.data.root_pos_w - scene.drawer.data.root_pos_w)
    q_rel = scene_mod._qmul(_qconj(dq), scene.cube.data.root_quat_w)
    hover_local = (c.box_l / 2 + 0.055, 0.0, zk + 0.006)
    bp, bq = board_frame()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b2w(hover_local)
    st[:, 3:7] = bq
    scene.drawer.write_root_state_to_sim(st, all_ids)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b2w(hover_local) + quat_apply(bq, off)
    st[:, 3:7] = scene_mod._qmul(bq, q_rel)
    scene.cube.write_root_state_to_sim(st, all_ids)
    # PD-hold at the hover pose (the grasp stabilizing before the slide).
    drive(scene.drawer, m_asm, lambda i: b2w(hover_local),
          kp=60.0, kd=12.0, clamp=15.0, ko=0.40, kw=0.08, clampt=0.60, steps=90,
          board_aligned=True, label="hover hold")
    report("hover")
    assert bool(scene.cube_in_drawer()[0]), "block must survive the transport"
    up, fx = scene._axes_dots()
    assert float(up[0]) > 0.98 and float(fx[0]) > 0.98, "drawer must hover aligned"
    s = print_score("P2 loaded drawer held at the tagged bay's mouth")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    s_prev = s

    # ---------------- phase 3: INSERTION — force-driven slide to flush ---------------------------
    # Three stages so the box can never wedge-climb the sill lip (a fully
    # gravity-compensated carry has nothing pulling it back down once a lip catch
    # nudges it up): (A) enter HIGH — nose crosses the face plane 6 mm above the
    # sill; (B) drop — the box floor is set down onto the bay floor and PRESSED
    # (feedforward at 0.9x weight = a steady ~0.4 N down-preload); (C) slide flush
    # along the floor under the same down-preload, jambs guiding laterally, until
    # the plate lands on the jamb fronts.
    x0, x_in, x1 = c.box_l / 2 + 0.055, -0.030, -c.box_l / 2 - 0.004
    z_hi, z_lo = zk + 0.006, zk - 0.003   # z_lo below the floor: PD presses down
    travel_a, travel_c = 300, 280

    def tgt_a(i: int) -> torch.Tensor:
        s01 = min(1.0, i / travel_a)
        return b2w((x0 + s01 * (x_in - x0), 0.0, z_hi))

    drive(scene.drawer, m_asm, tgt_a,
          kp=60.0, kd=12.0, clamp=15.0, ko=0.40, kw=0.08, clampt=0.60,
          steps=travel_a + 120, board_aligned=True,
          done=lambda: float(scene._board_local(scene.drawer.data.root_pos_w)[0, 0]) < x_in + 0.004,
          label="insert A: enter high")
    drive(scene.drawer, 0.90 * m_asm, lambda i: b2w((x_in, 0.0, z_lo)),
          kp=60.0, kd=12.0, clamp=15.0, ko=0.40, kw=0.08, clampt=0.60,
          steps=90, board_aligned=True, label="insert B: set down on the bay floor")

    def tgt_c(i: int) -> torch.Tensor:
        s01 = min(1.0, i / travel_c)
        return b2w((x_in + s01 * (x1 - x_in), 0.0, z_lo))

    def ins_done() -> bool:
        return float(scene.plate_gap()[0]) < 0.004 \
            and float(scene.drawer.data.root_lin_vel_w[0].norm()) < 0.03

    drive(scene.drawer, 0.90 * m_asm, tgt_c,
          kp=60.0, kd=12.0, clamp=15.0, ko=0.40, kw=0.08, clampt=0.60,
          steps=travel_c + 200, board_aligned=True, done=ins_done,
          label="insert C: slide flush")
    step(180)   # release; friction and the flush stop hold the seated pose
    report("flush")
    assert bool(scene._eng_l[0]), "engagement latch must fire during the slide"
    assert float(scene._ins_f[0]) >= 0.95, \
        f"insertion fraction latched only {float(scene._ins_f[0]):.3f}"
    assert bool(scene.drawer_seated()[0]), \
        f"drawer must sit flush: gap={float(scene.plate_gap()[0]):+.4f}"
    assert bool(scene.cube_in_drawer()[0]), "block must be inside the closed drawer"
    assert bool(scene.success()[0]), "success() must hold after the slide settles"
    s = print_score("P3 drawer slid home through the opening; plate flush on the jambs")
    assert s >= s_prev - 1e-6, "score decreased across P3"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
