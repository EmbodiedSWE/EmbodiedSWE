"""Smoke / rubric-REJECTION battery for DrawerBeadPourScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i134`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — slide-force open, held-bowl pour, park — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled state and
asserts the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: drawer shut, every present
                            bead resting inside the bowl at its floor slot, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: present bead count varies,
                            the bowl's LEFT/RIGHT slot flips (Bernoulli), and per-slot
                            xy jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed task's terminal relation (the BLACK BOWL placed
                            INSIDE the open bottom drawer), beads still in the bowl:
                            zero deposit credit, NOT success, score well below the cap
                            (the role inversion is load-bearing);
  7.  shelf decoy         — all beads dropped in the always-open TOP SHELF compartment
                            and the bowl parked -> score ~0, no success (the zero-
                            effort receptacle earns nothing);
  8.  floor scatter       — beads dumped loose on the floor, bowl parked -> score ~0,
                            no success;
  9.  partial near-miss   — drawer open, all present beads but ONE on the drawer
                            floor, the last one still in the bowl -> NOT success,
                            deposit credit strictly partial, score <= 0.85;
  10. bowl on its side    — every bead deposited but the bowl LYING on its side on
                            the floor (tilt >> 20 deg) -> the park clause alone
                            rejects: NOT success;
  11. bowl in the shelf   — every bead deposited, bowl UPRIGHT but standing in the
                            shelf compartment (not at floor height) -> park clause
                            rejects: NOT success;
  12. latched credit      — teleporting the deposited beads back OUT of the drawer
                            leaves the latched score unchanged (credit does not
                            evaporate), still no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_bead_pour")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.05, 0.85)) + o),
                                tuple(np.array((0.45, 0.00, 0.12)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def open_mm() -> float:
        return float(scene.drawer_open()[0]) * 1000.0

    def pres_idx() -> list[int]:
        return [i for i in range(c.n_beads) if bool(scene.present[0, i])]

    def n_aboard() -> int:
        return int((scene.beads_in_bowl() & scene.present)[0].sum())

    def n_dep() -> int:
        return int((scene.beads_deposited() & scene.present)[0].sum())

    def report(tag: str) -> None:
        b = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | open={open_mm():5.1f}mm "
              f"bowl=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
              f"aboard={n_aboard()}/{len(pres_idx())} dep={n_dep()}/{len(pres_idx())} "
              f"opened={bool(scene._opened[0])} carry={float(scene._carry_max[0]):.3f} "
              f"depmax={float(scene._dep_max[0]):.3f} park={bool(scene._park[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def pose_drawer(open_m: float) -> None:
        """Teleport the drawer along its own prismatic slide coordinate (a joint-arc
        write: the joint stays consistent; the springless slide then holds the pose)."""
        place(scene.drawer, c.front_x - open_m, 0.0, c.drawer_z)

    def bowl_xyz() -> tuple[float, float, float]:
        p = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def beads_into_bowl_at(bx: float, by: float, bz_center: float) -> None:
        """Ring the PRESENT beads just above the floor of a bowl standing at
        (bx, by, bz_center) (probe constructor)."""
        floor_top = bz_center - c.bowl_h / 2 + c.bowl_floor_t
        for j, i in enumerate(pres_idx()):
            ang = 2 * math.pi * j / max(len(pres_idx()), 1)
            place(scene.beads[i], bx + 0.020 * math.cos(ang), by + 0.020 * math.sin(ang),
                  floor_top + c.bead_r + 0.004 + 0.001 * j)

    def beads_onto_drawer_floor(idxs: list[int]) -> None:
        """Spread beads on the CURRENT drawer's floor (probe constructor)."""
        d = (scene.drawer.data.root_pos_w - scene.env_origins)[0]
        z = float(d[2]) + c.drawer_t + c.bead_r + 0.004
        for j, i in enumerate(idxs):
            place(scene.beads[i], float(d[0]) + 0.06 + 0.055 * (j // 2),
                  -0.045 + 0.09 * (j % 2), z)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    step(90)
    report("reset-settle")
    fin0 = (torch.isfinite(scene.drawer.data.root_state_w).all()
            and torch.isfinite(scene.bowl.data.root_state_w).all()
            and all(torch.isfinite(b.data.root_state_w).all() for b in scene.beads))
    _, _, bz = bowl_xyz()
    still = (float(scene.bowl.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float((scene._bead_speed() * scene.present.float())[0].max()) < c.settle_speed)
    check("settle: states finite, drawer shut, every present bead resting inside the "
          "bowl, all still",
          bool(fin0) and open_mm() < 5.0 and n_aboard() == len(pres_idx())
          and len(pres_idx()) >= c.min_beads and abs(bz - c.bowl_h / 2) < 0.012 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        bx, by, _ = bowl_xyz()
        reads.append((bx, by, float(len(pres_idx())), 1.0 if by > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bowl_x, bowl_y, n_present, on_left):\n{arr}",
          flush=True)
    flags = arr[:, 3]
    check("randomization: bowl slot side flips across seeded resets AND the present "
          "bead count varies (readback)",
          0.0 < flags.mean() < 1.0 and arr[:, 2].min() < arr[:, 2].max()
          and arr[:, 2].min() >= c.min_beads and arr[:, 2].max() <= c.n_beads)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    check("randomization: per-slot spawn jitter is real (readback spread > 4 mm within "
          "a slot group)", jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: the BOWL into the drawer =================
    # The seed task's entire terminal relation — the black bowl resting inside the open
    # bottom drawer (beads still in it). Here that is a FAILURE state: beads in the bowl
    # are not deposited even inside the drawer, the bowl is not parked, no success, and
    # the score stays well below the cap.
    torch.manual_seed(41)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.005)
    step(20)
    d = (scene.drawer.data.root_pos_w - scene.env_origins)[0]
    in_x, in_z = float(d[0]) + 0.13, float(d[2]) + c.drawer_t + c.bowl_h / 2 + 0.004
    place(scene.bowl, in_x, 0.0, in_z)
    beads_into_bowl_at(in_x, 0.0, in_z)
    step(150)
    report("seed-strategy")
    s6, ok = judge()
    check("seed strategy: black bowl settled INSIDE the open drawer with its beads — "
          "bowl_in_drawer holds, ZERO deposit credit, NOT success, score <= 0.5 "
          "(the role inversion is load-bearing)",
          bool(scene.bowl_in_drawer()[0]) and n_aboard() == len(pres_idx())
          and n_dep() == 0 and float(scene._dep_max[0]) <= 0.01 and not ok and s6 <= 0.5)

    # =========================== 7. shelf decoy =============================================
    # Beads dropped in the always-open TOP SHELF compartment, bowl parked on the floor:
    # the zero-effort receptacle earns nothing. Drawer stays SHUT the whole time.
    torch.manual_seed(51)
    env.reset()
    step(30)
    shelf_z = c.plinth_h + c.shell_t + c.bay_h + c.shell_t  # panel top = shelf floor
    for j, i in enumerate(pres_idx()):
        place(scene.beads[i], c.front_x + 0.10 + 0.05 * (j // 2),
              -0.045 + 0.09 * (j % 2), shelf_z + c.bead_r + 0.004)
    place(scene.bowl, 0.20, -0.35, c.bowl_h / 2 + 0.002)
    step(150)
    report("shelf-decoy")
    s, ok = judge()
    check("shelf decoy: all beads settled in the TOP SHELF compartment, bowl parked — "
          "score ~0 (<= 0.02), no success", n_dep() == 0 and open_mm() < 5.0
          and not ok and s <= 0.02)

    # =========================== 8. floor scatter ===========================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    for j, i in enumerate(pres_idx()):
        place(scene.beads[i], 0.15 + 0.07 * j, 0.45, c.bead_r + 0.002)
    place(scene.bowl, 0.20, -0.35, c.bowl_h / 2 + 0.002)
    step(150)
    report("floor-scatter")
    s, ok = judge()
    check("floor scatter: beads dumped loose on the floor, bowl parked — score ~0 "
          "(<= 0.02), no success", n_dep() == 0 and not ok and s <= 0.02)

    # =========================== 9. partial near-miss =======================================
    # Drawer open, every present bead but ONE on the drawer floor, the last one still in
    # the bowl: deposit credit is strictly partial and success is out of reach.
    torch.manual_seed(71)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.005)
    step(20)
    idxs = pres_idx()
    k = len(idxs)
    beads_onto_drawer_floor(idxs[:-1])
    step(150)
    report("partial")
    s9, ok = judge()
    check("partial near-miss: drawer open, all present beads but ONE deposited, the "
          "last still in the bowl — NOT success, deposit fraction strictly partial, "
          "score <= 0.85",
          n_dep() == k - 1 and n_aboard() == 1 and not ok
          and abs(float(scene._dep_max[0]) - (k - 1) / k) < 0.01 and s9 <= 0.85)

    # =========================== 10. bowl on its side =======================================
    # Fresh episode: everything delivered, but the emptied bowl LYING on its side on the
    # floor. The park clause (upright cone) alone rejects.
    torch.manual_seed(81)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.005)
    step(20)
    idxs = pres_idx()
    beads_onto_drawer_floor(idxs)
    r_out = c.bowl_inner_r + c.bowl_t
    place(scene.bowl, 0.16, -0.32, r_out + 0.004,
          quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))  # 90 deg about y
    step(180)
    report("bowl-on-side")
    s10, ok = judge()
    from isaaclab.utils.math import quat_apply
    up = quat_apply(scene.bowl.data.root_quat_w,
                    torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(up[0, 2])))))
    check("bowl on side: every bead deposited but the bowl lying on its side "
          f"(tilt {tilt_deg:.0f} deg > {c.park_tilt_max_deg:.0f}) — park clause "
          "rejects: NOT success, no park latch",
          n_dep() == len(idxs) and tilt_deg > c.park_tilt_max_deg + 10.0
          and not bool(scene.bowl_parked()[0]) and not bool(scene._park[0]) and not ok
          and s10 <= 0.85)

    # =========================== 11. bowl upright in the SHELF ==============================
    # Same episode: stand the emptied bowl UPRIGHT — but in the decoy shelf compartment.
    # Upright passes; the floor-height clause rejects.
    place(scene.bowl, c.front_x + 0.14, 0.0, shelf_z + c.bowl_h / 2 + 0.004)
    step(150)
    report("bowl-in-shelf")
    s11, ok = judge()
    _, _, bz11 = bowl_xyz()
    up = quat_apply(scene.bowl.data.root_quat_w,
                    torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
    check("bowl in shelf: every bead deposited, bowl UPRIGHT but standing in the shelf "
          "compartment (not at floor height) — park clause rejects: NOT success",
          n_dep() == len(idxs) and float(up[0, 2]) > 0.94 and bz11 > 0.10
          and not bool(scene.bowl_parked()[0]) and not bool(scene._park[0]) and not ok
          and s11 <= 0.85)

    # =========================== 12. latched credit survives regression =====================
    for j, i in enumerate(idxs):
        place(scene.beads[i], 0.10 + 0.07 * j, 0.45, c.bead_r + 0.002)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: teleporting the deposited beads back OUT of the drawer "
          f"leaves the latched score unchanged ({s11:.3f} -> {s12:.3f}), still no "
          "success", abs(s12 - s11) < 1e-3 and n_dep() == 0 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.drawer.data.root_state_w).all()
           and torch.isfinite(scene.bowl.data.root_state_w).all()
           and torch.isfinite(scene.shell.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all() for b in scene.beads))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drawer_bead_pour")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
