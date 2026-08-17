"""Smoke / rubric-REJECTION battery for WobblyBistroScene (sim_gen task
`set_the_table_i410`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — wedge pushed under the hovering foot by a
velocity-limited floor-level force, dishes released over their seats under gravity
— is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: empty table LEVEL with the
                           short foot hovering ~25 mm, wedge/dishes flat on the
                           floor, all still, score ~0;
  3.  mass readback      — PhysX get_masses() matches the authored explicit
                           masses (custom spawners: density mass would silently
                           break every moment-margin statics claim);
  4-5. randomization     — READBACK over 8 seeded resets: table xy jitter + free
                           yaw are real (the short corner points anywhere); wedge/
                           plate/cup floor slots and yaws are real;
  6.  null policy        — 240 idle steps -> score ~0, no success;
  7.  seed strategy      — the end state the SEED's plan produces here (just
                           arrange the dishes, ignore the furniture): both dishes
                           dropped onto their marked seats with NO shim -> the
                           table ROCKS onto the short leg (max tilt > 3 deg,
                           readback) and the polished top SHEDS both dishes off
                           their seats -> no dish credit, score ~0, no success
                           (repair-first is physics-forced);
  8.  wrong place (floor)— wedge parked BESIDE the short leg (in front of the gap,
                           not under it) + dishes dropped -> tips and sheds, no
                           shore credit, no success (footprint gating is real);
  9.  wrong leg          — wedge shoved under a LONG grounded leg + dishes dropped
                           -> tips and sheds, shored() (which reads the SHORT
                           foot) never fires, no success (fault identification is
                           load-bearing);
  10. ballast cheat      — wedge parked ON the tabletop's far corner as a
                           counterweight + dishes dropped -> still tips (asserted
                           moment margin, verified physically), no success;
  11. swapped dishes     — shim properly constructed, then plate dropped on the
                           BLUE DISC and cup on the RED RING -> table stays level
                           but neither dish is on its OWN seat: only the shore
                           credit (0.30), no dish credit, no success;
  12. near-miss seat     — shim + plate dropped 50 mm off the red ring (outside
                           the 35 mm tolerance, still on the table) -> no plate
                           credit, score stays 0.30, no success;
  13. partial + regression — shim + plate correctly seated -> score exactly 0.50,
                           no success; then the WEDGE is teleported away (table
                           and plate woken by identity state re-writes — a
                           settled stack sleeps, and moving the WEDGE does not
                           wake the bodies it was supporting): the table TIPS
                           (> 3 deg, readback — the shim really was load-bearing)
                           and sheds the plate; the latched 0.50 survives, still
                           no success (non-success cap holds);
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wobbly_bistro")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.95, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    step_i = 0
    max_tilt = [0.0]  # window-max tabletop tilt (deg), reset per probe

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            max_tilt[0] = max(max_tilt[0], float(scene.tilt_deg()[0]))
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

    def dish_loc(body) -> torch.Tensor:
        return scene._in_table_frame(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        pl, cl = dish_loc(scene.plate), dish_loc(scene.cup)
        s, ok = judge()
        print(f"[smoke] {tag:18s} | tilt={float(scene.tilt_deg()[0]):.2f} "
              f"max_tilt={max_tilt[0]:.2f} "
              f"foot_z={float(scene.leg_tip_w()[0, 2]):.4f} "
              f"plate_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},"
              f"{float(pl[2]):.3f}) cup_loc=({float(cl[0]):+.3f},"
              f"{float(cl[1]):+.3f},{float(cl[2]):.3f}) "
              f"shored={bool(scene.shored()[0])} "
              f"latches=({bool(scene._shored_ever[0])},"
              f"{bool(scene._plate_ever[0])},{bool(scene._cup_ever[0])}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def table_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w, quat_w, s_dir): table pose + world unit push line toward the
        short corner (table-frame (+1,+1) diagonal, z-projected)."""
        t_pos = scene.table.data.root_pos_w
        t_quat = scene.table.data.root_quat_w
        diag = torch.tensor([1.0, 1.0, 0.0], device=device) / math.sqrt(2.0)
        s = quat_apply(t_quat, diag.expand(n, 3)).clone()
        s[:, 2] = 0.0
        s = s / s.norm(dim=-1, keepdim=True)
        return t_pos, t_quat, s

    def drop_dish(body, seat: tuple, half_h: float, *, off=(0.0, 0.0)) -> None:
        """Teleport a dish to hover 20 mm above a table-frame point and release."""
        t_pos, t_quat, _s = table_frame()
        loc = torch.tensor([seat[0] + off[0], seat[1] + off[1],
                            c.top_face_z + half_h + 0.020],
                           device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = t_pos + quat_apply(t_quat, loc)
        st[:, 3:7] = t_quat
        body.write_root_state_to_sim(st, all_ids)

    def construct_shim() -> None:
        """PROBE CONSTRUCTOR: write the wedge directly into the shored pose (tip
        under the hovering foot, bearing edge at supporting height) and settle."""
        _t_pos, _t_quat, s = table_frame()
        foot = scene.leg_tip_w()
        yaw = math.atan2(float(s[0, 1]), float(s[0, 0]))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = foot[:, 0:2] - s[:, 0:2] * 0.004  # foot_in_wedge x = +0.004
        st[:, 2] = 0.001 + scene.env_origins[:, 2]
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.wedge.write_root_state_to_sim(st, all_ids)
        step(60)

    def park_wedge_floor(dist: float = 0.55) -> None:
        """Teleport the wedge to open floor far from the table."""
        t_pos, _t_quat, s = table_frame()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = t_pos[:, 0:2] - s[:, 0:2] * dist
        st[:, 2] = 0.001 + scene.env_origins[:, 2]
        st[:, 3] = 1.0
        scene.wedge.write_root_state_to_sim(st, all_ids)

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.table.data.root_state_w).all()
                    and torch.isfinite(scene.wedge.data.root_state_w).all()
                    and torch.isfinite(scene.plate.data.root_state_w).all()
                    and torch.isfinite(scene.cup.data.root_state_w).all())

    def dishes_off_seats() -> bool:
        pl, cl = dish_loc(scene.plate), dish_loc(scene.cup)
        p_off = math.hypot(float(pl[0]) - c.seat_plate[0],
                           float(pl[1]) - c.seat_plate[1]) > c.seat_tol
        c_off = math.hypot(float(cl[0]) - c.seat_cup[0],
                           float(cl[1]) - c.seat_cup[1]) > c.seat_tol
        return p_off and c_off

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(120)
    report("show")
    foot_z = float(scene.leg_tip_w()[0, 2] - scene.env_origins[0, 2])
    wz = float(scene.wedge.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    still = (float(scene.table.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.wedge.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.plate.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.cup.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, empty table LEVEL with the short foot hovering "
          "~25 mm (the fault is visible only as the gap), wedge flat on the floor, "
          "all still",
          finite_all() and float(scene.tilt_deg()[0]) < c.level_tol_deg
          and 0.015 < foot_z < 0.035 and wz < 0.05 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3. mass readback ===========================================
    # Custom spawn funcs bypass Isaac's mass_props schema path: if the explicit
    # MassAPI mass did NOT take, PhysX falls back to density mass and every
    # moment-margin claim (dish out-tips the CoM bias, ballast insufficiency)
    # silently breaks. Read the masses actually simulated.
    m_tab = float(scene.table.root_physx_view.get_masses().flatten()[0])
    m_wed = float(scene.wedge.root_physx_view.get_masses().flatten()[0])
    m_pla = float(scene.plate.root_physx_view.get_masses().flatten()[0])
    m_cup = float(scene.cup.root_physx_view.get_masses().flatten()[0])
    print(f"[smoke] mass readback: table={m_tab:.4f}/{c.table_mass} "
          f"wedge={m_wed:.4f}/{c.wedge_mass} plate={m_pla:.4f}/{c.plate_mass} "
          f"cup={m_cup:.4f}/{c.cup_mass}", flush=True)
    check("masses: PhysX simulates the authored explicit masses (each within 2% "
          "of cfg — the statics story is real, not density-mass accident)",
          abs(m_tab - c.table_mass) < 0.02 * c.table_mass
          and abs(m_wed - c.wedge_mass) < 0.02 * c.wedge_mass
          and abs(m_pla - c.plate_mass) < 0.02 * c.plate_mass
          and abs(m_cup - c.cup_mass) < 0.02 * c.cup_mass)

    # =========================== 4-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        tp = (scene.table.data.root_pos_w - scene.env_origins)[0]
        tq = scene.table.data.root_quat_w[0]
        t_yaw = math.degrees(2.0 * math.atan2(float(tq[3]), float(tq[0])))
        wp = (scene.wedge.data.root_pos_w - scene.env_origins)[0]
        pp = (scene.plate.data.root_pos_w - scene.env_origins)[0]
        cp = (scene.cup.data.root_pos_w - scene.env_origins)[0]
        wq = scene.wedge.data.root_quat_w[0]
        w_yaw = math.degrees(2.0 * math.atan2(float(wq[3]), float(wq[0])))
        reads.append((float(tp[0]), float(tp[1]), t_yaw, float(wp[0]), float(wp[1]),
                      float(pp[0]), float(pp[1]), float(cp[0]), float(cp[1]), w_yaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (table_x, table_y, table_yaw_deg, "
          f"wedge_xy, plate_xy, cup_xy, wedge_yaw_deg):\n{arr}", flush=True)
    yaw_spread = float(np.ptp(arr[:, 2]))
    xy_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    check("randomization: table xy jitter (> 8 mm spread) and FREE yaw (> 40 deg "
          "spread — the short corner points anywhere) are real (readback)",
          xy_spread > 0.008 and yaw_spread > 40.0)
    item_spread = float(min(np.ptp(arr[:, 3]) + np.ptp(arr[:, 4]),
                            np.ptp(arr[:, 5]) + np.ptp(arr[:, 6]),
                            np.ptp(arr[:, 7]) + np.ptp(arr[:, 8])))
    check("randomization: wedge/plate/cup floor slots (> 10 mm xy spread each) and "
          "wedge yaw (> 40 deg spread) are real (readback)",
          item_spread > 0.010 and float(np.ptp(arr[:, 9])) > 40.0)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: set the table, skip the repair ===========
    # The seed's whole plan — arrange the dishes at their designated spots — executed
    # verbatim: both dishes released over their OWN seats with NO shim. The payload
    # moment rocks the table onto the short leg and the polished top sheds them.
    torch.manual_seed(41)
    env.reset()
    step(60)
    max_tilt[0] = 0.0
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2)
    drop_dish(scene.cup, c.seat_cup, c.cup_h / 2)
    step(600)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: dishes dropped on their marked seats with NO shim — the "
          f"table ROCKS onto the short leg (max tilt {max_tilt[0]:.1f} deg > 3) and "
          "the slick top sheds both dishes off their seats: no dish credit, score "
          "~0 (<= 0.02), no success (repair-first is physics-forced)",
          max_tilt[0] > 3.0 and dishes_off_seats()
          and not bool(scene._plate_ever[0]) and not bool(scene._cup_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 7. wedge beside the gap (not under) ========================
    # Wedge parked on the push line but 60 mm SHORT of the foot: in front of the
    # gap, not under it — the footprint gating must not read this as shored.
    torch.manual_seed(51)
    env.reset()
    step(60)
    _t_pos, _t_quat, s_dir = table_frame()
    foot = scene.leg_tip_w()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = foot[:, 0:2] + s_dir[:, 0:2] * (c.wedge_len / 2 + 0.060)
    st[:, 2] = 0.001 + scene.env_origins[:, 2]
    yaw = math.atan2(float(s_dir[0, 1]), float(s_dir[0, 0]))
    st[:, 3], st[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
    scene.wedge.write_root_state_to_sim(st, all_ids)
    step(30)
    max_tilt[0] = 0.0
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2)
    drop_dish(scene.cup, c.seat_cup, c.cup_h / 2)
    step(600)
    report("beside-gap")
    s, ok = judge()
    check("wrong place: wedge parked just IN FRONT of the gap (not under the foot) "
          f"— table still tips (max tilt {max_tilt[0]:.1f} deg > 3) and sheds: no "
          "shore credit, score ~0, no success (footprint gating is real)",
          max_tilt[0] > 3.0 and not bool(scene._shored_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 8. wedge under the WRONG leg ===============================
    # Wedge written under the diagonally opposite LONG leg (grounded): shoring a
    # grounded leg fixes nothing — the short corner still tips.
    torch.manual_seed(61)
    env.reset()
    step(60)
    t_pos, t_quat, s_dir = table_frame()
    wrong_tip = torch.tensor([-c.leg_xy, -c.leg_xy, -c.top_t / 2 - c.leg_len],
                             device=device).expand(n, 3)
    wrong_w = t_pos + quat_apply(t_quat, wrong_tip)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = wrong_w[:, 0:2] - s_dir[:, 0:2] * 0.004  # same insertion depth
    st[:, 2] = 0.001 + scene.env_origins[:, 2]
    yaw = math.atan2(float(-s_dir[0, 1]), float(-s_dir[0, 0]))
    st[:, 3], st[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
    scene.wedge.write_root_state_to_sim(st, all_ids)
    step(30)
    max_tilt[0] = 0.0
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2)
    drop_dish(scene.cup, c.seat_cup, c.cup_h / 2)
    step(600)
    report("wrong-leg")
    s, ok = judge()
    check("wrong leg: wedge under the diagonally opposite LONG (grounded) leg — "
          f"table still tips (max tilt {max_tilt[0]:.1f} deg > 3): shored() (which "
          "reads the SHORT foot) never fires, score ~0, no success (fault "
          "identification is load-bearing)",
          max_tilt[0] > 3.0 and not bool(scene._shored_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 9. ballast cheat ===========================================
    # Wedge parked ON the tabletop's far corner as a counterweight: the asserted
    # moment margin says both dishes still out-tip it — verified physically here.
    torch.manual_seed(71)
    env.reset()
    step(60)
    t_pos, t_quat, _s = table_frame()
    ballast = torch.tensor([-c.top_side / 2 + 0.045, -c.top_side / 2 + 0.045,
                            c.top_face_z + 0.012], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = t_pos + quat_apply(t_quat, ballast)
    st[:, 3:7] = t_quat
    scene.wedge.write_root_state_to_sim(st, all_ids)
    step(60)
    max_tilt[0] = 0.0
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2)
    drop_dish(scene.cup, c.seat_cup, c.cup_h / 2)
    step(600)
    report("ballast-cheat")
    s, ok = judge()
    check("ballast cheat: wedge parked ON the far corner of the top as a "
          f"counterweight — the dishes still out-tip it (max tilt {max_tilt[0]:.1f} "
          "deg > 3), no shore credit, no success (the repair cannot be replaced by "
          "counterweighting)",
          max_tilt[0] > 3.0 and not bool(scene._shored_ever[0])
          and s <= 0.02 and not ok)

    # =========================== 10. swapped dishes =========================================
    # Proper shim first (probe-constructed), then the dishes SWAPPED: plate on the
    # blue disc, cup on the red ring. Level holds — but each seat check is per-dish.
    torch.manual_seed(81)
    env.reset()
    step(60)
    construct_shim()
    ok_shim = bool(scene.shored()[0])
    drop_dish(scene.plate, c.seat_cup, c.plate_h / 2)   # plate -> cup's disc
    drop_dish(scene.cup, c.seat_plate, c.cup_h / 2)     # cup -> plate's ring
    step(480)
    report("swapped")
    s, ok = judge()
    check("swapped dishes: with a proper shim, plate dropped on the BLUE DISC and "
          "cup on the RED RING settle level — but neither is on its OWN seat: only "
          f"the shore credit remains (score {s:.3f} <= 0.32), no dish latches, no "
          "success (dish identity is load-bearing)",
          ok_shim and float(scene.tilt_deg()[0]) < c.level_tol_deg
          and not bool(scene._plate_ever[0]) and not bool(scene._cup_ever[0])
          and c.w_shore - 0.02 <= s <= c.w_shore + 0.02 and not ok)

    # =========================== 11. near-miss seat =========================================
    torch.manual_seed(91)
    env.reset()
    step(60)
    construct_shim()
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2, off=(-0.050, 0.0))
    step(480)
    report("near-miss")
    pl = dish_loc(scene.plate)
    s, ok = judge()
    check("near-miss seat: shim + plate dropped 50 mm off the red ring (toward "
          "the table centre — outside the 35 mm tolerance, still on the table) "
          "— no plate credit "
          f"(score {s:.3f} stays at the shore credit), no success",
          float(pl[2]) > 0.0 and not bool(scene._plate_ever[0])
          and c.w_shore - 0.02 <= s <= c.w_shore + 0.02 and not ok)

    # =========================== 12. partial credit + regression ============================
    # Shim + plate correctly seated -> exactly 0.50. Then the wedge is yanked away:
    # the table TIPS (the shim really was load-bearing) and sheds the plate; the
    # latched 0.50 survives, still no success.
    torch.manual_seed(101)
    env.reset()
    step(60)
    construct_shim()
    drop_dish(scene.plate, c.seat_plate, c.plate_h / 2)
    step(480)
    report("shim+plate")
    s12a, ok = judge()
    got_partial = (bool(scene._shored_ever[0]) and bool(scene._plate_ever[0])
                   and abs(s12a - (c.w_shore + c.w_plate)) < 0.01 and not ok)
    max_tilt[0] = 0.0
    park_wedge_floor()
    # WAKE the table and plate with identity re-writes of their CURRENT states:
    # a fully settled stack is asleep in PhysX, and teleporting the WEDGE away
    # does not wake the bodies it was supporting — without this the pair freezes
    # in a statically impossible pose and the probe is vacuous.
    scene.table.write_root_state_to_sim(
        scene.table.data.root_state_w.clone(), all_ids)
    scene.plate.write_root_state_to_sim(
        scene.plate.data.root_state_w.clone(), all_ids)
    step(600)
    report("wedge-yanked")
    s12b, ok = judge()
    check("partial + regression: shim + plate seated earns exactly "
          f"0.50 (got {s12a:.3f}), no success; yanking the WEDGE away tips the "
          f"table (max tilt {max_tilt[0]:.1f} deg > 3 — the shim was load-bearing) "
          f"and sheds the plate; the latched credit survives ({s12a:.3f} -> "
          f"{s12b:.3f}), still no success (non-success cap holds)",
          got_partial and max_tilt[0] > 3.0 and abs(s12b - s12a) < 1e-3
          and not bool(scene.plate_seated()[0]) and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.wobbly_bistro")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
