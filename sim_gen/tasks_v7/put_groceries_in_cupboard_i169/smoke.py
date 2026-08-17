"""Smoke / rubric-REJECTION battery for CarouselCupboardScene — NullRobot, teleported probes.

solve.py is the acceptance proof (knob-torque the carousel into alignment,
force-carry the carton in through the window, knob-torque the loaded bay behind
the wall). This battery proves the rubric REJECTS wrong outcomes and that the
CLOSED CUPBOARD — the task's strategic differentiator from the seed — is
physically load-bearing: there is no passive shelf that scores, and the wall
really arrests a carton pressed at an unaligned bay. Every probe is CONSTRUCTED
as a settled state (teleport, real physics steps, judge); constructed partial
states may earn latched partial credit but none may reach success() unless the
carton genuinely ends stocked AND stowed.

Checks:
   1. settle/no-NaN    — seeded reset settles finite; the carousel holds its
                         random bearing on the damped free pivot, carton upright
                         on its stand, cans seated; score ~0, no success;
   2. randomization    — two seeded resets: READBACK cabinet yaw, cabinet xy,
                         empty-bay bearing and stand xy all differ;
   3. bay shuffle      — across seeds the empty-bay index takes >= 2 values and
                         BOTH decoy-to-bay assignments occur; the stored indices
                         match the turntable-local can readbacks and the empty
                         bay is physically empty;
   4. null-policy      — 240 idle steps: bearing holds, score ~0, no success;
   5. SEED STRATEGY    — the seed's whole plan ("set the grocery down on the
                         cupboard shelf"): the carton set down on the ROOF rests
                         there and scores no success; dropped at the window
                         mouth on the ground it also scores no success — there
                         is no passive put-it-down surface;
   6. wall-load-bearing— with the empty bay still rotated away, a real force
                         press (the solve's own carry gains) drives the carton
                         radially at its bay from outside: the probe MOVES,
                         then ARRESTS on the wall outside the interior — no
                         entry, no credit;
   7. wrong bay        — carton constructed settled in an OCCUPIED bay (beside
                         the resident can), carousel stowed: no success;
   8. not stowed       — carton constructed loaded in the empty bay with the
                         bay still ALIGNED to the window (the natural stopping
                         point of a no-stow policy): no success, score <= 0.70;
   9. latch regression — from state 8, the carton is removed to the ground:
                         latched partial credit is kept (score unchanged) but
                         success stays False — latches never fake an outcome;
  10. stow boundary    — same loaded construction at ~85 deg (< stow_min 100):
                         still not stowed, no success;
  11. toppled          — carton lying on its SIDE in the stowed empty bay:
                         upright/z-band reject it, no success;
  12. decoy displaced  — carton perfectly stocked and stowed but the blue can
                         dumped out of the cupboard: no success;
  13. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.put_groceries_in_cupboard_i169.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qz, _wrap_deg = task_scene._qmul, task_scene._qz, task_scene._wrap_deg

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    b = float(scene.target_bearing_deg()[0])
    loc = scene._tt_local(scene.carton.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | bearing={b:+7.2f}deg "
          f"carton_tt=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
          f"in_target={bool(scene.carton_in_target()[0])} "
          f"stowed={bool(scene.stowed()[0])} decoys={bool(scene.decoys_home()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_cupboard")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.90, 0.95)) + o),
                                tuple(np.array((0.50, 0.00, 0.25)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def bearing() -> float:
        _refresh()
        return float(scene.target_bearing_deg()[0])

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def cab_local(pos_w: torch.Tensor) -> torch.Tensor:
        _refresh()
        return quat_apply_inverse(scene.cabinet.data.root_quat_w,
                                  pos_w - scene.cabinet.data.root_pos_w)

    def set_bearing(deg: float) -> None:
        """Teleport the carousel so the EMPTY bay's bearing is `deg` — the
        turntable origin sits ON the pivot axis, so this is a joint-consistent
        pure yaw write (zero velocity)."""
        e = float(scene.empty_bay[0])
        phi = math.radians(deg - 120.0 * e)
        q = _qmul(scene.cabinet.data.root_quat_w,
                  _qz(torch.full((n,), phi, device=device)))
        _write_body(scene.turntable, scene.cabinet.data.root_pos_w, q)

    def put_tt_local(body, loc_xyz, extra_quat=None) -> None:
        """Teleport a body to a turntable-local point (orientation = turntable
        yaw, optionally composed with extra_quat on the right)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.turntable.data.root_pos_w \
            + quat_apply(scene.turntable.data.root_quat_w, loc)
        q = scene.turntable.data.root_quat_w
        if extra_quat is not None:
            q = _qmul(q, extra_quat.expand(n, 4))
        _write_body(body, pos, q)

    def put_cab_local(body, loc_xyz, quat=None) -> None:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.cabinet.data.root_pos_w \
            + quat_apply(scene.cabinet.data.root_quat_w, loc)
        _write_body(body, pos, quat if quat is not None else scene.cabinet.data.root_quat_w)

    def bay_xy(bay: float, r: float, off_deg: float = 0.0) -> tuple:
        a = math.radians(120.0 * bay + off_deg)
        return r * math.cos(a), r * math.sin(a)

    def press_toward(tgt_local_tt, *, steps: int, clamp: float = 6.0) -> None:
        """Real force press on the carton toward a turntable-local target (the
        solve's own carry gains): PD + gravity feedforward + righting torque.
        Wrench zeroed afterwards."""
        no_action = torch.empty(0, device=device)
        tgt = torch.tensor(tgt_local_tt, device=device, dtype=torch.float).expand(n, 3)
        m = float(scene.carton.root_physx_view.get_masses()[0].sum())
        for _ in range(steps):
            _refresh()
            q = scene.carton.data.root_quat_w
            p = scene.carton.data.root_pos_w
            v = scene.carton.data.root_lin_vel_w
            w = scene.carton.data.root_ang_vel_w
            tgt_w = scene.turntable.data.root_pos_w \
                + quat_apply(scene.turntable.data.root_quat_w, tgt)
            f_w = m * 9.81 * ez + 15.0 * (tgt_w - p) - 6.0 * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            axis = quat_apply(q, ez)
            w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
            t_w = 0.03 * torch.cross(axis, ez, dim=-1) - 0.012 * w_perp
            t_b = quat_apply_inverse(q, t_w)
            scene.carton.set_external_force_and_torque(f_b.reshape(n, 1, 3),
                                                       t_b.reshape(n, 1, 3))
            rec = _REC["on"] and _REC["annot"] is not None
            env.step(no_action, render=rec)
            if rec and _REC["i"] % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(_REC["annot"].get_data())
                if arr.size:
                    _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
            _REC["i"] += 1
        scene.carton.set_external_force_and_torque(zero, zero)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    b0 = float(scene.bearing0[0])
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    up_z = float(quat_apply(scene.carton.data.root_quat_w, ez)[0, 2])
    carton_z = float(scene.carton.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    check("settle/no-NaN: layout settles finite; the carousel holds its random "
          "bearing on the damped pivot, carton upright on its stand, cans seated; "
          "score ~0, no success",
          bool(scene._finite()[0]) and abs(bearing() - b0) < 8.0
          and up_z > 0.95 and 0.14 < carton_z < 0.22
          and bool(scene.decoys_home()[0])
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        cyaw = yaw_of(scene.cabinet.data.root_quat_w[0])
        cp = scene.cabinet.data.root_pos_w[0, :2].clone()
        sp = scene.stand.data.root_pos_w[0, :2].clone()
        return cyaw, cp, float(scene.bearing0[0]), sp

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_b0, a_sp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_b0, b_sp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_cp = float((a_cp - b_cp).norm())
    d_b0 = abs(a_b0 - b_b0)
    d_sp = float((a_sp - b_sp).norm())
    print(f"[smoke] randomization deltas: cab_yaw={d_yawv:.1f}deg "
          f"cab_xy={d_cp * 1000:.1f}mm bearing0={d_b0:.1f}deg "
          f"stand_xy={d_sp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: cabinet yaw, cabinet xy, empty-bay bearing and "
          "stand position readback all differ across seeds",
          d_yawv > 1.5 and d_cp > 0.003 and d_b0 > 3.0 and d_sp > 0.02)

    # ================= 3. randomization: bay shuffle occurs and reads back ========================
    empties = set()
    swaps = set()
    consistent = True
    for sd in range(300, 312):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        _refresh()
        e = int(scene.empty_bay[0])
        bb, gb = int(scene.blue_bay[0]), int(scene.green_bay[0])
        empties.add(e)
        swaps.add(bb == (e + 1) % 3)
        consistent &= {e, bb, gb} == {0, 1, 2}
        # can positions must physically match their assigned bays, and the
        # empty bay must be physically empty
        for body, bay in ((scene.can_blue, bb), (scene.can_green, gb)):
            loc = scene._tt_local(body.data.root_pos_w)[0]
            ang = math.degrees(math.atan2(float(loc[1]), float(loc[0])))
            d = abs((ang - 120.0 * bay + 180.0) % 360.0 - 180.0)
            de = abs((ang - 120.0 * e + 180.0) % 360.0 - 180.0)
            consistent &= d < 10.0 and de > 60.0
    print(f"[smoke] shuffle over seeds 300-311: empty bays={sorted(empties)} "
          f"swap variants={len(swaps)} consistent={consistent}", flush=True)
    check("randomization (bay shuffle): the empty-bay index takes >= 2 values, "
          "both decoy assignments occur, and the indices match the turntable-"
          "local can readbacks (empty bay physically empty)",
          len(empties) >= 2 and len(swaps) == 2 and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the carousel keeps its bearing, "
          "carton stays on the stand, score ~0, no success",
          abs(bearing() - b0) < 8.0 and float(scene.score()[0]) <= 0.03
          and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the grocery in the cupboard" seed-style: grasp it and SET IT DOWN on
    # the cupboard. The only set-down surfaces here are the ROOF and the ground
    # at the window mouth — construct both settled rest states; neither scores.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_cab_local(scene.carton, (0.20, 0.12, 0.42))  # clear of the crank sweep
    _step(240)
    _report("seed-roof")
    zloc = float(cab_local(scene.carton.data.root_pos_w)[0, 2])
    roof_rest = abs(zloc - (c.roof_z1 + c.carton_h / 2)) < 0.03
    roof_no = not bool(scene.success()[0]) \
        and float(scene.score()[0]) <= c.w_lift + 1e-4  # teleport height latches `lifted` only
    put_cab_local(scene.carton, (0.33, 0.0, c.carton_h / 2 + 0.003))
    _step(180)
    _report("seed-ground")
    ground_no = not bool(scene.success()[0]) \
        and not bool(scene.carton_in_target()[0])
    _REC["on"] = False
    check("negative (SEED strategy): the carton set down on the cupboard ROOF "
          "rests there without success, and left on the ground at the window "
          "mouth it also scores nothing — no passive shelf exists",
          roof_rest and roof_no and ground_no)

    # ================= 6. wall-is-load-bearing (real force, arrested) =============================
    # Empty bay still rotated away (fresh reset): press the carton radially at
    # ITS bay from outside with the solve's own carry gains. The probe must
    # MOVE (approach the wall) and then ARREST outside the interior.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _refresh()
    beta = bearing()  # empty-bay direction in cabinet frame (|beta| in 75..170)
    br = math.radians(beta)
    start = (0.30 * math.cos(br), 0.30 * math.sin(br), 0.21)
    q_face = _qmul(scene.cabinet.data.root_quat_w,
                   _qz(torch.full((n,), br, device=device)))
    put_cab_local(scene.carton, start, q_face)
    _step(30)
    r_start = float(cab_local(scene.carton.data.root_pos_w)[0, :2].norm())
    _REC["on"] = True
    e = float(scene.empty_bay[0])
    press_toward((*bay_xy(e, c.slot_r), 0.19), steps=300)
    _step(90)
    _report("wall-arrest")
    _REC["on"] = False
    r_end = float(cab_local(scene.carton.data.root_pos_w)[0, :2].norm())
    moved = r_start - r_end
    print(f"[smoke] wall probe: r {r_start:.3f} -> {r_end:.3f} m "
          f"(moved {moved * 1000:.0f} mm inward; wall outer face at "
          f"{c.wall_r + c.wall_t / 2:.3f})", flush=True)
    check("wall-is-load-bearing: a real 6 N press drives the carton inward at "
          "its (unaligned) bay — it moves, then arrests on the wall outside "
          "the interior; no entry, no credit, no success",
          moved > 0.015 and r_end > c.wall_r - 0.01
          and not bool(scene.carton_in_target()[0])
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 7. negative: WRONG BAY =====================================================
    # Carton settled in an OCCUPIED bay (beside its resident can), carousel
    # stowed: in_bay holds for the WRONG bay — the rubric must reject it.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_bearing(150.0)
    _step(30)
    e = float(scene.empty_bay[0])
    o = (e + 1) % 3
    put_tt_local(scene.carton, (*bay_xy(o, 0.14, 35.0),
                                c.disc_top + c.carton_h / 2 + 0.003))
    _step(180)
    _report("wrong-bay")
    check("negative (wrong bay): carton settled upright in an OCCUPIED bay "
          "beside the resident can, carousel stowed — no success",
          bool(scene.in_bay(scene.carton,
                            torch.full_like(scene.empty_bay, int(o)))[0])
          and not bool(scene.carton_in_target()[0])
          and bool(scene.decoys_home()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.03)

    # ================= 8. near-miss: loaded but NOT stowed ========================================
    # The natural stopping point of a policy that aligns and loads but never
    # stows: bay aligned to the window, carton upright inside. Partial credit
    # latches (aligned + loaded) but success must stay False.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_bearing(0.0)
    _step(30)
    e = float(scene.empty_bay[0])
    put_tt_local(scene.carton, (*bay_xy(e, c.slot_r),
                                c.disc_top + c.carton_h / 2 + 0.003))
    _step(180)
    _report("not-stowed")
    _REC["on"] = False
    s8 = float(scene.score()[0])
    check("near-miss (not stowed): carton loaded in the empty bay with the bay "
          "still facing the window — in_target holds, stowed does not, no "
          "success, score <= 0.70",
          bool(scene.carton_in_target()[0]) and not bool(scene.stowed()[0])
          and not bool(scene.success()[0]) and s8 <= 0.7000005)

    # ================= 9. latch regression: credit is memory, success is live ====================
    # Remove the carton from state 8 to the open ground: the latched credit
    # must survive (score unchanged) while success stays False.
    put_cab_local(scene.carton, (0.0, -0.60, c.carton_h / 2 + 0.003))
    _step(120)
    _report("latch-regress")
    s9 = float(scene.score()[0])
    check("latch-regression: carton removed to the ground after loading — "
          "latched partial credit is kept (score unchanged) but success and "
          "in_target read the live state: False",
          abs(s9 - s8) < 1e-4 and not bool(scene.carton_in_target()[0])
          and not bool(scene.success()[0]))

    # ================= 10. near-miss: stow boundary (85 < 100) ====================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_bearing(85.0)
    _step(30)
    e = float(scene.empty_bay[0])
    put_tt_local(scene.carton, (*bay_xy(e, c.slot_r),
                                c.disc_top + c.carton_h / 2 + 0.003))
    _step(180)
    _report("boundary-85")
    check("near-miss (stow boundary): carton loaded, bay rotated only ~85 deg "
          "(< stow_min 100) — not stowed, no success, score <= 0.70",
          bool(scene.carton_in_target()[0]) and not bool(scene.stowed()[0])
          and abs(abs(bearing()) - 85.0) < 8.0
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 11. negative: toppled carton in the stowed bay =============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_bearing(150.0)
    _step(30)
    e = float(scene.empty_bay[0])
    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device)
    put_tt_local(scene.carton, (*bay_xy(e, c.slot_r),
                                c.disc_top + c.carton_w / 2 + 0.003), qy90)
    _step(180)
    _report("toppled")
    up_t = float(quat_apply(scene.carton.data.root_quat_w, ez)[0, 2])
    check("negative (toppled): carton lying on its side in the stowed empty "
          "bay — upright and z-band reject it, no success",
          up_t < 0.5 and not bool(scene.carton_in_target()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.03)

    # ================= 12. negative: decoy displaced ==============================================
    # Displace the blue can FIRST, then construct the otherwise-perfect stocked
    # and stowed carton (never passing through a true success state): the
    # missing decoy must veto success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_cab_local(scene.can_blue, (0.0, 0.55, c.can_h / 2 + 0.003))
    _step(60)
    set_bearing(150.0)
    _step(30)
    e = float(scene.empty_bay[0])
    put_tt_local(scene.carton, (*bay_xy(e, c.slot_r),
                                c.disc_top + c.carton_h / 2 + 0.003))
    _step(180)
    _report("decoy-out")
    check("negative (decoy displaced): carton perfectly stocked AND stowed but "
          "the blue can dumped out of the cupboard — decoys_home vetoes, no "
          "success",
          bool(scene.carton_in_target()[0]) and bool(scene.stowed()[0])
          and not bool(scene.decoys_home()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_cupboard")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
