"""Smoke / rubric-REJECTION battery for StovePropScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real open->brace->place
construction and the latched credit is monotone along it). This battery proves the
rubric REJECTS wrong outcomes and that the claims the task rests on — gravity
always closes an unbraced lid, the two-point rod brace (base in WELL + tip in
POCKET) is the only thing that may hold the lid open, and the copper-pot placement
clauses — are load-bearing. Every probe is CONSTRUCTED as a settled state
(teleport transport, external wrench with verified non-vacuous holds, real physics
steps, judge) — instrumentation, never a solution: no probe here reaches success().

Checks:
  1.  settle/no-NaN   — seeded reset settles finite; lid CLOSED sealing the tub,
                        pots outside on the counter, score ~0, no success;
  2.  randomization   — two seeded resets: READBACK rod pose (base xy + axis
                        azimuth) and pot yaws all differ;
  3.  side coin-flip  — over 10 resets the COPPER pot spawns on BOTH sides
                        (identity is appearance, not position);
  4.  null-policy     — 240 idle steps: lid stays shut, score ~0, no success;
  5.  seed-strategy   — the copper pot set down ON TOP of the closed lid, right
                        over the burner (the seed's own move class: pot onto the
                        stove surface): settles, scores ZERO here;
  6.  anti-pinning    — an external wrench (the hand) holds the lid open and CALM
                        at 65 deg (hold verified non-vacuous): the open latch may
                        fire but NO prop credit accrues; on release gravity SLAMS
                        the lid shut — score <= 0.205, never success;
  7.  pot-wedge       — copper pot upright on the burner, lid gently lowered onto
                        it: the lid parks at ~13 deg (reads CLOSED); pot latch may
                        fire but no prop credit, no success, score <= 0.455;
  8.  wrong pot       — the REAL brace built (rod in well + pocket, hands off,
                        propped verified), then the STEEL pot placed on the
                        burner: wrong_in_tub refuses success, no pot latch,
                        score <= 0.505;
  9.  pot off-pad     — braced state restored; copper pot dropped inside the tub
                        but OFF the burner pad: no pot latch, no success;
  10. pot tipped      — braced state restored; copper pot lying on its SIDE on
                        the pad: upright clause refuses, no pot latch, no success;
  11. pot-prop cheat  — the steel pot wedged as close to the hinge as it fits and
                        the lid lowered onto it: the pot visibly holds the lid
                        off the tub, but the rod clauses (well + pocket) refuse
                        all prop credit — score <= 0.205, no success;
  12. strut-beside    — the SAME rod braced 6 cm beside the well/pocket line (base
                        on bare floor, tip on bare panel): with no fence catch the
                        loaded tip slides and the strut COLLAPSES — the lid falls
                        shut, no prop latch, score <= 0.205;
  13. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i167.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=24)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects
# RTX -> the annotator returns EMPTY frames. Disable the check.
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

_qy = task_scene._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOLD_DEG = 68.0
G = 9.81

# ----- module state wired up in main() ----------------------------------------------------------
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
    print(f"[smoke] {tag:16s} | lid={math.degrees(float(scene.lid_angle()[0])):+6.2f}deg "
          f"tip={bool(scene.tip_in_pocket()[0])} well={bool(scene.base_in_well()[0])} "
          f"propped={bool(scene.propped()[0])} pot={bool(scene.pot_on_burner()[0])} "
          f"wrong_in_tub={bool(scene.wrong_in_tub()[0])} "
          f"latches=({int(scene._l_open[0])},{int(scene._l_prop[0])},{int(scene._l_pot[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main -------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stove_prop")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.05, 0.92)) + o),
                                tuple(np.array((0.38, 0.00, 0.15)) + o),
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

    # ----- shared machinery (mirrors solve.py) ---------------------------------------------------
    zero3 = torch.zeros(n, 1, 3, device=device)
    r_com = c.lid_len / 2

    def lid_wrench(theta_des: float, scale: float = 1.0, kp: float = 2.0,
                   kd: float = 0.35) -> None:
        th = scene.lid_angle()
        om_open = -scene.lid.data.root_ang_vel_w[:, 1]
        tau_open = (c.lid_mass * G * r_com * torch.cos(th)
                    + kp * (theta_des - th) - kd * om_open).clamp(-0.2, 1.5)
        t3 = zero3.clone()
        t3[:, 0, 1] = -tau_open * scale
        scene.lid.set_external_force_and_torque(zero3, t3)
        _step(1)

    def drop_wrench() -> None:
        scene.lid.set_external_force_and_torque(zero3, zero3)

    def open_lid(tgt_deg: float, ramp: int = 300, hold: int = 240) -> None:
        th0 = float(scene.lid_angle()[0])
        tgt = math.radians(tgt_deg)
        for i in range(ramp):
            lid_wrench(th0 + (tgt - th0) * (i + 1) / ramp)
        for _ in range(hold):
            lid_wrench(tgt)

    def lower_and_release(tgt_low: float, ramp: int = 360, bleed: int = 360,
                          settle: int = 360) -> None:
        th_now = float(scene.lid_angle()[0])
        for i in range(ramp):
            lid_wrench(th_now + (tgt_low - th_now) * (i + 1) / ramp)
        for i in range(bleed):
            lid_wrench(tgt_low, scale=1.0 - (i + 1) / bleed)
        drop_wrench()
        _step(settle)

    def place_rod(y_off: float = 0.0) -> None:
        """Rod base hovering 1 mm above the (well) floor, tip aimed just inside
        the held-open lid's pocket line, offset `y_off` sideways."""
        tp = scene.tub.data.root_pos_w
        base = tp.clone()
        base[:, 0] += c.well_x
        base[:, 1] += c.well_y + y_off
        base[:, 2] += c.floor_t + 0.001
        pk = torch.tensor([c.pocket_r, c.pocket_y + y_off, -0.006],
                          device=device).expand(n, 3)
        tip_tgt = scene.lid.data.root_pos_w + quat_apply(scene.lid.data.root_quat_w, pk)
        d = tip_tgt - base
        d = d / d.norm(dim=-1, keepdim=True)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        v = torch.cross(ez, d, dim=-1)
        w = 1.0 + d[:, 2:3]
        q = torch.cat([w, v], dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        _write_body(scene.rod, base, q)

    def build_brace() -> bool:
        """The solve's construction: open, stand the rod, lower, release."""
        open_lid(HOLD_DEG)
        for _attempt in range(3):
            place_rod()
            for _ in range(240):
                lid_wrench(math.radians(HOLD_DEG))
            _refresh()
            if not (bool(scene.base_in_well()[0]) and bool(scene.tip_in_pocket()[0])):
                continue
            lower_and_release(c.catch_angle + math.radians(0.5))
            _refresh()
            if bool(scene.propped()[0]):
                return True
            open_lid(HOLD_DEG, ramp=240, hold=120)
        return False

    def tub_spot(x: float, y: float, dz: float) -> torch.Tensor:
        """World point at tub-local xy, `dz` above the tub floor top."""
        p = scene.tub.data.root_pos_w.clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += c.floor_t + dz
        return p

    def lid_top_spot(x: float, y: float) -> torch.Tensor:
        """World point just above the CLOSED lid's top surface at tub-local xy."""
        p = scene.tub.data.root_pos_w.clone()
        p[:, 0] += x
        p[:, 1] += y
        p[:, 2] += c.hinge_z + c.lid_t + 0.002
        return p

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def rod_azimuth() -> float:
        u = quat_apply(scene.rod.data.root_quat_w,
                       torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0]
        return math.degrees(math.atan2(float(u[1]), float(u[0])))

    origins = env.iscene.env_origins

    # ================= 1. settle / no-NaN (closed lid seals the tub) ============================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.lid.data.root_pos_w).all()
               and torch.isfinite(scene.rod.data.root_pos_w).all()
               and torch.isfinite(scene.pot_t.data.root_lin_vel_w).all())
    ang = abs(math.degrees(float(scene.lid_angle()[0])))
    check("settle/no-NaN: lid CLOSED sealing the tub, rod and pots on the counter, "
          "score ~0, no success",
          fin and ang < 3.0 and not bool(scene.wrong_in_tub()[0])
          and not bool(scene.pot_on_burner()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        return (scene.rod.data.root_pos_w[0, :2] - origins[0, :2],
                rod_azimuth(),
                yaw_of(scene.pot_t.data.root_quat_w[0]),
                yaw_of(scene.pot_w.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_r, a_az, a_t, a_w = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_r, b_az, b_t, b_w = readback()
    d_r = float((a_r - b_r).norm())
    d_az = dyaw(a_az, b_az)
    d_t, d_w = dyaw(a_t, b_t), dyaw(a_w, b_w)
    print(f"[smoke] randomization deltas: rod_base={d_r * 1000:.1f}mm "
          f"rod_azim={d_az:.1f}deg pot_t_yaw={d_t:.1f}deg pot_w_yaw={d_w:.1f}deg",
          flush=True)
    check("randomization-is-real: rod base xy, rod axis azimuth and both pot "
          "yaws readback differ",
          d_r > 0.003 and d_az > 2.0 and d_t > 5.0 and d_w > 5.0)

    # ================= 3. copper-side coin flip =================================================
    sides = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.append(1 if float(scene.pot_t.data.root_pos_w[0, 1]
                                - origins[0, 1]) > 0 else -1)
    print(f"[smoke] copper-pot sides over 10 resets: {sides}", flush=True)
    check("side coin-flip: the copper pot spawns on BOTH sides over 10 resets "
          "(identity by appearance, not position)",
          1 in sides and -1 in sides)

    # ================= 4. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, lid stays shut, score ~0, no success",
          abs(math.degrees(float(scene.lid_angle()[0]))) < 3.0
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. seed strategy: pot onto the (closed) stove top ========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.pot_t, lid_top_spot(c.pad_pos[0], c.pad_pos[1]))
    _step(300)
    _report("pot-on-lid")
    _REC["on"] = False
    check("seed-strategy rejected: copper pot set down on TOP of the closed lid, "
          "right over the burner — score ~0, no success",
          not bool(scene.pot_on_burner()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 6. anti-pinning: wrench-held lid, then slam shut =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    open_lid(65.0)
    _refresh()
    held_ang = math.degrees(float(scene.lid_angle()[0]))
    held = held_ang > 55.0 and bool(scene.lid_calm()[0])
    no_prop_while_held = not bool(scene.propped()[0]) and not bool(scene._l_prop[0])
    no_success_while_held = not bool(scene.success()[0])
    _report("wrench-held")
    _REC["on"] = True
    drop_wrench()
    _step(480)
    _report("slammed")
    _REC["on"] = False
    end_ang = math.degrees(float(scene.lid_angle()[0]))
    check("anti-pinning: wrench holds the lid open+calm (verified non-vacuous) — "
          "open latch only, NO prop credit; released lid slams shut, "
          "score <= 0.205, never success",
          held and no_prop_while_held and no_success_while_held
          and bool(scene._l_open[0]) and end_ang < 10.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.205)

    # ================= 7. pot-wedge: lid resting on the burner pot ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    open_lid(HOLD_DEG)
    _write_body(scene.pot_t, tub_spot(c.pad_pos[0], c.pad_pos[1], c.pad_h + 0.008))
    for _ in range(240):
        lid_wrench(math.radians(HOLD_DEG))
    _refresh()
    pot_seated = bool(scene.pot_on_burner()[0])
    lower_and_release(math.radians(16.0), ramp=480, bleed=240, settle=480)
    _report("pot-wedge")
    wedge_ang = math.degrees(float(scene.lid_angle()[0]))
    check("pot-wedge: lid lowered onto the burner pot parks at ~13 deg (reads "
          "CLOSED) — no prop credit, no success, score <= 0.455",
          pot_seated and wedge_ang < c.open_min_deg - 25.0
          and not bool(scene._l_prop[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.455)

    # ================= 8. real brace + WRONG pot on the burner ==================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    braced = build_brace()
    _report("braced")
    assert braced, "smoke could not reproduce the brace (solve-verified path)"
    state_braced = scene.get_state(_all_ids())
    _write_body(scene.pot_w, tub_spot(c.pad_pos[0], c.pad_pos[1], c.pad_h + 0.008))
    _step(360)
    _report("wrong-pot")
    _REC["on"] = False
    check("wrong-pot rejected: brace verified propped, STEEL pot on the burner — "
          "wrong_in_tub refuses success, no pot latch, score <= 0.505",
          bool(scene.propped()[0]) and bool(scene.wrong_in_tub()[0])
          and not bool(scene.pot_on_burner()[0]) and not bool(scene._l_pot[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.505)

    # ================= 9. braced, but copper pot OFF the pad ====================================
    scene.set_state(state_braced, _all_ids())
    _refresh()
    _write_body(scene.pot_t, tub_spot(0.09, -0.05, 0.008))
    _step(360)
    _report("pot-off-pad")
    check("off-pad rejected: braced lid, copper pot at rest INSIDE the tub but "
          "off the burner pad — no pot latch, no success, score <= 0.505",
          bool(scene.propped()[0]) and not bool(scene.pot_on_burner()[0])
          and not bool(scene._l_pot[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.505)

    # ================= 10. braced, but copper pot TIPPED on the pad =============================
    scene.set_state(state_braced, _all_ids())
    _refresh()
    _write_body(scene.pot_t, tub_spot(c.pad_pos[0], c.pad_pos[1], c.pad_h + 0.036),
                _qy(torch.full((n,), math.pi / 2, device=device)))
    _step(360)
    _report("pot-tipped")
    check("tipped rejected: copper pot lying on its SIDE on the pad — upright "
          "clause refuses, no pot latch, no success, score <= 0.505",
          not bool(scene.pot_on_burner()[0]) and not bool(scene._l_pot[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.505)

    # ================= 11. pot-prop cheat: steel pot wedged near the hinge ======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    open_lid(76.0)
    _write_body(scene.pot_w, tub_spot(-0.102, 0.0, 0.008))
    for _ in range(240):
        lid_wrench(math.radians(76.0))
    _REC["on"] = True
    # lower until the lid meets the pot (lag detection), then bleed off
    tgt0, tgt1 = math.radians(76.0), math.radians(35.0)
    for i in range(600):
        tgt = tgt0 + (tgt1 - tgt0) * (i + 1) / 600
        lid_wrench(tgt)
        if float(scene.lid_angle()[0]) - tgt > math.radians(6.0):
            break
    th_rest = float(scene.lid_angle()[0])
    for i in range(240):
        lid_wrench(th_rest, scale=1.0 - (i + 1) / 240)
    drop_wrench()
    _step(480)
    _report("pot-prop")
    _REC["on"] = False
    cheat_ang = math.degrees(float(scene.lid_angle()[0]))
    print(f"[smoke] pot-prop rest angle: {cheat_ang:.1f} deg", flush=True)
    check("pot-prop cheat: steel pot wedged at the hinge visibly holds the lid "
          "off the tub (verified) — rod clauses refuse all prop credit, "
          "score <= 0.205, no success",
          35.0 - 3.0 < cheat_ang < 60.0 and not bool(scene._l_prop[0])
          and not bool(scene.propped()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.205)

    # ================= 12. strut beside the well/pocket line ====================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    open_lid(HOLD_DEG)
    place_rod(y_off=0.06)
    for _ in range(240):
        lid_wrench(math.radians(HOLD_DEG))
    _refresh()
    aside_ok = not bool(scene.tip_in_pocket()[0]) and not bool(scene.base_in_well()[0])
    lower_and_release(c.catch_angle + math.radians(0.5), settle=600)
    _report("strut-beside")
    beside_ang = math.degrees(float(scene.lid_angle()[0]))
    check("strut-beside rejected: same rod braced 6 cm beside the well/pocket "
          "line — no fence catch, the strut collapses (lid falls shut), no prop "
          "latch, score <= 0.205",
          aside_ok and beside_ang < c.open_min_deg
          and not bool(scene._l_prop[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.205)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stove_prop")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
