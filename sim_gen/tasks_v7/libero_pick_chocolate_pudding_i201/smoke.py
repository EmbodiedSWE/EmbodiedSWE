"""Smoke / rubric-REJECTION battery for ShuttleVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the vault is sealed
against direct drops, the pocket conveys, the hole deposits — are load-bearing. Every
probe is CONSTRUCTED (teleport as transport/instrumentation, real physics steps,
judge); success() is monitored at EVERY step and must never turn True anywhere in the
battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; pudding + distractors OUTSIDE
                           the vault, shuttle on its rail; score 0, no success;
   2. randomization      — two seeded resets: READBACK pudding/ketchup/milk xy + yaw
                           and the shuttle rail coordinate all differ;
   3. rail coverage      — over 8 resets the shuttle start coordinate spans a real
                           range (and home is NOT guaranteed at spawn);
   4. null-policy        — 300 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed family's naive plan ("put the pudding into the
                           open receptacle"): pudding dropped through the window into
                           the HOME pocket — the only open-from-above vessel — and
                           left there, settled: no success, score <= 0.20 (loaded
                           latch only; the chamber is not fed);
   6. sealing (roof)     — pudding dropped from above DIRECTLY over the drop hole's
                           xy: it lands ON the roof — the offset apertures leave no
                           straight path; not in the vault, score 0, no success;
   7. sealing (offset)   — shuttle parked at B, pudding dropped through the window:
                           it lands on the GALLERY FLOOR at A (the pocket is away, the
                           chamber unreachable) — not in chamber, not loaded, score 0;
   8. near-miss convey   — pudding loaded into the home pocket by gravity, shuttle
                           FORCE-pushed only to mid-rail: loaded+conveyed latch
                           (score ~0.35) but no deposit, no success;
   9. MECHANISM deposit  — the same push continued to B: the pocket DRAGS the pudding
                           across the gallery floor and gravity drops it through the
                           hole into the chamber (rail moved, pudding conveyed,
                           chamber entered — all by readback);
  10. not-home rejection — that state (pudding deposited, distractors out, settled)
                           with the shuttle still at B: no success, score <= 0.65;
  11. latched credit     — pudding teleported back out of the chamber: deposit is no
                           longer live but the latched credit survives (score still
                           >= 0.65), no success;
  12. wrong object       — the MILK cube resting in the chamber instead (shuttle
                           home): no success, and the pudding-specific latches give
                           score ~0;
  13. contamination      — pudding AND milk both in the chamber, shuttle home,
                           settled — every clause but the distractor one holds: no
                           success, score <= 0.75;
  14. settle gate        — the exact success state but the pudding still sliding at
                           ~0.4 m/s: success refuses while anything moves (probe
                           dismantled before it can settle into a real success);
  15. rejection audit    — success() was never True at any step of this battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -u smoke.py --headless
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

_qapply, _qz, encode_force = task_scene._qapply, task_scene._qz, task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    pl = scene._housing_local(scene.pudding.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | q={float(scene.shuttle_q()[0]):+.4f} "
          f"home={bool(scene.home()[0])} "
          f"pud_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):+.3f}) "
          f"gal={bool(scene.in_gallery(scene.pudding)[0])} "
          f"pock={bool(scene.in_pocket(scene.pudding)[0])} "
          f"cham={bool(scene.in_chamber(scene.pudding)[0])} "
          f"clear={bool(scene.distractors_clear()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shuttle_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)

    def housing_pt(loc) -> torch.Tensor:
        _refresh()
        loc_t = torch.tensor(loc, device=device, dtype=torch.float).expand(n, 3)
        return scene.housing.data.root_pos_w + _qapply(scene.housing.data.root_quat_w,
                                                       loc_t)

    def hquat() -> torch.Tensor:
        _refresh()
        return scene.housing.data.root_quat_w.clone()

    def write_shuttle_q(q: float) -> None:
        """Park the shuttle at rail coordinate q (transport along its own rail)."""
        _write_body(scene.shuttle, housing_pt((c.x_a + q, 0.0, c.rail_z)), hquat())

    def drop_pudding_at(loc_xy, body=None) -> None:
        """Gravity-drop a body from a free hover above the roof at housing-local xy."""
        _write_body(body if body is not None else scene.pudding,
                    housing_pt((loc_xy[0], loc_xy[1], c.roof_z1 + 0.065)), hquat())
        _step(150)

    def push_shuttle(q_tgt: float, tag: str, timeout: int = 900,
                     tol: float = 0.006) -> None:
        """FORCE-push the shuttle along its rail toward q_tgt (the mechanism probe:
        a knob wrench, velocity-regulated, mode-probed — never a teleport)."""
        mode = 0
        q_ref = scene.shuttle.data.root_quat_w.clone()
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        floor_f = 0.4
        win_i, win_err = 0, abs(q_tgt - float(scene.shuttle_q()[0]))
        for i in range(timeout):
            q = float(scene.shuttle_q()[0])
            err = q_tgt - q
            if abs(err) <= tol:
                break
            sgn = 1.0 if err > 0 else -1.0
            xhat = _qapply(scene.housing.data.root_quat_w, ex)
            v_along = float((scene.shuttle.data.root_lin_vel_w * xhat).sum(dim=-1)[0])
            v_des = sgn * (0.10 if abs(err) > 0.030 else 0.04)
            f_along = 4.0 * (v_des - v_along)
            if abs(v_along) < 0.02 and f_along * sgn < floor_f:
                f_along = sgn * floor_f
            f_along = max(-6.0, min(6.0, f_along))
            f_arg = encode_force(mode, q_ref, scene.shuttle.data.root_quat_w,
                                 xhat * f_along)
            scene.shuttle.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            if i - win_i >= 45:
                e = abs(q_tgt - float(scene.shuttle_q()[0]))
                if e > win_err + 0.004:
                    mode = 1 - mode
                    print(f"[smoke] {tag}: moving away; force-frame mode -> {mode}",
                          flush=True)
                elif e > win_err - 0.002:
                    floor_f = min(floor_f + 0.3, 2.5)
                win_i, win_err = i, e
        scene.shuttle.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        _step(45)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.70, 0.55)) + o),
                                tuple(np.array((0.42, 0.00, 0.10)) + o),
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

    vb = scene._vault_build
    print(f"[smoke] vault build pose: ({vb[0]:+.3f},{vb[1]:+.3f},yaw={vb[2]:+.3f}rad)",
          flush=True)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    q0 = float(scene.shuttle_q()[0])
    check("settle/no-NaN: layout settles finite; pudding + distractors OUTSIDE the "
          "vault, shuttle on its rail; score 0, no success",
          bool(scene._finite()[0]) and not bool(scene.in_vault(scene.pudding)[0])
          and bool(scene.distractors_clear()[0]) and -0.005 <= q0 <= c.travel + 0.005
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.pudding.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pudding.data.root_quat_w[0]),
                scene.ketchup.data.root_pos_w[0, :2].clone(),
                scene.milk.data.root_pos_w[0, :2].clone(),
                float(scene.shuttle_q()[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_pp, a_py, a_kp, a_mp, a_q = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_pp, b_py, b_kp, b_mp, b_q = readback()
    d_pp, d_kp = float((a_pp - b_pp).norm()), float((a_kp - b_kp).norm())
    d_mp, d_py = float((a_mp - b_mp).norm()), dyaw(a_py, b_py)
    d_q = abs(a_q - b_q)
    print(f"[smoke] randomization deltas: pudding_xy={d_pp * 1000:.1f}mm "
          f"pudding_yaw={d_py:.1f}deg ketchup_xy={d_kp * 1000:.1f}mm "
          f"milk_xy={d_mp * 1000:.1f}mm shuttle_q={d_q * 1000:.1f}mm", flush=True)
    check("randomization-is-real: pudding xy+yaw, ketchup xy, milk xy and the shuttle "
          "rail coordinate readback all differ",
          d_pp > 0.003 and d_py > 2.0 and d_kp > 0.003 and d_mp > 0.003 and d_q > 0.005)

    # ================= 3. rail coverage ===========================================================
    qs = []
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        qs.append(float(scene.shuttle_q()[0]))
    print(f"[smoke] shuttle q over 8 resets: min={min(qs):.3f} max={max(qs):.3f}",
          flush=True)
    check("rail coverage: over 8 resets the shuttle start coordinate spans a real "
          "range and home is NOT guaranteed",
          max(qs) - min(qs) > 0.030 and max(qs) > c.home_tol)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(300)
    _report("null-policy")
    check("null-policy-fails: 300 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED FAMILY'S OWN STRATEGY ================================
    # The seed drops the pudding into an OPEN basket. The only open-from-above vessel
    # here is the shuttle pocket under the window: drop the pudding in and stop.
    # Settled and tidy — and refused: the chamber was never fed.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_shuttle_q(0.0)
    _step(30)
    drop_pudding_at((c.x_a, 0.0))
    _step(60)
    _report("seed-strategy")
    check("negative (SEED strategy): pudding dropped into the open pocket and left "
          "there — no success, score <= 0.20 (loaded latch only)",
          bool(scene.in_gallery(scene.pudding)[0])
          and bool(scene.in_pocket(scene.pudding)[0])
          and not bool(scene.in_chamber(scene.pudding)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= c.w_load + 1e-4)
    _REC["on"] = False

    # ================= 6. sealing (roof): no straight path to the hole ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_pudding_at((c.x_b, 0.0))  # directly over the DROP HOLE's xy — lands on the roof
    _report("sealing-roof")
    pl = scene._housing_local(scene.pudding.data.root_pos_w)[0]
    print(f"[smoke] roof-parked pudding local z={float(pl[2]) * 1000:.1f}mm "
          f"(roof top {c.roof_z1 * 1000:.0f}mm)", flush=True)
    check("sealing (roof): pudding dropped directly over the drop hole's xy lands ON "
          "the roof — no straight path into the vault: score 0, no success",
          float(pl[2]) > c.roof_z1 - 0.002 and not bool(scene.in_vault(scene.pudding)[0])
          and not bool(scene.in_chamber(scene.pudding)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 7. sealing (offset): window feeds only the pocket ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_shuttle_q(c.travel)  # pocket parked away at B
    _step(30)
    drop_pudding_at((c.x_a, 0.0))  # through the window onto the bare gallery floor
    _step(60)
    _report("sealing-offset")
    pl = scene._housing_local(scene.pudding.data.root_pos_w)[0]
    check("sealing (offset apertures): with the pocket away, a window drop strands "
          "the pudding on the GALLERY floor — chamber unreachable, not loaded, score 0",
          bool(scene.in_gallery(scene.pudding)[0])
          and not bool(scene.in_pocket(scene.pudding)[0])
          and not bool(scene.in_chamber(scene.pudding)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 8+9+10. mechanism: convey + deposit; not-home rejection ====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_shuttle_q(0.0)
    _step(30)
    drop_pudding_at((c.x_a, 0.0))
    assert bool(scene.in_pocket(scene.pudding)[0]), "probe setup: pocket must load"
    p0 = scene._housing_local(scene.pudding.data.root_pos_w)[0, 0]
    # The pudding trails the pocket center by the internal slack (~16.5 mm/side), so
    # the conveyed latch (pudding local x > conv_x_min = +0.005) fires around
    # q ~ 0.074 while the deposit needs pudding x ~ +0.021 (q ~ 0.089). Push in
    # <= 4 mm rungs and STOP at the first latch readback: between consecutive checks
    # the pudding advances at most one rung, so it cannot jump from below the latch
    # threshold to the deposit point unchecked.
    for tgt in (0.066, 0.070, 0.074, 0.078, 0.082, 0.086, 0.090, 0.094):
        push_shuttle(tgt, "mid-rail", tol=0.002)
        if bool(scene._conveyed[0]):
            break
    _report("near-miss-convey")
    q_mid = float(scene.shuttle_q()[0])
    check("near-miss (convey): pudding loaded and pushed only to mid-rail — "
          "loaded+conveyed latch (score ~0.35), no deposit, no success",
          bool(scene.in_pocket(scene.pudding)[0])
          and not bool(scene.in_chamber(scene.pudding)[0])
          and q_mid > 0.030
          and abs(float(scene.score()[0]) - (c.w_load + c.w_conv)) < 1e-3
          and not bool(scene.success()[0]))
    push_shuttle(c.travel - 0.004, "to-B")
    _step(120)
    _report("mech-deposit")
    q_b = float(scene.shuttle_q()[0])
    p1 = scene._housing_local(scene.pudding.data.root_pos_w)[0, 0]
    print(f"[smoke] convey: rail {q_mid:.3f} -> {q_b:.3f}, pudding local x "
          f"{float(p0) * 1000:.0f} -> {float(p1) * 1000:.0f}mm", flush=True)
    check("MECHANISM (convey+deposit): the force-pushed pocket dragged the pudding "
          "across the gallery floor and gravity dropped it through the hole into the "
          "chamber",
          q_b > c.travel - 0.020 and float(p1 - p0) > 0.060
          and bool(scene.in_chamber(scene.pudding)[0]))
    check("not-home rejection: pudding deposited, distractors out, settled — but the "
          "shuttle still at B: no success, score <= 0.65",
          bool(scene.in_chamber(scene.pudding)[0]) and bool(scene.distractors_clear()[0])
          and not bool(scene.home()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_load + c.w_conv + c.w_dep + 1e-4)
    _REC["on"] = False

    # ================= 11. latched credit survives ================================================
    park = housing_pt((0.0, 0.0, 0.0))
    park[:, 0] -= 0.45  # well outside the vault, on the open floor
    park[:, 2] = env.iscene.env_origins[:, 2] + c.pudding_size[2] / 2 + 0.003
    _write_body(scene.pudding, park, _qz(torch.zeros(n, device=device)))
    _step(60)
    _report("latched-credit")
    check("latched credit: pudding teleported back out of the chamber — deposit no "
          "longer live, latched credit survives (score >= 0.65), no success",
          not bool(scene.in_chamber(scene.pudding)[0])
          and float(scene.score()[0]) >= c.w_load + c.w_conv + c.w_dep - 1e-4
          and not bool(scene.success()[0]))

    # ================= 12. negative: wrong object =================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_shuttle_q(0.0)
    _step(30)
    _write_body(scene.milk, housing_pt((c.x_b, 0.0, 0.060)), hquat())
    _step(90)
    _report("wrong-object")
    check("negative (wrong object): the MILK cube resting in the chamber instead — "
          "no success, score ~0",
          bool(scene.in_chamber(scene.milk)[0]) and bool(scene.home()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 13. contamination ==========================================================
    # Continue: ALSO put the pudding in the chamber (instrumentation) — every clause
    # but the distractor one now holds. The distractor clause must refuse alone.
    _write_body(scene.pudding, housing_pt((-0.045, 0.0, 0.060)), hquat())
    _step(120)
    _report("contamination")
    check("contamination: pudding AND milk both in the chamber, shuttle home, settled "
          "— the distractor clause alone refuses: no success, score <= 0.75",
          bool(scene.in_chamber(scene.pudding)[0]) and bool(scene.in_vault(scene.milk)[0])
          and bool(scene.home()[0]) and bool(scene.settled()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75 + 1e-6)

    # ================= 14. settle gate ============================================================
    # The exact success state — pudding in the chamber, distractors out, shuttle home
    # — but the pudding still SLIDING. success() must refuse while anything moves; the
    # probe is dismantled (transport) well before friction could settle it.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_shuttle_q(0.0)
    _step(30)
    vel = torch.zeros(n, 3, device=device)
    vel[:, :2] = _qapply(hquat(), torch.tensor([0.40, 0.0, 0.0],
                                               device=device).expand(n, 3))[:, :2]
    _write_body(scene.pudding, housing_pt((-0.060, 0.0, 0.040)), hquat(), lin_vel=vel)
    moving_ok = True
    for _ in range(5):
        _step(1)
        v = float(scene.pudding.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.05 and bool(scene.in_chamber(scene.pudding)[0]) \
            and not bool(scene.settled()[0]) and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    _write_body(scene.pudding, park, _qz(torch.zeros(n, device=device)))
    _step(30)
    check("settle gate: the exact success state with the pudding still sliding at "
          "~0.4 m/s is refused while anything moves",
          moving_ok)

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shuttle_vault")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die loudly, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
