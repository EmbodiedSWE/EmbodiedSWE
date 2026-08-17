"""Smoke / rubric-REJECTION battery for TiltLabyrinthScene — NullRobot, teleported probes.

solve.py is the acceptance proof (tilt-servo the gimballed case, route the sealed
checker through the baffle gap into the beacon-side pocket, release). This battery
proves the rubric REJECTS wrong outcomes and that the two strategic differentiators
from the seed — the SEAL (the piece can never be touched) and the DECOY (a wrong
pocket is a permanent trap) — are physically load-bearing. Every probe is
CONSTRUCTED as a settled state (teleport, real physics steps, judge); constructed
partial states may earn latched partial credit but none may reach success() unless
the full outcome is genuinely built and the case genuinely released.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; springs hold the case level,
                           checker parked in the start chamber; score ~0, no success;
   2. randomization      — three seeded resets: READBACK stand yaw, stand xy and the
                           checker's start cell all differ across seeds;
   3. side swap          — across seeds BOTH target sides occur, and the stored sign
                           matches the beacon's stand-local position readback;
   4. null-policy        — 240 idle steps: the checker stays parked (no phantom
                           creep), the case stays level, score ~0, no success;
   5. SEAL reality       — a 1.5 N upward pull (50x the checker's weight — far more
                           than any grasp could apply through the 12 mm slots) lifts
                           the checker INTO the grille and no further: it visibly
                           rises (anti-vacuous), is arrested below the grille line,
                           never leaves the case, and falls back when released;
   6. SEED end-state     — the seed's own plan ("lay the checker flat on the marked
                           square"): checker CONSTRUCTED settled ON TOP of the case
                           roof directly over the target pocket — the z-band voids
                           it: no latch, score ~0, no success;
   7. decoy permanence   — checker constructed settled in the DECOY pocket: no
                           success, score capped at the gap credit; then a full-limit
                           rim press toward the target side AND back toward the start
                           cannot extract it — the wrong pocket is a permanent trap;
   8. near-miss wall     — checker settled on the far-chamber floor against the far
                           wall (past the gap, beside the pocket): gap credit only,
                           score <= 0.25, no success;
   9. held-tilt no-success — checker constructed in the TARGET pocket but the rim
                           held pressed at the stop: pocket credit latches (0.60) yet
                           success stays false until the press is RELEASED and the
                           springs re-level — then success (positive control), and a
                           subsequent full-limit press back toward the start chamber
                           cannot pull the checker out of the target pocket;
  10. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_checkers_i211.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

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
    p = scene.disc_local()[0]
    print(f"[smoke] {tag:16s} | disc_loc=({float(p[0]):+.3f},{float(p[1]):+.3f},"
          f"{float(p[2]):+.3f}) tilt={float(scene.tilt_deg()[0]):5.2f}deg "
          f"gap={bool(scene._gap_latch[0])} pocket={bool(scene._pocket_latch[0])} "
          f"decoy={bool(scene.in_decoy_pocket()[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_labyrinth")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.80, 0.85)) + o),
                                tuple(np.array((0.38, 0.00, 0.24)) + o),
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

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def disc_loc():
        _refresh()
        return scene.disc_local()[0]

    def press_steps(d_local, steps: int, tau: float = 1.2) -> None:
        """Constant rim press: torque `tau` about the world axis that tilts the
        case downhill toward the STAND-LOCAL direction d_local, recomputed each
        step (body-frame wrench). tau=1.2 N.m drives the hinges onto their 12 deg
        stops — the hardest press the rim geometry admits."""
        d = torch.zeros(n, 3, device=device)
        nm = math.hypot(d_local[0], d_local[1])
        d[:, 0] = d_local[0] / nm
        d[:, 1] = d_local[1] / nm
        for _ in range(steps):
            d_w = quat_apply(scene.stand.data.root_quat_w, d)
            t_b = quat_apply_inverse(scene.tray.data.root_quat_w,
                                     tau * torch.cross(ez, d_w, dim=-1))
            scene.tray.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            _step(1)
        scene.tray.set_external_force_and_torque(zero, zero)

    def put_disc_tray_local(loc_xyz) -> None:
        """Teleport-construct: the checker at a tray-local point, tray attitude."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        pos = scene.tray.data.root_pos_w + quat_apply(scene.tray.data.root_quat_w, loc)
        _write_body(scene.disc, pos, scene.tray.data.root_quat_w)

    # pocket-centre construction points (tray-local)
    pk_cx = (c.pk_x0 + c.ix) / 2
    pk_cy = (c.pk_y0 + c.iy) / 2
    z_in_pocket = c.pocket_z1 + c.disc_h / 2 + 0.002
    z_on_floor = c.floor_z1 + c.disc_h / 2 + 0.002
    z_on_roof = c.grille_z1 + c.disc_h / 2 + 0.002

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    p = disc_loc()
    check("settle/no-NaN: layout settles finite; springs hold the case level, the "
          "checker parked on the start-chamber floor; score ~0, no success",
          bool(scene._finite()[0]) and float(scene.tilt_deg()[0]) < c.level_max_deg
          and float(p[0]) < c.fun_x0 + 0.02 and abs(float(p[2]) + 0.006) < 0.004
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        syaw = yaw_of(scene.stand.data.root_quat_w[0])
        sp = scene.stand.data.root_pos_w[0, :2].clone()
        dl = scene.disc_local()[0, :2].clone()
        return syaw, sp, dl

    rb = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(30)
        rb.append(readback())
    d_yaw = max(dyaw(a[0], b[0]) for a, b in ((rb[0], rb[1]), (rb[0], rb[2]), (rb[1], rb[2])))
    d_sp = max(float((a[1] - b[1]).norm()) for a, b in ((rb[0], rb[1]), (rb[0], rb[2]),
                                                        (rb[1], rb[2])))
    d_dl = max(float((a[2] - b[2]).norm()) for a, b in ((rb[0], rb[1]), (rb[0], rb[2]),
                                                        (rb[1], rb[2])))
    print(f"[smoke] randomization deltas (max over 3 seeds): stand_yaw={d_yaw:.1f}deg "
          f"stand_xy={d_sp * 1000:.1f}mm disc_cell={d_dl * 1000:.1f}mm", flush=True)
    check("randomization-is-real: stand yaw, stand xy and the checker's start cell "
          "readback all differ across seeds",
          d_yaw > 2.0 and d_sp > 0.005 and d_dl > 0.010)

    # ================= 3. randomization: the target side swaps and reads back =====================
    seen = {True: 0, False: 0}
    consistent = True
    for sd in range(300, 312):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        _refresh()
        sgn = float(scene.target_sign[0])
        bloc = quat_apply_inverse(scene.stand.data.root_quat_w,
                                  scene.beacon.data.root_pos_w - scene.stand.data.root_pos_w)[0]
        consistent &= (float(bloc[1]) > 0) == (sgn > 0) \
            and abs(abs(float(bloc[1])) - c.beacon_y) < 0.02
        seen[sgn > 0] += 1
        if seen[True] and seen[False] and sd >= 305:
            break
    print(f"[smoke] target-side counts over seeds: +y={seen[True]} -y={seen[False]} "
          f"consistent={consistent}", flush=True)
    check("randomization (side swap): both target sides occur across seeds and the "
          "sign matches the beacon's stand-local position readback",
          seen[True] > 0 and seen[False] > 0 and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    p_a = disc_loc().clone()
    _step(240)
    _report("null-policy")
    p_b = disc_loc()
    drift = float((p_b - p_a).norm())
    check("null-policy-fails: 240 idle steps, the checker stays parked (drift "
          f"{drift * 1000:.1f} mm), the case stays level, score ~0, no success",
          drift < 0.01 and float(scene.tilt_deg()[0]) < c.level_max_deg
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 5. SEAL reality: the checker cannot be touched/extracted ===================
    # A 1.5 N upward pull (50x the checker's 0.29 N weight — far beyond anything a
    # fingertip could transmit through a 12 mm slot) is applied to the checker.
    # Anti-vacuous: the checker must visibly RISE off the floor; the grille must
    # arrest it inside the case; released, it must fall back to the floor.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    z_max = -1.0
    for _ in range(240):
        f_b = quat_apply_inverse(scene.disc.data.root_quat_w,
                                 1.5 * ez)
        scene.disc.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
        _step(1)
        z_max = max(z_max, float(disc_loc()[2]))
    scene.disc.set_external_force_and_torque(zero, zero)
    _report("seal-pull")
    p_pull = disc_loc().clone()
    inside = abs(float(p_pull[0])) < c.ix + 0.01 and abs(float(p_pull[1])) < c.iy + 0.01
    _step(180)
    _report("seal-released")
    p_back = disc_loc()
    print(f"[smoke] seal probe: z_max={z_max * 1000:.1f}mm (floor rest -6.0, grille "
          f"line {c.inner_top * 1000:.0f})", flush=True)
    check("SEAL-reality: a 1.5 N upward pull lifts the checker into the grille "
          "(it moved — anti-vacuous), the roof arrests it inside the case, and it "
          "falls back when released; never a success",
          z_max > -0.001 and z_max < c.inner_top + 0.004 and inside
          and abs(float(p_back[2]) + 0.006) < 0.004
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.03)

    # ================= 6. negative: the SEED'S OWN END STATE ======================================
    # The seed task lays checkers flat onto marked squares of an open board. Its
    # nearest analog here: the checker laid flat ON TOP of the case, directly over
    # the target pocket. CONSTRUCT it settled there — the z-band must void it.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    sgn = float(scene.target_sign[0])
    put_disc_tray_local((pk_cx, sgn * pk_cy, z_on_roof))
    _step(180)
    _report("seed-on-roof")
    _REC["on"] = False
    p = disc_loc()
    on_roof = float(p[2]) > c.inner_top and abs(float(p[0]) - pk_cx) < 0.05
    check("negative (SEED end state): checker laid flat ON the case roof over the "
          "marked pocket — settled there but OUTSIDE the play volume: no latch, "
          "score ~0, no success",
          on_roof and not bool(scene.entered_far()[0])
          and not bool(scene.in_target_pocket()[0])
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 7. negative: the DECOY pocket is a permanent trap ==========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    sgn = float(scene.target_sign[0])
    put_disc_tray_local((pk_cx, -sgn * pk_cy, z_in_pocket))
    _step(180)
    _report("decoy-dropped")
    in_decoy = bool(scene.in_decoy_pocket()[0])
    decoy_scored = float(scene.score()[0])
    no_succ_a = not bool(scene.success()[0])
    # escape attempt 1: full-limit press toward the TARGET side (the correction a
    # solver would try); escape attempt 2: full-limit press back toward the start.
    press_steps((0.0, sgn), 400)
    press_steps((-1.0, 0.0), 400)
    _step(240)
    _report("decoy-escape")
    _REC["on"] = False
    check("negative (decoy permanence): checker settled in the WRONG pocket earns "
          "no pocket credit (score <= gap credit) and full-limit presses toward "
          "the target and back cannot extract it — permanent failure",
          in_decoy and no_succ_a and decoy_scored <= c.w_gap + 0.0000005
          and bool(scene.in_decoy_pocket()[0])
          and not bool(scene.in_target_pocket()[0])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_gap + 0.0000005)

    # ================= 8. near-miss: far chamber, beside the pocket ===============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    put_disc_tray_local((c.ix - c.disc_r - 0.004, 0.0, z_on_floor))
    _step(180)
    _report("wall-nearmiss")
    p = disc_loc()
    check("near-miss (far wall): checker settled against the far wall BESIDE the "
          "pockets — gap credit only, score <= 0.25, no success",
          float(p[0]) > c.gap_x_min and abs(float(p[2]) + 0.006) < 0.004
          and bool(scene._gap_latch[0])
          and float(scene.score()[0]) <= c.w_gap + 0.0000005
          and not bool(scene.success()[0]))

    # ================= 9. release is judged: held tilt never succeeds; then it does ===============
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    sgn = float(scene.target_sign[0])
    # hold the rim pressed while constructing the pocketed state: pocket credit
    # may latch, success must NOT fire while the case is tilted
    press_steps((1.0, 0.0), 240)
    press_steps((1.0, 0.0), 1, tau=1.2)  # leave the press applied
    put_disc_tray_local((pk_cx, sgn * pk_cy, z_in_pocket))
    for _ in range(240):
        d_w = quat_apply(scene.stand.data.root_quat_w,
                         torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        t_b = quat_apply_inverse(scene.tray.data.root_quat_w,
                                 1.2 * torch.cross(ez, d_w, dim=-1))
        scene.tray.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
        _step(1)
    _report("held-tilt")
    held_ok = bool(scene.in_target_pocket()[0]) \
        and float(scene.tilt_deg()[0]) > c.level_max_deg + 2.0 \
        and not bool(scene.success()[0]) \
        and float(scene.score()[0]) <= 0.6000005
    # release: springs re-level, everything rests -> genuine success (positive
    # control that the judge accepts the true end state)
    scene.tray.set_external_force_and_torque(zero, zero)
    _step(300)
    _report("released")
    released_ok = bool(scene.success()[0]) and float(scene.score()[0]) >= 0.999
    # retention: a full-limit press back toward the start chamber cannot pull the
    # checker out of the target pocket; released again, success re-establishes
    press_steps((-1.0, 0.0), 400)
    _step(300)
    _report("retention")
    _REC["on"] = False
    check("release-is-judged + retention: pocketed but HELD tilted is never success "
          "(score <= 0.60); released, the springs re-level and success holds; a "
          "full-limit reverse press cannot extract the checker from the pocket",
          held_ok and released_ok and bool(scene.in_target_pocket()[0])
          and bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_labyrinth")
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
