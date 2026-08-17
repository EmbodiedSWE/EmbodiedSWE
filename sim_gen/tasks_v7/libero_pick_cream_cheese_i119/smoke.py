"""Smoke / rubric-REJECTION battery for SiloTipPourScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a real stage-then-pour trajectory
with monotone latched credit, two+ seeds). This battery proves the rubric REJECTS
wrong outcomes, and that the mechanism claims the task rests on — the lever tips the
silo, a released silo self-returns, a box can only leave legitimately while tipped —
are physics, not fiat. Every probe is CONSTRUCTED (teleport, real physics steps,
judge) — instrumentation, never a solution; force/torque probes assert the actuator
actually moved (no vacuous rejections).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; silos at rest tilt, boxes
                           caged, basket upright on the open ground; score ~0;
   2. randomization      — two seeded resets: READBACK stand yaw/xy and basket
                           pose all differ;
   3. side swap          — over 10 resets the cream box occupies BOTH silos;
   4. null-policy        — 240 idle steps: boxes stay caged, score ~0, no success;
   5. lever mechanism    — hinge torque tips the cream silo past +15 deg (assert
                           it moved); released, it swings back to its rest tilt by
                           its own CoM bias alone;
   6. out-of-order pour  — that same pour ran with the basket UNSTAGED: the box
                           grounds -> permanent fail; even CONSTRUCTED into the
                           basket afterwards, success stays False (score = latched
                           partial only);
   7. SEED STRATEGY      — the seed's plan ("grasp the box, place it in the
                           basket"): cream box teleported straight from the silo
                           into the basket = an exit from an UNTIPPED silo ->
                           breach latch, no success, score ~0;
   8. drag-out           — 1.5x-weight horizontal pull drags the box out of the
                           resting silo's open spout (assert it exited): breach ->
                           no success ever;
   9. staging near-miss  — basket 100 mm off the pad centre: no staged latch;
                           re-set onto the centre: latch (0.15);
  10. wrong silo         — full tidy episode on the DECOY: basket staged under the
                           brown silo, brown poured in -> no success, score ~0;
  11. legal pour         — stage + torque pour on the cream side -> success; and
                           success never fired while the box was airborne
                           (consecutive-still counter);
  12. decoy contamination— brown dropped into the succeeded basket -> success
                           flips OFF (live clause); removed -> success returns;
  13. no post-hoc        — cream lifted back OUT to the ground: grounded latch ->
                           success gone, score capped at 0.60; re-constructed into
                           the basket -> success STAYS False (fail is permanent);
  14. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_cream_cheese_i119.smoke --headless
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
    from . import scene as task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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
    cl = scene._stand_local(scene.cream.data.root_pos_w)[0]
    bl = scene._stand_local(scene.basket.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | pitch={float(scene.cream_silo_pitch_deg()[0]):+.1f}deg "
          f"cream_std=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
          f"basket_std=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
          f"in_silo={bool(scene.cream_in_silo()[0])} "
          f"staged={bool(scene._staged[0])} poured={bool(scene._poured[0])} "
          f"landed={bool(scene._landed[0])} breach={bool(scene._breach[0])} "
          f"grounded={bool(scene._grounded[0])} "
          f"in_basket={bool(scene.in_basket(scene.cream.data.root_pos_w)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.silo_tip_pour")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.60, -0.80, 0.62)) + o),
                                tuple(np.array((0.28, 0.00, 0.12)) + o),
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

    def stand_world(loc_xyz) -> torch.Tensor:
        """Stand-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.stand.data.root_pos_w + quat_apply(scene.stand.data.root_quat_w, loc)

    def hinge_axis_w() -> torch.Tensor:
        _refresh()
        return quat_apply(scene.stand.data.root_quat_w,
                          torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))

    def cream_silo():
        return scene.silo_p if float(scene.cream_side[0]) > 0 else scene.silo_n

    def brown_silo():
        return scene.silo_n if float(scene.cream_side[0]) > 0 else scene.silo_p

    def apply_tau(silo, mag: float) -> None:
        zero = torch.zeros(n, 1, 3, device=device)
        if mag == 0.0:
            silo.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        else:
            t = mag * hinge_axis_w().view(n, 1, 3)
            silo.set_external_force_and_torque(zero, t, env_ids=_all_ids(), is_global=True)

    _SIN0 = math.sin(math.radians(55.0 + 12.5))

    def grav_tau(p: float) -> float:
        """Keel restoring torque about the hinge (decays to zero at the ~55 deg
        balance angle) — a constant hold torque is a runaway push at large pitch."""
        return 0.34 * math.sin(math.radians(55.0 - p)) / _SIN0

    def torque_pour(silo, box, target: float = 22.0, budget: int = 900):
        """Drive the silo with a gravity-feedforward PD hinge torque until `box`
        exits it; hold the tilt briefly, release. Returns (max_pitch_seen, exited,
        airborne_success_seen)."""
        kp, kd, tau_max = 0.010, 0.15, 0.80
        max_pitch, exited, air_succ = -90.0, False, False
        last_probe = float(scene.silo_pitch_deg(silo)[0])
        for i in range(budget):
            _refresh()
            p = float(scene.silo_pitch_deg(silo)[0])
            max_pitch = max(max_pitch, p)
            if not bool(scene._in_silo(silo, box.data.root_pos_w)[0]):
                exited = True
                break
            w = float((silo.data.root_ang_vel_w[0] * hinge_axis_w()[0]).sum())
            tau = grav_tau(p) + kp * (target - p) - kd * w
            apply_tau(silo, max(0.0, min(tau, tau_max)))
            _step(1)
            if i % 90 == 89:
                p_now = float(scene.silo_pitch_deg(silo)[0])
                if p_now < last_probe + 1.0 and p_now < target - 2.0:
                    kp = min(kp + 0.005, 0.030)
                    tau_max = min(tau_max + 0.20, 1.6)
                last_probe = p_now
        _refresh()
        hold_at = float(scene.silo_pitch_deg(silo)[0])
        for _ in range(25):
            _refresh()
            p = float(scene.silo_pitch_deg(silo)[0])
            max_pitch = max(max_pitch, p)
            w = float((silo.data.root_ang_vel_w[0] * hinge_axis_w()[0]).sum())
            tau = grav_tau(p) + kp * (hold_at - p) - kd * w
            apply_tau(silo, max(0.0, min(tau, tau_max)))
            _step(1)
            if bool(scene.success()[0]) and \
                    float(box.data.root_lin_vel_w[0].norm()) > 0.15:
                air_succ = True
        apply_tau(silo, 0.0)
        return max_pitch, exited, air_succ

    def stage_basket(side_sign: float) -> None:
        """Teleport the basket 30 mm above the given side's pad; gravity seats it."""
        side = torch.full((n,), side_sign, device=device)
        pad = scene.pad_center_w(side)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pad
        st[:, 2] += 0.030
        st[:, 3:7] = scene.stand.data.root_quat_w
        scene.basket.write_root_state_to_sim(st, _all_ids())
        _refresh()
        _step(120)

    def basket_drop_point(dz: float = 0.10) -> torch.Tensor:
        _refresh()
        p = scene.basket.data.root_pos_w.clone()
        p[:, 2] += dz
        return p

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    pit_p = float(scene.silo_pitch_deg(scene.silo_p)[0])
    pit_n = float(scene.silo_pitch_deg(scene.silo_n)[0])
    brown_in = bool(scene._in_silo(brown_silo(), scene.brown.data.root_pos_w)[0])
    check("settle/no-NaN: layout settles finite; both silos at their rest tilt, both "
          "boxes caged, basket upright on the open ground; score ~0, no success",
          bool(scene._finite()[0])
          and abs(pit_p - c.rest_pitch_deg) < 3.0 and abs(pit_n - c.rest_pitch_deg) < 3.0
          and bool(scene.cream_in_silo()[0]) and brown_in
          and bool(scene.basket_upright()[0]) and bool(scene.basket_on_ground()[0])
          and not bool(scene.basket_staged()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.stand.data.root_quat_w[0]),
                scene.stand.data.root_pos_w[0, :2].clone(),
                scene.basket.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.basket.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_sp, a_bp, a_by = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_sp, b_bp, b_by = readback()
    d_yaw, d_sp = dyaw(a_yaw, b_yaw), float((a_sp - b_sp).norm())
    d_bp, d_by = float((a_bp - b_bp).norm()), dyaw(a_by, b_by)
    print(f"[smoke] randomization deltas: stand_yaw={d_yaw:.1f}deg "
          f"stand_xy={d_sp * 1000:.1f}mm basket_xy={d_bp * 1000:.1f}mm "
          f"basket_yaw={d_by:.1f}deg", flush=True)
    check("randomization-is-real: stand yaw, stand xy and the basket's pose readback "
          "differ across seeds",
          d_yaw > 1.0 and d_sp > 0.002 and d_bp > 0.005 and d_by > 3.0)

    # ================= 3. cream side swap =========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.cream_side[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: cream silo sides {sorted(sides)}", flush=True)
    check("side swap: the cream box occupies BOTH silos over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, boxes stay caged, score ~0, no success",
          bool(scene.cream_in_silo()[0]) and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 5+6. lever mechanism + out-of-order pour grounds ===========================
    # Pour the cream silo WITHOUT staging the basket: (5) the hinge torque tips the
    # silo (assert it moved) and the released silo swings back by its own CoM bias;
    # (6) the unreceived box grounds — a PERMANENT fail that even a constructed
    # in-basket state cannot undo.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    max_p, exited, _air = torque_pour(cream_silo(), scene.cream)
    _step(300)  # hands-off: box falls to the pad/ground, silo swings back
    _report("bare-pour")
    ret_p = float(scene.silo_pitch_deg(cream_silo())[0])
    print(f"[smoke] mechanism: tipped to {max_p:+.1f}deg (exit={exited}), returned to "
          f"{ret_p:+.1f}deg after release", flush=True)
    check("lever mechanism: the hinge torque tips the silo past +15deg and the "
          "RELEASED silo swings back to its rest tilt by its own CoM bias",
          max_p > 15.0 and exited and ret_p < -8.0)
    grounded_now = bool(scene._grounded[0])
    _write_body(scene.cream, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(150)
    _report("oo-constructed")
    _REC["on"] = False
    check("out-of-order pour grounds: box poured with the basket unstaged lands on "
          "the bare pad -> grounded latch; even CONSTRUCTED into the basket, "
          "success stays False and score is the latched partial only",
          grounded_now and bool(scene.in_basket(scene.cream.data.root_pos_w)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.26)

    # ================= 7. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "grasp the box and place it in the basket". The box cannot
    # be grasped; teleporting it straight from the caged rest state into the basket
    # is an exit from an UNTIPPED silo -> breach.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    _write_body(scene.cream, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(150)
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): cream box teleported from the resting silo "
          "straight into the basket -> breach latch, no success, score ~0",
          bool(scene._breach[0]) and bool(scene.in_basket(scene.cream.data.root_pos_w)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 8. drag-out is a breach ====================================================
    # The spout is open: a horizontal pull CAN extract the box from the resting
    # silo — and doing so is exactly what the breach latch punishes.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    out_w = quat_apply(scene.stand.data.root_quat_w,
                       torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    _push(scene.cream, 1.5 * c.box_mass * 9.81 * out_w, 180)
    _step(180)
    _report("drag-out")
    dragged_out = not bool(scene.cream_in_silo()[0])
    check("drag-out: a 1.5x-weight horizontal pull drags the box out of the resting "
          "silo's open spout (the probe moved it) -> breach latch, no success",
          dragged_out and bool(scene._breach[0]) and not bool(scene.success()[0]))

    # ================= 9. staging near-miss =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    side0 = float(scene.cream_side[0])
    off = stand_world((c.pad_x + 0.10, side0 * c.pad_y, 0.030))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = off
    st[:, 3:7] = scene.stand.data.root_quat_w
    scene.basket.write_root_state_to_sim(st, _all_ids())
    _refresh()
    _step(120)
    _report("stage-miss")
    missed = (not bool(scene._staged[0])) and float(scene.score()[0]) <= 0.01
    stage_basket(side0)
    _report("stage-hit")
    check("staging near-miss: basket 100 mm off the pad centre earns NOTHING; "
          "re-set onto the centre latches staged (0.15)",
          missed and bool(scene._staged[0])
          and abs(float(scene.score()[0]) - c.w_staged) < 0.01)

    # ================= 10. negative: wrong silo (a tidy decoy episode) ============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    stage_basket(-float(scene.cream_side[0]))  # stage under the BROWN silo
    max_pb, exited_b, _ = torque_pour(brown_silo(), scene.brown)
    _step(400)
    _report("wrong-silo")
    _REC["on"] = False
    check("negative (wrong silo): basket staged under the DECOY and the brown box "
          "poured cleanly into it — cream still caged, no success, score ~0",
          exited_b and bool(scene.in_basket(scene.brown.data.root_pos_w)[0])
          and bool(scene.cream_in_silo()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 11. legal pour succeeds (and never mid-air) ================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    stage_basket(float(scene.cream_side[0]))
    assert bool(scene._staged[0]), "probe setup: basket must stage on the cream pad"
    max_pc, exited_c, air_succ = torque_pour(cream_silo(), scene.cream)
    got = False
    for j in range(700):
        _step(1)
        _refresh()
        if bool(scene.success()[0]) and \
                float(scene.cream.data.root_lin_vel_w[0].norm()) > 0.15:
            air_succ = True
        if bool(scene.success()[0]):
            got = True
            break
    _step(60)
    _report("legal-pour")
    check("legal pour: staged basket + hinge-torque pour on the cream side reaches "
          "success, and success never fired while the box was still moving fast",
          exited_c and (got or bool(scene.success()[0])) and not air_succ)

    # ================= 12. decoy contamination is judged live =====================================
    _write_body(scene.brown, basket_drop_point(0.12), scene.basket.data.root_quat_w)
    _step(180)
    _report("decoy-in")
    dirty = bool(scene.in_basket(scene.brown.data.root_pos_w)[0]) \
        and not bool(scene.success()[0])
    _write_body(scene.brown, stand_world((0.60, 0.30, 0.030)))
    _step(180)
    _report("decoy-out")
    _REC["on"] = False
    check("decoy contamination: brown dropped into the succeeded basket flips "
          "success OFF (live clause); removed, success returns",
          dirty and bool(scene.success()[0]))

    # ================= 13. no post-hoc: the end state is judged, fails are forever ================
    _write_body(scene.cream, stand_world((0.55, -0.30, 0.030)))
    _step(150)
    _report("post-hoc-out")
    capped = bool(scene._grounded[0]) and not bool(scene.success()[0]) \
        and abs(float(scene.score()[0]) - 0.60) < 1e-3
    _write_body(scene.cream, basket_drop_point(), scene.basket.data.root_quat_w)
    _step(150)
    _report("post-hoc-back")
    check("no post-hoc: cream lifted back out grounds (success gone, score capped "
          "at the 0.60 latch); re-constructed into the basket it STAYS failed",
          capped and bool(scene.in_basket(scene.cream.data.root_pos_w)[0])
          and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.silo_tip_pour")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
