"""Smoke / rubric-REJECTION battery for PryLidVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts an insert / pry / lid-off /
withdraw / place episode and the latched credit is monotone along it, two seeds). This
battery proves the rubric REJECTS wrong outcomes, and that the physics claims the task
rests on — the seated lid is slide-captive, the seed's lower-from-above plan dead-ends
on the flush lid, the pry credential only fires under a genuine bar-under-lid lever —
are real, not fiat. Every probe is CONSTRUCTED as a settled state (teleport, real
physics steps, judge) — instrumentation, never a solution. Smoke constructs may write
bodies inside rubric volumes (that is the point of a rubric probe); only solve.py is
barred from doing so.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; lid seated flush, ketchup
                           outside the cavity; score ~0, no success;
   2. randomization      — two seeded resets: READBACK chest yaw, chest xy, the bar
                           pose and the ketchup bottle's pose all differ;
   3. slot shuffle       — over 10 resets the ketchup bottle occupies >= 2 different
                           table slots;
   4. null-policy        — 240 idle steps: lid stays seated (edge never proud), score
                           ~0, no success;
   5. SEED STRATEGY      — the seed's plan ("carry the bottle over the tray and lower
                           it in from above"): the bottle ends standing ON THE CLOSED
                           LID — not in the cavity, no success, score ~0;
   6. lid slide-captive  — a 10 N horizontal shove (2.5 s) on the SEATED lid leaves it
                           in the recess (< 8 mm travel, edge never proud, no credit);
                           control: the same class of push moves the free BAR >= 30 mm
                           (the probe force demonstrably works);
   7. NO-PRY CHEAT       — lid teleported straight off to the table + ketchup dropped
                           into the open cavity: the END STATE looks complete (lid
                           flat+clear, bottle in, bar clear, settled — all asserted
                           live) yet success is FALSE and score ~0 — the pry
                           credential was never earned;
   8. positive control   — the honest order, constructed: bar teleported into the
                           slot, a REAL ramped lever torque pries the lid proud (the
                           `pried` latch is earned by contact dynamics), lid off, bar
                           parked, ketchup dropped in -> success TRUE, score 1.0;
   9. wrong object       — from the success state: MUSTARD also dropped in the cavity
                           -> success revoked;
  10. bar left in cavity — from the success state: the bar laid inside the cavity ->
                           success revoked;
  11. lid back on chest  — from the success state: the lid balanced back on the chest
                           -> success revoked;
  12. bottle out         — from the success state: the ketchup moved out to the table
                           -> success revoked, latched credit capped at 0.40;
  13. barless tilt       — the lid HELD tilted proud with the bar nowhere near: the
                           pried credential does NOT fire (it demands the bar tip
                           under the lid), and the released lid falls back to seat;
  14. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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

_qmul = task_scene._qmul
encode_force = task_scene.encode_force

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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel_w is not None:
        st[:, 7:10] = lin_vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    ll = scene.lid_loc()[0]
    bl = scene._chest_local(scene.bar.data.root_pos_w)[0]
    kl = scene._chest_local(scene.ketchup.data.root_pos_w)[0]
    ez = float(scene._chest_local(scene._lid_edge_w())[0, 2])
    print(f"[smoke] {tag:16s} | lid=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
          f"{float(ll[2]):+.3f}) edge_z={ez:+.3f} bar=({float(bl[0]):+.3f},"
          f"{float(bl[1]):+.3f},{float(bl[2]):+.3f}) ketchup=({float(kl[0]):+.3f},"
          f"{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
          f"pried={bool(scene._pried[0])} lid_off={bool(scene._lid_off[0])} "
          f"in_cav={bool(scene.ketchup_in_cavity()[0])} "
          f"bar_clear={bool(scene.bar_clear()[0])} "
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
    env = ENVS.get("simgen.pry_lid_vault")().build(num_envs=args.num_envs,
                                                  device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.42, -0.72, 0.60)) + o),
                                tuple(np.array((0.40, 0.00, 0.08)) + o),
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

    def chest_world(loc_xyz) -> torch.Tensor:
        """Chest-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.chest.data.root_pos_w + quat_apply(
            scene.chest.data.root_quat_w, loc)

    def chest_quat() -> torch.Tensor:
        _refresh()
        return scene.chest.data.root_quat_w

    def dir_w(v_xyz) -> torch.Tensor:
        """Chest-local direction -> world unit vector (env 0)."""
        _refresh()
        v = torch.tensor(v_xyz, device=device, dtype=torch.float).expand(n, 3)
        return quat_apply(scene.chest.data.root_quat_w, v)[0]

    def edge_z() -> float:
        _refresh()
        return float(scene._chest_local(scene._lid_edge_w())[0, 2])

    def lid_xy() -> torch.Tensor:
        _refresh()
        return scene.lid_loc()[0, :2].clone()

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    ll = scene.lid_loc()[0]
    check("settle/no-NaN: layout settles finite; lid seated flush (edge below the "
          "proud gate), ketchup outside the cavity; score ~0, no success",
          bool(scene._finite()[0])
          and abs(float(ll[0])) < 0.01 and abs(float(ll[1])) < 0.01
          and 0.060 < float(ll[2]) < 0.072
          and edge_z() < c.rim_top + 0.002
          and not bool(scene.ketchup_in_cavity()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.chest.data.root_quat_w[0]),
                scene.chest.data.root_pos_w[0, :2].clone(),
                scene.bar.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.bar.data.root_quat_w[0]),
                scene.ketchup.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.ketchup.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_bp, a_by, a_kp, a_ky = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_bp, b_by, b_kp, b_ky = readback()
    d_yaw, d_cp = dyaw(a_yaw, b_yaw), float((a_cp - b_cp).norm())
    d_bp, d_by = float((a_bp - b_bp).norm()), dyaw(a_by, b_by)
    d_kp, d_ky = float((a_kp - b_kp).norm()), dyaw(a_ky, b_ky)
    print(f"[smoke] randomization deltas: chest_yaw={d_yaw:.1f}deg "
          f"chest_xy={d_cp * 1000:.1f}mm bar_xy={d_bp * 1000:.1f}mm "
          f"bar_yaw={d_by:.1f}deg ketchup_xy={d_kp * 1000:.1f}mm "
          f"ketchup_yaw={d_ky:.1f}deg", flush=True)
    check("randomization-is-real: chest yaw, chest xy, the bar pose and the ketchup "
          "bottle's pose readback differ across seeds",
          d_yaw > 2.0 and d_cp > 0.003 and d_bp > 0.005 and d_kp > 0.005
          and d_ky > 2.0)

    # ================= 3. bottle slot shuffle =====================================================
    slots_seen = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slots_seen.add(int(scene.ketchup_slot[0]))
    print(f"[smoke] over 10 resets: ketchup slots {sorted(slots_seen)}", flush=True)
    check("slot shuffle: the ketchup bottle occupies >= 2 different table slots "
          "over 10 resets", len(slots_seen) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(20)
    xy0 = lid_xy()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, lid stays seated (< 5 mm, edge never "
          "proud), score ~0, no success",
          float((lid_xy() - xy0).norm()) < 0.005
          and edge_z() < c.rim_top + 0.002
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's plan is "carry the bottle over the tray and lower it in from
    # above". Here the cavity is CLOSED by the flush lid: the plan's end state is
    # the bottle standing on the lid. Must not be in the cavity, not success, ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.ketchup,
                chest_world((0.0, 0.0, c.rim_top + c.bottle_size[2] / 2 + 0.008)),
                chest_quat())
    _step(180)
    _report("seed-strategy")
    kz = float(scene._chest_local(scene.ketchup.data.root_pos_w)[0, 2])
    print(f"[smoke] seed strategy: bottle ends at chest z={kz * 1000:.0f}mm "
          f"(closed lid top at {c.rim_top * 1000:.0f}mm)", flush=True)
    check("negative (SEED strategy): bottle lowered-from-above ends ON THE CLOSED "
          "LID — not in the cavity, no success, score ~0",
          kz > c.rim_top and not bool(scene.ketchup_in_cavity()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 8. positive control: the honest order accepts ==============================
    # NOTE: executed BEFORE checks 6/7 — the positive control's precision force work
    # must run on a wrench-clean env (exactly solve.py's condition): an earlier
    # `_push` leaves this pod's wrench frame-dragged with a stale reference that
    # survives env.reset() and misdirects the insertion PD (observed live).
    # Constructed honest episode: bar teleported into the slot (a transport smoke is
    # allowed), then a REAL ramped lever torque earns `pried` by contact dynamics,
    # lid off to the table, bar parked clear, ketchup dropped in. Success TRUE, 1.0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    # -- carry the bar to the slot MOUTH (transport of the free tool), then run the
    # SAME held-carry PD insertion and ramped lever pry that solve.py demonstrates --
    state = {"mode": 0, "q_ref": scene.bar.data.root_quat_w.clone()}
    zero = torch.zeros(n, 1, 3, device=device)
    bm8 = float(scene.bar.root_physx_view.get_masses().sum())
    Z_CARRY = 0.0550

    def bar_loc8() -> torch.Tensor:
        _refresh()
        return scene._chest_local(scene.bar.data.root_pos_w)[0]

    def enc8(f_world: torch.Tensor) -> torch.Tensor:
        return encode_force(state["mode"], state["q_ref"],
                            scene.bar.data.root_quat_w, f_world).view(n, 1, 3)

    def pd_hold8(z_ref: float, f_fwd: float) -> None:
        bl = bar_loc8()
        v_w = scene.bar.data.root_lin_vel_w
        vz = float(v_w[0, 2])
        vy = float((v_w[0] * dir_w((0.0, 1.0, 0.0))).sum())
        a_z = 9.81 + 600.0 * (z_ref - float(bl[2])) - 50.0 * vz
        a_y = 600.0 * (0.0 - float(bl[1])) - 50.0 * vy
        f = (bm8 * a_z) * torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3) \
            + (bm8 * a_y) * dir_w((0.0, 1.0, 0.0)).expand(n, 3) \
            - f_fwd * dir_w((1.0, 0.0, 0.0)).expand(n, 3)
        f = f.clamp(min=-6.0, max=6.0)
        scene.bar.set_external_force_and_torque(enc8(f), zero, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)

    # -- FRAME CALIBRATION: after the earlier probe pushes this pod can apply a
    # "global" wrench dragged into the body frame (stale wrench reference across
    # resets — the failed run showed the y-servo pushing the bar OFF the slot
    # axis). Measure in free air which encoding actually moves the bar along
    # +x_chest and lock it in before the precision insertion.
    q_id = torch.zeros(n, 4, device=device)
    q_id[:, 0] = 1.0
    zvec = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    best_mode, best_qref, best_a = 0, None, -2.0
    for cand in (0, 1):
        _write_body(scene.bar, chest_world((0.0, 0.0, 0.45)), chest_quat())
        state["mode"] = cand
        # mode 1 with an IDENTITY reference is the continuous body-frame encode
        state["q_ref"] = (scene.bar.data.root_quat_w.clone() if cand == 0 else q_id)
        p0 = scene.bar.data.root_pos_w[0].clone()
        for _ in range(8):
            f = (bm8 * 9.81) * zvec + 2.0 * dir_w((1.0, 0.0, 0.0)).expand(n, 3)
            scene.bar.set_external_force_and_torque(enc8(f), zero,
                                                    env_ids=_all_ids(),
                                                    is_global=True)
            _step(1)
        scene.bar.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        _refresh()
        d = scene.bar.data.root_pos_w[0] - p0
        a = float((d * dir_w((1.0, 0.0, 0.0))).sum()) / max(float(d[:2].norm()), 1e-9)
        print(f"[smoke] check8 frame probe mode {cand}: alignment {a:+.2f} "
              f"(moved {float(d.norm()) * 1000:.0f}mm)", flush=True)
        if a > best_a:
            best_mode, best_qref, best_a = cand, state["q_ref"].clone(), a
    state["mode"], state["q_ref"] = best_mode, best_qref
    print(f"[smoke] check8 force-frame locked: mode {best_mode} "
          f"(alignment {best_a:+.2f})", flush=True)

    def insert8(max_steps: int) -> bool:
        """Solve's proven held-carry insertion: hover-converge, gentle velocity-
        capped forward drive, peck-retry on stall; abort (for a fresh attempt) if
        the bar ever veers off the slot axis or drops out of the carry."""
        for _ in range(40):
            pd_hold8(Z_CARRY, 0.0)
        z_ref, fwd, retract, pecks = Z_CARRY, 0.6, 0, 0
        nudge = (0.0, +0.0015, -0.0015, +0.0025)
        probe_x = float(bar_loc8()[0])
        for i in range(max_steps):
            bl = bar_loc8()
            if float(bl[0]) < 0.120 and abs(float(bl[1])) < 0.020 \
                    and float(bl[2]) > 0.040:
                return True
            if abs(float(bl[1])) > 0.040 or float(bl[2]) < 0.030:
                print(f"[smoke] check8 insert veered: bar=({float(bl[0]):+.3f},"
                      f"{float(bl[1]):+.3f},{float(bl[2]):+.3f})", flush=True)
                return False
            v_fwd = -float((scene.bar.data.root_lin_vel_w[0]
                            * dir_w((1.0, 0.0, 0.0))).sum())
            if retract > 0:
                f_fwd = -0.8
                retract -= 1
            else:
                f_fwd = fwd if v_fwd < 0.06 else (0.2 * fwd if v_fwd < 0.12 else -0.1)
            pd_hold8(z_ref, f_fwd)
            if i % 45 == 44:
                x_new = float(bar_loc8()[0])
                if x_new > probe_x - 0.002:
                    pecks += 1
                    z_ref = Z_CARRY + nudge[pecks % 4]
                    fwd = min(fwd + 0.2, 1.4)
                    retract = 18
                probe_x = x_new
        return False

    inserted8 = False
    for attempt in range(3):
        _write_body(scene.bar,
                    chest_world((c.outer_half + 0.010 + c.bar_size[0] / 2, 0.0,
                                 Z_CARRY)),
                    chest_quat())
        if state["mode"] == 0:
            state["q_ref"] = scene.bar.data.root_quat_w.clone()
        if insert8(900):
            inserted8 = True
            break
        print(f"[smoke] check8 insert attempt {attempt + 1} failed; retrying",
              flush=True)
    scene.bar.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(30)
    bl8 = bar_loc8()
    print(f"[smoke] check8 insertion: done={inserted8} bar=({float(bl8[0]):+.4f},"
          f"{float(bl8[1]):+.4f},{float(bl8[2]):+.4f})", flush=True)
    # -- ramped lever torque about chest +y presses the tail down; -x hold force --
    tau = 0.15
    pried_earned = False
    for i in range(700):
        _refresh()
        if bool(scene._pried[0]):
            pried_earned = True
            break
        t_vec = enc8(tau * dir_w((0.0, 1.0, 0.0)).expand(n, 3))
        f_vec = enc8(-1.5 * dir_w((1.0, 0.0, 0.0)).expand(n, 3))
        scene.bar.set_external_force_and_torque(f_vec, t_vec, env_ids=_all_ids(),
                                                is_global=True)
        _step(1)
        e0, e1 = scene._bar_ends_w()
        tips = [scene._chest_local(e)[0] for e in (e0, e1)]
        tip = min(tips, key=lambda t: float(t[0]))
        if i % 60 == 0:
            bl8 = bar_loc8()
            bav = float(scene.bar.data.root_ang_vel_w[0].norm())
            print(f"[smoke] pry[{i}] tau={tau:.2f} edge_z={edge_z():.4f} "
                  f"tip=({float(tip[0]):+.4f},{float(tip[1]):+.4f},"
                  f"{float(tip[2]):+.4f}) bar=({float(bl8[0]):+.4f},"
                  f"{float(bl8[1]):+.4f},{float(bl8[2]):+.4f}) |w|={bav:.3f}",
                  flush=True)
        if i % 30 == 29:
            if float(tip[2]) < 0.040:
                # tip dived below the slot floor plane -> torque acting the wrong
                # way for this pod's wrench convention; flip the encoding
                state["mode"] ^= 1
                state["q_ref"] = scene.bar.data.root_quat_w.clone()
                print(f"[smoke] pry torque acted the WRONG way; mode -> "
                      f"{state['mode']}", flush=True)
            else:
                tau = min(tau + 0.08, 3.0)
    _report("smoke-pry")
    print(f"[smoke] positive control: pried earned by lever = {pried_earned} "
          f"(tau reached {tau:.2f} N*m)", flush=True)
    # grasp the (held-proud) lid off to the table, release the lever
    _write_body(scene.lid, chest_world((-0.02, -0.30, c.lid_size[2] / 2 + 0.002)),
                chest_quat())
    scene.bar.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    # park the bar clear of the cavity, drop the ketchup in
    _write_body(scene.bar, chest_world((0.30, -0.34, c.bar_size[2] / 2 + 0.002)),
                chest_quat())
    _step(60)
    _write_body(scene.ketchup,
                chest_world((0.0, 0.0, c.rim_top + c.bottle_size[2] / 2 + 0.012)),
                chest_quat())
    ok8 = False
    for _ in range(360):
        _step(1)
        if bool(scene.success()[0]):
            ok8 = True
            break
    _step(60)
    _report("positive")
    s8 = float(scene.score()[0])
    check("positive control: bar-lever pry (earned by contact dynamics), lid off, "
          "bar parked, ketchup in -> success TRUE and score 1.0",
          pried_earned and (ok8 or bool(scene.success()[0])) and s8 >= 0.999)
    _REC["on"] = False

    # --- snapshot the success state for the revocation probes -------------------------------------
    snap = scene.get_state(_all_ids())

    def restore() -> None:
        scene.set_state(snap, _all_ids())
        _refresh()

    # ================= 6. the seated lid is slide-captive =========================================
    # (Executed after check 8 — see the note there. The snapshot above fully
    # reconstructs the success state for checks 9-12, latches included.)
    # A 10 N horizontal shove (2.5 s, toward the open -x and then +y) must leave the
    # lid in the recess: < 8 mm of travel (1.5 mm design gap + compliance), edge
    # never proud, nothing scores. Control (non-vacuous force): the same class of
    # push moves the free BAR >= 30 mm on the table.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    xy0 = lid_xy()
    max_edge = edge_z()
    for d in ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        f = 10.0 * dir_w(d)
        zero = torch.zeros(n, 1, 3, device=device)
        fw = f.view(1, 1, 3).expand(n, 1, 3).contiguous()
        for _ in range(150):
            scene.lid.set_external_force_and_torque(fw, zero, env_ids=_all_ids(),
                                                    is_global=True)
            _step(1)
            max_edge = max(max_edge, edge_z())
        scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        _step(30)
    trav = float((lid_xy() - xy0).norm())
    print(f"[smoke] lid shove: travel {trav * 1000:.1f}mm, max edge z "
          f"{max_edge * 1000:.1f}mm (proud gate {(c.rim_top + c.pried_proud) * 1000:.1f}mm)",
          flush=True)
    b0 = scene.bar.data.root_pos_w[0, :2].clone()
    _push(scene.bar, 3.0 * dir_w((0.0, 1.0, 0.0)), 120)
    _step(30)
    _refresh()
    b_trav = float((scene.bar.data.root_pos_w[0, :2] - b0).norm())
    print(f"[smoke] control push: bar travelled {b_trav * 1000:.0f}mm", flush=True)
    _report("lid-captive")
    check("lid slide-captive: 10 N shoves leave the seated lid in the recess "
          "(< 8 mm, edge never proud, no credit) while the same class of push "
          "moves the free bar >= 30 mm (probe force is real)",
          trav < 0.008 and max_edge < c.rim_top + c.pried_proud
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and b_trav >= 0.030)
    _REC["on"] = False

    # ================= 7. FLAGSHIP: the no-pry cheat scores nothing ===============================
    # Lid teleported straight off to the table + ketchup dropped into the open
    # cavity. The END STATE is byte-for-byte the goal state (asserted live: lid
    # flat+clear, bottle in, bar clear, settled) — but the pry credential was never
    # earned, so success is FALSE and the score stays ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.lid, chest_world((-0.02, -0.30, c.lid_size[2] / 2 + 0.002)),
                chest_quat())
    _step(90)
    _write_body(scene.ketchup,
                chest_world((0.0, 0.0, c.rim_top + c.bottle_size[2] / 2 + 0.012)),
                chest_quat())
    _step(240)
    _report("no-pry-cheat")
    end_state_complete = (bool(scene.lid_off_now()[0])
                          and bool(scene.ketchup_in_cavity()[0])
                          and bool(scene.bar_clear()[0]) and bool(scene.settled()[0]))
    check("FLAGSHIP (no-pry cheat): the lid lifted straight off + bottle placed in "
          "— the end state looks complete (asserted live) yet success is FALSE and "
          "score ~0: the pry credential was never earned",
          end_state_complete and not bool(scene._pried[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 9. wrong object also in the cavity =========================================
    restore()
    # drop the mustard BESIDE the standing ketchup (90 deg yaw: its 36 mm side runs
    # along chest-y, spanning y 0.032..0.068 — clear of the ketchup at |y|<0.026 and
    # of the wall at 0.080) so it lands standing inside the cavity, not on top.
    qz90 = torch.zeros(n, 4, device=device)
    qz90[:, 0] = qz90[:, 3] = math.sqrt(0.5)
    _write_body(scene.mustard,
                chest_world((0.0, 0.050, c.rim_top + c.bottle_size[2] / 2 + 0.012)),
                _qmul(chest_quat(), qz90))
    _step(240)
    _report("wrong-object")
    check("negative (wrong object): MUSTARD dropped into the cavity too — success "
          "revoked",
          bool(scene.in_cavity(scene.mustard.data.root_pos_w)[0])
          and not bool(scene.success()[0]))

    # ================= 10. bar left in the cavity =================================================
    restore()
    qz45 = torch.zeros(n, 4, device=device)
    ang = math.pi / 4
    qz45[:, 0], qz45[:, 3] = math.cos(ang / 2), math.sin(ang / 2)
    _write_body(scene.bar, chest_world((0.0, 0.0, 0.045)),
                _qmul(chest_quat(), qz45))
    _step(120)
    _report("bar-in-cavity")
    check("negative (tool left behind): the bar laid inside the open cavity — "
          "success revoked",
          not bool(scene.bar_clear()[0]) and not bool(scene.success()[0]))

    # ================= 11. lid balanced back on the chest =========================================
    restore()
    _write_body(scene.lid, chest_world((0.05, 0.0, 0.085)), chest_quat())
    _step(150)
    _report("lid-back-on")
    check("negative (lid back on the chest): the lid balanced over the chest again "
          "— lid_off_now false, success revoked",
          not bool(scene.lid_off_now()[0]) and not bool(scene.success()[0]))

    # ================= 12. bottle out: latched credit is capped ===================================
    restore()
    _write_body(scene.ketchup, chest_world((0.0, 0.35, c.bottle_size[2] / 2 + 0.002)),
                chest_quat())
    _step(90)
    _report("bottle-out")
    s12 = float(scene.score()[0])
    check("near-miss (bottle back out): all three latches stand but the live state "
          "is broken — no success, score capped at 0.40",
          not bool(scene.success()[0]) and 0.39 <= s12 <= 0.401)  # float32 +eps

    # ================= 13. the pry credential demands the bar =====================================
    # The lid HELD tilted proud by a direct write, bar far away on the table: the
    # `pried` predicate must NOT fire (no bar tip under the lid), the latch stays
    # clear, and the released lid falls back into its seat.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    qy = torch.zeros(n, 4, device=device)
    ang = math.radians(-8.0)  # about +y: negative angle raises the +x (front) edge
    qy[:, 0], qy[:, 2] = math.cos(ang / 2), math.sin(ang / 2)
    for _ in range(30):
        _write_body(scene.lid, chest_world((0.0, 0.0, 0.078)),
                    _qmul(chest_quat(), qy))
        _step(1)
    _refresh()
    ez13 = edge_z()
    proud13 = ez13 > c.rim_top + c.pried_proud
    fired13 = bool(scene.pried_now()[0]) or bool(scene._pried[0])
    _step(150)  # released: the lid falls back into the recess
    _report("barless-tilt")
    ll = scene.lid_loc()[0]
    reseated = abs(float(ll[0])) < 0.012 and abs(float(ll[1])) < 0.012 \
        and float(ll[2]) < 0.075
    check("pry credential demands the bar: the lid held tilted proud with the bar "
          "far away never fires `pried`, and the released lid falls back to seat",
          proud13 and not fired13 and reseated
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pry_lid_vault")
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
