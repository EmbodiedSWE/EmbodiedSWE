"""Smoke / rubric-REJECTION battery for HoodPropScene (sim_gen task
`lift_peg_upright_i116`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — torque-servo the lid onto the back-stop, drop the
red peg into the socket, lower the lid onto it — is the acceptance evidence that the
rubric ACCEPTS a correct outcome; it passes on forge seeds 0/1/2 covering both start
states). Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it — plus physics probes
proving the mechanism is real (the propped band is gravitationally unstable in BOTH
directions without the peg). No probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1.  settle/no-NaN      — closed-start reset settles finite, lid RESTS shut on the
                           rim (~0 deg), score ~0, no success;
  2a-b. randomization    — READBACK over 12 seeded resets: chest xy+yaw and peg
                           positions vary; BOTH lid start states (closed / parked on
                           the back-stop) occur;
  3.  null policy closed — 300 idle steps -> score ~0, no success;
  4.  null policy open   — open-start episode, 300 idle steps: the pre-parked lid
                           earns NO opened-credit (latch gated on started_closed),
                           score ~0, no success;
  5.  seed strategy      — maniskill/lift_peg_upright's whole goal (the red peg stood
                           upright FREE on the ground, correct height band, settled)
                           earns ~0 here: no socket, lid untouched, no success;
  6.  peg on the lid     — red peg stood upright ON TOP of the closed lid: upright
                           but not in the socket -> no credit, no success;
  7.  near-miss xy       — red peg upright ON the sill RIGHT BESIDE the collar
                           (~9 cm off the socket axis, inside the box): outside
                           sock_xy_tol -> no peg credit, no success;
  8.  out-of-order end   — red peg properly seated in the socket but the lid LEFT ON
                           THE BACK-STOP (too far open): partial credit only (~0.25),
                           no success;
  9.  short-peg decoy    — blue peg seated in the socket, lid released just above it:
                           it settles propped ~31 deg — genuinely BELOW band_min —
                           and the decoy earns no peg credit, no success;
  10. band unstable down — lid POSED inside the band (55 deg) with NO peg: judged
                           immediately it is already not-success, and hands-off it
                           FALLS SHUT (the band cannot be occupied without support);
  11. hinge_ok clause    — lid POSED at a band angle but LIFTED OFF the hinge (8 cm
                           above, peg in socket, zero velocity): rejected by the
                           hinge check — a dislodged lid cannot fake the prop;
  12. band unstable up   — lid posed at 100 deg falls BACK onto the stop (~106 deg >
                           band_max), not into the band: no success from above either;
  13. rejection audit    — success() was never True at ANY judged point;
  14. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.lift_peg_upright_i116.smoke --headless
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
# RTX recipe: kit mis-decodes the pod driver version and silently rejects RTX -> the
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qmul = scene_mod._qmul

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hood_prop")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.15, -0.95, 0.85)) + o),
                                tuple(np.array((0.05, 0.00, 0.12)) + o),
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

    def theta_deg() -> float:
        return math.degrees(float(scene.lid_angle()[0]))

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | theta={theta_deg():+7.2f} "
              f"hinge_ok={bool(scene.hinge_ok()[0])} "
              f"pegged={bool(scene.peg_socketed()[0])} in_band={bool(scene.in_band()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, settle_steps: int = 45) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def chest_frame(local) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return scene.chest.data.root_pos_w + quat_apply(
            scene.chest.data.root_quat_w,
            torch.tensor(local, device=device, dtype=torch.float32).expand(n, 3))

    def stand_peg(body, half_l: float, local_xy, dz: float = 0.003,
                  settle_steps: int = 150) -> None:
        """Stand a peg upright at chest-frame xy, foot `dz` above the surface at
        chest-frame z=`local_xy[2]`, and release it."""
        pos = chest_frame([local_xy[0], local_xy[1], local_xy[2] + half_l + dz])
        teleport(body, pos, scene.chest.data.root_quat_w, settle_steps=settle_steps)

    def pose_lid(theta: float, dz: float = 0.0, settle_steps: int = 0) -> None:
        """Write the lid at opening angle `theta` (deg) on (or `dz` above) its hinge."""
        ph = -math.radians(theta) / 2
        q_pitch = torch.tensor([math.cos(ph), 0.0, math.sin(ph), 0.0],
                               device=device).expand(n, 4)
        q_lid = _qmul(scene.chest.data.root_quat_w, q_pitch)
        pos = scene.chest.data.root_pos_w.clone()
        pos[:, 2] += 0.0605 + dz
        teleport(scene.lid, pos, q_lid, settle_steps=settle_steps)

    def find_seed(start: int, want_closed: bool) -> int:
        """Probe seeded resets until the wanted lid start state holds (readback)."""
        for sd in range(start, start + 16):
            env.reset(seed=sd)
            step(2)
            if bool(scene.started_closed[0]) == want_closed:
                return sd
        raise AssertionError("no seed with the wanted lid start state in 16 probes")

    def all_finite() -> bool:
        bodies = [scene.chest, scene.lid, scene.peg_long, scene.peg_short]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def peg_up(body) -> bool:
        from isaaclab.utils.math import quat_apply

        ax = quat_apply(body.data.root_quat_w,
                        torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
        return float(ax[0, 2].abs()) > math.cos(math.radians(15.0))

    # =========================== 1. settle / no-NaN (closed start) =======================
    sd_closed = find_seed(2, want_closed=True)
    step(240)
    report("reset-closed")
    s, ok = judge()
    check(f"settle (closed-start seed {sd_closed}): all states finite, lid RESTS shut "
          f"on the rim (theta={theta_deg():+.2f} deg ~ 0), score ~0, no success",
          all_finite() and abs(theta_deg()) < 5.0 and bool(scene.lid_settled()[0])
          and s <= 0.02 and not ok)

    # =========================== 2. randomization is real ================================
    reads, starts = [], []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(2)
        cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
        q = scene.chest.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        pl = (scene.peg_long.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(cp[0]), float(cp[1]), yaw, float(pl[0]), float(pl[1])))
        starts.append(bool(scene.started_closed[0]))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (chest_x, chest_y, yaw, peg_x, peg_y):\n"
          f"{arr.round(3)}\n[smoke] started_closed: {starts}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: chest pose (xy + yaw) and peg positions vary across seeded "
          f"resets (readback: dx={spread[0]:.3f} dy={spread[1]:.3f} "
          f"dyaw={spread[2]:.2f} dpeg={spread[3]:.3f})",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.5 and spread[3] > 0.10)
    check("randomization: BOTH lid start states occur (closed "
          f"{starts.count(True)}/12, parked-open {starts.count(False)}/12)",
          starts.count(True) >= 2 and starts.count(False) >= 2)

    # =========================== 3. null policy (closed start) ===========================
    find_seed(2, want_closed=True)
    step(300)
    report("null-closed")
    s, ok = judge()
    check("null policy, closed start: score ~0 and no success after 300 idle steps",
          s <= 0.02 and not ok)

    # =========================== 4. null policy (open start, no freebie) =================
    sd_open = find_seed(0, want_closed=False)
    step(300)
    report("null-open")
    s, ok = judge()
    check(f"null policy, open start (seed {sd_open}): the pre-parked lid "
          f"(theta={theta_deg():+.1f}) earns NO opened-credit — score ~0, no success",
          s <= 0.02 and not ok and theta_deg() > 90.0)

    # =========================== 5. seed strategy (peg upright, free) ====================
    sd2 = find_seed(2, want_closed=True)
    step(120)
    # maniskill/lift_peg_upright's SUCCESS state: the red peg stood upright on the
    # open ground (correct height band, settled). Here: worthless.
    pos = chest_frame([0.90, 0.90, 0.0])
    pos[:, 2] = scene.env_origins[:, 2] + c.peg_l_long / 2 + 0.003
    teleport(scene.peg_long, pos, scene.chest.data.root_quat_w, settle_steps=150)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (the seed task's goal: red peg stood upright FREE on the "
          f"ground, standing={peg_up(scene.peg_long)}): score ~0, lid untouched "
          f"(theta={theta_deg():+.1f}), no success",
          peg_up(scene.peg_long) and s <= 0.02 and not ok and abs(theta_deg()) < 5.0)

    # =========================== 6. peg upright ON the closed lid ========================
    stand_peg(scene.peg_long, c.peg_l_long / 2, (0.15, 0.0, 0.068), settle_steps=150)
    report("peg-on-lid")
    s, ok = judge()
    check("red peg stood upright ON TOP of the closed lid: upright but not in the "
          "socket -> no peg credit, no success",
          s <= 0.02 and not ok and not bool(scene.peg_socketed()[0]))

    # =========================== 7. near-miss: upright on the sill, beside the collar ====
    sd_open2 = find_seed(0, want_closed=False)
    step(60)
    stand_peg(scene.peg_long, c.peg_l_long / 2, (0.15, 0.09, 0.012), settle_steps=150)
    d_xy = float((scene.peg_long.data.root_pos_w[0, :2]
                  - scene._sock_center_w()[0, :2]).norm())
    report("near-miss-xy")
    s, ok = judge()
    check("near-miss: red peg upright ON the sill beside the collar "
          f"(|xy-socket|={d_xy:.3f} m > tol {c.sock_xy_tol}): no peg credit, no success",
          peg_up(scene.peg_long) and d_xy > c.sock_xy_tol and s <= 0.02 and not ok)

    # =========================== 8. out-of-order: pegged but lid left on the stop ========
    stand_peg(scene.peg_long, c.peg_l_long / 2, (c.sock_x, 0.0, 0.012), settle_steps=180)
    report("pegged-lid-parked")
    s, ok = judge()
    check("out-of-order end state: red peg seated in the socket but the lid LEFT ON "
          f"the back-stop (theta={theta_deg():+.1f} > band_max {c.band_max:.0f}): "
          f"partial credit only (score={s:.2f} ~ 0.25), no success",
          bool(scene.peg_socketed()[0]) and theta_deg() > c.band_max + 5.0
          and 0.20 <= s <= 0.30 and not ok)

    # =========================== 11 (setup shared): hinge_ok clause ======================
    # Lid lifted OFF the hinge to a band angle, peg in socket, zero velocity: every
    # other predicate holds at the judged instant, only hinge_ok rejects it.
    pose_lid(55.0, dz=0.08, settle_steps=0)
    step(2)  # refresh buffers only — judge before it can fall
    in_band_now = bool(scene.in_band()[0])
    pegged_now = bool(scene.peg_socketed()[0])
    hinge_now = bool(scene.hinge_ok()[0])
    s, ok = judge()
    check("hinge_ok clause: lid POSED at a band angle but 8 cm OFF its hinge (peg in "
          f"socket, zero vel; in_band={in_band_now}, pegged={pegged_now}, "
          f"hinge_ok={hinge_now}): a dislodged lid cannot fake the prop — no success",
          in_band_now and pegged_now and not hinge_now and not ok and s <= 0.30)
    # destroy the construction: park the lid flat on the floor far from the chest
    pos = chest_frame([0.0, -0.90, 0.0])
    pos[:, 2] = scene.env_origins[:, 2] + 0.05
    teleport(scene.lid, pos, scene.chest.data.root_quat_w, settle_steps=60)

    # =========================== 9. short-peg decoy near-miss ============================
    find_seed(0, want_closed=False)
    step(60)
    stand_peg(scene.peg_short, c.peg_l_short / 2, (c.sock_x, 0.0, 0.012),
              settle_steps=180)
    assert bool(scene.peg_socketed(scene.peg_short)[0]), "decoy failed to seat"
    pose_lid(c.prop_deg_short + 3.5, settle_steps=0)  # released just above the decoy
    step(420)  # hands-off: the lid drops the ~3 deg onto the short peg and rests
    report("decoy-propped")
    s, ok = judge()
    check("short-peg decoy: blue peg seated in the socket genuinely props the lid — "
          f"but at theta={theta_deg():+.1f} deg, BELOW band_min {c.band_min:.0f} "
          "(and it earns no peg credit): no success",
          bool(scene.peg_socketed(scene.peg_short)[0]) and 15.0 < theta_deg() < c.band_min
          and s <= 0.02 and not ok)

    # =========================== 10. band unstable downward ==============================
    find_seed(2, want_closed=True)
    step(60)
    pose_lid(55.0, settle_steps=0)
    step(2)
    s_immediate, ok_immediate = judge()
    step(420)  # hands-off: nothing under the lid — it must fall shut
    report("fake-prop-fell")
    s, ok = judge()
    check("band unstable downward: lid POSED at 55 deg with NO peg is not success at "
          f"the judged instant (success={ok_immediate}) and hands-off it FALLS SHUT "
          f"(settled theta={theta_deg():+.2f} deg < 5)",
          not ok_immediate and s_immediate <= 0.02 and abs(theta_deg()) < 5.0
          and not ok and s <= 0.02)

    # =========================== 12. band unstable upward ================================
    pose_lid(100.0, settle_steps=0)
    step(420)  # hands-off: gravity pulls it BACK onto the stop, not into the band
    report("posed-100")
    s, ok = judge()
    check("band unstable upward: lid posed at 100 deg settles BACK onto the back-stop "
          f"(theta={theta_deg():+.1f} > band_max {c.band_max:.0f}), not into the band "
          "— no success (score may hold the opened stage credit only)",
          theta_deg() > c.band_max + 5.0 and not ok and s <= 0.16)

    # =========================== 13-14. audit + no-NaN ===================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict ==========================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hood_prop")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
