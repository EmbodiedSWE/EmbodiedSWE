"""Smoke / rubric-REJECTION battery for CoveredDishScene (sim_gen task
`track_bowl_i261`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — gravity-seat the bowl's boss in the stand's
socket, gravity-drop the egg inside, gravity-cap with the real lid — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts
the rubric REJECTS it; no probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: bowl / egg / both lids at
                            their ground rest heights (readback); score ~0, no
                            success;
  3-4. randomization      — READBACK over 6 seeded resets: stand xy + yaw move,
                            bowl / egg / lid / decoy stand-local xy all move, and
                            the lid/decoy slot coin flip takes BOTH values;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("grasp the bowl, carry it along
                            a path, set it down at the goal") = the bowl alone
                            transported and gravity-SEATED in the socket: seat
                            credit only (score ~0.20), NOT success;
  7.  fouled socket       — the egg dropped into the BARE socket, then the bowl
                            seated on top: the boss rides on the debris ~30 mm
                            above the seat band (readback) -> bowl_seated False,
                            score ~0;
  8.  inverted bowl       — the bowl dropped UPSIDE-DOWN over the socket: fails
                            the upright cone + height band -> no seat credit;
  9.  egg on capped lid   — bowl seated, real lid capped (empty), egg dropped on
                            TOP of the lid: lid genuinely seated, but the egg is
                            OUTSIDE (physics forces egg-before-lid) -> egg_in
                            False, asm never latched, score ~0.20, NOT success;
  10. lid rim-perch       — the real lid dropped 45 mm OFF-AXIS onto the seated
                            bowl: disk on the rim / skirt fouling the wall, far
                            outside the concentric tolerance -> lid_seated False;
  11. decoy trap          — the decoy dropped on the mouth falls INSIDE (readback)
                            and the real lid dropped after rides on its knob ~11 mm
                            above the rim band -> lid_seated False, capping is
                            physically blocked, NOT success;
  12. off-fixture dish    — bowl on the GROUND, egg genuinely inside, lid genuinely
                            seated (both readback) -> the missing SEAT clause alone
                            rejects: score ~0, NOT success;
  13. latched credit      — after real seat + fill credit, the egg teleported back
                            OUT of the bowl: the latched score survives unchanged
                            (and still no success);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.track_bowl_i261.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.covered_dish")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.40, -0.95, 0.80)) + o),
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

    def report(tag: str) -> None:
        bl = scene._stand_local(scene.bowl.data.root_pos_w)[0]
        el = scene._bowl_local(scene.egg.data.root_pos_w)[0]
        ll = scene._bowl_local(scene.lid.data.root_pos_w)[0]
        dl = scene._bowl_local(scene.decoy.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bowl_stand=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) egg_bowl=({float(el[0]):+.3f},{float(el[1]):+.3f},"
              f"{float(el[2]):+.3f}) lid_bowl=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
              f"{float(ll[2]):+.3f}) decoy_bowl_z={float(dl[2]):+.3f} "
              f"seated={bool(scene.bowl_seated()[0])} egg_in={bool(scene.egg_in_bowl()[0])} "
              f"lid_on={bool(scene.lid_seated()[0])} decoy_in={bool(scene.decoy_in_bowl()[0])} "
              f"settled={bool(scene.settled()[0])} seatL={float(scene.seat_latch[0]):.2f} "
              f"eggL={float(scene.egg_latch[0]):.2f} asmL={float(scene.asm_latch[0]):.2f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def hover_local(body, ref, lx: float, ly: float, lz: float,
                    quat: tuple | None = None, settle_steps: int = 60) -> None:
        """Kinematic probe placement in `ref`'s BODY frame (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap).
        Orientation: `ref`'s own orientation unless a world quat is given."""
        from isaaclab.utils.math import quat_apply

        tgt = torch.tensor([lx, ly, lz], device=device).expand(n, 3)
        pos = ref.data.root_pos_w + quat_apply(ref.data.root_quat_w, tgt)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        if quat is None:
            st[:, 3:7] = ref.data.root_quat_w
        else:
            st[:, 3:7] = torch.tensor(quat, device=device).expand(n, 4)
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def seat_bowl() -> None:
        """The solution's P1 move: hover over the socket, gravity-drop the boss in."""
        hover_local(scene.bowl, scene.stand, 0.004, 0.0,
                    c.collar_h + 0.015 + c.bowl_org_h, settle_steps=90)

    def fin_all() -> bool:
        return bool(torch.isfinite(scene.stand.data.root_state_w).all()
                    and torch.isfinite(scene.bowl.data.root_state_w).all()
                    and torch.isfinite(scene.lid.data.root_state_w).all()
                    and torch.isfinite(scene.decoy.data.root_state_w).all()
                    and torch.isfinite(scene.egg.data.root_state_w).all())

    def ground_z(body) -> float:
        return float((body.data.root_pos_w - scene.env_origins)[0][2])

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    check("settle: states finite; bowl / egg / both lids resting at their ground rest "
          "heights (readback), everything settled",
          fin_all() and abs(ground_z(scene.bowl) - c.bowl_org_h) < 0.008
          and abs(ground_z(scene.egg) - c.egg_r) < 0.008
          and abs(ground_z(scene.lid) - c.lid_skirt_h) < 0.008
          and abs(ground_z(scene.decoy) - c.dec_skirt_h) < 0.008
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.03), no success", s <= 0.03 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        q = scene.stand.data.root_quat_w[0]
        syaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        bl = scene._stand_local(scene.bowl.data.root_pos_w)[0]
        el = scene._stand_local(scene.egg.data.root_pos_w)[0]
        ll = scene._stand_local(scene.lid.data.root_pos_w)[0]
        dl = scene._stand_local(scene.decoy.data.root_pos_w)[0]
        reads.append((float(sp[0]), float(sp[1]), syaw, float(scene.lid_slot[0]),
                      float(bl[0]), float(bl[1]), float(el[0]), float(el[1]),
                      float(ll[0]), float(ll[1]), float(dl[0]), float(dl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, lid_slot, "
          f"bowl_x, bowl_y, egg_x, egg_y, lid_x, lid_y, decoy_x, decoy_y):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw vary AND the lid/decoy slot coin flip takes "
          "both values across seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05
          and arr[:, 3].min() < -0.5 and arr[:, 3].max() > 0.5)
    check("randomization: bowl, egg, lid, and decoy stand-local xy all vary across "
          "seeded resets (readback)",
          spread[4] > 0.01 and spread[5] > 0.01 and spread[6] > 0.01
          and spread[7] > 0.01 and spread[8] > 0.01 and spread[9] > 0.01
          and spread[10] > 0.01 and spread[11] > 0.01)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "grasp the bowl, carry it along a path, set it down at
    # the goal". The best that plan can earn here — the bowl alone transported and
    # genuinely gravity-seated in the socket — is the 0.20 seat credit, nothing more.
    env.reset(seed=41)
    step(10)
    seat_bowl()
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (the bowl alone carried and set down seated in the socket): "
          "seat credit only — bowl_seated True, 0.19 <= score <= 0.21, NOT success",
          bool(scene.bowl_seated()[0]) and 0.19 <= s <= 0.21 and not ok)

    # =========================== 7. fouled socket ===========================================
    # Debris in the fixture: the egg dropped into the BARE socket, then the bowl
    # seated exactly as the solution does — the boss rides on the egg ~30 mm above
    # the seat band. bowl_seated must read False and no seat credit may latch.
    env.reset(seed=51)
    step(10)
    hover_local(scene.egg, scene.stand, 0.0, 0.0, c.collar_h + 0.020 + c.egg_r,
                quat=(1.0, 0.0, 0.0, 0.0), settle_steps=60)
    egg_in_socket = float(scene._stand_local(scene.egg.data.root_pos_w)[0][2])
    seat_bowl()
    report("fouled-socket")
    bowl_z = float(scene._stand_local(scene.bowl.data.root_pos_w)[0][2])
    s, ok = judge()
    check("fouled socket: egg debris in the bare socket (readback), the seated bowl "
          "rides on it above the seat band (readback) -> bowl_seated False, score ~0",
          egg_in_socket < c.collar_h and bowl_z > c.seat_z + c.seat_z_tol + 0.010
          and not bool(scene.bowl_seated()[0]) and s <= 0.03 and not ok)

    # =========================== 8. inverted bowl ===========================================
    # The bowl dropped UPSIDE-DOWN over the socket: its rim lands on the collar, the
    # boss points at the sky — upright cone and height band both reject.
    env.reset(seed=61)
    step(10)
    hover_local(scene.bowl, scene.stand, 0.0, 0.0, c.collar_h + c.wall_h + 0.015,
                quat=(0.0, 1.0, 0.0, 0.0), settle_steps=90)
    report("inverted-bowl")
    s, ok = judge()
    check("inverted bowl over the socket: bowl_seated False (upright cone + height "
          "band), no seat credit, NOT success",
          not bool(scene.bowl_seated()[0]) and s <= 0.03 and not ok)

    # =========================== 9. egg on the capped lid ===================================
    # Seat the bowl, cap the EMPTY bowl with the real lid (a genuine seat — readback),
    # then drop the egg on TOP: physics forces egg-before-lid, so the egg stays
    # OUTSIDE, the assembly latch never fires, and the score stays at seat credit.
    env.reset(seed=71)
    step(10)
    seat_bowl()
    hover_local(scene.lid, scene.bowl, 0.004, 0.0, c.wall_h + c.lid_skirt_h + 0.012,
                settle_steps=120)
    lid_on_empty = bool(scene.lid_seated()[0])
    hover_local(scene.egg, scene.bowl, 0.010, 0.0,
                c.wall_h + c.lid_disk_t + c.knob_h + c.cap_t + c.egg_r + 0.020,
                quat=(1.0, 0.0, 0.0, 0.0), settle_steps=120)
    report("egg-on-lid")
    s, ok = judge()
    check("egg on the capped lid: the lid seats genuinely on the empty bowl "
          "(readback) but the egg dropped on top stays OUTSIDE -> egg_in False, "
          "asm never latched, score <= 0.21, NOT success",
          lid_on_empty and not bool(scene.egg_in_bowl()[0])
          and float(scene.asm_latch[0]) < 0.5 and s <= 0.21 and not ok)

    # =========================== 10. lid rim-perch ==========================================
    # The real lid dropped 45 mm OFF-AXIS onto the seated bowl: the disk lands on
    # the rim / the skirt fouls the wall — far outside the 12 mm concentric band.
    env.reset(seed=81)
    step(10)
    seat_bowl()
    hover_local(scene.lid, scene.bowl, 0.045, 0.0, c.wall_h + c.lid_skirt_h + 0.012,
                settle_steps=120)
    report("rim-perch")
    lid_xy = float(scene._bowl_local(scene.lid.data.root_pos_w)[0][:2].norm())
    s, ok = judge()
    check("lid rim-perch: the real lid dropped 45 mm off-axis rests far outside the "
          "concentric tolerance (readback) -> lid_seated False, score <= 0.21",
          lid_xy > c.lid_xy_tol and not bool(scene.lid_seated()[0])
          and s <= 0.21 and not ok)

    # =========================== 11. decoy trap =============================================
    # The wrong-object failure is physical: the decoy dropped on the mouth falls
    # INSIDE the bowl (readback), and the real lid dropped after rides on the decoy's
    # tall knob above the rim band — capping is blocked until the decoy is removed.
    env.reset(seed=91)
    step(10)
    seat_bowl()
    hover_local(scene.decoy, scene.bowl, 0.002, 0.0,
                c.wall_h + c.dec_skirt_h + 0.020, settle_steps=120)
    decoy_z = float(scene._bowl_local(scene.decoy.data.root_pos_w)[0][2])
    decoy_in = bool(scene.decoy_in_bowl()[0])
    hover_local(scene.lid, scene.bowl, 0.004, 0.0, c.wall_h + c.lid_skirt_h + 0.012,
                settle_steps=120)
    report("decoy-trap")
    lid_z = float(scene._bowl_local(scene.lid.data.root_pos_w)[0][2])
    s, ok = judge()
    check("decoy trap: the decoy dropped on the mouth falls INSIDE (readback, rest "
          "below the rim plane) and the real lid dropped after rides on its knob "
          "above the rim band (readback) -> lid_seated False, capping blocked",
          decoy_in and decoy_z < c.wall_h - 0.020
          and lid_z > c.lid_z + c.lid_z_tol - 0.001
          and not bool(scene.lid_seated()[0]) and not ok)

    # =========================== 12. off-fixture dish =======================================
    # A complete covered dish assembled on the GROUND: egg genuinely inside, lid
    # genuinely seated (both readback) — the missing SEAT clause alone rejects it,
    # and no latch fires (egg/asm credit is gated on the seat).
    env.reset(seed=101)
    step(10)
    hover_local(scene.egg, scene.bowl, 0.005, 0.0, c.wall_h + 0.040 + c.egg_r,
                quat=(1.0, 0.0, 0.0, 0.0), settle_steps=90)
    hover_local(scene.lid, scene.bowl, 0.004, 0.0, c.wall_h + c.lid_skirt_h + 0.012,
                settle_steps=120)
    report("off-fixture")
    s, ok = judge()
    check("off-fixture dish: egg inside + lid seated on the GROUNDED bowl (readback) "
          "but the bowl is not in the socket -> score ~0, NOT success",
          bool(scene.egg_in_bowl()[0]) and bool(scene.lid_seated()[0])
          and not bool(scene.bowl_seated()[0]) and s <= 0.03 and not ok)

    # =========================== 13. latched credit survives regression =====================
    env.reset(seed=111)
    step(10)
    seat_bowl()
    hover_local(scene.egg, scene.bowl, 0.005, 0.0, c.wall_h + 0.040 + c.egg_r,
                quat=(1.0, 0.0, 0.0, 0.0), settle_steps=120)
    report("seat+fill")
    s_in, _ok = judge()
    hover_local(scene.egg, scene.stand, -0.35, 0.0, -c.stand_h + c.egg_r + 0.002,
                quat=(1.0, 0.0, 0.0, 0.0), settle_steps=60)
    report("egg-removed")
    s_out, ok = judge()
    check("latched credit: teleporting the egg back OUT of the seated+filled bowl "
          "leaves the latched score unchanged (and still no success)",
          s_in >= 0.44 and abs(s_out - s_in) < 0.02
          and not bool(scene.egg_in_bowl()[0]) and not ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.covered_dish")
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
    main()
