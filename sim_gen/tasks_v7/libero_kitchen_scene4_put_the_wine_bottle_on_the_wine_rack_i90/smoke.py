"""Smoke / rubric-REJECTION battery for LipHangRackScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i90`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — bottle staged at the slot mouth, lip caught on the
rails by a real drop, force-slid to the tower stop — is the acceptance evidence that
the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN     — reset layout settles finite: bottle standing on the floor,
                           tags at OPPOSITE slots (readback), all still, score ~0;
  3-4. randomization     — READBACK over 8 seeded resets: the GREEN side flips (tag
                           bodies physically swap, matching _green_v) and the bottle
                           spawn side flips; rack yaw spread and xy jitter are real;
  5.  null policy        — 240 idle steps -> score ~0, no success;
  6.  seed strategy      — the end state the SEED's plan produces here (put the
                           bottle ON the rack): the bottle laid HORIZONTALLY across
                           the rack's rail tops (body bridging the two slots' inner
                           rails), resting ON the rack -> NOT success, score <= 0.20
                           (the rubric demands the HANG: support from above, body in
                           free air);
  7.  tower top          — bottle standing upright ON the tower top (the other
                           "on the rack" reading) -> NOT success, score <= 0.20;
  8.  under the slot     — bottle standing on the FLOOR directly below the green
                           slot -> score ~0, no success (position under the goal is
                           not the goal);
  9.  near-miss depth    — GENUINE hang in the green slot (lip on the rails, built by
                           a real 3 mm drop) but at mid-rail depth, SHORT of the
                           tower stop -> NOT success, score <= 0.601;
  10. latched credit     — teleporting the bottle from that hang back to the floor
                           leaves the latched score unchanged, still no success;
  11. wrong slot         — GENUINE hang at stop depth but in the RED-tagged slot ->
                           NOT success, score <= 0.20 (tag identification is
                           load-bearing; threading credit is green-gated);
  12. false middle slot  — bottle released upright BETWEEN the two slots (the 76 mm
                           middle span is wider than the 50 mm lip): it falls
                           STRAIGHT THROUGH to the floor -> NOT success (the metric
                           interlock is physically real);
  13. transient motion   — bottle in the success pose but MOVING (judged one step
                           after a 0.6 m/s injection) -> the settle gate rejects at
                           that instant; the probe is removed without settling;
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
    env = ENVS.get("simgen.lip_hang_rack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, -1.15, 0.85)) + o),
                                tuple(np.array((0.55, 0.00, 0.22)) + o),
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
        loc = scene._bottle_loc()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) depth={float(scene._depth()[0]):+.3f} "
              f"up={float(scene._bottle_up()[0]):+.3f} green_v={float(scene._green_v[0]):+.3f} "
              f"lift={bool(scene._lift_ever[0])} mouth={bool(scene._mouth_ever[0])} "
              f"depth_max={float(scene._depth_max[0]):.3f} "
              f"settled={bool(scene._settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_rack_local(x: float, v: float, z: float, quat_local=None,
                         vel_w=None) -> None:
        """Teleport the bottle to a rack-local pose of the rack's CURRENT frame
        (probe constructor), optionally with a local orientation and world velocity."""
        r_pos = scene.rack.data.root_pos_w
        r_quat = scene.rack.data.root_quat_w
        loc = torch.tensor([x, v, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = r_pos + quat_apply(r_quat, loc)
        if quat_local is None:
            st[:, 3] = 1.0
        else:
            ql = torch.tensor(quat_local, device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(r_quat, ql)
        if vel_w is not None:
            st[:, 7:10] = torch.tensor(vel_w, device=device).expand(n, 3)
        scene.bottle.write_root_state_to_sim(st, all_ids)

    def place_floor(x: float, y: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, c.stand_z + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.bottle.write_root_state_to_sim(st, all_ids)

    def rack_yaw() -> float:
        q = scene.rack.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.rack.data.root_state_w).all()
                    and torch.isfinite(scene.bottle.data.root_state_w).all()
                    and torch.isfinite(scene.green_tag.data.root_state_w).all()
                    and torch.isfinite(scene.red_tag.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    bz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    gt_v = float(scene._rack_local(scene.green_tag.data.root_pos_w)[0, 1])
    rt_v = float(scene._rack_local(scene.red_tag.data.root_pos_w)[0, 1])
    gv = float(scene._green_v[0])
    check("settle: states finite, bottle STANDING on the floor, green/red tags at "
          "OPPOSITE slots and the green tag at the green slot (readback), all still",
          finite_all() and abs(bz - c.stand_z) < 0.010
          and abs(gt_v - gv) < 0.005 and abs(rt_v + gv) < 0.005
          and bool(scene._settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        b = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        r = (scene.rack.data.root_pos_w - scene.env_origins)[0]
        gtv = float(scene._rack_local(scene.green_tag.data.root_pos_w)[0, 1])
        reads.append((float(scene._green_v[0]), gtv, float(b[1]),
                      float(r[0]), float(r[1]), rack_yaw()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (green_v, green_tag_v, bottle_y, rack_x, "
          f"rack_y, rack_yaw_deg):\n{arr}", flush=True)
    g_flags = (arr[:, 0] > 0).astype(float)
    b_flags = (arr[:, 2] > 0).astype(float)
    check("randomization A: the GREEN side flips across seeded resets AND the tag "
          "body readback matches _green_v every time AND the bottle spawn side flips",
          0.0 < g_flags.mean() < 1.0 and 0.0 < b_flags.mean() < 1.0
          and all(abs(r[0] - r[1]) < 0.005 for r in reads))
    yaw_spread = float(arr[:, 5].max() - arr[:, 5].min())
    rack_jit = float((arr[:, 3:5].max(axis=0) - arr[:, 3:5].min(axis=0)).max())
    check("randomization B: rack yaw spread (> 2 deg) and rack xy jitter (> 4 mm) "
          "are real (readback)", yaw_spread > 2.0 and rack_jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: rest the bottle ON the rack ==============
    # The seed's relation — bottle supported from below by the rack — constructed
    # here: the bottle laid HORIZONTALLY across the rack, its (uniform-radius) BODY
    # bridging the two slots' INNER rails symmetrically, so it rests level and
    # stable ON the rail tops. The rubric demands the hang, judged as-is.
    torch.manual_seed(41)
    env.reset()
    step(30)
    q_lie = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)  # +z -> rack +y
    # body centre sits 0.0425 m from the origin along -z: offset the origin so the
    # BODY (not the assembly origin) is centred between the two inner rails
    place_rack_local(c.face_x + 0.06, 0.0425,
                     c.rail_top_z + c.r_body + 0.004, quat_local=q_lie)
    step(180)
    report("seed-strategy")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("seed strategy: bottle laid ACROSS the rack's rail tops (body bridging the "
          "two inner rails) rests ON the rack, level and still — NOT success, "
          "score <= 0.20 (the task demands the HANG, not the seed's rest-on-top)",
          float(loc[2]) > c.rail_top_z - 0.02 and abs(float(scene._bottle_up()[0])) < 0.5
          and not ok and s <= 0.20)

    # =========================== 7. tower top ===============================================
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_rack_local(0.0, 0.0, c.tower_h + c.stand_z + 0.004)
    step(240)
    report("tower-top")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("tower top: bottle standing upright ON the tower top — NOT success, "
          "score <= 0.20 (still the seed's support-from-below relation)",
          float(loc[2]) > c.tower_h - 0.02 and not ok and s <= 0.20)

    # =========================== 8. standing under the slot =================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    gv = float(scene._green_v[0])
    place_rack_local(c.face_x + 0.06, gv, c.stand_z + 0.002)
    step(180)
    report("under-slot")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("under the slot: bottle standing on the FLOOR directly below the green "
          "slot — score ~0 (<= 0.02), no success (right (x,y), wrong relation)",
          float(loc[2]) < 0.15 and not ok and s <= 0.02)

    # =========================== 9. near-miss: hang SHORT of the stop =======================
    # GENUINE hang (real 3 mm drop onto the rails) but at mid-rail depth 0.075 —
    # outside the stop band [d_min, d_stop]: the slide-to-the-stop is load-bearing.
    torch.manual_seed(71)
    env.reset()
    step(30)
    gv = float(scene._green_v[0])
    place_rack_local(c.face_x + 0.075, gv, c.hang_z0 + 0.003)
    step(240)
    report("short-of-stop")
    loc = scene._bottle_loc()[0]
    d9 = float(scene._depth()[0])
    s9, ok = judge()
    check("near-miss depth: GENUINE lip-hang in the green slot but at mid-rail depth "
          "(short of the tower stop) — NOT success, score <= 0.601",
          abs(float(loc[2]) - c.hang_z0) < 0.012 and d9 > c.d_stop + 0.01
          and not ok and s9 <= 0.601)

    # =========================== 10. latched credit survives regression =====================
    place_floor(0.30, -0.25)
    step(90)
    report("regressed")
    s10, ok = judge()
    loc = scene._bottle_loc()[0]
    check("latched credit: teleporting the bottle from that hang back to the floor "
          f"leaves the latched score unchanged ({s9:.3f} -> {s10:.3f}), still no "
          "success", abs(s10 - s9) < 1e-3 and float(loc[2]) < 0.15 and not ok)

    # =========================== 11. wrong slot (RED) =======================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    gv = float(scene._green_v[0])
    place_rack_local(c.face_x + 0.038, -gv, c.hang_z0 + 0.003)  # RED slot, stop depth
    step(240)
    report("red-slot")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("wrong slot: GENUINE lip-hang at STOP depth but in the RED-tagged slot — "
          "NOT success, score <= 0.20 (threading credit is green-gated; tag "
          "identification is load-bearing)",
          abs(float(loc[2]) - c.hang_z0) < 0.012 and abs(float(loc[1]) + gv) < 0.012
          and float(scene._depth()[0]) < c.d_stop and not ok and s <= 0.20)

    # =========================== 12. false middle slot ======================================
    # Released upright BETWEEN the slots: the 76 mm middle span is wider than the
    # 50 mm lip — the bottle falls straight through to the floor. The interlock that
    # makes "some pair of rails will do" physically false.
    torch.manual_seed(91)
    env.reset()
    step(30)
    place_rack_local(c.face_x + 0.075, 0.0, c.hang_z0 + 0.003)
    step(240)
    report("middle-slot")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("false middle slot: bottle released upright BETWEEN the two slots falls "
          "STRAIGHT THROUGH (middle span 76 mm > lip 50 mm) and ends on the floor — "
          "NOT success",
          float(loc[2]) < 0.20 and not ok)

    # =========================== 13. transient motion (settle gate) =========================
    # Success pose but MOVING: inject the pose with a 0.6 m/s escape velocity along
    # the slot and judge ONE step later — the settle gate must reject even though
    # the pose bands all pass at that instant. The probe is then removed unsettled.
    torch.manual_seed(101)
    env.reset()
    step(30)
    gv = float(scene._green_v[0])
    out_w = quat_apply(scene.rack.data.root_quat_w,
                       torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    place_rack_local(c.face_x + 0.030, gv, c.hang_z0 + 0.001,
                     vel_w=tuple(float(x) * 0.6 for x in out_w))
    env.step(no_action)
    report("moving-probe")
    spd = float(scene.bottle.data.root_lin_vel_w[0].norm())
    s, ok = judge()
    check("transient motion: bottle in the success pose but MOVING (0.6 m/s along "
          "the slot) — the settle gate rejects at the judged instant",
          spd > c.settle_speed and not ok)
    place_floor(0.30, 0.25)  # remove the probe before it can settle anywhere scored
    step(60)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.lip_hang_rack")
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
