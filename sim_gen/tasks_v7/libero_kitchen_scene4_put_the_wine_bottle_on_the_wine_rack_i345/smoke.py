"""Smoke / rubric-REJECTION battery for MissingRailRackScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i345`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — crossbar dropped into its brackets, bottle laid
across the two rails by a real drop — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS
a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN    — reset layout settles finite: bar on the floor, bottle
                          standing, on OPPOSITE sides, all still, score ~0;
  3-4. randomization    — READBACK over 8 seeded resets: the bar/bottle side sign
                          flips (always opposite each other), bar yaw spread is
                          real; rack yaw spread and xy jitter are real;
  5.  null policy       — 240 idle steps -> score ~0, no success;
  6.  seed strategy     — the end state the SEED's plan produces here (put the
                          bottle straight ON the rack): the bottle released at the
                          EXACT goal pose while the crossbar still lies on the
                          floor. Supported on ONE side only, it TIPS THROUGH the
                          missing-rail gap and falls -> NOT success, score <= 0.105
                          (the missing-rail order-forcer is physically real);
  7.  bar conjunct      — the crossbar dropped onto the tower tops BETWEEN the
                          rails (resting at seat height but NOT in its brackets)
                          and the bottle laid level bridging fixed rail + mis-laid
                          bar INSIDE the goal window: bottle_racked reads True yet
                          NOT success, score <= 0.205 (the repair is load-bearing
                          by conjunct, not just as scaffolding);
  8.  near-miss saddle  — crossbar GENUINELY drop-seated, bottle laid across both
                          rails at height but OFF the chock saddle (|y| ~ 95 mm)
                          -> NOT success, score <= 0.505;
  9.  latched credit    — teleporting the bottle from that rest back to the floor
                          leaves the latched score unchanged, still no success;
  10. under the cradle  — bottle standing on the FLOOR directly below the saddle
                          -> score ~0, no success (right (x,y), wrong relation);
  11. transient motion  — bar seated + bottle in the success pose but MOVING
                          (judged one step after a 0.5 m/s injection) -> the settle
                          gate rejects at that instant; probe removed unsettled;
  12. rejection audit   — success() was never True at ANY judged point;
  13. final no-NaN      — all task-object states finite at the end.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.missing_rail_rack")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.50, -1.15, 0.80)) + o),
                                tuple(np.array((0.50, 0.00, 0.15)) + o),
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
        bl = scene._bar_loc()[0]
        ol = scene._bottle_loc()[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | bar=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) seated={bool(scene.bar_seated()[0])} "
              f"bottle=({float(ol[0]):+.3f},{float(ol[1]):+.3f},{float(ol[2]):.3f}) "
              f"racked={bool(scene.bottle_racked()[0])} "
              f"latches=({bool(scene._bar_lift_ever[0])},{bool(scene._bar_seat_ever[0])},"
              f"{bool(scene._bottle_lift_ever[0])},{bool(scene._cradle_ever[0])}) "
              f"settled={bool(scene._settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    q_y90 = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # +z -> +x

    def place_bottle_rack_local(x: float, y: float, z: float, quat_local=None,
                                vel_w=None) -> None:
        """Teleport the bottle to a rack-local pose of the rack's CURRENT frame
        (probe constructor), optionally with a local orientation and world velocity."""
        r_pos = scene.rack.data.root_pos_w
        r_quat = scene.rack.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
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

    def place_bar_rack_local(x: float, z: float) -> None:
        """Teleport the bar to a rack-local pose, long axis along the rack y-axis."""
        r_pos = scene.rack.data.root_pos_w
        r_quat = scene.rack.data.root_quat_w
        loc = torch.tensor([x, 0.0, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = r_pos + quat_apply(r_quat, loc)
        st[:, 3:7] = r_quat
        scene.bar.write_root_state_to_sim(st, all_ids)

    def place_bottle_floor(x: float, y: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, c.stand_z + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.bottle.write_root_state_to_sim(st, all_ids)

    def seat_bar_genuinely() -> bool:
        """Drop the bar into its brackets (transport hover + gravity), like solve."""
        hover_z = c.tower_h + c.wall_h + c.bar_s / 2 + 0.006
        place_bar_rack_local(c.rail_x, hover_z)
        for _ in range(300):
            env.step(no_action)
            if bool(scene.bar_seated()[0]) and bool(scene._settled()[0]):
                return True
        return False

    def rack_yaw() -> float:
        q = scene.rack.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def bar_yaw() -> float:
        q = scene.bar.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        return bool(torch.isfinite(scene.rack.data.root_state_w).all()
                    and torch.isfinite(scene.bar.data.root_state_w).all()
                    and torch.isfinite(scene.bottle.data.root_state_w).all())

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    bz = float((scene.bar.data.root_pos_w - scene.env_origins)[0, 2])
    by = float((scene.bar.data.root_pos_w - scene.env_origins)[0, 1])
    oz = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 2])
    oy = float((scene.bottle.data.root_pos_w - scene.env_origins)[0, 1])
    check("settle: states finite, bar lying on the FLOOR, bottle STANDING on the "
          "floor, on OPPOSITE sides of the centreline, all still",
          finite_all() and abs(bz - c.bar_floor_z) < 0.010
          and abs(oz - c.stand_z) < 0.010 and by * oy < 0
          and bool(scene._settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        b = (scene.bar.data.root_pos_w - scene.env_origins)[0]
        o = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        r = (scene.rack.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(b[1]), float(o[1]), bar_yaw(),
                      float(r[0]), float(r[1]), rack_yaw()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bar_y, bottle_y, bar_yaw_deg, rack_x, "
          f"rack_y, rack_yaw_deg):\n{arr}", flush=True)
    side_flags = (arr[:, 0] > 0).astype(float)
    opposite = all(r[0] * r[1] < 0 for r in reads)
    yaw_wrapped = np.mod(arr[:, 2] + 90.0, 180.0) - 90.0  # bar yaw, mod pi symmetry
    bar_yaw_spread = float(yaw_wrapped.max() - yaw_wrapped.min())
    check("randomization A: the bar/bottle side sign flips across seeded resets "
          "(bar and bottle ALWAYS on opposite sides, readback) and the bar spawn "
          "yaw spread is real (> 5 deg)",
          0.0 < side_flags.mean() < 1.0 and opposite and bar_yaw_spread > 5.0)
    ryaw = arr[:, 5]
    ryaw_wrapped = np.mod(ryaw - ryaw.mean() + 180.0, 360.0) - 180.0
    yaw_spread = float(ryaw_wrapped.max() - ryaw_wrapped.min())
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

    # =========================== 6. seed strategy: bottle straight ON the rack ==============
    # The SEED's plan transplanted verbatim — put the bottle on the rack, skipping
    # the repair: released at the EXACT goal pose (horizontal, centred on the
    # saddle) while the crossbar still lies on the floor. Supported by the fixed
    # rail only, it TIPS THROUGH the missing-rail gap. Judged as-is on the settled
    # outcome: the order-forcer is physical, not conventional.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_bottle_rack_local(0.0, 0.0, c.rest_z + 0.004, quat_local=q_y90)
    step(300)
    report("seed-strategy")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("seed strategy: bottle released at the EXACT goal pose with the crossbar "
          "still on the floor tips through the missing-rail gap and falls — NOT "
          "success, score <= 0.105 (the repair-first order is physically forced)",
          float(loc[2]) < c.rest_z - c.z_tol - 0.01 and not ok and s <= 0.105)

    # =========================== 7. bar conjunct: bridging a MIS-LAID bar ===================
    # The bar dropped onto the open tower tops BETWEEN the two rail positions: it
    # rests at seat HEIGHT (tower top + half bar) but is NOT in its brackets. The
    # bottle then laid level bridging fixed rail + mis-laid bar, its origin INSIDE
    # the goal window: bottle_racked reads True — yet success must refuse, because
    # the repair is a conjunct of the goal, not scaffolding.
    torch.manual_seed(51)
    env.reset()
    step(30)
    place_bar_rack_local(-0.010, c.tower_h + c.bar_s / 2 + 0.030)
    step(180)
    bl7 = scene._bar_loc()[0]
    place_bottle_rack_local(-0.020, 0.0, c.rest_z + 0.004, quat_local=q_y90)
    step(240)
    report("mis-laid-bar")
    loc = scene._bottle_loc()[0]
    racked7 = bool(scene.bottle_racked()[0])
    seated7 = bool(scene.bar_seated()[0])
    s, ok = judge()
    check("bar conjunct: bar resting on the tower tops at seat HEIGHT but NOT in "
          "its brackets, bottle laid level bridging fixed rail + mis-laid bar "
          "INSIDE the goal window — bottle_racked True yet NOT success, "
          "score <= 0.205 (the seated-bar conjunct is load-bearing)",
          abs(float(bl7[2]) - c.seat_z) < 0.010 and not seated7 and racked7
          and abs(float(loc[2]) - c.rest_z) <= c.z_tol and not ok and s <= 0.205)

    # =========================== 8. near-miss: off the chock saddle =========================
    # Crossbar GENUINELY seated (real drop into the brackets), bottle laid across
    # both rails at height — but at |y| ~ 95 mm, outside the chock saddle.
    torch.manual_seed(61)
    env.reset()
    step(30)
    seated8 = seat_bar_genuinely()
    s_bar, ok_bar = judge()
    place_bottle_rack_local(0.0, 0.095, c.rest_z + 0.004, quat_local=q_y90)
    step(240)
    report("off-saddle")
    loc = scene._bottle_loc()[0]
    s8, ok = judge()
    check("near-miss saddle: crossbar genuinely drop-seated (alone NOT success, "
          "score 0.40) + bottle resting across BOTH rails at height but OFF the "
          "chock saddle (|y| ~ 95 mm) — NOT success, score <= 0.505",
          seated8 and not ok_bar and abs(s_bar - 0.40) < 0.005
          and abs(float(loc[2]) - c.rest_z) <= 0.015 and abs(float(loc[1])) > c.y_tol
          and not ok and s8 <= 0.505)

    # =========================== 9. latched credit survives regression ======================
    place_bottle_floor(0.30, -0.30)
    step(90)
    report("regressed")
    s9, ok = judge()
    loc = scene._bottle_loc()[0]
    check("latched credit: teleporting the bottle from that rest back to the floor "
          f"leaves the latched score unchanged ({s8:.3f} -> {s9:.3f}), still no "
          "success", abs(s9 - s8) < 1e-3 and float(loc[2]) < 0.12 and not ok)

    # =========================== 10. standing under the cradle ==============================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_bottle_rack_local(0.0, 0.0, c.stand_z + 0.002)
    step(180)
    report("under-cradle")
    loc = scene._bottle_loc()[0]
    s, ok = judge()
    check("under the cradle: bottle standing on the FLOOR directly below the "
          "saddle — score ~0 (<= 0.02), no success (right (x,y), wrong relation)",
          float(loc[2]) < 0.12 and not ok and s <= 0.02)

    # =========================== 11. transient motion (settle gate) =========================
    # Bar genuinely seated, bottle in the success pose but MOVING: inject the pose
    # with a 0.5 m/s velocity along the rails and judge ONE step later — the settle
    # gate must reject even though the pose bands all pass at that instant. The
    # probe is then removed unsettled (before it can stop inside the goal).
    torch.manual_seed(81)
    env.reset()
    step(30)
    seated11 = seat_bar_genuinely()
    lat_w = quat_apply(scene.rack.data.root_quat_w,
                       torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    place_bottle_rack_local(0.0, 0.0, c.rest_z + 0.001, quat_local=q_y90,
                            vel_w=tuple(float(x) * 0.5 for x in lat_w))
    env.step(no_action)
    report("moving-probe")
    spd = float(scene.bottle.data.root_lin_vel_w[0].norm())
    s, ok = judge()
    check("transient motion: bar seated + bottle in the success pose but MOVING "
          "(0.5 m/s along the rails) — the settle gate rejects at the judged "
          "instant", seated11 and spd > c.settle_speed and not ok)
    place_bottle_floor(0.30, 0.30)  # remove the probe before it can settle anywhere scored
    step(60)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.missing_rail_rack")
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
    except BaseException:  # noqa: BLE001 - die fast, don't idle to the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(4)
