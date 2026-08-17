"""Smoke / rubric-REJECTION battery for PlateSlotSwapScene (sim_gen task
`put_plate_in_colored_dish_rack_i205`) — NullRobot, constructed probe states, RECORDED.

This is NOT a solution (solve.py — force-servo vertical extractions and insertions
with free-air transport teleports — is the acceptance evidence that the rubric ACCEPTS
a correct swap; it passes on forge seeds 0 and 1). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a judged state and
asserts the rubric REJECTS it — plus a capacity-reality probe proving the seed's
single-transport strategy (lower the plate into its colored slot) physically fails
while the slot is occupied. No probe in this battery ever reaches success(), and a
final audit check asserts exactly that.

   1. settle             — reset settles finite; readback: blue plate seated in the
                           YELLOW slot and yellow plate in the BLUE slot (the swapped
                           start), both ON EDGE, neither matched (the color/x-band
                           binding), everything settled;
   2. fresh score        — score ~0 at reset, no success;
   3. rack randomization — READBACK over 8 seeded resets: rack yaw varies widely
                           (both signs), live quat agrees with the cached sample,
                           xy jitter is real;
   4. cradle+plate rand. — cradle yaw/xy vary; the plates' in-slot x lean seed and
                           y slide jitter are real (rack-frame readback);
   5. null policy        — 240 idle steps -> score ~0, no success, still swapped;
   6. capacity drop      — the SEED's strategy: blue plate lowered straight onto its
                           OCCUPIED color slot falls, lands perched/deflected (z or
                           axis readback), NOT seated in the blue slot, NOT matched,
                           score stays ~0 — a pocket holds exactly one plate;
   7. flat fin-top perch — with the pocket EMPTIED, a plate laid FLAT across the fin
                           tops sits INSIDE the z window yet is rejected by the
                           on-edge axis test alone;
   8. cradle is a buffer — yellow plate seated in the transfer cradle -> exactly the
                           +0.20 buffer credit latches, NOT matched, NOT success;
   9. half done          — blue plate then seated in its freed BLUE slot -> match
                           credit latches, score capped at 0.55, NOT success (yellow
                           still parked in the cradle);
  10. latched credit     — removing the matched blue plate to the floor keeps the
                           latched 0.55 while live matched drops — score never
                           decreases;
  11. settle gate        — both plates written into their matched slots and judged
                           immediately: matched live but NOT settled -> NOT success;
                           the probe is dismantled before the stillness latch can
                           complete (audit safety);
  12. rejection audit    — success() was never True at ANY judged point;
  13. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i205.smoke \
    --headless
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

BLUE_SLOT_SIGN = scene_mod.BLUE_SLOT_SIGN

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.plate_slot_swap")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    blue_cx = BLUE_SLOT_SIGN * c.slot_dx
    yellow_cx = -BLUE_SLOT_SIGN * c.slot_dx

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.55, 0.55)) + o),
                                tuple(np.array((0.00, -0.08, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def loc(name: str, frame) -> torch.Tensor:
        return scene._local(scene.plates[name], frame)[0]

    def axis_z(name: str) -> float:
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(scene.plates[name].data.root_quat_w, ez)[0, 2].abs())

    def yaw_of(q: torch.Tensor) -> float:
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def report(tag: str) -> None:
        s, ok = judge()
        lb = loc("blue", scene.rack)
        ly = loc("yellow", scene.rack)
        print(f"[smoke] {tag:16s} | blue_rack=({float(lb[0]):+.3f},{float(lb[1]):+.3f},"
              f"{float(lb[2]):+.3f}) yellow_rack=({float(ly[0]):+.3f},"
              f"{float(ly[1]):+.3f},{float(ly[2]):+.3f}) | "
              f"mB={bool(scene.matched('blue')[0])} "
              f"mY={bool(scene.matched('yellow')[0])} "
              f"crB={bool(scene.seated_in_cradle('blue')[0])} "
              f"crY={bool(scene.seated_in_cradle('yellow')[0])} "
              f"stl={bool(scene.settled()[0])} | "
              f"latch c={float(scene.cradle_latch[0]):.0f}/m="
              f"{float(scene.match_latch[0]):.0f} | score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_plate(name: str, pos_env, quat, settle_steps: int = 60) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL
        physics steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = torch.as_tensor(pos_env, device=device, dtype=torch.float32) \
            + scene.env_origins[0]
        st[:, 3:7] = torch.as_tensor(quat, device=device, dtype=torch.float32)
        scene.plates[name].write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def frame_pose(frame, local_xyz) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_env, seat_quat) for a frame-local point, plate standing on edge."""
        q_f = frame.data.root_quat_w
        local = torch.tensor(local_xyz, device=device).expand(n, 3)
        pos = (frame.data.root_pos_w + quat_apply(q_f, local))[0] \
            - scene.env_origins[0]
        return pos, scene._seat_quat(q_f)[0]

    def hover_drop(name: str, frame, cx: float, settle_steps: int = 300,
                   z: float = 0.16) -> None:
        """Free-air hover over the pocket + pure GRAVITY drop (no force)."""
        pos, q = frame_pose(frame, (cx, 0.0, z))
        write_plate(name, pos, q, settle_steps=settle_steps)

    def park_on_floor(name: str, xy, settle_steps: int = 60) -> None:
        """Plate laid FLAT on the open floor, far from rack and cradle."""
        write_plate(name, (xy[0], xy[1], c.plate_t / 2 + 0.004),
                    (1.0, 0.0, 0.0, 0.0), settle_steps=settle_steps)

    # =========================== 1-2. settle / fresh score ==================================
    env.reset(seed=11)
    step(90)
    report("reset")
    bodies = (scene.rack, scene.cradle, scene.plate_blue, scene.plate_yellow)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    swapped = bool(scene.seated_in_slot("blue", "yellow")[0]) \
        and bool(scene.seated_in_slot("yellow", "blue")[0])
    check("settle: all states finite, plates seated ON EDGE in each other's color "
          f"slots (swapped start; axis_z blue={axis_z('blue'):.2f} "
          f"yellow={axis_z('yellow'):.2f}), NEITHER matched (the color binding), "
          "everything settled",
          fin and swapped and not bool(scene.matched("blue")[0])
          and not bool(scene.matched("yellow")[0])
          and axis_z("blue") <= c.axis_z_max and axis_z("yellow") <= c.axis_z_max
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh: score ~0 at reset, no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    rack_reads, cradle_reads, plate_reads = [], [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        yaw_live = yaw_of(scene.rack.data.root_quat_w[0])
        yaw_cache = float(scene.rack_yaw0[0])
        d = math.degrees(math.atan2(math.sin(yaw_live - yaw_cache),
                                    math.cos(yaw_live - yaw_cache)))
        rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
        rack_reads.append((math.degrees(yaw_live), d, float(rp[0]), float(rp[1])))
        cq = yaw_of(scene.cradle.data.root_quat_w[0])
        cp = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        cradle_reads.append((math.degrees(cq), float(cp[0]), float(cp[1])))
        lb = loc("blue", scene.rack)
        ly = loc("yellow", scene.rack)
        plate_reads.append((float(lb[0]) - yellow_cx, float(lb[1]),
                            float(ly[0]) - blue_cx, float(ly[1])))
        print(f"[smoke] seed {sd}: rack yaw={math.degrees(yaw_live):+7.1f} "
              f"(cache delta {d:+.2f}) pos=({float(rp[0]):+.3f},{float(rp[1]):+.3f})"
              f" | cradle yaw={math.degrees(cq):+7.1f} "
              f"pos=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) | plate jit "
              f"bx={plate_reads[-1][0] * 1000:+.1f}mm by={plate_reads[-1][1] * 1000:+.1f}mm",
              flush=True)
    yaws = [r[0] for r in rack_reads]
    agree = max(abs(r[1]) for r in rack_reads)
    rxs = [r[2] for r in rack_reads]
    rys = [r[3] for r in rack_reads]
    check("randomization: rack yaw varies across 8 seeded resets (spread "
          f"{max(yaws) - min(yaws):.0f} deg, both signs: "
          f"{len({y > 0 for y in yaws}) == 2}), live quat agrees with the cached "
          f"sample (max delta {agree:.2f} deg), xy jitter real (x spread "
          f"{(max(rxs) - min(rxs)) * 1000:.0f} mm, y spread "
          f"{(max(rys) - min(rys)) * 1000:.0f} mm)",
          (max(yaws) - min(yaws)) > 90.0 and len({y > 0 for y in yaws}) == 2
          and agree < 1.0 and (max(rxs) - min(rxs)) > 0.008
          and (max(rys) - min(rys)) > 0.008)
    cyaws = [r[0] for r in cradle_reads]
    cxs = [r[1] for r in cradle_reads]
    cys = [r[2] for r in cradle_reads]
    bxs = [r[0] for r in plate_reads]
    bys = [r[1] for r in plate_reads]
    yxs = [r[2] for r in plate_reads]
    yys = [r[3] for r in plate_reads]
    check("randomization: cradle yaw/xy vary (yaw spread "
          f"{max(cyaws) - min(cyaws):.0f} deg, x {(max(cxs) - min(cxs)) * 1000:.0f} mm, "
          f"y {(max(cys) - min(cys)) * 1000:.0f} mm); plate in-slot jitter real "
          f"(blue x {(max(bxs) - min(bxs)) * 1000:.1f} mm / y "
          f"{(max(bys) - min(bys)) * 1000:.1f} mm, yellow x "
          f"{(max(yxs) - min(yxs)) * 1000:.1f} mm / y "
          f"{(max(yys) - min(yys)) * 1000:.1f} mm)",
          (max(cyaws) - min(cyaws)) > 90.0 and (max(cxs) - min(cxs)) > 0.010
          and (max(cys) - min(cys)) > 0.005 and (max(bys) - min(bys)) > 0.004
          and (max(yys) - min(yys)) > 0.004 and (max(bxs) - min(bxs)) > 0.0015
          and (max(yxs) - min(yxs)) > 0.0015)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps — the episode "
          "STARTS swapped (both slots wrong, cradle empty)",
          s <= 0.02 and not ok and bool(scene.seated_in_slot("blue", "yellow")[0])
          and bool(scene.seated_in_slot("yellow", "blue")[0]))

    # =========================== 6. the seed's strategy: capacity drop ======================
    env.reset(seed=41)
    step(30)
    # The seed-analog shortcut: carry the blue plate straight to its color slot and
    # lower it in — while the yellow plate still OCCUPIES that slot. The hover is
    # ABOVE the occupant's top rim (z=0.24: rim-to-rim contact is at 0.225), so the
    # drop is a real free fall onto the occupied pocket, not a depenetration blast.
    hover_drop("blue", scene.rack, blue_cx, settle_steps=360, z=0.24)
    report("capacity-drop")
    s, ok = judge()
    lb = loc("blue", scene.rack)
    fell = float(lb[2]) < 0.20  # left the hover AND toppled off the rim balance
    seated_blue = bool(scene.seated_in_slot("blue", "blue")[0])
    check("capacity: the seed's single-transport strategy fails — blue plate dropped "
          f"onto its OCCUPIED color slot toppled off (z_loc={float(lb[2]):+.3f}, "
          f"axis_z={axis_z('blue'):.2f}), ended NOT seated in the blue slot, NOT "
          f"matched, score<=0.02 (got {s:.2f}); the occupant is still seated: "
          f"{bool(scene.seated_in_slot('yellow', 'blue')[0])}",
          fell and not seated_blue and not bool(scene.matched("blue")[0])
          and not ok and s <= 0.02)

    # =========================== 7. flat fin-top perch (axis test) ==========================
    env.reset(seed=51)
    step(30)
    # Empty the blue pocket (park its occupant, the yellow plate, on the floor), then
    # lay the blue plate FLAT across the blue pocket's fin tops.
    park_on_floor("yellow", (0.65, 0.65))
    pos, _q = frame_pose(scene.rack, (blue_cx, 0.0, c.fin_h + c.plate_t / 2 + 0.004))
    q_rack = scene.rack.data.root_quat_w[0]
    write_plate("blue", pos, q_rack, settle_steps=240)  # flat: plate axis vertical
    report("flat-perch")
    s, ok = judge()
    lb = loc("blue", scene.rack)
    z_in_window = c.z_lo <= float(lb[2]) <= c.z_hi
    check("flat fin-top perch: plate laid FLAT across the emptied pocket's fin tops "
          f"sits INSIDE the z window (z_loc={float(lb[2]):+.3f} in "
          f"[{c.z_lo:.3f},{c.z_hi:.3f}]) yet the on-edge axis test alone rejects it "
          f"(axis_z={axis_z('blue'):.2f} > {c.axis_z_max:.2f}) -> NOT seated, NOT "
          "success",
          z_in_window and axis_z("blue") > 0.9
          and not bool(scene.seated_in_slot("blue", "blue")[0]) and not ok
          and s <= 0.02)

    # ============ 8-11. buffer credit, half done, latched credit, settle gate ===============
    env.reset(seed=61)
    step(30)
    # Yellow plate: blue slot -> transfer cradle (the unavoidable buffer move).
    hover_drop("yellow", scene.cradle, 0.0, settle_steps=240)
    for _ in range(12):
        if float(scene.cradle_latch[0]) > 0.5:
            break
        step(15)
    report("buffer-parked")
    s_buf, ok = judge()
    check("cradle is a buffer, not a goal: yellow plate seated in the transfer "
          f"cradle latches EXACTLY the +0.20 buffer credit (got {s_buf:.3f}), is "
          "NOT matched, NOT success",
          bool(scene.seated_in_cradle("yellow")[0]) and 0.18 <= s_buf <= 0.21
          and not bool(scene.matched("yellow")[0]) and not ok)
    # Blue plate: yellow slot -> its freed BLUE slot (matched).
    hover_drop("blue", scene.rack, blue_cx, settle_steps=240)
    for _ in range(12):
        if float(scene.match_latch[0]) > 0.5:
            break
        step(15)
    report("half-done")
    s_half, ok = judge()
    check("half done: blue plate seated in its freed BLUE slot latches the match "
          f"credit — score capped at the 0.55 partial ceiling (got {s_half:.3f}) "
          "and NOT success (the yellow plate is still parked in the cradle)",
          bool(scene.matched("blue")[0]) and 0.53 <= s_half <= 0.5601 and not ok)
    # Latched credit: remove the matched plate — the score must NOT decrease.
    park_on_floor("blue", (0.65, -0.65))
    report("latch-remove")
    s_after, ok = judge()
    check("latched credit: removing the matched blue plate to the floor keeps the "
          f"latched score ({s_half:.3f} -> {s_after:.3f}) while live matched drops",
          abs(s_after - s_half) < 1e-3 and not bool(scene.matched("blue")[0])
          and not ok)
    # Settle gate: BOTH plates written into their matched slots, judged immediately —
    # matched live, but the stillness latch has not persisted -> NOT success.
    pos_b, q_b = frame_pose(scene.rack, (blue_cx, 0.0, c.plate_r + 0.004))
    write_plate("blue", pos_b, q_b, settle_steps=0)
    pos_y, q_y = frame_pose(scene.rack, (yellow_cx, 0.0, c.plate_r + 0.004))
    write_plate("yellow", pos_y, q_y, settle_steps=0)
    step(5)
    report("settle-gate")
    s, ok = judge()
    both_live = bool(scene.matched("blue")[0]) and bool(scene.matched("yellow")[0])
    check("settle gate: both plates matched LIVE immediately after placement, but "
          f"stillness has not persisted (settled={bool(scene.settled()[0])}) -> "
          "NOT success at the judged moment",
          both_live and not bool(scene.settled()[0]) and not ok)
    # Dismantle BEFORE the stillness latch can complete (audit safety).
    park_on_floor("yellow", (0.65, 0.65), settle_steps=30)

    # =========================== 12-13. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.plate_slot_swap")
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
