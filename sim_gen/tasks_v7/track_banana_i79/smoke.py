"""Smoke / rubric-REJECTION battery for BananaLineScene (sim_gen task
`track_banana_i79`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — stage the crate, squeeze the clamp tails so the
mechanism releases the banana into it, deliver, all through contact — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
CONSTRUCTS a wrong (or partial) outcome as a settled state and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit
check asserts exactly that.

  1-2. settle/no-NaN      — all three bananas hang from their clamps, crate on the
                            floor, everything still and finite, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: gantry xy+yaw vary; the
                            GREEN banana's station takes >= 2 values; crate and
                            depot xy jitter is real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  captivity           — the SEED strategy (pull the pre-grasped banana along a
                            path) is impossible: 12 N yanks in 5 directions visibly
                            disturb the hanging banana (measured displacement — the
                            probe is not vacuous) yet it re-settles HANGING each
                            time (roof/stops/slot form closure, spring-held);
  7.  release control     — driving the tail levers (the squeeze analog) opens the
                            clamp > 10 deg and the banana falls; with no crate under
                            it, the floor landing latches an irreversible BRUISE;
                            the spring re-closes the clamp; score stays ~0;
  8.  empty delivery      — the crate pushed to the depot WITHOUT harvesting: no
                            delivery latch, no success, score ~0;
  9.  green harvested     — all three bananas in the delivered crate: the unripe
                            green must HANG, so no success; score capped 0.70;
  10. rim height          — a banana held at rim height over the crate centre is
                            NOT contained (z bound), while the same xy inside the
                            volume IS (positive control);
  11. depot near-miss     — loaded crate 10 cm off the depot centre: no delivery,
                            no success, score 0.50 (the two in-crate latches);
  12. latched credit      — removing both yellows from the crate does not evaporate
                            the latched 0.50;
  13. bruise is fatal     — a perfect final configuration reached AFTER a bruise:
                            every other conjunct holds, success still False;
  14. settle gate         — the perfect configuration moving at 0.45 m/s is refused
                            at that instant;
  15. rejection audit     — success() never True at ANY judged point;
  16. final no-NaN        — all task-object states finite; frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_banana_i79.smoke --headless
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
import traceback

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
    env = ENVS.get("simgen.banana_line")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.35, 1.05)) + o),
                                tuple(np.array((0.28, -0.08, 0.15)) + o),
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        b = [loc(bn) for bn in scene.bananas]
        pc = loc(scene.crate)
        ang = torch.rad2deg(scene.lever_angles()[0])
        inc = scene.in_crate()[0]
        hng = scene.hanging()[0]
        s, ok = judge()
        vmax = max(float(bn.data.root_lin_vel_w[0].norm()) for bn in scene.bananas)
        wmax = max(float(bn.data.root_ang_vel_w[0].norm()) for bn in scene.bananas)
        print(f"[smoke] {tag:14s} | crate=({float(pc[0]):+.3f},{float(pc[1]):+.3f}) "
              f"bz=({float(b[0][2]):.3f},{float(b[1][2]):.3f},{float(b[2][2]):.3f}) "
              f"|ang|max={float(ang.abs().max()):.1f} bv={vmax:.3f} bw={wmax:.2f} "
              f"in_crate={[bool(v) for v in inc]} hang={[bool(v) for v in hng]} "
              f"bruised={bool(scene.bruised()[0])} deliv={bool(scene._delivered[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
              vel=(0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def depot_xy() -> tuple[float, float]:
        return float(scene._depot_xy[0, 0]), float(scene._depot_xy[0, 1])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = torch.isfinite(scene.crate.data.root_state_w).all()
    for bn in scene.bananas:
        fin0 = fin0 and torch.isfinite(bn.data.root_state_w).all()
    hng = scene.hanging()[0]
    check("settle: states finite, all three bananas HANGING from their clamps, crate "
          "low on the floor, everything still",
          bool(fin0) and all(bool(v) for v in hng) and float(loc(scene.crate)[2]) < 0.02
          and bool(scene._still()[0]) and bool(scene.clamps_closed()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        gx, gy = float(scene._gantry_xy[0, 0]), float(scene._gantry_xy[0, 1])
        gyaw = math.degrees(float(scene._gantry_yaw[0]))
        green_st = int(scene._ban_station[0, 2])
        pcx, pcy = float(loc(scene.crate)[0]), float(loc(scene.crate)[1])
        dx, dy = depot_xy()
        reads.append((gx, gy, gyaw, green_st, pcx, pcy, dx, dy))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (gantry_x, gantry_y, gantry_yaw_deg, "
          f"green_station, crate_x, crate_y, depot_x, depot_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: gantry xy and yaw vary across seeded resets (readback: yaw "
          "spread > 8 deg within the declared band, xy spread > 1 cm)",
          spread[2] > 8.0 and (spread[0] > 0.01 or spread[1] > 0.01)
          and np.abs(arr[:, 2]).max() <= c.gantry_yaw_deg + 1.0)
    check("randomization: the GREEN banana's station takes >= 2 values across seeds "
          "and crate/depot xy jitter is real (> 2 cm spread)",
          len(set(int(v) for v in arr[:, 3])) >= 2
          and (spread[4] > 0.02 or spread[5] > 0.02)
          and (spread[6] > 0.02 or spread[7] > 0.02))

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. captivity: yanks cannot extract =========================
    # The seed's strategy — take hold of the banana and pull it along a path — is
    # constructed and fails: 12 N in every direction visibly disturbs the banana but
    # the closed clamp (roof, rear stop, converged slot, spring) never lets go.
    torch.manual_seed(41)
    env.reset()
    step(40)
    b0 = scene.bananas[0]
    p_ref = loc(b0).clone()
    max_disp = 0.0
    all_held = True
    for d_ in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)):
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0], f[:, 0, 1], f[:, 0, 2] = (c.yank_hold * v for v in d_)
        b0.set_external_force_and_torque(f, zero3, env_ids=all_ids, is_global=True)
        d_disp = 0.0
        a_max = 0.0
        lx = ly = 0.0
        for _ in range(35):
            env.step(no_action)
            a_max = max(a_max, float(torch.rad2deg(scene.lever_angles()[0]).abs().max()))
            dd = float((loc(b0) - p_ref).norm())
            if dd > d_disp:  # gantry-local xy at the extreme -> which exit it probed
                d_disp = dd
                gy = float(scene_mod._yaw_of(scene.gantry.data.root_quat_w)[0])
                dv = loc(b0) - loc(scene.gantry)
                lx = math.cos(gy) * float(dv[0]) + math.sin(gy) * float(dv[1])
                ly = -math.sin(gy) * float(dv[0]) + math.cos(gy) * float(dv[1])
        max_disp = max(max_disp, d_disp)
        b0.set_external_force_and_torque(zero3, zero3, env_ids=all_ids)
        step(60)
        held = bool(scene.hanging()[0, 0])
        print(f"[smoke] yank {d_}: disp {d_disp * 1000:.1f} mm  "
              f"lever|ang|max {a_max:.1f} deg  local=({lx:+.3f},{ly:+.3f})  "
              f"z={float(loc(b0)[2]):.3f}  hang={held}", flush=True)
        all_held = all_held and held
        if not held:
            break            # escaped: stop yanking a free banana across the floor
    report("post-yanks")
    s, ok = judge()
    print(f"[smoke] yank probe: max displacement {max_disp * 1000:.1f} mm", flush=True)
    check("captivity: 12 N yanks (+-x, +-y, up) measurably disturb the hanging banana "
          f"(max {max_disp * 1000:.1f} mm >= 2 mm — not vacuous) yet it re-settles "
          "HANGING every time; still score ~0, no success",
          max_disp >= 0.002 and all_held and s <= 0.02 and not ok)

    # =========================== 7. release control + floor bruise ==========================
    # Positive control for the mechanism (squeeze -> pads part -> banana falls) and
    # construction of the irreversible failure: nothing underneath -> BRUISE.
    torch.manual_seed(51)
    env.reset()
    step(30)
    st0 = int(scene._ban_station[0, 0])
    kl, kr = 2 * st0, 2 * st0 + 1
    fell = False
    max_open = 0.0
    for i in range(240):
        ramp = min(1.0, (i + 1) / 30.0)
        scene.lever_drive[:, kl] = +2.2 * ramp
        scene.lever_drive[:, kr] = -2.2 * ramp
        env.step(no_action)
        ang = torch.rad2deg(scene.lever_angles()[0])
        max_open = max(max_open, float(ang[kl]), float(-ang[kr]))
        if float(loc(scene.bananas[0])[2]) < 0.16:
            fell = True
            break
    scene.lever_drive[:, :] = 0.0
    step(120)
    report("floor-drop")
    s, ok = judge()
    ang = torch.rad2deg(scene.lever_angles()[0])
    check("release control: driving the tail levers opens the clamp "
          f"({max_open:.1f} deg >= 10) and the banana FALLS; on the open floor it "
          "latches a BRUISE; the spring re-closes the clamp; score stays ~0, no "
          "success",
          fell and max_open >= 10.0 and float(loc(scene.bananas[0])[2]) < 0.05
          and not bool(scene.hanging()[0, 0]) and bool(scene.bruised()[0])
          and float(ang.abs().max()) < c.closed_deg and s <= 0.02 and not ok)

    # =========================== 8. empty delivery is worthless =============================
    torch.manual_seed(61)
    env.reset()
    step(10)
    dx, dy = depot_xy()
    place(scene.crate, dx, dy, 0.001)
    step(80)
    report("empty-deliv")
    s, ok = judge()
    check("empty delivery: the crate at the depot WITHOUT bananas aboard — the "
          "delivery latch stays False, no success, score ~0",
          bool(scene.crate_in_depot()[0]) and not bool(scene._delivered[0])
          and s <= 0.02 and not ok)

    # =========================== 9. harvesting the green too ================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    dx, dy = depot_xy()
    place(scene.crate, dx, dy, 0.001)
    # side by side along the crate's SHORT axis: bananas are 129 mm tip-to-tip, so
    # x offsets would clip the walls (interior half-length 105 mm) and pop
    for j, off in ((0, -0.045), (1, 0.0), (2, 0.045)):
        place(scene.bananas[j], dx, dy + off, 0.032)
    step(100)
    report("green-too")
    s, ok = judge()
    inc = scene.in_crate()[0]
    check("green harvested: ALL THREE bananas settled in the delivered crate — the "
          "green must hang, so no success; latched credit capped at 0.70",
          bool(inc[0]) and bool(inc[1]) and bool(inc[2]) and not ok
          and 0.68 <= s <= 0.701)

    # =========================== 10. rim height is not contained ============================
    torch.manual_seed(81)
    env.reset()
    step(10)
    pcx, pcy = float(loc(scene.crate)[0]), float(loc(scene.crate)[1])
    rim_never = True
    for _ in range(25):
        place(scene.bananas[0], pcx, pcy, 0.100)
        env.step(no_action)
        rim_never = rim_never and not bool(scene.in_crate()[0, 0])
    # positive control: same xy INSIDE the volume reads contained
    place(scene.bananas[0], pcx, pcy, 0.032)
    env.step(no_action)
    inside_reads = bool(scene.in_crate()[0, 0])
    report("rim-height")
    check("rim height: a banana held over the crate centre at rim height (z bound) "
          "is NEVER contained, while the same xy inside the volume IS (positive "
          "control)", rim_never and inside_reads)

    # =========================== 11. depot near-miss ========================================
    torch.manual_seed(91)
    env.reset()
    step(10)
    dx, dy = depot_xy()
    place(scene.crate, dx + 0.10, dy, 0.001)
    place(scene.bananas[0], dx + 0.10, dy - 0.035, 0.032)
    place(scene.bananas[1], dx + 0.10, dy + 0.035, 0.032)
    step(80)
    report("near-miss")
    s11, ok = judge()
    check("depot near-miss: loaded crate 10 cm off the depot centre — no delivery "
          "latch, no success, score exactly the two in-crate latches (0.50)",
          not bool(scene.crate_in_depot()[0]) and not bool(scene._delivered[0])
          and not ok and 0.48 <= s11 <= 0.52)

    # =========================== 12. latched credit survives regression =====================
    place(scene.bananas[0], 0.60, -0.60, 0.030)
    place(scene.bananas[1], 0.66, -0.60, 0.030)
    step(60)
    report("regressed")
    s12, ok = judge()
    check("latched credit: removing both yellows from the crate does not evaporate "
          f"the latched credit ({s11:.3f} -> {s12:.3f}), never success",
          s12 >= s11 - 1e-3 and not ok)

    # =========================== 13. a bruise is fatal ======================================
    torch.manual_seed(101)
    env.reset()
    step(10)
    # first: bruise yellow 0 on the open floor
    place(scene.bananas[0], 0.62, -0.55, 0.030)
    step(40)
    assert bool(scene.bruised()[0]), "bruise construction failed"
    # then: build the otherwise-perfect final configuration
    dx, dy = depot_xy()
    place(scene.crate, dx, dy, 0.001)
    place(scene.bananas[0], dx, dy - 0.035, 0.032)
    place(scene.bananas[1], dx, dy + 0.035, 0.032)
    step(100)
    report("bruised-final")
    s, ok = judge()
    inc = scene.in_crate()[0]
    conj = {"y0_in": bool(inc[0]), "y1_in": bool(inc[1]),
            "green_out": not bool(inc[2]), "green_hangs": bool(scene.hanging()[0, 2]),
            "in_depot": bool(scene.crate_in_depot()[0]),
            "clamps": bool(scene.clamps_closed()[0]), "still": bool(scene._still()[0])}
    print(f"[smoke] bruised-final conjuncts: {conj}", flush=True)
    others = all(conj.values())
    check("bruise is fatal: the perfect final configuration reached AFTER a bruise — "
          "every other success conjunct verified True, success still False, score "
          "capped 0.70", others and not ok and s <= 0.701)

    # =========================== 14. settle gate ============================================
    torch.manual_seed(111)
    env.reset()
    step(10)
    dx, dy = depot_xy()
    place(scene.bananas[0], dx, dy - 0.035, 0.032)
    place(scene.bananas[1], dx, dy + 0.035, 0.032)
    place(scene.crate, dx, dy, 0.001, vel=(0.45, 0.0, 0.0))
    env.step(no_action)
    in_dep = bool(scene.crate_in_depot()[0])
    speed = float(scene.crate.data.root_lin_vel_w[0].norm())
    s, ok = judge()
    gate = in_dep and speed > 0.20 and not ok
    # deconstruct BEFORE it can settle into a genuine success
    place(scene.crate, 0.60, -0.75, 0.001)
    place(scene.bananas[0], 0.55, -0.60, 0.030)
    place(scene.bananas[1], 0.66, -0.60, 0.030)
    step(30)
    check("settle gate: the perfect configuration still sliding at "
          f"{speed:.2f} m/s inside the depot zone is refused at that instant", gate)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = torch.isfinite(scene.crate.data.root_state_w).all() \
        and torch.isfinite(scene.gantry.data.root_state_w).all()
    for bn in scene.bananas:
        fin = fin and torch.isfinite(bn.data.root_state_w).all()
    for lv in scene.levers:
        fin = fin and torch.isfinite(lv.data.root_state_w).all()
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.banana_line")
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
    except BaseException:  # noqa: BLE001 — Kit teardown hangs on exception; die loudly
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        threading.Timer(10.0, lambda: os._exit(2)).start()
        os._exit(2)
