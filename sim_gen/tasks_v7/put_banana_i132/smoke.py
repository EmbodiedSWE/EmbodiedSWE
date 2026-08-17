"""Smoke / rubric-REJECTION battery for HutchFeedScene (sim_gen task `put_banana_i132`) —
NullRobot, teleported/forced probe states, RECORDED.

This is NOT a solution (solve.py — kinematic door lift/lower + force-pushed doorway
passage — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
probe here CONSTRUCTS a wrong (or partial) outcome — as a settled state or as an honest
force attempt — and asserts the rubric REJECTS it; no probe reaches success(), and a
final audit asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: door seated CLOSED in its
                            channel, block outside on the doorway side, all still;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 8 seeded resets: hutch xy + doorway-heading
                            yaw really move, block spawn distance + lateral offset
                            really vary (hutch-local readback); the door spawns seated
                            CLOSED in every one of them (it follows the hutch);
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's aerial-drop plan: the block released from above
                            the hutch centre lands ON THE ROOF and stays there (the
                            sealed top is real) — not inside, score ~0, no success;
  7.  out-of-order push   — the block force-pushed against the CLOSED door with the
                            solve's own bounded servo: it demonstrably reaches the door
                            (movement asserted — no vacuous probe) and is stopped;
                            never inside, opened never latches, score ~0, no success;
  8.  near-miss           — door parked away (open), block settled just OUTSIDE the
                            doorway mouth: NOT success, score < 0.9;
  9.  straddle            — door open, block settled IN the doorway (centre on the wall
                            plane): the inside margin rejects it — NOT success;
  10. open hutch          — block fully INSIDE but the door left parked away: the
                            closed-door clause rejects — NOT success, score < 0.9;
  11. latched credit      — dragging the block back OUT of the hutch afterwards leaves
                            the latched score unchanged (credit does not evaporate),
                            still no success;
  12. empty-hutch close   — door lifted out and re-seated with the block still outside:
                            door_closed_now holds but the close-credit term stays
                            unlatched — NOT success, score < 0.5;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.put_banana_i132.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hutch_feed")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.85)) + o),
                                tuple(np.array((0.50, 0.00, 0.05)) + o),
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
        b = (scene.block.data.root_pos_w - scene.env_origins)[0]
        d = (scene.door.data.root_pos_w - scene.env_origins)[0]
        bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | block=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) block_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) door=({float(d[0]):+.3f},{float(d[1]):+.3f},"
              f"{float(d[2]):.3f}) blocking={bool(scene.door_blocking()[0])} "
              f"closed={bool(scene.door_closed_now()[0])} "
              f"inside={bool(scene.block_inside_now()[0])} "
              f"opened={bool(scene._opened[0])} app={float(scene._app_max[0]):.3f} "
              f"in={bool(scene._inside[0])} closed_after={bool(scene._closed_after[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def local_to_world(loc) -> torch.Tensor:
        p = torch.tensor(loc, device=device).expand(n, 3)
        return scene.hutch.data.root_pos_w + quat_apply(scene.hutch.data.root_quat_w, p)

    def place_block_local(lx: float, ly: float, z: float, align: bool = True) -> None:
        """Teleport the block to a hutch-local xy at world height z (probe constructor)."""
        p = local_to_world((lx, ly, 0.0))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p[:, :2]
        st[:, 2] = z
        st[:, 3:7] = scene.hutch.data.root_quat_w if align \
            else torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        scene.block.write_root_state_to_sim(st, all_ids)

    def park_door_away() -> None:
        """Teleport the door out of the channel, lying flat clear of the doorway
        (constructs the OPEN-hutch relation as a settled state)."""
        q_y90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                             device=device).expand(n, 4)
        p = local_to_world((0.30, 0.30, 0.0))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p[:, :2]
        st[:, 2] = c.door_t / 2 + 0.02
        st[:, 3:7] = quat_mul(scene.hutch.data.root_quat_w, q_y90)
        scene.door.write_root_state_to_sim(st, all_ids)

    def seat_door() -> None:
        """Teleport the door back to its channel seat pose (probe constructor)."""
        p = local_to_world((c.door_seat_lx, 0.0, 0.0))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = p[:, :2]
        st[:, 2] = c.door_seat_lz + 0.002
        st[:, 3:7] = scene.hutch.data.root_quat_w
        scene.door.write_root_state_to_sim(st, all_ids)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.hutch.data.root_state_w).all()
            and torch.isfinite(scene.door.data.root_state_w).all()
            and torch.isfinite(scene.block.data.root_state_w).all())
    still = (float(scene.block.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.door.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
    check("settle: states finite, door seated CLOSED in its channel, block outside on "
          "the doorway side, everything still",
          bool(fin0) and bool(scene.door_closed_now()[0])
          and not bool(scene.block_inside_now()[0]) and float(bl[0]) > 0.25 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    door_ok = True
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(30)
        h = (scene.hutch.data.root_pos_w - scene.env_origins)[0]
        hq = scene.hutch.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(hq[3]), float(hq[0]))
        dev_deg = math.degrees(math.atan2(math.sin(yaw - math.pi),
                                          math.cos(yaw - math.pi)))
        bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
        door_ok = door_ok and bool(scene.door_closed_now()[0])
        reads.append((float(h[0]), float(h[1]), dev_deg, float(bl[0]), float(bl[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (hutch_x, hutch_y, yaw_dev_deg, "
          f"block_loc_x, block_loc_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: hutch xy jitter AND doorway-heading yaw really move across "
          "seeded resets (readback)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 15.0)
    check("randomization: block spawn distance + lateral offset really vary "
          "(hutch-local readback) AND the door spawns seated CLOSED in every reset",
          spread[3] > 0.02 and spread[4] > 0.03 and door_ok)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: aerial drop ==============================
    # The seed task's whole plan — carry the object over the container and release it —
    # executed literally: the block dropped from above the hutch centre. The sealed roof
    # catches it: it settles ON the roof, outside the interior, worth ~nothing.
    torch.manual_seed(41)
    env.reset()
    step(10)
    drop_z = c.wall_h + c.roof_t + c.block_s / 2 + 0.05
    place_block_local(0.0, 0.0, drop_z)
    step(120)
    report("roof-drop")
    s, ok = judge()
    bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
    on_roof = (abs(float(bl[0])) < c.hx + c.wall_t and abs(float(bl[1])) < c.hy + c.wall_t
               and float(bl[2]) > c.wall_h)
    check("seed strategy: block dropped from above the hutch centre lands ON THE ROOF "
          "and stays there — not inside, score ~0 (<= 0.02), no success",
          on_roof and not bool(scene.block_inside_now()[0]) and s <= 0.02 and not ok)

    # =========================== 7. out-of-order: push at the CLOSED door ===================
    # The solve's own bounded push servo (same caps) aimed at the hutch centre with the
    # door still seated: the block must reach the door (asserted — the probe is not
    # vacuous) and be STOPPED there; opened never latches, nothing is earned.
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_block_local(0.26, 0.0, c.block_rest_z + 0.002)
    step(30)
    q_ref = scene.block.data.root_quat_w.clone()
    mode = 0

    def encode(f_world: torch.Tensor) -> torch.Tensor:
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_inv(scene.block.data.root_quat_w)),
                          f_world)

    target_xy = scene.hutch.data.root_pos_w[:, :2].clone()
    min_locx = float(scene._hutch_local(scene.block.data.root_pos_w)[0, 0])
    win_i, win_dist = 0, float((target_xy[0] - scene.block.data.root_pos_w[0, :2]).norm())
    for i in range(480):
        d_vec = target_xy[0] - scene.block.data.root_pos_w[0, :2]
        dist = float(d_vec.norm())
        u = d_vec / max(dist, 1e-6)
        v = scene.block.data.root_lin_vel_w[0, :2]
        f_xy = 2.0 * (u * 0.10 - v)
        f_along = float((f_xy * u).sum())
        if float(v.norm()) < 0.02 and f_along < 0.45:
            f_xy = f_xy + u * (0.45 - f_along)
        fn = float(f_xy.norm())
        if fn > 0.9:
            f_xy = f_xy * (0.9 / fn)
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :2] = f_xy
        scene.block.set_external_force_and_torque(
            encode(f_world).view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
        env.step(no_action)
        min_locx = min(min_locx, float(scene._hutch_local(scene.block.data.root_pos_w)[0, 0]))
        if i - win_i >= 60:  # force-frame probe: toggle if moving AWAY
            if dist > win_dist + 0.008:
                mode = 1 - mode
                print(f"[smoke] push probe: moving away; force-frame mode -> {mode}",
                      flush=True)
            win_i, win_dist = i, dist
    scene.block.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)
    report("closed-door-push")
    s, ok = judge()
    print(f"[smoke] closed-door push: min block_loc_x reached = {min_locx:.3f} "
          f"(door outer face ~{c.door_seat_lx + c.door_t / 2:.3f})", flush=True)
    check("out-of-order: the push demonstrably reached the closed door and was STOPPED "
          "there (movement asserted; never entered the interior), opened never latched, "
          "score ~0 (<= 0.02), no success",
          0.13 < min_locx < 0.20 and not bool(scene.block_inside_now()[0])
          and not bool(scene._opened[0]) and bool(scene.door_blocking()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. near-miss: just outside the mouth =======================
    torch.manual_seed(61)
    env.reset()
    step(10)
    park_door_away()
    step(60)  # opened latches
    place_block_local(0.17, 0.0, c.block_rest_z + 0.002)
    step(60)
    report("near-miss")
    s, ok = judge()
    check("near-miss: door open, block settled just OUTSIDE the doorway mouth — "
          "NOT success, score < 0.9",
          bool(scene._opened[0]) and not bool(scene.block_inside_now()[0])
          and not ok and s < 0.9)

    # =========================== 9. straddle: block IN the doorway ==========================
    # (same episode) block centred on the front wall plane, halfway through the aperture:
    # the inside margin (3.5 cm past the inner faces) rejects a block left in the door.
    place_block_local(c.hx, 0.0, c.block_rest_z + 0.002)
    step(60)
    report("straddle")
    s, ok = judge()
    bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
    check("straddle: block settled IN the doorway (centre on the wall plane) — the "
          "inside margin rejects it, NOT success",
          abs(float(bl[0]) - c.hx) < 0.03 and not bool(scene.block_inside_now()[0])
          and not ok and s < 0.9)

    # =========================== 10. open hutch: inside but door parked away ================
    torch.manual_seed(71)
    env.reset()
    step(10)
    park_door_away()
    step(60)
    place_block_local(0.0, 0.0, c.block_rest_z + 0.002)
    step(60)
    report("open-hutch")
    s10, ok = judge()
    check("open hutch: block fully INSIDE but the door left parked away — the "
          "closed-door clause rejects, NOT success, score < 0.9",
          bool(scene.block_inside_now()[0]) and not bool(scene.door_closed_now()[0])
          and not ok and s10 < 0.9)

    # =========================== 11. latched credit survives regression =====================
    # (same episode) drag the block back OUT, well BEYOND its latched best approach
    # (local x=0.48: farther from the mouth than any previously latched position, so no
    # new approach credit can accrue): the latched inside/approach credit must not
    # evaporate, and the regressed state is of course no success.
    place_block_local(0.48, 0.0, c.block_rest_z + 0.002)
    step(60)
    report("regressed")
    s11, ok = judge()
    check("latched credit: dragging the block back OUT leaves the latched score "
          f"unchanged ({s10:.3f} -> {s11:.3f}), still no success",
          abs(s11 - s10) < 1e-3 and not bool(scene.block_inside_now()[0]) and not ok)

    # =========================== 12. empty-hutch close ======================================
    torch.manual_seed(81)
    env.reset()
    step(10)
    park_door_away()
    step(60)  # opened latches
    seat_door()
    step(90)
    report("empty-close")
    s, ok = judge()
    bl = scene._hutch_local(scene.block.data.root_pos_w)[0]
    check("empty-hutch close: door lifted out and re-seated with the block still "
          "outside — door_closed_now holds but the close term stays unlatched: "
          "NOT success, score < 0.5",
          bool(scene.door_closed_now()[0]) and float(bl[0]) > 0.25
          and not bool(scene._closed_after[0]) and not ok and s < 0.5)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.hutch.data.root_state_w).all()
           and torch.isfinite(scene.door.data.root_state_w).all()
           and torch.isfinite(scene.block.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hutch_feed")
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
