"""Smoke / rubric-REJECTION battery for CrownSocketScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i246`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the force-driven plug extraction + gravity seating
— is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: plug seated in the socket,
                            bowl standing on the floor, everything still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the cabinet frame moves
                            (seated-plug world xy + rim-wall yaw from sim state) and
                            the bowl spawn side swaps and jitters;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (set the bowl down on top of the cabinet)
                            executed on the roof slope: the slick pitched sheet SHEDS
                            it over the eave onto the floor -> no success;
  7.  ridge perch         — bowl balanced on the ridge line beside the tower: sheds
                            down one slope and off -> the roof holds nothing;
  8.  occupied socket     — bowl dropped dead-centre onto the SEATED plug: it can
                            only stack on the knob / tumble off — there is no room in
                            the well, nothing seats -> no success, no credit;
  9.  partial extraction  — plug raised 2 cm (still inside the well) -> it still
                            OCCUPIES the socket: vacated stays False, no credit;
  10. vacated grading     — plug discarded to the floor -> score == w_vacated only,
                            no success (the bowl is still on the floor);
  11. re-occupied socket  — plug dropped BACK into the well -> the vacated credit is
                            honestly REVOKED (state-based, not latched): score ~0;
  12. wrong orientation   — plug out, bowl lying on its SIDE in the well (centre
                            inside the xy tol and the seated z band) -> the upright
                            clause rejects: no seated credit beyond vacated;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i246.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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

PLUG_DISCARD = (0.10, -0.45)  # empty floor: clear of the cabinet and both bowl slots


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crown_socket")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.60, -1.20, 1.10)) + o),
                                tuple(np.array((0.50, 0.00, 0.30)) + o),
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
        lp, lb = scene._local(scene.plug)[0], scene._local(scene.bowl)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | plug_loc=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):.3f}) bowl_loc=({float(lb[0]):+.3f},{float(lb[1]):+.3f},"
              f"{float(lb[2]):.3f}) occ={bool(scene.plug_occupies()[0])} "
              f"seated={bool(scene.bowl_seated()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Pose write in the CURRENT episode's cabinet frame (xy rotated by yaw)."""
        cosy = float(torch.cos(scene._cab_yaw[0]))
        siny = float(torch.sin(scene._cab_yaw[0]))
        wx = float(scene._cab_xy[0, 0]) + cosy * lx - siny * ly
        wy = float(scene._cab_xy[0, 1]) + siny * lx + cosy * ly
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_world(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def w_pos(body) -> tuple[float, float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1]), float(p[2])

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.plug.data.root_state_w).all()
            and torch.isfinite(scene.bowl.data.root_state_w).all()
            and all(torch.isfinite(b.data.root_state_w).all() for b in scene.parts.values()))
    lp = scene._local(scene.plug)[0]
    _, _, bz = w_pos(scene.bowl)
    still = (float(scene.plug.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and bool(scene.bowl_still()[0]))
    check("settle: states finite, plug seated in the socket (bottom on the well "
          "floor), bowl standing on the floor, everything still",
          bool(fin0) and bool(scene.plug_occupies()[0])
          and abs(float(lp[2]) - c.floor_top) < 0.01 and bz < 0.05 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(10)  # let the written poses take and the plug seat
        px_, py_, _ = w_pos(scene.plug)  # seated plug = the socket's world position
        q = scene.parts["wall_xp"].data.root_quat_w[0]  # rim wall carries the yaw
        yaw_deg = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
        bx_, by_, _ = w_pos(scene.bowl)
        reads.append((px_, py_, yaw_deg, bx_, by_, 1.0 if by_ > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (plug_x, plug_y, yaw_deg, bowl_x, bowl_y, "
          f"bowl_left):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: the cabinet frame really moves — seated-plug world xy "
          "spread > 1.5 cm and rim-wall yaw spread > 5 deg across seeded resets "
          "(sim-state readback)",
          (spread[0] > 0.015 or spread[1] > 0.015) and spread[2] > 5.0
          and abs(arr[:, 2]).max() <= c.cab_yaw_deg + 1.0)
    check("randomization: bowl spawn side swaps and xy jitter is real across seeded "
          "resets (readback)",
          0.0 < arr[:, 5].mean() < 1.0 and spread[3] > 0.015)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: set the bowl down on top ================
    # The seed's plan verbatim: carry the black bowl up and set it down on top of the
    # cabinet. Here the top is the slick 22-deg roof: the bowl slides over the eave
    # onto the floor.
    torch.manual_seed(41)
    env.reset()
    step(10)
    # bowl 2 cm above the +y roof slope, clear of the rim tower (local y = 0.13)
    roof_top = c.ridge_z - (c.ridge_z - c.roof_eave_z) / (c.carcass[1] / 2) * 0.13
    place_local(scene.bowl, 0.0, 0.13, roof_top + c.bowl_h / 2 + 0.02)
    step(360)
    report("seed-strategy")
    s, ok = judge()
    _, _, bz = w_pos(scene.bowl)
    lb = scene._local(scene.bowl)[0]
    off_cab = max(abs(float(lb[0])), abs(float(lb[1]))) > c.carcass[0] / 2 - 0.02
    check("seed strategy: bowl set down on top of the cabinet (roof slope) is SHED "
          "over the eave onto the floor — off the footprint or grounded, no success, "
          "score <= 0.02",
          bz < 0.15 and off_cab and not ok and s <= 0.02)

    # =========================== 7. the ridge is not a shelf either =========================
    torch.manual_seed(43)
    env.reset()
    step(10)
    place_local(scene.bowl, 0.14, 0.0, c.ridge_z + c.bowl_h / 2 + 0.02)  # astride the peak
    step(360)
    report("ridge-perch")
    s, ok = judge()
    _, _, bz = w_pos(scene.bowl)
    check("ridge perch: bowl balanced astride the ridge line beside the tower sheds "
          "down a slope — it does not rest on the roof: no success, score <= 0.02",
          bz < 0.15 and not ok and s <= 0.02)

    # =========================== 8. the occupied socket admits nothing ======================
    torch.manual_seed(51)
    env.reset()
    step(10)
    knob_top = c.floor_top + c.plug_body_h + c.plug_knob_h
    place_local(scene.bowl, 0.0, 0.0, knob_top + c.bowl_h / 2 + 0.02)
    step(300)
    report("stack-on-plug")
    s, ok = judge()
    check("occupied socket: bowl dropped dead-centre over the SEATED plug can only "
          "stack on the knob or tumble off — the plug still occupies, nothing seats: "
          "no success, score <= 0.02",
          bool(scene.plug_occupies()[0]) and not bool(scene.bowl_seated()[0])
          and not ok and s <= 0.02)

    # =========================== 9. partial extraction still occupies =======================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_local(scene.plug, 0.0, 0.0, c.floor_top + 0.02)  # raised 2 cm, inside the well
    step(2)  # judge the raised state before gravity re-seats it
    report("partial-pull")
    s, ok = judge()
    check("partial extraction: plug raised 2 cm but still inside the well STILL "
          "occupies the socket — vacated stays False, score <= 0.02",
          bool(scene.plug_occupies()[0]) and s <= 0.02 and not ok)
    step(60)  # let it drop back and re-seat

    # =========================== 10. vacated grading ========================================
    place_world(scene.plug, PLUG_DISCARD[0], PLUG_DISCARD[1], 0.01)
    step(60)
    report("vacated")
    s10, ok = judge()
    check("vacated grading: plug discarded to the floor earns exactly the vacated "
          "credit (score in [0.35, 0.45]), no success (bowl still on the floor)",
          bool(scene.vacated()[0]) and 0.35 <= s10 <= 0.45 and not ok)

    # =========================== 11. re-occupying revokes the credit ========================
    place_local(scene.plug, 0.0, 0.0, 0.52)  # hover above the empty well, drop back in
    step(240)
    report("re-seated")
    s11, ok = judge()
    check("re-occupied socket: plug dropped back into the well re-seats and the "
          "vacated credit is honestly REVOKED (state-based, not latched): score "
          "<= 0.02, no success",
          bool(scene.plug_occupies()[0]) and s11 <= 0.02 and not ok)

    # =========================== 12. wrong orientation in the well ==========================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_world(scene.plug, PLUG_DISCARD[0], PLUG_DISCARD[1], 0.01)
    step(30)
    r2 = math.sqrt(0.5)
    place_local(scene.bowl, 0.0, 0.0, c.floor_top + c.bowl_r + 0.003, quat=(r2, r2, 0.0, 0.0))
    step(240)
    report("side-lying")
    s12, ok = judge()
    lb = scene._local(scene.bowl)[0]
    in_band = (abs(float(lb[0])) <= c.seat_xy_tol and abs(float(lb[1])) <= c.seat_xy_tol
               and c.seat_z_lo <= float(lb[2]) <= c.seat_z_hi)
    q = scene.bowl.data.root_quat_w[0]
    r33 = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    check("wrong orientation: bowl lying on its SIDE in the vacated well — centre "
          "inside the xy tol and the seated z band but NOT upright — the upright "
          "clause rejects: no seated credit beyond vacated, no success",
          in_band and r33 < c.up_min and not bool(scene.bowl_seated()[0])
          and 0.35 <= s12 <= 0.45 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.plug.data.root_state_w).all()
           and torch.isfinite(scene.bowl.data.root_state_w).all()
           and all(torch.isfinite(b.data.root_state_w).all() for b in scene.parts.values()))
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.crown_socket")
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
