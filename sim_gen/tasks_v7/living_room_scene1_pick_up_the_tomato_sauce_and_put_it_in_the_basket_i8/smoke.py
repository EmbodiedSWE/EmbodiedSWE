"""Smoke / rubric-REJECTION battery for SlideLidHamperScene (sim_gen task
`living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i8`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — force-slide open, gravity insertion, force-slide
shut — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
settled state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: lid fully shut in its channel,
                            cans upright on the ground, score ~0 at rest;
  3-4. randomization      — READBACK over 6 seeded resets: hamper yaw + xy, both can
                            positions (including band swaps) all move; cans never
                            spawn within 9 cm of each other;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  seed strategy       — the seed's plan (release the can above the container):
                            dropped over the CLOSED hamper the can settles ON the lid,
                            never inside (the aperture is physically blocked -> the
                            open->insert order is forced by geometry), score ~0;
  7.  wrong object        — BEIGE distractor inside the shut hamper, red can on the
                            ground -> no success, score ~0;
  8.  near-miss ajar      — red can inside but the lid settled 30 mm from shut
                            (tol 12 mm) -> NOT success, score < 0.9;
  9.  wrong place         — red can settled on the DECK (on the hamper, correct
                            height, outside the cavity) -> NOT success, score ~0;
  10. incomplete          — red can inside but the lid left fully OPEN -> NOT success,
                            score <= 0.85 (the re-close stage is load-bearing);
  11. monotonicity        — a half-open lid latches strictly less open credit than a
                            fully-open lid;
  12. latched credit + close gating — sliding the lid back shut with NO can inside
                            leaves the latched score unchanged (open credit does not
                            evaporate; closing an empty hamper earns nothing);
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i8.smoke --headless
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slidelid_hamper")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.25, 0.95)) + o),
                                tuple(np.array((0.35, 0.00, 0.10)) + o),
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

    def hamper_pose() -> tuple[torch.Tensor, float]:
        hp = (scene.hamper.data.root_pos_w - scene.env_origins)[0]
        q = scene.hamper.data.root_quat_w[0]
        return hp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        x_l = float(scene._lid_travel()[0])
        tl = scene._local(scene.tomato)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | lid_x={x_l:+.3f} tomato_local=({float(tl[0]):+.3f},"
              f"{float(tl[1]):+.3f},{float(tl[2]):.3f}) open_latch={float(scene._open_max[0]):.3f} "
              f"in={bool(scene._in[0])} close_latch={float(scene._close_max[0]):.3f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def to_world(x_l: float, y_l: float) -> tuple[float, float, float]:
        hp, hyaw = hamper_pose()
        wx = float(hp[0]) + math.cos(hyaw) * x_l - math.sin(hyaw) * y_l
        wy = float(hp[1]) + math.sin(hyaw) * x_l + math.cos(hyaw) * y_l
        return wx, wy, hyaw

    def place_obj(obj, x_l: float, y_l: float, z: float, yaw_local: bool = True,
                  settle_steps: int = 60) -> None:
        """Kinematic probe placement in HAMPER-LOCAL coordinates (instrumentation, not
        a solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy, hyaw = to_world(x_l, y_l)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        if yaw_local:
            st[:, 3], st[:, 6] = math.cos(hyaw / 2), math.sin(hyaw / 2)
        else:
            st[:, 3] = 1.0
        st[:, 0:3] += env.iscene.env_origins
        obj.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def place_lid(x_l: float, settle_steps: int = 60) -> None:
        place_obj(scene.lid, x_l, 0.0, c.lid_z0, settle_steps=settle_steps)

    def place_can_inside(can, settle_steps: int = 60) -> None:
        place_obj(can, 0.0, 0.0, c.floor_t + c.can_h / 2 + 0.002, yaw_local=False,
                  settle_steps=settle_steps)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.lid.data.root_state_w).all()
            and torch.isfinite(scene.tomato.data.root_state_w).all()
            and torch.isfinite(scene.distractor.data.root_state_w).all())
    x_l0 = float(scene._lid_travel()[0])
    lid_z = float(scene._local(scene.lid)[0, 2])
    tz = float((scene.tomato.data.root_pos_w - scene.env_origins)[0, 2])
    still = (float(scene.lid.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.tomato.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, lid fully shut in its channel, red can upright on the "
          "ground, everything at rest",
          bool(fin0) and abs(x_l0) < 0.006 and abs(lid_z - c.lid_z0) < 0.006
          and abs(tz - c.can_h / 2) < 0.01 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        hp, hyaw = hamper_pose()
        tp = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
        dp = (scene.distractor.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(hp[0]), float(hp[1]), hyaw, float(tp[0]), float(tp[1]),
                      float(dp[0]), float(dp[1]),
                      float((tp[:2] - dp[:2]).norm())))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (hamper_x, hamper_y, hamper_yaw, tomato_x, "
          f"tomato_y, distractor_x, distractor_y, can_sep):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: hamper yaw + xy vary across seeded resets (readback)",
          spread[2] > 0.04 and spread[0] > 0.008 and spread[1] > 0.008)
    check("randomization: both can positions vary (incl. band swaps) and the cans never "
          "spawn within 9 cm of each other (readback)",
          spread[3] > 0.015 and spread[4] > 0.05 and spread[6] > 0.05
          and float(arr[:, 7].min()) >= 0.09)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy: drop onto the CLOSED hamper ==============
    # The seed's whole plan is "carry the can over the container and release". Executed
    # against the shut hamper, the can lands ON the lid: the aperture is physically
    # blocked, so the open->insert order is forced by geometry, not just unscored.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place_obj(scene.tomato, -0.05, 0.0, c.z_top + c.lid_t + 0.05 + c.can_h / 2,
              yaw_local=False, settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    tl = scene._local(scene.tomato)[0]
    check("seed strategy: can released above the SHUT hamper settles ON the lid — never "
          "inside (aperture physically blocked), no success, score <= 0.05",
          float(tl[2]) > c.z_top and not bool(scene._in[0]) and not ok and s <= 0.05)

    # =========================== 7. wrong object: distractor inside =========================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place_can_inside(scene.distractor, settle_steps=90)
    report("wrong-object")
    s, ok = judge()
    dl = scene._local(scene.distractor)[0]
    check("wrong object: BEIGE distractor enclosed in the shut hamper, red can outside "
          "— no success, score <= 0.05",
          float(dl[2]) < c.inside_z_max and not ok and s <= 0.05)

    # =========================== 8. near-miss: lid ajar 30 mm ===============================
    torch.manual_seed(61)
    env.reset()
    step(10)
    place_lid(0.055, settle_steps=30)     # make room so the constructed insert is clean
    place_can_inside(scene.tomato, settle_steps=60)
    place_lid(0.030, settle_steps=60)     # ajar: just outside the 12 mm closed tolerance
    report("ajar")
    s_ajar, ok = judge()
    x_l = float(scene._lid_travel()[0])
    check("near-miss: red can inside but lid settled ~30 mm from shut (tol 12 mm) — NOT "
          "success, score < 0.9",
          bool(scene._in[0]) and x_l > c.closed_tol + 0.005 and not ok and s_ajar < 0.9)

    # =========================== 9. wrong place: on the DECK ================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place_obj(scene.tomato, 0.19, 0.0, c.z_top + c.can_h / 2 + 0.002, yaw_local=False,
              settle_steps=90)
    report("wrong-place")
    s, ok = judge()
    tl = scene._local(scene.tomato)[0]
    check("wrong place: red can settled on the hamper's DECK (on the fixture, outside "
          "the cavity) — not inside, no success, score <= 0.05",
          not bool(scene._in[0]) and float(tl[2]) > c.inside_z_max and not ok and s <= 0.05)

    # =========================== 10. incomplete: can inside, lid left OPEN ==================
    torch.manual_seed(81)
    env.reset()
    step(10)
    place_lid(c.open_ref, settle_steps=30)
    place_can_inside(scene.tomato, settle_steps=90)
    report("left-open")
    s_open, ok = judge()
    check("incomplete: red can inside but lid left fully OPEN — the re-close stage is "
          "load-bearing: NOT success, score <= 0.85",
          bool(scene._in[0]) and not ok and s_open <= 0.85)

    # =========================== 11. monotonicity of open credit ============================
    torch.manual_seed(91)
    env.reset()
    step(10)
    place_lid(0.06, settle_steps=40)      # half-open
    report("half-open")
    s_half, _ok = judge()
    op_half = float(scene._open_max[0])
    place_lid(c.open_ref, settle_steps=40)  # fully open
    report("fully-open")
    s_full, _ok = judge()
    op_full = float(scene._open_max[0])
    check("monotonicity: fully-open probe latched strictly more open credit than the "
          f"half-open probe ({op_half:.3f} < {op_full:.3f}) and a higher score",
          op_half + 0.05 < op_full and s_half < s_full)

    # =========================== 12. latched credit + close gating ==========================
    place_lid(0.0, settle_steps=40)       # back shut, hamper still EMPTY
    report("reshut-empty")
    s_back, ok = judge()
    check("latched credit + gating: sliding the lid back shut with NO can inside leaves "
          "the latched score unchanged (open credit kept, close credit gated on "
          "insertion), still no success", abs(s_back - s_full) < 1e-3 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.lid.data.root_state_w).all()
           and torch.isfinite(scene.hamper.data.root_state_w).all()
           and torch.isfinite(scene.tomato.data.root_state_w).all()
           and torch.isfinite(scene.distractor.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.slidelid_hamper")
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
