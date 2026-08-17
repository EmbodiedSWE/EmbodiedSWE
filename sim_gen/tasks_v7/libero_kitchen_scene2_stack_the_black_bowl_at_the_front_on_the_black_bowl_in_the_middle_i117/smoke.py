"""Smoke / rubric-REJECTION battery for WaitersPullScene (sim_gen task
libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i117)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the force-yank tablecloth pull — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and asserts the
rubric's verdict on it. No probe below constructs full success.

  1-2. settle/no-NaN     — the authored sandwich (slat across the rim, bowl riding it)
                           settles finite AND HOLDS: the bowl stays ~20 mm above the seated
                           z band (riding height), nothing seated/stowed, score 0;
  3.  determinism        — the same seed twice -> identical layout readback;
  4-6. randomization     — READBACK over 8 seeded resets: pull-axis yaw varies, pedestal
                           centre jitters, tray side takes both values and tray yaw varies;
  7.  null policy        — 240 idle steps -> score ~0, no success (the sandwich does not
                           self-seat);
  8.  seed strategy      — the SEED's end state built by PLACEMENT: slat set aside on the
                           bare counter, bowl stacked seated on the pedestal by hand ->
                           seated() geometrically TRUE, yet no draw credit, no stow credit,
                           success False, score pinned at 0.25;
  9.  seat twin/near-miss— a 12 mm off-centre seat counts; a settled 22 mm off-centre seat
                           (foot still physically inside the recess) does NOT (xy gate);
 10.  stow-without-seat  — slat stowed in the tray while the bowl was dumped on the counter
                           (the "draw failed, stow anyway" outcome) -> success False, score
                           pinned at 0.25;
 11.  stow twin          — slat dropped centred in the tray counts as stowed;
 12.  slat on the rails  — slat laid CROSSWISE over the tray, resting flat on both rails
                           (24 mm above the floor band) -> not stowed (z band + yaw gate);
 13.  slat half out      — bowl legitimately-seated pose + slat sticking out of the tray's
                           open end (60 mm out of the x band) -> not stowed, success False,
                           score pinned at 0.25 (the latched seat credit);
 14.  latched credit     — the 0.25 seat credit survives the bowl being knocked back off
                           the pedestal (latches never evaporate);
 15.  finite at the end.

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
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the annotator
# returns EMPTY frames. Disable the driver check.
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

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.waiters_pull")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -1.15, 0.90)) + o),
                                tuple(np.array((0.06, 0.0, 0.24)) + o),
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

    def settle(max_steps: int = 500, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:18s} | seated={bool(scene.seated()[0])} "
              f"stowed={bool(scene.stowed()[0])} clear={bool(scene.slat_clear()[0])} "
              f"home={bool(scene.pedestal_home()[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    def place(body, x: float, y: float, z: float, quat: tuple | None = None,
              settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def yaw_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def ped_xy() -> tuple:
        p = scene.pedestal.data.root_pos_w[0, :2] - origin[0, :2]
        return float(p[0]), float(p[1])

    def bowl_dz() -> float:
        return float(scene.bowl.data.root_pos_w[0, 2] - scene.pedestal.data.root_pos_w[0, 2])

    def park_slat_on_counter() -> None:
        place(scene.slat, -0.30, -0.20 * float(scene.tray_side[0]),
              z0 + c.slat_t / 2 + 0.002, yaw_quat(0.4), settle_steps=20)

    def seat_bowl(dx: float = 0.0, dy: float = 0.0) -> None:
        px, py = ped_xy()
        place(scene.bowl, px + dx, py + dy, z0 + c.seat_dz + 0.006, settle_steps=20)
        settle()

    def stow_slat(dx_l: float = 0.0, crosswise: bool = False, high: float = 0.0) -> None:
        tyaw = float(scene.tray_yaw[0])
        txy = scene.tray_xy[0]
        yaw = tyaw + (math.pi / 2 if crosswise else 0.0)
        ux, uy = math.cos(tyaw), math.sin(tyaw)
        place(scene.slat, float(txy[0]) + dx_l * ux, float(txy[1]) + dx_l * uy,
              z0 + c.tray_floor_t + c.slat_t / 2 + 0.015 + high, yaw_quat(yaw),
              settle_steps=20)
        settle()

    def layout_readback() -> tuple:
        return (float(scene.pull_yaw[0]), float(scene.pedestal_xy[0, 0]),
                float(scene.pedestal_xy[0, 1]), float(scene.tray_xy[0, 0]),
                float(scene.tray_xy[0, 1]), float(scene.tray_yaw[0]),
                float(scene.tray_side[0]))

    def all_finite() -> bool:
        bodies = [scene.pedestal, scene.bowl, scene.slat, scene.tray]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    check("settle: authored sandwich finite, settled and HOLDING (bowl riding "
          "~20 mm above the seated z band; slat spanning the rim); nothing seated/stowed",
          all_finite() and bool(scene.settled()[0])
          and bowl_dz() > c.seat_dz + c.seat_z_tol + 0.006
          and not bool(scene.seated()[0]) and not bool(scene.stowed()[0])
          and not bool(scene.slat_clear()[0]) and bool(scene.pedestal_home()[0]))
    check("rubric clean at reset: score 0, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 3. determinism =============================================
    env.reset(seed=777)
    step(3)
    read_a = layout_readback()
    env.reset(seed=777)
    step(3)
    read_b = layout_readback()
    print(f"[smoke] determinism readback: {read_a} vs {read_b}", flush=True)
    check("determinism: same seed -> identical layout readback",
          all(abs(a - b) < 1e-5 for a, b in zip(read_a, read_b)))

    # =========================== 4-6. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pull_yaw, ped_x, ped_y, tray_x, tray_y, "
          f"tray_yaw, side):\n{arr}", flush=True)
    check("randomization: pull-axis yaw varies across seeds (readback)",
          arr[:, 0].max() - arr[:, 0].min() > 0.05)
    check("randomization: pedestal centre jitters across seeds (readback)",
          (arr[:, 1].max() - arr[:, 1].min()) > 0.008
          or (arr[:, 2].max() - arr[:, 2].min()) > 0.008)
    check("randomization: tray side takes both values and tray yaw varies (readback)",
          arr[:, 6].max() - arr[:, 6].min() > 1.0
          and arr[:, 5].max() - arr[:, 5].min() > 0.03)

    # =========================== 7. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps (sandwich does not "
          "self-seat)",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0])
          and bowl_dz() > c.seat_dz + c.seat_z_tol + 0.006)

    # =========================== 8. seed strategy control =====================================
    # The seed's whole plan: PLACE one black bowl on the other. Constructed here by hand
    # (slat set aside on the bare counter first — NOT stowed), the stack is geometrically a
    # perfect seat, yet it earns no draw credit and no stow credit: success False.
    env.reset(seed=41)
    settle()
    park_slat_on_counter()
    seat_bowl()
    report("seed-strategy")
    check("seed strategy (bowl PLACED seated on the pedestal, slat left on the counter): "
          "seat geometrically TRUE but no draw/stow credit -> success False, score 0.25",
          bool(scene.seated()[0]) and not bool(scene.stowed()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 9. seat tolerance twin + near miss ==========================
    env.reset(seed=51)
    settle()
    park_slat_on_counter()
    seat_bowl(dx=0.012)
    report("twin-12mm")
    check("tolerance twin: a 12 mm off-centre seat counts as seated",
          bool(scene.seated()[0]) and not bool(scene.success()[0]))
    env.reset(seed=52)
    settle()
    park_slat_on_counter()
    seat_bowl(dx=0.0, dy=0.022)
    report("near-miss-22mm")
    dz = bowl_dz()
    check("near miss: a settled 22 mm off-centre seat (foot physically inside the recess, "
          "flange on the rim) is NOT seated (xy gate)",
          not bool(scene.seated()[0]) and bool(scene.settled()[0])
          and abs(dz - c.seat_dz) < 0.012)

    # =========================== 10. stow without the seat ===================================
    # "The draw failed and dumped the bowl on the counter; stow the slat anyway."
    env.reset(seed=61)
    settle()
    place(scene.bowl, -0.22, 0.14, z0 + 0.002, settle_steps=20)
    settle()
    stow_slat()
    report("stow-no-seat")
    check("stow without the seat: slat stowed, bowl dumped on the counter -> stowed TRUE "
          "but success False, score 0.25",
          bool(scene.stowed()[0]) and not bool(scene.seated()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 11-12. stow twin / slat on the rails ========================
    env.reset(seed=71)
    settle()
    place(scene.bowl, -0.22, -0.14, z0 + 0.002, settle_steps=20)
    stow_slat(dx_l=0.008)
    report("stow-twin")
    check("stow twin: slat dropped centred (8 mm off) in the tray counts as stowed",
          bool(scene.stowed()[0]))
    stow_slat(crosswise=True, high=0.030)
    report("slat-on-rails")
    slat_z = float(scene.slat.data.root_pos_w[0, 2] - origin[0, 2])
    check("slat on the rails: slat laid CROSSWISE resting on both tray rails (~24 mm above "
          "the floor band) -> NOT stowed",
          not bool(scene.stowed()[0]) and bool(scene.settled()[0])
          and slat_z > z0 + c.tray_floor_t + c.slat_t / 2 + 0.015)

    # =========================== 13. slat half out of the open end ===========================
    env.reset(seed=81)
    settle()
    park_slat_on_counter()
    seat_bowl()  # legit seat credit latches (0.25)
    stow_slat(dx_l=-0.060)  # sticking out of the open (-x) end: x band is +/-22 mm
    report("slat-half-out")
    check("slat half out: seat + slat protruding 60 mm from the tray's open end -> not "
          "stowed, success False, score pinned at 0.25",
          bool(scene.seated()[0]) and not bool(scene.stowed()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.25) < 1e-3)

    # =========================== 14. latched credit survives =================================
    place(scene.bowl, -0.25, 0.10, z0 + 0.002, settle_steps=20)
    settle()
    report("bowl-knocked-off")
    check("latched credit: the 0.25 seat credit survives the bowl being knocked back off "
          "the pedestal",
          not bool(scene.seated()[0]) and abs(float(scene.score()[0]) - 0.25) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 15. finite at the end ========================================
    check("finite: all states finite at the end", all_finite())

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.waiters_pull")
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
