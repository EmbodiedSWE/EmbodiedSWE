"""Smoke / rubric-REJECTION battery for CounterweightVaultScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i146) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — the hover-release counterweight + gravity-delivery
run — is the acceptance evidence that the rubric ACCEPTS a correct outcome). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partially-right) outcome
as a settled state and asserts the rubric's verdict on it. No probe below constructs
full success (the battery tops out at the latched 0.50).

  1-2. settle/no-NaN     — the authored layout settles finite at the gravity-closed
                           rest (~13 deg < closed band 25), white bowl home, nothing
                           loaded/held/delivered; rubric clean: score 0, no success;
  3.  determinism        — the same seed twice -> identical layout readback;
  4-5. randomization     — READBACK over 8 seeded resets: black-bowl MASS spreads,
                           park sides take both values, park centres jitter;
  6.  null policy        — 240 idle steps -> gate stays closed, score ~0, no success;
  7.  teleported-open    — the EMPTY rotor written to open 95 deg (on the joint
                           manifold): `held` never latches (no bowl in the pan) and
                           gravity re-closes the gate by itself -> score 0;
  8.  closed-gate drop   — cube released over the vault with the gate CLOSED: it comes
                           to rest ON the flap ABOVE the rim (the mouth is physically
                           blocked), delivered stays False, score 0;
  9.  teleport-through   — cube written directly INSIDE the white bowl UNDER the closed
                           flap: geometrically in, but `delivered` (gate-open clause)
                           never latches -> success False, score 0;
 10.  seed strategy      — the SEED's move (carry the black bowl to the goal and put it
                           down there) lands it on the closed flap/vault: nothing
                           latches, score 0;
 11.  wrong counterweight— the 15 g cube dropped INTO the pan barely moves the gate
                           (predicted ~4 deg): never opens, no credit;
 12.  real stage 1       — black bowl hover-released into the pan: the gate swings open
                           and HOLDS under the load -> loaded+held latch, score 0.50;
 13.  displaced target   — with the gate held open, the white bowl moved OUT of the
                           recess and the cube dropped INTO it 12 mm off-centre: cube_in
                           True (tolerance twin) but vbowl_home False -> `delivered`
                           does NOT latch, score pinned at 0.50;
 14.  near miss          — cube settled on the counter beside that bowl -> not cube_in;
 15.  unload + latch     — bowl re-parked: the emptied gate re-closes BY ITSELF, and the
                           latched 0.50 survives (credit never evaporates), no success;
 16.  finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweight_vault")().build(num_envs=args.num_envs,
                                                         device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z
    vx, vy = c.vault_xy
    hx, hy = c.hinge_xy

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -1.10, 0.95)) + o),
                                tuple(np.array((0.08, 0.06, 0.28)) + o),
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
        print(f"[smoke] {tag:18s} | open={float(scene.open_angle_deg()[0]):6.1f}deg "
              f"in_pan={bool(scene.bowl_in_pan()[0])} cube_in={bool(scene.cube_in_vbowl()[0])} "
              f"home={bool(scene.vbowl_home()[0])} "
              f"L/H/D={int(scene.ever_loaded[0])}{int(scene.ever_held[0])}"
              f"{int(scene.ever_delivered[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    def place(body, x: float, y: float, z: float, quat: tuple | None = None,
              settle_steps: int = 30) -> None:
        """Probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def open_quat(open_deg: float) -> tuple:
        """Rotor orientation at gate opening `open_deg` — ON the joint manifold
        (rotation about the authored tilted hinge axis through the rotor origin)."""
        b = math.radians(c.tilt_deg)
        ax = (0.0, -math.sin(b), math.cos(b))
        half = math.radians(-open_deg) / 2.0
        s = math.sin(half)
        return (math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s)

    def drop_bowl_into_pan(max_wait: int = 900) -> bool:
        """The solve's stage-1 move: hover-release the black bowl 18 mm above the pan
        floor and wait for the gravity swing to open AND hold."""
        pan = scene.pan_center_w()[0] - origin[0]
        place(scene.cw_bowl, float(pan[0]), float(pan[1]), float(pan[2]) + 0.018,
              settle_steps=0)
        waited = 0
        while waited < max_wait:
            step(10)
            waited += 10
            if (bool(scene.gate_open()[0])
                    and float(scene.rotor_axis_w()[0]) < c.settle_rotor_w
                    and bool(scene.bowl_in_pan()[0])):
                break
        settle(max_steps=400)
        return (bool(scene.bowl_in_pan()[0]) and bool(scene.gate_open()[0])
                and bool(scene.settled()[0]))

    def layout_readback() -> tuple:
        bw = scene.cw_bowl.data.root_pos_w[0] - origin[0]
        cu = scene.cube.data.root_pos_w[0] - origin[0]
        vb = scene.v_bowl.data.root_pos_w[0] - origin[0]
        return (float(scene.open_angle_deg()[0]), float(scene.bowl_mass[0]),
                float(bw[0]), float(bw[1]), float(cu[0]), float(cu[1]),
                float(scene.bowl_side[0]), float(scene.cube_side[0]),
                float(vb[0]), float(vb[1]))

    def all_finite() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in (scene.rotor, scene.cw_bowl, scene.v_bowl, scene.cube))

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    check("settle: authored layout finite and settled; gate at its gravity-closed rest "
          "(< closed band); white bowl home; nothing loaded/held/delivered",
          all_finite() and bool(scene.settled()[0])
          and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and bool(scene.vbowl_home()[0]) and not bool(scene.bowl_in_pan()[0])
          and not bool(scene.cube_in_vbowl()[0]) and not bool(scene.ever_held[0]))
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

    # =========================== 4-5. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (open, mass, bx, by, cx, cy, bside, cside, "
          f"vbx, vby):\n{arr}", flush=True)
    check("randomization: black-bowl MASS spreads across seeds (readback)",
          arr[:, 1].max() - arr[:, 1].min() > 0.03)
    check("randomization: park sides each take both values; park centres jitter",
          arr[:, 6].max() - arr[:, 6].min() > 1.0
          and arr[:, 7].max() - arr[:, 7].min() > 1.0
          and (arr[:, 2].max() - arr[:, 2].min()) > 0.005
          and (arr[:, 4].max() - arr[:, 4].min()) > 0.005)

    # =========================== 6. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: gate stays closed, score ~0, no success after 240 idle steps",
          float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 7. teleported-open EMPTY gate ===============================
    env.reset(seed=35)
    settle()
    place(scene.rotor, hx, hy, z0 + c.hinge_h, open_quat(95.0), settle_steps=0)
    step(5)
    open_now = float(scene.open_angle_deg()[0])
    print(f"[smoke] rotor written open: open={open_now:.1f}deg", flush=True)
    waited = 0
    while waited < 2400:  # the empty pendulum needs several swings to decay back
        step(20)
        waited += 20
        if (float(scene.open_angle_deg()[0]) < c.closed_max_deg
                and float(scene.rotor_axis_w()[0]) < c.settle_rotor_w):
            break
    report("teleported-open")
    check("teleported-open EMPTY gate: `held` never latches (no bowl in the pan) and "
          "gravity re-closes the gate by itself; score 0",
          open_now > 80.0 and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and not bool(scene.ever_held[0]) and not bool(scene.ever_loaded[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. closed gate physically blocks the drop ===================
    env.reset(seed=40)
    settle()
    place(scene.cube, vx, vy, z0 + 0.16, settle_steps=10)
    settle()
    report("cube-on-flap")
    cube_z = float(scene.cube.data.root_pos_w[0, 2] - origin[0, 2])
    check("closed-gate drop: the cube released over the vault rests ON the flap ABOVE "
          "the rim — mouth physically blocked; delivered False, score 0",
          cube_z > z0 + c.vault_h and not bool(scene.cube_in_vbowl()[0])
          and not bool(scene.ever_delivered[0])
          and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 9. cube teleported THROUGH the closed gate ==================
    env.reset(seed=44)
    settle()
    vb = scene.v_bowl.data.root_pos_w[0] - origin[0]
    place(scene.cube, float(vb[0]), float(vb[1]),
          float(vb[2]) + c.bowl_floor_t + c.cube_size / 2 + 0.002, settle_steps=20)
    settle()
    report("teleport-through")
    check("teleport-through: cube written INSIDE the white bowl UNDER the closed flap "
          "is geometrically in, but `delivered` (gate-open clause) never latches -> "
          "success False, score 0",
          bool(scene.cube_in_vbowl()[0]) and not bool(scene.ever_delivered[0])
          and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 10. seed strategy control ====================================
    # The seed's whole plan: pick the black bowl up, carry it to the goal, put it down.
    # Here that lands it on the closed flap over the vault — nothing latches.
    env.reset(seed=48)
    settle()
    place(scene.cw_bowl, vx, vy, z0 + 0.135, settle_steps=10)
    settle()
    report("seed-strategy")
    check("seed strategy (black bowl carried onto the goal): rests on the closed "
          "flap/vault, gate stays closed, score 0, no success",
          not bool(scene.bowl_in_pan()[0]) and not bool(scene.ever_loaded[0])
          and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 11. wrong counterweight (the cube) ==========================
    env.reset(seed=52)
    settle()
    pan = scene.pan_center_w()[0] - origin[0]
    place(scene.cube, float(pan[0]), float(pan[1]),
          float(pan[2]) + c.cube_size / 2 + 0.012, settle_steps=10)
    settle()
    report("cube-in-pan")
    from isaaclab.utils.math import quat_apply_inverse
    loc = quat_apply_inverse(scene.rotor.data.root_quat_w,
                             scene.cube.data.root_pos_w - scene.pan_center_w())[0]
    check("wrong counterweight: the 15 g cube IN the pan (readback: it landed) moves "
          "the gate only a few degrees — never opens, no credit",
          float(loc[2]) < 0.05  # the probe really delivered its weight to the pan
          and float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and not bool(scene.gate_open()[0]) and not bool(scene.ever_loaded[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 12. real stage-1 physics ====================================
    env.reset(seed=56)
    settle()
    held = drop_bowl_into_pan() or drop_bowl_into_pan()  # one retry for drop luck
    report("counterweight-in")
    check("real stage 1: the black bowl hover-released into the pan swings the gate "
          "open and HOLDS it (loaded+held latch, score 0.50, no success)",
          held and bool(scene.ever_loaded[0]) and bool(scene.ever_held[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 13. displaced target + tolerance twin =======================
    # Gate legitimately open (bowl still in the pan): white bowl moved OUT of the
    # recess onto the counter, cube dropped INTO it 12 mm off-centre.
    place(scene.v_bowl, -0.28, 0.0, z0 + 0.002, settle_steps=20)
    place(scene.cube, -0.28 + 0.012, 0.0,
          z0 + 0.002 + c.bowl_floor_t + c.cube_size / 2 + 0.003, settle_steps=20)
    settle()
    report("displaced-target")
    check("displaced target: cube 12 mm off-centre INSIDE the moved white bowl counts "
          "as cube_in (tolerance twin) but vbowl_home is False -> `delivered` does "
          "NOT latch, score pinned at 0.50",
          bool(scene.cube_in_vbowl()[0]) and not bool(scene.vbowl_home()[0])
          and bool(scene.gate_open()[0]) and not bool(scene.ever_delivered[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 14. near miss ================================================
    place(scene.cube, -0.28, 0.15, z0 + c.cube_size / 2 + 0.002, settle_steps=20)
    settle()
    report("near-miss")
    check("near miss: cube settled on the counter beside that bowl is NOT cube_in; "
          "score stays 0.50",
          not bool(scene.cube_in_vbowl()[0])
          and abs(float(scene.score()[0]) - 0.50) < 1e-3)

    # =========================== 15. unload: self re-close + latched credit ==================
    place(scene.cw_bowl, c.bowl_park[0], c.bowl_park[1] * float(scene.bowl_side[0]),
          z0 + 0.002, settle_steps=0)
    waited = 0
    while waited < 2400:
        step(20)
        waited += 20
        if (float(scene.open_angle_deg()[0]) < c.closed_max_deg
                and float(scene.rotor_axis_w()[0]) < c.settle_rotor_w):
            break
    report("unloaded")
    check("unload: the emptied gate re-closes BY ITSELF, and the latched 0.50 "
          "survives the mishap (credit never evaporates); no success",
          float(scene.open_angle_deg()[0]) < c.closed_max_deg
          and abs(float(scene.score()[0]) - 0.50) < 1e-3
          and not bool(scene.ever_delivered[0]) and not bool(scene.success()[0]))

    # =========================== 16. finite at the end ========================================
    check("finite: all states finite at the end", all_finite())

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counterweight_vault")
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
