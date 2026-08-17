"""Smoke battery for WeightAirlockScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py: weigh the plate with the blue cube,
push the red cargo through the open doorway, remove the weight so the gate locks it
in; the Franka strategy is TASK.md's embodiment argument). Probes here use the same
fingertip-scale push numbers as solve.py (shared numbers = shared honesty); outcomes
are CONSTRUCTED through the live joints + interlock plant — teleports only transport
cubes across free space, never into a judged pose.

One linear run, 13 named checks:
  1. settle    — clean reset: finite state everywhere, gate physically at the closed
                 stop, plate at the top stop, cubes AT their sampled slots (readback),
                 score ~0, no success;
  2. plant     — mass readback: blue is the ONLY heavy object (red/decoy under the
                 spring preload margin, blue far over it);
  3. random    — cube-to-slot permutation, xy jitter and yaw all vary across 8 seeds;
                 poses verified by READBACK against the sampled values;
  4. null      — 2 s of nothing: score < 0.05, no success, gate stays closed;
  5. negative  — the SEED's plan (lift the red cube to the target and drop it): the
                 roof rejects it — the cube never gets inside, ~0;
  6. negative  — the closed gate is physical: solve-scale pushes cannot force the red
                 cube through the barred doorway; no entry latch;
  7. negative  — wrong weight: the same-size GRAY decoy dropped on the plate does not
                 depress it; the gate stays shut (only the blue cube is heavy enough);
  8. negative  — wrong cargo: full interlock operation but the DECOY pushed inside
                 instead of the red cube -> no success, only the interlock credit;
  9. negative  — near miss: red cargo left straddling the doorway when the weight is
                 removed -> the closing gate shoves it out / stalls; either way no
                 success and no inside credit;
 10. negative  — end-state order: red inside but the weight LEFT on the plate (gate
                 open) -> no success, partial credit only;
 11. exactness — removing the weight completes the lock-up -> success() and
                 score == 1.0, still true 1 s later;
 12. latch     — dropping the weight back on the plate reopens the gate and revokes
                 success (success is live state); latched credit remains;
 13. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.pick_cube_v1_i170.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.pick_cube_v1_i170 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale push numbers as solve.py (shared numbers = shared honesty).
V_DES = 0.12
KV = 6.0  # K*dt/m = 0.83 < 1 against the one-substep wrench delay
F_FF = 0.45  # N friction feedforward (matches solve.py's starting value)
F_CAP = 2.0
KX = 2.0
KDX = 1.0
SEED = 5  # fixed probe seed (no geometric preconditions needed)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weight_airlock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.68, 1.05)) + o),
                                tuple(np.array((0.02, 0.04, 0.45)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def local(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        rp = local(scene.red)
        print(f"[smoke] {tag:14s} | red=({float(rp[0]):+.3f},{float(rp[1]):+.3f},"
              f"{float(rp[2]):.3f}) gate={float(scene.gate_disp()[0]) * 1000:6.1f}mm "
              f"plate_z={float(scene.plate_z()[0]):.3f} "
              f"closed={bool(scene.gate_closed()[0])} up={bool(scene.plate_up()[0])} "
              f"inside={bool(scene.red_inside()[0])} "
              f"latch=({int(scene.latch_open[0])},{int(scene.latch_inside[0])}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def teleport(body, x: float, y: float, z: float, cube_idx: int | None = None) -> None:
        """TRANSPORT ONLY: free-space set-down, identity yaw, zero velocity."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = x, y, z
        st[0, 3] = 1.0
        st[0, 0:3] += scene.env_origins[0]
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        if cube_idx is not None:
            scene._q_ref[0, cube_idx] = st[0, 3:7]

    def push_to(cube_idx: int, y_stop: float, budget: int = 1500) -> bool:
        """solve.py's capped-force velocity servo along +y with lateral centering."""
        body = scene.cubes[cube_idx]
        for _ in range(budget):
            p = local(body)
            v = body.data.root_lin_vel_w[0]
            if float(p[1]) >= y_stop:
                scene.push_f[0, cube_idx] = 0.0
                return True
            fy = max(-F_CAP, min(F_CAP, KV * (V_DES - float(v[1])) + F_FF))
            fx = max(-1.0, min(1.0, KX * (0.0 - float(p[0])) - KDX * float(v[0])))
            scene.push_f[0, cube_idx, 0] = fx
            scene.push_f[0, cube_idx, 1] = fy
            scene.push_f[0, cube_idx, 2] = 0.0
            step(1)
        scene.push_f[0, cube_idx] = 0.0
        return False

    def wait_for(fn, budget_steps: int, chunk: int = 5) -> bool:
        for _ in range(max(1, budget_steps // chunk)):
            if bool(fn()):
                return True
            step(chunk)
        return bool(fn())

    def blue_on_plate() -> bool:
        teleport(scene.blue, c.ped_center[0], c.ped_center[1],
                 c.plate_rest_z + c.plate_size[2] / 2 + c.blue_size / 2 + 0.008, cube_idx=1)
        return wait_for(lambda: scene.plate_depressed()[0] & scene.gate_open()[0], 480)

    def fresh(seed: int) -> None:
        torch.manual_seed(seed)
        env.reset()
        step(30)

    # ========================= 1. settle / clean-slate ========================================
    fresh(SEED)
    step(30)
    report("reset")
    slot_err = max(float((local(b)[0:2] - scene.spawn_xy[0, i]).norm())
                   for i, b in enumerate(scene.cubes))
    check("settle: clean reset (finite, gate closed, plate up, cubes at sampled slots, ~0)",
          finite_all() and bool(scene.gate_closed()[0]) and bool(scene.plate_up()[0])
          and slot_err < 0.005 and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 2. plant readback (masses vs spring preload) ===================
    masses = {nm: float(b.root_physx_view.get_masses().reshape(-1)[0])
              for nm, b in (("red", scene.red), ("blue", scene.blue),
                            ("decoy", scene.decoy), ("plate", scene.plate))}
    margin = c.spring_force - masses["plate"] * 9.81  # spare spring force on the empty plate
    print(f"[smoke] masses={masses} spring={c.spring_force}N empty-margin={margin:.2f}N "
          f"decoy_w={masses['decoy'] * 9.81:.2f}N blue_w={masses['blue'] * 9.81:.2f}N",
          flush=True)
    check("plant: blue is the only object heavier than the spring margin "
          "(red/decoy below, blue far above)",
          masses["red"] * 9.81 < 0.7 * margin and masses["decoy"] * 9.81 < 0.7 * margin
          and masses["blue"] * 9.81 > 3.0 * margin)

    # ========================= 3. randomization across seeds ==================================
    slots_t = torch.tensor(c.slots, device=device)
    draws = []
    ok_rb = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        # READBACK: each cube physically at its sampled pose
        for i, b in enumerate(scene.cubes):
            ok_rb &= float((local(b)[0:2] - scene.spawn_xy[0, i]).norm()) < 0.005
        red_slot = int((slots_t - scene.spawn_xy[0, 0]).norm(dim=1).argmin())
        blue_slot = int((slots_t - scene.spawn_xy[0, 1]).norm(dim=1).argmin())
        draws.append((red_slot, blue_slot,
                      round(float(scene.spawn_xy[0, 0, 0]), 3),
                      round(float(scene.spawn_xy[0, 0, 1]), 3),
                      round(float(scene.spawn_yaw[0, 0]), 2)))
    print(f"[smoke] draws (red_slot, blue_slot, red_x, red_y, red_yaw): {draws}", flush=True)
    check("randomization is real (slot deal varies, jitter varies, yaw varies; READBACK)",
          ok_rb and len({d[0] for d in draws}) >= 2 and len({d[1] for d in draws}) >= 2
          and len({d[2] for d in draws}) >= 5 and len({d[4] for d in draws}) >= 5)

    # ========================= 4. null policy =================================================
    fresh(SEED)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05, no success, gate stays closed",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0])
          and bool(scene.gate_closed()[0]))

    # ========================= 5. negative: the SEED's plan (lift + drop at target) ===========
    fresh(SEED)
    teleport(scene.red, c.vault_center[0], c.vault_center[1], c.z0 + 0.30, cube_idx=0)
    step(300)  # drop onto the roof, settle / skitter
    report("seed-drop")
    check("negative (seed strategy): red dropped at the target from above never gets in "
          "— the roof rejects the seed's plan",
          finite_all() and not bool(scene.red_inside()[0]) and not bool(scene.success()[0])
          and not bool(scene.latch_inside[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 6. negative: the closed gate is physical =======================
    fresh(SEED)
    teleport(scene.red, 0.0, -0.08, c.z0 + c.red_size / 2 + 0.003, cube_idx=0)
    step(30)
    push_to(0, 0.05, budget=360)  # solve-scale push straight at the barred doorway
    step(60)
    report("barred-push")
    ry = float(local(scene.red)[1])
    check("negative (closed gate): a solve-scale push cannot force the doorway "
          "(gate holds, cube stays outside, no entry latch)",
          bool(scene.gate_closed()[0]) and ry < 0.005
          and not bool(scene.red_inside()[0]) and not bool(scene.latch_inside[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 7. negative: the decoy is too light ============================
    fresh(SEED)
    teleport(scene.decoy, c.ped_center[0], c.ped_center[1],
             c.plate_rest_z + c.plate_size[2] / 2 + c.decoy_size / 2 + 0.008, cube_idx=2)
    step(240)
    report("decoy-weigh")
    check("negative (wrong weight): the same-size gray decoy cannot depress the plate; "
          "the gate stays shut",
          bool(scene.plate_up()[0]) and bool(scene.gate_closed()[0])
          and not bool(scene.latch_open[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 8. negative: wrong cargo pushed inside =========================
    fresh(SEED)
    ok_w = blue_on_plate()
    teleport(scene.decoy, 0.0, -0.06, c.z0 + c.decoy_size / 2 + 0.003, cube_idx=2)
    step(30)
    ok_p = push_to(2, 0.125)
    step(90)
    teleport(scene.blue, -0.25, -0.28, c.z0 + c.blue_size / 2 + 0.003, cube_idx=1)
    ok_c = wait_for(lambda: scene.gate_closed()[0] & scene.plate_up()[0], 480)
    step(120)
    report("wrong-cargo")
    dy = float(local(scene.decoy)[1])
    check("negative (wrong cargo): decoy locked in instead of red -> no success, "
          "interlock credit only",
          ok_w and ok_p and ok_c and dy > c.inside_y_min  # the construct really executed
          and not bool(scene.red_inside()[0]) and not bool(scene.success()[0])
          and 0.20 < float(scene.score()[0]) < 0.30)

    # ========================= 9. negative: cargo left straddling the doorway =================
    fresh(SEED)
    ok_w = blue_on_plate()
    teleport(scene.red, 0.0, -0.06, c.z0 + c.red_size / 2 + 0.003, cube_idx=0)
    step(30)
    ok_p = push_to(0, 0.010)  # stop in the gate plane — a doorway straddle
    step(60)
    teleport(scene.blue, -0.25, -0.28, c.z0 + c.blue_size / 2 + 0.003, cube_idx=1)
    step(480)  # the closing gate shoves the straddler / stalls on it
    report("straddle")
    check("negative (near miss): cargo left in the doorway when the weight comes off -> "
          "shoved out or gate stalled; no success, no inside credit",
          finite_all() and not bool(scene.red_inside()[0])
          and not bool(scene.latch_inside[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.30)

    # ========================= 10-12. order / exactness / revocation ==========================
    fresh(SEED)
    ok_w = blue_on_plate()
    step(60)
    teleport(scene.red, 0.0, -0.06, c.z0 + c.red_size / 2 + 0.003, cube_idx=0)
    step(30)
    ok_p = push_to(0, 0.125)
    step(90)
    report("weight-still-on")
    check("negative (end-state order): red inside but the weight still on the plate "
          "(gate open) -> no success, partial credit",
          ok_w and ok_p and bool(scene.red_inside()[0]) and bool(scene.latch_inside[0])
          and not bool(scene.gate_closed()[0]) and not bool(scene.success()[0])
          and 0.55 < float(scene.score()[0]) < 0.65)

    teleport(scene.blue, -0.25, -0.28, c.z0 + c.blue_size / 2 + 0.003, cube_idx=1)
    ok_c = wait_for(lambda: scene.gate_closed()[0] & scene.plate_up()[0], 480)
    ok_s = wait_for(lambda: scene.success()[0], 480)
    step(120)
    report("locked")
    good = ok_c and ok_s and bool(scene.success()[0]) \
        and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off
    check("exactness: weight off -> gate locks the cargo in -> success() and "
          "score == 1.0, stable",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    ok_r = blue_on_plate()  # drop the weight back on: the gate reopens
    step(60)
    report("revoked")
    check("achievement latch: re-weighing the plate reopens the gate and revokes success; "
          "latched credit remains",
          ok_r and not bool(scene.success()[0])
          and 0.55 < float(scene.score()[0]) < 0.70)

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weight_airlock")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
