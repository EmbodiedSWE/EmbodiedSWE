"""Smoke battery for SpitRoastScene — REJECTION tests for the rubric, NullRobot, RECORDED.

This is NOT a solution (the solution is solve.py — a grasp-wrench servo threads the rod
through the bore and seats it in the rack through live contact dynamics; the Franka
strategy is TASK.md's embodiment argument). Teleported constructs here are rubric
INSTRUMENTATION: build an outcome as a settled state under live physics, then assert the
rubric's verdict on it. Linkage teleports write BOTH bodies coherently.

One linear run, 13 named checks:
  1. settle     — clean reset: finite, block resting on the cradle (readback), spit lying
                  on the floor clear of rack + station, score ~0;
  2. random     — randomization by READBACK across 3 seeds: station side/yaw/xy (stored
                  tensors vs body pose) and spit floor pose pairwise differ;
  3. null       — 2 s of nothing: score < 0.05, no success;
  4. negative A — the SEED's plan (grasp + haul through a door-like arc): a sustained
                  lift-and-swing wrench on the BARE rod moves it plenty (asserted — the
                  probe is not vacuous) but earns ~0: there is no door here, and the rod
                  alone is not the goal;
  5. negative B — wrong object in place: the BARE rod laid into both rack seats — the
                  seat predicates read TRUE (asserted, probe non-vacuous) yet success
                  stays False and score ~0 (threading is load-bearing);
  6. negative C — wrong topology: the block dropped ONTO the racked rod — the bore is a
                  closed square, lateral capture is geometrically impossible; threaded
                  stays False, score ~0;
  7. partial    — tip inserted 25 mm into the bore, released: `entered` latches 0.15,
                  threaded False, no success;
  8. stage cap  — full threaded assembly built ON THE GROUND away from the rack:
                  threaded latches 0.40, not seated, not carried, no success;
  9. capture    — funnel forgiveness calibration: the loaded assembly dropped over the
                  rack at y-offsets 20 and 40 mm is gathered into the seats (both ends
                  seated);
 10. exactness  — dropped at offset 0: settles to success() and score == 1.0 exactly;
 11. miss + cap — dropped at offset 80 mm (outside the 64 mm funnel half-mouth): not
                  seated, no success, and the score sits at the latched 0.65 carried
                  stage — the non-success cap;
 12. knock-off  — from the success state, an axial shove on the block: success revoked
                  (live state), score stays latched at 0.65;
 13. frames     — video frames recorded; saved as frames.npz in the CWD.

Not probed, defended by construction (cfg asserts): a block hanging OUTSIDE the posts on
a seated rod is unconstructible — the 410 mm rod covering both notches leaves no outboard
rod for the 90 mm block, and the post + cheek assembly blocks axial passage.

Run (forge): python -u -m simgen_tasks.open_oven_i378.smoke --headless
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
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

try:
    from simgen_tasks.open_oven_i378 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

assert scene_mod.SpitRoastScene is not None  # keep the import explicit

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.spit_roast")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    ex = torch.tensor([1.0, 0.0, 0.0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.85, 0.80)) + o),
                                tuple(np.array((0.42, 0.0, 0.14)) + o),
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
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        p_s = scene.spit.data.root_pos_w[0] - scene.env_origins[0]
        p_r = scene.roast.data.root_pos_w[0] - scene.env_origins[0]
        print(f"[smoke] {tag:14s} | rod=({p_s[0]:+.3f},{p_s[1]:+.3f},{p_s[2]:+.3f}) "
              f"roast_z={p_r[2]:.3f} ent={int(scene.entered()[0])} "
              f"thr={int(scene.threaded()[0])} "
              f"seat=({int(scene.seated(0)[0])},{int(scene.seated(1)[0])}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 600, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def teleport_rod(x: float, y: float, z: float, yaw: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor([x, y, z], device=device)
        st[:, 3] = math.cos(yaw / 2.0)
        st[:, 6] = math.sin(yaw / 2.0)
        scene.spit.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def teleport_roast(x: float, y: float, z: float, yaw: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor([x, y, z], device=device)
        st[:, 3] = math.cos(yaw / 2.0)
        st[:, 6] = math.sin(yaw / 2.0)
        scene.roast.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def place_assembly(x: float, y: float, z_rod: float, yaw: float) -> None:
        """Construct the loaded spit as ONE coherent linkage write: rod at (x,y,z_rod)
        with the given yaw, block threaded and hanging centred on it (its bore top rides
        the rod: block centre z = z_rod - (bore_half - rod_r)). Zero velocities."""
        teleport_rod(x, y, z_rod, yaw)
        teleport_roast(x, y, z_rod - (c.bore_half - c.rod_r), yaw)

    # ========================= 1. settle / clean-slate ============================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("settled")
    p_r = scene.roast.data.root_pos_w[0] - scene.env_origins[0]
    p_s = scene.spit.data.root_pos_w[0] - scene.env_origins[0]
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.spit, scene.roast, scene.station))
    on_cradle = abs(float(p_r[2]) - c.roast_rest_z) < 0.01
    rod_low = float(p_s[2]) < 0.03
    check("settle: clean reset (finite, block on cradle, rod on floor, score ~0)",
          finite and on_cradle and rod_low and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization by readback =======================================
    st_sigs, rod_sigs = [], []
    readback_err = 0.0
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        env.reset()
        step(10)
        side = float(scene.side[0])
        yaw = float(scene.station_yaw[0])
        sxy = scene.station_xy[0]
        got = scene.station.data.root_pos_w[0] - scene.env_origins[0]
        readback_err = max(readback_err, float((got[:2] - sxy).norm()))
        p_s = scene.spit.data.root_pos_w[0] - scene.env_origins[0]
        st_sigs.append((round(side, 0), round(yaw, 2),
                        round(float(sxy[0]), 3), round(float(sxy[1]), 3)))
        rod_sigs.append((round(float(p_s[0]), 3), round(float(p_s[1]), 3)))
    print(f"[smoke] station draws: {st_sigs} | rod draws: {rod_sigs} | "
          f"station readback err={readback_err * 1000:.1f}mm", flush=True)
    check("randomization is real (station + rod poses differ across seeds; readback ok)",
          len(set(st_sigs)) >= 2 and len(set(rod_sigs)) >= 2 and readback_err < 0.005)

    # ========================= 3. null policy =====================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the seed's plan =====================================
    # open_oven's strategy — grasp a handle and haul it through a door-like arc. Here: a
    # sustained lift-and-swing wrench on the bare rod. It moves plenty (no door resists),
    # and it earns nothing. (The seed's literal end state — an oven door standing open —
    # does not exist in this scene: N/A, there is no articulation at all.)
    torch.manual_seed(7)
    env.reset()
    step(30)
    pos_before = (scene.spit.data.root_pos_w[0] - scene.env_origins[0]).clone()
    scene.spit_force[0] = torch.tensor([-5.0, 0.0, 7.0], device=device)  # away from the rack
    scene.spit_torque[0] = torch.tensor([0.0, 0.0, 0.4], device=device)  # swung, door-like
    step(240)
    moved = float((scene.spit.data.root_pos_w[0] - scene.env_origins[0]
                   - pos_before).norm())  # measured AT force-off (before gravity restores)
    scene.spit_force[0] = 0.0
    scene.spit_torque[0] = 0.0
    step(120)
    report("haul-test")
    print(f"[smoke]   haul displacement at force-off: {moved * 1000:.0f}mm", flush=True)
    check("negative (seed strategy): hauling the rod through an arc moves it yet scores ~0",
          moved > 0.05 and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative B: bare rod racked =====================================
    torch.manual_seed(8)
    env.reset()
    step(30)
    teleport_rod(c.rack_x, 0.0, c.z_seat + 0.004, 0.0)
    step(180)
    report("bare-racked")
    seated_both = bool(scene.seated(0)[0]) and bool(scene.seated(1)[0])
    check("negative (wrong object): bare rod seated in BOTH slots yet score ~0, no success",
          seated_both and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. negative C: block dropped onto the racked rod ===================
    teleport_roast(c.rack_x, 0.0, 0.26, 0.0)  # centred right above the seated rod
    step(300)
    report("perch-drop")
    check("negative (wrong topology): block dropped onto the racked rod is never threaded",
          not bool(scene.threaded()[0]) and float(scene.score()[0]) < 0.05
          and not bool(scene.success()[0]))

    # ========================= 7. partial insertion: entered latches 0.15 =========================
    torch.manual_seed(9)
    env.reset()
    step(30)
    p_r = scene.roast.data.root_pos_w[0] - scene.env_origins[0]
    a = quat_apply(scene.roast.data.root_quat_w[0:1], ex.unsqueeze(0))[0]
    a[2] = 0.0
    a = a / a.norm()
    yaw_a = math.atan2(float(a[1]), float(a[0]))
    ins = p_r + (c.rod_half + c.roast_s / 2.0 - 0.025) * a  # tip 25 mm inside the bore
    teleport_rod(float(ins[0]), float(ins[1]), float(p_r[2]), yaw_a)
    step(240)
    report("partial-ins")
    s_val = float(scene.score()[0])
    check("partial: 25 mm tip insertion latches `entered` (score ~0.15), no threaded/success",
          c.w_entered - 0.01 <= s_val <= c.w_entered + 0.02
          and not bool(scene.threaded()[0]) and not bool(scene.success()[0]))

    # ========================= 8. stage cap: threaded assembly on the ground ======================
    torch.manual_seed(10)
    env.reset()
    step(30)
    place_assembly(0.15, 0.0, 0.052, 0.0)  # on open floor, far from rack + both stations
    step(240)
    report("ground-thread")
    s_val = float(scene.score()[0])
    check("stage: threaded-on-ground latches 0.40 — not seated, not carried, no success",
          c.w_threaded - 0.01 <= s_val <= c.w_threaded + 0.02
          and not (bool(scene.seated(0)[0]) and bool(scene.seated(1)[0]))
          and not bool(scene.success()[0]))

    # ========================= 9. capture calibration: funnels gather 20/40 mm ===================
    torch.manual_seed(4)
    env.reset()
    step(30)
    cap_ok = {}
    for off in (0.020, 0.040):
        place_assembly(c.rack_x, off, 0.26, 0.0)  # above the funnel tops (0.267 > drop start ok)
        settle_until(lambda: bool(scene.seated(0)[0]) and bool(scene.seated(1)[0]),
                     max_steps=480)
        step(60)
        cap_ok[off] = bool(scene.seated(0)[0]) and bool(scene.seated(1)[0])
        report(f"capture@{off * 1000:.0f}mm")
    check("calibration: 20 and 40 mm lateral drop errors are funnelled into the seats",
          cap_ok[0.020] and cap_ok[0.040])

    # ========================= 11. miss + latch cap: 80 mm is outside the mouth ===================
    place_assembly(c.rack_x, 0.080, 0.26, 0.0)
    step(420)
    report("miss@80mm")
    s_val = float(scene.score()[0])
    check("calibration: 80 mm misses the 64 mm funnel half-mouth — no seat, score capped 0.65",
          not (bool(scene.seated(0)[0]) and bool(scene.seated(1)[0]))
          and not bool(scene.success()[0])
          and abs(s_val - c.w_carried) < 0.01)

    # ========================= 10. exactness: 0 mm drop -> success, score == 1.0 ==================
    place_assembly(c.rack_x, 0.0, 0.26, 0.0)
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=720)
    step(30)
    report("hung")
    check("exactness: constructed hang settles to success() and score == 1.0",
          ok and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 12. knock-off: success is live state; latch keeps 0.65 =============
    d = quat_apply(scene.spit.data.root_quat_w[0:1], ex.unsqueeze(0))[0]
    r_before = (scene.roast.data.root_pos_w[0] - scene.env_origins[0]).clone()
    scene.roast_force[0] = -25.0 * d  # axial shove toward the tip end
    step(60)
    moved = float((scene.roast.data.root_pos_w[0] - scene.env_origins[0] - r_before).norm())
    scene.roast_force[0] = 0.0
    step(180)
    report("knock-off")
    print(f"[smoke]   knock-off displacement: {moved * 1000:.0f}mm", flush=True)
    check("latch: axial shove revokes success (live state), score stays at latched 0.65",
          moved > 0.03 and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - c.w_carried) < 0.01)

    # ========================= 13. save + verdict =================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spit_roast")
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
