"""Smoke battery for CatwalkBridgeScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py: seat the plank as a spanning bridge,
then push the cube across it into the island shelter; the Franka strategy is TASK.md's
embodiment argument). Probes here use the same capped push numbers as solve.py (shared
numbers = shared honesty); outcomes are CONSTRUCTED through contact dynamics —
teleports only transport bodies across free space, never into a judged pose.

One linear run, 13 named checks:
  1. settle    — clean reset: finite state everywhere, all three bodies physically AT
                 their sampled slots (readback), no bridge, score ~0, no success;
  2. plant     — mass readback for plank/cube/rod, and the load-bearing geometry facts
                 (plank half-length < gap < plank length; a riding cube fits under the
                 roof) hold in the built cfg;
  3. random    — slot deal (swap), xy jitter and yaw all vary across 8 seeds; poses
                 verified by READBACK against the sampled values;
  4. null      — 2 s of nothing: score < 0.05, no success, no latches;
  5. negative  — the SEED's plan (grasp the red cube, lift, carry to the target, drop):
                 the drop lands ON THE ROOF — never inside, ~0 (physical roof readback);
  6. negative  — crossing without a bridge: the same push that solves the task shoves
                 the cube off the sill INTO THE TRENCH; no transit latch, no success;
  7. negative  — the plank cannot be SHOVED in lengthwise (half-length < gap): a
                 decisive push tips it into the trench; bridge latch never fires;
  8. negative  — near miss: plank set down fully ON THE ISLAND (level, right height,
                 still — but covering neither sill pair) -> bridge_ok false, no latch;
  9. negative  — wrong object: the blue rod seated spanning the trench earns nothing
                 (the rubric reads the PLANK);
 10. partial   — real bridge built, cube pushed to mid-island and STOPPED ON THE PLANK:
                 transit latch fires, score ~0.60, but no success (on-plank z excluded);
 11. exactness — continuing the push drops the cube off the plank's far end onto the
                 island floor -> success() and score == 1.0, still true 1 s later;
 12. latch     — teleporting the cube back to the bench revokes success (success is
                 live state); the latched 0.60 remains;
 13. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.pick_cube_i295.smoke --headless
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
    from simgen_tasks.pick_cube_i295 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same rod-tip-scale cube-push numbers as solve.py (shared numbers = shared honesty).
V_DES = 0.08  # m/s
KV = 0.4  # N*s/m; K*dt/m = 0.067 << 1 against the one-substep wrench delay
F_FF0 = 0.16  # N friction feedforward (solve.py's starting value)
F_FF_MAX = 0.32
F_CAP = 0.40  # N — under the cube's mg = 0.49 N tipping bound
KX = 3.0
KDX = 0.6
F_LAT = 0.15
# Plank-shove probe (check 7): a DECISIVE fingertip shove, well above the plank's
# sliding friction (~0.44 N) — the point is that even a strong shove fails.
P_V = 0.10
P_KV = 0.6  # K*dt/m = 0.033 << 1
P_FF = 0.50
P_CAP = 0.90
SEED = 3  # fixed probe seed (probes teleport their own starts; no preconditions needed)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.catwalk_bridge")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.82, -0.80, 0.78)) + o),
                                tuple(np.array((-0.06, 0.0, 0.10)) + o),
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

    def up_z(body) -> float:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        return float(quat_apply(body.data.root_quat_w, ez)[0, 2])

    def report(tag: str) -> None:
        pp, cp = local(scene.plank), local(scene.cube)
        print(f"[smoke] {tag:14s} | plank=({float(pp[0]):+.3f},{float(pp[1]):+.3f},"
              f"{float(pp[2]):.3f}) cube=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
              f"{float(cp[2]):.3f}) bridge={bool(scene.bridge_ok()[0])} "
              f"on_island={bool(scene.cube_on_island()[0])} "
              f"latch=({int(scene.latch_bridge[0])},{int(scene.latch_transit[0])}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._named_bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    def teleport(idx: int, x: float, y: float, z: float) -> None:
        """TRANSPORT ONLY: free-space set-down, identity yaw, zero velocity."""
        body = scene.bodies[idx]
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = x, y, z
        st[0, 3] = 1.0
        st[0, 0:3] += scene.env_origins[0]
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        scene._q_ref[0, idx] = st[0, 3:7]

    def push_cube_to(x_stop: float, budget: int = 2400) -> bool:
        """solve.py's capped-force velocity servo + friction feedforward along +x with
        lateral centering on the corridor axis y = 0."""
        ff = F_FF0
        last_x = float(local(scene.cube)[0])
        last_ck = 0
        for i in range(budget):
            p = local(scene.cube)
            v = scene.cube.data.root_lin_vel_w[0]
            if float(p[0]) >= x_stop:
                scene.push_f[0, 1] = 0.0
                return True
            fx = max(-F_CAP, min(F_CAP, KV * (V_DES - float(v[0])) + ff))
            fy = max(-F_LAT, min(F_LAT, KX * (0.0 - float(p[1])) - KDX * float(v[1])))
            scene.push_f[0, 1, 0] = fx
            scene.push_f[0, 1, 1] = fy
            scene.push_f[0, 1, 2] = 0.0
            step(1)
            if i - last_ck >= 120:  # stall watch: escalate the FEEDFORWARD, not the cap
                x = float(local(scene.cube)[0])
                if x - last_x < 0.04 and ff < F_FF_MAX:
                    ff = min(F_FF_MAX, ff * 1.4)
                last_x, last_ck = x, i
        scene.push_f[0, 1] = 0.0
        return False

    def wait_for(fn, budget_steps: int, chunk: int = 5) -> bool:
        for _ in range(max(1, budget_steps // chunk)):
            if bool(fn()):
                return True
            step(chunk)
        return bool(fn())

    def build_bridge() -> bool:
        """solve.py's PHASE 1: hover the plank 1 cm above the spanning pose, release,
        gravity + the two sills do the seating."""
        teleport(0, c.span_center_x, 0.0, c.plank_seat_z + 0.010)
        return wait_for(lambda: scene.bridge_ok()[0], 360)

    def fresh(seed: int) -> None:
        torch.manual_seed(seed)
        env.reset()
        step(30)

    # ========================= 1. settle / clean-slate ========================================
    fresh(SEED)
    step(30)
    report("reset")
    slot_err = max(float((local(b)[0:2] - scene.spawn_xy[0, i]).norm())
                   for i, b in enumerate(scene.bodies))
    check("settle: clean reset (finite, bodies at sampled slots by readback, no bridge, ~0)",
          finite_all() and slot_err < 0.005 and not bool(scene.bridge_ok()[0])
          and not bool(scene.latch_bridge[0]) and not bool(scene.latch_transit[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. plant readback (masses + forcing geometry) ==================
    masses = {nm: float(b.root_physx_view.get_masses().reshape(-1)[0])
              for nm, b in (("plank", scene.plank), ("cube", scene.cube), ("rod", scene.rod))}
    print(f"[smoke] masses={masses} gap={c.gap:.3f} plank_len={c.plank_size[0]:.3f} "
          f"roof_underside={c.roof_z0:.3f} cube_ride_top={c.cube_ride_z + c.cube_size / 2:.3f}",
          flush=True)
    check("plant: masses as authored; plank half-length < gap < plank length (un-shovable "
          "but spanning); a riding cube fits under the roof",
          abs(masses["plank"] - c.plank_mass) < 0.1 * c.plank_mass
          and abs(masses["cube"] - c.cube_mass) < 0.1 * c.cube_mass
          and abs(masses["rod"] - c.rod_mass) < 0.1 * c.rod_mass
          and c.plank_size[0] / 2 < c.gap < c.plank_size[0]
          and c.cube_ride_z + c.cube_size / 2 < c.roof_z0 - 0.01)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    ok_rb = True
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        # READBACK: each body physically at its sampled pose
        for i, b in enumerate(scene.bodies):
            ok_rb &= float((local(b)[0:2] - scene.spawn_xy[0, i]).norm()) < 0.005
        draws.append((int(scene.spawn_swap[0]),
                      round(float(scene.spawn_xy[0, 1, 0]), 3),
                      round(float(scene.spawn_xy[0, 1, 1]), 3),
                      round(float(scene.spawn_yaw[0, 0]), 2),
                      round(float(scene.spawn_yaw[0, 2]), 2)))
    print(f"[smoke] draws (swap, cube_x, cube_y, plank_yaw, rod_yaw): {draws}", flush=True)
    check("randomization is real (slot deal varies, jitter varies, yaws vary; READBACK)",
          ok_rb and len({d[0] for d in draws}) == 2
          and len({d[1] for d in draws}) >= 5 and len({d[2] for d in draws}) >= 5
          and len({d[3] for d in draws}) >= 5 and len({d[4] for d in draws}) >= 5)

    # ========================= 4. null policy =================================================
    fresh(SEED)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05, no success, no latches",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0])
          and not bool(scene.latch_bridge[0]) and not bool(scene.latch_transit[0]))

    # ========================= 5. negative: the SEED's plan (carry + drop at target) ==========
    fresh(SEED)
    teleport(1, 0.16, 0.0, c.roof_z0 + c.roof_t + 0.12)  # carry above the target, drop
    step(300)
    report("seed-drop")
    cz = float(local(scene.cube)[2])
    check("negative (seed strategy): the cube carried over the target and dropped lands ON "
          "THE ROOF (physical roof readback) — never inside, ~0",
          finite_all() and cz > c.roof_z0 and not bool(scene.cube_on_island()[0])
          and not bool(scene.success()[0]) and not bool(scene.latch_transit[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 6. negative: crossing without a bridge =========================
    fresh(SEED)
    teleport(1, -0.12, 0.0, c.cube_deck_z + 0.003)  # on the bench, before the mouth
    step(30)
    push_cube_to(0.17, budget=480)  # the SOLVING push — but no bridge exists
    step(60)
    report("no-bridge")
    cz = float(local(scene.cube)[2])
    check("negative (no bridge): the solving push shoves the cube off the sill INTO THE "
          "TRENCH; no transit latch, no success",
          finite_all() and cz < 0.05 and not bool(scene.latch_transit[0])
          and not bool(scene.cube_on_island()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 7. negative: the plank cannot be shoved in =====================
    fresh(SEED)
    teleport(0, -0.26, 0.0, c.z0 + c.plank_size[2] / 2 + 0.003)  # aligned, on the bench
    step(30)
    px0 = float(local(scene.plank)[0])
    for _ in range(1200):  # decisive lengthwise shove toward the island
        p = local(scene.plank)
        v = scene.plank.data.root_lin_vel_w[0]
        if up_z(scene.plank) < 0.90 or float(p[2]) < 0.070 or float(p[0]) > 0.05:
            break
        fx = max(-P_CAP, min(P_CAP, P_KV * (P_V - float(v[0])) + P_FF))
        scene.push_f[0, 0, 0] = fx
        scene.push_f[0, 0, 1] = max(-F_LAT, min(F_LAT, KX * (0.0 - float(p[1]))
                                                - KDX * float(v[1])))
        step(1)
    scene.push_f[0, 0] = 0.0
    step(240)  # let the tip-over finish
    report("plank-shove")
    px1, pz1 = float(local(scene.plank)[0]), float(local(scene.plank)[2])
    check("negative (lengthwise shove): the plank moved but TIPPED INTO THE TRENCH "
          "(half-length < gap) — never a bridge, latch never fired",
          finite_all() and px1 > px0 + 0.05  # the shove really executed
          and (up_z(scene.plank) < 0.95 or pz1 < c.bridge_z_band[0])
          and not bool(scene.bridge_ok()[0]) and not bool(scene.latch_bridge[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 8. negative: near-miss plank fully on the island ===============
    fresh(SEED)
    teleport(0, 0.13, 0.0, c.plank_seat_z + 0.010)  # level, right height — wrong place
    step(240)
    report("island-rest")
    check("negative (near miss): plank level and still at sill height but fully ON THE "
          "ISLAND (covers neither sill) -> bridge_ok false, no latch",
          finite_all() and up_z(scene.plank) > 0.98
          and abs(float(local(scene.plank)[2]) - c.plank_seat_z) < 0.006  # construct is real
          and not bool(scene.bridge_ok()[0]) and not bool(scene.latch_bridge[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 9. negative: wrong object spanning =============================
    fresh(SEED)
    # Park the plank aligned on a clear bench corner first — a free-yaw spawn can stick a
    # corner into the rod's y=0 set-down line (transport only, free-space set-down).
    teleport(0, -0.34, 0.10, c.z0 + c.plank_size[2] / 2 + 0.003)
    step(30)
    teleport(2, c.span_center_x, 0.0, c.z0 + c.rod_size[2] / 2 + 0.010)  # rod hover-drop
    step(240)
    report("rod-span")
    rp = local(scene.rod)
    print(f"[smoke] rod=({float(rp[0]):+.3f},{float(rp[1]):+.3f},{float(rp[2]):.3f}) "
          f"up_z={up_z(scene.rod):.3f}", flush=True)
    rz = float(rp[2])
    check("negative (wrong object): the blue rod seated spanning the trench earns nothing "
          "— the rubric reads the PLANK",
          finite_all() and abs(rz - (c.z0 + c.rod_size[2] / 2)) < 0.006  # rod really spans
          and not bool(scene.bridge_ok()[0]) and not bool(scene.latch_bridge[0])
          and float(scene.score()[0]) < 0.05)

    # ========================= 10. partial: stop ON the plank =================================
    fresh(SEED)
    ok_b = build_bridge()
    step(30)
    teleport(1, -0.105, 0.0, c.cube_ride_z + 0.006)  # onto the plank's back seat
    step(30)
    ok_p = push_cube_to(0.05)  # cross the trench, then STOP on the plank
    scene.push_f[0, 1] = 0.0
    step(120)
    report("stop-short")
    check("partial: cube stopped ON the plank past mid-island -> bridge+transit latched, "
          "score ~0.60, but NO success (on-plank z excluded from the goal band)",
          ok_b and ok_p and bool(scene.latch_bridge[0]) and bool(scene.latch_transit[0])
          and not bool(scene.cube_on_island()[0]) and not bool(scene.success()[0])
          and 0.55 < float(scene.score()[0]) < 0.65)

    # ========================= 11. exactness: finish the crossing =============================
    ok_p = push_cube_to(0.17)
    step(60)
    ok_s = wait_for(lambda: scene.success()[0], 480)
    report("landed")
    good = ok_p and ok_s and bool(scene.success()[0]) \
        and abs(float(scene.score()[0]) - 1.0) < 1e-3
    step(120)  # ...and it persists hands-off
    check("exactness: continuing the push drops the cube onto the island floor -> "
          "success() and score == 1.0, stable",
          good and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 12. achievement latch / revocation =============================
    teleport(1, -0.30, 0.07, c.cube_deck_z + 0.003)  # transport the cube back to the bench
    step(120)
    report("revoked")
    check("achievement latch: cube moved back to the bench revokes success (live state); "
          "the latched 0.60 remains",
          not bool(scene.success()[0]) and bool(scene.latch_bridge[0])
          and bool(scene.latch_transit[0]) and 0.55 < float(scene.score()[0]) < 0.65)

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.catwalk_bridge")
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
