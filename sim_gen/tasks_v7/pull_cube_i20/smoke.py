"""Smoke / rubric-REJECTION battery for BeamScaleScene (sim_gen task `pull_cube_i20`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop cubes one at a time into the raised pan until
the beam swings over — is the acceptance evidence that the rubric ACCEPTS a correct
outcome; it passes on seeds 0/1/2 covering counterweights needing 3 and 1 cubes).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
as a settled state and asserts the rubric REJECTS it — plus physics probes that prove
the mechanism is real (the counterweight genuinely holds the beam down against k-1
cubes, and genuinely restores a beam that was tipped without load). No probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset settles finite; the beam RESTS fully tilted onto the
                            counterweight side; block in its pan, cubes flat on the
                            floor; score ~0, no success;
  3-4. randomization      — READBACK over 6 seeded resets: counterweight choice and
                            side vary; stand pose (xy + yaw) and cube positions vary;
  5.  null policy         — 300 idle steps -> score ~0, no success;
  6.  seed strategy       — maniskill/pull_cube's move (drag cubes across the floor
                            to a spot near the robot): all three cubes teleported to a
                            floor cluster in front of the stand -> score ~0, beam
                            undisturbed, no success;
  7.  on the beam, no pan — a cube parked on the beam's SPINE (on the beam, outside
                            both pan windows) -> no cube credit, beam stays down;
  8.  wrong pan           — a cube dropped into the COUNTERWEIGHT's pan -> no credit,
                            the beam digs in deeper, no success;
  9.  near-miss underload — on a seed needing k >= 2 cubes, k-1 cubes properly banked
                            in the target pan: the beam genuinely STAYS on the
                            counterweight side (the tip threshold is real mass, not a
                            step function on 'a cube arrived'), partial credit only;
  10. mechanism: slick    — readback of the deterministic lever arms the margins are
                            computed from: the block rests at its pan's OUTER wall,
                            the banked cubes at the raised pan's INNER wall;
  11. remove-CW exploit   — the counterweight teleported out to the floor: the beam
                            now tips from the banked cubes alone, but success stays
                            False (block must be in its pan) and the tilt term is
                            GATED (no tilt credit for tipping an unloaded beam);
  12. physics restores    — a beam teleported to the cube-side stop with the block
                            aboard but NO cubes swings BACK onto the counterweight
                            side on its own (the goal tilt cannot be faked by posing);
  13. settle gate         — a statically-correct success layout written with the beam
                            spinning: judged immediately it is NOT success (the tilt
                            must be AT REST), state destroyed before it can settle;
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.pull_cube_i20.smoke --headless
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
    env = ENVS.get("simgen.beam_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    sin_t = math.sin(math.radians(c.tip_deg))

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.08)) + o),
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

    def tilt() -> float:
        return float((scene.cw_side * scene.beam_tilt())[0])

    def side() -> float:
        return float(scene.cw_side[0])

    def cw_body():
        return scene.cws[scene.cw_names[int(scene.cw_idx[0])]]

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | cw_idx={int(scene.cw_idx[0])} side={side():+.0f} "
              f"s*u={tilt():+.3f} cubes_in={int(scene.n_cubes_target()[0])} "
              f"cw_in={bool(scene.cw_in_place()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def beam_frame_pose(local, pitch: float = 0.0):
        """(pos, quat) world pose for `local` in the beam's CURRENT frame, with an
        optional extra pitch about the beam's y axis composed on the beam's yaw."""
        from isaaclab.utils.math import quat_apply, quat_mul

        q_b = scene.beam.data.root_quat_w
        pos = scene.beam.data.root_pos_w + quat_apply(q_b, torch.tensor(
            local, device=device).expand(n, 3))
        if pitch:
            q_p = torch.tensor([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0],
                               device=device).expand(n, 4)
            return pos, quat_mul(q_b, q_p)
        return pos, q_b

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def teleport_floor(body, x: float, y: float, z: float, settle_steps: int = 45) -> None:
        pos = torch.tensor([x, y, z], device=device).expand(n, 3) + scene.env_origins
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        teleport(body, pos, quat, settle_steps=settle_steps)

    def drop_cube(i: int, pan_side: float, dy: float = 0.0, settle_steps: int = 300) -> None:
        """Solve's transport move: release cube `i` at rest above the pan on
        `pan_side`, aligned with the beam; contact does the rest."""
        pos, quat = beam_frame_pose([pan_side * 0.20, dy, 0.095])
        teleport(scene.cubes[i], pos, quat, settle_steps=settle_steps)

    def settle_all(max_steps: int = 720) -> None:
        for _ in range(max_steps // 30):
            step(30)
            ok = bool(scene.beam_settled()[0])
            for b in scene.cubes:
                ok = ok and bool(scene.settled(b)[0])
            if ok:
                break

    def find_seed(start: int, want) -> int:
        """Probe seeded resets until `want(cw_idx)` holds (readback-driven)."""
        for sd in range(start, start + 12):
            env.reset(seed=sd)
            step(2)
            if want(int(scene.cw_idx[0])):
                return sd
        raise AssertionError("no seed with the wanted counterweight in 12 probes")

    def all_finite() -> bool:
        bodies = [scene.stand, scene.beam, *scene.cubes, *scene.cws.values()]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    report("reset")
    cubes_flat = all(abs(float(b.data.root_pos_w[0, 2]) - c.cube_size / 2) < 0.01
                     for b in scene.cubes)
    check("settle: all states finite and the beam RESTS fully tilted onto the "
          f"counterweight side (s*u={tilt():+.3f} ~ -0.24)",
          all_finite() and tilt() < -0.15 and bool(scene.beam_settled()[0]))
    s, ok = judge()
    check("settle: block in its pan, cubes flat on the floor, score ~0, no success",
          bool(scene.cw_in_place()[0]) and cubes_flat and s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    from isaaclab.utils.math import quat_apply  # noqa: PLC0415

    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
        env.reset(seed=sd)
        step(20)
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        ex = quat_apply(scene.stand.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        c0 = (scene.cubes[0].data.root_pos_w - scene.env_origins)[0]
        reads.append((int(scene.cw_idx[0]), side(), float(sp[0]), float(sp[1]), yaw,
                      float(c0[0]), float(c0[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (cw_idx, side, stand_x, stand_y, stand_yaw, "
          f"cube0_x, cube0_y):\n{arr.round(3)}", flush=True)
    idxs = {int(r[0]) for r in reads}
    combos = {(int(r[0]), int(r[1])) for r in reads}
    check("randomization: the counterweight choice and its side vary across seeded "
          f"resets (readback: {len(idxs)} masses, {len(combos)} mass-side combos)",
          len(idxs) >= 2 and len(combos) >= 3)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand pose (xy + yaw) and cube positions vary (readback: "
          f"dx={spread[2]:.3f} dyaw={spread[4]:.2f} dcube={spread[5]:.3f})",
          spread[2] > 0.01 and spread[4] > 0.15 and spread[5] > 0.02)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 300 idle steps", s <= 0.02 and not ok)

    # =========================== 6. seed strategy (pull to a floor spot) ====================
    # maniskill/pull_cube's move: drag the cube across the floor into a goal region.
    # There is no floor goal here: park all three cubes in a tight cluster on the
    # floor just in front of the stand -> nothing counts, the beam does not move.
    env.reset(seed=41)
    step(30)
    for i, b in enumerate(scene.cubes):
        teleport_floor(b, 0.0 + 0.07 * i, -0.42, c.cube_size / 2 + 0.002, settle_steps=20)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (pull cubes to a floor spot): score ~0, beam undisturbed "
          f"(s*u={tilt():+.3f}), no success", s <= 0.02 and tilt() < -0.15 and not ok)

    # =========================== 7. on the beam but outside both pans =======================
    env.reset(seed=51)
    settle_all(480)
    pos, quat = beam_frame_pose([-side() * 0.06, 0.0, 0.06])
    teleport(scene.cubes[0], pos, quat, settle_steps=120)
    report("on-spine")
    on_beam_z = float(scene._local(scene.cubes[0])[0, 2])
    s, ok = judge()
    check("cube parked on the beam's SPINE (on the beam, outside both pan windows, "
          f"z_loc={on_beam_z:.3f}): no cube credit, beam stays down, no success",
          s <= 0.02 and tilt() < -0.15 and int(scene.n_cubes_target()[0]) == 0 and not ok)

    # =========================== 8. wrong pan ===============================================
    env.reset(seed=61)
    settle_all(480)
    drop_cube(0, side(), dy=-0.055, settle_steps=240)
    report("wrong-pan")
    s, ok = judge()
    check("wrong pan: a cube dropped into the COUNTERWEIGHT's pan earns nothing and "
          f"the beam stays on that side (s*u={tilt():+.3f})",
          s <= 0.02 and tilt() < -0.15 and int(scene.n_cubes_target()[0]) == 0 and not ok)

    # =========================== 9-10. near-miss underload + slick readback =================
    sd = find_seed(71, lambda k: k >= 1)  # cw_idx >= 1 -> needs at least 2 cubes
    k_need = int(scene.cw_idx[0]) + 1
    settle_all(480)
    cw_x0 = float((scene.cw_side[0] * scene._local(cw_body())[0, 0]))
    for i in range(k_need - 1):
        drop_cube(i, -side(), dy=(0.0, 0.056, -0.056)[i], settle_steps=300)
    settle_all(480)
    report("underload")
    s, ok = judge()
    n_in = int(scene.n_cubes_target()[0])
    check(f"near-miss underload (seed {sd} needs {k_need} cubes): {k_need - 1} cube(s) "
          f"properly banked yet the beam STAYS on the counterweight side "
          f"(s*u={tilt():+.3f}) — partial credit only, no success",
          n_in == k_need - 1 and tilt() < -0.15 and not ok
          and s <= 0.15 * (k_need - 1) + 0.02)
    cube_x = float((-scene.cw_side[0]) * scene._local(scene.cubes[0])[0, 0])
    check("mechanism (slick pans): cargo self-locates at the deterministic walls the "
          f"margins assume — block at the OUTER wall (|x|={cw_x0:.3f} >= 0.21), banked "
          f"cube at the raised pan's INNER wall (|x|={cube_x:.3f} <= 0.19)",
          cw_x0 >= 0.21 and 0.15 <= cube_x <= 0.19)

    # =========================== 11. remove-counterweight exploit ===========================
    teleport_floor(cw_body(), 0.9, 0.6, c.cw_size[2] / 2 + 0.002, settle_steps=10)
    settle_all(720)
    report("cw-removed")
    s, ok = judge()
    check("remove-CW exploit: with the block gone the banked cubes tip the beam "
          f"(s*u={tilt():+.3f} >= {sin_t:.3f}) — but success stays False and the tilt "
          "term is gated (score = cube credit only)",
          tilt() >= sin_t and not ok and not bool(scene.cw_in_place()[0])
          and s <= 0.15 * (k_need - 1) + 0.02)

    # =========================== 12. physics restores an unloaded fake tilt =================
    env.reset(seed=81)
    settle_all(480)
    pitch = -side() * math.radians(13.0)  # tipped toward the EMPTY (cube) side
    from isaaclab.utils.math import quat_mul  # noqa: PLC0415

    q_b = scene.beam.data.root_quat_w
    beam_pos = scene.beam.data.root_pos_w.clone()
    q_p = torch.tensor([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0],
                       device=device).expand(n, 4)
    q_tipped = quat_mul(q_b, q_p)
    # place BOTH bodies from the tipped quat directly (the data buffer still holds the
    # pre-write pose until the next step — never read it back between writes)
    teleport(scene.beam, beam_pos, q_tipped, settle_steps=0)
    cw_pos = beam_pos + quat_apply(q_tipped, torch.tensor(
        [side() * 0.19, 0.0, 0.058], device=device).expand(n, 3))
    teleport(cw_body(), cw_pos, q_tipped, settle_steps=0)
    step(30)
    faked = tilt()
    settle_all(720)
    report("fake-tilt")
    s, ok = judge()
    check("physics restores: a beam POSED at the cube-side stop with the block aboard "
          f"but no cubes (posed s*u={faked:+.3f}) swings back onto the counterweight "
          f"side on its own (settled s*u={tilt():+.3f}) — the goal tilt cannot be "
          "faked", tilt() < -0.15 and not ok and s <= 0.02)

    # =========================== 13. settle gate ============================================
    sd = find_seed(91, lambda k: k == 0)  # k=1: one cube IS statically sufficient
    settle_all(480)
    # statically-correct success layout, but the beam is SPINNING when judged
    pitch = -side() * math.radians(13.0)
    q_b = scene.beam.data.root_quat_w
    beam_pos = scene.beam.data.root_pos_w.clone()
    q_p = torch.tensor([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0],
                       device=device).expand(n, 4)
    q_tipped = quat_mul(q_b, q_p)
    ax = quat_apply(q_b, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    spin = [float(v) * 3.0 for v in ax]
    teleport(scene.beam, beam_pos, q_tipped, ang=spin, settle_steps=0)
    cw_pos = beam_pos + quat_apply(q_tipped, torch.tensor(
        [side() * 0.19, 0.0, 0.058], device=device).expand(n, 3))
    teleport(cw_body(), cw_pos, q_tipped, settle_steps=0)
    cube_pos = beam_pos + quat_apply(q_tipped, torch.tensor(
        [-side() * 0.175, 0.0, 0.045], device=device).expand(n, 3))
    teleport(scene.cubes[0], cube_pos, q_tipped, settle_steps=0)
    step(2)  # refresh buffers only — judge while still spinning
    av = float(scene.beam.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    gate_ok = av > c.beam_settle_ang and not ok
    # destroy the construction BEFORE it can settle into a real success
    teleport_floor(scene.cubes[0], -0.8, 0.6, c.cube_size / 2 + 0.002, settle_steps=10)
    settle_all(720)
    report("settle-gate")
    check("settle gate: a statically-correct success layout judged with the beam "
          f"spinning ({av:.2f} rad/s) is NOT success (the tilt must be AT REST); "
          "state destroyed before it could settle", gate_ok)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_scale")
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
