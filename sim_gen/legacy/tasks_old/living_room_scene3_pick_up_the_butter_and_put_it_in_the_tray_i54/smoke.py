"""Smoke / oracle test for CliffCatchScene (sim_gen task
`living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i54`) — NullRobot,
teleport-oracle, RECORDED.

Battery (compass_crate / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite: butter at rest in the pen on the
                          ledge, tray upright at its parking spot, score ~0, no latches;
  2. randomization      — READBACK: fixture heading, drop-zone position, tray parking spot
                          and butter start all move across seeded resets; the tray never
                          starts anywhere near the drop zone;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success, no latches;
  4. oracle x3 seeds    — align + slide the tray onto the drop pad, push the butter off
                          the edge, free-fall catch, settle -> success() and score 1.0;
  5. monotonicity       — 0 (idle) < 0.30 (tray staged) < ~0.65 (caught, still settling)
                          < 1.0 (success); partials < 1.0;
  6. negative A (seed)  — the seed's own strategy, carry-and-place: the butter is carried
                          out kinematically and LOWERED gently into the tray — identical
                          final pose, but no free-fall crossing: caught never latches,
                          score ~0, no success (the roof also blocks this plan physically);
  7. negative B (miss)  — tray staged 130 mm off the drop zone: the butter lands on the
                          floor, permanent `dropped` latch, score capped at 0.10; dropping
                          the butter into the tray afterwards can NOT repair it;
  8. negative C (order) — push first, stage afterwards: dropped fires, perfect staging
                          later still scores <= 0.10 (execution order is enforced);
  9. calibration probe  — lateral staging-offset sweep 0/0/30/60/130 mm -> published
                          catch table; catches at <= 30 mm, clean miss at 130 mm.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i54.smoke --headless
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
    from simgen_tasks.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i54 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _axes(scene) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, l): fixture outward (drop) and lateral unit vectors, env 0, xy world."""
    fy = float(scene.fix_yaw[0])
    dev = scene.env.device
    u = torch.tensor([math.cos(fy), math.sin(fy)], device=dev)
    lat = torch.tensor([-math.sin(fy), math.cos(fy)], device=dev)
    return u, lat


def _butter_fixture_x(scene) -> float:
    """Butter center, fixture-local x (env 0) — the drop edge sits at cfg.edge_x."""
    u, _l = _axes(scene)
    p = (scene.butter.data.root_pos_w - scene.env_origins)[0, :2]
    return float(torch.dot(p - scene.fix_xy[0], u))


def _pin_path(scene, body, waypoints, step_fn, step_len: float = 0.0025,
              quat=None, final_vel=None) -> None:
    """Kinematic paced move: pin `body` through straight segments between env-local
    `waypoints` (x, y, z), one pose write + 1 physics step per `step_len` increment,
    velocities zeroed (the i30 paced-carry lesson: <= ~2.5 mm/substep). Optional final
    write sets `final_vel` (env-local, m/s) at the last waypoint, then releases."""
    dev = scene.env.device
    all_ids = torch.arange(scene.env.num_envs, device=dev)
    origin = scene.env_origins[all_ids]
    st0 = body.data.root_state_w[all_ids].clone()
    if quat is not None:
        st0[:, 3:7] = torch.tensor(quat, device=dev)
    cur = st0[:, 0:3] - origin
    for wp in waypoints:
        tgt = torch.tensor([float(v) for v in wp], device=dev).expand_as(cur)
        seg = float((tgt - cur).norm())
        k = max(1, int(math.ceil(seg / step_len)))
        for i in range(1, k + 1):
            st = st0.clone()
            st[:, 0:3] = origin + cur + (tgt - cur) * (i / k)
            st[:, 7:13] = 0.0
            body.write_root_state_to_sim(st, all_ids)
            step_fn(1)
        cur = tgt.clone()
    if final_vel is not None:
        st = st0.clone()
        st[:, 0:3] = origin + cur
        st[:, 7:13] = 0.0
        st[:, 7:10] = torch.tensor([float(v) for v in final_vel], device=dev)
        body.write_root_state_to_sim(st, all_ids)
        step_fn(1)


def _rotate_tray_to(scene, target_yaw: float, step_fn, inc_deg: float = 12.0) -> None:
    """Incremental in-place kinematic yaw of the (upright) tray to `target_yaw`."""
    all_ids = torch.arange(scene.env.num_envs, device=scene.env.device)
    st0 = scene.tray.data.root_state_w[all_ids].clone()
    yaw0 = 2.0 * math.atan2(float(st0[0, 6]), float(st0[0, 3]))
    dyaw = (target_yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
    k = max(1, int(math.ceil(abs(dyaw) / math.radians(inc_deg))))
    for i in range(1, k + 1):
        half = (yaw0 + dyaw * i / k) / 2
        st = st0.clone()
        st[:, 3] = math.cos(half)
        st[:, 4:6] = 0.0
        st[:, 6] = math.sin(half)
        st[:, 7:13] = 0.0
        scene.tray.write_root_state_to_sim(st, all_ids)
        step_fn(2)


def stage_tray(scene, step_fn, lateral_off: float = 0.0) -> None:
    """Slide the tray (kinematic, paced) from its parking spot onto the drop zone —
    yaw-aligned to the fixture first, approach from straight out front (both legs stay in
    the open front sector, clear of the fixture) — then release and settle briefly."""
    c = scene.cfg
    u, lat = _axes(scene)
    fy = float(scene.fix_yaw[0])
    dz = scene.dropzone_xy[0] + lat * lateral_off
    z = c.tray_rest_z + 0.001
    way = dz + u * 0.25
    _rotate_tray_to(scene, fy, step_fn)
    _pin_path(scene, scene.tray,
              [(float(way[0]), float(way[1]), z), (float(dz[0]), float(dz[1]), z)],
              step_fn)
    step_fn(30)


def push_butter_off(scene, step_fn, exit_speed: float = 0.15) -> None:
    """Slide the butter (kinematic, paced 1.5 mm/substep, z pinned at ledge height) along
    the pen toward the front opening until its CoM is 28 mm past the drop edge, then
    release with a small outward exit velocity — a real push-off; everything after the
    release is free physics."""
    c = scene.cfg
    u, _l = _axes(scene)
    p0 = (scene.butter.data.root_pos_w - scene.env_origins)[0]
    dx = (c.edge_x + 0.028) - _butter_fixture_x(scene)
    tgt = p0[:2] + u * max(0.0, dx)
    _pin_path(scene, scene.butter,
              [(float(tgt[0]), float(tgt[1]), float(p0[2]))],
              step_fn, step_len=0.0015,
              final_vel=(float(u[0]) * exit_speed, float(u[1]) * exit_speed, 0.0))


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle: solve the CURRENT episode — stage the tray on the drop pad, then
    push the butter over the edge and let gravity load the tray. Returns True iff
    scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    no_action = torch.empty(0, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    stage_tray(scene, _step)
    if verbose:
        print(f"[oracle] staged={bool(scene.staged[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)
    push_butter_off(scene, _step)
    waited = 0
    while waited < 400 and not bool(scene.success()[0]):
        _step(10)
        waited += 10
    if verbose:
        print(f"[oracle] caught={bool(scene.caught[0])} dropped={bool(scene.dropped[0])} "
              f"in_tray={bool(scene.in_tray()[0])} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cliff_catch")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 0.90)) + o),
                                tuple(np.array((0.0, 0.0, 0.10)) + o),
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

    def settle_until(pred, max_steps: int = 400, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        bp = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        print(f"[smoke] {tag:14s} | butter_x={_butter_fixture_x(scene):+.3f} "
              f"z={float(bp[2]):.3f} staged={bool(scene.staged[0])} "
              f"caught={bool(scene.caught[0])} dropped={bool(scene.dropped[0])} "
              f"in_tray={bool(scene.in_tray()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def carry_butter_to(scene, target_xy, drop_h: float | None = None) -> None:
        """The SEED'S plan, as a teleport control. drop_h None (butter starts in the pen):
        carry it out through the front opening at ledge height, over to `target_xy`, then
        LOWER it gently to rest. drop_h set (butter on the floor): lift it vertically to
        `drop_h`, carry over, release into free fall."""
        u, _l = _axes(scene)
        p0 = (scene.butter.data.root_pos_w - scene.env_origins)[0]
        if drop_h is None:
            z_carry = float(p0[2])
            exit_xy = scene.fix_xy[0] + u * (c.edge_x + 0.18)
            gentle_z = c.tray_rest_z + c.tray_floor_t / 2 + c.butter_size[2] / 2 + 0.004
            _pin_path(scene, scene.butter,
                      [(float(exit_xy[0]), float(exit_xy[1]), z_carry),
                       (float(target_xy[0]), float(target_xy[1]), z_carry),
                       (float(target_xy[0]), float(target_xy[1]), gentle_z)],
                      step, step_len=0.0012)
        else:
            _pin_path(scene, scene.butter,
                      [(float(p0[0]), float(p0[1]), drop_h),
                       (float(target_xy[0]), float(target_xy[1]), drop_h)],
                      step, step_len=0.0025)
        step(1)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.butter.data.root_state_w
    bz = float((scene.butter.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle: states finite, butter at rest in the pen on the ledge, tray upright "
          "on the floor",
          bool(torch.isfinite(st0).all()) and (c.ledge_h < bz < c.ledge_h + 0.05)
          and bool(scene.settled()[0]) and bool(scene.tray_upright()[0])
          and bool(scene.tray_on_floor()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0]) and not bool(scene.staged[0])
          and not bool(scene.caught[0]) and not bool(scene.dropped[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(4)
        dz = scene.dropzone_xy[0]
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0, :2]
        reads.append((float(scene.fix_yaw[0]), float(dz[0]), float(dz[1]),
                      float(tp[0]), float(tp[1]), _butter_fixture_x(scene),
                      float((tp - dz).norm())))
    arr = np.array(reads)
    print("[smoke] randomization readback (fix_yaw, dz_x, dz_y, tray_x, tray_y, "
          f"butter_x, park_dist):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture heading, drop zone, tray park and butter start all "
          "move (readback); tray never parks near the drop zone",
          spread[0] > 0.5 and (spread[1] > 0.10 or spread[2] > 0.10)
          and (spread[3] > 0.05 or spread[4] > 0.05) and spread[5] > 0.005
          and all(d >= 0.30 for d in arr[:, 6]))

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, no latches after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]) and not bool(scene.caught[0])
          and not bool(scene.dropped[0]) and not bool(scene.staged[0]))

    # =========================== 4. oracle on 3 seeds =======================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 5. rubric monotonicity =====================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    stage_tray(scene, step)
    settle_until(lambda: bool(scene.staged[0]), max_steps=100)
    s1 = sc()
    report("staged")
    push_butter_off(scene, step)
    s_mid = 0.0
    for _ in range(60):
        step(2)
        if bool(scene.success()[0]):
            break
        s_mid = max(s_mid, sc())
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    s3 = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} staged={s1:.3f} "
          f"caught={s_mid:.3f} success={s3:.3f}", flush=True)
    check("monotonicity: idle < staged < caught < success, strictly increasing",
          s0 < 0.02 and 0.28 <= s1 <= 0.32 and s_mid >= 0.60 and s3 == 1.0
          and s0 < s1 < s_mid < s3)
    check("monotonicity: partial states score < 1.0", max(s0, s1, s_mid) < 1.0)

    # =========================== 6. negative A: the seed's own strategy =====================
    # The seed picks the butter up and PLACES it in the tray. Physically the roof forbids
    # lifting it out of the pen; executed as a teleport-carry anyway, the gentle lowered
    # placement produces the identical final pose but no free-fall rim crossing — `caught`
    # never latches, so the rubric scores ~0 with no success.
    torch.manual_seed(51)
    env.reset()
    step(20)
    tray_xy = (scene.tray.data.root_pos_w - scene.env_origins)[0, :2].clone()
    carry_butter_to(scene, tray_xy)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    report("carry-place")
    check("negative A (seed strategy): butter carried + gently placed IN the tray -> "
          "in_tray but caught never latched, no success, score ~0",
          bool(scene.in_tray()[0]) and not bool(scene.caught[0])
          and not bool(scene.success()[0]) and sc() <= 0.05)

    # =========================== 7. negative B: near-miss staging ===========================
    torch.manual_seed(61)
    env.reset()
    step(20)
    stage_tray(scene, step, lateral_off=0.13)
    report("stage-offset")
    check("negative B: tray 130 mm off the drop zone does NOT latch staged",
          not bool(scene.staged[0]))
    push_butter_off(scene, step)
    settle_until(lambda: bool(scene.dropped[0]), max_steps=300)
    step(60)
    report("missed")
    check("negative B: butter misses the offset tray -> permanent dropped latch, "
          "score <= 0.10, no success",
          bool(scene.dropped[0]) and not bool(scene.success()[0]) and sc() <= 0.101)
    # permanence: dropping the (ruined) butter into the tray afterwards cannot repair it
    tray_xy = (scene.tray.data.root_pos_w - scene.env_origins)[0, :2].clone()
    carry_butter_to(scene, tray_xy, drop_h=0.30)
    step(240)
    report("post-drop")
    check("negative B: latch is permanent — a later real drop into the tray still "
          "scores <= 0.10, no success",
          bool(scene.dropped[0]) and not bool(scene.caught[0])
          and not bool(scene.success()[0]) and sc() <= 0.101)

    # =========================== 8. negative C: order violation =============================
    torch.manual_seed(71)
    env.reset()
    step(20)
    push_butter_off(scene, step)
    settle_until(lambda: bool(scene.dropped[0]), max_steps=300)
    report("pushed-first")
    ok_dropped = bool(scene.dropped[0])
    # move the ruined butter clear of the drop zone (it landed on the pad; staging would
    # otherwise plow it into the cliff gap — the rubric outcome is identical either way)
    u, lat = _axes(scene)
    aside = scene.dropzone_xy[0] + lat * 0.30
    st_b = scene.butter.data.root_state_w[all_ids].clone()
    st_b[:, 0] = aside[0] + scene.env_origins[0, 0]
    st_b[:, 1] = aside[1] + scene.env_origins[0, 1]
    st_b[:, 2] = scene.env_origins[0, 2] + c.butter_size[2] / 2 + 0.002
    st_b[:, 3] = 1.0
    st_b[:, 4:7] = 0.0
    st_b[:, 7:13] = 0.0
    scene.butter.write_root_state_to_sim(st_b, all_ids)
    step(20)
    stage_tray(scene, step)
    settle_until(lambda: bool(scene.staged[0]), max_steps=100)
    report("staged-late")
    check("negative C: butter pushed BEFORE staging is lost for good — perfect staging "
          "afterwards stays capped <= 0.10, no success",
          ok_dropped and bool(scene.staged[0]) and not bool(scene.success()[0])
          and sc() <= 0.101)

    # =========================== 9. calibration probe =======================================
    print("[smoke] CALIBRATION: lateral staging offset -> catch outcome "
          "(fresh reset each; landing = fixture-local x past the edge)", flush=True)
    cal = []
    for seed, off in ((91, 0.0), (92, 0.0), (93, 0.030), (94, 0.060), (95, 0.130)):
        torch.manual_seed(seed)
        env.reset()
        step(20)
        stage_tray(scene, step, lateral_off=off)
        push_butter_off(scene, step)
        settle_until(lambda: bool(scene.success()[0]) or bool(scene.dropped[0]),
                     max_steps=300)
        step(60)
        land = _butter_fixture_x(scene) - c.edge_x
        cal.append((off, bool(scene.success()[0]), bool(scene.caught[0]),
                    bool(scene.dropped[0]), sc()))
        print(f"[smoke]   off={off * 1000:5.0f}mm landed_out={land * 1000:6.1f}mm "
              f"caught={cal[-1][2]} dropped={cal[-1][3]} success={cal[-1][1]} "
              f"score={cal[-1][4]:.3f}", flush=True)
    check("calibration: catches at lateral offsets <= 30 mm (3/3 success)",
          all(su for off, su, _c, _d, _s in cal if off <= 0.031))
    check("calibration: clean miss at 130 mm — dropped, no success, capped",
          all((not su) and dr and s <= 0.101 for off, su, _c, dr, s in cal if off >= 0.12))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cliff_catch")
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
