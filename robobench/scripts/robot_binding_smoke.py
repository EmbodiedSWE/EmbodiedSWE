"""Generic robot-binding stress smoke — boot, hold, wiggle, reach, RECORDED.

Verifies ANY registered `suite.scene.robot.mode` binding (the per-embodiment
counterpart of each suite's NullRobot smoke), one binding per run:

  1. boot the env and settle — no NaN, no explosion;
  2. hold the home posture 80 steps — end-effector drift < 5 cm;
  3. wiggle the hand/gripper (composite controller's LAST sub-controller) toward the
     far joint limits and back — >=80% of driven joints track within 0.15 rad;
  4. reach: march the end-effector to a hover above `--reach_body` (an iscene key,
     e.g. 'barrel', 'cup') — residual < 8 cm.
     Supported control modes: `pink_ik` (humanoids: absolute wrist poses) and `osc`
     (Franka: end-effector pose deltas). `joint` bindings skip the reach.

    python -m robobench.scripts.robot_binding_smoke --env puzzle.syringe.franka.osc \
        --reach_body barrel --headless
"""

from __future__ import annotations

import pinocchio  # noqa: F401  # MUST precede AppLauncher (pink_ik eigenpy converters)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True,
                    help="registered env name, e.g. puzzle.syringe.franka.osc")
parser.add_argument("--reach_body", type=str, default="",
                    help="iscene key to reach a hover above ('' = skip the reach)")
parser.add_argument("--via", type=str, default="",
                    help="optional waypoint offset dx,dy,dz from the body, visited BEFORE "
                         "the approach point (OSC has no collision avoidance: a straight "
                         "line from home to the safe dial beaches the hand on the box top)")
parser.add_argument("--hover", type=str, default="0,0,0.10",
                    help="approach offset dx,dy,dz from the body (m); e.g. the safe dial "
                         "needs 0,-0.12,0 — a hover ABOVE it is inside the door")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--reach_steps", type=int, default=350)
parser.add_argument("--max_step", type=float, default=0.008)
parser.add_argument("--out", type=str, default="robot_binding_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[binding-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    robobench.discover()
    parts = args.env.split(".")
    robot_name, mode = parts[2], parts[3]
    env = ENVS.get(args.env)().build(num_envs=1, seed=args.seed)
    art = env.robot.articulation
    # End-effector body from the robot class: EE_BODIES = (left, right) on the
    # humanoids (take the right), EE_BODY on the single-arm grippers.
    ee_body = (getattr(env.robot, "EE_BODIES", None) or (env.robot.EE_BODY,))[-1]
    origins = env.iscene.env_origins
    device = env.device
    print(f"[binding-smoke] {args.env} action_dim={env.robot.action_dim}", flush=True)

    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origins[0].detach().cpu().numpy().astype(float)
        # Per-robot framing: the original single camera (eye z=1.2 -> tgt z=0.5)
        # crops the head off the ~1.75 m humanoids (user flagged the GR1T2 videos).
        # Humanoids get the crate_robot_smoke framing, pulled back a touch more.
        if robot_name == "franka":
            eye, tgt = (1.3, -1.4, 1.2), (0.0, 0.0, 0.5)
        elif robot_name == "gr1t2":
            # tallest robot (~1.7 m standing upright): needs the highest camera or the
            # head crown clips the frame top (still did at eye z=1.55).
            eye, tgt = (1.75, -1.85, 1.85), (0.0, 0.0, 0.85)
        else:
            eye, tgt = (1.55, -1.65, 1.55), (0.0, 0.0, 0.75)
        env.sim.set_camera_view(tuple(np.array(eye) + o),
                                tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[binding-smoke] camera ready shape={np.asarray(annot.get_data()).shape}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[binding-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0

    def step(action: torch.Tensor, k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(action, render=True)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    env.reset()
    ee_i = art.find_bodies(ee_body)[0][0]
    grip_ctrl = env.robot.controller.controllers[-1]
    grip_ids = grip_ctrl.joint_ids
    grip0 = art.data.joint_pos[:, grip_ids].clone()
    ee0 = art.data.body_link_state_w[:, ee_i, :7].clone()
    if getattr(env.robot, "EE_BODIES", None):  # humanoids only (pink_ik uses the LEFT wrist pose);
        # single-arm grippers carry EE_BODY alone
        li = art.find_bodies(env.robot.EE_BODIES[0])[0][0]
        l0 = art.data.body_link_state_w[:, li, :7].clone()
    arm_ids = (env.robot.controller.controllers[0].joint_ids
               if mode == "joint" else None)

    def build_action(ee_pos_delta_or_abs: torch.Tensor, grips: torch.Tensor) -> torch.Tensor:
        """pink_ik: absolute wrist poses (L pose7 + R pose7 + hands). osc: 6 EE deltas
        + grips. joint: current arm positions + grips (hold)."""
        if mode == "pink_ik":
            return torch.cat([l0[:, :3] - origins, l0[:, 3:7],
                              ee_pos_delta_or_abs, ee0[:, 3:7], grips], dim=1)
        if mode == "osc":
            return torch.cat([ee_pos_delta_or_abs,
                              torch.zeros(1, 3, device=device), grips], dim=1)
        return torch.cat([art.data.joint_pos[:, arm_ids], grips], dim=1)

    def hold_action() -> torch.Tensor:
        if mode == "pink_ik":
            return build_action(ee0[:, :3] - origins, grip0)
        if mode == "osc":
            return build_action(torch.zeros(1, 3, device=device), grip0)
        return build_action(None, grip0)

    # 1. boot
    q = art.data.joint_pos
    check("boot-no-nan", bool(torch.isfinite(q).all()),
          f"root_h={float(art.data.root_pos_w[0, 2]):.2f}")

    # 2. hold home
    before = art.data.body_link_state_w[:, ee_i, :3].clone()
    step(hold_action(), 80)
    drift = float((art.data.body_link_state_w[:, ee_i, :3] - before).norm())
    check("hold-home", drift < 0.05 and bool(torch.isfinite(art.data.joint_pos).all()),
          f"ee drift {drift * 100:.1f}cm")

    # 3. hand/gripper wiggle at home. GR1T2's *_intermediate_* / thumb_distal joints
    # are converted MIMIC joints (no independent actuator) — driving them is
    # meaningless, so they are masked out (same as the crate robot smoke; my first
    # generic version dropped the mask and "failed" every GR1T2 binding on joints
    # that cannot be commanded).
    names = [art.joint_names[j] for j in grip_ids]
    print(f"[binding-smoke] gripper joints ({len(names)}): {names}", flush=True)
    drive_mask = torch.tensor(
        [not ("intermediate" in nm or "thumb_distal" in nm) for nm in names],
        dtype=torch.bool, device=device)
    lo = art.data.soft_joint_pos_limits[:, grip_ids, 0]
    hi = art.data.soft_joint_pos_limits[:, grip_ids, 1]
    lim = torch.where((grip0 - lo).abs() > (grip0 - hi).abs(), lo, hi)
    far = grip0 + 0.6 * (lim - grip0)
    far = torch.where(drive_mask.view(1, -1), far, grip0)

    def grip_step(target: torch.Tensor) -> None:
        if mode == "pink_ik":
            cur = art.data.body_link_state_w[:, ee_i, :3] - origins
            step(build_action(cur, target), 120)
        elif mode == "osc":
            step(build_action(torch.zeros(1, 3, device=device), target), 120)
        else:
            step(build_action(None, target), 120)

    start = art.data.joint_pos[:, grip_ids].clone()
    grip_step(far)
    at_far = art.data.joint_pos[:, grip_ids].clone()
    grip_step(grip0)
    at_back = art.data.joint_pos[:, grip_ids].clone()
    # Criterion: ACTUATION, not convergence — commanded travel must be substantially
    # followed on average (>=50% toward far, >=70% of the way back home). Absolute
    # convergence over-fails dexterous hands whose fingers legitimately stop on
    # self-collision partway (GR1T2 middle/ring/pinky at the 60% close target).
    cmd = (far - start).abs().clamp_min(1e-6)
    frac_far = (((at_far - start).abs() / cmd).clamp(0, 1) * drive_mask)[0]
    frac_back = ((1.0 - (at_back - grip0).abs() / cmd).clamp(0, 1) * drive_mask)[0]
    n_act = int(drive_mask.sum())
    mean_far = float(frac_far.sum() / n_act)
    mean_back = float(frac_back.sum() / n_act)
    worst = sorted(zip(frac_far.tolist(), names))[:3]
    check("hand-wiggle", mean_far >= 0.5 and mean_back >= 0.7,
          f"mean travel-followed: far {mean_far:.2f}, back {mean_back:.2f} "
          f"(n_act={n_act}); least far: "
          + ", ".join(f"{nm}={e:.2f}" for e, nm in worst))

    # 4. reach an approach point offset from the task body, optionally through a
    # --via waypoint (OSC has no collision avoidance; a straight line from home to
    # the safe dial beaches the hand on the box top). VIRTUAL CARRIER target (the
    # crate smoke's method): marching from the LIVE wrist throttles progress by the
    # tracking lag every step and stalls short (first generic version).
    if args.reach_body and mode in ("pink_ik", "osc"):
        body_pos = env.iscene[args.reach_body].data.root_pos_w.clone() - origins
        off = torch.tensor([[float(v) for v in args.hover.split(",")]], device=device)
        goals = []
        if args.via:
            voff = torch.tensor([[float(v) for v in args.via.split(",")]], device=device)
            goals.append(body_pos + voff)
        goals.append(body_pos + off)
        cur = (art.data.body_link_state_w[:, ee_i, :3] - origins).clone()
        print(f"[binding-smoke] reach start: ee={[round(float(v), 3) for v in cur[0]]} "
              f"goals={[[round(float(v), 3) for v in g[0]] for g in goals]}", flush=True)
        it_tot = 0
        for gi, goal in enumerate(goals):
            for _it in range(args.reach_steps):
                it_tot += 1
                d = goal - cur
                if float(d.norm()) < 0.002 and gi < len(goals) - 1:
                    break  # waypoint reached
                cur = cur + d * (args.max_step
                                 / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
                if mode == "pink_ik":
                    step(build_action(cur, grip0))
                else:
                    # OSC actions are in ACTION UNITS (target latch = ee +
                    # action*pos_scale, pos_scale=0.02 m): sending METERS as the first
                    # versions did latches targets ~1 mm out and the reach crawls.
                    # pos_scale comes from the OSC controller cfg, so gain retunes
                    # show up in this trajectory instead of a stale hardcode.
                    arm_cfg = env.robot.controller.controllers[0].cfg
                    pos_scale = float(getattr(arm_cfg, "pos_scale", 0.02))
                    live = art.data.body_link_state_w[:, ee_i, :3] - origins
                    act = ((cur - live) / pos_scale).clamp(-1.0, 1.0)
                    step(build_action(act, grip0))
                if it_tot % 80 == 0:
                    live = art.data.body_link_state_w[:, ee_i, :3] - origins
                    print(f"[binding-smoke]   reach t={it_tot}: ee="
                          f"{[round(float(v), 3) for v in live[0]]} carrier="
                          f"{[round(float(v), 3) for v in cur[0]]} goal_i={gi}", flush=True)
        goal = goals[-1]
        live = art.data.body_link_state_w[:, ee_i, :3] - origins
        resid = float((live - goal).norm())
        dax = (live - goal)[0]
        if robot_name == "franka":
            arm_ids7 = env.robot.controller.controllers[0].joint_ids
            jp = art.data.joint_pos[0, arm_ids7]
            jlo = art.data.soft_joint_pos_limits[0, arm_ids7, 0]
            jhi = art.data.soft_joint_pos_limits[0, arm_ids7, 1]
            at_lim = [(art.joint_names[j], round(float(q), 2))
                      for j, q, lo, hi in zip(arm_ids7, jp, jlo, jhi)
                      if float(q - lo) < 0.05 or float(hi - q) < 0.05]
            print(f"[binding-smoke]   arm joints at limits: {at_lim}", flush=True)
        check("reach-body", resid < 0.08,
              f"body={args.reach_body} residual {resid * 100:.1f}cm "
              f"(dx,dy,dz)=({float(dax[0]):+.3f},{float(dax[1]):+.3f},{float(dax[2]):+.3f})")
    else:
        print("[binding-smoke] reach skipped (joint mode or no --reach_body)", flush=True)

    print(f"[binding-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env=args.env)
        print(f"[binding-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[binding-smoke] hdfs upload rc={rc}", flush=True)
    print("ROBOT_BINDING_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()
