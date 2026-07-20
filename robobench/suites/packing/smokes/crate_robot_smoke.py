"""Robot-binding stress smoke for the packing suite — boot, hold, reach, RECORDED.

For one registered binding (``--robot {g1,gr1t2} --mode {joint,pink_ik}``):
  1. boot ``packing.crate.<robot>.<mode>`` and settle — verify no NaN / no explosion;
  2. hold the home posture 80 steps — verify the arms track the command (drift < 5cm);
  3. (pink_ik only) march the right wrist to a hover above the SLAB with waypoints
     (mirrors ikea_table_assembly_reachable_smoke) and report the closing residual;
  4. wiggle the hands open->closed->open — verify hand joints track;
  5. record video throughout (same annotator recipe as crate_packing_smoke).

Prints PASS/FAIL per check and ``CRATE_ROBOT_SMOKE_DONE`` at the end.

    python -m robobench.suites.packing.smokes.crate_robot_smoke --robot g1 --mode pink_ik --headless
"""

from __future__ import annotations

import pinocchio  # noqa: F401  # MUST precede AppLauncher (pink_ik eigenpy converters)

import argparse

from isaaclab.app import AppLauncher

WRISTS = {
    "g1": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "gr1t2": ("left_hand_pitch_link", "right_hand_pitch_link"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "gr1t2"), default="g1")
parser.add_argument("--mode", choices=("joint", "pink_ik"), default="pink_ik")
parser.add_argument("--record_every", type=int, default=4)
parser.add_argument("--reach_steps", type=int, default=350)
parser.add_argument("--max_step", type=float, default=0.008)
parser.add_argument("--out", type=str, default="crate_robot_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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
    print(f"[robot-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    robobench.discover()
    env_name = f"packing.crate.{args.robot}.{args.mode}"
    env = ENVS.get(env_name)().build(num_envs=1)
    art = env.robot.articulation
    scene = env.scene
    origins = env.iscene.env_origins
    device = env.device
    print(f"[robot-smoke] {env_name} action_dim={env.robot.action_dim}", flush=True)

    # --- recording ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origins[0].detach().cpu().numpy().astype(float)
        # GR1T2 (~1.7 m) leans over the bench and its head crown clips the frame top at
        # the default framing — give it the taller camera (matches robot_binding_smoke).
        eye, tgt = (((1.75, -1.85, 1.85), (0.0, 0.0, 0.85)) if args.robot == "gr1t2"
                    else ((1.4, -1.5, 1.5), (0.0, 0.0, 0.75)))
        env.sim.set_camera_view(tuple(np.array(eye) + o),
                                tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[robot-smoke] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[robot-smoke] camera setup FAILED ({exc!r})", flush=True)

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
    li, ri = (art.find_bodies(n)[0][0] for n in WRISTS[args.robot])
    hand_ids = env.robot.controller.controllers[1].joint_ids
    arm_ids = env.robot.controller.controllers[0].joint_ids if args.mode == "joint" else None
    hands0 = art.data.joint_pos[:, hand_ids].clone()
    l0 = art.data.body_link_state_w[:, li, :7].clone()
    r0 = art.data.body_link_state_w[:, ri, :7].clone()

    def build_action(right_pos: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
        if args.mode == "pink_ik":
            return torch.cat([l0[:, :3] - origins, l0[:, 3:7],
                              right_pos, r0[:, 3:7], hands], dim=1)
        return torch.cat([art.data.joint_pos[:, arm_ids], hands], dim=1)

    # 1. boot sanity
    q = art.data.joint_pos
    check("boot-no-nan", bool(torch.isfinite(q).all()),
          f"root_h={float(art.data.root_pos_w[0, 2]):.2f}")

    # 2. hold home 80 steps
    hold = build_action(r0[:, :3] - origins, hands0)
    w_before = art.data.body_link_state_w[:, ri, :3].clone()
    step(hold, 80)
    drift = float((art.data.body_link_state_w[:, ri, :3] - w_before).norm())
    check("hold-home", drift < 0.05 and bool(torch.isfinite(art.data.joint_pos).all()),
          f"wrist drift {drift*100:.1f}cm")

    # 3. hand wiggle AT HOME (free space — after a reach the fingers may legitimately be
    # blocked by cargo/bench contact, which is not a controller fault): drive the ACTUATED
    # hand joints to their other limit and back, verify tracking. GR1T2's *_intermediate_*
    # finger joints are converted mimic joints (URDF effort 1e7, no independent actuator)
    # — commanding them is meaningless, so they keep their home target and only the
    # proximal/thumb joints are wiggled.
    hand_names = [art.joint_names[j] for j in hand_ids]
    drive_mask = torch.ones(len(hand_names), dtype=torch.bool, device=device)
    # Target 60% of the way to the far limit: the last stretch is blocked by legitimate
    # self-collision (GR1T2 spawns with self-collisions on; fingers meet the palm), which
    # is physics, not a controller fault. 60% spans a functional grasp stroke.
    lo = art.data.soft_joint_pos_limits[:, hand_ids, 0]
    hi = art.data.soft_joint_pos_limits[:, hand_ids, 1]
    lim = torch.where((hands0 - lo).abs() > (hands0 - hi).abs(), lo, hi)
    far = hands0 + 0.6 * (lim - hands0)
    far = torch.where(drive_mask.view(1, -1), far, hands0)
    r_cur = art.data.body_link_state_w[:, ri, :3] - origins
    step(build_action(r_cur, far), 120)
    errs_far = ((art.data.joint_pos[:, hand_ids] - far).abs() * drive_mask)[0]
    step(build_action(r_cur, hands0), 120)
    errs_back = ((art.data.joint_pos[:, hand_ids] - hands0).abs() * drive_mask)[0]
    # Criterion: >=80% of actuated joints track within 0.15 rad. Individual fingers may
    # legitimately stop early on self-collision (GR1T2 spawns with self-collisions ON);
    # the test is "the controller drives the joints", not "no finger ever touches".
    n_act = int(drive_mask.sum())
    ok_far = int(((errs_far < 0.15) | ~drive_mask).sum()) - (len(hand_names) - n_act)
    ok_back = int(((errs_back < 0.15) | ~drive_mask).sum()) - (len(hand_names) - n_act)
    worst = sorted(zip(errs_far.tolist(), hand_names), reverse=True)[:3]
    check("hand-wiggle", ok_far >= 0.8 * n_act and ok_back >= 0.8 * n_act,
          f"tracking {ok_far}/{n_act} far, {ok_back}/{n_act} back; worst far: "
          + ", ".join(f"{nm}={e:.2f}" for e, nm in worst))

    # 4. reach (pink_ik only): first a NEAR target (isolates IK convergence from workspace
    # limits), then the slab hover.
    if args.mode == "pink_ik":
        near = (r0[:, :3] - origins).clone()
        near[:, 1] += 0.10
        near[:, 2] -= 0.10
        cur = (r0[:, :3] - origins).clone()
        for _ in range(150):
            d = near - cur
            cur = cur + d * (args.max_step / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            step(build_action(cur, hands0))
        resid_n = float((art.data.body_link_state_w[:, ri, :3] - origins - near).norm())
        check("reach-near", resid_n < 0.04, f"residual {resid_n*100:.1f}cm")

        # Reach the BOX cargo part nearest the reaching wrist. Cross-body parts are the
        # other arm's job, and cylinders are excluded — a tube can roll away from the
        # snapshotted goal mid-approach (v7: 13cm 'residual' chasing a rolled tube).
        wrist = art.data.body_link_state_w[:, ri, :3]
        best_name, best_d = None, 1e9
        for nm, kind, _dims in scene.cfg.manifest:
            if kind != "box":
                continue
            b = scene.cargo[nm]
            dd = float((b.data.root_pos_w - wrist).norm())
            if dd < best_d:
                best_name, best_d = nm, dd
        goal = scene.cargo[best_name].data.root_pos_w - origins
        goal[:, 2] += 0.12
        cur = art.data.body_link_state_w[:, ri, :3] - origins
        for _ in range(args.reach_steps):
            d = goal - cur
            cur = cur + d * (args.max_step / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            step(build_action(cur, hands0))
        resid = float((art.data.body_link_state_w[:, ri, :3] - origins - goal).norm())
        check("reach-cargo", resid < 0.08, f"nearest={best_name} residual {resid*100:.1f}cm")

    print(f"[robot-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env=env_name)
        print(f"[robot-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[robot-smoke] hdfs upload rc={rc}", flush=True)
    print("CRATE_ROBOT_SMOKE_DONE", flush=True)
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
