"""Shoe_tying smoke — the bimanual ALOHA (2x WidowX 250 6DOF) LIFTS both shoelaces, RTX-renderable.

This is a capability check, not a knot attempt: on the registered env
`deformable.knot.aloha.joint` (bimanual WidowX 250 + Newton rods on the suite's proxy-coupled
MJWarp+VBD substrate), each arm lifts its lace's free end off the table with its real fingers
(pure contact through the gripper proxy bodies — no weld, no kinematic handles:
`kinematic_ends=False` leaves the ends dynamic) and holds it. A PASS demonstrates the
substrate: both arms reach the lace workspace, finger-lace contact carries load, and the
coupled rod dynamics stay stable through contact.

Verdict: at the END of the hold each lace's end section (last 6 bodies) must still be
>= 60 mm above the table top (the pinch genuinely carried the lace up and kept it there),
the laces must stay intact (no cable joint torn past 2.5x the rest segment length), and the
rod state must stay finite.

Runs ONLY under the Newton venv (newton >= 1.6 @ f4209981 — see the README):
  env_newton/bin/python -m robobench.suites.deformable.smokes.aloha_lift_smoke --headless

RTX video (visual USD shoe + robot + per-segment lace capsules synced from body_q):
  env_newton/bin/python scripts/record_video.py \\
      robobench.suites.deformable.smokes.aloha_lift_smoke \\
      --video robobench/suites/deformable/videos/aloha_lift_smoke.mp4 \\
      --eye 0.92 -0.86 0.78 --target-at 0.0 0.02 0.08
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="the demo driver is single-env; keep 1")
parser.add_argument("--lift", type=float, default=0.12, help="fingertip lift height above the grasp [m]")
parser.add_argument("--hold", type=float, default=1.5, help="hold time at the top of the lift [s]")
parser.add_argument("--print_every", type=int, default=60, help="progress print period [frames]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total frames (debugging)")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    # Recording path (scripts/record_video.py): rendering on develop is pumped by visualizers,
    # and an explicit --headless force-disables them — make headless+kit-visualizer the defaults.
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
render_on = (args.livestream > 0) or bool(args.visualizer and "none" not in args.visualizer)

app = AppLauncher(args).app

# Disable the cubric GPU transform hierarchy (renders moving prims frozen on this stack);
# the CPU update_world_xforms() fallback renders correctly.
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.deformable.scenes.shoe_knot import LaceVisuals  # noqa: E402

FPS = 60  # must match RodSimCfg.dt
MAX_DQ = 0.04  # per-tick joint reference step clamp [rad]
LEAD_MAX = 0.30  # max lead of the commanded reference over the ACTUAL joints [rad]
# WX250s finger targets [m], LEFT-finger coordinate (the right mirrors NEGATED, see the robot
# module).
OPEN, CLOSE = 0.037, 0.0101
PRECLOSE = 0.0216  # intermediate finger target [m] commanded before CLOSE
# Fingertip point in the gripper_link frame (the fingertip contact spheres sit at x=0.108 in
# the converted USD).
TIP_OFFSET = (0.108, 0.0, 0.0)
# Ry(90 deg): gripper_link +x (the finger axis) points straight down; a world-z yaw is
# composed on top.
Q_DOWN = (0.0, 0.7071068, 0.0, 0.7071068)  # xyzw
END_SECTION = 6  # bodies of the lace tail whose height carries the verdict
GRASP_WINDOW = (-5, -1)  # candidate tail bodies for the target
MOUTH_DEPTH = 0.008  # pad-tip depth below the target rod center [m] (clamped above the table)


class Arm:
    """DiffIK servo toward a world-frame gripper_link pose; returns the 8-wide joint action
    (6 arm position targets + 2 finger targets, right = negated left)."""

    ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

    def __init__(self, robot, device: str) -> None:
        art = robot.articulation
        self.art = art
        self.device = device
        self.arm_ids = art.find_joints(self.ARM_JOINTS, preserve_order=True)[0]
        self.hand_idx = art.find_bodies([".*gripper_link.*"])[0][0]
        num_base = getattr(art, "num_base_dofs", 0)
        self.jacobi_body = self.hand_idx - 1 if art.is_fixed_base else self.hand_idx
        self.jacobi_joints = [j + num_base for j in self.arm_ids]
        self.grip = OPEN
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.05}
            ),
            num_envs=1,
            device=device,
        )
        self.limits = art.data.joint_pos_limits.torch[:, self.arm_ids, :]
        self.q_ref: torch.Tensor | None = None
        self.target_p: torch.Tensor | None = None
        self.target_q: torch.Tensor | None = None

    def hand_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.art.data.body_pos_w.torch[:, self.hand_idx], self.art.data.body_quat_w.torch[:, self.hand_idx]

    def tip_pos_w(self) -> torch.Tensor:
        hp, hq = self.hand_pose_w()
        return hp + math_utils.quat_apply(hq, torch.tensor([TIP_OFFSET], device=self.device))

    def hand_for_tip(self, tip_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        """gripper_link position putting the fingertip point at `tip_w` under `quat_w`."""
        return tip_w - math_utils.quat_apply(quat_w, torch.tensor([TIP_OFFSET], device=self.device))

    def solve(self) -> torch.Tensor:
        tp, tq = math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, self.target_p, self.target_q
        )
        ee_p, ee_q = math_utils.subtract_frame_transforms(
            self.art.data.root_pos_w.torch, self.art.data.root_quat_w.torch, *self.hand_pose_w()
        )
        self.ik.set_command(torch.cat([tp, tq], dim=-1), ee_p, ee_q)
        J = self.art.data.body_link_jacobian_w.torch[:, self.jacobi_body, :, self.jacobi_joints]
        R = math_utils.matrix_from_quat(math_utils.quat_inv(self.art.data.root_quat_w.torch))
        J[:, :3, :] = torch.bmm(R, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
        q_arm = self.art.data.joint_pos.torch[:, self.arm_ids]
        q_des = self.ik.compute(ee_p, ee_q, J, q_arm)
        if self.q_ref is None:
            self.q_ref = q_arm.clone()
        q_ref = (self.q_ref + (q_des - self.q_ref).clamp(-MAX_DQ, MAX_DQ)).clamp(
            self.limits[..., 0], self.limits[..., 1]
        )
        q_ref = q_ref.clamp(q_arm - LEAD_MAX, q_arm + LEAD_MAX)
        self.q_ref = q_ref
        # both fingers: the right finger's coordinate is the mirror of the left's
        grip = torch.tensor([[self.grip, -self.grip]], device=self.device)
        return torch.cat([q_ref, grip], dim=-1)


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class _Done(Exception):
    pass


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("deformable.knot.aloha.joint")().build(num_envs=1, device=device)
    scene = env.scene
    env.reset()

    R_env, t_env = scene.world_frames[0]  # task frame: env origin lifted to the table top

    def local(p: np.ndarray) -> np.ndarray:  # world point(s) -> task-local (z=0 = table top)
        return (p - t_env) @ R_env

    arms = {name: Arm(env.robot[name], device) for name in ("left", "right")}
    slices = env.robot.action_slices
    seg_len = scene.seg_len
    print(f"[aloha-lift] laces: {[len(b) for b in scene.lace_bodies_w[0]]} bodies | "
          f"action dim {env.robot.action_dim} | slices {slices}", flush=True)

    visuals = LaceVisuals(env.stage, scene) if render_on else None

    tick = 0
    action = torch.zeros((1, env.robot.action_dim), device=device)

    def lace_pts(i: int) -> np.ndarray:
        return scene.lace_points(i, 0)

    def end_height(i: int) -> float:
        """Highest task-local z of the lace's end section [m above the table top]."""
        return float(local(lace_pts(i))[-END_SECTION:, 2].max())

    def check_state() -> None:
        q = scene._nm._state_0.body_q.numpy()
        qd = scene._nm._state_0.body_qd.numpy()
        assert np.isfinite(q).all(), "non-finite body_q"
        assert np.isfinite(qd).all(), "non-finite body_qd"
        assert np.abs(qd).max() < 1.0e3, f"velocity blow-up: {np.abs(qd).max():.1f}"

    def proxy_contacts() -> int:
        """Gripper-lace contact count from the proxy coupling's own collision pipeline."""
        c = scene._nm._solver.get_proxy_contacts("mjc", "vbd")
        return int(c.rigid_contact_count.numpy()[0]) if c is not None else -1

    def step() -> None:
        nonlocal tick
        for name, arm in arms.items():
            action[:, slices[name]] = arm.solve()
        if visuals is not None:
            visuals.sync()
        env.step(action, render=render_on)
        tick += 1
        if tick % args.print_every == 0:
            check_state()
            hs = [end_height(i) for i in (0, 1)]
            errs = [float((arm.hand_pose_w()[0] - arm.target_p).norm()) for arm in arms.values()]
            print(
                f"  step {tick:5d} t={tick / FPS:5.1f}s | end z {hs[0] * 1e3:+5.0f}/{hs[1] * 1e3:+5.0f} mm"
                f" | hand err {errs[0] * 1e3:4.0f}/{errs[1] * 1e3:4.0f} mm | proxy contacts {proxy_contacts()}",
                flush=True,
            )
        if args.max_steps is not None and tick >= args.max_steps:
            raise _Done

    def glide(targets: dict[str, tuple[torch.Tensor, torch.Tensor]], dur: float) -> None:
        """Simultaneously glide each arm's (pos, quat) target; quats are held, not slerped."""
        start = {n: arms[n].target_p.clone() for n in targets}
        steps = max(1, int(round(dur * FPS)))
        for k in range(steps):
            a = _smoothstep((k + 1) / steps)
            for n, (p1, q1) in targets.items():
                arms[n].target_p = start[n] + (p1 - start[n]) * a
                arms[n].target_q = q1
            step()

    def dwell(dur: float) -> None:
        for _ in range(int(round(dur * FPS))):
            step()

    z_end = {0: 0.0, 1: 0.0}
    z_peak = {0: 0.0, 1: 0.0}
    try:
        # park the arms where they spawned and let the freed lace tails settle onto the table
        for name, arm in arms.items():
            p0, q0 = arm.hand_pose_w()
            arm.target_p, arm.target_q = p0.clone(), q0.clone()
        dwell(2.0)
        z_rest = [end_height(i) for i in (0, 1)]

        # plan each arm's target from its lace's settled end (lace i -> arm: 0/left, 1/right)
        plan: dict[str, dict] = {}
        for i, name in ((0, "left"), (1, "right")):
            pts = lace_pts(i)
            lo, hi = GRASP_WINDOW
            k = int(np.argmin(pts[lo:hi, 2])) + (len(pts) + lo)  # target tail body
            grasp_w = pts[k].copy()
            tang = pts[min(k + 2, len(pts) - 1)] - pts[max(k - 2, 0)]
            yaw = float(np.arctan2(tang[1], tang[0]))
            q_yaw = math_utils.quat_from_angle_axis(
                torch.tensor([yaw], device=device), torch.tensor([[0.0, 0.0, 1.0]], device=device)
            )
            q_grasp = math_utils.quat_mul(q_yaw, torch.tensor([Q_DOWN], device=device))
            # pad tips MOUTH_DEPTH below the rod center (floor: 1.5 mm above the table)
            tip_z = max(float(grasp_w[2]) - MOUTH_DEPTH, float(t_env[2]) + 0.0015)
            tip_grasp = torch.tensor(
                [[float(grasp_w[0]), float(grasp_w[1]), tip_z]],
                dtype=torch.float32, device=device,
            )
            plan[name] = {"lace": i, "q": q_grasp, "grasp": tip_grasp, "body": k,
                          "hover": tip_grasp + torch.tensor([[0.0, 0.0, 0.10]], device=device),
                          "lift": tip_grasp + torch.tensor([[0.0, 0.0, args.lift]], device=device)}
            g_l = local(grasp_w)
            print(f"  [plan] {name} arm -> lace {i + 1} body {k - len(pts)} at "
                  f"({g_l[0]:+.3f}, {g_l[1]:+.3f}, {g_l[2] * 1e3:.0f} mm) local, "
                  f"yaw {np.degrees(yaw):+.0f} deg | settled end z {z_rest[i] * 1e3:.0f} mm", flush=True)

        # reach over the lace ends, both arms together
        glide({n: (arms[n].hand_for_tip(plan[n]["hover"], plan[n]["q"]), plan[n]["q"]) for n in arms}, 3.0)
        # settle the wrist at hover; if the planned yaw branch did not converge, use the
        # mirrored one
        dwell(1.0)
        z180 = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)  # xyzw: yaw 180 deg
        for name, arm in arms.items():
            err = float(math_utils.quat_error_magnitude(arm.hand_pose_w()[1], arm.target_q))
            if err > np.radians(60.0):
                plan[name]["q"] = math_utils.quat_mul(z180, plan[name]["q"])
                print(f"  [orient] {name}: near yaw branch unreachable (wrist err {np.degrees(err):.0f} deg), "
                      f"flipping 180", flush=True)
        glide({n: (arms[n].hand_for_tip(plan[n]["hover"], plan[n]["q"]), plan[n]["q"]) for n in arms}, 2.0)
        for name, arm in arms.items():
            err = float(math_utils.quat_error_magnitude(arm.hand_pose_w()[1], arm.target_q))
            print(f"  [orient] {name}: settled at hover, wrist err {np.degrees(err):.0f} deg", flush=True)
        # descend to 15 mm above the target
        mid = {n: plan[n]["grasp"] + torch.tensor([[0.0, 0.0, 0.015]], device=device) for n in arms}
        glide({n: (arms[n].hand_for_tip(mid[n], plan[n]["q"]), plan[n]["q"]) for n in arms}, 1.5)
        # re-target on the live position of the chosen body
        for name, arm in arms.items():
            g = lace_pts(plan[name]["lace"])[plan[name]["body"]]
            tip = arm.tip_pos_w()[0].cpu().numpy()
            tip_z = max(float(g[2]) - MOUTH_DEPTH, float(t_env[2]) + 0.0015)
            for key, z in (("grasp", tip_z), ("lift", tip_z + args.lift)):
                t_new = plan[name][key].clone()
                t_new[0, 0], t_new[0, 1], t_new[0, 2] = float(g[0]), float(g[1]), z
                plan[name][key] = t_new
            print(f"  [recenter] {name}: pinch body at {np.round(local(g), 3)} "
                  f"({np.linalg.norm(g[:2] - tip[:2]) * 1e3:.1f} mm off)", flush=True)
        glide({n: (arms[n].hand_for_tip(plan[n]["grasp"], plan[n]["q"]), plan[n]["q"]) for n in arms}, 1.5)
        # close in two stages
        for arm in arms.values():
            arm.grip = PRECLOSE
        dwell(0.6)
        for arm in arms.values():
            arm.grip = CLOSE
        dwell(1.5)
        for name, arm in arms.items():
            tip = arm.tip_pos_w()[0].cpu().numpy()
            pts = lace_pts(plan[name]["lace"])
            d = np.linalg.norm(pts - tip, axis=1).min()
            fq = arm.art.data.joint_pos.torch[0, arm.art.find_joints([".*finger.*"])[0]]
            print(f"  [close] {name}: tip local {np.round(local(tip), 3)} | nearest lace pt "
                  f"{d * 1e3:.1f} mm | fingers q {[round(float(v), 4) for v in fq]} | "
                  f"proxy contacts {proxy_contacts()}", flush=True)
        # lift straight up and hold
        glide({n: (arms[n].hand_for_tip(plan[n]["lift"], plan[n]["q"]), plan[n]["q"]) for n in arms}, 4.0)
        for _ in range(int(round(args.hold * FPS))):
            step()
            for i in (0, 1):
                z_peak[i] = max(z_peak[i], end_height(i))
        for i in (0, 1):
            z_end[i] = end_height(i)  # measured at the END of the hold
        print(f"  [hold] end z at hold end {z_end[0] * 1e3:.0f}/{z_end[1] * 1e3:.0f} mm "
              f"(peaks {z_peak[0] * 1e3:.0f}/{z_peak[1] * 1e3:.0f})", flush=True)
    except _Done:
        print(f"  [cap] stopped at step {tick}", flush=True)

    # ----- verdict ----------------------------------------------------------------------------
    check_state()
    gaps_ok = True
    for i in (0, 1):
        pts = lace_pts(i)
        gap = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).max())
        gaps_ok &= gap <= 2.5 * seg_len
        if gap > 2.5 * seg_len:
            print(f"  lace {i + 1} torn: max joint gap {gap * 1e3:.1f} mm (rest {seg_len * 1e3:.1f})", flush=True)
    lifted = all(z_end[i] >= 0.06 for i in (0, 1))
    ok = lifted and gaps_ok
    print(
        f"ALOHA-LIFT {'PASS' if ok else 'FAIL'} | lace end heights at hold end "
        f"{z_end[0] * 1e3:+.0f}/{z_end[1] * 1e3:+.0f} mm (gate >= 60 both) | intact={gaps_ok}",
        flush=True,
    )
    env.close()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    app.close()
