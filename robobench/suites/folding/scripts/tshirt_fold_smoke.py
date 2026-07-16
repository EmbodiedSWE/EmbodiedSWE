"""Fold smoke for the folding suite — scene + Franka driving a scripted key-pose fold.

The Franka folds the T-shirt in three moves (left sleeve to the center line, right sleeve to the
center line, bottom hem up to the collar), then tucks the crease and pats the bundle flat,
following the timed key-pose table `KEY_POSES`. Two adaptive elements on top of the fixed table:

  - **Adaptive grasp anchoring**: each grasp/tuck/pat block re-anchors its approach rows on the
    MEASURED cloth extremes at approach time (`retarget_grasp`) — the cloth's settle position
    shifts with material params and with each completed fold, so fixed coordinates pinch air.
    Fold DESTINATIONS stay scripted.
  - **Tilted near-base approach** (`QR_NEAR`) for the right sleeve: a straight-down pinch 0.19 m
    from the base axis stalls ~14 cm short of the target on joint limits.

Actuation:

  - The Cartesian target is low-pass filtered with a ~1 s time constant (``--tau``) and tracked
    by absolute-pose **damped-least-squares differential IK** (isaaclab's
    `DifferentialIKController`, the same method the in-tree Newton cloth task uses), whose
    joint-position output feeds the robot's "joint" control mode.
  - Frames: `KEY_POSES` rows are authored as cm world positions and mapped to robot-root-frame
    `panda_hand` poses: positions ``(x,y,z)/100 + (0.5, 0.5, 0)`` (base at (-0.5,-0.5,0)),
    fingertip offset ``(0, 0, 0.113)`` in the hand frame. NOTE: isaaclab develop uses **xyzw**
    quaternions throughout (warp convention) — data layer, math utils, and DiffIK commands alike.
  - Gripper: finger targets open 0.032 m, close 0.004 m (pinch gap).

Verdict: cloth footprint < 0.30 m² (a completed 3-fold run lands ~0.17; failure modes stay >=
0.41; settled unfolded ~0.50), particles in bounds, and the first hover pose reached within
~2 cm (validates the frame/quat conventions).

Runs ONLY under the Newton venv:
  env_newton/bin/python -m robobench.suites.folding.scripts.tshirt_fold_smoke --headless
  env_newton/bin/python -m robobench.suites.folding.scripts.tshirt_fold_smoke --livestream 2
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--tau", type=float, default=0.3, help="target low-pass time constant [s]; the DLS+servo stack adds its own ~1-2 s lag, so keep this small")
parser.add_argument("--time_scale", type=float, default=1.0, help="multiply every key-pose duration (slower, more settled folds)")
parser.add_argument("--lam", type=float, default=0.05, help="DLS damping lambda (small: workspace-edge poses need it)")
parser.add_argument("--k_null", type=float, default=1.0, help="nullspace pull gain toward home posture [1/s]")
parser.add_argument("--hold", type=float, default=3.0, help="extra settle time after the last key pose [s]")
parser.add_argument("--print_every", type=int, default=120, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--debug", action="store_true", help="verbose frame/IK prints")
parser.add_argument("--density", type=float, default=None, help="override cfg.cloth_density")
parser.add_argument("--edge_ke", type=float, default=None, help="override cfg.edge_ke (bending stiffness)")
parser.add_argument("--scene", nargs="*", default=None, metavar="K=V", help="generic scene cfg overrides, e.g. cloth_density=15 soft_contact_mu=0.5")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    # Recording path (scripts/record_video.py): on isaaclab develop, rendering is pumped by
    # VISUALIZERS — without a kit visualizer the replicator render products stay EMPTY. And an
    # EXPLICIT `--headless` flag (deprecated there) force-disables all visualizers. So: turn
    # --headless into a parser DEFAULT and default the kit visualizer on — a headless app with a
    # kit visualizer fills the render products.
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

# Disable the cubric GPU transform hierarchy: its shim is pinned to IAdapter v0.1 while the
# isaacsim 6.0.0.1 plugin reports v0.2, and the drift breaks hierarchy propagation — robot link
# VISUALS render detached/frozen while physics is correct. Forcing `_cubric = None` selects the
# CPU `update_world_xforms()` fallback, which renders correctly (fine at video cadence).
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

from typing import TYPE_CHECKING

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.controllers.differential_ik import DifferentialIKController  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

if TYPE_CHECKING:
    from robobench.suites.folding.scenes import TshirtFoldingScene

FPS = 60
OPEN, CLOSE = 0.8 * 0.04, 0.1 * 0.04  # 80% / 10% of the 4 cm finger stroke, in m
# Gripper-down grasp quats for the panda_hand frame (link7-frame downward flips right-multiplied
# by Rz(-pi/4), the hand frame's -45° yaw vs link7). Quats in **xyzw** (isaaclab develop convention):
QL = (0.92388, 0.0, 0.38268, 0.0)  # grasps on the +x half of the table: 180° about (cos22.5°, 0, sin22.5°)
QR = (1.0, 0.0, 0.0, 0.0)  # grasps on the -x half / center line: 180° about x
# Near-base grasp (right sleeve): approach tilted 45° away from the base so the wrist clears the
# joint-limit envelope — a straight-down pinch 0.19 m from the base axis stalls ~14 cm short.
QR_NEAR = (-0.92388, 0.0, 0.38268, 0.0)  # 180° about (-cos22.5°, 0, sin22.5°)
TIP_OFFSET = (0.0, 0.0, 0.113)  # fingertip point in the panda_hand frame (link7 +0.22 m = hand 0.107 + 0.113)


def _row(dur: float, x_cm: float, y_cm: float, z_cm: float, quat: tuple, grip_m: float) -> tuple:
    """One key pose: cm world position -> robot-root-frame meters (base at (-0.5,-0.5,0))."""
    return (dur, x_cm / 100.0 + 0.5, y_cm / 100.0 + 0.5, z_cm / 100.0, *quat, grip_m)


# The timed key-pose table. Sleeve GRASP coordinates target where the tuned cloth parameters
# settle the shirt (measured: sleeve tips (+0.379,-0.62) / (-0.374,-0.611), hem y=-0.16,
# collar y=-0.83), pinching ~3 cm inside the tip; `retarget_grasp` re-measures at run time.
KEY_POSES = [
    # wait for the cloth to settle, hover above the left sleeve
    _row(3.5, 35, -62, 28.0, QL, OPEN),
    # --- fold left sleeve to the middle ---
    _row(2.5, 35, -62, 20.0, QL, OPEN),  # descend onto sleeve tip
    _row(2.0, 35, -62, 20.0, QL, CLOSE),  # pinch sleeve
    _row(2.0, 33, -62, 26.0, QL, CLOSE),  # lift gently
    _row(2.0, 18, -62, 30.0, QL, CLOSE),  # carry hop 1
    _row(2.0, 6, -62, 31.0, QL, CLOSE),  # carry hop 2
    _row(2.0, -4, -62, 28.0, QL, CLOSE),  # carry hop 3 to center line
    _row(1.5, -4, -62, 24.0, QL, CLOSE),  # lower
    _row(1.0, -4, -62, 24.0, QL, OPEN),  # release
    _row(1.5, -4, -62, 38.0, QL, OPEN),  # retreat upward
    # --- fold right sleeve to the middle (tilted approach: near-base reach) ---
    _row(2.5, -34, -61, 28.0, QR_NEAR, OPEN),  # hover above right sleeve
    _row(2.5, -34, -61, 20.0, QR_NEAR, OPEN),  # descend onto sleeve tip
    _row(2.0, -34, -61, 20.0, QR_NEAR, CLOSE),  # pinch sleeve
    _row(2.0, -32, -61, 26.0, QR_NEAR, CLOSE),  # lift gently
    _row(2.0, -16, -61, 30.0, QR, CLOSE),  # carry hop 1
    _row(2.0, -5, -61, 31.0, QR, CLOSE),  # carry hop 2
    _row(2.0, 4, -61, 28.0, QR, CLOSE),  # carry hop 3 to center line
    _row(1.5, 4, -61, 24.0, QR, CLOSE),  # lower
    _row(1.0, 4, -61, 24.0, QR, OPEN),  # release
    _row(1.5, 4, -61, 38.0, QR, OPEN),  # retreat upward
    # --- fold bottom hem up to the collar ---
    _row(2.5, 0, -18, 28.0, QR, OPEN),  # hover above bottom hem
    _row(2.5, 0, -16, 18.5, QR, OPEN),  # descend onto hem
    _row(2.0, 0, -16, 18.5, QR, CLOSE),  # pinch hem
    _row(2.0, 0, -22, 26.0, QR, CLOSE),  # lift and start the fold
    _row(2.0, 0, -34, 27.0, QR, CLOSE),  # arc hop 1 (kept low to drag the hem up over the body,
    _row(2.0, 0, -48, 27.0, QR, CLOSE),  # arc hop 2  rather than lift it into a tent)
    _row(2.0, 0, -68, 24.0, QR, CLOSE),  # arc hop 3
    _row(2.0, 0, -84, 21.0, QR, CLOSE),  # lay the hem over the collar and press to crease
    _row(1.5, 0, -84, 22.0, QR, OPEN),  # release
    # tuck the trailing crease material toward the bundle
    _row(1.5, 0, -42, 32.0, QR, CLOSE),  # move above the crease
    _row(1.5, 0, -42, 22.5, QR, CLOSE),  # press down
    _row(2.0, 0, -52, 22.5, QR, CLOSE),  # slide toward the bundle
    _row(1.5, 0, -48, 34.0, QR, CLOSE),  # lift
    # pat the folded bundle flat with closed fingers
    _row(1.5, 0, -62, 34.0, QR, CLOSE),  # move above the bundle
    _row(1.5, 0, -62, 23.0, QR, CLOSE),  # press down
    _row(1.5, 0, -62, 34.0, QR, CLOSE),  # lift
    _row(2.0, -22, -60, 42.0, QR, OPEN),  # move clear of the folded shirt
]

# Verdict thresholds (env-local meters).
# Smoke gate: a completed 3-fold run lands ~0.17 m^2 (settled unfolded ~0.50); every observed
# failure mode (missed/slipped grasps, spring-back) stays >= 0.41.
FOOTPRINT_TARGET = 0.30  # m^2
BOUNDS_LO = (-0.45, -1.00, -0.05)
BOUNDS_HI = (0.45, 0.10, 0.60)


def _quat_step(q_from: torch.Tensor, q_to: torch.Tensor, alpha: float) -> torch.Tensor:
    """Normalized lerp with sign correction — fine for small per-tick steps."""
    dot = (q_from * q_to).sum(-1, keepdim=True)
    q_to = torch.where(dot < 0, -q_to, q_to)
    q = q_from + (q_to - q_from) * alpha
    return q / q.norm(dim=-1, keepdim=True)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    cfg = ENVS.get("folding.tshirt.franka.joint")()
    kw: dict = {}
    if args.density is not None:
        kw["cloth_density"] = args.density
    if args.edge_ke is not None:
        kw["edge_ke"] = args.edge_ke
    for kv in args.scene or []:
        k, v = kv.split("=", 1)
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    if kw:
        from robobench.suites.folding.scenes import TshirtFoldingSceneCfg

        cfg.scene_cfg = TshirtFoldingSceneCfg(**kw)
        print(f"[fold] scene overrides: {kw}", flush=True)
    env = cfg.build(num_envs=args.num_envs, device=device)
    scene: TshirtFoldingScene = env.scene  # type: ignore[assignment]
    render = (not args.headless) or livestream_on
    n = env.num_envs

    # -- robot handles + IK setup (frame/jacobian handling mirrors isaaclab's DiffIK action term) --
    art = env.robot.articulation
    arm_ids = art.find_joints(["panda_joint[1-7]"])[0]
    hand_idx = art.find_bodies(["panda_hand"])[0][0]
    num_base = getattr(art, "num_base_dofs", 0)
    jacobi_body = hand_idx - 1 if art.is_fixed_base else hand_idx
    jacobi_joints = [j + num_base for j in arm_ids]
    off_pos = torch.tensor(TIP_OFFSET, device=device).repeat(n, 1)
    off_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(n, 1)  # identity, xyzw
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": args.lam}
        ),
        num_envs=n,
        device=device,
    )

    def ee_pose_b() -> tuple[torch.Tensor, torch.Tensor]:
        """Current tip pose in the robot root frame (hand body + tip offset)."""
        p, q = math_utils.subtract_frame_transforms(
            art.data.root_pos_w.torch,
            art.data.root_quat_w.torch,
            art.data.body_pos_w.torch[:, hand_idx],
            art.data.body_quat_w.torch[:, hand_idx],
        )
        return math_utils.combine_frame_transforms(p, q, off_pos, off_rot)

    def jacobian_b() -> torch.Tensor:
        """Tip jacobian in the root frame (offset-corrected; rot offset is identity)."""
        J = art.data.body_link_jacobian_w.torch[:, jacobi_body, :, jacobi_joints]  # (n, 6, 7) copy
        R = math_utils.matrix_from_quat(math_utils.quat_inv(art.data.root_quat_w.torch))
        J[:, :3, :] = torch.bmm(R, J[:, :3, :])
        J[:, 3:, :] = torch.bmm(R, J[:, 3:, :])
        J[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(off_pos), J[:, 3:, :])
        return J

    # -- schedule --
    durs = np.array([r[0] for r in KEY_POSES]) * args.time_scale
    t_edges = np.cumsum(durs)
    poses = torch.tensor([r[1:8] for r in KEY_POSES], dtype=torch.float32, device=device)  # (K, 7)
    grips = torch.tensor([r[8] for r in KEY_POSES], dtype=torch.float32, device=device)  # (K,)
    total_steps = int(round((t_edges[-1] + args.hold) * FPS))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    beta = 1.0 - math.exp(-(1.0 / FPS) / args.tau)
    arm_limits = art.data.joint_pos_limits.torch[:, arm_ids, :]  # (n, 7, 2)
    if args.debug:
        try:
            print("    dbg effort limits:", art.data.joint_effort_limits.torch[0].cpu().numpy().round(1), flush=True)
        except Exception as e:  # noqa: BLE001
            print("    dbg effort limits unavailable:", e, flush=True)

    env.reset()
    # Seed the target filter at the arm's ACTUAL tip pose so the command ramps smoothly from the
    # home configuration to the first key pose with the tau time constant; seeding at the key
    # pose would command a ~40 cm jump and blow up the contacts.
    ee_p0, ee_q0 = ee_pose_b()
    filt_pos = ee_p0.clone()
    filt_quat = ee_q0.clone()
    hover_err = None
    last_k = -1

    def cloth_root() -> torch.Tensor:
        """Cloth particles in the robot ROOT frame (base rot is identity) — poses live there too."""
        return scene.cloth.data.nodal_pos_w.torch[0] - art.data.root_pos_w.torch[0]

    def retarget_grasp(k_new: int) -> None:
        """Adaptive grasp anchoring: each grasp block re-anchors on the MEASURED cloth extreme
        when its approach begins (the cloth drifts as folds progress and with material params) —
        fixed coordinates pinch air. Only the approach/pinch/lift rows move — the fold
        destinations stay scripted. Uses env 0 (this smoke is single-env)."""
        p = cloth_root()
        if k_new == 1:  # left sleeve: descend rows 1-2 at tip - 3 cm (x), lift row 3 another 2 cm in
            tip = p[p[:, 0].argmax()]
            for i, inset in ((1, -0.03), (2, -0.03), (3, -0.05)):
                poses[i, 0] = tip[0] + inset
                poses[i, 1] = tip[1]
            print(f"  [adapt] left sleeve tip (root) {tip[:2].cpu().numpy().round(3)}", flush=True)
        elif k_new == 10:  # right sleeve: hover 10 + descend/pinch 11-12 at tip + 6 cm, lift 13 at +8 cm
            tip = p[p[:, 0].argmin()]
            for i, inset in ((10, 0.06), (11, 0.06), (12, 0.06), (13, 0.08)):
                poses[i, 0] = tip[0] + inset
                poses[i, 1] = tip[1]
            print(f"  [adapt] right sleeve tip (root) {tip[:2].cpu().numpy().round(3)}", flush=True)
        elif k_new in (20, 21):  # bottom hem: max-y particle near the center line; re-measure at
            # the descend row too (the edge drifts during the 2.5 s hover approach)
            mid = p[p[:, 0].sub(0.5).abs() < 0.12]  # root x = world x + 0.5
            hem = mid[mid[:, 1].argmax()]
            if k_new == 20:
                poses[20, 0] = hem[0]
                poses[20, 1] = hem[1] - 0.05
            for i in (21, 22):
                poses[i, 0] = hem[0]
                poses[i, 1] = hem[1] - 0.05  # pinch 5 cm inside the hem edge for a solid bite of the flat fabric
            print(f"  [adapt] hem edge (root) {hem[:2].cpu().numpy().round(3)} (at pose {k_new})", flush=True)
        elif k_new == 29:  # tuck + pat: re-anchor on the folded bundle. The crease (trailing edge
            # of the hem fold) is the current max-y point; the pat lands on the bundle centroid.
            crease = p[p[:, 1].argmax()]
            centroid = p.mean(dim=0)
            for i, dy in ((29, -0.02), (30, -0.02), (31, -0.12), (32, -0.08)):
                poses[i, 0] = crease[0]
                poses[i, 1] = crease[1] + dy
            for i in (33, 34, 35):
                poses[i, 0] = centroid[0]
                poses[i, 1] = centroid[1]
            print(
                f"  [adapt] crease (root) {crease[:2].cpu().numpy().round(3)}"
                f" bundle centroid {centroid[:2].cpu().numpy().round(3)}",
                flush=True,
            )
    max_dq = 0.20  # per-tick joint step clamp [rad] (~12 rad/s commanded; the servo is the real limiter)
    # Nullspace posture target: pull joints 2..7 back toward home, joint 1 free.
    q_home = art.data.joint_pos.torch[:, arm_ids].clone()
    eye7 = torch.eye(len(arm_ids), device=device).expand(n, -1, -1)

    for step in range(total_steps):
        t = step / FPS
        k = int(np.searchsorted(t_edges, t)) if t < t_edges[-1] else len(KEY_POSES) - 1
        if k != last_k:
            retarget_grasp(k)
        filt_pos += (poses[k, :3] - filt_pos) * beta
        filt_quat = _quat_step(filt_quat, poses[k, 3:7].expand_as(filt_quat), beta)

        ee_p, ee_q = ee_pose_b()
        ik.set_command(torch.cat([filt_pos, filt_quat], dim=-1), ee_p, ee_q)
        q_arm = art.data.joint_pos.torch[:, arm_ids]
        J = jacobian_b()
        q_des = ik.compute(ee_p, ee_q, J, q_arm)
        # Nullspace posture pull: project (home - q) through I - J+J, leaving joint 1 free.
        dq_null = (q_home - q_arm) * (args.k_null / FPS)
        dq_null[:, 0] = 0.0
        J_pinv = torch.linalg.pinv(J)
        dq_null = (dq_null.unsqueeze(1) @ (eye7 - J_pinv @ J).transpose(1, 2)).squeeze(1)
        q_cmd = q_arm + (q_des - q_arm + dq_null).clamp(-max_dq, max_dq)
        q_cmd = q_cmd.clamp(arm_limits[..., 0], arm_limits[..., 1])
        action = torch.cat([q_cmd, grips[k].expand(n, 2)], dim=-1)  # 7 arm + 2 fingers
        env.step(action, render=render)
        if args.debug and step % 30 == 0:
            qe = math_utils.quat_error_magnitude(ee_q, filt_quat)
            print(
                f"    dbg step {step:4d} | filt {filt_pos[0].cpu().numpy().round(3)}"
                f" | ee {ee_p[0].cpu().numpy().round(3)} | rot_err {float(qe.max()):.2f} rad"
                f" | q {q_arm[0].cpu().numpy().round(2)} | dq {(q_des - q_arm)[0].cpu().numpy().round(3)}"
                f" | cmd-q {(q_cmd - q_arm)[0].cpu().numpy().round(3)}",
                flush=True,
            )

        if k != last_k:  # phase transition: footprint + max z tell the fold/grasp story
            zmax = float(scene.cloth.data.nodal_pos_w.torch[..., 2].max())
            extra = ""
            if args.debug:
                try:
                    import numpy as _np

                    from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import (
                        NewtonCoupledMJWarpVBDManager as _MGR,
                    )
                    from isaaclab_newton.physics.newton_manager import NewtonManager as _NM

                    _c = _MGR._contacts
                    _cnt = int(_c.soft_contact_count.numpy().ravel()[0])
                    _shapes = _c.soft_contact_particle.numpy()[:1]  # touch to keep warp happy
                    _sh = _c.soft_contact_shape.numpy()[:_cnt]
                    _rob = int((_NM.get_model().shape_body.numpy()[_sh] >= 0).sum())
                    _pw = scene.cloth.data.nodal_pos_w.torch
                    _tip_w = art.data.body_pos_w.torch[:, hand_idx] + math_utils.quat_apply(
                        art.data.body_quat_w.torch[:, hand_idx], off_pos
                    )
                    _dists = (_pw - _tip_w[:, None, :]).norm(dim=-1)
                    _d = float(_dists.min())
                    _near = _pw[0, int(_dists[0].argmin())].cpu().numpy().round(3)
                    _tw = _tip_w[0].cpu().numpy().round(3)
                    extra = (
                        f" | contacts {_cnt} (robot {_rob}) | tip-cloth dist {_d:.3f}"
                        f" | tip_w {_tw} nearest {_near}"
                    )
                except Exception as _e:  # noqa: BLE001
                    extra = f" | diag failed: {_e}"
            print(
                f"  t={t:5.1f}s pose {k:2d}/{len(KEY_POSES) - 1} | footprint {scene.footprint().mean():.3f} m^2"
                f" | cloth zmax {zmax:.3f}{extra}",
                flush=True,
            )
            last_k = k
        if step == int(t_edges[0] * FPS) - 1:  # end of the settle/hover pose: validates the frame/quat conventions
            hover_err = float((ee_pose_b()[0] - poses[0, :3]).norm(dim=-1).max())
            print(f"  hover reached | ee err {hover_err * 100:.1f} cm (want < ~2 cm)", flush=True)
        if step % args.print_every == 0:
            err = float((ee_p - filt_pos).norm(dim=-1).max())
            print(
                f"  step {step:5d} t={t:5.1f}s pose {k:2d} | ee err {err * 100:4.1f} cm"
                f" | footprint {scene.footprint().mean():.3f} m^2",
                flush=True,
            )

    # -- verdict --
    p = scene.nodal_pos_local()
    lo = torch.tensor(BOUNDS_LO, device=p.device)
    hi = torch.tensor(BOUNDS_HI, device=p.device)
    in_bounds = bool(((p >= lo) & (p <= hi)).all())
    fp = float(scene.footprint().mean())
    ext_x = float((p[..., 0].amax() - p[..., 0].amin()))
    ext_y = float((p[..., 1].amax() - p[..., 1].amin()))
    folded = fp < FOOTPRINT_TARGET
    hover_ok = hover_err is not None and hover_err < 0.03
    verdict = "PASS" if (folded and in_bounds and hover_ok) else "FAIL"
    print(
        f"TSHIRT-FOLD {verdict} | footprint {fp:.3f} m^2 = {ext_x:.2f} x {ext_y:.2f} m"
        f" (gate < {FOOTPRINT_TARGET}, unfolded ~0.50) | in_bounds={in_bounds}"
        f" | hover_err={None if hover_err is None else round(hover_err * 100, 1)} cm",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
