"""Franka Panda embodiment for sim_gen tasks.

Injects the MuJoCo Menagerie Panda (vendored under assets/, BSD-licensed) into any
task scene via MjSpec composition, and exposes the CoSiGen-loop-style control API
(``move_to`` / ``set_gripper`` / ...) that solving agents write code against.

The arm is position-actuated (the Menagerie model's own actuators); ``move_to`` solves
IK with damped least squares on the MuJoCo Jacobian and glides the joint targets there
while physics runs.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
PANDA_XML = ASSETS / "franka_emika_panda" / "panda.xml"

# Panda TCP: ~0.1034 m along the hand's z (between the fingertips).
TCP_OFFSET = (0.0, 0.0, 0.1034)
HOME_QPOS = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)


def attach_franka(scene_xml: str, base_pos=(-0.45, 0.0, 0.0), base_quat=(1, 0, 0, 0),
                  prefix: str = "robot_") -> mujoco.MjModel:
    """Compile ``scene_xml`` with a Panda attached at ``base_pos``.

    Returns the combined model; robot names carry ``prefix``.
    """
    spec = mujoco.MjSpec.from_string(scene_xml)
    robot = mujoco.MjSpec.from_file(str(PANDA_XML))
    # TCP site on the hand, so IK has a well-defined end-effector frame.
    hand = robot.body("hand")
    hand.add_site(name="tcp", pos=TCP_OFFSET, size=[0.005, 0.005, 0.005])
    frame = spec.worldbody.add_frame(pos=list(base_pos), quat=list(base_quat))
    frame.attach_body(robot.body("link0"), prefix, "")
    return spec.compile()


class FrankaArm:
    """Control wrapper: position-actuated arm + gripper on a combined scene model."""

    def __init__(self, scene, prefix: str = "robot_"):
        """``scene`` is a BaseScene whose model was built with attach_franka()."""
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        self.prefix = prefix
        self.joint_ids = [self.model.joint(f"{prefix}joint{i}").id for i in range(1, 8)]
        self.qadr = [int(self.model.jnt_qposadr[j]) for j in self.joint_ids]
        self.dofadr = [int(self.model.jnt_dofadr[j]) for j in self.joint_ids]
        self.act_ids = [self.model.actuator(f"{prefix}actuator{i}").id
                        for i in range(1, 8)]
        self.grip_act = self.model.actuator(f"{prefix}actuator8").id
        self.tcp_site = self.model.site(f"{prefix}tcp").id
        self.ctrl_lo = self.model.actuator_ctrlrange[self.act_ids, 0]
        self.ctrl_hi = self.model.actuator_ctrlrange[self.act_ids, 1]

    # ---- setup ------------------------------------------------------------------
    def go_home(self) -> None:
        """Write the home configuration directly (used at reset, before physics)."""
        for adr, q in zip(self.qadr, HOME_QPOS):
            self.data.qpos[adr] = q
        for adr in self.dofadr:
            self.data.qvel[adr] = 0.0
        for aid, q in zip(self.act_ids, HOME_QPOS):
            self.data.ctrl[aid] = q
        self.data.ctrl[self.grip_act] = 255.0  # open
        mujoco.mj_forward(self.model, self.data)

    # ---- kinematics --------------------------------------------------------------
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def _ik(self, target_pos, target_quat=None, iters: int = 100, damping: float = 1e-3,
            tol: float = 1e-3) -> np.ndarray:
        """Damped-least-squares IK for the TCP site; returns 7 joint positions.
        Runs on a scratch copy of qpos so the live state is untouched."""
        qpos0 = self.data.qpos.copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            err_pos = target_pos - self.data.site_xpos[self.tcp_site]
            if target_quat is not None:
                site_q = np.empty(4)
                mujoco.mju_mat2Quat(site_q, self.data.site_xmat[self.tcp_site])
                dq = np.empty(4)
                mujoco.mju_mulQuat(dq, np.asarray(target_quat, dtype=np.float64),
                                   _quat_conj(site_q))
                err_rot = 2.0 * dq[1:4] * np.sign(dq[0])
                err = np.concatenate([err_pos, err_rot])
            else:
                err = err_pos
            if np.linalg.norm(err_pos) < tol:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site)
            J = np.vstack([jacp, jacr])[: len(err), self.dofadr]
            dq_step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(len(err)), err)
            for k, adr in enumerate(self.qadr):
                self.data.qpos[adr] = np.clip(self.data.qpos[adr] + dq_step[k],
                                              self.ctrl_lo[k], self.ctrl_hi[k])
        sol = np.array([self.data.qpos[adr] for adr in self.qadr])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)
        return sol

    # ---- the solving-agent API ---------------------------------------------------
    def move_to(self, pos, quat=None, steps: int = 300, pos_tol: float = 0.01) -> float:
        """Move the TCP to ``pos`` (optionally with orientation ``quat``, wxyz).
        Glides joint position targets to the IK solution while physics runs; returns
        the final TCP position error (m)."""
        sol = self._ik(pos, quat)
        start = np.array([self.data.ctrl[a] for a in self.act_ids])
        for i in range(1, steps + 1):
            t = i / steps
            tgt = (1 - t) * start + t * sol
            for aid, q in zip(self.act_ids, tgt):
                self.data.ctrl[aid] = q
            self.scene.step(1)
            if t == 1.0 or (i % 25 == 0
                            and np.linalg.norm(self.tcp_pos() - np.asarray(pos)) < pos_tol):
                break
        # settle at the final target
        for _ in range(50):
            self.scene.step(1)
            if np.linalg.norm(self.tcp_pos() - np.asarray(pos)) < pos_tol:
                break
        return float(np.linalg.norm(self.tcp_pos() - np.asarray(pos)))

    def set_gripper(self, open_frac: float, steps: int = 100) -> None:
        """0.0 = closed, 1.0 = fully open (255 in actuator units)."""
        self.data.ctrl[self.grip_act] = float(np.clip(open_frac, 0, 1)) * 255.0
        self.scene.step(steps)

    def functions(self) -> dict:
        """The API surface a solving agent gets (mirrors the CoSiGen loop style)."""
        return {
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "tcp_pos": self.tcp_pos,
            "step": self.scene.step,
            "body_pos": self.scene.body_pos,
            "body_quat": self.scene.body_quat,
            "success": self.scene.success,
            "score": self.scene.score,
            "describe": self.scene.describe,
        }


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def bind_franka(scene_cls, cfg=None, base_pos=(-0.45, 0.0, 0.0)):
    """Instantiate a task scene with a Franka attached; returns (scene, arm).

    The scene keeps its full contract (reset/success/score/...); after every
    ``scene.reset()`` call ``arm.go_home()`` to restore the robot posture.
    """
    scene = scene_cls(cfg) if cfg is not None else scene_cls()
    scene.model = attach_franka(scene.xml(), base_pos=base_pos)
    scene.model.opt.timestep = scene.cfg.timestep
    scene.model.vis.global_.offwidth = max(scene.model.vis.global_.offwidth, 1280)
    scene.model.vis.global_.offheight = max(scene.model.vis.global_.offheight, 800)
    scene.data = mujoco.MjData(scene.model)
    arm = FrankaArm(scene)
    return scene, arm
