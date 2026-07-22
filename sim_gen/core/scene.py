"""BaseScene — the contract every sim_gen task implements.

A task is one Scene subclass: it owns the MJCF model (``xml()``), the per-episode
instance distribution (``reset_instance()``), and the task semantics (``success()`` /
``score()`` / ``describe()``).  Mirrors the robobench scene-is-task convention, on plain
MuJoCo instead of Isaac.

No robot embodiment lives here: tasks are validated scene-first with a *null-robot*
solution that moves objects kinematically (``carry()``) and lets physics settle the
rest, exactly like the robobench null-robot smokes.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

SCENES: dict[str, type["BaseScene"]] = {}


def register_scene(name: str):
    """Class decorator: register a Scene under ``name``."""

    def deco(cls):
        if name in SCENES and SCENES[name] is not cls:
            raise ValueError(f"scene {name!r} already registered to {SCENES[name]}")
        SCENES[name] = cls
        cls.scene_name = name
        return cls

    return deco


@dataclass
class SceneCfg:
    """Every tunable init parameter of a scene lives in its cfg (house rule)."""

    episode_steps: int = 2000            # physics-step budget per episode
    timestep: float = 0.002              # MuJoCo dt
    camera_distance: float = 1.6         # default free-camera framing
    camera_azimuth: float = 110.0
    camera_elevation: float = -28.0
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.15)


class BaseScene:
    """Owns MjModel/MjData plus the helpers null solutions and checks are built from."""

    scene_name = "base"

    def __init__(self, cfg: SceneCfg | None = None):
        self.cfg = cfg or SceneCfg()
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.model.opt.timestep = self.cfg.timestep
        # Offscreen framebuffer large enough for the recorder (tasks shouldn't need
        # to remember the <visual><global offwidth/></visual> boilerplate).
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 800)
        self.data = mujoco.MjData(self.model)
        self.steps_used = 0
        self.recorder = None  # attached by smoke/harness via attach_recorder()
        self._rng = np.random.default_rng(0)

    # ---- authoring surface (subclasses implement) -------------------------------
    def xml(self) -> str:
        """Return the full MJCF model string."""
        raise NotImplementedError

    def reset_instance(self, rng: np.random.Generator) -> None:
        """Sample ONE episode instance and write it into ``self.data`` (poses, masses,
        goals...).  Must genuinely randomize: different rng seeds => different
        instances (gated by validate.py)."""
        raise NotImplementedError

    def success(self) -> bool:
        """Binary task completion, judged from the current physics state."""
        raise NotImplementedError

    def score(self) -> float:
        """Graded progress in [0, 1]; monotone in real progress; 1.0 iff success()."""
        raise NotImplementedError

    def describe(self) -> str:
        """The natural-language task statement a solving agent receives."""
        raise NotImplementedError

    # ---- lifecycle ---------------------------------------------------------------
    def reset(self, seed: int = 0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.steps_used = 0
        self._rng = np.random.default_rng(seed)
        self.reset_instance(self._rng)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.steps_used += 1
            if self.recorder is not None:
                self.recorder.maybe_capture(self)

    def settle(self, steps: int = 300) -> None:
        """Step with zero controls until things come to rest (bounded)."""
        self.data.ctrl[:] = 0
        self.step(steps)

    # ---- state snapshot / restore --------------------------------------------------
    def get_state(self) -> np.ndarray:
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        buf = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return buf

    def set_state(self, buf: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, self.data)

    def state_hash(self) -> str:
        return hashlib.sha256(self.get_state().tobytes()).hexdigest()[:16]

    # ---- named access helpers -------------------------------------------------------
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def body_quat(self, name: str) -> np.ndarray:
        return self.data.body(name).xquat.copy()

    def _free_joint_adr(self, body: str) -> int:
        """qpos address of the body's free joint (bodies moved by the null robot must
        have a <freejoint/>)."""
        b = self.model.body(body)
        j = self.model.joint(b.jntadr[0])
        if j.type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"body {body!r} has no free joint")
        return int(j.qposadr[0])

    def set_body_pose(self, body: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Write a free body's pose directly (zeroing its velocity)."""
        adr = self._free_joint_adr(body)
        self.data.qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=np.float64)
        b = self.model.body(body)
        dadr = int(self.model.joint(b.jntadr[0]).dofadr[0])
        self.data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- null-robot primitive ---------------------------------------------------------
    def carry(self, body: str, to_pos, to_quat=None, steps: int = 150) -> None:
        """Kinematically carry a free body along a straight line while physics runs
        (the null-robot move: the carried body is pose-driven with zero velocity each
        step so the rest of the world reacts normally), then release."""
        adr = self._free_joint_adr(body)
        p0 = self.data.qpos[adr : adr + 3].copy()
        q0 = self.data.qpos[adr + 3 : adr + 7].copy()
        p1 = np.asarray(to_pos, dtype=np.float64)
        q1 = np.asarray(to_quat, dtype=np.float64) if to_quat is not None else q0
        dadr = int(self.model.joint(self.model.body(body).jntadr[0]).dofadr[0])
        for i in range(1, steps + 1):
            t = i / steps
            self.data.qpos[adr : adr + 3] = (1 - t) * p0 + t * p1
            q = (1 - t) * q0 + t * q1
            self.data.qpos[adr + 3 : adr + 7] = q / (np.linalg.norm(q) + 1e-12)
            self.data.qvel[dadr : dadr + 6] = 0.0
            self.step(1)

    # ---- misc -------------------------------------------------------------------------
    def attach_recorder(self, recorder) -> None:
        self.recorder = recorder

    def cfg_dict(self) -> dict:
        return dataclasses.asdict(self.cfg)
