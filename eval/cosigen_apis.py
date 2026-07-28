"""CoSiGen control-API classes: one privileged code-as-policy surface per
embodiment (G1 assembly base class + Franka / Kuka / bimanual / packing /
generic / null-scene variants). Prompt text lives in cosigen_prompts; the
per-env oracle view and state utils in cosigen_view."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import re as _re
import subprocess
import time as _time
import traceback
import uuid as _uuid

import numpy as np
import torch

import robobench
from robobench.core import ENVS

from cosigen_config import _FEATURES_OFF, _RELOAD_HDFS_DIR, _import_sibling
from cosigen_prompts import (
    CODE_DIRECTIVE, DOC_CHECKPOINT_TREE, DOC_CONTROL_TOOLKIT, DOC_HEAD,
    DOC_RAW_ACCESS, DOC_READ_ONLY, DOC_TAIL, DOC_TUNING, EMB_FRANKA_BIMANUAL,
    EMB_FRANKA_SINGLE, EMB_G1_ASSEMBLY, EMB_IKEA_BIMANUAL, EMB_KUKA_ALLEGRO,
    SEATED_LINE_ASSEMBLY, SEATED_LINE_GENERIC,
    _compose_doc, _emb_screw,
)
from cosigen_view import (AssemblyView, _clip_words, _np, _overwrite_replica_rows,
                          _q_wxyz, expand_state_tree, zero_root_velocities)


class AssemblyApi:
    """Privileged code-as-policy control API for the robobench IKEA-table assembly
    suite (humanoid, pink_ik wrist-pose control). Object poses are WORLD frame;
    pink_ik wants ENV-frame wrist poses, handled internally.

    N-ENV MODEL: env 0 is authoritative (all queries/grading read env 0); control
    targets are kept IDENTICAL across envs during scripted phases so the replicas
    mirror env 0. A batch search (the optimize tool) temporarily de-syncs the
    replicas (its candidate population), then restores everything from the
    pre-search snapshot."""

    STUD_OFFSETS = [(0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25)]

    # Wrist bodies the IK frames track (per embodiment; G1 default). GR1T2 subclasses
    # override — its pink_ik frames are the hand pitch links.
    WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    def __init__(self, env, max_steps: int = 3000):
        self._init_common(env, max_steps)
        self._wrist_idx = [
            self.art.find_bodies(self.WRIST_BODIES[0])[0][0],
            self.art.find_bodies(self.WRIST_BODIES[1])[0][0],
        ]
        self._hand_ids = env.robot.controller.controllers[1].joint_ids
        self._init_hand_limits()
        self.reset_state()

    def _init_common(self, env, max_steps: int) -> None:
        self._env = env
        self.art = env.robot.articulation
        self.scene = env.scene
        self.origin = env.iscene.env_origins
        self.device = env.device
        self.n = int(env.num_envs)
        self.max_steps = int(max_steps)
        self._joint_id_cache: dict = {}
        self._snapshots: dict[str, dict] = {}
        self._snap_seq = 0
        self._event_log: list[dict] = []
        # Checkpoint TREE for backtracking: auto node per turn + manual checkpoint();
        # goto(cid) restores the world (namespace persists). Branching = goto + act.
        self._ckpt_nodes: dict[str, dict] = {}
        self._ckpt_seq = 0
        self._ckpt_current: str | None = None
        # Trial/checkpoint protocol: each execution starts from the current node and only
        # a checkpointed, cleanly-exiting program becomes a child node (set from turn meta).
        self._trial_mode = False
        # Set for one turn by the driver's goto tool (turn meta): the only route by which
        # a trial-mode session may move the world to another node.
        self._goto_allowed = False
        self._pending_checkpoint: dict | None = None
        # Parameter-search request (driver's optimize tool): run_policy executes the
        # search instead of code for that turn; the result rides the turn payload.
        self._pending_optimize: dict | None = None
        self._last_optimize_result: dict | None = None
        # Live per-generation search progress, readable over HTTP (/ping) while the
        # search occupies the main thread — the async optimize tool's data channel.
        self._opt_progress: dict | None = None
        # Replica parking (2026-07-24): replicas exist ONLY as the optimize tool's
        # candidate batch. During scripted execution they used to mirror env 0, paying
        # n copies of every contact solve for nothing (a 1195s regrip turn at n=512).
        # Parked replicas hold a quiescent pose; a batch search re-seeds them from the
        # env-0 snapshot. Opt-out: CAPX_PARK_REPLICAS=0.
        self._park_replicas = False  # enabled by apis that implement the parked _action
        self._replicas_active = False
        self._park_states = None
        self._turn_no = 0
        self._turn_frame_start = 0
        # Optional durable tree storage. Configured by the harness through the
        # request meta side-channel; disabled until a session id is supplied.
        self._persist_session_id: str | None = None
        self._persist_hdfs_root = os.environ.get(
            "CAPX_COSIGEN_SESSION_DIR",
            "hdfs://haruna/tmp/zeyu.shen/cosigen_sessions",
        )
        self._persist_path: str | None = None
        # optional RTX recorder + viewport camera (attached by the server; see look())
        self._rec_cam = None
        self._viewport_cam = None
        self._rec_every = 20
        self._rec_frames: list = []   # EPISODE VIDEO: default camera only, captured during step()
        self._rec_parts: list[dict] = []  # zero-truncation storage: flushed npz chunks
        self._rec_total = 0               # frames captured EVER (global index space)
        self._rec_epoch = _uuid.uuid4().hex  # disambiguates indices across server restarts
        self._edge_frame_start = 0        # first frame after parent/goto for next checkpoint
        self._rec_turn_part_start = 0     # first part index belonging to the current turn
        self._aux_frames: list = []   # look()/render_checkpoint() captures: agent-facing only,
        self._turn_images: list[dict] = []  # ...NEVER spliced into the episode video

    # Proven flex targets from the reference G1 wheel pick (eval_result/Fable_5/
    # G1_Upper_body_IK): LEFT fingers curl toward NEGATIVE joint values (0 = straight,
    # extended); the RIGHT hand mirrors (curl positive). thumb_0 is a yaw joint
    # (1.0 = aside, 0.0 = opposing). The previous uniform "closed = upper limit"
    # mapping therefore EXTENDED the left fingers on close_gripper — every left-hand
    # grasp in earlier sessions was closing an inverted hand.
    _G1_HAND_OPEN = {"index_0": 0.0, "middle_0": 0.0, "index_1": 0.0, "middle_1": 0.0,
                     "thumb_0": 1.0, "thumb_1": 0.0, "thumb_2": 0.0}
    _G1_HAND_CLOSED = {"index_0": -1.25, "middle_0": -1.25, "index_1": -1.5, "middle_1": -1.5,
                       "thumb_0": 0.0, "thumb_1": 0.8, "thumb_2": 1.2}

    def _init_hand_limits(self) -> None:
        lim = None
        for attr in ("joint_pos_limits", "soft_joint_pos_limits", "default_joint_pos_limits"):
            if hasattr(self.art.data, attr):
                lim = getattr(self.art.data, attr)[0, self._hand_ids]
                break
        if lim is None:
            raise RuntimeError("no joint position limits on articulation data")
        self._hand_lower = lim[:, 0].clone()
        self._hand_upper = lim[:, 1].clone()
        # Per-joint open/closed target vectors (fallback: old lower->upper behavior).
        open_t = self._hand_lower.clone()
        closed_t = self._hand_upper.clone()
        names = list(self.art.data.joint_names)
        for col, jid in enumerate(self._hand_ids):
            m = _re.match(r"(left|right)_hand_(\w+)_joint", names[jid])
            if not m:
                continue
            side, suffix = m.groups()
            if suffix not in self._G1_HAND_OPEN:
                continue
            sign = 1.0 if side == "left" else -1.0
            lo, hi = float(self._hand_lower[col]), float(self._hand_upper[col])
            open_t[col] = float(np.clip(sign * self._G1_HAND_OPEN[suffix], lo, hi))
            closed_t[col] = float(np.clip(sign * self._G1_HAND_CLOSED[suffix], lo, hi))
        self._hand_open_t = open_t
        self._hand_closed_t = closed_t

    # ----------------- internal state helpers -----------------
    def reset_state(self):
        self._tgt = [
            self.art.data.body_link_state_w[:, self._wrist_idx[0], :7].clone(),
            self.art.data.body_link_state_w[:, self._wrist_idx[1], :7].clone(),
        ]
        pos = self.art.data.joint_pos[:, self._hand_ids]
        denom = self._hand_closed_t - self._hand_open_t
        safe = denom.abs() > 1e-4
        per_joint = torch.where(safe.unsqueeze(0), (pos - self._hand_open_t) / denom, torch.zeros_like(pos))
        self._hand_frac = per_joint.mean(dim=1).clamp(0.0, 1.0)
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))
        self._steps_used = 0
        self._policy_last_action = None

    def _action(self):
        def env_frame(p7):
            p = p7.clone()
            p[:, :3] = p7[:, :3] - self.origin
            return p
        return torch.cat([env_frame(self._tgt[0]), env_frame(self._tgt[1]), self._hands], dim=1)

    def _object_root(self, name: str) -> torch.Tensor:
        """(N,13) world root state of a named object (synthetic for studs)."""
        if name == "table":
            return self.scene.table.data.root_state_w
        if name.startswith("leg_"):
            return self.scene.legs[int(name.split("_")[1])].data.root_state_w
        if name.startswith("stud_"):
            i = int(name.split("_")[1])
            t = self.scene.table.data.root_state_w
            s = torch.zeros(self.n, 13, device=t.device)
            ox, oy = self.STUD_OFFSETS[i]
            s[:, 0] = t[:, 0] + ox
            s[:, 1] = t[:, 1] + oy
            s[:, 2] = t[:, 2]
            s[:, 3] = 1.0
            return s
        raise ValueError(f"unknown object {name!r}; use leg_i / stud_i / table")

    def _resolve_joint_ids(self, names):
        """names: None (all), a regex string, or a list of joint names/regexes."""
        key = names if isinstance(names, (str, type(None))) else tuple(names)
        if key not in self._joint_id_cache:
            if names is None:
                ids = list(range(self.art.num_joints))
            else:
                pats = [names] if isinstance(names, str) else list(names)
                ids = self.art.find_joints(pats)[0]
            self._joint_id_cache[key] = ids
        return self._joint_id_cache[key]

    def _api_state_env0(self) -> dict:
        return {"tgt": [t[0:1].clone() for t in self._tgt],
                "hand_frac": self._hand_frac[0:1].clone(),
                "steps_used": int(self._steps_used)}

    def _restore_api_state(self, s: dict) -> None:
        """Restore control targets/hands. NOTE: steps_used is NOT restored -- sim time is
        monotonic (jumping back in the tree is itself a step forward in time)."""
        for arm in (0, 1):
            self._tgt[arm][:] = s["tgt"][arm].repeat(self.n, 1)
        self._hand_frac[:] = s["hand_frac"].repeat(self.n)
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))

    # ----------------- sim stepping + video -----------------
    # Per-turn wall-clock budget (2026-07-24): a runaway agent turn (huge step loops)
    # has NO execution bound server-side — only the HTTP reply window 504s. Turns from
    # wall-killed eval sessions kept grinding for hours, GIL-starving the pod and
    # blocking the NEXT session's first turn (observed on 5 pods at once). The guard
    # lives at the shared step() choke point so it hot-deploys via /reload.
    _TURN_WALL_S = float(os.environ.get("CAPX_TURN_WALL_S", "3600"))

    def _check_turn_wall(self) -> None:
        deadline = getattr(self, "_turn_wall_deadline", None)
        if deadline is not None and _time.time() > deadline:
            self._turn_wall_deadline = None  # raise once; let cleanup/end_turn step freely
            raise RuntimeError(
                f"turn wall-clock budget exceeded ({self._TURN_WALL_S:.0f}s of sim "
                "stepping in one turn). Split the work across turns and re-plan with "
                "shorter, checkable moves.")

    def step(self, n: int = 1):
        """Advance the sim n control steps, re-applying the current wrist + hand targets."""
        for _ in range(int(n)):
            if self._steps_used >= self.max_steps:
                break
            if self._steps_used % 100 == 0:
                self._check_turn_wall()
            self._env.step(self._action(), render=False)
            if self._rec_cam is not None and self._steps_used % self._rec_every == 0:
                self._grab_frame()  # uncapped: chunks stream to disk (_flush_rec_chunk)
            self._steps_used += 1

    def attach_camera(self, camera) -> None:
        """Server-side: keep the viewport annotator available for look() even when
        full-video recording is off."""
        self._viewport_cam = camera

    # ZERO-TRUNCATION recording (user directive 2026-07-22: never drop frames, ever).
    # Frames stream to per-session npz PART FILES on local disk every _REC_CHUNK
    # captures; memory holds at most one chunk. Nothing is capped, nothing is cleared
    # without being flushed to disk first. A million frames = a million frames stored.
    _REC_CHUNK = int(os.environ.get("CAPX_VIDEO_CHUNK", "400"))

    def enable_recording(self, camera, every: int = 20, max_frames: int | None = None) -> None:
        """Turn on inline RGB capture using `camera` (a replicator rgb annotator). None disables.
        Re-arming NEVER discards footage: any buffered frames are flushed to a part file
        first. `max_frames` is IGNORED — every frame is kept (zero-truncation storage);
        the parameter survives only because pre-2026-07-26 server files (loaded at pod
        boot, not hot-reloadable) still pass it."""
        if getattr(self, "_rec_frames", None):
            self._flush_rec_chunk()
        self._rec_cam = camera
        self._rec_every = max(1, int(every))
        self._rec_frames = []
        self._aux_frames = []

    def _flush_rec_chunk(self) -> None:
        """Write the in-memory frame buffer to a local npz part file (never raises; on
        write failure the buffer is KEPT so no frame is ever lost silently)."""
        if not self._rec_frames:
            return
        try:
            import uuid as _uuid
            part_dir = f"/tmp/cosigen_rec_{id(self)}"
            os.makedirs(part_dir, exist_ok=True)
            path = f"{part_dir}/part_{len(self._rec_parts):06d}_{_uuid.uuid4().hex[:6]}.npz"
            arr = np.stack(self._rec_frames, axis=0)
            np.savez_compressed(path, frames=arr)
            self._rec_parts.append({"path": path,
                                    "start": self._rec_total - len(self._rec_frames),
                                    "n": len(self._rec_frames)})
            self._rec_frames = []
        except Exception:
            import sys as _sys
            print("frame chunk flush FAILED — keeping frames in memory (no loss):\n"
                  + traceback.format_exc(), file=_sys.stderr, flush=True)

    def drain_turn_frame_parts(self) -> list[dict]:
        """Server hook at turn end: flush the buffer and return THIS turn's part files
        [{path, start, n}, ...]. Parts stay on local disk for review_rollout()."""
        self._flush_rec_chunk()
        parts = self._rec_parts[self._rec_turn_part_start:]
        self._rec_turn_part_start = len(self._rec_parts)
        return list(parts)

    def _frames_at(self, idxs) -> list:
        """Fetch frames by GLOBAL capture index, transparently across part files and the
        live buffer (zero-truncation storage means any frame ever captured is loadable)."""
        out = []
        buf_start = self._rec_total - len(self._rec_frames)
        for i in idxs:
            i = int(i)
            if i >= buf_start:
                out.append(self._rec_frames[i - buf_start])
                continue
            for part in self._rec_parts:
                if part["start"] <= i < part["start"] + part["n"]:
                    out.append(np.load(part["path"])["frames"][i - part["start"]])
                    break
        return out

    # Renders per frame-grab: RTX temporal accumulation blends new renders with history,
    # so a single render after LARGE pose jumps (a goto() teleport, a look() camera cut)
    # shows a ghost of the previous view. Those aux grabs flush hard (16). INLINE video
    # frames are captured every couple of control steps of CONTINUOUS motion — history
    # decay across consecutive frames is tiny, so 2 renders suffice; this is what makes
    # dense (watchable, real-time) capture affordable at all. The 16x-flush-everywhere
    # era forced 15-20-step sampling and produced the 26x "teleporting robot" videos.
    _RENDER_FLUSH_AUX = int(os.environ.get("CAPX_VIDEO_FLUSH_RENDERS", "16"))
    # 1 render per capture: FXAA is enforced at the capture boundary (no temporal
    # accumulation), so the extra flush renders that fought TAA/DLSS ghosting are
    # dead cost. The post-jump ghost window below still flushes 6x as a safety net.
    _RENDER_FLUSH_INLINE = int(os.environ.get("CAPX_VIDEO_FLUSH_INLINE", "1"))
    _GHOST_FLUSH_CAPTURES = int(os.environ.get("CAPX_VIDEO_GHOST_CAPTURES", "12"))

    def _grab_frame(self, cam=None, aux: bool = False) -> bool:
        """Render the viewport and read one RGB frame into the episode-video buffer
        (aux=False) or the agent-facing aux buffer (aux=True: look()/render_checkpoint()
        captures, which must NOT pollute the fixed-camera episode video).
        Errors printed, never raised."""
        import sys as _sys
        cam = cam or self._rec_cam
        try:
            # Replicator may restore its configured renderer (DLSS/TAA, op=3) when a
            # recording turn starts, AFTER build_record_camera() selected FXAA. Enforce
            # the non-temporal mode at the actual capture boundary so no lifecycle hook
            # can silently re-enable temporal history and bake ghost silhouettes into RGB.
            import carb.settings

            _aa_want = int(os.environ.get("CAPX_CAPTURE_AA_OP", "2"))
            _settings = carb.settings.get_settings()
            _settings.set("/rtx/post/aa/op", _aa_want)
            # VERIFY, don't trust: read the setting back. A nonzero violation count is
            # surfaced in every turn payload — a capture-quality regression must scream
            # in the turn logs, not wait for someone to eyeball a video.
            if int(_settings.get("/rtx/post/aa/op") or -1) != _aa_want:
                self._aa_violations = getattr(self, "_aa_violations", 0) + 1
            flush = self._RENDER_FLUSH_AUX if aux else self._RENDER_FLUSH_INLINE
            # elevated flush right after a state JUMP: teleported bodies leave temporal-
            # accumulation ghosts for a stretch of captures otherwise
            if not aux and self._rec_total < getattr(self, "_ghost_flush_until", 0):
                flush = max(flush, 6)
            for _ in range(max(1, flush)):
                self._env.sim.render()
            data = cam.get_data()
            arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)
            if arr.size:
                if aux:
                    self._aux_frames.append(arr[:, :, :3].copy())
                else:
                    self._rec_frames.append(arr[:, :, :3].copy())
                    self._rec_total += 1
                    if len(self._rec_frames) >= self._REC_CHUNK:
                        self._flush_rec_chunk()  # stream to disk; memory stays bounded
                return True
        except Exception:
            print("frame grab error:\n" + traceback.format_exc(), file=_sys.stderr, flush=True)
        return False

    def collect_frames(self) -> np.ndarray:
        """The CURRENT TURN's frames (parts + live buffer). Prefer
        drain_turn_frame_parts() on the server: it avoids re-loading chunks into RAM."""
        start = self._turn_frame_start
        idxs = range(start, self._rec_total)
        frames = self._frames_at(idxs)
        if not frames:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)

    # ----------------- agent-facing: control -----------------
    def get_object_pose(self, name: str):
        """World pose of an assembly object. name: 'leg_0'..'leg_3', 'stud_0'..'stud_3', 'table'.
        Returns (position xyz (3,), quaternion wxyz (4,))."""
        s = self._object_root(name)[0]
        return _np(s[:3]), _q_wxyz(_np(s[3:7]))

    def get_eef_pose(self, arm: int = 0):
        """Current wrist (end-effector) WORLD pose: (position xyz, quaternion wxyz). arm 0=left, 1=right."""
        s = self.art.data.body_link_state_w[0, self._wrist_idx[arm], :7]
        return _np(s[:3]), _q_wxyz(_np(s[3:7]))

    def get_seated(self):
        """List of 4 booleans: whether each leg is seated on a stud (task done when all True)."""
        return [bool(v) for v in self.scene.seated()[0].tolist()]

    def move_to(self, arm: int, position, quaternion_wxyz=None, max_steps: int = 300,
                pos_tol: float = 0.02, max_step: float = 0.01):
        """Waypoint-move the arm's wrist to a WORLD-frame target position (optional orientation),
        so the IK tracks smoothly. arm 0=left, 1=right. Returns the final ACTUAL wrist distance
        to the target (m) -- if this stays large, the pose is unreachable for the IK (target
        commands converged but the arm physically cannot track them)."""
        goal_w = torch.tensor(np.asarray(position, dtype=np.float64), device=self.device,
                              dtype=torch.float32).reshape(1, 3)
        if quaternion_wxyz is not None:
            q = torch.tensor(np.asarray(quaternion_wxyz, dtype=np.float64),
                             device=self.device, dtype=torch.float32).reshape(1, 4)
            self._tgt[arm][:, 3:7] = q.repeat(self.n, 1)
        for _ in range(int(max_steps)):
            cur0 = self._tgt[arm][0:1, :3]  # env 0 authoritative; targets stay mirrored
            delta = goal_w - cur0
            pos_ok = float(delta.norm()) <= pos_tol
            # BOTH criteria must converge: a rotation-only move_to (same position, new
            # quaternion) must still step until the LIVE eef orientation tracks the target
            # (an early break here silently skipped all rotation commands).
            rot_ok = True
            if quaternion_wxyz is not None:
                q_live = self.art.data.body_link_state_w[0, self._wrist_idx[arm], 3:7]
                rot_ok = float(torch.abs((q_live * self._tgt[arm][0, 3:7]).sum())) > 0.998
            if pos_ok and rot_ok:
                break
            if not pos_ok:
                new0 = cur0 + delta * min(1.0, max_step / max(float(delta.norm()), 1e-6))
                self._tgt[arm][:, :3] = new0.repeat(self.n, 1)
            self.step(1)
            if self._steps_used >= self.max_steps:
                break
        eef = self.art.data.body_link_state_w[0:1, self._wrist_idx[arm], :3]
        return float((eef - goal_w).norm())

    def set_hand_frac_tensor(self, frac: torch.Tensor) -> None:
        """Set per-env hand closure in [0,1] (embodiment-specific joint mapping).
        Used by the RL ActionAdapter; G1: interpolates the per-joint OPEN->CLOSED flex
        vectors (left fingers curl negative, right positive; see _G1_HAND_CLOSED)."""
        self._hand_frac = frac
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))

    def _set_hands(self, frac: float, steps: int):
        self.set_hand_frac_tensor(torch.full_like(self._hand_frac, float(np.clip(frac, 0.0, 1.0))))
        self.step(steps)

    def close_gripper(self, arm: int = 0, steps: int = 30):
        """Close the three-finger hand (curl fingers in, thumb opposing)."""
        self._set_hands(1.0, steps)

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        """PER-JOINT hand control (asymmetric grasps: hook grasps, thumb-aside approaches,
        scoop-curls -- things the scalar open/close cannot express). `targets` maps joint
        suffixes to FRACTIONS in [0,1] of that joint's [lower, upper] range, e.g.
        {'index_0': 0.9, 'index_1': 0.9, 'middle_0': 0.9, 'middle_1': 0.9,
         'thumb_0': 0.2, 'thumb_1': 0.0, 'thumb_2': 0.0}
        Unlisted joints keep their current command. Steps the sim `steps` times."""
        names = list(self.art.data.joint_names)
        side = "left" if arm == 0 else "right"
        for suffix, frac in targets.items():
            jn = f"{side}_hand_{suffix}_joint"
            if jn not in names:
                raise KeyError(f"unknown hand joint {jn!r}")
            col = list(self._hand_ids).index(names.index(jn))
            lo = float(self._hand_lower[col])
            hi = float(self._hand_upper[col])
            self._hands[:, col] = lo + float(np.clip(frac, 0.0, 1.0)) * (hi - lo)
        self.step(steps)

    def open_gripper(self, arm: int = 0, steps: int = 20):
        """Open the three-finger hand (extend toward the joint lower limit)."""
        self._set_hands(0.0, steps)

    # ----------------- agent-facing: read-only introspection -----------------
    def list_objects(self):
        """Names queryable via get_state/get_object_pose."""
        return [f"leg_{i}" for i in range(4)] + [f"stud_{i}" for i in range(4)] + ["table"]

    def get_state(self, name: str):
        """Full 13-dim world root state of an object: [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]."""
        return _np(self._object_root(name)[0])

    def get_robot_state(self):
        """Dict snapshot of the robot: joint names/pos/vel/limits, wrist poses+vels, hand state,
        current control targets. All numpy copies."""
        d = self.art.data
        out = {
            "joint_names": list(d.joint_names),
            "joint_pos": _np(d.joint_pos[0]),
            "joint_vel": _np(d.joint_vel[0]),
            "hand_joint_ids": list(map(int, self._hand_ids)),
            "hand_frac": float(self._hand_frac[0]),
            "steps_used": int(self._steps_used),
            "max_steps": int(self.max_steps),
        }
        for arm, tag in ((0, "left"), (1, "right")):
            s = d.body_link_state_w[0, self._wrist_idx[arm]]
            out[f"{tag}_wrist_pose"] = _np(s[:7])
            out[f"{tag}_wrist_vel"] = _np(s[7:13])
            out[f"{tag}_wrist_target"] = _np(self._tgt[arm][0])
        return out

    def describe_scene(self, structured: bool = False):
        """Natural-language scene+goal description, or (structured=True) the USD prim dump
        [{path,type,pos,size}, ...] for arbitrary structural queries.
        WARNING: structured positions are the AUTHORED stage transforms — robot links there
        do NOT track live physics. For live robot link poses use get_link_positions()."""
        return self._env.describe_stage() if structured else self._env.describe()

    def get_link_positions(self, pattern: str = ".*hand.*"):
        """LIVE world positions of robot links whose name matches `pattern` (regex).
        Returns {link_name: pos_xyz(3,)}. Use this (not describe_scene) to see where the
        fingers/palm actually are, e.g. get_link_positions('left_hand.*')."""
        names = list(self.art.data.body_names)
        rx = _re.compile(pattern)
        return {nm: _np(self.art.data.body_link_state_w[0, i, :3])
                for i, nm in enumerate(names) if rx.search(nm)}

    # Camera presets in ENV-frame [eye, target]; world = preset + env-0 origin.
    VIEWS = {
        "default": ((1.6, 1.6, 1.9), (-0.25, 0.0, 1.0)),
        "front":   ((1.8, 0.0, 1.5), (-0.3, 0.0, 0.9)),
        "left":    ((0.2, -1.9, 1.5), (-0.3, 0.0, 0.9)),
        "right":   ((0.2, 1.9, 1.5), (-0.3, 0.0, 0.9)),
        "top":     ((0.05, 0.0, 2.7), (-0.3, 0.0, 0.8)),
        "close":   ((0.75, 0.75, 1.35), (-0.3, 0.0, 0.9)),
    }

    def _set_view(self, eye, target) -> None:
        o = _np(self.origin[0])
        self._env.sim.set_camera_view(tuple(np.asarray(eye, dtype=float) + o),
                                      tuple(np.asarray(target, dtype=float) + o),
                                      camera_prim_path="/OmniverseKit_Persp")

    def look(self, view: str = "default", eye=None, target=None) -> str:
        """Render the scene RIGHT NOW from a viewpoint and attach the image to your next
        feedback (VLM callers see it; it is also saved to the episode video). view is one of
        'default','front','left','right','top','close', or pass custom ENV-frame eye=(x,y,z),
        target=(x,y,z). Use this to visually verify the scene between plan stages."""
        cam = self._rec_cam or self._viewport_cam
        if cam is None:
            return "no camera available"
        if eye is not None or target is not None:
            vp = (tuple(eye or self.VIEWS["default"][0]), tuple(target or self.VIEWS["default"][1]))
            view = "custom"
        else:
            vp = self.VIEWS.get(view)
            if vp is None:
                return f"unknown view {view!r}; use one of {list(self.VIEWS)} or eye=/target="
        try:
            self._set_view(*vp)
            ok = self._grab_frame(cam, aux=True)
            if ok:
                self._turn_images.append({"view": view, "buf": "aux",
                                          "frame_idx": len(self._aux_frames) - 1})
        finally:
            self._set_view(*self.VIEWS["default"])  # restore the episode-video angle
        return (f"captured view {view!r} (image attached to your next feedback)"
                if ok else "capture failed")

    def _node_thumb(self) -> str:
        """Small JPEG (b64) of the current view, stored per node so the harness-side
        annotator can diff any node against its parent visually. Best-effort."""
        cam = self._rec_cam or self._viewport_cam
        if cam is None:
            return ""
        try:
            import base64
            from io import BytesIO

            from PIL import Image
            for _ in range(max(1, self._RENDER_FLUSH_INLINE)):
                self._env.sim.render()
            data = cam.get_data()
            arr = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)[:, :, :3]
            im = Image.fromarray(arr)
            w = 480
            im = im.resize((w, int(im.height * w / im.width)))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            import sys as _sys
            print("node thumb error:\n" + traceback.format_exc(), file=_sys.stderr, flush=True)
            return ""

    def _diff_context(self) -> dict:
        """Parent-vs-current oracle context for the out-of-trajectory node annotator:
        each side carries the scene summary, the structured pose snapshot, and a small
        frame. Task-generic: uses whatever scene_summary()/obs_snapshot() report."""
        def side(nd: dict) -> dict:
            def clean(x):
                if isinstance(x, dict):
                    return {k: clean(v) for k, v in x.items()}
                if isinstance(x, np.ndarray):
                    return [round(float(v), 3) for v in x.ravel().tolist()]
                if isinstance(x, (list, tuple)):
                    return [clean(v) for v in x]
                if isinstance(x, (float, np.floating)):
                    return round(float(x), 3)
                return x
            return {"scene": nd.get("scene", ""), "obs": clean(nd.get("obs", {})),
                    "thumb_b64": nd.get("thumb_b64", "")}

        cur = self._ckpt_nodes.get(self._ckpt_current)
        if cur is None:
            return {}
        parent = self._ckpt_nodes.get(cur.get("parent") or "")
        return {"current": side(cur), "parent": side(parent) if parent else {}}

    def drain_turn_payload(self) -> dict:
        """Server hook: extra per-turn response fields -- look()/review/checkpoint-render images
        (JPEG base64) + the checkpoint-tree cursor for the harness feedback footer, and the
        scene's PARTIAL-CREDIT score when it exposes one (score@k eval protocol, 2026-07-24:
        constructed problems are graded on their staged score() in [0,1], not just success)."""
        payload: dict = {}
        try:
            if hasattr(self.scene, "score"):
                payload["score"] = float(self.scene.score()[0].item())
        except Exception:  # noqa: BLE001 -- a scene score bug must not kill the turn
            traceback.print_exc()
        # Capture-quality self-report: count of frames where the FXAA readback failed
        # (should be 0 forever; anything else = temporal AA leaked into stored footage).
        if getattr(self, "_aa_violations", 0):
            payload["capture_aa_violations"] = int(self._aa_violations)
        if getattr(self, "_last_optimize_result", None) is not None:
            payload["optimize_result"] = self._last_optimize_result
            self._last_optimize_result = None
        if self._ckpt_current and "checkpoint" not in _FEATURES_OFF:
            cur = self._ckpt_nodes[self._ckpt_current]
            payload["tree"] = {"current": self._ckpt_current, "depth": cur["depth"],
                               "n_nodes": len(self._ckpt_nodes),
                               # only set when the agent CHECKPOINTED this turn — the
                               # annotator (action/state_diff/scene_diff) targets it
                               "new_node": getattr(self, "_turn_new_node", None)}
            if payload["tree"]["new_node"]:
                payload["diff_context"] = self._diff_context()
        imgs, self._turn_images = self._turn_images, []
        if not imgs:
            return payload
        out = []
        try:
            import base64
            from io import BytesIO

            from PIL import Image
            for im in imgs:
                if im.get("buf") == "aux":
                    arr = self._aux_frames[im["frame_idx"]]
                else:
                    # rec-buffer indices are GLOBAL capture indices (chunked storage)
                    got = self._frames_at([im["frame_idx"]])
                    if not got:
                        continue
                    arr = got[0]
                buf = BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG", quality=85)
                out.append({"view": im["view"],
                            "jpeg_b64": base64.b64encode(buf.getvalue()).decode()})
        except Exception:
            import sys as _sys
            print("turn image encode error:\n" + traceback.format_exc(),
                  file=_sys.stderr, flush=True)
        payload["turn_images"] = out
        return payload

    # ------------- internal: raw snapshot/restore (used by optimize + the tree) ---------
    def _ensure_park_states(self) -> None:
        """Capture the replicas' quiescent parked state once (they are at rest from
        episode reset until the first batch search de-syncs them)."""
        if self._park_states is None:
            ids = torch.arange(self.n, device=self.device, dtype=torch.long)
            self._park_states = self._env.get_states(ids)

    def _states_for_restore(self, env0_state):
        """Expand an env-0 snapshot for set_states. With replica parking active, rows
        1.. get the replicas' quiescent parked state (captured lazily while they are
        still at rest) instead of a broadcast copy of env 0. All written root states
        get their velocities zeroed (restore hygiene, 2026-07-26): residual velocities
        broadcast into many envs seed the simultaneous-jitter PhysX slow state that
        ran v19_full's post-search turns at 0.85-3.0 s/step."""
        full = expand_state_tree(env0_state, self.n)
        if (getattr(self, "_park_replicas", False)
                and not getattr(self, "_replicas_active", False)):
            self._ensure_park_states()
            _overwrite_replica_rows(full, self._park_states)
        zero_root_velocities(full)
        return full

    def snapshot(self) -> str:
        """INTERNAL. Save the CURRENT full sim state (env 0); returns a raw snapshot id."""
        self._snap_seq += 1
        sid = f"s{self._snap_seq}"
        ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self._snapshots[sid] = {"sim": self._env.get_states(ids), "api": self._api_state_env0()}
        if len(self._snapshots) > 16:
            self._snapshots.pop(next(iter(self._snapshots)))
        return sid

    def _flush_render_accum(self) -> None:
        """Clear the RTX temporal-accumulation history after a state JUMP (goto/restore).
        The per-frame inline flush (2 renders) is enough for continuous motion but not
        for teleported state: stale accumulation blends the pre-jump scene into the next
        captured frames as overlapping 'ghost' shadows (user report 2026-07-24)."""
        try:
            for _ in range(max(1, self._RENDER_FLUSH_AUX)):
                self._env.sim.render()
            # keep captures extra-flushed for a window after the jump
            self._ghost_flush_until = self._rec_total + self._GHOST_FLUSH_CAPTURES
        except Exception:  # noqa: BLE001 -- rendering must never break state ops
            import sys as _sys
            print("render-accum flush error:\n" + traceback.format_exc(),
                  file=_sys.stderr, flush=True)

    def restore(self, sid: str) -> str:
        """INTERNAL. Restore a raw snapshot (optimize's auto-restore around a search)."""
        snap = self._snapshots.get(sid)
        if snap is None:
            raise KeyError(f"unknown snapshot {sid!r}; have {list(self._snapshots)}")
        self._env.set_states(self._states_for_restore(snap["sim"]))
        self._restore_api_state(snap["api"])
        self._flush_render_accum()
        # Footage before an internal restore belongs to the abandoned attempt, not
        # to a later checkpoint edge reached from the restored state.
        self._edge_frame_start = self._rec_total
        # Tag the state JUMP with its capture offset (like goto): lets the video
        # builder slice footage frame-exactly at internal restores.
        self._event_log.append({"event": "restore", "sid": sid,
                             "frame": max(0, self._rec_total
                                          - getattr(self, "_turn_frame_start", 0))})
        return sid

    # ----------------- durable checkpoint-tree storage -----------------------------------
    def configure_persistence(self, session_id: str) -> str:
        """Enable HDFS persistence for this tree. Safe to call repeatedly."""
        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))[:120]
        if not safe:
            raise ValueError("session_id must contain at least one safe character")
        self._persist_session_id = safe
        self._persist_path = f"{self._persist_hdfs_root.rstrip('/')}/{safe}/tree_manifest.pt"
        subprocess.run(
            ["hdfs", "dfs", "-mkdir", "-p",
             f"{self._persist_path.rsplit('/', 1)[0]}/nodes"],
            capture_output=True,
            check=False,
        )
        return self._persist_path

    def _persist_tree(self, new_cid: str | None = None) -> None:
        """Publish an incremental node snapshot plus a lightweight tree manifest.

        State tensors are written once per node; captions/goto only rewrite the
        small manifest. This avoids uploading the entire growing tree every turn.
        """
        if not self._persist_path:
            return
        root = self._persist_path.rsplit("/", 1)[0]
        if new_cid:
            node = self._ckpt_nodes[new_cid]
            local_node = f"/tmp/cosigen_{self._persist_session_id}_{new_cid}.pt"
            torch.save({"sim": node["sim"], "api": node["api"]}, local_node)
            subprocess.run(
                ["hdfs", "dfs", "-put", "-f", local_node, f"{root}/nodes/{new_cid}.pt"],
                capture_output=True,
                text=True,
            )
        manifest_nodes = {
            cid: {k: v for k, v in node.items() if k not in ("sim", "api")}
            for cid, node in self._ckpt_nodes.items()
        }
        local = f"/tmp/cosigen_tree_{self._persist_session_id}_manifest.pt"
        payload = {
            "version": 1,
            "activity": getattr(self._env.cfg, "scene", ""),
            "nodes": manifest_nodes,
            "seq": self._ckpt_seq,
            "current": self._ckpt_current,
            "turn": self._turn_no,
            "total_steps": int(self._steps_used),
            "saved_at": _time.time(),
        }
        torch.save(payload, local)
        proc = subprocess.run(
            ["hdfs", "dfs", "-put", "-f", local, self._persist_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"checkpoint persistence failed: {proc.stderr}",
                flush=True,
            )

    def load_persisted_tree(self) -> str:
        """Load this session's tree, restore its current world state, and preserve total steps."""
        if not self._persist_path:
            raise RuntimeError("configure_persistence(session_id) must be called first")
        local = f"/tmp/cosigen_tree_{self._persist_session_id}.pt"
        proc = subprocess.run(
            ["hdfs", "dfs", "-get", "-f", self._persist_path, local],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return f"no persisted tree found for {self._persist_session_id}"
        payload = torch.load(local, map_location=self.device, weights_only=False)
        if int(payload.get("version", 0)) != 1:
            raise ValueError(f"unsupported persisted tree version: {payload.get('version')}")
        self._ckpt_nodes = payload["nodes"]
        root = self._persist_path.rsplit("/", 1)[0]
        for cid, node in self._ckpt_nodes.items():
            local_node = f"/tmp/cosigen_{self._persist_session_id}_{cid}.pt"
            got = subprocess.run(
                ["hdfs", "dfs", "-get", "-f", f"{root}/nodes/{cid}.pt", local_node],
                capture_output=True,
                text=True,
            )
            if got.returncode != 0:
                raise RuntimeError(f"missing persisted state for checkpoint {cid}: {got.stderr}")
            state = torch.load(local_node, map_location=self.device, weights_only=False)
            node["sim"], node["api"] = state["sim"], state["api"]
        self._ckpt_seq = int(payload["seq"])
        self._ckpt_current = payload["current"]
        self._turn_no = int(payload["turn"])
        self._steps_used = int(payload.get("total_steps", 0))
        if self._ckpt_current:
            node = self._ckpt_nodes[self._ckpt_current]
            self._env.set_states(self._states_for_restore(node["sim"]))
            self._restore_api_state(node["api"])
            # _restore_api_state deliberately does not rewind steps; restore the
            # monotonic persisted total explicitly after a server restart.
            self._steps_used = int(payload.get("total_steps", self._steps_used))
        # Capture indices restart with this server process. Any new child edge starts
        # in the current recording epoch; old checkpoint ranges retain their own epoch.
        self._edge_frame_start = self._rec_total
        return (
            f"loaded {len(self._ckpt_nodes)} checkpoints; current={self._ckpt_current}; "
            f"steps={self._steps_used}"
        )

    # ----------------- agent-facing: CHECKPOINT TREE (backtracking skill) -----------------
    # The tree is OUR data structure, separate from the agent's trajectory: nodes = world
    # states reached (auto: one per turn; manual: checkpoint()), edges = the code that led
    # there. goto(cid) jumps the WORLD to a node; the agent's variables/policies/conversation
    # are untouched and freely reusable. Everything is kept (nodes are a few KB each).
    def _caption_from_code(self, code: str | None) -> str:
        """Edge caption fallback: the agent's '# plan:' comment, else an API-call digest."""
        if not code or not code.strip():
            return "start"
        for line in code.splitlines():
            s = line.strip()
            if s.startswith("#") and "plan:" in s.lower():
                return _clip_words(s.split(":", 1)[1].strip()) or "unlabeled plan"
        calls = _re.findall(
            r"\b(move_to|open_gripper|close_gripper|step|goto|"
            r"checkpoint|look|review_rollout|render_checkpoint)\s*\(", code)
        if calls:
            seen: dict[str, int] = {}
            for c in calls:
                seen[c] = seen.get(c, 0) + 1
            return _clip_words(", ".join(f"{v}x {k}" if v > 1 else k for k, v in seen.items()))
        return _clip_words(code.strip().splitlines()[0])

    def _make_node(self, code: str | None, rc: int, label: str | None = None,
                   stdout: str = "", code_name: str = "") -> str:
        self._ckpt_seq += 1
        cid = f"n{self._ckpt_seq}"
        parent = self._ckpt_current
        ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self._ckpt_nodes[cid] = {
            "cid": cid, "parent": parent,
            "depth": (self._ckpt_nodes[parent]["depth"] + 1) if parent else 0,
            "sim": self._env.get_states(ids), "api": self._api_state_env0(),
            "code": code or "", "code_name": code_name, "rc": int(rc), "label": label,
            "caption": label or self._caption_from_code(code),
            # Three-field node annotation, filled by the out-of-trajectory annotator via
            # the meta side-channel: action (parent->node intent+outcome), state_diff
            # (oracle-state change vs parent), scene_diff (visual change vs parent; VLMs).
            "ann": {"action": label or self._caption_from_code(code),
                    "state_diff": "", "scene_diff": ""},
            # The printed log up to checkpoint time, IN FULL (zero-truncation directive
            # 2026-07-22). Never surfaced in the tree/digests (state_diff replaces it
            # there) but agent-readable on demand via get_checkpoint_log(cid). The key
            # name is historical.
            "stdout_tail": (stdout or "").strip(),
            "turn": self._turn_no,
            "success": bool(self.success()), "steps_used": int(self._steps_used),
            "scene": self.scene_summary(), "obs": self.obs_snapshot(),
            "thumb_b64": self._node_thumb(),
            # Legacy per-turn range plus native checkpoint-edge lineage. The latter is
            # the exact footage since the parent checkpoint (or most recent goto/restore),
            # and is sufficient to render root->current without turn-history heuristics.
            "frames": (self._turn_frame_start, self._rec_total),
            "edge_frames": {"epoch": self._rec_epoch,
                            "start": int(self._edge_frame_start),
                            "end": int(self._rec_total)},
            "t": _time.time(), "children": [],
        }
        if parent:
            self._ckpt_nodes[parent]["children"].append(cid)
        self._ckpt_current = cid
        self._edge_frame_start = self._rec_total
        self._persist_tree(new_cid=cid)
        return cid

    def begin_turn(self, code: str) -> None:
        """Server-side hook at the start of each executed turn."""
        if not self._ckpt_nodes:  # lazy root = the reset state, before any code ran
            self._turn_no = 0
            self._turn_frame_start = self._rec_total
            self._make_node(None, 0, label="start")
        if getattr(self, "_trial_mode", False) and self._ckpt_current:
            # Restore the anchor BEFORE running (not after): a crashed or wall-timed-out
            # turn otherwise leaves the world dirty for the next program.
            self._restore_node_state(self._ckpt_current)
        self._turn_no += 1
        self._turn_frame_start = self._rec_total
        self._turn_wall_deadline = _time.time() + self._TURN_WALL_S
        self._turn_code = code      # attached to any checkpoint the agent takes this turn
        self._turn_new_node = None  # set by checkpoint(); read by drain_turn_payload

    def capture_end_state(self) -> None:
        """Grab one image of where the program left the world, always.

        The anchor is restored at the START of a turn, so at this point the world still
        holds this run's end state and a single frame is enough to show it. Before this,
        an image existed only if the program had thought to call look() while writing it
        — 45 of 51 stepping runs in the v21 ckpt session left nothing to see, which is
        how an agent spends twenty runs on a leg that a glance would have shown lying
        flat on the table."""
        # Remember this run's footage range before the next turn overwrites it, so the
        # agent can ask for more frames of the run it just saw (the view tool).
        self._last_run_frames = (getattr(self, "_turn_frame_start", 0), self._rec_total)
        if self._turn_images:  # the program already captured something this turn
            return
        cam = self._rec_cam or self._viewport_cam
        if cam is None:
            return
        try:
            if self._grab_frame(cam, aux=True):
                self._turn_images.append({"view": "end of run", "buf": "aux",
                                          "frame_idx": len(self._aux_frames) - 1})
        except Exception:  # noqa: BLE001 -- a missing image must never fail the turn
            import sys as _sys
            print("end-state capture failed:\n" + traceback.format_exc(),
                  file=_sys.stderr, flush=True)

    def end_turn(self, code: str, rc: int, stdout: str = "") -> str | None:
        """Server-side hook after each executed turn. Creates the child node when the
        driver asked to checkpoint this program and it exited cleanly; a program that
        raised never becomes a node (the world is already back at the anchor for the
        next try). Also back-fills the log onto a node the program checkpointed itself."""
        ckpt = getattr(self, "_pending_checkpoint", None)
        if ckpt and not self._turn_new_node:
            if rc == 0:
                spec = ckpt if isinstance(ckpt, dict) else {}
                cid = self._make_node(code, rc, label=spec.get("label"), stdout=stdout,
                                      code_name=str(spec.get("name") or ""))
                self._turn_new_node = cid
            else:
                self._event_log.append({"event": "checkpoint_rejected",
                                     "reason": "program raised"})
        self._pending_checkpoint = None
        if self._turn_new_node:
            node = self._ckpt_nodes.get(self._turn_new_node)
            if node is not None:
                if not node.get("stdout_tail"):
                    node["stdout_tail"] = (stdout or "").strip()  # stored in full
                node["rc"] = int(rc)
        return self._turn_new_node

    def set_annotation(self, cid: str, ann) -> None:
        """Server-side hook: out-of-trajectory node annotation from the harness.
        ``ann`` is {"action":..., "state_diff":..., "scene_diff":...} (any subset),
        or a plain string treated as the action."""
        node = self._ckpt_nodes.get(cid)
        if node is None or not ann:
            return
        if isinstance(ann, str):
            ann = {"action": ann}
        for key in ("action", "state_diff", "scene_diff"):
            if ann.get(key):
                node["ann"][key] = _clip_words(ann[key])
        if node["ann"].get("action"):
            node["caption"] = node["ann"]["action"]  # caption stays = action
        self._persist_tree()

    @staticmethod
    def _require_feature(name: str) -> None:
        """Ablation gate: the api object is agent-reachable (raw access), so disabled
        harness features must refuse at the METHOD, not just vanish from the prompt."""
        if name in _FEATURES_OFF:
            raise RuntimeError(f"the {name} facility is not available in this session")

    def checkpoint(self, label: str | None = None) -> str:
        """Save the current world state as a node under `label`; returns its id.
        Not called from inside programs under the trial/checkpoint protocol: there, a
        node is created by checkpointing a program with the checkpoint TOOL, so that a
        node's code is exactly its parent->node edge."""
        self._require_feature("checkpoint")
        if getattr(self, "_trial_mode", False):
            raise RuntimeError(
                "checkpoint is a TOOL: call it as a tool call (like execute), with a "
                "label and the workspace path of the program to save — not from inside "
                "your program. That way the node's code is exactly the program that "
                "reaches it.")
        # Log up to THIS moment (run_policy shares its stdout buffer) — so the node's
        # log describes the state at checkpoint time, not later output from the turn.
        buf = getattr(self, "_turn_stdout_buf", None)
        stdout_now = buf.getvalue() if buf is not None else ""
        cid = self._make_node(getattr(self, "_turn_code", None), 0,
                              label=(_clip_words(label) if label else None),
                              stdout=stdout_now)
        self._turn_new_node = cid
        return cid

    def _tried_from(self, cid: str) -> str:
        """Digest of the branches ALREADY explored from a node (their intent, the exact code
        tried, and the realized outcome), so goto() reminds the agent what it tried here and
        how it turned out. Every branch shows its FULL code — zero truncation (user directive
        2026-07-26; the old 16k-char budget degraded older branches to caption+outcome)."""
        node = self._ckpt_nodes.get(cid)
        kids = node["children"] if node else []
        if not kids:
            return ""
        blocks = []
        for ch in kids:
            c = self._ckpt_nodes[ch]
            ann = c.get("ann") or {}
            action = ann.get("action") or c["label"] or c["caption"]
            outcome = ann.get("state_diff") or ""
            flags = (" [SUCCESS]" if c.get("success") else "") + (" [CRASHED]" if c["rc"] else "")
            head = f"  [{ch}] {action}{flags}"
            detail = f"\n     changed: {outcome}" if outcome else ""
            if ann.get("scene_diff"):
                detail += f"\n     visual: {ann['scene_diff']}"
            code = (c.get("code") or "").strip()
            if code:
                indented = "\n".join("       " + ln for ln in code.splitlines())
                blocks.append(f"{head}{detail}\n     code:\n{indented}")
            else:
                blocks.append(f"{head}{detail}")
        header = f"previously tried from here ({len(kids)} branch(es), newest last):"
        return header + "\n" + "\n".join(blocks)

    def goto(self, cid: str) -> str:
        """Jump the WORLD back to a checkpoint. Your variables and everything else you
        created remain available -- only the sim state changes. New actions branch the
        tree from that node. The return value REMINDS YOU which branches you already
        tried from this node and how each turned out -- do not repeat a failed
        approach; change strategy or tune the failing program's constants."""
        self._require_feature("checkpoint")
        if getattr(self, "_trial_mode", False) and not self._goto_allowed:
            # Under the trial protocol the jump belongs to the driver's goto tool, which
            # shows the node's image and log first and asks for confirmation. In-program
            # jumps skipped that: v21 agents backtracked nine times having looked at the
            # target zero times.
            raise RuntimeError(
                "goto is a TOOL: call it as a tool call (like execute), so you see the "
                "node's image and log before the world moves there.")
        node = self._ckpt_nodes.get(cid)
        if node is None:
            raise KeyError(f"unknown checkpoint {cid!r}; call list_checkpoints()")
        self._restore_node_state(cid)
        self._ckpt_current = cid
        # 'frame' = capture offset within this turn's footage: lets the path-only video
        # builder slice a mixed goto+work turn exactly at the jump (2026-07-24).
        self._event_log.append({"event": "goto", "cid": cid, "depth": node["depth"],
                             "frame": max(0, self._rec_total
                                          - getattr(self, "_turn_frame_start", 0))})
        self._persist_tree()
        ann = node.get("ann") or {}
        header = (f"world is now at {cid} (parent {node.get('parent') or 'root'}, "
                  f"\"{ann.get('action') or node['label'] or node['caption']}\")")
        if ann.get("state_diff"):
            header += f"\n  state vs its parent: {ann['state_diff']}"
        tried = self._tried_from(cid)
        return f"{header}\n{tried}" if tried else header

    def _restore_node_state(self, cid: str) -> None:
        """Put the world back at node `cid` (sim state + control targets) and start a new
        edge here, so any footage recorded from now on belongs to the next program only."""
        node = self._ckpt_nodes[cid]
        self._env.set_states(self._states_for_restore(node["sim"]))
        self._restore_api_state(node["api"])
        self._flush_render_accum()
        self._edge_frame_start = self._rec_total

    def list_checkpoints(self) -> str:
        """The checkpoint tree as text: every world state reached, what was tried from it,
        and where you are now. Use it to decide where to backtrack."""
        self._require_feature("checkpoint")
        if not self._ckpt_nodes:
            return "(no checkpoints yet)"
        lines = []

        def walk(cid: str, indent: int) -> None:
            nd = self._ckpt_nodes[cid]
            here = "   <== YOU ARE HERE" if cid == self._ckpt_current else ""
            ann = nd.get("ann") or {}
            cap = ann.get("action") or nd["label"] or nd["caption"]
            flags = (" SUCCESS" if nd.get("success") else "") + (" CRASHED" if nd["rc"] else "")
            parent = f" (parent {nd['parent']})" if nd.get("parent") else " (root)"
            lines.append(f"{'  ' * indent}{cid}{parent} t{nd['turn']}{flags} "
                         f"\"{cap}\"{here}")
            for ch in nd["children"]:
                walk(ch, indent + 1)

        roots = [c for c, nd in self._ckpt_nodes.items() if nd["parent"] is None]
        for r in roots:
            walk(r, 0)
        lines.append("(render_checkpoint(cid) to SEE any of these states before jumping)")
        return "\n".join(lines)

    def get_checkpoint_code(self, cid: str) -> str:
        """The exact code that produced a checkpoint (empty for manual/root nodes)."""
        self._require_feature("checkpoint")
        return self._ckpt_nodes[cid]["code"]

    def get_checkpoint_scene(self, cid: str) -> dict:
        """What the world looked like at a checkpoint: scene text, pose snapshot, and the
        node's annotation (action / state_diff / scene_diff)."""
        self._require_feature("checkpoint")
        nd = self._ckpt_nodes[cid]
        return {"scene": nd["scene"], "objects": copy.deepcopy(nd["obs"]),
                "annotation": dict(nd.get("ann") or {}), "depth": nd["depth"]}

    def get_checkpoint_log(self, cid: str) -> str:
        """The printed log (your prints / oracle readouts) up to the moment this checkpoint
        was taken, in full. Read it to reconstruct exactly what had happened by then."""
        self._require_feature("checkpoint")
        return self._ckpt_nodes[cid].get("stdout_tail") or "(no log recorded at this checkpoint)"

    def render_checkpoint(self, cid: str) -> str:
        """SEE a past checkpoint: briefly restore it, render one frame (attached to your next
        feedback like look()), then return to the current state."""
        self._require_feature("checkpoint")
        node = self._ckpt_nodes.get(cid)
        if node is None:
            raise KeyError(f"unknown checkpoint {cid!r}")
        cam = self._rec_cam or self._viewport_cam
        if cam is None:
            return "no camera available"
        ids = torch.tensor([0], device=self.device, dtype=torch.long)
        cur = {"sim": self._env.get_states(ids), "api": self._api_state_env0()}
        try:
            self._env.set_states(self._states_for_restore(node["sim"]))
            ok = self._grab_frame(cam, aux=True)
            if ok:
                self._turn_images.append({"view": f"checkpoint {cid}", "buf": "aux",
                                          "frame_idx": len(self._aux_frames) - 1})
        finally:
            self._env.set_states(self._states_for_restore(cur["sim"]))
            self._restore_api_state(cur["api"])
        return (f"rendered {cid} (image attached to your next feedback)" if ok
                else "render failed")

    def review_last(self, k: int = 4) -> str:
        """Sample k frames evenly from the run just executed, checkpointed or not.

        review_rollout works off a node's frame range, so it can only show runs that were
        saved; this reads the range recorded at the end of the last run, which is what the
        agent actually wants to look at."""
        f0, f1 = getattr(self, "_last_run_frames", (0, 0))
        if f1 <= f0:
            return "the last run recorded no frames (it did not move anything)"
        idxs = np.linspace(f0, f1 - 1, min(int(k), f1 - f0)).astype(int)
        for i in idxs:
            self._turn_images.append({"view": f"frame {int(i) - int(f0)} of the last run",
                                      "buf": "rec", "frame_idx": int(i)})
        return f"{len(idxs)} frames from the last run attached"

    def review_rollout(self, k: int = 4, cid: str | None = None) -> str:
        """Sample k frames evenly from the video recorded during a turn (default: the last
        executed turn) and attach them to your next feedback -- watch what your code did."""
        nd = self._ckpt_nodes.get(cid or self._ckpt_current or "")
        if nd is None:
            return "no checkpoints yet"
        # Frame ranges are GLOBAL capture indices; footage is chunked to disk and never
        # dropped, so ANY turn's rollout — however old — remains reviewable.
        f0, f1 = nd["frames"]
        if f1 <= f0:
            return "no frames recorded during that turn (video off or no sim steps)"
        idxs = np.linspace(f0, f1 - 1, min(int(k), f1 - f0)).astype(int)
        for i in idxs:
            self._turn_images.append({"view": f"rollout frame {int(i)}", "buf": "rec",
                                      "frame_idx": int(i)})
        return f"{len(idxs)} rollout frames attached to your next feedback"

    def apply_turn_meta(self, meta: dict) -> None:
        """Server-side hook for harness->api side-channel data (NOT agent-facing): e.g.
        {'annotations': {cid: caption}} from the out-of-trajectory annotation call, and
        {'max_steps': N} to set the episode step budget per session (the server's boot-time
        default is otherwise fixed; unbounded-turn sessions need a much larger budget).
        A session_id enables durable HDFS checkpoint persistence; resume_tree restores it."""
        if meta.get("session_id"):
            self.configure_persistence(str(meta["session_id"]))
            if meta.get("resume_tree"):
                msg = self.load_persisted_tree()
                self._event_log.append({"event": "load_persisted_tree", "message": msg})
        for cid, ann in (meta.get("annotations") or {}).items():
            self.set_annotation(cid, ann)
        if meta.get("max_steps"):
            self.max_steps = int(meta["max_steps"])
        if "trial_mode" in meta:
            # Trial/checkpoint protocol (2026-07-25): every execution starts from the
            # current node's state, so a node's code is exactly the program that leads
            # from its parent to it, and a node's footage is exactly that program's run.
            self._trial_mode = bool(meta["trial_mode"])
        # A checkpoint request turns THIS execution into the edge parent->child: the node
        # is created only if the program exits without error (see end_turn).
        self._pending_checkpoint = meta.get("checkpoint") or None
        # This turn is the driver's goto tool acting on a confirmed node.
        self._goto_allowed = bool(meta.get("goto_allowed"))
        # An optimize request replaces THIS turn's code execution with a parameter
        # search over the shipped program/objective sources (driver's optimize tool).
        self._pending_optimize = meta.get("optimize") or None
        if meta.get("arm_recorder"):
            # Re-arm inline video capture mid-session: the server only arms the recorder in
            # its episode-reset branch, so reset=false turns could never (re)enable video.
            cam = self._rec_cam or self._viewport_cam
            if cam is not None:
                spec = meta["arm_recorder"] if isinstance(meta["arm_recorder"], dict) else {}
                self.enable_recording(cam, every=int(spec.get("every", 15)))

    # ----------------- agent-facing: parameter search (the optimize tool) -----------------
    def optimize_program(self, program_src: str, objective_src: str, space: dict,
                         **kw) -> dict:
        """THE agent-facing parameter search (2026-07-25, user-mandated interface): the
        agent's own single-env program file runs as one copy per env (lockstep
        scheduler in cosigen_opt), each copy with its own values for the top-level
        constants named in `space`, scored by objective(v) from the objective file.
        The driver's `optimize` tool reads both files and lands here via turn meta."""
        self._require_feature("opt")
        O = _import_sibling("cosigen_opt")
        # Capture the parked reference before the batch de-syncs the replicas, and
        # restore through the parking-aware snapshot path so replicas return to their
        # quiescent poses (a plain env-0 broadcast leaves the replica arms fighting
        # their parked targets). This bracketing is the proven 512-env path.
        if getattr(self, "_park_replicas", False):
            self._ensure_park_states()
        sid = self.snapshot()
        # a search is a legitimately long turn: give the runaway-turn guard its budget
        if getattr(self, "_turn_wall_deadline", None):
            self._turn_wall_deadline += float(kw.get("budget_s", 3600.0))

        # Per-generation progress is published here so the SERVER can serve it from its
        # HTTP thread while this (main-thread) search is still running — that is what
        # makes the driver's optimize tool async: the agent keeps working and reads
        # results as each generation lands, instead of blocking for the whole budget.
        meta = dict(kw.pop("progress_meta", None) or {})
        progress_hdfs = str(kw.pop("progress_hdfs", "") or "")
        self._opt_progress = {"state": "running", "started": _time.time(), **meta}

        def _publish(record: dict) -> None:
            self._opt_progress = {"state": "running", "started": self._opt_progress
                                  .get("started"), **meta, **record}
            self._publish_opt_progress(progress_hdfs)

        try:
            out = O.optimize_program(self._env, self, program_src=program_src,
                                     objective_src=objective_src, space=space,
                                     progress_cb=_publish, **kw)
            self._opt_progress = {"state": "done", **meta, **out}
            self._publish_opt_progress(progress_hdfs)
            return out
        except Exception as exc:  # noqa: BLE001 -- publish the failure too
            self._opt_progress = {"state": "failed", "error": repr(exc), **meta}
            self._publish_opt_progress(progress_hdfs)
            raise
        finally:
            self.restore(sid)

    def _publish_opt_progress(self, hdfs_path: str) -> None:
        """Mirror live search progress to HDFS. The /ping payload is the primary channel,
        but it lives in the render SERVER file, which is loaded at pod boot and never
        hot-reloads — so on a reloaded pod the driver would see no progress at all. HDFS
        is written from this (hot-reloadable) module, so streaming works on every pod."""
        if not hdfs_path:
            return
        try:
            import json as _json
            local = f"/tmp/opt_progress_{os.getpid()}.json"
            with open(local, "w") as fh:
                _json.dump(self._opt_progress, fh, default=str)
            os.system(f"hdfs dfs -mkdir -p {os.path.dirname(hdfs_path)} 2>/dev/null; "
                      f"hdfs dfs -put -f {local} {hdfs_path} >/dev/null 2>&1")
        except Exception:  # noqa: BLE001 -- streaming must never break a search
            import sys as _sys
            print("optimize progress publish failed:\n" + traceback.format_exc(),
                  file=_sys.stderr, flush=True)

    def optimize(self, *_a, **_kw) -> dict:
        """Namespace guard: optimize is a TOOL (one name, always parallel). A program
        calling it in code cannot ship its workspace files to the simulator host, so
        redirect the agent to the tool call."""
        raise RuntimeError(
            "optimize is a TOOL: call it as a tool call (like execute and checkpoint), "
            "not from inside a program — your workspace files are not visible to "
            "the simulator host, so the tool must ship them here for you.")

    def drain_event_log(self) -> list[dict]:
        log, self._event_log = self._event_log, []
        return log

    # Wire-compat alias (2026-07-26 rename): a pod's running server file is loaded at
    # boot and cannot hot-reload, so servers older than the rename still call
    # api.drain_rl_log() after a /reload swaps in this module. Drop this alias once
    # every pod has been rebooted on the renamed server.
    drain_rl_log = drain_event_log

    # ----------------- namespace / prompt / grading -----------------
    def functions(self) -> dict:
        return {
            # control (the ONLY way to affect the world)
            "get_object_pose": self.get_object_pose,
            "get_eef_pose": self.get_eef_pose,
            "get_seated": self.get_seated,
            "move_to": self.move_to,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "set_hand_joints": self.set_hand_joints,
            "step": self.step,
            # read-only introspection
            "list_objects": self.list_objects,
            "get_state": self.get_state,
            "get_robot_state": self.get_robot_state,
            "get_link_positions": self.get_link_positions,
            "describe_scene": self.describe_scene,
            # No look() any more: every run that moves the world returns an image of
            # where it ended, and the view tool samples more frames on request. These two
            # stay reachable because the driver's view/goto tools call them on the pod.
            "review_last": self.review_last,
            "review_rollout": self.review_rollout,
            # checkpoint tree (backtracking)
            "checkpoint": self.checkpoint,
            "goto": self.goto,
            "list_checkpoints": self.list_checkpoints,
            "get_checkpoint_code": self.get_checkpoint_code,
            "get_checkpoint_scene": self.get_checkpoint_scene,
            "get_checkpoint_log": self.get_checkpoint_log,
            "render_checkpoint": self.render_checkpoint,
            # parameter search: `optimize` is a TOOL; the namespace entry only
            # redirects in-program calls to it (workspace files live driver-side)
            "optimize": self.optimize,
        }

    def api_doc(self) -> str:
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS="'leg_0'..'leg_3','stud_0'..'stud_3','table' (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_G1_ASSEMBLY,
        )
    def success(self) -> bool:
        return bool(self.scene.seated()[0].all().item())

    def scene_summary(self) -> str:
        lines = [f"seated={self.get_seated()}  steps_used={self._steps_used}/{self.max_steps}"]
        for i in range(4):
            p, _ = self.get_object_pose(f"leg_{i}")
            s, _ = self.get_object_pose(f"stud_{i}")
            lines.append(f"leg_{i} pos={np.round(p, 3).tolist()}  stud_{i} pos={np.round(s, 3).tolist()}")
        return "\n".join(lines)

    def obs_snapshot(self) -> dict:
        return {
            "legs": {f"leg_{i}": self.get_object_pose(f"leg_{i}")[0] for i in range(4)},
            "studs": {f"stud_{i}": self.get_object_pose(f"stud_{i}")[0] for i in range(4)},
            "eef": {arm: self.get_eef_pose(arm)[0] for arm in (0, 1)},
            "seated": self.get_seated(),
        }


class Gr1t2HandMixin:
    """GR1T2 (Fourier 6-DOF hand) overrides for the AssemblyApi machinery.

    Differences from G1 handled here:
      - pink_ik frames track the HAND PITCH links (not wrist_yaw; G1 naming crashes find_bodies);
      - hand joints are named L_/R_<suffix>_joint with suffixes like index_proximal;
      - fingers curl toward the LOWER limit (0 = straight): the generic lower->upper
        fallback would command an INVERTED hand (the G1 left-hand bug all over again);
      - thumb: proximal_yaw rotates the thumb across the palm (0 = aside, negative =
        opposing), proximal_pitch/distal curl positive."""

    WRIST_BODIES = ("left_hand_pitch_link", "right_hand_pitch_link")

    # Per-suffix OPEN/CLOSED joint targets (rad). Same sign convention on both hands
    # (URDF L_/R_ limits are identical — verified 2026-07-16).
    _GR1T2_HAND_OPEN = {
        "index_proximal": 0.0, "index_intermediate": 0.0,
        "middle_proximal": 0.0, "middle_intermediate": 0.0,
        "ring_proximal": 0.0, "ring_intermediate": 0.0,
        "pinky_proximal": 0.0, "pinky_intermediate": 0.0,
        "thumb_proximal_yaw": -0.5, "thumb_proximal_pitch": 0.0, "thumb_distal": 0.0,
    }
    _GR1T2_HAND_CLOSED = {
        "index_proximal": -1.3, "index_intermediate": -1.5,
        "middle_proximal": -1.3, "middle_intermediate": -1.5,
        "ring_proximal": -1.3, "ring_intermediate": -1.5,
        "pinky_proximal": -1.3, "pinky_intermediate": -1.5,
        "thumb_proximal_yaw": -1.0, "thumb_proximal_pitch": 1.1, "thumb_distal": 1.2,
    }

    def _init_hand_limits(self) -> None:
        lim = None
        for attr in ("joint_pos_limits", "soft_joint_pos_limits", "default_joint_pos_limits"):
            if hasattr(self.art.data, attr):
                lim = getattr(self.art.data, attr)[0, self._hand_ids]
                break
        if lim is None:
            raise RuntimeError("no joint position limits on articulation data")
        self._hand_lower = lim[:, 0].clone()
        self._hand_upper = lim[:, 1].clone()
        open_t = torch.zeros_like(self._hand_lower)
        closed_t = torch.zeros_like(self._hand_lower)
        names = list(self.art.data.joint_names)
        for col, jid in enumerate(self._hand_ids):
            m = _re.match(r"[LR]_(\w+)_joint", names[jid])
            suffix = m.group(1) if m else None
            lo, hi = float(self._hand_lower[col]), float(self._hand_upper[col])
            if suffix in self._GR1T2_HAND_OPEN:
                open_t[col] = float(np.clip(self._GR1T2_HAND_OPEN[suffix], lo, hi))
                closed_t[col] = float(np.clip(self._GR1T2_HAND_CLOSED[suffix], lo, hi))
            else:  # unknown joint: hold at current default
                cur = float(self.art.data.default_joint_pos[0, jid])
                open_t[col] = closed_t[col] = cur
        self._hand_open_t = open_t
        self._hand_closed_t = closed_t

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        """PER-JOINT hand control, GR1T2 naming. `targets` maps suffixes (index_proximal,
        index_intermediate, ..., thumb_proximal_yaw, thumb_proximal_pitch, thumb_distal)
        to FRACTIONS in [0,1] of that joint's OPEN->CLOSED stroke (not raw limits)."""
        names = list(self.art.data.joint_names)
        side = "L" if arm == 0 else "R"
        hand_cols = {names[jid]: col for col, jid in enumerate(self._hand_ids)}
        for suffix, frac in targets.items():
            jn = f"{side}_{suffix}_joint"
            if jn not in hand_cols:
                raise KeyError(f"unknown hand joint {jn!r}; suffixes={list(self._GR1T2_HAND_OPEN)}")
            col = hand_cols[jn]
            o, cl = float(self._hand_open_t[col]), float(self._hand_closed_t[col])
            self._hands[:, col] = o + float(np.clip(frac, 0.0, 1.0)) * (cl - o)
        self.step(steps)


class FrankaTabletopApi(AssemblyApi):
    """Control API for the Franka tabletop assembly scenes (nut_thread / bulb) under the
    task-space 'osc' control mode (action = 6 EE pose deltas + 2 gripper).

    VIRTUAL-TARGET architecture: we keep a persistent EE pose target `_tgt[0]` (world frame)
    exactly like the G1 API keeps wrist targets, and `step()` converts (target - live EE pose)
    into clipped OSC deltas each control step. This keeps move_to / the checkpoint tree /
    the optimizer's per-env facades semantically IDENTICAL across embodiments (everything
    drives the same virtual target). Single arm: arm arg accepted but ignored."""

    # Camera presets anchored on the FRANKA TABLETOP workspace (table at env-frame
    # (0.55, 0); nut/bolt near z~0.1). The ikea presets aim 0.85m away -> filmed floor.
    VIEWS = {
        "default": ((1.55, 1.05, 0.85), (0.5, 0.0, 0.12)),
        "front":   ((1.7, 0.0, 0.7), (0.5, 0.0, 0.1)),
        "left":    ((0.55, -1.3, 0.8), (0.55, 0.0, 0.1)),
        "right":   ((0.55, 1.3, 0.8), (0.55, 0.0, 0.1)),
        "top":     ((0.6, 0.02, 1.6), (0.55, 0.0, 0.1)),
        "close":   ((0.95, 0.45, 0.45), (0.55, 0.0, 0.1)),
    }

    def __init__(self, env, max_steps: int = 3000):
        self._init_common(env, max_steps)
        ee_idx = self.art.find_bodies(env.robot.EE_BODY)[0][0]
        self._wrist_idx = [ee_idx, ee_idx]  # View.eef_pose(arm) works for arm 0 or 1
        self._hand_ids = self.art.find_joints(list(env.robot.GRIPPER_JOINTS))[0]
        self._init_hand_limits()
        # OSC action scaling (must mirror the controller cfg to convert target error -> action)
        ctrl = env.robot.controller
        arm_ctrl = getattr(ctrl, "controllers", [ctrl])[0]
        self._pos_scale = float(getattr(arm_ctrl.cfg, "pos_scale", 0.02))
        self._rot_scale = float(getattr(arm_ctrl.cfg, "rot_scale", 0.097))
        try:  # aim the episode camera at THIS scene's workspace (safe if renderer not up yet)
            self._set_view(*self.VIEWS["default"])
        except Exception:
            pass
        self.reset_state()

    # -- state: single virtual EE target + gripper fraction ------------------------------
    def reset_state(self):
        ee = self.art.data.body_link_state_w[:, self._wrist_idx[0], :7].clone()
        self._tgt = [ee, ee.clone()]  # [0] is authoritative; [1] aliases for arm-agnostic code
        pos = self.art.data.joint_pos[:, self._hand_ids]
        span = (self._hand_upper - self._hand_lower).clamp_min(1e-6)
        # hand_frac: 0 = open (fingers at upper/wide), 1 = closed (lower). Panda fingers:
        # joint value == finger opening, so closed = lower limit.
        self._hand_frac = (1.0 - (pos - self._hand_lower) / span).mean(dim=1).clamp(0.0, 1.0)
        self._hands = self._hand_lower + (1.0 - self._hand_frac).unsqueeze(1) * span
        self._steps_used = 0
        self._policy_last_action = None

    def _restore_api_state(self, s: dict) -> None:
        for arm in (0, 1):
            self._tgt[arm][:] = s["tgt"][arm].repeat(self.n, 1)
        self._hand_frac[:] = s["hand_frac"].repeat(self.n)
        span = (self._hand_upper - self._hand_lower).clamp_min(1e-6)
        self._hands = self._hand_lower + (1.0 - self._hand_frac).unsqueeze(1) * span

    def set_hand_frac_tensor(self, frac: torch.Tensor) -> None:
        """Panda fingers: joint value == opening width, so closed = LOWER limit."""
        self._hand_frac = frac
        span = (self._hand_upper - self._hand_lower).clamp_min(1e-6)
        self._hands = self._hand_lower + (1.0 - self._hand_frac).unsqueeze(1) * span

    def _set_hands(self, frac: float, steps: int):
        self.set_hand_frac_tensor(torch.full_like(self._hand_frac, float(np.clip(frac, 0.0, 1.0))))
        self.step(steps)

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        """Franka override (the inherited G1 joint naming cannot address the Panda fingers).
        `targets` maps 'finger_1'/'finger_2' (or 'width' for the total jaw opening in m,
        0..0.08) to a fraction in [0,1] of that finger joint's [lower, upper] range
        (0 = closed, 1 = fully open). Partial openings enable thin-edge pinches where a
        fully open jaw would collide with the support surface. Steps the sim `steps` times."""
        for key, val in targets.items():
            if key == "width":
                frac = float(np.clip(val / 0.08, 0.0, 1.0))
                cols = range(len(self._hand_ids))
            elif key in ("finger_1", "finger_2"):
                frac = float(np.clip(val, 0.0, 1.0))
                cols = [int(key[-1]) - 1]
            else:
                raise KeyError(f"unknown franka gripper key {key!r}; use finger_1/finger_2/width")
            for col in cols:
                lo, hi = float(self._hand_lower[col]), float(self._hand_upper[col])
                self._hands[:, col] = lo + frac * (hi - lo)
        self.step(steps)

    def _action(self):
        """(N, 8) OSC action: clipped deltas from live EE pose toward the virtual target,
        plus direct gripper finger targets."""
        from isaaclab.utils.math import quat_mul  # noqa: PLC0415

        ee = self.art.data.body_link_state_w[:, self._wrist_idx[0], :7]
        tgt = self._tgt[0]
        dpos = ((tgt[:, :3] - ee[:, :3]) / max(self._pos_scale, 1e-6)).clamp(-1.0, 1.0)
        # orientation error as axis-angle of tgt_quat * conj(ee_quat)
        q_ee = ee[:, 3:7]
        q_conj = torch.cat([q_ee[:, :1], -q_ee[:, 1:]], dim=1)
        q_err = quat_mul(tgt[:, 3:7], q_conj)
        w = q_err[:, 0].clamp(-1.0, 1.0)
        angle = 2.0 * torch.acos(w.abs())
        axis = q_err[:, 1:] * torch.sign(w).unsqueeze(1)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
        drot = (axis * angle.unsqueeze(1) / max(self._rot_scale, 1e-6)).clamp(-1.0, 1.0)
        return torch.cat([dpos, drot, self._hands], dim=1)

    # -- objects ---------------------------------------------------------------------------
    def _pairs(self):
        """(loose_objects, fixed_objects, loose_name, fixed_name) for the current scene."""
        s = self.scene
        if hasattr(s, "nuts"):
            return s.nuts, s.bolts, "nut", "bolt"
        if hasattr(s, "bulbs"):
            return s.bulbs, s.sockets, "bulb", "socket"
        raise RuntimeError(f"unsupported franka scene {type(s).__name__}")

    def _object_root(self, name: str) -> torch.Tensor:
        loose, fixed, ln, fn = self._pairs()
        if name.startswith(ln + "_"):
            return loose[int(name.rsplit("_", 1)[1])].data.root_state_w
        if name.startswith(fn + "_"):
            return fixed[int(name.rsplit("_", 1)[1])].data.root_state_w
        if name == "table" and hasattr(self.scene, "table"):
            return self.scene.table.data.root_state_w
        raise ValueError(f"unknown object {name!r}; use {ln}_i / {fn}_i")

    def list_objects(self):
        loose, fixed, ln, fn = self._pairs()
        return ([f"{ln}_{i}" for i in range(len(loose))]
                + [f"{fn}_{i}" for i in range(len(fixed))])

    def get_seated(self):
        return [bool(v) for v in self.scene.seated()[0].tolist()]

    def success(self) -> bool:
        return bool(self.scene.seated()[0].all().item())

    def scene_summary(self) -> str:
        loose, fixed, ln, fn = self._pairs()
        lines = [f"seated={self.get_seated()}  steps_used={self._steps_used}/{self.max_steps}"]
        for i in range(len(loose)):
            p, _ = self.get_object_pose(f"{ln}_{i}")
            q, _ = self.get_object_pose(f"{fn}_{i}")
            lines.append(f"{ln}_{i} pos={np.round(p, 3).tolist()}  {fn}_{i} pos={np.round(q, 3).tolist()}")
        ee, _ = self.get_eef_pose(0)
        lines.append(f"ee pos={np.round(ee, 3).tolist()}  gripper_closed_frac={float(self._hand_frac[0]):.2f}")
        return "\n".join(lines)

    def obs_snapshot(self) -> dict:
        loose, fixed, ln, fn = self._pairs()
        return {
            "loose": {f"{ln}_{i}": self.get_object_pose(f"{ln}_{i}")[0] for i in range(len(loose))},
            "fixed": {f"{fn}_{i}": self.get_object_pose(f"{fn}_{i}")[0] for i in range(len(fixed))},
            "eef": {0: self.get_eef_pose(0)[0]},
            "seated": self.get_seated(),
        }

    def debug_step_raw(self, action6, n: int = 100):
        """TEMPORARY diagnostic: bypass the virtual-target layer and feed a RAW 6-dim OSC
        action (+ current gripper) straight to env.step for n control steps. Returns the
        EE pose before/after + joint7 before/after."""
        a = torch.tensor(np.asarray(action6, dtype=np.float32), device=self.device).reshape(1, 6)
        a = a.repeat(self.n, 1).clamp(-1.0, 1.0)
        j7 = self.art.find_joints(["panda_joint7"])[0]
        before = (_np(self.art.data.body_link_state_w[0, self._wrist_idx[0], :7]),
                  float(self.art.data.joint_pos[0, j7[0]]))
        for _ in range(int(n)):
            self._env.step(torch.cat([a, self._hands], dim=1), render=False)
            self._steps_used += 1
        after = (_np(self.art.data.body_link_state_w[0, self._wrist_idx[0], :7]),
                 float(self.art.data.joint_pos[0, j7[0]]))
        return {"before_pose": before[0].round(4).tolist(), "after_pose": after[0].round(4).tolist(),
                "j7_before": round(before[1], 4), "j7_after": round(after[1], 4)}

    def api_doc(self) -> str:
        loose, fixed, ln, fn = self._pairs()
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS=f"{ln}_0..{ln}_{len(loose)-1}, {fn}_0..{fn}_{len(fixed)-1} (WORLD)",
            SEATED_LINE=(f"  get_seated() -> [bool x{len(loose)}]             "
                         f"# per-{ln} seated; task done when all True"),
            N_ENVS=str(self.n),
            EMBODIMENT=_emb_screw(EMB_FRANKA_SINGLE, ln, fn),
        )


class KukaAllegroApi(FrankaTabletopApi):
    """Control API for the Kuka iiwa7 + Allegro DEXTEROUS HAND on the assembly tabletop
    scenes. FrankaTabletopApi's virtual-target machinery drives the palm; the 16 hand
    joints are per-joint position targets with named per-finger control instead of a
    single open/close width.

    Hand model: fingers index/middle/ring/thumb, joints 0..3 each. joint_0 = ab/adduction
    (thumb: opposition), joints 1..3 = proximal/middle/distal curl. `set_hand_frac`
    interpolates every joint between the OPEN and CLOSED postures below; `set_hand_joints`
    sets individual joints by name fraction (0 = open target, 1 = closed target)."""

    # Per-joint OPEN/CLOSED targets (rad). OPEN = the preset ready pose (slight curl,
    # thumb opposed); CLOSED = a full power wrap. Fractions interpolate between them.
    _ALLEGRO_OPEN = {
        "index_joint_0": 0.0, "index_joint_1": 0.3, "index_joint_2": 0.3, "index_joint_3": 0.3,
        "middle_joint_0": 0.0, "middle_joint_1": 0.3, "middle_joint_2": 0.3, "middle_joint_3": 0.3,
        "ring_joint_0": 0.0, "ring_joint_1": 0.3, "ring_joint_2": 0.3, "ring_joint_3": 0.3,
        "thumb_joint_0": 1.5, "thumb_joint_1": 0.6, "thumb_joint_2": 0.34, "thumb_joint_3": 0.6,
    }
    _ALLEGRO_CLOSED = {
        "index_joint_0": 0.0, "index_joint_1": 1.25, "index_joint_2": 1.35, "index_joint_3": 1.2,
        "middle_joint_0": 0.0, "middle_joint_1": 1.25, "middle_joint_2": 1.35, "middle_joint_3": 1.2,
        "ring_joint_0": 0.0, "ring_joint_1": 1.25, "ring_joint_2": 1.35, "ring_joint_3": 1.2,
        "thumb_joint_0": 1.35, "thumb_joint_1": 1.1, "thumb_joint_2": 1.0, "thumb_joint_3": 1.0,
    }

    def _init_hand_limits(self) -> None:
        lim = None
        for attr in ("joint_pos_limits", "soft_joint_pos_limits", "default_joint_pos_limits"):
            if hasattr(self.art.data, attr):
                lim = getattr(self.art.data, attr)[0, self._hand_ids]
                break
        if lim is None:
            raise RuntimeError("no joint position limits on articulation data")
        self._hand_lower = lim[:, 0].clone()
        self._hand_upper = lim[:, 1].clone()
        names = list(self.art.data.joint_names)
        open_t = torch.zeros_like(self._hand_lower)
        closed_t = torch.zeros_like(self._hand_lower)
        for col, jid in enumerate(self._hand_ids):
            nm = names[jid]
            lo, hi = float(self._hand_lower[col]), float(self._hand_upper[col])
            open_t[col] = float(np.clip(self._ALLEGRO_OPEN.get(nm, 0.0), lo, hi))
            closed_t[col] = float(np.clip(self._ALLEGRO_CLOSED.get(nm, 0.0), lo, hi))
        self._hand_open_t = open_t
        self._hand_closed_t = closed_t

    # -- hand state: fraction interpolates open->closed posture ---------------------------
    def reset_state(self):
        ee = self.art.data.body_link_state_w[:, self._wrist_idx[0], :7].clone()
        self._tgt = [ee, ee.clone()]
        pos = self.art.data.joint_pos[:, self._hand_ids]
        denom = (self._hand_closed_t - self._hand_open_t)
        safe = denom.abs() > 1e-4
        per_joint = torch.where(safe.unsqueeze(0), (pos - self._hand_open_t) / denom,
                                torch.zeros_like(pos))
        self._hand_frac = per_joint.mean(dim=1).clamp(0.0, 1.0)
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))
        self._steps_used = 0
        self._policy_last_action = None

    def _restore_api_state(self, s: dict) -> None:
        for arm in (0, 1):
            self._tgt[arm][:] = s["tgt"][arm].repeat(self.n, 1)
        self._hand_frac[:] = s["hand_frac"].repeat(self.n)
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))

    def set_hand_frac_tensor(self, frac: torch.Tensor) -> None:
        self._hand_frac = frac
        self._hands = (self._hand_open_t
                       + self._hand_frac.unsqueeze(1) * (self._hand_closed_t - self._hand_open_t))

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        """PER-JOINT dexterous control. `targets` maps Allegro joint names
        ('index_joint_1', 'thumb_joint_0', ... — or a finger prefix like 'index' to set
        that finger's joints 1..3 together) to FRACTIONS in [0,1] of that joint's
        OPEN->CLOSED stroke. `arm` accepted for API symmetry (single hand)."""
        names = [list(self.art.data.joint_names)[j] for j in self._hand_ids]
        col_of = {nm: c for c, nm in enumerate(names)}
        expanded: dict[str, float] = {}
        for key, frac in targets.items():
            if key in col_of:
                expanded[key] = frac
            elif any(nm.startswith(key + "_joint") for nm in names):
                for j in (1, 2, 3):
                    expanded[f"{key}_joint_{j}"] = frac
            else:
                raise KeyError(f"unknown hand joint/finger {key!r}; joints={names}")
        for nm, frac in expanded.items():
            col = col_of[nm]
            o, cl = float(self._hand_open_t[col]), float(self._hand_closed_t[col])
            self._hands[:, col] = o + float(np.clip(frac, 0.0, 1.0)) * (cl - o)
        self.step(steps)

    def get_hand_joints(self) -> dict:
        """Live hand joint positions (rad) AND their open->closed fractions, by name."""
        names = [list(self.art.data.joint_names)[j] for j in self._hand_ids]
        pos = self.art.data.joint_pos[0, self._hand_ids]
        out = {}
        for c, nm in enumerate(names):
            o, cl = float(self._hand_open_t[c]), float(self._hand_closed_t[c])
            frac = (float(pos[c]) - o) / (cl - o) if abs(cl - o) > 1e-4 else 0.0
            out[nm] = {"rad": round(float(pos[c]), 4), "frac": round(frac, 3)}
        return out

    def move_to(self, arm: int, position, quaternion_wxyz=None, max_steps: int = 300,
                pos_tol: float = 0.02, max_step: float = 0.01):
        """Base move_to + a LIVE-tracking settle. The iiwa7 under OSC (EMA 0.2, heavy
        links) lags the virtual target by 10-20 cm mid-move; the base loop exits when
        the TARGET reaches the goal, which returned 0.2 m 'residuals' that were pure
        transit lag (diag_08: the same pose converged to 6 mm given settle time).
        Here we keep stepping until the LIVE palm converges or the budget runs out."""
        d = super().move_to(arm, position, quaternion_wxyz=quaternion_wxyz,
                            max_steps=max_steps, pos_tol=pos_tol, max_step=max_step)
        goal = torch.tensor(np.asarray(position, dtype=np.float64), device=self.device,
                            dtype=torch.float32).reshape(1, 3)
        for _ in range(int(max_steps)):
            live = self.art.data.body_link_state_w[0:1, self._wrist_idx[arm], :3]
            d = float((live - goal).norm())
            if d <= pos_tol * 1.2 or self._steps_used >= self.max_steps:
                break
            self.step(1)
        return d

    def set_hand_frac(self, arm: int, frac: float, steps: int = 30):
        """Whole-hand posture fraction: 0 = open/ready, 1 = full power wrap."""
        self._set_hands(float(frac), int(steps))

    def functions(self) -> dict:
        fns = super().functions()
        fns["get_hand_joints"] = self.get_hand_joints
        fns["set_hand_frac"] = self.set_hand_frac
        return fns

    def api_doc(self) -> str:
        loose, fixed, ln, fn = self._pairs()
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS=f"{ln}_0..{ln}_{len(loose)-1}, {fn}_0..{fn}_{len(fixed)-1} (WORLD)",
            SEATED_LINE=(f"  get_seated() -> [bool x{len(loose)}]             "
                         f"# per-{ln} seated; task done when all True"),
            N_ENVS=str(self.n),
            EMBODIMENT=_emb_screw(EMB_KUKA_ALLEGRO, ln, fn),
        )


class PackingApi(AssemblyApi):
    """Control API for the packing suite's crate task (G1/GR1T2 embodiments). Reuses the
    G1 arm/hand machinery from AssemblyApi; overrides the OBJECT world (manifest cargo +
    lid instead of legs/studs/table) and the success/summary/verifier surface."""

    # Camera presets framed on the bench-height crate workspace (env frame).
    VIEWS = {
        "default": ((1.5, -1.5, 1.7), (0.0, 0.1, 0.75)),
        "front":   ((0.0, -1.7, 1.5), (0.0, 0.15, 0.8)),
        "left":    ((-1.7, -0.4, 1.5), (0.0, 0.15, 0.8)),
        "right":   ((1.7, -0.4, 1.5), (0.0, 0.15, 0.8)),
        "top":     ((0.0, 0.1, 2.6), (0.0, 0.15, 0.75)),
        "close":   ((0.8, -0.8, 1.3), (0.0, 0.15, 0.85)),
    }

    def list_objects(self):
        """Names queryable via get_state/get_object_pose: the cargo parts + 'lid'."""
        return [name for name, _, _ in self.scene.cfg.manifest] + ["lid"]

    def _object_root(self, name: str) -> torch.Tensor:
        if name == "lid":
            return self.scene.lid.data.root_state_w
        if name in self.scene.cargo:
            return self.scene.cargo[name].data.root_state_w
        raise ValueError(f"unknown object {name!r}; use one of {self.list_objects()}")

    def get_seated(self):
        """Packing has no seats; kept for API compatibility — returns per-part inside()."""
        return [bool(v) for v in self.scene.inside()[0].tolist()]

    def success(self) -> bool:
        return bool(self.scene.packed()[0])

    def scene_summary(self) -> str:
        inside = {n: bool(v) for n, v in zip([m[0] for m in self.scene.cfg.manifest],
                                             self.scene.inside()[0].tolist())}
        return (f"inside={inside}\n"
                f"lid_angle={float(self.scene.lid_angle_deg()[0]):.1f} deg  "
                f"lid_seated={bool(self.scene.lid_seated()[0])}  "
                f"packed={bool(self.scene.packed()[0])}  "
                f"steps_used={self._steps_used}/{self.max_steps}")

    def obs_snapshot(self):
        out = {}
        for name in self.list_objects():
            s = self._object_root(name)[0]
            out[name] = {"pos": _np(s[:3]), "quat": _q_wxyz(_np(s[3:7]))}
        out["lid_angle_deg"] = float(self.scene.lid_angle_deg()[0])
        return out

    def api_doc(self) -> str:
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS="'slab','tube_0'..'tube_3','brick','lid' (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_G1_ASSEMBLY,
        )


class Gr1t2PackingApi(Gr1t2HandMixin, PackingApi):
    """PackingApi with the GR1T2 hand/wrist mapping (see Gr1t2HandMixin)."""


class FrankaPackingApi(FrankaTabletopApi):
    """Franka OSC control against the packing crate scene: FrankaTabletopApi's
    virtual-target machinery + PackingApi's object world/success surface."""

    list_objects = PackingApi.list_objects
    _object_root = PackingApi._object_root
    get_seated = PackingApi.get_seated
    success = PackingApi.success
    scene_summary = PackingApi.scene_summary
    obs_snapshot = PackingApi.obs_snapshot

    # Franka crate is at GROUND level (no bench): reuse the packing views but lower.
    VIEWS = {
        "default": ((1.3, -1.4, 1.2), (0.0, 0.1, 0.15)),
        "front":   ((0.0, -1.6, 1.0), (0.0, 0.1, 0.15)),
        "left":    ((-1.6, -0.3, 1.0), (0.0, 0.1, 0.15)),
        "right":   ((1.6, -0.3, 1.0), (0.0, 0.1, 0.15)),
        "top":     ((0.0, 0.1, 2.2), (0.0, 0.12, 0.1)),
        "close":   ((0.7, -0.7, 0.7), (0.0, 0.1, 0.15)),
    }

    def api_doc(self) -> str:
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS="'slab','tube_0'..'tube_3','brick','lid' (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_FRANKA_SINGLE,
        )


class GenericSceneApi(PackingApi):
    """Suite-generic control API (articulated / tool_use suites): the object surface
    comes straight from the InteractiveScene rigid-object registry, success from the
    scene's own success(), and scene telemetry from a whitelist of well-known scene
    methods (only the ones this scene actually has). Arm/hand machinery is inherited
    (G1 default; GR1T2/Franka variants below)."""

    # Known scene telemetry methods across the articulated/tool_use scenes:
    # safe: dial/handle/door angles + combo/door/prize latches; scale: beam angle/
    # settled/verdict/slot_of_box; syringe: travel/liquid/doses/spilled/...;
    # whiteboard: word/letter_scores/stray_frac/worst_letter.
    _TELEMETRY = ("dial_reading_deg", "handle_angle_deg", "door_angle_deg",
                  "combo_entered", "door_open", "prize_out", "opened",
                  "beam_angle_deg", "beam_settled", "verdict", "slot_of_box",
                  "travel", "liquid", "doses", "drawn", "spilled", "doses_ok", "parked",
                  "seated_reservoir", "seated_well",
                  "word", "letter_scores", "stray_frac", "worst_letter")
    _SKIP_OBJECTS = ("ground", "light", "bench")

    def list_objects(self):
        reg = getattr(self._env.iscene, "rigid_objects", {}) or {}
        return sorted(n for n in reg if n not in self._SKIP_OBJECTS)

    def _object_root(self, name: str) -> torch.Tensor:
        reg = getattr(self._env.iscene, "rigid_objects", {}) or {}
        if name in reg:
            return reg[name].data.root_state_w
        raise ValueError(f"unknown object {name!r}; use one of {self.list_objects()}")

    def get_seated(self):
        return []

    def success(self) -> bool:
        return bool(self.scene.success()[0])

    def _telemetry_lines(self) -> list:
        lines = []
        for name in self._TELEMETRY:
            fn = getattr(self.scene, name, None)
            if not callable(fn):
                continue
            try:
                v = fn()
            except TypeError:
                continue  # needs arguments -> not zero-arg telemetry
            if isinstance(v, str):
                lines.append(f"{name}={v}")
                continue
            if torch.is_tensor(v):
                v0 = v[0] if (v.dim() > 0 and v.shape[0] == self.n) else v
                arr = np.asarray(v0.detach().cpu())
                val = arr.tolist() if arr.dtype == np.bool_ \
                    else np.round(arr.astype(float), 3).tolist()
                lines.append(f"{name}={val}")
        return lines

    def scene_summary(self) -> str:
        return (f"success={self.success()}  steps_used={self._steps_used}/{self.max_steps}\n"
                + "  ".join(self._telemetry_lines()))

    def obs_snapshot(self):
        out = {}
        for name in self.list_objects():
            s = self._object_root(name)[0]
            out[name] = {"pos": _np(s[:3]), "quat": _q_wxyz(_np(s[3:7]))}
        return out

    def api_doc(self) -> str:
        objs = self.list_objects()
        shown = ",".join(f"'{o}'" for o in objs[:10]) + (",..." if len(objs) > 10 else "")
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS=f"{shown} (WORLD)",
            SEATED_LINE=SEATED_LINE_GENERIC,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_G1_ASSEMBLY,
        )


class Gr1t2GenericApi(Gr1t2HandMixin, GenericSceneApi):
    """GenericSceneApi with the GR1T2 hand/wrist mapping (see Gr1t2HandMixin)."""


# NullSceneApi (no-robot, sim_gen constructed problems) moved to legacy/cosigen_null.py
# (2026-07-26, user decision): no session ever ran on it — construction is standalone
# teleport-oracle scripts, evals are robot-bound. resolve_api loads it from legacy.


class FrankaGenericApi(FrankaTabletopApi):
    """Franka OSC control against the articulated/tool_use scenes (ground-level
    franka layouts): FrankaTabletopApi's virtual-target machinery + the generic
    object/success/telemetry surface."""

    list_objects = GenericSceneApi.list_objects
    _object_root = GenericSceneApi._object_root
    get_seated = GenericSceneApi.get_seated
    success = GenericSceneApi.success
    _telemetry_lines = GenericSceneApi._telemetry_lines
    scene_summary = GenericSceneApi.scene_summary
    obs_snapshot = GenericSceneApi.obs_snapshot
    _TELEMETRY = GenericSceneApi._TELEMETRY
    _SKIP_OBJECTS = GenericSceneApi._SKIP_OBJECTS

    # ground-level franka layouts: same framing as the franka crate
    VIEWS = FrankaPackingApi.VIEWS

    def api_doc(self) -> str:
        objs = self.list_objects()
        shown = ",".join(f"'{o}'" for o in objs[:10]) + (",..." if len(objs) > 10 else "")
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS=f"{shown} (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_FRANKA_SINGLE,
        )




class FrankaBimanualApi(FrankaGenericApi):
    """Bimanual MultiRobot (two Franka-style children) against the articulated/tool_use
    scenes: TWO independent virtual EE pose targets (arm 0 = "left" child, 1 = "right"),
    each converted to that child's OSC action over its `action_slices` block. Object /
    success / telemetry surface inherited from the generic scene api. ADDITIVE
    (2026-07-17, balance_scale agent)."""

    def __init__(self, env, max_steps: int = 3000):
        self._init_common(env, max_steps)
        # Store wrist targets in ENV-0 world coordinates for every row. `_action`
        # re-offsets each row to its replica origin. This makes both toolkit writes
        # and agent-authored `api._tgt[:] = env0_target.repeat(n, 1)` correct; the
        # previous raw-world representation pulled 511/512 arms toward env 0.
        self._targets_env0_world = True
        self._park_replicas = (self.n > 1
                               and os.environ.get("CAPX_PARK_REPLICAS", "1") == "1")
        robot = env.robot
        self._children = list(robot.robots.values())
        self._child_names = list(robot.robots.keys())
        self._arts = [r.articulation for r in self._children]
        self._wrist = [r.articulation.find_bodies(r.EE_BODY)[0][0] for r in self._children]
        self._hids = [r.articulation.find_joints(list(r.GRIPPER_JOINTS))[0] for r in self._children]
        self._hlow, self._hup, self._scales = [], [], []
        for r, art, ids in zip(self._children, self._arts, self._hids):
            lim = art.data.joint_pos_limits[0, ids]
            self._hlow.append(lim[:, 0].clone())
            self._hup.append(lim[:, 1].clone())
            ctrl = r.controller
            arm_ctrl = getattr(ctrl, "controllers", [ctrl])[0]
            self._scales.append((float(getattr(arm_ctrl.cfg, "pos_scale", 0.02)),
                                 float(getattr(arm_ctrl.cfg, "rot_scale", 0.097))))
        # base-class compat shims (checkpoint tree paths poke these)
        self.art = self._arts[0]
        self._wrist_idx = self._wrist
        self._hand_ids = self._hids[0]
        try:
            self._set_view(*self.VIEWS["default"])
        except Exception:
            pass
        self.reset_state()

    # ----- state -----
    def reset_state(self):
        self._tgt = [a.data.body_link_state_w[0:1, w, :7].repeat(self.n, 1).clone()
                     for a, w in zip(self._arts, self._wrist)]
        self._hfrac = []
        self._handt = []
        for i, (art, ids) in enumerate(zip(self._arts, self._hids)):
            pos = art.data.joint_pos[:, ids]
            span = (self._hup[i] - self._hlow[i]).clamp_min(1e-6)
            f = (1.0 - (pos - self._hlow[i]) / span).mean(dim=1).clamp(0.0, 1.0)
            self._hfrac.append(f)
            self._handt.append(self._hlow[i] + (1.0 - f).unsqueeze(1) * span)
        self._hand_frac = self._hfrac[0]
        self._steps_used = 0
        self._policy_last_action = None
        # Parking references: replicas hold these hand targets while parked; the full
        # parked sim state is re-captured lazily from the fresh quiescent replicas.
        self._park_handt = [h.clone() for h in self._handt]
        self._park_states = None

    def _api_state_env0(self) -> dict:
        return {"tgt": [t[0:1].clone() for t in self._tgt],
                "hfrac": [f[0:1].clone() for f in self._hfrac],
                "steps_used": int(self._steps_used)}

    def _restore_api_state(self, s: dict) -> None:
        for i in range(2):
            self._tgt[i][:] = s["tgt"][i].repeat(self.n, 1)
            if "hfrac" in s:
                self._hfrac[i][:] = s["hfrac"][i].repeat(self.n)
                span = (self._hup[i] - self._hlow[i]).clamp_min(1e-6)
                self._handt[i] = self._hlow[i] + (1.0 - self._hfrac[i]).unsqueeze(1) * span

    def set_hand_frac_tensor_arm(self, frac: torch.Tensor, arms) -> None:
        """Per-arm gripper drive for batch consumers (frac (N,) in [0,1], 1 = closed;
        Panda finger joint value == opening width, so closed = LOWER limit). The inherited
        single-arm set_hand_frac_tensor uses _hand_upper/_hand_lower, which this bimanual
        api does not define (it keeps PER-ARM _hup/_hlow) — calling that crashed every
        batched hand drive (AttributeError, 2026-07-23)."""
        for i in arms:
            self._hfrac[i] = frac
            span = (self._hup[i] - self._hlow[i]).clamp_min(1e-6)
            self._handt[i] = self._hlow[i] + (1.0 - frac).unsqueeze(1) * span
        self._hand_frac = self._hfrac[0]

    def set_hand_frac_tensor(self, frac: torch.Tensor) -> None:
        """Arm-ambiguous single-tensor form: drive BOTH grippers (documented bimanual
        semantics; batch consumers pass explicit arms via set_hand_frac_tensor_arm)."""
        self.set_hand_frac_tensor_arm(frac, (0, 1))

    # ----- action -----
    def _action(self):
        from isaaclab.utils.math import quat_mul  # noqa: PLC0415
        outs = []
        origin_delta = self.origin - self.origin[0:1]
        # Parked replicas get ZERO pose deltas + their parked hand targets: OSC holds
        # them motionless at rest (their cells sleep), so scripted contact turns cost
        # ~1 env of contact solving instead of n. A batch search unparks
        # (_replicas_active).
        parked = self._park_replicas and not self._replicas_active
        for i in range(2):
            ee = self._arts[i].data.body_link_state_w[:, self._wrist[i], :7]
            tgt = self._tgt[i]
            ps, rs = self._scales[i]
            target_pos_w = tgt[:, :3] + origin_delta
            dpos = ((target_pos_w - ee[:, :3]) / max(ps, 1e-6)).clamp(-1.0, 1.0)
            q_ee = ee[:, 3:7]
            q_conj = torch.cat([q_ee[:, :1], -q_ee[:, 1:]], dim=1)
            q_err = quat_mul(tgt[:, 3:7], q_conj)
            w = q_err[:, 0].clamp(-1.0, 1.0)
            angle = 2.0 * torch.acos(w.abs())
            axis = q_err[:, 1:] * torch.sign(w).unsqueeze(1)
            axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
            drot = (axis * angle.unsqueeze(1) / max(rs, 1e-6)).clamp(-1.0, 1.0)
            hand = self._handt[i]
            if parked:
                dpos = dpos.clone()
                drot = drot.clone()
                dpos[1:] = 0.0
                drot[1:] = 0.0
                hand = hand.clone()
                hand[1:] = self._park_handt[i][1:]
            outs.append(torch.cat([dpos, drot, hand], dim=1))
        return torch.cat(outs, dim=1)

    # ----- per-arm control -----
    def get_eef_pose(self, arm: int = 0):
        s = self._arts[arm].data.body_link_state_w[0, self._wrist[arm], :7]
        return _np(s[:3]), _q_wxyz(_np(s[3:7]))

    def move_to(self, arm: int, position, quaternion_wxyz=None, max_steps: int = 300,
                pos_tol: float = 0.02, max_step: float = 0.01):
        goal_w = torch.tensor(np.asarray(position, dtype=np.float64), device=self.device,
                              dtype=torch.float32).reshape(1, 3)
        if quaternion_wxyz is not None:
            q = torch.tensor(np.asarray(quaternion_wxyz, dtype=np.float64),
                             device=self.device, dtype=torch.float32).reshape(1, 4)
            self._tgt[arm][:, 3:7] = q.repeat(self.n, 1)
        for _ in range(int(max_steps)):
            cur0 = self._tgt[arm][0:1, :3]
            delta = goal_w - cur0
            pos_ok = float(delta.norm()) <= pos_tol
            rot_ok = True
            if quaternion_wxyz is not None:
                q_live = self._arts[arm].data.body_link_state_w[0, self._wrist[arm], 3:7]
                rot_ok = float(torch.abs((q_live * self._tgt[arm][0, 3:7]).sum())) > 0.998
            if pos_ok and rot_ok:
                break
            if not pos_ok:
                new0 = cur0 + delta * min(1.0, max_step / max(float(delta.norm()), 1e-6))
                self._tgt[arm][:, :3] = new0.repeat(self.n, 1)
            self.step(1)
            if self._steps_used >= self.max_steps:
                break
        # Live-tracking settle: the loop above converges the VIRTUAL target; the
        # OSC-tracked arm lags several cm behind it. Keep stepping until the physical
        # EE arrives (the KukaAllegro fix — without it every reach silently undershoots).
        for _ in range(int(max_steps)):
            eef = self._arts[arm].data.body_link_state_w[0:1, self._wrist[arm], :3]
            if float((eef - goal_w).norm()) <= pos_tol * 1.2 or self._steps_used >= self.max_steps:
                break
            self.step(1)
        eef = self._arts[arm].data.body_link_state_w[0:1, self._wrist[arm], :3]
        return float((eef - goal_w).norm())

    def _set_arm_frac(self, arm: int, frac: float, steps: int):
        f = float(np.clip(frac, 0.0, 1.0))
        self._hfrac[arm] = torch.full_like(self._hfrac[arm], f)
        span = (self._hup[arm] - self._hlow[arm]).clamp_min(1e-6)
        self._handt[arm] = self._hlow[arm] + (1.0 - self._hfrac[arm]).unsqueeze(1) * span
        self.step(steps)

    def close_gripper(self, arm: int = 0, steps: int = 30):
        self._set_arm_frac(arm, 1.0, steps)

    def open_gripper(self, arm: int = 0, steps: int = 20):
        self._set_arm_frac(arm, 0.0, steps)

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        raise NotImplementedError("parallel grippers: use open_gripper/close_gripper(arm)")

    def get_robot_state(self):
        out = {"steps_used": int(self._steps_used), "max_steps": int(self.max_steps)}
        for i, nm in enumerate(self._child_names):
            s = self._arts[i].data.body_link_state_w[0, self._wrist[i]]
            out[f"{nm}_ee_pose"] = _np(s[:7])
            out[f"{nm}_ee_target"] = _np(self._tgt[i][0])
            out[f"{nm}_grip_frac"] = float(self._hfrac[i][0])
        return out

    def get_link_positions(self, pattern: str = ".*hand.*"):
        rx = _re.compile(pattern)
        out = {}
        for i, nm in enumerate(self._child_names):
            names = list(self._arts[i].data.body_names)
            for j, bn in enumerate(names):
                if rx.search(bn):
                    out[f"{nm}/{bn}"] = _np(self._arts[i].data.body_link_state_w[0, j, :3])
        return out

    def api_doc(self) -> str:
        objs = self.list_objects()
        shown = ",".join(f"'{o}'" for o in objs[:10]) + (",..." if len(objs) > 10 else "")
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS=f"{shown} (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_FRANKA_BIMANUAL,
        )




class IkeaBimanualApi(FrankaBimanualApi):
    """Bimanual Franka on the IKEA table-assembly scene (the solve-verified
    assembly.ikea_table.bimanual_franka.* presets): FrankaBimanualApi's two-arm
    virtual-target OSC control + the ASSEMBLY task surface (legs/studs/table objects,
    success = all legs seated via scene.seated() — the generic scene.success() surface
    does not exist on this scene)."""

    VIEWS = AssemblyApi.VIEWS
    STUD_OFFSETS = AssemblyApi.STUD_OFFSETS
    list_objects = AssemblyApi.list_objects
    _object_root = AssemblyApi._object_root
    get_seated = AssemblyApi.get_seated
    success = AssemblyApi.success
    scene_summary = AssemblyApi.scene_summary
    obs_snapshot = AssemblyApi.obs_snapshot

    def api_doc(self) -> str:
        return _compose_doc(
            DOC_HEAD, DOC_RAW_ACCESS, DOC_CONTROL_TOOLKIT, DOC_READ_ONLY,
            DOC_CHECKPOINT_TREE, DOC_TUNING, DOC_TAIL,
            OBJECTS="'leg_0'..'leg_3','stud_0'..'stud_3','table' (WORLD)",
            SEATED_LINE=SEATED_LINE_ASSEMBLY,
            N_ENVS=str(self.n),
            EMBODIMENT=EMB_IKEA_BIMANUAL,
        )


class BimanualFrankaPackingApi(FrankaPackingApi):
    """ADDITIVE (2026-07-17, crate_packing agent): control API for MultiRobot bimanual
    franka bindings on the packing crate. Each child arm keeps its own virtual EE
    target; `arm` selects the child (0 = first-declared, usually "left"; 1 = second,
    "right"). The composed action is the children's 8-dim OSC actions concatenated per
    `env.robot.action_slices`. Object surface / success inherited from FrankaPackingApi."""

    def __init__(self, env, max_steps: int = 3000):
        self._init_common(env, max_steps)
        self._names = list(env.robot.cfg.robots.keys())
        self._arms = [env.robot[n] for n in self._names]
        self._arts = [a.articulation for a in self._arms]
        self.art = self._arts[-1]  # primary telemetry = last child ("right", the worker)
        self._slices = [env.robot.action_slices[n] for n in self._names]
        self._ee_idx = [a.articulation.find_bodies(a.EE_BODY)[0][0] for a in self._arms]
        self._grip_ids = [a.articulation.find_joints(list(a.GRIPPER_JOINTS))[0]
                          for a in self._arms]
        self._grip_lims = []
        for art, gids in zip(self._arts, self._grip_ids):
            lim = art.data.joint_pos_limits[0, gids]
            self._grip_lims.append((lim[:, 0].clone(), lim[:, 1].clone()))
        # keep base-class attrs coherent for inherited helpers (hands of the worker arm)
        self._hand_ids = self._grip_ids[-1]
        self._hand_lower, self._hand_upper = self._grip_lims[-1]
        self._hand_open_t = self._hand_upper.clone()
        self._hand_closed_t = self._hand_lower.clone()
        ctrl0 = self._arms[0].controller
        arm_ctrl = getattr(ctrl0, "controllers", [ctrl0])[0]
        self._pos_scale = float(getattr(arm_ctrl.cfg, "pos_scale", 0.02))
        self._rot_scale = float(getattr(arm_ctrl.cfg, "rot_scale", 0.097))
        try:
            self._set_view(*self.VIEWS["default"])
        except Exception:
            pass
        self.reset_state()

    def reset_state(self):
        self._tgt = [art.data.body_link_state_w[:, ei, :7].clone()
                     for art, ei in zip(self._arts, self._ee_idx)]
        self._hands_bi = []
        self._grip_frac = []
        for art, gids, (lo, hi) in zip(self._arts, self._grip_ids, self._grip_lims):
            pos = art.data.joint_pos[:, gids]
            span = (hi - lo).clamp_min(1e-6)
            self._grip_frac.append((1.0 - (pos - lo) / span).mean(dim=1).clamp(0.0, 1.0))
            self._hands_bi.append(pos.clone())
        self._hand_frac = self._grip_frac[-1]
        self._steps_used = 0
        self._policy_last_action = None

    def _api_state_env0(self) -> dict:
        return {"tgt": [t[0:1].clone() for t in self._tgt],
                "hands": [h[0:1].clone() for h in self._hands_bi],
                "steps_used": int(self._steps_used)}

    def _restore_api_state(self, s: dict) -> None:
        for i in range(len(self._tgt)):
            self._tgt[i][:] = s["tgt"][i].repeat(self.n, 1)
            if "hands" in s:
                self._hands_bi[i][:] = s["hands"][i].repeat(self.n, 1)

    def _arm_action(self, i):
        from isaaclab.utils.math import quat_mul  # noqa: PLC0415
        art, ei = self._arts[i], self._ee_idx[i]
        ee = art.data.body_link_state_w[:, ei, :7]
        tgt = self._tgt[i]
        dpos = ((tgt[:, :3] - ee[:, :3]) / max(self._pos_scale, 1e-6)).clamp(-1.0, 1.0)
        q_ee = ee[:, 3:7]
        q_conj = torch.cat([q_ee[:, :1], -q_ee[:, 1:]], dim=1)
        q_err = quat_mul(tgt[:, 3:7], q_conj)
        w = q_err[:, 0].clamp(-1.0, 1.0)
        angle = 2.0 * torch.acos(w.abs())
        axis = q_err[:, 1:] * torch.sign(w).unsqueeze(1)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
        drot = (axis * angle.unsqueeze(1) / max(self._rot_scale, 1e-6)).clamp(-1.0, 1.0)
        return torch.cat([dpos, drot, self._hands_bi[i]], dim=1)

    def _action(self):
        total = self._env.robot.action_dim
        out = torch.zeros(self.n, total, device=self.device)
        for i, sl in enumerate(self._slices):
            out[:, sl] = self._arm_action(i)
        return out

    def get_eef_pose(self, arm: int = 0):
        s = self._arts[arm].data.body_link_state_w[0, self._ee_idx[arm], :7]
        return _np(s[:3]), _q_wxyz(_np(s[3:7]))

    def move_to(self, arm: int, position, quaternion_wxyz=None, max_steps: int = 300,
                pos_tol: float = 0.02, max_step: float = 0.01):
        goal_w = torch.tensor(np.asarray(position, dtype=np.float64), device=self.device,
                              dtype=torch.float32).reshape(1, 3)
        if quaternion_wxyz is not None:
            q = torch.tensor(np.asarray(quaternion_wxyz, dtype=np.float64),
                             device=self.device, dtype=torch.float32).reshape(1, 4)
            self._tgt[arm][:, 3:7] = q.repeat(self.n, 1)
        for _ in range(int(max_steps)):
            cur0 = self._tgt[arm][0:1, :3]
            delta = goal_w - cur0
            if float(delta.norm()) <= pos_tol:
                break
            new0 = cur0 + delta * min(1.0, max_step / max(float(delta.norm()), 1e-6))
            self._tgt[arm][:, :3] = new0.repeat(self.n, 1)
            self.step(1)
            if self._steps_used >= self.max_steps:
                break
        eef = self._arts[arm].data.body_link_state_w[0:1, self._ee_idx[arm], :3]
        return float((eef - goal_w).norm())

    def _set_arm_frac(self, arm: int, frac: float, steps: int):
        lo, hi = self._grip_lims[arm]
        span = (hi - lo).clamp_min(1e-6)
        closed = torch.where(lo.abs() <= hi.abs(), lo, hi)
        opened = torch.where(lo.abs() <= hi.abs(), hi, lo)
        self._hands_bi[arm][:] = opened + float(np.clip(frac, 0.0, 1.0)) * (closed - opened)
        self.step(steps)

    def open_gripper(self, arm: int = 0, steps: int = 20):
        self._set_arm_frac(arm, 0.0, steps)

    def close_gripper(self, arm: int = 0, steps: int = 30):
        self._set_arm_frac(arm, 1.0, steps)

    def set_hand_joints(self, arm: int, targets: dict, steps: int = 20):
        """'width' (total jaw opening, m) / 'finger_N' fraction — per arm."""
        lo, hi = self._grip_lims[arm]
        for key, val in targets.items():
            if key == "width":
                span = float((hi - lo).abs().sum())
                frac = float(np.clip(val / max(span, 1e-6), 0.0, 1.0))
                closed = torch.where(lo.abs() <= hi.abs(), lo, hi)
                opened = torch.where(lo.abs() <= hi.abs(), hi, lo)
                self._hands_bi[arm][:] = closed + frac * (opened - closed)
            else:
                raise KeyError(f"unknown bimanual gripper key {key!r}; use 'width'")
        self.step(steps)

    def get_link_positions(self, pattern: str = ".*hand.*"):
        out = {}
        for name, art in zip(self._names, self._arts):
            names = list(art.data.body_names)
            rx = _re.compile(pattern)
            for i, nm in enumerate(names):
                if rx.search(nm):
                    out[f"{name}/{nm}"] = _np(art.data.body_link_state_w[0, i, :3])
        return out

    def get_robot_state(self):
        out = {"steps_used": int(self._steps_used), "max_steps": int(self.max_steps),
               "hand_frac": float(self._grip_frac[-1][0])}
        for i, (name, art) in enumerate(zip(self._names, self._arts)):
            s = art.data.body_link_state_w[0, self._ee_idx[i]]
            out[f"{name}_wrist_pose"] = _np(s[:7])
            out[f"{name}_wrist_target"] = _np(self._tgt[i][0])
        out["joint_names"] = list(self._arts[-1].data.joint_names)
        out["joint_pos"] = _np(self._arts[-1].data.joint_pos[0])
        out["joint_vel"] = _np(self._arts[-1].data.joint_vel[0])
        out["hand_joint_ids"] = list(map(int, self._grip_ids[-1]))
        return out


