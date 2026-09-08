"""pen_holder: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (measured grasp geometry, per-finger grasp, settled lift baseline, curriculum, finger PD)."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class PenHolderDenseReward(TaskReward):
    """Dense potential for the pen-holder scene.

    reach    finger pads at the nearest not-yet-inserted present pen
    lift     that pen raised off the table
    carry    that pen's bottom end over the holder axis (holder frame)
    tip_up   that pen's axis along the holder axis, tip up
    insert   pen bottom lowered below the rim toward depth_min (when near the axis)
    all_in   fraction of present pens counted (the scene's own rubric term)
    "Nearest uninserted pen" focuses the dense terms on one pen at a time; `all_in` keeps credit
    for the ones already done."""

    LIFT_FULL = 0.10

    WEIGHTS = {"reach": 0.1, "lift": 0.1, "carry": 0.15, "tip_up": 0.1, "insert": 0.25, "all_in": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos, _, _ = self.scene._pen_tensors()
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
        self._z0[env_ids] = pos[env_ids, :, 2]

    def terms(self) -> dict[str, torch.Tensor]:
        sc, c = self.scene, self.scene.cfg
        pos, _quat, _vel = sc._pen_tensors()  # (n,P,3)
        b_loc, t_loc = sc._pen_ends_local()  # holder frame (n,P,3)
        present = sc.present
        counted = sc.counted() & present
        pinch = pinch_point(self.env)
        n, P = pos.shape[:2]
        # focus: the nearest present, not-yet-counted pen (fallback: any present pen)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        d_focus = torch.where(present & ~counted, d, torch.full_like(d, 1e3))
        d_focus = torch.where(present.any(dim=1, keepdim=True) & (d_focus >= 1e3).all(dim=1, keepdim=True),
                              torch.where(present, d, torch.full_like(d, 1e3)), d_focus)
        k = d_focus.argmin(dim=1)  # (n,)
        idx = torch.arange(n, device=pos.device)
        done_all = counted.sum(1) >= present.sum(1)
        reach = reach_kernel(d[idx, k])
        lift = ((pos[idx, k, 2] - self._z0[idx, k]) / self.LIFT_FULL).clamp(0.0, 1.0)
        bxy = b_loc[idx, k, :2].norm(dim=-1)
        carry = kernel(bxy, 0.10)
        axis = t_loc[idx, k] - b_loc[idx, k]
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = axis[:, 2].clamp(0.0, 1.0)
        depth = c.holder_h / 2 - b_loc[idx, k, 2]  # below the rim
        insert = (bxy <= 2 * c.xy_tol).float() * (depth / c.depth_min).clamp(0.0, 1.0)
        # once every present pen is in, the focused pen is a done one: hold the dense terms at 1
        one = torch.ones_like(reach)
        reach, lift, carry, tip_up, insert = [torch.where(done_all, one, t) for t in (reach, lift, carry, tip_up, insert)]
        all_in = counted.sum(1).float() / present.sum(1).clamp(min=1).float()
        return dict(reach=reach, lift=lift, carry=carry, tip_up=tip_up, insert=insert, all_in=all_in)


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: <scene>_tuned) -----


class PenHolderTunedReward(TaskReward):
    """Dense potential for the pen-holder scene.

    reach    finger pads at the nearest not-yet-inserted present pen
    lift     that pen raised off the table
    carry    that pen's bottom end over the holder axis (holder frame)
    tip_up   that pen's axis along the holder axis, tip up
    insert   pen bottom lowered below the rim toward depth_min (when near the axis)
    all_in   fraction of present pens counted (the scene's own rubric term)
    "Nearest uninserted pen" focuses the dense terms on one pen at a time; `all_in` keeps credit
    for the ones already done."""

    LIFT_FULL = 0.10
    LIFT_BONUS_Z = 0.04
    # MEASURED (Zoo probe, 2026-09-06): the pencil's collider is far thinner than its 11.6 mm visual barrel at the grip point
    # (open pads pass over it into a 10 mm gap; a hold reads a finger-joint gap of 7.5 mm at the preset's kp 2000, crushed to
    # 1.5 mm at kp 8000). With 0.0116 the per-finger not-through floor (2.8 mm) sat above the real hold, so grasp paid nothing
    # for a real pinch. A fast close from fully open ejects the pen; a pre-narrowed close holds it (7/8 probe envs lifted it
    # 54 mm at kp 2000, 5/8 at kp 8000).
    PEN_WIDTH = 0.0075  # m, the measured hold gap at gripper_stiffness 2000
    PINCH_OFFSET = 0.099
    HELD_NEAR = 0.06

    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "lift": 0.1, "carry": 0.15, "tip_up": 0.1, "insert": 0.2, "all_in": 0.25}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos, _, _ = self.scene._pen_tensors()
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
        self._z0[env_ids] = pos[env_ids, :, 2]

    def focus(self) -> torch.Tensor:
        """(n,) the pen the dense terms track: nearest present, not-yet-counted (fallback: any present)."""
        sc = self.scene
        pos, _, _ = sc._pen_tensors()
        present, counted = sc.present, sc.counted() & sc.present
        d = (pos - pinch_point(self.env, self.PINCH_OFFSET)[:, None, :]).norm(dim=-1)
        d_focus = torch.where(present & ~counted, d, torch.full_like(d, 1e3))
        d_focus = torch.where(present.any(dim=1, keepdim=True) & (d_focus >= 1e3).all(dim=1, keepdim=True),
                              torch.where(present, d, torch.full_like(d, 1e3)), d_focus)
        return d_focus.argmin(dim=1)

    def hover_target(self) -> torch.Tensor:
        pos, _, _ = self.scene._pen_tensors()
        k = self.focus()
        return pos[torch.arange(pos.shape[0], device=pos.device), k] + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    def terms(self) -> dict[str, torch.Tensor]:
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, self.scene._pen_tensors()[0][..., 2])
        sc, c = self.scene, self.scene.cfg
        pos, _quat, _vel = sc._pen_tensors()  # (n,P,3)
        b_loc, t_loc = sc._pen_ends_local()  # holder frame (n,P,3)
        present = sc.present
        counted = sc.counted() & present
        pinch = pinch_point(self.env, self.PINCH_OFFSET)
        n, P = pos.shape[:2]
        k = self.focus()
        idx = torch.arange(n, device=pos.device)
        done_all = counted.sum(1) >= present.sum(1)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        pk, qk = pos[idx, k], _quat[idx, k]
        # nominal grasp pose on the focused pen: approach down, fingers opening ACROSS the pen axis
        down = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand(n, 3)
        axis_w = local_axis(qk, 2)
        across = torch.cross(axis_w, down, dim=-1) * torch.tensor([1.0, 1.0, 0.0], device=pos.device)
        across = torch.where(across.norm(dim=-1, keepdim=True) < 1e-3, torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, 3), across)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)
        across = torch.where((across * hand_y).sum(-1, keepdim=True) < 0, -across, across)
        t_pos, t_quat = hand_target_from_axes(pk, down, across, pinch_offset=self.PINCH_OFFSET)
        reach = reach_kernel(keypoint_distance(self.env, t_pos, t_quat))
        grasp = grasp_term_fingers(d[idx, k], finger_positions(self.env), self.PEN_WIDTH)
        held = (d[idx, k] <= self.HELD_NEAR).float()
        dz = pk[:, 2] - self._z0[idx, k]
        lift = held * (0.5 * (dz / self.LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > self.LIFT_BONUS_Z).float())
        bxy = b_loc[idx, k, :2].norm(dim=-1)
        carry = held * kernel(bxy, 0.10)  # carry / tip-up / insert count only while HELD (else: shoving exploit)
        axis = t_loc[idx, k] - b_loc[idx, k]
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tip_up = held * axis[:, 2].clamp(0.0, 1.0)
        depth = c.holder_h / 2 - b_loc[idx, k, 2]  # below the rim
        insert = held * (bxy <= 2 * c.xy_tol).float() * (depth / c.depth_min).clamp(0.0, 1.0)
        # once every present pen is in, the focused pen is a done one: hold the dense terms at 1
        one = torch.ones_like(reach)
        reach, grasp, lift, carry, tip_up, insert = [torch.where(done_all, one, t) for t in (reach, grasp, lift, carry, tip_up, insert)]
        all_in = counted.sum(1).float() / present.sum(1).clamp(min=1).float()
        return dict(reach=reach, grasp=grasp, lift=lift, carry=carry, tip_up=tip_up, insert=insert, all_in=all_in, dist_m=d[idx, k])


class PenTunedEnv(RoboBenchEnv):
    """Pen-holder TUNED env. The grader's first stage (`pens_in`) already needs a pen inserted tip-up in the cup, so
    the tuned reward keeps all rungs (reach, per-finger grasp, lift, carry, tip-up, insert) and the curriculum
    warm-starts above the focused pen."""
    name = "pen_tuned"
    reward_cls = PenHolderTunedReward

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 14},           # the bulb/nut pick horizon; insertion rungs come after a graded pick
            "robot": {"gripper_stiffness": 2000},      # the preset's own finger PD: 8000 crushes into the pen's thin collider (see PEN_WIDTH)
            # fingers 0-20 mm each (the pen's collider needs ~8 mm of gap; 40 mm made every noisy step a drop) and a first-order
            # filter on the finger targets (finger_ema; 0 = off): MEASURED on pen_r8, an in-hand start (held, lift 0.76) was
            # opened to a 30 mm gap by ONE std-1 finger action and dropped within the first policy step, every episode, so PPO
            # never saw a hold last — the filter makes a hold survive single noisy steps while a sustained command still opens.
            "action": {"affine": [{"dims": [6, 7], "lo": 0.0, "hi": 0.02}], "finger_ema": 0.9},
            # hover_z_min < 0.10 spreads the warm-start height over [hover_z_min, 0.10] above the pen; align_frac re-yaws the
            # focused pen across the hand's finger axis after the pre-roll; inhand_frac starts with the pen pinched over the
            # holder (all training-only init widening; the graded reset is the scene's own)
            "curriculum": {"hover_start_frac": 0.5, "hover_steps": 90, "hover_jitter": 0.03,
                           "hover_z_min": 0.10, "hover_z_pow": 1.0, "align_frac": 0.0, "inhand_frac": 0.0, "inhand_at": "pen"},
            # first-stage weights = the bulb/nut pick recipe (measured on tuned2/tuned3: with 7 rungs at ~0.1 each,
            # grasp peaked 0.04 and lift 0.0003 over 300 it — the pick was never learned, so nothing downstream could be)
            "reward": {"weights": {"reach": 0.2, "grasp": 0.2, "lift": 0.6}},
            "ppo": {"num_steps_per_env": 32, "algorithm": {"entropy_coef": 0.006}},
        }


    def _cur(self, key: str) -> float:
        return float((self.cfg.get("curriculum") or {}).get(key, 0.0))

    def _process_actions(self, a: torch.Tensor) -> torch.Tensor:
        """Finger dims go through a first-order filter (see defaults); the scripted pre-roll phases write it directly so the
        warm/in-hand starts land exactly where they aim. Applies in training AND in the exported solve (same class)."""
        alpha = float((self.cfg.get("action") or {}).get("finger_ema", 0.0))
        if alpha > 0:
            if not hasattr(self, "_fa"):
                self._fa = torch.ones(a.shape[0], a.shape[1] - 6, device=a.device)  # open
            self._fa = a[:, 6:].clone() if getattr(self, "_in_preroll", False) else alpha * self._fa + (1 - alpha) * a[:, 6:]
            a = torch.cat([a[:, :6], self._fa], dim=1)
        return self.action_map(self.env, a)

    @torch.no_grad()
    def _hover_preroll(self, ids: torch.Tensor) -> None:
        """Warm start in two legs (training only; the graded reset is the scene's own). Leg 1: the warm envs servo to
        10 cm above the focused pen (30 steps; body poses are fresh from here on — env.reset writes joints without
        stepping). Then, for a random `align_frac` of ALL envs, the focused pen is re-yawed in place so its axis runs
        across the hand's finger-opening axis (the scene draws that yaw uniformly; this only reweights the start).
        Leg 2 (60 steps): the warm envs descend to a height drawn from [hover_z_min, 0.10] above the pen, xy jitter
        scaled with the height, so a 1 cm start puts the pen between the open pads. Measured on pen_r1 model_100-150
        (play --trace): from a fixed 0.10 start the policy parks at the hover height and never explores the descent."""
        if ids.numel() != self.num_envs:
            print("[rl] hover curriculum needs lockstep resets; partial reset -> skipped", flush=True)
            return
        sc, task, n, dev = self.env.scene, self.reward_fn.task, self.num_envs, self.device
        self._in_preroll = True
        warm = torch.rand(n, device=dev) < self.hover_frac
        idx = torch.arange(n, device=dev)
        jit = (torch.rand(n, 3, device=dev) * 2 - 1) * self.hover_jitter * torch.tensor([1.0, 1.0, 0.0], device=dev)
        k = task.focus()
        up = torch.tensor([0.0, 0.0, 0.10], device=dev)
        target = sc._pen_tensors()[0][idx, k] + up + jit
        for _ in range(30):
            self.env.step(self._process_actions(self._hover_action(target, warm)))
        if self._cur("align_frac") > 0:
            self._align_pen(self._cur("align_frac"), k)
        zmin = self._cur("hover_z_min") or 0.10
        z = zmin + (0.10 - zmin) * torch.rand(n, device=dev) ** (self._cur("hover_z_pow") or 1.0)  # pow > 1 biases low
        target = sc._pen_tensors()[0][idx, k] + z[:, None] * up / 0.10 + jit * (z / 0.10)[:, None]
        for _ in range(60):
            self.env.step(self._process_actions(self._hover_action(target, warm)))
        self.preroll_warm = warm
        self._in_preroll = False

    def _reset_idx(self, ids: torch.Tensor) -> None:
        if hasattr(self, "_fa"):
            self._fa[ids] = 1.0  # fingers open at the scene's own reset
        super()._reset_idx(ids)
        if self._cur("inhand_frac") > 0 and not self._in_preroll and ids.numel() == self.num_envs:
            self._inhand_start(self._cur("inhand_frac"))

    @torch.no_grad()
    def _align_pen(self, frac: float, k: torch.Tensor) -> None:
        """Re-yaw the focused pen `k` (flat on the table, same spot, at rest) across the hand's finger axis in a random
        `frac` of the envs."""
        sc, task = self.env.scene, self.reward_fn.task
        n, dev = self.num_envs, self.device
        sel = torch.rand(n, device=dev) < frac
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, task._hand_idx], 1)
        yaw = torch.atan2(hand_y[:, 0], -hand_y[:, 1])  # pen axis = hand_y rotated 90 deg about z
        c45, half = math.cos(math.pi / 4), yaw / 2
        for i, b in enumerate(sc.pens.values()):
            m = sel & (k == i) & sc.present[:, i]
            if m.any():
                st = b.data.root_state_w[m].clone()
                st[:, 3], st[:, 4], st[:, 5], st[:, 6] = torch.cos(half[m]) * c45, -torch.sin(half[m]) * c45, torch.cos(half[m]) * c45, torch.sin(half[m]) * c45
                st[:, 7:] = 0.0
                b.write_root_state_to_sim(st, m.nonzero(as_tuple=False).squeeze(-1))

    @torch.no_grad()
    def _inhand_start(self, frac: float, steps: int = 15) -> None:
        """Init-state widening (TRAINING only; grading resets the real scene): a random `frac` of the envs start with the
        focused pen already pinched in the fingers (Factory-style in-hand start). `curriculum.inhand_at` = "pen": held
        horizontally across the pads 5 cm above its spot on the table (the pick stage: teaches the value of held + up);
        "holder": held tip-up with its bottom 3 cm over the rim (the insert stage); "mix": half and half. Sequence measured on the Zoo probe:
        hand servoed to the spot with open pads, pads narrowed to a 10 mm gap (a close from fully open ejects the pen),
        pen teleported into the gap, pads closed for `steps` control steps."""
        sc, task = self.env.scene, self.reward_fn.task
        n, dev = self.num_envs, self.device
        sel = torch.rand(n, device=dev) < frac
        idx, k = torch.arange(n, device=dev), task.focus()
        at = (self.cfg.get("curriculum") or {}).get("inhand_at", "pen")
        # "mix": half of the in-hand envs start over the holder (tip-up), half at the pen's spot (horizontal)
        over_holder = torch.full((n,), at == "holder", dtype=torch.bool, device=dev)
        if at == "mix":
            over_holder = torch.rand(n, device=dev) < 0.5
        lift = sc.cfg.holder_h / 2 + 0.03 + float(sc._half_l[0])
        t_holder = sc.holder.data.root_pos_w + torch.tensor([0.0, 0.0, lift + task.PINCH_OFFSET], device=dev)
        t_pen = sc._pen_tensors()[0][idx, k] + torch.tensor([0.0, 0.0, 0.05 + task.PINCH_OFFSET], device=dev)
        target = torch.where(over_holder[:, None], t_holder, t_pen)
        self._in_preroll = True
        for _ in range(self.hover_steps):
            self.env.step(self._process_actions(self._hover_action(target, sel)))
        a = torch.zeros(n, self.num_actions, device=dev)
        a[:, 6:] = torch.where(sel[:, None], -0.75, 1.0)  # 10 mm gap (sel) / stay open (rest)
        for _ in range(10):
            self.env.step(self._process_actions(a))
        from isaaclab.utils.math import quat_from_matrix

        pinch = pinch_point(self.env, task.PINCH_OFFSET)
        hq = self.env.robot.articulation.data.body_quat_w[:, task._hand_idx]
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = pinch
        hy, hz = local_axis(hq, 1), local_axis(hq, 2)
        ax = torch.cross(hy, hz, dim=-1)  # pen axis across the finger-opening axis (hand y), in the pad plane
        q_flat = quat_from_matrix(torch.stack([hy, torch.cross(ax, hy, dim=-1), ax], dim=-1))
        q_up = torch.zeros(n, 4, device=dev)
        q_up[:, 0] = 1.0  # identity: pen +z (the tip) up
        st[:, 3:7] = torch.where(over_holder[:, None], q_up, q_flat)
        for i, b in enumerate(sc.pens.values()):
            m = sel & (k == i) & sc.present[:, i]
            if m.any():
                b.write_root_state_to_sim(st[m], m.nonzero(as_tuple=False).squeeze(-1))
        a[:, 6:] = torch.where(sel[:, None], -1.0, 1.0)
        for _ in range(steps):
            self.env.step(self._process_actions(a))
        self._in_preroll = False
