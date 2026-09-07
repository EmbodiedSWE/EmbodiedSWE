"""tool_packing: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (drawer pull ladder -> per-item pick ladder -> drop into the exposed tray, curriculum, finger PD)."""
from __future__ import annotations

import math

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class ToolPackingDenseReward(TaskReward):
    """Dense potential for the tool-packing scene.

    reach     finger pads at the nearest not-yet-stowed item
    lift      that item raised off the table
    carry     that item's origin over its assigned tray centre (drawer body frame)
    stowed    fraction of items inside their trays (scene's `_item_in_tray`)
    drawers   mean drawer shut-ness (1 = shut, 0 = fully out)
    doors     mean door shut-ness"""

    LIFT_FULL = 0.08

    WEIGHTS = {"reach": 0.1, "lift": 0.1, "carry": 0.2, "stowed": 0.3, "drawers": 0.2, "doors": 0.1}

    def reset(self, env_ids: torch.Tensor) -> None:
        pos = torch.stack([b.data.root_pos_w for b in self.scene.items.values()], dim=1)
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
        self._z0[env_ids] = pos[env_ids, :, 2]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse

        sc, c = self.scene, self.scene.cfg
        pos = torch.stack([b.data.root_pos_w for b in sc.items.values()], dim=1)  # (n,I,3)
        n, I = pos.shape[:2]
        in_tray = sc._item_in_tray()  # (n,I)
        pinch = pinch_point(self.env)
        d = (pos - pinch[:, None, :]).norm(dim=-1)
        d_focus = torch.where(~in_tray, d, torch.full_like(d, 1e3))
        k = d_focus.argmin(dim=1)
        idx = torch.arange(n, device=pos.device)
        done_all = in_tray.all(dim=1)
        # tray-centre offset of the focused item, in its drawer body frame
        bpos, bquat = sc.box.data.body_pos_w, sc.box.data.body_quat_w
        offs = []
        for i in range(I):
            b = sc._drawer_b[sc._assigned[i]]
            loc = quat_apply_inverse(bquat[:, b], pos[:, i] - bpos[:, b]) - sc._tray_c
            offs.append(loc)
        off = torch.stack(offs, dim=1)[idx, k]  # (n,3)
        reach = reach_kernel(d[idx, k])
        lift = ((pos[idx, k, 2] - self._z0[idx, k]) / self.LIFT_FULL).clamp(0.0, 1.0)
        carry = kernel(off.norm(dim=-1), 0.15)
        one = torch.ones_like(reach)
        reach, lift, carry = [torch.where(done_all, one, t) for t in (reach, lift, carry)]
        stowed = in_tray.float().mean(dim=1)
        travel = float(c.drawer_travel or 0.2)
        drawers = (1.0 - sc.drawer_pos().abs() / travel).clamp(0.0, 1.0).mean(dim=1)
        door_max = math.radians(90.0)
        doors = (1.0 - sc.door_pos().abs() / door_max).clamp(0.0, 1.0).mean(dim=1) if sc.door_pos().shape[1] else one
        return dict(reach=reach, lift=lift, carry=carry, stowed=stowed, drawers=drawers, doors=doors)


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: tool_packing_tuned) -----


class ToolPackingTunedReward(TaskReward):
    """Per-item ladders, averaged over the three items (a stowed item holds its whole ladder at 1, so finishing one
    never costs potential when the next begins). Each item's ladder, on the chest (all drawers start SHUT):
    reach_h   hand at the handle pinch pose of the item's ASSIGNED drawer: pads at the handle bar, approaching
              horizontally into the drawer front, fingers opening vertically ACROSS the 18 mm bar (keypoint distance)
    grasp_h   pads at the bar AND each finger closed onto it (per-finger window)
    open      that drawer pulled out: half a ramp to OPEN_FULL, half a threshold bonus at OPEN_OK (the pull that exposes
              enough tray for a drop). While opened, reach_h/grasp_h hold at 1 (letting go of the handle is free).
    Gated on `opened` (pick + place only pay once the drawer is out):
    reach     hand at the item's top-down grasp pose (pads at its grasp height, fingers across its narrow width)
    grasp     per-finger closure onto the item width
    lift      item raised off its settled height while at the pads (bulb recipe)
    carry     item origin at the drop point above the EXPOSED tray window (drawer body frame), while held
    orient    the item's long axis across the tray (drawer x), while held — the exposed window is shorter than the item
    stowed    the scene's own `_item_in_tray` (origin inside the assigned tray box)."""

    # drawer-frame pinch point on the handle bar (probe 2026-09-06: bar top at z 0.048, the pads must stay in FRONT of the
    # panel face at y -0.076 — pad centre at -0.087 caught the bar's lip; at -0.083 the fingertips hit the panel and closed
    # on nothing). The pads close to ~3 mm each on the lip: the scanned pull is far thinner than its 18 mm slab model.
    HANDLE_LOCAL = (0.0, -0.087, 0.039)
    HANDLE_OFFSET = 0.1034  # pad centre (PANDA_PINCH_OFFSET): the handle pinch is judged at the pads themselves
    HANDLE_WIDTH = 0.006  # m, the lip thickness the pads actually close onto (measured finger stop 0.003 each)
    # approach yaw in the drawer frame: a horizontal hand pointing straight into the front (+y local) stalls on the
    # Franka's wrist limits (the handle is 0.45 m from the base at shoulder height); yawed 40 deg toward the box side,
    # fingers axis pointing DOWN (hand +y = world -z), the pose converges exactly (probe: keypoint error 0.000)
    APPROACH_LOCAL = (-0.6428, 0.7660, 0.0)
    OPEN_W = (0.0, 0.0, -1.0)
    OPEN_OK = 0.07  # m of pull: ~5 cm of tray exposed beyond the carcass front (drop window)
    OPEN_FULL = 0.09  # m of pull that counts as fully open (probe: the third drawer pulls to 0.093 before the elbow/wrist limits)
    # Ladders are built for ACTIVE items only (the rubric's `stowed` still counts every item). Probe 2026-09-06 from this
    # base: the pinch converges on all three handles, but pulling the top/second drawer slips at ~5.6 cm when joints 4/5
    # hit their limits (handles at shoulder height, 0.45 m out); the third drawer (knife) pulls to 9.3 cm and holds.
    # ... and the knife itself cannot be pinched there: with the hand pointing down at the knife's slot the arm settles on
    # its joint-1-limit branch and the pads bottom out ~2 cm above the table, level with the knife's 20 mm top (15 scripted
    # pinches, all four pitches/yaws/along-axis offsets: fingers close to 0 and the knife squirts). The stapler (44 mm tall,
    # slot 14 cm nearer the base where the pads reach the table) is the one pickable item; its drawer is the top one, which
    # pulls 5.6-8 cm (wrist limits) — enough window for the stapler ONLY turned 90 deg (long axis across the tray), so the
    # ladder carries an `orient` rung. 2026-09-07 03:50: switched with ~5 h left; the knife ladder solved the drawer only.
    ACTIVE = ("stapler",)
    # per manifest item: grasp height above the origin, the local axis the fingers open ACROSS, pinch width (probe bboxes:
    # stapler 107x38x44 mm long along x; scissors 97x222x15 along y; knife 40x194x21 along y)
    # per manifest item: (grasp height above the origin, local axis the fingers open ACROSS, pinch width, local LONG axis,
    # lo, hi): the grasp point slides along the long axis to the point nearest the pads (clamped to the full-width body), so
    # the policy may pick the most reachable spot. Mesh slices (probe 2026-09-07): knife body 32.7 x 20 mm, full width for
    # local y in [-0.10, 0.065] (rounded end cap beyond); stapler 30.5 mm wide at mid-length, 44 mm tall. The Franka's
    # low reach at the knife's slot is at its joint-1 limit branch (pads bottom out 2-3 cm above the table there, lower
    # toward the handle end nearer the base), so the along-axis freedom matters.
    # stapler: scripted pinch 2026-09-07 05:10 — at its mid-length (x 0, pads 2.0-2.8 cm up) the fingers close to ~1 mm each
    # and the stapler LIFTS (12.5 cm, held); at x +-2 cm it slips out. The 30.5 mm slice width is the shell's footprint,
    # not what the pads meet at that height, so the pinch width is the measured finger stop (~2 mm), and the along-axis
    # window is +-1 cm. (v9b's policy had learned to hover the fingers at the 30 mm window without touching: grasp 0.9, lift 0.)
    ITEM = {"stapler": (0.024, (0.0, 1.0, 0.0), 0.004, (1.0, 0.0, 0.0), -0.01, 0.01),
            "scissors": (0.006, (1.0, 0.0, 0.0), 0.03, (0.0, 1.0, 0.0), -0.08, 0.08),
            "knife": (0.012, (1.0, 0.0, 0.0), 0.033, (0.0, 1.0, 0.0), -0.09, 0.06)}
    PINCH_OFFSET = 0.099
    HELD_NEAR = 0.06
    # v11 (2300 it.): the policy pinches the stapler exactly (pads within 2 mm, fingers 1 mm, 290/450 steps closed) and
    # SLIDES it 11 cm along the table toward the cabinet without ever lifting — carry paid for sliding, a 1 cm lift paid
    # 0.0075. So: the lift ramp saturates at 4 cm and carry only pays once the item is off the table.
    LIFT_FULL = 0.04
    LIFT_BONUS_Z = 0.04
    CARRY_MIN_DZ = 0.03
    DROP_LOCAL = (0.0, -0.03, 0.062)  # drawer-frame target for the item ORIGIN: over the exposed window, 5 cm above the tray floor
    HOVER_BACK = 0.08  # curriculum: pads this far in front of the handle along the approach, hand already horizontal

    WEIGHTS = {"reach_h": 0.05, "grasp_h": 0.05, "open": 0.15, "reach": 0.1, "grasp": 0.15, "lift": 0.2, "orient": 0.05,
               "carry": 0.1, "stowed": 0.15}

    def _items(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.stack([b.data.root_pos_w for b in self.scene.items.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.scene.items.values()], dim=1)
        return pos, quat

    def reset(self, env_ids: torch.Tensor) -> None:
        pos, _ = self._items()
        if not hasattr(self, "_z0"):
            self._z0 = pos[..., 2].clone()
            sc, dev = self.scene, self.device
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
            self._bidx = [sc._drawer_b[a] for a in sc._assigned]  # drawer body per item
            names = [m[0] for m in sc.cfg.manifest]
            self._active = torch.tensor([nm in self.ACTIVE for nm in names], device=dev)  # (I,) ladder mask
            self._active_ids = self._active.nonzero().squeeze(-1)
            self._gz = torch.tensor([[0.0, 0.0, self.ITEM[nm][0]] for nm in names], device=dev)  # (I,3)
            self._across = torch.tensor([self.ITEM[nm][1] for nm in names], device=dev)  # (I,3)
            self._width = [self.ITEM[nm][2] for nm in names]
            self._long = torch.tensor([self.ITEM[nm][3] for nm in names], device=dev)  # (I,3)
            self._lo = torch.tensor([self.ITEM[nm][4] for nm in names], device=dev)
            self._hi = torch.tensor([self.ITEM[nm][5] for nm in names], device=dev)
            self._handle = torch.tensor(self.HANDLE_LOCAL, device=dev)
            self._appr = torch.tensor(self.APPROACH_LOCAL, device=dev)
            self._drop = torch.tensor(self.DROP_LOCAL, device=dev)
            self._hover_quat = None
            self._focus = torch.zeros(env_ids.numel(), dtype=torch.long, device=dev)  # curriculum item per env (env sets it)
        self._z0[env_ids] = pos[env_ids, :, 2]

    def _drawers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assigned drawer per item: body pos (n,I,3), quat (n,I,4), pull-out metres (n,I)."""
        sc = self.scene
        bp, bq = sc.box.data.body_pos_w[:, self._bidx], sc.box.data.body_quat_w[:, self._bidx]
        pull = (-sc.drawer_pos()[:, sc._assigned]).clamp(0.0, float(sc.cfg.drawer_travel))
        return bp, bq, pull

    def _handle_pose(self, bp: torch.Tensor, bq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle pinch point (n,I,3) + the nominal hand pose pinching it: horizontal approach yawed 40 deg into the
        front (APPROACH_LOCAL, drawer frame), finger axis pointing down. Returns handle_w, t_pos, t_quat (flat n*I)."""
        from isaaclab.utils.math import quat_apply

        n, I = bp.shape[:2]
        q = bq.reshape(-1, 4)
        handle_w = bp.reshape(-1, 3) + quat_apply(q, self._handle.expand(n * I, 3))
        approach = quat_apply(q, self._appr.expand(n * I, 3))
        down = torch.tensor(self.OPEN_W, device=bp.device).expand(n * I, 3)
        t_pos, t_quat = hand_target_from_axes(handle_w, approach, down, pinch_offset=self.HANDLE_OFFSET)
        return handle_w.reshape(n, I, 3), t_pos, t_quat

    def hover_target(self) -> torch.Tensor:
        """Curriculum pose for the env's focus item: above it when its drawer is already out (fingers open, hand
        vertical: quat NaN = position only), else in front of its handle with the hand already horizontal (quat set)."""
        from isaaclab.utils.math import quat_apply

        pos, quat = self._items()
        n, I = pos.shape[:2]
        bp, bq, pull = self._drawers()
        k = self._focus
        idx = torch.arange(n, device=pos.device)
        opened = pull[idx, k] >= self.OPEN_OK
        gp, i_pos, i_quat = self._item_pose(pos, quat)
        above = i_pos.reshape(n, I, 3)[idx, k] + torch.tensor([0.0, 0.0, 0.03], device=pos.device)  # pads 3 cm above the grasp point
        handle_w, t_pos, t_quat = self._handle_pose(bp, bq)
        front = t_pos.reshape(n, I, 3)[idx, k] - quat_apply(bq[idx, k], self._appr.expand(n, 3)) * self.HOVER_BACK
        self._hover_quat = torch.where(opened[:, None], i_quat.reshape(n, I, 4)[idx, k], t_quat.reshape(n, I, 4)[idx, k])
        return torch.where(opened[:, None], above, front)  # both are HAND-ORIGIN targets (the base servo drives the hand)

    def _item_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per item: grasp point (n,I,3) — on the item's long axis at the point nearest the pads, clamped to the body — and
        the nominal top-down pinch pose there (flat n*I): approach straight down, fingers opening ACROSS the item's narrow
        width (ITEM axis), signed to the nearer of the two grasps by the hand's finger axis."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        n, I = pos.shape[:2]
        q = quat.reshape(-1, 4)
        pinch = pinch_point(self.env, self.PINCH_OFFSET)[:, None, :].expand(n, I, 3)
        loc = quat_apply_inverse(q, (pinch - pos).reshape(-1, 3)).reshape(n, I, 3)
        along = (loc * self._long[None]).sum(-1).clamp(self._lo[None], self._hi[None])  # (n,I)
        off = self._gz[None] + along[..., None] * self._long[None]  # (n,I,3) item-frame grasp offset
        grasp_pt = pos + quat_apply(q, off.reshape(-1, 3)).reshape(n, I, 3)
        down = torch.tensor([0.0, 0.0, -1.0], device=pos.device).expand(n * I, 3)
        across = quat_apply(q, self._across[None].expand(n, I, 3).reshape(-1, 3)).reshape(n, I, 3)
        across = across * torch.tensor([1.0, 1.0, 0.0], device=pos.device)
        across = torch.where(across.norm(dim=-1, keepdim=True) < 1e-3,
                             torch.tensor([0.0, 1.0, 0.0], device=pos.device).expand(n, I, 3), across)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)[:, None, :]
        across = torch.where((across * hand_y).sum(-1, keepdim=True) < 0, -across, across)
        i_pos, i_quat = hand_target_from_axes(grasp_pt.reshape(-1, 3), down, across.reshape(-1, 3), pinch_offset=self.PINCH_OFFSET)
        return grasp_pt, i_pos, i_quat

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        sc = self.scene
        pos, quat = self._items()  # (n,I,3), (n,I,4)
        n, I = pos.shape[:2]
        self._z0 = torch.minimum(self._z0, pos[..., 2])
        in_tray = sc._item_in_tray()  # (n,I)
        pinch = pinch_point(self.env, self.PINCH_OFFSET)[:, None, :]
        pads = pinch_point(self.env, self.HANDLE_OFFSET)[:, None, :]
        fq = finger_positions(self.env)
        # --- drawer ladder ---
        bp, bq, pull = self._drawers()
        handle_w, t_pos, t_quat = self._handle_pose(bp, bq)
        reach_h = reach_kernel(torch.stack([keypoint_distance(self.env, t_pos.reshape(n, I, 3)[:, i], t_quat.reshape(n, I, 4)[:, i])
                                            for i in range(I)], dim=1))
        d_h = (handle_w - pads).norm(dim=-1)
        grasp_h = torch.stack([grasp_term_fingers(d_h[:, i], fq, self.HANDLE_WIDTH) for i in range(I)], dim=1)
        # smooth gate on the pull (a bump that pushes an open drawer in a little costs a little, not a cliff — the step
        # version taught v4 to keep the hand away from the cabinet altogether)
        gate = (pull / self.OPEN_OK).clamp(0.0, 1.0)
        open_ = 0.5 * (pull / self.OPEN_FULL).clamp(0.0, 1.0) + 0.5 * ((pull - 0.05) / (self.OPEN_OK - 0.05)).clamp(0.0, 1.0)
        one = torch.ones_like(open_)
        reach_h, grasp_h = [t + (one - t) * gate for t in (reach_h, grasp_h)]  # blend to 1 as the drawer comes out
        # --- pick ladder (gated on the drawer being out) ---
        grasp_pt, i_pos, i_quat = self._item_pose(pos, quat)
        d = (grasp_pt - pinch).norm(dim=-1)
        reach = reach_kernel(torch.stack([keypoint_distance(self.env, i_pos.reshape(n, I, 3)[:, i], i_quat.reshape(n, I, 4)[:, i])
                                          for i in range(I)], dim=1))
        grasp = torch.stack([grasp_term_fingers(d[:, i], fq, self._width[i]) for i in range(I)], dim=1)
        held = (d <= self.HELD_NEAR).float()
        dz = pos[..., 2] - self._z0
        lift = held * (0.5 * (dz / self.LIFT_FULL).clamp(0.0, 1.0) + 0.5 * (dz > self.LIFT_BONUS_Z).float())
        loc = quat_apply_inverse(bq.reshape(-1, 4), (pos - bp).reshape(-1, 3)).reshape(n, I, 3) - self._drop
        carry = held * (dz > self.CARRY_MIN_DZ).float() * kernel(loc.norm(dim=-1), 0.10)
        # orient: the item's long axis across the tray (drawer x) — the exposed window is shorter than the stapler
        long_w = quat_apply(quat.reshape(-1, 4), self._long[None].expand(n, I, 3).reshape(-1, 3))
        long_d = quat_apply_inverse(bq.reshape(-1, 4), long_w).reshape(n, I, 3)
        orient = held * long_d[..., 0].abs().clamp(0.0, 1.0)
        reach, grasp, lift, orient, carry = [gate * t for t in (reach, grasp, lift, orient, carry)]
        out = dict(reach_h=reach_h, grasp_h=grasp_h, open=open_, reach=reach, grasp=grasp, lift=lift, orient=orient, carry=carry)
        act = self._active[None].expand(n, I)
        # a stowed item's ladder holds at 1; inactive items contribute nothing (mean over the active ones)
        out = {k: (torch.where(in_tray, one, v) * act).sum(dim=1) / act.sum(dim=1) for k, v in out.items()}
        out["stowed"] = (in_tray & act).float().sum(dim=1) / act.sum(dim=1)
        idx = torch.arange(n, device=pos.device)
        out["dist_h"] = d_h[idx, self._focus]  # diagnostics (not in WEIGHTS), focus item: pads -> handle, pinch -> item, pull
        out["dist_m"] = d[idx, self._focus]
        out["pull_m"] = pull[idx, self._focus]
        return out


class ToolPackingTunedEnv(RoboBenchEnv):
    """Tool-packing TUNED env: 24 s horizon (pull + pick + place), finger PD 8000, fingers in metres, the pose warm-start
    curriculum (hand horizontal in front of the handle, or above the item when its drawer is already out), a training-only
    drawer curriculum (a fraction of envs resets with one assigned drawer fully out so pick + place train in parallel;
    grading still starts shut)."""
    name = "tool_packing_tuned"
    reward_cls = ToolPackingTunedReward
    OPEN_START_FRAC = 0.5

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 24},
            "robot": {"gripper_stiffness": 8000},
            "action": {"affine": [{"dims": [6, 7], "lo": 0.0, "hi": 0.04}]},
            "curriculum": {"hover_start_frac": 0.5, "hover_steps": 90, "hover_jitter": 0.03},
            "ppo": {"num_steps_per_env": 32, "algorithm": {"entropy_coef": 0.006}},
        }

    def _on_reset(self, ids: torch.Tensor) -> None:
        """Training-only curriculum: a random focus item per env (the hover target's item); for `open_start_frac` of the
        envs its assigned drawer starts fully out."""
        sc, task = self.env.scene, self.reward_fn.task
        I = len(sc._assigned)
        m = ids.numel()
        task._focus[ids] = task._active_ids[torch.randint(0, task._active_ids.numel(), (m,), device=self.device)]
        frac = float((self.cfg.get("curriculum") or {}).get("open_start_frac", self.OPEN_START_FRAC))
        pick = torch.rand(m, device=self.device) < frac
        if not pick.any():
            return
        ids = ids[pick]
        jp = sc.box.data.joint_pos[ids].clone()
        cols = torch.tensor([sc._drawer_j[a] for a in sc._assigned], device=self.device)[task._focus[ids]]
        jp[torch.arange(ids.numel(), device=self.device), cols] = -(float(sc.cfg.drawer_travel) - 0.003)
        sc.box.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)

    def _hover_action(self, target: torch.Tensor, warm: torch.Tensor) -> torch.Tensor:
        """Position servo (base) + a rotation servo toward the reward's hover quaternion where it is set."""
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        a = super()._hover_action(target, warm)
        q_t = getattr(self.reward_fn.task, "_hover_quat", None)
        if q_t is None:
            return a
        art = self.env.robot.articulation
        ee_q = art.data.body_quat_w[:, list(art.data.body_names).index(self.env.robot.EE_BODY)]
        has = torch.isfinite(q_t).all(dim=1) & warm
        q_t = torch.where(torch.isfinite(q_t), q_t, ee_q)
        q_t = torch.where((q_t * ee_q).sum(-1, keepdim=True) >= 0, q_t, -q_t)
        ctrl = self.env.robot.controller
        arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
        rot_scale = float(getattr(getattr(arm, "cfg", None), "rot_scale", 0.097))
        rot = (axis_angle_from_quat(quat_mul(q_t, quat_conjugate(ee_q))) / rot_scale).clamp(-1.0, 1.0)
        a[:, 3:6] = rot * has[:, None].float()
        return a

    # no early termination: one env terminating (a knocked-off item) desynchronises the lockstep resets and the hover
    # preroll is skipped for the rest of training (v4: 11 partial resets in the first 100 iterations, no warm starts after)
