"""slice: DENSE reward = the up-front full-task proposal (never iterated); TUNED reward + env = the expert-iterated
first-stage recipe (measured grasp geometry, per-finger grasp, settled lift baseline, curriculum, finger PD)."""
from __future__ import annotations

import torch

from . import TaskReward
from .common import (finger_positions, grasp_term_fingers, hand_target_from_axes, kernel, keypoint_distance,
                     local_axis, pinch_point, reach_kernel)
from ..vec_env import RoboBenchEnv


class SliceDenseReward(TaskReward):
    """Dense potential for the slice scene (knife on a rest, scored food on a board).

    reach   finger pads at the knife handle
    taken   knife lifted off its rest
    carry   the blade's edge centre over the food (food frame xy)
    align   blade plane normal parallel to the cut axis (food x)
    press   for the next uncut plane: edge sample nearest the plane within the plane and
            pressed toward the board (food-frame depth)
    cut     fraction of planes released (the scene's live weld mask)"""

    HANDLE_LOCAL = (-0.09, 0.0, 0.0)  # knife frame: the handle grip point [CALIBRATE; edge runs x 0..0.22]
    LIFT_FULL = 0.06

    WEIGHTS = {"reach": 0.1, "taken": 0.1, "carry": 0.15, "align": 0.15, "press": 0.2, "cut": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = self.scene.knife.data.root_pos_w[:, 2]
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            dev = z.device
            self._edge = torch.tensor(self.scene.knife_edge(), device=dev, dtype=torch.float32)  # (E,3)
            b0, b1 = self.scene.manifest["bounds"]
            self._food_c = torch.tensor([(a + b) / 2 for a, b in zip(b0, b1)], device=dev)
            self._food_zmin = float(b0[2])
            self._planes_x = torch.tensor([p for _, p in self.scene.planes], device=dev)
        self._z0[env_ids] = z[env_ids]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        sc = self.scene
        n = self.env.num_envs
        kp, kq = sc.knife.data.root_pos_w, sc.knife.data.root_quat_w
        pinch = pinch_point(self.env)
        handle = kp + quat_apply(kq, torch.tensor(self.HANDLE_LOCAL, device=kp.device).expand(n, 3))
        reach = reach_kernel((handle - pinch).norm(dim=-1))
        taken = ((kp[:, 2] - self._z0) / self.LIFT_FULL).clamp(0.0, 1.0)
        # edge samples into the food frame (anchored on the mid piece, as the scene does)
        ref = sc.pieces[len(sc.pieces) // 2]
        rp, rq = ref.data.root_pos_w, ref.data.root_quat_w
        ref_cent = torch.tensor(sc._cents[len(sc.pieces) // 2], device=kp.device)
        E = self._edge.shape[0]
        edge_w = kp[:, None, :] + quat_apply(kq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                             self._edge[None].expand(n, E, 3).reshape(-1, 3)).reshape(n, E, 3)
        edge_f = quat_apply_inverse(rq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                    (edge_w - rp[:, None, :]).reshape(-1, 3)).reshape(n, E, 3) + ref_cent
        centre = edge_f.mean(dim=1)  # (n,3)
        carry = kernel((centre[:, :2] - self._food_c[:2]).norm(dim=-1), 0.10)
        bn_f = quat_apply_inverse(rq, local_axis(kq, 2))  # blade normal in the food frame
        align = bn_f[:, 0].abs().clamp(0.0, 1.0)
        # next uncut plane per env (all cut -> hold press at 1)
        cut = sc.cut  # (n,K) bool
        big = torch.full_like(self._planes_x[None].expand(n, -1), 1e3)
        order = torch.where(cut, big, torch.arange(cut.shape[1], device=kp.device, dtype=torch.float32)[None].expand(n, -1))
        k = order.argmin(dim=1)
        px = self._planes_x[k]  # (n,)
        dx = (edge_f[..., 0] - px[:, None]).abs()  # (n,E)
        e = dx.argmin(dim=1)
        pt = edge_f[torch.arange(n, device=kp.device), e]  # (n,3)
        in_plane = kernel((pt[:, 0] - px).abs(), 0.01)
        depth = ((self._food_c[2] - pt[:, 2]) / (self._food_c[2] - self._food_zmin)).clamp(0.0, 1.0)
        press = in_plane * (0.3 + 0.7 * depth)
        press = torch.where(cut.all(dim=1), torch.ones_like(press), press)
        return dict(reach=reach, taken=taken, carry=carry, align=align, press=press, cut=cut.float().mean(dim=1))


# ----- TUNED condition: expert-iterated first-stage reward + env (task_env: <scene>_tuned) -----


class SliceTunedReward(TaskReward):
    """Dense potential for the slice scene (knife on a rest, scored food on a board).

    reach   hand at the nominal grasp POSE: pads at the handle centre, approaching from above, fingers
            opening across the plate (keypoint distance, so orientation counts — a side-on pinch
            of the handle butt with the hand lying on the island was the measured local optimum)
    grasp   pads at the handle AND fingers closed to the handle width (a pinch, not a bat)
    taken   knife lifted off its rest while it stays at the pads (held, whatever the fingers read):
            gating lift on the aperture window made holding a pinch strictly safer than trying to
            lift (a slip drops the grasp term) — measured: grasp 0.76, lift 0.00 after 300 iterations
    carry   the blade's edge centre over the food (food frame xy)
    align   blade plane normal parallel to the cut axis (food x)
    press   for the next uncut plane: edge sample nearest the plane within the plane and
            pressed toward the board (food-frame depth)
    cut     fraction of planes released (the scene's live weld mask)"""

    HANDLE_LOCAL = (-0.05, 0.03, 0.0)
    HANDLE_WIDTH = 0.005  # m pad-to-pad when stalled on the 4 mm plate (finger-joint sum ~ pad gap)
    # m of lift (above the settled rest) that counts as fully taken. MEASURED: the knife settles 2.1 cm into the
    # notches after reset (root 0.923 -> 0.902 within 10 steps) and the grader's knife_taken wants a rise of
    # rest_rail_h (2.5 cm) above the UNSETTLED reset height, i.e. 4.6 cm above the settled rest — the old 0.04
    # paid full credit for lifts the grader never counted (training: taken 0.5 vs knife_taken 0.004)
    LIFT_FULL = 0.08
    HELD_NEAR = 0.06  # m: knife handle point within this of the pads = "held" for the lift term (loose on purpose)

    WEIGHTS = {"reach": 0.1, "grasp": 0.1, "taken": 0.1, "carry": 0.1, "align": 0.1, "press": 0.2, "cut": 0.3}

    def reset(self, env_ids: torch.Tensor) -> None:
        z = self.scene.knife.data.root_pos_w[:, 2]
        if not hasattr(self, "_z0"):
            self._z0 = z.clone()
            art = self.env.robot.articulation
            self._hand_idx = list(art.data.body_names).index(self.env.robot.EE_BODY)
            dev = z.device
            self._edge = torch.tensor(self.scene.knife_edge(), device=dev, dtype=torch.float32)  # (E,3)
            b0, b1 = self.scene.manifest["bounds"]
            self._food_c = torch.tensor([(a + b) / 2 for a, b in zip(b0, b1)], device=dev)
            self._food_zmin = float(b0[2])
            self._planes_x = torch.tensor([p for _, p in self.scene.planes], device=dev)
        self._z0[env_ids] = z[env_ids]

    def terms(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        # the part settles after reset (measured: the bulb drops 1.9 cm) — track the lowest height seen
        # since reset as the lift baseline, or lift credit only starts above the unsettled spawn height
        self._z0 = torch.minimum(self._z0, self.scene.knife.data.root_pos_w[:, 2])

        sc = self.scene
        n = self.env.num_envs
        kp, kq = sc.knife.data.root_pos_w, sc.knife.data.root_quat_w
        pinch = pinch_point(self.env)
        handle = kp + quat_apply(kq, torch.tensor(self.HANDLE_LOCAL, device=kp.device).expand(n, 3))
        d_handle = (handle - pinch).norm(dim=-1)
        # nominal grasp pose: approach straight down, fingers opening along the plate normal (knife +z)
        down = torch.tensor([0.0, 0.0, -1.0], device=kp.device).expand(n, 3)
        plate_n = local_axis(kq, 2)
        hand_y = local_axis(self.env.robot.articulation.data.body_quat_w[:, self._hand_idx], 1)  # finger-opening axis
        plate_n = torch.where((plate_n * hand_y).sum(-1, keepdim=True) < 0, -plate_n, plate_n)  # nearer of the 2 grasps
        t_pos, t_quat = hand_target_from_axes(handle, down, plate_n)
        reach = reach_kernel(keypoint_distance(self.env, t_pos, t_quat))
        grasp = grasp_term_fingers(d_handle, finger_positions(self.env), self.HANDLE_WIDTH)
        held = (d_handle <= self.HELD_NEAR).float()
        taken = held * ((kp[:, 2] - self._z0) / self.LIFT_FULL).clamp(0.0, 1.0)
        # edge samples into the food frame (anchored on the mid piece, as the scene does)
        ref = sc.pieces[len(sc.pieces) // 2]
        rp, rq = ref.data.root_pos_w, ref.data.root_quat_w
        ref_cent = torch.tensor(sc._cents[len(sc.pieces) // 2], device=kp.device)
        E = self._edge.shape[0]
        edge_w = kp[:, None, :] + quat_apply(kq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                             self._edge[None].expand(n, E, 3).reshape(-1, 3)).reshape(n, E, 3)
        edge_f = quat_apply_inverse(rq[:, None, :].expand(n, E, 4).reshape(-1, 4),
                                    (edge_w - rp[:, None, :]).reshape(-1, 3)).reshape(n, E, 3) + ref_cent
        centre = edge_f.mean(dim=1)  # (n,3)
        carry = kernel((centre[:, :2] - self._food_c[:2]).norm(dim=-1), 0.10)
        bn_f = quat_apply_inverse(rq, local_axis(kq, 2))  # blade normal in the food frame
        align = bn_f[:, 0].abs().clamp(0.0, 1.0)
        # next uncut plane per env (all cut -> hold press at 1)
        cut = sc.cut  # (n,K) bool
        big = torch.full_like(self._planes_x[None].expand(n, -1), 1e3)
        order = torch.where(cut, big, torch.arange(cut.shape[1], device=kp.device, dtype=torch.float32)[None].expand(n, -1))
        k = order.argmin(dim=1)
        px = self._planes_x[k]  # (n,)
        dx = (edge_f[..., 0] - px[:, None]).abs()  # (n,E)
        e = dx.argmin(dim=1)
        pt = edge_f[torch.arange(n, device=kp.device), e]  # (n,3)
        in_plane = kernel((pt[:, 0] - px).abs(), 0.01)
        depth = ((self._food_c[2] - pt[:, 2]) / (self._food_c[2] - self._food_zmin)).clamp(0.0, 1.0)
        press = in_plane * (0.3 + 0.7 * depth)
        press = torch.where(cut.all(dim=1), torch.ones_like(press), press)
        return dict(reach=reach, grasp=grasp, taken=taken, carry=carry, align=align, press=press,
                    cut=cut.float().mean(dim=1))


class SliceTunedEnv(RoboBenchEnv):
    """Slice TUNED env (first-stage target: the grader's `knife_taken` — knife root lifted above the rail height).
    Joint-mode Franka preset: arm dims are joint deltas relative to the CURRENT joint position; the warm-start
    servo therefore takes a damped least-squares Jacobian step toward a hover pose above the handle. Per-finger
    grasp on the 4 mm plate, lift gated on proximity from the settled height.
    MEASURED (2026-09-06): a joint delta of `scale` per step against the arm PD (kp 400 / kd 80) caps the joint
    speed at scale*kp/kd — 0.25 rad/s at the original 0.05 (the hover servo needed all 240 steps for a 0.47 m
    move); 0.15 (0.75 rad/s) learned the pick from the hover start in 100 iterations and scored from home at
    200 (graded 2/8). Wall-clock note: 20 s episodes = 960 steps = 15 PPO iterations, so `zoo.sh tb`/tb_summary
    rows at multiples of 15 are episode STARTS (stage terms ~0), not policy collapses."""
    name = "slice_tuned"
    reward_cls = SliceTunedReward

    @classmethod
    def defaults(cls):
        return {
            "task": {"episode_seconds": 20},
            "action": {"affine": [{"dims": [0, 1, 2, 3, 4, 5, 6], "from": "joint_delta", "scale": 0.15},
                                  {"dims": [7, 8], "lo": 0.0, "hi": 0.04}]},
            "curriculum": {"hover_start_frac": 0.6, "hover_steps": 100, "hover_jitter": 0.02,
                           "hover_partial_frac": 0.6, "hover_partial_steps": 100},
            "reward": {"weights": {"reach": 0.2, "grasp": 0.2, "taken": 0.6}},
            "ppo": {"num_steps_per_env": 64, "algorithm": {"entropy_coef": 0.006}},
        }

    def _handle(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        k = self.env.scene.knife
        return k.data.root_pos_w + quat_apply(k.data.root_quat_w, torch.tensor(SliceTunedReward.HANDLE_LOCAL, device=self.device).expand(self.num_envs, 3))

    def _get_terminated(self):
        return self.env.scene.knife.data.root_pos_w[:, 2] < self.env.scene.cfg.island_top - 0.10

    def _hover_target(self):
        return self._handle() + torch.tensor([0.0, 0.0, 0.10], device=self.device)

    @torch.no_grad()
    def _hover_preroll(self, ids):
        """Bridging warm start: a `hover_partial_frac` share of the warm envs is servoed for only a random number
        of steps (<= hover_partial_steps), so episode starts are spread ALONG the home->hover servo path instead
        of sitting at its two ends. Measured (tuned1-3): the pick was learned from the hover start (4/8) but never
        from home (0/8) — the home envs closed to ~9 cm keypoint distance and stalled there; a start distribution
        with no states between home and hover gives the policy nothing to chain."""
        target = self._hover_target()
        if ids.numel() != self.num_envs:
            return super()._hover_preroll(ids)  # prints the lockstep warning
        cur = self.cfg.get("curriculum", {}) or {}
        pfrac, psteps = float(cur.get("hover_partial_frac", 0.0)), int(cur.get("hover_partial_steps", self.hover_steps))
        self._in_preroll = True
        n, dev = self.num_envs, self.device
        warm = torch.rand(n, device=dev) < self.hover_frac
        steps = torch.full((n,), self.hover_steps, device=dev)
        partial = torch.rand(n, device=dev) < pfrac
        steps[partial] = (torch.rand(int(partial.sum()), device=dev) * psteps).long()
        target = target + (torch.rand(n, 3, device=dev) * 2 - 1) * self.hover_jitter * torch.tensor([1.0, 1.0, 0.0], device=dev)
        for t in range(self.hover_steps):
            self.env.step(self._process_actions(self._hover_action(target, warm & (t < steps))))
        self.preroll_warm = warm
        self._in_preroll = False

    def _hover_action(self, target, warm):
        """Joint-space servo: dq = J^+ (target - ee) (position rows, damped), scaled into the joint-delta action."""
        art = self.env.robot.articulation
        ee = list(art.data.body_names).index(self.env.robot.EE_BODY)
        jd = self.action_map
        J = art.root_physx_view.get_jacobians()[:, ee - 1, 0:3, :][:, :, jd.jd_joint_ids]  # fixed base: body idx - 1
        err = (target - art.data.body_pos_w[:, ee]).clamp(-0.08, 0.08)  # 8 cm step cap
        lam = 0.05
        JJt = J @ J.transpose(1, 2) + lam**2 * torch.eye(3, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
        a = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        a[:, jd.jd_dims] = (dq / jd.jd_scale).clamp(-1.0, 1.0) * warm[:, None].float()
        a[:, 7:] = 1.0  # fingers open
        return a
