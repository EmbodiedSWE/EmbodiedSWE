"""ChannelRunScene — slide a puck along the floor of a walled S-channel and drop it into a
roofed pocket (seed: pick_place/hand_trajectory, strategically inverted).

The seed task grasps a box and carries it through FREE SPACE along five prescribed floating
waypoints, in order, judged on the HAND's distance to each virtual marker. Here every element
of that strategy is inverted:

- The route is PHYSICAL, not virtual: a walled S-channel (two 90-degree corners) milled into a
  track slab. There are no markers to memorize — the route is the visible channel itself, and
  per episode the whole track is jittered, yawed, AND randomly MIRRORED (left-handed vs
  right-handed S), so a memorized waypoint list fails; the solver must follow the geometry.
- The object is never carried: ordered progress checkpoints (start bay -> mid-leg -> corner ->
  mid-leg -> corner -> mid-leg) latch ONLY while the puck rides the channel floor (a z gate:
  the puck's centre must stay below the wall-top band). Lifting the puck over the 42 mm walls
  and flying it along the exact route latches nothing.
- The goal is a gravity event, not a hover: the last stretch of the channel dead-ends under an
  amber ROOF slab that overhangs a recessed pocket. The pocket cannot be reached from the air
  (the roof covers it); the only way in is to push the puck under the roof lip until its centre
  of mass passes the pocket edge and it TIPS AND DROPS in by gravity — a contact-dynamics
  event with a settled containment outcome.
- The seed's plan is "grasp, then track hand waypoints"; the plan here is "plan a pushing
  route with two direction changes, keep the object grounded, finish with a blind push under
  an overhang". No grasp is required anywhere.

Geometry honesty is asserted in `__post_init__`: the roof face sits closer to the pocket lip
than one puck radius (so a flush push drops the CoM past the lip with 16 mm margin), the roof
underside clears the standing puck by 20 mm, the channel is 44 mm wider than the puck, the
carried-puck z is 25+ mm above the checkpoint z gate, and the pocket admits the puck standing
or lying.

Rubric: score() = 0.09 per ordered checkpoint latched (5 of them, 0.45 total) + 0.25 latched
once the puck is observed inside the pocket WITH all checkpoints already latched; exactly 1.0
iff success() = all checkpoints latched (in order, on the floor) AND the puck is inside the
pocket now AND settled. Null policy scores 0 (the first checkpoint is 55+ mm from every spawn
pose); latched credit never evaporates.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChannelRunSceneCfg(BaseCfg):
    """Config for `ChannelRunScene`. All geometry lives in the TRACK-LOCAL frame: origin at the
    track slab's centre, z = 0 at the slab (well-floor) top, channel floor at `floor_t`. The
    canonical (unmirrored) route runs +x along the south leg, +y up the east leg, -x along the
    north leg into the roofed pocket; a mirrored episode flips every local y."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cp_radius: float = tunable(0.045)  # checkpoint latch radius (track-frame xy, m)
    cp_zmax: float = tunable(0.070)  # checkpoint z gate: puck centre below this (local z, m)
    well_zmax: float = tunable(0.045)  # in-pocket gate: puck centre below this (local z, m)
    settle_speed: float = tunable(0.12)  # max puck |lin vel| when judging success (m/s)
    # (0.12 sits above the GPU phantom-velocity readback artifact band, below any real motion)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    track_pos: tuple = tunable((0.42, 0.0))  # nominal track centre on the ground (m)
    track_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the track per episode
    track_yaw_deg: float = tunable(15.0)  # uniform +/- yaw of the track per episode
    mirror_random: bool = tunable(True)  # 50/50 left-handed vs right-handed S per episode
    start_x: tuple = tunable((-0.175, -0.10))  # puck spawn x range along the start leg (local)
    start_y_jitter: float = tunable(0.012)  # puck spawn lateral jitter (local y, m)

    # --- info: structure (track-local constants; pieces derived in __post_init__) ------------
    base_size: tuple = info((0.48, 0.40, 0.02))  # the track slab under everything
    floor_t: float = info(0.025)  # raised channel floor thickness = pocket depth
    wall_h: float = info(0.042)  # wall height above the channel floor
    wall_t: float = info(0.03)
    chan_w: float = info(0.10)  # channel width between wall faces
    leg_y: float = info(0.12)  # centreline y of the south/north legs
    leg_x: float = info(0.16)  # centreline x of the east leg
    lip_x: float = info(-0.135)  # pocket east lip (west face of the north-leg floor)
    roof_face_x: float = info(-0.123)  # roof east face (12 mm east of the lip)
    roof_clear: float = info(0.080)  # roof underside height above the channel floor
    roof_t: float = info(0.008)
    puck_r: float = info(0.028)
    puck_h: float = info(0.060)
    puck_mass: float = info(0.15)
    puck_color: tuple = info((0.85, 0.10, 0.08))
    contact_offset: float = info(0.004)
    # ordered checkpoints, canonical local xy: mid south leg, SE corner, mid east leg,
    # NE corner, mid north leg (the pocket run-up).
    checkpoints: tuple = info(((0.0, -0.12), (0.16, -0.12), (0.16, 0.0), (0.16, 0.12), (0.0, 0.12)))
    well_rect: tuple = info((-0.205, -0.138, 0.075, 0.165))  # x0, x1, y0, y1 (inset gates)

    # Derived (filled in __post_init__).
    pieces: tuple = field(default=None, init=False)  # ((name, size, local centre, rgb), ...)
    chan_floor_z: float = field(default=None, init=False)  # = floor_t
    z_off: float = field(default=None, init=False)  # local z=0 above the ground plane

    def __post_init__(self) -> None:
        cf = self.floor_t
        wt = self.wall_t
        wtop = cf + self.wall_h  # 0.067
        r_under = cf + self.roof_clear  # 0.105
        slate = (0.42, 0.45, 0.52)
        grey = (0.60, 0.60, 0.62)
        amber = (0.95, 0.62, 0.08)
        ly, lx, cw = self.leg_y, self.leg_x, self.chan_w
        roof_w = self.roof_face_x - (-0.21)  # 0.087, west end flush with the west wall face
        roof_cx = -0.21 + roof_w / 2
        self.pieces = (
            ("base", self.base_size, (0.0, 0.0, -self.base_size[2] / 2), (0.30, 0.30, 0.33)),
            # raised channel floors (the pocket = the missing rectangle in the north leg)
            ("floor_a", (0.42, cw, cf), (0.0, -ly, cf / 2), grey),
            ("floor_b", (cw, 2 * ly - cw, cf), (lx, 0.0, cf / 2), grey),
            ("floor_c", (0.21 - self.lip_x, cw, cf),
             ((0.21 + self.lip_x) / 2, ly, cf / 2), grey),
            # boundary walls + the centre island (leg separator)
            ("wall_s", (0.48, wt, wtop), (0.0, -(ly + cw / 2 + wt / 2), wtop / 2), slate),
            ("wall_n", (0.48, wt, wtop), (0.0, ly + cw / 2 + wt / 2, wtop / 2), slate),
            ("wall_w", (wt, 2 * (ly + cw / 2), wtop), (-(0.21 + wt / 2), 0.0, wtop / 2), slate),
            ("wall_e", (wt, 2 * (ly + cw / 2), wtop), (0.21 + wt / 2, 0.0, wtop / 2), slate),
            ("island", (0.21 + lx - cw / 2, 2 * ly - cw, wtop),
             ((lx - cw / 2 - 0.21) / 2, 0.0, wtop / 2), slate),
            # roof flank pillars (on the island / north wall tops) + the amber roof slab
            ("pillar_a", (roof_w, 0.02, r_under - wtop),
             (roof_cx, ly - cw / 2 - 0.01, (r_under + wtop) / 2), amber),
            ("pillar_b", (roof_w, 0.02, r_under - wtop),
             (roof_cx, ly + cw / 2 + 0.01, (r_under + wtop) / 2), amber),
            ("roof", (roof_w, cw + 0.04, self.roof_t),
             (roof_cx, ly, r_under + self.roof_t / 2), amber),
        )
        self.chan_floor_z = cf
        self.z_off = self.base_size[2]  # slab bottom rests on the ground plane
        # --- honesty asserts -----------------------------------------------------------------
        # A flush push (puck east face at the roof face) puts the CoM past the pocket lip.
        drop_margin = self.puck_r - (self.roof_face_x - self.lip_x)
        assert drop_margin >= 0.012, f"flush push must drop the CoM past the lip ({drop_margin})"
        # The standing puck passes under the roof overhang.
        assert r_under - (cf + self.puck_h) >= 0.015, "roof must clear the standing puck"
        # Channel clearance and carried-puck rejection band.
        assert cw - 2 * self.puck_r >= 0.040, "channel must clear the puck generously"
        assert self.cp_zmax >= cf + self.puck_h / 2 + 0.010, "riding puck must pass the z gate"
        assert wtop + self.puck_h / 2 - self.cp_zmax >= 0.025, "carried puck must fail the z gate"
        # The pocket admits the puck standing or lying, and the gates cover its rest poses.
        well_len = self.lip_x - (-0.21)
        assert well_len > self.puck_h + 0.010 and well_len > 2 * self.puck_r + 0.015
        assert self.well_zmax >= self.puck_r + 0.012, "lying pocket rest must pass the z gate"
        assert cf + self.puck_h / 2 - self.well_zmax >= 0.010, "on-floor rest must fail it"
        # The whole spawn band keeps the puck clear of the closed (west) end of the start leg.
        assert self.start_x[0] + 0.21 >= self.puck_r + 0.005, "spawn must clear the west wall"
        # The first checkpoint is unreachable by doing nothing (null policy scores 0).
        assert abs(self.start_x[1] - self.checkpoints[0][0]) >= self.cp_radius + 0.050


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("channel_run")
class ChannelRunScene(BaseScene):
    cfg: ChannelRunSceneCfg

    def __init__(self, cfg: ChannelRunSceneCfg | None = None) -> None:
        super().__init__(cfg or ChannelRunSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the 12 kinematic track pieces at their canonical pose (reset()
        re-places everything, including mirror), and the dynamic puck."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        hard = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.40, dynamic_friction=0.30, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for name, size, ctr, rgb in c.pieces:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Track_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=hard,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.track_pos[0] + ctr[0], c.track_pos[1] + ctr[1], c.z_off + ctr[2])),
            )
        out["puck"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Puck",
            spawn=sim_utils.CylinderCfg(
                radius=c.puck_r,
                height=c.puck_h,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,  # the capsule-creep fix band
                    max_depenetration_velocity=0.5,
                    linear_damping=0.10,
                    angular_damping=0.30,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.puck_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=hard,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.puck_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.track_pos[0] - 0.15, c.track_pos[1] - 0.12,
                     c.z_off + c.floor_t + c.puck_h / 2 + 0.002)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-env track pose (pos, yaw, mirror) and the latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.track: dict[str, RigidObject] = {name: env.iscene[name]
                                              for name, _s, _c, _rgb in c.pieces}
        self.puck: RigidObject = env.iscene["puck"]
        self.env_origins = env.iscene.env_origins
        self.t_pos = torch.zeros(n, 2, device=dev)  # track centre xy (env-local)
        self.t_yaw = torch.zeros(n, device=dev)
        self.t_mir = torch.ones(n, device=dev)  # +1 canonical / -1 mirrored (local y flip)
        self.cp_latch = torch.zeros(n, len(c.checkpoints), dtype=torch.bool, device=dev)
        self.delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cps = torch.tensor(c.checkpoints, device=dev)  # (5, 2)

    # ----- track-frame transforms ---------------------------------------------------------------
    def local_to_world(self, p_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(m, 3) canonical track-local points -> world, applying mirror, yaw, origin."""
        c, s = torch.cos(self.t_yaw[env_ids]), torch.sin(self.t_yaw[env_ids])
        ly = self.t_mir[env_ids] * p_local[:, 1]
        out = torch.empty_like(p_local)
        out[:, 0] = self.t_pos[env_ids, 0] + c * p_local[:, 0] - s * ly
        out[:, 1] = self.t_pos[env_ids, 1] + s * p_local[:, 0] + c * ly
        out[:, 2] = p_local[:, 2] + self.cfg.z_off
        return out + self.env_origins[env_ids]

    def world_to_local(self, p_world: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> canonical track-local (un-mirrors), all envs."""
        p = p_world - self.env_origins
        c, s = torch.cos(self.t_yaw), torch.sin(self.t_yaw)
        dx = p[:, 0] - self.t_pos[:, 0]
        dy = p[:, 1] - self.t_pos[:, 1]
        out = torch.empty_like(p)
        out[:, 0] = c * dx + s * dy
        out[:, 1] = self.t_mir * (-s * dx + c * dy)
        out[:, 2] = p[:, 2] - self.cfg.z_off
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample track pose (xy jitter + yaw + 50/50 mirror), re-pin all 12
        kinematic pieces, spawn the puck standing in the start bay (x along-leg sampling +
        lateral jitter + free spin), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)

        self.t_pos[env_ids, 0] = c.track_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.track_jitter
        self.t_pos[env_ids, 1] = c.track_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.track_jitter
        self.t_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.track_yaw_deg)
        if c.mirror_random:
            self.t_mir[env_ids] = torch.where(torch.rand(m, device=dev) < 0.5,
                                              torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            self.t_mir[env_ids] = 1.0

        half = self.t_yaw[env_ids] / 2
        qw, qz = torch.cos(half), torch.sin(half)
        for name, _size, ctr, _rgb in c.pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            self.track[name].write_root_state_to_sim(st, env_ids)

        # puck: standing in the start bay of the south leg
        px = c.start_x[0] + torch.rand(m, device=dev) * (c.start_x[1] - c.start_x[0])
        py = -c.leg_y + (torch.rand(m, device=dev) * 2 - 1) * c.start_y_jitter
        pz = torch.full((m,), c.floor_t + c.puck_h / 2 + 0.002, device=dev)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.local_to_world(torch.stack([px, py, pz], dim=1), env_ids)
        spin = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3] = torch.cos(spin / 2)
        st[:, 6] = torch.sin(spin / 2)
        self.puck.write_root_state_to_sim(st, env_ids)

        self.cp_latch[env_ids] = False
        self.delivered[env_ids] = False

    # ----- geometry queries ---------------------------------------------------------------------
    def puck_local(self) -> torch.Tensor:
        """(N, 3) puck centre in the canonical track-local frame."""
        return self.world_to_local(self.puck.data.root_pos_w)

    def in_well(self) -> torch.Tensor:
        """(N,) bool, geometric: puck centre inside the roofed pocket (track frame, inset
        rectangle, centre below `well_zmax` — an on-channel-floor puck sits 10 mm above it)."""
        c = self.cfg
        p = self.puck_local()
        x0, x1, y0, y1 = c.well_rect
        return ((p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
                & (p[:, 2] < c.well_zmax))

    def settled(self) -> torch.Tensor:
        return self.puck.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Ordered checkpoint latching (z-gated to the channel floor band) + delivery latch."""
        c = self.cfg
        p = self.puck_local()
        on_floor = p[:, 2] < c.cp_zmax
        near = (p[:, None, :2] - self._cps[None, :, :]).norm(dim=-1) < c.cp_radius  # (N, 5)
        prev = torch.cat([torch.ones_like(self.cp_latch[:, :1]), self.cp_latch[:, :-1]], dim=1)
        self.cp_latch |= near & on_floor.unsqueeze(1) & prev
        self.delivered |= self.in_well() & self.cp_latch.all(dim=1)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "puck": self.puck.data.root_state_w[env_ids].clone(),
            "track": {k: v[env_ids].clone()
                      for k, v in (("t_pos", self.t_pos), ("t_yaw", self.t_yaw),
                                   ("t_mir", self.t_mir))},
            "latches": {"cp": self.cp_latch[env_ids].clone(),
                        "delivered": self.delivered[env_ids].clone()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.t_pos[env_ids] = state["track"]["t_pos"]
        self.t_yaw[env_ids] = state["track"]["t_yaw"]
        self.t_mir[env_ids] = state["track"]["t_mir"]
        dev = self.env.device
        m = len(env_ids)
        half = self.t_yaw[env_ids] / 2
        for name, _size, ctr, _rgb in self.cfg.pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.track[name].write_root_state_to_sim(st, env_ids)
        self.puck.write_root_state_to_sim(state["puck"], env_ids)
        self.cp_latch[env_ids] = state["latches"]["cp"]
        self.delivered[env_ids] = state["latches"]["delivered"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A flat track slab ({c.base_size[0] * 100:.0f} x {c.base_size[1] * 100:.0f} cm) "
            f"lies on the ground, carrying a walled S-shaped channel: three straight legs "
            f"({c.chan_w * 100:.0f} cm wide, walls {c.wall_h * 1000:.0f} mm above the channel "
            f"floor) joined by two 90-degree corners. A red cylindrical puck "
            f"({2 * c.puck_r * 1000:.0f} mm across, {c.puck_h * 1000:.0f} mm tall) stands near "
            f"the closed end of one outer leg (the start bay). The other outer leg dead-ends "
            f"under an AMBER ROOF slab; hidden beneath that roof the channel floor drops away "
            f"into a recessed pocket ({c.floor_t * 1000:.0f} mm deep). The track's position and "
            f"heading vary per episode, and the S may be built left- or right-handed — read the "
            f"layout, do not memorize a route.\n"
            f"Goal: slide the red puck along the channel floor from the start bay, through both "
            f"corners, to the roofed end, and push it under the amber roof until it drops into "
            f"the recessed pocket and comes to rest there. The puck must TRAVEL THE CHANNEL ON "
            f"ITS FLOOR the whole way, in route order: progress only counts while the puck "
            f"rides the channel floor (below the wall tops), so lifting it over the walls, "
            f"carrying it along the route in the air, or dropping it near the goal does not "
            f"count — and the pocket itself is covered by the roof, so it cannot be entered "
            f"from above. Pushing the puck with any part of the gripper is enough; no grasp is "
            f"required."
        )

    def instruction(self) -> str:
        return (
            "Slide the red puck along the walled channel floor from its start bay, through "
            "both corners, then push it under the amber roof at the far end so it drops into "
            "the hidden pocket and rests there. The puck must stay on the channel floor the "
            "whole way; lifting it over the walls or dropping it in from the air fails."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every checkpoint latched in route order on the channel floor, AND the
        puck is inside the roofed pocket now, AND it is settled."""
        return self.cp_latch.all(dim=1) & self.in_well() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.09 per ordered checkpoint latched (max 0.45) + 0.25 latched
        once delivered (in the pocket with the full route already latched); 1.0 iff success().
        Latched credit never evaporates; the null policy scores 0."""
        s = 0.09 * self.cp_latch.sum(dim=1).float() + 0.25 * self.delivered.float()
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="channel_run", robot="null", env_spacing=3))
