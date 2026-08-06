"""ShapeSorterScene — post three color-coded pieces through a sorter box's matching
color-rimmed lid openings so each ends up CONTAINED in its own compartment.

Derived from the maniskill/draw_triangle seed (a Franka holding a rigid stick TRACES a
triangle outline on a canvas — continuous, contact-maintained path-following of a given
curve, judged on the traced path). This variant shares neither the artifact nor the
mechanism: nothing is drawn and no path is followed. A closed sorter box (a toddler
"post box") stands on the floor: three side-by-side compartments, fully lidded, each lid
tile carrying ONE opening framed by a brightly colored rim — a red square hole, a green
square hole, and a blue slot. Three loose pieces lie scattered on the floor in front:
a red cube, a green cylinder, and a blue rectangular card. The color->compartment
arrangement is PERMUTED per episode, so the solver must first perceive which rim sits
where, then execute three discrete post operations: grasp a piece, transport it above
the opening whose rim matches the piece's color, orient it so it passes (the card only
fits the slot within a ~10 deg yaw window; misaligned attitudes bridge the opening and
rest on the lid), and release so it falls through and settles INSIDE. The lid seals
everything but the openings, so containment is physically reachable only through the
matching aperture passage.

Judged on PHYSICAL outcomes only (settled poses read back from sim, in the box's frame
read back from the floor panel): a piece counts when its centre lies inside its ASSIGNED
compartment's interior footprint, its topmost point lies BELOW the lid underside (exact
per-shape extent — a piece wedged in an opening or resting on the lid does not count),
its centre sits above the box floor, and it is settled. success() = all three pieces
counted simultaneously. score(): 0.25 per counted piece; exactly 1.0 iff success() —
containment credit is physically persistent (a posted piece cannot leave a sealed
compartment), so credit never evaporates under correct behavior. The null policy scores
0 by construction (pieces spawn on open floor, clear of the box).

Fully procedural (no external assets): the box is 19 kinematic cuboid panels (shell +
per-compartment colored rim strips) re-posed coherently per episode (pose jitter + yaw +
rim permutation); the pieces are primitive rigid bodies. Heavy imports (isaaclab) are
deferred so importing this module — and registering the scene — stays app-free.
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

COLORS = ("red", "green", "blue")
PIECES = ("cube", "cylinder", "card")  # piece i is color i


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShapeSorterSceneCfg(BaseCfg):
    """Config for `ShapeSorterScene`. Aperture clearances are generous by design (7-10 mm
    per side laterally; the card's slot allows ~11 deg of yaw error) — an order of
    magnitude above closed-loop arm noise, while the lid still physically forces every
    piece to enter through an opening."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    xy_margin: float = tunable(0.002)  # centre-in-compartment inset from the interior walls (m)
    top_clear: float = tunable(0.002)  # piece top must sit below lid underside by this (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| (every piece) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    box_center: tuple = tunable((0.34, 0.0))  # nominal box centre on the floor
    box_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the box centre per episode
    box_yaw_deg: float = tunable(15.0)  # uniform +/- box yaw per episode
    slot_x: float = tunable(0.08)  # piece staging column (world x)
    slot_ys: tuple = tunable((-0.16, 0.0, 0.16))  # staging rows (world y); pieces permuted
    slot_jitter: float = tunable(0.030)  # uniform +/- xy jitter per piece
    piece_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per piece

    # --- info: box structure (all panels are axis-aligned cuboids in the box frame) ----------
    bin_half: float = info(0.055)  # compartment interior half width (x AND y)
    bin_pitch: float = info(0.118)  # compartment centre spacing = 2*bin_half + wall_t
    wall_t: float = info(0.008)
    floor_t: float = info(0.008)
    wall_h: float = info(0.085)  # interior depth floor-top -> lid underside
    lid_t: float = info(0.008)
    # apertures per color (ax along box x, ay along box y), centred on each lid tile
    apertures: tuple = info(((0.058, 0.058), (0.054, 0.054), (0.092, 0.034)))
    rim_colors: tuple = info(((0.85, 0.15, 0.15), (0.15, 0.65, 0.22), (0.20, 0.35, 0.90)))
    shell_color: tuple = info((0.35, 0.35, 0.38))
    # pieces: cube edge / cylinder r, h / card size (x, y, z at drop attitude)
    cube_edge: float = info(0.038)
    cyl_r: float = info(0.020)
    cyl_h: float = info(0.050)
    card_size: tuple = info((0.080, 0.018, 0.042))
    piece_mass: tuple = info((0.10, 0.12, 0.08))
    contact_offset: float = info(0.002)  # default ~2 cm offsets would eat the 5-10 mm
    # aperture clearances and glue pieces to the rims
    surface_z: float = info(0.0)  # ground plane height

    # Derived (filled in __post_init__).
    shell: tuple = field(default=None, init=False)  # ((name, size, local_center), ...)
    strips: tuple = field(default=None, init=False)  # per color: ((size, (dx, dy)), ...) x4
    lid_z0: float = field(default=None, init=False)  # lid underside height (box floor frame)
    lid_z1: float = field(default=None, init=False)  # lid top height

    def __post_init__(self) -> None:
        bh, wt, ft, wh, lt = self.bin_half, self.wall_t, self.floor_t, self.wall_h, self.lid_t
        outer_x = 2 * bh + 2 * wt
        outer_y = 3 * (2 * bh) + 4 * wt
        self.lid_z0 = ft + wh
        self.lid_z1 = self.lid_z0 + lt
        wall_zc = ft + wh / 2
        end_y = self.bin_pitch + bh + wt / 2
        self.shell = (
            ("floor", (outer_x, outer_y, ft), (0.0, 0.0, ft / 2)),
            ("wall_xn", (wt, outer_y, wh), (-(bh + wt / 2), 0.0, wall_zc)),
            ("wall_xp", (wt, outer_y, wh), (bh + wt / 2, 0.0, wall_zc)),
            ("wall_yn", (2 * bh, wt, wh), (0.0, -end_y, wall_zc)),
            ("wall_yp", (2 * bh, wt, wh), (0.0, end_y, wall_zc)),
            ("div_n", (2 * bh, wt, wh), (0.0, -(bh + wt / 2), wall_zc)),
            ("div_p", (2 * bh, wt, wh), (0.0, bh + wt / 2, wall_zc)),
        )
        strips = []
        for ax, ay in self.apertures:
            wx = (outer_x - ax) / 2
            wy = (self.bin_pitch - ay) / 2
            strips.append((
                ((wx, self.bin_pitch, lt), (-(ax / 2 + wx / 2), 0.0)),
                ((wx, self.bin_pitch, lt), (ax / 2 + wx / 2, 0.0)),
                ((ax, wy, lt), (0.0, -(ay / 2 + wy / 2))),
                ((ax, wy, lt), (0.0, ay / 2 + wy / 2)),
            ))
        self.strips = tuple(strips)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shape_sorter")
class ShapeSorterScene(BaseScene):
    cfg: ShapeSorterSceneCfg

    def __init__(self, cfg: ShapeSorterSceneCfg | None = None) -> None:
        super().__init__(cfg or ShapeSorterSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, 19 kinematic box panels (shell + colored rim strips) and the
        three pieces (reset() re-places everything coherently)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.8, restitution=0.0)

        def panel(size, pos, rgb) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path=None,  # filled by caller
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

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
        bx, by = c.box_center
        for name, size, (lx, ly, lz) in c.shell:
            p = panel(size, (bx + lx, by + ly, c.surface_z + lz), c.shell_color)
            p.prim_path = "{ENV_REGEX_NS}/Box_" + name
            out[name] = p
        lid_zc = c.lid_z0 + c.lid_t / 2
        for ci, color in enumerate(COLORS):
            bin_y = (ci - 1) * c.bin_pitch  # nominal; reset permutes
            for k, (size, (dx, dy)) in enumerate(c.strips[ci]):
                p = panel(size, (bx + dx, by + bin_y + dy, c.surface_z + lid_zc),
                          c.rim_colors[ci])
                p.prim_path = "{ENV_REGEX_NS}/Rim_" + f"{color}_{k}"
                out[f"rim_{color}_{k}"] = p

        piece_props = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.10)
        e = c.cube_edge
        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Piece_cube",
            spawn=sim_utils.CuboidCfg(
                size=(e, e, e), rigid_props=piece_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.piece_mass[0]),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.rim_colors[0]),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.slot_x, c.slot_ys[0], c.surface_z + e / 2 + 0.002)),
        )
        out["cylinder"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Piece_cylinder",
            spawn=sim_utils.CylinderCfg(
                radius=c.cyl_r, height=c.cyl_h, rigid_props=piece_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.piece_mass[1]),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.rim_colors[1]),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.slot_x, c.slot_ys[1], c.surface_z + c.cyl_h / 2 + 0.002)),
        )
        out["card"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Piece_card",
            spawn=sim_utils.CuboidCfg(
                size=c.card_size, rigid_props=piece_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.piece_mass[2]),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.rim_colors[2]),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.slot_x, c.slot_ys[2], c.surface_z + c.card_size[1] / 2 + 0.002),
                rot=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)),  # lying flat
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={"solver_type": 1, "bounce_threshold_velocity": 0.2},
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.panels: dict[str, RigidObject] = {name: env.iscene[name]
                                               for name, _s, _p in c.shell}
        for ci, color in enumerate(COLORS):
            for k in range(4):
                self.panels[f"rim_{color}_{k}"] = env.iscene[f"rim_{color}_{k}"]
        self.pieces: list[RigidObject] = [env.iscene[n] for n in PIECES]
        self.env_origins = env.iscene.env_origins
        # assign[e, c]: compartment slot (0/1/2, -y to +y) of color c in episode e
        self.assign = torch.tensor([[0, 1, 2]], device=env.device).expand(
            env.num_envs, 3).clone()
        # box_pose[e]: (x, y, yaw) of the box frame (floor centre) in episode e
        self.box_pose = torch.zeros(env.num_envs, 3, device=env.device)
        self.box_pose[:, 0] = c.box_center[0]
        self.box_pose[:, 1] = c.box_center[1]

    def _write_fixture(self, env_ids: torch.Tensor) -> None:
        """Re-pose all 19 kinematic panels coherently from box_pose + assign."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        bx, by, yaw = self.box_pose[env_ids, 0], self.box_pose[env_ids, 1], \
            self.box_pose[env_ids, 2]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        half = yaw / 2
        qw, qz = torch.cos(half), torch.sin(half)

        def write(body, lx: torch.Tensor, ly: torch.Tensor, lz: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = bx + cy * lx - sy * ly
            st[:, 1] = by + sy * lx + cy * ly
            st[:, 2] = c.surface_z + lz
            st[:, 3] = qw
            st[:, 6] = qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for name, _size, (lx, ly, lz) in c.shell:
            write(self.panels[name],
                  torch.full((m,), lx, device=dev), torch.full((m,), ly, device=dev), lz)
        lid_zc = c.lid_z0 + c.lid_t / 2
        for ci, color in enumerate(COLORS):
            bin_y = (self.assign[env_ids, ci].float() - 1.0) * c.bin_pitch
            for k, (_size, (dx, dy)) in enumerate(c.strips[ci]):
                write(self.panels[f"rim_{color}_{k}"],
                      torch.full((m,), dx, device=dev), bin_y + dy, lid_zc)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the box, permute the color->compartment assignment
        (re-posing the rim strips), scatter the pieces over permuted staging rows with
        jitter + free yaw, clear of the box under every draw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        self.box_pose[env_ids, 0] = c.box_center[0] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        self.box_pose[env_ids, 1] = c.box_center[1] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        self.box_pose[env_ids, 2] = (torch.rand(m, device=dev) * 2 - 1) \
            * math.radians(c.box_yaw_deg)
        self.assign[env_ids] = torch.argsort(torch.rand(m, 3, device=dev), dim=1)
        self._write_fixture(env_ids)

        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)  # piece i -> row perm[:, i]
        yaw_amp = math.radians(c.piece_yaw_deg)
        rows = torch.tensor(c.slot_ys, device=dev)
        rest_z = (c.cube_edge / 2, c.cyl_h / 2, c.card_size[1] / 2)
        for i in range(3):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.slot_x
            st[:, 1] = rows[perm[:, i]]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st[:, 2] = c.surface_z + rest_z[i] + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            if i == 2:  # card lies flat: q = qz(yaw) * qx(90 deg)
                c45 = math.cos(math.pi / 4)
                st[:, 3] = torch.cos(half) * c45
                st[:, 4] = torch.cos(half) * c45
                st[:, 5] = torch.sin(half) * c45
                st[:, 6] = torch.sin(half) * c45
            else:
                st[:, 3] = torch.cos(half)
                st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.pieces[i].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pieces": [p.data.root_state_w[env_ids].clone() for p in self.pieces],
            "box_pose": self.box_pose[env_ids].clone(),
            "assign": self.assign[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box_pose[env_ids] = state["box_pose"]
        self.assign[env_ids] = state["assign"]
        self._write_fixture(env_ids)
        for p, st in zip(self.pieces, state["pieces"]):
            p.write_root_state_to_sim(st, env_ids)

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A closed rectangular sorter box (a 'post box', about 36 cm long, 13 cm wide, "
            "10 cm tall) stands on the floor. Inside it are three side-by-side compartments "
            "in a row along the box's long axis, fully covered by a lid. Each compartment's "
            "lid section has ONE opening, framed by a brightly colored rim: a RED-rimmed "
            f"square hole ({c.apertures[0][0] * 1000:.0f} mm), a GREEN-rimmed square hole "
            f"({c.apertures[1][0] * 1000:.0f} mm), and a BLUE-rimmed slot "
            f"({c.apertures[2][0] * 1000:.0f} x {c.apertures[2][1] * 1000:.0f} mm, long side "
            "across the box's short axis). Which color sits over which compartment CHANGES "
            "every episode — read the rims, not remembered positions. Scattered on the floor "
            "on the near side of the box lie three loose pieces, color-matched to the rims: "
            f"a red cube ({c.cube_edge * 1000:.0f} mm), a green cylinder "
            f"({2 * c.cyl_r * 1000:.0f} mm diameter, {c.cyl_h * 1000:.0f} mm tall), and a "
            f"blue rectangular card ({c.card_size[0] * 1000:.0f} x "
            f"{c.card_size[2] * 1000:.0f} x {c.card_size[1] * 1000:.0f} mm thick).\n"
            "Goal: post every piece into its own compartment — drop each piece through the "
            "opening whose rim matches the piece's color, so it ends up resting INSIDE that "
            "compartment (fully below the lid). The card only passes the blue slot when its "
            "long axis is lined up with the slot; a piece left on the lid, wedged in an "
            "opening, dropped into a wrong-colored compartment, or left outside the box does "
            "not count. Any posting order is fine."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Post each piece into the sorter box through the lid opening whose colored rim "
            "matches it: the red cube through the red-rimmed hole, the green cylinder "
            "through the green-rimmed hole, and the blue card through the blue-rimmed slot "
            "(line the card up with the slot). Every piece must end up resting inside its "
            "own compartment, fully below the lid — a piece on the lid, in a wrong "
            "compartment, or outside the box fails."
        )

    # ----- readouts / rubric ----------------------------------------------------------------
    def box_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """READBACK box frame: ((N,2) floor centre xy local to env origin, (N,) yaw)."""
        from isaaclab.utils.math import quat_apply

        p = (self.panels["floor"].data.root_pos_w - self.env_origins)[:, :2]
        q = self.panels["floor"].data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=p.device).expand(p.shape[0], 3)
        ax = quat_apply(q, ex)
        return p, torch.atan2(ax[:, 1], ax[:, 0])

    def _piece_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos (N,3,3) local to env origin, quat (N,3,4), |lin vel| (N,3))."""
        pos = torch.stack([p.data.root_pos_w for p in self.pieces], dim=1)
        pos = pos - self.env_origins[:, None, :]
        quat = torch.stack([p.data.root_quat_w for p in self.pieces], dim=1)
        vel = torch.stack([p.data.root_lin_vel_w.norm(dim=-1) for p in self.pieces], dim=1)
        return pos, quat, vel

    def piece_top_z(self) -> torch.Tensor:
        """(N,3) exact topmost world-z (local to env origin) of each piece."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos, quat, _v = self._piece_tensors()
        n = pos.shape[0]
        dev = pos.device
        tops = []
        halves = {
            0: (c.cube_edge / 2, c.cube_edge / 2, c.cube_edge / 2),
            2: (c.card_size[0] / 2, c.card_size[1] / 2, c.card_size[2] / 2),
        }
        for i in range(3):
            q = quat[:, i]
            if i == 1:  # cylinder
                ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
                az = quat_apply(q, ez)[:, 2].abs().clamp(max=1.0)
                off = az * c.cyl_h / 2 + torch.sqrt((1 - az * az).clamp(min=0.0)) * c.cyl_r
            else:  # box: sum of |world-z component of each body axis| * half extent
                off = torch.zeros(n, device=dev)
                for k, h in enumerate(halves[i]):
                    e = torch.zeros(3, device=dev)
                    e[k] = 1.0
                    off = off + quat_apply(q, e.expand(n, 3))[:, 2].abs() * h
            tops.append(pos[:, i, 2] + off)
        return torch.stack(tops, dim=1)

    def piece_bin(self) -> torch.Tensor:
        """(N,3) long: compartment slot (0/1/2) CONTAINING each piece, judged in the
        box frame read back from the floor panel; -1 when not contained (outside the
        interior footprint, on/through the lid line, or below the floor top)."""
        c = self.cfg
        pos, _q, _v = self._piece_tensors()
        (bxy, yaw) = self.box_frame()
        d = pos[:, :, :2] - bxy[:, None, :]
        cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        lx = cy * d[:, :, 0] + sy * d[:, :, 1]
        ly = -sy * d[:, :, 0] + cy * d[:, :, 1]
        slot = torch.round(ly / c.bin_pitch).clamp(-1, 1)
        in_x = lx.abs() <= c.bin_half - c.xy_margin
        in_y = (ly - slot * c.bin_pitch).abs() <= c.bin_half - c.xy_margin
        below_lid = self.piece_top_z() <= c.surface_z + c.lid_z0 - c.top_clear
        above_floor = pos[:, :, 2] >= c.surface_z + c.floor_t
        ok = in_x & in_y & below_lid & above_floor
        return torch.where(ok, slot.long() + 1, torch.full_like(slot.long(), -1))

    def settled(self) -> torch.Tensor:
        """(N,3) bool: piece |lin vel| below `settle_speed`."""
        _p, _q, vel = self._piece_tensors()
        return vel < self.cfg.settle_speed

    def counted(self) -> torch.Tensor:
        """(N,3) bool: piece contained in its ASSIGNED compartment AND settled."""
        return (self.piece_bin() == self.assign) & self.settled()

    def success(self) -> torch.Tensor:
        """(N,) bool: all three pieces posted into their own compartments, settled."""
        return self.counted().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 per counted piece; exactly 1.0 iff success() now.
        Containment credit is physically persistent (sealed compartments), so correct
        behavior's credit cannot evaporate; the null policy scores 0 by construction."""
        base = 0.25 * self.counted().sum(dim=1).float()
        return torch.where(self.success(), torch.ones_like(base), base)

    def status_report(self) -> str:
        """Human-readable env-0 readout (used by solve/smoke prints)."""
        bins = self.piece_bin()[0].tolist()
        want = self.assign[0].tolist()
        cnt = self.counted()[0].tolist()
        return (f"bins={bins} assigned={want} counted={[bool(x) for x in cnt]} "
                f"score={float(self.score()[0]):.3f} success={bool(self.success()[0])}")


register_env("simgen", lambda: EnvCfg(scene="shape_sorter", robot="null"))
