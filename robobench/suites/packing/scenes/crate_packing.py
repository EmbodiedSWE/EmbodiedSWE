"""CratePackingScene — pack odd-shaped parts into a crate so the hinged lid closes flush.

The object world for the packing task: an open-top crate (floor + 4 walls, kinematic)
with a dynamic lid on a revolute hinge along the +y top edge, and a manifest of loose
cargo parts scattered on the ground around it.
**Goal (carried here, no task layer): get every cargo part inside the crate and the lid
closed flush on the rim.**

Judged by physics, not predicates: the crate depth is sized so the REFERENCE packing
fills it to within ~1 cm of the rim — any part misplaced/stacked sloppily pokes above
the rim and the falling lid rests on it instead of on the rim. The `packed()` check
reads (a) every part inside the interior volume, (b) lid hinge angle ~ closed, and
(c) the lid settled at rim height (a lid propped on proud cargo sits measurably higher).

v0 cargo is PROCEDURAL (slab + 4 tubes + box, dimensioned so flat side-by-side layouts
exceed the floor area and only the layered packing fits). v1 swaps in furniture-bench
meshes via the same manifest cfg. Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class CratePackingSceneCfg(BaseCfg):
    """Config for `CratePackingScene`. Interior dims are DERIVED from the manifest with a
    designed slack (`oversize`), so required packing precision is a dial, not an accident."""

    # --- tunable: difficulty dials -----------------------------------------------------------
    # 1.12 measured TOO FORGIVING (lid seated 3/3 at 25mm jitter, sweep 2026-07-14);
    # tightened to 1.06 with a slimmer rim clearance to move the knee toward ~1.5cm.
    oversize: float = tunable(1.06)  # interior = reference-packing bbox * oversize (1.05 hard)
    lid_close_deg: float = tunable(2.0)  # lid counts as closed at/below this hinge angle
    lid_seat_tol: float = tunable(0.012)  # max lid-center height above rim when closed (m)
    settle_speed: float = tunable(0.05)  # max lid |v| when judging lid_seated (m/s; cargo is not speed-checked)
    reset_pos_jitter: float = tunable(0.02)  # uniform +/- xy jitter per part at reset (m)

    # --- tunable: placement (robot embodiments raise the work onto a bench) ------------------
    surface_z: float = tunable(0.0)  # work-surface height; 0 = on the ground (null smoke)

    # --- info: manifest + crate structure ----------------------------------------------------
    bench_size: tuple = info((1.1, 0.9))  # procedural bench top (x, y), used when surface_z > 0
    # Cargo spawn arc (deg): full circle for the null smoke; robot bindings use a front arc
    # so every part lands on the reachable side of the bench.
    spawn_arc: tuple = info((0.0, 360.0))
    # v0 procedural manifest: name -> ("box", (sx, sy, sz)) or ("cyl", (radius, length)).
    # Dimensioned so: reference packing = slab flat, 4 tubes side-by-side on the slab, box on
    # the tubes; flat single-layer footprint (~0.20 m^2) exceeds the floor (~0.14 m^2).
    manifest: tuple = info((
        ("slab", "box", (0.40, 0.30, 0.04)),
        ("tube_0", "cyl", (0.025, 0.28)),
        ("tube_1", "cyl", (0.025, 0.28)),
        ("tube_2", "cyl", (0.025, 0.28)),
        ("tube_3", "cyl", (0.025, 0.28)),
        ("brick", "box", (0.16, 0.12, 0.08)),
    ))
    # Reference packing heights: slab (0.04) + tube layer (0.05) + brick (0.08) = 0.17.
    ref_stack_h: float = info(0.17)
    wall_t: float = info(0.015)  # crate wall/floor thickness
    lid_t: float = info(0.02)
    part_mass: float = info(0.4)  # per cargo part (slab gets 2x)
    crate_pos: tuple = info((0.0, 0.0))  # crate centre on the ground plane
    # Cargo spawn ring: parts scattered around the crate at this radius, angle-indexed.
    spawn_radius: float = info(0.65)

    # Derived (filled in __post_init__): interior (ix, iy, iz), rim height.
    interior: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Interior footprint from the largest part (slab) + tube row; height from ref stack.
        slab = self.manifest[0][2]
        ix = slab[0] * self.oversize
        iy = slab[1] * self.oversize
        iz = self.ref_stack_h + 0.006  # slim rim clearance over the reference packing
        self.interior = (round(ix, 4), round(iy, 4), round(iz, 4))


@SCENES.register("crate")
class CratePackingScene(BaseScene):
    cfg: CratePackingSceneCfg

    def __init__(self, cfg: CratePackingSceneCfg | None = None) -> None:
        super().__init__(cfg or CratePackingSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the crate (kinematic compound: floor + 4 walls), the dynamic lid,
        and the manifest cargo parts as free rigid bodies."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ix, iy, iz = c.interior
        t = c.wall_t
        cx, cy = c.crate_pos

        z0 = c.surface_z
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
        if z0 > 0:  # procedural workbench: a kinematic slab whose TOP lands at surface_z
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0 / 2)),
            )

        # Crate = 5 kinematic boxes (exact interior collision; no mesh decomposition needed).
        # Floor at z = t/2; walls rise to rim height iz + t (floor top at t).
        wall_h = iz + t
        crate_parts = {
            "floor": ((ix + 2 * t, iy + 2 * t, t), (cx, cy, z0 + t / 2)),
            "wall_xp": ((t, iy + 2 * t, wall_h), (cx + ix / 2 + t / 2, cy, z0 + wall_h / 2)),
            "wall_xn": ((t, iy + 2 * t, wall_h), (cx - ix / 2 - t / 2, cy, z0 + wall_h / 2)),
            "wall_yp": ((ix, t, wall_h), (cx, cy + iy / 2 + t / 2, z0 + wall_h / 2)),
            "wall_yn": ((ix, t, wall_h), (cx, cy - iy / 2 - t / 2, z0 + wall_h / 2)),
        }
        for name, (size, pos) in crate_parts.items():
            out[f"crate_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

        # Lid: dynamic slab, hinged along the +y top edge (joint authored in bind()).
        # JOINT CONVENTION (bug-fixed after v0): identity joint frames make joint-zero =
        # lid CLOSED (flat on the rim); rotation about +x sends POSITIVE angles down INTO
        # the crate, so OPENING is NEGATIVE. Limits are [-115, 0] and the lid spawns at
        # -110 deg: leaning back past vertical behind the +y wall, where gravity presses
        # it into the -115 stop (it stays open until deliberately closed).
        lid_len_y = iy + 2 * t
        rim_z = wall_h
        a = math.radians(-110.0)
        # Hinge line: x along the +y wall top edge at (cy + iy/2 + t, z = rim_z). The lid's
        # +y edge rides the hinge; closed (angle 0) it extends -y, covering the crate.
        hy, hz = cy + iy / 2 + t, z0 + rim_z
        # Lid centre = hinge + Rx(a) @ (0, -off, +lid_t/2)   [closed-pose offset]
        off = lid_len_y / 2
        ca, sa = math.cos(a), math.sin(a)
        lid_pos = (cx,
                   hy + (-off) * ca - (c.lid_t / 2) * sa,
                   hz + (-off) * sa + (c.lid_t / 2) * ca)
        qw, qx = math.cos(a / 2), math.sin(a / 2)
        out["lid"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Lid",
            spawn=sim_utils.CuboidCfg(
                size=(ix + 2 * t, lid_len_y, c.lid_t),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.20)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=lid_pos, rot=(qw, qx, 0.0, 0.0)),
        )

        # Cargo parts scattered in a ring around the crate (reset() re-places them anyway).
        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        for k, (name, kind, dims) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (k + 0.5) / len(c.manifest)
            px, py = cx + c.spawn_radius * math.cos(ang), cy + c.spawn_radius * math.sin(ang)
            if kind == "box":
                spawn = sim_utils.CuboidCfg(
                    size=dims,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=c.part_mass * (2.0 if name == "slab" else 1.0)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.55)),
                )
                part_z = dims[2] / 2
            else:  # cyl: (radius, length), spawned lying on its side (axis along x)
                spawn = sim_utils.CylinderCfg(
                    radius=dims[0], height=dims[1],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.part_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.55)),
                )
                part_z = dims[0]
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo_" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.surface_z + part_z)),
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
        """Grab handles, cache origins, author the lid hinge (revolute, +x axis,
        limits [-115, 0] deg: 0 = closed flat on the rim, opening negative — see the
        joint-convention comment in assets()). Authored in bind(), i.e. right after
        sim.reset(), before any stepping; the hinge topology is fixed for the episode."""
        super().bind(env)
        c = self.cfg
        self.lid: RigidObject = env.iscene["lid"]
        self.cargo: dict[str, RigidObject] = {name: env.iscene[name] for name, _, _ in c.manifest}
        self.crate_floor: RigidObject = env.iscene["crate_floor"]
        self.env_origins = env.iscene.env_origins
        self._author_lid_hinges()

    def _author_lid_hinges(self) -> None:
        """One revolute joint per env: lid <-> +y wall, axis +x along the rim edge, limited to
        [0, 115] deg (0 = closed flat on the rim). Authored ENABLED before play (fixed topology,
        unlike the ikea welds which toggle)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        ix, iy, iz = c.interior
        t = c.wall_t
        wall_h = iz + t
        stage = omni.usd.get_context().get_stage()
        self._hinge_paths: list[str] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            jp = f"{base}/lid_hinge"
            j = UsdPhysics.RevoluteJoint.Define(stage, jp)
            j.CreateBody0Rel().SetTargets([f"{base}/Crate_wall_yp"])
            j.CreateBody1Rel().SetTargets([f"{base}/Lid"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            # Joint frame on the wall: its top inner edge (wall local frame: centre at
            # (cx, cy+iy/2+t/2, wall_h/2)) -> hinge at local (0, +t/2, +wall_h/2).
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, t / 2, wall_h / 2))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # Joint frame on the lid: its +y edge, bottom face (lid local frame) ->
            # (0, +lid_len_y/2, -lid_t/2). With identity rotations on both frames,
            # joint angle 0 == closed lid; opening = NEGATIVE rotation about +x.
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, (iy + 2 * t) / 2, -c.lid_t / 2))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-115.0)
            j.CreateUpperLimitAttr(0.0)
            self._hinge_paths.append(jp)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh start: lid open (~110 deg), cargo scattered in the spawn ring with jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        ix, iy, iz = c.interior
        t = c.wall_t
        cx, cy = c.crate_pos

        # Lid: reset to its authored open pose, -110 deg about +x (leaning back past
        # vertical, so gravity holds it against the -115 stop). Mirrors assets() math.
        wall_h = iz + t
        a = math.radians(-110.0)
        off = (iy + 2 * t) / 2
        ca, sa = math.cos(a), math.sin(a)
        lid = torch.zeros(m, 13, device=dev)
        lid[:, 0] = origin[:, 0] + cx
        lid[:, 1] = origin[:, 1] + cy + iy / 2 + t + (-off) * ca - (c.lid_t / 2) * sa
        lid[:, 2] = origin[:, 2] + c.surface_z + wall_h + (-off) * sa + (c.lid_t / 2) * ca
        lid[:, 3] = math.cos(a / 2)
        lid[:, 4] = math.sin(a / 2)
        self.lid.write_root_state_to_sim(lid, env_ids)

        a0, a1 = (math.radians(v) for v in c.spawn_arc)
        for k, (name, kind, dims) in enumerate(c.manifest):
            ang = a0 + (a1 - a0) * (k + 0.5) / len(c.manifest)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = origin[:, 0] + cx + c.spawn_radius * math.cos(ang)
            st[:, 1] = origin[:, 1] + cy + c.spawn_radius * math.sin(ang)
            st[:, 2] = origin[:, 2] + c.surface_z + (dims[2] / 2 if kind == "box" else dims[0])
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            if kind == "cyl":  # lying on its side: cylinder local z (its axis) -> world x
                st[:, 3] = math.cos(math.pi / 4)
                st[:, 5] = math.sin(math.pi / 4)  # 90 deg about y
            else:
                st[:, 3] = 1.0
            self.cargo[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "cargo": {n: b.data.root_state_w[env_ids].clone() for n, b in self.cargo.items()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for n, b in self.cargo.items():
            b.write_root_state_to_sim(state["cargo"][n], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        ix, iy, iz = c.interior
        parts = ", ".join(f"{n} ({kind} {dims})" for n, kind, dims in c.manifest)
        return (
            f"A wooden shipping crate (interior {ix:.2f} x {iy:.2f} x {iz:.2f} m) stands open on "
            f"the floor, lid hinged at its back edge. {len(c.manifest)} loose parts lie scattered "
            f"around it: {parts} (sizes in m). Packed carelessly they overflow; they only all fit "
            f"layered — the big slab flat on the crate floor, the tubes side by side on the slab, "
            f"the brick on the tubes.\n"
            f"Goal: place EVERY part fully inside the crate and close the lid so it rests flat on "
            f"the rim. A part sticking up past the rim will block the lid; the task is done only "
            f"when the lid is closed flush with everything inside."
        )

    # ----- progress -----------------------------------------------------------------------------
    def lid_angle_deg(self) -> torch.Tensor:
        """Lid opening angle per env (deg): 0 = flat on the rim."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        lid_up = quat_apply(self.lid.data.root_quat_w, ez)
        cosang = lid_up[:, 2].clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(cosang))

    def inside(self) -> torch.Tensor:
        """Whether each cargo part's centre is inside the crate interior volume,
        shape (num_envs, num_parts)."""
        c = self.cfg
        ix, iy, iz = c.interior
        t = c.wall_t
        lo = torch.tensor([c.crate_pos[0] - ix / 2, c.crate_pos[1] - iy / 2,
                           c.surface_z + t - 0.005], device=self.env.device)
        hi = torch.tensor([c.crate_pos[0] + ix / 2, c.crate_pos[1] + iy / 2,
                           c.surface_z + t + iz], device=self.env.device)
        cols = []
        for n, b in self.cargo.items():
            p = b.data.root_pos_w - self.env_origins
            cols.append(((p >= lo) & (p <= hi)).all(dim=-1))
        return torch.stack(cols, dim=1)

    def lid_seated(self) -> torch.Tensor:
        """Lid closed AND resting at rim height (not propped on proud cargo), per env."""
        c = self.cfg
        rim_z = c.surface_z + c.interior[2] + c.wall_t
        closed = self.lid_angle_deg() <= c.lid_close_deg
        lid_z = self.lid.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        at_rim = (lid_z - (rim_z + c.lid_t / 2)).abs() <= c.lid_seat_tol
        settled = self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return closed & at_rim & settled

    def packed(self) -> torch.Tensor:
        """Task success per env: everything inside + lid seated flush."""
        return self.inside().all(dim=1) & self.lid_seated()

    def success(self) -> torch.Tensor:
        """Scene-level success alias for `packed()` (matches the other suites' surface)."""
        return self.packed()

