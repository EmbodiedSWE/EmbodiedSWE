"""ToolHangupScene — sort three screwdrivers onto a SIZE-KEYED hanging rack (seed inversion).

Derived from pick_place/track_screwdriver but strategically inverted. The seed is pure
TRAJECTORY TRACKING: the screwdriver starts already grasped, five waypoint markers show
the exact free-space path, and the whole task is following that prescribed curve while
never touching the environment (the episode even terminates if the object leaves the
gripper). Here every one of those pillars is flipped:

  - NOTHING is pre-grasped and NO path is given: three screwdrivers of graded sizes lie
    flat on the floor and the solver must derive its own plan;
  - the goal is CONTACT-RICH SUSPENSION, not free-space motion: each tool must be
    reoriented shaft-down and threaded through an elevated eyelet ring until it HANGS —
    shaft through the aperture, handle caught on the rim, dangling under gravity;
  - the correspondence must be REASONED: the three ring apertures are size-keyed
    (shaft_r < aperture < handle_r only for the matching tool) and the stand order along
    the rack is PERMUTED per episode, so there is no fixed sequence to memorize. The
    keying is honest physics at the extremes: the small tool's handle fits through the
    large ring (it falls straight through to the floor) and the large tool's shaft
    cannot enter the small ring (both are smoke negative controls);
  - the seed's own plan — carry the tool along a nice aerial path and keep holding it —
    latches only the tiny "raised" credit and can never reach success (negative control).

Three stages of breadth (one keyed hang per present tool, any order; each hang is its
own sub-sequence: reorient flat->vertical, align over the matching ring, thread, release).

Rubric (graded, in [0, 1]; transients latched in post_step):
  0.00      nothing (null policy);
  0.10/P    per present tool ever RAISED to rack height (latched — carrying counts);
  0.80/P    per present tool CURRENTLY hanging in ITS OWN ring and settled;
  1.00      iff success(): every present tool hanging in its matching ring, settled.
Success is a physical, current-state predicate: a tool that fell through, tipped off the
rack, or lies on the floor scores only its latched raise credit.

Assets are fully procedural, one rigid body each (compound spawners, the pen_holder /
tee_up pattern): three KINEMATIC eyelet stands (foot + post + arm + 8-box octagonal
ring; fixed furniture, re-POSED per reset with row jitter + yaw + slot permutation,
verified by readback) and three dynamic screwdrivers (shaft cylinder + fat handle
cylinder, explicit LOW center of mass in the steel shaft so a hung tool is a stable
pendulum). Ring band colors match the handle colors, so the keying is also perceptible.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(xform, translation, orientation, kinematic: bool, mass: float,
                com_z: float | None = None):
    """Shared root-body authoring: xform ops (authored fresh on the newly defined prim, so
    cloning never sees a duplicate op), RigidBodyAPI (+kinematic flag), explicit mass and —
    for the tools — an explicit center of mass down in the shaft (steel shaft, plastic
    handle) so a hung tool is a stable pendulum."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com_z is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(com_z)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    # Cap the contact-solver pop: a handle landing on the ring rim penetrates a little in
    # one 120 Hz step and the default 3 m/s depenetration would eject it ballistically.
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        # a whiff of damping so the hung pendulum crosses the settle gate promptly
        px.CreateLinearDampingAttr(0.08)
        px.CreateAngularDampingAttr(0.20)
    return root


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, prim_path: str, *, size: tuple, center: tuple, color,
         contact_offset: float, yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    b = UsdGeom.Cube.Define(stage, prim_path)
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        xf.AddRotateZOp().Set(yaw_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    b.CreateDisplayColorAttr([color])
    _collide(b.GetPrim(), contact_offset)


def _cyl(stage, prim_path: str, *, radius: float, height: float, z_center: float, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    d = UsdGeom.Cylinder.Define(stage, prim_path)
    d.CreateRadiusAttr(radius)
    d.CreateHeightAttr(height)
    d.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                        Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(d.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_center))
    d.CreateDisplayColorAttr([color])
    _collide(d.GetPrim(), contact_offset)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Kinematic eyelet stand, root frame at the GROUND point directly under the ring
    axis: foot plate + square post (offset -y) + horizontal arm + 8 box segments forming
    an octagonal eyelet ring of inner inradius `aperture` centered on the root axis with
    its band spanning z in [ring_mid - band_h/2, ring_mid + band_h/2]."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = _apply_root(xform, translation, orientation, True, 1.0)
    gray = Gf.Vec3f(*cfg.frame_color)
    ring_rgb = Gf.Vec3f(*cfg.ring_color)
    a, wt, bh = cfg.aperture, cfg.wall_t, cfg.band_h
    ring_top = cfg.ring_mid + bh / 2
    dp = cfg.d_post  # post center distance from the ring axis (local -y)

    _box(stage, f"{prim_path}/foot", size=(0.12, 0.10, 0.010),
         center=(0.0, -dp, 0.005), color=gray, contact_offset=cfg.contact_offset)
    _box(stage, f"{prim_path}/post", size=(0.032, 0.032, ring_top - 0.010),
         center=(0.0, -dp, (0.010 + ring_top) / 2), color=gray,
         contact_offset=cfg.contact_offset)
    arm_y0, arm_y1 = -(a + wt / 2), -dp
    _box(stage, f"{prim_path}/arm",
         size=(0.026, abs(arm_y1 - arm_y0) + 0.020, bh),
         center=(0.0, (arm_y0 + arm_y1) / 2, cfg.ring_mid), color=ring_rgb,
         contact_offset=cfg.contact_offset)
    # 8 wall boxes forming the eyelet: inner aperture = regular octagon of inradius `a`
    r_mid = a + wt / 2
    seg_len = 2 * (a + wt) * math.tan(math.pi / 8) + 0.002
    for k in range(8):
        ang = 2 * math.pi * k / 8
        _box(stage, f"{prim_path}/ring_{k}", size=(wt, seg_len, bh),
             center=(r_mid * math.cos(ang), r_mid * math.sin(ang), cfg.ring_mid),
             color=ring_rgb, contact_offset=cfg.contact_offset,
             yaw_deg=math.degrees(ang))
    return root


def _spawn_driver(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One screwdriver, root frame at the shaft TIP, axis = local +z toward the handle:
    a slim shaft cylinder (z in [0, shaft_l]) and a fat handle cylinder (z in
    [shaft_l, shaft_l + handle_l]). Explicit CoM low in the shaft (see _apply_root)."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = _apply_root(xform, translation, orientation, False, cfg.mass_props.mass,
                       com_z=cfg.com_frac * cfg.shaft_l)
    _cyl(stage, f"{prim_path}/shaft", radius=cfg.shaft_r, height=cfg.shaft_l,
         z_center=cfg.shaft_l / 2, color=Gf.Vec3f(0.75, 0.75, 0.78),
         contact_offset=cfg.contact_offset)
    _cyl(stage, f"{prim_path}/handle", radius=cfg.handle_r, height=cfg.handle_l,
         z_center=cfg.shaft_l + cfg.handle_l / 2, color=Gf.Vec3f(*cfg.color),
         contact_offset=cfg.contact_offset)
    return root


def _stand_spawner_cfg(c: ToolHangupSceneCfg, k: int) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            aperture: float = 0.017
            wall_t: float = 0.008
            band_h: float = 0.012
            ring_mid: float = 0.22
            d_post: float = 0.045
            frame_color: tuple = (0.35, 0.35, 0.38)
            ring_color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg

    handle_r = c.tools[k][3]
    return _SPAWNER_CACHE["stand"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        aperture=c.apertures[k], wall_t=c.ring_wall_t, band_h=c.ring_band_h,
        ring_mid=c.ring_mid_z, d_post=handle_r + 0.022, frame_color=c.frame_color,
        ring_color=c.tools[k][5], contact_offset=c.contact_offset,
    )


def _driver_spawner_cfg(c: ToolHangupSceneCfg, k: int) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "driver" not in _SPAWNER_CACHE:

        @configclass
        class DriverSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_driver)
            shaft_r: float = 0.011
            shaft_l: float = 0.120
            handle_r: float = 0.021
            handle_l: float = 0.050
            com_frac: float = 0.35
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["driver"] = DriverSpawnerCfg

    _name, sr, sl, hr, hl, rgb = c.tools[k]
    return _SPAWNER_CACHE["driver"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tool_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        shaft_r=sr, shaft_l=sl, handle_r=hr, handle_l=hl, com_frac=c.com_frac,
        color=rgb, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ToolHangupSceneCfg(BaseCfg):
    """Config for `ToolHangupScene`. The size-keying chain is asserted in __post_init__:
    shaft_r[k] < aperture[k] < handle_r[k] for every tool (only the matching tool can
    hang), the small handle passes the large ring (falls through), and the large shaft
    cannot enter the small ring (blocked)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    hang_tilt_max_deg: float = tunable(35.0)  # tool axis within this of world-up when hung
    hang_z_tol: float = tunable(0.015)  # handle bottom at most this ABOVE the ring top
    hang_z_low: float = tunable(0.012)  # ... and at most this BELOW it (rim-edge tilt slack)
    xy_slack: float = tunable(0.006)  # xy tolerance = aperture + this (physics bounds it)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging hung (m/s)
    raise_z: float = tunable(0.17)  # handle bottom above this = "raised to rack height"

    # --- tunable: randomization (task-family knobs) ------------------------------------------
    rack_pos: tuple = tunable((0.30, 0.0))  # rack row center (xy)
    tools_center: tuple = tunable((-0.16, 0.0))  # scatter row center (xy)
    reset_pos_jitter: float = tunable(0.05)  # uniform +/- xy jitter (rack row AND tools)
    rack_yaw_deg: float = tunable(25.0)  # uniform +/- yaw of the whole rack row
    tool_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per tool (lying flat)
    permute_stands: bool = tunable(True)  # shuffle the ring-size order along the row
    subset_sample: bool = tunable(True)  # per-episode tool-count sampling
    min_present: int = tunable(2)  # lower bound of sampled tool count

    # --- info: structure ---------------------------------------------------------------------
    # (name, shaft_r, shaft_l, handle_r, handle_l, rgb) — sizes form the keying chain.
    tools: tuple = info((
        ("driver_s", 0.006, 0.110, 0.016, 0.045, (0.20, 0.40, 0.90)),
        ("driver_m", 0.011, 0.120, 0.021, 0.050, (0.15, 0.70, 0.30)),
        ("driver_l", 0.016, 0.130, 0.028, 0.055, (0.85, 0.20, 0.15)),
    ))
    apertures: tuple = info((0.012, 0.017, 0.023))  # eyelet inner inradii, keyed to tools
    ring_wall_t: float = info(0.008)
    ring_band_h: float = info(0.012)
    ring_mid_z: float = info(0.22)  # eyelet band mid height; large tool tip hangs ~0.10
    stand_pitch: float = info(0.16)  # slot spacing along the rack row
    tool_pitch: float = info(0.15)  # scatter row spacing
    tool_mass: float = info(0.12)
    com_frac: float = info(0.35)  # CoM at this fraction of shaft_l above the tip
    contact_offset: float = info(0.002)
    frame_color: tuple = info((0.35, 0.35, 0.38))
    parking_pos: tuple = info((1.1, 1.1))  # off-camera depot for absent tools

    # Derived (filled in __post_init__).
    ring_top_z: float = field(default=None, init=False)  # ring band top, stand frame
    n_tools: int = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.ring_top_z = round(self.ring_mid_z + self.ring_band_h / 2, 4)
        self.n_tools = len(self.tools)
        assert len(self.apertures) == self.n_tools
        for k, (_n, sr, sl, hr, _hl, _rgb) in enumerate(self.tools):
            a = self.apertures[k]
            assert sr < a - 0.004, f"tool {k}: shaft must pass its ring (funnel >= 4 mm)"
            assert hr >= a + 0.004, f"tool {k}: handle must catch on its ring"
            assert sl > self.ring_band_h + 0.05, f"tool {k}: shaft must dangle through"
        # honest keying at the extremes (both are smoke negative controls):
        assert self.tools[0][3] <= self.apertures[-1] - 0.005, \
            "small handle must fall through the large ring"
        assert self.tools[-1][1] >= self.apertures[0] + 0.003, \
            "large shaft must not enter the small ring"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tool_hangup")
class ToolHangupScene(BaseScene):
    cfg: ToolHangupSceneCfg

    def __init__(self, cfg: ToolHangupSceneCfg | None = None) -> None:
        super().__init__(cfg or ToolHangupSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
        for k, (name, _sr, _sl, hr, _hl, _rgb) in enumerate(c.tools):
            out[f"stand_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand_" + str(k),
                spawn=_stand_spawner_cfg(c, k),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1] + (k - 1) * c.stand_pitch, 0.0005)),
            )
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tool_" + name,
                spawn=_driver_spawner_cfg(c, k),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tools_center[0], c.tools_center[1] + (k - 1) * c.tool_pitch,
                         hr + 0.004),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
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
                "gpu_max_rigid_contact_count": 2**21,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.stands: list[RigidObject] = [env.iscene[f"stand_{k}"] for k in range(c.n_tools)]
        self.tool_bodies: list[RigidObject] = [env.iscene[t[0]] for t in c.tools]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        self.present = torch.ones(n, c.n_tools, dtype=torch.bool, device=env.device)
        self._latch_raised = torch.zeros(n, c.n_tools, dtype=torch.bool, device=env.device)
        self._shaft_l = torch.tensor([t[2] for t in c.tools], device=env.device)
        self._xy_tol = torch.tensor([a + c.xy_slack for a in c.apertures], device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the present-tool subset, re-pose the rack row (xy jitter +
        row yaw + per-episode slot PERMUTATION of the three keyed stands — kinematic
        teleport, verified by readback in the smoke), scatter present tools lying flat with
        free yaw, park absent ones off-camera, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- subset sampling: k ~ U{min_present .. n} present tools ---
        if c.subset_sample:
            kcnt = torch.randint(c.min_present, c.n_tools + 1, (m,), device=dev)
        else:
            kcnt = torch.full((m,), c.n_tools, dtype=torch.long, device=dev)
        rank = torch.rand(m, c.n_tools, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < kcnt.unsqueeze(1)

        # --- rack row: jitter + yaw + slot permutation ---
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        if c.permute_stands:
            perm = torch.rand(m, c.n_tools, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(c.n_tools, device=dev).expand(m, -1)
        half = yaw / 2
        for k in range(c.n_tools):
            d = (perm[:, k].float() - (c.n_tools - 1) / 2) * c.stand_pitch
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.rack_pos[0] + jit[:, 0] - sy * d
            st[:, 1] = c.rack_pos[1] + jit[:, 1] + cy * d
            st[:, 2] = 0.0005
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.stands[k].write_root_state_to_sim(st, env_ids)

        # --- tools: scatter row, lying flat with free yaw; absent -> parking depot ---
        tool_yaw_amp = math.radians(c.tool_yaw_deg)
        c45 = math.cos(math.pi / 4)
        for k, (_n, _sr, _sl, hr, _hl, _rgb) in enumerate(c.tools):
            scat = torch.zeros(m, 3, device=dev)
            scat[:, 0] = c.tools_center[0]
            scat[:, 1] = c.tools_center[1] + (k - 1) * c.tool_pitch
            scat[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            scat[:, 2] = hr + 0.004
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + 0.2 * k
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = hr + 0.004
            pres = self.present[env_ids, k].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, scat, park)
            # lying flat: q = qz(yaw) * qy(90 deg) (tool axis -> horizontal)
            th = (torch.rand(m, device=dev) * 2 - 1) * tool_yaw_amp / 2
            st[:, 3] = torch.cos(th) * c45
            st[:, 4] = -torch.sin(th) * c45
            st[:, 5] = torch.cos(th) * c45
            st[:, 6] = torch.sin(th) * c45
            self.tool_bodies[k].write_root_state_to_sim(st, env_ids)

        self._latch_raised[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the transient raise achievement at sim rate (buffers are fresh here)."""
        self._latch_raised |= self._raised_now()

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stands": [s.data.root_state_w[env_ids].clone() for s in self.stands],
            "tools": [t.data.root_state_w[env_ids].clone() for t in self.tool_bodies],
            "present": self.present[env_ids].clone(),
            "latch_raised": self._latch_raised[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for s, st in zip(self.stands, state["stands"]):
            s.write_root_state_to_sim(st, env_ids)
        for t, st in zip(self.tool_bodies, state["tools"]):
            t.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self._latch_raised[env_ids] = state["latch_raised"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        sizes = ", ".join(
            f"a {name.split('_')[1].upper()}-size one ({2 * sr * 1000:.0f} mm shaft, "
            f"{2 * hr * 1000:.0f} mm handle)"
            for name, sr, _sl, hr, _hl, _rgb in c.tools)
        aps = ", ".join(f"{2 * a * 1000:.0f} mm" for a in c.apertures)
        return (
            f"A hanging rack stands on the floor: three eyelet stands in a row, each an "
            f"open ring (openings {aps}) held {c.ring_mid_z * 1000:.0f} mm up on a post. "
            f"On the other side, screwdrivers lie flat on the floor: {sizes}. Between two "
            f"and three of them are present in any episode, and the order of the rings "
            f"along the rack is shuffled — ring colors match the handle they fit.\n"
            f"Goal: hang every screwdriver on the rack by threading its shaft down "
            f"through the ONE ring that fits it — the shaft passes and the fat handle "
            f"catches on the rim, so the tool hangs dangling. A tool dropped through a "
            f"too-big ring falls to the floor; a too-small ring will not admit the shaft; "
            f"a tool balanced on top of a ring does not count. Done when every present "
            f"screwdriver hangs settled in its matching ring."
        )

    # ----- predicates / rubric --------------------------------------------------------------
    def _tool_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(root/tip pos_w (N,T,3), axis_w (N,T,3), |lin vel| (N,T)) for all tools."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([t.data.root_pos_w for t in self.tool_bodies], dim=1)
        quat = torch.stack([t.data.root_quat_w for t in self.tool_bodies], dim=1)
        vel = torch.stack([t.data.root_lin_vel_w.norm(dim=-1)
                           for t in self.tool_bodies], dim=1)
        n, t = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * t, 3)
        axis = quat_apply(quat.reshape(n * t, 4), ez).reshape(n, t, 3)
        return pos, axis, vel

    def _stand_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(ring axis xy_w (N,T,2), ring band TOP z_w (N,T)) read back from the kinematic
        stand poses, so the rubric follows whatever randomization placed them."""
        sp = torch.stack([s.data.root_pos_w for s in self.stands], dim=1)
        return sp[:, :, :2], sp[:, :, 2] + self.cfg.ring_top_z

    def hung(self) -> torch.Tensor:
        """(N, T) bool, geometric: tool k judged ONLY at ITS OWN stand k — axis tip-down
        near-vertical, handle bottom in a narrow band at the ring top, shaft dangling well
        below the band, clear of the floor."""
        c = self.cfg
        pos, axis, _v = self._tool_tensors()
        ring_xy, ring_top = self._stand_tensors()
        hb = pos + axis * self._shaft_l[None, :, None]  # handle bottom point
        upright = axis[:, :, 2] >= math.cos(math.radians(c.hang_tilt_max_deg))
        near = (hb[:, :, :2] - ring_xy).norm(dim=-1) < self._xy_tol[None, :]
        dz = hb[:, :, 2] - ring_top
        in_band = (dz > -c.hang_z_low) & (dz < c.hang_z_tol)
        through = pos[:, :, 2] < ring_top - c.ring_band_h - 0.03  # tip well below the band
        clear = pos[:, :, 2] > 0.02  # not standing on the floor
        return upright & near & in_band & through & clear

    def settled(self) -> torch.Tensor:
        _p, _a, vel = self._tool_tensors()
        return vel < self.cfg.settle_speed

    def counted(self) -> torch.Tensor:
        """(N, T) bool: hung in the matching ring, settled AND present."""
        return self.hung() & self.settled() & self.present

    def _raised_now(self) -> torch.Tensor:
        """(N, T) bool: handle bottom lifted above `raise_z` (rack height) — the transient
        partial-progress achievement (carrying counts; lying on the floor never fires)."""
        pos, axis, _v = self._tool_tensors()
        hb_z = pos[:, :, 2] + axis[:, :, 2] * self._shaft_l[None, :]
        return hb_z > self.cfg.raise_z

    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT tool currently hanging in its matching ring, settled —
        a physical, present-state outcome judged on the sampled subset."""
        return (self.counted() | ~self.present).all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10/P per present tool ever raised to rack height
        (latched), 0.80/P per present tool currently hanging; 1.0 iff success()."""
        pn = self.present.sum(dim=1).clamp(min=1).float()
        raised = (self._latch_raised | self._raised_now()) & self.present
        s = 0.10 * raised.sum(dim=1).float() / pn + 0.80 * self.counted().sum(dim=1).float() / pn
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="tool_hangup", robot="null"))
