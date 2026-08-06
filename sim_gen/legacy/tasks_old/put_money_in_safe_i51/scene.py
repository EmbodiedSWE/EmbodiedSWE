"""VaultChuteScene — dam the vault's return chute, then bank the cash (sim_gen task
`put_money_in_safe_i51`, derived from rlbench/put_money_in_safe).

The seed grasps a dollar stack and sets it down on a shelf inside an open safe: the whole
task is one carry-and-place into a passive, welcoming container. Here the container is
ADVERSARIAL and placement alone is physically futile: the vault's only "shelf" is a
polished steel chute tilted toward the permanently open doorway (a night-depository
return ramp), and the cash stacks are slick banded bundles (friction_combine_mode="min",
mu ~0.1 << tan(18 deg)) — anything laid on the chute accelerates back out of the vault
and ends up on the floor outside. The seed's own strategy is therefore the tested
failing control, rejected by construction (the cfg asserts tan(tilt) >= 2x the cash's
static mu). The required PLAN is different in kind, not in numbers: first FASHION A
RETAINER — a high-friction rubber chock bar lies on the floor and, laid across the chute
(its friction margin against the ramp is asserted >= 1.5x the gravity load of every cash
stack pushing on it), it becomes a dam; then lay the cash uphill AGAINST the dam, where
normal contact — not friction — holds it. Execution order is physically forced: cash
placed before the dam exists has left the vault long before the chock arrives.

Judged on PHYSICAL outcomes only:
  - success() : every PRESENT cash stack currently rests INSIDE the vault on the chute
    (safe-body-frame containment box + height band above the chute surface), settled,
    AND has held that state for `secure_substeps` consecutive substeps at least once
    (the `secured` latch) — a teleported-in stack without a dam starts sliding within a
    few substeps and never accrues the latch, so kinematic flashes never count.
  - score()   : 0 for doing nothing; 0.25 for the chock currently deployed as a dam on
    the chute; +0.10 * (fraction of present cash that ever entered the vault — latched,
    so the seed's slide-through attempt earns a sliver, far below success); +0.55 *
    (fraction of present cash currently secured); exactly 1.0 iff success().

Per-episode randomization: vault pose (xy jitter + FULL yaw, so the doorway direction
must be read from the scene), cash / chock scatter poses on the floor in front of the
doorway (bearing + radius + free yaw), and cash-count subset sampling (1 or 2 stacks
present; absent stacks park in an off-camera depot) — success is judged on the sampled
subset, so a memorized fixed sequence fails.

Assets are fully procedural: the vault is ONE kinematic compound body (walls, roof,
header, tilted chute slab, visual-only dial), re-posed per reset (pose-only writes — no
joints anywhere, so the never-teleport-jointed-pairs rule is moot); cash stacks are
compound bodies (box collider + visual-only band strap) with a slick bound material;
the chock is a plain cuboid with a grippy material. Heavy imports (isaaclab, pxr) are
deferred so importing this module stays app-free.
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


# ----- compound spawners -----------------------------------------------------------------------
# One rigid body each, several child colliders + visual-only decoration, authored with raw pxr
# APIs; `isaaclab.sim.utils.clone` provides the per-env replicate machinery (the pen_holder
# pattern — child colliders of one body never self-collide).

_SPAWNER_CACHE: dict[str, Any] = {}


def _box_child(stage, prim_path: str, name: str, size, center, color, contact_offset: float,
               rot_y_deg: float = 0.0, collide: bool = True):
    """Author one box child collider (Cube prim scaled to `size`) at `center`."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_y_deg != 0.0:
        xf.AddRotateYOp().Set(float(rot_y_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return seg


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault at `prim_path`: KINEMATIC root Xform, box walls / back / roof /
    front header, the tilted chute slab (rotated box, descending toward the +x doorway and
    running all the way to the ground outside), and a visual-only gold dial. The chute
    material (bound at the root AFTER the colliders exist — binding first silently no-ops)
    is medium-friction, combine "average": the cash's "min" material wins on the cash-chute
    pair (slick), the chock's "multiply" wins on the chock-chute pair (grippy)."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    hd, hw = cfg.hd, cfg.hw
    wt, wh = cfg.wall_t, cfg.wall_h
    co = cfg.contact_offset
    span_x = 2 * hd + 2 * wt  # wall/roof footprint along x
    span_y = 2 * hw + 2 * wt
    wall_col = cfg.color
    _box_child(stage, prim_path, "wall_l", (span_x, wt, wh), (0.0, hw + wt / 2, wh / 2),
               wall_col, co)
    _box_child(stage, prim_path, "wall_r", (span_x, wt, wh), (0.0, -(hw + wt / 2), wh / 2),
               wall_col, co)
    _box_child(stage, prim_path, "back", (wt, span_y, wh), (-(hd + wt / 2), 0.0, wh / 2),
               wall_col, co)
    _box_child(stage, prim_path, "roof", (span_x, span_y, cfg.roof_t),
               (0.0, 0.0, wh + cfg.roof_t / 2), wall_col, co)
    _box_child(stage, prim_path, "header", (wt, span_y, cfg.header_h),
               (hd + wt / 2, 0.0, wh - cfg.header_h / 2), wall_col, co)

    # chute slab: top surface z_s(x) = z_sill + (hd - x) * tan(tilt), spanning the cavity
    # x in [-hd, hd + tongue]; the tongue length makes the surface meet the ground exactly
    # at its tip, so rejected cash slides out onto the floor in one continuous motion.
    th = math.radians(cfg.tilt_deg)
    x0, x1 = -hd, hd + cfg.tongue
    x_mid = (x0 + x1) / 2
    length = (x1 - x0) / math.cos(th)
    z_mid = cfg.z_sill + (hd - x_mid) * math.tan(th)
    ctr = (x_mid - (cfg.ramp_t / 2) * math.sin(th), 0.0, z_mid - (cfg.ramp_t / 2) * math.cos(th))
    _box_child(stage, prim_path, "chute", (length, span_y, cfg.ramp_t), ctr,
               cfg.chute_color, co, rot_y_deg=math.degrees(th))

    # visual-only gold dial on the header (safe provenance nod) — NO CollisionAPI
    dial = UsdGeom.Cylinder.Define(stage, f"{prim_path}/dial")
    dial.CreateRadiusAttr(0.022)
    dial.CreateHeightAttr(0.010)
    dial.CreateExtentAttr([Gf.Vec3f(-0.022, -0.022, -0.005), Gf.Vec3f(0.022, 0.022, 0.005)])
    dxf = UsdGeom.Xformable(dial.GetPrim())
    dxf.AddTranslateOp().Set(Gf.Vec3d(hd + wt + 0.001, 0.06, wh - cfg.header_h / 2))
    dxf.AddRotateYOp().Set(90.0)
    dial.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.68, 0.15)])

    # chute material — bind AFTER the colliders are authored (earlier binds no-op silently)
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.ramp_mu, dynamic_friction=cfg.ramp_mu, restitution=0.0,
        friction_combine_mode="average", restitution_combine_mode="min")
    mat.func(f"{prim_path}/physmat", mat)
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _spawn_cash(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one cash stack at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, one box collider, plus a visual-only band strap across the middle (no
    CollisionAPI — the pen-tip pattern). The slick bill-wrapper material is bound at the
    root with combine "min", so the stack is slippery on EVERYTHING it touches."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.05)

    sx, sy, sz = cfg.size
    _box_child(stage, prim_path, "stack", (sx, sy, sz), (0.0, 0.0, 0.0),
               cfg.color, cfg.contact_offset)
    band = UsdGeom.Cube.Define(stage, f"{prim_path}/band")  # visual only — NO CollisionAPI
    band.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(band.GetPrim())
    bxf.AddScaleOp().Set(Gf.Vec3f(sx + 0.002, 0.030, sz + 0.002))
    band.CreateDisplayColorAttr([Gf.Vec3f(*cfg.band_color)])

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.mu_s, dynamic_friction=cfg.mu_d, restitution=0.0,
        friction_combine_mode="min", restitution_combine_mode="min")
    mat.func(f"{prim_path}/physmat", mat)
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _vault_spawner_cfg(c: Any) -> Any:
    """Build (lazily, app required) the vault spawner cfg from the scene cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            hd: float = 0.12
            hw: float = 0.12
            wall_t: float = 0.012
            wall_h: float = 0.26
            roof_t: float = 0.012
            header_h: float = 0.06
            z_sill: float = 0.013
            ramp_t: float = 0.015
            tilt_deg: float = 18.0
            tongue: float = 0.04
            ramp_mu: float = 0.8
            color: tuple = (0.30, 0.32, 0.38)
            chute_color: tuple = (0.62, 0.66, 0.72)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg

    return _SPAWNER_CACHE["vault"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        hd=c.hd, hw=c.hw, wall_t=c.wall_t, wall_h=c.wall_h, roof_t=c.roof_t,
        header_h=c.header_h, z_sill=c.z_sill, ramp_t=c.ramp_t, tilt_deg=c.tilt_deg,
        tongue=c.tongue, ramp_mu=c.ramp_mu, color=c.vault_color,
        chute_color=c.chute_color, contact_offset=c.contact_offset,
    )


def _cash_spawner_cfg(c: Any, size: tuple, color: tuple, band_color: tuple) -> Any:
    """Build (lazily, app required) one cash-stack spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cash" not in _SPAWNER_CACHE:

        @configclass
        class CashSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cash)
            size: tuple = (0.065, 0.130, 0.024)
            color: tuple = (0.18, 0.52, 0.26)
            band_color: tuple = (0.92, 0.92, 0.88)
            mu_s: float = 0.10
            mu_d: float = 0.08
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cash"] = CashSpawnerCfg

    return _SPAWNER_CACHE["cash"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cash_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        size=size, color=color, band_color=band_color,
        mu_s=c.cash_mu_s, mu_d=c.cash_mu_d, contact_offset=0.002,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class VaultChuteSceneCfg(BaseCfg):
    """Config for `VaultChuteScene`. Vault body frame: doorway faces +x, root at the ground
    center of the cavity footprint; chute surface z_s(x) = z_sill + (hd - x) * tan(tilt)."""

    # --- tunable: chute / material physics (the honesty knobs, asserted below) --------------
    tilt_deg: float = tunable(18.0)      # chute pitch toward the doorway
    cash_mu_s: float = tunable(0.10)     # slick cash, combine "min": mu_s << tan(tilt)
    cash_mu_d: float = tunable(0.08)
    chock_mu_s: float = tunable(1.2)     # grippy rubber chock, combine "multiply"
    chock_mu_d: float = tunable(1.0)
    ramp_mu: float = tunable(0.8)        # chute's own material (combine "average")

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging settled (m/s)
    secure_substeps: int = tunable(30)   # consecutive inside+settled substeps to latch secured
    side_margin: float = tunable(0.02)   # strict containment: |y| < hw - this
    back_margin: float = tunable(0.03)   # strict containment: x > -hd + this
    front_margin: float = tunable(0.075)  # strict containment: x < hd - this (fully behind
    # the doorway plane: asserted >= the largest cash stack's horizontal half-diagonal)
    z_band: tuple = tunable((0.004, 0.055))  # cash center height above the chute surface
    dam_x_margin: float = tunable(0.02)  # dam counts only with the chock fully inside
    dam_z_band: tuple = tunable((0.004, 0.05))

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    safe_jitter: float = tunable(0.05)   # vault center xy jitter (m)
    safe_yaw_deg: float = tunable(180.0)  # vault yaw: uniform +/- (FULL yaw)
    item_radius: tuple = tunable((0.30, 0.38))  # scatter radius band from the vault center
    item_bearings: tuple = tunable((25.0, 68.0, -38.0))  # deg off the doorway axis:
    # cash_a, cash_b, chock (spacing dry-checked against half-diagonals + jitter)
    bearing_jitter_deg: float = tunable(5.0)
    item_yaw_deg: float = tunable(180.0)  # free yaw per item
    subset_sample: bool = tunable(True)   # per-episode cash-count sampling (smoke toggles)
    min_present: int = tunable(1)

    # --- info: structure ---------------------------------------------------------------------
    hd: float = info(0.12)               # cavity half-depth (x)
    hw: float = info(0.12)               # cavity half-width (y)
    wall_t: float = info(0.012)
    wall_h: float = info(0.26)
    roof_t: float = info(0.012)
    header_h: float = info(0.06)
    z_sill: float = info(0.013)          # chute surface height at the doorway plane
    ramp_t: float = info(0.015)
    vault_color: tuple = info((0.30, 0.32, 0.38))
    chute_color: tuple = info((0.62, 0.66, 0.72))
    # (name, size, color, band rgb) — two bill bundles, slightly different denominations
    cash_dims: tuple = info(((0.065, 0.130, 0.024), (0.060, 0.120, 0.022)))
    cash_colors: tuple = info(((0.18, 0.52, 0.26), (0.14, 0.44, 0.30)))
    band_colors: tuple = info(((0.92, 0.92, 0.88), (0.90, 0.78, 0.30)))
    cash_mass: float = info(0.08)
    chock_size: tuple = info((0.034, 0.140, 0.034))
    chock_mass: float = info(0.12)
    chock_color: tuple = info((0.55, 0.12, 0.10))
    contact_offset: float = info(0.003)
    parking_pos: tuple = info((1.4, 1.4))  # off-camera depot for absent cash

    # Derived (filled in __post_init__).
    tan_t: float = field(default=None, init=False)
    sin_t: float = field(default=None, init=False)
    cos_t: float = field(default=None, init=False)
    tongue: float = field(default=None, init=False)
    cash_names: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        th = math.radians(self.tilt_deg)
        self.tan_t, self.sin_t, self.cos_t = math.tan(th), math.sin(th), math.cos(th)
        # tongue length makes the chute surface meet the ground exactly at its tip
        self.tongue = round(self.z_sill / self.tan_t, 4)
        self.cash_names = ("cash_a", "cash_b")
        # --- honesty by construction ---------------------------------------------------------
        # 1) the seed's plan is physically rejected: slick cash cannot rest on the chute
        assert self.tan_t >= 2.0 * self.cash_mu_s, \
            f"chute must reject bare cash: tan({self.tilt_deg}) < 2*mu_s"
        # 2) the chock dam holds BOTH stacks: friction reserve vs the summed gravity load
        hold = self.chock_mu_d * self.ramp_mu * self.chock_mass * self.cos_t
        push = 2 * self.cash_mass * (self.sin_t - self.cash_mu_d * self.cos_t)
        assert hold >= 1.5 * push, f"chock dam under-margined: hold={hold:.3f} push={push:.3f}"
        # 3) a strictly-contained stack is fully behind the doorway plane, any yaw
        half_diag = max(math.hypot(d[0] / 2, d[1] / 2) for d in self.cash_dims)
        assert self.front_margin >= half_diag - 1e-6, \
            f"front_margin {self.front_margin} < cash half-diagonal {half_diag:.4f}"

    # --- pure-math helpers (torch-free scalars; tensors also work elementwise) --------------
    def surf_z(self, x):
        """Chute top-surface height at body-frame x (valid for x <= hd + tongue)."""
        return self.z_sill + (self.hd - x) * self.tan_t

    def rest_local(self, x_surf: float, y: float, half_t: float,
                   gap: float = 0.002) -> tuple:
        """Body-frame center of a slab of half-thickness `half_t` resting flush on the
        chute, its center-normal projection touching the surface at `x_surf`."""
        off = half_t + gap
        return (x_surf + off * self.sin_t, y, self.surf_z(x_surf) + off * self.cos_t)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("vault_chute")
class VaultChuteScene(BaseScene):
    cfg: VaultChuteSceneCfg

    def __init__(self, cfg: VaultChuteSceneCfg | None = None) -> None:
        super().__init__(cfg or VaultChuteSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic vault (re-posed by reset), two cash stacks and the
        rubber chock (reset scatters them in front of the sampled doorway)."""
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
                spawn=sim_utils.DomeLightCfg(intensity=2600.0, color=(0.9, 0.9, 0.9)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=_vault_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "chock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chock",
                spawn=sim_utils.CuboidCfg(
                    size=c.chock_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.chock_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.chock_mu_s, dynamic_friction=c.chock_mu_d,
                        restitution=0.0, friction_combine_mode="multiply",
                        restitution_combine_mode="min"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.chock_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.35, -0.25, c.chock_size[2] / 2 + 0.002)),
            ),
        }
        for i, name in enumerate(c.cash_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cash_" + name,
                spawn=_cash_spawner_cfg(c, c.cash_dims[i], c.cash_colors[i],
                                        c.band_colors[i]),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.35, 0.15 + 0.2 * i, c.cash_dims[i][2] / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the presence mask and the rubric latches."""
        super().bind(env)
        c = self.cfg
        n, dev = env.num_envs, env.device
        self.vault: RigidObject = env.iscene["vault"]
        self.chock: RigidObject = env.iscene["chock"]
        self.cash: list[RigidObject] = [env.iscene[nm] for nm in c.cash_names]
        self.env_origins = env.iscene.env_origins
        k = len(c.cash_names)
        self.present = torch.ones(n, k, dtype=torch.bool, device=dev)
        self.entered = torch.zeros(n, k, dtype=torch.bool, device=dev)   # ever inside (latch)
        self.secured = torch.zeros(n, k, dtype=torch.bool, device=dev)   # persistence latch
        self.secure_cnt = torch.zeros(n, k, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the vault pose (xy jitter + full yaw, POSE-only kinematic
        write), scatter chock + present cash on the floor in front of the doorway, park
        absent cash in the depot, clear every latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        sxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.safe_jitter
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.safe_yaw_deg)

        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = sxy
        st[:, 3] = torch.cos(psi / 2)
        st[:, 6] = torch.sin(psi / 2)
        st[:, 0:3] += origin
        self.vault.write_root_pose_to_sim(st, env_ids)

        # cash-count subset sampling: k ~ U{min_present..2}; judged on the sampled subset
        k = len(c.cash_names)
        if c.subset_sample:
            kk = torch.randint(c.min_present, k + 1, (m,), device=dev)
        else:
            kk = torch.full((m,), k, dtype=torch.long, device=dev)
        rank = torch.rand(m, k, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < kk.unsqueeze(1)

        def scatter(base_bearing_deg: float, half_h: float) -> torch.Tensor:
            b = psi + math.radians(base_bearing_deg) \
                + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bearing_jitter_deg)
            r = c.item_radius[0] + (c.item_radius[1] - c.item_radius[0]) \
                * torch.rand(m, device=dev)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sxy[:, 0] + r * torch.cos(b)
            st[:, 1] = sxy[:, 1] + r * torch.sin(b)
            st[:, 2] = half_h + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.item_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            return st

        for i, body in enumerate(self.cash):
            st = scatter(c.item_bearings[i], c.cash_dims[i][2] / 2)
            park = torch.zeros(m, 13, device=dev)
            park[:, 0] = c.parking_pos[0] + 0.2 * i
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.cash_dims[i][2] / 2 + 0.002
            park[:, 3] = 1.0
            park[:, 0:3] += origin
            pres = self.present[env_ids, i].unsqueeze(1)
            body.write_root_state_to_sim(torch.where(pres, st, park), env_ids)

        self.chock.write_root_state_to_sim(
            scatter(c.item_bearings[2], c.chock_size[2] / 2), env_ids)

        self.entered[env_ids] = False
        self.secured[env_ids] = False
        self.secure_cnt[env_ids] = 0

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: latch `entered` (ever inside the cavity) and run the
        `secured` persistence counter — inside AND settled for `secure_substeps`
        consecutive substeps. A dam-less teleported stack starts sliding within a few
        substeps, so it never accrues the latch."""
        c = self.cfg
        pl = self.cash_pos_safe()
        loose = ((pl[..., 1].abs() < c.hw) & (pl[..., 0] > -c.hd)
                 & (pl[..., 0] < c.hd + 0.01)
                 & (pl[..., 2] > 0.0) & (pl[..., 2] < c.wall_h))
        self.entered |= loose
        ok = self.inside_strict(pl) & self.cash_settled()
        self.secure_cnt = torch.where(ok, self.secure_cnt + 1,
                                      torch.zeros_like(self.secure_cnt))
        self.secured |= self.secure_cnt >= c.secure_substeps

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "chock": self.chock.data.root_state_w[env_ids].clone(),
            "cash": [b.data.root_state_w[env_ids].clone() for b in self.cash],
            "present": self.present[env_ids].clone(),
            "entered": self.entered[env_ids].clone(),
            "secured": self.secured[env_ids].clone(),
            "secure_cnt": self.secure_cnt[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_pose_to_sim(state["vault"][:, 0:7], env_ids)
        self.chock.write_root_state_to_sim(state["chock"], env_ids)
        for b, st in zip(self.cash, state["cash"]):
            b.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self.entered[env_ids] = state["entered"]
        self.secured[env_ids] = state["secured"]
        self.secure_cnt[env_ids] = state["secure_cnt"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel vault stands on the floor, doorway permanently open; its only shelf "
            f"is a polished chute tilted {c.tilt_deg:.0f} deg TOWARD the doorway, running "
            f"out to the ground — slick banded cash laid on it slides straight back out of "
            f"the vault. In front of the doorway lie one or two cash bundles (count what "
            f"you see) and a dark-red rubber chock bar ({c.chock_size[1] * 100:.0f} cm); "
            f"rubber grips the chute, cash does not.\n"
            f"Goal: bank every cash bundle — each must come to REST fully inside the vault "
            f"on the chute. Lay the chock across the chute first as a dam, then rest the "
            f"cash uphill against it; cash placed on the bare chute is returned to sender. "
            f"A bundle straddling the doorway, resting outside, or parked on the vault "
            f"roof does not count."
        )

    # ----- frames / predicates ----------------------------------------------------------------
    def _to_safe(self, p_w: torch.Tensor) -> torch.Tensor:
        """World positions (..., 3) -> vault body frame (vault is a pure-yaw kinematic)."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        sp = self.vault.data.root_pos_w
        sq = self.vault.data.root_quat_w
        if p_w.dim() == 3:
            k = p_w.shape[1]
            sq = sq[:, None, :].expand(n, k, 4).reshape(-1, 4)
            return quat_apply_inverse(sq, (p_w - sp[:, None, :]).reshape(-1, 3)) \
                .reshape(n, k, 3)
        return quat_apply_inverse(sq, p_w - sp)

    def cash_pos_safe(self) -> torch.Tensor:
        """(N, 2, 3) cash centers in the vault body frame."""
        return self._to_safe(torch.stack([b.data.root_pos_w for b in self.cash], dim=1))

    def chock_pos_safe(self) -> torch.Tensor:
        """(N, 3) chock center in the vault body frame."""
        return self._to_safe(self.chock.data.root_pos_w)

    def cash_settled(self) -> torch.Tensor:
        """(N, 2) bool: cash |lin vel| below `settle_speed`."""
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cash], dim=1)
        return vel < self.cfg.settle_speed

    def inside_strict(self, pl: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 2) bool, geometric: cash center inside the cavity containment box (fully
        behind the doorway plane, off the walls) AND within `z_band` of the chute surface
        directly below it — i.e. actually resting on/over the chute, not floating."""
        c = self.cfg
        if pl is None:
            pl = self.cash_pos_safe()
        dz = pl[..., 2] - c.surf_z(pl[..., 0])
        return ((pl[..., 1].abs() < c.hw - c.side_margin)
                & (pl[..., 0] > -c.hd + c.back_margin)
                & (pl[..., 0] < c.hd - c.front_margin)
                & (dz > c.z_band[0]) & (dz < c.z_band[1]))

    def dam_ok(self) -> torch.Tensor:
        """(N,) bool: the chock currently deployed as a dam — resting on the chute, fully
        inside the cavity, settled."""
        c = self.cfg
        p = self.chock_pos_safe()
        dz = p[:, 2] - c.surf_z(p[:, 0])
        still = self.chock.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return ((p[:, 1].abs() < c.hw - 0.01)
                & (p[:, 0] > -c.hd + c.dam_x_margin) & (p[:, 0] < c.hd - c.dam_x_margin)
                & (dz > c.dam_z_band[0]) & (dz < c.dam_z_band[1]) & still)

    def secured_now(self) -> torch.Tensor:
        """(N, 2) bool: latched secured AND currently inside + settled (a stack that later
        slid out stops counting — the latch gates, the present state judges)."""
        return self.secured & self.inside_strict() & self.cash_settled()

    def success(self) -> torch.Tensor:
        """(N,) bool: every PRESENT cash stack is secured_now."""
        return (self.secured_now() | ~self.present).all(dim=1) & self.present.any(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * dam deployed now + 0.10 * fraction ever entered
        (latched — the seed's slide-through earns a sliver) + 0.55 * fraction secured now;
        exactly 1.0 iff success()."""
        pres = self.present.float()
        denom = pres.sum(dim=1).clamp(min=1.0)
        ent = (self.entered & self.present).float().sum(dim=1) / denom
        sec = (self.secured_now() & self.present).float().sum(dim=1) / denom
        base = (0.25 * self.dam_ok().float() + 0.10 * ent + 0.55 * sec).clamp(max=0.9)
        return torch.where(self.success(), torch.ones_like(base), base)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="vault_chute", robot="null", env_spacing=4.0))
