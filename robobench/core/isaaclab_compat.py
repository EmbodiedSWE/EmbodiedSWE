"""Shims that let robobench run on Isaac Lab 2.3 (Isaac Sim 5.1, main) and 3.0 (Isaac Sim 6, branch
isaacsim-6). Every cross-version difference in how a spawner cfg is edited lives here, so robots and
scenes never branch on the Isaac Lab version themselves. (The sim/render cfg split is in
`robobench.core.config.SimCfg.to_isaaclab` / `apply_render_settings`.)"""
from __future__ import annotations

from typing import Any


def set_fix_root_link(spawn: Any, fixed: bool) -> None:
    """Weld (or free) an articulation's root link at spawn.

    Isaac Lab 3.0 moved `fix_root_link` onto the spawner cfg (`UsdFileCfg.fix_root_link`) and turned
    `spawn.articulation_props` into a LIST of schema fragments (the stock isaaclab_assets robots ship
    it that way, so `spawn.articulation_props.fix_root_link = ...` dies with "'list' object has no
    attribute"). Isaac Lab 2.3 keeps the flag on `spawn.articulation_props`
    (an `ArticulationRootPropertiesCfg`)."""
    if hasattr(spawn, "fix_root_link"):  # Isaac Lab >= 3.0 spawner field
        spawn.fix_root_link = fixed
        return
    props = spawn.articulation_props
    if props is None or isinstance(props, (list, tuple)):
        raise TypeError(f"cannot set fix_root_link on spawn.articulation_props={props!r}")
    props.fix_root_link = fixed


def set_spawn_prop(spawn: Any, group: str, name: str, value: Any) -> None:
    """`spawn.<group>.<name> = value` for a props slot that is one cfg (Isaac Lab 2.3, or a 3.0
    single fragment) or a list of schema fragments (Isaac Lab 3.0): the first fragment carrying the
    attribute takes the value."""
    props = getattr(spawn, group)
    if isinstance(props, (list, tuple)):
        for frag in props:
            if hasattr(frag, name):
                setattr(frag, name, value)
                return
        raise AttributeError(
            f"no fragment in spawn.{group} carries '{name}' (fragments: {[type(f).__name__ for f in props]})")
    if props is None:
        raise AttributeError(f"spawn.{group} is None; cannot set '{name}'")
    setattr(props, name, value)


def add_app_launcher_args(parser: Any) -> None:
    """`AppLauncher.add_app_launcher_args(parser)` plus the two 2.3 flags Isaac Lab 3.0 dropped.

    3.0 runs headless unless `--visualizer` / `--livestream` asks for a view, and renders cameras
    whenever a CameraCfg exists, so `--headless` and `--enable_cameras` are accepted as no-ops there
    (defaults: headless=True, enable_cameras=False). Every documented CLI keeps working on both stacks,
    and `args.headless` exists before `AppLauncher(args)` on both."""
    from isaaclab.app import AppLauncher as _AppLauncher

    _AppLauncher.add_app_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", default=True,
                            help="no-op on Isaac Lab 3.0: headless unless --visualizer / --livestream")
    if "--enable_cameras" not in parser._option_string_actions:
        parser.add_argument("--enable_cameras", action="store_true", default=False,
                            help="no-op on Isaac Lab 3.0: cameras render whenever a CameraCfg exists")


# ----- quaternion / state convention boundary -----------------------------------------------------
# robobench is wxyz everywhere (cfg literals, recorded datasets, controllers, action spaces, the main
# branch's Isaac Sim 5.1 results; see robobench.core.quat). Isaac Lab 3.0 is xyzw everywhere. The
# helpers below are the ONLY place the two meet: no-ops on Isaac Lab 2.3.
def is_isaaclab_30() -> bool:
    """True on the Isaac Lab 3.0 line (Isaac Sim 6): the backend packages exist there."""
    global _IS_30
    if _IS_30 is None:
        try:
            import isaaclab_physx  # noqa: F401

            _IS_30 = True
        except ImportError:
            _IS_30 = False
    return _IS_30


_IS_30: bool | None = None
_WXYZ_TO_XYZW = (1, 2, 3, 0)
_XYZW_TO_WXYZ = (3, 0, 1, 2)


_PROXY_ARRAY: Any = None  # resolved lazily: isaaclab.utils.warp.ProxyArray on 3.0, else False


def _is_proxy_array(x: Any) -> bool:
    """ProxyArray forwards unknown attributes to its tensor (so `hasattr(x, "requires_grad")` is True on
    it); detect it by class, not by duck typing."""
    global _PROXY_ARRAY
    if _PROXY_ARRAY is None:
        try:
            from isaaclab.utils.warp import ProxyArray

            _PROXY_ARRAY = ProxyArray
        except ImportError:
            _PROXY_ARRAY = False
    return bool(_PROXY_ARRAY) and isinstance(x, _PROXY_ARRAY)


def _as_tensor(x: Any):
    """A torch tensor from a torch tensor or an Isaac Lab 3.0 ProxyArray (zero-copy view)."""
    return x.torch if _is_proxy_array(x) else x


def _permute_last(x: Any, order: tuple[int, int, int, int]):
    if isinstance(x, (tuple, list)):
        return type(x)(x[i] for i in order)
    return x[..., list(order)]


def to_isaaclab_quat(q: Any) -> Any:
    """wxyz (robobench) -> Isaac Lab's convention: xyzw on 3.0, unchanged on 2.3. Tuples or tensors."""
    return _permute_last(q, _WXYZ_TO_XYZW) if is_isaaclab_30() else q


def from_isaaclab_quat(q: Any) -> Any:
    """Isaac Lab's convention -> wxyz: the inverse of `to_isaaclab_quat`."""
    return _permute_last(q, _XYZW_TO_WXYZ) if is_isaaclab_30() else q


rot_cfg = to_isaaclab_quat  # alias for cfg literals: `InitialStateCfg(rot=rot_cfg((1, 0, 0, 0)))`


def quat_w(x: Any):
    """A wxyz tensor from an asset data quaternion property (`data.root_quat_w`, `data.body_quat_w`, ...)."""
    t = _as_tensor(x)
    return t[..., list(_XYZW_TO_WXYZ)] if is_isaaclab_30() else t


def state_w(x: Any):
    """A wxyz-ordered tensor from a state property whose quaternion sits at [..., 3:7]
    (`data.root_state_w` (…,13), `data.body_link_state_w` (…,13), `data.default_root_state`, poses (…,7)).
    On 3.0 this is a converted COPY; on 2.3 the property itself."""
    t = _as_tensor(x)
    if not is_isaaclab_30():
        return t
    out = t.clone()
    out[..., 3:7] = t[..., [6, 3, 4, 5]]
    return out


def state_to_isaaclab(state: Any):
    """wxyz state/pose (quat at [..., 3:7]) -> Isaac Lab's convention (a copy on 3.0)."""
    if not is_isaaclab_30():
        return state
    out = state.clone()
    out[..., 3:7] = state[..., [4, 5, 6, 3]]
    return out


def write_root_state(asset: Any, state: Any, env_ids: Any = None) -> None:
    """`asset.write_root_state_to_sim(state_wxyz, env_ids)` on both stacks (3.0: via the `_index`
    method, so no per-call deprecation warning; the quaternion is converted)."""
    s = state_to_isaaclab(state)
    fn = getattr(asset, "write_root_state_to_sim_index", None)
    if fn is not None:
        fn(root_state=s, env_ids=env_ids)  # keyword-only on 3.0
    else:
        asset.write_root_state_to_sim(s, env_ids)


def write_root_pose(asset: Any, pose: Any, env_ids: Any = None) -> None:
    """`asset.write_root_pose_to_sim(pose_wxyz, env_ids)` on both stacks (see `write_root_state`)."""
    p = state_to_isaaclab(pose)
    fn = getattr(asset, "write_root_pose_to_sim_index", None)
    if fn is not None:
        fn(root_pose=p, env_ids=env_ids)  # keyword-only on 3.0
    else:
        asset.write_root_pose_to_sim(p, env_ids)


# ----- PhysX tensor views ---------------------------------------------------------------------------
class _PhysxViewShim:
    """2.3-style access to a PhysX tensor view on Isaac Lab 3.0: torch in, torch out. The 3.0 views run
    the warp backend (wp.array results, wp.array values, int32 index arrays); robobench code keeps
    passing / editing torch tensors (`mats = view.get_material_properties(); mats[..., :2] = f;
    view.set_material_properties(mats, ids)`). Conversions are zero-copy where torch allows it."""

    __slots__ = ("_view",)

    def __init__(self, view: Any) -> None:
        object.__setattr__(self, "_view", view)

    def __getattr__(self, name: str):
        attr = getattr(self._view, name)
        if not callable(attr):
            return _wp_to_torch(attr)

        def call(*args, **kwargs):
            return _wp_to_torch(attr(*[_torch_to_wp(a) for a in args], **{k: _torch_to_wp(v) for k, v in kwargs.items()}))

        return call


def _torch_to_wp(x: Any) -> Any:
    import torch

    if torch.is_tensor(x):
        import warp as wp

        if not x.is_floating_point() and x.dtype != torch.bool and x.dtype != torch.int32:
            x = x.to(torch.int32)  # index arrays
        return wp.from_torch(x.contiguous())
    return x


def _wp_to_torch(x: Any) -> Any:
    try:
        import warp as wp
    except ImportError:
        return x
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def physx_view(asset: Any) -> Any:
    """The asset's PhysX tensor view with 2.3 (torch) semantics on both stacks: 3.0's `root_view`
    wrapped in `_PhysxViewShim`, 2.3's `root_physx_view` as is."""
    if is_isaaclab_30():
        view = getattr(asset, "root_view", None)
        if view is None:
            view = asset.root_physx_view
        return _PhysxViewShim(view)
    return asset.root_physx_view


# ----- actuator cfg fields --------------------------------------------------------------------------
def set_effort_limit(actuator_cfg: Any, value: Any) -> None:
    """Set an actuator group's simulation effort limit on both stacks. Isaac Lab 3.0 renamed
    `effort_limit_sim` -> `joint_effort_limit` and REFUSES a cfg carrying both (the stock robot cfgs
    already set `joint_effort_limit`), so there the old field is cleared; 2.3 only knows the old one."""
    if hasattr(actuator_cfg, "joint_effort_limit"):
        actuator_cfg.joint_effort_limit = value
        actuator_cfg.effort_limit_sim = None
    else:
        actuator_cfg.effort_limit_sim = value


# ----- prim paths -----------------------------------------------------------------------------------
def env_prim_path(prim_path: str, env_i: int) -> str:
    """The concrete prim path of env `env_i` from an asset cfg's `prim_path`. Isaac Lab 2.3 leaves
    the `{ENV_REGEX_NS}` macro in `cfg.prim_path` (it stands for `/World/envs/env_.*`); 3.0 expands
    it at scene build to `/World/envs/env_[^/]+` (prim-path expressions are whole-path regexes
    there). Both forms resolve here."""
    import re

    p = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    return re.sub(r"env_(\.\*|\[\^/\]\+)", f"env_{env_i}", p)


def T(x: Any):
    """A torch tensor from an asset data property on both stacks: Isaac Lab 3.0 returns `ProxyArray`
    (a Warp-backed view whose `.device` is a Warp device, whose implicit tensor use is deprecated);
    `.torch` is the zero-copy tensor view. Anything that is not a ProxyArray (2.3 tensors, name
    lists, floats) passes through unchanged."""
    return x.torch if _is_proxy_array(x) else x


# ----- external wrenches --------------------------------------------------------------------------
def set_external_force_and_torque(asset: Any, forces: Any, torques: Any, *, positions: Any = None,
                                  body_ids: Any = None, env_ids: Any = None, is_global: bool = False) -> None:
    """`asset.set_external_force_and_torque(...)` on both stacks, with a WORLD-frame wrench rotated into
    the body frame on Isaac Lab 3.0.

    On Isaac Lab 3.0 the `is_global=True` path turns a press + twist on a staged part into pure spin
    with no descent, while the same wrench applied in the body frame reproduces the 2.3 behaviour.
    With `positions=None` the wrench acts at the body either way, so applying it in the body frame is
    exact. World-frame `positions` are not supported on 3.0 here."""
    if is_global and is_isaaclab_30():
        import torch

        from robobench.core.quat import quat_apply_inverse

        if positions is not None:
            raise NotImplementedError("is_global wrench with world-frame positions is not supported on Isaac Lab 3.0")
        data = asset.data
        q = quat_w(data.body_quat_w) if hasattr(data, "body_quat_w") else quat_w(data.root_quat_w).unsqueeze(1)
        if body_ids is not None:
            q = q[:, body_ids]
        if env_ids is not None:
            q = q[env_ids]

        def to_body(v):
            if v is None:
                return None
            v = torch.as_tensor(v, device=q.device, dtype=q.dtype)
            return quat_apply_inverse(q.expand(v.shape[0], v.shape[1], 4), v)

        forces, torques, is_global = to_body(forces), to_body(torques), False
    asset.set_external_force_and_torque(forces, torques, positions=positions, body_ids=body_ids, env_ids=env_ids,
                                        is_global=is_global)
