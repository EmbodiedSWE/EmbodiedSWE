"""Task scenery attached after physics initialization, with no simulation schemas."""
from __future__ import annotations

import copy
import math
import re
import time

from robobench.core.assets import asset_path, ensure_assets, PACKAGE_ROOT

def room_pose(room):
    yaw = math.radians(room.get("yaw", 0.0))
    x, y = room["anchor"]
    return (-math.cos(yaw)*x + math.sin(yaw)*y,
            -math.sin(yaw)*x - math.cos(yaw)*y,
            room["floor_world_z"] - room["room_floor_z"])


def physics_prims(root):
    from pxr import Usd

    for prim in Usd.PrimRange(root):
        schemas = set(prim.GetAppliedSchemas())
        op = prim.GetMetadata("apiSchemas")
        if op:
            schemas.update(str(s) for s in op.GetAddedOrExplicitItems())
        if prim.GetTypeName().startswith(("Physics", "Physx")) or any(
            s.startswith(("Physics", "Physx")) for s in schemas
        ):
            yield str(prim.GetPath())


def prepare_room(cfg, scene, robot):
    """Resolve assets and room spacing before constructing the physics world."""
    setting = cfg.room
    if setting is None:
        return None, cfg.env_spacing
    spec = copy.deepcopy(setting(cfg, scene, robot) if callable(setting) else setting)
    room_id = spec["room"]["id"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", room_id):
        raise ValueError(f"Invalid room asset id: {room_id!r}")
    spec["name"] = cfg.scene
    prefix = f"robobench/assets/rooms/{room_id}"
    ensure_assets([prefix])
    usd = asset_path(PACKAGE_ROOT / prefix / "scene_visual.usd")
    if not usd.is_file():
        raise FileNotFoundError(f"Room is not published in the asset manifest: {usd}")
    spec["room"]["usd"] = str(usd)
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd))
    root = stage.GetDefaultPrim() if stage else None
    if not root:
        raise ValueError(f"Room USD has no default prim: {usd}")
    live = list(physics_prims(root))
    if live:
        raise ValueError(f"Room contains simulation schemas: {live[:5]}")
    spacing = cfg.env_spacing
    if cfg.num_envs > 1:
        # Whole rooms would overlap at the benchmark's usual 2–3 m task spacing.
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(root).ComputeAlignedRange()
        size = bounds.GetSize()
        spacing = max(spacing, float(size[0])+1, float(size[1])+1)
    return spec, spacing


def spawn_room(env, spec):
    """Spawn physics-free room copies outside the initialized simulation model."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = env.stage
    stage.DefinePrim("/World/Rooms", "Xform")
    room = spec["room"]
    pos = room_pose(room)
    half_yaw = math.radians(room.get("yaw", 0.0))/2
    quat = (0., 0., math.sin(half_yaw), math.cos(half_yaw)) if spec["backend"] == "newton" else (math.cos(half_yaw), 0., 0., math.sin(half_yaw))
    origins = env.iscene.env_origins.detach().cpu().tolist()
    for index, origin in enumerate(origins):
        path = f"/World/Rooms/env_{index}"
        config = sim_utils.UsdFileCfg(usd_path=room["usd"])
        config.func(path, config, translation=tuple(pos[i]+origin[i] for i in range(3)), orientation=quat)
        for rel in room.get("hide", []):
            prim = stage.GetPrimAtPath(f"{path}/{rel}")
            if prim:
                prim.SetActive(False)
        for rel, name, value in room.get("attrs", []):
            prim = stage.GetPrimAtPath(f"{path}/{rel}")
            attr = prim.GetAttribute(name) if prim else None
            if attr:
                if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = Sdf.AssetPath(value)
                elif isinstance(value, list) and len(value) == 3:
                    value = Gf.Vec3f(*value)
                attr.Set(value)
        zones = spec.get("clear_zones", [])
        if zones:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                depth = prim.GetPath().pathElementCount - Sdf.Path(path).pathElementCount
                if not 1 <= depth <= spec.get("clear_depth", 3) or prim.GetTypeName() not in ("Xform", "Mesh", ""):
                    continue
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if bounds.IsEmpty() or max(bounds.GetSize()) > spec.get("clear_max_size", 1.6):
                    continue
                lo, hi = bounds.GetMin(), bounds.GetMax()
                if any(all(lo[i] <= b[i]+origin[i] and hi[i] >= a[i]+origin[i] for i in range(3)) for a, b in zones):
                    prim.SetActive(False)
        for i, pedestal in enumerate(spec.get("pedestals", [])):
            floor, top = room["floor_world_z"], pedestal.get("top", 0.)
            if top-floor <= .01:
                continue
            sx, sy = pedestal.get("size", [.18, .18])
            config = sim_utils.CuboidCfg(size=(sx, sy, top-floor), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.16,.16,.18)))
            config.func(f"/World/Rooms/Stand_{index}_{i}", config,
                        translation=(pedestal["pos"][0]+origin[0], pedestal["pos"][1]+origin[1], (top+floor)/2+origin[2]))
    patterns = [re.compile(p) for p in spec.get("hide", [])]
    for prim in stage.Traverse():
        if any(p.fullmatch(str(prim.GetPath())) for p in patterns):
            imageable = UsdGeom.Imageable(prim)
            # Hiding a parent also hides its descendants and inherited clones. Redundant
            # overrides on cloned collision meshes trigger PhysX shape reattachment.
            if imageable and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                imageable.MakeInvisible()  # retain the task's colliders
    live = list(physics_prims(stage.GetPrimAtPath("/World/Rooms")))
    if live:
        raise RuntimeError(f"Spawned room contains physics: {live[:5]}")
    if "camera" in spec:
        eye, target = (tuple(v[i]+origins[0][i] for i in range(3)) for v in spec["camera"])
        env.sim.set_camera_view(eye, target)
    print(f"[room] {spec['name']}: {len(origins)} room(s), zero physics prims", flush=True)


def wait_for_room(env, timeout=120):
    """Optional render warmup for previews; does not step the physics simulation."""
    import omni.usd

    start = time.monotonic()
    while True:
        _, loaded, total = omni.usd.get_context().get_stage_loading_status()
        if total == 0 or loaded >= total:
            break
        if time.monotonic()-start > timeout:
            raise TimeoutError("Room assets did not finish streaming")
        env.sim.render()
    for _ in range(8):
        env.sim.render()
