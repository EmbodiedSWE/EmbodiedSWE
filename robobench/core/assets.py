"""Resolve and fetch asset groups without importing the simulator.

COSIGEN_ASSET_DIR selects a writable mirror root (containing ``robobench/``).
The default is the checkout, matching the existing fetch_assets command.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def asset_root() -> Path:
    return Path(os.environ.get("COSIGEN_ASSET_DIR", str(PACKAGE_ROOT))).expanduser().resolve()


def asset_path(path: str | Path) -> Path:
    path = Path(path)
    try:
        return asset_root() / path.relative_to(PACKAGE_ROOT)
    except ValueError:
        return path  # explicit external asset overrides stay external


def _paths(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (str, Path)):
        yield Path(value)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _paths(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _paths(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _paths(item, seen)


def ensure_assets(prefixes) -> None:
    """Fetch only selected manifest groups; imports/registry listing never call this."""
    from robobench.scripts import fetch_assets

    prefixes = sorted(set(str(p).rstrip("/") for p in prefixes))
    if not prefixes:
        return
    # Text USD/material layers remain in Git; preserve their relative layout in an override root.
    if asset_root() != PACKAGE_ROOT:
        for prefix in prefixes:
            source = PACKAGE_ROOT / prefix
            if source.is_dir():
                for src in source.rglob("*"):
                    if src.is_file() and src.suffix.lower() in {".usda", ".mdl", ".urdf", ".xml", ".json"}:
                        dst = asset_path(src)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists() or src.read_bytes() != dst.read_bytes():
                            fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name+".")
                            try:
                                with os.fdopen(fd, "wb") as f:
                                    f.write(src.read_bytes())
                                os.replace(tmp, dst)
                            finally:
                                Path(tmp).unlink(missing_ok=True)
    if fetch_assets.fetch(prefixes=prefixes):
        raise RuntimeError(f"Asset verification failed for {prefixes}")


def scene_asset_groups(scene_cls) -> set[str]:
    parts = scene_cls.__module__.split(".")
    if len(parts) > 2 and parts[:2] == ["robobench", "suites"]:
        return {f"robobench/suites/{parts[2]}/assets"}
    return set()


def ensure_environment_assets(scene, robot) -> None:
    groups = scene_asset_groups(type(scene))
    # Tables and ground layers are shared across suites, including inline assets() paths.
    groups.add("robobench/suites/assembly/assets/props")

    def robot_cfgs(current):
        yield getattr(current, "cfg", None)
        for child in getattr(current, "robots", {}).values():
            yield from robot_cfgs(child)

    for path in _paths((getattr(scene, "cfg", None), list(robot_cfgs(robot)))):
        try:
            parts = path.relative_to(asset_root()).parts
        except ValueError:
            continue
        if "assets" in parts:
            i = parts.index("assets")
            # Robot models are independent directories; suite bundles preserve their shared layers.
            end = i + 2 if parts[:2] == ("robobench", "robots") and len(parts) > i + 1 else i + 1
            if parts[:2] == ("robobench", "robots") and "composites" in parts:
                end = i + 1  # attached arms reference sibling arm and gripper directories
            groups.add("/".join(parts[:end]))
    ensure_assets(groups)
