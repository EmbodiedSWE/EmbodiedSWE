"""Turn CLI intent (scene, robot, maybe controller) into a REGISTERED preset name.

robobench's registries are the task database — configs/envs.py presets carry the
verified-feasible layouts (reach bands, frictions, dt). Anything not registered
is not a valid experiment target: fail loudly at build time, never at agent time.
"""

from __future__ import annotations

# When the controller is unspecified, pick by this order and say so.
PREFERRED_MODES = ("osc", "impedance", "joint")

_discovered = False


def _ensure_discovered() -> None:
    global _discovered
    if not _discovered:
        import robobench

        robobench.discover()  # app-free; populates SCENES/ROBOTS/CONTROLLERS/ENVS
        _discovered = True


def list_envs() -> list[str]:
    _ensure_discovered()
    from robobench.core.registries import ENVS

    return ENVS.list()


def resolve_preset(scene: str, robot: str, controller: str | None = None, suite: str = "assembly") -> str:
    """Return the registered env name for scene+robot(+controller), or raise with the menu.

    Registered names follow `suite.scene[.robot[.control_mode]]`.
    """
    _ensure_discovered()
    names = list_envs()

    if controller:
        want = f"{suite}.{scene}.{robot}.{controller}"
        if want in names:
            return want
        near = [n for n in names if n.startswith(f"{suite}.{scene}.")]
        raise SystemExit(
            f"'{want}' is not registered — registration in configs/envs.py is the feasibility "
            f"gate (register + verify it first).\nRegistered for scene '{scene}': {near or 'none'}"
        )

    candidates = [n for n in names if n.startswith(f"{suite}.{scene}.{robot}")]
    exact_or_moded = [n for n in candidates if n == f"{suite}.{scene}.{robot}" or n.count(".") == 3]
    if not exact_or_moded:
        near = [n for n in names if n.startswith(f"{suite}.{scene}.")]
        raise SystemExit(
            f"no registered env for scene '{scene}' + robot '{robot}'.\n"
            f"Registered for scene '{scene}': {near or 'none'}"
        )
    for mode in PREFERRED_MODES:
        want = f"{suite}.{scene}.{robot}.{mode}"
        if want in exact_or_moded:
            print(f"[resolve] controller unspecified -> '{mode}' (preference {PREFERRED_MODES})")
            return want
    return sorted(exact_or_moded)[0]
