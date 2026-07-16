"""Pour smoke for the pouring suite — scripted kinematic milk-cup pour into the coffee cup.

The milk cup (kinematic rigid collider) follows a scripted trajectory: settle, lift off the
table, traverse to a pose next to the coffee cup, then tip — the cup is positioned **by its lip**
during the pour (`pos = lip_target - R(theta) @ lip_local`), so the pouring edge stays anchored
above the coffee cup mouth while the body swings. Poses + finite-difference twists are written
through the rigid-object API every physics tick (the same driving pattern as IsaacLab's in-tree
MPM pour demo).

Quats are **xyzw** (isaaclab develop / warp convention); the tilt is a pure rotation about +y by
``-theta`` so the local ``-x`` lip (the side facing the coffee cup) dips.

Verdict: milk transfer fraction >= 0.70 (milk particles inside the coffee cup), coffee retention
>= 0.90, milk spilled on the table <= 0.05.

Runs ONLY under the Newton venv (isaaclab develop runs headless unless a kit visualizer is
requested; an explicit --headless force-disables visualizers, so don't combine it with --viz):
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_pour_smoke                  # headless
  env_newton/bin/python -m robobench.suites.pouring.scripts.latte_pour_smoke --viz kit        # GUI (Kit window)
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="MPM grid is scene-wide; keep 1")
parser.add_argument("--time_scale", type=float, default=1.0, help="multiply every phase duration (slower pour)")
parser.add_argument("--tilt_deg", type=float, default=118.0, help="full pour tilt [deg]")
parser.add_argument("--lip_height", type=float, default=0.06, help="lip anchor height above the coffee-cup rim [m]")
parser.add_argument("--lip_x", type=float, default=0.02, help="lip anchor x over the coffee-cup mouth [m]")
parser.add_argument("--hold", type=float, default=2.5, help="extra settle time after the trajectory [s]")
parser.add_argument("--print_every", type=int, default=200, help="progress print period [steps]")
parser.add_argument("--max_steps", type=int, default=None, help="cap total steps (debugging)")
parser.add_argument("--scene", nargs="*", default=None, metavar="K=V", help="scene cfg overrides, e.g. voxel_size=0.004")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" in sys.argv:
    # Recording path (scripts/record_video.py): rendering on develop is pumped by visualizers, and
    # an explicit --headless force-disables them — so make headless+kit-visualizer the DEFAULTS.
    sys.argv = [a for a in sys.argv if a != "--headless"]
    parser.set_defaults(headless=True, visualizer=["kit"])
args = parser.parse_args()
livestream_on = args.livestream > 0
# Pump rendering only when someone is watching: a livestream client or a requested visualizer.
# (A bare run is headless on develop; stepping with render=True there is pure overhead.)
render_on = livestream_on or bool(args.visualizer and "none" not in args.visualizer)

app = AppLauncher(args).app

# Disable the cubric GPU transform hierarchy (isaacsim 6.0.0.1 IAdapter version drift renders
# moving prims frozen/detached); the CPU update_world_xforms() fallback renders correctly.
from isaaclab_newton.physics import newton_manager as _nm  # noqa: E402


def _no_cubric(cls) -> None:
    cls._cubric = None


_nm.NewtonManager._setup_cubric_bindings = classmethod(_no_cubric)

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

FPS = 200  # must match MpmSimCfg.dt


def _smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    cfg = ENVS.get("pouring.latte")()
    kw: dict = {}
    for kv in args.scene or []:
        k, v = kv.split("=", 1)
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    if kw:
        from robobench.suites.pouring.scenes import LatteSceneCfg

        cfg.scene_cfg = LatteSceneCfg(**kw)
        print(f"[latte] scene overrides: {kw}", flush=True)
    env = cfg.build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    render = render_on
    n = env.num_envs
    print(
        f"[latte] particles: coffee {scene.coffee.particles_per_object}, milk {scene.milk.particles_per_object}",
        flush=True,
    )

    from robobench.suites.pouring.scenes.latte import TABLE_TOP_Z

    # -- trajectory geometry (env-local; num_envs=1) --
    mx, my = c.milk_cup_pos
    r_outer = c.milk_cup_r + c.cup_wall
    lip_local = torch.tensor([-r_outer, 0.0, c.milk_cup_h], device=device)  # pouring edge, cup-local
    rim_z = TABLE_TOP_Z + c.coffee_cup_h
    lip_target = torch.tensor([args.lip_x, my, rim_z + args.lip_height], device=device)
    theta_max = math.radians(args.tilt_deg)

    p_start = torch.tensor([mx, my, TABLE_TOP_Z], device=device)
    p_travel = torch.tensor([mx, my, rim_z + args.lip_height + 0.02], device=device)

    def anchor_pos(theta: float) -> torch.Tensor:
        """Cup root position that keeps the lip at lip_target under tilt theta (rotation -theta about +y)."""
        st, ct = math.sin(-theta), math.cos(-theta)
        lx, lz = lip_local[0], lip_local[2]
        lip_rot = torch.tensor([ct * lx + st * lz, 0.0, -st * lx + ct * lz], device=device)
        return lip_target - lip_rot

    p_anchor0 = anchor_pos(0.0)

    # (duration [s], pose_fn(alpha) -> (pos, theta)) — alpha is smoothstepped phase progress.
    def lerp(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (b - a) * s

    phases: list[tuple[str, float, object]] = [
        ("settle", 1.0, lambda s: (p_start, 0.0)),
        ("lift", 1.2, lambda s: (lerp(p_start, p_travel, s), 0.0)),
        ("traverse", 1.8, lambda s: (lerp(p_travel, p_anchor0, s), 0.0)),
        ("pour", 3.5, lambda s: (anchor_pos(theta_max * s), theta_max * s)),
        ("drain", 1.8, lambda s: (anchor_pos(theta_max), theta_max)),
        ("recover", 1.2, lambda s: (anchor_pos(theta_max * (1.0 - s)), theta_max * (1.0 - s))),
        ("return", 1.8, lambda s: (lerp(p_anchor0, p_travel, s), 0.0)),
        ("set_down", 1.2, lambda s: (lerp(p_travel, p_start, s), 0.0)),
    ]
    durs = [d * args.time_scale for _, d, _ in phases]
    t_edges = [sum(durs[: i + 1]) for i in range(len(durs))]
    total_steps = int(round((t_edges[-1] + args.hold) * FPS))
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    def pose_at(t: float) -> tuple[torch.Tensor, float]:
        for (name, _, fn), edge, dur in zip(phases, t_edges, durs):
            if t < edge:
                s = _smoothstep(1.0 - (edge - t) / max(dur, 1e-9))
                return fn(s)
        return phases[-1][2](1.0)

    env.reset()
    env_origin = env.iscene.env_origins[0]
    action = torch.zeros((n, 0), device=device)
    prev_pos, prev_theta = pose_at(0.0)
    last_phase = ""

    def status() -> str:
        return (
            f"transfer {float(scene.transfer_fraction().mean()):.3f}"
            f" | retention {float(scene.retention_fraction().mean()):.3f}"
            f" | spilled {float(scene.spilled_fraction().mean()):.3f}"
        )

    for step in range(total_steps):
        t = step / FPS
        pos, theta = pose_at(t)
        # xyzw quat for rotation of -theta about +y (tips the local -x lip toward the coffee cup)
        half = -theta / 2.0
        quat = torch.tensor([0.0, math.sin(half), 0.0, math.cos(half)], device=device)
        pose = torch.cat([pos + env_origin, quat]).unsqueeze(0)
        lin_vel = (pos - prev_pos) * FPS
        ang_vel = torch.tensor([0.0, -(theta - prev_theta) * FPS, 0.0], device=device)
        twist = torch.cat([lin_vel, ang_vel]).unsqueeze(0)  # Newton spatial vectors: (linear, angular)
        scene.milk_cup.write_root_link_pose_to_sim_index(root_pose=pose)
        scene.milk_cup.write_root_link_velocity_to_sim_index(root_velocity=twist)
        prev_pos, prev_theta = pos, theta

        env.step(action, render=render)

        phase = next((nm for (nm, _, _), edge in zip(phases, t_edges) if t < edge), "hold")
        if phase != last_phase:
            print(f"  t={t:5.1f}s phase {phase:9s} | tilt {math.degrees(theta):5.1f} deg | {status()}", flush=True)
            last_phase = phase
        if step % args.print_every == 0:
            print(f"  step {step:5d} t={t:5.1f}s {phase:9s} | {status()}", flush=True)

    transfer = float(scene.transfer_fraction().mean())
    retention = float(scene.retention_fraction().mean())
    spilled = float(scene.spilled_fraction().mean())
    ok = transfer >= 0.70 and retention >= 0.90 and spilled <= 0.05
    print(
        f"LATTE-POUR {'PASS' if ok else 'FAIL'} | transfer {transfer:.3f} (gate >= 0.70)"
        f" | retention {retention:.3f} (>= 0.90) | spilled {spilled:.3f} (<= 0.05)",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
