"""Generic config smoke test — build any registered env (or scene+robot combo) and step it with
random actions, to check it composes and runs. One script for every combination.

Two ways to say what to build, both resolved through `EnvCfg`:
  - by **named env config** (preferred): `--env assembly.ikea_table` — a suite-registered
    binding of scene + robot + control mode + sim (see `ENVS`; names are `suite.scene[.robot[.mode]]`);
  - by **ad-hoc combo**: `--scene NAME --robot NAME --mode MODE` — wired on the fly.

The random actions only exercise the action -> controller -> sim pipeline; they are NOT meaningful
behaviour (for an IK control mode the action is a wrist pose, so random values are nonsense but still
prove the path runs). Use a suite's own scripts for meaningful demos / regression tests.

List what's registered (no app launch):
    python -m robobench.scripts.smoke --list

Smoke a named env, or an ad-hoc combo:
    python -m robobench.scripts.smoke --env assembly.ikea_table --livestream 2
    python -m robobench.scripts.smoke --scene ikea_table --robot g1 --mode joint --headless
"""

from __future__ import annotations

import argparse
import sys
import time


def _print_registries() -> None:
    """Discover and print registries without importing Isaac Lab's native app stack."""
    import robobench

    robobench.discover()
    from robobench.core import CONTROLLERS, ENVS, ROBOTS, SCENES

    print("envs       :", ENVS.list())
    print("scenes     :", SCENES.list())
    print("robots     :", ROBOTS.list())
    print("controllers:", CONTROLLERS.list())


def main() -> None:
    # Keep the documented registry-only path genuinely app-free. Importing
    # isaaclab.app loads native Kit libraries even before AppLauncher() is called;
    # that made `--list` crash on a broken driver/runtime and hid useful registry
    # evidence behind a grep pipe.
    if sys.argv[1:] == ["--list"]:
        _print_registries()
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=str, default="", help="registered env-config name (see --list)")
    parser.add_argument("--scene", type=str, default="", help="ad-hoc: registered scene name")
    parser.add_argument("--robot", type=str, default="null", help="ad-hoc: registered robot name")
    parser.add_argument("--mode", type=str, default="", help="robot control_mode ('' = the robot's first)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action_scale", type=float, default=0.3, help="random action ~ U(-s, s) per dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room", choices=("default", "none"), default="default",
                        help="use the task configuration room (default), or disable scenery")
    parser.add_argument("--list", action="store_true", help="list everything registered, then exit")

    # pink_ik needs pinocchio imported BEFORE AppLauncher (its eigenpy STL converters must register in a
    # clean process; see controllers/pink_ik.py). Optional dependency, so tolerate its absence.
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pass

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # Capture render intent BEFORE AppLauncher(args) — the launcher consumes/strips these attrs.
    want_render = (not args.headless) or args.livestream > 0

    import robobench

    robobench.discover()
    from robobench.core import ENVS

    if args.list or not (args.env or args.scene):
        _print_registries()
        if not args.list:
            print("\nGive --env NAME (or --scene NAME [--robot/--mode]) to build + smoke it.")
        return

    # Resolve to one EnvCfg (the single build path), from a named config or an ad-hoc combo.
    from robobench.core import EnvCfg

    if args.env:
        cfg = ENVS.get(args.env)()
        if args.mode:
            cfg.control_mode = args.mode  # let --mode override the named config's mode
    else:
        cfg = EnvCfg(scene=args.scene, robot=args.robot, control_mode=args.mode)

    # --- build needs the app ---------------------------------------------------------------------
    import os
    import traceback

    app = AppLauncher(args).app
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        room_override = {"room": None} if args.room == "none" else {}
        env = cfg.build(num_envs=args.num_envs, device=device, seed=args.seed, **room_override)
        robot = env.robot
        n, dim = env.num_envs, robot.action_dim

        print(f"BUILT {args.env or '(ad-hoc)'} -> {cfg.describe()}  mode='{robot.control_mode}'")
        print(f"action_dim={dim}  (random U(-{args.action_scale}, {args.action_scale}))")
        print(f"describe(): {env.describe()}")

        env.reset()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            action = (torch.rand(n, dim, device=device) * 2.0 - 1.0) * args.action_scale
            env.step(action, render=want_render)
        elapsed = time.perf_counter() - t0
        fps = args.steps / elapsed  # sim steps/s for a single env (rendering, if on, slows this)
        print(
            f"RAN {args.steps} steps, no crash | {fps:.0f} fps/env "
            f"({n * fps:.0f} env-steps/s total over {n} env{'s' if n != 1 else ''}, {elapsed:.1f}s)"
        )
    except BaseException:  # noqa: BLE001 — any failure after the app is up must be a non-zero exit
        # Kit's own shutdown path otherwise turns a crashed smoke into exit code 0 (seen with the pink_ik
        # pinocchio registration error), which hides failures from automated checks. Report and hard-exit.
        traceback.print_exc()
        print("SMOKE FAILED (see traceback above)", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Kit teardown regularly hangs INSIDE env.close()/app.close() (a 100% CPU spin), wedging headless
    # runs — the same hard-exit as `suites/assembly/smokes.close_and_exit`. Everything is printed by
    # now; `os._exit` is looked up at call time so scripts/record_video.py can patch it to flush the
    # mp4 first.
    import threading

    sys.stdout.flush()
    watchdog = threading.Timer(10.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
