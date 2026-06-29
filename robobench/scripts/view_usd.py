"""Open an arbitrary USD stage and keep the app alive so you can inspect it — meant for livestream.

Not part of the suite/scene machinery: it just boots `AppLauncher`, opens the USD, and idles the
render loop until you Ctrl-C. Use it to eyeball any `.usd`/`.usda`/`.usdc` (an asset you built, a
robot USD, a scene) without wiring it into a scene config.

    python -m robobench.scripts.view_usd path/to/asset.usd --livestream 2

Then connect the Isaac Sim WebRTC Streaming Client (set OMNI_KIT_ACCEPT_EULA=YES first).
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("usd", type=str, help="path (or URI) to the USD stage to open")
parser.add_argument("--colliders", action="store_true",
                    help="overlay PhysX collision geometry (wireframe) on top of the visual mesh")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import carb  # noqa: E402
import omni.usd  # noqa: E402


def main() -> None:
    # Anchor as an ABSOLUTE path: a relative root-layer identifier makes USD mis-resolve the stage's
    # own `../`-relative references (it double-anchors them against the layer dir).
    usd = args.usd if "://" in args.usd else os.path.abspath(args.usd)
    ctx = omni.usd.get_context()
    if not ctx.open_stage(usd):  # returns a plain bool in Isaac Sim 5.1
        raise RuntimeError(f"failed to open {usd}")
    print(f"opened {usd} — Ctrl-C to quit")

    if args.colliders:
        # PhysX draws the collision approximation as a wireframe over the visual mesh: 2 = all, 0 = off.
        carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)

    while app.is_running():
        app.update()


if __name__ == "__main__":
    main()
