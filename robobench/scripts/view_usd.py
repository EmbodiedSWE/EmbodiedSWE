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
import signal
import sys

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

    # `visualizationDisplayColliders` is a `/persistent/` setting (saved to user.config.json), so left
    # on it leaks the overlay into every future Isaac Sim session. We turn it off again on exit, always
    # settling on 0 (the shipped default) so even an earlier hard-killed run self-heals (2 = all, 0 = off).
    settings = carb.settings.get_settings()
    COLLIDERS_KEY = "/persistent/physics/visualizationDisplayColliders"

    def _disable_colliders() -> None:
        settings.set_int(COLLIDERS_KEY, 0)

    if args.colliders:
        settings.set_int(COLLIDERS_KEY, 2)
        # Isaac's own SIGINT handler shuts the app down — which FLUSHES /persistent settings to disk —
        # before any `finally` here would run, so a plain try/finally can't stop the value leaking.
        # Restore ahead of that flush in our own handler, then hand off to a normal shutdown.
        def _on_sigint(signum, frame):  # noqa: ANN001
            _disable_colliders()
            app.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, _on_sigint)

    try:
        while app.is_running():  # normal exit (e.g. window close): finally restores while app is alive
            app.update()
    finally:
        if args.colliders:
            _disable_colliders()


if __name__ == "__main__":
    main()
