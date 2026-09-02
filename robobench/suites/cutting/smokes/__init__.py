"""Runnable smoke tests for the cutting suite.

Each smoke boots `AppLauncher` at import, so nothing in the package imports these — run
standalone, e.g. `python -m robobench.suites.cutting.smokes.slice_smoke --headless`.
"""


def close_and_exit(env, app) -> None:
    """Tear down and hard-exit — every smoke's last call (Kit teardown regularly hangs)."""
    import os
    import threading

    watchdog = threading.Timer(10.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    env.close()
    app.close()
    os._exit(0)
