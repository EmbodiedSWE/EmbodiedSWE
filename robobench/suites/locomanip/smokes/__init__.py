"""Runnable smoke tests for the locomanip suite.

Each smoke boots `AppLauncher` at import, so nothing in the package imports these — run them
standalone, e.g. `python -m robobench.suites.locomanip.smokes.wheel_carry_smoke --headless`.

Naming: `<scene>_smoke.py` = a scene smoke test (scene + NullRobot, or scene + a robot where the
mechanic needs one). One per scene keeps them unambiguous.
"""


def close_and_exit(env, app) -> None:
    """Tear down and hard-exit — every smoke's last call.

    Kit teardown regularly hangs INSIDE env.close()/app.close(), so a plain return can wedge
    headless runs. Call this only once everything is printed (use flush=True): it arms a
    force-exit watchdog before closing, then hard-exits. `os._exit` is looked up at call time
    on purpose — scripts/record_video.py patches it to flush the mp4 first.
    """
    import os
    import threading

    watchdog = threading.Timer(10.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    env.close()
    app.close()
    os._exit(0)
