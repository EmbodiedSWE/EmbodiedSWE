"""Runnable entrypoints for the assembly suite (smoke tests + demos).

Each script boots `AppLauncher` at import, so nothing in the package imports these — run them
standalone, e.g. `python -m robobench.suites.assembly.scripts.ikea_table_assembly_smoke --headless`.

Naming: `<scene>_smoke.py` = a scene smoke test (scene + NullRobot); `<config>_demo.py` = a full-env
demo (scene + a real robot from a config). One per scene/config keeps them unambiguous.
"""
