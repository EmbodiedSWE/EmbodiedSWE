#!/usr/bin/env python3
"""Ground-truth check of the supervisor's process detection.

The supervisor decides whether to relaunch a driver, relay or side-car purely from
this predicate, so a false "already running" silently disables the whole supervisor
(which is exactly what the first pgrep-based version did). Run it while a v21 run is
up: it must see what is running and not see what is not.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "sup", "/home/tiger/cap-x/scripts/cosigen_arm_supervisor.py")
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

TAG = sys.argv[1] if len(sys.argv) > 1 else "v21"
cases = [
    (("cosigen_harness.py", f"--session-id ikea-{TAG}-ckpt"), True, "ckpt driver"),
    (("cosigen_harness.py", f"--session-id ikea-{TAG}-full"), True, "full driver"),
    (("server.py", "--port 8118"), True, "ckpt relay"),
    (("server.py", "--port 8119"), True, "full relay"),
    (("cosigen_harness.py", "--session-id ikea-v00-nonexistent"), False,
     "driver that is not running"),
    (("definitely_not_a_process_xyz.py",), False, "process that cannot exist"),
]
ok = True
for needles, want, label in cases:
    got = sup.running(*needles)
    ok &= got == want
    print(f"   {'PASS' if got == want else 'FAIL'}  {label}: detected={got} expected={want}")
print("SUPERVISOR DETECTION " + ("PASSED" if ok else "FAILED"))
