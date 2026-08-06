"""Named-check harness for task smokes (the robobench ``check()`` pattern).

Every assertion in a smoke is a named check; the verdict is the conjunction and is
printed as the machine-readable ``SIM_GEN_SMOKE: ALL PASS n/n`` / ``FAIL k/n`` marker
that pipeline/validate.py greps for.  No orphaned booleans: a check that is computed
is a check that is counted.
"""

from __future__ import annotations


class Checks:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        ok = bool(cond)
        self.results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[check] {mark:4s} {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    def finish(self) -> bool:
        n = len(self.results)
        good = sum(1 for _, ok, _ in self.results if ok)
        if good == n:
            print(f"SIM_GEN_SMOKE: ALL PASS {good}/{n}", flush=True)
            return True
        for name, ok, detail in self.results:
            if not ok:
                print(f"SIM_GEN_SMOKE_FAILED_CHECK: {name} {detail}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {good}/{n}", flush=True)
        return False
