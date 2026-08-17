"""Smoke / rubric-REJECTION battery for ClutchValveStoveScene — NullRobot, wrench probes.

solve.py is the acceptance proof (the rubric accepts the correct outcome on
seeds 0 and 1 and the score is monotone along a real trajectory). This battery
proves the rubric REJECTS wrong outcomes, and that the claims the task rests
on — the free-spin child-safety decoy, the sustained lift-AND-turn clutch, the
one-way direction, and the release-to-finish clause — are load-bearing. Every
probe is CONSTRUCTED (wrench-driven physics, real steps, judge) —
instrumentation, never a solution: no probe here reaches success() (audited at
every step; the one probe that shuts the valve HOLDS the knob up throughout
and is torn down by an immediate reset, never a release).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; dial in the OPEN window,
                          knob parked at the bottom stop, pots on the apron row;
                          score 0, no success;
   2. randomization     — 3 seeded resets, max-PAIRWISE readback deltas: dial
                          start angle, knob yaw, pot0 xy all real;
   3. permutation       — over 8 resets pot0 occupies >= 2 distinct slots
                          (the 3-slot shuffle is real);
   4. null-policy       — 240 idle steps: score ~0, no success, dial does not
                          creep off its start angle;
   5. SEED-NAIVE spin   — the seed's move, "just twist the knob": the UN-LIFTED
                          knob is spun >= 180 deg (readback — non-vacuous); the
                          rest air gap transmits NOTHING — dial moves < 3 deg,
                          engage latch never fires, score ~0;
   6. engage-only       — lift+creep engages the clutch (latched 0.15) but the
                          dial stays far from OFF -> no success, score capped
                          low; released, the knob drops back, credit keeps;
   7. wrong direction   — engaged knob wound COUNTER-clockwise: the dial pins at
                          its 90-deg OPEN stop (readback), never approaches OFF
                          -> no success, no progress credit;
   8. near-miss         — engaged, wound clockwise to ~28 deg — SHORT of
                          theta_off — braked and released: the dial settles
                          outside tolerance -> partial credit only, no success;
   9. held-at-stop      — wound all the way to the shut stop but the knob KEPT
                          HELD UP: the release clause refuses success at every
                          held step (score caps at 0.70); torn down by an
                          immediate reset (zero wrench, NO step in between);
  10. wrong object      — a stock pot parked on the burner plate changes
                          nothing: dial unchanged, score ~0, no success;
  11. rejection audit   — success() was never True at ANY step of the battery;
  12. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene8_turn_off_the_stove_i363.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401 — registers the scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WDT = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                        os._exit(3)))
_WDT.daemon = True
_WDT.start()

_G = 9.81

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"hit": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["hit"] = _AUDIT["hit"] or bool(env.scene.success().any())
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_state(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | dial={math.degrees(float(scene.rotor_angle()[0])):+.1f}deg "
          f"lift={1000 * float(scene.knob_lift()[0]):.1f}mm "
          f"engaged={bool(scene._engaged[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.clutch_valve_stove")().build(num_envs=args.num_envs,
                                                        device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    mg = c.knob_mass * _G
    dt = 1.0 / 120.0

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.78, -0.85, 0.80)) + o),
                                tuple(np.array((0.02, 0.05, 0.25)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def deg() -> float:
        _refresh()
        return math.degrees(float(scene.rotor_angle()[0]))

    def lift() -> float:
        _refresh()
        return float(scene.knob_lift()[0])

    def dang(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    # --- the same grip emulation the solve uses (lift PD + P yaw servo) ---
    Z_TGT, KP, KD = 0.019, 250.0, 8.0
    # saturating P velocity servo, mirror of the solve (no feedforward: a ff
    # term overruns w_des and dynamic probes coast through their targets)
    KW, T_MAX = 0.08, 0.40
    swept = {"acc": 0.0}  # |knob rotation| integrated from VELOCITY readback

    def drive(w_des: float, *, hold: bool, f_cap: float, steps: int = 1) -> None:
        for _ in range(steps):
            f = 0.0
            if hold:
                z = float(scene.knob_lift()[0])
                vz = float(scene.knob.data.root_lin_vel_w[0, 2])
                f = max(0.0, min(f_cap, mg + KP * (Z_TGT - z) - KD * vz))
            w = float(scene.knob.data.root_ang_vel_w[0, 2])
            t = 0.0
            if w_des != 0.0 or abs(w) > 0.05:
                t = max(-T_MAX, min(T_MAX, KW * (w_des - w)))
            fb = torch.zeros(n, 1, 3, device=device)
            tb = torch.zeros(n, 1, 3, device=device)
            fb[:, 0, 2] = f
            tb[:, 0, 2] = t
            scene.knob.set_external_force_and_torque(fb, tb, env_ids=_all_ids())
            _step(1)
            swept["acc"] += abs(float(scene.knob.data.root_ang_vel_w[0, 2])) * dt

    def zero_knob() -> None:
        zero = torch.zeros(n, 1, 3, device=device)
        scene.knob.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    def engage() -> bool:
        """Solve-style engage: press straight up with NO twist (a pressed
        twist friction-drags the dial through the vane-bottom thrust
        contact); if a post parks under a vane, release fully (air gap ->
        zero coupling), re-phase the knob ~24 deg while DOWN, press again."""
        for _ in range(12):
            streak = 0
            for _ in range(60):
                drive(0.0, hold=True, f_cap=mg + 2.0)
                streak = streak + 1 if lift() >= c.engage_z + 0.0005 else 0
                if streak >= 20:
                    return True
            zero_knob()
            for _ in range(40):
                _step(1)
                if lift() <= 0.002:
                    break
            drive(-2.0, hold=False, f_cap=0.0, steps=25)
            drive(0.0, hold=False, f_cap=0.0, steps=15)
        return False

    def turn_to(target_deg: float, w_des: float, max_steps: int) -> None:
        t = 0
        while deg() > target_deg and t < max_steps:
            drive(w_des, hold=True, f_cap=10.0)
            t += 1

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    _refresh()
    fin = bool(torch.isfinite(scene.rotor.data.root_pos_w).all()) \
        and bool(torch.isfinite(scene.knob.data.root_pos_w).all())
    on_row = True
    for b in scene.pots:
        fin = fin and bool(torch.isfinite(b.data.root_pos_w).all())
        py = float((b.data.root_pos_w - scene.env_origins)[0, 1])
        on_row = on_row and abs(py - c.slot_y) < c.slot_jitter + 0.02
    th0_settle = deg()
    check("settle/no-NaN: dial in the OPEN window, knob parked at the bottom stop, "
          "pots on the apron row; score 0, no success",
          fin and on_row
          and c.theta0_lo_deg - 3.0 <= th0_settle <= c.theta0_hi_deg + 3.0
          and lift() <= 0.002 and not bool(scene.valve_off()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ====================
    def readback():
        _refresh()
        return (deg(),
                math.degrees(float(scene.knob_yaw()[0])),
                scene.pots[0].data.root_pos_w[0, :2].clone())

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        obs.append(readback())
    d_th = max(dang(a[0], b[0]) for a, b in ((obs[0], obs[1]), (obs[0], obs[2]),
                                             (obs[1], obs[2])))
    d_yaw = max(dang(a[1], b[1]) for a, b in ((obs[0], obs[1]), (obs[0], obs[2]),
                                              (obs[1], obs[2])))
    d_pot = max(float((a[2] - b[2]).norm()) for a, b in ((obs[0], obs[1]),
                                                         (obs[0], obs[2]),
                                                         (obs[1], obs[2])))
    print(f"[smoke] randomization max-pairwise deltas: dial={d_th:.1f}deg "
          f"knob_yaw={d_yaw:.1f}deg pot0_xy={d_pot * 1000:.1f}mm", flush=True)
    check("randomization-is-real: dial start angle, knob yaw, pot0 xy readback "
          "all differ across 3 seeds", d_th > 2.0 and d_yaw > 5.0 and d_pot > 0.005)

    # ================= 3. permutation: pot0 visits >= 2 distinct slots ============================
    slots_seen = set()
    xs = torch.tensor(c.slot_xs)
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sx = float((scene.pots[0].data.root_pos_w - scene.env_origins)[0, 0])
        slot = int(torch.argmin((xs - sx).abs()))
        assert abs(float(xs[slot]) - sx) < c.slot_jitter + 0.02, \
            "pot0 must land on one of the 3 slots"
        slots_seen.add(slot)
    print(f"[smoke] pot0 slots over 8 resets: {sorted(slots_seen)}", flush=True)
    check("permutation: pot0 occupies >= 2 distinct slots over 8 resets "
          "(the 3-slot shuffle is real)", len(slots_seen) >= 2)

    # ================= 4. null policy fails (and the dial does not creep) =========================
    torch.manual_seed(100)
    env.reset()
    _step(5)
    th0 = deg()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, dial stays "
          "on its start angle",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and dang(deg(), th0) < 3.0)

    # ================= 5. SEED-NAIVE: just twist the knob (no lift) ===============================
    # The seed's whole plan is "grasp the stove knob and rotate it". Here that
    # move is a DECOY: the un-lifted knob spins freely below the vanes and
    # transmits nothing through the rest air gap.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    th0 = deg()
    _REC["on"] = True
    swept["acc"] = 0.0
    for _ in range(420):
        drive(-3.0, hold=False, f_cap=0.0)
    spin_deg = math.degrees(swept["acc"])
    zero_knob()
    _step(120)
    _report("seed-naive-spin")
    _REC["on"] = False
    print(f"[smoke] free-spin: knob swept {spin_deg:.0f} deg, dial moved "
          f"{dang(deg(), th0):.2f} deg", flush=True)
    check("SEED-NAIVE spin: the un-lifted knob provably spun >= 180 deg yet the "
          "dial moved < 3 deg — the clutch air gap is real; engage latch never "
          "fired, score ~0, no success",
          spin_deg >= 180.0 and dang(deg(), th0) < 3.0
          and not bool(scene._engaged[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. engage-only: lifting is not turning off =================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    th0 = deg()
    _REC["on"] = True
    ok_eng = engage()
    # hold up WITHOUT winding for a while — nothing should progress
    drive(0.0, hold=True, f_cap=10.0, steps=180)
    held_lift = lift()
    _report("engage-only")
    zero_knob()
    _step(150)  # released: knob drops, engage credit latches
    _report("engage-drop")
    _REC["on"] = False
    s6 = float(scene.score()[0])
    check("engage-only: clutch engaged (lift readback held above engage_z) but the "
          "dial stays far from OFF — score is the 0.15 engage credit (+ at most "
          "incidental creep), no success; the dropped knob keeps the latch",
          ok_eng and held_lift >= c.engage_z - 0.001
          and bool(scene._engaged[0]) and dang(deg(), th0) < 12.0
          and not bool(scene.valve_off()[0])
          and 0.14 <= s6 <= 0.30 and not bool(scene.success()[0]))

    # ================= 7. wrong direction: counter-clockwise pins at the OPEN stop ================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    th0 = deg()
    _REC["on"] = True
    assert engage(), "probe rig: engage failed"
    swept["acc"] = 0.0
    drive(1.2, hold=True, f_cap=10.0, steps=600)  # wind the WRONG way
    ccw_deg = deg()
    ccw_spin = math.degrees(swept["acc"])
    _report("wrong-direction")
    zero_knob()
    _step(150)
    _REC["on"] = False
    print(f"[smoke] ccw: knob swept {ccw_spin:.0f} deg, dial {th0:+.1f} -> "
          f"{ccw_deg:+.1f} deg", flush=True)
    check("wrong direction: the engaged knob wound counter-clockwise (swept "
          ">= 20 deg readback) drives the dial UP against its 90-deg OPEN stop, "
          "never toward OFF — no progress credit beyond engage, no success",
          ccw_spin >= 20.0 and ccw_deg >= th0 - 1.0 and ccw_deg <= 91.0
          and not bool(scene.valve_off()[0])
          and float(scene.score()[0]) <= 0.30 and not bool(scene.success()[0]))

    # ================= 8. near-miss: stopped short of theta_off ===================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    th0 = deg()
    _REC["on"] = True
    assert engage(), "probe rig: engage failed"
    turn_to(28.0, -0.5, 3000)          # slow wind to ~28 deg — SHORT of 8 deg
    drive(0.0, hold=True, f_cap=10.0, steps=100)  # brake the wrist (kill coast)
    zero_knob()
    _step(240)                          # release: knob drops, dial rests
    _report("near-miss")
    _REC["on"] = False
    near_deg = deg()
    s8 = float(scene.score()[0])
    check("near-miss: wound to ~28 deg (short of theta_off) and released — the "
          "dial settles outside tolerance: partial credit only, no success",
          c.theta_off_deg + 3.0 <= near_deg <= 40.0
          and bool(scene.released()[0]) and not bool(scene.valve_off()[0])
          and 0.30 <= s8 <= 0.75 and not bool(scene.success()[0]))

    # ================= 9. held-at-stop: the release clause is load-bearing ========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    assert engage(), "probe rig: engage failed"
    _REC["on"] = False  # the long wind duplicates check 8 visually
    turn_to(1.5, -1.2, 3000)
    # valve is OFF but the knob is STILL HELD UP: success must refuse throughout
    held_hit = False
    for _ in range(120):
        drive(0.0, hold=True, f_cap=10.0)
        held_hit = held_hit or bool(scene.success()[0])
    _report("held-at-stop")
    held_off = bool(scene.valve_off()[0])
    held_lift = lift()
    s9 = float(scene.score()[0])
    held_ok = held_off and held_lift >= c.engage_z - 0.001 and not held_hit \
        and not bool(scene.success()[0]) and 0.69 <= s9 <= 0.705
    # TEAR-DOWN: zero the wrench and reset IMMEDIATELY — a single free step here
    # would release the knob into genuine success and poison the audit.
    zero_knob()
    torch.manual_seed(100)
    env.reset()
    check("held-at-stop: dial wound to the shut stop with the knob still held up "
          "— the release clause refuses success at EVERY held step, score caps "
          "at 0.70; torn down by immediate reset (no free step)", held_ok)

    # ================= 10. wrong object: a pot on the burner does nothing =========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    th0 = deg()
    _REC["on"] = True
    t = scene.stove.data.root_pos_w[0]
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = float(t[0]) + c.burner_x
    pos[:, 1] = float(t[1])
    pos[:, 2] = float(t[2]) + c.burner_h + 0.006
    _write_state(scene.pots[0], pos)
    _step(240)
    _report("pot-on-burner")
    _REC["on"] = False
    check("wrong object: a stock pot parked on the burner plate changes nothing — "
          "dial unchanged, score ~0, no success",
          dang(deg(), th0) < 3.0 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 11. rejection audit ========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.clutch_valve_stove")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
