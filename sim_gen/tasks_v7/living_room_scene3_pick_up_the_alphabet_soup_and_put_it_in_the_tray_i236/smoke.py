"""Smoke / rubric-REJECTION battery for AirlockTransferScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct open-hatch / drop /
cycle-airlock episode and the latched credit is monotone along it, two seeds). This
battery proves the rubric REJECTS wrong outcomes, and that the geometry claims the
task rests on — the vault is sealed from above, the hatch and the window are never
simultaneously passable, the sealed window holds against a real push, the gate is
captive on its stroke — are physics, not fiat. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a
solution. Smoke constructs may write bodies inside rubric volumes (that is the point
of a rubric probe); only solve.py is barred from doing so.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; gate on its stroke, cans on
                           the ground, tray on its seat; score ~0, no success;
   2. randomization      — two seeded resets: READBACK station yaw, station xy, gate
                           start c and the alphabet can's xy all differ;
   3. slot shuffle       — over 10 resets the alphabet can occupies >= 2 different
                           ground slots;
   4. null-policy        — 240 idle steps: gate and can stay put, score ~0;
   5. SEED STRATEGY      — the seed's plan ("carry the can over the tray and lower it
                           in from above"): the can ends ON THE SEALED VAULT ROOF —
                           not in the tray, no success, score ~0;
   6. anti-phase A       — gate slid LEFT (window open): a can dropped over the hatch
                           lands ON the closed roof blade — never enters the chamber;
   7. anti-phase B       — gate slid RIGHT (hatch open): a can placed mid-ramp slides
                           >= 50 mm down and rests against the CLOSED window blade
                           (staged partial credit 0.45); a 5 N push toward the vault
                           cannot put it through — never in the vault, no success;
   8. latch persistence  — the staged can teleported back to the ground: the latched
                           0.45 survives (score never decreases) but success stays
                           False — success is judged live;
   9. DEAD BAND          — gate at c=+0.012 (neither opening passes): the staged push
                           still cannot reach the vault, and with the gate HELD there
                           (an agent gripping the handle) a can dropped over the 57 mm
                           uncovered hatch strip cannot END inside the chamber (no
                           credit in a fresh episode; unheld, the impact just shoves
                           the captive slider to its stop = opening the hatch);
  10. wrong object       — CORN delivered into the tray, alphabet on the ground -> no
                           success, no credit;
  11. captive gate       — 8 N yanks both ways: the gate traverses its full stroke
                           (the drive the solve and a Franka perform is real) but
                           never leaves the stops, the rails or the roof plane;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.<task>.smoke --headless
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel_w is not None:
        st[:, 7:10] = lin_vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    kl = scene._station_local(scene.alphabet.data.root_pos_w)[0]
    tl = scene._tray_local(scene.alphabet.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | c={float(scene.gate_c()[0]):+.3f} "
          f"can_st=({float(kl[0]):+.3f},{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
          f"can_tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},{float(tl[2]):+.3f}) "
          f"cham={bool(scene._in_chamber[0])} staged={bool(scene._staged[0])} "
          f"vault={bool(scene._in_vault[0])} in={bool(scene.alphabet_in_tray()[0])} "
          f"A={bool(scene.hatch_passes()[0])} B={bool(scene.window_passes()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# force-frame convention (some pods rotate a "global" wrench by the body's rotation
# since reset; probed ONCE on the gate, then reused for every push)
_FR = {"mode": 0, "q_ref": None}


def main() -> None:  # noqa: PLR0915 - a linear battery reads best linear
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.airlock_transfer")().build(num_envs=args.num_envs,
                                                     device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    dev = env.device
    all_ids = _all_ids()
    zero_w = torch.zeros(n, 1, 3, device=dev)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.60, 0.65)) + o),
                                tuple(np.array((0.40, 0.00, 0.15)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)
    _REC["on"] = True

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok)))
        print(f"[smoke] CHECK {'PASS' if ok else 'FAIL'} - {name}"
              + (f" ({detail})" if detail else ""), flush=True)

    def st_world(x: float, y: float, z: float) -> torch.Tensor:
        loc = torch.tensor([x, y, z], device=dev).expand(n, 3)
        return scene.station.data.root_pos_w \
            + quat_apply(scene.station.data.root_quat_w, loc)

    def gate_to(cv: float, settle: int = 30) -> None:
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = st_world(cv, 0.0, 0.0)
        st[:, 3:7] = scene.station.data.root_quat_w
        scene.gate.write_root_state_to_sim(st, all_ids)
        _refresh()
        _step(settle)

    def can_to(body, x: float, y: float, z: float, settle: int = 0) -> None:
        _write_body(body, st_world(x, y, z), quat=scene.station.data.root_quat_w)
        if settle:
            _step(settle)

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def push_body(body, dir_local, mag: float, steps: int, watch=None) -> bool:
        """Constant CoM force along a station-local direction; returns True if
        `watch()` ever fired during the push. Clears the force afterwards."""
        fdir = quat_apply(scene.station.data.root_quat_w,
                          torch.tensor(dir_local, device=dev).expand(n, 3))
        _FR["q_ref"] = body.data.root_quat_w.clone()
        fired = False
        for _ in range(steps):
            f = task_scene.encode_force(_FR["mode"], _FR["q_ref"],
                                        body.data.root_quat_w,
                                        mag * fdir).view(n, 1, 3)
            body.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                               is_global=True)
            _step(1)
            if watch is not None and bool(watch()):
                fired = True
        clear_force(body)
        return fired

    def yank_gate(sign: float, mag: float, steps: int,
                  vcap: float = 0.5) -> tuple[bool, float]:
        """8 N-class yank along station-local sign*x (velocity-capped so speculative
        contacts cannot tunnel). Samples EVERY step that the gate stays captive:
        |c| on stroke, |y| < 5 cm, z on the ground plane band. Returns
        (captive_all_steps, extreme c reached in the yank direction)."""
        xdir = quat_apply(scene.station.data.root_quat_w,
                          torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        _FR["q_ref"] = scene.gate.data.root_quat_w.clone()
        captive = True
        extreme = float(scene.gate_c()[0]) * sign
        for _ in range(steps):
            v = float((scene.gate.data.root_lin_vel_w[0] * xdir[0]).sum())
            m = mag if sign * v < vcap else 0.0
            f = task_scene.encode_force(_FR["mode"], _FR["q_ref"],
                                        scene.gate.data.root_quat_w,
                                        (m * sign) * xdir).view(n, 1, 3)
            scene.gate.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                                     is_global=True)
            _step(1)
            gl = scene._station_local(scene.gate.data.root_pos_w)[0]
            captive = captive and bool(scene._finite()[0]) \
                and abs(float(gl[0])) < c.stroke + 0.012 \
                and abs(float(gl[1])) < 0.05 and -0.012 < float(gl[2]) < 0.05
            extreme = max(extreme, float(gl[0]) * sign)
        clear_force(scene.gate)
        _step(30)
        return captive, extreme * sign

    def layout() -> tuple:
        sp = (scene.station.data.root_pos_w - scene.env_origins)[0]
        sq = scene.station.data.root_quat_w[0]
        yaw = math.atan2(float(sq[3]), float(sq[0])) * 2.0
        kp = (scene.alphabet.data.root_pos_w - scene.env_origins)[0]
        return (yaw, float(sp[0]), float(sp[1]), float(scene.gate_c()[0]),
                float(kp[0]), float(kp[1]))

    def score0() -> float:
        return float(scene.score()[0])

    # --- probe the force-frame convention once (scratch episode) ---
    env.reset(seed=99)
    _step(60)
    gate_to(0.0)
    xdir = quat_apply(scene.station.data.root_quat_w,
                      torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    for mag in (1.5, 3.0):
        c0 = float(scene.gate_c()[0])
        _FR["q_ref"] = scene.gate.data.root_quat_w.clone()
        for _ in range(60):
            f = task_scene.encode_force(_FR["mode"], _FR["q_ref"],
                                        scene.gate.data.root_quat_w,
                                        mag * xdir).view(n, 1, 3)
            scene.gate.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                                     is_global=True)
            _step(1)
        clear_force(scene.gate)
        dc = float(scene.gate_c()[0]) - c0
        if dc < -0.004:
            _FR["mode"] ^= 1
            print(f"[smoke] force-frame mode -> {_FR['mode']} (dc={dc:+.4f})",
                  flush=True)
        elif dc > 0.004:
            break
    print(f"[smoke] force-frame mode = {_FR['mode']}", flush=True)

    # ---------------- 1. settle / no-NaN ---------------------------------------------------------
    env.reset(seed=0)
    _step(120)
    _report("settle")
    cans_ok = True
    for b in scene.items:
        bl = scene._station_local(b.data.root_pos_w)[0]
        cans_ok = cans_ok and 0.03 < float(bl[2]) < 0.07 \
            and not bool(scene.in_chamber(b.data.root_pos_w)[0]) \
            and not bool(scene.in_vault(b.data.root_pos_w)[0])
    tl = scene._station_local(scene.tray.data.root_pos_w)[0]
    tray_ok = abs(float(tl[1]) - c.tray_seat[1]) < 0.02 \
        and 0.01 < float(tl[2]) < 0.05
    check("settle/no-NaN", bool(scene._finite()[0])
          and abs(float(scene.gate_c()[0])) < c.stroke + 0.006
          and cans_ok and tray_ok and score0() <= 0.03
          and not bool(scene.success()[0]),
          f"score={score0():.3f}")

    # ---------------- 2. randomization readback --------------------------------------------------
    env.reset(seed=0)
    _step(15)
    la = layout()
    env.reset(seed=1)
    _step(15)
    lb = layout()
    dyaw = abs(la[0] - lb[0])
    dxy = max(abs(la[1] - lb[1]), abs(la[2] - lb[2]))
    dc_ = abs(la[3] - lb[3])
    dkan = max(abs(la[4] - lb[4]), abs(la[5] - lb[5]))
    check("randomization readback",
          dyaw > 0.01 and dxy > 0.003 and dc_ > 0.003 and dkan > 0.005,
          f"dyaw={dyaw:.3f} dxy={dxy:.3f} dc={dc_:.3f} dcan={dkan:.3f}")

    # ---------------- 3. slot shuffle ------------------------------------------------------------
    slots_seen = set()
    for s in range(100, 110):
        env.reset(seed=s)
        _refresh()
        slots_seen.add(int(scene.alphabet_slot[0]))
    check("slot shuffle", len(slots_seen) >= 2, f"slots={sorted(slots_seen)}")

    # ---------------- 4. null policy -------------------------------------------------------------
    env.reset(seed=0)
    _step(30)
    cg0 = float(scene.gate_c()[0])
    kp0 = scene.alphabet.data.root_pos_w[0].clone()
    _step(240)
    kdrift = float((scene.alphabet.data.root_pos_w[0] - kp0).norm())
    check("null policy",
          abs(float(scene.gate_c()[0]) - cg0) < 0.005 and kdrift < 0.010
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"gate_dc={float(scene.gate_c()[0]) - cg0:+.4f} can_drift={kdrift:.4f} "
          f"score={score0():.3f}")

    # ---------------- 5. SEED STRATEGY: carry over the top -> sealed vault roof ------------------
    env.reset(seed=1)
    _step(60)
    can_to(scene.alphabet, 0.0, -0.098, c.roof_top + c.can_height / 2 + 0.004,
           settle=180)
    _report("seed-strategy")
    kl = scene._station_local(scene.alphabet.data.root_pos_w)[0]
    check("seed strategy rejected (can rests ON the sealed vault roof)",
          float(kl[2]) > 0.28
          and not bool(scene.in_vault(scene.alphabet.data.root_pos_w)[0])
          and not bool(scene.alphabet_in_tray()[0])
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"can_z={float(kl[2]):.3f} score={score0():.3f}")

    # ---------------- 6. anti-phase A: window open -> hatch drop lands ON blade A ----------------
    env.reset(seed=2)
    _step(30)
    gate_to(-0.055)
    can_to(scene.alphabet, 0.0, 0.147, 0.315, settle=240)
    _report("anti-phase-A")
    kl = scene._station_local(scene.alphabet.data.root_pos_w)[0]
    check("anti-phase A (hatch drop lands on the closed roof blade)",
          float(kl[2]) > 0.29 and not bool(scene._in_chamber[0])
          and not bool(scene.in_chamber(scene.alphabet.data.root_pos_w)[0])
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"can_z={float(kl[2]):.3f} score={score0():.3f}")

    # ---------------- 7. anti-phase B: ramp slide stages; 5 N push cannot enter the vault --------
    env.reset(seed=3)
    _step(30)
    gate_to(+0.060)
    can_to(scene.alphabet, 0.0, 0.130, 0.176, settle=300)
    _report("staged")
    kl = scene._station_local(scene.alphabet.data.root_pos_w)[0]
    slid = 0.130 - float(kl[1])
    s_stage = score0()
    stage_ok = slid >= 0.050 and bool(scene._staged[0]) \
        and 0.449 <= s_stage <= 0.46 \
        and not bool(scene.in_vault(scene.alphabet.data.root_pos_w)[0])
    fired = push_body(scene.alphabet, (0.0, -1.0, 0.0), 5.0, 240,
                      watch=lambda: scene.in_vault(scene.alphabet.data.root_pos_w)[0])
    _step(60)
    _report("pushed")
    kl = scene._station_local(scene.alphabet.data.root_pos_w)[0]
    push_ok = (not fired) \
        and not bool(scene.in_vault(scene.alphabet.data.root_pos_w)[0]) \
        and float(kl[1]) < 0.075 and abs(float(kl[0])) < 0.055 \
        and float(scene.gate_c()[0]) > 0.02 \
        and not bool(scene.success()[0])
    check("anti-phase B (slide stages 0.45; 5 N push never enters the vault)",
          stage_ok and push_ok,
          f"slid={slid:.3f} score={s_stage:.3f} fired={fired}")

    # ---------------- 8. latch persistence (score keeps, success stays live) ---------------------
    can_to(scene.alphabet, 0.0, 0.340, 0.047, settle=90)
    _report("latch-persist")
    check("latch persistence (staged credit survives, success stays live-judged)",
          score0() >= 0.449 and bool(scene._staged[0])
          and not bool(scene.alphabet_in_tray()[0])
          and not bool(scene.success()[0]),
          f"score={score0():.3f}")

    # ---------------- 9. DEAD BAND: neither opening passes ---------------------------------------
    env.reset(seed=4)
    _step(30)
    gate_to(+0.012)
    can_to(scene.alphabet, 0.0, 0.130, 0.176, settle=300)
    fired = push_body(scene.alphabet, (0.0, -1.0, 0.0), 5.0, 240,
                      watch=lambda: scene.in_vault(scene.alphabet.data.root_pos_w)[0])
    _step(60)
    _report("deadband-push")
    a_ok = (not fired) \
        and not bool(scene.in_vault(scene.alphabet.data.root_pos_w)[0])
    env.reset(seed=5)
    _step(30)
    # gate HELD at the dead band (pose re-written every step — the probe analogue
    # of an agent gripping the handle): a drop onto the blade edge otherwise
    # SHOVES the free captive slider to its stop, which is just "opening the
    # hatch" by impact, not a dead-band leak (observed on the forge)
    gst = torch.zeros(n, 13, device=dev)
    gst[:, 0:3] = st_world(+0.012, 0.0, 0.0)
    gst[:, 3:7] = scene.station.data.root_quat_w
    scene.gate.write_root_state_to_sim(gst, all_ids)
    _refresh()
    can_to(scene.alphabet, -0.0265, 0.140, 0.320)
    for _ in range(300):
        scene.gate.write_root_state_to_sim(gst, all_ids)
        _step(1)
    _report("deadband-drop")
    b_ok = not bool(scene._in_chamber[0]) \
        and not bool(scene.in_chamber(scene.alphabet.data.root_pos_w)[0]) \
        and score0() <= 0.03 and not bool(scene.success()[0])
    check("dead band (push cannot reach the vault; 57 mm strip drop never enters)",
          a_ok and b_ok, f"a={a_ok} b={b_ok} score={score0():.3f}")

    # ---------------- 10. wrong object -----------------------------------------------------------
    env.reset(seed=6)
    _step(60)
    tq = scene.tray.data.root_quat_w
    tpos = scene.tray.data.root_pos_w \
        + quat_apply(tq, torch.tensor([0.0, 0.0, 0.047], device=dev).expand(n, 3))
    _write_body(scene.corn, tpos, quat=tq)
    _step(120)
    _report("wrong-object")
    check("wrong object (corn in the tray earns nothing)",
          bool(scene.in_tray(scene.corn.data.root_pos_w)[0])
          and not bool(scene.alphabet_in_tray()[0])
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"corn_in_tray={bool(scene.in_tray(scene.corn.data.root_pos_w)[0])} "
          f"score={score0():.3f}")

    # ---------------- 11. captive gate under 8 N yanks -------------------------------------------
    env.reset(seed=7)
    _step(60)
    ok_l, c_min = yank_gate(-1.0, 8.0, 300)
    ok_r, c_max = yank_gate(+1.0, 8.0, 300)
    _report("yanked")
    check("captive gate (8 N yanks traverse the stroke, never derail)",
          ok_l and ok_r and c_min <= -0.060 and c_max >= +0.060
          and bool(scene._finite()[0]),
          f"c_min={c_min:+.3f} c_max={c_max:+.3f} captive={ok_l and ok_r}")

    # ---------------- 12. frames.npz -------------------------------------------------------------
    frames_ok = False
    try:
        if len(_REC["frames"]) >= 8:
            arr = np.stack(_REC["frames"], axis=0)
            np.savez_compressed(args.out, frames=arr)
            frames_ok = os.path.isfile(args.out) and os.path.getsize(args.out) > 0 \
                and int(arr.max()) > 10
            print(f"[smoke] saved {arr.shape} -> {args.out} "
                  f"({os.path.getsize(args.out)} bytes)", flush=True)
        else:
            print(f"[smoke] only {len(_REC['frames'])} frames captured", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] frame save FAILED ({exc!r})", flush=True)
    check("frames.npz video", frames_ok, f"n={len(_REC['frames'])}")

    # ---------------- verdict --------------------------------------------------------------------
    npass = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for name, ok in checks:
        if not ok:
            print(f"[smoke] FAILED: {name}", flush=True)
    verdict_ok = npass == total
    if verdict_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {npass}/{total}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {npass}/{total}", flush=True)

    code = 0 if verdict_ok else 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
