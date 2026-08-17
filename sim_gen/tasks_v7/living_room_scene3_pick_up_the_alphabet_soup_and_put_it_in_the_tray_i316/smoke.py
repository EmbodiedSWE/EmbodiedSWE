"""Smoke / rubric-REJECTION battery for CarafePourScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct carry / pour / set-down
episode and the latched credit is monotone along it, two seeds). This battery proves
the rubric REJECTS wrong outcomes, and that the geometry claims the task rests on —
the can is captive in the bore against real pushes, a partial tilt does NOT discharge
it, dumping the whole carafe into the tray earns nothing — are physics, not fiat.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution. Smoke constructs may write bodies inside rubric
volumes (that is the point of a rubric probe); only solve.py is barred from doing so.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; both cans captive in their
                           carafes, tray on the ground; score ~0, no success;
   2. randomization      — three seeded resets: READBACK tray xy + yaw, carafe xy and
                           the red can's xy — max-pairwise deltas all nonzero;
   3. red-jar shuffle    — over 10 resets the red can starts in BOTH carafes;
   4. null-policy        — 240 idle steps: can and carafes stay put, score ~0;
   5. JAR DUMP rejected  — the transport-only cheat (lay the whole carafe, can still
                           inside, INTO the tray): the can's position IS inside the
                           tray volume, yet no credit and no success — the rubric
                           demands the can be free of every carafe;
   6. captive bore       — a real 5 N lateral push on the captive can: the carafe
                           scoots along the ground but the can NEVER leaves the bore;
   7. partial tilt       — the carafe HELD at 60 deg (pose re-written every step, the
                           probe analogue of a wrist grip): the can stays in the bore
                           — only a past-horizontal pour discharges it;
   8. decant near-miss   — the red can settled on open ground (out of every carafe,
                           not in the tray): decant credit 0.30 only, no success;
   9. goal + live judge  — the exact goal state constructed -> success TRUE (positive
                           control); the can then removed to the ground -> success
                           back to False while the latched 0.60 survives;
  10. wrong object       — CORN settled in the tray: zero credit; and with the red
                           can ALSO in the tray, success is STILL False (the corn
                           must not be in the tray);
  11. carafe in tray     — red can delivered but the red carafe parked standing
                           INSIDE the tray: no success;
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
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
    rp = (scene.alphabet.data.root_pos_w - scene.env_origins)[0]
    tl = scene._tray_local(scene.alphabet.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | can_w=({float(rp[0]):+.3f},{float(rp[1]):+.3f},"
          f"{float(rp[2]):+.3f}) can_tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
          f"{float(tl[2]):+.3f}) in_jar={bool(scene.in_any_jar(scene.alphabet.data.root_pos_w)[0])} "
          f"in_tray={bool(scene.in_tray(scene.alphabet.data.root_pos_w)[0])} "
          f"dec={bool(scene._decanted[0])} del={bool(scene._delivered[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# force-frame convention (some pods rotate a "global" wrench by the body's rotation
# since reset; probed ONCE on a free can, then reused for every push)
_FR = {"mode": 0, "q_ref": None}


def main() -> None:  # noqa: PLR0915 - a linear battery reads best linear
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carafe_pour")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    n = env.num_envs
    dev = env.device
    all_ids = _all_ids()
    zero_w = torch.zeros(n, 1, 3, device=dev)

    from robobench.core import BaseScene  # noqa: F401  (import parity with siblings)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -0.85, 0.75)) + o),
                                tuple(np.array((0.33, 0.00, 0.10)) + o),
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

    def tray_pt(x: float, y: float, z: float) -> torch.Tensor:
        loc = torch.tensor([x, y, z], device=dev).expand(n, 3)
        return scene.tray.data.root_pos_w \
            + task_scene._qapply(scene.tray.data.root_quat_w, loc)

    def ground_pt(x: float, y: float, z: float) -> torch.Tensor:
        p = torch.tensor([x, y, z], device=dev).expand(n, 3)
        return p + scene.env_origins

    def red_jar():
        return scene.jars[int(scene.red_jar[0])]

    def move_jar_with_can(jar, can, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Rigid re-pose of a carafe WITH its resident can (relative pose kept)."""
        jp, jq = jar.data.root_pos_w.clone(), jar.data.root_quat_w.clone()
        rel_p = task_scene._qapply(task_scene._qinv(jq), can.data.root_pos_w - jp)
        rel_q = task_scene._qmul(task_scene._qinv(jq), can.data.root_quat_w)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        jar.write_root_state_to_sim(st, all_ids)
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = pos_w + task_scene._qapply(quat_w, rel_p)
        st[:, 3:7] = task_scene._qmul(quat_w, rel_q)
        can.write_root_state_to_sim(st, all_ids)
        _refresh()

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def push_body(body, dir_w, mag: float, steps: int, vcap: float = 0.35,
                  watch=None) -> bool:
        """Velocity-capped constant CoM force along a WORLD direction; returns True
        if `watch()` ever fired during the push. Clears the force afterwards."""
        fdir = torch.tensor(dir_w, device=dev, dtype=torch.float).expand(n, 3)
        _FR["q_ref"] = body.data.root_quat_w.clone()
        fired = False
        for _ in range(steps):
            v = float((body.data.root_lin_vel_w[0] * fdir[0]).sum())
            m = mag if v < vcap else 0.0
            f = task_scene.encode_force(_FR["mode"], _FR["q_ref"],
                                        body.data.root_quat_w,
                                        m * fdir).view(n, 1, 3)
            body.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                               is_global=True)
            _step(1)
            if watch is not None and bool(watch()):
                fired = True
        clear_force(body)
        return fired

    def layout() -> tuple:
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        tq = scene.tray.data.root_quat_w[0]
        yaw = math.atan2(float(tq[3]), float(tq[0])) * 2.0
        ja = (scene.jar_a.data.root_pos_w - scene.env_origins)[0]
        rp = (scene.alphabet.data.root_pos_w - scene.env_origins)[0]
        return (yaw, float(tp[0]), float(tp[1]), float(ja[0]), float(ja[1]),
                float(rp[0]), float(rp[1]))

    def score0() -> float:
        return float(scene.score()[0])

    # --- probe the force-frame convention once (scratch episode, free can) ---
    env.reset(seed=99)
    _step(60)
    _write_body(scene.alphabet, ground_pt(0.0, 0.50, 0.045))
    _step(30)
    xdir = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
    for mag in (1.5, 3.0):
        x0 = float(scene.alphabet.data.root_pos_w[0, 0])
        _FR["q_ref"] = scene.alphabet.data.root_quat_w.clone()
        for _ in range(60):
            f = task_scene.encode_force(_FR["mode"], _FR["q_ref"],
                                        scene.alphabet.data.root_quat_w,
                                        mag * xdir).view(n, 1, 3)
            scene.alphabet.set_external_force_and_torque(f, zero_w, env_ids=all_ids,
                                                         is_global=True)
            _step(1)
        clear_force(scene.alphabet)
        dx = float(scene.alphabet.data.root_pos_w[0, 0]) - x0
        if dx < -0.004:
            _FR["mode"] ^= 1
            print(f"[smoke] force-frame mode -> {_FR['mode']} (dx={dx:+.4f})",
                  flush=True)
        elif dx > 0.004:
            break
    print(f"[smoke] force-frame mode = {_FR['mode']}", flush=True)

    # ---------------- 1. settle / no-NaN ---------------------------------------------------------
    env.reset(seed=0)
    _step(120)
    _report("settle")
    red_ok = bool(scene.in_jar(red_jar(), scene.alphabet.data.root_pos_w)[0])
    corn_ok = bool(scene.in_jar(scene.jars[1 - int(scene.red_jar[0])],
                                scene.corn.data.root_pos_w)[0])
    tz = float((scene.tray.data.root_pos_w - scene.env_origins)[0, 2])
    check("settle/no-NaN (cans captive, tray grounded)",
          bool(scene._finite()[0]) and red_ok and corn_ok and -0.01 < tz < 0.03
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"red_in_jar={red_ok} corn_in_jar={corn_ok} tray_z={tz:.3f} "
          f"score={score0():.3f}")

    # ---------------- 2. randomization readback (3 seeds, max-pairwise) --------------------------
    lays = []
    for s in (0, 1, 2):
        env.reset(seed=s)
        _step(15)
        lays.append(layout())
    dyaw = max(abs(a[0] - b[0]) for a in lays for b in lays)
    dtray = max(max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a in lays for b in lays)
    djar = max(max(abs(a[3] - b[3]), abs(a[4] - b[4])) for a in lays for b in lays)
    dcan = max(max(abs(a[5] - b[5]), abs(a[6] - b[6])) for a in lays for b in lays)
    check("randomization readback",
          dyaw > 0.01 and dtray > 0.003 and djar > 0.003 and dcan > 0.005,
          f"dyaw={dyaw:.3f} dtray={dtray:.3f} djar={djar:.3f} dcan={dcan:.3f}")

    # ---------------- 3. red-jar shuffle ---------------------------------------------------------
    seen = set()
    for s in range(100, 110):
        env.reset(seed=s)
        _refresh()
        seen.add(int(scene.red_jar[0]))
    check("red-jar shuffle", len(seen) >= 2, f"red_jar values={sorted(seen)}")

    # ---------------- 4. null policy -------------------------------------------------------------
    env.reset(seed=0)
    _step(30)
    kp0 = scene.alphabet.data.root_pos_w[0].clone()
    jp0 = red_jar().data.root_pos_w[0].clone()
    _step(240)
    kdrift = float((scene.alphabet.data.root_pos_w[0] - kp0).norm())
    jdrift = float((red_jar().data.root_pos_w[0] - jp0).norm())
    check("null policy",
          kdrift < 0.010 and jdrift < 0.010 and score0() <= 0.03
          and not bool(scene.success()[0]),
          f"can_drift={kdrift:.4f} jar_drift={jdrift:.4f} score={score0():.3f}")

    # ---------------- 5. JAR DUMP rejected (transport-only cheat) --------------------------------
    env.reset(seed=1)
    _step(60)
    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=dev).expand(n, 4)
    q_dump = task_scene._qmul(scene.tray.data.root_quat_w, qy90)
    move_jar_with_can(red_jar(), scene.alphabet,
                      tray_pt(-0.100, 0.0, 0.050), q_dump)
    _step(300)
    _report("jar-dump")
    can_pos_in_tray = bool(scene.in_tray(scene.alphabet.data.root_pos_w)[0])
    check("jar dump rejected (can POSITION in the tray, still captive: no credit)",
          can_pos_in_tray
          and bool(scene.in_any_jar(scene.alphabet.data.root_pos_w)[0])
          and not bool(scene._decanted[0]) and not bool(scene._delivered[0])
          and score0() <= 0.03 and not bool(scene.success()[0]),
          f"pos_in_tray={can_pos_in_tray} score={score0():.3f}")

    # ---------------- 6. captive bore under a real lateral push ----------------------------------
    env.reset(seed=2)
    _step(60)
    kp0 = scene.alphabet.data.root_pos_w[0].clone()
    push_body(scene.alphabet, (-1.0, 0.0, 0.0), 5.0, 240)
    _step(60)
    _report("bore-push")
    moved = float((scene.alphabet.data.root_pos_w[0] - kp0).norm())
    check("captive bore (5 N lateral push scoots the carafe, can never exits)",
          moved > 0.020
          and bool(scene.in_jar(red_jar(), scene.alphabet.data.root_pos_w)[0])
          and not bool(scene._decanted[0]) and score0() <= 0.03
          and not bool(scene.success()[0]),
          f"moved={moved:.3f} score={score0():.3f}")

    # ---------------- 7. partial tilt (60 deg, held) retains the can -----------------------------
    env.reset(seed=3)
    _step(60)
    q_tilt = torch.tensor([math.cos(math.radians(30.0)), 0.0,
                           math.sin(math.radians(30.0)), 0.0],
                          device=dev).expand(n, 4)
    hold_p = ground_pt(0.00, -0.50, 0.150)
    move_jar_with_can(red_jar(), scene.alphabet, hold_p, q_tilt)
    gst = torch.zeros(n, 13, device=dev)
    gst[:, 0:3] = hold_p
    gst[:, 3:7] = q_tilt
    for _ in range(240):
        red_jar().write_root_state_to_sim(gst, all_ids)
        _step(1)
    _report("tilt-60")
    check("partial tilt (60 deg held: the can stays in the bore)",
          bool(scene.in_jar(red_jar(), scene.alphabet.data.root_pos_w)[0])
          and not bool(scene._decanted[0]) and score0() <= 0.03
          and not bool(scene.success()[0]),
          f"score={score0():.3f}")

    # ---------------- 8. decant-only near miss (0.30) --------------------------------------------
    env.reset(seed=4)
    _step(60)
    _write_body(scene.alphabet, ground_pt(0.05, 0.50, 0.045))
    _step(150)
    _report("decant-only")
    check("decant near-miss (can on open ground: 0.30, no success)",
          bool(scene._decanted[0]) and not bool(scene._delivered[0])
          and 0.29 <= score0() <= 0.31 and not bool(scene.success()[0]),
          f"score={score0():.3f}")

    # ---------------- 9. goal construct + live re-judge ------------------------------------------
    env.reset(seed=5)
    _step(60)
    _write_body(scene.alphabet, tray_pt(0.0, 0.0, 0.050),
                quat=scene.tray.data.root_quat_w)
    _step(180)
    _report("goal-construct")
    goal_ok = bool(scene.success()[0]) and score0() >= 0.999
    _write_body(scene.alphabet, ground_pt(0.05, 0.50, 0.045))
    _step(120)
    _report("goal-removed")
    latch_ok = (0.60 - 1e-4) <= score0() <= 0.61 \
        and not bool(scene.success()[0]) and bool(scene._delivered[0])
    check("goal construct succeeds; removal drops success, keeps the 0.60 latch",
          goal_ok and latch_ok, f"goal={goal_ok} latch={latch_ok} "
          f"score={score0():.3f}")

    # ---------------- 10. wrong object -----------------------------------------------------------
    env.reset(seed=6)
    _step(60)
    _write_body(scene.corn, tray_pt(0.055, 0.0, 0.050),
                quat=scene.tray.data.root_quat_w)
    _step(150)
    _report("corn-in-tray")
    corn_in = bool(scene.in_tray(scene.corn.data.root_pos_w)[0])
    a_ok = corn_in and score0() <= 0.03 and not bool(scene.success()[0])
    _write_body(scene.alphabet, tray_pt(-0.055, 0.0, 0.050),
                quat=scene.tray.data.root_quat_w)
    _step(150)
    _report("both-in-tray")
    b_ok = bool(scene.in_tray(scene.alphabet.data.root_pos_w)[0]) \
        and bool(scene.in_tray(scene.corn.data.root_pos_w)[0]) \
        and not bool(scene.success()[0]) and score0() <= 0.61
    check("wrong object (corn earns nothing; corn in the tray blocks success)",
          a_ok and b_ok, f"a={a_ok} b={b_ok} score={score0():.3f}")

    # ---------------- 11. carafe parked in the tray ----------------------------------------------
    env.reset(seed=7)
    _step(60)
    _write_body(scene.alphabet, tray_pt(-0.060, 0.0, 0.050),
                quat=scene.tray.data.root_quat_w)
    _step(120)
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = tray_pt(0.055, 0.0, 0.020)
    st[:, 3:7] = scene.tray.data.root_quat_w
    red_jar().write_root_state_to_sim(st, all_ids)
    _refresh()
    _step(240)
    _report("jar-in-tray")
    check("carafe parked in the tray blocks success (delivered can or not)",
          bool(scene.in_tray(scene.alphabet.data.root_pos_w)[0])
          and bool(scene.jar_in_tray(red_jar())[0])
          and not bool(scene.success()[0]) and score0() <= 0.61,
          f"jar_in_tray={bool(scene.jar_in_tray(red_jar())[0])} "
          f"score={score0():.3f}")

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
