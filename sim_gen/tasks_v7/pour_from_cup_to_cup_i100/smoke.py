"""Smoke / rubric-REJECTION battery for TipDumpScene — NullRobot, physical probes.

solve.py is the acceptance proof (the rubric accepts the press-hold-drain-release
outcome and the printed credit is monotone along a real trajectory). This battery
proves the rubric REJECTS wrong outcomes, and that the claims the task rests on — the
~57 deg drain angle is real, the wrong pedal is an irrecoverable spill, the release
endgame is load-bearing — are physics, not fiat. Force probes write the scene's own
`press_tau` input (clamped by the scene to the fingertip bound); every force-driven
rejection check ASSERTS THE ACTUATOR ACTUALLY MOVED (a probe that never tipped the
hopper proves nothing). Constructed states are teleported, settled with real physics
steps, then judged. No probe here is asserted to reach success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite; every present ball rests inside
                          the hopper, absent balls parked in the depot, hinge near
                          vertical; score ~0, no success;
   2. randomization     — two seeded resets: READBACK of basin pose + ball_0 position
                          + start tilt differ;
   3. coverage          — over 10 resets BOTH basin sides appear (>= 2x each) and
                          >= 2 distinct present-ball counts appear;
   4. null-policy       — 240 idle steps: hinge stays near vertical, every present
                          ball still in the hopper, score ~0, no success;
   5. SUB-DRAIN TIP     — press the correct pedal to 45 deg (< the ~57 deg drain) and
                          hold ~2 s: hinge VERIFIED >= 40 deg, yet every ball stays in
                          the hopper and score stays ~0 (partial tips spill nothing);
                          release: the spring returns the hopper upright hands-off;
   6. WRONG-SIDE DUMP   — press the WRONG pedal past 55 deg and hold ~4 s: hinge
                          VERIFIED <= -55 deg toward the bare floor, balls leave the
                          hopper, ZERO land in the basin, the signed latch never arms
                          -> score ~0, no success (the direction commitment is real
                          and irrecoverable);
   7. HELD-OPEN         — correct-side press-and-hold until every ball IS in the
                          basin, judged WHILE STILL HELD: success must refuse (the
                          hopper is not upright) and score caps at latch+frac = 0.75
                          (the release endgame is load-bearing);
   8. OVERSHOT SPILL    — present balls constructed settled on the bare floor just
                          PAST the basin's far wall (hopper untouched, upright):
                          containment refuses, score ~0, no success;
   9. PARTIAL DRAIN     — all but one present ball constructed settled inside the
                          basin, one left in the hopper: success refuses and the score
                          sits in the 0.60*(k-1)/k band (credit is proportional,
                          never full);
  10. frames.npz        — video captured and saved to the CWD.

The seed-strategy end state (grasp the source vessel, carry it over the target, pour)
has NO analog here — there is no carryable vessel; the hopper is trunnion-mounted to
the stand. Documented as N/A in TASK.md.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pour_from_cup_to_cup_i100.smoke --headless
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
    pres = scene.present[0]
    print(f"[smoke] {tag:16s} | hinge={float(scene.hinge_deg()[0]):+7.2f}deg "
          f"in_basin={int((scene.in_basin()[0] & pres).sum())}/{int(pres.sum())} "
          f"in_hopper={int((scene.in_hopper()[0] & pres).sum())} "
          f"latch={bool(scene._tip_latch[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tip_dump_station")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -1.05, 0.85)) + o),
                                tuple(np.array((0.05, 0.0, 0.22)) + o),
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

    origin = env.iscene.env_origins  # (n, 3)

    # --- pedal-press servo (writes ONLY the scene's own clamped press_tau input) ---
    TAU_MAX, KP, KD, SLEW_DPS = 2.2, 6.0, 0.5, 40.0

    def servo(target_deg: float, seconds: float, stop=None, rock: bool = False,
              slew_dps: float = SLEW_DPS) -> None:
        """Feedforward+PD press toward `target_deg` with a slewed target — the same
        fingertip-honest law as solve.py. Leaves the press HELD (caller releases)."""
        th_des = float(scene.hinge_deg()[0])
        slew = slew_dps * env.dt
        sgn = 1.0 if target_deg >= th_des else -1.0
        t = 0.0
        while t < seconds:
            th_des = (min(th_des + slew, target_deg) if sgn > 0
                      else max(th_des - slew, target_deg))
            tgt = th_des
            if rock and t > 3.0 and abs(th_des - target_deg) < 1e-6:
                tgt = target_deg + math.copysign(
                    4.0 * math.sin(2 * math.pi * 1.2 * (t - 3.0)), target_deg)
            th = float(scene.hinge_rad()[0])
            w = float(scene.hinge_rate()[0])
            tau = c.spring_k * math.radians(tgt) + KP * (math.radians(tgt) - th) - KD * w
            scene.press_tau[0] = max(-TAU_MAX, min(TAU_MAX, tau))
            _step(1)
            t += env.dt
            if stop is not None and stop():
                return

    def release_and_settle(steps: int = 240) -> None:
        scene.press_tau[0] = 0.0
        _step(steps)

    def n_present() -> int:
        return int(scene.present[0].sum())

    def n_in_basin() -> int:
        return int((scene.in_basin()[0] & scene.present[0]).sum())

    def n_in_hopper() -> int:
        return int((scene.in_hopper()[0] & scene.present[0]).sum())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    ball_pos = torch.stack([b.data.root_pos_w[0] for b in scene.balls])  # (4, 3)
    depot_ok = True
    for i in range(c.n_max):
        if not bool(scene.present[0, i]):
            d = ball_pos[i, :2] - origin[0, :2] - torch.tensor(
                [c.depot[0], c.depot[1] + 0.06 * i], device=device)
            depot_ok = depot_ok and float(d.norm()) < 0.05
    check("settle/no-NaN: present balls rest inside the hopper, absent balls parked "
          "in the depot, hinge near vertical, score ~0, no success",
          bool(torch.isfinite(ball_pos).all())
          and n_in_hopper() == n_present() and depot_ok
          and abs(float(scene.hinge_deg()[0])) < 6.0
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return torch.cat([scene.basin.data.root_pos_w[0, :2] - origin[0, :2],
                          scene.balls[0].data.root_pos_w[0, :2] - origin[0, :2],
                          scene.hinge_deg()[0:1]]).clone()

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b = readback()
    d = (a - b).abs()
    print(f"[smoke] randomization deltas: basin={float(d[:2].norm()) * 1000:.1f}mm "
          f"ball0={float(d[2:4].norm()) * 1000:.1f}mm tilt={float(d[4]):.2f}deg", flush=True)
    check("randomization-is-real: basin pose, ball_0 position and start tilt "
          "readback differ between two seeded resets",
          float(d[:2].norm()) > 0.003 and float(d[2:4].norm()) > 0.003)

    # ================= 3. coverage: both sides, variable ball count ===============================
    sides, counts = [], []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.append(int(scene.side[0]))
        counts.append(n_present())
    print(f"[smoke] over 10 resets: sides={sides} counts={counts}", flush=True)
    check("coverage: both basin sides appear >= 2x and >= 2 distinct present-ball "
          "counts appear over 10 resets",
          sides.count(1) >= 2 and sides.count(-1) >= 2 and len(set(counts)) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — hinge near vertical, every present "
          "ball still in the hopper, score ~0, no success",
          abs(float(scene.hinge_deg()[0])) < 6.0 and n_in_hopper() == n_present()
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 5. SUB-DRAIN TIP spills nothing ============================================
    torch.manual_seed(100)
    env.reset()
    _step(90)
    side = int(scene.side[0])
    _REC["on"] = True
    # Quasi-static approach (8 deg/s): the ~57 deg drain angle is a *static* geometric
    # threshold. A fast slew lets the balls roll ~10 cm downhill and slam the 9 mm lip
    # with enough momentum to vault it well below the static angle (a real hopper
    # sloshes when jerked). The sub-drain claim under test is the static one: balls
    # resting against the lip at 45 deg stay put.
    servo(side * 45.0, 9.0, slew_dps=8.0)
    peak = side * float(scene.hinge_deg()[0])
    _report("sub-drain-held")
    held_ok = (peak >= 40.0  # the probe actually tipped the hopper (non-vacuous)
               and n_in_hopper() == n_present() and n_in_basin() == 0
               and float(scene.score()[0]) <= 0.02)
    release_and_settle(300)
    _report("sub-drain-rel")
    _REC["on"] = False
    check("SUB-DRAIN TIP: held at 45 deg (VERIFIED >= 40) — every ball stays in the "
          "hopper, score ~0; on release the spring returns the hopper upright",
          held_ok and abs(float(scene.hinge_deg()[0])) < c.upright_tol_deg
          and not bool(scene.success()[0]))

    # ================= 6. WRONG-SIDE DUMP is an irrecoverable ~0 ==================================
    torch.manual_seed(100)
    env.reset()
    _step(90)
    side = int(scene.side[0])
    _REC["on"] = True
    servo(-side * c.hold_deg, 5.0)
    wrong_peak = -side * float(scene.hinge_deg()[0])
    _report("wrong-side-held")
    dumped = n_in_hopper()
    release_and_settle(300)
    _report("wrong-side-rel")
    _REC["on"] = False
    check("WRONG-SIDE DUMP: pressed the wrong pedal past the drain angle (VERIFIED "
          ">= 55 deg away from the basin) — balls leave the hopper, ZERO in the "
          "basin, latch never arms, score ~0, no success",
          wrong_peak >= 55.0 and dumped == 0 and n_in_basin() == 0
          and not bool(scene._tip_latch[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 7. HELD-OPEN: the release endgame is load-bearing ==========================
    torch.manual_seed(100)
    env.reset()
    _step(90)
    side = int(scene.side[0])
    _REC["on"] = True

    def all_in() -> bool:
        return n_in_basin() == n_present()

    servo(side * c.hold_deg, 16.0, stop=all_in, rock=True)
    # judge WHILE STILL HELD (press_tau untouched, hopper still tipped)
    _step(60)  # let the last ball settle in the basin under the held tip
    _report("held-open")
    held_tilt = side * float(scene.hinge_deg()[0])
    check("HELD-OPEN near-miss: every ball drained into the basin but the pedal is "
          "still held (VERIFIED tip >= 55 deg) — success refuses, score <= 0.75",
          all_in() and held_tilt >= 55.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_latch + c.w_frac + 1e-3)
    release_and_settle(300)
    _report("held-open-rel")
    _REC["on"] = False

    # ================= 8. OVERSHOT SPILL: containment refuses =====================================
    torch.manual_seed(100)
    env.reset()
    _step(90)
    side = int(scene.side[0])
    _refresh()
    bp = scene.basin.data.root_pos_w[0] - origin[0]
    far_x = float(bp[0]) + side * (c.basin_in_x_half + c.basin_wall_t + 0.045)
    for i in range(c.n_max):
        if bool(scene.present[0, i]):
            pos = origin + torch.tensor([far_x, float(bp[1]) - 0.10 + 0.07 * i,
                                         c.ball_r + 0.002], device=device)
            _write_body(scene.balls[i], pos)
    _step(180)
    _report("overshot")
    check("OVERSHOT SPILL near-miss: present balls settled on the bare floor just "
          "past the basin's far wall — none in the basin, score ~0, no success",
          n_in_basin() == 0 and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # ================= 9. PARTIAL DRAIN: proportional credit, no success ==========================
    torch.manual_seed(100)
    env.reset()
    _step(90)
    _refresh()
    bp = scene.basin.data.root_pos_w[0] - origin[0]
    k = n_present()
    moved = 0
    for i in range(c.n_max):
        if bool(scene.present[0, i]) and moved < k - 1:
            pos = origin + torch.tensor(
                [float(bp[0]) - 0.06 + 0.06 * moved, float(bp[1]),
                 c.basin_floor_t + c.ball_r + 0.003], device=device)
            _write_body(scene.balls[i], pos)
            moved += 1
    _step(180)
    _report("partial")
    expect = c.w_frac * (k - 1) / k
    s_now = float(scene.score()[0])
    print(f"[smoke] partial drain: {moved} of {k} constructed in basin, "
          f"score={s_now:.3f} expected~{expect:.3f}", flush=True)
    check("PARTIAL DRAIN: all but one ball in the basin, one still in the hopper — "
          "success refuses, score sits in the proportional 0.60*(k-1)/k band",
          n_in_basin() == k - 1 and n_in_hopper() == 1
          and abs(s_now - expect) <= 0.02 and not bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tip_dump_station")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
