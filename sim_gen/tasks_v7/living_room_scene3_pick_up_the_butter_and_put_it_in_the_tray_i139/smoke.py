"""Smoke / rubric-REJECTION battery for BeamBalanceTrayScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real load-and-balance trajectory, two seeds). This battery
proves the rubric REJECTS wrong outcomes, and that the mechanism claim the task rests
on — the balance actually discriminates mass — is physics, not fiat. Every probe is
CONSTRUCTED as a settled state (teleport above the pan volume gate, free fall, real
physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN    — seeded reset settles finite; the empty beam self-centers
                         level; butter + cubes on the ground; score ~0, no success;
   2. randomization    — two seeded resets: READBACK base yaw, base xy and the
                         butter's pose all differ;
   3. class+slots vary — over 12 resets the butter size class takes >= 2 values AND
                         cube_0 occupies >= 2 different ground slots (permutation);
   4. null-policy      — 240 idle steps: beam stays level, score ~0, no success
                         (the balance being level is NOT the task — the tray is empty);
   5. SEED STRATEGY    — the seed's whole plan ("put the butter in the tray"): butter
                         dropped into the tray pan and abandoned -> the beam slams to
                         its stop (tilt > balanced_deg, ~stop_deg) with the butter
                         retained in the pan -> NO success, score = the in_tray latch
                         alone (0.20). The open-top drop is necessary but no longer
                         sufficient;
   6. wrong mass (light)— 600 g butter in the tray + 300 g cube on the ballast pan:
                         the beam stays pinned tray-side past the gate -> no success,
                         score <= 0.40;
   7. wrong mass (heavy)— 150 g butter in the tray + 300 g cube on the ballast pan:
                         the beam tips BALLAST-side past the gate -> no success;
   8. mirrored pans    — butter on the BALLAST pan + matching cube in the TRAY pan:
                         the beam levels (masses match!) but the butter is in the
                         wrong pan -> no success (pan identity is load-bearing);
   9. cube-only        — matching cube on the ballast pan, butter left on the ground:
                         beam pinned ballast-side, score = the on_pan latch alone
                         (0.20), no success;
  10. overload         — butter in the tray + ALL THREE cubes on the ballast pan
                         (1.05 kg vs <= 0.6): pinned ballast-side -> no success,
                         score capped at 0.40;
  11. frames.npz       — video captured and saved to the CWD.

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
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from . import scene as _scene  # noqa: F401 - registers simgen.beam_balance_tray
except ImportError:  # pragma: no cover - direct-script fallback
    import sys  # noqa: E402

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _scene  # noqa: F401

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
    bl = scene._beam_local(scene.butter_pos_w())[0]
    print(f"[smoke] {tag:16s} | butter_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
          f"{float(bl[2]):+.3f}) beam={float(scene.beam_deg()[0]):+.1f}deg "
          f"in_tray={bool(scene.butter_in_tray()[0])} "
          f"balanced={bool(scene.balanced()[0])} settled={bool(scene.settled()[0])} "
          f"latch_tray={bool(scene._in_tray[0])} latch_pan={bool(scene._on_pan[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance_tray")().build(num_envs=args.num_envs,
                                                       device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.85, 0.70)) + o),
                                tuple(np.array((0.50, 0.00, 0.22)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def beam_point_w(loc_xyz) -> torch.Tensor:
        """Beam-local point -> world (per-env, beam's CURRENT pose incl. tilt)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)

    def drop_into_pan(body, sgn: float, half_h: float, settle: int = 300) -> None:
        """CONSTRUCT 'load in pan': teleport ABOVE the pan volume gate (floor +
        95 mm), free-fall in, settle. Same construction discipline as solve.py."""
        pos = beam_point_w((sgn * c.pan_x, 0.0, c.pan_floor_top + half_h + 0.095))
        _refresh()
        _write_body(body, pos, scene.beam.data.root_quat_w)
        _step(settle)

    def reset_with_class(want: int, seed0: int) -> int:
        """Reseed until the present butter has size class `want` (readback)."""
        for s in range(seed0, seed0 + 60):
            torch.manual_seed(s)
            env.reset()
            _refresh()
            if int(scene.butter_class[0]) == want:
                return s
        raise AssertionError(f"no seed in [{seed0},{seed0 + 60}) gives class {want}")

    def butter_body():
        return scene.butters[int(scene.butter_class[0])]

    def butter_half_h() -> float:
        return c.butter_dims[int(scene.butter_class[0])][2] / 2

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; empty beam self-centers level; "
          "butter and cubes on the ground; score ~0, no success",
          bool(scene._finite()[0]) and abs(float(scene.beam_deg()[0])) < 3.0
          and not bool(scene.butter_in_tray()[0])
          and not bool(scene.cubes_on_ballast()[0].any())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.base.data.root_quat_w[0]),
                scene.base.data.root_pos_w[0, :2].clone(),
                scene.butter_pos_w()[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_bp, a_bu = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_bp, b_bu = readback()
    d_yawv, d_bp = dyaw(a_yaw, b_yaw), float((a_bp - b_bp).norm())
    d_bu = float((a_bu - b_bu).norm())
    print(f"[smoke] randomization deltas: base_yaw={d_yawv:.1f}deg "
          f"base_xy={d_bp * 1000:.1f}mm butter_xy={d_bu * 1000:.1f}mm", flush=True)
    check("randomization-is-real: base yaw, base xy and the butter's position "
          "readback differ across seeds",
          d_yawv > 2.0 and d_bp > 0.003 and d_bu > 0.005)

    # ================= 3. butter class + cube slot permutation vary ================================
    classes = set()
    slots_seen = set()
    from isaaclab.utils.math import quat_apply_inverse
    slot_ref = torch.tensor(c.cube_slots, device=device, dtype=torch.float)
    for s in range(12):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        classes.add(int(scene.butter_class[0]))
        loc = quat_apply_inverse(scene.base.data.root_quat_w,
                                 scene.cubes[0].data.root_pos_w
                                 - scene.base.data.root_pos_w)[0, :2]
        slots_seen.add(int((slot_ref - loc).norm(dim=1).argmin()))
    print(f"[smoke] over 12 resets: butter classes {sorted(classes)}, "
          f"cube_0 slots {sorted(slots_seen)}", flush=True)
    check("class+slots vary: butter size class takes >= 2 values and cube_0 "
          "occupies >= 2 different slots over 12 resets",
          len(classes) >= 2 and len(slots_seen) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, beam level but tray empty — score ~0, "
          "no success (a level balance is not the task)",
          abs(float(scene.beam_deg()[0])) < 3.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # The seed's complete plan is "pick up the butter and put it in the tray". Here
    # that exact end state — butter resting in the tray pan, nothing else done —
    # slams the beam to its stop. Must be retained-but-rejected: in_tray latch only.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_into_pan(butter_body(), +1.0, butter_half_h(), settle=420)
    _report("seed-strategy")
    tilt = float(scene.beam_deg()[0])
    print(f"[smoke] butter-only tilt: {tilt:+.1f}deg (gate {c.balanced_deg:.0f}, "
          f"stop {c.stop_deg:.0f})", flush=True)
    check("negative (SEED strategy): butter in the tray and nothing else — the beam "
          "slams to its stop with the butter retained in the pan; in_tray latch only "
          "(score 0.20), no success",
          bool(scene.butter_in_tray()[0]) and tilt > c.balanced_deg
          and abs(float(scene.score()[0]) - c.w_in_tray) < 0.01
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. near-miss: wrong mass, too LIGHT ========================================
    s6 = reset_with_class(2, 400)  # 600 g butter
    print(f"[smoke] wrong-light probe uses seed {s6} (600 g butter)", flush=True)
    _step(60)
    _REC["on"] = True
    drop_into_pan(butter_body(), +1.0, butter_half_h(), settle=360)
    drop_into_pan(scene.cubes[1], -1.0, c.cube_sides[1] / 2, settle=480)  # 300 g
    _report("wrong-light")
    tilt = float(scene.beam_deg()[0])
    print(f"[smoke] 600g butter vs 300g cube: tilt {tilt:+.1f}deg", flush=True)
    check("near-miss (wrong mass, light): 600 g butter vs 300 g cube — beam stays "
          "pinned tray-side past the gate, no success, score <= 0.40",
          tilt > c.balanced_deg and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.40 + 1e-6)
    _REC["on"] = False

    # ================= 7. near-miss: wrong mass, too HEAVY ========================================
    s7 = reset_with_class(0, 500)  # 150 g butter
    print(f"[smoke] wrong-heavy probe uses seed {s7} (150 g butter)", flush=True)
    _step(60)
    drop_into_pan(butter_body(), +1.0, butter_half_h(), settle=360)
    drop_into_pan(scene.cubes[1], -1.0, c.cube_sides[1] / 2, settle=480)  # 300 g
    _report("wrong-heavy")
    tilt = float(scene.beam_deg()[0])
    print(f"[smoke] 150g butter vs 300g cube: tilt {tilt:+.1f}deg", flush=True)
    check("near-miss (wrong mass, heavy): 150 g butter vs 300 g cube — beam tips "
          "BALLAST-side past the gate, no success",
          tilt < -c.balanced_deg and not bool(scene.success()[0]))

    # ================= 8. negative: mirrored pans (level but wrong pan) ===========================
    # Matching masses on SWAPPED pans: the beam levels — physically a perfect
    # weighing — but the butter is not in the TRAY. Pan identity must refuse alone.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    cls = int(scene.butter_class[0])
    drop_into_pan(butter_body(), -1.0, butter_half_h(), settle=360)
    drop_into_pan(scene.cubes[cls], +1.0, c.cube_sides[cls] / 2, settle=540)
    _report("mirrored")
    check("negative (mirrored pans): butter on the BALLAST pan + matching cube in "
          "the TRAY — beam level and settled, but butter in the wrong pan: no "
          "success",
          bool(scene.balanced()[0]) and bool(scene.butter_on_ballast()[0])
          and not bool(scene.butter_in_tray()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. near-miss: cube only ====================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    cls = int(scene.butter_class[0])
    drop_into_pan(scene.cubes[cls], -1.0, c.cube_sides[cls] / 2, settle=420)
    _report("cube-only")
    check("near-miss (cube only): matching cube on the ballast pan, butter left on "
          "the ground — on_pan latch only (score 0.20), no success",
          bool(scene.cubes_on_ballast()[0, cls])
          and abs(float(scene.score()[0]) - c.w_on_pan) < 0.01
          and not bool(scene.success()[0]))

    # ================= 10. negative: overload =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_into_pan(butter_body(), +1.0, butter_half_h(), settle=300)
    for i in range(3):  # all three cubes (1.05 kg) onto the ballast pan
        off = (-c.pan_x, (i - 1) * 0.042, c.pan_floor_top + c.cube_sides[i] / 2 + 0.095)
        _write_body(scene.cubes[i], beam_point_w(off), scene.beam.data.root_quat_w)
        _step(120)
    _step(360)
    _report("overload")
    tilt = float(scene.beam_deg()[0])
    print(f"[smoke] overload tilt: {tilt:+.1f}deg", flush=True)
    check("negative (overload): butter in the tray + ALL THREE cubes on the ballast "
          "pan — pinned ballast-side past the gate, no success, score capped at 0.40",
          tilt < -c.balanced_deg and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.40 + 1e-6)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.beam_balance_tray")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
