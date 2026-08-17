"""Smoke / rubric-REJECTION battery for WeighbridgeScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the real accumulated-load tip and
the latched credit is monotone along that trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that the physical claims the task rests on — foam cannot
tip the beam, one iron cube cannot tip the beam, weight outside the pan does nothing,
an empty tip does not survive — are load-bearing physics, not fiat. Every probe is
CONSTRUCTED as a settled state (teleport, real physics steps, judge) — instrumentation,
never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; beam on its counterweight stop
                           (theta < -9 deg), six cubes flat on the ground; score ~0,
                           no success;
   2. randomization      — two seeded resets: READBACK slot permutation differs and
                           iron_0's ground xy moves by > 3 mm;
   3. slot permutation   — over 10 resets: >= 5 distinct permutations and the IRON
                           TRIPLE occupies >= 4 distinct slot-sets (the mass-identity
                           scatter is real);
   4. null-policy        — 240 idle steps: score ~0, no success, beam never moves;
   5. FOAM CANNOT TIP    — all three foam cubes constructed inside the pan tray: the
                           beam stays counterweight-down (theta < -9), no success,
                           score ~0 (foam earns nothing anywhere);
   6. ONE IRON SHORT     — 1 iron + all 3 foam in the pan (the most generous
                           insufficient load): beam never crosses horizontal, no
                           success; only legitimate appr+iron1 credit (<= 0.26);
   7. EMPTY TIP REVERTS  — the beam teleported to the pan-down stop with NOTHING in
                           the pan: the counterweight rights it within seconds; the
                           load-gated angle latches earn nothing (score ~0), no
                           success (the seed-analog "carry/hold aloft with no load
                           path" state cannot be faked);
   8. WRONG SIDE         — two iron cubes set on top of the COUNTERWEIGHT block: they
                           help the wrong arm; beam stays down on the rest stop, not
                           in pan, no success, score ~0;
   9. ARM OUTSIDE PAN    — two iron cubes on the pan-side ARM near the hinge (real
                           weight, wrong lever arm, outside the tray): torque budget
                           says the beam must NOT tip; in-pan count 0, no success,
                           score <= 0.11 (approach credit alone may latch);
  10. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.scoop_with_spatula_i97.smoke --headless
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
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

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
    th = float(scene.theta_deg()[0])
    print(f"[smoke] {tag:16s} | theta={th:+6.2f}deg iron_in_pan={int(scene.iron_in_pan()[0])} "
          f"appr={bool(scene._appr[0])} i1={bool(scene._iron1[0])} i2={bool(scene._iron2[0])} "
          f"cross={bool(scene._cross[0])} high={bool(scene._high[0])} "
          f"settled={bool(scene.settled()[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weighbridge")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.35, -1.05, 0.85)) + o),
                                tuple(np.array((0.32, -0.08, 0.12)) + o),
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

    def beam_put(body, local, drop: float = 0.030) -> None:
        """CONSTRUCT: write the body at a point given in the BEAM'S BODY FRAME,
        `drop` metres above it along the beam-local z (probe instrumentation; the
        caller settles when the arrangement is complete). Zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = float(local[0]), float(local[1]), float(local[2]) + drop
        pos = scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)
        _write_body(body, pos)

    # beam-local rest heights
    z_pan = c.arm_t / 2 + c.pan_floor_t + c.cube_size / 2       # cube on the pan floor
    z_cw = c.arm_t / 2 + c.cw_h + c.cube_size / 2               # cube on the counterweight block
    z_arm = c.arm_t / 2 + c.cube_size / 2                       # cube on the bare arm

    def pan_put(body, y_off: float, dz: float = 0.0) -> None:
        beam_put(body, (c.pan_x, y_off, z_pan + dz))

    def settle(max_steps: int = 480, tail: int = 60) -> None:
        for i in range(max_steps):
            _step(1)
            if i > 30 and bool(scene.settled()[0]):
                break
        _step(tail)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    pos, _v = scene._cube_tensors(scene.CUBE_NAMES)
    hz = pos[0, :, 2] - env.iscene.env_origins[0, 2]
    th0 = float(scene.theta_deg()[0])
    check("settle/no-NaN: layout settles finite; beam on its counterweight stop, six "
          "cubes flat on the ground; score ~0, no success",
          bool(torch.isfinite(pos).all()) and th0 < -9.0
          and bool((hz > 0.005).all() and (hz < 0.05).all())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (tuple(scene.slot_of[0].tolist()),
                scene.cubes["iron_0"].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_perm, a_xy = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_perm, b_xy = readback()
    d_xy = float((a_xy - b_xy).norm())
    print(f"[smoke] randomization: perm {a_perm} -> {b_perm}, "
          f"iron_0 xy moved {d_xy * 1000:.1f}mm", flush=True)
    check("randomization-is-real: slot permutation and iron_0 ground xy readback differ",
          a_perm != b_perm and d_xy > 0.003)

    # ================= 3. slot permutation coverage ===============================================
    perms, iron_sets = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        p = tuple(scene.slot_of[0].tolist())
        perms.add(p)
        iron_sets.add(tuple(sorted(p[:3])))
    print(f"[smoke] over 10 resets: {len(perms)} distinct permutations, "
          f"{len(iron_sets)} distinct iron slot-sets", flush=True)
    check("slot permutation: >= 5 distinct permutations and the iron triple occupies "
          ">= 4 distinct slot-sets over 10 resets",
          len(perms) >= 5 and len(iron_sets) >= 4)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, beam still on its "
          "rest stop",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and float(scene.theta_deg()[0]) < -9.0)

    # ================= 5. FOAM CANNOT TIP (the mass-identity claim is physics) ====================
    # All three foam cubes constructed inside the pan tray — the very move that wins
    # with iron. Their summed weight is ~2 % of the counter-torque: the beam must not
    # stir off its stop, and foam earns NOTHING in the rubric.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pan_put(scene.cubes["foam_0"], -0.026)
    pan_put(scene.cubes["foam_1"], +0.026)
    settle(240)
    pan_put(scene.cubes["foam_2"], 0.0, dz=c.cube_size + 0.004)  # third rests on the pair
    settle(360)
    _report("foam-in-pan")
    _REC["on"] = False
    check("FOAM CANNOT TIP: all three foam cubes inside the pan tray — beam stays "
          "counterweight-down, no success, score ~0",
          float(scene.theta_deg()[0]) < -9.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 6. ONE IRON IS NOT ENOUGH ==================================================
    # The most generous INSUFFICIENT load: one iron cube plus ALL the foam in the pan.
    # The torque budget says the counterweight still wins — the beam must never cross
    # horizontal, so only the legitimate approach+iron1 credit may latch.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    pan_put(scene.cubes["iron_0"], -0.026)
    settle(360)
    pan_put(scene.cubes["foam_0"], +0.026)
    settle(240)
    pan_put(scene.cubes["foam_1"], -0.020, dz=c.cube_size + 0.004)
    pan_put(scene.cubes["foam_2"], +0.020, dz=c.cube_size + 0.004)
    settle(360)
    _report("one-iron+foam")
    _REC["on"] = False
    check("ONE IRON SHORT: 1 iron + all 3 foam in the pan — beam never crosses "
          "horizontal, no success, only appr+iron1 credit (score <= 0.26)",
          int(scene.iron_in_pan()[0]) == 1 and float(scene.theta_deg()[0]) < 0.0
          and not bool(scene._cross[0]) and not bool(scene._high[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.26)

    # ================= 7. EMPTY TIP REVERTS (angle latches are load-gated) ========================
    # The beam teleported to the pan-down stop with an EMPTY pan (the state a "just
    # press the pan down" policy would leave behind — and the nearest settleable
    # analog of the seed's unsupported carry-aloft end state). The counterweight must
    # right it within seconds, and the load-gated cross/high latches must stay dark.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = c.bridge_pos[0], c.bridge_pos[1], c.hinge_z
    st[:, 3:7] = _qy(torch.full((n,), math.radians(c.limit_deg - 0.5), device=device))
    st[:, 0:3] += env.iscene.env_origins
    scene.beam.write_root_state_to_sim(st, _all_ids())
    _refresh()
    th_now = float(scene.theta_deg()[0])
    print(f"[smoke] beam FORCED to theta={th_now:+.2f}deg with an empty pan", flush=True)
    _step(360)
    _report("empty-tip")
    _REC["on"] = False
    check("EMPTY TIP REVERTS: beam forced to the pan-down stop with nothing aboard — "
          "the counterweight rights it, angle latches stay dark, score ~0, no success",
          th_now > 9.0 and float(scene.theta_deg()[0]) < -9.0
          and not bool(scene._cross[0]) and not bool(scene._high[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 8. WRONG SIDE (weight must go IN THE PAN) ==================================
    # Two iron cubes set on top of the COUNTERWEIGHT block: real weight, wrong arm —
    # it presses the beam INTO its rest stop. Nothing may latch.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    beam_put(scene.cubes["iron_0"], (c.cw_x - 0.011, 0.0, z_cw))
    beam_put(scene.cubes["iron_1"], (c.cw_x + 0.011, 0.0, z_cw + c.cube_size + 0.004))
    settle(360)
    _report("wrong-side")
    _REC["on"] = False
    check("WRONG SIDE: two iron cubes on the counterweight block — beam stays on its "
          "rest stop, nothing in the pan, no success, score ~0",
          float(scene.theta_deg()[0]) < -9.0 and int(scene.iron_in_pan()[0]) == 0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 9. ARM OUTSIDE PAN (the lever arm is load-bearing) =========================
    # Two iron cubes on the pan-side ARM close to the hinge — real weight on the
    # correct side, but at ~0.05/0.10 m arm the summed torque (~0.47 N m) is far below
    # the 0.86 N m counter-torque AND the cubes are outside the tray. The beam must
    # not tip and the in-pan clauses must refuse; only approach credit may latch.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    beam_put(scene.cubes["iron_0"], (0.050, 0.0, z_arm))
    beam_put(scene.cubes["iron_1"], (0.100, 0.0, z_arm))
    settle(360)
    _report("arm-outside")
    _REC["on"] = False
    check("ARM OUTSIDE PAN: two iron cubes on the arm near the hinge — insufficient "
          "torque, in-pan count 0, no success, score <= 0.11",
          float(scene.theta_deg()[0]) < 0.0 and int(scene.iron_in_pan()[0]) == 0
          and not bool(scene._iron1[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.11)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weighbridge")
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
    main()
