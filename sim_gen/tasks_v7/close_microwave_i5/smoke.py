"""Smoke / rubric-REJECTION battery for BayonetCanisterScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claims the task rests on — the twist is what
makes the lid captive; the decoy physically cannot enter — are load-bearing. Every
probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; both lids standing on their
                           feet away from the canister; score 0, no success;
   2. randomization      — two seeded resets: READBACK canister yaw, canister xy and
                           blue-lid xy all differ;
   3. slot swap          — over 10 resets the blue lid occupies BOTH ground slots;
   4. null-policy        — 240 idle steps: nothing moves to the canister, score ~0,
                           no success;
   5. SEED STRATEGY      — the seed's plan is "push the cover closed and stop":
                           CONSTRUCT the blue lid seated on the mouth UNTWISTED
                           (psi ~ +2 deg) -> seated but NO success, score <= 0.75;
                           then a 3x-weight upward pull removes it (> 20 mm rise):
                           an untwisted lid is not locked, physically;
   6. LOCK CAPTURE       — CONSTRUCT the locked state (seated, psi ~ +48 deg); the
                           same 3x-weight pull raises it < 8 mm (the lug feet jam
                           into the flange bottom) and it is still locked after
                           release: the twist is load-bearing, not rubric fiat;
   7. near-miss twist    — seated but twisted only to psi ~ +30 deg (below the
                           40 deg lock gate) -> no success, score <= 0.75;
   8. wrong lid          — the RED four-winged decoy dropped centred and aligned on
                           the canister: its four feet cannot line up with three
                           notches, it PERCHES ~19 mm high -> readback high, no
                           success;
   9. wrong place        — blue lid standing on the GROUND beside the canister with
                           a lock-looking relative yaw (psi ~ +48) -> no success
                           (seat clauses refuse);
  10. perch              — blue lid dropped on the canister MISALIGNED (psi ~ +60:
                           wings over the flange arcs) -> rests high on the flange,
                           no success;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_microwave_i5.smoke --headless
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

_qz, _yaw_deg = task_scene._qz, task_scene._yaw_deg

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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
    loc = scene._can_local(scene.blue.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | lid_z={float(loc[2]):+.3f} "
          f"psi={float(scene.psi_deg()[0]):+6.1f} "
          f"over={bool(scene.lid_over()[0])} seated={bool(scene.lid_seated()[0])} "
          f"locked={bool(scene.lid_locked()[0])} "
          f"lat=({int(scene._over[0])},{int(scene._seated[0])},{int(scene._rotated[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _pull(body, force_n: float, steps: int) -> float:
    """Apply a world +z force at the body's CoM for `steps` steps; return the max rise
    of the body origin above its starting height. Clears the wrench afterwards."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = zero.clone()
    f[:, 0, 2] = force_n
    z0 = float(body.data.root_pos_w[0, 2])
    zmax = z0
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
        zmax = max(zmax, float(body.data.root_pos_w[0, 2]))
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    return zmax - z0


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_canister")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.75, 0.65)) + o),
                                tuple(np.array((0.35, 0.0, 0.10)) + o),
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

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def lid_on_canister(body, psi_deg: float, dz: float = 0.001) -> None:
        """CONSTRUCT: teleport `body` onto the canister axis at seat height + dz with
        relative twist `psi_deg`, then settle."""
        _refresh()
        can_yaw = _yaw_deg(scene.canister.data.root_quat_w)
        p = scene.canister.data.root_pos_w.clone()
        p[:, 2] += c.z_seat + dz
        _write_body(body, p, _qz(torch.deg2rad(can_yaw + psi_deg)))
        _step(90)

    def lid_z_loc() -> float:
        _refresh()
        return float(scene._can_local(scene.blue.data.root_pos_w)[0, 2])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    d0 = float((scene.blue.data.root_pos_w[0, :2]
                - scene.canister.data.root_pos_w[0, :2]).norm())
    check("settle/no-NaN: layout settles finite; both lids on their feet away from "
          "the canister; score 0, no success",
          bool(scene._finite()[0]) and d0 > 0.20
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (float(_yaw_deg(scene.canister.data.root_quat_w)[0]),
                scene.canister.data.root_pos_w[0, :2].clone(),
                scene.blue.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_bp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_bp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_cp, d_bp = float((a_cp - b_cp).norm()), float((a_bp - b_bp).norm())
    print(f"[smoke] randomization deltas: canister_yaw={d_yawv:.1f}deg "
          f"canister_xy={d_cp * 1000:.1f}mm blue_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: canister yaw, canister xy, blue-lid xy readback differ",
          d_yawv > 2.0 and d_cp > 0.003 and d_bp > 0.003)

    # ================= 3. lid slot swap ===========================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add(int(scene.blue_slot[0]))
    print(f"[smoke] over 10 resets: blue lid slots {sorted(sides)}", flush=True)
    check("slot swap: the blue lid occupies BOTH ground slots over 10 resets",
          sides == {0, 1})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, nothing approaches the canister, "
          "score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY: cover it and stop (untwisted lid) ========================
    # The seed's whole plan is "push the cover closed". CONSTRUCT that end state: the
    # blue lid seated on the mouth, feet through the notches, NOT twisted. It must
    # not be success — and a 3x-weight upward pull must remove it (nothing holds it).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    lid_on_canister(scene.blue, psi_deg=2.0)
    _report("seed-strategy")
    seated_untwisted = bool(scene.lid_seated()[0])
    no_succ = not bool(scene.success()[0])
    sc = float(scene.score()[0])
    rise = _pull(scene.blue, 3.0 * c.lid_mass * 9.81, 40)
    _step(30)
    _report("untwisted-pull")
    print(f"[smoke] untwisted lid rise under 3x-weight pull: {rise * 1000:.1f} mm", flush=True)
    check("SEED strategy (cover, no twist): seated lid without the twist is NOT "
          "success (score <= 0.75) and a 3x-weight pull removes it",
          seated_untwisted and no_succ and sc <= 0.75 and rise > 0.020)
    _REC["on"] = False

    # ================= 6. LOCK CAPTURE: the twist is load-bearing =================================
    # CONSTRUCT the locked state (seated, psi ~ +48: feet under the flange, against
    # the stops). The same 3x-weight pull must NOT remove it: the feet jam into the
    # flange bottom (3 mm play), and after release it is still locked.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    lid_on_canister(scene.blue, psi_deg=48.0)
    _report("locked")
    locked0 = bool(scene.lid_locked()[0])
    rise = _pull(scene.blue, 3.0 * c.lid_mass * 9.81, 40)
    _step(60)
    _report("locked-pull")
    print(f"[smoke] locked lid rise under 3x-weight pull: {rise * 1000:.1f} mm", flush=True)
    check("LOCK CAPTURE: locked lid resists a 3x-weight pull (< 8 mm rise) and is "
          "still locked after release",
          locked0 and rise < 0.008 and bool(scene.lid_locked()[0]))
    _REC["on"] = False

    # ================= 7. near-miss: twisted short of the lock gate ===============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    lid_on_canister(scene.blue, psi_deg=30.0)
    _report("short-twist")
    psi = float(scene.psi_deg()[0])
    check("near-miss (short twist): seated at psi ~ +30 deg (< 40 lock gate) — no "
          "success, score <= 0.75",
          bool(scene.lid_seated()[0]) and psi < c.psi_lock_min
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 8. wrong lid: the decoy cannot enter =======================================
    # Drop the RED four-winged decoy centred on the canister, one wing aligned with a
    # notch. 4-fold lugs vs 3-fold notches: it must PERCH high on the flange.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    can_yaw = _yaw_deg(scene.canister.data.root_quat_w)
    p = scene.canister.data.root_pos_w.clone()
    p[:, 2] += c.z_seat + c.hover
    _write_body(scene.red, p, _qz(torch.deg2rad(can_yaw)))
    _step(150)
    _report("decoy-drop")
    red_z = float(scene._can_local(scene.red.data.root_pos_w)[0, 2])
    print(f"[smoke] decoy rest height on canister: z_loc={red_z * 1000:.1f} mm "
          f"(seat {c.z_seat * 1000:.0f} mm)", flush=True)
    check("wrong lid: the four-winged decoy dropped aligned on the canister perches "
          "high (>= seat + 8 mm), never seats — no success",
          red_z > c.z_seat + 0.008 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. wrong place: lock-looking yaw on the ground =============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    can_yaw = _yaw_deg(scene.canister.data.root_quat_w)
    p = scene.canister.data.root_pos_w.clone()
    p[:, 0] += 0.25  # standing on the ground beside the canister
    p[:, 2] = env.iscene.env_origins[:, 2] + c.z_rest + 0.002
    _write_body(scene.blue, p, _qz(torch.deg2rad(can_yaw + 48.0)))
    _step(90)
    _report("wrong-place")
    psi = float(scene.psi_deg()[0])
    check("wrong place: blue lid on the ground beside the canister with a "
          "lock-looking yaw (psi ~ +48) — no success",
          abs(psi - 48.0) < 10.0 and not bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 10. perch: misaligned wings rest ON the flange =============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    can_yaw = _yaw_deg(scene.canister.data.root_quat_w)
    p = scene.canister.data.root_pos_w.clone()
    p[:, 2] += c.z_seat + c.hover
    _write_body(scene.blue, p, _qz(torch.deg2rad(can_yaw + 60.0)))
    _step(150)
    _report("perch")
    z_loc = lid_z_loc()
    print(f"[smoke] misaligned blue lid rest height: z_loc={z_loc * 1000:.1f} mm "
          f"(seat {c.z_seat * 1000:.0f} mm)", flush=True)
    check("perch: blue lid dropped misaligned (psi ~ +60) rests high on the flange "
          "— not seated, no success",
          z_loc > c.z_seat + 0.008 and not bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bayonet_canister")
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
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)
