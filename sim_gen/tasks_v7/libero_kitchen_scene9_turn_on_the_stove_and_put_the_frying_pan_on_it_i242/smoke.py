"""Smoke / rubric-REJECTION battery for BallastStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the
rubric REJECTS wrong outcomes and that the claims the task rests on — the
mass discrimination, the gravity-held level stop, the fence/tray windows, the
settle gate and the LIVE deadman — are load-bearing. Every probe is CONSTRUCTED
(teleport transports, real physics steps, judge); success() is audited at every
step and allowed ONLY inside the sanctioned deadman window of check 11.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; beam on its trivet-down tilt
                          stop, ingot and decoy in opposite apron slots, pan upright
                          on the apron; score 0, no success;
   2. randomization     — two seeded resets: READBACK ingot xy, decoy xy, pan xy and
                          pan yaw all differ;
   3. slot coin         — over 10 resets the ingot appears in BOTH the west and the
                          east slot (readback, not flag);
   4. null-policy       — 240 idle steps: score ~0, no success, the beam holds its
                          tilt stop (no creep toward level);
   5. SEED-NAIVE        — the seed-family move: pan set straight onto the TILTED,
                          unlit trivet without any ballast. The beam-frame seat
                          clause reads it (delivery credit 0.15, the positive
                          control) but the stove stays tipped and OFF: score
                          <= 0.151, never success;
   6. DECOY             — the same-size wood block dropped into the ballast tray:
                          15x too light — the beam does NOT leave its tilt stop
                          (physics, not decree) and the ingot-specific tray clause
                          gives nothing: score ~0, no success;
   7. SWAP              — ingot dropped onto the TRIVET and pan into the TRAY (both
                          arms loaded, roles swapped): the beam stays tipped (the
                          ingot now REINFORCES the bias) and nothing is judged:
                          score ~0, no success;
   8. fence-perch       — legit ballast (beam level), pan dropped 65 mm north: it
                          rests half-over the rim fence — the xy window / upright
                          cone refuse the perch: not seated, no success, <= 0.401;
   9. settle gate       — legit ballast, the exact success pose written WITH
                          0.4 m/s pan velocity: geometry alone is not success —
                          refused while moving (dismantled immediately);
  10. MECHANISM         — from a fresh reset, the ingot released from a hover ABOVE
                          the judged tray band (asserted: the write satisfies
                          nothing): pure gravity + tray contact rotate the beam
                          > 14 deg onto its LEVEL stop -> score 0.40, no success;
  11. DEADMAN           — continue: pan dropped onto the trivet -> live success,
                          score 1.0 (sanctioned window); then the ingot is lifted
                          OUT (transport) -> success collapses INSTANTLY, gravity
                          re-tilts the beam back onto the tilt stop with the pan
                          still riding the trivet, and the score falls back to the
                          latched 0.55 cap;
  12. audit             — success() was never True OUTSIDE the sanctioned deadman
                          window;
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i242.smoke --headless
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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
# success() may be True ONLY inside the sanctioned deadman window (check 11).
_AUDIT = {"hit": False, "allow": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if not _AUDIT["allow"]:
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


def _write_state(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                 vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | pitch={float(scene.beam_pitch_deg()[0]):+.2f}deg "
          f"level={bool(scene.beam_level()[0])} "
          f"tray={bool(scene.ingot_in_tray()[0])} "
          f"pan={bool(scene.pan_on_trivet()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -1.00, 0.85)) + o),
                                tuple(np.array((0.00, 0.00, 0.20)) + o),
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

    def pitch() -> float:
        _refresh()
        return float(scene.beam_pitch_deg()[0])

    def up_z(body) -> float:
        _refresh()
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        return float(quat_apply(body.data.root_quat_w, ez)[0, 2])

    def rel(body) -> torch.Tensor:
        _refresh()
        return (body.data.root_pos_w - scene.env_origins)[0]

    def score() -> float:
        _refresh()
        return float(scene.score()[0])

    def beam_pt(local_xyz) -> torch.Tensor:
        """A beam-frame point in world coords via the LIVE beam pose."""
        _refresh()
        loc = torch.tensor(local_xyz, device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)

    def beam_quat(yaw: float | None = None) -> torch.Tensor:
        """The live beam attitude, optionally composed with a local yaw."""
        _refresh()
        q = scene.beam.data.root_quat_w.clone()
        if yaw is None:
            return q
        return quat_mul(q, _qz(torch.full((n,), yaw, device=device)))

    def apron_park(body, x: float, y: float, z_half: float) -> None:
        """Transport a body to a free apron spot on the deck and settle it."""
        pos = scene.env_origins.clone()
        pos[:, 0] += x
        pos[:, 1] += y
        pos[:, 2] += c.deck_h + z_half + 0.004
        _write_state(body, pos)
        _step(40)

    # The solve's transport hovers (asserted to satisfy nothing at the write):
    ING_HOVER = (-c.arm, 0.0, c.plate_top + 0.055)   # above the tray z band
    PAN_HOVER = (c.arm, 0.0, c.pan_seat_z + 0.020)   # above the seat z band

    def drop_ingot_in_tray(settle: int = 240) -> None:
        _write_state(scene.ingot, beam_pt(ING_HOVER), beam_quat())
        _step(settle)

    def drop_pan_on_trivet(dy: float = 0.0, settle: int = 240) -> None:
        pos = beam_pt((c.arm, dy, c.pan_seat_z + 0.020))
        _write_state(scene.pan, pos, beam_quat(-math.pi / 2))
        _step(settle)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    ip, bp = rel(scene.ingot), rel(scene.block)
    fin = bool(torch.isfinite(scene.beam.data.root_pos_w).all()
               and torch.isfinite(scene.ingot.data.root_pos_w).all()
               and torch.isfinite(scene.pan.data.root_pos_w).all())
    check("settle/no-NaN: beam on its tilt stop, ingot+decoy in opposite slots, "
          "pan upright on the apron; score 0, no success",
          fin and c.tilt_deg - 1.2 <= pitch() <= c.tilt_deg + 1.2
          and abs(abs(float(ip[0])) - c.slot_x) <= c.slot_jitter + 0.01
          and abs(abs(float(bp[0])) - c.slot_x) <= c.slot_jitter + 0.01
          and float(ip[0]) * float(bp[0]) < 0
          and up_z(scene.pan) > 0.95
          and score() <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (rel(scene.ingot)[:2].clone(), rel(scene.block)[:2].clone(),
                rel(scene.pan)[:2].clone(), yaw_of(scene.pan.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_i, a_b, a_p, a_y = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_i, b_b, b_p, b_y = readback()
    d_i, d_b = float((a_i - b_i).norm()), float((a_b - b_b).norm())
    d_p, d_y = float((a_p - b_p).norm()), dyaw(a_y, b_y)
    print(f"[smoke] randomization deltas: ingot={d_i * 1000:.1f}mm "
          f"decoy={d_b * 1000:.1f}mm pan={d_p * 1000:.1f}mm yaw={d_y:.1f}deg",
          flush=True)
    check("randomization-is-real: ingot xy, decoy xy, pan xy, pan yaw readback differ",
          d_i > 0.005 and d_b > 0.005 and d_p > 0.005 and d_y > 3.0)

    # ================= 3. slot coin: the ingot deals to BOTH slots ================================
    sides = []
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        ix = float(rel(scene.ingot)[0])
        assert (ix > 0) == bool(scene.swap[0]), "swap flag vs ingot slot mismatch"
        sides.append(1 if ix > 0 else -1)
    print(f"[smoke] ingot slot over 10 resets: {sides}", flush=True)
    check("slot coin: the ingot appears in BOTH apron slots over 10 resets",
          (1 in sides) and (-1 in sides))

    # ================= 4. null policy fails (and the beam holds its stop) =========================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, the beam holds "
          "its tilt stop (no creep toward level)",
          score() <= 0.01 and not bool(scene.success()[0])
          and pitch() >= c.tilt_deg - 1.2)

    # ================= 5. SEED-NAIVE: pan straight onto the tilted, unlit trivet ==================
    # The seed-family move — "put the pan on the stove" without turning it on.
    # The beam-frame seat clause DOES read it (delivery credit, the positive
    # control for the trivet-relative rubric) but the stove stays tipped + OFF.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_pan_on_trivet()
    _report("seed-naive")
    _REC["on"] = False
    check("SEED-NAIVE: pan set on the TILTED unlit trivet — seat credit only "
          "(0.15, beam-frame positive control), beam still on its tilt stop: "
          "no success, score <= 0.151",
          bool(scene.pan_on_trivet()[0]) and pitch() >= c.tilt_deg - 1.5
          and not bool(scene.success()[0]) and 0.149 <= score() <= 0.151)

    # ================= 6. DECOY: same size, 15x too light — physics refuses =======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p_before = pitch()
    _write_state(scene.block, beam_pt(ING_HOVER), beam_quat())
    _step(240)
    _report("decoy")
    _REC["on"] = False
    print(f"[smoke] decoy: pitch {p_before:+.2f} -> {pitch():+.2f}deg "
          f"(decoy tau {c.decoy_mass * 9.81 * c.arm:.3f} vs bias "
          f"{c.bias_torque:.3f} N*m)", flush=True)
    check("DECOY: the wood block in the ballast tray cannot move the beam off its "
          "tilt stop, and the ingot-specific tray clause gives nothing: score ~0, "
          "no success",
          pitch() >= c.tilt_deg - 1.5 and not bool(scene.beam_level()[0])
          and not bool(scene.ingot_in_tray()[0])
          and score() <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. SWAP: ingot on the trivet, pan in the tray ==============================
    # Both arms loaded but roles swapped: the ingot now REINFORCES the tilt bias
    # and the pan (in the tray) is far too light to fight ingot + bias.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_state(scene.ingot,
                 beam_pt((c.arm, 0.0, c.plate_top + c.block_size[2] / 2 + 0.020)),
                 beam_quat())
    _step(120)
    _write_state(scene.pan, beam_pt((-c.arm, 0.0, c.plate_top + 0.055)),
                 beam_quat(-math.pi / 2))
    _step(240)
    _report("swap")
    check("SWAP: ingot on the trivet + pan in the tray — the beam stays tipped "
          "(the ingot reinforces the bias) and nothing is judged: score ~0, "
          "no success",
          pitch() >= c.tilt_deg - 1.5 and not bool(scene.ingot_in_tray()[0])
          and not bool(scene.pan_on_trivet()[0])
          and score() <= 0.01 and not bool(scene.success()[0]))

    # ================= 8. fence-perch: the seat windows are load-bearing ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_ingot_in_tray()
    assert bool(scene.beam_level()[0]), "ballast must level the beam for the perch probe"
    drop_pan_on_trivet(dy=0.065)
    _report("fence-perch")
    p_beam = float(scene._in_beam(scene.pan)[0, 1])
    print(f"[smoke] fence-perch: pan beam-frame dy={p_beam * 1000:+.1f}mm "
          f"(tol {c.pan_xy_tol * 1000:.0f}mm) up_z={up_z(scene.pan):+.3f}",
          flush=True)
    check("fence-perch: pan dropped 65 mm north rests half-over the rim fence — "
          "the xy window / upright cone refuse: not seated, no success, <= 0.401",
          not bool(scene.pan_on_trivet()[0]) and not bool(scene.success()[0])
          and score() <= 0.401)

    # ================= 9. settle gate: the success pose in motion is refused ======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_ingot_in_tray()
    assert bool(scene.beam_level()[0]), "ballast must level the beam for the gate probe"
    _write_state(scene.pan, beam_pt((c.arm, 0.0, c.pan_seat_z)),
                 beam_quat(-math.pi / 2), vel_x=0.40)
    _step(1)
    _report("settle-gate")
    moving_refused = bool(scene.pan_on_trivet()[0]) and not bool(scene.success()[0]) \
        and float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]) > c.settle_speed
    # dismantle IMMEDIATELY (before the slide damps into a genuine success)
    apron_park(scene.pan, 0.0, c.pan_y_nom, c.pan_base_t / 2)
    check("settle gate: the exact success geometry moving at 0.4 m/s is refused "
          "(geometry alone is not success; dismantled before it could calm)",
          moving_refused and not bool(scene.success()[0]))

    # ================= 10. MECHANISM: gravity ballast really levels the beam ======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    p_start = pitch()
    _write_state(scene.ingot, beam_pt(ING_HOVER), beam_quat())
    _step(1)
    hover_clean = not bool(scene.ingot_in_tray()[0]) and not bool(scene.beam_level()[0])
    _step(300)
    _report("mechanism")
    print(f"[smoke] mechanism: pitch {p_start:+.2f} -> {pitch():+.2f}deg "
          f"(hover write satisfied nothing: {hover_clean})", flush=True)
    check("MECHANISM: the ingot released ABOVE the judged band rotates the beam "
          "> 14 deg onto its LEVEL stop by pure gravity + contact — score 0.40, "
          "no success",
          hover_clean and p_start - pitch() > 14.0
          and bool(scene.beam_level()[0]) and bool(scene.ingot_in_tray()[0])
          and 0.39 <= score() <= 0.401 and not bool(scene.success()[0]))

    # ================= 11. DEADMAN: live success, then the ballast is lifted out ==================
    # Continue from the mechanism state (beam level, ingot in the tray). The pan
    # drop reaches genuine success (sanctioned window) — then the ingot is
    # lifted OUT and the valve must shut ITSELF: success collapses instantly,
    # gravity re-tilts the beam onto the tilt stop with the pan still aboard,
    # and only the latched 0.55 survives.
    _AUDIT["allow"] = True
    drop_pan_on_trivet()
    _report("deadman-armed")
    armed = bool(scene.success()[0]) and score() >= 0.999
    apron_park(scene.ingot, 0.0, c.pan_y_nom, c.block_size[2] / 2)  # lift the ballast out
    _AUDIT["allow"] = False
    dead_now = not bool(scene.success()[0])
    _step(240)
    _report("deadman-cut")
    _REC["on"] = False
    check("DEADMAN: success held live with the ballast in (score 1.0), collapsed "
          "the moment the ingot left the tray, and the beam re-tilted itself onto "
          "the tilt stop — score falls back to the latched 0.55",
          armed and dead_now and not bool(scene.success()[0])
          and pitch() >= c.tilt_deg - 1.5
          and 0.549 <= score() <= 0.5501)

    # ================= 12. audit ==================================================================
    check("audit: success() was never True outside the sanctioned deadman window",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_stove")
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
    except BaseException as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
