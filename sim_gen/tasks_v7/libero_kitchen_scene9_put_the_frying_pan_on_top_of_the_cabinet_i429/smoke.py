"""Smoke battery for RatchetPortcullisScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a settled wrong outcome and asserts the
rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — gate shut, pan resting at its sampled outside pose,
                        bowl on the roof, score ~0, no success.
 2. randomization A   — the pan's sampled start (x, y, yaw) varies across
                        resets and the settled pan tracks it every time.
 3. randomization B   — the bowl's roof pose varies across resets (readback).
 4. null policy       — 240 idle steps -> gate stays shut, nothing scores.
 5. SEED strategy     — the seed task's outcome (pan ON TOP of the cabinet):
                        pan constructed settled on the roof beside the bowl —
                        score ~0, no success. The seed's whole skill is worth
                        nothing here.
 6. teleport bypass   — gate STOLEN open (written at the stop; it settles onto
                        a pawl catch) and the pan STOLEN onto the stove pad
                        without ever crossing the doorway: on_pad IS true and
                        the state IS settled, but the transit credential never
                        fired -> no success, score capped at stolen latch
                        credit. The doorway crossing cannot be teleported.
 7. closed-gate push  — REAL force on the pan straight at the doorway with the
                        gate SHUT: the pan slides (it moved — the force pathway
                        is live) and is WALLED by the closed gate; transit
                        never fires, score ~0, no success (order forcing).
 8. ratchet one-way   — REAL lift to the stop + gentle set-down: the pawl
                        carries the gate hands-off (partial credit only, no
                        success); then a REAL 2x-weight downward drag cannot
                        pull the ratcheted gate back down (one-way under
                        hand-scale force).
 9. near miss + gates — REAL full mechanism run: gate ratcheted open, pan slid
                        through the doorway (transit latches) to an interior
                        point OFF the pad -> capped 0.65, no success. Then
                        judged-without-stepping isolation constructs: pan
                        TILTED 30 deg on the pad (upright gate refuses), pan
                        flat on the pad but MOVING (settle gate refuses); then
                        the pan is retracted and finally stolen back OUTSIDE:
                        latches survive, success does not.
10. wrong object      — the bystander bowl shoved off the roof to the ground:
                        score ~0, no success (and nothing to gain).
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i429.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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


def _write_body(body, pos_w: torch.Tensor, quat=(1.0, 0.0, 0.0, 0.0),
                vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = quat[0]
    st[:, 4] = quat[1]
    st[:, 5] = quat[2]
    st[:, 6] = quat[3]
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene.pan_pos()[0]
    print(f"[smoke] {tag:16s} | lift={float(scene.gate_lift()[0]):+.4f} "
          f"pan=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"latch(o/p/t)=({int(scene._opened[0])},{int(scene._prop[0])},"
          f"{int(scene._transit[0])}) on_pad={bool(scene.on_pad()[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ratchet_portcullis")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.95, 0.80)) + o),
                                tuple(np.array((-0.08, 0.00, 0.14)) + o),
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

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    zero = torch.zeros(n, 1, 3, device=device)

    def servo(body, tgt_xyz, *, kp: float, kd: float, clamp: float, steps: int,
              done=None) -> None:
        """REAL actuation: clamped world-frame PD force on `body`'s CoM toward
        a world target (None entries unservoed), force-limited to hand scale."""
        tgt = torch.zeros(n, 3, device=device)
        mask = torch.zeros(3, device=device)
        for a, v in enumerate(tgt_xyz):
            if v is not None:
                tgt[:, a] = v
                mask[a] = 1.0
        for _ in range(steps):
            pos = body.data.root_pos_w - scene.env_origins
            vel = body.data.root_lin_vel_w
            f_w = (kp * (tgt - pos) - kd * vel) * mask
            f_w = f_w.clamp(min=-clamp, max=clamp)
            f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        _step(1)  # flush the cleared wrench

    def open_gate_for_real() -> None:
        """Ratchet the gate open with honest force: lift to the stop, then ease
        it down onto the pawl (same move as the solve)."""
        servo(scene.gate, (None, None, c.gate_hi + 0.01), kp=300.0, kd=30.0,
              clamp=12.0, steps=600,
              done=lambda: float(scene.gate_lift()[0]) >= c.gate_hi - 0.004)
        servo(scene.gate, (None, None, c.gate_hi), kp=200.0, kd=25.0,
              clamp=8.0, steps=60)  # hold: let the flicked pawl re-seat
        servo(scene.gate, (None, None, c.gate_hi - 0.015), kp=120.0, kd=25.0,
              clamp=8.0, steps=300)
        _step(150)  # hands off: the pawl carries the gate

    # ================= 1. settle / no-NaN =========================================================
    env.reset(seed=11)
    _step(150)
    _report("reset")
    p1 = scene.pan_pos()[0]
    check("settle/no-NaN: gate shut, pan resting at its sampled outside pose, "
          "bowl on the roof, score ~0, no success",
          bool(scene._finite()[0])
          and float(scene.gate_lift()[0]) < 0.005
          and abs(float(p1[0]) - float(scene.p0[0, 0])) < 0.01
          and abs(float(p1[1]) - float(scene.p0[0, 1])) < 0.01
          and float((scene.bowl.data.root_pos_w - scene.env_origins)[0, 2]) > c.roof_z1
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    p0s, pans, b0s = [], [], []
    for k in range(8):
        env.reset(seed=100 + k)
        _step(60)
        _refresh()
        p0s.append(scene.p0[0].tolist())
        pans.append(scene.pan_pos()[0].tolist())
        b0s.append(scene.b0[0].tolist())
    p0a = np.asarray(p0s)
    spans = p0a.max(axis=0) - p0a.min(axis=0)
    tracks = all(abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01
                 for p, q in zip(pans, p0s))
    b0a = np.asarray(b0s)
    bspan = (b0a.max(axis=0) - b0a.min(axis=0)).tolist()
    print(f"[smoke] readback: pan spans x={spans[0]:.4f} y={spans[1]:.4f} "
          f"yaw={spans[2]:.4f} tracks={tracks} bowl spans x={bspan[0]:.4f} "
          f"y={bspan[1]:.4f}", flush=True)
    check("randomization A: the pan's sampled start (x, y, yaw) varies across "
          "resets and the settled pan tracks it every time",
          spans[0] > 0.020 and spans[1] > 0.030 and spans[2] > 0.15 and tracks)
    check("randomization B: the bowl's roof pose varies across resets",
          bspan[0] > 0.030 and bspan[1] > 0.015)

    # ================= 4. null policy =============================================================
    env.reset(seed=31)
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> gate stays shut, pan stays outside, "
          "score ~0, no success",
          float(scene.gate_lift()[0]) < 0.005
          and float(scene.pan_pos()[0, 0]) > 0.15
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (pan on TOP of the cabinet) ===============================
    env.reset(seed=41)
    _step(120)
    b0x, b0y = float(scene.b0[0, 0]), float(scene.b0[0, 1])
    px = b0x + 0.125 if b0x < -0.20 else b0x - 0.125
    # handle to the side (yaw 90 deg) so only the disc needs roof room
    _write_body(scene.pan, origin(px, 0.0, c.roof_z1 + 0.003),
                quat=(0.7071068, 0.0, 0.0, 0.7071068))
    _step(150)
    _report("seed-on-roof")
    pz = float(scene.pan_pos()[0, 2])
    check("SEED strategy: the pan constructed settled ON TOP of the cabinet "
          "beside the bowl (the seed task's goal) — score ~0, no success",
          pz > c.roof_z0 and bool(scene.settled()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. teleport bypass (the credential check) ==================================
    env.reset(seed=51)
    _step(120)
    _write_body(scene.gate, origin(0.0, 0.0, c.gate_hi))
    _step(150)  # drops a few mm onto a pawl catch and stays
    lift6 = float(scene.gate_lift()[0])
    _write_body(scene.pan, origin(c.pad_x, c.pad_y, 0.001))
    _step(120)
    _report("teleport-bypass")
    check("teleport bypass: gate stolen open (settles onto a pawl catch) and "
          "the pan stolen onto the stove pad without ever crossing the doorway "
          "— on_pad and settled are TRUE but the transit credential never "
          "fired: no success, score capped",
          lift6 >= 0.105
          and bool(scene.on_pad()[0]) and bool(scene.settled()[0])
          and not bool(scene._transit[0])
          and float(scene.score()[0]) <= 0.65 + 1e-4 and not succ())

    # ================= 7. closed-gate push (order forcing) ========================================
    env.reset(seed=61)
    _step(120)
    x_start = float(scene.pan_pos()[0, 0])
    servo(scene.pan, (0.02, 0.0, None), kp=60.0, kd=12.0, clamp=8.0, steps=480)
    _step(60)
    _report("closed-gate")
    x_end = float(scene.pan_pos()[0, 0])
    check("closed-gate push: a real 8 N drag at the doorway moves the pan (the "
          "force is live) but the shut gate WALLS it outside; transit never "
          "fires, score ~0, no success",
          x_start - x_end >= 0.05 and x_end >= 0.045
          and float(scene.gate_lift()[0]) < 0.02
          and not bool(scene._transit[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. ratchet one-way (real lift, then a real pull-down) =====================
    env.reset(seed=71)
    _step(120)
    open_gate_for_real()
    lift8 = float(scene.gate_lift()[0])
    held = lift8 >= 0.105 and bool(scene._opened[0]) and bool(scene._prop[0])
    s8 = float(scene.score()[0])
    # 2x-weight downward drag on the ratcheted gate
    servo(scene.gate, (None, None, -0.05), kp=400.0, kd=20.0, clamp=6.0, steps=240)
    _step(90)
    _report("one-way")
    lift8b = float(scene.gate_lift()[0])
    check("ratchet one-way: a real lift + set-down leaves the pawl carrying the "
          "gate hands-off (partial credit only), and a real 6 N downward drag "
          "cannot pull the ratcheted gate back down; no success",
          held and c.w_open + c.w_prop - 1e-4 <= s8 <= 0.65 + 1e-4
          and lift8b >= lift8 - 0.006 and not succ())

    # ================= 9. near miss + judged-state gates + latch steal ============================
    env.reset(seed=81)
    _step(120)
    open_gate_for_real()
    servo(scene.pan, (0.15, 0.0, None), kp=60.0, kd=12.0, clamp=8.0, steps=480,
          done=lambda: abs(float(scene.pan_pos()[0, 0]) - 0.15) < 0.02
          and abs(float(scene.pan_pos()[0, 1])) < 0.015)
    servo(scene.pan, (-0.12, 0.0, None), kp=50.0, kd=14.0, clamp=6.0, steps=720,
          done=lambda: float(scene.pan_pos()[0, 0]) < -0.115)
    _step(90)
    _report("near-miss")
    near = (bool(scene._transit[0])
            and not bool(scene.on_pad()[0])
            and float(scene.score()[0]) <= 0.65 + 1e-4 and not succ())
    check("near miss: the REAL mechanism run end-to-end but the pan left short "
          "of the pad — transit latched, capped 0.65, no success", near)
    # tilted on the pad: transit + position + settled all fine, upright fails
    _write_body(scene.pan, origin(c.pad_x, c.pad_y, 0.020),
                quat=(0.9659258, 0.0, 0.2588190, 0.0))  # pitch 30 deg
    tilt_rejected = not succ() and not bool(scene.on_pad()[0])
    # flat on the pad but MOVING: the settle gate refuses (judged w/o stepping)
    _write_body(scene.pan, origin(c.pad_x, c.pad_y, 0.001), vel_x=0.30)
    moving_rejected = not succ() and bool(scene.on_pad()[0])
    # retract BEFORE any stepping (a flat still pan on the pad would be genuine
    # success here — the transit credential is legitimately earned)
    _write_body(scene.pan, origin(-0.12, -0.05, 0.001))
    _step(60)
    check("judged-state gates: pan TILTED 30 deg on the pad refused (upright "
          "gate), pan flat on the pad but MOVING refused (settle gate)",
          tilt_rejected and moving_rejected)
    # steal the pan back OUTSIDE: latches survive, success does not
    _write_body(scene.pan, origin(0.30, 0.10, 0.001))
    _step(90)
    _report("latch-steal")
    check("latched credit: after the real open+transit the pan is stolen back "
          "outside — the latches survive (score stays 0.65), success does not",
          bool(scene._transit[0])
          and 0.65 - 1e-4 <= float(scene.score()[0]) <= 0.65 + 1e-4
          and not succ())

    # ================= 10. wrong object (bystander bowl shoved off the roof) =====================
    env.reset(seed=91)
    _step(120)
    _write_body(scene.bowl, origin(0.50, -0.40, 0.002))
    _step(120)
    _report("wrong-object")
    check("wrong object: the bystander bowl shoved off the roof to the ground "
          "— score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ratchet_portcullis")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
