"""Smoke battery for VaultDepositScene (sim_gen task `obstacle_i284`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a settled wrong outcome and asserts the
rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — vault locked (bolt at its sampled lock, lid in its
                        sampled closed window), cubes outside, score ~0.
 2. randomization A   — housing xy jitters AND both yaw flip clusters (track
                        pointing either way) appear across resets (readback).
 3. randomization B   — bolt lock position and lid closed position jitter, and
                        the red/blue slot assignment SWAPS across resets.
 4. null policy       — 240 idle steps -> nothing moves, score ~0, no success.
 5. SEED strategy     — the seed task's whole plan (carry the payload over the
                        blocker, set it down at the goal) executed to its end
                        state: the red cube set down ON the closed lid, exactly
                        over the bay. Settled there it is NOT in the bay (the
                        z window rejects it); score ~0, no success.
 6. locked interlock  — REAL force probe, non-vacuous: the solve's own lid
                        servo pushes the closed lid toward open while the bolt
                        is still LOCKED. The lid MOVES (>= 2 mm — the force
                        pathway is live) but JAMS on the blade far short of
                        open; the bolt is not retracted by it; score ~0.
 7. wrong object      — the full correct sequence constructed with the DECOY:
                        bolt retracted, lid opened, BLUE cube dropped through
                        the mouth (it lands and settles in the bay — the
                        containment window demonstrably accepts real deposits),
                        lid re-closed. No success, capped stage credit only.
 8. both cubes in     — red added beside the settled decoy, lid closed: every
                        other gate holds, the decoy gate alone refuses success.
 9. unsecured deposit — red cube settled in the bay but the lid left OPEN:
                        deposit credit only (<= 0.55...), no success.
10. lid ajar          — same deposit, lid parked at x=+0.035 (outside the
                        closed window, mouth still mostly covered): no success.
11. latched credit    — the bolt REALLY retracted by force (w_bolt latches),
                        then teleport-stolen back to its lock: the latch
                        survives, success does not.
12. settle gate       — full success layout constructed but the cube written
                        WITH velocity: judged without stepping, the settle gate
                        refuses; the layout is destroyed before it can settle.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.obstacle_i284.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ----------------------------------------------------------
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | bolt={float(scene.bolt_travel()[0]):+.4f} "
          f"lid={float(scene._local(scene.lid)[0, 0]):+.4f} "
          f"cube_in={bool(scene.in_bay(scene.cube)[0])} "
          f"decoy_in={bool(scene.in_bay(scene.decoy)[0])} "
          f"lid_closed={bool(scene.lid_closed()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.vault_deposit")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.00, 0.85)) + o),
                                tuple(np.array((0.05, 0.00, 0.12)) + o),
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

    def finite() -> bool:
        return all(torch.isfinite(b.data.root_pos_w).all()
                   for b in (scene.lid, scene.bolt, scene.cube, scene.decoy))

    zero = torch.zeros(n, 1, 3, device=device)

    def h_axes():
        """Housing world pose + world x/y axes (read back live)."""
        hp = scene.housing.data.root_pos_w.clone()
        hq = scene.housing.data.root_quat_w.clone()
        ex = quat_apply(hq, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(n, 1))
        ey = quat_apply(hq, torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(n, 1))
        return hp, hq, ex, ey

    def write_local(body, x: float, y: float, z: float, *, vel_w=None) -> None:
        """Teleport a body to a housing-local position, housing-aligned, given vel."""
        hp, hq, _, _ = h_axes()
        local = torch.tensor([[x, y, z]], device=device).repeat(n, 1)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = hp + quat_apply(hq, local)
        st[:, 3:7] = hq
        if vel_w is not None:
            st[:, 7:10] = torch.tensor([vel_w], device=device)
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def servo(body, axis_w: torch.Tensor, v_des: float, *, kv: float, clamp: float,
              steps: int, done=None) -> None:
        """REAL actuation: the solve's own force-limited velocity servo."""
        for _ in range(steps):
            v = (body.data.root_lin_vel_w * axis_w).sum(dim=-1)
            f = (kv * (v_des - v)).clamp(min=-clamp, max=clamp)
            f_w = axis_w * f.unsqueeze(-1)
            f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN ========================================================
    env.reset(seed=11)
    _step(150)
    _report("reset")
    check("settle/no-NaN: vault locked — bolt at its sampled lock, lid in its "
          "sampled closed window, cubes outside, score ~0, no success",
          finite()
          and float(scene.bolt_travel()[0]) < 0.005
          and float(scene.lid_open()[0]) < 0.005
          and bool(scene.lid_closed()[0])
          and not bool(scene.in_bay(scene.cube)[0])
          and not bool(scene.in_bay(scene.decoy)[0])
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 2+3. randomization readback ===============================================
    hxys, yaw_flip, bolt_refs, lid_refs, red_side = [], [], [], [], []
    for k in range(10):
        env.reset(seed=100 + k)
        _step(50)
        _refresh()
        hp, hq, ex, _ = h_axes()
        hxys.append((hp[0] - scene.env_origins[0])[:2].tolist())
        yaw_flip.append(bool(ex[0, 0] < 0))  # True = 180-flipped cluster
        bolt_refs.append(float(scene._bolt_ref[0]))
        lid_refs.append(float(scene._lid_ref[0]))
        red_side.append(bool(scene._local(scene.cube)[0, 1] > 0))
    xystd = float(np.std(np.asarray(hxys), axis=0).mean())
    b_span = max(bolt_refs) - min(bolt_refs)
    l_span = max(lid_refs) - min(lid_refs)
    print(f"[smoke] readback: house_xystd={xystd:.4f} flips={yaw_flip} "
          f"bolt_span={b_span:.4f} lid_span={l_span:.4f} red_side={red_side}",
          flush=True)
    check("randomization A: housing xy jitters and BOTH yaw flip clusters (track "
          "pointing either way) appear across resets",
          xystd > 0.008 and any(yaw_flip) and not all(yaw_flip))
    check("randomization B: bolt lock and lid closed positions jitter, and the "
          "red/blue slot assignment swaps across resets",
          b_span > 0.002 and l_span > 0.003
          and any(red_side) and not all(red_side))

    # ================= 4. null policy ============================================================
    env.reset(seed=31)
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> bolt locked, lid closed, nothing "
          "deposited, score ~0, no success",
          float(scene.bolt_travel()[0]) < 0.005
          and float(scene.lid_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 5. SEED strategy (carry over the blocker, set down at the goal) ===========
    env.reset(seed=41)
    _step(120)
    # the seed task's end state: payload set down at the goal xy — here that means ON the
    # closed lid, over the bay (y=+0.055 stays clear of the lid's handle knob, |y| span
    # 0.033-0.078 vs handle 0.015, and is still inside the bay's xy window)
    lid_top = float(scene._local(scene.lid)[0, 2]) + 0.006 + c.cube_size / 2
    write_local(scene.cube, 0.0, 0.055, lid_top + 0.002)
    _step(150)
    _report("seed-skill")
    cube_z = float(scene._local(scene.cube)[0, 2])
    check("SEED strategy: the red cube set down ON the closed lid exactly over "
          "the bay (the seed's carry-over-and-set-down end state) settles there "
          "but is NOT in the bay — z window rejects it; score ~0, no success",
          cube_z > c.bay_z_win[1] + 0.02
          and not bool(scene.in_bay(scene.cube)[0])
          and not bool(scene._deposited[0])
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 6. locked interlock (REAL force probe, non-vacuous) ======================
    env.reset(seed=51)
    _step(120)
    _, _, ex_w, _ = h_axes()
    lid_before = float(scene.lid_open()[0])
    servo(scene.lid, ex_w, 0.10, kv=20.0, clamp=6.0, steps=300)
    _report("locked-probe")
    lid_jam = float(scene.lid_open()[0])
    check("locked interlock: the solve's own lid servo pushes the closed lid "
          "while the bolt is LOCKED — the lid moves (force pathway live) but "
          "jams on the blade far short of open; the bolt stays locked; score ~0",
          lid_jam - lid_before >= 0.002
          and lid_jam <= 0.035
          and float(scene.bolt_travel()[0]) < 0.02
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. wrong object (full sequence with the DECOY) ===========================
    env.reset(seed=61)
    _step(120)
    write_local(scene.bolt, 0.116, float(scene._bolt_ref[0]) + c.bolt_clear + 0.004,
                0.100 + 0.012 + 0.001)
    write_local(scene.lid, c.lid_open_need + 0.008, 0.0, 0.146 + 0.001)
    _step(60)
    write_local(scene.decoy, 0.0, 0.0, 0.22)
    _step(200)
    decoy_landed = bool(scene.in_bay(scene.decoy)[0])
    write_local(scene.lid, 0.0, 0.0, 0.146 + 0.001)
    _step(120)
    _report("decoy-deposit")
    check("wrong object: the full correct sequence executed with the DECOY — it "
          "lands and settles in the bay (containment window accepts real "
          "deposits) and the lid re-closes, yet no success and no deposit credit",
          decoy_landed and bool(scene.in_bay(scene.decoy)[0])
          and bool(scene.lid_closed()[0])
          and not bool(scene._deposited[0])
          and float(scene.score()[0]) <= 0.25 + 1e-4 and not succ())

    # ================= 8. both cubes in (decoy gate alone refuses) ==============================
    # continue from check 7's state: decoy settled in the bay, lid closed
    write_local(scene.cube, -0.050, 0.0, 0.055)
    _step(120)
    _report("both-in")
    check("both cubes in: red settled beside the decoy, lid closed — every other "
          "gate holds, the decoy gate ALONE refuses success (capped credit)",
          bool(scene.in_bay(scene.cube)[0]) and bool(scene.in_bay(scene.decoy)[0])
          and bool(scene.lid_closed()[0])
          and float(scene.score()[0]) <= 0.55 + 1e-4 and not succ())

    # ================= 9. unsecured deposit (lid left open) =====================================
    env.reset(seed=71)
    _step(120)
    write_local(scene.bolt, 0.116, float(scene._bolt_ref[0]) + c.bolt_clear + 0.004,
                0.100 + 0.012 + 0.001)
    write_local(scene.lid, c.lid_open_need + 0.008, 0.0, 0.146 + 0.001)
    _step(60)
    write_local(scene.cube, 0.0, 0.0, 0.22)
    _step(200)
    _report("unsecured")
    check("unsecured deposit: red cube settled in the bay but the lid left OPEN "
          "— deposit credit only, no success",
          bool(scene.in_bay(scene.cube)[0])
          and not bool(scene.lid_closed()[0])
          and 0.30 - 1e-4 <= float(scene.score()[0]) <= 0.55 + 1e-4 and not succ())

    # ================= 10. lid ajar (outside the closed window) =================================
    write_local(scene.lid, 0.035, 0.0, 0.146 + 0.001)
    _step(90)
    _report("lid-ajar")
    check("lid ajar: same deposit, lid parked at x=+0.035 — outside the closed "
          "window though the mouth is mostly covered: no success",
          bool(scene.in_bay(scene.cube)[0])
          and not bool(scene.lid_closed()[0])
          and float(scene.score()[0]) <= 0.55 + 1e-4 and not succ())

    # ================= 11. latched credit (real retraction, then stolen back) ===================
    env.reset(seed=81)
    _step(120)
    _, _, _, ey_w = h_axes()
    servo(scene.bolt, ey_w, 0.10, kv=5.0, clamp=2.0, steps=600,
          done=lambda: float(scene.bolt_travel()[0]) >= c.bolt_clear + 0.002)
    really_retracted = float(scene.bolt_travel()[0]) >= c.bolt_clear
    write_local(scene.bolt, 0.116, float(scene._bolt_ref[0]),
                0.100 + 0.012 + 0.001)
    _step(90)
    _report("latch-steal")
    check("latched credit: the bolt REALLY retracted by force (w_bolt latches), "
          "then teleport-stolen back to its lock — the latch survives, success "
          "does not",
          really_retracted
          and float(scene.bolt_travel()[0]) < 0.010
          and c.w_bolt - 1e-4 <= float(scene.score()[0]) <= 0.55 + 1e-4
          and not succ())

    # ================= 12. settle gate (success layout, moving cube) ============================
    env.reset(seed=91)
    _step(120)
    write_local(scene.bolt, 0.116, float(scene._bolt_ref[0]) + c.bolt_clear + 0.004,
                0.100 + 0.012 + 0.001)
    write_local(scene.lid, 0.0, 0.0, 0.146 + 0.001)
    write_local(scene.cube, 0.0, 0.0, 0.055, vel_w=(0.0, 0.0, 0.40))
    in_bay_now = bool(scene.in_bay(scene.cube)[0])
    moving_rejected = not succ()  # judged WITHOUT stepping: in the bay but moving
    write_local(scene.cube, 0.10, -0.30, c.cube_size / 2 + 0.003)  # destroy the layout
    _step(90)
    _report("settle-gate")
    check("settle gate: full success layout with the cube written MOVING — "
          "judged without stepping, in the bay but not settled: refused; layout "
          "destroyed before it can settle",
          in_bay_now and moving_rejected and not succ())

    # ================= 13-15. audit, no-NaN, video ===============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", finite())

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.vault_deposit")
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
