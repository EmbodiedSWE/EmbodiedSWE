"""Smoke / rubric-REJECTION battery for CrateDockScene — NullRobot, teleported probes.

solve.py is the acceptance proof (blocker pushed into the siding, payload pushed down
the lane through the mouth into the roofed dock, all by contact forces, score 1.0).
This battery proves the rubric REJECTS wrong outcomes and that the physical claims the
task rests on are load-bearing: the seed's carry-and-drop plan is dead (a payload
delivered straight into the dock is refused for skipping the mouth), the ordering is
geometry-enforced (pushing the payload first only wedges the blocker against the roof's
proud front edge; the blocker itself can never enter the dock), and partial outcomes
earn only their latched slice. Every probe is CONSTRUCTED as a state (teleport, real
physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; payload parked in the entry
                          lane, blocker plugging the junction; score 0, no success;
   2. randomization     — three seeded resets, MAX-PAIRWISE readback deltas: rig yaw,
                          rig xy, payload lane slot (rig-local x) and blocker world
                          position all differ (the push axes must be read, not assumed);
   3. null-policy       — 240 idle steps: nothing moves, score ~0, no success;
   4. SEED STRATEGY     — the seed's plan is grasp-carry-drop into the container. Its
                          end state CONSTRUCTED: payload teleported into the dock
                          window, settled — a REAL contained equilibrium, REJECTED:
                          it never crossed the mouth (score <= 0.26, no success). The
                          pathway latch is load-bearing;
   5. cleared-only      — blocker transported into the siding and left: exactly the
                          clearing credit (~0.15), payload untouched, no success;
   6. WRONG ORDER       — payload pushed down the lane with the blocker still in the
                          junction: the payload only rams the blocker into the roof's
                          front edge, where it JAMS (23 mm taller than the opening);
                          the payload advances to contact but never crosses the
                          junction band; score 0, no success;
   7. mouth straddle    — payload teleported into the mouth window and settled there,
                          half in, half out: mouth credit only (<= 0.21), not success
                          (containment window starts a full crate past the mouth);
   8. blocker-into-dock — blocker set on the open lane short of the mouth and pushed
                          +x with a hard grind: it advances freely, then jams at the
                          roof front edge and never enters the dock — the "wrong
                          crate" can never satisfy the goal; score stays ~0.15 (its
                          clearing latch never fires either; no success);
   9. rejection audit   — success() never fired at any judged step during the
                          negative probes (checks 4-8);
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i307.smoke --headless
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

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"on": False, "fired": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUDIT["on"]:
            _AUDIT["fired"] |= bool(env.scene.success()[0])
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


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crate_dock")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.00, -0.85, 0.75)) + o),
                                tuple(np.array((0.05, 0.10, 0.06)) + o),
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

    def rig_xyz(body) -> tuple[float, float, float]:
        _refresh()
        p = scene._rig_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def speed(body) -> float:
        _refresh()
        return float(body.data.root_lin_vel_w.norm(dim=-1)[0])

    def rig_pose(x: float, y: float, z: float) -> torch.Tensor:
        """(N,13) root state at rig-frame (x, y, z), rig-aligned quat, zero velocity."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x, y, z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.rig.data.root_pos_w + quat_apply(scene.rig.data.root_quat_w, loc)
        st[:, 3:7] = scene.rig.data.root_quat_w
        return st

    def report(tag: str) -> None:
        _refresh()
        px, py, pz = rig_xyz(scene.payload)
        bx, by, bz = rig_xyz(scene.blocker)
        print(f"[smoke] {tag:16s} | payload=({px:+.3f},{py:+.3f},{pz:+.3f}) "
              f"blocker=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"in_dock={bool(scene.payload_in_dock()[0])} "
              f"cleared={bool(scene.blocker_cleared()[0])} "
              f"latch=[c {int(scene._cleared[0])} j {int(scene._junction[0])} "
              f"m {int(scene._mouth[0])} d {int(scene._dock_ever[0])}] "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
              flush=True)

    def push_rig_axis(body, ax: float, ay: float, newtons: float, bouts: int,
                      stop) -> None:
        """Bouted CoM force along the rig-frame (ax, ay) axis (the same push style
        solve.py uses), cleared between bouts, stopped once stop() is True."""
        zero = torch.zeros(n, 1, 3, device=device)
        e = torch.tensor([ax, ay, 0.0], device=device).expand(n, 3)
        for _ in range(bouts):
            if stop():
                break
            f = quat_apply(scene.rig.data.root_quat_w, e).reshape(n, 1, 3) * newtons
            body.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                               is_global=True)
            _step(30)
            body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
            _step(15)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    report("settle")
    _REC["on"] = False
    px, py, pz = rig_xyz(scene.payload)
    bx, by, bz = rig_xyz(scene.blocker)
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all())
                 for b in (scene.payload, scene.blocker))
    check("settle/no-NaN: seeded reset settles finite; payload parked in the entry "
          "lane, blocker plugging the junction; score 0, no success",
          finite and px < -0.15 and abs(py) < 0.05
          and 0.02 < bx < 0.16 and abs(by) < 0.08
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return (yaw_of(scene.rig.data.root_quat_w[0]),
                scene.rig.data.root_pos_w[0, :2].clone(),
                rig_xyz(scene.payload)[0],
                scene.blocker.data.root_pos_w[0, :2].clone())

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_yawv = max(dyaw(obs[i][0], obs[j][0]) for i, j in pairs)
    d_rp = max(float((obs[i][1] - obs[j][1]).norm()) for i, j in pairs)
    d_px = max(abs(obs[i][2] - obs[j][2]) for i, j in pairs)
    d_bp = max(float((obs[i][3] - obs[j][3]).norm()) for i, j in pairs)
    print(f"[smoke] randomization max-pairwise deltas: rig_yaw={d_yawv:.1f}deg "
          f"rig_xy={d_rp * 1000:.1f}mm payload_slot_x={d_px * 1000:.1f}mm "
          f"blocker_xy={d_bp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rig yaw, rig xy, payload lane slot and blocker "
          "world position readback all differ across three seeded resets",
          d_yawv > 4.0 and d_rp > 0.003 and d_px > 0.006 and d_bp > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    px, _py, _pz = rig_xyz(scene.payload)
    check("null-policy-fails: 240 idle steps — nothing moves, payload still in the "
          "entry lane; score ~0, no success",
          px < -0.15 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. SEED STRATEGY: carry the payload and drop it in the dock ================
    # The seed's plan (grasp the target, carry it, release it inside the container),
    # CONSTRUCTED at its end state: the payload teleported straight into the dock
    # containment window (which the roof makes physically impossible from above). It
    # settles there — genuinely contained, genuinely at rest — and is REJECTED
    # because it never crossed the mouth: the pathway latch is load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.payload.write_root_state_to_sim(
        rig_pose(0.270, 0.0, c.deck_top + c.payload_h / 2 + 0.003), _all_ids())
    _step(240)
    report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy / bypass): payload delivered straight into the "
          "dock window and settled — real containment WITHOUT the mouth pathway — "
          "success refused, score <= 0.26",
          bool(scene.payload_in_dock()[0]) and speed(scene.payload) < c.settle_lin
          and not bool(scene._mouth[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.26)

    # ================= 5. cleared-only: blocker moved, payload untouched ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    scene.blocker.write_root_state_to_sim(
        rig_pose(0.090, 0.210, c.deck_top + c.blocker_h / 2 + 0.003), _all_ids())
    _step(180)
    report("cleared-only")
    check("cleared-only: blocker transported into the siding and left — exactly the "
          "clearing credit (~0.15), payload untouched in the lane, no success",
          bool(scene.blocker_cleared()[0])
          and 0.14 <= float(scene.score()[0]) <= 0.16
          and rig_xyz(scene.payload)[0] < -0.15
          and not bool(scene.success()[0]))

    # ================= 6. WRONG ORDER: push the payload with the blocker in place =================
    # Same push style as solve.py phase 2, but the junction is still plugged: the
    # payload advances to contact and only rams the blocker against the roof's proud
    # front edge, where it JAMS (135 mm tall vs the 112 mm underside). The payload
    # can never cross the junction band; no credit is earned.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    push_rig_axis(scene.payload, 1.0, 0.0, newtons=4.0, bouts=20,
                  stop=lambda: rig_xyz(scene.payload)[0] > 0.05)
    _step(120)
    report("wrong-order")
    _REC["on"] = False
    px, _py, _pz = rig_xyz(scene.payload)
    bx, _by, bz = rig_xyz(scene.blocker)
    b_rest = c.deck_top + c.blocker_h / 2
    check("negative (WRONG ORDER): payload pushed down the lane with the junction "
          "still plugged — it advances to contact (probe is live) but the blocker "
          "jams at the roof edge; junction band never crossed, score 0, no success",
          px > -0.12 and px < c.junction_band[0]
          and not bool(scene._junction[0]) and bx < 0.16
          and abs(bz - b_rest) < 0.010  # still at deck rest: never climbed the roof
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. mouth straddle: in the doorway is not in the dock =======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    scene.blocker.write_root_state_to_sim(
        rig_pose(0.090, 0.210, c.deck_top + c.blocker_h / 2 + 0.003), _all_ids())
    _step(60)
    scene.payload.write_root_state_to_sim(
        rig_pose(0.190, 0.0, c.deck_top + c.payload_h / 2 + 0.003), _all_ids())
    _step(180)
    report("mouth-straddle")
    px, _py, _pz = rig_xyz(scene.payload)
    check("near-miss (mouth straddle): payload settled in the mouth doorway, half "
          "in half out — mouth + clearing credit only (<= 0.36), centimetres from "
          "the window, not success",
          speed(scene.payload) < c.settle_lin and abs(px - 0.19) < 0.03
          and bool(scene._mouth[0]) and not bool(scene.payload_in_dock()[0])
          and float(scene.score()[0]) <= 0.36 and not bool(scene.success()[0]))

    # ================= 8. the blocker can never enter the dock ====================================
    # Blocker set on the open lane short of the mouth, then ground forward with a
    # hard push: it advances freely (probe is live), then hits the roof's proud
    # front edge and stops dead — the tall crate can never be pushed into the dock.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.blocker.write_root_state_to_sim(
        rig_pose(0.030, 0.0, c.deck_top + c.blocker_h / 2 + 0.003), _all_ids())
    _step(60)
    start_bx = rig_xyz(scene.blocker)[0]
    push_rig_axis(scene.blocker, 1.0, 0.0, newtons=6.0, bouts=14,
                  stop=lambda: rig_xyz(scene.blocker)[0] > 0.13)
    _step(120)
    report("blocker-grind")
    _REC["on"] = False
    bx, _by, bz = rig_xyz(scene.blocker)
    b_rest = c.deck_top + c.blocker_h / 2
    check("negative (blocker into the dock): a hard +x grind advances the blocker "
          "freely (moved > 30 mm, probe is live) then jams it at the roof front "
          "edge — it never reaches the mouth plane, never pops over the roof, "
          "score ~0, no success",
          bx - start_bx > 0.030 and bx < 0.13
          and abs(bz - b_rest) < 0.010  # still at deck rest: never climbed the roof
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 9. rejection audit =========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-8)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.crate_dock")
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
    except BaseException as exc:  # noqa: BLE001
        print(f"[smoke] EXCEPTION: {exc!r}", flush=True)
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
