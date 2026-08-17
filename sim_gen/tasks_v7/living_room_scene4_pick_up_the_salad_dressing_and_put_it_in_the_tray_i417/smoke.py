"""Smoke / rubric-REJECTION battery for DecantReturnScene — NullRobot, teleported probes.

solve.py is the acceptance proof (held-pose wrench pour of the marble out of the mouth
into the tray, then a drop-in seat of the empty bottle in the socket, score 1.0). This
battery proves the rubric REJECTS wrong outcomes and that the physical claims the task
rests on are load-bearing: the SEED's plan (put the bottle in the tray) is a dead end,
an end state IDENTICAL to success but reached by teleporting the marble out is refused
by the decant continuity latch, the tray walls really do refuse a ground-rolling
marble, and partial outcomes earn only their latched slice. Every probe is CONSTRUCTED
as a state (teleport, real physics steps, judge) — instrumentation, never a solution.

Checks:
   1. settle/no-NaN     — seeded reset settles finite; marble sealed inside the bottle
                          at the jittered start; score 0, no success;
   2. randomization     — three seeded resets, MAX-PAIRWISE readback deltas: rig yaw,
                          rig xy, bottle rig-local xy, bottle free yaw and the marble's
                          in-cavity offset all differ (fixture bearings must be read);
   3. null-policy       — 240 idle steps: marble stays sealed, bottle stays parked;
                          score ~0, no success;
   4. SEED STRATEGY     — the seed's goal verbatim: the bottle (marble still inside)
                          teleported INTO the tray and settled — a REAL contained
                          equilibrium, REJECTED: the marble never left the bottle
                          (in-bottle marble does not count as in-tray; score ~0);
   5. BYPASS (flagship) — END-STATE-IDENTICAL to success: marble teleported to rest in
                          the tray, bottle drop-seated in the socket. Every LIVE
                          predicate of success holds — refused because the decant was
                          never observed (the teleport-out trips the continuity guard);
                          only the in-tray slice is credited (<= 0.16);
   6. WALLS + aim-miss  — marble teleported to the ground and GROUND toward the tray
                          with a real push (probe is live, moved > 30 mm): it stops at
                          the outer wall, never enters; bottle seated; no success;
   7. socket near-miss  — everything else satisfied (marble in tray; decant latch
                          FORCED true as pure instrumentation) but the bottle standing
                          upright 60 mm off the socket: bottle_seated() refuses, no
                          success;
   8. never-poured seat — bottle WITH the marble still inside drop-seated in the
                          socket: seat credit is decant-gated, score ~0, no success;
   9. rejection audit   — success() never fired at any judged step during the negative
                          probes (checks 4-8);
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i417.smoke --headless
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
    env = ENVS.get("simgen.decant_return")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.90, 0.80)) + o),
                                tuple(np.array((0.05, 0.02, 0.08)) + o),
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
        bx, by, bz = rig_xyz(scene.bottle)
        mx, my, mz = rig_xyz(scene.marble)
        print(f"[smoke] {tag:16s} | bottle=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"marble=({mx:+.3f},{my:+.3f},{mz:+.3f}) "
              f"in_bottle={bool(scene.marble_in_bottle()[0])} "
              f"in_tray={bool(scene.marble_in_tray()[0])} "
              f"seated={bool(scene.bottle_seated()[0])} "
              f"latch=[l {int(scene._lifted[0])} h {int(scene._hover[0])} "
              f"d {int(scene._decant[0])} t {int(scene._tray_ever[0])} "
              f"s {int(scene._seated[0])}] "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
              flush=True)

    def drop_bottle_in_socket(with_marble: bool) -> None:
        """Transport the bottle to 20 mm above the socket plate (yaw-aligned) and let
        the drop-in registration seat it; optionally keep the marble inside."""
        st = rig_pose(c.sock_x, -c.tray_y, c.sock_plate + c.bottle_zmid + 0.020)
        scene.bottle.write_root_state_to_sim(st, _all_ids())
        if with_marble:
            loc = torch.tensor([0.0, 0.0, c.floor_t + c.marble_r + 0.004 - c.bottle_zmid],
                               device=device).expand(n, 3)
            stm = torch.zeros(n, 13, device=device)
            stm[:, 0:3] = st[:, 0:3] + quat_apply(st[:, 3:7], loc)
            stm[:, 3] = 1.0
            scene.marble.write_root_state_to_sim(stm, _all_ids())
        _step(150)

    def kick_marble_rig_y(speed_mps: float, kicks: int, stop) -> None:
        """Roll the ground marble at the tray by writing a horizontal velocity along
        rig +y. Velocity kicks are drag-proof: this pod rotates applied external
        FORCES by a body's rotation since a stale reference orientation (measured
        by the solve's kick probe), and a rolling ball accumulates arbitrary
        rotation, so a force push drifts off-line — a velocity write needs no force
        frame at all. The kick speed is capped below the wall-hop bound
        (sqrt(2 g dh) ~ 0.9 m/s for the ball CoM to top the 55 mm wall), so arrest
        at the wall is a genuine geometry verdict, and the check's moved-assert
        keeps the probe non-vacuous."""
        e = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        for _ in range(kicks):
            if stop():
                break
            _refresh()
            st = scene.marble.data.root_state_w.clone()
            st[:, 7:10] = quat_apply(scene.rig.data.root_quat_w, e) * speed_mps
            st[:, 10:13] = 0.0
            scene.marble.write_root_state_to_sim(st, _all_ids())
            _step(90)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    report("settle")
    _REC["on"] = False
    bx, by, _bz = rig_xyz(scene.bottle)
    finite = all(bool(torch.isfinite(b.data.root_pos_w).all())
                 for b in (scene.bottle, scene.marble))
    check("settle/no-NaN: seeded reset settles finite; marble sealed inside the "
          "bottle at the jittered start; score 0, no success",
          finite and bool(scene.marble_in_bottle()[0])
          and abs(bx - c.bottle_start[0]) < 0.08 and abs(by - c.bottle_start[1]) < 0.08
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        mloc = scene._bottle_local(scene.marble.data.root_pos_w)[0, :2].clone()
        return (yaw_of(scene.rig.data.root_quat_w[0]),
                scene.rig.data.root_pos_w[0, :2].clone(),
                torch.tensor(rig_xyz(scene.bottle)[:2]),
                yaw_of(scene.bottle.data.root_quat_w[0]),
                mloc)

    obs = []
    for sd in (101, 202, 303):
        torch.manual_seed(sd)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]
    d_ry = max(dyaw(obs[i][0], obs[j][0]) for i, j in pairs)
    d_rp = max(float((obs[i][1] - obs[j][1]).norm()) for i, j in pairs)
    d_bx = max(float((obs[i][2] - obs[j][2]).norm()) for i, j in pairs)
    d_by = max(dyaw(obs[i][3], obs[j][3]) for i, j in pairs)
    d_mm = max(float((obs[i][4] - obs[j][4]).norm()) for i, j in pairs)
    print(f"[smoke] randomization max-pairwise deltas: rig_yaw={d_ry:.1f}deg "
          f"rig_xy={d_rp * 1000:.1f}mm bottle_xy={d_bx * 1000:.1f}mm "
          f"bottle_yaw={d_by:.1f}deg marble_incav={d_mm * 1000:.1f}mm", flush=True)
    check("randomization-is-real: rig yaw, rig xy, bottle rig-local xy, bottle free "
          "yaw and marble in-cavity offset all differ across three seeded resets",
          d_ry > 3.0 and d_rp > 0.003 and d_bx > 0.006 and d_by > 5.0 and d_mm > 0.0015)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    report("null-policy")
    bx, by, _bz = rig_xyz(scene.bottle)
    check("null-policy-fails: 240 idle steps — marble stays sealed inside, bottle "
          "stays parked at the start; score ~0, no success",
          bool(scene.marble_in_bottle()[0]) and abs(bx - c.bottle_start[0]) < 0.08
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = True  # ---- negative probes below: success must never fire ----

    # ================= 4. SEED STRATEGY: put the bottle in the tray ===============================
    # The seed task's goal verbatim, CONSTRUCTED at its end state: the bottle (marble
    # still sealed inside) teleported into the tray and settled — a REAL contained
    # equilibrium. REJECTED: an in-bottle marble does not count as in-tray, no decant
    # ever happened, nothing is seated. Score ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    st = rig_pose(c.tray_x, c.tray_y, c.tray_floor + c.bottle_zmid + 0.003)
    scene.bottle.write_root_state_to_sim(st, _all_ids())
    loc = torch.tensor([0.0, 0.0, c.floor_t + c.marble_r + 0.004 - c.bottle_zmid],
                       device=device).expand(n, 3)
    stm = torch.zeros(n, 13, device=device)
    stm[:, 0:3] = st[:, 0:3] + quat_apply(st[:, 3:7], loc)
    stm[:, 3] = 1.0
    scene.marble.write_root_state_to_sim(stm, _all_ids())
    _step(240)
    report("seed-strategy")
    _REC["on"] = False
    bx, by, bz = rig_xyz(scene.bottle)
    in_tray_xy = abs(bx - c.tray_x) < c.tray_in / 2 and abs(by - c.tray_y) < c.tray_in / 2
    check("negative (SEED strategy): bottle with the marble inside delivered into "
          "the tray and settled — a real contained equilibrium — refused: in-bottle "
          "marble is not in-tray, no decant, score ~0, no success",
          in_tray_xy and bz < 0.12 and speed(scene.bottle) < c.settle_lin
          and bool(scene.marble_in_bottle()[0]) and not bool(scene.marble_in_tray()[0])
          and not bool(scene._decant[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. BYPASS flagship: end-state-identical, decant never observed =============
    # Marble teleported to rest in the tray, bottle drop-seated in the socket: every
    # LIVE predicate of success() holds (marble in tray + settled, bottle seated +
    # settled) — refused because the marble's exit was a teleport, which the decant
    # continuity guard (per-step travel bound) rejects. Only the in-tray slice credits.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    scene.marble.write_root_state_to_sim(
        rig_pose(c.tray_x, c.tray_y, c.tray_floor + c.marble_r + 0.003), _all_ids())
    _step(60)
    drop_bottle_in_socket(with_marble=False)
    _step(120)
    report("bypass")
    _REC["on"] = False
    check("negative (BYPASS flagship): end state IDENTICAL to success — marble at "
          "rest in the tray AND bottle seated in the socket — refused because the "
          "decant was never observed (teleport-out trips the continuity guard); "
          "score <= 0.16",
          bool(scene.marble_in_tray()[0]) and speed(scene.marble) < c.settle_lin
          and bool(scene.bottle_seated()[0]) and speed(scene.bottle) < c.settle_lin
          and not bool(scene._decant[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.16)

    # ================= 6. tray walls are load-bearing: ground marble cannot enter =================
    # Marble teleported to the ground outside the tray, bottle seated (everything
    # else right), then the marble is GROUND toward the tray with a real push: it
    # moves freely (probe is live), hits the outer wall and stops — a spilled marble
    # is lost. The 55 mm walls make AIMING the pour load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_bottle_in_socket(with_marble=False)
    scene.marble.write_root_state_to_sim(
        rig_pose(c.tray_x, -0.010, c.marble_r + 0.002), _all_ids())
    _step(60)
    my0 = rig_xyz(scene.marble)[1]
    wall_outer_y = c.tray_y - c.tray_in / 2 - c.tray_wall_t
    _REC["on"] = True
    kick_marble_rig_y(speed_mps=0.7, kicks=3,
                      stop=lambda: rig_xyz(scene.marble)[1] > wall_outer_y + 0.02)
    _step(90)
    report("wall-refusal")
    _REC["on"] = False
    mx, my, mz = rig_xyz(scene.marble)
    check("negative (WALLS + aim-miss): a ground marble rolled at the tray moves "
          "freely (> 30 mm, probe is live) but is arrested by the outer wall and "
          "never enters; marble lost, no success",
          my - my0 > 0.030 and not bool(scene.marble_in_tray()[0])
          and mz < 0.045 and my < c.tray_y - c.tray_in / 2 + 0.005
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)

    # ================= 7. socket near-miss: upright but 60 mm off ================================
    # Everything else satisfied — marble in the tray, decant latch FORCED true (pure
    # instrumentation, isolating the seat gate) — but the bottle stands upright on
    # the ground 60 mm from the socket: bottle_seated() refuses (xy tolerance 10 mm,
    # base z at plate height), so no success and no seat credit... the drop-in
    # registration is real.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    scene.marble.write_root_state_to_sim(
        rig_pose(c.tray_x, c.tray_y, c.tray_floor + c.marble_r + 0.003), _all_ids())
    scene.bottle.write_root_state_to_sim(
        rig_pose(c.sock_x, -c.tray_y + c.sock_in / 2 + c.sock_lip_t + 0.060,
                 c.bottle_zmid + 0.002), _all_ids())
    scene._decant[:] = True  # instrumentation: isolate the seat predicate
    _step(180)
    report("seat-near-miss")
    check("near-miss (socket): marble in tray and decant latch forced true, but the "
          "bottle stands upright 60 mm off the socket — bottle_seated() refuses, "
          "no success",
          bool(scene.marble_in_tray()[0]) and not bool(scene.bottle_seated()[0])
          and speed(scene.bottle) < c.settle_lin
          and not bool(scene.success()[0]))

    # ================= 8. never-poured: seat the still-loaded bottle ==============================
    # The bottle with the marble STILL INSIDE drop-seated in the socket: the seating
    # is genuine, but the marble is not in the tray and the seat credit is
    # decant-gated — score ~0, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_bottle_in_socket(with_marble=True)
    _step(120)
    report("never-poured")
    _REC["on"] = False
    check("negative (never poured): bottle with the marble still sealed inside "
          "drop-seated in the socket — genuine seat, but no decant and no marble in "
          "the tray: seat credit is decant-gated, score ~0, no success",
          bool(scene.bottle_seated()[0]) and bool(scene.marble_in_bottle()[0])
          and not bool(scene._seated[0]) and not bool(scene._decant[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    _AUDIT["on"] = False  # ---- end of negative probes ----

    # ================= 9. rejection audit =========================================================
    check("rejection audit: success() never fired at any step of the negative "
          "probes (checks 4-8)", not _AUDIT["fired"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.decant_return")
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
