"""Smoke / oracle test for BoardLeanScene (sim_gen task `pick_single_ycb_i53`) —
NullRobot, teleport-oracle, RECORDED.

Battery (compass_crate / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, board flat, at rest, score 0;
  2. randomization      — READBACK: rack yaw/position and board spawn pose all move
                          across seeded resets; board always spawns OUTSIDE the lane
                          (staged latch never fires at reset);
  3. null-policy-fails  — 240 idle steps -> score ~0, no success;
  4-6. oracle x3 seeds  — teleport-place a 62 deg lean against the wall, release, real
                          physics settles it; success() and score 1.0 on 3 seeds;
  7-8. monotonicity     — ladder 0 (null) -> 0.2 (flat staged in lane) -> 0.6 (leaned
                          then knocked back flat: propped latch persists) -> 1.0
                          (re-leaned): strictly increasing, final is success;
  9-10. negative A      — the SEED's own strategy (hoist the object and hold it up):
                          board held hovering scores ~0 with no latch; releasing it
                          drops a flat board -> still no success;
 11. negative B         — the pick-and-place instinct: board laid FLAT at the target,
                          end touching the wall base, settled -> staging credit only;
 12. negative C         — near-vertical park (86 deg, overhang < 40 mm): physically
                          stable but NOT a lean -> no success, no propped latch;
 13. near-miss physics  — 25 deg lean is BELOW the slip angle: the board slides out and
                          falls flat on its own -> no success;
 14-15. calibration     — lean-angle sweep 25/40/55/70 deg -> hold/slip table; the
                          25 deg probe slips, 55 and 70 hold as success; hold-set is
                          monotone (upward-closed) in angle;
 16. state roundtrip    — get_state/set_state restores a success lean after the board
                          was teleported away.

Run (forge): python -u -m simgen_tasks.pick_single_ycb_i53.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from simgen_tasks.pick_single_ycb_i53 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

_G_DT = 9.81 / 120.0  # gravity-compensation vz for kinematic holds (state idx 9)


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _rack_pose(scene):
    """(rack_pos_w (N,3), rack_quat_w (N,4))."""
    return scene.rack.data.root_pos_w, scene.rack.data.root_quat_w


def _q_pitch(alpha: float, n: int, device) -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = math.cos(alpha / 2)
    q[:, 2] = math.sin(alpha / 2)
    return q


def _lean_state(scene, theta_deg: float, lat: float = 0.0,
                gap: float = 0.002) -> torch.Tensor:
    """Root state (N,13) for a lean at `theta_deg` against the wall: top edge `gap` off
    the wall face plane, bottom edge `gap` above the floor, zero velocities. Rack-frame
    geometry: board axis (bottom->top) = (-cos th, 0, sin th); board local +z -> that
    via a pitch of (th - 90 deg) about y, composed under the rack yaw."""
    from isaaclab.utils.math import quat_apply, quat_mul

    c = scene.cfg
    env = scene.env
    n = env.num_envs
    th = math.radians(theta_deg)
    rp, rq = _rack_pose(scene)
    cx = c.wall_t / 2 + gap + (c.board_l / 2) * math.cos(th) + (c.board_t / 2) * math.sin(th)
    cz_world = (c.board_l / 2) * math.sin(th) + (c.board_t / 2) * math.cos(th) + gap
    loc = torch.zeros(n, 3, device=env.device)
    loc[:, 0] = cx
    loc[:, 1] = lat
    loc[:, 2] = cz_world - c.wall_h / 2  # rack root sits at z = wall_h/2
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = rp + quat_apply(rq, loc)
    st[:, 3:7] = quat_mul(rq, _q_pitch(th - math.pi / 2, n, env.device))
    return st


def _flat_state(scene, center_x_from_face: float, lat: float = 0.0) -> torch.Tensor:
    """Root state (N,13) for the board lying FLAT, long axis along rack +x, center at
    `center_x_from_face` in front of the wall face plane, zero velocities."""
    from isaaclab.utils.math import quat_apply, quat_mul

    c = scene.cfg
    env = scene.env
    n = env.num_envs
    rp, rq = _rack_pose(scene)
    loc = torch.zeros(n, 3, device=env.device)
    loc[:, 0] = c.wall_t / 2 + center_x_from_face
    loc[:, 1] = lat
    loc[:, 2] = (c.board_t / 2 + 0.003) - c.wall_h / 2
    st = torch.zeros(n, 13, device=env.device)
    st[:, 0:3] = rp + quat_apply(rq, loc)
    st[:, 3:7] = quat_mul(rq, _q_pitch(-math.pi / 2, n, env.device))
    return st


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle: solve the CURRENT episode. Reads the (randomized) rack pose back
    from sim, places the board as a 62 deg lean with a 2 mm standoff, releases, and lets
    real physics carry it into wall contact and settle. One retry at 58 deg if the first
    release did not judge success. Returns True iff scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    all_ids = torch.arange(env.num_envs, device=env.device)
    no_action = torch.empty(0, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def settle(max_steps: int = 300) -> None:
        _step(20)
        waited = 20
        while waited < max_steps and not bool(scene.settled()[0]):
            _step(10)
            waited += 10

    for attempt, th in enumerate((62.0, 58.0)):
        scene.board.write_root_state_to_sim(_lean_state(scene, th), all_ids)
        settle()
        if verbose:
            print(f"[oracle] pass {attempt} (th={th:.0f}): theta={float(scene.theta_deg()[0]):.1f} "
                  f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])}",
                  flush=True)
        if bool(scene.success()[0]):
            return True
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.board_lean")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.55, -1.35, 0.95)) + o),
                                tuple(np.array((0.25, 0.0, 0.15)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def settle_until(pred, max_steps: int = 400, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def theta() -> float:
        return float(scene.theta_deg()[0])

    def report(tag: str) -> None:
        bot_loc, top_loc = scene._ends_rack_frame()
        print(f"[smoke] {tag:16s} | theta={theta():5.1f}deg "
              f"bot_x={float(bot_loc[0, 0]) - c.wall_t / 2:+.3f} "
              f"top_x={float(top_loc[0, 0]) - c.wall_t / 2:+.3f} "
              f"staged={bool(scene.staged[0])} propped={bool(scene.propped[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    print(f"[smoke] describe():\n{scene.describe()}", flush=True)

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st0 = scene.board.data.root_state_w
    check("settle: board state finite, flat, at rest, score 0, no success",
          bool(torch.isfinite(st0).all()) and theta() < 5.0
          and bool(scene.settled()[0]) and float(scene.score()[0]) <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(6)
        rp, rq = _rack_pose(scene)
        rloc = (rp - scene.env_origins)[0]
        yaw = 2.0 * math.atan2(float(rq[0, 3]), float(rq[0, 0]))
        bloc = (scene.board.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(rloc[0]), float(rloc[1]), yaw, float(bloc[0]), float(bloc[1]),
                      bool(scene.staged[0])))
    arr = np.array([r[:5] for r in reads])
    print("[smoke] randomization readback (rack_x, rack_y, rack_yaw, board_x, board_y):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rack pose (xy+yaw) and board spawn all move (readback); "
          "board never spawns staged",
          (spread[0] > 0.01 or spread[1] > 0.01) and spread[2] > 0.05
          and (spread[3] > 0.03 or spread[4] > 0.03)
          and not any(r[5] for r in reads))

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4-6. oracle on 3 seeds =====================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 7-8. rubric monotonicity ===================================
    # Ladder: null 0 -> flat staged in the lane 0.2 -> full lean achieved then knocked back
    # flat (propped latch persists) 0.6 -> re-leaned 1.0. Strictly increasing.
    torch.manual_seed(41)
    env.reset()
    step(20)
    # NOTE: every teleport is followed by FORCED physics steps before judging — a freshly
    # written state reads back settled (zero velocities) with zero steps run, and the
    # post_step latches only fire during real substeps (the vault_unstack lesson).
    scores = [round(float(scene.score()[0]), 3)]
    scene.board.write_root_state_to_sim(_flat_state(scene, 0.24 - c.wall_t / 2), all_ids)
    step(30)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    report("ladder-staged")
    scores.append(round(float(scene.score()[0]), 3))
    scene.board.write_root_state_to_sim(_lean_state(scene, 62.0), all_ids)
    step(30)
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    report("ladder-leaned")
    scene.board.write_root_state_to_sim(_flat_state(scene, 0.24 - c.wall_t / 2), all_ids)
    step(30)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    report("ladder-knocked")
    scores.append(round(float(scene.score()[0]), 3))
    scene.board.write_root_state_to_sim(_lean_state(scene, 62.0), all_ids)
    step(30)
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    report("ladder-releaned")
    scores.append(round(float(scene.score()[0]), 3))
    print(f"[smoke] monotonicity ladder [null, staged, propped-lost, releaned] -> {scores}",
          flush=True)
    check("monotonicity: score strictly increases across the ladder (0, 0.2, 0.6, 1.0)",
          all(b > a for a, b in zip(scores, scores[1:]))
          and scores[0] <= 0.005 and abs(scores[1] - 0.2) < 0.01
          and abs(scores[2] - 0.6) < 0.01)
    check("monotonicity: ladder final state is success at exactly 1.0",
          bool(scene.success()[0]) and scores[-1] == 1.0)

    # =========================== 9-10. negative A: the seed's own strategy ==================
    # maniskill/pick_single_ycb lifts the object 7.5 cm and HOLDS it (its checker is a +z
    # position shift; the dense variant holds it at a goal, grasped). Here: a held,
    # hovering board earns nothing — and simply releasing it earns no lean either.
    torch.manual_seed(51)
    env.reset()
    step(20)
    hold = _flat_state(scene, 0.30, lat=0.0)  # ends 0.09-0.51 from the face: clear of the wall
    hold[:, 2] += 0.25 - (c.board_t / 2 + 0.003)  # hover 25 cm up, over the lane
    hold[:, 9] = _G_DT  # gravity-compensated kinematic hold
    for _ in range(60):
        scene.board.write_root_state_to_sim(hold, all_ids)
        step(1)
    report("hoist-held")
    check("negative A (seed strategy): board hoisted and held aloft scores ~0, "
          "no latch, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.staged[0])
          and not bool(scene.success()[0]))
    step(60)  # real release: let the board actually fall before judging
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("hoist-released")
    check("negative A: releasing the hoisted board -> flat drop, still no success, "
          "at most staging credit",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.21)

    # =========================== 11. negative B: flat at the target =========================
    # The generic pick-and-place instinct: set the object down AT the goal. A flat board
    # in the lane with its end touching the wall base is staging credit only.
    torch.manual_seed(61)
    env.reset()
    step(20)
    scene.board.write_root_state_to_sim(
        _flat_state(scene, 0.02 + c.board_l / 2), all_ids)
    step(30)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=200)
    report("flat-at-wall")
    check("negative B: board laid FLAT against the wall base (in lane, touching) is "
          "staging credit only (0.2), no success",
          not bool(scene.success()[0]) and not bool(scene.propped[0])
          and 0.15 <= float(scene.score()[0]) <= 0.25)

    # =========================== 12. negative C: near-vertical park ==========================
    # Parking the board ~upright against the wall (86 deg: overhang 29 mm < 40 mm and
    # theta > 80 deg) is physically stable but is NOT a lean.
    torch.manual_seed(71)
    env.reset()
    step(20)
    scene.board.write_root_state_to_sim(_lean_state(scene, 86.0), all_ids)
    step(40)  # real physics must validate the park's stability
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    report("vertical-park")
    check("negative C: near-vertical park (86 deg) is stable but rejected — no success, "
          "no propped latch, score <= 0.25",
          theta() > 80.0 and bool(scene.settled()[0])
          and not bool(scene.success()[0]) and not bool(scene.propped[0])
          and float(scene.score()[0]) <= 0.25)

    # =========================== 13. near-miss: shallow lean slips ==========================
    torch.manual_seed(81)
    env.reset()
    step(20)
    scene.board.write_root_state_to_sim(_lean_state(scene, 25.0), all_ids)
    step(60)
    settle_until(lambda: bool(scene.settled()[0]), max_steps=500)
    report("shallow-25deg")
    check("near-miss (physics): 25 deg lean is below the slip angle — the board slides "
          "out and falls flat, no success",
          theta() < 30.0 and not bool(scene.success()[0])
          and not bool(scene.propped[0]))

    # =========================== 14-15. calibration probe ===================================
    print("[smoke] CALIBRATION: lean angle -> settled theta / hold / score "
          "(fresh reset each; analytic slip cliff ~38 deg for mu_f 0.6, mu_w 0.1)",
          flush=True)
    cal = []
    for tgt in (25.0, 40.0, 55.0, 70.0):
        torch.manual_seed(91)
        env.reset()
        step(10)
        scene.board.write_root_state_to_sim(_lean_state(scene, tgt), all_ids)
        step(60)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=500)
        th_f, su, sc = theta(), bool(scene.success()[0]), float(scene.score()[0])
        cal.append((tgt, th_f, su, sc))
        print(f"[smoke]   target={tgt:5.1f}deg settled={th_f:5.1f}deg "
              f"hold={su} score={sc:.3f}", flush=True)
    check("calibration: slip cliff — 25 deg slips flat, 55 and 70 deg hold as success",
          (cal[0][1] < 30.0 and not cal[0][2]) and cal[2][2] and cal[3][2])
    holds = [1 if su else 0 for _t, _th, su, _s in cal]
    check("calibration: hold-set is monotone (upward-closed) in lean angle",
          all(b >= a for a, b in zip(holds, holds[1:])))

    # =========================== 16. state roundtrip ========================================
    torch.manual_seed(101)
    env.reset()
    step(20)
    ok = oracle_solution(env, step_fn=step, verbose=False)
    saved = scene.get_state(all_ids)
    scene.board.write_root_state_to_sim(_flat_state(scene, 0.55), all_ids)
    step(30)
    away = not bool(scene.success()[0])
    scene.set_state(saved, all_ids)
    step(4)
    restored = settle_until(lambda: bool(scene.success()[0]), max_steps=100)
    report("state-roundtrip")
    check("state roundtrip: success lean saved, board moved away, set_state restores "
          "a success state", ok and away and restored)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.board_lean")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
