"""Smoke / rubric-REJECTION battery for KeystoneCascadeScene (sim_gen task
`poke_cube_i83`) — NullRobot, teleported/flicked probe states, RECORDED.

This is NOT a solution (solve.py — place the keystone on the pad, calibrated poke,
hands-off cascade — is the acceptance evidence that the rubric ACCEPTS a correct
outcome). Every teleport/velocity-write here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it — plus physics probes
that prove the mechanism is real (the cascade genuinely relays from the first domino
all the way to the potted ball, and the guard geometry genuinely stops a short decoy
and an off-pad keystone). Force-style probes assert the actuator MOVED (a flicked
piece must actually topple), so no rejection is vacuous. No probe in this battery
ever reaches success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN      — reset settles finite: run + hammer stand, keystone lies
                             flat in its scatter slot, ball perched ABOVE the pit-z
                             gate; score ~0, no success;
  3-4.  randomization      — READBACK over 12 seeded resets: keystone slot shuffles;
                             alley anchor + yaw and scatter positions vary;
  5.   null policy         — 300 idle steps -> score ~0, run intact, no success;
  6.   seed strategy       — maniskill/poke_cube's move (poke the object a few cm
                             across the floor to a spot): keystone slid to an open
                             floor spot -> score ~0, nothing armed, no success;
  7.   decoy on the pad    — a SHORT domino stood on the pad and flicked downstream
                             genuinely topples yet falls SHORT of d1 (guard is real),
                             arms nothing, no success;
  8.   off-pad near miss   — the keystone stood 7.5 cm upstream (outside the armed
                             window) and flicked downstream genuinely topples yet
                             falls short of d1; no arm, no credit;
  9.   wand bypass         — d1 flicked directly (unreachable for a hand; pure
                             instrumentation): the WHOLE cascade runs to the potted
                             ball (mechanism is real) but armed stays False ->
                             partial credit only, no success;
  10.  out-of-order        — after that cascade, the keystone placed on the pad and
                             poked over: latch can no longer arm (run is down), no
                             success;
  11.  ball teleport       — ball dropped straight into the pit: in-pit credit only,
                             no arm, run intact, no success;
  12.  sideways poke       — honest arming (keystone settled on pad, run intact ->
                             armed latches), then the keystone flicked SIDEWAYS: it
                             topples on the open floor, run intact -> armed credit
                             only, no success;
  13.  fly-through no-arm  — the keystone written moving fast THROUGH the pad window
                             does not latch (settled-streak gate);
  14.  geometry audit      — alley-frame readback of the spawned layout matches the
                             guard margins (mouth < hand, d1 beyond fingertips,
                             decoy tip circle short of d1, keystone circle past d1's
                             top corner);
  15.  rejection audit     — success() never True at any judged point;
  16.  final no-NaN        — all task-object states finite.

Run (forge): python -u -m simgen_tasks.poke_cube_i83.smoke --headless
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_PAD_X = scene_mod._PAD_X
_MOUTH_X = scene_mod._MOUTH_X
_D1_X = scene_mod._D1_X
_T = scene_mod._T
_H = scene_mod._H
_KT = scene_mod._KT
_KH = scene_mod._KH
_SDIAG = scene_mod._SDIAG
_KDIAG = scene_mod._KDIAG
_PERCH_X = scene_mod._PERCH_X
_PERCH_H = scene_mod._PERCH_H
_BALL_R = scene_mod._BALL_R

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.keystone_cascade")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.55, -1.15, 0.95)) + o),
                                tuple(np.array((0.60, 0.00, 0.10)) + o),
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

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def upz(body) -> float:
        return float(scene.up_z(body)[0])

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:18s} | armed={bool(scene.armed[0])} "
              f"key_upz={upz(scene.keystone):+.3f} "
              f"run_frac={float(scene.run_toppled_frac()[0]):.2f} "
              f"ball_pit={bool(scene.ball_in_pit()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def alley_pose(local, yaw_extra: float = 0.0, flat: bool = False):
        """(pos, quat) world pose for an alley-frame `local` position; quat = alley
        yaw (+optional extra yaw), standing or lying-flat (long axis horizontal)."""
        pos = scene.env_origins.clone()
        pos[:, 0:2] += scene.anchor
        pos += scene.alley_dir([local[0], local[1], 0.0])
        pos[:, 2] = local[2]
        yaw = scene.yaw + yaw_extra
        ch, sh = torch.cos(yaw / 2), torch.sin(yaw / 2)
        z = torch.zeros_like(ch)
        if flat:
            c45 = math.sqrt(0.5)
            quat = torch.stack([ch * c45, -sh * c45, ch * c45, sh * c45], dim=-1)
        else:
            quat = torch.stack([ch, z, z, sh], dim=-1)
        return pos, quat

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = vel
        if ang is not None:
            st[:, 10:13] = ang
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    def flick(body, axis_alley, omega: float, settle_steps: int = 60) -> None:
        """Deterministic probe actuator: keep the body's pose, write an angular
        velocity about an alley-frame axis (a 'wand tap'), then let physics run."""
        pos = body.data.root_pos_w.clone()
        quat = body.data.root_quat_w.clone()
        ang = scene.alley_dir(axis_alley) * omega
        teleport(body, pos, quat, ang=ang, settle_steps=settle_steps)

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            vmax = max(float(b.data.root_lin_vel_w[0].norm())
                       for b in [*scene.run, scene.keystone, *scene.decoys, scene.ball])
            if vmax < 0.03:
                break

    def all_finite() -> bool:
        bodies = [scene.arcade, *scene.run, scene.keystone, *scene.decoys, scene.ball]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    def run_standing() -> bool:
        return bool(scene.run_standing()[0])

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(360)
    report("reset")
    ball_loc = scene.to_alley(scene.ball.data.root_pos_w)[0]
    key_flat = abs(upz(scene.keystone)) < 0.3
    check("settle: all states finite, run + hammer stand intact, keystone lies flat "
          f"in its slot (up_z={upz(scene.keystone):+.2f})",
          all_finite() and run_standing() and key_flat)
    s, ok = judge()
    check("settle: ball perched ABOVE the pit-z gate (alley z="
          f"{float(ball_loc[2]):.3f} > {c.pit_z}), score ~0, no success",
          float(ball_loc[2]) > c.pit_z + 0.01 and not bool(scene.ball_in_pit()[0])
          and s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in range(21, 33):
        env.reset(seed=sd)
        step(10)
        loc = scene.to_alley(scene.keystone.data.root_pos_w)[0]
        reads.append((int(scene.key_slot[0]), float(scene.anchor[0, 0]),
                      float(scene.anchor[0, 1]), float(scene.yaw[0]),
                      float(loc[0]), float(loc[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (key_slot, anchor_x, anchor_y, yaw, "
          f"key_alley_x, key_alley_y):\n{arr.round(3)}", flush=True)
    slots = {int(r[0]) for r in reads}
    spread = arr.max(axis=0) - arr.min(axis=0)
    check(f"randomization: the keystone's scatter slot shuffles (readback: {len(slots)} "
          "distinct slots)", len(slots) >= 2)
    check("randomization: alley anchor + yaw and keystone scatter vary (readback: "
          f"dx={spread[1]:.3f} dyaw={spread[3]:.2f} dkey={spread[4]:.3f})",
          spread[1] > 0.01 and spread[3] > 0.15 and spread[4] > 0.03)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    report("null-policy")
    s, ok = judge()
    check("null policy: 300 idle steps -> run intact, latch clear, score ~0, no success",
          run_standing() and not bool(scene.armed[0]) and s <= 0.02 and not ok)

    # =========================== 6. seed strategy (poke across the floor) ===================
    # maniskill/poke_cube's move: push the poked object a few cm along the floor to a
    # goal spot. There is no floor goal here: slide the keystone (still flat) to an
    # open floor spot -> nothing arms, nothing counts.
    env.reset(seed=41)
    step(30)
    pos, quat = alley_pose([0.30, -0.10, _KT / 2 + 0.003], flat=True)
    teleport(scene.keystone, pos, quat, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (poke the keystone a few cm across the floor to a spot): "
          "score ~0, latch clear, run intact, no success",
          s <= 0.02 and not bool(scene.armed[0]) and run_standing() and not ok)

    # =========================== 7. decoy on the pad cannot bridge ==========================
    env.reset(seed=51)
    settle_all(240)
    pos, quat = alley_pose([_PAD_X, 0.0, _H / 2 + 0.004])
    teleport(scene.decoys[0], pos, quat, settle_steps=60)
    armed_before = bool(scene.armed[0])
    # 16 rad/s: the CoM-spin -> edge-pivot collision keeps only ~omega/4, and the
    # SHORT 80 mm piece needs ~12 rad/s pre-collision to clear its balance angle.
    flick(scene.decoys[0], [0.0, 1.0, 0.0], 16.0, settle_steps=180)
    report("decoy-on-pad")
    s, ok = judge()
    d1_up = upz(scene.run[0])
    check("decoy on the pad: a SHORT domino stood on the pad and flicked downstream "
          f"genuinely topples (up_z={upz(scene.decoys[0]):+.2f}) yet d1 still stands "
          f"(up_z={d1_up:+.2f}) — too short to bridge; no arm, no credit, no success",
          upz(scene.decoys[0]) < c.topple_upz and d1_up > c.stand_upz and run_standing()
          and not armed_before and not bool(scene.armed[0]) and s <= 0.02 and not ok)

    # =========================== 8. off-pad near miss =======================================
    env.reset(seed=61)
    settle_all(240)
    pos, quat = alley_pose([_PAD_X - 0.075, 0.0, _KH / 2 + 0.004])
    teleport(scene.keystone, pos, quat, settle_steps=60)
    armed_mid = bool(scene.armed[0])
    flick(scene.keystone, [0.0, 1.0, 0.0], 8.0, settle_steps=180)
    report("near-miss")
    s, ok = judge()
    check("off-pad near miss: the keystone stood 7.5 cm upstream (outside the armed "
          f"window) and flicked downstream topples (up_z={upz(scene.keystone):+.2f}) "
          f"yet d1 still stands (up_z={upz(scene.run[0]):+.2f}); no arm, no credit",
          upz(scene.keystone) < c.topple_upz and run_standing() and not armed_mid
          and not bool(scene.armed[0]) and s <= 0.02 and not ok)

    # =========================== 9. wand bypass proves the mechanism ========================
    env.reset(seed=71)
    settle_all(240)
    flick(scene.run[0], [0.0, 1.0, 0.0], 16.0, settle_steps=0)  # short piece: see check 7
    for _ in range(40):  # up to 10 s: let the cascade relay to the potted ball
        step(30)
        if float(scene.run_toppled_frac()[0]) >= 1.0 and bool(scene.ball_in_pit()[0]) \
                and bool(scene.ball_settled()[0]):
            break
    report("wand-bypass")
    s, ok = judge()
    check("wand bypass (d1 flicked directly — unreachable for a hand): the cascade "
          f"genuinely relays to the end (run_frac={float(scene.run_toppled_frac()[0]):.2f}, "
          f"ball potted={bool(scene.ball_in_pit()[0])}) — mechanism is REAL — but the "
          "latch never armed: partial credit only, no success",
          float(scene.run_toppled_frac()[0]) >= 1.0 and bool(scene.ball_in_pit()[0])
          and bool(scene.ball_settled()[0]) and not bool(scene.armed[0])
          and s <= 0.42 and not ok)

    # =========================== 10. out-of-order ===========================================
    pos, quat = alley_pose([_PAD_X, 0.0, _KH / 2 + 0.004])
    teleport(scene.keystone, pos, quat, settle_steps=90)
    armed_after_cascade = bool(scene.armed[0])
    flick(scene.keystone, [0.0, 1.0, 0.0], 8.0, settle_steps=180)
    report("out-of-order")
    s, ok = judge()
    check("out-of-order: cascade first, keystone placed + poked afterwards — the "
          "latch can no longer arm (the run was already down), no success",
          not armed_after_cascade and not bool(scene.armed[0])
          and upz(scene.keystone) < c.topple_upz and s <= 0.42 and not ok)

    # =========================== 11. ball teleport bypass ===================================
    env.reset(seed=81)
    settle_all(240)
    pos, quat = alley_pose([1.10, 0.0, _BALL_R + 0.002])
    teleport(scene.ball, pos, quat, settle_steps=90)
    report("ball-teleport")
    s, ok = judge()
    check("ball teleport bypass: the ball dropped straight onto the pit floor counts "
          f"only its own term (ball_pit={bool(scene.ball_in_pit()[0])}, score={s:.2f} "
          "<= 0.12); run intact, no arm, no success",
          bool(scene.ball_in_pit()[0]) and run_standing() and not bool(scene.armed[0])
          and s <= 0.12 and not ok)

    # =========================== 12. sideways poke (armed but no cascade) ===================
    env.reset(seed=91)
    settle_all(240)
    pos, quat = alley_pose([_PAD_X, 0.0, _KH / 2 + 0.004])
    teleport(scene.keystone, pos, quat, settle_steps=0)
    armed_now = False
    for _ in range(16):
        step(15)
        if bool(scene.armed[0]):
            armed_now = True
            break
    # Sideways the keystone tips about its WIDE 40 mm face (higher balance angle):
    # needs ~11 rad/s pre-collision, so 16 (the downstream tip about the 22 mm face
    # clears at 8 — checks 8 and 10 keep that).
    flick(scene.keystone, [1.0, 0.0, 0.0], 16.0, settle_steps=180)
    report("sideways-poke")
    s, ok = judge()
    check("sideways poke: honest arming latches (keystone settled on the pad, run "
          f"intact -> armed={armed_now}), but a sideways topple "
          f"(up_z={upz(scene.keystone):+.2f}) starts nothing: run intact, armed "
          f"credit only (score={s:.2f} in [0.20, 0.27]), no success",
          armed_now and upz(scene.keystone) < c.topple_upz and run_standing()
          and 0.20 <= s <= 0.27 and not ok)

    # =========================== 13. fly-through cannot arm =================================
    env.reset(seed=101)
    settle_all(240)
    pos, quat = alley_pose([_PAD_X - 0.02, 0.0, _KH / 2 + 0.010])
    vel = scene.alley_dir([1.0, 0.0, 0.0]) * 0.8
    teleport(scene.keystone, pos, quat, vel=vel, settle_steps=0)
    in_win = False
    for _ in range(10):
        step(1)
        loc = scene.to_alley(scene.keystone.data.root_pos_w)[0]
        in_win = in_win or (c.pad_win_x[0] <= float(loc[0]) <= c.pad_win_x[1]
                            and abs(float(loc[1])) <= c.pad_win_y)
    step(120)
    report("fly-through")
    s, ok = judge()
    check("fly-through: the keystone written moving fast THROUGH the pad window "
          f"(window transited={in_win}) does not latch — the settled-streak gate "
          "holds; no success",
          in_win and not bool(scene.armed[0]) and not ok)

    # =========================== 14. geometry audit =========================================
    env.reset(seed=111)
    step(20)
    d1 = float(scene.to_alley(scene.run[0].data.root_pos_w)[0, 0])
    ham = float(scene.to_alley(scene.run[6].data.root_pos_w)[0, 0])
    ball_loc = scene.to_alley(scene.ball.data.root_pos_w)[0]
    depth = (d1 - _T / 2) - _MOUTH_X
    decoy_reach = _PAD_X + _T / 2 + _SDIAG
    key_corner = math.hypot((d1 - _T / 2) - (_PAD_X + _KT / 2), _H)
    print(f"[smoke] geometry readback: d1_x={d1:.3f} ham_x={ham:.3f} "
          f"ball=({float(ball_loc[0]):.3f},{float(ball_loc[1]):.3f},"
          f"{float(ball_loc[2]):.3f}) depth={depth:.3f} decoy_reach={decoy_reach:.3f} "
          f"key_corner={key_corner:.3f} (KDIAG={_KDIAG:.3f})", flush=True)
    check("geometry audit (alley readback): mouth 58 mm < hand 63 mm; d1 stands "
          f"{depth * 1000:.0f} mm behind the mouth (> 54 mm fingertips); a decoy's tip "
          f"circle ({decoy_reach:.3f}) falls short of d1's face ({d1 - _T / 2:.3f}); "
          f"the keystone's circle ({_KDIAG:.3f}) covers d1's top corner "
          f"({key_corner:.3f}); ball perched at its design pose",
          depth > 0.054 + 0.008 and decoy_reach + 0.006 < d1 - _T / 2
          and key_corner <= _KDIAG - 0.008
          and abs(float(ball_loc[0]) - _PERCH_X) < 0.01
          and abs(float(ball_loc[2]) - (_PERCH_H + _BALL_R)) < 0.008)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.keystone_cascade")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
