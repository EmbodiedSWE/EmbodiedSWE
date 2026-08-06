"""Smoke / oracle test for VaultUnstackScene (sim_gen task `stack_pyramid_i17`) —
NullRobot, teleport-oracle, RECORDED.

Battery (pen_holder / tunnel_shuffle smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite; score exactly 0 at rest; the
                          tower stands and covers the well;
  2. reset sanity       — prize inside the well, uncovered latch clear;
  3. randomization      — READBACK: vault, tray (pose+yaw) and pad all move across
                          seeded resets;
  4. null-policy-fails  — 240 idle steps -> score ~0, no success, well still covered;
  5-7. oracle x3 seeds  — top-down un-stack into tray slots + prize to pad reaches
                          success() and score exactly 1.0 on 3 seeds;
  8-9. monotonicity     — staged ladder (top->tray, mid->tray, cap->tray, prize out on
                          ground, prize->pad): score strictly climbs the declared
                          milestones [0.15, 0.30, 0.55, 0.70, 1.0]; every partial < 1;
 10. negative A (seed)  — the seed's own strategy: re-build the stack on open ground
                          (executed perfectly) scores <= 0.15, no success;
 11. negative B (dump)  — all blocks dumped on the ground BESIDE the tray + prize on
                          the pad: no block counts, score ~0.5, no success (tidy
                          storage is load-bearing);
 12. negative C (stack) — a block stacked ON another block INSIDE the tray does not
                          count (rest-on-tray-floor gate);
 13-14. near-miss       — folded into the ladder: prize beside the pad = 0.70 and no
                          success; final placement recovers to success;
 15-16. calibration     — occlusion probe: a 1.5 m/s vertical kick cannot raise the
                          prize past the rim while the tower stands (measured max
                          height), and the SAME kick clears the rim once the blocks
                          are removed — the cover is the physical blocker.

Run (forge): python -u -m simgen_tasks.stack_pyramid_i17.smoke --headless
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
    from simgen_tasks.stack_pyramid_i17 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # standalone fallback (run from inside the task dir)
    import scene as scene_mod  # noqa: F401

BLOCK_NAMES = ("cap", "mid", "top")


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _tray_slots(scene) -> dict[str, float]:
    """Tray-frame x for each block's storage slot (side-by-side on the tray floor)."""
    return {"cap": -0.085, "mid": 0.010, "top": 0.095}


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle for the CURRENT episode: un-stack the tower TOP-DOWN — each block
    is set down 10 mm above its tray-floor slot (aligned with the tray's yaw) and dropped;
    the landing, resting and settling are real physics — then the prize is lifted out of
    the open well and dropped onto the delivery pad. Returns True iff scene.success()."""
    from isaaclab.utils.math import quat_apply

    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    dev = env.device
    no_action = torch.empty(0, device=dev)
    all_ids = torch.arange(env.num_envs, device=dev)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def until(pred, max_steps: int = 400, poll: int = 10) -> bool:
        # ALWAYS step before the first poll: a freshly-teleported body reads zero
        # velocity, so settle-gated predicates would otherwise pass with the body still
        # airborne and zero physics steps run (run-1 failure mode).
        _step(15)
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return False

    slots = _tray_slots(scene)
    halves = dict(zip(BLOCK_NAMES, c.halves))
    order = ("top", "mid", "cap")  # top-down: the physical removal order
    for i, name in enumerate(order):
        tq = scene.tray.data.root_quat_w
        tp = scene.tray.data.root_pos_w
        loc = torch.zeros(env.num_envs, 3, device=dev)
        loc[:, 0] = slots[name]
        # 15 mm of air: outside the 12 mm rest-on-floor band, so `stored` can only turn
        # true after the block has REALLY landed and settled.
        loc[:, 2] = c.tray_floor_top + halves[name] + 0.015
        st = torch.zeros(env.num_envs, 13, device=dev)
        st[:, 0:3] = tp + quat_apply(tq, loc)
        st[:, 3:7] = tq  # align the block with the tray so the slot margins hold
        scene.block_bodies[name].write_root_state_to_sim(st, all_ids)
        idx = BLOCK_NAMES.index(name)
        ok = until(lambda idx=idx: bool(scene.stored()[0, idx]))
        if verbose:
            print(f"[oracle] store {name}: stored={ok} "
                  f"score={float(scene.score()[0]):.3f}", flush=True)
        if not ok:
            return False

    gp = scene.pad.data.root_pos_w
    st = torch.zeros(env.num_envs, 13, device=dev)
    st[:, 0:3] = gp
    st[:, 2] += c.pad_t / 2 + c.prize_size / 2 + 0.015  # outside pad_z_tol until landed
    st[:, 3] = 1.0
    scene.prize.write_root_state_to_sim(st, all_ids)
    ok = until(lambda: bool(scene.success()[0]))
    if verbose:
        print(f"[oracle] deliver prize: success={ok} "
              f"score={float(scene.score()[0]):.3f}", flush=True)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.vault_unstack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    halves = dict(zip(BLOCK_NAMES, c.halves))

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.25, 0.95)) + o),
                                tuple(np.array((0.10, 0.0, 0.10)) + o),
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

    def report(tag: str) -> None:
        stored = scene.stored()[0].int().tolist()
        print(f"[smoke] {tag:14s} | stored={stored} covered={bool(scene.covered()[0])} "
              f"uncovered_latch={bool(scene.uncovered[0])} "
              f"in_well={bool(scene.prize_in_well()[0])} "
              f"on_pad={bool(scene.prize_on_pad()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_block(name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        scene.block_bodies[name].write_root_state_to_sim(st, all_ids)

    def place_prize(pos) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3] = 1.0
        scene.prize.write_root_state_to_sim(st, all_ids)

    def block_to_tray_slot(name: str, x_loc: float, z_extra: float = 0.015) -> None:
        tq = scene.tray.data.root_quat_w
        tp = scene.tray.data.root_pos_w
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = x_loc
        loc[:, 2] = c.tray_floor_top + halves[name] + z_extra
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = tp + quat_apply(tq, loc)
        st[:, 3:7] = tq
        scene.block_bodies[name].write_root_state_to_sim(st, all_ids)

    def prize_z_local() -> float:
        return float((scene.prize.data.root_pos_w - scene.env_origins)[0, 2])

    def kick_prize(vz: float) -> None:
        st = scene.prize.data.root_state_w[all_ids].clone()
        st[:, 7:9] = 0.0
        st[:, 9] = vz
        st[:, 10:13] = 0.0
        scene.prize.write_root_state_to_sim(st, all_ids)

    def track_max_prize_z(steps: int = 90) -> float:
        zmax = prize_z_local()
        for _ in range(steps // 3):
            step(3)
            zmax = max(zmax, prize_z_local())
        return zmax

    slots = _tray_slots(scene)

    # =========================== 1-2. settle / no-NaN + reset sanity =========================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in list(scene.block_bodies.values()) + [scene.prize])
    check("settle: states finite, score exactly 0, tower covers the well",
          finite and float(scene.score()[0]) == 0.0 and bool(scene.covered()[0]))
    check("reset sanity: prize inside the well, uncovered latch clear",
          bool(scene.prize_in_well()[0]) and not bool(scene.uncovered[0]))

    # =========================== 3. randomization is real ====================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(5)
        v = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        t = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        g = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(v[0]), float(v[1]), float(t[0]), float(t[1]),
                      float(scene.tray.data.root_quat_w[0, 3]), float(g[0]), float(g[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (vault_xy, tray_xy, tray_qz, pad_xy):\n{arr}",
          flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: vault, tray (pose+yaw) and pad all move on reset (readback)",
          spread[0] > 0.008 and spread[2] > 0.008 and spread[4] > 0.02
          and spread[5] > 0.008)

    # =========================== 4. null policy fails ========================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, well still covered after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0])
          and bool(scene.covered()[0]))

    # =========================== 5-7. oracle on 3 seeds ======================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} (score exactly 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 8-9 + 13-14. monotonicity ladder ============================
    # Milestones: top stored 0.15 -> mid 0.30 -> cap 0.55 (uncovered latch fires) ->
    # prize out on the ground 0.70 (near-miss beside the pad, NOT success) -> on pad 1.0.
    torch.manual_seed(41)
    env.reset()
    step(30)
    ladder: list[float] = [float(scene.score()[0])]

    def do_stage(fn, pred) -> None:
        fn()
        step(15)  # real physics before the first poll (see `until` in the oracle)
        settle_until(pred)
        ladder.append(round(float(scene.score()[0]), 3))

    do_stage(lambda: block_to_tray_slot("top", slots["top"]),
             lambda: bool(scene.stored()[0, 2]))
    do_stage(lambda: block_to_tray_slot("mid", slots["mid"]),
             lambda: bool(scene.stored()[0, 1]))
    do_stage(lambda: block_to_tray_slot("cap", slots["cap"]),
             lambda: bool(scene.stored()[0, 0]) and bool(scene.uncovered[0]))
    g = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    beside = (float(g[0]) + c.pad_half + 0.05, float(g[1]),
              c.prize_size / 2 + 0.002)
    do_stage(lambda: place_prize(beside),
             lambda: not bool(scene.prize_in_well()[0]) and bool(scene.prize_settled()[0]))
    near_miss_score = ladder[-1]
    near_miss_ok = not bool(scene.success()[0])
    do_stage(lambda: place_prize((float(g[0]), float(g[1]),
                                  c.pad_t + c.prize_size / 2 + 0.015)),
             lambda: bool(scene.success()[0]))
    print(f"[smoke] monotonicity ladder scores: {ladder}", flush=True)
    expect = [0.0, 0.15, 0.30, 0.55, 0.70, 1.0]
    check("monotonicity: score strictly climbs the declared milestones",
          all(b > a for a, b in zip(ladder, ladder[1:]))
          and all(abs(s - e) <= 0.03 for s, e in zip(ladder, expect)))
    check("monotonicity: every partial score < 1.0", max(ladder[:-1]) < 1.0)
    check("near-miss: prize on the ground beside the pad -> 0.70, no success",
          near_miss_ok and abs(near_miss_score - 0.70) <= 0.03)
    check("ladder recovery: final placement on the pad reaches success (1.0)",
          bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)

    # =========================== 10. negative A: the seed's own strategy =====================
    # maniskill/stack_pyramid STACKS the cubes. Execute that plan perfectly here — rebuild
    # the tower on open ground away from the vault — and it is worth <= 0.15 (only the
    # incidental uncovered latch) with the prize still imprisoned.
    torch.manual_seed(51)
    env.reset()
    step(30)
    # All three states written in ONE pass (no stepping between): teleporting the cap
    # away first would leave mid/top dangling over the vault mid-battery.
    base = (0.02, 0.02)
    zc = 0.0
    for name in BLOCK_NAMES:
        h = halves[name]
        place_block(name, (base[0], base[1], zc + h + 0.002))
        zc += 2 * h + 0.003
    step(40)  # real landing + post_step latching before judging
    settle_until(lambda: bool(scene.blocks_settled()[0].all()))
    report("seed-restack")
    check("negative A (seed strategy): a perfect re-stack on open ground scores <= 0.15, "
          "no success, prize still imprisoned",
          float(scene.score()[0]) <= 0.15 and not bool(scene.success()[0])
          and bool(scene.prize_in_well()[0]))

    # =========================== 11. negative B: dump beside the tray ========================
    # Un-stacking alone is not the task: blocks dumped on the GROUND next to the tray count
    # for nothing even with the prize delivered — tidy storage is load-bearing.
    torch.manual_seed(61)
    env.reset()
    step(30)
    t = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    for i, name in enumerate(BLOCK_NAMES):  # one pass, no stepping between (see negative A)
        h = halves[name]
        place_block(name, (float(t[0]) - 0.12 + 0.12 * i, float(t[1]) - 0.22, h + 0.002))
    g = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    place_prize((float(g[0]), float(g[1]), c.pad_t + c.prize_size / 2 + 0.015))
    step(40)  # real landing + post_step latching before judging
    settle_until(lambda: bool(scene.prize_on_pad()[0]))
    settle_until(lambda: bool(scene.blocks_settled()[0].all()), max_steps=200)
    report("dump-beside")
    check("negative B: blocks dumped beside the tray never count (score ~0.5, no success)",
          int(scene.stored()[0].sum()) == 0 and 0.42 <= float(scene.score()[0]) <= 0.58
          and not bool(scene.success()[0]))

    # =========================== 12. negative C: stacking INSIDE the tray ====================
    # The seed's skill is rejected even inside the goal container: a block stacked on
    # another block in the tray fails the rest-on-tray-floor gate.
    torch.manual_seed(71)
    env.reset()
    step(30)
    # One pass (see negative A): cap and mid to their floor slots, top ON TOP of the cap
    # (6 mm above the still-airborne cap's upper face so the pair settles without overlap).
    block_to_tray_slot("cap", slots["cap"])
    block_to_tray_slot("mid", slots["mid"])
    block_to_tray_slot("top", slots["cap"], z_extra=2 * halves["cap"] + 0.024)
    step(40)  # real landing + post_step latching before judging
    settle_until(lambda: bool(scene.stored()[0, 0]) and bool(scene.stored()[0, 1]))
    settle_until(lambda: bool(scene.blocks_settled()[0].all()), max_steps=200)
    report("stack-in-tray")
    check("negative C: a block stacked on another block inside the tray does NOT count",
          int(scene.stored()[0].sum()) == 2 and not bool(scene.stored()[0, 2])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.48)

    # =========================== 15-16. occlusion calibration probe ==========================
    # The imprisonment is PHYSICS, not fiction: kick the prize straight up at 1.5 m/s
    # (free rise ~11 cm). Under the intact tower it cannot pass the rim; with the blocks
    # removed the same kick clears the rim.
    torch.manual_seed(81)
    env.reset()
    step(30)
    kick_prize(1.5)
    zmax_covered = track_max_prize_z(90)
    settle_until(lambda: bool(scene.prize_settled()[0]), max_steps=200)
    report("kick-covered")
    print(f"[smoke] occlusion probe: covered max prize z = {zmax_covered * 1000:.1f} mm "
          f"(rim at {c.rim_top * 1000:.0f} mm)", flush=True)
    check("occlusion probe (covered): the capped well physically imprisons the prize",
          zmax_covered < c.rim_top + 0.005 and bool(scene.prize_in_well()[0])
          and float(scene.score()[0]) <= 0.02)
    # remove the tower to parking, same kick
    for i, name in enumerate(BLOCK_NAMES):
        h = halves[name]
        place_block(name, (-0.35, -0.12 + 0.14 * i, h + 0.002))
    step(20)
    kick_prize(1.5)
    zmax_open = track_max_prize_z(90)
    report("kick-open")
    print(f"[smoke] occlusion probe: open max prize z = {zmax_open * 1000:.1f} mm "
          f"(delta vs covered = {(zmax_open - zmax_covered) * 1000:.1f} mm)", flush=True)
    check("occlusion probe (open): the same kick clears the rim once uncovered",
          zmax_open > c.rim_top + 0.02)

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.vault_unstack")
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
