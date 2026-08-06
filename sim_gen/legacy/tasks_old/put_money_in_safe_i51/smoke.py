"""Smoke / oracle test for VaultChuteScene (sim_gen task `put_money_in_safe_i51`) —
NullRobot, teleport-oracle, RECORDED.

Battery (compass_crate / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout settles finite, score exactly 0, no success;
  2. randomization      — READBACK: vault yaw + position and cash scatter all move across
                          seeded resets; cash-count subset sampling produces both 1 and 2;
  3. null-policy-fails  — 240 idle substeps -> score ~0, no success;
  4. negative A (seed)  — the seed's own strategy, "set the cash down on the safe shelf":
                          cash laid on the bare chute slides OUT of the vault; the exit
                          time must sit in the analytic mu/tilt window (slide calibration);
  5. calibration B      — the rubber chock, laid on the same chute, GRIPS (moves < 8 mm in
                          2 s) and scores exactly the 0.25 dam credit;
  6. oracle x3 seeds    — dam the chute, rest the present cash uphill against the chock;
                          success() and score 1.0 on 3 seeds with subset sampling ON;
  7. chain              — subset OFF (both stacks present): dam + two-stack chain succeeds;
  8. monotonicity       — 0 -> dam (0.25) -> one stack secured (~0.575) -> both (1.0):
                          strictly increasing, partials < 0.9, final is success;
  9. near-miss          — chock dammed on the TONGUE (outside the doorway): the cash is
                          caught straddling the threshold -> no dam credit, no success;
                          re-damming properly and re-laying the cash recovers success;
 10. negative C (roof)  — cash parked on the vault roof: never entered, score ~0;
 11. anti-flash         — cash teleported to a perfect in-vault pose WITHOUT a dam is not
                          success on the spot (persistence latch) and has slid out 0.75 s
                          later (physics agrees).

Run (forge): python -u -m simgen_tasks.put_money_in_safe_i51.smoke --headless
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
    from simgen_tasks.put_money_in_safe_i51 import scene as scene_mod
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _all_ids(env):
    return torch.arange(env.num_envs, device=env.device)


def _place_on_chute(scene, body, x_surf: float, half_t: float, y: float = 0.0,
                    gap: float = 0.002) -> None:
    """Kinematic write: rest `body` flush on the chute at surface station `x_surf` (vault
    body frame), zero velocities — computed per env from each vault's own live pose."""
    from isaaclab.utils.math import quat_apply, quat_mul

    c, env = scene.cfg, scene.env
    n, dev = env.num_envs, env.device
    sp = scene.vault.data.root_pos_w
    sq = scene.vault.data.root_quat_w
    th = math.radians(c.tilt_deg)
    qy = torch.tensor([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0],
                      device=dev).expand(n, 4)
    pl = torch.tensor(c.rest_local(x_surf, y, half_t, gap), device=dev).expand(n, 3)
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = sp + quat_apply(sq, pl)
    st[:, 3:7] = quat_mul(sq, qy)
    body.write_root_state_to_sim(st, _all_ids(env))


def _place_local_flat(scene, body, local_xyz) -> None:
    """Kinematic write: `body` level (vault-yaw aligned) at a vault-body-frame position."""
    from isaaclab.utils.math import quat_apply

    env = scene.env
    n, dev = env.num_envs, env.device
    sp = scene.vault.data.root_pos_w
    sq = scene.vault.data.root_quat_w
    pl = torch.tensor(list(local_xyz), device=dev).expand(n, 3)
    st = torch.zeros(n, 13, device=dev)
    st[:, 0:3] = sp + quat_apply(sq, pl)
    st[:, 3:7] = sq
    body.write_root_state_to_sim(st, _all_ids(env))


def oracle_solution(scene_or_env, step_fn=None, verbose: bool = True) -> bool:
    """Teleport-oracle for the CURRENT episode: (1) lay the rubber chock across the chute
    just inside the doorway (the dam), (2) rest every PRESENT cash stack uphill against
    it, chained one above the other, (3) let real physics settle until the persistence
    latch confirms every stack. Returns True iff scene.success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=env.device)

    def _step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def until(pred, max_steps: int = 400, poll: int = 10) -> bool:
        waited = 0
        while waited < max_steps:
            _step(poll)
            waited += poll
            if pred():
                return True
        return bool(pred())

    _place_on_chute(scene, scene.chock, x_surf=0.075, half_t=c.chock_size[2] / 2)
    ok_dam = until(lambda: bool(scene.dam_ok()[0]), max_steps=200)
    if verbose:
        print(f"[oracle] dam deployed ok={ok_dam} score={float(scene.score()[0]):.3f}",
              flush=True)

    xs = 0.075 - c.chock_size[0] / 2
    for i, body in enumerate(scene.cash):
        if not bool(scene.present[0, i]):
            continue
        half_len = c.cash_dims[i][0] / 2
        xs = xs - half_len - 0.008
        _place_on_chute(scene, body, x_surf=xs, half_t=c.cash_dims[i][2] / 2)
        ok_i = until(lambda i=i: bool(scene.secured[0, i]), max_steps=400)
        if verbose:
            print(f"[oracle] cash {i} laid at x_surf={xs:.3f} secured={ok_i} "
                  f"score={float(scene.score()[0]):.3f}", flush=True)
        xs = xs - half_len
    until(lambda: bool(scene.success()[0]), max_steps=300)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.vault_chute")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=scene_mod.VaultChuteSceneCfg(subset_sample=False))
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.45, 1.00)) + o),
                                tuple(np.array((0.0, 0.0, 0.12)) + o),
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

    def cash_x(i: int = 0) -> float:
        return float(scene.cash_pos_safe()[0, i, 0])

    def report(tag: str) -> None:
        pl = scene.cash_pos_safe()[0]
        print(f"[smoke] {tag:14s} | cash_x=({float(pl[0, 0]):+.3f},{float(pl[1, 0]):+.3f}) "
              f"dam={bool(scene.dam_ok()[0])} "
              f"entered={scene.entered[0].int().tolist()} "
              f"secured={scene.secured[0].int().tolist()} "
              f"present={scene.present[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def vault_yaw() -> float:
        q = scene.vault.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in [scene.vault, scene.chock, *scene.cash])
    check("settle: all states finite, everything at rest",
          finite and bool(scene.cash_settled()[0].all()))
    check("settle: score exactly 0 at reset, no success",
          float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(4)
        p = (scene.cash[0].data.root_pos_w - scene.env_origins)[0]
        v = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        reads.append((vault_yaw(), float(v[0]), float(v[1]), float(p[0]), float(p[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (vault_yaw, vx, vy, cash_x, cash_y):\n"
          f"{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: vault yaw + pose and cash scatter all move (readback)",
          spread[0] > 0.5 and (spread[1] > 0.02 or spread[2] > 0.02)
          and (spread[3] > 0.05 or spread[4] > 0.05))
    scene.cfg.subset_sample = True
    counts = []
    for s in range(201, 209):
        torch.manual_seed(s)
        env.reset()
        counts.append(int(scene.present[0].sum()))
    scene.cfg.subset_sample = False
    print(f"[smoke] subset-sampled cash counts over 8 resets: {counts}", flush=True)
    check("randomization: cash-count subset sampling yields both 1 and 2",
          1 in counts and 2 in counts)

    # =========================== 3. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle substeps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4. negative A: the seed's own strategy =====================
    # rlbench/put_money_in_safe sets the dollar stack down on the safe's shelf. Here the
    # shelf is the tilted chute: the same plan — lay the cash inside, no dam — must fail
    # PHYSICALLY, and on the analytic clock: a = g(sin t - mu_d cos t) ~= 2.29 m/s^2 from
    # station x=0.02 puts the center past the doorway plane around substep ~35 @120 Hz.
    torch.manual_seed(51)
    env.reset()
    step(20)
    _place_on_chute(scene, scene.cash[0], x_surf=0.020, half_t=c.cash_dims[0][2] / 2)
    exit_step = None
    for t in range(240):
        step(1)
        if cash_x(0) > c.hd:
            exit_step = t + 1
            break
    print(f"[smoke] seed-strategy slide: exit at substep {exit_step} "
          f"(analytic ~35 @120 Hz)", flush=True)
    step(100)
    report("seed-strategy")
    check("negative A (seed strategy): cash laid on the bare chute leaves the vault "
          "(no dam, no success, score <= 0.15)",
          exit_step is not None and cash_x(0) > c.hd
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.15)
    check("negative A: exit time inside the analytic mu/tilt window [15, 150] substeps",
          exit_step is not None and 15 <= exit_step <= 150)

    # =========================== 5. calibration B: the chock grips ==========================
    torch.manual_seed(61)
    env.reset()
    step(20)
    _place_on_chute(scene, scene.chock, x_surf=0.075, half_t=c.chock_size[2] / 2)
    step(4)
    p0 = scene.chock.data.root_pos_w[0].clone()
    step(240)
    disp = float((scene.chock.data.root_pos_w[0] - p0).norm())
    report("chock-grips")
    check("calibration: rubber chock GRIPS the same chute (moved < 8 mm in 2 s) and "
          "earns exactly the 0.25 dam credit",
          disp < 0.008 and bool(scene.dam_ok()[0])
          and 0.24 <= float(scene.score()[0]) <= 0.26)

    # =========================== 6. oracle on 3 seeds (subset ON) ===========================
    scene.cfg.subset_sample = True
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        n_present = int(scene.present[0].sum())
        ok = oracle_solution(env, step_fn=step)
        report(f"oracle-seed{s}")
        check(f"oracle reaches success() on seed {s} ({n_present} stack(s) present, "
              f"score 1.0)", ok and bool(scene.success()[0])
              and float(scene.score()[0]) == 1.0)
    scene.cfg.subset_sample = False

    # =========================== 7. two-stack chain (subset OFF) ============================
    torch.manual_seed(71)
    env.reset()
    step(20)
    ok = oracle_solution(env, step_fn=step)
    report("chain-both")
    check("chain: with BOTH stacks present the dam holds the two-stack chain -> success",
          ok and int(scene.present[0].sum()) == 2)

    # =========================== 8. rubric monotonicity =====================================
    torch.manual_seed(41)
    env.reset()
    step(20)
    s0 = float(scene.score()[0])
    _place_on_chute(scene, scene.chock, x_surf=0.075, half_t=c.chock_size[2] / 2)
    settle_until(lambda: bool(scene.dam_ok()[0]), max_steps=200)
    s1 = float(scene.score()[0])
    _place_on_chute(scene, scene.cash[0], x_surf=0.020, half_t=c.cash_dims[0][2] / 2)
    settle_until(lambda: bool(scene.secured_now()[0, 0]), max_steps=400)
    s2 = float(scene.score()[0])
    _place_on_chute(scene, scene.cash[1], x_surf=-0.050, half_t=c.cash_dims[1][2] / 2)
    settle_until(lambda: bool(scene.success()[0]), max_steps=400)
    s3 = float(scene.score()[0])
    print(f"[smoke] monotonicity ladder: {[round(v, 3) for v in (s0, s1, s2, s3)]}",
          flush=True)
    check("monotonicity: 0 -> dam -> one secured -> both: strictly increasing",
          s0 < s1 < s2 < s3)
    check("monotonicity: partials below 0.9, final exactly 1.0 and success",
          s0 == 0.0 and max(s1, s2) < 0.9 and s3 == 1.0 and bool(scene.success()[0]))

    # =========================== 9. near-miss: dam at the threshold =========================
    # Chock laid on the TONGUE (outside the doorway plane): it still stops the cash, but
    # the bundle comes to rest straddling the threshold — outside strict containment, no
    # dam credit. Re-damming properly and re-laying the cash recovers success.
    torch.manual_seed(81)
    env.reset()
    step(20)
    _place_on_chute(scene, scene.chock, x_surf=c.hd + 0.012,
                    half_t=c.chock_size[2] / 2)
    step(30)
    _place_on_chute(scene, scene.cash[0], x_surf=0.020, half_t=c.cash_dims[0][2] / 2)
    step(150)
    report("threshold-dam")
    check("near-miss: dam on the tongue catches the cash STRADDLING the doorway — "
          "no dam credit, no success, score <= 0.15",
          not bool(scene.dam_ok()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.15
          and bool(scene.cash_settled()[0, 0]) and cash_x(0) > c.hd - c.front_margin)
    # clear the doorway before re-damming (the dam station overlaps where the cash came
    # to rest — a teleport into that overlap would depenetration-pop both bodies)
    _place_local_flat(scene, scene.cash[0], (0.42, 0.15, c.cash_dims[0][2] / 2 + 0.002))
    step(20)
    ok = oracle_solution(env, step_fn=step)
    report("re-dammed")
    check("near-miss recovery: re-damming inside and re-laying the cash -> success", ok)

    # =========================== 10. negative C: cash on the roof ===========================
    torch.manual_seed(91)
    env.reset()
    step(20)
    _place_local_flat(scene, scene.cash[0],
                      (0.0, 0.0, c.wall_h + c.roof_t + c.cash_dims[0][2] / 2 + 0.002))
    step(120)
    report("on-roof")
    check("negative C: cash parked on the vault roof never entered, score ~0, no success",
          not bool(scene.entered[0, 0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # =========================== 11. anti-flash persistence =================================
    # A perfect in-vault pose teleported WITHOUT a dam: not success on the spot (the
    # secured latch needs 30 consecutive settled substeps), and physics evicts it anyway.
    torch.manual_seed(101)
    env.reset()
    step(20)
    _place_on_chute(scene, scene.cash[0], x_surf=0.020, half_t=c.cash_dims[0][2] / 2)
    env.iscene.update(0.0)
    check("anti-flash: teleported-in cash is NOT success at the written state "
          "(persistence latch gates)",
          bool(scene.inside_strict()[0, 0]) and not bool(scene.secured[0, 0])
          and not bool(scene.success()[0]))
    step(90)
    report("anti-flash")
    check("anti-flash: 0.75 s later the dam-less cash has slid out (still no success)",
          cash_x(0) > c.hd and not bool(scene.success()[0]))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.vault_chute")
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
