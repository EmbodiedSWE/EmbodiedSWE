"""Smoke / rubric-rejection battery for LatchDrawerScene (sim_gen task
`libero_kitchen_scene1_open_top_drawer_i15`) — NullRobot, teleported probe states
(instrumentation, NOT a solution; no probe reaches success — solve.py is the
acceptance proof), RECORDED.

Battery (all rejection + mechanism calibration):
  1. settle/no-NaN       — reset settles finite: drawer CLOSED, red cube enclosed in the
                           cavity, score ~0, no latches, no success;
  2. cube enclosed       — the cargo starts sealed under the carcass top behind the flush
                           plate (the physical order-enforcer);
  3. randomization       — READBACK across 6 seeds: pad polar slot moves and covers BOTH
                           flanks, cube-in-drawer offset moves, decoy moves; drawer always
                           starts closed;
  4. null-policy         — 240 idle steps -> score ~0, drawer closed, no success;
  5. negative A (seed strategy) — the seed's plan is PULLING the drawer open: a drawer
                           state-teleported to 11 cm open WITHOUT the press snaps shut
                           (latch spring), earns no armed/ejected credit, no success;
  6. negative B (shallow poke) — an 8 mm press (below the 14 mm arming depth) springs
                           back closed: not armed, no credit;
  7. mechanism click     — a quasi-static 17 mm press probe arms the latch (0.15), then
                           releases and EJECTS >= 9 cm on its own; the cube rides the
                           drawer out and ends exposed in front of the carcass; drawer
                           open alone is NOT success (the seed's goal state is rejected);
  8. persistence         — 240 further steps: latched credit does not evaporate, the
                           drawer stays parked open;
  9. negative C (near-miss) — red cube settled on the floor just outside the pad
                           tolerance -> no success;
 10. negative D (wrong object) — the BLUE decoy centered on the pad -> no success;
 11. negative E (stacked) — red cube resting ON the decoy which sits on the pad (right
                           xy, wrong height — not resting on the pad) -> no success;
 12. negative F (wrong place) — red cube on the cabinet TOP -> no success (the lift
                           latch fires, giving the monotonicity ladder's 0.50 rung);
 13. monotonicity        — 0 (idle) < 0.15 (armed) < 0.30 (ejected) < 0.50 (+lifted),
                           all < 1.0;
 14. calibration         — click-press probe on 2 more seeds: the drawer ejects and the
                           cube ends exposed every time (published table).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i15.smoke --headless
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

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_open_top_drawer_i15 import (  # noqa: F401
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.latch_drawer")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.75, -0.85, 0.70)) + o),
                                tuple(np.array((0.28, 0.0, 0.10)) + o),
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

    def sc() -> float:
        return float(scene.score()[0])

    def disp() -> float:
        return float(scene.drawer_disp()[0])

    def cube_w():
        return (scene.cube.data.root_pos_w - scene.env_origins)[0]

    def decoy_w():
        return (scene.decoy.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cp = cube_w()
        px = scene._pad_xy[0]
        print(f"[smoke] {tag:16s} disp={disp() * 1000:+7.1f}mm armed={bool(scene._armed[0])} "
              f"released={bool(scene._released[0])} "
              f"cube=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f}) "
              f"pad=({px[0]:+.3f},{px[1]:+.3f}) in_drw={bool(scene.cube_in_drawer()[0])} "
              f"on_pad={bool(scene.cube_on_pad()[0])} score={sc():.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tp(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(
            [float(v) for v in pos_env], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def shift_drawer(d: float) -> None:
        """Teleport the drawer follower to displacement `d`, co-shifting the cube by the
        same delta (preserves the relative pose — pure instrumentation)."""
        cur = disp()
        cp = cube_w()
        tp(scene.cube, [float(cp[0]) + (d - cur), float(cp[1]), float(cp[2])],
           tuple(float(v) for v in scene.cube.data.root_quat_w[0]))
        st = scene.drawer_pose_at(torch.full((n,), d, device=device), all_ids)
        scene.drawer.write_root_state_to_sim(st, all_ids)

    def drawer_settled() -> bool:
        return bool(scene.drawer_still()[0]) and bool(scene.cube_settled()[0])

    cube_exposed_x = c.cab_face_x - c.cube_size / 2  # cube center in front of the carcass

    # =========================== 1-2. settle / no-NaN / enclosed =================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.drawer.data.root_state_w, scene.cube.data.root_state_w,
                     scene.decoy.data.root_state_w], dim=-1)
    check("settle: states finite, drawer CLOSED, cube riding the cavity floor, everything "
          "settled, score ~0, no latches, no success",
          bool(torch.isfinite(st0).all()) and abs(disp()) < 0.006
          and bool(scene.cube_in_drawer()[0]) and drawer_settled()
          and sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._armed[0]) and not bool(scene._released[0]))
    cp = cube_w()
    check("enclosed: at reset the cube sits behind the flush face, under the carcass top "
          "(sealed in — the physical order-enforcer)",
          float(cp[0]) > c.cab_face_x + 0.01 and float(cp[2]) + c.cube_size / 2 < c.ap_z_hi)

    # =========================== 3. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        cp = cube_w()
        dp = decoy_w()
        px = scene._pad_xy[0]
        reads.append((float(px[0]), float(px[1]), float(cp[0]), float(cp[1]),
                      float(dp[0]), float(dp[1]), disp()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (pad_x, pad_y, cube_x, cube_y, decoy_x, decoy_y, "
          f"disp):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    flanks = {v > 0 for v in arr[:, 1]}
    check("randomization: pad polar slot moves and covers BOTH flanks, cube-in-drawer "
          "offset moves, decoy moves; drawer always starts closed (readback)",
          (spread[0] + spread[1]) > 0.06 and len(flanks) == 2
          and (spread[2] + spread[3]) > 0.015 and (spread[4] + spread[5]) > 0.05
          and all(abs(v) < 0.006 for v in arr[:, 6]))

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, drawer still closed, no success after 240 idle steps",
          sc() <= 0.02 and abs(disp()) < 0.006 and not bool(scene.success()[0]))

    # =========================== 5. negative A: the seed's own strategy ==========================
    # The seed opens the drawer by PULLING. Emulate its end state: drawer teleported to
    # 11 cm open with the latch never pressed -> the latch spring snaps it shut.
    torch.manual_seed(51)
    env.reset()
    step(30)
    shift_drawer(-0.11)
    settle_until(drawer_settled, max_steps=400)
    step(60)
    report("seed-yank")
    check("negative A (seed strategy): a drawer yanked/teleported open WITHOUT the press "
          "snaps back shut; no armed/ejected credit, cube still enclosed, no success, "
          "score ~0",
          abs(disp()) < 0.02 and not bool(scene._armed[0]) and not bool(scene._released[0])
          and bool(scene.cube_in_drawer()[0]) and not bool(scene.success()[0])
          and sc() <= 0.02)

    # =========================== 6. negative B: shallow poke =====================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    shift_drawer(0.008)
    settle_until(drawer_settled, max_steps=300)
    step(30)
    report("shallow-poke")
    check("negative B (shallow poke): an 8 mm press (below the 14 mm arming depth) "
          "springs back closed; not armed, no credit",
          abs(disp()) < 0.006 and not bool(scene._armed[0]) and not bool(scene._released[0])
          and sc() <= 0.02)

    # =========================== 7-8. mechanism click + persistence ==============================
    torch.manual_seed(71)
    env.reset()
    step(30)
    s_null = sc()
    shift_drawer(0.017)
    step(2)
    s_arm = sc()  # armed latched, release not yet fired (drawer still deep)
    ok_eject = settle_until(
        lambda: bool(scene._released[0]) and float(scene.drawer_open()[0]) > c.open_gate
        and drawer_settled(), max_steps=500)
    s_eject = sc()
    report("click-eject")
    check("mechanism click: a quasi-static 17 mm press arms the latch (0.15 latched), "
          "then releases and ejects >= 9 cm on its own; the cube rides the drawer out "
          "and ends exposed; drawer open alone is NOT success (seed goal state rejected)",
          ok_eject and 0.13 <= s_arm <= 0.17 and 0.28 <= s_eject <= 0.32
          and bool(scene.cube_in_drawer()[0]) and float(cube_w()[0]) < cube_exposed_x
          and not bool(scene.success()[0]))
    step(240)
    report("persistence")
    check("persistence: 240 further steps — latched credit does not evaporate and the "
          "drawer stays parked open",
          sc() >= 0.295 and float(scene.drawer_open()[0]) > 0.10)

    # =========================== 9. negative C: near-miss beside the pad =========================
    px = scene._pad_xy[0]
    off = c.pad_tol + 0.025
    tp(scene.cube, [float(px[0]) + off, float(px[1]), c.cube_size / 2 + 0.003])
    settle_until(lambda: bool(scene.cube_settled()[0]), max_steps=200)
    step(30)
    report("near-miss")
    check("negative C (near-miss): red cube settled on the floor just outside the pad "
          "tolerance -> no success, no full credit",
          not bool(scene.cube_on_pad()[0]) and not bool(scene.success()[0]) and sc() <= 0.35)

    # =========================== 10. negative D: wrong object ====================================
    tp(scene.decoy, [float(px[0]), float(px[1]), c.pad_t + c.cube_size / 2 + 0.003])
    settle_until(lambda: float(scene.decoy.data.root_lin_vel_w[0].norm()) < 0.05,
                 max_steps=200)
    step(30)
    report("wrong-object")
    check("negative D (wrong object): the BLUE decoy centered on the pad -> no success "
          "(the rubric tracks the RED cube)",
          not bool(scene.success()[0]) and sc() <= 0.35)

    # =========================== 11. negative E: stacked, not resting on the pad ================
    tp(scene.cube, [float(px[0]), float(px[1]),
                    c.pad_t + c.cube_size + c.cube_size / 2 + 0.004])
    settle_until(lambda: bool(scene.cube_settled()[0]), max_steps=200)
    step(30)
    report("stacked")
    check("negative E (stacked): red cube resting ON the decoy over the pad (right xy, "
          "wrong height — not resting on the pad) -> no success",
          not bool(scene.cube_on_pad()[0]) and not bool(scene.success()[0]))

    # =========================== 12. negative F: wrong place (cabinet top) =======================
    tp(scene.cube, [c.cab_face_x + c.cab_depth / 2, 0.0,
                    c.ap_z_hi + c.cab_top_t + c.cube_size / 2 + 0.004])
    settle_until(lambda: bool(scene.cube_settled()[0]), max_steps=200)
    step(30)
    s_lift = sc()
    report("wrong-place")
    check("negative F (wrong place): red cube on the cabinet top -> no success (the lift "
          "latch fires; placement is judged on the pad only)",
          not bool(scene.cube_on_pad()[0]) and not bool(scene.success()[0])
          and 0.48 <= s_lift <= 0.52)

    # =========================== 13. monotonicity ================================================
    print(f"[smoke] monotonicity ladder: idle={s_null:.3f} armed={s_arm:.3f} "
          f"ejected={s_eject:.3f} +lifted={s_lift:.3f}", flush=True)
    check("monotonicity: idle < armed (0.15) < ejected (0.30) < +lifted (0.50), all "
          "partials < 1.0",
          s_null <= 0.02 and s_null < s_arm < s_eject < s_lift < 1.0)

    # =========================== 14. calibration: click across seeds =============================
    print("[smoke] CALIBRATION: click-press probe -> opening, cube exposure", flush=True)
    cal = []
    for s in (0, 1):
        torch.manual_seed(s)
        env.reset()
        step(30)
        shift_drawer(0.017)
        ok = settle_until(
            lambda: bool(scene._released[0]) and float(scene.drawer_open()[0]) > c.open_gate
            and drawer_settled(), max_steps=500)
        opening = float(scene.drawer_open()[0])
        exposed = bool(scene.cube_in_drawer()[0]) and float(cube_w()[0]) < cube_exposed_x
        cal.append((s, ok, opening, exposed))
        print(f"[smoke]   seed={s} ejected={ok} opening={opening * 1000:.0f}mm "
              f"cube_exposed={exposed}", flush=True)
    check("calibration: the click mechanism ejects the drawer and exposes the riding "
          "cube on every probed seed",
          all(ok and exp for _s, ok, _o, exp in cal))

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.latch_drawer")
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
