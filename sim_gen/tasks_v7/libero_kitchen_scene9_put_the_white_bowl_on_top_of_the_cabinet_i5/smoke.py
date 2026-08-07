"""Smoke / rubric-rejection battery for CounterweightShelfScene (sim_gen task
`libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5`) — NullRobot,
teleported probe states (instrumentation, NOT a solution), RECORDED.

Battery:
  1. settle/no-NaN      — reset settles finite: tray resting on its TIPPED stop, bowl and
                          block on the floor, score ~0, no latches, no success;
  2. randomization      — READBACK across seeds: bowl/block xy and yaw move, the flank
                          sides swap, the tray always starts tipped;
  3. null-policy-fails  — 240 idle steps -> score ~0, no success, tray still tipped;
  4-6. oracle x3 seeds  — block seated in the socket (tray swings LEVEL), bowl set on the
                          level platform -> success() and score 1.0; success PERSISTS over
                          240 further steps (no flicker);
  7. monotonicity       — 0 (idle) < 0.30 (counterweight seated + shelf leveled) < 0.45
                          (bowl lifted) < 1.0 (placed = success); partials < 1.0;
  8. negative A (seed)  — the seed's own plan verbatim: set the bowl on the (tipped)
                          shelf top without counterweighting -> the bowl slides off onto
                          the floor, the shelf stays tipped, no success, score <= 0.15
                          (only the lift latch);
  9. negative B (wrong place) — bowl set over the rear SOCKET instead of the platform ->
                          no success (and the bowl cannot even level the shelf: it slides
                          off the tilted socket rim);
 10. negative C (block misplaced) — the counterweight set on the FRONT platform instead
                          of the socket -> slides off the ramp, shelf stays tipped, no
                          seat/level credit;
 11. negative D (near-miss y) — block seated, bowl resting just outside the lateral
                          tolerance -> no success;
 12. negative E (near-miss x) — block seated, bowl resting over the hinge, outside the
                          platform x band -> no success;
 13. negative F (inverted) — block seated, bowl UPSIDE-DOWN on the platform -> no success;
 14. calibration       — bowl drop x-offset sweep across the platform band (published
                          in/out table).

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet_i5 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweight_shelf")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.65, -0.95, 0.75)) + o),
                                tuple(np.array((0.30, 0.0, 0.18)) + o),
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

    def ang() -> float:
        return math.degrees(float(scene.tray_angle()[0]))

    def report(tag: str) -> None:
        bl = scene._to_tray_frame(scene.block.data.root_pos_w)[0]
        wl = scene._to_tray_frame(scene.bowl.data.root_pos_w)[0]
        print(f"[smoke] {tag:16s} ang={ang():+7.2f} "
              f"block_loc=({bl[0]:+.3f},{bl[1]:+.3f},{bl[2]:+.3f}) "
              f"bowl_loc=({wl[0]:+.3f},{wl[1]:+.3f},{wl[2]:+.3f}) "
              f"seated={bool(scene.block_seated()[0])} level={bool(scene.tray_level()[0])} "
              f"on_plat={bool(scene.bowl_on_platform()[0])} settled={bool(scene.settled()[0])} "
              f"score={sc():.2f} success={bool(scene.success()[0])} frames={len(frames)}",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tray_local_to_world(loc):
        from isaaclab.utils.math import quat_apply

        p = quat_apply(scene.tray.data.root_quat_w[0:1],
                       torch.tensor([loc], device=device))[0]
        return p + scene.tray.data.root_pos_w[0] - scene.env_origins[0]

    def tp(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(
            [float(v) for v in pos_env], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def tilt_quat():
        th = float(scene.tray_angle()[0])
        return (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)

    def seat_block() -> None:
        """Teleport the block into the (tipped) socket, tilt-matched; wait out the swing."""
        pos = tray_local_to_world([c.sock_cx, 0.0, c.block_size / 2 + 0.012])
        tp(scene.block, [float(v) for v in pos], tilt_quat())
        settle_until(lambda: bool(scene.tray_level()[0]) and bool(scene.tray_still()[0]),
                     max_steps=300)

    def place_bowl(x_loc: float, y_loc: float = 0.0, inverted: bool = False,
                   drop: float = 0.006) -> None:
        """Set the bowl just above the (level) platform at a tray-local xy, release."""
        q = (0.0, 1.0, 0.0, 0.0) if inverted else (1.0, 0.0, 0.0, 0.0)
        pos = tray_local_to_world([x_loc, y_loc, c.bowl_h / 2 + drop])
        tp(scene.bowl, [float(v) for v in pos], q)
        settle_until(lambda: bool(scene.settled()[0]), max_steps=300)

    def bowl_w():
        return (scene.bowl.data.root_pos_w - scene.env_origins)[0]

    def block_w():
        return (scene.block.data.root_pos_w - scene.env_origins)[0]

    # =========================== 1. settle / no-NaN ==============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.tray.data.root_state_w, scene.bowl.data.root_state_w,
                     scene.block.data.root_state_w], dim=-1)
    check("settle: states finite, tray resting TIPPED on its stop (< -30 deg), bowl and "
          "block on the floor, everything settled",
          bool(torch.isfinite(st0).all()) and ang() < -30.0
          and float(bowl_w()[2]) < 0.06 and float(block_w()[2]) < 0.06
          and bool(scene.settled()[0]))
    check("settle: score ~0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._seated_ever[0]) and not bool(scene._lifted_ever[0])
          and not bool(scene._placed_ever[0]))

    # =========================== 2. randomization is real ========================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        bp = bowl_w()
        kp = block_w()
        from isaaclab.utils.math import quat_apply

        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        reads.append((float(bp[0]), float(bp[1]), float(kp[0]), float(kp[1]),
                      yaw, ang()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bowl_x, bowl_y, block_x, block_y, bowl_yaw, "
          f"tray_ang):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    flanks = {v > 0 for v in arr[:, 1]}
    check("randomization: bowl/block positions and bowl yaw move across seeds (readback); "
          "the bowl spawns on BOTH flanks; the tray always starts tipped",
          spread[0] > 0.02 and spread[2] > 0.02 and spread[4] > 0.5
          and len(flanks) == 2 and all(v < -28.0 for v in arr[:, 5]))

    # =========================== 3. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: score ~0, no success, tray still tipped after 240 idle steps",
          sc() <= 0.02 and not bool(scene.success()[0]) and ang() < -30.0)

    # =========================== 4-6. oracle on 3 seeds ==========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(30)
        seat_block()
        report(f"oracle{s}-level")
        place_bowl(-0.07, 0.0)
        ok = settle_until(lambda: bool(scene.success()[0]), max_steps=300)
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: counterweight seated -> shelf level; bowl on the platform "
              f"-> success, score 1.0, persists 240 steps",
              ok and bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 7. rubric monotonicity ==========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    s0 = sc()
    seat_block()
    s_seat = sc()
    report("mono-seated")
    # lift the bowl (latches lifted), then set it back on the floor
    bp0 = bowl_w().clone()
    tp(scene.bowl, [float(bp0[0]), float(bp0[1]), c.lift_z + 0.08])
    step(4)
    s_lift = sc()
    tp(scene.bowl, [float(bp0[0]), float(bp0[1]), c.bowl_h / 2 + 0.003])
    settle_until(lambda: bool(scene.settled()[0]), max_steps=150)
    place_bowl(-0.07, 0.0)
    settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    s3 = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} seated+level={s_seat:.3f} "
          f"lifted={s_lift:.3f} success={s3:.3f}", flush=True)
    check("monotonicity: idle < seated+leveled < +lifted < success (1.0), latched credit "
          "does not evaporate",
          s0 <= 0.02 and 0.28 <= s_seat <= 0.32 and 0.43 <= s_lift <= 0.47
          and s3 == 1.0 and s0 < s_seat < s_lift < s3)
    check("monotonicity: partial states score < 1.0", max(s0, s_seat, s_lift) < 1.0)

    # =========================== 8. negative A: the seed's own strategy ==========================
    # Grasp the bowl, set it down on top of the "cabinet" — verbatim, with the shelf left
    # unconfigured. The bowl starts flat on the tipped platform and slides straight off.
    torch.manual_seed(51)
    env.reset()
    step(30)
    pos = tray_local_to_world([-0.07, 0.0, c.bowl_h / 2 + 0.004])
    tp(scene.bowl, [float(v) for v in pos], tilt_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("seed-strategy")
    check("negative A (seed strategy): bowl set on the TIPPED shelf top slides off onto "
          "the floor; shelf stays tipped; no success; score <= 0.15 (lift latch only)",
          not bool(scene.success()[0]) and ang() < -30.0
          and float(bowl_w()[2]) < 0.09 and not bool(scene.bowl_on_platform()[0])
          and sc() <= 0.155)

    # =========================== 9. negative B: bowl over the socket =============================
    torch.manual_seed(61)
    env.reset()
    step(30)
    pos = tray_local_to_world([c.sock_cx, 0.0, c.sock_wall_h + c.bowl_h / 2 + 0.006])
    tp(scene.bowl, [float(v) for v in pos], tilt_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("bowl-in-socket")
    check("negative B (wrong place): bowl set over the rear socket instead of the platform "
          "-> no success (it slides off the tilted rim; the goal region is the platform)",
          not bool(scene.success()[0]) and not bool(scene.bowl_on_platform()[0]))

    # =========================== 10. negative C: counterweight on the platform ===================
    torch.manual_seed(71)
    env.reset()
    step(30)
    pos = tray_local_to_world([-0.07, 0.0, c.block_size / 2 + 0.004])
    tp(scene.block, [float(v) for v in pos], tilt_quat())
    settle_until(lambda: bool(scene.settled()[0]), max_steps=300)
    step(60)
    report("block-on-plat")
    check("negative C (block misplaced): counterweight set on the front platform slides "
          "off; shelf stays tipped; no seat/level credit",
          not bool(scene.tray_level()[0]) and ang() < -30.0
          and not bool(scene._seated_ever[0]) and not bool(scene._leveled_ever[0])
          and sc() <= 0.02)

    # =========================== 11-13. near-misses on a LEVEL shelf =============================
    torch.manual_seed(81)
    env.reset()
    step(30)
    seat_block()
    place_bowl(-0.07, c.place_y_abs + 0.025)  # just outside the lateral tolerance
    step(60)
    report("near-miss-y")
    check("negative D (near-miss y): bowl resting just outside the lateral tolerance on "
          "the level shelf -> no success",
          bool(scene.tray_level()[0]) and not bool(scene.bowl_on_platform()[0])
          and not bool(scene.success()[0]) and sc() < 1.0)

    place_bowl(c.place_x_hi + 0.022, 0.0)  # over the hinge, outside the x band
    step(60)
    report("near-miss-x")
    check("negative E (near-miss x): bowl resting over the hinge, outside the platform "
          "band -> no success",
          not bool(scene.bowl_on_platform()[0]) and not bool(scene.success()[0])
          and sc() < 1.0)

    place_bowl(-0.07, 0.0, inverted=True)
    step(60)
    report("inverted")
    check("negative F (inverted): bowl UPSIDE-DOWN on the level platform -> no success",
          bool(scene.tray_level()[0]) and not bool(scene.bowl_upright()[0])
          and not bool(scene.bowl_on_platform()[0]) and not bool(scene.success()[0])
          and sc() < 1.0)

    # =========================== 14. calibration =================================================
    print("[smoke] CALIBRATION: bowl rest x (tray frame) -> counts as placed "
          "(block seated, shelf level)", flush=True)
    cal = []
    for x_off in (-0.07, -0.100, -0.135, -0.005):
        torch.manual_seed(91)
        env.reset()
        step(20)
        seat_block()
        place_bowl(x_off, 0.0)
        step(120)
        loc = scene._to_tray_frame(scene.bowl.data.root_pos_w)[0]
        cal.append((x_off, float(loc[0]), bool(scene.bowl_on_platform()[0]),
                    bool(scene.success()[0])))
        print(f"[smoke]   drop_x={x_off * 1000:+5.0f}mm rest_x={float(loc[0]) * 1000:+5.0f}mm "
              f"placed={cal[-1][2]} success={cal[-1][3]}", flush=True)
    check("calibration: bowl resting inside the platform band counts as placed; outside "
          "(past the front-edge band or over the hinge) does not",
          cal[0][2] and cal[1][2] and not cal[2][2] and not cal[3][2])

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counterweight_shelf")
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
