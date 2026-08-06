"""Smoke / rubric-REJECTION battery for TrestleServiceScene (sim_gen task
libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl_i7) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — the force-push + gravity-landing run — is the acceptance
evidence that the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation
that CONSTRUCTS a wrong (or partially-right) outcome as a settled state and asserts the
rubric's verdict on it. No probe below constructs full success.

  1-2. settle/no-NaN     — the authored layout settles finite; nothing seated/bridged/
                           topped; the board rests on the counter; score 0;
  3.  determinism        — the same seed twice -> identical layout readback;
  4-6. randomization     — READBACK over 8 seeded resets: bridge-axis yaw varies, bridge
                           centre jitters, all three side signs (bowl swap, board park,
                           cube park) take both values;
  7.  null policy        — 240 idle steps -> score ~0, no success;
  8.  seed strategy      — the SEED's end state (one black bowl stacked on the other
                           black bowl) -> success False, score 0;
  9.  flat service       — board laid flat on the COUNTER between the pads with the cube
                           on its deck (the whole "bridge" skipped) -> no bridge credit,
                           no cube credit, score 0;
 10.  ramp               — both bowls seated, board with one end on a bowl foot and one
                           end on the counter (settled ramp) -> not bridged, score pinned
                           at 0.30;
 11.  seat tolerance     — a 15 mm off-centre bowl seat counts (tolerance twin), a
                           settled 45 mm off-centre bowl does NOT;
 12.  flipped bowl       — a bowl set opening-UP on its pad (a stable pose) -> not seated;
 13.  off-pad assembly   — the complete bridge + cube built AWAY from the pads ->
                           success False, score 0 (position-to-mark is load-bearing);
 14.  grip-bar cube      — legit bridge, cube settled on the GRIP BAR top (30 mm above
                           the deck) -> not on_deck, score pinned at 0.65;
 15.  cube on counter    — cube beside the finished bridge -> no cube credit;
 16.  latched credit     — bridge credit (0.65) survives removing the board again;
 17.  finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Kit teardown hangs leave GPU zombies: global watchdog long before the forge timeout.
threading.Timer(1350.0, lambda: os._exit(4)).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.trestle_service")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    origin = env.iscene.env_origins
    z0 = c.surface_z

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.25, -1.25, 0.95)) + o),
                                tuple(np.array((0.02, 0.0, 0.25)) + o),
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

    def settle(max_steps: int = 500, poll: int = 10) -> None:
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if bool(scene.settled()[0]):
                return

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        sm = scene.seated_matrix()[0].flatten().tolist()
        print(f"[smoke] {tag:18s} | seated={sm} pads={bool(scene.pads_seated()[0])} "
              f"bridged={bool(scene.bridged()[0])} on_deck={bool(scene.cube_on_deck()[0])} "
              f"topped={bool(scene.topped()[0])} settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    def place(body, x: float, y: float, z: float, quat: tuple | None = None,
              settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics steps
        before judging."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3:7] = torch.tensor(quat or (1.0, 0.0, 0.0, 0.0), device=device)
        st[:, 0:3] += origin
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def yaw_quat(yaw: float) -> tuple:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def pad(j: int) -> tuple:
        p = scene.pad_xy[0, j]
        return float(p[0]), float(p[1])

    def bridge_axis() -> tuple:
        psi = float(scene.bridge_yaw[0])
        return math.cos(psi), math.sin(psi)

    def seat_bowl(i: int, j: int, dx: float = 0.0, dy: float = 0.0) -> None:
        px, py = pad(j)
        place(scene.bowls[i], px + dx, py + dy, z0 + 0.002, settle_steps=20)

    def drop_board_between_bowls() -> None:
        a = scene.bowls[0].data.root_pos_w[0, :2] - origin[0, :2]
        b = scene.bowls[1].data.root_pos_w[0, :2] - origin[0, :2]
        mid = (a + b) / 2
        yaw = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
        place(scene.board, float(mid[0]), float(mid[1]), z0 + c.bowl_h + c.board_t / 2 + 0.03,
              yaw_quat(yaw), settle_steps=10)
        settle()

    def layout_readback() -> tuple:
        p0 = pad(0)
        return (float(scene.bridge_yaw[0]), float(scene.bridge_cxy[0, 0]),
                float(scene.bridge_cxy[0, 1]), float(scene.swap[0]),
                float(scene.board_side[0]), float(scene.cube_side[0]), p0[0], p0[1])

    def all_finite() -> bool:
        bodies = scene.bowls + [scene.board, scene.cube]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN / clean rubric =========================
    env.reset(seed=11)
    settle()
    report("reset")
    board_z = float(scene.board.data.root_pos_w[0, 2] - origin[0, 2])
    check("settle: authored layout finite and settled; nothing seated/bridged/topped; "
          "board resting flat on the counter",
          all_finite() and bool(scene.settled()[0])
          and not bool(scene.seated_matrix()[0].any()) and not bool(scene.bridged()[0])
          and not bool(scene.topped()[0]) and abs(board_z - (z0 + c.board_t / 2)) < 0.005)
    check("rubric clean at reset: score 0, no success",
          float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 3. determinism =============================================
    env.reset(seed=777)
    step(3)
    read_a = layout_readback()
    env.reset(seed=777)
    step(3)
    read_b = layout_readback()
    print(f"[smoke] determinism readback: {read_a} vs {read_b}", flush=True)
    check("determinism: same seed -> identical layout readback",
          all(abs(a - b) < 1e-5 for a, b in zip(read_a, read_b)))

    # =========================== 4-6. randomization is real ==================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(3)
        reads.append(layout_readback())
    arr = np.array(reads)
    print(f"[smoke] randomization readback (psi, bc_x, bc_y, swap, board, cube, "
          f"p0x, p0y):\n{arr}", flush=True)
    check("randomization: bridge-axis yaw varies across seeds (readback)",
          arr[:, 0].max() - arr[:, 0].min() > 0.03)
    check("randomization: bridge centre jitters across seeds (readback)",
          (arr[:, 1].max() - arr[:, 1].min()) > 0.005
          or (arr[:, 2].max() - arr[:, 2].min()) > 0.005)
    check("randomization: bowl-swap, board-park and cube-park sides each take both values",
          all(arr[:, k].max() - arr[:, k].min() > 1.0 for k in (3, 4, 5)))

    # =========================== 7. null policy fails ========================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 8. seed strategy control =====================================
    # The seed's whole plan and end state: one black bowl stacked on the other black bowl.
    env.reset(seed=41)
    settle()
    b0 = scene.bowls[0].data.root_pos_w[0] - origin[0]
    place(scene.bowls[1], float(b0[0]), float(b0[1]), z0 + c.bowl_h + 0.002, settle_steps=20)
    settle()
    b1z = float(scene.bowls[1].data.root_pos_w[0, 2] - origin[0, 2])
    report("seed-strategy")
    check("seed strategy (one black bowl stacked on the other): stack is stable but scores "
          "0, no success",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02
          and b1z > z0 + c.bowl_h - 0.010)  # genuinely resting on the lower bowl

    # =========================== 9. flat service (bridge skipped) ============================
    env.reset(seed=45)
    settle()
    bcx, bcy = float(scene.bridge_cxy[0, 0]), float(scene.bridge_cxy[0, 1])
    psi = float(scene.bridge_yaw[0])
    place(scene.board, bcx, bcy, z0 + c.board_t / 2 + 0.002, yaw_quat(psi), settle_steps=10)
    settle()
    bp = scene.board.data.root_pos_w[0] - origin[0]
    place(scene.cube, float(bp[0]), float(bp[1]) + 0.02,
          float(bp[2]) + c.board_t / 2 + c.cube_size / 2 + 0.01, settle_steps=10)
    settle()
    report("flat-service")
    check("flat service: board on the COUNTER between the pads with the cube on its deck "
          "-> not bridged, no cube credit, score 0",
          not bool(scene.bridged()[0]) and not bool(scene.topped()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 10. ramp (one end on a bowl, one on the counter) ============
    env.reset(seed=51)
    settle()
    seat_bowl(0, 0)
    seat_bowl(1, 1)
    settle()
    s_pads = float(scene.score()[0])
    ax, ay = bridge_axis()
    p1x, p1y = pad(1)
    place(scene.board, p1x + 0.06 * ax, p1y + 0.06 * ay,
          z0 + c.bowl_h + c.board_t / 2 + 0.03, yaw_quat(float(scene.bridge_yaw[0])),
          settle_steps=10)
    settle()
    report("ramp")
    check("ramp: both bowls seated (0.30 latched) but board one-end-on-foot, "
          "one-end-on-counter -> not bridged, no success, score pinned at 0.30",
          abs(s_pads - 0.30) < 1e-3 and not bool(scene.bridged()[0])
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.30) < 1e-3)

    # =========================== 11. seat tolerance twin + near miss =========================
    env.reset(seed=61)
    settle()
    seat_bowl(0, 0, dx=0.015)
    settle()
    report("twin-15mm")
    check("tolerance twin: a 15 mm off-centre bowl counts as seated",
          bool(scene.seated_matrix()[0, 0, 0]) and not bool(scene.success()[0]))
    ax, ay = bridge_axis()
    p0x, p0y = pad(0)
    place(scene.bowls[0], p0x - 0.045 * ay, p0y + 0.045 * ax, z0 + 0.002, settle_steps=20)
    settle()
    report("near-miss-45mm")
    check("near miss: a settled 45 mm off-centre bowl is NOT seated",
          not bool(scene.seated_matrix()[0, 0, 0]) and bool(scene.settled()[0]))

    # =========================== 12. flipped bowl on the pad =================================
    env.reset(seed=71)
    settle()
    p0x, p0y = pad(0)
    place(scene.bowls[0], p0x, p0y, z0 + c.bowl_h + 0.002, (0.0, 1.0, 0.0, 0.0),
          settle_steps=20)
    settle()
    report("flipped-bowl")
    up_z = float(scene._up_z(scene.bowls[0].data.root_quat_w)[0])
    check("flipped bowl: a bowl set opening-UP on its pad (stable, centred) is NOT seated",
          up_z < -0.5 and not bool(scene.seated_matrix()[0, 0, 0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 13. complete assembly built OFF the pads ====================
    env.reset(seed=81)
    settle()
    place(scene.bowls[0], -0.03, 0.15, z0 + 0.002, settle_steps=20)
    place(scene.bowls[1], 0.13, 0.15, z0 + 0.002, settle_steps=20)
    settle()
    drop_board_between_bowls()
    bp = scene.board.data.root_pos_w[0] - origin[0]
    place(scene.cube, float(bp[0]) + 0.08, float(bp[1]),
          float(bp[2]) + c.board_t / 2 + c.cube_size / 2 + 0.02, settle_steps=10)
    settle()
    report("off-pad-assembly")
    board_elev = bool(scene.board_level()[0]) and bool(scene.board_elevated()[0])
    check("off-pad assembly: the complete elevated bridge + cube built AWAY from the pads "
          "-> not bridged, success False, score 0",
          board_elev and not bool(scene.bridged()[0]) and not bool(scene.topped()[0])
          and float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 14-16. legit bridge, wrong cube, latched credit =============
    env.reset(seed=91)
    settle()
    seat_bowl(0, 0)
    seat_bowl(1, 1)
    settle()
    drop_board_between_bowls()
    report("bridge-built")
    s_bridge = float(scene.score()[0])
    bp = scene.board.data.root_pos_w[0] - origin[0]
    place(scene.cube, float(bp[0]), float(bp[1]),
          float(bp[2]) + c.board_t / 2 + c.grip_h + c.cube_size / 2 + 0.002,
          settle_steps=40)
    settle()
    report("cube-on-grip")
    check("grip-bar cube: bridge legitimately built (0.65 latched), cube settled on the "
          "GRIP BAR top (30 mm above the deck) -> not on_deck, no success, score 0.65",
          abs(s_bridge - 0.65) < 1e-3 and bool(scene.bridged()[0])
          and not bool(scene.cube_on_deck()[0]) and not bool(scene.topped()[0])
          and abs(float(scene.score()[0]) - 0.65) < 1e-3 and not bool(scene.success()[0]))
    place(scene.cube, 0.25, -0.18, z0 + c.cube_size / 2 + 0.002, settle_steps=20)
    settle()
    report("cube-on-counter")
    check("cube on counter beside the finished bridge: no cube credit, no success, "
          "score 0.65",
          not bool(scene.topped()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.65) < 1e-3)
    place(scene.board, 0.30, -0.25, z0 + c.board_t / 2 + 0.002, yaw_quat(1.2),
          settle_steps=20)
    settle()
    report("board-removed")
    check("latched credit: bridge credit (0.65) survives removing the board again",
          not bool(scene.bridged()[0]) and abs(float(scene.score()[0]) - 0.65) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 17. finite at the end ========================================
    check("finite: all states finite at the end", all_finite())

    # =========================== save + verdict ==============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.trestle_service")
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
