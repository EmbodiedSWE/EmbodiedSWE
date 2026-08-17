"""Smoke / rubric-REJECTION battery for BreachDoorwayScene (sim_gen task
`obstacle_i400`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — slide the three bricks out of the doorway top
brick first with contact forces, then push the cargo through the bore — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here
is instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the
rubric REJECTS it; no probe in this battery ever reaches success(), and a final
audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: three bricks stacked in the
                            doorway plug, cargo resting on the near floor; score ~0
                            at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: fixture xy + yaw move,
                            cargo scatter and brick slot jitter move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("carry the payload OVER the wall,
                            set it down at the goal") = cargo teleported directly to
                            rest INSIDE the court (physically impossible for the
                            embodied agent: the court is roofed and sealed;
                            constructed by teleport as pure instrumentation): the
                            end state is verified through+in-court+settled, yet NOT
                            success (bore-transit latch never set) and score ~0;
  7.  blocked door        — plug intact: the cargo is force-pushed at the doorway
                            for 240 steps; it genuinely presses the stack (moved-
                            assert) but never gets near the bore, bricks stay put,
                            no entry credit, no success;
  8.  out-of-order        — the BOTTOM brick is magicked out (teleport): the bricks
                            above DROP into the vacancy (readback: the middle brick
                            lands at floor height, still in the plug) and the
                            doorway stays plugged — a subsequent force-push is
                            still blocked; score is exactly the one-brick credit;
  9.  bore-rest partial   — bricks cleared, cargo placed at rest INSIDE the bore:
                            entry latch + demolition credit = 0.55 exactly, NOT
                            success (not through);
  10. near-miss           — cargo at rest 8 mm SHORT of the through threshold (past
                            the inner face but not fully): NOT success;
  11. wrong object        — a BRICK teleported into the court instead of the cargo:
                            NOT success, only the brick-out credit;
  12. latched credit      — full partial progress constructed (bricks out, cargo in
                            bore), then the cargo removed AND a brick re-plugged:
                            the latched 0.55 survives unchanged, still no success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.obstacle_i400.smoke --headless
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
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.breach_doorway")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -0.95, 0.70)) + o),
                                tuple(np.array((0.00, 0.10, 0.10)) + o),
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

    def fix_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = fix_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def report(tag: str) -> None:
        cl = scene._fix_local(scene.cargo.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | cargo_local=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) out={float(scene.bricks_out_now()[0]):.0f} "
              f"out_latch={float(scene.out_latch[0]):.0f} "
              f"enter_latch={float(scene.enter_latch[0]):.0f} "
              f"through={bool(scene.cargo_through()[0])} "
              f"in_court={bool(scene.cargo_in_court()[0])} "
              f"settled={bool(scene.settled()[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                    settle_steps: int = 30) -> None:
        """Kinematic probe placement in FIXTURE-LOCAL xy (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy = to_world(lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def park_bricks(*ks: int) -> None:
        """Instrumentation: magic bricks out of the doorway to parking spots."""
        for j, k in enumerate(ks):
            place_local(scene.bricks[k], -0.62, -0.30 - 0.15 * j,
                        c.brick_h / 2 + 0.002, settle_steps=5)
        step(30)

    def push_cargo(steps: int) -> float:
        """The solve's forward force servo (velocity-regulated, lateral PD hold),
        run against whatever blocks the doorway. Returns the max fixture-local y
        the cargo centre ever reached. Ends force-free + settled."""
        _fp, fyaw = fix_pose()
        cy, sy = math.cos(fyaw), math.sin(fyaw)
        fwd = torch.tensor([-sy, cy, 0.0], device=device)
        lat = torch.tensor([cy, sy, 0.0], device=device)
        max_y = -10.0
        for _ in range(steps):
            cl = scene._fix_local(scene.cargo.data.root_pos_w)[0]
            max_y = max(max_y, float(cl[1]))
            v_w = scene.cargo.data.root_lin_vel_w[0]
            v_fwd = float(torch.dot(v_w, fwd))
            v_lat = float(torch.dot(v_w, lat))
            f_fwd = max(0.0, min(0.90, 0.55 + c.cargo_mass * 40.0 * (0.12 - v_fwd)))
            f_lat = max(-0.6, min(0.6, c.cargo_mass * (-50.0 * float(cl[0]) - 10.0 * v_lat)))
            f_world = f_fwd * fwd + f_lat * lat
            scene.cargo.set_external_force_and_torque(
                f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            step(1)
        scene.cargo.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)
        step(40)
        return max(max_y, float(scene._fix_local(scene.cargo.data.root_pos_w)[0][1]))

    def fin_all() -> bool:
        ok = bool(torch.isfinite(scene.fixture.data.root_state_w).all()
                  and torch.isfinite(scene.cargo.data.root_state_w).all())
        for b in scene.bricks:
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    cz = c.cargo_size / 2 + 0.002
    bz = c.brick_h / 2 + 0.002

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    cargo_z = float(scene._fix_local(scene.cargo.data.root_pos_w)[0][2])
    plugged = all(bool(scene._in_plug(b)[0]) for b in scene.bricks)
    check("settle: states finite; three bricks stacked in the doorway plug, cargo "
          "resting on the near floor (readback)",
          fin_all() and plugged and float(scene.bricks_out_now()[0]) == 0.0
          and abs(cargo_z - c.cargo_size / 2) < 0.008 and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = fix_pose()
        cl = scene._fix_local(scene.cargo.data.root_pos_w)[0]
        b0 = scene._fix_local(scene.bricks[0].data.root_pos_w)[0]
        b2 = scene._fix_local(scene.bricks[2].data.root_pos_w)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(cl[0]), float(cl[1]),
                      float(b0[0]), float(b2[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, cargo_x, cargo_y, "
          f"brick0_x, brick2_x):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: fixture xy + yaw vary across seeded resets (readback)",
          spread[0] > 0.02 and spread[1] > 0.008 and spread[2] > 0.05)
    check("randomization: cargo scatter and brick slot jitter vary across seeded "
          "resets (readback)",
          spread[3] > 0.03 and spread[4] > 0.010 and spread[5] > 0.0012
          and spread[6] > 0.0012)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "grasp the payload, carry it OVER the wall, set it down
    # at the goal". Here that END STATE — the cargo at rest inside the court, having
    # never gone through the doorway (impossible for the embodied agent: the court is
    # roofed and sealed; constructed by teleport as pure instrumentation) — must NOT
    # be success: the bore-transit latch was never set.
    env.reset(seed=41)
    step(10)
    place_local(scene.cargo, 0.0, 0.18, cz, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (cargo SET DOWN at rest inside the court, through+in-court+"
          "settled all verified by readback): NOT success — it never transited the "
          "doorway bore — and score ~0",
          bool(scene.cargo_through()[0]) and bool(scene.cargo_in_court()[0])
          and bool(scene.settled()[0]) and float(scene.enter_latch[0]) < 0.5
          and not ok and s <= 0.02)

    # =========================== 7. blocked door ============================================
    env.reset(seed=51)
    step(10)
    place_local(scene.cargo, 0.0, -0.16, cz, settle_steps=20)
    max_y = push_cargo(240)
    report("blocked-door")
    s, ok = judge()
    check("blocked door: force-pushed cargo genuinely presses the intact plug (moved "
          f"from -0.16 to max_y={max_y:+.3f}) but never nears the bore; bricks stay "
          "put, no entry credit, no success",
          -0.12 < max_y < -0.02 and float(scene.bricks_out_now()[0]) == 0.0
          and float(scene.enter_latch[0]) < 0.5 and not ok and s <= 0.02)

    # =========================== 8. out-of-order ============================================
    # Magic the BOTTOM brick out (teleport instrumentation): gravity drops the bricks
    # above into the vacancy and the doorway stays plugged — extraction order is
    # enforced by physics, and a push is still blocked.
    env.reset(seed=61)
    step(10)
    place_local(scene.bricks[0], -0.62, -0.30, bz, settle_steps=120)
    b1z = float(scene._fix_local(scene.bricks[1].data.root_pos_w)[0][2])
    replugged = bool(scene._in_plug(scene.bricks[1])[0]) and bool(
        scene._in_plug(scene.bricks[2])[0])
    place_local(scene.cargo, 0.0, -0.16, cz, settle_steps=20)
    max_y = push_cargo(200)
    report("out-of-order")
    s, ok = judge()
    check("out-of-order: bottom brick magicked out -> the bricks above DROP into the "
          f"vacancy (middle brick lands at floor height z={b1z:.3f}, still in the "
          "plug), the push is still blocked, and score is exactly the one-brick "
          "credit (0.10)",
          replugged and b1z < c.brick_h / 2 + 0.012 and max_y < -0.02
          and float(scene.enter_latch[0]) < 0.5 and not ok and abs(s - 0.10) < 0.005)

    # =========================== 9. bore-rest partial credit ================================
    env.reset(seed=71)
    step(10)
    park_bricks(2, 1, 0)
    place_local(scene.cargo, 0.0, c.wall_t / 2 + 0.005, cz, settle_steps=60)
    report("bore-rest")
    s, ok = judge()
    in_bore = bool(scene.cargo_in_bore()[0])
    check("bore-rest: bricks cleared + cargo at rest INSIDE the bore = demolition + "
          "entry credit exactly 0.55, NOT success (not through)",
          in_bore and float(scene.out_latch[0]) == 3.0
          and float(scene.enter_latch[0]) > 0.5 and not ok and abs(s - 0.55) < 0.002)

    # =========================== 10. near-miss ==============================================
    place_local(scene.cargo, 0.0, c.through_y - 0.008, cz, settle_steps=60)
    report("near-miss")
    s, ok = judge()
    check("near-miss: cargo at rest 8 mm SHORT of the through threshold (past the "
          "inner face, not fully inside): NOT success, credit stays 0.55",
          not bool(scene.cargo_through()[0]) and bool(scene.settled()[0]) and not ok
          and abs(s - 0.55) < 0.002)

    # =========================== 11. wrong object ===========================================
    env.reset(seed=81)
    step(10)
    place_local(scene.bricks[2], 0.0, 0.18, bz, settle_steps=60)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: a BRICK delivered into the court instead of the cargo: NOT "
          "success, only the brick-out credit (<= 0.15)",
          not bool(scene.cargo_through()[0]) and not ok and s <= 0.15)

    # =========================== 12. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    park_bricks(2, 1, 0)
    place_local(scene.cargo, 0.0, c.wall_t / 2 + 0.005, cz, settle_steps=40)
    s_in, _ = judge()
    place_local(scene.cargo, 0.0, -0.35, cz, settle_steps=40)
    place_local(scene.bricks[0], 0.0, c.wall_t / 2, bz, settle_steps=40)  # re-plug one
    report("regressed")
    s_out, ok = judge()
    check("latched credit: cargo removed from the bore AND a brick re-plugged into "
          f"the doorway — the latched credit survives unchanged ({s_in:.3f} -> "
          f"{s_out:.3f}), still no success",
          abs(s_in - 0.55) < 0.002 and abs(s_out - s_in) < 0.005 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.breach_doorway")
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
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die NOW, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
