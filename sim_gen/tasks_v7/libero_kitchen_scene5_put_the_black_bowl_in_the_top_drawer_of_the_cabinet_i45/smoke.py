"""Smoke / rubric-REJECTION battery for DominoRelayScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i45`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — teleport the tiles into a standing relay, nudge
the trigger once, hands-off cascade — is the acceptance evidence that the rubric
ACCEPTS a correct outcome). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: trigger standing on the pad,
                            ball seated on the shelf, tiles lying flat on the floor;
                            score ~0 at rest, no success;
  3-4. randomization      — READBACK over 6 seeded resets: gallery xy + yaw, pad
                            span + bearing, trigger yaw jitter, tile scatter all move;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  SEED strategy       — the seed's whole plan ("grasp the payload, carry it,
                            set it down inside the receptacle") = ball SET DOWN at
                            rest on the pit floor: ball_in_pit verified True, yet NOT
                            success and score ~0 — a hand-delivered ball earns
                            NOTHING (pit credit is causal-gated);
  7.  out of order        — the complete-looking END STATE assembled in the WRONG
                            ORDER (ball delivered long before the trigger fell,
                            trigger + k tiles then laid fallen in the corridor):
                            causal stays 0, NOT success;
  8.  too slow (tau)      — trigger felled, ball dislodged only AFTER tau_s has
                            passed (the one-by-one / manual-ferry timescale):
                            causal 0, NOT success;
  9.  relay clause        — trigger felled and the ball knocked in WITHIN tau
                            (causal genuinely latches) but ZERO tiles fallen in the
                            corridor: NOT success, score <= 0.75;
  10. near-miss seat      — ball dislodged but resting ON the shelf (not in the
                            pit): ball_in_pit False;
  11. near-miss outside   — ball on the floor OUTSIDE the front wall: ball_in_pit
                            False;
  12. wrong-place relay   — k tiles genuinely FALLEN but laterally OUTSIDE the
                            pad->window corridor: fallen_in_corridor() = 0, NOT
                            success even with causal + pit constructed;
  13. latched credit      — a standing relay built in the corridor latches `built`;
                            removing the tiles leaves the latched score unchanged
                            (and no success);
  14. monotonicity        — standing MORE relay tiles in the corridor latches
                            strictly more construction credit;
  15. rejection audit     — success() was never True at ANY judged point;
  16. final no-NaN        — all task-object states finite at the end.

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
    env = ENVS.get("simgen.domino_relay")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.85, 0.65)) + o),
                                tuple(np.array((0.00, -0.05, 0.08)) + o),
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

    def gallery_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.gallery.data.root_pos_w - scene.env_origins)[0]
        q = scene.gallery.data.root_quat_w[0]
        return fp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def to_world(lx: float, ly: float) -> tuple[float, float]:
        fp, yaw = gallery_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (float(fp[0]) + lx * cy - ly * sy, float(fp[1]) + lx * sy + ly * cy)

    def report(tag: str) -> None:
        bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) trig_upz={float(scene._up_z(scene.trigger)[0]):+.3f} "
              f"standing={float(scene.standing_in_corridor()[0]):.0f} "
              f"fallen={float(scene.fallen_in_corridor()[0]):.0f} "
              f"built={float(scene.built[0]):.0f} causal={float(scene.causal[0]):.0f} "
              f"pit={bool(scene.ball_in_pit()[0])} settled={bool(scene.settled()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, lx: float, ly: float, z: float,
                    quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
                    settle_steps: int = 30) -> None:
        """Kinematic probe placement in GALLERY-LOCAL xy (instrumentation, not a
        solution) + REAL physics steps before judging (the zero-step trap)."""
        wx, wy = to_world(lx, ly)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 7], st[:, 8], st[:, 9] = vel
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)
        step(settle_steps)

    def lying_quat(world_yaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qy(90 deg): tile local +z -> horizontal (lying flat on its
        largest face, thickness axis vertical)."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(world_yaw / 2), math.sin(world_yaw / 2)
        return (cy2 * c45, -sy2 * c45, cy2 * c45, sy2 * c45)

    def chain_frame() -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """(pad_local (2,), u_local (2,), span, world chain yaw) for env 0."""
        _fp, fyaw = gallery_pose()
        pad = scene.pad_local[0]
        u = scene.u_local[0]
        chain_yaw = fyaw + math.atan2(float(u[1]), float(u[0]))
        return pad, u, float(scene.span[0]), chain_yaw

    def chain_local(s: float, lat: float = 0.0) -> tuple[float, float]:
        """Gallery-local xy of the point at distance `s` along the chain, offset
        `lat` to its left."""
        pad, u, _span, _cy = chain_frame()
        return (float(pad[0]) + float(u[0]) * s - float(u[1]) * lat,
                float(pad[1]) + float(u[1]) * s + float(u[0]) * lat)

    def topple_trigger() -> None:
        """Instrumentation: lay the trigger tile flat on its pad spot (constructs
        `trigger fallen` without a nudge)."""
        pad, _u, _span, chain_yaw = chain_frame()
        place_local(scene.trigger, float(pad[0]), float(pad[1]), c.tile_t / 2 + 0.002,
                    quat=lying_quat(chain_yaw), settle_steps=30)

    def fin_all() -> bool:
        ok = (torch.isfinite(scene.gallery.data.root_state_w).all()
              and torch.isfinite(scene.pad.data.root_state_w).all()
              and torch.isfinite(scene.trigger.data.root_state_w).all()
              and torch.isfinite(scene.ball.data.root_state_w).all())
        for t in scene.tiles:
            ok = ok and torch.isfinite(t.data.root_state_w).all()
        return bool(ok)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    bl = scene._fix_local(scene.ball.data.root_pos_w)[0]
    tile_z = [float(scene._z_rel(t)[0]) for t in scene.tiles]
    check("settle: states finite; trigger standing on its pad, ball seated on the "
          "shelf at seat height, all 8 tiles lying flat on the floor (readback)",
          fin_all() and float(scene._up_z(scene.trigger)[0]) > 0.99
          and abs(float(scene._z_rel(scene.trigger)[0]) - c.tile_z0) < 0.010
          and abs(float(bl[2]) - c.seat_z) < 0.008 and abs(float(bl[1]) - c.seat_y) < 0.010
          and all(abs(z - c.tile_t / 2) < 0.008 for z in tile_z)
          and bool(scene.settled()[0]))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(5)
        fp, fyaw = gallery_pose()
        pad = scene.pad_local[0]
        bear = math.atan2(float(pad[0]), -float(pad[1]))
        tq = scene.trigger.data.root_quat_w[0]
        tyaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
        t0 = scene._fix_local(scene.tiles[0].data.root_pos_w)[0]
        t1 = scene._fix_local(scene.tiles[1].data.root_pos_w)[0]
        reads.append((float(fp[0]), float(fp[1]), fyaw, float(scene.span[0]), bear,
                      tyaw, float(t0[0]), float(t0[1]), float(t1[0]), float(t1[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (fix_x, fix_y, fix_yaw, span, bearing, "
          f"trig_yaw, t0_x, t0_y, t1_x, t1_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: gallery xy + yaw, pad span + bearing all vary across "
          "seeded resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.05
          and spread[3] > 0.01 and spread[4] > 0.05)
    check("randomization: trigger yaw and tile scatter vary across seeded resets "
          "(readback)",
          spread[5] > 0.02 and spread[6] > 0.02 and spread[7] > 0.02
          and spread[8] > 0.02 and spread[9] > 0.02)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. SEED strategy ===========================================
    # The seed's whole plan is "grasp the payload, carry it across free space, set it
    # down inside the receptacle". Here that end state — the ball placed at rest on the
    # pit floor (physically impossible for the embodied agent: roofed gallery, letterbox
    # window under an eave; constructed by teleport as pure instrumentation) — must earn
    # NOTHING: pit credit is causal-gated and success requires the cascade.
    env.reset(seed=41)
    step(10)
    place_local(scene.ball, 0.0, 0.13, c.ball_r + 0.002, settle_steps=90)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (ball SET DOWN at rest on the pit floor, in-pit delivery "
          "verified): NOT success and score <= 0.02 — hand delivery without the "
          "cascade earns nothing",
          bool(scene.ball_in_pit()[0]) and not ok and s <= 0.02)

    # =========================== 7. out-of-order end state ==================================
    # Same episode: the ball dislodged LONG BEFORE the trigger fell. Now lay the trigger
    # flat and put k tiles genuinely fallen in the corridor — the end state LOOKS
    # complete (trigger down, relay down, ball in pit, settled) but the order was wrong.
    topple_trigger()
    _pad, _u, span, chain_yaw = chain_frame()
    for i in range(c.k_relay):
        lx, ly = chain_local(0.06 + 0.06 * i)
        place_local(scene.tiles[i], lx, ly, c.tile_t / 2 + 0.002,
                    quat=lying_quat(chain_yaw), settle_steps=5)
    step(90)
    report("out-of-order")
    s, ok = judge()
    check("out of order: trigger down + k tiles fallen in the corridor + ball in pit "
          "+ settled, but the ball moved BEFORE the trigger fell — causal stays 0, "
          "NOT success",
          bool(scene.trigger_fallen()[0])  # trigger genuinely down
          and float(scene.fallen_in_corridor()[0]) >= float(c.k_relay)
          and bool(scene.ball_in_pit()[0]) and bool(scene.settled()[0])
          and float(scene.causal[0]) < 0.5 and not ok)

    # =========================== 8. too slow (tau clause) ===================================
    # Trigger felled first, ball dislodged only after tau_s has passed — the timescale
    # of toppling tiles one by one or ferrying by hand. causal must NOT latch.
    env.reset(seed=51)
    step(10)
    topple_trigger()
    wait = int((c.tau_s + 0.8) / env.dt)
    step(wait)
    place_local(scene.ball, 0.0, 0.13, c.ball_r + 0.002, settle_steps=60)
    report("too-slow")
    s, ok = judge()
    check(f"too slow: ball dislodged {c.tau_s + 0.8:.1f}s after the trigger fell "
          f"(> tau {c.tau_s:.2f}s) — causal 0, no pit credit, NOT success",
          bool(scene.trigger_fallen()[0]) and bool(scene.ball_in_pit()[0])
          and float(scene.causal[0]) < 0.5 and float(scene.pit_latch[0]) < 0.5 and not ok)

    # =========================== 9. relay clause ============================================
    # Trigger felled and the ball knocked in WITHIN tau — causal genuinely latches —
    # but no relay tile ever fell in the corridor: the fallen-relay clause alone
    # must reject success.
    env.reset(seed=61)
    step(10)
    topple_trigger()
    step(30)  # 0.25 s < tau
    place_local(scene.ball, 0.0, 0.13, c.ball_r + 0.002, settle_steps=60)
    report("no-relay")
    s, ok = judge()
    check("relay clause: causal genuinely latched (ball in within tau of the trigger "
          "falling) and ball in pit, but ZERO tiles fallen in the corridor — NOT "
          "success, score <= 0.75",
          float(scene.causal[0]) > 0.5 and bool(scene.ball_in_pit()[0])
          and float(scene.fallen_in_corridor()[0]) < 1.0 and not ok and s <= 0.75)

    # =========================== 10-11. near-miss ball states ===============================
    env.reset(seed=71)
    step(10)
    # dislodged but still ON the shelf (against the front wall corner of the shelf)
    place_local(scene.ball, 0.030, c.seat_y, c.seat_z + 0.002, settle_steps=60)
    report("on-shelf")
    check("near miss: ball dislodged from its seat but resting ON the shelf — "
          "ball_in_pit() False",
          bool(scene.ball_dislodged()[0]) and not bool(scene.ball_in_pit()[0]))
    # on the floor OUTSIDE the front wall (as if it fell out of the window)
    place_local(scene.ball, 0.0, -0.10, c.ball_r + 0.002, settle_steps=45)
    report("outside")
    s, ok = judge()
    check("near miss: ball on the floor OUTSIDE the front wall — ball_in_pit() "
          "False, NOT success",
          not bool(scene.ball_in_pit()[0]) and not ok)

    # =========================== 12. wrong-place relay ======================================
    # Causal + pit constructed as in check 9, and k tiles genuinely FALLEN — but
    # laterally OUTSIDE the corridor: fallen_in_corridor() must count zero.
    env.reset(seed=81)
    step(10)
    topple_trigger()
    step(30)
    place_local(scene.ball, 0.0, 0.13, c.ball_r + 0.002, settle_steps=45)
    _pad, _u, span, chain_yaw = chain_frame()
    for i in range(c.k_relay):
        lx, ly = chain_local(0.08 + 0.06 * i, lat=c.corr_halfw + 0.07)
        place_local(scene.tiles[i], lx, ly, c.tile_t / 2 + 0.002,
                    quat=lying_quat(chain_yaw), settle_steps=5)
    step(60)
    report("relay-offside")
    s, ok = judge()
    check("wrong place: k tiles genuinely fallen but laterally OUTSIDE the corridor "
          "— fallen_in_corridor() 0, NOT success even with causal + pit",
          float(scene.causal[0]) > 0.5 and bool(scene.ball_in_pit()[0])
          and float(scene.fallen_in_corridor()[0]) < 1.0 and not ok)

    # =========================== 13. latched credit survives regression =====================
    env.reset(seed=91)
    step(10)
    _pad, _u, span, chain_yaw = chain_frame()
    for i in range(c.k_relay):
        lx, ly = chain_local(0.06 + 0.055 * i)
        place_local(scene.tiles[i], lx, ly, c.tile_z0 + 0.002,
                    quat=(math.cos(chain_yaw / 2), 0.0, 0.0, math.sin(chain_yaw / 2)),
                    settle_steps=5)
    step(60)
    report("chain-built")
    s_in, _ = judge()
    for i in range(c.k_relay):
        place_local(scene.tiles[i], -0.35 + 0.12 * i, -0.70, c.tile_t / 2 + 0.002,
                    quat=lying_quat(chain_yaw), settle_steps=5)
    step(45)
    report("chain-removed")
    s_out, ok = judge()
    check("latched credit: a standing relay in the corridor latches `built` "
          f"(score {s_in:.2f} >= 0.28); removing the tiles leaves it unchanged "
          "(and still no success)",
          s_in >= 0.28 and abs(s_out - s_in) < 0.02 and not ok)

    # =========================== 14. construction monotonicity ==============================
    env.reset(seed=101)
    step(10)
    _pad, _u, span, chain_yaw = chain_frame()
    stand_q = (math.cos(chain_yaw / 2), 0.0, 0.0, math.sin(chain_yaw / 2))
    for i in range(2):
        lx, ly = chain_local(0.06 + 0.055 * i)
        place_local(scene.tiles[i], lx, ly, c.tile_z0 + 0.002, quat=stand_q,
                    settle_steps=5)
    step(30)
    s_two = float(scene.score()[0])
    for i in range(2, 4):
        lx, ly = chain_local(0.06 + 0.055 * i)
        place_local(scene.tiles[i], lx, ly, c.tile_z0 + 0.002, quat=stand_q,
                    settle_steps=5)
    step(30)
    s_four = float(scene.score()[0])
    judge()
    check("monotonicity: standing MORE relay tiles in the corridor latches strictly "
          f"more construction credit ({s_two:.3f} < {s_four:.3f})",
          s_two + 0.10 < s_four)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.domino_relay")
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
    except BaseException as exc:  # noqa: BLE001 - die loudly, never idle to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
