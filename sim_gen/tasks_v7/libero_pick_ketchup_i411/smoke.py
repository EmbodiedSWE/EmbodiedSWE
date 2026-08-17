"""Smoke / rubric-REJECTION battery for BasculeKeepScene (sim_gen task
`libero_pick_ketchup_i411`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — lift both counterweights out of the tray, let
gravity deploy the bridge, push the ketchup across through the doorway — is the
acceptance evidence). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it, plus
physics probes that prove the mechanism is real (one block alone genuinely holds the
bridge raised; an empty tray genuinely deploys it; ballast aboard genuinely re-raises
it; the raised machine genuinely blocks the road). One construct DOES build the
genuine end state on purpose — the acceptance construct — and three flip pairs leave
and re-enter it; every other judged point must stay success()=False and a final audit
asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; bridge raised on its heel, both
                            blocks in the tray, bottles standing far out on the
                            ground; score ~0, no success;
  3-5.  randomization     — READBACK over 8 seeded resets: rig xy + yaw vary; the
                            bottles' ground slots genuinely SWAP (both orders occur,
                            always opposite); per-bottle jitter varies;
  6.    null policy       — 300 idle steps -> score ~0, no success;
  7.    raised interlock  — the ketchup laid along the raised deck channel and
        is real             SLAMMED 2.5 m/s up the slope reaches the tongue-tip zone
                            (readback) but NEVER passes the doorway: the raised
                            tongue-to-lintel aperture undercuts the bottle. No
                            success;
  8.    one block holds   — block A alone lifted out (teleport transport): the
                            bridge stays RAISED — either single counterweight holds
                            it, so BOTH must be removed;
  9.    empty tray        — block B lifted out too: gravity swings the bridge level
        deploys             onto the doorway sill BY ITSELF (no force ever applied
                            to the machine); readback tilt ~35 -> ~0 deg;
  10.   seed strategy     — the seed's move (carry the bottle over the receptacle
        fails               and drop): released over the keep it lands ON THE ROOF —
                            the keep is roofed; not inside, score ~0, no success;
  11.   wrong object      — the YELLOW mustard stood inside the keep (by fiat,
                            through the wall) with the ketchup left outside: the
                            identification + exclusion clauses reject it;
  12.   ballast-hold      — the leaf HELD level kinematically with BOTH blocks still
        cheat               in the tray and the ketchup placed inside: end state
                            LOOKS complete, but the tray-empty clause rejects it and
                            the latched score stays ~0 (deploy credit is gated on
                            the tray being empty) — pressing the bridge down is not
                            unballasting it;
  13.   ballast re-raises — the hold released: the ballasted leaf swings itself back
                            RAISED (readback) — level+settled+ballasted is not a
                            physical rest state, the machine works both ways;
  14.   acceptance        — both blocks lifted out of the tray: the bridge deploys
                            itself WITH the cargo already inside -> the genuine end
                            state, success TRUE, score 1.0;
  15-16. mustard flips    — mustard added into the keep: success flips FALSE
                            (exclusion clause); state restored via set_state:
                            success returns TRUE;
  17-18. bridge flips     — the whole bridge (pillar+leaf, coherently) dragged
                            aside off the doorway: success flips FALSE (the road no
                            longer spans — hinge-position clause); restored via
                            set_state: success returns TRUE;
  19.   settle gate       — the delivered ketchup kicked and judged while moving:
                            NOT success (must be at rest); resettled -> success
                            again;
  20.   near miss (deck)  — bridge deployed, ketchup standing ON the deck, never
                            pushed through: partial credit only (~0.45), no success;
  21.   near miss (wall)  — ketchup standing on the ground against the keep's
                            pedestal face: outside the interior box, no success;
  22.   rejection audit   — success() was never True at any judged point EXCEPT the
                            acceptance construct, the flip-backs and the resettle
                            (14, 16, 18, 19b);
  23.   final no-NaN      — all task-object states finite at the end.

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
    from .scene import (  # noqa: F401
        BLOCK, BOT_L, BOT_R, DECK_T, FLOOR_TOP, PIV_Z, ROOF_BOT, THETA_UP, TRAY_X0,
        TRAY_X1, WALL_X0, WALL_X1, _qapply, _qmul, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BLOCK, BOT_L, BOT_R, DECK_T, FLOOR_TOP, PIV_Z, ROOF_BOT, THETA_UP, TRAY_X0,
        TRAY_X1, WALL_X0, WALL_X1, _qapply, _qmul, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DEPOT = {"a": (-0.42, 0.42), "b": (-0.42, -0.42)}  # rig-frame ground depots


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bascule_keep")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.20, 0.95)) + o),
                                tuple(np.array((-0.05, 0.00, 0.15)) + o),
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

    ever_bad_success = [False]

    def judge() -> tuple[float, bool]:
        """Judge a REJECTION probe: success here is a rubric failure."""
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_bad_success[0] = ever_bad_success[0] or ok
        return s, ok

    def judge_accept() -> tuple[float, bool]:
        """Judge an ACCEPTANCE construct: success here is expected and allowed."""
        return float(scene.score()[0]), bool(scene.success()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        print(f"[smoke] {tag:18s} | tilt={float(scene.leaf_tilt_deg()[0]):+.2f}deg "
              f"hinge={bool(scene.hinge_intact()[0])} "
              f"down={bool(scene.bridge_down()[0])} "
              f"tray_a={bool(scene.in_tray(scene.block_a)[0])} "
              f"tray_b={bool(scene.in_tray(scene.block_b)[0])} "
              f"keep={bool(scene.in_keep(scene.ketchup)[0])} "
              f"mus_out={bool(scene.mustard_out()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def teleport(body, pos, quat, vel=None, ang=None, settle_steps: int = 45) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = quat
        if vel is not None:
            st[:, 7:10] = torch.tensor(vel, device=device)
        if ang is not None:
            st[:, 10:13] = torch.tensor(ang, device=device)
        body.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    def rig_pose(local) -> torch.Tensor:
        return scene.rig.data.root_pos_w + _qapply(
            scene.rig.data.root_quat_w, torch.tensor(local, device=device).expand(n, 3))

    def blocks_to_depots(settle_steps: int = 60) -> None:
        """Both counterweights out of the open-top tray to the ground depots (the
        solve's transport teleports)."""
        for body, key in ((scene.block_a, "a"), (scene.block_b, "b")):
            dx, dy = DEPOT[key]
            teleport(body, rig_pose((dx, dy, BLOCK[2] / 2 + 0.003)),
                     scene.rig.data.root_quat_w, settle_steps=0)
        step(settle_steps)

    def wait_deployed(max_steps: int = 720) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.bridge_down()[0]) and bool(scene.settled(scene.leaf)[0]):
                break

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in
                   (scene.leaf, scene.block_a, scene.block_b, scene.ketchup,
                    scene.mustard)):
                break

    def all_finite() -> bool:
        bodies = [scene.rig, scene.pillar, scene.leaf, scene.block_a, scene.block_b,
                  scene.ketchup, scene.mustard]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    up_deg = math.degrees(THETA_UP)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    tilt0 = float(scene.leaf_tilt_deg()[0])
    check("settle: all states finite, bridge raised on its heel "
          f"(tilt={tilt0:+.1f} ~ {up_deg:.1f} deg), hinge intact, both blocks in the "
          "tray, both bottles standing far outside on the open ground",
          all_finite() and abs(tilt0 - up_deg) < 2.0
          and bool(scene.hinge_intact()[0])
          and bool(scene.in_tray(scene.block_a)[0])
          and bool(scene.in_tray(scene.block_b)[0])
          and float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 0]) < -0.3
          and float(scene._rig_local(scene.mustard.data.root_pos_w)[0, 0]) < -0.3)
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.rig.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        kl = scene._rig_local(scene.ketchup.data.root_pos_w)[0]
        ml = scene._rig_local(scene.mustard.data.root_pos_w)[0]
        reads.append((float(rp[0]), float(rp[1]), yaw, float(kl[0]), float(kl[1]),
                      float(ml[0]), float(ml[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (rig_x, rig_y, rig_yaw, ket_x, ket_y, "
          f"mus_x, mus_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: rig pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.10)
    ket_pos = arr[:, 4] > 0
    mus_pos = arr[:, 6] > 0
    check("randomization: the bottles' ground slots genuinely swap (readback: ketchup "
          f"lands on +y in {int(ket_pos.sum())}/8 seeds, both orders occur, and the "
          "mustard always takes the opposite slot)",
          0 < int(ket_pos.sum()) < 8 and bool(np.all(ket_pos != mus_pos)))
    check("randomization: per-bottle jitter varies within the slots (readback: "
          f"dket_x={spread[3]:.3f} dmus_x={spread[5]:.3f})",
          spread[3] > 0.02 and spread[5] > 0.02)

    # =========================== 6. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 7. the raised interlock blocks the road ====================
    env.reset(seed=35)
    settle_all(300)
    q_leaf = scene.leaf.data.root_quat_w
    lay = scene.leaf.data.root_pos_w + _qapply(
        q_leaf, torch.tensor([0.06, 0.0, DECK_T / 2 + BOT_R + 0.003],
                             device=device).expand(n, 3))
    # lying ALONG the channel (axis leaf-x): its 5.5 cm diameter fits the channel
    # and slides lengthwise up to the tip, where every exit undercuts 5.5 cm;
    # spawned mid-channel so the 2.5 m/s slam carries it right INTO the tip zone
    q_across = _qmul(q_leaf, _qy(torch.full((n,), math.pi / 2, device=device)))
    kick_v = _qapply(q_leaf, torch.tensor([2.5, 0.0, 0.0], device=device).expand(n, 3))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = lay
    st[:, 3:7] = q_across
    st[:, 7:10] = kick_v
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    max_leaf_x, ever_past = -1.0, False
    for _ in range(300):
        step(1)
        max_leaf_x = max(max_leaf_x,
                         float(scene._leaf_local(scene.ketchup.data.root_pos_w)[0, 0]))
        ever_past = ever_past or bool(scene.past_doorway(scene.ketchup)[0])
    s, ok = judge()
    report("ramp-slam", s, ok)
    check("mechanism (the raised interlock is real): the ketchup laid along the "
          f"RAISED deck channel and slammed 2.5 m/s up the slope reaches the "
          f"tongue-tip zone (max leaf-frame x {max_leaf_x:+.3f}) but never passes "
          "the doorway — every raised-machine exit undercuts the bottle; no success",
          max_leaf_x > 0.14 and not ever_past
          and not bool(scene.in_keep(scene.ketchup)[0]) and not ok)

    # =========================== 8. one counterweight still holds ===========================
    teleport(scene.block_a, rig_pose((DEPOT["a"][0], DEPOT["a"][1],
                                      BLOCK[2] / 2 + 0.003)),
             scene.rig.data.root_quat_w, settle_steps=360)
    tilt_one = float(scene.leaf_tilt_deg()[0])
    s, ok = judge()
    report("one-block", s, ok)
    check("mechanism (one block holds): block A alone lifted out — the bridge stays "
          f"RAISED on the single remaining counterweight (tilt={tilt_one:+.1f} deg "
          f">= {scene.cfg.raised_min_deg:.0f}); BOTH must be removed; no success",
          tilt_one >= scene.cfg.raised_min_deg
          and not bool(scene.in_tray(scene.block_a)[0])
          and bool(scene.in_tray(scene.block_b)[0]) and not ok)

    # =========================== 9. the empty tray deploys the bridge =======================
    teleport(scene.block_b, rig_pose((DEPOT["b"][0], DEPOT["b"][1],
                                      BLOCK[2] / 2 + 0.003)),
             scene.rig.data.root_quat_w, settle_steps=10)
    wait_deployed(720)
    tilt_dep = float(scene.leaf_tilt_deg()[0])
    s, ok = judge()
    report("self-deploy", s, ok)
    check("mechanism (the deploy is real): with the tray empty, gravity swings the "
          f"bridge level onto the doorway sill BY ITSELF (tilt {tilt_one:+.1f} -> "
          f"{tilt_dep:+.1f} deg) — no force was ever applied to the machine; deploy "
          f"credit latched (score={s:.3f}), no success (no cargo delivered)",
          bool(scene.bridge_down()[0]) and abs(tilt_dep) < scene.cfg.deploy_max_deg + 1e-6
          and s >= 0.45 - 1e-6 and not ok)

    # =========================== 10. the seed strategy fails ================================
    env.reset(seed=41)
    settle_all(300)
    drop = rig_pose((0.33, 0.0, ROOF_BOT + 0.012 + BOT_L / 2 + 0.08))
    teleport(scene.ketchup, drop, scene.rig.data.root_quat_w, settle_steps=300)
    kz = float(scene._rig_local(scene.ketchup.data.root_pos_w)[0, 2])
    s, ok = judge()
    report("roof-drop", s, ok)
    check("seed strategy fails: the bottle carried over the receptacle and dropped "
          f"(the seed's whole plan) lands ON the keep's roof (rig-frame z={kz:+.3f}) "
          f"— the keep is roofed; not inside, score ~0, no success (score={s:.3f})",
          not bool(scene.in_keep(scene.ketchup)[0]) and kz > ROOF_BOT - 0.005
          and s <= 0.02 and not ok)

    # =========================== 11. wrong object ===========================================
    teleport(scene.mustard, rig_pose((0.32, 0.0, FLOOR_TOP + BOT_L / 2 + 0.004)),
             scene.rig.data.root_quat_w, settle_steps=150)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object (identification): the YELLOW mustard standing inside the "
          "keep (placed through the wall by fiat) with the RED ketchup outside is "
          f"rejected — exclusion clause violated, no success (score={s:.3f})",
          not ok and not bool(scene.mustard_out()[0]) and s <= 0.02)

    # =========================== 12. ballast-hold cheat =====================================
    env.reset(seed=51)
    settle_all(300)
    piv = rig_pose((0.0, 0.0, PIV_Z))
    q_level = scene.rig.data.root_quat_w
    # construct: leaf held level by fiat, blocks placed in the LEVEL tray, cargo in
    for body, sy in ((scene.block_a, 1.0), (scene.block_b, -1.0)):
        loc = torch.tensor([(TRAY_X0 + TRAY_X1) / 2, sy * 0.0305,
                            DECK_T / 2 + BLOCK[2] / 2 + 0.003], device=device).expand(n, 3)
        teleport(body, piv + _qapply(q_level, loc), q_level, settle_steps=0)
    teleport(scene.ketchup, rig_pose((0.32, 0.0, FLOOR_TOP + BOT_L / 2 + 0.004)),
             q_level, settle_steps=0)
    for _ in range(150):  # kinematic hold: the "pressing the bridge down" state
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = piv
        st[:, 3:7] = q_level
        scene.leaf.write_root_state_to_sim(st, all_ids)
        step(1)
    s, ok = judge()
    report("ballast-held", s, ok)
    check("ballast-hold cheat: the leaf HELD level with BOTH blocks still in the "
          "tray and the ketchup inside the keep — the end state LOOKS complete but "
          "the tray-empty clause rejects it AND the latched score stays ~0 "
          f"(score={s:.3f}: deploy credit is gated on the tray being empty)",
          not ok and s <= 0.02 and bool(scene.in_keep(scene.ketchup)[0])
          and bool(scene.in_tray(scene.block_a)[0])
          and bool(scene.in_tray(scene.block_b)[0]))

    # =========================== 13. the ballasted leaf re-raises ===========================
    settle_all(600)
    tilt_back = float(scene.leaf_tilt_deg()[0])
    s, ok = judge()
    report("ballast-released", s, ok)
    check("mechanism (both ways): the hold released — the still-ballasted leaf "
          f"swings itself back RAISED (tilt 0 -> {tilt_back:+.1f} deg >= "
          f"{scene.cfg.raised_min_deg:.0f}): level+settled+ballasted is not a "
          "physical rest state; no success",
          tilt_back >= scene.cfg.raised_min_deg and not ok)

    # =========================== 14. acceptance construct ===================================
    blocks_to_depots(settle_steps=30)
    wait_deployed(720)
    s, ok = judge_accept()
    report("accept-construct", s, ok)
    check("acceptance construct: both blocks lifted out — the bridge deploys itself "
          "WITH the cargo already resting inside -> the genuine end state, success "
          f"TRUE (score={s:.3f})", ok and s >= 0.99
          and bool(scene.bridge_down()[0]) and bool(scene.tray_empty()[0]))
    snap = scene.get_state(all_ids)

    # =========================== 15-16. mustard clause flips success ========================
    teleport(scene.mustard, rig_pose((0.40, 0.0, FLOOR_TOP + BOT_L / 2 + 0.004)),
             scene.rig.data.root_quat_w, settle_steps=120)
    s, ok = judge()
    report("mustard-in", s, ok)
    check("exclusion clause: the mustard added into the keep beside the delivered "
          "ketchup flips success FALSE", not ok and not bool(scene.mustard_out()[0]))
    scene.set_state(snap, all_ids)
    step(30)
    s, ok = judge_accept()
    report("mustard-restored", s, ok)
    check("state restored via set_state -> success returns TRUE (15's rejection was "
          "the exclusion clause and nothing else)", ok)

    # =========================== 17-18. bridge clause flips success =========================
    shift = _qapply(scene.rig.data.root_quat_w,
                    torch.tensor([0.0, 0.55, 0.0], device=device).expand(n, 3))
    for body in (scene.pillar, scene.leaf):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = body.data.root_pos_w + shift
        st[:, 3:7] = body.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)
    step(60)
    s, ok = judge()
    report("bridge-aside", s, ok)
    check("bridge clause: the whole bridge (pillar+leaf, coherently) dragged aside "
          "off the doorway (cargo still inside... but the road no longer spans the "
          "moat) flips success FALSE (hinge-position clause)",
          not ok and not bool(scene.hinge_intact()[0]))
    scene.set_state(snap, all_ids)
    step(30)
    s, ok = judge_accept()
    report("bridge-restored", s, ok)
    check("state restored via set_state -> success returns TRUE (17's rejection was "
          "the bridge clause and nothing else)", ok)

    # =========================== 19. settle gate ============================================
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.ketchup.data.root_pos_w
    st[:, 3:7] = scene.ketchup.data.root_quat_w
    st[:, 7:10] = torch.tensor([0.0, 0.0, 0.35], device=device)
    st[:, 10:13] = torch.tensor([0.0, 0.0, 6.0], device=device)
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.ketchup.data.root_lin_vel_w[0].norm())
    av = float(scene.ketchup.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    moving = lv > scene.cfg.settle_lin or av > scene.cfg.settle_ang
    settle_all(300)
    s2, ok2 = judge_accept()
    report("re-settled", s2, ok2)
    check("settle gate: the delivered bottle kicked (lin={:.2f} m/s, ang={:.1f} "
          "rad/s) and judged immediately is NOT success (must be at rest); once "
          "resettled inside, success returns".format(lv, av), moving and not ok and ok2)

    # =========================== 20-21. near misses =========================================
    env.reset(seed=61)
    settle_all(300)
    blocks_to_depots(settle_steps=30)
    wait_deployed(720)
    teleport(scene.ketchup,
             rig_pose((-0.02, 0.0, PIV_Z + DECK_T / 2 + BOT_L / 2 + 0.004)),
             scene.rig.data.root_quat_w, settle_steps=150)
    s, ok = judge()
    report("on-deck", s, ok)
    check("near miss (stopped short): bridge deployed and the ketchup standing ON "
          f"the deck, never pushed through — partial credit only (score={s:.3f}), "
          "no success", not ok and s <= 0.45 + 1e-4
          and not bool(scene.past_doorway(scene.ketchup)[0]))
    teleport(scene.ketchup, rig_pose((WALL_X0 - BOT_R - 0.006, 0.10,
                                      BOT_L / 2 + 0.003)),
             scene.rig.data.root_quat_w, settle_steps=150)
    s, ok = judge()
    report("beside-wall", s, ok)
    check("near miss (right place, wrong side of the wall): the ketchup standing on "
          "the ground against the keep's pedestal face is outside the interior box — "
          "no success", not ok and not bool(scene.in_keep(scene.ketchup)[0]))

    # =========================== 22-23. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "acceptance construct, the flip-backs and the resettle",
          not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bascule_keep")
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
