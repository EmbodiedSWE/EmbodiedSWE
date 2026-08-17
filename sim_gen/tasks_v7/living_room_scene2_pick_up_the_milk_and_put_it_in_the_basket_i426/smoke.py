"""Smoke / rubric-REJECTION battery for BallastGateScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i426`) — NullRobot,
teleported probe states, RECORDED.

This is NOT a solution (solve.py — ballast the gate open, carry the milk through,
unload the ballast — is the acceptance evidence). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome as a judged state and
asserts the rubric REJECTS it, plus physics probes that prove the mechanism is real
(the roof genuinely bars the seed's top-drop, the closed gate genuinely resists both
pressed cargo and an under-weight load yet genuinely sinks under the ballast weight
and genuinely springs back shut). Three probes DO construct the genuine end state on
purpose — the acceptance construct (14) and the decoy-removed flip (18) judge it, and
the settle-kick (15) starts from it; every other judged point must stay
success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; gate pressed SHUT (readback
                            q > -0.01), milk outside; score ~0, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: vault xy + free yaw
                            vary; milk and juice poses vary;
  5.   null policy        — 300 idle steps -> score ~0, no success;
  6.   authored masses    — root_physx_view.get_masses readback: vault ~30 kg,
                            gate ~0.12 kg, milk 0.35, juice 0.45 (the custom
                            spawners' MassAPI really applied — the whole spring
                            balance rides on these numbers);
  7.   roof (seed means)  — the seed's strategy, physically denied: the milk
                            dropped from ABOVE the vault lands ON the roof
                            (readback z ~ roof height), never enters — no credit;
  8.   locked door        — the milk held at panel height and PRESSED 2 N against
                            the CLOSED gate for 2 s: the gate barely moves
                            (q >= -0.02 — friction drag < spring hold, the
                            anti-drop-chute probe: this door does NOT yield to
                            cargo), the milk never enters, no credit;
  9.   under-weight       — 0.8 N pressed DOWN on the gate for 2 s: it stays shut
                            (q >= -0.02) — the spring preload is real;
  10.  over-weight        — 3.0 N pressed DOWN: the gate visibly sinks (min q <=
                            -0.15, the 9/10 pair is non-vacuous), and RELEASED it
                            springs back shut on its own (q >= -0.02);
  11.  ballast opens      — the juice laid LYING on the tray: the gate sinks to
                            the bottom stop (q <= -0.20) and STAYS there hands-
                            free for 2.5 s; gate-open credit (~0.25), no success;
  12.  mid-corridor       — the milk posed mid-doorway at carry height, judged
                            immediately: NOT in the crib, no extra credit;
  13.  incomplete end     — milk settled in the crib but the juice STILL on the
                            tray (gate held open): success FALSE (the vault must
                            be RE-SEALED — delivery alone is not the task);
  14.  acceptance         — the juice moved off to the far floor: the spring
                            re-seals the gate on its own -> success TRUE (the
                            rubric accepts exactly the delivered-and-sealed end
                            state, reached by the mechanism itself);
  15.  settle gate        — the delivered milk kicked and judged immediately:
                            NOT success (must be at rest);
  16.  decoy on the roof  — the juice parked ON the vault roof: success FALSE
                            (decoy clause);
  17.  decoy inside       — the juice standing on the vault floor BESIDE the
                            crib: success FALSE (decoy clause);
  18.  decoy removed      — juice back on the far floor: success returns TRUE
                            (16/17 were the decoy clause and nothing else);
  19.  wrong object       — fresh reset, the JUICE in the crib while the milk
                            stands on the floor: score ~0, no success;
  20.  beside the crib    — the milk INSIDE the vault but on the floor BESIDE
                            the crib (a stable near-miss): no crib credit, no
                            success;
  21.  rejection audit    — success() was never True at any judged point EXCEPT
                            the constructed acceptance probes (14, 18);
  22.  final no-NaN       — all task-object states finite at the end.

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
        BASE_H, CRIB_CX, CRIB_PAD_T, GATE_PLANE_X, GATE_T, GATE_TRAVEL, GATE_Z0,
        MILK_H, MILK_W, ROOF_T, ROOF_Z0, TRAY_CX, TRAY_CY, TRAY_CZ, TRAY_T,
        _qapply, _qinv, _qmul, _qx, _qy, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BASE_H, CRIB_CX, CRIB_PAD_T, GATE_PLANE_X, GATE_T, GATE_TRAVEL, GATE_Z0,
        MILK_H, MILK_W, ROOF_T, ROOF_Z0, TRAY_CX, TRAY_CY, TRAY_CZ, TRAY_T,
        _qapply, _qinv, _qmul, _qx, _qy, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

TRAY_TOP_CLOSED = GATE_Z0 + TRAY_CZ + TRAY_T / 2  # 0.535 (vault frame, gate closed)
PAD_TOP = BASE_H + CRIB_PAD_T  # 0.266
CRIB_LIE_Z = PAD_TOP + MILK_W / 2 + 0.004  # milk lying on the crib pad
MILK_MG = 0.35 * 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_gate")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.50, -1.30, 0.95)) + o),
                                tuple(np.array((0.05, 0.00, 0.28)) + o),
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

    def gate_q() -> float:
        return float(scene.gate_q()[0])

    def report(tag: str, s: float, ok: bool) -> None:
        ml = scene.vault_local(scene.milk.data.root_pos_w)[0]
        ju = scene.vault_local(scene.juice.data.root_pos_w)[0]
        print(f"[smoke] {tag:18s} | q={gate_q():+.4f} "
              f"milk_v=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"juice_v=({float(ju[0]):+.3f},{float(ju[1]):+.3f},{float(ju[2]):+.3f}) "
              f"in_crib={bool(scene.milk_in_crib()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"open={bool(scene.gate_open_deep()[0])} "
              f"decoy_out={bool(scene.decoy_out()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def vault_pose(local, extra_quat=None):
        q_vault = scene.vault.data.root_quat_w
        pos = scene.vault.data.root_pos_w + _qapply(
            q_vault, torch.tensor(local, device=device).expand(n, 3))
        q = q_vault if extra_quat is None else _qmul(q_vault, extra_quat)
        return pos, q

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

    def settle_all(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.juice)[0]) \
                    and bool(scene.settled(scene.gate)[0]):
                break

    def clear_forces() -> None:
        scene.milk.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def all_finite() -> bool:
        bodies = [scene.vault, scene.gate, scene.milk, scene.juice]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    up_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    half_pi = torch.full((n,), math.pi / 2, device=device)
    lie_x_quat = _qy(half_pi)  # carton long axis along vault x
    lie_y_quat = _qx(half_pi)  # carton long axis along vault y (the tray pose)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    check("settle: all states finite, gate pressed SHUT (readback "
          f"q={gate_q():+.4f} > -0.01), milk outside the crib",
          all_finite() and gate_q() > -0.01 and not bool(scene.milk_in_crib()[0]))
    check(f"settle: score ~0 and no success on a fresh reset (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(20)
        vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        ex = _qapply(scene.vault.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        ml = (scene.milk.data.root_pos_w - scene.env_origins)[0]
        ju = (scene.juice.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(vp[0]), float(vp[1]), yaw, float(ml[0]), float(ml[1]),
                      float(ju[0]), float(ju[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (vault_x, vault_y, vault_yaw, milk_x, "
          f"milk_y, ju_x, ju_y):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: vault pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: milk and juice poses vary (readback: "
          f"dmilk={max(spread[3], spread[4]):.3f} djuice={max(spread[5], spread[6]):.3f})",
          max(spread[3], spread[4]) > 0.10 and max(spread[5], spread[6]) > 0.10)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 6. authored masses are real ================================
    # The whole spring balance (2.4 N preload vs 1.18 N gate vs 4.4 N ballast) rides
    # on the custom spawners' MassAPI being applied — read the masses back.
    masses = {nm: float(getattr(scene, nm).root_physx_view.get_masses().flatten()[0])
              for nm in ("vault", "gate", "milk", "juice")}
    print(f"[smoke] mass readback: {masses}", flush=True)
    check("authored masses applied (readback: vault={vault:.1f}kg~30, gate={gate:.3f}"
          "kg~0.12, milk={milk:.2f}kg~0.35, juice={juice:.2f}kg~0.45)".format(**masses),
          abs(masses["vault"] - 30.0) < 3.0 and abs(masses["gate"] - 0.12) < 0.012
          and abs(masses["milk"] - 0.35) < 0.035
          and abs(masses["juice"] - 0.45) < 0.045)

    # =========================== 7. the roof denies the seed's top-drop =====================
    # The seed's whole delivery — carry the item OVER the container and release it
    # from above — attempted as physics: the milk dropped over the vault's center.
    settle_all(300)
    pos, q = vault_pose((0.0, 0.0, ROOF_Z0 + ROOF_T + MILK_W / 2 + 0.05), lie_x_quat)
    teleport(scene.milk, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("roof-drop", s, ok)
    ml = scene.vault_local(scene.milk.data.root_pos_w)[0]
    check("roof (the seed's means, denied): the milk dropped from ABOVE the vault "
          f"lands ON the roof (readback z={float(ml[2]) * 1000:.0f}mm >= "
          f"{(ROOF_Z0 - 0.02) * 1000:.0f}mm) — never enters, no credit, no success "
          f"(score={s:.3f})",
          float(ml[2]) >= ROOF_Z0 - 0.02 and not bool(scene.milk_in_crib()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. the closed gate is weight-LOCKED ========================
    # The drop-chute move — press the cargo against the door and let it yield — done
    # as physics: the milk held at panel height and pressed 2 N against the CLOSED
    # gate. Friction drag (~0.4 N) is far below the spring hold (~1.2 N net).
    env.reset(seed=41)
    settle_all(300)
    press_x0 = GATE_PLANE_X + GATE_T / 2 + MILK_H / 2 + 0.004
    pos, q = vault_pose((press_x0, 0.0, 0.40), lie_x_quat)
    teleport(scene.milk, pos, q, settle_steps=0)
    min_q, min_x = 0.0, press_x0
    for _ in range(240):
        q_vault = scene.vault.data.root_quat_w
        f_vault = torch.tensor([-2.0, 0.0, MILK_MG], device=device).expand(n, 3)
        f_world = _qapply(q_vault, f_vault)
        f_body = _qapply(_qinv(scene.milk.data.root_quat_w), f_world)
        scene.milk.set_external_force_and_torque(f_body.view(n, 1, 3), zero_wrench,
                                                 env_ids=all_ids)
        step(1)
        min_q = min(min_q, gate_q())
        min_x = min(min_x, float(scene.vault_local(scene.milk.data.root_pos_w)[0, 0]))
    clear_forces()
    settle_all(300)
    s, ok = judge()
    report("locked-door", s, ok)
    check("locked door: 2 N of cargo press against the CLOSED gate for 2 s barely "
          f"moves it (min q={min_q:+.4f} >= -0.02) and the milk never enters (min "
          f"center x={min_x * 1000:.0f}mm >= 250) — this door does NOT yield to "
          f"pushed cargo (score={s:.3f})",
          min_q >= -0.02 and min_x >= 0.250 and s <= 0.02 and not ok)

    # =========================== 9-10. under- / over-weight pair ============================
    # milk out of the way first (it fell at the door base during 8)
    pos, q = vault_pose((0.55, -0.55, MILK_H / 2 + 0.003))
    teleport(scene.milk, pos, up_quat, settle_steps=60)
    min_q = 0.0
    for _ in range(240):
        f_dn = torch.tensor([0.0, 0.0, -0.8], device=device).expand(n, 3)
        scene.gate.set_external_force_and_torque(f_dn.view(n, 1, 3), zero_wrench,
                                                 env_ids=all_ids)
        step(1)
        min_q = min(min_q, gate_q())
    clear_forces()
    settle_all(240)
    check(f"under-weight: 0.8 N pressed down on the gate leaves it shut (min "
          f"q={min_q:+.4f} >= -0.02) — the spring preload is real", min_q >= -0.02)
    min_q = 0.0
    for _ in range(360):
        f_dn = torch.tensor([0.0, 0.0, -3.0], device=device).expand(n, 3)
        scene.gate.set_external_force_and_torque(f_dn.view(n, 1, 3), zero_wrench,
                                                 env_ids=all_ids)
        step(1)
        min_q = min(min_q, gate_q())
    clear_forces()
    settle_all(360)
    s, ok = judge()
    report("weight-pair", s, ok)
    check(f"over-weight: 3.0 N visibly sinks the gate (min q={min_q:+.4f} <= -0.15, "
          f"the pair is non-vacuous) and RELEASED it springs back shut on its own "
          f"(q={gate_q():+.4f} >= -0.02)", min_q <= -0.15 and gate_q() >= -0.02)

    # =========================== 11. the ballast opens it hands-free ========================
    pos, q = vault_pose((GATE_PLANE_X + TRAY_CX, TRAY_CY,
                         TRAY_TOP_CLOSED + MILK_W / 2 + 0.006), lie_y_quat)
    teleport(scene.juice, pos, q, settle_steps=0)
    settle_all(900)
    q_after_sink = gate_q()
    step(300)  # hands-free hold: 2.5 s, nothing touching anything
    s, ok = judge()
    report("ballasted", s, ok)
    check("ballast opens: the juice laid on the tray sank the gate to the bottom "
          f"stop (q={q_after_sink:+.4f} <= -0.20) and it STAYED open hands-free "
          f"for 2.5 s (q={gate_q():+.4f}) — gate-open credit (score={s:.3f} in "
          f"[0.24,0.26]), no success",
          q_after_sink <= -0.20 and gate_q() <= -0.20
          and 0.24 <= s <= 0.26 and not ok)

    # =========================== 12. mid-corridor is NOT in the crib ========================
    pos, q = vault_pose((0.19, 0.0, 0.39), lie_x_quat)
    teleport(scene.milk, pos, q, settle_steps=2)  # judge mid-transit
    s, ok = judge()
    report("mid-corridor", s, ok)
    check("mid-corridor: the milk posed mid-doorway at carry height earns no crib "
          f"credit (score={s:.3f} <= 0.26, gate-open credit only), no success",
          not bool(scene.milk_in_crib()[0]) and s <= 0.26 and not ok)

    # =========================== 13. delivered but NOT re-sealed ============================
    pos, q = vault_pose((CRIB_CX, 0.0, CRIB_LIE_Z), lie_x_quat)
    teleport(scene.milk, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("not-resealed", s, ok)
    check("incomplete end state: milk settled in the crib but the juice STILL on "
          f"the tray holds the gate open (q={gate_q():+.4f}) -> success FALSE — "
          f"delivery alone is not the task, the vault must be re-sealed "
          f"(score={s:.3f} in [0.64,0.66])",
          bool(scene.milk_in_crib()[0]) and not bool(scene.gate_closed()[0])
          and 0.64 <= s <= 0.66 and not ok)

    # =========================== 14. acceptance construct ===================================
    pos, q = vault_pose((0.75, -0.45, MILK_H / 2 + 0.003))
    teleport(scene.juice, pos, up_quat, settle_steps=0)
    settle_all(600)
    s, ok = judge_accept()
    report("accept", s, ok)
    check("acceptance: ballast removed -> the spring re-sealed the gate on its own "
          f"(q={gate_q():+.4f} > -0.01), milk in the crib, decoy away -> success "
          f"TRUE (score={s:.3f})", ok and s >= 0.99 and gate_q() > -0.01)

    # =========================== 15. settle gate ============================================
    pos = scene.milk.data.root_pos_w
    q = scene.milk.data.root_quat_w
    teleport(scene.milk, pos, q, vel=[0.0, 0.0, 0.35], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.milk.data.root_lin_vel_w[0].norm())
    av = float(scene.milk.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check("settle gate: the delivered milk kicked (lin={:.2f} m/s, ang={:.1f} "
          "rad/s) and judged immediately is NOT success (must be at rest)"
          .format(lv, av), (lv > c.settle_lin or av > c.settle_ang) and not ok)
    settle_all(420)

    # =========================== 16-18. decoy clause flips success ==========================
    pos, q = vault_pose((0.0, 0.0, ROOF_Z0 + ROOF_T + MILK_H / 2 + 0.004))
    teleport(scene.juice, pos, up_quat, settle_steps=150)
    s, ok = judge()
    report("decoy-roof", s, ok)
    check("decoy on the roof: the juice parked ON the vault roof flips success "
          "FALSE (decoy clause), milk untouched",
          not ok and not bool(scene.decoy_out()[0]) and bool(scene.milk_in_crib()[0]))
    pos, q = vault_pose((0.0, 0.148, BASE_H + MILK_H / 2 + 0.004))
    teleport(scene.juice, pos, q, settle_steps=0)
    settle_all(300)
    s, ok = judge()
    report("decoy-inside", s, ok)
    check("decoy inside: the juice standing on the vault floor BESIDE the crib "
          "flips success FALSE (decoy clause)",
          not ok and not bool(scene.decoy_out()[0]) and bool(scene.milk_in_crib()[0]))
    pos, q = vault_pose((0.75, -0.45, MILK_H / 2 + 0.003))
    teleport(scene.juice, pos, up_quat, settle_steps=150)
    s, ok = judge_accept()
    report("decoy-removed", s, ok)
    check("decoy removed: juice back on the far floor -> success returns TRUE "
          "(16/17 were the decoy clause and nothing else)", ok)

    # =========================== 19. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    pos, q = vault_pose((CRIB_CX, 0.0, CRIB_LIE_Z), lie_x_quat)
    teleport(scene.juice, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object: the JUICE in the crib while the milk stands on the floor "
          f"is rejected twice over — milk not in the crib AND the decoy clause "
          f"(score={s:.3f} <= 0.02)",
          not ok and s <= 0.02 and not bool(scene.milk_in_crib()[0])
          and not bool(scene.decoy_out()[0]))

    # =========================== 20. inside the vault is not in the crib ====================
    env.reset(seed=91)
    settle_all(300)
    pos, q = vault_pose((0.0, 0.148, BASE_H + MILK_H / 2 + 0.004))
    teleport(scene.milk, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("beside-crib", s, ok)
    ml = scene.vault_local(scene.milk.data.root_pos_w)[0]
    check("beside the crib: the milk INSIDE the vault but standing on the floor "
          f"beside the crib (readback y={float(ml[1]) * 1000:.0f}mm, a stable "
          f"near-miss) earns no crib credit (score={s:.3f} <= 0.02), no success",
          not bool(scene.milk_in_crib()[0]) and s <= 0.02 and not ok)

    # =========================== 21-22. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probes", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_gate")
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
