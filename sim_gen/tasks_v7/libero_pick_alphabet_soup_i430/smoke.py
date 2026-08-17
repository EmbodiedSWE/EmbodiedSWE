"""Smoke / rubric-REJECTION battery for PressLatchVaultScene (sim_gen task
`libero_pick_alphabet_soup_i430`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — bridge both pin caps with the bar, press, let the
gate drive itself open, push the can through — is the acceptance evidence). Every
teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a
judged state and asserts the rubric REJECTS it, plus physics probes that prove the
interlock is real: the full roof genuinely bars the seed's top-drop; the closed gate
genuinely resists pushed cargo and a direct sideways shove; ONE pressed pin (either
one, and even both pressed one-AFTER-the-other) genuinely does NOT release the gate —
only a SIMULTANEOUS two-point press does; parked weight on the caps genuinely cannot
press them out; the short rod genuinely cannot bridge the caps; and once open, the
ratchet genuinely refuses to re-seal. Two probes DO construct the genuine end state
on purpose — the direct-press positive control (15) proves the mechanism opens, and
the acceptance construct (18) judges success TRUE; every other judged point must stay
success()=False and a final audit asserts exactly that.

  1-2.  settle/no-NaN     — reset settles finite; gate SEALED (readback q < 0.02,
                            the drive is stopped by the pinned strips), both pins UP,
                            can outside; score ~0, no success;
  3-4.  randomization     — READBACK over 8 seeded resets: vault xy + free yaw vary;
                            red can / bar / rod / green can world poses vary;
  5.   null policy        — 300 idle steps -> score ~0, no success;
  6.   authored masses    — root_physx_view.get_masses readback: vault ~30 kg, gate
                            ~0.30, pins ~0.20 each, bar 0.60, rod 0.25, can 0.30
                            (the whole force budget rides on these numbers);
  7.   roof (seed means)  — the seed's strategy, physically denied: the can dropped
                            from ABOVE the vault lands ON the roof and never enters;
  8.   cargo vs door      — the can pressed 2 N against the CLOSED gate for 2 s:
                            the door is not its DOF — the can never crosses the gate
                            plane, the gate stays sealed, no credit;
  9-10. single pin        — 15 N pressed on pin A alone (then B alone): the pin
                            bottoms out but the OTHER collar stops the gate within
                            millimetres (q <= 0.02); no unlock latch; the released
                            pin springs back up;
  11.  sequential press   — pin A pressed FULLY and released, THEN pin B pressed
                            fully and released: both pins were down, but never AT
                            THE SAME TIME -> the gate never opens (max q <= 0.02),
                            no unlock latch — the simultaneity requirement is real;
  12.  gate shove         — 5 N sideways on the gate itself (its DOF) for 2 s on
                            top of its own drive: the pinned strips hold it sealed
                            (max q <= 0.02), score still ~0;
  13.  parked weight      — the bar RESTED across both caps hands-free: pin preload
                            ignores it (pins stay far above the clear line), gate
                            sealed, no unlock latch — pressing must be ACTIVE;
  14.  short rod          — the rod teleported to bridge the caps: it is too short,
                            falls between the stands (readback z far below the cap
                            band), pins untouched, gate sealed;
  15.  positive control   — 15 N pressed on BOTH pins at once: the gate drives
                            ITSELF fully open (q >= 0.15), unlock+open latches set,
                            gate-open credit (score 0.50), NOT success (can outside);
  16.  ratchet            — 15 N shoving the OPEN gate back toward closed for 2 s:
                            the popped-up pins block it far short of sealed (min q
                            >= 0.02 — the doorway can never be re-sealed), and
                            released, the drive re-opens it (q >= 0.15);
  17.  doorway near-miss  — the can posed IN the doorway (settled, straddling the
                            sill): outside the interior window -> no entry credit
                            (score stays 0.50), no success;
  18.  acceptance         — the can placed standing on the vault floor: settles ->
                            success TRUE, score 1.0 (the rubric accepts exactly the
                            delivered end state);
  19.  settle gate        — the delivered can kicked and judged immediately: NOT
                            success (must be at rest);
  20.  wrong object       — fresh reset, the GREEN decoy can inside the vault while
                            the red can stands on the porch: score ~0, no success;
  21.  order gate         — fresh reset, the gate TELEPORTED fully open without any
                            pin ever pressed: open door alone earns NO credit
                            (score ~0 — the unlock latch gates the open latch);
  22.  rejection audit    — success() was never True at any judged point EXCEPT the
                            constructed acceptance probe (18);
  23.  final no-NaN       — all task-object states finite at the end.

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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from .scene import (  # noqa: F401
        BAR_H, CAN_H, CAN_R, CAP_C, CAP_ZB, CAP_ZT, GATE_TRAVEL, GATE_X1, PIN_TRAVEL,
        PLINTH_H, PORCH_TOP, ROOF_T, ROOF_Z0, _qapply, _qinv, _qmul, _qz,
    )
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import (  # noqa: F401
        BAR_H, CAN_H, CAN_R, CAP_C, CAP_ZB, CAP_ZT, GATE_TRAVEL, GATE_X1, PIN_TRAVEL,
        PLINTH_H, PORCH_TOP, ROOF_T, ROOF_Z0, _qapply, _qinv, _qmul, _qz,
    )
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOVER_Z = CAP_ZT + BAR_H / 2 + 0.008   # bar resting height on the caps + drop gap
CAN_STAND_Z = PLINTH_H + CAN_H / 2 + 0.003


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.press_latch_vault")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.10, 0.85)) + o),
                                tuple(np.array((0.08, 0.00, 0.13)) + o),
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

    def pins_q() -> tuple[float, float]:
        return float(scene.pin_q(scene.pin_a)[0]), float(scene.pin_q(scene.pin_b)[0])

    def latches() -> tuple[bool, bool, bool]:
        return (bool(scene.l_unlock[0]), bool(scene.l_open[0]), bool(scene.l_enter[0]))

    def report(tag: str, s: float, ok: bool) -> None:
        cv = scene.vault_local(scene.can_red.data.root_pos_w)[0]
        pa, pb = pins_q()
        print(f"[smoke] {tag:16s} | q={gate_q():+.4f} pins=({pa:+.4f},{pb:+.4f}) "
              f"can_v=({float(cv[0]):+.3f},{float(cv[1]):+.3f},{float(cv[2]):+.3f}) "
              f"in={bool(scene.can_in_vault()[0])} L={latches()} "
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
            if bool(scene.settled(scene.can_red)[0]) and bool(scene.settled(scene.bar)[0]) \
                    and bool(scene.settled(scene.gate)[0]) \
                    and bool(scene.settled(scene.rod)[0]):
                break

    def clear_forces() -> None:
        for body in (scene.can_red, scene.can_green, scene.bar, scene.rod,
                     scene.gate, scene.pin_a, scene.pin_b):
            body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_world(body, f_world_vec) -> None:
        """One substep's external force, WORLD vector applied in the BODY frame
        (set_external_force_and_torque is body-frame; world refs go stale)."""
        f_w = torch.tensor(f_world_vec, device=device).expand(n, 3)
        f_b = _qapply(_qinv(body.data.root_quat_w), f_w)
        body.set_external_force_and_torque(f_b.view(n, 1, 3), zero_wrench, env_ids=all_ids)

    def push_vault_frame(body, f_vault_vec) -> None:
        """One substep's external force, VAULT-frame vector -> world -> body frame."""
        f_v = torch.tensor(f_vault_vec, device=device).expand(n, 3)
        f_w = _qapply(scene.vault.data.root_quat_w, f_v)
        f_b = _qapply(_qinv(body.data.root_quat_w), f_w)
        body.set_external_force_and_torque(f_b.view(n, 1, 3), zero_wrench, env_ids=all_ids)

    def all_finite() -> bool:
        bodies = [scene.vault, scene.gate, scene.pin_a, scene.pin_b,
                  scene.bar, scene.rod, scene.can_red, scene.can_green]
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    settle_all(480)
    s, ok = judge()
    report("reset", s, ok)
    pa, pb = pins_q()
    check("settle: all states finite, gate SEALED (readback "
          f"q={gate_q():+.4f} <= {c.gate_shut_q}), both pins UP "
          f"(({pa:+.4f},{pb:+.4f}) >= -0.005), can outside the vault",
          all_finite() and gate_q() <= c.gate_shut_q and pa >= -0.005 and pb >= -0.005
          and not bool(scene.can_in_vault()[0]))
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
        cr = (scene.can_red.data.root_pos_w - scene.env_origins)[0]
        br = (scene.bar.data.root_pos_w - scene.env_origins)[0]
        rd = (scene.rod.data.root_pos_w - scene.env_origins)[0]
        gc = (scene.can_green.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(vp[0]), float(vp[1]), yaw, float(cr[0]), float(cr[1]),
                      float(br[0]), float(br[1]), float(rd[0]), float(rd[1]),
                      float(gc[0]), float(gc[1])))
    arr = np.array(reads)
    print("[smoke] randomization readback (vault_x, vault_y, yaw, can_xy, bar_xy, "
          f"rod_xy, green_xy):\n{arr.round(3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: vault pose varies across seeded resets (readback: "
          f"dx={spread[0]:.3f} dy={spread[1]:.3f} dyaw={spread[2]:.2f} rad)",
          spread[0] > 0.02 and spread[1] > 0.02 and spread[2] > 0.8)
    check("randomization: can/bar/rod/green world poses vary (readback spreads: "
          f"can={max(spread[3], spread[4]):.3f} bar={max(spread[5], spread[6]):.3f} "
          f"rod={max(spread[7], spread[8]):.3f} green={max(spread[9], spread[10]):.3f})",
          max(spread[3], spread[4]) > 0.10 and max(spread[5], spread[6]) > 0.10
          and max(spread[7], spread[8]) > 0.10 and max(spread[9], spread[10]) > 0.10)

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(300)
    s, ok = judge()
    report("null-policy", s, ok)
    check(f"null policy: score ~0 and no success after 300 idle steps (score={s:.3f})",
          s <= 0.02 and not ok)

    # =========================== 6. authored masses are real ================================
    # The whole interlock budget (7 N pin preload vs 5.9 N bar weight; ~20 N to clear
    # both pins vs 14 N of ALL loose parts) rides on the spawners' MassAPI.
    masses = {nm: float(getattr(scene, nm).root_physx_view.get_masses().flatten()[0])
              for nm in ("vault", "gate", "pin_a", "pin_b", "bar", "rod", "can_red")}
    print(f"[smoke] mass readback: {masses}", flush=True)
    check("authored masses applied (readback: vault={vault:.1f}kg~30, gate={gate:.2f}"
          "kg~0.30, pins=({pin_a:.2f},{pin_b:.2f})kg~0.20, bar={bar:.2f}kg~0.60, "
          "rod={rod:.2f}kg~0.25, can={can_red:.2f}kg~0.30)".format(**masses),
          abs(masses["vault"] - 30.0) < 3.0 and abs(masses["gate"] - 0.30) < 0.03
          and abs(masses["pin_a"] - 0.20) < 0.02 and abs(masses["pin_b"] - 0.20) < 0.02
          and abs(masses["bar"] - 0.60) < 0.06 and abs(masses["rod"] - 0.25) < 0.025
          and abs(masses["can_red"] - 0.30) < 0.03)

    # =========================== 7. the roof denies the seed's top-drop =====================
    # The seed's whole delivery — carry the item over the container and release from
    # above — attempted as physics: the can dropped over the vault's center.
    pos, q = vault_pose((0.0, 0.0, ROOF_Z0 + ROOF_T + CAN_H / 2 + 0.015))
    teleport(scene.can_red, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("roof-drop", s, ok)
    cv = scene.vault_local(scene.can_red.data.root_pos_w)[0]
    check("roof (the seed's means, denied): the can dropped from ABOVE the vault "
          f"lands ON the roof (readback z={float(cv[2]) * 1000:.0f}mm >= "
          f"{ROOF_Z0 * 1000:.0f}mm) — never enters, no credit, no success "
          f"(score={s:.3f})",
          float(cv[2]) >= ROOF_Z0 and not bool(scene.can_in_vault()[0])
          and s <= 0.02 and not ok)

    # =========================== 8. pushed cargo cannot breach the door =====================
    env.reset(seed=41)
    settle_all(300)
    press_x0 = GATE_X1 + CAN_R + 0.004
    pos, q = vault_pose((press_x0, 0.0, PORCH_TOP + CAN_H / 2 + 0.003))
    teleport(scene.can_red, pos, q, settle_steps=20)
    min_x, max_q = press_x0, gate_q()
    for _ in range(240):
        push_vault_frame(scene.can_red, (-2.0, 0.0, 0.0))
        step(1)
        min_x = min(min_x, float(scene.vault_local(scene.can_red.data.root_pos_w)[0, 0]))
        max_q = max(max_q, gate_q())
    clear_forces()
    settle_all(300)
    s, ok = judge()
    report("cargo-vs-door", s, ok)
    check("cargo vs door: 2 N pressing the can against the CLOSED gate for 2 s — "
          f"the can never crosses the gate plane (min center x={min_x * 1000:.0f}mm "
          f">= 160) and the gate stays sealed (max q={max_q:+.4f} <= "
          f"{c.gate_shut_q}); no credit (score={s:.3f})",
          min_x >= 0.160 and max_q <= c.gate_shut_q and s <= 0.02 and not ok)
    # Probe 8 left the can pressed against the gate at x~0.172 — directly under the
    # cap zone where 13/14 drop the bar/rod. Return it to its porch station first.
    pos, q = vault_pose((0.29, 0.0, PORCH_TOP + CAN_H / 2 + 0.003))
    teleport(scene.can_red, pos, q, settle_steps=45)

    # =========================== 9-10. one pin alone does NOT unlock ========================
    for tag, pin in (("A", scene.pin_a), ("B", scene.pin_b)):
        max_q, min_p = gate_q(), 0.0
        for _ in range(180):
            push_world(pin, (0.0, 0.0, -15.0))
            step(1)
            max_q = max(max_q, gate_q())
            min_p = min(min_p, float(scene.pin_q(pin)[0]))
        clear_forces()
        step(90)
        p_back = float(scene.pin_q(pin)[0])
        s, ok = judge()
        report(f"single-pin-{tag}", s, ok)
        check(f"single pin {tag}: 15 N bottoms the pin out (min q={min_p:+.4f} <= "
              f"-0.020) yet the OTHER collar stops the gate within millimetres "
              f"(max q={max_q:+.4f} <= {c.gate_shut_q}); no unlock latch; released, "
              f"the pin springs back up (q={p_back:+.4f} >= -0.010)",
              min_p <= -0.020 and max_q <= c.gate_shut_q
              and not bool(scene.l_unlock[0]) and p_back >= -0.010)

    # =========================== 11. sequential press is NOT simultaneous ===================
    max_q = gate_q()
    both_down = []
    for pin in (scene.pin_a, scene.pin_b):
        for _ in range(150):
            push_world(pin, (0.0, 0.0, -15.0))
            step(1)
            max_q = max(max_q, gate_q())
            pa, pb = pins_q()
            both_down.append(pa <= c.pin_clear_q and pb <= c.pin_clear_q)
        clear_forces()
        step(90)
        max_q = max(max_q, gate_q())
    s, ok = judge()
    report("sequential", s, ok)
    check("sequential press: pin A pressed FULLY and released, THEN pin B — both "
          "were down, never at the same time (readback confirms) -> the gate never "
          f"opened (max q={max_q:+.4f} <= {c.gate_shut_q}), no unlock latch — the "
          f"simultaneity requirement is real (score={s:.3f})",
          not any(both_down) and max_q <= c.gate_shut_q
          and not bool(scene.l_unlock[0]) and s <= 0.02 and not ok)

    # =========================== 12. shoving the gate itself fails ==========================
    max_q = gate_q()
    for _ in range(240):
        push_vault_frame(scene.gate, (0.0, 5.0, 0.0))
        step(1)
        max_q = max(max_q, gate_q())
    clear_forces()
    step(60)
    s, ok = judge()
    report("gate-shove", s, ok)
    check("gate shove: 5 N sideways on the gate (its own DOF, on top of its drive) "
          f"for 2 s — the pinned strips hold it sealed (max q={max_q:+.4f} <= "
          f"{c.gate_shut_q}), score still ~0 (score={s:.3f})",
          max_q <= c.gate_shut_q and s <= 0.02 and not ok)

    # =========================== 13. parked weight cannot unlock ============================
    pos, q = vault_pose((CAP_C, 0.0, HOVER_Z))
    teleport(scene.bar, pos, q, settle_steps=120)
    pa, pb = pins_q()
    s, ok = judge()
    report("parked-bar", s, ok)
    check("parked weight: the bar RESTED across both caps hands-free — the pin "
          f"preload ignores it (pins=({pa:+.4f},{pb:+.4f}) >= -0.015, far above the "
          f"clear line {c.pin_clear_q}), gate sealed (q={gate_q():+.4f}), no unlock "
          "latch — pressing must be ACTIVE",
          pa >= -0.015 and pb >= -0.015 and gate_q() <= c.gate_shut_q
          and not bool(scene.l_unlock[0]))
    pos, q = vault_pose((0.55, 0.45, BAR_H / 2 + 0.004))
    teleport(scene.bar, pos, q, settle_steps=45)  # park the bar clear

    # =========================== 14. the short rod cannot bridge ============================
    pos, q = vault_pose((CAP_C, 0.0, HOVER_Z))
    teleport(scene.rod, pos, q, settle_steps=0)
    settle_all(300)
    rz = float(scene.vault_local(scene.rod.data.root_pos_w)[0, 2])
    pa, pb = pins_q()
    s, ok = judge()
    report("short-rod", s, ok)
    check("short rod: teleported to bridge the caps, it is too short and falls "
          f"between the stands (readback z={rz * 1000:.0f}mm <= "
          f"{(CAP_ZB - 0.010) * 1000:.0f}mm), pins untouched "
          f"(({pa:+.4f},{pb:+.4f}) >= -0.010), gate sealed (q={gate_q():+.4f})",
          rz <= CAP_ZB - 0.010 and pa >= -0.010 and pb >= -0.010
          and gate_q() <= c.gate_shut_q)
    pos, q = vault_pose((0.55, -0.45, BAR_H / 2 + 0.004))
    teleport(scene.rod, pos, q, settle_steps=45)  # park the rod clear

    # =========================== 15. positive control: simultaneous press ===================
    opened = False
    for i in range(360):
        push_world(scene.pin_a, (0.0, 0.0, -15.0))
        push_world(scene.pin_b, (0.0, 0.0, -15.0))
        step(1)
        if gate_q() >= 0.160:
            opened = True
            break
    clear_forces()
    step(90)
    s, ok = judge()
    report("both-pins", s, ok)
    check("positive control: 15 N on BOTH pins at once — the gate drove ITSELF "
          f"fully open (q={gate_q():+.4f} >= {c.gate_open_q}), unlock+open latches "
          f"set, gate-open credit (score={s:.3f} in [0.49,0.51]), NOT success "
          "(the can is still outside)",
          opened and gate_q() >= c.gate_open_q and bool(scene.l_unlock[0])
          and bool(scene.l_open[0]) and 0.49 <= s <= 0.51 and not ok)

    # =========================== 16. the ratchet refuses to re-seal =========================
    min_q = gate_q()
    for _ in range(240):
        push_vault_frame(scene.gate, (0.0, -15.0, 0.0))
        step(1)
        min_q = min(min_q, gate_q())
    clear_forces()
    settle_all(240)
    s, ok = judge()
    report("ratchet", s, ok)
    check("ratchet: 15 N shoving the OPEN gate back toward closed for 2 s — the "
          f"popped-up pins block it far short of sealed (min q={min_q:+.4f} >= "
          f"{c.gate_shut_q}: the doorway can never be re-sealed), and released, the "
          f"drive re-opens it (q={gate_q():+.4f} >= {c.gate_open_q})",
          min_q >= c.gate_shut_q and gate_q() >= c.gate_open_q)

    # =========================== 17. the doorway is not the interior ========================
    pos, q = vault_pose((0.105, 0.0, CAN_STAND_Z))
    teleport(scene.can_red, pos, q, settle_steps=0)
    settle_all(300)
    s, ok = judge()
    report("doorway", s, ok)
    cv = scene.vault_local(scene.can_red.data.root_pos_w)[0]
    check("doorway near-miss: the can settled IN the doorway (readback "
          f"x={float(cv[0]) * 1000:.0f}mm > {c.in_x_max * 1000:.0f}mm window edge) "
          f"is NOT inside -> no entry credit (score={s:.3f} <= 0.51), no success",
          not bool(scene.can_in_vault()[0]) and s <= 0.51 and not ok)

    # =========================== 18. acceptance construct ===================================
    pos, q = vault_pose((0.0, 0.0, CAN_STAND_Z))
    teleport(scene.can_red, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge_accept()
    report("accept", s, ok)
    check("acceptance: the can placed standing on the vault floor settles inside "
          f"-> success TRUE, score={s:.3f} >= 0.99 (the rubric accepts exactly the "
          "delivered end state)", ok and s >= 0.99)

    # =========================== 19. settle gate ============================================
    pos = scene.can_red.data.root_pos_w
    q = scene.can_red.data.root_quat_w
    teleport(scene.can_red, pos, q, vel=[0.0, 0.0, 0.30], ang=[0.0, 0.0, 6.0],
             settle_steps=0)
    step(1)  # refresh buffers only — judge while still moving
    lv = float(scene.can_red.data.root_lin_vel_w[0].norm())
    av = float(scene.can_red.data.root_ang_vel_w[0].norm())
    s, ok = judge()
    report("settle-gate", s, ok)
    check("settle gate: the delivered can kicked (lin={:.2f} m/s, ang={:.1f} rad/s) "
          "and judged immediately is NOT success (must be at rest)"
          .format(lv, av), (lv > c.settle_lin or av > c.settle_ang) and not ok)
    settle_all(300)

    # =========================== 20. wrong object ===========================================
    env.reset(seed=81)
    settle_all(300)
    pos, q = vault_pose((0.0, 0.0, CAN_STAND_Z))
    teleport(scene.can_green, pos, q, settle_steps=0)
    settle_all(420)
    s, ok = judge()
    report("wrong-object", s, ok)
    check("wrong object: the GREEN decoy can inside the vault while the RED can "
          f"stands on the porch earns nothing (score={s:.3f} <= 0.02), no success",
          not bool(scene.can_in_vault()[0]) and s <= 0.02 and not ok)

    # =========================== 21. open door alone earns NO credit ========================
    env.reset(seed=91)
    settle_all(300)
    pos, q = vault_pose((0.0, GATE_TRAVEL, 0.0))
    teleport(scene.gate, pos, q, settle_steps=60)
    s, ok = judge()
    report("order-gate", s, ok)
    check("order gate: the gate TELEPORTED fully open without any pin ever pressed "
          f"(readback q={gate_q():+.4f} >= {c.gate_open_q}) earns NO credit "
          f"(score={s:.3f} <= 0.02 — the unlock latch gates the open latch), "
          "no success",
          gate_q() >= c.gate_open_q and s <= 0.02
          and not bool(scene.l_unlock[0]) and not bool(scene.l_open[0]) and not ok)

    # =========================== 22-23. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point except the "
          "constructed acceptance probe", not ever_bad_success[0])
    check("final: all task-object states finite (no NaN)", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.press_latch_vault")
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
