"""Smoke / rubric-REJECTION battery for RollAwayVaultScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claim the task rests on — the seated roller
SEALS the slot — is physically load-bearing. Every probe is CONSTRUCTED (teleport,
real physics steps, judge) — instrumentation, never a solution: no probe here reaches
success(), and check 15 audits exactly that across every stepped state.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; roller SEATED over the slot,
                           both cubes inside the cavity; score 0, no success;
   2. randomization      — two seeded resets: READBACK vault yaw, vault xy, pad xy and
                           roller seat x all differ;
   3. arrangement swap   — over 10 resets the GOLD prize occupies BOTH interior slots;
   4. null-policy        — 480 idle steps: the roller stays in its seat POCKET (the
                           GPU solver walks a free cylinder a few mm on its neutral
                           seat; the detent lip bounds the walk well inside the seat
                           window — `unseated` never latches), score ~0, no success;
   5. SEAL raid          — with the roller seated, the prize is shoved UPWARD through
                           the slot at 3x its weight (then up+sideways into the edge
                           crescent) for 2 s: it demonstrably RISES >= 15 mm, is
                           stopped by the roller, and never leaves the cavity — "the
                           box must be opened first" is physics, not fiat;
   6. bare opening       — the seed family's naive end state (box open, nothing
                           retrieved): roller CONSTRUCTED at rest in the dock, cubes
                           untouched -> no success, score <= the 0.35 opening credit.
                           (The seed's literal end state — a lid rotated open on its
                           hinge — is inexpressible here: the scene has no joint.);
   7. bypass             — prize CONSTRUCTED on the pad while the roller stays SEATED:
                           `out`/`placed` latch NOTHING (out requires an open mouth),
                           score ~0, no success — teleporting past the closure is
                           worthless by construction;
   8. near-miss coverage — roller stalled just past the lip (seat x ~72 mm, still
                           overlapping the slot), prize perfect on the pad, distractor
                           in — judged instantly: still `covered`, no success;
   9. near-miss place    — prize upright on the GROUND ~65 mm from the pad axis (just
                           outside `pad_xy_tol`), roller docked -> no success;
  10. wrong object       — GREY distractor delivered to the pad, GOLD prize left
                           inside, roller docked -> no success;
  11. restraint          — prize correctly delivered but the distractor ALSO taken
                           out: distractor-in clause refuses; score capped 0.75;
  12. settle gate        — the exact success layout with the prize sliding at
                           0.45 m/s, judged instantly (no stepping): refused;
  13. MECHANISM roll     — an escalating ~4-8 N CoM push (the solve's own act) pops
                           the roller over the lip; force CUT at the ramp edge; it
                           self-rolls down between the fences and rests in the dock
                           (|seat y| < 20 mm, ground height) -> `cleared` latches,
                           score 0.35;
  14. latch persistence  — the roller teleported BACK onto its seat: the mouth is
                           covered again, success stays False, but the latched 0.35
                           opening credit survives (credit never evaporates);
  15. rejection audit    — success() was never True at ANY stepped state of the
                           battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.open_box_i184.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qz = task_scene._qmul, task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"stepped": 0, "success_hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["stepped"] += 1
        if bool(env.scene.success()[0]):
            _AUDIT["success_hits"] += 1
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if vel_w is not None:
        st[:, 7:10] = vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    rl = scene._vault_local(scene.roller.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | roller=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
          f"{float(rl[2]):+.3f}) covered={bool(scene.mouth_covered()[0])} "
          f"seated={bool(scene.roller_seated()[0])} out={bool(scene.prize_out()[0])} "
          f"on_pad={bool(scene.prize_on_pad()[0])} "
          f"distr_in={bool(scene.distractor_in()[0])} "
          f"L(u/c/o/p)={int(scene._unseated[0])}{int(scene._cleared[0])}"
          f"{int(scene._out[0])}{int(scene._placed[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roll_away_vault")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.35, -0.85, 0.70)) + o),
                                tuple(np.array((0.50, -0.05, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def vault_world(loc_xyz) -> torch.Tensor:
        """Vault-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, loc)

    def vault_quat() -> torch.Tensor:
        _refresh()
        return scene.vault.data.root_quat_w

    def roller_loc() -> torch.Tensor:
        _refresh()
        return scene._vault_local(scene.roller.data.root_pos_w)[0]

    def dock_roller(settle: int = 90) -> None:
        """CONSTRUCT the roller at rest on the ground in the dock, then settle."""
        _write_body(scene.roller, vault_world((0.36, 0.0, c.roller_r + 0.002)),
                    vault_quat())
        _step(settle)

    def prize_to_pad(settle: int = 60) -> None:
        p = scene.pad.data.root_pos_w.clone()
        p[:, 2] += c.pad_t / 2 + c.cube_s / 2 + 0.010
        _write_body(scene.prize, p, None)
        _step(settle)

    def zero_v() -> torch.Tensor:
        return torch.zeros(n, 3, device=device)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; roller SEATED over the slot, both "
          "cubes inside the cavity; score 0, no success",
          bool(scene._finite()[0]) and bool(scene.roller_seated()[0])
          and bool(scene.mouth_covered()[0]) and not bool(scene.prize_out()[0])
          and bool(scene.distractor_in()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.vault.data.root_quat_w[0]),
                scene.vault.data.root_pos_w[0, :2].clone(),
                scene.pad.data.root_pos_w[0, :2].clone(),
                float(scene._vault_local(scene.roller.data.root_pos_w)[0, 0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_vp, a_pp, a_rx = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_vp, b_pp, b_rx = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_vp, d_pp = float((a_vp - b_vp).norm()), float((a_pp - b_pp).norm())
    d_rx = abs(a_rx - b_rx)
    print(f"[smoke] randomization deltas: vault_yaw={d_yawv:.1f}deg "
          f"vault_xy={d_vp * 1000:.1f}mm pad_xy={d_pp * 1000:.1f}mm "
          f"roller_x={d_rx * 1000:.1f}mm", flush=True)
    check("randomization-is-real: vault yaw, vault xy, pad xy, roller seat readback "
          "differ across seeds",
          d_yawv > 1.5 and d_vp > 0.003 and d_pp > 0.003 and d_rx > 0.001)

    # ================= 3. cube arrangement swap ===================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+" if float(scene.prize_slot[0]) > 0 else "-")
    print(f"[smoke] over 10 resets: prize slots {sorted(sides)}", flush=True)
    check("arrangement swap: the GOLD prize occupies BOTH interior slots over 10 resets",
          sides == {"+", "-"})

    # ================= 4. null policy fails (creep bounded by the detent pocket) ==================
    # A free cylinder on its flat seat is a neutral equilibrium and the GPU solver
    # walks it a few mm/s; the DETENT LIP exists precisely to bound that walk. The
    # honest claim is boundedness: over 4 s hands-off the roller must stay inside the
    # seat window (lip contact is at x ~20 mm < seat_x 25 mm < unseat_x 30 mm), the
    # `unseated` latch must never fire, and the score must stay ~0.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    x0 = float(roller_loc()[0])
    _step(480)
    _report("null-policy")
    x1 = float(roller_loc()[0])
    print(f"[smoke] null policy: roller seat x {x0 * 1000:+.1f} -> {x1 * 1000:+.1f} mm "
          f"(walk {abs(x1 - x0) * 1000:.1f} mm; lip contact ~20 mm, seat window "
          f"{c.seat_x * 1000:.0f} mm, unseat latch {c.unseat_x * 1000:.0f} mm)", flush=True)
    check("null-policy-fails: 480 idle steps, the roller stays in its seat pocket "
          "(the lip bounds the solver walk), `unseated` never latches, score ~0, "
          "no success",
          bool(scene.roller_seated()[0]) and bool(scene.mouth_covered()[0])
          and abs(x1) < c.seat_x and not bool(scene._unseated[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SEAL raid: the shut box cannot be robbed ================================
    # "while the roller is seated nothing passes the slot": centre the prize under the
    # slot, shove it UP at 3x its weight (then up+sideways into the edge crescent).
    # The probe must be NON-VACUOUS: the cube demonstrably rises and is stopped.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.prize,
                vault_world((0.0, 0.0, c.floor_top + c.cube_s / 2 + 0.002)), vault_quat())
    _step(30)
    z_start = float(scene._vault_local(scene.prize.data.root_pos_w)[0, 2])
    zero = torch.zeros(n, 1, 3, device=device)
    up = 3.0 * c.cube_mass * 9.81
    z_max = z_start
    out_ever = False
    for i in range(240):  # 2 s: first straight up, then up + toward the edge crescent
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = up
        if i >= 120:
            fx = quat_apply(scene.vault.data.root_quat_w,
                            torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
            f[:, 0, :] += 1.5 * fx
        scene.prize.set_external_force_and_torque(f, zero, env_ids=_all_ids(),
                                                  is_global=True)
        _step(1)
        z_max = max(z_max, float(scene._vault_local(scene.prize.data.root_pos_w)[0, 2]))
        out_ever = out_ever or bool(scene.prize_out()[0])
    scene.prize.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("seal-raid")
    _REC["on"] = False
    print(f"[smoke] seal raid: prize centre z {z_start * 1000:.0f} -> max "
          f"{z_max * 1000:.0f} mm (plate top {c.top_z * 1000:.0f} mm); "
          f"escaped={out_ever}", flush=True)
    check("SEAL raid: a 3x-weight upward+sideways shove for 2 s raises the prize "
          ">= 15 mm into the slot but the seated roller never lets it out of the cavity",
          z_max > z_start + 0.015 and not out_ever and not bool(scene.prize_out()[0])
          and bool(scene.roller_seated()[0]) and not bool(scene._out[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 6. negative: bare opening (the seed family's naive plan) ===================
    # The seed's literal end state (a lid swung open about its hinge) does not exist
    # here — no joint anywhere. Its family's naive residue — "make the box open and
    # stop" — is constructed: roller at rest in the dock, cubes untouched.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    dock_roller()
    _report("bare-opening")
    check("negative (bare opening): roller at rest in the dock, nothing retrieved — "
          "no success, score <= the opening credit (0.35 + eps)",
          bool(scene.mouth_clear()[0]) and not bool(scene.prize_out()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.351)

    # ================= 7. negative: BYPASS the closure ============================================
    # Prize teleported straight onto the pad while the roller stays SEATED: `out`
    # requires an OPEN mouth, `placed` is gated on `out` — the bypass latches nothing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    prize_to_pad(settle=90)
    _report("bypass")
    check("negative (bypass): prize on the pad under a still-SEATED roller — out and "
          "placed latch NOTHING, score ~0, no success",
          bool(scene.prize_on_pad()[0]) and bool(scene.mouth_covered()[0])
          and not bool(scene._out[0]) and not bool(scene._placed[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 8. near-miss: roller stalled past the lip, still covering ==================
    # Everything else perfect (prize seated on the pad, distractor in), roller pushed
    # past the lip but stalled on the plate at seat x ~72 mm — its surface still
    # overlaps the slot. Judged INSTANTLY (zero velocities, no stepping): `covered`
    # must hold and success must refuse. The roller is then re-seated before stepping.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    prize_to_pad(settle=90)  # roller still seated: no success possible while settling
    _write_body(scene.roller, vault_world((0.072, 0.0, c.top_z + c.roller_r)),
                vault_quat())
    _report("near-miss-cover")
    rl = roller_loc()
    ok8 = (bool(scene.mouth_covered()[0]) and bool(scene.prize_on_pad()[0])
           and bool(scene.distractor_in()[0]) and bool(scene.settled()[0])
           and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)
    print(f"[smoke] near-miss coverage: roller x={float(rl[0]) * 1000:.0f}mm "
          f"(clear_x {c.clear_x * 1000:.0f}mm) covered={bool(scene.mouth_covered()[0])}",
          flush=True)
    check("near-miss (coverage): roller stalled just past the lip still COVERS the "
          "slot — no success even with the prize perfect on the pad", ok8)
    _write_body(scene.roller, vault_world((0.0, 0.0, c.top_z + c.roller_r + 0.002)),
                vault_quat())
    _step(30)  # re-seated (mouth covered): stepping is audit-safe

    # ================= 9. near-miss: prize beside the pad =========================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    dock_roller()
    p = scene.pad.data.root_pos_w.clone()
    p[:, 0] += c.pad_r + 0.010  # 65 mm from the axis: standing on the ground beside it
    p[:, 2] = env.iscene.env_origins[:, 2] + c.cube_s / 2 + 0.003
    _write_body(scene.prize, p, None)
    _step(60)
    _report("near-miss-pad")
    d_xy = float((scene.prize.data.root_pos_w[0, :2]
                  - scene.pad.data.root_pos_w[0, :2]).norm())
    print(f"[smoke] near-miss distance from pad axis: {d_xy * 1000:.0f}mm "
          f"(tol {c.pad_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (placement): prize upright on the ground just beside the pad — "
          "no success, score <= 0.75",
          d_xy > c.pad_xy_tol and not bool(scene.prize_on_pad()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 10. negative: wrong object =================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    dock_roller()
    p = scene.pad.data.root_pos_w.clone()
    p[:, 2] += c.pad_t / 2 + c.cube_s / 2 + 0.010
    _write_body(scene.distractor, p, None)
    _step(60)
    _report("wrong-object")
    check("negative (wrong object): GREY distractor delivered to the pad, GOLD prize "
          "left inside — no success",
          not bool(scene.prize_on_pad()[0]) and not bool(scene.distractor_in()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 11. negative: restraint (distractor also taken out) ========================
    # Construction order is audit-safe: distractor OUT first, so no prefix of the
    # construction ever satisfies the goal.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    d = scene.pad.data.root_pos_w.clone()
    d[:, 1] += c.pad_r + 0.070  # on the ground, off the pad, out of the box
    d[:, 2] = env.iscene.env_origins[:, 2] + c.cube_s / 2 + 0.003
    _write_body(scene.distractor, d, None)
    _step(30)
    dock_roller()
    prize_to_pad(settle=90)
    _report("restraint")
    _REC["on"] = False
    check("negative (restraint): prize correctly delivered but the distractor ALSO "
          "out of the box — no success, score capped at 0.75",
          bool(scene.prize_on_pad()[0]) and not bool(scene.distractor_in()[0])
          and bool(scene.mouth_clear()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)

    # ================= 12. settle gate: the success pose in motion is refused =====================
    # Exact success layout, but the prize is written sliding at 0.45 m/s. Judged
    # INSTANTLY (no stepping — a step could legitimately settle into success): the
    # settled clause must refuse. The prize is then parked back inside the cavity.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    dock_roller()
    p = scene.pad.data.root_pos_w.clone()
    p[:, 2] += c.pad_t / 2 + c.cube_s / 2
    v = zero_v()
    v[:, 0] = 0.45
    _write_body(scene.prize, p, None, vel_w=v)
    _report("settle-gate")
    ok12 = (bool(scene.prize_on_pad()[0]) and bool(scene.mouth_clear()[0])
            and bool(scene.distractor_in()[0]) and not bool(scene.settled()[0])
            and not bool(scene.success()[0]))
    check("settle gate: the exact success layout with the prize sliding at 0.45 m/s "
          "is refused by the settled clause", ok12)
    sgn = float(scene.prize_slot[0])
    _write_body(scene.prize,
                vault_world((0.0, sgn * c.cube_slot_y, c.floor_top + c.cube_s / 2 + 0.002)),
                vault_quat())
    _step(30)  # prize back in the cavity: no success possible

    # ================= 13. MECHANISM: the push-roll-dock chain works ==============================
    # The solve's own act, demonstrated as a smoke fact: an escalating CoM push pops
    # the roller over the lip; the force is CUT at the ramp edge; gravity rolls it
    # down between the fences into the dock and it STAYS there (retention wall).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    zero = torch.zeros(n, 1, 3, device=device)
    fmag = 4.0
    cut_at = None
    docked = False
    for i in range(900):
        rl = roller_loc()
        if float(rl[0]) <= c.box_hx + 0.005 and cut_at is None:
            fdir = quat_apply(scene.vault.data.root_quat_w,
                              torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
            scene.roller.set_external_force_and_torque(
                (fmag * fdir).view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            if i % 60 == 59:
                fmag = min(fmag + 1.0, 8.0)
        elif cut_at is None:
            cut_at = i
            scene.roller.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
            print(f"[smoke] mechanism: force cut at step {i} "
                  f"(x_loc={float(rl[0]):+.3f}, {fmag:.0f} N)", flush=True)
        _step(1)
        if cut_at is not None and bool(scene._cleared[0]) \
                and float(scene.roller.data.root_lin_vel_w[0].norm()) < 0.03:
            docked = True
            break
    scene.roller.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(90)
    _report("mechanism-roll")
    rl = roller_loc()
    print(f"[smoke] mechanism: roller rest loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
          f"{float(rl[2]):+.3f}); dock spans x ~0.29..{c.dock_x1:.2f}", flush=True)
    check("MECHANISM roll: ~4-8 N push pops the lip; hands-off the roller self-rolls "
          "down the fenced ramp and rests in the dock; `cleared` latches, score 0.35",
          docked and bool(scene.mouth_clear()[0]) and bool(scene._cleared[0])
          and float(rl[0]) > 0.25 and float(rl[0]) < c.dock_x1
          and abs(float(rl[1])) < 0.020 and float(rl[2]) < c.roller_r + 0.020
          and float(scene.score()[0]) >= 0.34 and float(scene.score()[0]) <= 0.351)

    # ================= 14. latch persistence under regression =====================================
    # Roller teleported straight back onto its seat: the mouth is covered again and
    # success stays impossible, but the latched opening credit survives.
    _write_body(scene.roller, vault_world((0.0, 0.0, c.top_z + c.roller_r + 0.002)),
                vault_quat())
    _step(60)
    _report("re-seated")
    _REC["on"] = False
    check("latch persistence: roller re-seated (mouth covered again) — success False, "
          "but the latched 0.35 opening credit survives",
          bool(scene.mouth_covered()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) >= 0.34)

    # ================= 15. rejection audit ========================================================
    print(f"[smoke] audit: {_AUDIT['stepped']} stepped states, "
          f"{_AUDIT['success_hits']} success hits", flush=True)
    check("rejection audit: success() was never True at ANY stepped state of the "
          "battery", _AUDIT["success_hits"] == 0 and _AUDIT["stepped"] > 1500)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.roll_away_vault")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
