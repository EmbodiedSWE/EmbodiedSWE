"""Smoke / rubric-REJECTION battery for LatchCanisterScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that the physical interlocks the task rests on — a
tab turned early keeps the lid from seating, a mis-seated lid blocks the tab, a
sealed mouth blocks a late ball — are load-bearing. Every probe is CONSTRUCTED
(teleport, real physics steps, judge) — instrumentation, never a solution:
success() is monitored at EVERY step and must never turn True anywhere in the
battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; tabs near OPEN, cavity
                           empty, lid on the ground; score 0, no success;
   2. randomization      — two seeded resets: READBACK canister xy+yaw, tab
                           angles, red-ball xy and lid xy all differ;
   3. slot swap          — over 10 resets the red ball occupies BOTH slots;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed's own outcome ("lid on the jar"): lid gravity-
                           dropped onto the seat of the EMPTY canister, tabs open,
                           settled -> lid_seated True yet score 0 (order-coupled
                           latches: no ball -> no lid credit), no success;
   6. wrong object       — the BLUE ball sealed inside instead (blue in cavity,
                           lid seated, tabs locked) -> no success, score 0;
   7. INTERLOCK tab-first— tabs turned LOCKED over the EMPTY recess, then the lid
                           dropped dead-centre: it lands ON the bars ~15 mm too
                           high -> lid_seated False (geometry forces the order);
   8. near-miss seat     — lid dropped 20 mm off-axis: one edge perches on the
                           lip ring -> lid_seated False, no success;
   9. INTERLOCK mis-seat — lid dropped yawed 45 deg perches on the lip rails; a
                           locking torque then CANNOT swing the tab into the
                           locked window (the standing lid blocks the swing);
  10. near-miss tab      — ball in, lid seated, east tab locked, west tab at
                           -45 deg (short of the -70 deg window) -> no success,
                           score <= 0.60;
  11. restraint          — BOTH balls sealed in, lid seated, tabs locked, still —
                           every clause but the decoy holds -> no success,
                           score <= 0.60: the blue-out clause is load-bearing;
  12. MECHANISM seal     — empty canister sealed (lid + both tabs), the RED ball
                           dropped onto it from above stays OUT of the cavity ->
                           ball_in False, score 0 (no late entry, omission earns
                           nothing);
  13. settle gate        — the exact success geometry constructed while the blue
                           ball is still ROLLING: every geometric clause holds
                           yet success refuses for 45 sampled steps (consecutive-
                           still counter, not an instantaneous gate); dismantled
                           before it can settle into a real success;
  14. latched credit     — after 13's dismantle (lid carried away) the live state
                           is broken but the latched credit survives: score still
                           0.60, no success;
  15. rejection audit    — success() was never True at any step of this battery;
  16. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_jar_i54.smoke --headless
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

_qz, _qmul, _qapply = task_scene._qz, task_scene._qmul, task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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
                lin_vel: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel is not None:
        st[:, 7:10] = lin_vel
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    th = scene.tab_theta()[0]
    lid_loc = scene._local(scene.lid.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | tabs=({float(th[0]):+6.1f},{float(th[1]):+6.1f})deg "
          f"lid_loc=({float(lid_loc[0]):+.3f},{float(lid_loc[1]):+.3f},"
          f"{float(lid_loc[2]):+.3f}) ball_in={bool(scene.ball_in()[0])} "
          f"lid_seated={bool(scene.lid_seated()[0])} "
          f"locked={[bool(v) for v in scene.tabs_locked()[0]]} "
          f"decoy_out={bool(scene.decoy_out()[0])} still={bool(scene.still()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.latch_canister")().build(num_envs=args.num_envs,
                                                   device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.20, -0.55, 0.50)) + o),
                                tuple(np.array((0.35, 0.00, 0.08)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def can_pt(loc) -> torch.Tensor:
        """Canister-frame point -> world (live canister pose), (N,3)."""
        _refresh()
        p = torch.tensor([float(v) for v in loc], device=device).expand(n, 3)
        return scene.canister.data.root_pos_w + _qapply(
            scene.canister.data.root_quat_w, p)

    def can_quat(yaw_extra: float = 0.0) -> torch.Tensor:
        _refresh()
        return _qmul(scene.canister.data.root_quat_w,
                     _qz(torch.full((n,), math.radians(yaw_extra), device=device)))

    def write_tab(idx: int, theta_deg: float) -> None:
        """Joint-consistent tab pose write at relative angle `theta_deg`
        (0 = open, -90 = locked)."""
        body, side = ((scene.tab_e, 1.0), (scene.tab_w, -1.0))[idx]
        open_local = -90.0 if side > 0 else 90.0
        _write_body(body, can_pt((side * c.post_x, 0.0, c.pivot_z)),
                    can_quat(open_local + theta_deg))

    def drop_lid(offset=(0.0, 0.0), yaw_extra: float = 0.0, settle: int = 150) -> None:
        """Gravity-drop the lid from an 18 mm hover over the seat (+ offset/yaw)."""
        _write_body(scene.lid, can_pt((offset[0], offset[1], 0.112)),
                    can_quat(yaw_extra))
        _step(settle)

    def write_lid_seated() -> None:
        _write_body(scene.lid, can_pt((0.0, 0.0, c.seat_z)), can_quat())

    def write_ball_in(body) -> None:
        _write_body(body if body is not None else scene.red,
                    can_pt((0.0, 0.0, 0.010 + c.ball_r + 0.002)))

    def drive_tab(idx: int, steps: int, tau_cap: float = 0.03) -> float:
        """The solve controller (stable gain), fixed torque cap; returns final theta."""
        body = (scene.tab_e, scene.tab_w)[idx]
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            th = float(scene.tab_theta()[0, idx])
            if th <= -80.0:
                break
            w = float(body.data.root_ang_vel_w[0, 2])
            tau = max(-tau_cap, min(tau_cap, 0.004 * (-0.9 - w)))
            t_world = torch.zeros(n, 1, 3, device=device)
            t_world[0, 0, 2] = tau
            body.set_external_force_and_torque(zero, t_world, env_ids=_all_ids(),
                                               is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        _step(60)
        return float(scene.tab_theta()[0, idx])

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    th = scene.tab_theta()[0]
    check("settle/no-NaN: layout settles finite; tabs near OPEN, cavity empty, lid "
          "on the ground; score 0, no success",
          bool(scene._finite()[0]) and float(th.abs().max()) < 25.0
          and not bool(scene.ball_in()[0]) and not bool(scene.lid_seated()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.canister.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.canister.data.root_quat_w[0]),
                scene.tab_theta()[0].clone(),
                scene.red.data.root_pos_w[0, :2].clone(),
                scene.lid.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_cp, a_cy, a_th, a_rp, a_lp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_cp, b_cy, b_th, b_rp, b_lp = readback()
    d_cp = float((a_cp - b_cp).norm())
    d_cy = dyaw(a_cy, b_cy)
    d_th = float((a_th - b_th).abs().max())
    d_rp, d_lp = float((a_rp - b_rp).norm()), float((a_lp - b_lp).norm())
    print(f"[smoke] randomization deltas: can_xy={d_cp * 1000:.1f}mm "
          f"can_yaw={d_cy:.1f}deg tab_theta={d_th:.1f}deg "
          f"red_xy={d_rp * 1000:.1f}mm lid_xy={d_lp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: canister xy+yaw, tab angles, red xy and lid xy "
          "readback differ between seeds",
          d_cp > 0.003 and d_cy > 2.0 and d_th > 1.0 and d_rp > 0.003
          and d_lp > 0.003)

    # ================= 3. slot swap ===============================================================
    slots = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        slots.add("A" if float(scene.red_slot[0]) > 0 else "B")
    print(f"[smoke] over 10 resets: red ball slots {sorted(slots)}", flush=True)
    check("slot swap: the red ball occupies BOTH slots over 10 resets",
          slots == {"A", "B"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN OUTCOME ========================================
    # rlbench/close_jar's outcome is "the lid on the jar". Reproduced here — lid
    # gravity-seated on the EMPTY canister, tabs left open — it is worth NOTHING:
    # the latches are order-coupled (no ball -> no lid credit) and success needs
    # the ball and both tabs.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_lid()
    _report("seed-strategy")
    check("negative (SEED strategy): lid seated on the EMPTY canister, tabs open — "
          "lid_seated True yet score 0 and no success (order-coupled credit)",
          bool(scene.lid_seated()[0]) and not bool(scene.ball_in()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: wrong object ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_ball_in(scene.blue)
    _step(30)
    write_lid_seated()
    _step(30)
    write_tab(0, -88.0)
    write_tab(1, -88.0)
    _step(90)
    _report("wrong-object")
    check("negative (wrong object): the BLUE ball sealed inside instead — no "
          "success, score 0",
          bool(scene.in_cavity(scene.blue.data.root_pos_w)[0])
          and not bool(scene.decoy_out()[0]) and bool(scene.lid_seated()[0])
          and not bool(scene.ball_in()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)

    # ================= 7. INTERLOCK: tab turned first blocks the seat =============================
    # Both tabs teleport-locked over the EMPTY recess, then the lid dropped dead-
    # centre and yaw-aligned — the exact drop that seats it in check 5. It lands ON
    # the bars, ~15 mm above the seat window: geometry forces ball -> lid -> tabs.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_tab(0, -88.0)
    write_tab(1, -88.0)
    _step(30)
    drop_lid()
    _report("tab-first")
    lid_z = float(scene._local(scene.lid.data.root_pos_w)[0, 2])
    print(f"[smoke] lid rests at canister-frame z={lid_z * 1000:.1f}mm "
          f"(seat window {c.seat_z_lo * 1000:.1f}..{c.seat_z_hi * 1000:.1f}mm)",
          flush=True)
    check("INTERLOCK (tab first): tabs locked over the EMPTY recess make the "
          "dead-centre lid drop rest ON the bars, too high to seat",
          lid_z > c.seat_z_hi and not bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 8. near-miss: lid dropped off-axis perches on the lip ======================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    drop_lid(offset=(0.0, 0.020))  # north edge lands on the lip rail
    _report("near-miss-seat")
    lid_loc = scene._local(scene.lid.data.root_pos_w)[0]
    print(f"[smoke] lid centre canister-frame=({float(lid_loc[0]) * 1000:+.0f},"
          f"{float(lid_loc[1]) * 1000:+.0f},{float(lid_loc[2]) * 1000:.1f})mm "
          f"(xy tol {c.seat_xy_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (seat): lid dropped 20 mm off-axis perches with an edge on "
          "the lip ring — lid_seated False, no success",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0]))

    # ================= 9. INTERLOCK: a mis-seated lid blocks the tab ==============================
    # Lid dropped yawed 45 deg: its corners ride the lip rails (rotated half-
    # diagonal 99 mm > 75 mm opening), so the plate stands ABOVE the tab swing
    # plane. The solve controller then CANNOT swing the east tab into the locked
    # window — the standing lid physically blocks the bar.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_lid(yaw_extra=45.0)
    lid_z = float(scene._local(scene.lid.data.root_pos_w)[0, 2])
    th_end = drive_tab(0, steps=600)
    _report("mis-seat-blocks")
    print(f"[smoke] yawed lid perched at z={lid_z * 1000:.1f}mm; east tab driven "
          f"to {th_end:+.1f} deg (locked window starts at "
          f"-{c.tab_locked_deg:.0f} deg)", flush=True)
    check("INTERLOCK (mis-seat): a 45 deg-yawed lid perched on the lip blocks the "
          "driven tab short of the locked window",
          lid_z > c.seat_z_hi and not bool(scene.lid_seated()[0])
          and th_end > -c.tab_locked_deg and not bool(scene.tabs_locked()[0, 0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 10. near-miss: one tab short of the window =================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_ball_in(scene.red)
    _step(30)
    write_lid_seated()
    _step(30)
    write_tab(0, -88.0)
    write_tab(1, -45.0)  # short of the -70 deg locked window
    _step(120)  # long enough for the still counter to latch
    _report("near-miss-tab")
    check("near-miss (tab): ball in, lid seated, east tab locked, west tab at "
          "-45 deg — no success, score <= 0.60",
          bool(scene.ball_in()[0]) and bool(scene.lid_seated()[0])
          and bool(scene.tabs_locked()[0, 0]) and not bool(scene.tabs_locked()[0, 1])
          and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.60 + 1e-6)  # f32 0.25+0.20+0.15 = 0.60000002

    # ================= 11. restraint: the decoy clause is load-bearing ============================
    # BOTH balls sealed in, lid seated, tabs locked, everything still: every other
    # success clause holds — only decoy_out refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.red, can_pt((0.025, 0.0, 0.010 + c.ball_r + 0.002)))
    _write_body(scene.blue, can_pt((-0.025, 0.0, 0.010 + c.ball_r + 0.002)))
    _step(30)
    write_lid_seated()
    _step(30)
    write_tab(0, -88.0)
    write_tab(1, -88.0)
    _step(120)
    _report("restraint")
    check("restraint: BOTH balls sealed in, lid seated, tabs locked, still — only "
          "the decoy clause refuses: no success, score <= 0.60",
          bool(scene.ball_in()[0]) and not bool(scene.decoy_out()[0])
          and bool(scene.lid_seated()[0]) and bool(scene.tabs_locked()[0].all())
          and bool(scene.still()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.60 + 1e-6)

    # ================= 12. MECHANISM: the sealed mouth blocks a late ball =========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_lid_seated()
    _step(30)
    write_tab(0, -88.0)
    write_tab(1, -88.0)
    _step(30)
    _write_body(scene.red, can_pt((0.0, 0.0, 0.160)))  # the P1 drop, now sealed
    _step(180)
    _report("late-ball")
    red_loc = scene._local(scene.red.data.root_pos_w)[0]
    print(f"[smoke] late ball ended at canister-frame z={float(red_loc[2]) * 1000:.1f}mm "
          f"(cavity window {c.cavity_z_lo * 1000:.0f}..{c.cavity_z_hi * 1000:.0f}mm)",
          flush=True)
    check("MECHANISM (seal): the red ball dropped onto the SEALED canister stays "
          "OUT of the cavity — no late entry, omission earns nothing",
          not bool(scene.ball_in()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.01)
    _REC["on"] = False

    # ================= 13+14. settle gate / latched credit ========================================
    # Construct the exact success geometry IN ORDER while the blue ball is rolling
    # (nudged away from the canister): every geometric clause holds, all three
    # credit latches fire, but success() refuses at every sampled step because the
    # consecutive-still counter never latches while anything moves. The probe is
    # dismantled before the scene can settle into a real success; the latched
    # credit must survive the dismantle.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _refresh()
    kick = torch.zeros(n, 3, device=device)
    kick[:, 0] = -0.5  # roll away from the canister (canister is at +x)
    _write_body(scene.blue, scene.blue.data.root_pos_w.clone(), None, lin_vel=kick)
    write_ball_in(scene.red)
    _step(20)
    write_lid_seated()
    _step(20)
    write_tab(0, -88.0)
    write_tab(1, -88.0)
    geom_ok, gate_ok = True, True
    for _ in range(45):
        _step(1)
        geom_ok = geom_ok and bool(scene.ball_in()[0]) \
            and bool(scene.lid_seated()[0]) and bool(scene.tabs_locked()[0].all()) \
            and bool(scene.decoy_out()[0])
        gate_ok = gate_ok and not bool(scene.still()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    blue_v = float(scene.blue.data.root_lin_vel_w[0].norm())
    print(f"[smoke] blue ball still rolling at {blue_v:.2f} m/s through the sample "
          f"window", flush=True)
    check("settle gate: the exact success geometry while the blue ball rolls — "
          "every geometric clause holds for 45 steps yet success refuses "
          "(consecutive-still counter)",
          geom_ok and gate_ok and blue_v > 0.05)
    # dismantle before it can settle into a real success: carry the lid away
    far = can_pt((0.0, -0.30, 0.004))
    _write_body(scene.lid, far, can_quat())
    _step(30)
    _report("dismantled")
    check("latched credit: after the dismantle the live state is broken "
          "(lid gone) but the latched credit survives: score 0.60, no success",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.60) < 1e-6)

    # ================= 15. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.latch_canister")
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
