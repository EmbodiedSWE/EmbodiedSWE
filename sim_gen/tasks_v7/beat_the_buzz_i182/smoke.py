"""Smoke battery for PressurePlateHeistScene — REJECTION tests for the rubric,
NullRobot, RECORDED.

This is NOT a solution (the solution is solve.py — counterweight the pressure plate
with the granite block FIRST, only then steal the idol; the Franka strategy is
TASK.md's embodiment argument). Probes use the same servo numbers as solve.py
(shared numbers = shared honesty); teleports are never used to CONSTRUCT outcomes —
every load-bearing state is reached by lifting with wrenches through the live
spring/D6 plant and releasing to settle by contact, exactly like the solve. What is
being tested is that WRONG settled end states are rejected.

One linear run, 14 named checks:
  1. settle    — clean reset: finite state on every body, plate seated at the bottom
                 of its travel, idol standing on it, no alarm, score ~0;
  2. readback  — physical masses (granite heavy / foam light / idol between, the
                 mass-threshold interlock is real) and poses: granite and foam on
                 opposite sampled y-sides, pad and blocks inside their cfg bands,
                 idol on the plate within its jitter;
  3. random    — 8 seeds: both granite sides drawn; granite, pad and idol-jitter
                 positions all vary (READBACK, not cfg echo);
  4. null      — 2 s of nothing: no alarm, no latch, score < 0.05, no success;
  5. negative  — the SEED-STRATEGY / wrong order: lift the idol first — however
                 slowly (solve.py's own gentle servo) — the plate physically rises
                 (readback > trigger) and the alarm LATCHES;
  6. negative  — permanence: the goal geometry is then fully reconstructed THROUGH
                 THE PLANT (granite dropped onto the risen plate pushes it back
                 down; idol delivered upright to the pad): every geometric
                 predicate holds, yet success is False and score == 0 — the alarm
                 is permanent, "put it back" does not help;
  7. negative  — wrong counterweight: the white FOAM block seated on the plate
                 (readback: actually on it, plate still down) does NOT hold it —
                 lifting the idol still fires the alarm (mass threshold, not
                 mere occupancy);
  8. negative  — wrong place: the granite parked on the TABLE right beside the
                 pedestal (moved — non-vacuous — but not on the plate) does not
                 help either: idol lift -> alarm;
  9. negative  — near miss: correct order, but the idol is landed on the bare table
                 ~13 cm past the pad — hold+clear credit only (score ~0.50), no
                 success;
 10. negative  — toppled delivery: correct order, idol released LYING DOWN over the
                 pad — it rests on the pad but not upright: no pad credit, no
                 success (score ~0.50);
 11. exactness — the full correct strategy -> success() and score == 1.0;
 12. latch     — an 8 N shove pushes the delivered idol off the pad: success is
                 revoked (live state) but the latched 0.65 of transient credit
                 remains;
 13. latch     — credit evaporates only under INCORRECT behavior: lifting the
                 granite off the plate afterwards frees the spring, the plate
                 rises, the alarm fires — score collapses to 0;
 14. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.beat_the_buzz_i182.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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
    from simgen_tasks.beat_the_buzz_i182 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same drive numbers as solve.py (shared numbers = shared honesty).
G = 9.81
V_CAP = 0.25
K_APP = 3.0
KV_IDOL = 10.0  # KV*dt/m = 0.167 (delay-stable)
KV_GRAN = 12.0  # 0.143
KV_FOAM = 2.0  # 0.278 (foam is 60 g — a solve-scale gain would be delay-unstable)
F_MIN, F_MAX = -3.0, 16.0
LIFT_Z = 0.70
HOVER = 0.008
V_DONE = 0.05
SHOVE_F = 8.0  # N lateral shove (friction on the 0.5 kg idol is ~3.4 N)

SLOT_IDOL, SLOT_GRAN, SLOT_FOAM = 0, 1, 2


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pressure_plate_heist")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    origin = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 1.15)) + o),
                                tuple(np.array((-0.05, 0.0, 0.50)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def epos(body) -> torch.Tensor:
        return body.data.root_pos_w[0] - origin

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        print(f"[smoke] {tag:14s} | ext={float(scene.plate_ext()[0]) * 1000:6.2f}mm "
              f"alarm={int(scene.alarm[0])} hold={float(scene.hold_latch[0]):.0f} "
              f"clear={float(scene.clear_latch[0]):.0f} pad={float(scene.pad_latch[0]):.0f} "
              f"score={sc():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- probe helpers (solve.py's lift / carry / release, slot-generic) ---
    def lift(slot: int, body, mass: float, kv: float, z_des: float,
             budget: int = 900) -> tuple[bool, float]:
        """solve.py's vertical velocity-servo lift; returns (converged, max plate ext)."""
        max_ext = 0.0
        ok = False
        for _ in range(budget):
            z = float(epos(body)[2])
            v = float(body.data.root_lin_vel_w[0, 2])
            if abs(z - z_des) < 0.02 and abs(v) < V_DONE:
                ok = True
                break
            v_des = max(-V_CAP, min(V_CAP, K_APP * (z_des - z)))
            f = mass * G + kv * (v_des - v)
            scene.drive_f[0, slot, 2] = max(F_MIN, min(F_MAX, f))
            step(1)
            max_ext = max(max_ext, float(scene.plate_ext()[0]))
        return ok, max_ext

    def carry(slot: int, body, mass: float, pos_env: torch.Tensor,
              quat: torch.Tensor | None = None) -> None:
        """TRANSPORT only: pose across free space, zero velocity, one gravity-hold step."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = pos_env + origin
        st[0, 3:7] = body.data.root_quat_w[0] if quat is None else quat
        scene.drive_f[0, slot] = 0.0
        body.write_root_state_to_sim(st, torch.tensor([0], device=device))
        scene.drive_f[0, slot, 2] = mass * G
        step(1)

    def release_and_settle(slot: int, body, budget: int = 600, streak_need: int = 30) -> bool:
        scene.drive_f[0, slot] = 0.0
        streak = 0
        for _ in range(budget):
            step(1)
            v = float(body.data.root_lin_vel_w[0].norm())
            w = float(body.data.root_ang_vel_w[0].norm())
            streak = streak + 1 if (v < 0.04 and w < 0.8) else 0
            if streak >= streak_need:
                return True
        return False

    def on_plate(body) -> bool:
        """Generic 'block seated on the plate' readback (granite_on_plate is granite-only)."""
        b = epos(body)
        p = epos(scene.plate)
        on_xy = bool(((b[0:2] - p[0:2]).abs() < c.on_plate_xy).all())
        dz = float((b[2] - c.block_s / 2) - (p[2] + c.plate_size[2] / 2))
        return on_xy and abs(dz) < c.on_plate_z_tol

    def counterweight_hover(block_half: float) -> torch.Tensor:
        """Hover spot over the plate, opposite side from the idol (solve.py geometry)."""
        ped = torch.tensor(c.pedestal_xy, device=device)
        off = epos(scene.idol)[0:2] - ped
        u = -off / max(float(off.norm()), 1e-6) if float(off.norm()) > 0.003 \
            else torch.tensor([1.0, 0.0], device=device)
        plate_top = float(epos(scene.plate)[2]) + c.plate_size[2] / 2
        h = torch.zeros(3, device=device)
        h[0:2] = ped + u * 0.062
        h[2] = plate_top + block_half + HOVER
        return h

    def place_granite() -> bool:
        """The correct PHASE-1 move: granite lifted, carried over the plate, released."""
        ok, _ = lift(SLOT_GRAN, scene.granite, c.granite_mass, KV_GRAN, LIFT_Z)
        if not ok:
            return False
        carry(SLOT_GRAN, scene.granite, c.granite_mass, counterweight_hover(c.block_s / 2),
              quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        return release_and_settle(SLOT_GRAN, scene.granite)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    up_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("reset")
    idol_p0 = epos(scene.idol).clone()
    ped = torch.tensor(c.pedestal_xy, device=device)
    check("settle: clean reset (finite, plate seated, idol standing on it, score ~0)",
          finite_all()
          and abs(float(scene.plate_ext()[0])) < 0.002
          and bool(scene.idol_up()[0])
          and float((idol_p0[0:2] - ped).norm()) < c.idol_jitter * 1.6
          and float(scene.alarm[0]) == 0.0 and sc() < 0.05
          and not bool(scene.success()[0]))

    # ========================= 2. mass + pose readback ========================================
    try:
        m_g = float(scene.granite.root_physx_view.get_masses().reshape(-1)[0])
        m_f = float(scene.foam.root_physx_view.get_masses().reshape(-1)[0])
        m_i = float(scene.idol.root_physx_view.get_masses().reshape(-1)[0])
        m_p = float(scene.plate.root_physx_view.get_masses().reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] mass readback unavailable ({exc!r})", flush=True)
        m_g = m_f = m_i = m_p = float("nan")
    print(f"[smoke] masses: granite={m_g:.3f} foam={m_f:.3f} idol={m_i:.3f} "
          f"plate={m_p:.3f} kg", flush=True)
    side = float(scene.granite_side[0])
    g = epos(scene.granite)
    f_ = epos(scene.foam)
    p = epos(scene.pad)
    spring_ok = (m_p + m_f) * G < c.spring_f < (m_p + m_i) * G < (m_p + m_g) * G
    pose_ok = (side * float(g[1]) >= c.block_y_range[0] - 0.02
               and -side * float(f_[1]) >= c.block_y_range[0] - 0.02
               and float(g[1]) * float(f_[1]) < 0
               and c.block_x_range[0] - 0.02 <= float(g[0]) <= c.block_x_range[1] + 0.02
               and c.pad_x_range[0] - 0.01 <= float(p[0]) <= c.pad_x_range[1] + 0.01
               and c.pad_y_range[0] - 0.01 <= float(p[1]) <= c.pad_y_range[1] + 0.01)
    check("readback: interlock masses ordered (foam < spring < idol < granite) + "
          "blocks on opposite sampled sides, pad in band",
          abs(m_g - c.granite_mass) < 0.02 and abs(m_f - c.foam_mass) < 0.01
          and abs(m_i - c.idol_mass) < 0.02 and spring_ok and pose_ok)

    # ========================= 3. randomization across seeds ==================================
    draws = []
    for seed in (11, 12, 13, 14, 15, 16, 17, 18):
        torch.manual_seed(seed)
        env.reset()
        step(2)
        gg = epos(scene.granite)
        pp = epos(scene.pad)
        ii = epos(scene.idol)
        draws.append((int(scene.granite_side[0]),
                      round(float(gg[0]), 3), round(float(gg[1]), 3),
                      round(float(pp[0]), 3), round(float(pp[1]), 3),
                      round(float(ii[0] - ped[0]), 3), round(float(ii[1] - ped[1]), 3)))
    print(f"[smoke] draws: {draws}", flush=True)
    sides = {d[0] for d in draws}
    g_pos = {(d[1], d[2]) for d in draws}
    p_pos = {(d[3], d[4]) for d in draws}
    i_off = {(d[5], d[6]) for d in draws}
    check("random: both granite sides drawn; granite, pad and idol-jitter all vary (8 seeds)",
          sides == {1, -1} and len(g_pos) >= 4 and len(p_pos) >= 4 and len(i_off) >= 4)

    # ========================= 4. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null: 2 s of no action — no alarm, no latch, score < 0.05, no success",
          float(scene.alarm[0]) == 0.0 and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. wrong order (the seed strategy) + permanence ================
    # Lift the idol FIRST — with the solve's own gentle servo — and the plate rises.
    z_i0 = float(epos(scene.idol)[2])
    ok_l, max_ext = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    moved = float(epos(scene.idol)[2]) - z_i0
    report("idol-first")
    alarm_fired = float(scene.alarm[0]) == 1.0
    # ...then physically RECONSTRUCT the goal: granite onto the (risen) plate pushes it
    # back down; the idol is delivered upright to the pad. Geometry == goal, alarm bites.
    carry(SLOT_IDOL, scene.idol, c.idol_mass,
          torch.tensor([float(epos(scene.pad)[0]), float(epos(scene.pad)[1]),
                        c.pad_top + c.idol_h / 2 + HOVER], device=device), quat=up_q)
    ok_i = release_and_settle(SLOT_IDOL, scene.idol)
    ok_g = place_granite()
    step(240)
    report("rebuilt")
    geom = (bool(scene.idol_on_pad()[0]) and bool(scene.plate_down()[0])
            and bool(scene.granite_on_plate()[0]) and bool(scene.settled()[0]))
    check("negative (seed strategy / order): gentle idol-first lift rises the plate "
          "past the trigger and latches the alarm",
          ok_l and moved > 0.10 and max_ext > c.trigger_h and alarm_fired)
    check("negative (permanence): goal geometry fully rebuilt through the plant — "
          "still no success, score == 0",
          ok_i and ok_g and geom and not bool(scene.success()[0]) and sc() < 0.05)

    # ========================= 6. wrong counterweight (foam) ==================================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ok_lf, _ = lift(SLOT_FOAM, scene.foam, c.foam_mass, KV_FOAM, LIFT_Z)
    carry(SLOT_FOAM, scene.foam, c.foam_mass, counterweight_hover(c.block_s / 2), quat=up_q)
    ok_sf = release_and_settle(SLOT_FOAM, scene.foam)
    foam_on = on_plate(scene.foam)
    plate_down0 = bool(scene.plate_down()[0]) and float(scene.alarm[0]) == 0.0
    report("foam-on")
    ok_li, max_ext = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    step(60)
    report("foam-failed")
    check("negative (wrong counterweight): foam seated on the plate does not hold it — "
          "idol lift still fires the alarm",
          ok_lf and ok_sf and foam_on and plate_down0
          and ok_li and max_ext > c.trigger_h and float(scene.alarm[0]) == 1.0
          and sc() < 0.05 and not bool(scene.success()[0]))

    # ========================= 7. wrong place (granite beside the pedestal) ===================
    torch.manual_seed(3)
    env.reset()
    step(30)
    g0 = epos(scene.granite).clone()
    ok_lg, _ = lift(SLOT_GRAN, scene.granite, c.granite_mass, KV_GRAN, LIFT_Z)
    beside = torch.tensor([c.pedestal_xy[0],
                           c.pedestal_size[1] / 2 + c.block_s / 2 + 0.03,
                           c.z0 + c.block_s / 2 + HOVER], device=device)
    carry(SLOT_GRAN, scene.granite, c.granite_mass, beside, quat=up_q)
    ok_sg = release_and_settle(SLOT_GRAN, scene.granite)
    g_moved = float((epos(scene.granite) - g0).norm())
    near_ped = float((epos(scene.granite)[0:2] - ped).norm()) < 0.25
    report("g-beside")
    ok_li, max_ext = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    step(60)
    report("beside-failed")
    check("negative (wrong place): granite parked beside the pedestal (moved, near, "
          "NOT on plate) — idol lift still fires the alarm",
          ok_lg and ok_sg and g_moved > 0.10 and near_ped
          and not on_plate(scene.granite)
          and ok_li and max_ext > c.trigger_h and float(scene.alarm[0]) == 1.0
          and sc() < 0.05)

    # ========================= 8. near miss (landed off the pad) ==============================
    torch.manual_seed(3)
    env.reset()
    step(30)
    g_side = float(scene.granite_side[0])  # granite's ORIGINAL side — vacant after phase 1
    ok_g = place_granite()
    ok_li, _ = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    pp = epos(scene.pad)
    miss = torch.tensor([float(pp[0]), float(pp[1]) + g_side * 0.13,
                         c.z0 + c.idol_h / 2 + HOVER], device=device)
    carry(SLOT_IDOL, scene.idol, c.idol_mass, miss, quat=up_q)
    ok_si = release_and_settle(SLOT_IDOL, scene.idol)
    step(240)
    report("near-miss")
    check("negative (near miss): idol landed ~13 cm off the pad — hold+clear credit "
          "only (~0.50), no success",
          ok_g and ok_li and ok_si and float(scene.alarm[0]) == 0.0
          and not bool(scene.idol_on_pad()[0]) and not bool(scene.success()[0])
          and abs(sc() - 0.50) < 0.02)

    # ========================= 9. toppled delivery ============================================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ok_g = place_granite()
    ok_li, _ = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    pp = epos(scene.pad)
    lie_q = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=device)  # 90 deg about y
    lay = torch.tensor([float(pp[0]), float(pp[1]),
                        c.pad_top + c.idol_base / 2 + HOVER], device=device)
    carry(SLOT_IDOL, scene.idol, c.idol_mass, lay, quat=lie_q)
    ok_si = release_and_settle(SLOT_IDOL, scene.idol)
    step(240)
    report("toppled")
    on_pad_xy = bool(((epos(scene.idol)[0:2] - pp[0:2]).abs() < c.pad_xy_tol).all())
    check("negative (toppled): idol LYING on the pad — not upright, no pad credit, "
          "no success (~0.50)",
          ok_g and ok_li and ok_si and float(scene.alarm[0]) == 0.0
          and on_pad_xy and not bool(scene.idol_up()[0])
          and not bool(scene.idol_on_pad()[0]) and not bool(scene.success()[0])
          and abs(sc() - 0.50) < 0.02)

    # ========================= 10. exactness ==================================================
    torch.manual_seed(3)
    env.reset()
    step(30)
    ok_g = place_granite()
    ok_li, max_ext = lift(SLOT_IDOL, scene.idol, c.idol_mass, KV_IDOL, LIFT_Z)
    pp = epos(scene.pad)
    goal = torch.tensor([float(pp[0]), float(pp[1]),
                         c.pad_top + c.idol_h / 2 + HOVER], device=device)
    carry(SLOT_IDOL, scene.idol, c.idol_mass, goal)
    ok_si = release_and_settle(SLOT_IDOL, scene.idol)
    got = False
    for _ in range(48):
        if bool(scene.success()[0]):
            got = True
            break
        step(10)
    report("goal-state")
    check("exactness: full correct strategy -> success() and score == 1.0",
          ok_g and ok_li and max_ext < c.trigger_h and ok_si and got
          and abs(sc() - 1.0) < 1e-3)

    # ========================= 11. success revoked, latches stay ==============================
    i0 = epos(scene.idol).clone()
    pp = epos(scene.pad)
    for _ in range(int(1.5 * 120)):  # shove +x (empty table) until clearly off the pad
        if float((epos(scene.idol)[0:2] - pp[0:2]).norm()) > 0.09:
            break
        scene.drive_f[0, SLOT_IDOL, 0] = SHOVE_F
        step(1)
    scene.drive_f[0, SLOT_IDOL] = 0.0
    step(240)
    i_moved = float((epos(scene.idol)[0:2] - i0[0:2]).norm())
    report("shoved-off")
    check("achievement latch: idol shoved off the pad — success revoked, latched 0.65 "
          "remains",
          i_moved > 0.05 and not bool(scene.idol_on_pad()[0])
          and not bool(scene.success()[0]) and abs(sc() - 0.65) < 0.02)

    # ========================= 12. credit evaporates on incorrect behavior ====================
    ok_lg, max_ext = lift(SLOT_GRAN, scene.granite, c.granite_mass, KV_GRAN, LIFT_Z)
    scene.drive_f[0, SLOT_GRAN] = 0.0
    step(120)
    report("g-removed")
    check("alarm latch: lifting the granite back OFF frees the spring — plate rises, "
          "alarm fires, all credit evaporates",
          ok_lg and max_ext > c.trigger_h and float(scene.alarm[0]) == 1.0 and sc() < 0.01)

    # ========================= 13. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.pressure_plate_heist")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
