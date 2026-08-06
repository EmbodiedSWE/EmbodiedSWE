"""Smoke / oracle test for SweeperGauntletScene (sim_gen task `track_banana_i48`) —
NullRobot, teleport-oracle, RECORDED.

Battery (rail_ferry / pen_holder smoke skeleton):
  1. settle/no-NaN      — reset layout finite, crates settled, score 0;
  2. sweeper drive      — sweepers actually patrol (physical readback moves) and track
                          the analytic triangle wave;
  3. randomization      — READBACK across 8 seeded resets: crate scatter, basket pose,
                          sweeper phases AND periods all move;
  4. null-policy-fails  — 300 idle substeps -> score ~0, no success (sweepers patrol,
                          crates untouched);
  5-7. oracle x3 seeds  — full gauntlet run (per crate: approach -> wait for a patrol
                          window -> dash corridor 1 -> wait -> dash corridor 2 -> place
                          in basket), reaching success() and score 1.0;
  8. monotonicity       — staged score strictly increases across the 9 t1/t2/deliver
                          milestones of the seed-0 run, ending exactly 1.0;
  9-11. negative A      — the seed's phase-blind CONSTANT-SPEED carry straight across
                          (its tracking plan, no timing): guaranteed to meet the sweeper
                          (`struck` latch), then a PERFECT basket pose is still refused,
                          and a clean window-timed re-run of the same crate stays refused
                          (latch permanence);
  12. negative B        — the seed's AERIAL WAYPOINT ARC (carry high over both corridors,
                          its waypoints rise to z 0.36+): `flew` latch, refused despite a
                          perfect basket pose;
  13. negative C        — teleport-to-basket with no transit: t1/t2 never latch, not
                          delivered (anti-teleport);
  14-15. near-miss      — a cleanly gauntleted crate set down BESIDE the basket earns
                          only the t2 credit; moving it in recovers delivery credit;
  16-17. calibration    — the timing cliff: a dash launched while the sweeper approaches
                          the lane is struck 3/3; a window-timed dash is clean 3/3
                          (published with per-seed period/speed/wait stats).

Run (forge): python -u -m simgen_tasks.track_banana_i48.smoke --headless
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
    from simgen_tasks.track_banana_i48 import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

DASH_STEP = 0.006   # dash x-advance per substep (0.72 m/s at 120 Hz)
CARRY_STEP = 0.005  # free-space carry step
CARRY_Z = 0.040     # corridor-legal carry height (crate center, under the 0.12 fly cap)
WAIT_X1 = -0.36     # pre-corridor-1 wait point (30+ mm outside the strike footprint)
MED_X = -0.03       # median wait point
EXIT_X = 0.345      # post-corridor-2 release point
LANE_Y = 0.0        # crossing lane (patrol center: passes every T/2)
TRIG_Y = 0.19       # dash trigger: sweeper receding beyond this |y| ...
TRIG_T = 0.70       # ... with at least this long before it can re-enter the lane danger
                    # zone (dry-computed: max available 0.91 s at the fastest period;
                    # dash danger exposure ends 0.52 s after launch)


def _env_of(scene_or_env):
    return scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env


def _mk_helpers(env, step_fn):
    """Shared staging/motion helpers bound to (env, step)."""
    scene = env.scene
    c = scene.cfg
    dev = env.device
    n = env.num_envs
    all_ids = torch.arange(n, device=dev)
    no_action = torch.empty(0, device=dev)

    def step(k: int) -> None:
        if step_fn is not None:
            step_fn(k)
        else:
            for _ in range(k):
                env.step(no_action)

    def write_body(body, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=dev)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=dev)
        st[:, 3:7] = torch.tensor(quat, device=dev)
        body.write_root_state_to_sim(st, all_ids)
        env.iscene.update(0.0)

    def loc(body):
        return (body.data.root_pos_w - scene.env_origins)[0]

    def carry(i: int, waypoints, step_len: float = CARRY_STEP) -> None:
        """Kinematic carry of crate i along straight segments (zero-velocity re-pins,
        one pose write per substep — the scene's latches watch the whole path)."""
        body = scene.crates[f"crate_{i}"]
        p = loc(body).tolist()
        st = torch.zeros(n, 13, device=dev)
        for wp in waypoints:
            d = math.dist(p, wp)
            kk = max(1, int(math.ceil(d / step_len)))
            for t in range(1, kk + 1):
                q = [p[a] + (wp[a] - p[a]) * t / kk for a in range(3)]
                st.zero_()
                st[:, 0:3] = env.iscene.env_origins + torch.tensor(q, device=dev)
                st[:, 3] = 1.0
                body.write_root_state_to_sim(st, all_ids)
                step(1)
            p = wp

    def sweeper_yv(j: int) -> tuple[float, float]:
        y, vy = scene.sweeper_analytic()
        return float(y[0, j]), float(vy[0, j])

    def t_return(j: int) -> float:
        """Time before sweeper j can next reach the lane's danger zone (analytic,
        triangle wave; exact — the patrol is scripted)."""
        a = c.patrol_amp
        danger = c.half_sum_y + c.strike_pad_y
        y, vy = sweeper_yv(j)
        v = abs(vy)
        if y * vy <= 0:  # approaching the lane
            return max(0.0, (abs(y) - danger) / v)
        return ((a - abs(y)) + (a - danger)) / v

    def wait_window(j: int, need: float = TRIG_T, max_s: float = 12.0) -> float:
        """Idle-step until sweeper j has just passed the lane, is receding beyond
        TRIG_Y, and cannot return within `need` seconds. Returns the wait (s)."""
        waited = 0
        cap = int(max_s / env.dt)
        while waited < cap:
            y, vy = sweeper_yv(j)
            if y * vy > 0 and abs(y) >= TRIG_Y and t_return(j) > need:
                return waited * env.dt
            step(2)
            waited += 2
        return waited * env.dt

    def wait_collision_course(j: int, max_s: float = 12.0) -> float:
        """Idle-step until sweeper j is APPROACHING the lane from 0.22-0.28 m out —
        the mistimed launch that guarantees a mid-dash strike (dry-computed)."""
        waited = 0
        cap = int(max_s / env.dt)
        while waited < cap:
            y, vy = sweeper_yv(j)
            if y * vy < 0 and 0.22 < abs(y) < 0.28:
                return waited * env.dt
            step(2)
            waited += 2
        return waited * env.dt

    def dash(i: int, x_to: float) -> None:
        """Low fast x-dash of crate i at the lane (kinematic, DASH_STEP per substep)."""
        body = scene.crates[f"crate_{i}"]
        p = loc(body).tolist()
        x0 = p[0]
        kk = max(1, int(math.ceil(abs(x_to - x0) / DASH_STEP)))
        st = torch.zeros(n, 13, device=dev)
        for t in range(1, kk + 1):
            x = x0 + (x_to - x0) * t / kk
            st.zero_()
            st[:, 0] = x
            st[:, 1] = LANE_Y
            st[:, 2] = CARRY_Z
            st[:, 0:3] += env.iscene.env_origins
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, all_ids)
            step(1)

    def settle_until(pred, max_steps: int = 240, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def basket_slot(t: int) -> tuple[float, float]:
        bx = float(scene.basket_xy[0, 0])
        by = float(scene.basket_xy[0, 1])
        return bx + (t - 1) * 0.065, by

    return {
        "step": step, "write_body": write_body, "loc": loc, "carry": carry,
        "sweeper_yv": sweeper_yv, "t_return": t_return, "wait_window": wait_window,
        "wait_collision_course": wait_collision_course, "dash": dash,
        "settle_until": settle_until, "basket_slot": basket_slot,
    }


def run_crate_gauntlet(env, h, i: int, slot: int, milestones=None,
                       waits=None) -> None:
    """One crate through the full gauntlet: approach, window-dash corridor 1, window-dash
    corridor 2, place into its basket slot."""
    scene = env.scene

    def mark() -> None:
        if milestones is not None:
            milestones.append(float(scene.score()[0]))

    h["carry"](i, [(WAIT_X1, LANE_Y, CARRY_Z), (WAIT_X1, LANE_Y, 0.030)])
    w = h["wait_window"](0)
    h["dash"](i, MED_X)
    h["settle_until"](lambda: bool(scene.t1[0, i]) and bool(scene.settled()[0, i]),
                      max_steps=80)
    mark()
    w2 = h["wait_window"](1)
    h["dash"](i, EXIT_X)
    h["settle_until"](lambda: bool(scene.t2[0, i]), max_steps=60)
    mark()
    if waits is not None:
        waits += [w, w2]
    bx, by = h["basket_slot"](slot)
    # rise FIRST at the lane (the basket's near wall is clear of y=0 by construction),
    # then translate at z 0.115 — above the 72 mm basket rim everywhere — and release.
    h["carry"](i, [(0.38, LANE_Y, 0.115), (bx, by, 0.115)])
    h["settle_until"](lambda: bool(scene.delivered()[0, i]), max_steps=200)
    mark()


def oracle_solution(scene_or_env, step_fn=None, milestones=None, waits=None,
                    verbose: bool = True) -> bool:
    """Teleport-oracle: solve the CURRENT episode by running the gauntlet crate by crate
    — observe each sweeper's patrol, dash through timed windows, deliver to the basket.
    Appends score() after every t1/t2/deliver milestone. Returns True iff success()."""
    env = _env_of(scene_or_env)
    scene = env.scene
    h = _mk_helpers(env, step_fn)
    for t, i in enumerate(range(scene.cfg.n_crates)):
        run_crate_gauntlet(env, h, i, slot=t, milestones=milestones, waits=waits)
        if verbose:
            print(f"[oracle] crate {i}: t1/t2/delivered="
                  f"{bool(scene.t1[0, i])}/{bool(scene.t2[0, i])}/"
                  f"{bool(scene.delivered()[0, i])} score={float(scene.score()[0]):.3f}",
                  flush=True)
    h["settle_until"](lambda: bool(scene.success()[0]), max_steps=200)
    return bool(scene.success()[0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sweeper_gauntlet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.35, -1.35, 1.05)) + o),
                                tuple(np.array((0.0, 0.0, 0.10)) + o),
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

    h = _mk_helpers(env, step)
    write_body, loc, carry = h["write_body"], h["loc"], h["carry"]
    settle_until = h["settle_until"]

    def report(tag: str) -> None:
        print(f"[smoke] {tag:14s} | t1={scene.t1[0].int().tolist()} "
              f"t2={scene.t2[0].int().tolist()} "
              f"struck={scene.struck[0].int().tolist()} "
              f"flew={scene.flew[0].int().tolist()} "
              f"in_basket={scene.in_basket()[0].int().tolist()} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def sw_phys_y(j: int) -> float:
        return float((scene.sweepers[j].data.root_pos_w - scene.env_origins)[0, 1])

    # =========================== 1. settle / no-NaN =========================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    st_all = torch.cat([b.data.root_state_w for b in scene.crates.values()]
                       + [s.data.root_state_w for s in scene.sweepers], dim=0)
    check("settle: states finite, crates settled, score 0, no success",
          bool(torch.isfinite(st_all).all()) and bool(scene.settled()[0].all())
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. sweeper drive ===========================================
    y0 = [sw_phys_y(j) for j in range(2)]
    max_disp = 0.0
    for _ in range(6):
        step(15)
        max_disp = max(max_disp, *(abs(sw_phys_y(j) - y0[j]) for j in range(2)))
    ya, _vya = scene.sweeper_analytic()
    drift = max(abs(sw_phys_y(j) - float(ya[0, j])) for j in range(2))
    print(f"[smoke] sweeper drive: max displacement {max_disp:.3f} m over 90 substeps, "
          f"physical-vs-analytic drift {drift * 1000:.1f} mm", flush=True)
    check("sweeper drive: sweepers patrol autonomously and track the analytic wave",
          max_disp > 0.08 and drift < 0.005)

    # =========================== 3. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(s)
        env.reset()
        step(2)
        pos, _v = scene._crate_tensors()
        scat = float(pos[0, :, :2].sum())
        row = [scat, float(scene.basket_xy[0, 0]), float(scene.basket_xy[0, 1]),
               sw_phys_y(0), sw_phys_y(1)]
        step(60)
        row.append(sw_phys_y(0))
        reads.append(row)
    arr = np.array(reads)
    print("[smoke] randomization readback (crate_scatter, basket_x, basket_y, sw0_y0, "
          f"sw1_y0, sw0_y@60):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: crate scatter, basket pose, sweeper phases all move (readback)",
          spread[0] > 0.02 and (spread[1] > 0.02 or spread[2] > 0.05)
          and spread[3] > 0.15 and spread[4] > 0.15)

    # =========================== 4. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(300)
    report("null-policy")
    check("null policy: score ~0 and no success after 300 idle substeps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 5-8. oracle x3 + monotonicity ==============================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        step(20)
        ms: list[float] = []
        waits: list[float] = []
        ok = oracle_solution(env, step_fn=step, milestones=ms, waits=waits)
        report(f"oracle-seed{s}")
        periods = [round(float(scene.sw_period[0, j]), 2) for j in range(2)]
        print(f"[smoke] oracle seed {s}: periods={periods}s "
              f"waits={[round(w, 2) for w in waits]}s", flush=True)
        check(f"oracle reaches success() on seed {s} (score 1.0)",
              ok and bool(scene.success()[0]) and float(scene.score()[0]) == 1.0)
        if s == 0:
            print(f"[smoke] milestones (seed 0): {[round(v, 3) for v in ms]}", flush=True)
            check("monotonicity: staged score strictly increases across the 9 "
                  "t1/t2/deliver milestones to 1.0",
                  len(ms) == 9 and all(b > a + 1e-6 for a, b in zip(ms, ms[1:]))
                  and ms[-1] == 1.0)

    # =========================== 9-11. negative A: the seed's own strategy ==================
    # pick_place/track_banana follows a prescribed path at steady tracking pace in a
    # STATIC world — no notion of timing. Express exactly that on the ground: a
    # phase-blind constant-speed carry straight across corridor 1 (0.06 m/s — exposed
    # ~4.1 s > half the slowest patrol period, so the sweeper ALWAYS arrives), then
    # carry on and place PERFECTLY in the basket.
    torch.manual_seed(51)
    env.reset()
    step(20)
    carry(0, [(WAIT_X1, LANE_Y, CARRY_Z)])
    carry(0, [(MED_X, LANE_Y, CARRY_Z)], step_len=0.0005)  # 0.06 m/s, phase-blind
    report("seed-carry")
    check("negative A (seed strategy): phase-blind constant-speed carry is struck",
          bool(scene.struck[0, 0]))
    bx, by = float(scene.basket_xy[0, 0]), float(scene.basket_xy[0, 1])
    carry(0, [(EXIT_X, LANE_Y, CARRY_Z), (0.38, LANE_Y, 0.115), (bx, by, 0.115)])
    settle_until(lambda: bool(scene.settled()[0, 0]), max_steps=150)
    report("seed-placed")
    check("negative A: perfect basket pose, yet refused (score <= 0.05, no success)",
          bool(scene.in_basket()[0, 0]) and not bool(scene.delivered()[0, 0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    # latch permanence: re-run the SAME crate cleanly (window-timed) — still refused.
    write_body(scene.crates["crate_0"], (WAIT_X1, LANE_Y, 0.030))
    step(10)
    h["wait_window"](0)
    h["dash"](0, MED_X)
    step(20)
    h["wait_window"](1)
    h["dash"](0, EXIT_X)
    carry(0, [(0.38, LANE_Y, 0.115), (bx, by, 0.115)])
    settle_until(lambda: bool(scene.settled()[0, 0]), max_steps=150)
    report("re-gauntleted")
    check("negative A permanence: clean window-timed re-run of the struck crate is "
          "still refused",
          bool(scene.in_basket()[0, 0]) and bool(scene.struck[0, 0])
          and not bool(scene.delivered()[0, 0]) and float(scene.score()[0]) <= 0.05)

    # =========================== 12. negative B: the seed's aerial arc ======================
    # The seed's waypoints rise to z 0.36-0.47: carry the crate HIGH over both corridors
    # (no sweeper can touch it up there) and lower it into the basket, perfectly.
    torch.manual_seed(61)
    env.reset()
    step(20)
    bx, by = float(scene.basket_xy[0, 0]), float(scene.basket_xy[0, 1])
    p0 = loc(scene.crates["crate_0"]).tolist()
    carry(0, [(p0[0], p0[1], 0.36), (bx, by, 0.36), (bx, by, 0.115)], step_len=0.02)
    settle_until(lambda: bool(scene.settled()[0, 0]), max_steps=150)
    report("aerial-arc")
    check("negative B (seed aerial arc): flying over a corridor trips `flew`; perfect "
          "basket pose refused",
          bool(scene.flew[0, 0]) and not bool(scene.struck[0, 0])
          and not bool(scene.t1[0, 0]) and bool(scene.in_basket()[0, 0])
          and not bool(scene.delivered()[0, 0]) and float(scene.score()[0]) <= 0.05)

    # =========================== 13. negative C: teleport (anti-cheat) ======================
    torch.manual_seed(71)
    env.reset()
    step(20)
    bx, by = float(scene.basket_xy[0, 0]), float(scene.basket_xy[0, 1])
    write_body(scene.crates["crate_0"], (bx, by, 0.06))
    step(60)
    report("teleport")
    check("negative C: teleported-to-basket crate has no transit latches -> not "
          "delivered, score ~0",
          bool(scene.in_basket()[0, 0]) and not bool(scene.t1[0, 0])
          and not bool(scene.delivered()[0, 0]) and float(scene.score()[0]) <= 0.02)

    # =========================== 14-15. near-miss + recovery ================================
    torch.manual_seed(81)
    env.reset()
    step(20)
    carry(0, [(WAIT_X1, LANE_Y, CARRY_Z), (WAIT_X1, LANE_Y, 0.030)])
    h["wait_window"](0)
    h["dash"](0, MED_X)
    step(20)
    h["wait_window"](1)
    h["dash"](0, EXIT_X)
    settle_until(lambda: bool(scene.t2[0, 0]), max_steps=60)
    bx, by = float(scene.basket_xy[0, 0]), float(scene.basket_xy[0, 1])
    side_y = by + math.copysign(0.15, by)  # beside the basket, on its far side
    carry(0, [(0.38, LANE_Y, 0.115), (bx, side_y, 0.115)])
    settle_until(lambda: bool(scene.settled()[0, 0]), max_steps=120)
    report("near-miss")
    sc_nm = float(scene.score()[0])
    check("near-miss: clean gauntlet but set down BESIDE the basket -> t2 credit only "
          "(~0.18), no success",
          bool(scene.t2[0, 0]) and not bool(scene.struck[0, 0])
          and not bool(scene.in_basket()[0, 0])
          and abs(sc_nm - c.credit_t2) < 0.03 and not bool(scene.success()[0]))
    pr = loc(scene.crates["crate_0"]).tolist()  # rise vertically from the true rest pose
    carry(0, [(pr[0], pr[1], 0.115), (bx, by, 0.115)])
    ok = settle_until(lambda: bool(scene.delivered()[0, 0]), max_steps=150)
    report("recovered")
    check("near-miss recovery: moved into the basket -> delivery credit (~0.30)",
          ok and abs(float(scene.score()[0]) - c.credit_delivered) < 0.03
          and float(scene.score()[0]) > sc_nm)

    # =========================== 16-17. calibration: the timing cliff =======================
    # The task's core resource is the patrol phase. Launch the SAME dash (a) while the
    # sweeper approaches the lane from 0.22-0.28 m out (arrives during the crate's
    # ~0.4 s danger exposure -> guaranteed hit, dry-computed) and (b) through the
    # oracle's window trigger. Published per-seed with period/speed stats.
    print("[smoke] CALIBRATION: mistimed vs window-timed dash, 3 fresh seeds each",
          flush=True)
    struck_bad = 0
    clean_good = 0
    for s in (91, 92, 93):
        torch.manual_seed(s)
        env.reset()
        step(10)
        periods = [round(float(scene.sw_period[0, j]), 2) for j in range(2)]
        carry(0, [(WAIT_X1, LANE_Y, CARRY_Z), (WAIT_X1, LANE_Y, 0.030)])
        h["wait_collision_course"](0)
        h["dash"](0, MED_X)
        step(10)
        hit = bool(scene.struck[0, 0])
        struck_bad += int(hit)
        print(f"[smoke]   seed {s} periods={periods}s mistimed dash -> struck={hit}",
              flush=True)
        # park the struck crate out of the dash lane, then run a fresh crate through
        # the SAME episode's window trigger — it must stay clean
        write_body(scene.crates["crate_0"], (-0.45, -0.35, 0.030))
        step(5)
        carry(1, [(WAIT_X1, LANE_Y, CARRY_Z), (WAIT_X1, LANE_Y, 0.030)])
        w = h["wait_window"](0)
        h["dash"](1, MED_X)
        step(10)
        clean = bool(scene.t1[0, 1]) and not bool(scene.struck[0, 1])
        clean_good += int(clean)
        print(f"[smoke]   seed {s} window-timed dash (wait {w:.2f}s) -> clean={clean}",
              flush=True)
    check("calibration: mistimed dash (sweeper approaching) struck 3/3", struck_bad == 3)
    check("calibration: window-timed dash clean 3/3", clean_good == 3)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sweeper_gauntlet")
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
