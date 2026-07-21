"""Smoke / oracle test for PouringScene — NullRobot, kinematic cup hold, RECORDED.

The metered split-pour pipeline plus the brief's robot-free feasibility spike:
  1. tilt-hysteresis sweep (the feasibility spike): a mounted cup over bowl 0 ramps
     its tilt up until pellets flow (theta_start) and back down until the stream dies
     (theta_stop), at three fill levels — the measured hysteresis curve IS the task's
     difficulty and gets published; pellet conservation (no tunneling at 120 Hz) is
     asserted throughout;
  2. oracle solve x N (fresh episode each): pick the cup up, closed-loop metered pour
     into the sampled MAJOR bowl until the live count hits the sampled target (ramp to
     flow, back off early, settle, top up), empty the rest into the minor bowl, park
     the cup upright — assert the split chain's 6 staged flags, score 100, success();
  3. negative controls: dumping EVERYTHING into the major bowl must fail the split
     goal (and, goal-switched at runtime, must PASS the faithful v0 "pour_all" —
     the spatula curriculum-knob check); >spill_max pellets at rest on the bench must
     block success, and a spilled pellet later knocked into a bowl must stay spilled
     (never counted as delivered).

Video is recorded throughout (viewport rgb annotator, the proven server recipe).

    python -m robobench.suites.articulated.smokes.pouring_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean split-pour oracle (no sweep, no negatives)")
parser.add_argument("--only_sweep", action="store_true", default=False,
                    help="skip straight to the tilt-hysteresis sweep (fast iteration "
                         "on the flow physics)")
parser.add_argument("--repeats", type=int, default=3,
                    help="oracle episodes (the brief's 20/20 spike raises this)")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="/tmp/pouring_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional hdfs dir to push the npz to (empty = skip)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import json
import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


# ---- float quat helpers (w, x, y, z tuples; env-0 staging math — the spatula kit) ----
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qx(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def qnlerp(a, b, f):
    if sum(u * v for u, v in zip(a, b)) < 0.0:
        b = tuple(-v for v in b)
    q = tuple(u + (v - u) * f for u, v in zip(a, b))
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return tuple(v / n for v in q)


def qang_vel(a, b, dt: float):
    d = qmul(b, (a[0], -a[1], -a[2], -a[3]))
    if d[0] < 0.0:
        d = tuple(-v for v in d)
    return (2.0 * d[1] / dt, 2.0 * d[2] / dt, 2.0 * d[3] / dt)


def ease(f: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * f)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[pour-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("articulated.pouring")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n_envs = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n_envs, device=device)

    # --- recording (viewport rgb annotator, the proven server recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        cam = (0.95, -1.00, 0.70) if not args.demo else (0.80, -0.90, 0.60)
        tgt = (0.0, 0.0, c.surface_z + 0.10)
        env.sim.set_camera_view(tuple(np.array(cam) + o), tuple(np.array(tgt) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[pour-smoke] camera ready shape={np.asarray(annot.get_data()).shape}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[pour-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0
    frame_steps: list[int] = []
    telem: list[tuple] = []
    phases: list[tuple[int, str]] = []

    def phase(label: str) -> None:
        phases.append((step_i, label))
        print(f"[pour-smoke] PHASE @{step_i}: {label}", flush=True)

    def cnt() -> dict[str, int]:
        d = scene.counts()
        return {"cup": int(d["in_cup"][0]), "a": int(d["in_bowl"][0, 0]),
                "b": int(d["in_bowl"][0, 1]), "major": int(d["in_major"][0]),
                "minor": int(d["in_minor"][0]), "spilled": int(d["spilled"][0]),
                "air": int(d["air"][0]), "lost": int(d["lost"][0])}

    # --- kinematic cup hold (the hard-won pen-holder/spatula velocity semantics, verbatim):
    # every pre-step write carries +g*dt so gravity integration cancels; a MOVING hold
    # writes the OLD pose + the transport velocity and lets integration carry the cup
    # (contacts stay continuous — the pellets are dragged, never phased through);
    # transport is single-step-scoped; after each step() the cup is re-pinned at zero
    # velocity + update(0.0) before any judging.
    hold_write: torch.Tensor | None = None
    hold_pin: torch.Tensor | None = None
    g_dt = 9.81 * env.dt

    def step(k: int = 1) -> None:
        nonlocal step_i, hold_write
        for _ in range(k):
            if hold_pin is not None:
                pre = hold_write.clone()
                pre[:, 9] += g_dt
                scene.cup.write_root_state_to_sim(pre, all_ids)
            env.step(no_action, render=True)
            if hold_pin is not None:
                hold_write = hold_pin.clone()
            d = cnt()
            telem.append((step_i, float(scene.cup_tilt_deg()[0]), d["cup"], d["a"],
                          d["b"], d["spilled"], d["air"], int(scene.score()[0])))
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
                    frame_steps.append(step_i)
            step_i += 1
        if hold_pin is not None:
            scene.cup.write_root_state_to_sim(hold_pin, all_ids)
            env.iscene.update(0.0)

    def make_state(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(n_envs, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        st[:, 3:7] = torch.tensor(quat, device=device)
        return st

    cur = {"pos": None, "quat": None}

    def hold(pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        nonlocal hold_write, hold_pin
        pos = tuple(float(v) for v in pos)
        quat = tuple(float(v) for v in quat)
        if cur["pos"] is None:
            hold_write = make_state(pos, quat)
        else:
            hold_write = make_state(cur["pos"], cur["quat"])
            hold_write[:, 7] = (pos[0] - cur["pos"][0]) / env.dt
            hold_write[:, 8] = (pos[1] - cur["pos"][1]) / env.dt
            hold_write[:, 9] = (pos[2] - cur["pos"][2]) / env.dt
            hold_write[:, 10:13] = torch.tensor(qang_vel(cur["quat"], quat, env.dt),
                                                device=device)
        hold_pin = make_state(pos, quat)
        cur["pos"], cur["quat"] = pos, quat

    def move_to(pos, quat, steps: int) -> None:
        p0, q0 = cur["pos"], cur["quat"]
        for t in range(steps):
            f = ease((t + 1) / steps)
            p = tuple(p0[i] + (pos[i] - p0[i]) * f for i in range(3))
            hold(p, qnlerp(q0, quat, f))
            step(1)
        hold(pos, quat)
        step(1)

    def release() -> None:
        nonlocal hold_write, hold_pin
        hold_write = None
        hold_pin = None
        cur["pos"], cur["quat"] = None, None

    def settle_until(pred, max_steps: int = 300, poll: int = 5) -> bool:
        for _ in range(0, max_steps, poll):
            if bool(pred()):
                return True
            step(poll)
        return bool(pred())

    # --- staging helpers (env-local coords) ---
    CARRY_Z = c.surface_z + 0.28  # cup-centre height for upright carries
    # Pour station: at flow angles (~97 deg) the lip sits ~5 cm -y and ~4 cm below the
    # cup centre, so this centre offset/height puts the exit right over the bowl axis
    # ~3 cm above the rim (GPU round 3: a higher, less-offset station let launched
    # pellets clear the far rim).
    POUR_Z = c.surface_z + c.bowl_h + 0.075  # cup-centre height while pouring
    POUR_DY = 0.052  # cup centre this far +y of the bowl axis; qx(+t) tips the lip -y

    def bowl_xy(b: int) -> tuple[float, float]:
        p = scene.bowls[b].data.root_pos_w[0] - env.iscene.env_origins[0]
        return (float(p[0]), float(p[1]))

    def pickup() -> None:
        """Grab the cup at its resting spot and raise it to an upright carry."""
        p = scene.cup.data.root_pos_w[0] - env.iscene.env_origins[0]
        hold((float(p[0]), float(p[1]), float(p[2])))
        step(2)
        move_to((float(p[0]), float(p[1]), CARRY_Z), (1.0, 0.0, 0.0, 0.0), 70)

    def over_bowl(b: int, tilt_deg: float = 0.0, steps: int = 90) -> None:
        """Travel UPRIGHT to the pour station, then tilt in place (tilting mid-travel
        both risks dribbling on the bench and pollutes the carry-tilt metric)."""
        bx, by = bowl_xy(b)
        move_to((bx, by + POUR_DY, POUR_Z), (1.0, 0.0, 0.0, 0.0), steps)
        if tilt_deg:
            ramp_tilt(tilt_deg, 1.0)

    def put_pellet(i: int, pos, settle: int = 0) -> None:
        st = make_state(pos)
        scene.pellets[i].write_root_state_to_sim(st, all_ids)
        if settle:
            step(settle)

    def load_cup(fill: int) -> None:
        """(Privileged, sweep only) re-seed `fill` pellets into the held, upright cup
        and reset the episode's latched masks — a clean fill-level measurement."""
        px, py, pz = cur["pos"]
        slots = scene._fill_slots_local(scene.cfg.max_pellets)
        for i in range(scene.cfg.max_pellets):
            if i < fill:
                lx, ly, lz = slots[i]
                put_pellet(i, (px + lx, py + ly, pz + lz))
            else:
                gx, gy = scene._reserve_slot(i)
                put_pellet(i, (gx, gy, c.pellet_r + 0.001))
        scene._active[:] = False
        scene._active[:, :fill] = True
        scene._n_active[:] = fill
        scene._spilled[:] = False
        scene._lost[:] = False
        scene._was_in_cup[:] = scene._active.clone()
        scene._since_out[:] = 10_000
        scene._stream_active[:] = False
        scene._flags[:] = False
        step(50)  # settle the stack

    # --- pour maneuvers (all through the kinematic hold) ---
    def ramp_tilt(to_deg: float, rate: float) -> None:
        t0 = _cur_tilt()
        n_steps = max(1, int(abs(to_deg - t0) / rate))
        px, py, pz = cur["pos"]
        for k in range(n_steps):
            t = t0 + (to_deg - t0) * (k + 1) / n_steps
            hold((px, py, pz), qx(t))
            step(1)

    def _cur_tilt() -> float:
        # commanded tilt (the hold's quat), not the measured one — exact for qx quats
        w = cur["quat"][0]
        return math.degrees(2 * math.acos(max(-1.0, min(1.0, w))))

    def measure_hysteresis(fill: int) -> tuple[float, float, bool, bool]:
        """Ramp tilt up over bowl 0 until pellets flow, then back down until the
        stream dies. Returns (theta_start, theta_stop, conserved, emptied) — a stop
        angle is only a hysteresis measurement if the cup still held pellets when the
        stream died (a starved stream at low fill dies at ANY angle; GPU round 1)."""
        over_bowl(0, 0.0, 90)
        load_cup(fill)
        ramp_tilt(35.0, 1.0)
        px, py, pz = cur["pos"]
        theta_start, last_out_tilt = None, None
        t = 35.0
        while t < 130.0:
            t += 0.30
            hold((px, py, pz), qx(t))
            step(1)
            if int(scene._since_out[0]) == 0:
                last_out_tilt = t
                if theta_start is None:
                    theta_start = t
                if t > theta_start + 4.0:
                    break  # committed stream measured; stop ramping
        theta_stop = None
        if theta_start is not None:
            quiet = 0
            while t > 5.0 and quiet < c.stream_gap + 20:
                t -= 0.60  # come back down briskly: preserve pellets at low fills
                hold((px, py, pz), qx(t))
                step(1)
                if int(scene._since_out[0]) == 0:
                    last_out_tilt = t
                    quiet = 0
                else:
                    quiet += 1
            theta_stop = last_out_tilt
        emptied = cnt()["cup"] == 0
        ramp_tilt(0.0, 1.0)
        settle_until(lambda: cnt()["air"] == 0, 240)
        d = cnt()
        accounted = d["cup"] + d["a"] + d["b"] + d["spilled"] + d["air"]
        conserved = (accounted == fill) and d["lost"] == 0
        print(f"[pour-smoke]   fill={fill}: theta_start={theta_start} "
              f"theta_stop={theta_stop}{' (UNMEASURED: cup emptied)' if emptied else ''} "
              f"counts={d} accounted={accounted}", flush=True)
        return theta_start, theta_stop, conserved, emptied

    def pour_into(b: int, stop_at: int, guard: int = 3600, label: str = "") -> int:
        """Closed-loop metered pour into bowl `b` until its count reaches `stop_at`.
        Control law (each clause bought by a GPU round):
          - creep the tilt up only while NOT flowing; once pellets stream, HOLD the
            angle so the pile surface relaxes instead of avalanching (round 5: creeping
            through the flow fed the avalanche a systematic +2..+3 overshoot);
          - within 3 of the target, CATCH the stream mid-flow (snap below the stop
            angle, let the in-flight land, recount) — pulsed 1-3 pellet deliveries;
          - if committed (= in bowl + in flight) covers the target, stop the stream
            and re-check settled; top up from below if short."""
        over_bowl(b, 30.0, 90)
        px, py, pz = cur["pos"]
        t, used, flow_tilt, pulses = 30.0, 0, None, 0
        key = "a" if b == 0 else "b"

        def catch(back: float, why: str, d: dict) -> None:
            nonlocal t, pulses
            pulses += 1
            print(f"[pour-smoke]   [{label}] catch#{pulses} ({why}) tilt={t:.1f} "
                  f"in_bowl={d[key]} air={d['air']} cup={d['cup']}", flush=True)
            ramp_tilt(max(30.0, t - back), 2.5)
            t = _cur_tilt()
            settle_until(lambda: cnt()["air"] == 0, 240)

        while used < guard:
            d = cnt()
            committed = d[key] + d["air"]
            if committed >= stop_at or d["cup"] == 0:
                catch(30.0, "committed>=target" if d["cup"] else "cup empty", d)
                d = cnt()
                if d[key] >= stop_at or d["cup"] == 0:
                    break
                used += 1
                continue
            flowing = int(scene._since_out[0]) <= 8
            if flowing:
                if flow_tilt is None:
                    flow_tilt = t
                    print(f"[pour-smoke]   [{label}] flow onset at {t:.1f}deg "
                          f"(in_bowl={d[key]})", flush=True)
                if stop_at - committed <= 3:
                    catch(25.0, f"near target ({committed}/{stop_at})", d)
                else:
                    step(1)  # HOLD the tilt: never ramp into an active stream
                used += 1
                continue
            # not flowing: fast approach while safely below the known flow angle,
            # creep near it (anticipatory metering)
            rate = 0.35 if flow_tilt is None or t < flow_tilt - 8.0 else 0.12
            t = min(t + rate, 128.0)
            hold((px, py, pz), qx(t))
            step(1)
            used += 1
        ramp_tilt(0.0, 1.5)
        settle_until(lambda: cnt()["air"] == 0, 240)
        got = cnt()[key]
        print(f"[pour-smoke]   [{label}] pour done: in_bowl={got} target={stop_at} "
              f"pulses={pulses} steps={used}", flush=True)
        return got

    def empty_into(b: int, guard: int = 8) -> None:
        """Pour the WHOLE remaining load into bowl `b` (the v0 move). At steep tilt
        the pellets slide the full cup length and LAUNCH along the axis (GPU round 3:
        a 132-135 deg jiggle at pour height fired half the load over the far rim and
        the escapees rolled off the bench) — so once tilted, sink the station until
        the lip hovers at the rim plane and top out at 115 deg: past vertical, the
        floor overhangs and drains, and released pellets have nowhere to fly."""
        over_bowl(b, 100.0, 90)
        bx, by = bowl_xy(b)
        # Sink until the lip rides AT/BELOW the rim plane (round 4: stragglers slide
        # the whole cup and exit at ~0.9 m/s; released above the rim they can clear
        # it — released below it, the bowl wall is geometrically in the way).
        move_to((bx, by + POUR_DY + 0.012, POUR_Z - 0.03), qx(100.0), 40)
        for _ in range(guard):
            if cnt()["cup"] == 0:
                break
            for jig in (106.0, 96.0, 112.0, 98.0):
                ramp_tilt(jig, 1.0)
                step(12)
        ramp_tilt(0.0, 1.5)
        move_to((bx, by + POUR_DY, POUR_Z + 0.02), (1.0, 0.0, 0.0, 0.0), 40)
        settle_until(lambda: cnt()["air"] == 0, 300)

    def park_cup() -> bool:
        move_to((cur["pos"][0], cur["pos"][1], CARRY_Z), (1.0, 0.0, 0.0, 0.0), 60)
        move_to((c.park_pos[0], c.park_pos[1], CARRY_Z), (1.0, 0.0, 0.0, 0.0), 80)
        move_to((c.park_pos[0], c.park_pos[1], c.surface_z + c.cup_h / 2 + 0.004),
                (1.0, 0.0, 0.0, 0.0), 80)
        release()
        return settle_until(lambda: bool(scene.cup_parked()[0]), 300)

    def dump_air(tag: str) -> None:
        """Pinpoint any unresolved 'air' pellets (position/velocity/region) — the
        round-2 failures were only diagnosable from exactly this."""
        if cnt()["air"] == 0:
            return
        o = env.iscene.env_origins[0]
        print(f"[pour-smoke] {tag}: unresolved AIR pellets:", flush=True)
        for i, p in enumerate(scene.pellets):
            if not bool(scene._active[0, i]) or bool(scene._spilled[0, i]):
                continue
            reg = int(scene._region[0, i])
            v = float(p.data.root_lin_vel_w[0].norm())
            counted = reg == 0 or (reg in (1, 2) and v < c.count_speed)
            if not counted:
                pp = (p.data.root_pos_w[0] - o).tolist()
                print(f"[pour-smoke]   pellet {i}: pos=({pp[0]:+.3f},{pp[1]:+.3f},"
                      f"{pp[2]:+.3f}) |v|={v:.3f} region={reg}", flush=True)

    def dump_spilled(tag: str) -> None:
        """Where did the losses come to rest? (round 4: spill counts alone could not
        say which pour phase shed them or where they landed)."""
        ids = [i for i in range(c.max_pellets)
               if bool(scene._active[0, i]) and bool(scene._spilled[0, i])]
        if not ids:
            return
        o = env.iscene.env_origins[0]
        print(f"[pour-smoke] {tag}: spilled pellet resting spots:", flush=True)
        for i in ids:
            pp = (scene.pellets[i].data.root_pos_w[0] - o).tolist()
            print(f"[pour-smoke]   pellet {i}: ({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:+.3f})",
                  flush=True)

    def oracle_once(tag: str) -> bool:
        env.reset()
        step(60)
        n = int(scene._n_active[0])
        major = int(scene._major[0])
        target = int(scene._target[0])
        phase(f"ORACLE {tag}: n={n} major={c.bowl_names[major]} target={target}"
              f"+/-{c.count_tol}")
        pickup()
        got = pour_into(major, target, label=tag)
        in_band = abs(got - target) <= c.count_tol
        check(f"{tag}-major-band", in_band,
              f"major={got} target={target}+/-{c.count_tol}")
        sp_meter = cnt()["spilled"]
        print(f"[pour-smoke] {tag}: spills during metered pour = {sp_meter}", flush=True)
        empty_into(1 - major)
        print(f"[pour-smoke] {tag}: spills during empty-into-minor = "
              f"{cnt()['spilled'] - sp_meter}", flush=True)
        parked = park_cup()
        # let straggler pellets resolve (in-bowl count or spill-window latch: the
        # displacement latch needs up to 2 x spill_grace substeps at rest)
        settle_until(lambda: cnt()["air"] == 0, 3 * c.spill_grace)
        dump_air(tag)
        dump_spilled(tag)
        d = cnt()
        ok = bool(scene.success()[0])
        chain_flags = int(scene.stage_flags()[0, list(c.CHAINS['split'])].long().sum())
        check(f"{tag}-flags-6", chain_flags == 6,
              f"{chain_flags}/6: {scene._flags[0].tolist()}")
        check(f"{tag}-score-100", int(scene.score()[0]) == 100,
              f"score={int(scene.score()[0])}")
        check(f"{tag}-success", ok, f"parked={parked} counts={d}")
        check(f"{tag}-conserved", d["lost"] == 0 and
              d["cup"] + d["a"] + d["b"] + d["spilled"] + d["air"] == n, f"{d}")
        print(f"[pour-smoke] {tag} metrics: pour_events={int(scene.pour_events()[0])} "
              f"carry_tilt_max={float(scene.carry_tilt_max()[0]):.1f}deg "
              f"spilled={d['spilled']}", flush=True)
        return ok and in_band

    # ---- run --------------------------------------------------------------------------------
    env.reset()
    step(60)
    print(f"[pour-smoke] sampled: n={int(scene._n_active[0])} "
          f"major={c.bowl_names[int(scene._major[0])]} target={int(scene._target[0])}",
          flush=True)
    print(f"[pour-smoke] describe():\n{scene.describe()}", flush=True)

    if args.demo:
        oracle_once("demo")
        save_npz(frames, frame_steps, telem, phases)
        print("POURING_SMOKE_DONE", flush=True)
        env.close()
        return

    # 1. feasibility spike: tilt hysteresis vs fill level (published curve)
    phase("hysteresis sweep: theta_start/theta_stop vs fill")
    pickup()
    curve = []
    for fill in (c.max_pellets, 16, 8):
        ts, tp, conserved, emptied = measure_hysteresis(fill)
        curve.append((fill, ts, tp if not emptied else None))
        check(f"sweep-fill{fill}-flows", ts is not None and ts < 130.0,
              f"theta_start={ts}")
        # A starved stream (cup emptied) dies at any angle — no stop measurement there.
        check(f"sweep-fill{fill}-hysteresis",
              emptied or (ts is not None and tp is not None and ts > tp - 1e-6),
              f"start={ts} stop={tp}" + (" (skipped: cup emptied)" if emptied else ""))
        check(f"sweep-fill{fill}-conserved", conserved)
    print(f"[pour-smoke] HYSTERESIS fill -> (theta_start, theta_stop): {curve} "
          f"(the published flow-valve curve; None = starved before the stop angle)",
          flush=True)
    release()

    if args.only_sweep:
        print(f"[pour-smoke] RESULT: "
              f"{'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}", flush=True)
        save_npz(frames, frame_steps, telem, phases)
        print("POURING_SMOKE_DONE", flush=True)
        env.close()
        return

    # 2. oracle solves (fresh episode each — the 20/20 spike raises --repeats)
    hits = sum(int(oracle_once(f"oracle{r}")) for r in range(args.repeats))
    check("oracle-hit-rate", hits == args.repeats, f"{hits}/{args.repeats}")

    # 3a. NEGATIVE: dump-all fails the split, passes the runtime-switched v0
    env.reset()
    step(60)
    major = int(scene._major[0])
    n = int(scene._n_active[0])
    phase("NEGATIVE: dump everything into the major bowl")
    pickup()
    empty_into(major)
    park_cup()
    d = cnt()
    check("negative-dump-fails-split", not bool(scene.success()[0])
          and int(scene.score()[0]) < 100,
          f"counts={d} score={int(scene.score()[0])}")
    scene.cfg.goal = "pour_all"  # the spatula runtime curriculum-knob switch
    need = math.ceil(c.pour_all_frac * n)
    check("v0-pour-all-succeeds", d["major"] >= need and bool(scene.success()[0])
          and int(scene.score()[0]) == 100,
          f"major={d['major']} need={need} score={int(scene.score()[0])}")
    scene.cfg.goal = "split"

    # 3b. NEGATIVE: spills block success and stay spilled forever
    env.reset()
    step(30)
    n = int(scene._n_active[0])
    major = int(scene._major[0])
    target = int(scene._target[0])
    phase(f"NEGATIVE: {c.spill_max + 2} pellets on the bench; perfect split otherwise")
    spill_ids = list(range(c.spill_max + 2))
    for k, i in enumerate(spill_ids):
        put_pellet(i, (0.30 + 0.05 * (k % 2), -0.30 - 0.05 * (k // 2),
                       c.surface_z + c.pellet_r + 0.002))
    step(3 * c.spill_grace)  # at rest through >= 2 displacement windows -> latch
    d = cnt()
    check("spill-latches", d["spilled"] == len(spill_ids), f"spilled={d['spilled']}")
    rest = [i for i in range(n) if i not in spill_ids]
    slot = [0, 0]  # per-bowl golden-spiral fill slot (16/layer keeps piles below rim)
    for k, i in enumerate(rest):
        b = major if k < target else 1 - major
        bx, by = bowl_xy(b)
        s = slot[b]
        slot[b] += 1
        ring = 0.045 * math.sqrt((s % 16) / 16)
        ang = s * 2.399
        z = c.surface_z + c.pad_size[2] + c.bowl_bot_t + c.pellet_r \
            + 0.017 * (s // 16) + 0.002
        put_pellet(i, (bx + ring * math.cos(ang), by + ring * math.sin(ang), z))
    scene.cup.write_root_state_to_sim(
        make_state((c.park_pos[0], c.park_pos[1], c.surface_z + c.cup_h / 2 + 0.002)),
        all_ids)
    step(120)
    d = cnt()
    # target may exceed len(rest) - minor needs >=1 either way; judge what landed
    check("negative-spill-blocks", not bool(scene.success()[0]),
          f"counts={d} (band met: {abs(d['major'] - target) <= c.count_tol})")
    before = cnt()["major"]
    bx, by = bowl_xy(major)
    put_pellet(spill_ids[0], (bx, by, c.surface_z + c.pad_size[2] + c.bowl_h + 0.03),
               settle=120)
    check("spilled-stays-spilled", cnt()["major"] == before
          and cnt()["spilled"] == len(spill_ids),
          f"major {before}->{cnt()['major']} spilled={cnt()['spilled']}")

    print(f"[pour-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)
    save_npz(frames, frame_steps, telem, phases)
    print("POURING_SMOKE_DONE", flush=True)
    env.close()


def save_npz(frames, frame_steps, telem, phases) -> None:
    if not frames:
        return
    arr = np.stack(frames, axis=0)
    t = np.array(telem, dtype=np.float32)
    np.savez_compressed(
        args.out, frames=arr, env="articulated.pouring",
        frame_steps=np.array(frame_steps, dtype=np.int64),
        telem_step=t[:, 0], telem_tilt=t[:, 1], telem_cup=t[:, 2], telem_a=t[:, 3],
        telem_b=t[:, 4], telem_spilled=t[:, 5], telem_air=t[:, 6], telem_score=t[:, 7],
        phases=json.dumps(phases))
    print(f"[pour-smoke] saved {arr.shape} (+telemetry) -> {args.out}", flush=True)
    if args.hdfs_dir:
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} "
                       f"{args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[pour-smoke] hdfs upload rc={rc}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
