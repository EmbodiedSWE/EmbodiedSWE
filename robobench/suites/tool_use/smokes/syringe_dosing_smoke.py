"""Smoke / oracle test for SyringeDosingScene — NullRobot, RECORDED.

The full triple-dose pipeline, NullRobot style (the barrel is CARRIED by teleport-glide
since there are no hands; the plunger is metered through `scene.plunger_drive`, the same
interface a robot thumb or an RL policy acts through):
  1. draw — glide the syringe over the reservoir, seat the tip, pull the plunger to a
     full draw; verify liquid == travel while seated;
  2. air check — pull/push over empty bench; verify NOTHING moves (no liquid gained,
     none dosed);
  3. dispense x3 — seat on each sample well, push exactly stroke/3 worth, verify each
     dose lands in [28%, 38%];
  4. park — glide back over the stand, drop in; verify `parked()` and full `success()`;
  5. dose-repeatability calibration — the dispense error across the 3 wells (the
     metering noise floor must be well inside the +/-5% band);
  6. negative control — a fresh episode where well #1 gets HALF the load: wells 2/3
     then cannot reach the band and success() must stay False (irreversibility is real).

Barrel carrying uses kinematic-style root-velocity writes (write_root_state 1cm-glide
per step) — gentle enough for the prismatic joint to follow. Video recorded throughout.

    python -m robobench.suites.tool_use.smokes.syringe_dosing_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (draw, 3 doses, park) "
                         "— no probes, no air check, no negative control")
parser.add_argument("--record_every", type=int, default=6)
parser.add_argument("--out", type=str, default="syringe_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[dose-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    from robobench.suites.tool_use.scenes import SyringeDosingSceneCfg

    env = ENVS.get("tool_use.syringe")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=SyringeDosingSceneCfg(debug_forensics=True))
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    n = env.num_envs
    ids = torch.arange(n, device=device)

    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.45, -0.55, 0.45)) + o),
                                tuple(np.array((0.03, 0.05, 0.12)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[dose-smoke] camera ready shape={np.asarray(annot.get_data()).shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[dose-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    # --- carrying: glide the whole syringe assembly (barrel-relative offsets preserved) ---
    SYR = ["barrel", "nozzle", "flange_0", "flange_1", "plunger", "thumb_ring"]

    def bodies():
        return {"barrel": scene.barrel, "nozzle": scene.nozzle,
                "flange_0": scene.flanges[0], "flange_1": scene.flanges[1],
                "plunger": scene.plunger, "thumb_ring": scene.ring}

    held: dict[str, torch.Tensor] = {}  # body -> pinned root pose (the "hand" holding it)

    def grab() -> None:
        """Record the current pose of EVERY syringe body as the held pose — including
        the plunger/ring: leaving them free while teleport-carrying the barrel makes
        the prismatic joint chase 1cm/step violations until it ends up 18cm outside
        its limits (v9). The hand carries the whole syringe; metering happens only
        while parked at a well, where meter() re-frees the plunger."""
        held.clear()
        for name, b in bodies().items():
            held[name] = b.data.root_state_w.clone()

    def hold() -> None:
        """Re-pin the held bodies (kinematic hold — the smoke's stand-in for a fixture
        hand; the plunger stays free so metering physics is real)."""
        for name, st in held.items():
            pin = st.clone()
            pin[:, 7:13] = 0.0
            bodies()[name].write_root_state_to_sim(pin, ids)

    def release() -> None:
        held.clear()

    def glide_to(tip_target: torch.Tensor, speed: float = 0.010, settle: int = 20) -> None:
        """Move the syringe so the nozzle TIP lands on tip_target: RIGID teleport-glide
        of every syringe body from a FORMATION captured at glide start, then HOLD.
        v0-v31 glided incrementally from each body's current pose with velocities
        zeroed — the un-compensated barrel free-falls ~0.3 mm/substep between
        teleports while the gravity-compensated plunger doesn't, so every carry
        ratcheted the plunger up ~0.3 mm/step (the draw started 40 mm 'pre-drawn'
        with air and liquid capped at ~0.64)."""
        ref = {nm: b.data.root_state_w.clone() for nm, b in bodies().items()}
        off = torch.zeros(1, 3, device=device)
        for _ in range(400):
            tip = scene.tip_pos()
            d = tip_target - tip
            dist = float(d.norm(dim=-1)[0])
            if dist < 0.001:
                break
            off = off + d * min(1.0, speed / max(dist, 1e-9))
            for nm, b in bodies().items():
                st = ref[nm].clone()
                st[:, 0:3] += off
                st[:, 7:13] = 0.0
                b.write_root_state_to_sim(st, ids)
            step(1)
        else:
            print(f"[dose-smoke]   WARNING glide_to NOT CONVERGED: residual "
                  f"{float((tip_target - scene.tip_pos()).norm(dim=-1)[0]) * 1000:.1f}mm "
                  f"(obstructed corridor?)", flush=True)
        grab()
        for _ in range(settle):
            hold()
            step(1)

    def move_tip(target: torch.Tensor, safe_h: float = 0.32) -> None:
        """Up-over-down tip move (never drags the nozzle laterally through pucks/posts).
        safe_h is TIP height: 0.32 puts the barrel bottom (tip+0.05) above the 0.26 m
        stand posts, so entering/leaving the pocket is clean."""
        cur = scene.tip_pos().clone()
        up = cur.clone()
        up[:, 2] = c.surface_z + safe_h
        glide_to(up, settle=2)
        over = target.clone()
        over[:, 2] = c.surface_z + safe_h
        glide_to(over, settle=2)
        glide_to(target)

    def meter(target_travel: float, force: float, guard: int = 900,
              seat_fn=None) -> None:
        """Drive the plunger toward target_travel (m) with a gentle capped push/pull,
        holding the barrel fixtured the whole time (v0 = stand/hand fixture per the
        brief; without it the free-floating syringe just falls over — first smoke run).
        The plunger/ring are freed here (grab() pins them during carries), and the
        WHOLE rig switches from kinematic pinning to the scene's compliant fixture:
        v16 showed per-step teleports + a driven plunger = violent solver limit-cycle
        (barrel thrash, 48 N phantom forces, no net metering)."""
        for name in ("plunger", "thumb_ring"):
            held.pop(name, None)
        pinned = dict(held)
        held.clear()  # stop teleport-pinning entirely while the fixture holds
        scene.fixture(True)
        f_cur = max(force, 0.9)  # trust the caller; escalation is the safety net
        x_win = float(scene.travel()[0])
        fine_brake = 0  # steps of drive-0 braking when first entering the fine band
        seat_hits = 0
        lat_max = dz_max_lo = dz_max_hi = 0.0
        for it in range(guard):
            x = float(scene.travel()[0])
            err = target_travel - x
            # seat-duty diagnostics (draw-loss forensics, cheap)
            tipv = scene.tip_pos()[0]
            rp = scene.reservoir.data.root_pos_w[0]
            lat = float((tipv[:2] - rp[:2]).norm())
            dz = float(tipv[2] - (rp[2] + scene.cfg.res_h / 2))
            seat_hits += int(bool(scene.seated_reservoir()[0]))
            lat_max = max(lat_max, lat)
            dz_max_lo = min(dz_max_lo, dz)
            dz_max_hi = max(dz_max_hi, dz)
            if abs(err) < 0.0010:
                break
            # Stall escalation, WINDOWED: v14/v15's per-step delta never fired because
            # the stuck plunger vibrates ~0.2 mm/step on whatever is blocking it — the
            # per-step motion looked alive while the 150-step net motion was zero.
            if it > 0 and it % 40 == 0:
                if abs(x - x_win) < 0.002:
                    f_cur = min(f_cur * 1.5, 4.0)
                    print(f"[dose-smoke]   meter escalate -> {f_cur:.2f}N "
                          f"(net {abs(x - x_win) * 1000:.2f}mm/40steps)", flush=True)
                x_win = x
            # two-level drive: full force far out; entering the 25 mm fine band,
            # BRAKE (drive 0, viscosity kills the coarse momentum in ~3 substeps —
            # v32 coasted 30+ mm through the band and overshot every dose), then a
            # CONTINUOUS barely-above-slip creep lands in tolerance.
            if abs(err) > 0.030:
                f_apply = f_cur
                fine_brake = 8
            elif fine_brake > 0:
                fine_brake -= 1
                f_apply = 0.0
            else:
                f_apply = min(f_cur, 0.78)
            # SEAT-GATED: never push liquid while the tip is off the vessel (v37:
            # well 2's dose leaked 0.067 to `spilled` through mid-meter seat dropouts)
            if seat_fn is not None and not bool(seat_fn()):
                f_apply = 0.0
            scene.plunger_drive[0] = f_apply * (1.0 if err > 0 else -1.0)
            hold()
            step(1)
            if it % 5 == 0:
                print(f"[dose-smoke]     ledger it={it}: x={x * 1000:+.1f} "
                      f"led={float(scene._x_led[0]) * 1000:+.1f} "
                      f"liq={float(scene._liquid[0]):.3f} "
                      f"seat={int(scene._res_recent[0])}", flush=True)
            if it % 150 == 0:
                v = scene.plunger.data.root_lin_vel_w[0]
                pz = float(scene.plunger.data.root_pos_w[0, 2])
                bz = float(scene.barrel.data.root_pos_w[0, 2])
                rz = float(scene.ring.data.root_pos_w[0, 2])
                axz = float(scene.barrel_axis()[0, 2])
                print(f"[dose-smoke]   meter t={it}: x={x * 1000:+.1f}mm drive="
                      f"{float(scene.plunger_drive[0]):+.2f}N applied="
                      f"{float(scene._dbg_force[0]):+.2f}N plunger_v={float(v[2]):+.3f} "
                      f"plunger_z={pz:.4f} barrel_z={bz:.4f} ring_z={rz:.4f} "
                      f"ax_z={axz:+.2f} engaged={bool(scene._hold_on[0])} "
                      f"hold_at={float(scene._hold_at[0]) * 1000:+.1f}mm", flush=True)
        scene.plunger_drive[0] = 0.0
        if seat_hits:
            print(f"[dose-smoke]   meter seat-duty: {seat_hits}/{it + 1} steps seated, "
                  f"lat_max={lat_max * 1000:.1f}mm dz=[{dz_max_lo * 1000:.1f}, "
                  f"{dz_max_hi * 1000:.1f}]mm", flush=True)
        for _ in range(20):
            step(1)
        scene.fixture(False)
        held.update(pinned)  # resume kinematic pinning for carries
        grab()  # re-pin the whole assembly (including the plunger at its new travel)

    def rig_report(tag: str) -> None:
        """Joint-integrity diagnostic: relative offsets of every syringe body from the
        barrel (fixed joints must hold these constant) + plunger travel."""
        bp = scene.barrel.data.root_pos_w[0]
        rel = {nm: [round(float(v), 4) for v in (b.data.root_pos_w[0] - bp)]
               for nm, b in bodies().items() if nm != "barrel"}
        print(f"[dose-smoke] RIG {tag}: barrel_z={float(bp[2]):.4f} "
              f"travel={float(scene.travel()[0]) * 1000:.1f}mm rel={rel}", flush=True)

    env.reset()
    if args.demo:
        step(60)
    else:
        rig_report("post-reset")
        trace = []
        for k in range(12):  # settle trace: is the sag a drift, an oscillation, or a jump?
            step(10)
            trace.append((round(float(scene.travel()[0]) * 1000, 1),
                          round(float(scene._dbg_force[0]), 2)))
        print(f"[dose-smoke] settle trace (travel mm, o-ring force N): {trace}", flush=True)
        rig_report("settled")
        print(f"[dose-smoke] home: travel={float(scene.travel()[0]) * 1000:.1f}mm "
              f"tip={scene.tip_pos()[0].tolist()}", flush=True)

    o = env.iscene.env_origins
    if not args.demo:
        # Joint probe with the barrel PINNED (v8 ran it free: the 3 N reaction yanked
        # the rig off the stand and the readings were toppling artifacts, not joint
        # behaviour). Free the plunger/ring after grab(): grab() pins the WHOLE
        # assembly for carries, which in v11 pinned the probe's own test subject
        # (travel froze at ~0 under full drive).
        grab()
        for _nm in ("plunger", "thumb_ring"):
            held.pop(_nm, None)
        _pinned = dict(held)
        held.clear()
        scene.fixture(True)  # compliant hold for the probe too (see meter() note)
        x0 = float(scene.travel()[0]) * 1000
        # Force ladder: v12 moved 53mm at 3.0 N yet the 1.2 N meter didn't move at all —
        # measure the actual breakaway force and the effective speed at each level.
        for F in (0.8, 1.2, 2.0, 3.0):
            xa = float(scene.travel()[0])
            scene.plunger_drive[0] = F
            for _ in range(120):
                hold()
                step(1)
            dx = (float(scene.travel()[0]) - xa) * 1000
            print(f"[dose-smoke]   ladder F={F}N: dx={dx:+.1f}mm in 1s "
                  f"(applied={float(scene._dbg_force[0]):+.2f}N)", flush=True)
        up = float(scene.travel()[0]) * 1000
        scene.plunger_drive[0] = -3.0
        for _ in range(180):
            hold()
            step(1)
            if float(scene.travel()[0]) < -0.02:
                break
        dn = float(scene.travel()[0]) * 1000
        scene.plunger_drive[0] = 0.0
        for _ in range(30):
            hold()
            step(1)
        release()
        step(20)
        rig_report("after-probe")
        scene.fixture(False)
        held.update(_pinned)
        print(f"[dose-smoke] probe: start={x0:.1f}mm +drive-> {up:.1f}mm -drive-> {dn:.1f}mm",
              flush=True)
        check("plunger-joint", up > x0 + 20.0 and dn < up - 10.0,
              f"travel up={up:.1f}mm down={dn:.1f}mm (prismatic joint must carry the drive)")
        for _ in range(40):  # let the rig ring down after the probe re-pin
            hold()
            step(1)
        check("boot-parked", bool(scene.parked()[0]),
              f"parked={bool(scene.parked()[0])} upright={float(scene.barrel_axis()[0, 2]):.2f} "
              f"vel={float(scene.barrel.data.root_lin_vel_w[0].norm()):.3f}")

        # 2. air check FIRST (empty syringe over the bench): pull+push far from any well
        bench_spot = o[:, :] + torch.tensor([0.30, -0.25, c.surface_z + 0.06], device=device)
        move_tip(bench_spot)
        meter(0.06, 1.0)
        meter(0.0, 1.0)
        check("air-noop", float(scene.liquid()[0]) < 0.01
              and float(scene.doses().sum()) < 0.01,
              f"liquid={float(scene.liquid()[0]):.3f} doses={scene.doses()[0].tolist()}")

    # 1. draw a full load from the reservoir
    res = scene.reservoir.data.root_pos_w.clone()
    res[:, 2] += c.res_h / 2 + 0.004
    move_tip(res)
    check("seated-res", bool(scene.seated_reservoir()[0]))
    for _t in ("_dbg_pos_seated", "_dbg_pos_unseated", "_dbg_neg"):
        getattr(scene, _t)[:] = 0.0  # DRAW-scoped forensics (cfg.debug_forensics)
    meter(c.stroke * 0.99, 1.0, guard=1500,
          seat_fn=lambda: scene.seated_reservoir()[0])
    liq = float(scene.liquid()[0])
    tip = scene.tip_pos()[0].tolist()
    print(f"[dose-smoke] draw-forensics: pos_seated={float(scene._dbg_pos_seated[0]) * 1000:.1f}mm "
          f"pos_unseated={float(scene._dbg_pos_unseated[0]) * 1000:.1f}mm "
          f"neg={float(scene._dbg_neg[0]) * 1000:.1f}mm "
          f"spilled={float(scene._spilled[0]):.3f}", flush=True)
    print(f"[dose-smoke] post-draw: travel={float(scene.travel()[0]) * 1000:.1f}mm "
          f"liquid={liq:.3f} seated_now={bool(scene.seated_reservoir()[0])} "
          f"tip={[round(v, 4) for v in tip]} "
          f"res={[round(float(v), 4) for v in scene.reservoir.data.root_pos_w[0]]}",
          flush=True)
    check("draw-full", bool(scene.drawn()[0]) and liq >= c.draw_min,
          f"liquid={liq:.3f} travel={float(scene.travel()[0]) / c.stroke:.3f}")

    # 3. dispense one third into each sample well
    third = c.stroke / 3.0
    for k in range(3):
        w = scene.wells[k].data.root_pos_w.clone()
        w[:, 2] += c.well_h / 2 + 0.004
        move_tip(w)
        for _retry in range(3):  # verify the seat before metering (v34: well 2's whole
            if bool(scene.seated_well()[0, k]):  # dose went to `spilled` off-seat)
                break
            print(f"[dose-smoke]   WARNING not seated at well {k} "
                  f"(tip={[round(float(v), 4) for v in scene.tip_pos()[0]]}) — re-gliding",
                  flush=True)
            move_tip(w)
        # ADAPTIVE dosing: aim each well at (remaining liquid)/(remaining wells),
        # UNDER-aimed by the measured ~0.02 approach overshoot; the LAST well simply
        # receives everything left (v39: two 0.35 doses left only 0.25 for well 2 —
        # per-well perfection starved the remainder).
        x_now = float(scene.travel()[0])
        liq_now = float(scene.liquid()[0])
        if k < 2:
            aim = max(liq_now / (3 - k) - 0.018, 0.0) * c.stroke
        else:
            aim = liq_now * c.stroke + 0.010  # past-empty: push out every drop
        target = x_now - aim + 0.0005
        meter(max(target, 0.0005), 1.0, guard=1500,
              seat_fn=lambda k=k: scene.seated_well()[0, k])
        d = scene.doses()[0].tolist()
        print(f"[dose-smoke] after well {k}: doses={[f'{x:.3f}' for x in d]} "
              f"liquid={float(scene.liquid()[0]):.3f} "
              f"spilled={float(scene.spilled()[0]):.3f}", flush=True)
    doses = scene.doses()[0]
    lo, hi = c.dose_band
    check("doses-in-band", bool(scene.doses_ok()[0]),
          f"doses={[f'{float(x):.3f}' for x in doses]} band=[{lo},{hi}]")
    spread = float(doses.max() - doses.min())
    check("repeatability", spread < 0.05, f"dose spread {spread:.3f} (metering noise floor)")

    # 4. park (move_tip rises above the posts, then descends into the pocket)
    sx, sy = c.stand_pos
    park_z = c.surface_z + c.barrel_home_h - c.barrel_l / 2 - c.nozzle_l + 0.002
    park = o[:, :] + torch.tensor([sx, sy, park_z], device=device)
    move_tip(park)
    release()
    step(80)
    check("parked", bool(scene.parked()[0]))
    check("success", bool(scene.success()[0]),
          f"drawn={bool(scene.drawn()[0])} doses_ok={bool(scene.doses_ok()[0])} "
          f"parked={bool(scene.parked()[0])}")

    if args.demo:  # deliverable video ends with the syringe parked after 3 clean doses
        step(30)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="tool_use.syringe")
            print(f"[dose-smoke] saved {arr.shape} -> {args.out}", flush=True)
            os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                      f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print("DOSE_SMOKE_DONE", flush=True)
        env.close()
        return

    # 6. negative control: over-dose well 1 with half the load -> success impossible
    env.reset()
    step(40)
    res = scene.reservoir.data.root_pos_w.clone()
    res[:, 2] += c.res_h / 2 + 0.004
    move_tip(res)
    meter(c.stroke * 0.99, 1.0, guard=1500)
    w = scene.wells[0].data.root_pos_w.clone()
    w[:, 2] += c.well_h / 2 + 0.004
    move_tip(w)
    meter(c.stroke * 0.49, 1.0, guard=1500)  # dump ~half into well 0
    for k in (1, 2):
        w = scene.wells[k].data.root_pos_w.clone()
        w[:, 2] += c.well_h / 2 + 0.004
        move_tip(w)
        meter(max(c.stroke * (0.49 - (k) * 0.33), 0.0), 1.0, guard=1500)
    d = [float(x) for x in scene.doses()[0]]
    check("negative-control", not bool(scene.doses_ok()[0]) and d[0] > 0.45,
          f"doses={[f'{x:.3f}' for x in d]} (well0 overdosed, 2/3 starved)")

    print(f"[dose-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="tool_use.syringe")
        print(f"[dose-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[dose-smoke] hdfs upload rc={rc}", flush=True)
    print("DOSE_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()
