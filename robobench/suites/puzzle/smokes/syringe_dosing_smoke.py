"""Smoke / oracle test for SyringeDosingScene — NullRobot, RECORDED.

The full triple-dose pipeline, NullRobot style (the barrel is CARRIED by teleport-glide
since there are no hands; the plunger is metered through `scene.plunger_drive`, the same
interface a robot thumb or an RL policy acts through):
  0. erect — the syringe starts LYING on the cart shelf (no stand): kinematically
     rotate the pinned assembly upright about the barrel centre;
  1. draw — glide the syringe over the reservoir beaker, seat the tip, pull the
     plunger to a full draw; verify liquid == travel while seated;
  2. air check — pull/push over an empty shelf spot; verify NOTHING moves (no liquid
     gained, none dosed);
  3. dispense x3 — seat on each glass tube's mouth, push exactly stroke/3 worth,
     verify each dose lands in the scene's dose band (33% +/- 10%);
  4. park — glide back over home, LAY the syringe back down flat; verify `parked()`
     and full `success()`;
  5. dose-repeatability calibration — the dispense error across the 3 tubes (the
     metering noise floor must be well inside the +/-5% band);
  6. negative control — a fresh episode where tube #1 gets HALF the load: tubes 2/3
     then cannot reach the band and success() must stay False (irreversibility is real).

Barrel carrying uses kinematic-style root-velocity writes (write_root_state 1cm-glide
per step) — gentle enough for the prismatic joint to follow. Video recorded throughout.

    python -m robobench.suites.puzzle.smokes.syringe_dosing_smoke --headless
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
    from robobench.suites.puzzle.scenes import SyringeDosingSceneCfg

    env = ENVS.get("puzzle.syringe")().build(
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
        # wide 3/4 view of the cart's west half: syringe home, rack tubes, beaker,
        # with the whole cart in frame (the work surface is the 0.75 m top shelf)
        env.sim.set_camera_view(tuple(np.array((-1.15, -0.95, 1.60)) + o),
                                tuple(np.array((-0.10, 0.02, 0.82)) + o),
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

    # --- carrying: glide the syringe ARTICULATION by its root (the barrel link).
    # A root write moves the whole articulation rigidly — the plunger rides its
    # prismatic joint, so the old two-body canonical-formation bookkeeping (and
    # its ratchet pathologies) is gone.
    held: list[torch.Tensor] = []  # pinned root pose (the "hand" holding it)

    def grab() -> None:
        held.clear()
        held.append(scene.syringe.data.root_state_w.clone())

    def hold() -> None:
        if held:
            pin = held[0].clone()
            pin[:, 7:13] = 0.0
            scene.syringe.write_root_state_to_sim(pin, ids)

    def release() -> None:
        held.clear()

    def glide_to(tip_target: torch.Tensor, speed: float = 0.010, settle: int = 20) -> None:
        """Move the syringe so the nozzle TIP lands on tip_target: rigid
        teleport-glide of the articulation root from a pose captured at glide
        start, then HOLD."""
        ref = scene.syringe.data.root_state_w.clone()
        off = torch.zeros(1, 3, device=device)
        for g_it in range(400):
            tip = scene.tip_pos()
            d = tip_target - tip
            dist = float(d.norm(dim=-1)[0])
            if dist < 0.001:
                break
            off = off + d * min(1.0, speed / max(dist, 1e-9))
            pin = ref.clone()
            pin[:, 0:3] += off
            pin[:, 7:13] = 0.0
            scene.syringe.write_root_state_to_sim(pin, ids)
            step(1)
            if g_it % 40 == 0:
                bp = scene.syringe.data.root_pos_w[0]
                print(f"[dose-smoke]     glide {g_it}: dist={dist * 1000:.1f}mm "
                      f"barrel=({float(bp[0]):+.3f},{float(bp[1]):+.3f},{float(bp[2]):+.3f}) "
                      f"travel={float(scene.travel()[0]) * 1000:+.1f}mm", flush=True)
        else:
            print(f"[dose-smoke]   WARNING glide_to NOT CONVERGED: residual "
                  f"{float((tip_target - scene.tip_pos()).norm(dim=-1)[0]) * 1000:.1f}mm "
                  f"(obstructed corridor?)", flush=True)
            # BACK OFF instead of leaving the rig pressed into the obstruction
            up_off = torch.tensor([[0.0, 0.0, 0.10]], device=device)
            ref2 = scene.syringe.data.root_state_w.clone()
            for s in range(20):
                pin = ref2.clone()
                pin[:, 0:3] += up_off * ((s + 1) / 20.0)
                pin[:, 7:13] = 0.0
                scene.syringe.write_root_state_to_sim(pin, ids)
                step(1)
        grab()
        for _ in range(settle):
            hold()
            step(1)

    def move_tip(target: torch.Tensor, safe_h: float = 0.32) -> None:
        """Up-over-down tip move (never drags the nozzle laterally through labware).
        safe_h is TIP height: 0.32 clears the rack's tube rims (0.27 above the shelf)
        by 5 cm during carries."""
        cur = scene.tip_pos().clone()
        up = cur.clone()
        up[:, 2] = c.surface_z + safe_h
        glide_to(up, settle=2)
        over = target.clone()
        over[:, 2] = c.surface_z + safe_h
        glide_to(over, settle=2)
        glide_to(target)

    def _set_tilt(s: float, center: torch.Tensor) -> None:
        """Write the articulation root at tilt fraction s (0 = lying along x,
        tip west, 1 = upright), the barrel centre at `center` (the plunger
        rides the joint). The boot lie is x-aligned (bimanual layout,
        2026-08-11), so the tilt is a rotation about world -y applied on it."""
        import math as _m

        from isaaclab.utils.math import quat_mul as _qm

        lie = torch.tensor([[0.5, -0.5, 0.5, -0.5]], device=device)
        ang = s * (_m.pi / 2)
        qy = torch.tensor([[_m.cos(ang / 2), 0.0, -_m.sin(ang / 2), 0.0]],
                          device=device)
        q = _qm(qy, lie)[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = center
        st[:, 3:7] = q
        scene.syringe.write_root_state_to_sim(st, ids)

    def erect() -> None:
        """Raise the LYING syringe upright (kinematic, the 'hand' motion): LIFT
        well clear of the shelf FIRST, then rotate upright at altitude (rotating
        at shelf height sweeps the tip through the shelf — kinematic
        interpenetration, measured 2026-08-10)."""
        base = scene.syringe.data.root_pos_w[0].clone()
        clear = c.barrel_l / 2 + c.nozzle_l + 0.06  # tip clears the shelf when tilted
        for i_s in range(31):  # phase 1: straight lift, still lying
            s = i_s / 30.0
            center = (base + torch.tensor(
                [0.0, 0.0, s * clear], device=device)).view(1, 3)
            _set_tilt(0.0, center)
            step(1)
        top = (base + torch.tensor([0.0, 0.0, clear], device=device)).view(1, 3)
        for i_s in range(41):  # phase 2: rotate upright at altitude
            s = i_s / 40.0
            _set_tilt(s, top)
            step(1)
        grab()
        for _ in range(10):
            hold()
            step(1)

    def lay_down() -> None:
        """Reverse of erect(): lower the carried syringe back to lying at home, then
        release and settle."""
        hx, hy = c.home_pos
        # end the descent HOVERING 1.5 cm above the rest height and hold the final
        # pose until the kinematic velocity flushes to zero: releasing mid-descent
        # handed the barrel ~1.2 m/s of teleport momentum 2 mm inside the shelf and
        # it tunneled through shelf and floor to z=-0.49 (measured 2026-08-09)
        home = o[0] + torch.tensor(
            [hx, hy, c.surface_z + c.barrel_home_h + 0.015], device=device)
        start = scene.syringe.data.root_pos_w[0].clone()
        for i_s in range(41):
            s = i_s / 40.0
            center = (start + (home - start) * s).view(1, 3)
            _set_tilt(1.0 - s, center)
            step(1)
        for _ in range(12):  # hold still: kinematic velocity -> 0 before release
            _set_tilt(0.0, home.view(1, 3))
            step(1)
        release()
        for i_r in range(60):
            step(1)
            if i_r % 10 == 0:
                bp = scene.syringe.data.root_pos_w[0]
                bv = scene.syringe.data.root_lin_vel_w[0]
                print(f"[dose-smoke]     lay-settle t={i_r}: z={float(bp[2]):.4f} "
                      f"vz={float(bv[2]):+.3f} xy=({float(bp[0]):+.3f},{float(bp[1]):+.3f})",
                      flush=True)

    def meter(target_travel: float, force: float, guard: int = 900,
              seat_fn=None) -> None:
        """Drive the plunger toward target_travel (m) with a gentle capped push/pull,
        holding the barrel PINNED the whole time (v0 = stand/hand fixture per the
        brief; without it the free-floating syringe just falls over — first smoke run).
        The plunger/ring are freed here (grab() pins them during carries), and the
        WHOLE rig switches from kinematic pinning to the scene's compliant fixture:
        v16 showed per-step teleports + a driven plunger = violent solver limit-cycle
        (barrel thrash, 48 N phantom forces, no net metering)."""
        # Metering configuration, MEASURED on the vendored rig (bisect 2026-08-09):
        # barrel PINNED kinematically + plunger free + drive = rock-stable for the
        # whole probe; the compliant fixture destabilized even at ZERO drive (the
        # explicit-force PD is the oscillator here — the exact inverse of the old
        # procedural rig, where pinning limit-cycled and the fixture was the fix).
        # PHYSICAL-JOINT constants: slip = the joint's Coulomb friction (1.5 N —
        # the old 0.78 N creep sat BELOW slip and moved nothing, then the stall
        # escalation burst through in jerks: doses 0.105/0.221/0.445). Creep just
        # above slip; gravity feedforward keeps up/down symmetric when upright
        # (the 0.06 kg plunger hands every downward push a free 0.59 N).
        slip = c.plunger_friction
        f_creep = slip + 0.25
        g_ff = c.plunger_mass * 9.81
        f_cur = max(force, slip + 0.9)
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
                    f_cur = min(f_cur * 1.5, 5.0)
                    # the creep floor escalates too: the authored 1.5 N Coulomb
                    # reads back as ~1.8-2.4 N of REAL static friction in PhysX,
                    # and a fixed creep below that froze the fine band forever
                    # (well 0 dosed 0.100: coarse 10 mm, then 1400 stalled steps)
                    f_creep = min(f_creep + 0.3, 4.0)
                    print(f"[dose-smoke]   meter escalate -> {f_cur:.2f}N "
                          f"creep -> {f_creep:.2f}N "
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
                f_apply = min(f_cur, f_creep)
            # SEAT-GATED: never push liquid while the tip is off the vessel (v37:
            # well 2's dose leaked 0.067 to `spilled` through mid-meter seat dropouts)
            if seat_fn is not None and not bool(seat_fn()):
                f_apply = 0.0
            # gravity feedforward only while roughly upright (metering posture)
            comp = g_ff if float(scene.barrel_axis()[0, 2]) > 0.7 else 0.0
            scene.plunger_drive[0] = (f_apply * (1.0 if err > 0 else -1.0)
                                      + (comp if f_apply > 0.0 else 0.0))
            hold()
            step(1)
            if it < 60 or it % 5 == 0:  # first 60 per-step: explosion forensics
                pv = float(scene.syringe.data.joint_vel[0, 0])
                bz = float(scene.syringe.data.root_pos_w[0, 2])
                tip = scene.tip_pos()[0]
                rr = scene.reservoir.data.root_pos_w[0]
                lat = float((tip[:2] - rr[:2]).norm()) * 1000
                dzz = float(tip[2] - (rr[2] + c.res_h / 2)) * 1000
                upz = float(scene.barrel_axis()[0, 2])
                print(f"[dose-smoke]     ledger it={it}: x={x * 1000:+.1f} "
                      f"led={float(scene._x_led[0]) * 1000:+.1f} "
                      f"v_ax={pv:+.3f} bz={bz:.4f} "
                      f"f={float(scene._dbg_force[0]):+.2f} "
                      f"liq={float(scene._liquid[0]):.3f} "
                      f"seat={int(scene._res_recent[0])} "
                      f"lat={lat:.1f} dz={dzz:+.1f} upz={upz:+.2f}", flush=True)
            if it % 150 == 0:
                jv = float(scene.syringe.data.joint_vel[0, 0])
                bz = float(scene.syringe.data.root_pos_w[0, 2])
                axz = float(scene.barrel_axis()[0, 2])
                print(f"[dose-smoke]   meter t={it}: x={x * 1000:+.1f}mm drive="
                      f"{float(scene.plunger_drive[0]):+.2f}N joint_v={jv:+.3f} "
                      f"barrel_z={bz:.4f} ax_z={axz:+.2f}", flush=True)
        scene.plunger_drive[0] = 0.0
        if seat_hits:
            print(f"[dose-smoke]   meter seat-duty: {seat_hits}/{it + 1} steps seated, "
                  f"lat_max={lat_max * 1000:.1f}mm dz=[{dz_max_lo * 1000:.1f}, "
                  f"{dz_max_hi * 1000:.1f}]mm", flush=True)
        for _ in range(20):
            hold()  # barrel stays pinned while the plunger rings down
            step(1)
        grab()  # re-pin the whole assembly (including the plunger at its new travel)

    def rig_report(tag: str) -> None:
        """Joint-integrity diagnostic: root height + measured joint travel."""
        bp = scene.syringe.data.root_pos_w[0]
        print(f"[dose-smoke] RIG {tag}: barrel_z={float(bp[2]):.4f} "
              f"travel={float(scene.travel()[0]) * 1000:.1f}mm", flush=True)

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
        # behaviour). Pinning the ROOT leaves the prismatic joint free — the drive
        # moves the real plunger link against joint friction.
        grab()
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
        # return ALL the way to seat: residual probe travel starts the main draw
        # short (12 mm left over cost the draw-full check a full stroke, measured)
        scene.plunger_drive[0] = -4.0
        for _ in range(400):
            hold()
            step(1)
            if float(scene.travel()[0]) < 0.0005:
                break
        dn = float(scene.travel()[0]) * 1000
        scene.plunger_drive[0] = 0.0
        for _ in range(30):
            hold()
            step(1)
        release()
        step(20)
        rig_report("after-probe")
        print(f"[dose-smoke] probe: start={x0:.1f}mm +drive-> {up:.1f}mm -drive-> {dn:.1f}mm",
              flush=True)
        check("plunger-joint", up > x0 + 20.0 and dn < up - 10.0,
              f"travel up={up:.1f}mm down={dn:.1f}mm (prismatic joint must carry the drive)")
        for _ in range(40):  # let the rig ring down after the probe re-pin
            hold()
            step(1)
        check("boot-parked", bool(scene.parked()[0]),
              f"parked={bool(scene.parked()[0])} lying={float(scene.barrel_axis()[0, 2]):.2f} "
              f"vel={float(scene.syringe.data.root_lin_vel_w[0].norm()):.3f}")

        def drop_probe(tag: str, at: torch.Tensor | None = None) -> None:
            """Contact-liveness probe: kinematically lift the LYING barrel 3 cm
            (from its current spot, or from `at`), hold, release, and watch whether
            it lands (contact alive) or free-falls through the shelf (the
            park-phase failure signature)."""
            grab()
            base_p = (at.clone() if at is not None
                      else scene.syringe.data.root_pos_w[0].clone())
            for i_s in range(13):
                center = (base_p + torch.tensor(
                    [0.0, 0.0, 0.03 * min(i_s / 8.0, 1.0)], device=device)).view(1, 3)
                _set_tilt(0.0, center)
                step(1)
            release()
            for i_r in range(40):
                step(1)
                if i_r % 8 == 0:
                    bp = scene.syringe.data.root_pos_w[0]
                    print(f"[dose-smoke]     drop[{tag}] t={i_r}: z={float(bp[2]):.4f} "
                          f"vz={float(scene.syringe.data.root_lin_vel_w[0, 2]):+.3f}",
                          flush=True)

        drop_probe("post-boot")

        # 0. erect the lying syringe (kinematic 'hand' raise)
        erect()
        check("erected", float(scene.barrel_axis()[0, 2]) > 0.95,
              f"axis_z={float(scene.barrel_axis()[0, 2]):.3f}")

        # 2. air check FIRST (empty syringe over a bare shelf spot): pull+push off-well.
        # Spot is INTERIOR — (-0.31,-0.20) sat on the west guard rail and the pinned
        # glide ground into it until the prismatic joint state broke (rig explosion)
        shelf_spot = o[:, :] + torch.tensor([-0.27, -0.16, c.surface_z + 0.06],
                                            device=device)
        move_tip(shelf_spot)
        meter(0.06, 1.0)
        meter(0.0, 1.0)
        scene.plunger_drive[0] = -4.0  # full return: the 1 mm meter tolerance
        for _ in range(120):           # would start the main draw short
            hold()
            step(1)
            if float(scene.travel()[0]) < 0.0005:
                break
        scene.plunger_drive[0] = 0.0
        check("air-noop", float(scene.liquid()[0]) < 0.01
              and float(scene.doses().sum()) < 0.01,
              f"liquid={float(scene.liquid()[0]):.3f} doses={scene.doses()[0].tolist()}")

    # 1. draw a full load from the reservoir
    if args.demo:
        erect()
    res = scene.reservoir.data.root_pos_w.clone()
    res[:, 2] += c.res_h / 2 + 0.004
    move_tip(res)
    check("seated-res", bool(scene.seated_reservoir()[0]))
    for _t in ("_dbg_pos_seated", "_dbg_pos_unseated", "_dbg_neg"):
        getattr(scene, _t)[:] = 0.0  # DRAW-scoped forensics (cfg.debug_forensics)
    meter(c.stroke * 0.99, 1.0, guard=1500,
          seat_fn=lambda: scene.seated_reservoir()[0])
    scene.plunger_drive[0] = 2.0  # top up to the hard stop (travel clamps at stroke)
    for _ in range(80):
        hold()
        step(1)
    scene.plunger_drive[0] = 0.0
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

    # 3. dispense one third into each glass tube
    for k in range(3):
        w = scene.well_mouths()[:, k].clone()
        w[:, 2] += 0.004
        move_tip(w)
        for _retry in range(3):  # verify the seat before metering (v34: well 2's whole
            if bool(scene.seated_well()[0, k]):  # dose went to `spilled` off-seat)
                break
            print(f"[dose-smoke]   WARNING not seated at well {k} "
                  f"(tip={[round(float(v), 4) for v in scene.tip_pos()[0]]}) — re-gliding",
                  flush=True)
            move_tip(w)
        # ADAPTIVE dosing: aim each well at (remaining liquid)/(remaining wells);
        # the LAST well simply receives everything left (v39: two 0.35 doses left
        # only 0.25 for well 2 — per-well perfection starved the remainder). No
        # under-aim: the kinematic rate law stops SHORT (meter tolerance), never
        # overshoots — the old rig's 0.018 correction inverted into a deficit.
        x_now = float(scene.travel()[0])
        liq_now = float(scene.liquid()[0])
        if k < 2:
            aim = max(liq_now / (3 - k), 0.0) * c.stroke
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

    # 4. park: carry home high, then LAY the syringe back down flat on the shelf
    hx, hy = c.home_pos
    over_home = o[:, :] + torch.tensor([hx, hy, c.surface_z + 0.30], device=device)
    move_tip(over_home)
    lay_down()
    if not args.demo and not bool(scene.parked()[0]):
        # park failed: bracket WHEN barrel<->cart contact died (post-boot probe
        # is the alive baseline)
        drop_probe("post-park-fail", at=o[0] + torch.tensor(
            [hx, hy, c.surface_z + 0.038], device=device))
    check("parked", bool(scene.parked()[0]),
          f"axis_z={float(scene.barrel_axis()[0, 2]):.2f} "
          f"z={float(scene.syringe.data.root_pos_w[0, 2]):.3f}")
    check("success", bool(scene.success()[0]),
          f"drawn={bool(scene.drawn()[0])} doses_ok={bool(scene.doses_ok()[0])} "
          f"parked={bool(scene.parked()[0])}")

    if args.demo:  # deliverable video ends with the syringe parked after 3 clean doses
        step(30)
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="puzzle.syringe")
            print(f"[dose-smoke] saved {arr.shape} -> {args.out}", flush=True)
            os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                      f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print("DOSE_SMOKE_DONE", flush=True)
        return

    # 6. negative control: over-dose tube 1 with half the load -> success impossible
    env.reset()
    step(40)
    erect()
    res = scene.reservoir.data.root_pos_w.clone()
    res[:, 2] += c.res_h / 2 + 0.004
    move_tip(res)
    meter(c.stroke * 0.99, 1.0, guard=1500)
    w = scene.well_mouths()[:, 0].clone()
    w[:, 2] += 0.004
    move_tip(w)
    meter(c.stroke * 0.49, 1.0, guard=1500)  # dump ~half into tube 0
    for k in (1, 2):
        w = scene.well_mouths()[:, k].clone()
        w[:, 2] += 0.004
        move_tip(w)
        meter(max(c.stroke * (0.49 - (k) * 0.33), 0.0), 1.0, guard=1500)
    d = [float(x) for x in scene.doses()[0]]
    check("negative-control", not bool(scene.doses_ok()[0]) and d[0] > 0.45,
          f"doses={[f'{x:.3f}' for x in d]} (tube0 overdosed, 2/3 starved)")

    print(f"[dose-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="puzzle.syringe")
        print(f"[dose-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[dose-smoke] hdfs upload rc={rc}", flush=True)
    print("DOSE_SMOKE_DONE", flush=True)


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends.
    env.close() must NOT be called inside main(): it hangs before the watchdog
    exists and wedges the pod after all results are printed (pod-9, 2026-08-11)."""
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
