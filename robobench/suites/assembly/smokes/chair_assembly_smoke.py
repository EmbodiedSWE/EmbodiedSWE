"""Smoke / oracle test for ChairAssemblyScene — NullRobot, teleport + drop staging, RECORDED.

One linear run (pen_holder/stacking smoke skeleton):
  1. show      — settle the reset layout (seat underside-up, five parts scattered on
                 permuted arc slots) so you can see it; score must read 0;
  2. oracle    — the source-order assembly (legs -> back -> nuts, no source oracle exists
                 for the chair so this run IS our authored reference): drop each leg into
                 a socket (a GENUINE drop, cone-guided), lower the backrest with a
                 gravity-compensated kinematic hold until both flange holes ride both
                 studs and release, then drop each nut over a protruding stud tip; score
                 climbing 20 -> 40 -> 60 -> 80 -> 100 with every transition checked,
                 success() at the end, zero ordering violations;
  3. repeat    — 3x re-reset (random seat yaw each time) + teleport-assemble the whole
                 chair directly and require success(): the feasibility spike's
                 "reference chair" check, proving the rel-pose predicates fire in the
                 SEAT frame whatever the seat's episode pose;
  4. negative A — ORDERING, the port's centerpiece: a nut dropped on a bare stud rides it
                 (violation counter fires) but never counts (`should_assembled_first`
                 gate), and then PHYSICALLY BLOCKS the backrest from seating (the flange
                 hole cannot pass the 60 mm ring); removing the nut recovers the episode;
  5. negative B — a leg standing NEXT to a socket, and one lying ACROSS the socket mouth,
                 must not count (xy / z+ori terms);
  6. negative C — the backrest with only ONE hole over a stud (yawed) must not count
                 (the two-point insertion term);
  7. shake     — lift the fully assembled chair by the seat (gravity-compensated hold)
                 and shake it: all 5 pairs must hold (the welded chair is rigid) and the
                 rubric must keep judging the MOVING chair correctly (parent-frame port);
  8. sweep     — calibration: (a) nut drop with growing xy offset, (b) backrest release
                 with growing xy offset; 3 seeds each, raw (no retry). Publishes
                 per-offset rates + the capture limit; raw drop outcomes are stochastic —
                 asserted statistically, never per-offset (the pen-holder lesson).
  --demo runs ONLY show + oracle and saves the deliverable video.

ALWAYS records video via the viewport rgb annotator (same recipe as crate_packing_smoke:
RTX driver-version override, 3-render ghost flush, npz -> HDFS). Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.assembly.smokes.chair_assembly_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no repeat phase, no "
                         "negative controls, no sweep) — the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="chair_assembly_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (same as the render server): kit mis-decodes the L20 driver version and
# silently rejects RTX -> annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import shutil

import numpy as np
import torch

import robobench
from robobench.core import ENVS
from robobench.suites.assembly.scenes import ChairAssemblySceneCfg


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("assembly.chair")().build(
        num_envs=args.num_envs, device=device, scene_cfg=ChairAssemblySceneCfg())
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    bodies = {"seat": scene.seat, "leg_0": scene.legs[0], "leg_1": scene.legs[1],
              "back": scene.back, "nut_0": scene.nuts[0], "nut_1": scene.nuts[1]}
    PAIR_OF = {"leg_0": 0, "leg_1": 1, "back": 2, "nut_0": 3, "nut_1": 4}

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.1, -1.1, 0.9)) + o),
                                tuple(np.array((0.0, 0.0, c.surface_z + 0.12)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe "
                  "(driver-version override, NVIDIA_DRIVER_CAPABILITIES)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    # While a body's name is in `holds`, it is "held": its root state is rewritten before
    # every physics step — a kinematic hold through the scene handle, no robot. The
    # pre-step write carries +g*dt of UPWARD velocity so PhysX's gravity integration
    # cancels to exactly zero (the pen-holder lesson: a naive zero-velocity re-pin leaves the
    # body free-falling g*dt every step and everything riding it fails the settle gate).
    holds: dict[str, torch.Tensor] = {}
    g_dt = 9.81 * env.dt  # one-step gravity velocity, the hold compensation term

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i
        for _ in range(k):
            for name, st in holds.items():
                pre = st.clone()
                pre[:, 9] += g_dt  # cancel the gravity kick -> truly static hold
                bodies[name].write_root_state_to_sim(pre, all_ids)
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                data = annot.get_data()
                arr = np.asarray(data)
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}", flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1
        if holds:
            # Re-pin at ZERO velocity before judging (the compensated pre-step write would
            # otherwise leave +g*dt in the held bodies' judged buffers).
            for name, st in holds.items():
                bodies[name].write_root_state_to_sim(st, all_ids)
            env.iscene.update(0.0)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | assembled={scene.assembled()[0].int().tolist()} "
              f"score={int(scene.score()[0])} "
              f"viol={int(scene.order_violations[0])} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def diagnose(tag: str) -> None:
        """Per-pair rubric-term breakdown — printed on a failed check so the log names the
        guilty term instead of leaving it to the video."""
        legs, back, nuts = scene._legs_seated()[0], scene._back_seated()[0], scene._nuts_seated()[0]
        lat = scene._nut_stud_lat()[0]
        loc = scene._nut_rel_seat()[0]
        print(f"[smoke]   {tag}: legs_seated={legs.int().tolist()} back_seated={bool(back)} "
              f"nuts_geo={nuts.int().tolist()} "
              f"nut_lat={[f'{v * 1000:.0f}mm' for v in lat.tolist()]} "
              f"nut_z={[f'{(v - c.nut_seat_z) * 1000:+.0f}mm' for v in loc[:, 2].tolist()]} "
              f"welded={scene.welded[0].int().tolist()}", flush=True)

    # --- staging helpers ------------------------------------------------------------------
    def seat_pose() -> tuple[torch.Tensor, torch.Tensor]:
        """Seat pos (3,) LOCAL to the env origin + quat (4,), env 0, live."""
        return (scene.seat.data.root_pos_w[0] - env.iscene.env_origins[0],
                scene.seat.data.root_quat_w[0])

    def seat_to_local(local) -> torch.Tensor:
        """A seat-frame point -> env-origin-local world coords, (3,)."""
        sp, sq = seat_pose()
        v = torch.tensor(local, device=device, dtype=torch.float32)
        return sp + quat_apply(sq.unsqueeze(0), v.unsqueeze(0))[0]

    def make_state(pos, quat) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.as_tensor(pos, device=device)
        st[:, 3:7] = torch.as_tensor(quat, device=device)
        return st

    def teleport(name: str, pos, quat) -> None:
        bodies[name].write_root_state_to_sim(make_state(pos, quat), all_ids)

    def settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
        """Step in `poll`-sized chunks until `pred()` is true or the budget runs out
        (settling time is physics, not what the checks are about — the pen-holder lesson)."""
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    drops = 0  # drop retries — a throughput metric

    def drop_leg(name: str, slot: int, xy_off=(0.0, 0.0), retry: bool = True,
                 settle: int = 60) -> bool:
        """Release `name` above socket `slot` (seat frame), cone tip down, let it fall in,
        wait for its pair to weld."""
        nonlocal drops
        k = PAIR_OF[name]
        _sp, sq = seat_pose()
        lx, ly = c.leg_slots[slot]
        for attempt in range(2 if retry else 1):
            z = c.slab_top + c.socket_h + 0.015 + c.leg_l / 2
            pos = seat_to_local((lx + xy_off[0], ly + xy_off[1], z))
            teleport(name, pos, sq)
            step(settle)
            if settle_until(lambda: bool(scene.welded[0, k])):
                return True
            if attempt == 0 and retry:
                drops += 1
                print(f"[smoke]   drop retry: {name} missed socket {slot}", flush=True)
                xy_off = (0.002, -0.002)
        return False

    def drop_nut(name: str, slot: int, xy_off=(0.0, 0.0), retry: bool = True,
                 settle: int = 60, want_weld: bool = True) -> bool:
        """Release `name` above stud `slot`'s tip cone, ring flat, let it fall over the
        stud; wait for its pair to weld (or, with want_weld=False, just settle)."""
        nonlocal drops
        k = PAIR_OF[name]
        _sp, sq = seat_pose()
        bx, by = c.stud_slots[slot]
        for attempt in range(2 if retry else 1):
            # release just 6 mm above the tip cone: less fall energy = less bounce/cocking
            # when the ring meets the flank (GPU sweep A run-1 lesson)
            z = c.slab_top + c.stud_l + 0.006 + c.nut_t / 2
            pos = seat_to_local((bx + xy_off[0], by + xy_off[1], z))
            teleport(name, pos, sq)
            step(settle)
            if not want_weld:
                return True
            if settle_until(lambda: bool(scene.welded[0, k])):
                return True
            if attempt == 0 and retry:
                drops += 1
                print(f"[smoke]   drop retry: {name} missed stud {slot}", flush=True)
                xy_off = (0.002, -0.002)
        return False

    def lower_back(xy_off=(0.0, 0.0), release_h: float = 0.006, hold_steps: int = 240,
                   settle: int = 80) -> None:
        """The two-point insertion: hold the backrest upright with both holes over both
        studs, descend from above the stud tips to `release_h` above the seated height
        (gravity-compensated kinematic hold — the pen-holder platform lesson), then release."""
        _sp, sq = seat_pose()
        sy = c.stud_slots[0][1]  # stud line y (both studs share it)
        z_hi = c.back_seat_z + c.stud_l + 0.020
        z_lo = c.back_seat_z + release_h
        for t in range(hold_steps):
            f = (t + 1) / hold_steps
            z = z_hi + (z_lo - z_hi) * f
            pos = seat_to_local((xy_off[0], sy + xy_off[1], z))
            holds["back"] = make_state(pos, sq)
            step(1)
        holds.pop("back")
        step(settle)

    def teleport_assembled() -> None:
        """Snap all five children straight to their assembled poses (the feasibility
        spike's reference chair); the weld reconcile picks them up on the next steps."""
        _sp, sq = seat_pose()
        for j, name in enumerate(("leg_0", "leg_1")):
            lx, ly = c.leg_slots[j]
            teleport(name, seat_to_local((lx, ly, c.slab_top + c.leg_l / 2)), sq)
        teleport("back", seat_to_local((0.0, c.stud_slots[0][1], c.back_seat_z)), sq)
        for j, name in enumerate(("nut_0", "nut_1")):
            bx, by = c.stud_slots[j]
            teleport(name, seat_to_local((bx, by, c.nut_seat_z)), sq)

    # =========================== 1. show ====================================================
    env.reset()
    report("reset")
    step(60)
    report("show")
    # Predicates must read a clean slate — if anything scores at reset, the rubric is broken
    # and everything downstream is meaningless. Fail loudly.
    assert int(scene.score()[0]) == 0, \
        f"score={int(scene.score()[0])} at reset — rubric predicates broken"

    # =========================== 2. oracle: legs -> back -> nuts ============================
    expect = [20, 40, 60, 80, 100]
    ok = drop_leg("leg_0", 0)
    check("rubric transition 20 after leg_0", ok and int(scene.score()[0]) == expect[0])
    report("leg_0")
    ok = drop_leg("leg_1", 1)
    check("rubric transition 40 after leg_1", ok and int(scene.score()[0]) == expect[1])
    report("leg_1")
    lower_back()
    ok = settle_until(lambda: bool(scene.welded[0, 2]))
    check("rubric transition 60 after backrest (two-point insertion)",
          ok and int(scene.score()[0]) == expect[2])
    report("back")
    if not ok:
        diagnose("after-back")
    ok = drop_nut("nut_0", 0)
    check("rubric transition 80 after nut_0", ok and int(scene.score()[0]) == expect[3])
    report("nut_0")
    ok = drop_nut("nut_1", 1)
    check("rubric transition 100 after nut_1", ok and int(scene.score()[0]) == expect[4])
    report("nut_1")
    if int(scene.score()[0]) != 100:
        diagnose("oracle-end")
    ok_success = bool(scene.success()[0])
    check("oracle solve reaches success()", ok_success)
    check("oracle run has zero ordering violations", int(scene.order_violations[0]) == 0)
    print(f"[smoke] RESULT: {'ASSEMBLED — SUCCESS' if ok_success else 'NOT ASSEMBLED — FAIL'} "
          f"(drop retries={drops})", flush=True)

    if args.demo:  # deliverable video = the one clean run above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="assembly.chair")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
                args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                          f"hdfs dfs -put -f {args.out} "
                          f"{args.hdfs_dir}/{os.path.basename(args.out)}")
        print("CHAIR_ASSEMBLY_SMOKE_DONE", flush=True)
        env.close()
        return

    # =========================== 3. repeat: teleport-assembled reference ====================
    # The spike's reference-chair check under the reset randomization: whatever yaw/jitter
    # the seat drew, snapping the parts to the seat-frame candidates must fire all five
    # predicates and weld the chair.
    for rep_i in range(3):
        torch.manual_seed(100 + rep_i)
        env.reset()
        step(20)
        teleport_assembled()
        ok = settle_until(lambda: bool(scene.success()[0]), max_steps=150)
        check(f"repeat {rep_i}: teleport-assembled reference reaches success", ok)
        if not ok:
            diagnose(f"repeat-{rep_i}")
    report("repeat")

    # =========================== 4. negative A: ordering (nut first) ========================
    # Drop a nut on a BARE stud: it rides the stud (violation) but never counts (gate), and
    # then physically blocks the backrest — the flange hole cannot pass the 60 mm ring.
    env.reset()
    step(40)
    drop_nut("nut_0", 0, want_weld=False, retry=False)
    settle_until(lambda: bool(scene._nut_on_stud()[0, 0]), max_steps=120)
    check("nut-first: nut rides the bare stud", bool(scene._nut_on_stud()[0, 0]))
    check("nut-first: ordering gate keeps the pair uncounted",
          not bool(scene.pair_seated()[0, 3]) and not bool(scene.welded[0, 3]))
    check("nut-first: ordering violation counted", int(scene.order_violations[0]) >= 1)
    # release ABOVE the blocking nut (flange bottom ~20 mm > nut top ~12 mm above the
    # slab): the kinematic hold must not plow through the free nut, the DROP shows the block
    lower_back(release_h=0.020, settle=100)
    blocked = not bool(scene.welded[0, 2]) and not bool(scene._back_seated()[0])
    check("nut-first: backrest physically blocked from seating", blocked)
    if not blocked:
        diagnose("nut-first")
    report("nut-first")
    # recovery: lift the nut away, retry the backrest — the episode is not dead
    teleport("nut_0", (0.45, 0.45, c.surface_z + c.nut_t / 2 + 0.003), (1.0, 0.0, 0.0, 0.0))
    step(30)
    lower_back()
    check("nut-first: removing the nut recovers the backrest stage",
          settle_until(lambda: bool(scene.welded[0, 2])))
    report("recovered")

    # =========================== 5. negative B: leg misplacements ===========================
    env.reset()
    step(40)
    _sp, sq = seat_pose()
    lx, ly = c.leg_slots[0]
    # standing NEXT to the socket (5 cm off): upright and at slab height, but xy fails
    pos = seat_to_local((lx + 0.05, ly + 0.05, c.slab_top + c.leg_l / 2))
    teleport("leg_0", pos, sq)
    env.iscene.update(0.0)  # judge the authored pose (no physics step)
    check("beside-socket: standing leg off the slot never counts",
          not bool(scene._legs_seated()[0, 0]))
    # lying ACROSS the socket mouth: on-axis in xy but shallow and sideways
    c45 = math.cos(math.pi / 4)
    q_across = torch.tensor([c45, 0.0, c45, 0.0], device=device)  # 90 deg about y
    sq4 = sq.tolist()
    w, x, y, z = sq4
    qa = q_across.tolist()
    q_mix = (w * qa[0] - x * qa[1] - y * qa[2] - z * qa[3],
             w * qa[1] + x * qa[0] + y * qa[3] - z * qa[2],
             w * qa[2] - x * qa[3] + y * qa[0] + z * qa[1],
             w * qa[3] + x * qa[2] - y * qa[1] + z * qa[0])
    pos = seat_to_local((lx, ly, c.slab_top + c.socket_h + c.leg_r))
    teleport("leg_0", pos, q_mix)
    env.iscene.update(0.0)
    check("across-socket: lying leg on the mouth never counts",
          not bool(scene._legs_seated()[0, 0]))
    step(60)
    report("bad-leg")

    # =========================== 6. negative C: one-post backrest ===========================
    # Yaw the backrest ~20 deg about the FIRST stud: hole 0 stays over its stud, hole 1
    # swings ~55 mm away — the two-point term must reject it.
    env.reset()
    step(40)
    _sp, sq = seat_pose()
    yaw = math.radians(20.0)
    cy, sy_ = math.cos(yaw / 2), math.sin(yaw / 2)
    w, x, y, z = sq.tolist()
    q_yawed = (cy * w - sy_ * z, cy * x - sy_ * y, cy * y + sy_ * x, cy * z + sy_ * w)
    # place hole 0 exactly on stud 0; the back origin swings around it by the yaw
    bx0, by0 = c.stud_slots[0]
    dx, dy = c.hole_sx * math.cos(yaw), c.hole_sx * math.sin(yaw)
    pos = seat_to_local((bx0 + dx, by0 + dy, c.back_seat_z))
    teleport("back", pos, q_yawed)
    env.iscene.update(0.0)
    check("one-hole backrest (yawed): two-point term rejects",
          not bool(scene._back_seated()[0]))
    step(60)
    report("one-hole")

    # =========================== 7. shake: the welded chair is rigid ========================
    env.reset()
    step(20)
    teleport_assembled()
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=150)
    check("shake setup: assembled chair welded", ok)
    seat0 = scene.seat.data.root_state_w[0:1].clone()
    hold = seat0.clone().expand(n, 13).clone()
    hold[:, 7:] = 0.0
    z0, x0 = float(hold[0, 2]), float(hold[0, 0])
    K_up, K_sh = 120, 240
    for t in range(K_up):  # lift 0.15 m
        hold[:, 2] = z0 + 0.15 * (t + 1) / K_up
        holds["seat"] = hold.clone()
        step(1)
    for t in range(K_sh):  # shake: +/- 3 cm x sinusoid, ~2 Hz at 120 Hz
        hold[:, 0] = x0 + 0.03 * math.sin(2 * math.pi * t / 60)
        hold[:, 2] = z0 + 0.15
        holds["seat"] = hold.clone()
        step(1)
    # the sticky weld flags are monotonic by construction — the rigidity proof is the LIVE
    # geometry: every pair must still sit at its seat-frame candidate while held mid-air
    check("shake: all 5 pairs LIVE-seated mid-air (rigid chair, moving-frame rubric)",
          bool(scene.pair_seated()[0].all()) and bool(scene.success()[0]))
    for t in range(K_up):  # set back down
        hold[:, 0] = x0
        hold[:, 2] = z0 + 0.15 * (1 - (t + 1) / K_up)
        holds["seat"] = hold.clone()
        step(1)
    holds.pop("seat")
    step(40)
    check("shake: still success after set-down", bool(scene.success()[0]))
    report("shake")

    # =========================== 8. calibration sweeps ======================================
    # Honest tolerance numbers, both drop-release, raw (no retry), 3 seeds per offset. Raw
    # outcomes are stochastic — per-offset rates are published, assertions statistical.
    # (a) nut over stud: pure-clearance funnel = nut_r_in - stud_r (5 mm); the tall tip
    # cone extends capture beyond it (published, only the funnel window is asserted).
    print("[smoke] SWEEP A (nut drop offset -> capture rate, 3 seeds each)", flush=True)
    nut_res: dict[float, int] = {}
    for off_mm in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0):
        hits = 0
        for seed in range(3):
            torch.manual_seed(seed)
            env.reset()
            step(15)
            teleport_assembled()  # nut_1 included; we re-drop nut_0 from scratch
            _sp, sq = seat_pose()
            teleport("nut_0", (0.45, -0.45, c.surface_z + c.nut_t / 2 + 0.003),
                     (1.0, 0.0, 0.0, 0.0))
            settle_until(lambda: bool(scene.welded[0, 2]), max_steps=120)
            ang = 2 * math.pi * (seed / 3.0)
            off = (off_mm / 1000.0 * math.cos(ang), off_mm / 1000.0 * math.sin(ang))
            hit = drop_nut("nut_0", 0, xy_off=off, retry=False)
            hits += int(hit)
            if not hit:  # name where the nut ended, so a failing band explains itself
                lat = float(scene._nut_stud_lat()[0, 0]) * 1000
                z_err = (float(scene._nut_rel_seat()[0, 0, 2]) - c.nut_seat_z) * 1000
                print(f"[smoke]   nut off={off_mm:.0f}mm seed={seed}: seated=False "
                      f"(ends lat={lat:.0f}mm z_err={z_err:+.0f}mm)", flush=True)
            else:
                print(f"[smoke]   nut off={off_mm:.0f}mm seed={seed}: seated=True", flush=True)
        nut_res[off_mm] = hits
    # (b) backrest release offset: hole clearance = 5 mm + the stud tip cones.
    print("[smoke] SWEEP B (backrest release offset -> seat rate, 3 seeds each)", flush=True)
    back_res: dict[float, int] = {}
    for off_mm in (0.0, 3.0, 6.0, 9.0):
        hits = 0
        for seed in range(3):
            torch.manual_seed(10 + seed)
            env.reset()
            step(15)
            ang = 2 * math.pi * (seed / 3.0)
            off = (off_mm / 1000.0 * math.cos(ang), off_mm / 1000.0 * math.sin(ang))
            lower_back(xy_off=off, settle=60)
            hit = settle_until(lambda: bool(scene.welded[0, 2]), max_steps=120)
            hits += int(hit)
            print(f"[smoke]   back off={off_mm:.0f}mm seed={seed}: seated={hit}", flush=True)
        back_res[off_mm] = hits

    def summarize(tag: str, res: dict[float, int], funnel_mm: float) -> float:
        limit = max((k for k, v in res.items() if v == 3), default=0.0)
        in_funnel = {k: v for k, v in res.items() if k <= funnel_mm}
        rate = sum(in_funnel.values()) / (3 * len(in_funnel))
        print(f"[smoke] SWEEP {tag}: " +
              " | ".join(f"{k:.0f}mm: {v}/3" for k, v in res.items()) +
              f"  -> capture limit (last 3/3) = {limit:.0f}mm, "
              f"within-funnel (<= {funnel_mm:.0f}mm) rate = {rate:.0%}", flush=True)
        return rate

    # Funnel windows are the PURE-CLEARANCE bands rounded to the tested grid (nut: 5 mm
    # clearance -> assert <= 6 mm; the cone-assisted tail beyond is published only).
    rate_a = summarize("A/nut", nut_res, 6.0)
    rate_b = summarize("B/back", back_res, 6.0)  # 5 mm clearance + cones, x2 coupling
    check("sweep A: within-funnel nut capture rate >= 60%", rate_a >= 0.60)
    check("sweep A: some nut offset is fully reliable (3/3)",
          any(v == 3 for v in nut_res.values()))
    check("sweep B: within-funnel backrest seat rate >= 60%", rate_b >= 0.60)
    check("sweep B: some backrest offset is fully reliable (3/3)",
          any(v == 3 for v in back_res.values()))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="assembly.chair")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        if shutil.which("hdfs"):  # optional archive channel; absent on RunPod
            rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                           f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/"
                           f"{os.path.basename(args.out)}")
            print(f"[smoke] hdfs upload rc={rc} -> "
                  f"{args.hdfs_dir}/{os.path.basename(args.out)}", flush=True)
    all_ok = all(okc for _name, okc in checks)
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"({sum(okc for _n, okc in checks)}/{len(checks)} checks, drops={drops})", flush=True)
    print("CHAIR_ASSEMBLY_SMOKE_DONE", flush=True)
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
