"""Smoke / oracle test for ChairAssemblyScene (real-asset backrest task) — NullRobot, RECORDED.

One linear run (pen_holder/stacking smoke skeleton), 3 pairs: (base,back), (base,nut_0/1):
  1. show      — settle the reset layout (chair base standing, backrest face-down + two nuts
                 on permuted arc slots); score must read 0;
  2. oracle    — the authored reference: lift the backrest with a gravity-compensated
                 kinematic hold, stand it upright behind the chair, slide BOTH holes onto
                 BOTH studs (the two-point insertion, riding the rear-leg rail), release;
                 then stage each nut on its exposed stud tip and PRESS + TWIST it on — real
                 threading on the HORIZONTAL axis (the run's central validation; nut_thread's
                 body-frame press/twist constants verbatim). Score 33 -> 66 -> 100 with every
                 transition checked, success() at the end, zero ordering violations;
  3. shake     — immediately on the oracle's REAL product: lift the assembled chair by the
                 base (gravity-compensated hold) and shake it: all 3 pairs must hold (the
                 welded chair is rigid) and the rubric must keep judging the MOVING chair
                 correctly (parent-frame port). (Teleport-staged assembly is impossible
                 under the screw mechanic: a nut only reaches depth by rotation.)
  4. repeat    — 2x re-reset (random base yaw each time) + teleport-seat the BACK and
                 require its weld: the rel-pose predicates fire in the BASE frame whatever
                 the base's episode pose (nut frame-independence is already exercised by
                 the oracle threading under the reset's random yaw);
  5. negative A — ORDERING: a nut threaded onto a bare stud rides it (violation counter
                 fires) but never counts (`should_assembled_first` gate); UNSCREWING it
                 (the mechanic is two-way) frees the stud and the episode recovers. The
                 predecessor's physical hole-block does not exist on the real asset — the
                 shell's rear face never sweeps the thread zone — so ordering is a logic
                 gate plus wasted work, not a wedge;
  6. negative B — the backrest with only ONE hole over a stud (yawed) must not count (the
                 two-point insertion term);
  7. sag       — pause mid-thread on a nut (wrench cleared): it must stay on the horizontal
                 stud (the engaged joint holds it) — the regrip-gap failure mode.
  --demo runs ONLY show + oracle and saves the deliverable video.

ALWAYS records video via the viewport rgb annotator (same recipe as crate_packing_smoke:
RTX driver-version override, 3-render ghost flush, npz out). Bodies are driven straight
through scene handles; the NullRobot applies nothing.

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
                         "negative controls) — the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--surface_z", type=float, default=None,
                    help="override the work-surface height (None = the scene default)")
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

# Assemble: twist about the stud axis, capping the spin rate — nut_thread's torque
# constants in the NUT'S BODY FRAME (its screw axis is local +z). The scene's measured-
# rotation screw joint turns that spin into helix advance; no press is needed.
TWIST, TARGET_W = -0.15, -3.0
# Stage each nut a few mm off the thread tip WITH an approach velocity: the gap avoids a
# crest-on-crest spawn (axially ramming mated threads pumps energy and ejects the nut);
# the motion keeps PhysX awake until contact, and the scene's sleep_threshold=0 nuts
# never sleep thereafter.
STAGE_GAP = 0.004
# Real threading pace: the spin cap (3 rad/s) on the 2 mm-pitch helix feeds ~0.95 mm/s,
# so the ~20 mm of nut travel needs ~21 s of sim = ~2500 steps.
SCREW_BUDGET = 3200
INSERT_STANDOFF = 0.060  # backrest staged this far out from the seated pose (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    cfg_kw = {} if args.surface_z is None else {"surface_z": args.surface_z}
    env = ENVS.get("assembly.chair")().build(
        num_envs=args.num_envs, device=device, scene_cfg=ChairAssemblySceneCfg(**cfg_kw))
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    bodies = {"base": scene.base, "back": scene.back,
              "nut_0": scene.nuts[0], "nut_1": scene.nuts[1]}
    # -90 deg about x maps local +z -> +y (the stud rigs use the same rotation): the nut's
    # screw axis points OUT along the stud, so the body-frame press (0,0,-|F|) drives it
    # toward the chair. (+90 was tried first: press = away, the nut backs off the tip.)
    QX90 = torch.tensor([math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0],
                        device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        # PARTIAL_RENDERING desyncs under the screw mechanic's per-substep root-state
        # writes (Isaac 5.1: the render transforms drift off and the annotator eventually
        # freezes — physics stays correct, 16/16 checks pass, but the footage dies). The
        # demo keeps FULL rendering; stress phases can afford the desync for speed.
        if not args.demo:
            env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        # rear-quarter view: the studs + nut work happen BEHIND the chair (+y side);
        # surface-relative so the platform preset frames the same way as the floor
        env.sim.set_camera_view(tuple(np.array((1.45, 1.55, c.surface_z + 0.95)) + o),
                                tuple(np.array((0.0, 0.15, c.surface_z + 0.42)) + o),
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
    # cancels to exactly zero (the pen-holder lesson).
    holds: dict[str, torch.Tensor] = {}
    g_dt = 9.81 * env.dt

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
            for name, st in holds.items():  # re-pin at ZERO velocity before judging
                bodies[name].write_root_state_to_sim(st, all_ids)
            env.iscene.update(0.0)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | assembled={scene.assembled()[0].int().tolist()} "
              f"score={int(scene.score()[0])} viol={int(scene.order_violations[0])} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def diagnose(tag: str) -> None:
        """Per-pair rubric-term breakdown — printed on a failed check so the log names the
        guilty term instead of leaving it to the video."""
        from isaaclab.utils.math import quat_apply_inverse

        back, nuts = bool(scene._back_seated()[0]), scene._nuts_seated()[0]
        lat = scene._nut_stud_lat()[0]
        loc = scene._nut_rel_base()[0]
        bp, bq = scene._base_frame()
        bloc = quat_apply_inverse(bq, scene.back.data.root_pos_w - bp)[0]
        b_up = scene._axes_w(scene.back.data.root_quat_w, (0.0, 0.0, 1.0))[0]
        s_up = scene._axes_w(bq, (0.0, 0.0, 1.0))[0]
        fwd_b = scene._axes_w(scene.back.data.root_quat_w, (0.0, 1.0, 0.0))[0]
        fwd_s = scene._axes_w(bq, (0.0, 1.0, 0.0))[0]
        relyaw = math.degrees(math.atan2(
            float(fwd_s[0] * fwd_b[1] - fwd_s[1] * fwd_b[0]), float((fwd_s * fwd_b).sum())))
        print(f"[smoke]   {tag}: back_seated={back} "
              f"back_loc=({bloc[0]:+.4f},{bloc[1]:+.4f},{bloc[2]:+.4f}) "
              f"(target {c.back_seat_pos}) upcos={float((b_up * s_up).sum()):+.3f} "
              f"relyaw={relyaw:+.1f}deg "
              f"nuts_geo={nuts.int().tolist()} "
              f"nut_lat={[f'{v * 1000:.0f}mm' for v in lat.tolist()]} "
              f"nut_depth={[f'{(c.thread_y1 - v) * 1000:+.0f}mm' for v in loc[:, 1].tolist()]} "
              f"welded={scene.welded[0].int().tolist()}", flush=True)

    # --- staging helpers ------------------------------------------------------------------
    def base_pose() -> tuple[torch.Tensor, torch.Tensor]:
        """Base pos (3,) LOCAL to the env origin + quat (4,), env 0, live."""
        return (scene.base.data.root_pos_w[0] - env.iscene.env_origins[0],
                scene.base.data.root_quat_w[0])

    def base_to_local(local) -> torch.Tensor:
        """A base-frame point -> env-origin-local world coords, (3,)."""
        bp, bq = base_pose()
        v = torch.tensor(local, device=device, dtype=torch.float32)
        return bp + quat_apply(bq.unsqueeze(0), v.unsqueeze(0))[0]

    def base_quat_mul(q_local: torch.Tensor) -> torch.Tensor:
        _bp, bq = base_pose()
        return quat_mul(bq.unsqueeze(0), q_local.unsqueeze(0))[0]

    def stud_axis_w() -> torch.Tensor:
        """The studs' +y axis in world coords, (n, 3)."""
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.base.data.root_quat_w, ey)

    def make_state(pos, quat) -> torch.Tensor:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.as_tensor(pos, device=device)
        st[:, 3:7] = torch.as_tensor(quat, device=device)
        return st

    def teleport(name: str, pos, quat) -> None:
        bodies[name].write_root_state_to_sim(make_state(pos, quat), all_ids)

    def settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def slide_back_on(y_to: float = 0.2205, steps: int = 160) -> None:
        """Kinematic-hold insertion: stand the backrest upright at the stud line, then slide
        it -y (base frame) onto the shanks, and release. Pose AND orientation are re-derived
        from the LIVE base every step: dragging the back across the leg rail scoots/yaws the
        free-standing chair a few degrees, and a quat frozen at slide-start leaves the back
        relatively yawed at release — one hole then misses its stud (measured 7 mm)."""
        y_from = c.back_seat_pos[1] + INSERT_STANDOFF
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        for k in range(steps + 1):
            y = y_from + (y_to - y_from) * k / steps
            pos = base_to_local((c.back_seat_pos[0], y, c.back_seat_pos[2] + 0.002))
            holds["back"] = make_state(pos, base_quat_mul(ident))
            step(1)
        step(10)
        del holds["back"]
        step(30)

    def stage_nut(k: int, extra_gap: float = 0.0) -> None:
        """Sweep nut k to its stud's thread tip with 12 kinematic position writes (a
        teleport-park leaves a long-slept body inert: pose writes show, but neither the
        stage velocity nor the tensor-API wrench acts on it — measured 1400 steps at
        2.5 N, zero displacement), handing off 1 mm short of the tip at 2 cm/s so the
        threads meet gently instead of crest-slamming."""
        sx = (-c.stud_x, c.stud_x)[k]
        quat = base_quat_mul(QX90)
        for j in range(12):
            gap = STAGE_GAP + extra_gap - (STAGE_GAP - 0.001) * j / 11
            st = make_state(base_to_local((sx, c.thread_y1 + gap, c.stud_z)), quat)
            st[:, 7:10] = -0.02 * stud_axis_w()
            bodies[f"nut_{k}"].write_root_state_to_sim(st, all_ids)
            step(1)
        loc = scene._nut_rel_base()[0, k]
        print(f"  [stage nut_{k}] loc=({loc[0]:+.4f},{loc[1]:+.4f},{loc[2]:+.4f}) "
              f"tip_y={c.thread_y1} engaged={scene._scr_eng[0].int().tolist()}", flush=True)

    def clear_wrench(k: int) -> None:
        z = torch.zeros(n, 1, 3, device=device)
        scene.nuts[k].set_external_force_and_torque(z, z)

    def screw_nut(k: int, until, budget: int = SCREW_BUDGET) -> bool:
        """Twist nut k about the stud axis (nut_thread's torque + spin cap) until `until()`
        or the budget runs out. The scene's measured-rotation screw joint owns engagement,
        lateral pinning and the helix advance — the oracle's only job is REAL rotation.
        Clears the wrench after."""
        done = False
        cum_spin = 0.0  # rad about the stud axis, env 0 — the helix telemetry
        depth0 = None
        for i in range(budget):
            w_axis = (scene.nuts[k].data.root_ang_vel_w * stud_axis_w()).sum(dim=-1)
            t = torch.zeros(n, 1, 3, device=device)
            t[w_axis > TARGET_W, 0, 2] = TWIST  # spin cap: twist only below the target rate
            scene.nuts[k].set_external_force_and_torque(torch.zeros(n, 1, 3, device=device), t)
            step(1)
            cum_spin += float(w_axis[0]) * env.dt
            depth = c.thread_y1 - float(scene._nut_rel_base()[0, k, 1])
            if depth0 is None and depth > 0.002:
                depth0, spin0 = depth, cum_spin  # helix accounting starts at engagement
            if i % 25 == 24:
                exp = ""
                if depth0 is not None:
                    trav = (depth - depth0) * 1000
                    spun = math.degrees(abs(cum_spin - spin0))
                    exp = (f" travel={trav:+.1f}mm spun={spun:.0f}deg "
                           f"(helix wants {abs(trav) * 180:.0f}deg)")
                print(f"  [screw nut_{k}] step {i + 1:4d} depth={depth * 1000:+.1f}mm "
                      f"lat={scene._nut_stud_lat()[0, k] * 1000:.1f}mm{exp}", flush=True)
            if until():
                if depth0 is not None:
                    trav = (depth - depth0) * 1000
                    spun = math.degrees(abs(cum_spin - spin0))
                    print(f"  [screw nut_{k}] SEATED: travel={trav:+.1f}mm spun={spun:.0f}deg "
                          f"(helix wants {abs(trav) * 180:.0f}deg)", flush=True)
                done = True
                break
        clear_wrench(k)
        step(20)
        return done

    def finish() -> None:
        # ======================== save + verdict ============================================
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="assembly.chair")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            if shutil.which("hdfs"):
                rc = args.hdfs_dir and os.system(
                    f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                    f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
                print(f"[smoke] hdfs upload rc={rc} -> "
                      f"{args.hdfs_dir}/{os.path.basename(args.out)}", flush=True)
        all_ok = all(okc for _name, okc in checks)
        print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
              f"({sum(okc for _n, okc in checks)}/{len(checks)} checks)", flush=True)
        print("CHAIR_ASSEMBLY_SMOKE_DONE", flush=True)
        env.close()

    # =========================== phase 1: show ==============================================
    env.reset()
    step(150)
    report("show")
    check("show: score 0 after reset settle", int(scene.score()[0]) == 0)
    check("show: nothing welded", not bool(scene.welded[0].any()))

    # =========================== phase 2: oracle ============================================
    # Backrest first (the ordering-legal path): two-point insertion onto the shanks.
    slide_back_on()
    ok = settle_until(lambda: bool(scene.welded[0, 0]), max_steps=180)
    report("oracle/back")
    check("oracle: backrest seats and welds (score 33)", ok and int(scene.score()[0]) == 33)
    if not ok:
        diagnose("back failed")

    # Nuts: genuine horizontal threading, one stud at a time.
    for k in (0, 1):
        stage_nut(k)
        step(15)  # let the stage pose take
        ok = screw_nut(k, until=lambda k=k: bool(scene.welded[0, 1 + k]))
        report(f"oracle/nut_{k}")
        check(f"oracle: nut_{k} threads on and welds", ok)
        if not ok:
            diagnose(f"nut_{k} failed")
    check("oracle: success, zero ordering violations",
          bool(scene.success()[0]) and int(scene.order_violations[0]) == 0)

    # =========================== phase 3: shake (on the oracle's real product) ==============
    ok = bool(scene.success()[0])
    check("shake: assembled before the shake", ok)
    bp0, bq0 = base_pose()
    for t_i in range(180):
        wob = 0.10 * math.sin(t_i / 9.0)
        pos = bp0 + torch.tensor([wob, 0.0, 0.30 + 0.05 * math.sin(t_i / 5.0)], device=device)
        holds["base"] = make_state(pos, bq0)
        step(1)
    still_ok = bool(scene.success()[0]) and bool(scene.welded[0].all())
    del holds["base"]
    step(60)
    check("shake: all 3 pairs hold while the chair swings (parent-frame rubric)", still_ok)

    if args.demo:
        finish()
        return

    # =========================== phase 4: repeat (frame independence) =======================
    for rep_i in range(2):
        env.reset()
        step(30)
        teleport("back", base_to_local(c.back_seat_pos), base_quat_mul(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)))
        ok = settle_until(lambda: bool(scene.welded[0, 0]), max_steps=240)
        check(f"repeat {rep_i}: teleport-seated back welds under a random base yaw", ok)
        if not ok:
            diagnose(f"repeat {rep_i}")

    # =========================== phase 5: negative A (ordering) =============================
    env.reset()
    step(30)
    stage_nut(0)
    step(15)
    screw_nut(0, until=lambda: scene._nut_rel_base()[0, 0, 1] < c.thread_y1 - 0.012,
              budget=SCREW_BUDGET)
    step(30)
    check("neg A: nut on a bare stud fires the ordering-violation counter",
          int(scene.order_violations[0]) >= 1)
    check("neg A: the premature nut never counts (gate)", not bool(scene.welded[0].any()))
    # recovery: UNSCREW (counter-rotate) until the joint releases at the tip, lift the
    # freed nut away, and the backrest goes on
    for i in range(SCREW_BUDGET):
        w_axis = (scene.nuts[0].data.root_ang_vel_w * stud_axis_w()).sum(dim=-1)
        t = torch.zeros(n, 1, 3, device=device)
        t[w_axis < -TARGET_W, 0, 2] = -TWIST  # counter-twist, same spin cap
        scene.nuts[0].set_external_force_and_torque(torch.zeros(n, 1, 3, device=device), t)
        step(1)
        if not bool(scene._scr_eng[0, 0]):
            break
    clear_wrench(0)
    check("neg A: counter-rotation unscrews the nut to release (two-way joint)",
          not bool(scene._scr_eng[0, 0]))
    teleport("nut_0", (0.5, -0.5, 0.02), torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
    step(30)
    slide_back_on()
    ok = settle_until(lambda: bool(scene.welded[0, 0]), max_steps=180)
    check("neg A: with the stud freed the episode recovers (backrest seats)", ok)
    if not ok:
        diagnose("neg A recovery")

    # =========================== phase 6: negative B (two-point term) =======================
    # Pure-rubric check, no stepping (the pose interpenetrates, so it is written, read, and
    # replaced without ever simulating): a back at the correct ORIGIN but yawed 14 deg has
    # its holes ~5 cm off the studs — the per-hole two-point term must reject it even though
    # 14 deg is inside the ori_cos cone (cos 14 = 0.970 >= 0.94).
    env.reset()
    step(30)
    yaw14 = torch.tensor([math.cos(math.radians(7.0)), 0.0, 0.0,
                          math.sin(math.radians(7.0))], device=device)
    teleport("back", base_to_local(c.back_seat_pos), base_quat_mul(yaw14))
    env.iscene.update(0.0)
    check("neg B: origin-correct but yawed back fails the per-hole two-point term",
          not bool(scene._back_seated()[0]))
    teleport("back", (0.6, -0.6, 0.10), base_quat_mul(yaw14))
    step(30)

    # =========================== phase 7: sag (horizontal-thread regrip gap) ================
    env.reset()
    step(30)
    slide_back_on()
    settle_until(lambda: bool(scene.welded[0, 0]), max_steps=180)
    stage_nut(0)
    step(15)
    screw_nut(0, until=lambda: (c.thread_y1 - scene._nut_rel_base()[0, 0, 1]) > 0.008,
              budget=SCREW_BUDGET)  # a few threads in, then STOP (wrench cleared)
    lat0 = float(scene._nut_stud_lat()[0, 0])
    step(240)  # two seconds hands-off
    lat1 = float(scene._nut_stud_lat()[0, 0])
    on_axis = lat1 <= c.tau_xy
    print(f"[smoke] sag: lat {lat0 * 1000:.1f}mm -> {lat1 * 1000:.1f}mm after 2 s "
          f"hands-off", flush=True)
    check("sag: a part-threaded nut stays on the horizontal stud unattended", on_axis)

    finish()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin) — the
    repo's standard hard-exit (see robobench/scripts/smoke.py): a watchdog guarantees the
    process ends."""
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
