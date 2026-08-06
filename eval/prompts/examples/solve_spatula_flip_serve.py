"""solve — spatula flip & serve (tool_use.spatula, goal v2 flip_serve) with a REAL Franka
arm, OSC servo loop on the FrankaSession substrate.

Strategy: TOP-DOWN pinch across the spatula's handle (the corpus's proven grasp
primitive; an along-the-axis "human grip" was tried first and abandoned — the descending
gripper body clipped the scan colliders' inflated tops and shoved the tool every try),
let the scene's weld-on-closure contract lock the tool, then run the tool through the
maneuver set the NullRobot smoke validated on this exact scene — every waypoint is
commanded as a TOOL pose and converted to a hand goal through the hand->tool transform
the weld recorded; the flip roll and tip pitch become arm arcs the OSC tracks:

  GRASP    hand slides down onto the handle's grip band, ramped close -> weld locks
  LIFT     tool level at altitude (rubric: lifted)
  WEDGE    pitched over-the-rim entry (28 deg), relax to the 14 deg run pitch about the
           planted tip, SLOW outward jab pinning the slice on the pan wall, fused
           lift+level out (rubric: wedged; retries re-aim, wall-adjacent -> tangential,
           escaped payload -> flat jab at its own plane)
  FLIP     hover over the pan, roll the blade past the commit point (150 deg), the slice
           peels off and lands inverted in the pan (rubric: flipped; escalating ladder)
  RE-WEDGE + CARRY  loaded traverse to the plate (rubric: loaded-after-flip)
  TIP      descend to just above the plate rim, pitch the blade, the slice slides into
           the recess (rubric: served) -> retreat, open, done

TOOL-ONLY holds by construction: the fingers only ever touch the handle. The arm never
dewinds while welded (a joint reset would drag the tool violently); wound-arm recovery
happens only before the grasp.

Run:  python -m solve_spatula_flip_serve --headless        (with eval/prompts/examples
      on PYTHONPATH, robobench importable, kitchen assets built)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=420.0, help="sim-time budget (s)")
parser.add_argument("--video", action="store_true", help="record an mp4 of the run")
parser.add_argument("--video_path", type=str, default="/tmp/spatula_franka_solve.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
    # RTX recipe (the smoke's L20 lesson): kit mis-decodes the driver version and
    # silently rejects RTX -> cameras return BLACK frames. Disable the check.
    if not getattr(args, "kit_args", None):
        args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import EnvCfg  # noqa: E402
from robobench.robots.franka import FrankaRobotCfg  # noqa: E402
from robobench.suites.tool_use.scenes import SpatulaFlipServeSceneCfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills"))
from franka_session import OPEN, FrankaSession, V3  # noqa: E402

GRIP = 0.006          # PER-FINGER close target: commanded well past the stick (16-21 mm
# across along the grip band) so the pads PRESS it and stall — a command that matches
# the stick width only kisses it: zero contact force, and the retreat dragged the tool
# 33 cm by friction (measured, run 2). 0.017/finger never even touched (run 1).
ENTRY_PITCH = 28.0    # tip-down entry over the pan rim (deg)
RUN_PITCH = 14.0      # jab pitch: must stay under the ~18 deg friction angle or the
# slice slides ahead of the blade instead of climbing (measured at 20). Rim contact is
# a non-issue here: the welded tool is DYNAMIC through the hand, so the pan pushes back
FLIP_LADDER = ((0.015, 110.0, 3.0, +1.0), (0.015, 110.0, 3.0, -1.0),
               (0.030, 120.0, 2.6, -1.0))
# WALL-ASSISTED topple, ALTERNATING ROLL SIGN: the arm's positive roll physically
# stalls near 70 deg (measured — 110-150 deg commands all achieve ~70 whatever the
# speed or the catch-up hold), and a slow roll peels the slice off at ~30 deg to land
# flat. The mirror direction may carry more wrist headroom, so the ladder tries both:
# park the blade so the slide-off side faces the pan wall (positive roll drops the
# MINUS-y edge — the measured sign lesson), nose the slice into the wall, lever over.
TIP_DEG = 40.0        # blade pitch that slides the slice off onto the plate (steeper
# exits spin the slice into a face-down landing — the smoke's measured lesson)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    robobench.discover()

    scene_cfg = SpatulaFlipServeSceneCfg(
        goal="flip_serve", sample_size=False,
        # the registered franka layout, with the spatula rest pulled 6 cm toward the
        # robot: at (0.02, -0.08) the grip band landed a 0.40 m side-reach where the
        # OSC parked 11 mm off target — too coarse for the weld's centred-pinch window
        pan_pos=(-0.14, 0.08), plate_pos=(0.14, 0.10), spatula_pos=(0.02, -0.14),
        surface_z=0.0,  # table-mounted arm at ground level (envs.py franka pattern);
        # the packing preset's native 0.994 would strand the ground-mounted base
    )
    if args.video:  # inject a spectator camera into the scene's assets before build
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        from robobench.suites.tool_use.scenes.spatula_flip_serve import SpatulaFlipServeScene

        _orig_assets = SpatulaFlipServeScene.assets

        def _assets_with_cam(self):
            out = _orig_assets(self)
            out["video_cam"] = CameraCfg(
                prim_path="/World/video_cam",
                update_period=0.0,
                height=480, width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, clipping_range=(0.05, 20.0)),
            )
            return out

        SpatulaFlipServeScene.assets = _assets_with_cam

    # Mounted WEST of the scene facing +x, the solve_musical_cans geometry class: every
    # target 0.3-0.75 m straight ahead, no base yaw. The registered binding's side-mount
    # guess (base at (0,-0.40), yaw 90) put the grasp in a side-reach where the OSC
    # parked 11-21 mm off target — envs.py itself marks those placements "re-verify
    # with the per-binding stress smoke". Gripper effort 25 N (the musical_cans value):
    # a 120 N pinch punts the tool when the close lands imperfectly.
    rcfg = FrankaRobotCfg(
        base_pos=(-0.60, 0.05, 0.0),
        nullspace_dof_pos=(),          # pen_holder lesson: default posture winds the arm.
        # (Tried ON to fight the wrist pinning at q7 -2.90 — it neither freed the wrist
        # nor left the flip its roll authority; measured, reverted.)
        gripper_effort_limit=25.0,
        gripper_stiffness=2000.0,
    )
    env = EnvCfg(scene="spatula", robot="franka", control_mode="osc",
                 env_spacing=3.0, robot_cfg=rcfg,
                 scene_cfg=scene_cfg).build(num_envs=1, device=device)
    env.reset()
    # kp 320: the corpus's stiff-contact retune — at the default 220 the OSC parked
    # 13 mm off the grasp target and the pinch missed the weld window (measured).
    s = FrankaSession(env, kp=(320.0, 320.0, 320.0, 600.0, 600.0, 600.0))
    scene, c = env.scene, env.scene.cfg
    # Aim grasp goals with the WELD's own pinch point (hand + 103.4 mm), not the pad
    # tip (112 mm): the 8.6 mm difference plus a few mm of servo error pushed the
    # contract's band-distance gate past its 10 mm limit (measured).
    s.finger_len = scene.GRASP_PINCH_OFFSET
    origin = s.origin
    PAN_FLOOR = c.pan_floor_z
    RIM = c.surface_z + c.pan_rim_top

    frames: list = []
    if args.video:
        cam = env.iscene["video_cam"]
        eye = origin + V3(1.05, -0.85, 0.80, device)
        tgt = origin + V3(0.0, 0.05, c.surface_z + 0.05, device)
        cam.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))
        cap = {"k": 0}

        def on_step(_s):
            cap["k"] += 1
            if cap["k"] % 4 == 0:  # 60 Hz ctrl -> 15 fps
                # the session ticks with render=False: without explicit render passes
                # the camera buffer never refreshes and every frame is black (measured:
                # a 322-frame mp4 of 16 KB). 3 passes flush accumulation ghosting.
                for _ in range(3):
                    env.sim.render()
                cam.update(0.0, force_recompute=True)
                rgb = cam.data.output["rgb"][0]
                frames.append(rgb[..., :3].detach().cpu().numpy())

        s.on_step = on_step

    # ----- readbacks --------------------------------------------------------------------
    def tool_pose():
        st = scene.spatula.data.root_state_w[0]
        return st[0:3] - origin, st[3:7]

    def bread_pos():
        i = int(scene._present[0].int().argmax())
        name = c.families[i][0]
        return (scene.breads[name].data.root_pos_w[0] - origin), i

    def plate_xy():
        return scene.plate.data.root_pos_w[0][:2] - origin[:2]

    def score() -> int:
        return int(scene.score()[0])

    def held() -> bool:
        return bool(getattr(scene, "grasp_held", torch.zeros(1, 1, dtype=torch.bool))[0].any())

    def report(tag: str) -> None:
        b, i = bread_pos()
        tp, tq = tool_pose()
        gw = tp + quat_apply(tq.unsqueeze(0), V3(-0.216, 0.0, 0.076, device).unsqueeze(0))[0]
        print(f"[solve] {tag:12s} score={score()} held={held()} "
              f"bread=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f}) "
              f"tool=({tp[0]:+.3f},{tp[1]:+.3f},{tp[2]:+.3f} "
              f"q={tq[0]:+.2f},{tq[1]:+.2f},{tq[2]:+.2f},{tq[3]:+.2f}) "
              f"grip=({gw[0]:+.3f},{gw[1]:+.3f},{gw[2]:+.3f}) "
              f"on_blade={bool(scene.on_blade()[0, i])} on_pan={bool(scene.on_pan()[0, i])} "
              f"on_plate={bool(scene.on_plate()[0, i])} success={bool(scene.success()[0])}",
              flush=True)

    # ----- tool-pose control (hand goals through the weld's recorded transform) ----------
    rel = {"p": None, "q": None}

    def cache_weld_transform() -> None:
        rel["p"] = scene._gw_rel_p[0, 0].clone()
        rel["q"] = scene._gw_rel_q[0, 0].clone()

    def hand_goal_for_tool(tp: torch.Tensor, tq: torch.Tensor):
        hq = quat_mul(tq.unsqueeze(0), quat_conjugate(rel["q"].unsqueeze(0)))[0]
        hp = tp + origin - quat_apply(hq.unsqueeze(0), rel["p"].unsqueeze(0))[0]
        return hp, hq

    def qz(deg):
        h = math.radians(deg) / 2
        return torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)], device=device)

    def qy(deg):
        h = math.radians(deg) / 2
        return torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=device)

    def qx(deg):
        h = math.radians(deg) / 2
        return torch.tensor([math.cos(h), math.sin(h), 0.0, 0.0], device=device)

    def tool_q(yaw_rad: float, pitch_deg: float = 0.0, roll_deg: float = 0.0):
        q = qz(math.degrees(yaw_rad))
        if pitch_deg:
            q = quat_mul(q.unsqueeze(0), qy(pitch_deg).unsqueeze(0))[0]
        if roll_deg:
            q = quat_mul(q.unsqueeze(0), qx(roll_deg).unsqueeze(0))[0]
        return q

    def hold_tool(tp, tq, secs: float) -> None:
        hp, hq = hand_goal_for_tool(tp, tq)
        s.hold(hp, hq, GRIP, secs)

    def goto_tool(tp_t, tq_t, secs: float) -> None:
        """Sweep the tool from its MEASURED pose to the target — an instant big-error
        hold makes the OSC wind the wrist onto its limit (measured: q7 pinned at -2.90
        by the re-wedge's entry hold, poisoning every later translation)."""
        tp0, tq0 = tool_pose()
        tp0 = tp0.clone()
        tq0 = tq0.clone()
        tq1 = tq_t.clone()
        if float((tq0 * tq1).sum()) < 0.0:  # hemisphere-align: nlerp takes the short way
            tq1 = -tq1

        def g(f):
            q_mix = torch.nn.functional.normalize(tq0 * (1 - f) + tq1 * f, dim=0)
            return (V3(float(tp0[0]) + (float(tp_t[0]) - float(tp0[0])) * f,
                       float(tp0[1]) + (float(tp_t[1]) - float(tp0[1])) * f,
                       float(tp0[2]) + (float(tp_t[2]) - float(tp0[2])) * f, device), q_mix)

        sweep_tool(g, secs)
        hold_tool(tp_t, tq_t, 0.3)

    def sweep_tool(goal_fn, secs: float, gate_fn=None) -> bool:
        """Ease the TOOL through goal_fn(f in 0..1); early-True on gate_fn."""
        n = s.SEC(secs)
        for k in range(n):
            f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            tp, tq = goal_fn(f)
            hp, hq = hand_goal_for_tool(tp, tq)
            a = torch.zeros(1, s.n_act, device=device)
            s.servo(hp, hq, GRIP, a)
            s.tick(a)
            if gate_fn is not None and gate_fn():
                return True
        return gate_fn() if gate_fn is not None else True

    def settle(pred, secs: float = 4.0) -> bool:
        deadline = s.sim_t + secs
        while s.sim_t < deadline:
            tp, tq = tool_pose()
            hold_tool(tp, tq, 0.1)
            if pred():
                return True
        return pred()

    # ----- phase 1: grasp the handle ------------------------------------------------------
    # TOP-DOWN pinch across the stick (the corpus's proven grasp primitive). The first
    # design approached ALONG the handle axis — elegant for the later roll, but the
    # descending gripper body clipped the scan colliders' inflated tops and shoved the
    # tool off target every try (measured, runs 3-5). Vertical fingers straddle the
    # 2 cm stick with 3 cm of clearance a side and the palm stays 9 cm above the stick;
    # nothing sweeps sideways. Every goal recomputes from the LIVE tool pose (the
    # corpus's core lesson — a stale target aimed one close 60 mm off).
    def grip_point():
        """Live world position of the grip-band centre + the stick's world heading."""
        sp, sq = tool_pose()
        gw = sp + quat_apply(sq.unsqueeze(0), V3(-0.216, 0.0, 0.076, device).unsqueeze(0))[0]
        stick = quat_apply(sq.unsqueeze(0), V3(-1.0, 0.0, 0.0, device).unsqueeze(0))[0]
        return gw, math.atan2(float(stick[1]), float(stick[0]))

    def grasp() -> bool:
        for attempt in range(3):
            s.dewind()  # safe: nothing held yet
            gw0, heading = grip_point()
            # jaw ACROSS the stick; alternate the hand yaw 180 deg between attempts —
            # the mirrored wrist configuration tracks where the first one sagged
            side = 1.0 if attempt % 2 == 0 else -1.0
            gq = s.tilt_quat(heading + side * math.pi / 2, gw0[:2] + origin[:2])

            def tip_goal(dz):
                gw, _h = grip_point()
                return s.hand_for_tip(gw + origin + V3(0.0, 0.0, dz, device), gq), gq

            if not s.run_phase(lambda: tip_goal(0.10),
                               lambda: (s.ee_pose()[0][:2] - (grip_point()[0][:2]
                                        + origin[:2])).norm() < 0.008,
                               OPEN, 8.0, tag=f"g-hover{attempt}"):
                continue
            # descend to the servo's own steady state (the OSC parks ~13 mm off at this
            # close-in pose — no gate tighter than that can ever pass), then cancel the
            # static residual by BIASING the commanded goal with the measured error: an
            # off-centre pinch blows the weld's summed-gap window (one finger 8 mm, the
            # other 30 — measured), so the close must happen centred to ~5 mm.
            s.run_phase(lambda: tip_goal(0.0),
                        lambda: (s.tip_pos() - (grip_point()[0] + origin)).norm() < 0.010,
                        OPEN, 8.0, tag=f"g-descend{attempt}", xy_boost=1.6)
            # ONE measured-residual bias (an accumulating loop diverged — the servo's
            # error is command-dependent, not a constant offset; measured 12 -> 21 mm)
            gp, _ = tip_goal(0.0)
            gw, _h = grip_point()
            bias = s.tip_pos() - (gw + origin)
            s.hold(gp - bias, gq, OPEN, 0.8, xy_boost=1.6)
            err = float((s.tip_pos() - (grip_point()[0] + origin)).norm())
            if err > 0.008:
                print(f"[solve]   centring failed ({err*1000:.0f}mm)", flush=True)
                continue
            s.close_ramp(gp - bias, gq, GRIP, secs=0.9)
            ok = settle_grasp()
            gw, _h = grip_point()
            tip_err = float((s.tip_pos() - (gw + origin)).norm())
            print(f"[solve] grasp attempt {attempt}: held={ok} width={s.width()*1000:.1f}mm "
                  f"tip-to-band={tip_err*1000:.0f}mm", flush=True)
            if ok:
                cache_weld_transform()
                ex0 = quat_apply(tool_pose()[1].unsqueeze(0),
                                 V3(1.0, 0.0, 0.0, device).unsqueeze(0))[0]
                state["axis_yaw"] = math.atan2(float(ex0[1]), float(ex0[0]))
                return True
            p, q = s.ee_pose()
            s.hold(p, q, OPEN, 0.6)  # open IN PLACE first (retreating while the pads
            # still kiss the stick dragged the tool 33 cm across the bench — measured)
            s.hold(p + V3(0, 0, 0.10, device), q, OPEN, 0.8)
        return False

    def settle_grasp() -> bool:
        for _ in range(s.SEC(1.5)):
            p, q = s.ee_pose()
            s.hold(p, q, GRIP, 0.05)
            if held():
                return True
        # weld never fired: print the contract's own view of the closure criterion
        try:
            from isaaclab.utils.math import quat_apply

            art = scene._gw_art
            hp = art.data.body_pos_w[:, scene._gw_hand_i]
            hq = art.data.body_quat_w[:, scene._gw_hand_i]
            gap = float(art.data.joint_pos[0, scene._gw_fingers].sum())
            vel = float(art.data.joint_vel[0, scene._gw_fingers].abs().sum())
            off = torch.zeros_like(hp)
            off[:, 2] = scene.GRASP_PINCH_OFFSET
            pinch = hp + quat_apply(hq, off)
            dist = float(scene._gw_site_dists(pinch)[0, 0])
            print(f"[solve]   weld-debug: on={scene._gw_on} gap={gap*1000:.1f}mm "
                  f"(win {scene._gw_sites[0][4]}) vel={vel:.4f} "
                  f"(stall<{scene.GRASP_STALL}) band-dist={dist*1000:.1f}mm "
                  f"(<{scene.cfg.grasp_weld_dist*1000:.0f}) count={int(scene._gw_count[0,0])}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[solve]   weld-debug unavailable: {exc!r}", flush=True)
        return held()

    # ----- phase 2+: the tool maneuvers ---------------------------------------------------
    def wedge() -> bool:
        """The smoke-validated wedge, arm edition: per attempt aim from the live slice
        location (in-pan outward wall-pin / tangential when wall-adjacent / flat when
        escaped), pitched entry, slow jab, fused lift+level out."""
        # Already riding? Hoist level and DONE — flip() lands the slice back on the
        # blade, and a fresh jab from here knocks it off and bulldozes it out of the
        # pan (measured: r_off 0.061 -> 0.111 across a ladder run that started loaded).
        if bool(scene.on_blade()[0].any()):
            tp, tq_meas = tool_pose()
            ex_w = quat_apply(tq_meas.unsqueeze(0),
                              V3(1.0, 0.0, 0.0, device).unsqueeze(0))[0]
            yaw_now = math.atan2(float(ex_w[1]), float(ex_w[0]))
            hold_tool(V3(float(tp[0]), float(tp[1]), c.surface_z + 0.15, device),
                      tool_q(yaw_now), 3.0)
            if bool(scene.on_blade()[0].any()):
                print("[solve]   wedge: slice already riding — skip the jab", flush=True)
                return True
        for attempt in range(3):
            b, i = bread_pos()
            r_i = c.bread_r(i)
            h_i = c.bread_h(i)
            rx, ry = float(b[0] - c.pan_pos[0]), float(b[1] - c.pan_pos[1])
            r_off = math.hypot(rx, ry)
            in_pan = r_off < c.pan_r_floor + 0.01
            if not in_pan:
                pl = plate_xy()
                on_plate = math.hypot(float(b[0] - pl[0]), float(b[1] - pl[1])) < c.plate_r + 0.02
                gx, gy = (float(pl[0]), float(pl[1])) if on_plate else (c.pan_pos[0], c.pan_pos[1])
                dx, dy = gx - float(b[0]), gy - float(b[1])
                nrm = math.hypot(dx, dy) or 1.0
                dirv = (dx / nrm, dy / nrm)
            elif float(b[2]) - h_i / 2 > PAN_FLOOR + 0.008:
                # stranded ON the wall slope (a failed flip leaves it leaning there):
                # a floor-level sweep passes under it — jab INWARD at the slice's own
                # resting plane instead, dragging it back down to the flat floor
                dirv = (-rx / max(r_off, 1e-6), -ry / max(r_off, 1e-6))
            elif r_off > 0.075:
                dirv = (-ry / r_off, rx / r_off)  # tangential sweep off the wall
            elif r_off > 0.03:
                dirv = (rx / r_off, ry / r_off)  # outward: pin the slice on the wall
            else:
                dirv = (1.0, 0.0)
            # (aim quantization to the grasp axis was tried against the wrist pinning
            # and broke the FLIP geometry — reverted; the wrist accumulation is paid
            # off with a put-down-and-regrasp after the flip instead)
            yaw = math.atan2(dirv[1], dirv[0])
            sinp = math.sin(math.radians(ENTRY_PITCH))
            cosp = math.cos(math.radians(ENTRY_PITCH))
            sinr = math.sin(math.radians(RUN_PITCH))

            if in_pan:
                back = cosp * c.blade_l / 2 + r_i + 0.015
                ax, ay = float(b[0]) - dirv[0] * back, float(b[1]) - dirv[1] * back
                arx, ary = ax - c.pan_pos[0], ay - c.pan_pos[1]
                a_off = math.hypot(arx, ary)
                if a_off > 0.06:
                    ax = c.pan_pos[0] + arx / a_off * 0.06
                    ay = c.pan_pos[1] + ary / a_off * 0.06
                # wedge plane from the slice's OWN resting height (round-3 lesson): a
                # slope-stranded slice sits above the floor and a floor-level tip
                # sweeps under it
                base_z = max(PAN_FLOOR, float(b[2]) - h_i / 2 - 0.001)
                z_entry = base_z + 0.001 + sinp * c.blade_l / 2
                z_run = base_z + 0.001 + sinr * c.blade_l / 2
                goto_tool(V3(ax, ay, RIM + 0.06, device), tool_q(yaw, ENTRY_PITCH), 2.0)
                sweep_tool(lambda f, x=ax, y=ay: (
                    V3(x, y, RIM + 0.06 + (z_entry - RIM - 0.06) * f, device),
                    tool_q(yaw, ENTRY_PITCH)), 1.5)
                # relax to the run pitch about the planted tip
                tip = (ax + dirv[0] * cosp * c.blade_l / 2,
                       ay + dirv[1] * cosp * c.blade_l / 2,
                       z_entry - sinp * c.blade_l / 2)

                def relax_goal(f):
                    ang = ENTRY_PITCH + (RUN_PITCH - ENTRY_PITCH) * f
                    sa, ca = math.sin(math.radians(ang)), math.cos(math.radians(ang))
                    return (V3(tip[0] - dirv[0] * ca * c.blade_l / 2,
                               tip[1] - dirv[1] * ca * c.blade_l / 2,
                               tip[2] + sa * c.blade_l / 2, device), tool_q(yaw, ang))

                sweep_tool(relax_goal, 0.8)
                b, i = bread_pos()
                # jab advance CAPPED so the slice is never squeezed into the wall face
                # (a wall-press squirts it over the rim — the smoke's measured lesson)
                prx = float(b[0]) - c.pan_pos[0]
                pry = float(b[1]) - c.pan_pos[1]
                room = (c.pan_r_floor - r_i - 0.006) - math.hypot(prx, pry)
                adv = min(0.01, max(0.0, room))
                ex, ey = float(b[0]) + dirv[0] * adv, float(b[1]) + dirv[1] * adv
                tip_r = math.hypot(ex + dirv[0] * c.blade_l / 2 - c.pan_pos[0],
                                   ey + dirv[1] * c.blade_l / 2 - c.pan_pos[1])
                over = tip_r - (c.pan_r_floor - 0.004)
                if over > 0:
                    ex, ey = ex - dirv[0] * over, ey - dirv[1] * over
                p0 = tool_pose()[0]

                def jab_goal(f):
                    return (V3(float(p0[0]) + (ex - float(p0[0])) * f,
                               float(p0[1]) + (ey - float(p0[1])) * f, z_run, device),
                            tool_q(yaw, RUN_PITCH))

                sweep_tool(jab_goal, 1.6)  # 1.6 s wedges reliably; 2.4 s loses the
                # inertia anchor and bulldozes (measured both ways)
                hold_tool(V3(ex, ey, z_run, device), tool_q(yaw, RUN_PITCH), 0.5)
                # retract from the wall before lifting: the grippy wall holds a pinned
                # slice better than the slick blade (the smoke's measured lesson)
                sweep_tool(lambda f: (
                    V3(ex - dirv[0] * 0.025 * f, ey - dirv[1] * 0.025 * f, z_run, device),
                    tool_q(yaw, RUN_PITCH)), 0.6)
                lift_pitch = RUN_PITCH
            else:
                back = c.blade_l / 2 + r_i + 0.02
                z = max(float(b[2]) - h_i / 2 + 0.001, 0.001)
                ax, ay = float(b[0]) - dirv[0] * back, float(b[1]) - dirv[1] * back
                goto_tool(V3(ax, ay, c.surface_z + 0.09, device), tool_q(yaw), 2.0)
                sweep_tool(lambda f, x=ax, y=ay, zz=z: (
                    V3(x, y, c.surface_z + 0.09 + (zz - c.surface_z - 0.09) * f, device),
                    tool_q(yaw)), 1.2)
                b, i = bread_pos()
                ex, ey = float(b[0]) + dirv[0] * 0.01, float(b[1]) + dirv[1] * 0.01
                p0 = tool_pose()[0]
                sweep_tool(lambda f: (
                    V3(float(p0[0]) + (ex - float(p0[0])) * f,
                       float(p0[1]) + (ey - float(p0[1])) * f, z, device), tool_q(yaw)), 1.2)
                hold_tool(V3(ex, ey, z, device), tool_q(yaw), 0.5)
                lift_pitch = 0.0

            # ITERATIVE centering: the jab leaves the slice riding the tip half, and an
            # overhung ride tips over the tip edge on the first lift (measured at
            # bf x +46 mm — the round-5 lesson). One slide under-corrects through servo
            # lag, so measure-and-slide up to three times: the slick blade slides
            # forward beneath while the cargo keeps its world spot.
            q_work = tool_q(yaw, lift_pitch)
            for _c in range(3):
                b, i = bread_pos()
                locx = float(scene._bread_in_blade_frame()[0, i][0])
                if locx < 0.018 or not (0.001 < float(
                        scene._bread_in_blade_frame()[0, i][2]) - h_i / 2 < 0.02):
                    break
                tpn = tool_pose()[0]
                sweep_tool(lambda f, x0=float(tpn[0]), y0=float(tpn[1]), z0=float(tpn[2]),
                           bx_=float(b[0]), by_=float(b[1]): (
                    V3(x0 + (bx_ - x0) * f, y0 + (by_ - y0) * f, z0, device), q_work), 0.8)
                hold_tool(V3(float(b[0]), float(b[1]), float(tpn[2]), device), q_work, 0.3)

            # fused lift + level out (ramp flattens as the cargo rises)
            p0 = tool_pose()[0]
            z1 = RIM + 0.06

            def out_goal(f, lp=lift_pitch):
                return (V3(float(p0[0]), float(p0[1]),
                           float(p0[2]) + (z1 - float(p0[2])) * f, device),
                        tool_q(yaw, lp * (1.0 - f)))

            sweep_tool(out_goal, 1.6)
            tp, _ = tool_pose()
            hold_tool(V3(float(tp[0]), float(tp[1]), z1, device), tool_q(yaw), 0.4)
            b, i = bread_pos()
            if bool(scene.on_blade()[0, i]):
                state["yaw"] = yaw
                return True
            print(f"[solve] wedge attempt {attempt}: missed (r_off={r_off:.3f})", flush=True)
        return False

    state = {"yaw": 0.0}

    def flip() -> bool:
        for attempt, (height, roll_deg, secs, sgn) in enumerate(FLIP_LADDER):
            yaw = state["yaw"]
            perp = (-math.sin(yaw), math.cos(yaw))
            # park so the roll's slide-off side faces the wall: positive roll drops the
            # MINUS-y edge (the measured sign lesson), so the park side flips with the
            # roll sign. Low edge at the floor's outer margin: the slice lands nose-on
            # the wall and the continuing roll levers it over the anchored edge.
            bx = c.pan_pos[0] - sgn * perp[0] * 0.065
            by = c.pan_pos[1] - sgn * perp[1] * 0.065
            goto_tool(V3(bx, by, RIM + 0.06, device), tool_q(yaw), 2.0)
            sweep_tool(lambda f: (
                V3(bx, by, RIM + 0.06 + (PAN_FLOOR + height - RIM - 0.06) * f, device),
                tool_q(yaw)), 1.2)

            def roll_goal(f):
                ang = roll_deg * f
                edge = (c.blade_w / 2) * math.sin(math.radians(min(ang, 179.0)))
                z = PAN_FLOOR + max(height, 0.008 + edge)
                # from 60% of the roll on, shovel toward the wall and rise IN THE SAME
                # MOTION: the slice leans at 75-80 deg pinned blade-to-wall late in the
                # roll but falls back within a second if the blade pauses (measured —
                # a separate catch-up hold + scoop arrived too late every time)
                g = max(0.0, (f - 0.6) / 0.4)
                return (V3(bx - sgn * perp[0] * 0.035 * g,
                           by - sgn * perp[1] * 0.035 * g,
                           z + 0.05 * g, device), tool_q(yaw, 0.0, sgn * ang))

            # traced roll: servo through the arc printing the slice's state — one run
            # then tells the whole departure story whatever the outcome
            b0, i0 = bread_pos()
            n_roll = s.SEC(secs)
            for k in range(n_roll):
                f = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n_roll)
                tp_g, tq_g = roll_goal(f)
                hp, hq = hand_goal_for_tool(tp_g, tq_g)
                a = torch.zeros(1, s.n_act, device=device)
                s.servo(hp, hq, GRIP, a)
                s.tick(a)
                if k % max(1, n_roll // 5) == 0 or k == n_roll - 1:
                    tqm = tool_pose()[1]
                    eym = quat_apply(tqm.unsqueeze(0), V3(0, 1, 0, device).unsqueeze(0))[0]
                    bm, im = bread_pos()
                    print(f"[solve]     roll f={f:.2f} ach="
                          f"{math.degrees(math.asin(max(-1, min(1, -sgn * float(eym[2]))))):.0f} "
                          f"bread=({bm[0]:+.3f},{bm[1]:+.3f},{bm[2]:+.3f}) "
                          f"up_z={float(scene._bread_up()[0, im, 2]):+.2f} "
                          f"onb={int(scene.on_blade()[0, im])}", flush=True)
            # brief catch-up on the final COMMANDED pose (the shovel+scoop already
            # happened inside the roll; this just lets the servo finish the arc)
            gp_end, gq_end = roll_goal(1.0)
            hold_tool(gp_end, gq_end, 1.0)
            tp, tq = tool_pose()
            ey = quat_apply(tq.unsqueeze(0), V3(0.0, 1.0, 0.0, device).unsqueeze(0))[0]
            b, i = bread_pos()
            print(f"[solve]   flip {attempt} (sgn {sgn:+.0f}): achieved roll="
                  f"{math.degrees(math.asin(max(-1.0, min(1.0, -sgn * float(ey[2]))))):.0f}deg "
                  f"bread up_z={float(scene._bread_up()[0, i, 2]):+.2f} "
                  f"on_blade={bool(scene.on_blade()[0, i])}", flush=True)
            # retreat UP at the rolled attitude, then level away from the landing zone
            sweep_tool(lambda f, z0=float(tp[2]): (
                V3(bx, by, z0 + (RIM + 0.09 - z0) * f, device),
                tool_q(yaw, 0.0, sgn * roll_deg)), 0.8)
            sweep_tool(lambda f: (
                V3(bx - math.cos(yaw) * 0.16 * f, by - math.sin(yaw) * 0.16 * f,
                   RIM + 0.07, device), tool_q(yaw, 0.0, sgn * roll_deg * (1.0 - f))), 1.2)
            if settle(lambda: score() >= 50, 6.0):
                return True
            print(f"[solve] flip attempt {attempt}: not flipped-at-rest, re-wedging", flush=True)
            ok_w = wedge()
            if score() >= 50:  # the flip can latch LATE, mid-re-wedge (measured: the
                # slice settled inverted while the retry wedge was still probing)
                return True
            if not ok_w:
                return False
        return score() >= 50

    def carry_and_tip() -> bool:
        yaw = state["yaw"]
        pl = plate_xy()
        dirv = (math.cos(yaw), math.sin(yaw))
        tx, ty = float(pl[0]) - dirv[0] * 0.03, float(pl[1]) - dirv[1] * 0.03
        tp, _ = tool_pose()
        sweep_tool(lambda f, x0=float(tp[0]), y0=float(tp[1]), z0=float(tp[2]): (
            V3(x0, y0, z0 + (c.surface_z + 0.16 - z0) * f, device), tool_q(yaw)), 1.0)
        tp, _ = tool_pose()
        sweep_tool(lambda f, x0=float(tp[0]), y0=float(tp[1]): (
            V3(x0 + (tx - x0) * f, y0 + (ty - y0) * f, c.surface_z + 0.16, device),
            tool_q(yaw)), 3.5)
        # descend INSIDE the dish to ~2 cm above the recess floor (clear of both the
        # rim ring and the rising dish slope), tip from there: ~1.5 cm drop, no tumble
        low_z = c.plate_rest_z + 0.020
        sweep_tool(lambda f: (
            V3(tx, ty, c.surface_z + 0.16 + (low_z - c.surface_z - 0.16) * f, device),
            tool_q(yaw)), 1.2)

        def tip_goal(f):
            ang = TIP_DEG * f
            z = c.plate_rest_z + 0.018 + math.sin(math.radians(ang)) * (c.blade_l / 2)
            return (V3(tx, ty, z, device), tool_q(yaw, ang))

        sweep_tool(tip_goal, 4.4)  # slow: a snappy pitch corner-spins the slice (measured)
        # catch-up hold on the final COMMANDED pitch (holding the measured pose would
        # freeze the servo's rotational lag — the flip's "70 deg cap" lesson)
        gp_end, gq_end = tip_goal(1.0)
        hold_tool(gp_end, gq_end, 1.2)
        sweep_tool(lambda f, x0=float(tp[0]), y0=float(tp[1]): (
            V3(x0 - dirv[0] * 0.20 * f, y0 - dirv[1] * 0.20 * f,
               c.surface_z + 0.12, device), tool_q(yaw, TIP_DEG * (1.0 - f))), 1.5)
        return settle(lambda: score() >= 100, 4.0)

    # =========================== run the chain ==============================================
    report("reset")
    assert score() == 0, f"score={score()} at reset — rubric broken"

    if not grasp():
        report("grasp-FAIL")
        print("[solve] RESULT: FAIL (never grasped the handle)", flush=True)
        env.close()
        return
    report("grasped")

    tp, _ = tool_pose()
    hold_tool(V3(float(tp[0]), float(tp[1]), c.surface_z + 0.15, device),
              tool_q(0.0), 2.0)
    report("lifted")

    def wound() -> bool:
        return (abs(float(s.art.data.joint_pos[0, 6])) > 2.0
                or abs(float(s.art.data.joint_pos[0, 3]) + 2.2) > 0.7)

    def put_down_and_regrasp() -> bool:
        """PUT DOWN AND RE-GRASP: the flip/jab arcs park the wrist on its limit
        (measured q7 -2.90), and every in-hand recovery tried — yaw unwinds, nullspace
        posture, joint resets — either failed or ended in catastrophe. With the blade
        FREE this is cheap and human: set the tool on the table, release, dewind the
        unloaded arm (safe now), and take a fresh grip. Every primitive is proven."""
        print(f"[solve]   wound (q7 {float(s.art.data.joint_pos[0, 6]):+.2f}) — "
              f"put down and re-grasp", flush=True)
        goto_tool(V3(c.spatula_pos[0], c.spatula_pos[1], c.surface_z + 0.10, device),
                  tool_q(0.0), 2.5)
        goto_tool(V3(c.spatula_pos[0], c.spatula_pos[1], c.surface_z + 0.004, device),
                  tool_q(0.0), 1.5)
        p, qh = s.ee_pose()
        s.hold(p, qh, OPEN, 0.8)          # weld releases on the wide-open jaw
        s.hold(p + V3(0, 0, 0.12, device), qh, OPEN, 1.0)
        s.dewind()                         # free hand: a joint reset is safe
        if grasp():
            return True
        print("[solve]   re-grasp failed", flush=True)
        return False

    ok = wedge()
    report("wedged")
    ok = ok and flip()
    report("flipped")
    # Re-wedge with wound-arm recovery BETWEEN attempts: a pinned wrist aims the jab
    # wrong (measured: r_off growing 0.028 -> 0.111 across a ladder run while q7 sat
    # at -2.89), so a fresh grip is a real retry lever, not just a carry precaution.
    if ok:
        ok_w = False
        for _ in range(3):
            if wound() and not put_down_and_regrasp():
                break
            if wedge():
                ok_w = True
                break
            if not wound():
                break  # wedge ladders internally; a fresh grip was the only new lever
        ok = ok_w
    # the flip arcs + re-wedge leave the wrist near its limit (measured: q7 -2.90 of
    # +/-2.90), and EVERY recovery that demanded rotation or reset joints ended in
    # catastrophe: a yaw-change hold blew through a singularity (tool 1 m out); a
    # joint reset while welded teleported the hand away from the un-teleported tool
    # and the weld constraint flung the assembly 31 m (both measured). The carry
    # needs NO particular yaw — the plate is round, the tip-off works in any world
    # direction — so demand zero rotation: adopt the tool's MEASURED yaw as the
    # carry heading and only translate.
    q = s.art.data.joint_pos[0]
    print(f"[solve]   pre-carry joints q1={float(q[0]):+.2f} q4={float(q[3]):+.2f} "
          f"q7={float(q[6]):+.2f}", flush=True)
    tp, tq_meas = tool_pose()
    ex_w = quat_apply(tq_meas.unsqueeze(0), V3(1.0, 0.0, 0.0, device).unsqueeze(0))[0]
    state["yaw"] = math.atan2(float(ex_w[1]), float(ex_w[0]))
    hold_tool(V3(float(tp[0]), float(tp[1]), c.surface_z + 0.25, device),
              tool_q(state["yaw"]), 2.0)
    report("loaded")
    ok = ok and carry_and_tip()
    report("served")

    # open wide to release the weld, retreat
    p, q = s.ee_pose()
    s.hold(p, q, OPEN, 1.0)
    s.hold(p + V3(0, 0, 0.10, device), q, OPEN, 1.0)
    success = bool(scene.success()[0])
    print(f"[solve] RESULT: {'SUCCESS' if success and ok else 'FAIL'} "
          f"score={score()} success={success} sim_t={s.sim_t:.0f}s", flush=True)

    if args.video and frames:
        import numpy as np

        arr = np.stack(frames, axis=0)
        try:
            import imageio_ffmpeg

            w = imageio_ffmpeg.write_frames(
                args.video_path, (arr.shape[2], arr.shape[1]), fps=15,
                codec="libx264", pix_fmt_out="yuv420p",
                output_params=["-movflags", "+faststart"])
            w.send(None)
            for fr in arr:
                w.send(np.ascontiguousarray(fr))
            w.close()
            print(f"[solve] video -> {args.video_path} ({len(frames)} frames)", flush=True)
        except Exception as exc:  # noqa: BLE001
            np.savez_compressed(args.video_path + ".npz", frames=arr)
            print(f"[solve] ffmpeg failed ({exc!r}); frames -> {args.video_path}.npz", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
