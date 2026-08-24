"""Step 5 concrete: construct the sim scene and record it.

A Franka beside the calibrated real-desk scene; random task-envelope motion;
records per frame: RGB + semantic masks (external + wrist cameras) and camera
poses in the scene's COLMAP frame.

    python demo_franka_scene.py <scene> <run_name> \
        [--object <name>[:x,y[,yaw_deg]]] ...

--object places a calibrated object USD (from calibrate_object.py, i.e.
objects/data/objects/<name>/<name>.usd — origin at bottom center) on the desk;
repeatable. Position defaults walk along the desk if omitted. Objects are
rigid bodies (they settle under physics) and are labeled for the composite.

Output: background/data/outputs/<run_name>/franka_desk/{external,wrist}/...
Runs inside the repo's Isaac venv (self-bootstraps).
"""

import argparse
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
REPO = str(_HERE.parents[3])
DATA = str(_HERE.parents[2] / "background" / "data")

_ISAAC_PY = pathlib.Path(REPO) / ".venv" / "bin" / "python"
if _ISAAC_PY.exists() and pathlib.Path(sys.executable).resolve() != _ISAAC_PY.resolve():
    os.execv(str(_ISAAC_PY), [str(_ISAAC_PY), *sys.argv])

import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("scene")
_ap.add_argument("run_name")
_ap.add_argument("--object", action="append", default=[], dest="objects",
                 metavar="NAME[:x,y[,yaw_deg]]",
                 help="calibrated object to place on the desk (repeatable)")
_args = _ap.parse_args()
SCENE, RUN = _args.scene, _args.run_name
OBJECTS_DATA = _HERE.parents[2] / "objects" / "data" / "objects"

def _parse_obj(spec, i):
    name, _, pose = spec.partition(":")
    default_spots = [(0.12, -0.10), (0.10, 0.12), (-0.05, -0.14), (-0.08, 0.10)]
    x, y = default_spots[i % len(default_spots)]
    yaw = 0.0
    if pose:
        parts = [float(v) for v in pose.split(",")]
        x, y = parts[0], parts[1]
        if len(parts) > 2:
            yaw = parts[2]
    usd = OBJECTS_DATA / name / f"{name}.usd"
    assert usd.exists(), f"missing {usd} (run calibrate_object.py {name} first)"
    return name, str(usd), x, y, yaw

OBJ_SPECS = [_parse_obj(s, i) for i, s in enumerate(_args.objects)]
WS = f"{DATA}/colmap/{SCENE}"
SETUP = f"{WS}/scene.json"
FRANKA_USD = f"{REPO}/robobench/robots/assets/franka/panda_instanceable.usd"
OUT = f"{DATA}/outputs/{RUN}/franka_desk"
N_FRAMES = 240  # 30 fps -> 8 s (pick mode overrides to 360)
PHYS_PER_FRAME = 2  # physics steps (dt=1/60) per captured frame
WAYPOINT_EVERY = 45

WRIST = {"fx": 620.0, "fy": 620.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720}

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

ok = False
try:
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.semantics import add_labels
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    cfg = json.load(open(SETUP))

    # External camera: synthetic look-at pose in the desk frame — lower and
    # more level than the auto-picked phone frame (less top-down).
    def lookat_usd(eye, target, up=(0.0, 0.0, 1.0)):
        eye, target, up = np.array(eye, float), np.array(target, float), np.array(up, float)
        f = target - eye
        f /= np.linalg.norm(f)
        x = np.cross(f, up); x /= np.linalg.norm(x)   # cam right
        y = np.cross(x, f)                            # cam up
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, -f, eye
        return T

    # Near-shell pose: f_0551's real pose pulled back + tilted up for headroom
    ext_real = json.load(open(f"{WS}/ext_cam_real.json"))
    T_ext = np.array(ext_real["T_usd_cam_isaac"])  # real pose, used as-is (on-shell)
    cfg = dict(cfg)
    ei = dict(ext_real["intrinsics"])
    scale_px = 1280.0 / ei["width"]  # render at 1280-wide regardless of capture res
    for k in ("fx", "fy", "cx", "cy"):
        ei[k] = ei[k] * scale_px
    ei["width"] = 1280
    ei["height"] = int(round(ei["height"] * scale_px))
    cfg["intrinsics"] = ei
    intr = cfg["intrinsics"]
    W, H = intr["width"], intr["height"]
    ex, ey = cfg["surface_extents_m"]
    R_ic = np.array(cfg["R_world_from_colmap"])  # rotation colmap->world (rows)
    s = cfg["scale_m_per_unit"]
    desk_center = np.array(cfg["surface_center_colmap"])

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    stage = world.stage

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(700)

    # Invisible home-desk collider (top at z=0)
    desk = UsdGeom.Cube.Define(stage, "/World/desk_col")
    desk.GetSizeAttr().Set(1.0)
    dxf = UsdGeom.XformCommonAPI(desk)
    dxf.SetScale(Gf.Vec3f(ex / 2, ey / 2, 0.02))
    dxf.SetTranslate(Gf.Vec3d(0, 0, -0.02))
    UsdPhysics.CollisionAPI.Apply(desk.GetPrim())
    UsdGeom.Imageable(desk.GetPrim()).MakeInvisible()



    # Calibrated objects on the desk (rigid bodies; origin = bottom center,
    # so z=0 stands them on the surface; +2 mm drop lets contacts resolve)
    for name, usd, ox, oy, oyaw in OBJ_SPECS:
        path = f"/World/obj_{name}"
        add_reference_to_stage(usd, path)
        oxf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(path))
        oxf.SetTranslate(Gf.Vec3d(ox, oy, 0.002))
        oxf.SetRotate(Gf.Vec3f(0, 0, oyaw))
        add_labels(stage.GetPrimAtPath(path), labels=["object"], instance_name="class")
        print(f"object: {name} at ({ox}, {oy}, yaw {oyaw})", flush=True)

    # Franka on the lab table, facing the desk (+x)
    add_reference_to_stage(FRANKA_USD, "/World/franka")
    robot = Robot("/World/franka", name="franka", position=np.array([-0.50, 0.0, 0.0]))
    world.scene.add(robot)
    add_labels(stage.GetPrimAtPath("/World/franka"), labels=["robot"], instance_name="class")

    # External camera (calibrated composite pose)
    def make_cam(path, fx, fy, w, h):
        cam = UsdGeom.Camera.Define(stage, path)
        cam.GetFocalLengthAttr().Set(float(fx) * 20.955 / w)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetVerticalApertureAttr().Set(20.955 * h / w)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        return cam

    ext_cam = make_cam("/World/cam_ext", intr["fx"], intr["fy"], W, H)
    UsdGeom.Xformable(ext_cam.GetPrim()).AddTransformOp().Set(
        Gf.Matrix4d(*[T_ext.T.flatten()[i] for i in range(16)])
    )

    # Wrist camera under the hand link: cam -z (forward) = hand +z (approach)
    hand_path = "/World/franka/panda_hand"
    assert stage.GetPrimAtPath(hand_path).IsValid(), f"missing {hand_path}"
    wrist_cam = make_cam(hand_path + "/wrist_cam", WRIST["fx"], WRIST["fy"], WRIST["width"], WRIST["height"])
    T_off = np.eye(4)
    ang = np.deg2rad(180.0)  # cam forward (-z) = hand +z (toward fingertips)
    Rx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    Rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # roll: hand at frame bottom
    T_off[:3, :3] = Rx @ Rz90
    T_off[:3, 3] = [0.065, 0.0, -0.02]  # to the side + slightly toward wrist
    UsdGeom.Xformable(wrist_cam.GetPrim()).AddTransformOp().Set(
        Gf.Matrix4d(*[T_off.T.flatten()[i] for i in range(16)])
    )

    rp_ext = rep.create.render_product("/World/cam_ext", (W, H), name="external")
    rp_wrist = rep.create.render_product(hand_path + "/wrist_cam", (WRIST["width"], WRIST["height"]), name="wrist")
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=OUT, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=False)

    world.reset()
    print("DOF:", robot.num_dof, flush=True)
    hand_xf = XFormPrim(hand_path)

    q0 = robot.get_joint_positions()
    # Task-like envelope: hand hovers over the desk, tool pointing down
    arm_default = np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.85, 0.78])
    lo = np.array([-0.5, -0.6, -0.5, -2.5, -0.6, 1.4, 0.2])
    hi = np.array([0.5, 0.25, 0.5, -1.5, 0.6, 2.3, 1.4])
    rng = np.random.default_rng(3)

    def random_waypoint():
        return np.clip(arm_default + rng.uniform(-0.35, 0.35, size=7), lo, hi)

    ctrl = robot.get_articulation_controller()

    PICK = bool(OBJ_SPECS)  # objects present -> the arm tries to pick the first one
    if PICK:
        N_FRAMES = 360  # noqa: F811 — pick choreography needs 12 s
        from isaacsim.robot_motion.motion_generation import (ArticulationMotionPolicy,
                                                             RmpFlow,
                                                             interface_config_loader)

        mp_cfg = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        rmp = RmpFlow(**mp_cfg)
        rmp.set_robot_base_pose(np.array([-0.50, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        amp = ArticulationMotionPolicy(robot, rmp, 1.0 / 60)
        pname, _, px, py, _ = OBJ_SPECS[0]
        obj_prim_path = f"/World/obj_{pname}"
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(
            stage.GetPrimAtPath(obj_prim_path)).ComputeAlignedRange()
        ztop = float(bb.GetMax()[2])
        print(f"pick target: {pname} top at z={ztop:.3f}", flush=True)
        DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down (wxyz)
        GRASP_Z = ztop - 0.035  # fingers wrap the top band deeply
        obj_xf = XFormPrim(obj_prim_path)

        def pick_phase(f):
            """-> (ee target xyz, gripper half-width)"""
            if f < 90:                     # hover above
                return np.array([px, py, ztop + 0.12]), 0.04
            if f < 160:                    # descend around the top
                a = (f - 90) / 70.0
                return np.array([px, py, ztop + 0.12 - a * (ztop + 0.12 - GRASP_Z)]), 0.04
            if f < 220:                    # close (give contacts time to settle)
                return np.array([px, py, GRASP_Z]), 0.0
            if f < 310:                    # lift
                a = (f - 220) / 90.0
                return np.array([px, py, GRASP_Z + a * 0.15]), 0.0
            return np.array([px, py, GRASP_Z + 0.15]), 0.0  # hold

    wp_from, wp_to = random_waypoint(), random_waypoint()

    # settle INTO the start pose during warmup so frame 0 is already
    # in the task envelope (wrist looking at the desk, not across the room)
    q_start = q0.copy()
    q_start[:7] = arm_default if PICK else wp_from
    if len(q_start) > 7:
        q_start[7:] = 0.04 if PICK else 0.035
    ctrl.apply_action(ArticulationAction(joint_positions=q_start))
    for _ in range(90):
        world.step(render=True)
    writer.attach([rp_ext, rp_wrist])

    flip = np.diag([1.0, -1.0, -1.0])  # USD cam <-> OpenCV cam
    wrist_poses = []
    for f in range(N_FRAMES):
        if PICK:
            ee_pos, grip = pick_phase(f)
            rmp.set_end_effector_target(ee_pos, DOWN)
            ctrl.apply_action(amp.get_next_articulation_action(1.0 / 60))
            ctrl.apply_action(ArticulationAction(joint_positions=np.array([grip, grip]),
                                                 joint_indices=np.array([7, 8])))
        else:
            k = f % WAYPOINT_EVERY
            if k == 0 and f > 0:
                wp_from, wp_to = wp_to, random_waypoint()
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / WAYPOINT_EVERY)
            full = q0.copy()
            full[:7] = (1 - a) * wp_from + a * wp_to
            if len(full) > 7:
                full[7:] = 0.035
            ctrl.apply_action(ArticulationAction(joint_positions=full))
        for _ in range(PHYS_PER_FRAME):
            world.step(render=False)

        # wrist cam world pose (fabric-truth hand pose ∘ fixed offset)
        pos, quat = hand_xf.get_world_poses()
        pos, quat = pos[0], quat[0]  # wxyz
        w_, x_, y_, z_ = [float(v) for v in quat]
        Rh = np.array(
            [
                [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)],
            ]
        )
        T_hand = np.eye(4)
        T_hand[:3, :3] = Rh
        T_hand[:3, 3] = np.array([float(v) for v in pos])
        T_wcam_usd = T_hand @ T_off
        # -> OpenCV convention, -> COLMAP frame (inverse of isaac= s*R*(x-c))
        R_cv = T_wcam_usd[:3, :3] @ flip
        t_isaac = T_wcam_usd[:3, 3]
        R_colmap = R_ic.T @ R_cv
        t_colmap = R_ic.T @ (t_isaac / s) + desk_center
        T_c = np.eye(4)
        T_c[:3, :3] = R_colmap
        T_c[:3, 3] = t_colmap
        wrist_poses.append(T_c.tolist())

        rep.orchestrator.step(delta_time=0.0, rt_subframes=4, pause_timeline=False)

    rep.orchestrator.wait_until_complete()
    json.dump({"poses": wrist_poses, "intrinsics": WRIST}, open(f"{OUT}/wrist_poses.json", "w"))
    R_cv_ext = T_ext[:3, :3] @ flip
    T_ext_colmap = np.eye(4)
    T_ext_colmap[:3, :3] = R_ic.T @ R_cv_ext
    T_ext_colmap[:3, 3] = R_ic.T @ (T_ext[:3, 3] / s) + desk_center
    json.dump({"T_cam_to_world": T_ext_colmap.tolist(), "intrinsics": cfg["intrinsics"]},
              open(f"{OUT}/ext_pose.json", "w"))
    counts = {}
    for d in ("external", "wrist"):
        rgbdir = os.path.join(OUT, d, "rgb")
        counts[d] = len(os.listdir(rgbdir)) if os.path.isdir(rgbdir) else 0
    print("CAPTURED:", counts, flush=True)
    ok = all(v >= N_FRAMES - 2 for v in counts.values()) and len(counts) == 2
    if PICK:
        obj_z = float(obj_xf.get_world_poses()[0][0][2])
        print(f"PICK: object bottom at z={obj_z:.3f} " +
              ("LIFTED" if obj_z > 0.05 else "NOT LIFTED"), flush=True)
        ok = ok and obj_z > 0.05
except Exception:
    import traceback

    traceback.print_exc()
    print("SCRIPT_ERROR ^^^", flush=True)
finally:
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    app.close()
