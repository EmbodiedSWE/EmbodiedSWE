"""Bake the three SO101 joint-3 assembly assets as STANDALONE (flattened) USDs.

Source: the official Isaac 5.1 asset `Robots/RobotStudio/so101_new_calib` (converted from
TheRobotStudio SO-ARM100 URDF). It is auto-downloaded into assets/so101/source/ when missing and
is only a BUILD-time dependency — each output is flattened (all composition arcs baked, meshes
embedded), so `source/` can be deleted after baking. Its structure is ideal for partial-assembly
surgery: every physical part (printed shell vs STS3215 motor) is a separate prim under each link's
instanceable `visuals`/`collisions`, and `ArticulationRootAPI` sits on the `root_joint` world-pin
(same rig as the factory bolt). Collisions are per-part convexDecomposition, so the motor pockets
stay open (insertable).

The benchmark task is the LeRobot SO101 tutorial's "Joint 3" (elbow_flex) step with everything
else pre-built: drop the elbow motor into the upper-arm pocket, screw it in, then clip the distal
half onto the motor horn. URDF ground truth for who owns which motor: base=m1(shoulder_pan),
shoulder=m2(shoulder_lift), upper_arm=m3(elbow_flex), lower_arm=m4(wrist_flex), wrist=m5(wrist_roll),
gripper=m6. Hence the split:

  - so101_proximal.usd  = base + shoulder + upper_arm shell; joints 1-2 live; fixed base
                          (keeps root_joint + its ArticulationRootAPI); the m3 motor part is
                          deactivated inside upper_arm's visuals AND collisions -> empty pocket.
  - so101_distal.usd    = lower_arm + wrist + gripper + moving_jaw; joints 4-5 + gripper live;
                          FLOATING base: root_joint (and the proximal links/joints) deactivated,
                          ArticulationRootAPI re-applied on lower_arm.
  - sts3215_03a.usd     = the bare elbow motor as a free rigid body: upper_arm link with the shell
                          part deactivated, all joints deactivated, no ArticulationRootAPI.
                          Mass overridden to the servo's 61 g with CoM at the motor mesh center
                          and the URDF link inertia blocked so PhysX recomputes it.

KEY FRAME FACT (the whole reason the surgery is done this way): the motor asset keeps the
upper_arm LINK frame as its rigid-body frame, so "motor seated in the pocket" is exactly
"motor root pose == proximal upper_arm body pose", and the distal attach frame is
upper_arm_pose * (elbow localPos0/localRot0). No hand-measured offsets anywhere.

Known deferred items (visual/mesh pass only):
  - proximal upper_arm keeps the URDF lumped inertia (shell+motor); subtract the motor later.
  - the motor asset's rigid-body prim is still named `upper_arm` (an artifact of the surgery).

  python -m robobench.suites.assembly.scripts.build_so101_assets
  # then view:
  python -m robobench.scripts.view_usd robobench/suites/assembly/assets/so101/so101_proximal.usd --livestream 2
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # a pure asset build; no view needed
app = AppLauncher(args).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

SO101 = Path(__file__).resolve().parents[1] / "assets" / "so101"
SOURCE = SO101 / "source" / "so101_new_calib.usd"
SOURCE_URL = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
              "/Assets/Isaac/5.1/Isaac/Robots/RobotStudio/so101_new_calib")
SOURCE_FILES = ["so101_new_calib.usd", "configuration/so101_new_calib_base.usd",
                "configuration/so101_new_calib_physics.usd", "configuration/so101_new_calib_robot.usd",
                "configuration/so101_new_calib_sensor.usd"]

LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PROXIMAL_JOINTS = {"shoulder_pan", "shoulder_lift"}
DISTAL_LINKS = {"lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"}
DISTAL_JOINTS = {"wrist_flex", "wrist_roll", "gripper"}
PROXIMAL_LINKS = set(LINKS) - DISTAL_LINKS
MOTOR_PART = "sts3215_03a_v1"  # the with-horn servo variant used at the elbow
SHELL_PART = "upper_arm_so101_v1"
STS3215_MASS_KG = 0.061  # Feetech STS3215 servo weight per BOM
# The URDF motor mesh ships ASSEMBLED: all FOUR M2 tab screws are modeled in place — the
# horn-side (near) pair with heads in the near-tab counterbores, and the far-tab pair threading
# up into the shell's pilot bosses. Driving screws IS the task, so the free-motor asset strips
# all of them: drop every face that lies fully inside a small cylinder around each screw axis
# (leaves the counterbores as empty, screwable holes). Measured in the baked asset's world
# frame, millimeters: near tab face plane x=5.31, head dome r 1.75 in a r 1.85 counterbore,
# seat at x=6.81, a shank sliver to x~6.95; the far pair occupies x ~34.5..38.9 on the same
# axes (their heads at the -x end, shafts threading toward the shell pilot at +x).
MOTOR_SCREW_AXES_MM = [(-48.293, 271.886), (-48.294, 251.417)]  # (y, z) of the two screw axes
MOTOR_SCREW_CUTS = [((5.33, 6.86), 1.80), ((6.86, 7.00), 1.00),  # near pair: head + sliver
                    ((34.5, 37.0), 1.85), ((36.9, 41.0), 1.55)]  # far pair: head zone, then the
# shaft through the tab bore (r<1.55 spares the Ø3.44 bore wall at r 1.72; zones overlap so no
# face can straddle a boundary with all verts outside one cylinder yet inside the union)
# Registration pins: KEPT in collision (historically cut because Ø1.74 pins in Ø1.86 shell
# holes = 0.06 mm clearance jammed the servo insertion at SDF resolution — but the shell pin
# holes are now DRILLED to r 1.5 (SHELL_NEAR_CUTS), giving the pins 0.6 mm of real clearance).
# The pins are the near-side DROP CATCH: a dropped M2's tip rests on the pin top, mirroring
# the far side's rest on the registration posts — both faces behave identically, no holding.
MOTOR_PIN_AXES_MM = [(-52.47, 271.78), (-52.47, 251.28)]  # (y, z) of the two pin axes
# The pins'/posts' OUTER ends are trimmed (collision-only) so the SEATED screws' tips — the
# seats put both heads ~2 mm proud of their walls — never touch them: near pin down to link
# z 4.0, far posts removed (they only span link z 34.1..35.8 and the far tip seats at ~32).
# The seated tips overlap the VISUAL pin/post ends inside the bores, where it cannot be seen.
MOTOR_PIN_CUTS = [((4.2, 6.5), 1.45),    # near pin outer tips, link z 1.7..4.0
                  ((36.0, 43.0), 1.45)]  # far posts, link z 33.5..40.5
MOTOR_COLLISION_SHRINK = 0.97  # collision-only slip-fit shrink (see strip_motor_screws)
# THE FAR HOLES MIRROR THE NEAR ONES (measured: the far pin line at link y=+0.001 has the same
# Ø1.9 pilot + cone + raised ring boss as the near drive holes) — except the kit drives that
# side from INSIDE, so the far ring boss is CAPPED at Ø1.9 where the near one is open Ø3.9.
# Per Haoxiang: drill the far line so both faces behave identically, screws driven from
# outside on both. The cap is drilled in the VISUAL mesh too (precedent: the modeled screws
# were stripped from visuals) — the carried screw must be SEEN to enter; everything else is
# collision-only. Cut-wall lore that shaped the radii: chords of sparse straddler faces dip to
# r*cos(theta/2), and removing whole regions poisons the open mesh's SDF (a full boss removal
# once stalled the servo insertion mid-corridor) — drill modest bores, project the verts.
# THE FAR HOLES MIRROR THE NEAR ONES (measured: the far pin line at link y=+0.001 has the same
# Ø1.9 pilot + cone + raised ring boss as the near drive holes) — except the kit drives that
# side from INSIDE, so the far ring boss is CAPPED at Ø1.9 where the near one is open Ø3.9.
# Per Haoxiang: drill the far line so both faces behave identically, screws driven from outside
# on both. The catch needs no holding on either side: near, the head rests in the countersink
# cone; far, the shaft rests on the servo's registration POST (the mirror of the near pin,
# kept intact in the motor collision — a dropped screw measured rock-stable on it).
# The far cap is drilled in the VISUAL mesh too (precedent: the modeled screws were stripped
# from visuals) — the head must be SEEN to pass; everything else stays collision-only.
# The far screws drive into the VISIBLE holes — the Ø3.2 openings of the tab-line ring bosses
# (a top-down render proved the pin line is visually solid plate on the far face; the eye must
# see the screw enter a real hole). The carried screw needs only bore clearance: the rim ring
# opened to r2.6 for the Ø4 head (visuals to r2.1 — a slight visible widening toward the near
# ring's Ø3.9 look), and a shaft bore through the boss layer below. The head seats INSIDE the
# ring, tip stopping just above the servo's far tab.
SHELL_FAR_CUTS = [((44.0, 47.0), 2.6),  # (stage x zone, r): the rim ring bore, link z 41.5..44.5
                  ((35.5, 42.0), 1.5)]  # the boss shaft bore, link z 33..39.5
SHELL_FAR_VISUAL_CUTS = [((44.0, 47.0), 2.1)]  # the visible rim opening, Ø3.2 -> Ø4.2
# NEAR pilot drilled to r 1.25 (collision-only): the stock Ø1.9 pilot vs the Ø2 shaft is a
# sub-voxel interference wedge — a dropped screw perches on it CHAOTICALLY (bake-dependent).
# r 1.25 gives the shaft real clearance (0.25) while GUIDING it coaxially onto the pin top
# (±0.25 mm on a Ø1.74 disc — it cannot slide off), and the Ø4 head cannot enter the hole
# (at r 1.5 the head wedged INTO the drilled bore). Same axes as the pins.
SHELL_NEAR_CUTS = [((2.5, 5.7), 1.45)]  # (stage x zone, r): pilot, link z 0..3.2 — real
# clearance for the driven shaft (r 1.0 + offset), no entry for the Ø4 head; the visual pilot
# stays the printed Ø1.9.
MOTOR_FAR_POCKET_CUTS: list = []  # no motor cuts on the far line — the far POSTS are the catch
# --- the elbow-horn interface: the lower_arm fork clips over the seated servo -------------------
# Every seated fit between the fork and the servo measured SUB-VOXEL (SDF ~0.26 mm voxels,
# 0.15 mm contact offsets): output-spline OD r2.85 vs fork hub bore r3.0; back-post top r3.15 vs
# far-plate hole r3.0; horn disc OD r9.94 vs fork rim ID r10.1; M3 shaft r1.5 vs fork skin bore
# r1.6. Each would phantom-contact-fight the enabled lower_arm weld (the far-seat z=0.0450
# lesson: weld-vs-contact fights jitter the assembly). Collision-only cuts, minimal zones, in
# LINK-frame mm (the frame all the horn measurements were taken in); visuals untouched — the
# fork envelopes every cut feature, nothing is visible.
MOTOR_HORN_AXIS = (-112.57, -28.0)  # the elbow axis in the upper_arm link frame (x, y), mm
MOTOR_HORN_CUTS = [((-3.0, -0.05), 2.95),  # the output-spline protrusion (the fork hub takes it)
                   ((34.4, 38.0), 6.2)]    # the back-post ring top + stub: trimmed 1.6 mm below
# the far-plate plane so the fork can SLIDE ON along the servo axis (the clevis attach) — the
# ring top is coplanar with the sliding plate otherwise. The far M3 hole layers sit at r 7.0
# from the elbow axis, outside this trim. The seated plate hovers contact-free (the weld holds).
LOWER_ARM_AXIS = (0.0, 0.0)  # the elbow axis in the lower_arm link frame (= the link origin)
LOWER_ARM_CUTS = [((-1.7, 0.1), 4.4),      # the hub TUBE below the cup — it sweeps through the
                  # horn-disc plane during the slide-on attach (and the spline it once located
                  # is trimmed; the weld locates the seated fork)
                  ((0.55, 2.3), 11.0)]     # the rim ring that closes around the horn disc —
# removed outright (thin locating ring; it blocks any lateral/axial entry of the disc)
# The FOUR peripheral M3 screw lines (the real SO101 horn fastening; the center bore is only
# driver access). Axes measured from the horn's Ø2.46 hole ring; the printed Ø2.96 pilots take
# the Ø3 shaft (self-tap interference in the kit) and the metal Ø2.46 layers take the tip —
# both sub-voxel for rigid bodies, so the COLLISION bores open to r 2.0 (visuals untouched).
MOTOR_M3_AXES = [(-107.60, -23.03), (-107.59, -32.95),  # (x, y) in the upper_arm link frame
                 (-117.52, -23.02), (-117.52, -32.90)]
MOTOR_M3_CUTS = [((-0.3, 3.3), 2.0),    # the horn's two hole layers (near side)
                 ((33.7, 36.7), 2.0)]   # the case-back's two hole layers (far side)
LOWER_ARM_M3_AXES = [(4.97, -4.97), (-4.95, -4.98),  # the same lines in the lower_arm frame
                     (4.98, 4.95), (-4.90, 4.95)]
LOWER_ARM_M3_CUTS = [((-3.9, 0.9), 2.0),    # inner-plate pilot + recess-floor pilot (near)
                     ((35.7, 40.3), 2.0)]   # the far plate's two pilot layers


def ensure_source() -> None:
    """Fetch the official NVIDIA asset into source/ if it isn't there (build-time only dep)."""
    for rel in SOURCE_FILES:
        dst = SOURCE.parent / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[source] downloading {rel} ...", flush=True)
        urllib.request.urlretrieve(f"{SOURCE_URL}/{rel}", dst)


def new_asset_stage(root_name: str) -> tuple[Usd.Stage, Usd.Prim]:
    """In-memory stage whose default prim references the vendored robot. The overrides authored on
    it are baked away by the flattening Export at the end of each build_*."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}").GetPrim()
    root.GetReferences().AddReference(str(SOURCE))  # resolves to source defaultPrim (so101_new_calib)
    stage.SetDefaultPrim(root)
    return stage, root


def export_standalone(stage: Usd.Stage, path: Path) -> None:
    """Flatten every composition arc (source reference, instancing, sublayers) into one
    self-contained crate file. Inactive prims are dropped, which auto-prunes the deactivated
    links/parts and any meshes only they used."""
    if path.exists():
        path.unlink()
    stage.Export(str(path))


def deactivate(stage: Usd.Stage, root: Usd.Prim, names: list[str], parent: str = "") -> None:
    for name in names:
        prim = stage.GetPrimAtPath(root.GetPath().AppendPath(f"{parent}{name}"))
        assert prim, f"missing prim {parent}{name}"
        prim.SetActive(False)


def strip_part(stage: Usd.Stage, root: Usd.Prim, link: str, part: str) -> None:
    """Deactivate one physical part inside a link's instanceable visuals + collisions."""
    for scope in ("visuals", "collisions"):
        holder = stage.GetPrimAtPath(root.GetPath().AppendPath(f"{link}/{scope}"))
        assert holder, f"missing {link}/{scope}"
        holder.SetInstanceable(False)  # instances are read-only; de-instance to override a child
        child = stage.GetPrimAtPath(holder.GetPath().AppendChild(part))
        assert child, f"missing {link}/{scope}/{part}"
        child.SetActive(False)


def build_proximal(path: Path) -> None:
    stage, root = new_asset_stage("so101_proximal")
    deactivate(stage, root, sorted(DISTAL_LINKS))
    deactivate(stage, root, sorted(set(JOINTS) - PROXIMAL_JOINTS), parent="joints/")
    strip_part(stage, root, "upper_arm", MOTOR_PART)  # empty elbow pocket
    # root_joint (fixed-base pin + ArticulationRootAPI) is kept as-is.
    export_standalone(stage, path)
    print(f"[proximal] wrote {path}", flush=True)


def build_distal(path: Path) -> None:
    stage, root = new_asset_stage("so101_distal")
    deactivate(stage, root, sorted(PROXIMAL_LINKS))
    deactivate(stage, root, sorted(set(JOINTS) - DISTAL_JOINTS), parent="joints/")
    deactivate(stage, root, ["root_joint"])  # drops the world pin AND its ArticulationRootAPI
    lower_arm = stage.GetPrimAtPath(root.GetPath().AppendChild("lower_arm"))
    UsdPhysics.ArticulationRootAPI.Apply(lower_arm)  # floating-base articulation rooted here
    for scope in ("visuals", "collisions"):  # de-instance so harden_distal can edit the meshes
        holder = stage.GetPrimAtPath(root.GetPath().AppendPath(f"lower_arm/{scope}"))
        assert holder, f"missing lower_arm/{scope}"
        holder.SetInstanceable(False)
    export_standalone(stage, path)
    print(f"[distal] wrote {path}", flush=True)


def harden_distal(path: Path) -> None:
    """Give the (not-yet-tested) distal half the SAME collision hygiene the debugged parts got:
    vertex-weld the STL soups, SDF colliders (512 for printed parts that will mate with the
    motor horn, 256 for the servo cases), and explicit tight contact offsets. Without this, its
    future tests would re-hit every gotcha the proximal/motor debugging already paid for."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim().GetName()
    holder = stage.GetPrimAtPath(f"/{root}/lower_arm/collisions")
    assert holder, "missing lower_arm collisions"
    for child in holder.GetChildren():
        prim = child.GetChild("mesh")
        if not prim or prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get())
        fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        uniq, inv = np.unique(pts.round(6), axis=0, return_inverse=True)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(uniq.astype(np.float32)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(inv[fvi].astype(np.int32)))
        res = 256 if "sts3215" in child.GetName() else 512
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
        prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
        prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(res)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        print(f"[distal] {child.GetName()}: welded {len(pts)} -> {len(uniq)} verts, SDF {res}",
              flush=True)
    stage.GetRootLayer().Save()


def build_proximal_free(path: Path) -> None:
    """The proximal stub as a FLOATING-base articulation (same surgery as build_distal): no
    world pin, ArticulationRootAPI re-rooted on `base`. A scene can then hold it kinematically
    (a fixture/hand "grasp" via root-pose writes) and RELEASE it — a baked fixed base cannot be
    freed at runtime (PhysX consumes the root joint as the fixed-base flag at parse)."""
    stage, root = new_asset_stage("so101_proximal_free")
    deactivate(stage, root, sorted(DISTAL_LINKS))
    deactivate(stage, root, sorted(set(JOINTS) - PROXIMAL_JOINTS), parent="joints/")
    strip_part(stage, root, "upper_arm", MOTOR_PART)  # empty elbow pocket
    deactivate(stage, root, ["root_joint"])
    base = stage.GetPrimAtPath(root.GetPath().AppendChild("base"))
    UsdPhysics.ArticulationRootAPI.Apply(base)
    export_standalone(stage, path)
    print(f"[proximal_free] wrote {path}", flush=True)


def build_motor(path: Path) -> None:
    stage, root = new_asset_stage("sts3215_03a")
    deactivate(stage, root, sorted(set(LINKS) - {"upper_arm"}))
    deactivate(stage, root, ["joints", "root_joint"])  # plain free rigid body, no articulation
    strip_part(stage, root, "upper_arm", SHELL_PART)  # keep only the servo meshes
    link = stage.GetPrimAtPath(root.GetPath().AppendChild("upper_arm"))
    # URDF mass properties lump shell+motor; override with the bare servo: real mass, CoM at the
    # motor mesh center, and an EXPLICIT box-approximation inertia about the CoM. (Blocking the
    # inertia so PhysX derives it from the collider does NOT work here: the stripped triangle-soup
    # convex decomposition yields an invalid tensor — "negative diagonal value" at parse — and the
    # body NaNs on the first step.)
    motor_vis = stage.GetPrimAtPath(link.GetPath().AppendPath(f"visuals/{MOTOR_PART}"))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(motor_vis).ComputeAlignedRange()
    center_world = (Gf.Vec3d(box.GetMin()) + Gf.Vec3d(box.GetMax())) / 2.0
    to_link = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    com = to_link.Transform(center_world)
    lo, hi = Gf.Vec3d(box.GetMin()), Gf.Vec3d(box.GetMax())
    corners = [to_link.Transform(Gf.Vec3d(x, y, z))
               for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    dims = [max(p[i] for p in corners) - min(p[i] for p in corners) for i in range(3)]
    m = STS3215_MASS_KG
    mass = UsdPhysics.MassAPI(link)
    mass.GetMassAttr().Set(m)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(com))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(m / 12.0 * (dims[1] ** 2 + dims[2] ** 2),
                                               m / 12.0 * (dims[0] ** 2 + dims[2] ** 2),
                                               m / 12.0 * (dims[0] ** 2 + dims[1] ** 2)))
    mass.GetPrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # box aligned with the link axes
    export_standalone(stage, path)
    print(f"[motor] wrote {path}  (CoM in link frame: {Gf.Vec3f(com)})", flush=True)


def sdf_shell_pocket(path: Path) -> None:
    """Post-process the baked proximal: give the upper-arm SHELL collision an SDF collider. The
    stock convexDecomposition FILLS the elbow pocket with hulls, so the seated (SDF) servo is in
    deep interpenetration and gets depenetration-kicked out a few mm every step; with SDF the
    pocket cavity (and the shell's screw holes) are real to the solver. Same recipe as the motor:
    weld the STL-style soup first (raw-soup SDF cooking is pathologically slow/unreliable).
    Also DRILLS the screw-hole lines (both lines sit on the pin axes, link y=+0.001): the far
    ring-boss cap + wall pilot (SHELL_FAR_CUTS — the cap in the VISUAL mesh too, the head must
    be seen to pass) and the near pilot (SHELL_NEAR_CUTS). Idempotent — safe to re-run on an
    already-processed file."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim().GetName()  # so101_proximal or so101_proximal_free

    def drill(scope: str, axes_cuts) -> "Usd.Prim":
        """axes_cuts: list of (axes, cuts) pairs — each cuts list is drilled about its axes."""
        prim = stage.GetPrimAtPath(f"/{root}/upper_arm/{scope}/{SHELL_PART}/mesh")
        assert prim, f"missing shell {scope} mesh"
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get())
        fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        cache = UsdGeom.XformCache()
        xf = np.array(cache.GetLocalToWorldTransform(prim))
        w = (pts @ xf[:3, :3] + xf[3, :3]) * 1000.0  # world mm (the measurement frame)
        # a face is dropped only if fully inside ONE cylinder — testing against the UNION lets
        # a long triangle spanning two holes (one vertex in each) qualify and carves a slot
        # between them
        drop = np.zeros(len(fvi), bool)
        for axes, cuts in axes_cuts:
            for ay, az in axes:
                r = np.hypot(w[:, 1] - ay, w[:, 2] - az)
                for (x0, x1), rad in cuts:
                    ins = (r < rad) & (w[:, 0] > x0) & (w[:, 0] < x1)
                    drop |= ins[fvi].all(axis=1)
        keep = ~drop
        fvi = fvi[keep]
        nrm = mesh.GetNormalsAttr().Get()
        if nrm is not None and len(nrm):  # faceVarying: 3 per face
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(
                np.array(nrm).reshape(-1, 3, 3)[keep].reshape(-1, 3).astype(np.float32)))
        used = np.unique(fvi.ravel())
        remap = np.full(len(pts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        pts, fvi = pts[used], remap[fvi]
        # project surviving in-cylinder vertices onto the cut wall (see strip_motor_screws:
        # shards from straddling faces otherwise protrude into the hole and grind the screw)
        w = (pts @ xf[:3, :3] + xf[3, :3]) * 1000.0
        for axes, cuts in axes_cuts:
            for ay, az in axes:
                for (x0, x1), rad in cuts:
                    dy, dz = w[:, 1] - ay, w[:, 2] - az
                    r = np.hypot(dy, dz)
                    ins = (r < rad) & (w[:, 0] > x0) & (w[:, 0] < x1)
                    if ins.any():
                        s = rad / np.maximum(r[ins], 1e-6)
                        w[ins, 1] = ay + dy[ins] * s
                        w[ins, 2] = az + dz[ins] * s
        pts = ((w / 1000.0 - xf[3, :3]) @ np.linalg.inv(xf[:3, :3])).astype(pts.dtype)
        uniq, inv = np.unique(pts.round(6), axis=0, return_inverse=True)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(uniq.astype(np.float32)))
        mesh.GetFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(inv[fvi].astype(np.int32).ravel()))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(fvi), 3, np.int32)))
        lo, hi = uniq.min(0), uniq.max(0)
        mesh.GetExtentAttr().Set([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])
        print(f"[proximal] shell {scope}: drilled {int((~keep).sum())} faces, "
              f"{len(pts)} -> {len(uniq)} verts", flush=True)
        return prim

    drill("visuals", [(MOTOR_SCREW_AXES_MM, SHELL_FAR_VISUAL_CUTS)])
    prim = drill("collisions", [(MOTOR_SCREW_AXES_MM, SHELL_FAR_CUTS),
                                (MOTOR_PIN_AXES_MM, SHELL_NEAR_CUTS)])
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
    prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
    # 512 over the ~110 mm shell ≈ 0.21 mm voxels: the servo's tabs slide through a channel
    # under the pocket wall with <0.5 mm design clearance for the last 18 mm of insertion — at
    # 256 (0.43 mm) that channel is voxel-pinched shut and the servo jams ~17 mm short.
    prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(512)
    # Explicit TIGHT offsets: unset, PhysX auto-computes mm-scale contact offsets on meshes this
    # size and generates phantom contacts wider than the pocket's clearances — the servo then
    # "collides" mid-corridor where the meshes are measurably clear. NOTE: the attributes are
    # IGNORED unless PhysxCollisionAPI is in the applied schemas.
    prim.AddAppliedSchema("PhysxCollisionAPI")
    # 0.15 mm ~= the SDF voxel: bigger offsets manufacture phantom contacts across every
    # sub-offset clearance (screw heads in bores, shafts in pilots). Fast-drop tunneling, the
    # original reason for 0.5, is handled by CCD on the screws (scene-side) instead.
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.00015)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()


def strip_motor_screws(path: Path) -> None:
    """Post-process the baked (flattened) motor file: cut the two modeled M2 tab screws out of
    the visual AND collision meshes (triangle soup with faceVarying normals, no UVs)."""
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    for scope in ("visuals", "collisions"):
        prim = stage.GetPrimAtPath(f"/sts3215_03a/upper_arm/{scope}/{MOTOR_PART}/mesh")
        assert prim, f"missing {scope} mesh"
        mesh = UsdGeom.Mesh(prim)
        pts_local = np.array(mesh.GetPointsAttr().Get())
        xf = np.array(cache.GetLocalToWorldTransform(prim))
        w = (pts_local @ xf[:3, :3] + xf[3, :3]) * 1000.0  # world mm (the measurement frame)
        fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        in_screw = np.zeros(len(w), bool)
        for ay, az in MOTOR_SCREW_AXES_MM:
            r = np.hypot(w[:, 1] - ay, w[:, 2] - az)
            for (x0, x1), rad in MOTOR_SCREW_CUTS:
                in_screw |= (r < rad) & (w[:, 0] > x0) & (w[:, 0] < x1)
        if scope == "collisions":  # pins: see MOTOR_PIN_CUTS — collision-only, visuals keep them
            for ay, az in MOTOR_PIN_AXES_MM:
                r = np.hypot(w[:, 1] - ay, w[:, 2] - az)
                for (x0, x1), rad in MOTOR_PIN_CUTS:
                    in_screw |= (r < rad) & (w[:, 0] > x0) & (w[:, 0] < x1)
        drop = in_screw[fvi].all(axis=1)  # only faces fully inside a cut cylinder
        keep = ~drop
        fvi = fvi[keep]
        used = np.unique(fvi.ravel())
        remap = np.full(len(pts_local), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        new_pts = pts_local[used].astype(np.float32)
        new_fvi = remap[fvi].astype(np.int32)
        nrm = np.array(mesh.GetNormalsAttr().Get()).reshape(-1, 3, 3)[keep]  # faceVarying: 3/face
        if scope == "collisions":
            # Weld the STL-style soup (every face has private duplicate verts) into a connected
            # mesh: PhysX SDF cooking on the raw soup is pathologically slow (tens of minutes) and
            # its inside/outside classification is unreliable. Exact-duplicate merge at 1 um.
            uniq, inv = np.unique(new_pts.round(6), axis=0, return_inverse=True)
            new_pts, new_fvi = uniq.astype(np.float32), inv[new_fvi].astype(np.int32)
            # COLLISION-ONLY uniform shrink (per Haoxiang): the pocket is a snap-fit that works
            # by print flex; rigid bodies need CLEARANCE instead. 3% about the case centroid
            # (~0.6 mm/side) turns the interference fit into a slip fit: the servo inserts along
            # a straight line and the seat no longer depenetration-ejects it. Visuals stay exact.
            # The registration PINS and POSTS are EXEMPT: they are the screw-drop catches (the
            # tips rest on them) and must sit exactly on the visual hole line — the centroid
            # shrink recenters them ~1.2 mm off it and the falling tip slips past. Their own
            # clearance comes from the DRILLED shell holes (SHELL_NEAR/FAR_CUTS), not shrinking.
            ctr = new_pts.mean(0)
            shrunk = (ctr + MOTOR_COLLISION_SHRINK * (new_pts - ctr)).astype(np.float32)
            w0 = (new_pts.astype(np.float64) @ xf[:3, :3] + xf[3, :3]) * 1000.0
            pin = np.zeros(len(w0), bool)
            for ay, az in MOTOR_PIN_AXES_MM:
                r0 = np.hypot(w0[:, 1] - ay, w0[:, 2] - az)
                pin |= (r0 < 1.45) & (w0[:, 0] > 3.0) & (w0[:, 0] < 45.0)
            new_pts = np.where(pin[:, None], new_pts, shrunk).astype(np.float32)
            print(f"[motor] collisions: {int(pin.sum())} pin/post verts exempt from the shrink",
                  flush=True)
            # FAR-TAB head pocket + shaft passage, cut AFTER the shrink about the REAL tab axes
            # (see MOTOR_FAR_POCKET_CUTS): evaluated on the shrunk points in world coordinates,
            # so the pocket lands where the (unshrunk) screw actually travels.
            w2 = (new_pts.astype(np.float64) @ xf[:3, :3] + xf[3, :3]) * 1000.0
            in_far = np.zeros(len(w2), bool)
            for ay, az in MOTOR_SCREW_AXES_MM:
                r2 = np.hypot(w2[:, 1] - ay, w2[:, 2] - az)
                for (x0, x1), rad in MOTOR_FAR_POCKET_CUTS:
                    in_far |= (r2 < rad) & (w2[:, 0] > x0) & (w2[:, 0] < x1)
            keep2 = ~in_far[new_fvi].all(axis=1)
            tri = new_fvi[keep2]
            nrm = nrm[keep2]
            used2 = np.unique(tri.ravel())
            remap2 = np.full(len(new_pts), -1, dtype=np.int64)
            remap2[used2] = np.arange(len(used2))
            new_pts, new_fvi = new_pts[used2], remap2[tri].astype(np.int32)
            # PROJECT surviving in-cylinder vertices onto the cut wall: face-dropping alone
            # leaves straddling-face shards protruding ~a triangle into the hole, which eats the
            # 0.5 mm clearance and randomly grinds the driven screw. Projection makes the wall
            # exact (collision-only; the mesh deforms sub-mm at the rims).
            w3 = (new_pts.astype(np.float64) @ xf[:3, :3] + xf[3, :3]) * 1000.0
            moved = 0
            for ay, az in MOTOR_SCREW_AXES_MM:
                for (x0, x1), rad in MOTOR_FAR_POCKET_CUTS:
                    dy, dz = w3[:, 1] - ay, w3[:, 2] - az
                    r3 = np.hypot(dy, dz)
                    inside = (r3 < rad) & (w3[:, 0] > x0) & (w3[:, 0] < x1)
                    if inside.any():
                        s = rad / np.maximum(r3[inside], 1e-6)
                        w3[inside, 1] = ay + dy[inside] * s
                        w3[inside, 2] = az + dz[inside] * s
                        moved += int(inside.sum())
            new_pts = (((w3 / 1000.0) - xf[3, :3]) @ np.linalg.inv(xf[:3, :3])).astype(np.float32)
            print(f"[motor] collisions: welded, shrunk x{MOTOR_COLLISION_SHRINK}, far pocket "
                  f"cut {int((~keep2).sum())} faces, projected {moved} shard verts "
                  f"-> {len(new_pts)} verts", flush=True)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_pts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(new_fvi.ravel()))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(new_fvi), 3, np.int32)))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(nrm.reshape(-1, 3).astype(np.float32)))
        lo, hi = new_pts.min(0), new_pts.max(0)
        mesh.GetExtentAttr().Set([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])
        if scope == "collisions":
            # SDF collision (the factory nut/bolt approach): the emptied Ø3.5 tab holes must be
            # REAL to the solver — a convex decomposition seals or misses them, so a dropped screw
            # either rests on phantom geometry or falls straight through. With SDF the shaft
            # enters and the Ø4.7 head catches on the rim. 160 over the 45 mm case ≈ 0.28 mm
            # voxels (~12 across the hole) — resolution is capped to keep the one-time cook short.
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("sdf")
            prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            prim.CreateAttribute("physxSDFMeshCollision:sdfResolution",
                                 Sdf.ValueTypeNames.Int).Set(256)  # tab-channel fit needs <0.3 mm
            prim.AddAppliedSchema("PhysxCollisionAPI")  # required or the attrs are ignored
            prim.CreateAttribute("physxCollision:contactOffset",
                                 Sdf.ValueTypeNames.Float).Set(0.00015)  # ~voxel; CCD on the
            # screws handles fast-drop tunneling (bigger offsets manufacture phantom contacts
            # across every sub-offset clearance)
            prim.CreateAttribute("physxCollision:restOffset",
                                 Sdf.ValueTypeNames.Float).Set(0.0)  # contacts from auto-offsets
        print(f"[motor] {scope}: stripped {drop.sum()} screw faces -> {keep.sum()} faces", flush=True)
    stage.GetRootLayer().Save()


def cut_link_cylinders(path: Path, link: str, part: str, axes_cuts: list,
                       spare: tuple | None = None) -> None:
    """Drop COLLISION faces fully inside link-frame cylinders and project surviving in-cylinder
    verts onto the cut wall — the same face-drop + shard-projection recipe as
    drill()/strip_motor_screws, but framed in the LINK frame so the zones read exactly like the
    probe measurements. `axes_cuts`: list of (axes, cuts) pairs — each cuts list ((z0, z1), r)
    in mm is applied about each of its axes (x, y). `spare`: optional (axes, radius) — verts
    within `radius` of any spare axis are exempt from every cut (protects e.g. drive-bore
    walls that a broad trim zone would gash open). Idempotent — safe to re-run."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim().GetName()
    link_prim = stage.GetPrimAtPath(f"/{root}/{link}")
    assert link_prim, f"missing link {link}"
    lxf = np.array(UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    prim = stage.GetPrimAtPath(f"/{root}/{link}/collisions/{part}/mesh")
    assert prim, f"missing {link} collision mesh {part}"
    mesh = UsdGeom.Mesh(prim)
    cache = UsdGeom.XformCache()
    xf = np.array(cache.GetLocalToWorldTransform(prim))
    pts = np.array(mesh.GetPointsAttr().Get())
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)

    def to_link_mm(p: np.ndarray) -> np.ndarray:
        w = p.astype(np.float64) @ xf[:3, :3] + xf[3, :3]
        return (w - lxf[3, :3]) @ np.linalg.inv(lxf[:3, :3]) * 1000.0

    def spared(P):
        if spare is None:
            return np.zeros(len(P), bool)
        s_axes, s_rad = spare
        m = np.zeros(len(P), bool)
        for ax, ay in s_axes:
            m |= np.hypot(P[:, 0] - ax, P[:, 1] - ay) < s_rad
        return m

    lk = to_link_mm(pts)
    ins = np.zeros(len(lk), bool)
    for axes, cuts in axes_cuts:
        for ax, ay in axes:
            r = np.hypot(lk[:, 0] - ax, lk[:, 1] - ay)
            for (z0, z1), rad in cuts:
                ins |= (r < rad) & (lk[:, 2] > z0) & (lk[:, 2] < z1)
    ins &= ~spared(lk)
    drop = ins[fvi].all(axis=1)
    keep = ~drop
    fvi = fvi[keep]
    nrm = mesh.GetNormalsAttr().Get()
    if nrm is not None and len(nrm):  # faceVarying: 3 per face
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(
            np.array(nrm).reshape(-1, 3, 3)[keep].reshape(-1, 3).astype(np.float32)))
    used = np.unique(fvi.ravel())
    remap = np.full(len(pts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts, fvi = pts[used], remap[fvi]
    lk = to_link_mm(pts)
    keep_out = spared(lk)
    moved = 0
    for axes, cuts in axes_cuts:  # project shards onto the cut walls
        for ax, ay in axes:
            for (z0, z1), rad in cuts:
                dx, dy = lk[:, 0] - ax, lk[:, 1] - ay
                rr = np.hypot(dx, dy)
                inside = (rr < rad) & (lk[:, 2] > z0) & (lk[:, 2] < z1) & ~keep_out
                if inside.any():
                    s = rad / np.maximum(rr[inside], 1e-6)
                    lk[inside, 0] = ax + dx[inside] * s
                    lk[inside, 1] = ay + dy[inside] * s
                    moved += int(inside.sum())
    w = (lk / 1000.0) @ lxf[:3, :3] + lxf[3, :3]
    pts = ((w - xf[3, :3]) @ np.linalg.inv(xf[:3, :3])).astype(np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(fvi.astype(np.int32).ravel()))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(fvi), 3, np.int32)))
    lo, hi = pts.min(0), pts.max(0)
    mesh.GetExtentAttr().Set([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])
    print(f"[{link}] cut {int(drop.sum())} faces, projected {moved} shard verts "
          f"({path.name}:{part})", flush=True)
    stage.GetRootLayer().Save()


def tighten_distal_offsets(path: Path) -> None:
    """0.5 mm contact offsets on the lower_arm colliders predate the phantom-contact finding:
    at the horn every clearance is sub-half-millimeter, so 0.5 manufactures permanent contacts.
    0.15 mm ~= the SDF voxel, same as the shell/motor."""
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim().GetName()
    for child in stage.GetPrimAtPath(f"/{root}/lower_arm/collisions").GetChildren():
        prim = child.GetChild("mesh")
        if not prim or prim.GetTypeName() != "Mesh":
            continue
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.00015)
    stage.GetRootLayer().Save()
    print(f"[distal] lower_arm contact offsets -> 0.15 mm", flush=True)


def inspect(path: Path, label: str) -> None:
    """Open the STANDALONE file (no source needed) and print bodies / live joints / physics APIs +
    a mesh count, to confirm both the recipe and the self-containedness."""
    stage = Usd.Stage.Open(str(path))
    meshes = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    print(f"\n===== {label}: {path.name}  ({path.stat().st_size / 1e6:.1f} MB, {meshes} meshes, "
          f"defaultPrim={stage.GetDefaultPrim().GetPath()}) =====", flush=True)
    for prim in stage.Traverse():
        apis = [a for a in ("ArticulationRootAPI", "RigidBodyAPI") if prim.HasAPI(getattr(UsdPhysics, a))]
        j = UsdPhysics.Joint(prim)
        if j:
            print(f"  JOINT {prim.GetPath()}  body0={list(j.GetBody0Rel().GetTargets())} "
                  f"body1={list(j.GetBody1Rel().GetTargets())}", flush=True)
        elif apis:
            mass = UsdPhysics.MassAPI(prim).GetMassAttr()
            info = f" mass={mass.Get()}" if mass and mass.HasAuthoredValue() else ""
            print(f"  BODY  {prim.GetPath()} APIs={apis}{info}", flush=True)
        elif prim.GetName() in ("visuals", "collisions") and not prim.IsInstanceable():
            kids = [(c.GetName(), c.IsActive()) for c in prim.GetChildren()]
            print(f"  PARTS {prim.GetPath()} -> {kids}", flush=True)


def main() -> None:
    ensure_source()
    build_proximal(SO101 / "so101_proximal.usd")
    sdf_shell_pocket(SO101 / "so101_proximal.usd")
    build_proximal_free(SO101 / "so101_proximal_free.usd")
    sdf_shell_pocket(SO101 / "so101_proximal_free.usd")
    build_distal(SO101 / "so101_distal.usd")
    harden_distal(SO101 / "so101_distal.usd")
    cut_link_cylinders(SO101 / "so101_distal.usd", "lower_arm", "under_arm_so101_v1",
                       [([LOWER_ARM_AXIS], LOWER_ARM_CUTS),
                        (LOWER_ARM_M3_AXES, LOWER_ARM_M3_CUTS)])
    tighten_distal_offsets(SO101 / "so101_distal.usd")
    build_motor(SO101 / "sts3215_03a.usd")
    strip_motor_screws(SO101 / "sts3215_03a.usd")
    cut_link_cylinders(SO101 / "sts3215_03a.usd", "upper_arm", MOTOR_PART,
                       [([MOTOR_HORN_AXIS], MOTOR_HORN_CUTS),
                        (MOTOR_M3_AXES, MOTOR_M3_CUTS)],
                       spare=(MOTOR_M3_AXES, 2.4))  # the back-ring trim must not gash the
    # far M3 drive-bore walls (they overlap its radius; a gashed wall SDF ejects the screw)
    inspect(SO101 / "so101_proximal.usd", "PROXIMAL (base+shoulder+upper_arm, joints 1-2, empty elbow pocket)")
    inspect(SO101 / "so101_distal.usd", "DISTAL (lower_arm..jaw, joints 4-5+gripper, floating)")
    inspect(SO101 / "sts3215_03a.usd", "MOTOR (free rigid body)")


if __name__ == "__main__":
    main()
    app.close()
