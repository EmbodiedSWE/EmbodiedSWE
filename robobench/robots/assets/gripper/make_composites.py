"""Write arm+gripper composite files into `assets/composites/` (same folder convention as the
composites already there).

A composite is a small usda that references a bare arm and a gripper USD posed so the gripper's
mount prim lands exactly on the arm's flange rest frame, welded with one fixed joint. The mount
pose is computed from the two stages' authored rest transforms — nothing is hand-typed.

`ARMS` describes arms (USD, root prim, flange body, how to make it bare). `GRIPPERS` describes
gripper assets (USD, mount prim, optional cleanup and mount corrections). `BUILD` lists which
pairs to bake. To weld any gripper onto a new arm: add an `ARMS` entry and a pair — then point
the arm's robot class at `composites/<arm>_<gripper>/`.

    python robobench/robots/assets/gripper/make_composites.py

Run with a plain `usd-core` Python (no AppLauncher needed).
"""

from __future__ import annotations

import os
import posixpath

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
COMPOSITES = os.path.join(ASSETS, "composites")

#: usd (under assets/), root prim, flange body (the weld target, relative to the root prim),
#: variants to select (e.g. to strip a built-in gripper), flange_pos (the mount FACE in the
#: flange body's frame, when it is offset from the body origin)
ARMS = {
    "xarm7": dict(usd="xarm7/xarm7.usd", root="/UF_ROBOT", flange="link7",
                  variants={"Variant_Set": "None"}),
    # right_j6 spins about l6's +z (= the tool axis; the authored right_hand tool frame
    # agrees); the flange face sits at +z 24.5 mm from the l6 body origin
    "sawyer": dict(usd="sawyer/sawyer_instanceable.usd", root="/sawyer", flange="right_l6",
                   flange_pos=(0.0, 0.0, 0.0245)),
}

#: usd (under assets/) and attach_prim (the mount body, relative to the USD's default prim).
#: Optional keys for assets that need corrections: mount_pos (the mount POINT in the attach
#: prim's frame, when the mounting face is not at its origin), mount_quat (wxyz: the gripper's
#: axes in the flange frame, when its fingers do not point along its own +z), api_prim (child
#: prim whose articulation-root APIs must be stripped; by default they are stripped from the
#: referenced root), deactivate (prims to switch off), add_linear_drives (prismatic joints that
#: ship without DriveAPI — PhysX creates no drive at parse time and gain writes move nothing),
#: pad_boxes ((body, translate, half-extents) collision plates for unusable fingertip geometry).
GRIPPERS = {
    "panda_hand": dict(usd="panda_hand/panda_hand.usd", attach_prim="panda_rig/panda_hand"),
}

#: (arm, gripper, composite name)
BUILD = (("xarm7", "panda_hand", "xarm7_panda_hand"), ("sawyer", "panda_hand", "sawyer_panda"))


def make(arm_name: str, gripper_name: str, name: str) -> str:
    from pxr import Gf, Usd, UsdGeom

    arm, grip = ARMS[arm_name], GRIPPERS[gripper_name]
    arm_usd = os.path.join(ASSETS, *arm["usd"].split("/"))
    grip_usd = os.path.join(ASSETS, *grip["usd"].split("/"))

    # the flange's rest transform on the bare arm (variant selections restored after)
    stage = Usd.Stage.Open(arm_usd)
    root_prim = stage.GetPrimAtPath(arm["root"])
    prev = {}
    for vs_name, choice in arm.get("variants", {}).items():
        vs = root_prim.GetVariantSet(vs_name)
        prev[vs_name] = vs.GetVariantSelection()
        vs.SetVariantSelection(choice)
    try:
        t_flange = UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath(f"{arm['root']}/{arm['flange']}"))
    finally:
        for vs_name, choice in prev.items():
            root_prim.GetVariantSet(vs_name).SetVariantSelection(choice)

    # pose /gripper so the mount POINT lands on the flange: T = (M_shift*T_mount)^-1 * R * T_flange
    # (row-vector matrices)
    gstage = Usd.Stage.Open(grip_usd)
    groot = gstage.GetDefaultPrim()
    t_mount = UsdGeom.XformCache().GetLocalToWorldTransform(
        gstage.GetPrimAtPath(f"{groot.GetPath()}/{grip['attach_prim']}"))
    mq = arm.get("mount_quat") or grip.get("mount_quat") or (1.0, 0.0, 0.0, 0.0)
    mp = grip.get("mount_pos", (0.0, 0.0, 0.0))
    fp = arm.get("flange_pos", (0.0, 0.0, 0.0))
    r_mount = Gf.Matrix4d().SetRotate(Gf.Quatd(mq[0], mq[1], mq[2], mq[3]))
    m_shift = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*mp))
    m_fshift = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*fp))
    tg = (m_shift * t_mount).GetInverse() * r_mount * m_fshift * t_flange
    t = tg.ExtractTranslation()
    q = tg.ExtractRotationQuat()

    out_dir = os.path.join(COMPOSITES, name)
    rel_arm = posixpath.relpath(arm_usd.replace(os.sep, "/"), out_dir.replace(os.sep, "/"))
    rel_grip = posixpath.relpath(grip_usd.replace(os.sep, "/"), out_dir.replace(os.sep, "/"))
    variants = "".join(f'\n        string {k} = "{v}"' for k, v in arm.get("variants", {}).items())
    var_meta = f"\n    variants = {{{variants}\n    }}" if variants else ""
    deact = "".join(
        f'\n        over "{p}" (\n            active = false\n        )\n        {{\n        }}\n'
        for p in grip.get("deactivate", ()))
    deact += "".join(
        f'\n        over "{p}" (\n            prepend apiSchemas = ["PhysicsDriveAPI:linear"]\n'
        f'        )\n        {{\n            float drive:linear:physics:damping = 100\n'
        f'            float drive:linear:physics:maxForce = 185\n'
        f'            float drive:linear:physics:stiffness = 2000\n        }}\n'
        for p in grip.get("add_linear_drives", ()))
    for jaw, tr, he in grip.get("pad_boxes", ()):
        deact += (
            f'\n        over "{jaw}"\n        {{\n'
            f'            def Cube "pad_fix" (\n'
            f'                prepend apiSchemas = ["PhysicsCollisionAPI"]\n            )\n'
            f'            {{\n                double size = 2\n'
            f'                double3 xformOp:translate = ({tr[0]}, {tr[1]}, {tr[2]})\n'
            f'                double3 xformOp:scale = ({he[0]}, {he[1]}, {he[2]})\n'
            f'                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
            f'            }}\n        }}\n')
    api_del = '["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]'
    api_prim = grip.get("api_prim", "")
    root_meta = f"\n        delete apiSchemas = {api_del}" if not api_prim else ""
    child_over = ("" if not api_prim else
                  f'\n        over "{api_prim}" (\n            '
                  f'delete apiSchemas = {api_del}\n        )\n        {{\n        }}\n')
    text = f'''#usda 1.0
(
    doc = "GENERATED by gripper/make_composites.py — edit its tables and re-run, do not hand-edit."
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}" (
    prepend references = @{rel_arm}@{var_meta}
)
{{
    def Xform "gripper" (
        prepend references = @{rel_grip}@{root_meta}
    )
    {{
        quatd xformOp:orient = ({q.GetReal()}, {q.GetImaginary()[0]}, {q.GetImaginary()[1]}, {q.GetImaginary()[2]})
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = ({t[0]}, {t[1]}, {t[2]})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{child_over}{deact}    }}

    def PhysicsFixedJoint "gripper_attach"
    {{
        rel physics:body0 = </{name}/{arm["flange"]}>
        rel physics:body1 = </{name}/gripper/{grip["attach_prim"]}>
        point3f physics:localPos0 = ({fp[0]}, {fp[1]}, {fp[2]})
        point3f physics:localPos1 = ({mp[0]}, {mp[1]}, {mp[2]})
        quatf physics:localRot0 = ({mq[0]}, {mq[1]}, {mq[2]}, {mq[3]})
        quatf physics:localRot1 = (1, 0, 0, 0)
    }}
}}
'''
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.usda")
    with open(out, "w") as f:
        f.write(text)
    return out


def main() -> None:
    for arm_name, gripper_name, name in BUILD:
        print("wrote", make(arm_name, gripper_name, name))


if __name__ == "__main__":
    main()
