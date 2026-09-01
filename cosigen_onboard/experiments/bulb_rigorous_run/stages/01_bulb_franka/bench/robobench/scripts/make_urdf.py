"""Regenerate the vendored kinematics URDFs from the vendored robot USDs (Pink IK's Pinocchio model).

General build-time tooling: convert the *same USD the sim spawns* into a URDF via Isaac's
`convert_usd_to_urdf`, then strip `<visual>`/`<collision>` so the vendored `.urdf` is pure kinematics
(links + joints + inertials) — no mesh refs, so Pinocchio never needs the mesh files the converter also
emits (otherwise `RobotWrapper.BuildFromURDF` errors trying to load them). Generating off the sim USD
guarantees the IK model matches the simulated bodies, and adding a robot is one `SPECS` entry.

URDF link names are the USD prim names prefixed with the USD default-prim name (e.g. GR1T2's
`right_hand_pitch_link` -> `GR1T2_fourier_hand_6dof_right_hand_pitch_link`); each robot's
`build_controller` uses that prefix.

NOTE — G1 is intentionally absent: its vendored `g1.usd` nests the three-finger hand as a referenced
sub-asset the converter can't fold into one tree (it errors on a doubled `left_hand_left_hand_*` root),
so G1 keeps its pre-vendored `g1_29dof_with_hand_only_kinematics.urdf`. Only add a robot here if its USD
converts cleanly.

Needs the app (the converter pulls isaacsim) — run with the venv python:
    python -m robobench.scripts.make_urdf               # all listed robots
    python -m robobench.scripts.make_urdf --robot gr1t2  # one
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

ASSETS = Path(__file__).resolve().parent.parent / "robots" / "assets"
# robot -> (source USD, output kinematics URDF), both relative to robots/assets/.
SPECS = {
    "gr1t2": ("gr1t2/GR1T2_fourier_hand_6dof.usd", "gr1t2/GR1T2_fourier_hand_6dof_kinematics.urdf"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=sorted(SPECS), default=None, help="which robot (default: all)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.controllers.utils import convert_usd_to_urdf  # noqa: E402


def main() -> None:
    for r in [args.robot] if args.robot else list(SPECS):
        usd, urdf = (ASSETS / p for p in SPECS[r])
        with tempfile.TemporaryDirectory() as tmp:  # converter also emits meshes/ — discarded with tmp
            generated, _ = convert_usd_to_urdf(str(usd), tmp, force_conversion=True)
            tree = ET.parse(generated)
            for link in tree.getroot().findall("link"):  # drop geometry -> pure kinematics, no mesh deps
                for el in link.findall("visual") + link.findall("collision"):
                    link.remove(el)
            tree.write(urdf)
        print(f"{r}: vendored kinematics URDF -> {urdf}")
    os._exit(0)


if __name__ == "__main__":
    main()
