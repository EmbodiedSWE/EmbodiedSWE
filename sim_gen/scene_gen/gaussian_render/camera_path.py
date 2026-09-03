"""Camera paths for splat QA renders: a generic auto-framable orbit
(poses_orbit) plus the hand-tuned kitchen walk-in (poses) as a worked example.

Pure numpy so both the isaacsim venv and the 3dgrut venv import it unchanged.
World frame = the handoff frame (z-up, floor at z=-1.06, eye height z=+0.45).

Path (24 fps):
  A  walk-in   : 10 s translate along +x from the west end to the room center
                 with a gentle yaw sway (parallax visible),
  B  look-round:  8 s full 360 yaw at the center,
  C  approach  :  6 s walk up to the south counter, tilting down onto the countertop.
Poses are OpenCV cam-to-world (x right, y down, z forward).
"""
import math
import numpy as np

FPS = 24
W, H = 1280, 720
FOCAL_MM, APERTURE_MM = 12.0, 20.955          # 12mm on a 20.955mm aperture (~82 deg HFOV robot camera)
FX = W * FOCAL_MM / APERTURE_MM                # 733.0 px  (HFOV 82 deg)
FY = FX
CX, CY = W / 2, H / 2
EYE_Z = 0.45
FLOOR_Z = -1.06


def look_at_cv(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)                  # forward
    x = np.cross(z, up); x /= np.linalg.norm(x)               # right
    y = np.cross(z, x)                                        # down
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _smooth(t):  # ease in/out on [0,1]
    return 0.5 - 0.5 * math.cos(math.pi * t)


def poses():
    out = []
    nA, nB, nC = 10 * FPS, 8 * FPS, 6 * FPS
    p0, p1 = np.array([-6.0, -0.5, EYE_Z]), np.array([0.5, -0.3, EYE_Z])
    for i in range(nA):
        t = i / (nA - 1)
        eye = p0 + (p1 - p0) * _smooth(t)
        yaw = math.radians(18.0) * math.sin(2 * math.pi * t)          # sway
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        out.append(look_at_cv(eye, eye + d))
    for i in range(nB):
        t = i / nB
        yaw = 2 * math.pi * t
        pitch = math.radians(-5.0) + math.radians(4.0) * math.sin(2 * yaw)
        d = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        out.append(look_at_cv(p1, p1 + d))
    # south counter: front at y~-1.75, wall at y~-2.6 (measured from the splat cloud)
    p2 = np.array([0.5, -1.15, EYE_Z])
    tgt0, tgt1 = p1 + np.array([0.0, -4.0, -0.3]), np.array([0.5, -2.2, -0.25])
    for i in range(nC):
        t = _smooth(i / (nC - 1))
        eye = p1 + (p2 - p1) * t
        out.append(look_at_cv(eye, tgt0 + (tgt1 - tgt0) * t))
    return np.stack(out)


def poses_orbit(center_xy=(0.0, 0.0), radius=2.6, eye_z=EYE_Z, target_z=0.0, frames=480):
    """Generic 360-degree orbit around a point - works for any scene; pair with
    auto-framing from the splat cloud (see render_splat_walk.py --path orbit)."""
    cx, cy = center_xy
    out = []
    for i in range(frames):
        a = 2 * math.pi * i / frames
        z = eye_z + 0.08 * math.sin(4 * math.pi * i / frames)
        eye = np.array([cx + radius * math.cos(a), cy + radius * math.sin(a), z])
        out.append(look_at_cv(eye, np.array([cx, cy, target_z])))
    return np.stack(out)


def cv_to_usd(T):
    """OpenCV cam-to-world -> USD camera xform (USD cam looks down -z, +y up)."""
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    return T @ F


if __name__ == "__main__":
    P = poses()
    print(len(P), "frames", len(P) / FPS, "s; fx", FX)
    print("first eye", P[0, :3, 3], "last eye", P[-1, :3, 3])
