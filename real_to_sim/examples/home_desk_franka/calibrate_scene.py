#!/usr/bin/env python3
"""Step 4: calibrate the scene — metric scale + world frame.

Turns the (unitless) reconstruction into a metric, oriented scene package:

    python scripts/calibrate_scene.py <scene> --height 0.77 [--extents 0.57 0.43]

Scale pin (pick one):
  --height <m>   work-surface height above the floor  (RECOMMENDED — robust)
  --span <m> [--major]   a measured span of the surface (fallback; the
                 reconstructed surface cluster may under/over-cover reality)

World frame: surface plane = z0, origin at the surface center, x = surface
major axis, z = up. Writes <ws>/scene.json (canonical) plus calib.json /
composite_setup.json kept for the usage tools.

Sanity checks printed: floor depth (should equal --height), camera heights
(should look like a human holding a phone), surface extents (must be <= the
real surface; larger means the plane cluster merged neighboring furniture).
"""

import argparse
import glob
import json
import os
import pathlib
import sys

STAGE = pathlib.Path(__file__).resolve().parents[2] / "background"
VENV_PY = STAGE / ".venv" / "bin" / "python"
if VENV_PY.exists() and pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])

import numpy as np
import pycolmap


def fit_surface(rec):
    """Dominant near-camera horizontal-ish plane + its largest point cluster."""
    centers = {i: img.projection_center() for i, img in rec.images.items()}
    track_imgs = {}
    for i, img in rec.images.items():
        for p2d in img.points2D:
            if p2d.has_point3D():
                track_imgs.setdefault(p2d.point3D_id, []).append(i)
    pids, pdist = [], []
    for pid, p in rec.points3D.items():
        if pid in track_imgs and p.error < 2.0 and len(track_imgs[pid]) >= 4:
            pids.append(pid)
            pdist.append(np.median([np.linalg.norm(p.xyz - centers[i]) for i in track_imgs[pid]]))
    pdist = np.array(pdist)
    d_ref = np.percentile(pdist, 5)  # ~closest camera approach: sets thresholds
    sel = [pid for pid, d in zip(pids, pdist) if d <= np.percentile(pdist, 35)]
    pts = np.array([rec.points3D[pid].xyz for pid in sel])

    rng = np.random.default_rng(0)
    thresh = 0.015 * d_ref
    best_inl, best_plane = None, None
    for _ in range(8000):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        inl = np.abs((pts - p0) @ n) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_plane = inl, (p0, n)
    p0, n = best_plane
    cam_mean = np.mean(list(centers.values()), axis=0)
    if np.dot(n, cam_mean - p0) < 0:
        n = -n  # normal points up toward the cameras
    inpts = pts[best_inl]

    # largest connected cluster on the plane (grid flood fill)
    u = np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = np.stack([(inpts - p0) @ u, (inpts - p0) @ v], axis=1)
    cell = 0.04 * d_ref
    grid = {}
    for i, q in enumerate(uv):
        grid.setdefault((int(q[0] // cell), int(q[1] // cell)), []).append(i)
    seen, best_cluster = set(), []
    for start in grid:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            comp.extend(grid[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(comp) > len(best_cluster):
            best_cluster = comp
    cl = uv[best_cluster]

    c_uv = cl.mean(axis=0)
    cl0 = cl - c_uv
    _, V = np.linalg.eigh(cl0.T @ cl0)  # ascending: V[:,1] = major axis
    proj = cl0 @ V
    spans = [np.percentile(proj[:, k], 99) - np.percentile(proj[:, k], 1) for k in (0, 1)]
    minor_u, major_u = sorted(spans)
    center = p0 + c_uv[0] * u + c_uv[1] * v
    major_ax = V[1, 1] * v + V[0, 1] * u
    major_ax /= np.linalg.norm(major_ax)
    return dict(p0=p0, n=n, center=center, major_ax=major_ax,
                minor_u=minor_u, major_u=major_u, d_ref=d_ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_name")
    ap.add_argument("--height", type=float, help="surface height above floor (m) — recommended pin")
    ap.add_argument("--span", type=float, help="measured surface span (m) — fallback pin")
    ap.add_argument("--major", action="store_true", help="--span refers to the LONG side")
    ap.add_argument("--extents", type=float, nargs=2, metavar=("X", "Y"),
                    help="true surface size (m); else the cluster estimate is used")
    ap.add_argument("--data-root", default=str(STAGE / "data"))
    args = ap.parse_args()
    if not args.height and not args.span:
        ap.error("need --height (recommended) or --span")

    ws = pathlib.Path(args.data_root) / "colmap" / args.scene_name
    rec = pycolmap.Reconstruction(ws / "sparse" / "0")
    S = fit_surface(rec)
    p0, n = S["p0"], S["n"]

    # ---- metric scale -----------------------------------------------------
    if args.height:
        pts = np.array([p.xyz for p in rec.points3D.values() if p.error < 2.0])
        dh = (pts - p0) @ n
        below = dh[(dh < -0.15 * S["d_ref"]) & (dh > -12 * S["d_ref"])]
        hist, edges = np.histogram(below, bins=200)
        fb = edges[np.argmax(hist)]
        d_floor = -np.median(below[(below > fb - 0.1 * S["d_ref"]) & (below < fb + 0.1 * S["d_ref"])])
        scale = args.height / d_floor
        pin = f"height {args.height} m over floor at {d_floor:.3f} u"
    else:
        matched = S["major_u"] if args.major else S["minor_u"]
        scale = args.span / matched
        pin = f"span {args.span} m on cluster {'major' if args.major else 'minor'}"

    # ---- world frame: surface = z0, x = surface major axis -----------------
    x_ax = S["major_ax"]
    z_ax = n / np.linalg.norm(n)
    y_ax = np.cross(z_ax, x_ax)
    R = np.stack([x_ax, y_ax, z_ax], axis=0)  # colmap -> world rotation (rows)

    ext_est = (S["major_u"] * scale, S["minor_u"] * scale)
    extents = tuple(args.extents) if args.extents else ext_est

    # ---- sanity prints ------------------------------------------------------
    hs = np.array([np.dot(img.projection_center() - p0, n) * scale for img in rec.images.values()])
    print(f"scale: {scale:.5f} m/unit  ({pin})")
    print(f"surface cluster: {ext_est[0]:.2f} x {ext_est[1]:.2f} m "
          f"(must be <= the real surface; using {extents[0]:.2f} x {extents[1]:.2f})")
    print(f"camera heights over surface: median {np.median(hs):.2f} m, max {hs.max():.2f} m "
          f"(should look human)")

    # ---- outputs ------------------------------------------------------------
    ckpts = sorted(glob.glob(str(pathlib.Path(args.data_root) / "runs" / args.scene_name / "*" / "ckpt_last.pt")))
    scene = {
        "scene": args.scene_name,
        "scale_m_per_unit": scale,
        "scale_pin": pin,
        "surface_plane_point": p0.tolist(),
        "surface_plane_normal": n.tolist(),
        "surface_center_colmap": S["center"].tolist(),
        "R_world_from_colmap": R.tolist(),
        "surface_extents_m": list(extents),
        "checkpoint": ckpts[-1] if ckpts else None,
    }
    json.dump(scene, open(ws / "scene.json", "w"), indent=1)

    # ---- external camera for the demo (this example: robot at x = -0.5 m) --
    def to_colmap(p_world):
        return R.T @ (np.array(p_world) / scale) + S["center"]

    robot_pts = [to_colmap((-0.5, 0, 0.15)), to_colmap((-0.5, 0, 0.40))]
    flip = np.diag([1.0, -1.0, -1.0])
    chosen, fallback = None, None
    for i, img in rec.images.items():
        C = img.projection_center()
        h = np.dot(C - p0, n) * scale
        dist = np.linalg.norm(C - S["center"]) * scale
        if not (0.2 < h < 1.1 and 0.5 < dist < 1.8):
            continue
        Tcw = np.eye(4)
        Tcw[:3, :3] = img.cam_from_world().rotation.matrix()
        Tcw[:3, 3] = img.cam_from_world().translation
        Twc = np.linalg.inv(Tcw)
        to_c = S["center"] - C
        aim = np.dot(Twc[:3, 2], to_c / np.linalg.norm(to_c))
        if aim < 0.6:
            continue
        cam_i = rec.cameras[img.camera_id]
        fx_, fy_, cx_, cy_ = cam_i.params[:4]
        sees_robot = True
        for P in robot_pts:
            pc = Tcw[:3, :3] @ P + Tcw[:3, 3]
            if pc[2] < 0.05:
                sees_robot = False
                break
            uu, vv = fx_ * pc[0] / pc[2] + cx_, fy_ * pc[1] / pc[2] + cy_
            if not (20 < uu < cam_i.width - 20 and 20 < vv < cam_i.height - 20):
                sees_robot = False
                break
        pitch = abs(np.dot(Twc[:3, 2] / np.linalg.norm(Twc[:3, 2]), n))
        entry = (pitch, i, img.name, Twc)
        if sees_robot and (chosen is None or entry[0] < chosen[0]):
            chosen = entry
        if fallback is None or aim > fallback[0]:
            fallback = (aim, i, img.name, Twc)
    if chosen is None:
        print("WARN: no capture pose frames surface+robot; using best-aim pose")
    _, i, name, Twc = chosen or fallback
    T_usd = np.eye(4)
    T_usd[:3, :3] = (R @ Twc[:3, :3]) @ flip
    T_usd[:3, 3] = scale * (R @ (Twc[:3, 3] - S["center"]))
    cam = rec.cameras[rec.images[i].camera_id]
    fx, fy, cx, cy = cam.params[:4]
    json.dump({"image": name, "T_usd_cam_isaac": T_usd.tolist(),
               "T_cam_to_world_colmap": Twc.tolist(),
               "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                              "width": int(cam.width), "height": int(cam.height)}},
              open(ws / "ext_cam_real.json", "w"), indent=1)
    print(f"external camera: {name}")
    print(f"wrote {ws}/scene.json")


if __name__ == "__main__":
    main()
