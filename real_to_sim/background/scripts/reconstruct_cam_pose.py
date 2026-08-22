#!/usr/bin/env python3
"""Step 2: capture -> camera poses (COLMAP workspace).

Takes a capture folder (photos in imgs/, videos in videos/, or loose files),
builds the curated image set, and runs structure-from-motion:

    python scripts/reconstruct_cam_pose.py data/captures/<scene> <scene>

Videos: frames are extracted (--fps, or auto from --target-frames) and only
the sharpest of every --cull-group is kept. Photos: HEIC converted, EXIF
orientation baked, used as-is. Everything fuses into ONE reconstruction
(exhaustive matching), so multiple clips/batches of the same static scene work.

Output: <data>/colmap/<scene>/{images/, database.db, sparse/0/}. Health: the
final "REGISTERED n of m" — expect >90% for a good capture.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

STAGE = pathlib.Path(__file__).resolve().parents[1]

# re-exec inside the stage venv if needed (so no manual activation required)
VENV_PY = STAGE / ".venv" / "bin" / "python"
if VENV_PY.exists() and pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])

import cv2
import pycolmap
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

VIDEO_EXT = {".mov", ".mp4"}
PHOTO_EXT = {".heic", ".heif", ".jpg", ".jpeg", ".png"}


def video_duration(path: pathlib.Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def sharpness(path: pathlib.Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("scene_name")
    ap.add_argument("--fps", type=float, default=None,
                    help="video frame extraction rate (default: auto from --target-frames)")
    ap.add_argument("--target-frames", type=int, default=600,
                    help="auto-fps aims for this many extracted frames across all videos")
    ap.add_argument("--cull-group", type=int, default=2,
                    help="keep the sharpest video frame per group of N (1 = keep all)")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--camera-mode", choices=["auto", "single"], default="auto",
                    help="'auto': one camera model per resolution group (mixed sources ok)")
    ap.add_argument("--data-root", default=str(STAGE / "data"))
    args = ap.parse_args()

    cap = pathlib.Path(args.capture_dir)
    ws = pathlib.Path(args.data_root) / "colmap" / args.scene_name
    images = ws / "images"
    images.mkdir(parents=True, exist_ok=True)

    def collect(sub: str, exts: set) -> list:
        roots = [cap / sub, cap]
        return sorted({f for r in roots if r.is_dir() for f in r.iterdir()
                       if f.is_file() and f.suffix.lower() in exts})

    videos = collect("videos", VIDEO_EXT)
    photos = collect("imgs", PHOTO_EXT)
    if not videos and not photos:
        sys.exit(f"no videos/photos found under {cap}")

    # --- videos: extract at fps, keep sharpest per group -------------------
    if videos:
        total = sum(video_duration(v) for v in videos)
        fps = args.fps or max(1.0, min(10.0, args.target_frames / max(total, 1.0)))
        print(f"videos: {len(videos)} files, {total:.0f}s total -> extracting at {fps:.2f} fps")
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = pathlib.Path(tmp)
            for i, v in enumerate(videos, 1):
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-n", "-i", str(v), "-vf", f"fps={fps}",
                     "-qscale:v", "2", str(tmpd / f"v{i}_%04d.jpg")],
                    check=True,
                )
            frames = sorted(tmpd.iterdir())
            kept = 0
            for k in range(0, len(frames), args.cull_group):
                best = max(frames[k:k + args.cull_group], key=sharpness)
                shutil.copy2(best, images / best.name)
                kept += 1
            print(f"videos: kept {kept}/{len(frames)} frames (sharpest per {args.cull_group})")

    # --- photos: convert + bake orientation --------------------------------
    for f in photos:
        img = ImageOps.exif_transpose(Image.open(f))
        img.convert("RGB").save(images / (f.stem + ".jpg"), quality=96)
    if photos:
        print(f"photos: prepared {len(photos)}")

    # --- structure-from-motion ---------------------------------------------
    n_img = len(list(images.iterdir()))
    print(f"{n_img} images -> SfM ({args.matcher} matching; the slow part)")
    db = ws / "database.db"
    sparse = ws / "sparse"
    sparse.mkdir(exist_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    mode = pycolmap.CameraMode.AUTO if args.camera_mode == "auto" else pycolmap.CameraMode.SINGLE
    pycolmap.extract_features(db, images, camera_mode=mode, reader_options=reader)
    if args.matcher == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        pycolmap.match_sequential(db)
    maps = pycolmap.incremental_mapping(db, images, sparse)
    if not maps:
        sys.exit("SfM failed: no reconstruction (capture too sparse/blurry?)")
    best = max(maps, key=lambda k: maps[k].num_reg_images())
    if best != 0:
        shutil.rmtree(sparse / "0")
        (sparse / str(best)).rename(sparse / "0")
    print(maps[best].summary())
    print(f"REGISTERED: {maps[best].num_reg_images()} of {n_img}  ->  {ws}")


if __name__ == "__main__":
    main()
