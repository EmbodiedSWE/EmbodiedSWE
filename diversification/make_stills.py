"""Turn a batch's videos into stills you can look at without opening 100 players.

Two outputs:
  filmstrip — one successful episode sampled across its length, so the whole threading sequence
              (carry, touchdown, wind strokes, seated) reads at a glance
  grid      — one frame per successful episode, late in the run, to see that the successes are
              genuinely different rollouts rather than the same motion 26 times

  python make_stills.py --batch nut_b2 [--episode 1] [--frames 6] [--cols 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/tiger/cap-x/eval_result/diversification")
_crop = None


CROP = (0.30, 0.20, 0.76, 0.95)   # the work area; the camera frames a lot of bare table
CROP_BY_BATCH = {"pen": (0.18, 0.10, 0.88, 0.98)}


def crop(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = _crop

    return arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def read_frames(path: Path, fracs: list[float]) -> list[np.ndarray]:
    """Frames at the given fractions of the video, one sequential pass."""
    rd = imageio.get_reader(str(path))
    n = rd.count_frames()
    want = sorted({min(n - 1, max(0, int(f * (n - 1)))) for f in fracs})
    out, wi = [], 0
    for i, frame in enumerate(rd):
        while wi < len(want) and want[wi] == i:
            out.append(crop(np.asarray(frame)))
            wi += 1
        if wi >= len(want):
            break
    rd.close()
    return out


def _brief(v: dict) -> str:
    """One-line summary from whichever fields this task's verdict carries."""
    if "dz_mm" in v:
        return f"dz={v['dz_mm']:.0f}mm sw={v['theta']['sweep_deg']:.0f}"
    return f"score={v.get('score')} {v.get('counted')}/{v.get('present')} {v['theta']['order_rule']}"


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, max(120, 8 * len(text)), 22], fill=(0, 0, 0))
    d.text((5, 5), text, fill=(255, 255, 255))
    return img


def tile(images: list[Image.Image], cols: int, pad: int = 4) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad),
                      (24, 24, 28))
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--episode", type=int, default=-1, help="filmstrip episode; -1 = first success")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb-w", type=int, default=460)
    a = ap.parse_args()
    global _crop
    d = ROOT / a.batch
    _crop = next((v for k, v in CROP_BY_BATCH.items() if a.batch.startswith(k)), CROP)
    verdicts = {}
    for f in sorted(d.glob("ep0*.json")):
        v = json.loads(f.read_text())
        verdicts[v["index"]] = v
    ok = sorted(i for i, v in verdicts.items() if v.get("success"))
    have = {int(p.stem[2:6]) for p in d.glob("ep*.mp4")}
    ok_with_video = [i for i in ok if i in have]
    print(f"{len(verdicts)} verdicts, {len(ok)} successes, {len(have)} videos present, "
          f"{len(ok_with_video)} successes with video")

    ep = a.episode if a.episode >= 0 else (ok_with_video[0] if ok_with_video else -1)
    if ep >= 0:
        path = d / f"ep{ep:04d}.mp4"
        v = verdicts[ep]
        fr = read_frames(path, [i / (a.frames - 1) for i in range(a.frames)])
        ims = []
        for k, arr in enumerate(fr):
            im = Image.fromarray(arr).resize((a.thumb_w, a.thumb_w * arr.shape[0] // arr.shape[1]))
            ims.append(label(im, f"{100 * k / (len(fr) - 1):.0f}%"))
        out = d / f"filmstrip_ep{ep:04d}.png"
        tile(ims, len(ims)).save(out)
        print(f"filmstrip: {out}  (ep{ep:04d} success={v['success']} " + _brief(v) + ")")

    if ok_with_video:
        ims = []
        for i in ok_with_video:
            v = verdicts[i]
            arr = read_frames(d / f"ep{i:04d}.mp4", [0.8])[0]
            im = Image.fromarray(arr).resize((a.thumb_w, a.thumb_w * arr.shape[0] // arr.shape[1]))
            ims.append(label(im, f"ep{i:04d} " + _brief(v)))
        out = d / "successes_grid.png"
        tile(ims, a.cols).save(out)
        print(f"grid: {out}  ({len(ims)} successful episodes)")


if __name__ == "__main__":
    main()
