"""Report low-motion (freeze) windows in a run video (pod-side helper)."""
import sys

import imageio
import numpy as np

src = sys.argv[1]
reader = imageio.get_reader(src)
n = reader.count_frames()
fps = reader.get_meta_data()["fps"]
step = max(1, int(fps // 4))  # sample 4 Hz
prev = None
times, deltas = [], []
for fi in range(0, n, step):
    img = reader.get_data(fi).astype(np.int16)
    if prev is not None:
        deltas.append(float(np.abs(img - prev).mean()))
        times.append(fi / fps)
    prev = img
deltas = np.array(deltas)
times = np.array(times)
thresh = 0.10  # mean abs pixel delta below this ~= no visible motion
print(f"video {n / fps:.1f}s; motion mean={deltas.mean():.2f} p10={np.percentile(deltas, 10):.2f}")
in_freeze = False
start = 0.0
for t, d in zip(times, deltas):
    if d < thresh and not in_freeze:
        in_freeze, start = True, t
    elif d >= thresh and in_freeze:
        in_freeze = False
        if t - start >= 2.0:
            print(f"freeze {start:.1f}s -> {t:.1f}s ({t - start:.1f}s)")
if in_freeze and times[-1] - start >= 2.0:
    print(f"freeze {start:.1f}s -> end ({times[-1] - start:.1f}s)")
print("done")
