"""Launch N L20 pods, one diversification episode each, in parallel.

The job spec is scripts/launch_cosigen_render_pool.py's, unchanged where it matters: the same
image, the same L20 pool/group/quota, the same apt list, the same RTX recipe (the Vulkan ICD
pointed at the real libGLX_nvidia plus the GLVND EGL vendor file — decisive even headless), the
same staged Isaac venv and CoSiGen tarballs from HDFS. That launcher is proven on these pods, so
only the payload changes: instead of booting a render server and registering it, each pod runs
ONE episode of run_episode.py and pushes its video, trajectory and verdict to HDFS.

One episode per pod, addressed by index, means no coordination at all: pod k runs episode k, and
a re-run of pod k reproduces the same theta. The diversification tree ships as its own small
tarball so the shared cosigen.tar.gz other jobs stage is left alone.

  # build + upload the code tarball, then launch 100 pods for batch <name>
  python launch_pool.py --batch nut_v1 --n 100 --upload
  python launch_pool.py --batch nut_v1 --n 4 --upload      # smoke first
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/tiger/cap-x")
DIV_TGZ_HDFS = "hdfs://haruna/tmp/zeyu.shen/cosigen_diversification.tar.gz"
OUT_ROOT = "hdfs://haruna/tmp/zeyu.shen/cosigen_div"

# Same pool the render servers use: the RTX recipe is proven on L20 (see resource-pools rule).
POOL = {"cluster": "palm-wlby", "group": "1804",
        "quota": "nvidia-l20-16.hpccluster-ydxtmq090qm5d2qqu2ss.ai", "gpuv": "NVIDIA-L20-16"}

YAML_TMPL = r"""caption: '[div] __BATCH__ ep__IDX__'
jobDefVersion:
  entrypointMode: FULL_SCRIPT
  gitRepo:
    commitSha: eba665296fbb353876907cffd1a742a9c647c951
    mnt: /opt/tiger/alpha-seed
    repoName: seed/alpha-seed
  imageMeta:
    imageVid: csost2bc77u68h428eb0
  pip3: []
  lazyDownloadCode: true
  name: '[div] __BATCH__ ep__IDX__'
  resource:
    arnoldConfig:
      clusterName: __CLUSTER__
      groupIds:
        - __GROUP__
      keepMins: __KEEPMINS__
      preemptible: true
      quotaPool: __QUOTA__
      roles:
        - cpu: 16
          gpu: 1
          gpuv: __GPUV__
          memory: 65536
          name: worker
          num: 1
          ports: 5
    backend: ARNOLD
jobRunParams:
  entrypointFullScript: |
    LOG=/tmp/div___LOGKEY__.log
    HDFS_LOG=__OUT__/logs/__LOGKEY__.log
    ( while true; do sleep 30; hdfs dfs -mkdir -p __OUT__/logs 2>/dev/null; hdfs dfs -put -f "$LOG" "$HDFS_LOG" 2>/dev/null; done ) &
    UPLOADER=$!
    {
      set -x
      nvidia-smi
      sudo apt-get update
      sudo apt-get install -y --no-install-recommends \
        libglu1-mesa libgl1 libegl1 libgles2 libglvnd0 libglx0 \
        libxrender1 libxext6 libsm6 libice6 libxkbcommon0 libxt6 \
        libxrandr2 libxi6 libxcursor1 libxinerama1 \
        libvulkan1 vulkan-tools libgomp1
      GLX=$(find /usr/lib /usr/local/nvidia /usr/local/lib /run/nvidia 2>/dev/null -name "libGLX_nvidia.so*" | head -1)
      NVLIB=${GLX:-$(ldconfig -p | grep -m1 'libGLX_nvidia.so.0' | awk '{print $NF}')}
      echo "libGLX_nvidia: ${NVLIB:-NONE}"
      sudo mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
      printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.3.277"}}\n' "${NVLIB:-libGLX_nvidia.so.0}" | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json
      printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json
      vulkaninfo --summary 2>&1 | grep -aiE "deviceName|driverName|ERROR" | head -6 || true

      cd /home/tiger
      time hdfs dfs -get -f hdfs://haruna/tmp/zeyu.shen/isaaclab_env.tar.gz /tmp/isaaclab_env.tar.gz
      time tar xzf /tmp/isaaclab_env.tar.gz -C /home/tiger && rm -f /tmp/isaaclab_env.tar.gz
      VENV=/home/tiger/isaaclab_build/env_isaaclab
      hdfs dfs -get -f hdfs://haruna/tmp/zeyu.shen/cosigen.tar.gz /tmp/cosigen.tar.gz
      tar xzf /tmp/cosigen.tar.gz -C /home/tiger && rm -f /tmp/cosigen.tar.gz
      # The diversification tree ships separately so the shared cosigen tarball is untouched.
      hdfs dfs -get -f __DIVTGZ__ /tmp/div.tar.gz
      mkdir -p /home/tiger/CoSiGen/diversification
      tar xzf /tmp/div.tar.gz -C /home/tiger/CoSiGen && rm -f /tmp/div.tar.gz
      # The cached Isaac venv has imageio but NOT its ffmpeg backend, so writing an mp4 failed
      # with "Could not find a backend" on the first smoke: no video, which is the deliverable.
      # The venv has no pip, so install the backend beside it and put it on PYTHONPATH — the
      # package is a thin wrapper plus a bundled ffmpeg binary, so imageio picks it up from there.
      pip3 install --no-cache-dir --target /tmp/pylibs imageio-ffmpeg || \
        pip3 install --no-cache-dir --target /tmp/pylibs \
          -i https://bytedpypi.byted.org/simple imageio-ffmpeg
      export PYTHONPATH=/tmp/pylibs:/home/tiger/CoSiGen:/home/tiger/CoSiGen/eval:$PYTHONPATH
      export IMAGEIO_FFMPEG_EXE=$(find /tmp/pylibs -name 'ffmpeg*' -type f -perm -u+x | head -1)
      echo "imageio-ffmpeg binary: ${IMAGEIO_FFMPEG_EXE:-NONE}"
      export OMNI_KIT_ACCEPT_EULA=YES
      export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      cd /home/tiger/CoSiGen/diversification
      $VENV/bin/python -u __SCRIPT__ __SCRIPTARGS__
      echo "[div] episode exited rc=$?"
      ls -l /tmp/div || true
    } > "$LOG" 2>&1
    kill $UPLOADER 2>/dev/null
    hdfs dfs -mkdir -p __OUT__/logs 2>/dev/null
    hdfs dfs -put -f "$LOG" "$HDFS_LOG"
  envsList:
    NVIDIA_DRIVER_CAPABILITIES: all
namespace: /user/zeyu.shen
"""


def pack_and_upload() -> None:
    tgz = Path(tempfile.gettempdir()) / "cosigen_diversification.tar.gz"
    # The reference solves ride along at their real repo paths: Phase-0 baselines run them
    # unmodified on the same pod recipe, and they are not in the shared cosigen tarball.
    subprocess.run(["tar", "czf", str(tgz), "--exclude=__pycache__", "--exclude=*.pyc",
                    "diversification", "examples/solve_nut_and_bolt.py",
                    "examples/solve_pen_holder.py"],
                   cwd=REPO / "CoSiGen", check=True)
    print(f"[div] packed {tgz} ({tgz.stat().st_size / 1e3:.0f} kB)")
    r = subprocess.run(["hdfs", "dfs", "-put", "-f", str(tgz), DIV_TGZ_HDFS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[div] upload failed: {r.stderr.strip()[:300]}")
    print(f"[div] uploaded -> {DIV_TGZ_HDFS}")


EPISODE_ARGS = ("--index __IDX__ --total __TOTAL__ --out /tmp/div --hdfs __OUT__ "
                "--max-sec __MAXSEC__ --headless --enable_cameras")


def launch(index: int, total: int, batch: str, max_sec: float, keep_mins: int,
           script: str = "run_episode.py", script_args: str = "", log_key: str = "") -> str:
    # Repeats of ONE index (same theta, different draw) would otherwise all write
    # logs/ep0.log and overwrite each other every 30 s, leaving no per-run diagnosis.
    out = f"{OUT_ROOT}/{batch}"
    # The injected args carry placeholders of their own, so they must go in FIRST — substituting
    # them last leaves a literal __IDX__ on the command line, which is how a 100-pod batch once
    # died at argparse before a single episode ran.
    yaml = (YAML_TMPL
            .replace("__SCRIPTARGS__", script_args or EPISODE_ARGS)
            .replace("__IDX__", str(index)).replace("__TOTAL__", str(total))
            .replace("__BATCH__", batch).replace("__OUT__", out)
            .replace("__DIVTGZ__", DIV_TGZ_HDFS).replace("__MAXSEC__", str(max_sec))
            .replace("__SCRIPT__", script)
            .replace("__LOGKEY__", log_key or f"ep{index}")
            .replace("__KEEPMINS__", str(int(keep_mins)))
            .replace("__CLUSTER__", POOL["cluster"]).replace("__QUOTA__", POOL["quota"])
            .replace("__GPUV__", POOL["gpuv"]).replace("__GROUP__", POOL["group"]))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"div_{batch}_{index}_",
                                     delete=False) as f:
        f.write(yaml)
        path = f.name
    r = subprocess.run(["mlx", "job", "submitv2", "-p", path], capture_output=True, text=True)
    line = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    return line[-1] if line else "(no output)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="names the HDFS output dir and the captions")
    ap.add_argument("--n", type=int, default=100, help="episodes = pods")
    ap.add_argument("--total", type=int, default=0, help="theta batch size (default: --n)")
    ap.add_argument("--start", type=int, default=0, help="first episode index (for topping up)")
    ap.add_argument("--max-sec", type=float, default=300.0)
    ap.add_argument("--keep-mins", type=int, default=180)
    ap.add_argument("--upload", action="store_true", help="pack + upload the code tarball first")
    ap.add_argument("--script", default="run_episode.py", help="payload to run on each pod")
    ap.add_argument("--script-args", default="", help="override the payload's flags")
    ap.add_argument("--log-key", default="", help="log file name (defaults to ep<index>)")
    a = ap.parse_args()
    total = a.total or a.n
    if a.upload:
        pack_and_upload()
    print(f"[div] batch={a.batch} episodes {a.start}..{a.start + a.n - 1} of {total} "
          f"-> {OUT_ROOT}/{a.batch}")
    for i in range(a.start, a.start + a.n):
        print(f"  ep{i:04d}: "
              f"{launch(i, total, a.batch, a.max_sec, a.keep_mins, a.script, a.script_args, a.log_key)}",
              flush=True)


if __name__ == "__main__":
    main()
