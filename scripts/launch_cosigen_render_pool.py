"""Launch a pool of CoSiGen / robobench GPU render servers -- one Arnold job per env.

Mirrors launch_behavior_render_pool.py: each job stages the cached Isaac Lab env
(8.5G HDFS tarball) + the CoSiGen repo, boots one robobench env via
CoSiGen/eval/cosigen_render_server.py, runs a self-test, and registers its URL to
``hdfs://haruna/tmp/zeyu.shen/cosigen_render/<env>.txt``. The RL val job's
CoSiGenBackend discovers each server there (CAPX_COSIGEN_REGISTRY_DIR).

Usage:
  python scripts/launch_cosigen_render_pool.py                                  # default eval envs
  python scripts/launch_cosigen_render_pool.py assembly.ikea_table.g1.joint     # one env
  python scripts/launch_cosigen_render_pool.py env1 env2 ...                    # explicit subset
  python scripts/launch_cosigen_render_pool.py --num-envs 16 <env>              # N-env batch for
                                                  # the optimize tool's parallel search
  python scripts/launch_cosigen_render_pool.py --forge 10 [--offset K]          # sim_gen Isaac
                                                  # FORGE pods (task-construction smoke runners),
                                                  # register to hdfs .../simgen_forge/forge_<i>.txt
"""
import os
import subprocess
import sys
import tempfile

# Keep in sync with COSIGEN_ENVS in capx/integrations/swalm/prepare_swalm_dataset.py.
DEFAULT_ENVS = [
    "assembly.ikea_table.g1.pink_ik",
    "assembly.ikea_table.g1.joint",
]

REGISTRY_DIR = "hdfs://haruna/tmp/zeyu.shen/cosigen_render"

# GPU pool presets -- submit wherever there is quota (see also CAPX_GPU_POOL env var).
# NOTE: the RTX render recipe is PROVEN on L20 (graphics-capable). H20 rendering is
# unvalidated -- watch the boot log's "record camera ready" line on first H20 use.
POOLS = {
    "l20": {"cluster": "palm-wlby",
            "group": "1804",
            "quota": "nvidia-l20-16.hpccluster-ydxtmq090qm5d2qqu2ss.ai",
            "gpuv": "NVIDIA-L20-16"},
    "h20-gve6": {"cluster": "soil-wlby",
                 "group": "2006",
                 "quota": "nvidia-h20.hpccluster-yecd0gve6pioxhtqvap6.ai",
                 "gpuv": "NVIDIA-H20"},
}

# __TAG__ is the caption prefix. It MUST distinguish the RL training pool from the
# eval render pods: both come from this launcher, so a shared caption makes the two
# indistinguishable in `mlx job list` -- and picking jobs to stop by caption then
# risks killing someone's eval run. RL pools pass --tag SimGen-RL (user directive
# 2026-07-30: "when you cancel jobs, look for simgen-rl, not [cosigen-server]").
#
# The job caption/name deliberately does NOT contain a scene (user directive
# 2026-07-31): pods are fungible -- engines /switch scenes on demand -- so a
# per-scene job name misleads. The scene lives only in the pod-internal slot
# key (__ENV__ -> registry filename + HDFS log name), which the keeper uses to
# track slots and pick a resubmitted pod's BOOT scene.
YAML_TMPL = r"""caption: '[__TAG__] fungible engine pod'
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
  name: '[__TAG__] fungible engine pod'
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
    LOG=/tmp/cosigen_server.log
    HDFS_LOG=hdfs://haruna/tmp/zeyu.shen/cosigen_server___ENV__.log
    ( while true; do sleep 30; hdfs dfs -put -f "$LOG" "$HDFS_LOG" 2>/dev/null; done ) &
    UPLOADER=$!
    ENVNAME=__ENV__
    SRV=0
    {
      set -x
      nvidia-smi
      sudo apt-get update
      sudo apt-get install -y --no-install-recommends \
        libglu1-mesa libgl1 libegl1 libgles2 libglvnd0 libglx0 \
        libxrender1 libxext6 libsm6 libice6 libxkbcommon0 libxt6 \
        libxrandr2 libxi6 libxcursor1 libxinerama1 \
        libvulkan1 vulkan-tools libgomp1
      # RTX rendering recipe (proven on these L20s by launch_isaac_g1.py): graphics libs come
      # from NVIDIA_DRIVER_CAPABILITIES=all (envsList); register the NVIDIA Vulkan ICD pointing
      # at the ACTUAL libGLX_nvidia + the GLVND EGL vendor dispatcher (decisive even headless).
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
      hdfs dfs -get -f __COSIGEN_TARBALL__ /tmp/cosigen.tar.gz
      tar xzf /tmp/cosigen.tar.gz -C /home/tiger && rm -f /tmp/cosigen.tar.gz
      # cached venv has no pip; robobench imports from the repo dir on PYTHONPATH.
      # simgen_rl/render: the render server + session stack, MOVED out of
      # eval/legacy into the RL package (user directive 2026-08-02) -- the
      # tarball ships CoSiGen/ AND simgen_rl/. Bare sibling imports inside the
      # server resolve via its own directory (sys.path[0] when run as a script;
      # listed here too for any subprocess).
      export PYTHONPATH=/home/tiger/CoSiGen:/home/tiger/CoSiGen/eval:/home/tiger/simgen_rl/render:$PYTHONPATH
      export OMNI_KIT_ACCEPT_EULA=YES
      export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
__SERVER_BLOCK__
    } > "$LOG" 2>&1
    kill $UPLOADER 2>/dev/null
    hdfs dfs -put -f "$LOG" "$HDFS_LOG"
    wait $SRV
  envsList:
    NVIDIA_DRIVER_CAPABILITIES: all
    CAPX_VIDEO_EVERY: '2'
    # Reply window for one /run_policy turn. Long agent turns (optimize searches)
    # legitimately exceed the old 1800s default; past it the server 504s while the
    # turn keeps executing (drivers now recover via /last_result, but a wider window
    # keeps results flowing inline).
    CAPX_RUN_TIMEOUT_S: '7200'
    # RL-training batch size: env 0 authoritative + N-1 replicas (pink-IK cost ~linear in N).
    CAPX_NUM_ENVS: '__NUM_ENVS__'
namespace: /user/zeyu.shen
"""


RENDER_BLOCK = r"""      IFS=',' read -ra PORTS <<< "${ARNOLD_WORKER_0_PORT:-10355,10356,10357,10358,10359}"
      NENG=${CAPX_ENGINES_PER_POD:-1}
      cd /home/tiger/simgen_rl/render
      # NENG independent engine PROCESSES share this pod's GPU: each is its own Isaac
      # sim + HTTP server + registry entry (engine 0 keeps the plain name; extras
      # register as <name>_e<k>). One engine hosts ONE live episode (lockstep
      # vectorized physics), so engines-per-pod is the per-GPU session-concurrency
      # lever -- size it to GPU memory (see nvidia-smi on a booted pod).
      #
      # ENV-FUNGIBLE ENGINES: each engine runs in a RELAUNCH LOOP reading its target
      # env from /tmp/engine_<k>_env.txt. POST /switch rewrites that file and
      # hard-exits the process; the loop reboots it on the new scene (~2-3 min on
      # warm kit caches -- staging is NOT repeated). This also auto-recovers engine
      # crashes without waiting for a whole-pod keeper resubmit.
      for k in $(seq 0 $((NENG-1))); do
        REG=__REGDIR__/${ENVNAME}_e${k}.txt
        if [ "$k" = "0" ]; then REG=__REGDIR__/${ENVNAME}.txt; fi
        ENVFILE=/tmp/engine_${k}_env.txt
        echo "__BOOTENV__" > $ENVFILE
        (
          while true; do
            CUR=$(head -1 $ENVFILE)
            echo "[engine-$k] launching env=$CUR" >> /tmp/server_proc_${k}.log
            $VENV/bin/python -u /home/tiger/simgen_rl/render/cosigen_render_server.py --env $CUR --max-steps 3000 \
              --num-envs ${CAPX_NUM_ENVS:-512} \
              --port ${PORTS[$k]} --registry $REG --env-file $ENVFILE \
              >> /tmp/server_proc_${k}.log 2>&1
            echo "[engine-$k] server exited rc=$?; relaunching in 10s" >> /tmp/server_proc_${k}.log
            sleep 10
          done
        ) &
        SRV=$!
      done
      # Boot gate: wait for every engine's /ping to report booted. The old
      # selftest_server.py/selftest_policy.py pair was retired with the eval reorg
      # (2026-08-03: pods that pulled the post-reorg tarball errored right here),
      # and a ping wait is all the boot script actually needs from it. printf-built
      # script: a heredoc or multi-line python -c cannot survive this YAML block.
      printf 'import json,sys,time,urllib.request\nport=int(sys.argv[1]); deadline=time.time()+3600\nwhile time.time()<deadline:\n    try:\n        j=json.loads(urllib.request.urlopen("http://127.0.0.1:%%d/ping"%%port,timeout=8).read())\n        if j.get("booted"): print("[bootwait]",port,"booted; activity=",j.get("activity")); sys.exit(0)\n        if j.get("boot_error"): print("[bootwait]",port,"BOOT_ERROR:",j.get("boot_error")); sys.exit(1)\n    except SystemExit: raise\n    except Exception: pass\n    time.sleep(15)\nprint("[bootwait]",port,"not booted after 3600s"); sys.exit(1)\n' > /tmp/bootwait.py
      for k in $(seq 0 $((NENG-1))); do
        $VENV/bin/python -u /tmp/bootwait.py ${PORTS[$k]}
      done
      tail -40 /tmp/server_proc_0.log"""

# The forge payload was lost when the pod-agent block was removed. The forge itself never boots
# Isaac: it is a thin HTTP server that submits/runs task modules with the staged venv, so a Kit
# crash in one run cannot take the pod with it (see sim_gen/isaac/forge_server.py).
FORGE_BLOCK = r"""      IFS=',' read -ra PORTS <<< "${ARNOLD_WORKER_0_PORT:-10355,10356,10357,10358,10359}"
      export MY_HOST_IPV6=${MY_HOST_IPV6:-$ARNOLD_WORKER_0_HOST}
      cd /home/tiger/CoSiGen
      $VENV/bin/python -u sim_gen/isaac/forge_server.py --port ${PORTS[0]} \
        --workdir /home/tiger/forge_work --registry __REGDIR__/${ENVNAME}.txt &
      SRV=$!
      sleep 20
      curl -s --max-time 10 "http://127.0.0.1:${PORTS[0]}/ping" || echo "(forge ping failed)"
      wait $SRV"""

# sim_gen Isaac forges register in their own HDFS dir, apart from render servers.
FORGE_REGISTRY_DIR = "hdfs://haruna/tmp/zeyu.shen/simgen_forge"


# Repo tarball the pod extracts at boot. Parameterized 2026-08-03 (new-task-wave
# migration): the OLD pool's pods must keep booting the old tarball (its scenes are
# gone from the repo tree), while the NEW pool boots cosigen_v2.tar.gz -- two pools,
# two frozen artifacts, zero cross-contamination.
DEFAULT_TARBALL = "hdfs://haruna/tmp/zeyu.shen/cosigen.tar.gz"


def launch(env: str, num_envs: int = 16, pool: str = "l20", forge: bool = False,
           alias: str = "", disable_features: str | None = None,
           registry_dir: str = "", keep_mins: int = 1440,
           engines: int = 1, tag: str = "CoSiGen-server",
           tarball: str = DEFAULT_TARBALL) -> None:
    """alias: distinct registry/log/caption key when several pods serve the SAME env
    (ablation trio, 2026-07-25; also the --count N pods-per-env mechanism -- aliases
    `<env>__<k>`, discovered by simgen_rl.client's `<env>__*.txt` glob).
    disable_features: bakes CAPX_DISABLE_FEATURES into the pod env (e.g.
    'checkpoint,rl' for the no-harness baseline). registry_dir: override the HDFS
    registry (the RL training pool registers apart from the eval pods).
    keep_mins: pod lifetime (RL pools want multi-day, eval default 1440)."""
    p = POOLS[pool]
    block = FORGE_BLOCK if forge else RENDER_BLOCK
    yaml = (YAML_TMPL
            .replace("__SERVER_BLOCK__", block)
            .replace("__TAG__", tag)
            .replace("__ENV__", alias or env)
            .replace("__REGDIR__", registry_dir or (FORGE_REGISTRY_DIR if forge else REGISTRY_DIR))
            .replace("__NUM_ENVS__", str(int(num_envs)))
            .replace("__KEEPMINS__", str(int(keep_mins)))
            # ENVNAME (= alias when set) keys the registry + logs; __BOOTENV__ is the
            # scene the engines BOOT (slot pods: initial env only -- /switch moves them).
            .replace("__BOOTENV__", env)
            .replace("__COSIGEN_TARBALL__", tarball)
            .replace("__CLUSTER__", p["cluster"]).replace("__QUOTA__", p["quota"])
            .replace("__GPUV__", p["gpuv"]).replace("__GROUP__", p["group"]))
    if disable_features is not None:
        yaml = yaml.replace("    CAPX_NUM_ENVS:",
                            f"    CAPX_DISABLE_FEATURES: '{disable_features}'\n"
                            "    CAPX_NUM_ENVS:")
    if int(engines) > 1:
        yaml = yaml.replace("    CAPX_NUM_ENVS:",
                            f"    CAPX_ENGINES_PER_POD: '{int(engines)}'\n"
                            "    CAPX_NUM_ENVS:")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=f"cosigen_{env}_", delete=False) as f:
        f.write(yaml)
        path = f.name
    kind = "FORGE" if forge else "render server"
    print(f"[pool] submitting {kind} for {env} num_envs={num_envs} ({path})")
    out = subprocess.run(["mlx", "job", "submitv2", "-p", path], capture_output=True, text=True)
    print((out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "(no output)")


def _fetch_registry(registry_dir: str) -> str | None:
    """ONE bulk `hdfs dfs -get` of the whole registry per keeper sweep. The previous
    per-key `hdfs dfs -cat` spawned a ~1GB JVM per pod per sweep (252 of them every
    10 min) -- enough to contribute to devbox overload (2026-07-24).

    Returns None when the fetch did not produce a usable registry. That distinction is
    the whole point: this used to return the path unconditionally, so a failed or
    partial `-get` looked exactly like "every pod is unregistered" and the keeper
    duplicated the ENTIRE pool. It also used a FIXED /tmp path and rmtree'd it, so two
    keeper processes racing on the same path erased each other's copy mid-read.
    Measured cost of those two bugs, 2026-07-29: 3 full resubmits of a HEALTHY
    160-pod pool (the launch watchdog saw 316 engines booted at the same moment the
    keeper called them all unregistered) -> 600 running pods, ~480 of them useless."""
    import tempfile
    # Per-process dir: concurrent keepers must not share (or delete) one copy.
    local = tempfile.mkdtemp(prefix=f"_keeper_registry_{os.getpid()}_")
    # Glob *.txt rather than fetching the DIRECTORY: a pod registering during the
    # sweep leaves a transient `<key>.txt._COPYING_`, and a whole-dir -get tries to
    # read it, fails with rc=255, and costs the entire sweep (seen 2026-07-30). The
    # glob only matches settled files, so registration and sweeping stop colliding.
    r = subprocess.run(["hdfs", "dfs", "-get", f"{registry_dir}/*.txt", local + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isdir(local):
        print(f"[keeper] registry fetch FAILED rc={r.returncode}: "
              f"{(r.stderr or '').strip()[:200]}", flush=True)
        return None
    if not [f for f in os.listdir(local) if f.endswith(".txt")]:
        print(f"[keeper] registry fetch returned NO .txt files from {registry_dir}; "
              f"treating as a read failure, not as an empty pool", flush=True)
        return None
    return local


def _ping_registered(registry_dir: str, key: str, local_dir: str | None = None) -> dict:
    """Read a pod's registry file (from the bulk-fetched local copy when given) and
    ping it. Returns {url, status} with status in {'up','booting','dead','unregistered'}."""
    import json as _json
    import urllib.request
    if local_dir:
        try:
            with open(os.path.join(local_dir, f"{key}.txt")) as f:
                out = f.read()
        except OSError:
            out = ""
    else:
        out = subprocess.run(["hdfs", "dfs", "-cat", f"{registry_dir}/{key}.txt"],
                             capture_output=True, text=True).stdout
    urls = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("http")]
    if not urls:
        return {"url": None, "status": "unregistered"}
    url = urls[-1].rstrip("/")
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        meta = _json.loads(opener.open(url + "/ping", timeout=8).read())
    except Exception:  # noqa: BLE001 -- unreachable pod
        return {"url": url, "status": "dead"}
    if meta.get("booted"):
        return {"url": url, "status": "up"}
    if meta.get("boot_error"):
        return {"url": url, "status": "dead"}
    return {"url": url, "status": "booting"}


def keeper(envs: list[str], count: int, num_envs: int, pool: str, registry_dir: str,
           keep_mins: int, engines: int = 1, interval_s: float = 600.0,
           boot_grace_s: float = 5400.0, boot_envs: list[str] | None = None,
           startup_grace_s: float | None = None,
           disable_features: str | None = None,
           tag: str = "CoSiGen-server",
           tarball: str = DEFAULT_TARBALL) -> None:
    """Liveness keeper for a training render pool: every `interval_s`, resubmit pods
    whose registered endpoint is unreachable/boot-failed (preemptions, keepMins expiry,
    Isaac crashes). A freshly (re)submitted pod gets `boot_grace_s` before it can be
    resubmitted again (Isaac + tarball staging takes tens of minutes). With
    engines > 1 a pod is resubmitted only when ALL its engines are dead (a resubmit
    kills the surviving engines' sessions). Slot pools (env-fungible): `envs` holds
    the SLOT KEYS and `boot_envs` the initial scenes (round-robin; a resubmitted
    slot's scene is arbitrary -- the scheduler /switch-es engines on demand)."""
    import time
    if boot_envs:
        pairs = [(boot_envs[i % len(boot_envs)], key) for i, key in enumerate(envs)]
    else:
        pairs = [(e, "" if k == 0 else f"{e}__{k}") for e in envs for k in range(max(1, count))]
    # STARTUP GRACE for every pod: the keeper does not know which pods were just
    # submitted externally (their registry entries do not exist yet) -- without
    # this it duplicates the entire freshly-launched pool on its first sweep
    # (happened 2026-07-24: 14 duplicate submissions before the kill).
    # ...but when the keeper is (re)started to RECOVER an already-decayed pool, the
    # pods are long dead and a 90-minute grace just extends the outage; pass a short
    # --startup-grace-s in that case.
    _sg = boot_grace_s if startup_grace_s is None else startup_grace_s
    grace: dict[str, float] = {alias or env: time.time() + _sg for env, alias in pairs}
    # RESUBMIT CAP (leak fix 2026-07-25): a pod whose scene fails to BUILD reports
    # boot_error forever, which _ping_registered classes as "dead", so the keeper
    # resubmitted it every boot_grace_s. Over 9h that grew a 126-pod pool to 254
    # running pods (128 wasted L20s). Give up on a key after this many attempts and
    # say so loudly -- a key that never boots is a scene/code bug, not a flaky pod.
    attempts: dict[str, int] = {}
    max_attempts = int(os.environ.get("CAPX_KEEPER_MAX_RESUBMITS", "3"))
    print(f"[keeper] watching {len(pairs)} pods ({len(envs)} envs x {max(1, count)}, "
          f"{max(1, engines)} engines each) registry={registry_dir}; startup grace "
          f"{_sg:.0f}s, post-resubmit grace {boot_grace_s:.0f}s", flush=True)
    while True:
        local_reg = _fetch_registry(registry_dir)  # ONE hdfs call per sweep
        if local_reg is None:
            print("[keeper] skipping this sweep (registry unreadable); resubmitting "
                  "NOTHING", flush=True)
            time.sleep(interval_s)
            continue
        # One ping pass per sweep, reused by both the guard and the resubmit loop.
        by_key = {}
        for env, alias in pairs:
            key = alias or env
            by_key[key] = [_ping_registered(registry_dir, key if j == 0 else f"{key}_e{j}",
                                            local_dir=local_reg)["status"]
                           for j in range(max(1, engines))]
        # BLAST-RADIUS GUARD, narrowed. The 2026-07-29 cascade (160 healthy pods ->
        # 600) came from a PARTIAL registry read: keys had no URL at all, so they read
        # as 'unregistered'. A pod that is genuinely gone reads as 'dead' -- it has a
        # registered URL that no longer answers. So suspect the registry only when the
        # pool looks *unregistered* en masse; never refuse to act on 'dead'.
        # The blanket "<50% up -> do nothing" version of this guard deadlocked recovery
        # on 2026-07-30: the pool decayed to 7% alive and the keeper sat out every
        # sweep for ~10h while training ran against 22 of 320 engines.
        unreg = sum(1 for s in by_key.values() if all(x == "unregistered" for x in s))
        if unreg > 0.5 * len(pairs):
            print(f"[keeper] {unreg}/{len(pairs)} pods have NO registry entry -- that is "
                  f"a partial registry read, not {unreg} deaths. Resubmitting NOTHING.",
                  flush=True)
            time.sleep(interval_s)
            continue
        # Bounded recovery rate: a real mass death still gets fixed, but over several
        # sweeps, so a mistake costs a fraction of the pool instead of all of it.
        per_sweep_cap = max(8, int(0.35 * len(pairs)))
        up = booting = resubmitted = 0
        for env, alias in pairs:
            key = alias or env
            statuses = by_key[key]
            if "up" in statuses:
                up += 1
                # The cap below is meant for scenes that never BUILD. A pod that booted
                # and later died (preemption, keepMins, Isaac crash) must stay eligible
                # forever, so only *consecutive* failures count toward giving up.
                attempts[key] = 0
                continue
            if "booting" in statuses or grace.get(key, 0) > time.time():
                booting += 1
                continue
            if attempts.get(key, 0) >= max_attempts:
                if attempts[key] == max_attempts:      # log the give-up once
                    print(f"[keeper] {key}: GIVING UP after {max_attempts} resubmits "
                          f"(engines={statuses}); this key needs a code/scene fix, "
                          f"not another pod", flush=True)
                    attempts[key] += 1
                continue
            if resubmitted >= per_sweep_cap:
                print(f"[keeper] per-sweep resubmit cap {per_sweep_cap} reached; the "
                      f"rest of the dead pods wait for the next sweep", flush=True)
                break
            attempts[key] = attempts.get(key, 0) + 1
            print(f"[keeper] {key}: engines={statuses} -> resubmitting "
                  f"(attempt {attempts[key]}/{max_attempts})", flush=True)
            launch(env, num_envs=num_envs, pool=pool, alias=alias,
                   registry_dir=registry_dir, keep_mins=keep_mins, engines=engines,
                   disable_features=disable_features, tag=tag, tarball=tarball)
            grace[key] = time.time() + boot_grace_s
            resubmitted += 1
        print(f"[keeper] sweep: up={up} booting/grace={booting} "
              f"resubmitted={resubmitted} / {len(pairs)}", flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    args = sys.argv[1:]
    # Default 512: num_envs is the optimize tool's search width, and width is what
    # makes searches efficient — measured 2026-07-26 on the pathological grasp search,
    # 512-wide ran 2048 evaluations in 831 s (0.41 s/eval) vs ~10 s/eval at 64, because
    # the pathological-contact penalty saturates while evaluations scale with width.
    # Video stays clean: cameras frame env 0 only.
    # Pass --num-envs 1 explicitly for demo/solution-video servers.
    num_envs = 512
    pool = os.environ.get("CAPX_GPU_POOL", "l20")
    count = 1
    registry_dir = ""
    keep_mins = 1440
    run_keeper = False

    def _take(flag, cast=str, default=None):
        global args
        if flag in args:
            i = args.index(flag)
            v = cast(args[i + 1])
            args = args[:i] + args[i + 2:]
            return v
        return default

    num_envs = _take("--num-envs", int, num_envs)
    pool = _take("--pool", str, pool)
    count = _take("--count", int, count)
    registry_dir = _take("--registry-dir", str, registry_dir)
    keep_mins = _take("--keep-mins", int, keep_mins)
    engines = _take("--engines", int, 1)
    # --slots N: ENV-FUNGIBLE pool mode. Submits N generic pods keyed slot000..N-1
    # in the registry, with initial scenes round-robined over the given envs; the
    # scheduler moves engines between tasks at lease time via POST /switch (process
    # restart on warm caches). This is how a 1000-task dataset runs on a few hundred
    # L20s: engine demand tracks the per-step batch, not the task count.
    slots = _take("--slots", int, 0)
    startup_grace_s = _take("--startup-grace-s", float, None)
    # e.g. --disable-features checkpoint,opt for the NATIVE-CC arm: the server drops
    # those sections from the prompt AND the program namespace, so the pod agrees with
    # a driver that registers no checkpoint/optimize tools.
    disable_features = _take("--disable-features", str, None)
    # Sweep period. The default paces a steady-state pool; recovering a mostly-dead
    # one wants a shorter one, because the per-sweep resubmit cap means the pool comes
    # back over ceil(dead / cap) sweeps.
    interval_s = _take("--interval-s", float, 600.0)
    # Caption prefix -- see YAML_TMPL. Pass --tag SimGen-RL for the training pool so
    # its pods are separable from eval render pods by name alone.
    tag = _take("--tag", str, "CoSiGen-server")
    # Repo tarball override (new-task-wave pools boot cosigen_v2.tar.gz; the old
    # pool keeps the default -- see DEFAULT_TARBALL comment).
    tarball = _take("--tarball", str, DEFAULT_TARBALL)
    if "--keeper" in args:
        args.remove("--keeper")
        run_keeper = True
    if "--forge" in args:
        i = args.index("--forge")
        n = int(args[i + 1])
        rest = args[:i] + args[i + 2:]
        offset = 0
        if "--offset" in rest:
            j = rest.index("--offset")
            offset = int(rest[j + 1])
        for k in range(offset, offset + n):
            launch(f"forge_{k}", pool=pool, forge=True)
    elif run_keeper and slots:
        envs = args or DEFAULT_ENVS
        keeper([f"slot{i:03d}" for i in range(slots)], 1, num_envs, pool,
               registry_dir or REGISTRY_DIR, keep_mins, engines=engines,
               boot_envs=envs, startup_grace_s=startup_grace_s,
               disable_features=disable_features, interval_s=interval_s, tag=tag,
               tarball=tarball)
    elif run_keeper:
        envs = args or DEFAULT_ENVS
        keeper(envs, count, num_envs, pool, registry_dir or REGISTRY_DIR, keep_mins,
               engines=engines, startup_grace_s=startup_grace_s,
               disable_features=disable_features, interval_s=interval_s, tag=tag,
               tarball=tarball)
    elif slots:
        envs = args or DEFAULT_ENVS
        for i in range(slots):
            launch(envs[i % len(envs)], num_envs=num_envs, pool=pool,
                   alias=f"slot{i:03d}", disable_features=disable_features, tag=tag,
                   registry_dir=registry_dir, keep_mins=keep_mins, engines=engines,
                   tarball=tarball)
    else:
        envs = args or DEFAULT_ENVS
        for e in envs:
            for k in range(max(1, count)):
                launch(e, num_envs=num_envs, pool=pool,
                       alias=("" if k == 0 else f"{e}__{k}"),
                       disable_features=disable_features, tag=tag,
                       registry_dir=registry_dir, keep_mins=keep_mins, engines=engines,
                       tarball=tarball)
