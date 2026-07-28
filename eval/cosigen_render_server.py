"""Persistent CoSiGen / robobench render/sim server (runs ON the GPU worker).

Mirrors Seed_behavior/behavior_render_server.py: boots ONE robobench env (Isaac
Lab) and keeps it warm, exposing a tiny HTTP API so the off-GPU CPU rollout
worker (capxrl CoSiGenBackend) can run model-written control programs against the
live sim with NO Isaac reboot per iteration.

Endpoints (IPv6):
  GET  /ping        -> {ok, booted, activity, prompt, boot_error, n_runs, num_envs}
  POST /run_policy   {code, max_steps?, num_frames?, reset?}
                    -> {ok, rc, stdout, stderr, success, scene_summary, event_log, n_runs}

MULTI-TURN SESSIONS: pass reset=true on the FIRST turn of an episode (fresh sim reset +
fresh exec namespace) and reset=false on later turns -- the episode AND the namespace
persist across turns (Jupyter-cell semantics), which is what the agent-trained-RL skill
needs (train a policy in one turn, look at feedback, use it in the next).

`activity` here is the robobench env name (e.g. "assembly.ikea_table.g1.pink_ik").
On boot it publishes its reachable address to an HDFS registry so the backend can
discover it (one server per env name, exactly like the BEHAVIOR pool).

Run (Isaac env_isaaclab venv, GPU worker, after the cached-env staging):
    python -u CoSiGen/eval/cosigen_render_server.py \
        --env assembly.ikea_table.g1.pink_ik --max-steps 3000 \
        --port $PORT --registry hdfs://haruna/tmp/zeyu.shen/cosigen_render/<env>.txt
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import socket
import subprocess
import threading
import time
import traceback

# pinocchio MUST precede AppLauncher (pink_ik / eigenpy converters), and the app
# MUST boot before importing robobench/torch (cosigen_loop). So we boot here first.
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

_ap = argparse.ArgumentParser()
_ap.add_argument("--env", default="assembly.ikea_table.g1.pink_ik")
_ap.add_argument("--max-steps", type=int, default=3000)
# N-env batch: env 0 is authoritative for control/grading; envs 1..N-1 are the RL-training
# batch (mirrored replicas during scripted phases). pink-IK cost is ~linear in N (per-env QP).
_ap.add_argument("--num-envs", type=int,
                 default=int(os.environ.get("CAPX_NUM_ENVS", "1")))
_ap.add_argument("--port", type=int,
                 default=int((os.environ.get("ARNOLD_WORKER_0_PORT", "10355").split(",")[0])))
_ap.add_argument("--advertise-host", default=os.environ.get("ARNOLD_WORKER_0_HOST", ""))
_ap.add_argument("--registry", default="")
# ENV-FUNGIBLE POOLS (SimGen-RL v2): the pod wrapper relaunches this server in a
# loop, reading the desired env from --env-file each iteration. POST /switch writes
# a new env there and HARD-EXITS the process (kit teardown hangs are a known repo
# pattern; the wrapper reboot IS the teardown). Line 2 of the file optionally names
# the session the reboot is reserved for: that session gets an automatic boot-time
# lease so nobody steals the freshly switched engine.
_ap.add_argument("--env-file", default="")
_ap.add_argument("--frames-hdfs-dir", default=os.environ.get(
    "CAPX_COSIGEN_FRAMES_DIR", "hdfs://haruna/tmp/zeyu.shen/cosigen_frames"))
# HOT-RELOAD: the harness control-API/prompt module (cosigen_loop.py) is pushed here by the dev
# loop; POST /reload re-downloads it + reimports it on the LIVE booted env (no Isaac reboot).
_ap.add_argument("--reload-src", default=os.environ.get(
    "CAPX_COSIGEN_RELOAD_SRC", "hdfs://haruna/tmp/zeyu.shen/cosigen_harness/cosigen_loop.py"))
AppLauncher.add_app_launcher_args(_ap)
ARGS = _ap.parse_args()

# RTX rendering DOES work on these L20s (proven by Fable_5 G1 pick-place). Two pieces required:
#  (a) NVIDIA_DRIVER_CAPABILITIES=all (set in the launch envsList) injects the graphics/GL libs,
#      else the renderer reports "no suitable CUDA GPU found" and PhysX drops to software/hangs;
#  (b) --/rtx/verifyDriverVersion/enabled=false: kit mis-decodes driver 535.261 -> 535.5 and
#      wrongly rejects it, so we disable the check.
ARGS.headless = True
ARGS.enable_cameras = True
ARGS.enable_pinocchio = True
if not getattr(ARGS, "kit_args", None):
    ARGS.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(ARGS).app

import cosigen_loop as L  # noqa: E402  (imports robobench/torch; safe post-AppLauncher)

STATE: dict = {
    "booted": False,
    "boot_error": None,
    "activity": ARGS.env,
    "lock": threading.Lock(),  # serialize sim access
    "ctx": None,
    "n_runs": 0,
    # Last COMPLETED /run_policy result, served on GET /last_result. Lets a driver
    # whose request 504'd (turn ran past CAPX_RUN_TIMEOUT_S while the main thread kept
    # executing) recover the finished turn's full result instead of losing the output
    # or re-POSTing a duplicate execution (2026-07-23 v7 opus driver crashes).
    "last_result": None,
    # Persistent multi-turn session: the agent's exec namespace, kept across /run_policy
    # calls with reset=false (cleared on reset=true). See module docstring.
    "session_ns": None,
    # Kit rendering must be driven from the MAIN thread (the one that launched the app); a
    # sim.render() from an HTTP worker thread hangs. So /run_policy enqueues a job here and the
    # main thread executes do_run (physics + render) and signals completion.
    "jobs": queue.Queue(),
    # EPISODE LEASE + FIFO WAIT QUEUE (RL training, 2026-07-24): one warm env = ONE
    # live episode, so concurrent RL rollouts must not interleave sessions on the same
    # engine. A rollout POSTs /acquire {session, ttl_s, wait_s} before its episode and
    # /release after; while a lease is active, /run_policy rejects other sessions with
    # 409. When the engine is busy, /acquire LONG-POLLS in a server-side FIFO queue:
    # the release hands the engine to the head waiter within ~2s (no client-side
    # poll-and-retry idle gap -- with far more episodes than engines, that gap is pure
    # throughput loss). Fully backward compatible: clients that never acquire (the
    # eval harness drivers, one per pod by convention) see no behavior change.
    "lease": {"session": None, "expires": 0.0, "busy": False,
              # FIFO waiters: {"session", "last_poll"}. A waiter that stops polling
              # (client died / gave up) is pruned after _QUEUE_WAITER_TTL_S.
              "queue": []},
    "lease_lock": threading.Lock(),  # NOT STATE["lock"]: /acquire must answer while a turn runs
}

# A live long-poller refreshes last_poll every ~2s; 45s bounds how long a stale
# HEAD entry (client died / switched engines without cancelling) can block the
# FIFO grant to the next waiter.
_QUEUE_WAITER_TTL_S = 45.0
_LEASE_MAX_QUEUE = int(os.environ.get("CAPX_LEASE_MAX_QUEUE", "8"))


def _queue_prune(now: float) -> None:
    """Caller holds lease_lock. Drop waiters whose client stopped polling."""
    q = STATE["lease"]["queue"]
    q[:] = [w for w in q if now - w["last_poll"] < _QUEUE_WAITER_TTL_S]


def _lease_active(now: float) -> bool:
    ls = STATE["lease"]
    return ls["session"] is not None and (ls["busy"] or ls["expires"] > now)


def _lease_status() -> dict:
    ls = STATE["lease"]
    now = time.time()
    active = _lease_active(now)
    return {"leased_by": ls["session"] if active else None,
            "lease_expires_in_s": round(max(0.0, ls["expires"] - now), 1) if active else 0.0,
            "lease_turn_running": bool(ls["busy"]) if active else False,
            "queue_len": len(ls["queue"])}


def lease_acquire(session: str, ttl_s: float, wait_s: float = 0.0) -> tuple[int, dict]:
    """Grant the lease to `session`, long-polling in FIFO order for up to `wait_s`.
    Returns 200 granted / 409 still-busy (queue position kept alive; re-poll to keep
    it) / 429 queue full (try a sibling engine). A lease is dead only when its TTL
    lapsed AND no turn of it is executing (a turn can legitimately run for hours;
    the blocked client cannot renew -- never reclaim mid-turn)."""
    deadline = time.time() + min(float(wait_s), 300.0)
    while True:
        with STATE["lease_lock"]:
            ls = STATE["lease"]
            now = time.time()
            _queue_prune(now)
            q = ls["queue"]
            if _lease_active(now) and ls["session"] == session:  # renew own lease
                q[:] = [w for w in q if w["session"] != session]
                ls["expires"] = now + float(ttl_s)
                return 200, {"ok": True, "session": session, "ttl_s": float(ttl_s)}
            mine = next((w for w in q if w["session"] == session), None)
            if not _lease_active(now) and (mine is None and not q or
                                           q and q[0]["session"] == session):
                # Engine free and it is our turn (FIFO): grant.
                if mine is not None:
                    q.remove(mine)
                ls["session"], ls["busy"] = session, False
                ls["expires"] = now + float(ttl_s)
                return 200, {"ok": True, "session": session, "ttl_s": float(ttl_s)}
            if mine is None:
                if len(q) >= _LEASE_MAX_QUEUE:
                    return 429, {"ok": False, "error": "lease queue full",
                                 **_lease_status()}
                q.append({"session": session, "last_poll": now})
            else:
                mine["last_poll"] = now
            position = next(i for i, w in enumerate(q) if w["session"] == session)
        if time.time() >= deadline:
            return 409, {"ok": False, "error": "engine busy; queued",
                         "queue_position": position, **_lease_status()}
        time.sleep(2.0)


def lease_release(session: str) -> dict:
    """Release a held lease AND/OR cancel the session's queue membership (a client
    that switches to a sibling engine cancels here so its stale entry never blocks
    the FIFO head)."""
    with STATE["lease_lock"]:
        ls = STATE["lease"]
        ls["queue"][:] = [w for w in ls["queue"] if w["session"] != session]
        if ls["session"] == session:
            ls["session"], ls["expires"], ls["busy"] = None, 0.0, False
            return {"ok": True, "released": True}
        return {"ok": True, "released": False, **_lease_status()}


def lease_check_and_touch(session: str | None) -> tuple[bool, dict]:
    """Gate a /run_policy request against the active lease. Requests carrying the
    owning session (or any request while NO lease is active -- eval-driver compat)
    pass; the owner's TTL is renewed and the in-flight turn marked."""
    with STATE["lease_lock"]:
        ls = STATE["lease"]
        now = time.time()
        active = ls["session"] is not None and (ls["busy"] or ls["expires"] > now)
        if active and session != ls["session"]:
            return False, {"ok": False, "error": "pod leased by another session",
                           **_lease_status()}
        if active:
            ls["expires"] = now + max(600.0, ls["expires"] - now)
            ls["busy"] = True
        return True, {}


def lease_turn_done(session: str | None) -> None:
    with STATE["lease_lock"]:
        ls = STATE["lease"]
        if ls["session"] is not None and session == ls["session"]:
            ls["busy"] = False
            ls["expires"] = time.time() + 1800.0  # post-turn grace for the next turn/release


def do_switch(new_env: str, session: str) -> tuple[int, dict]:
    """Switch this engine to `new_env` by PROCESS RESTART: write the new env (+ the
    requesting session as a boot-time lease reservation) to --env-file and hard-exit;
    the pod wrapper relaunches us on the new scene (~2-3 min on warm kit caches).
    In-process scene rebuild is deliberately NOT attempted: kit teardown hangs inside
    env.close() are a documented repo failure mode (robobench/scripts/smoke.py) --
    the restart IS the teardown. Allowed for the lease holder or on a free engine."""
    if not ARGS.env_file:
        return 400, {"ok": False, "error": "engine not switchable (no --env-file)"}
    with STATE["lease_lock"]:
        now = time.time()
        ls = STATE["lease"]
        if _lease_active(now) and ls["session"] != session:
            return 409, {"ok": False, "error": "pod leased by another session",
                         **_lease_status()}
    with open(ARGS.env_file, "w") as f:
        f.write(new_env + "\n" + session + "\n")
    print(f"[server] SWITCH -> {new_env} (reserved for {session}); hard-exiting for "
          "wrapper relaunch", flush=True)
    threading.Timer(1.0, lambda: os._exit(0)).start()  # reply first, then die
    return 200, {"ok": True, "switching_to": new_env,
                 "note": "engine rebooting; poll /ping until booted with the new activity"}


def _boot_reservation() -> None:
    """If the env-file names a reserving session (a /switch requester), lease the
    engine to it for 10 minutes at boot so its episode cannot be stolen while the
    client polls for the reboot to finish."""
    try:
        if not ARGS.env_file or not os.path.exists(ARGS.env_file):
            return
        lines = [ln.strip() for ln in open(ARGS.env_file)]
        if len(lines) >= 2 and lines[1]:
            code, obj = lease_acquire(lines[1], ttl_s=600.0)
            print(f"[server] boot lease reserved for {lines[1]}: {code} {obj}", flush=True)
            # One-shot: clear the reservation line so an ordinary crash/relaunch
            # does not re-reserve for a long-gone session.
            with open(ARGS.env_file, "w") as f:
                f.write(lines[0] + "\n")
    except Exception:  # noqa: BLE001 -- reservation is best-effort
        traceback.print_exc()


def boot(env_name: str, max_steps: int) -> dict:
    env, api = L.build_env(env_name, max_steps, num_envs=int(ARGS.num_envs))
    prompt = L.make_prompt(env, api)
    print(f"[server] booted env={env_name} action_dim={env.robot.action_dim} "
          f"num_envs={env.num_envs}", flush=True)
    camera = L.build_record_camera(env)  # None if it fails -> video off, grading unaffected
    api.attach_camera(camera)  # for mid-episode look()
    return {"env": env, "api": api, "prompt": prompt, "camera": camera}


# Files the dev loop can hot-push; /reload re-fetches ALL of them then reimports
# in dependency order (2026-07-26 reorg: the cosigen_loop monolith became purpose
# modules; the aggregator must be reloaded LAST so it re-exports the fresh ones).
_RELOAD_FILES = ["cosigen_config.py", "cosigen_prompts.py", "cosigen_view.py",
                 "cosigen_apis.py", "cosigen_session.py", "cosigen_opt.py",
                 "cosigen_loop.py"]


def do_reload() -> dict:
    """Hot-reload the harness modules on the LIVE booted env -- re-download the pushed
    files, reimport them in dependency order, and rebuild the control API + prompt on
    the existing env. No Isaac reboot, so a harness edit -> new behavior takes
    ~seconds. Runs on the main thread (via the job queue) since it mutates ctx the
    renderer uses."""
    import importlib
    src_dir = os.path.dirname(ARGS.reload_src)
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in _RELOAD_FILES:
        src = f"{src_dir}/{fname}"
        rc = os.system(f"hdfs dfs -get -f {src} {os.path.join(here, fname)}")
        if rc != 0 and fname == "cosigen_loop.py":  # loop is mandatory
            return {"ok": False, "error": f"hdfs get of {src} failed rc={rc}"}
    import glob
    import sys as _sys
    # Stale-bytecode guard: a shipped/old __pycache__ pyc can shadow the freshly downloaded
    # source (importlib may trust the cache) -- drop pycs + invalidate before reloading.
    for pyc in glob.glob(os.path.join(here, "__pycache__", "*.pyc")):
        try:
            os.remove(pyc)
        except OSError:
            pass
    importlib.invalidate_caches()
    # dependency order; missing modules (older pods) just skip
    for mod in ("cosigen_config", "cosigen_prompts", "cosigen_view",
                "cosigen_apis", "cosigen_session", "cosigen_opt"):
        if mod in _sys.modules:
            importlib.reload(_sys.modules[mod])
    importlib.reload(L)
    ctx = STATE["ctx"]
    env = ctx["env"]
    ctx["api"] = L.resolve_api(STATE["activity"])(env, max_steps=int(ARGS.max_steps))
    ctx["api"].attach_camera(ctx.get("camera"))
    ctx["prompt"] = L.make_prompt(env, ctx["api"])
    STATE["session_ns"] = None  # api objects changed; stale closures would misbehave
    print("[server] HARNESS RELOADED from", src_dir, flush=True)
    return {"ok": True, "reloaded_from": src_dir, "prompt_len": len(ctx["prompt"] or "")}


def _meta_payload() -> dict:
    c = STATE["ctx"] or {}
    api = c.get("api")
    return {
        "ok": True,
        "booted": STATE["booted"],
        "boot_error": STATE["boot_error"],
        "activity": STATE["activity"],
        "n_runs": STATE["n_runs"],
        "num_envs": int(ARGS.num_envs),
        "session_active": STATE["session_ns"] is not None,
        "prompt": c.get("prompt"),
        "switchable": bool(ARGS.env_file),
        # Live parameter-search progress. This handler runs on an HTTP thread, so it
        # answers WHILE a multi-generation search occupies the main thread — the
        # driver polls it to stream results to the agent (async optimize tool).
        "optimize_progress": getattr(api, "_opt_progress", None) if api else None,
        **_lease_status(),
    }


def _upload_frames_file(local: str) -> str:
    """Upload an existing local frames npz (a zero-truncation chunk) to HDFS; returns
    the HDFS path ('' on failure — the local file is KEPT either way, no loss)."""
    try:
        dst = ARGS.frames_hdfs_dir.rstrip("/") + "/" + os.path.basename(local)
        rc = os.system(f"hdfs dfs -mkdir -p {ARGS.frames_hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {local} {dst}")
        return dst if rc == 0 else ""
    except Exception:
        traceback.print_exc()
        return ""


def _save_frames_npz(frames) -> str:
    """Save (T,H,W,3) uint8 RGB frames to a compressed npz on HDFS; return the HDFS path ("" on
    failure/empty). numpy-only on the GPU side -- mp4 encoding happens off-box."""
    import sys as _sys
    try:
        import uuid

        import numpy as np
        if frames is None or getattr(frames, "size", 0) == 0:
            return ""
        local = f"/tmp/cosigen_frames_{uuid.uuid4().hex[:10]}.npz"
        np.savez_compressed(local, frames=frames, env=STATE["activity"])
        dst = ARGS.frames_hdfs_dir.rstrip("/") + "/" + os.path.basename(local)
        rc = os.system(f"hdfs dfs -mkdir -p {ARGS.frames_hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {local} {dst}")
        if rc != 0:
            print(f"[server] frames hdfs put failed rc={rc} dst={dst}", file=_sys.stderr, flush=True)
            return ""
        return dst
    except Exception:
        import traceback as _tb
        print("save_frames error:\n" + _tb.format_exc(), file=_sys.stderr, flush=True)
        return ""


def _record_frames_manifest(api, frames_parts: list[dict], n_frames: int) -> None:
    """Append this turn's uploaded frame parts to the session's durable video manifest
    (cosigen_sessions/<sid>/frames_manifest.jsonl). scripts/cosigen_session_video.py
    renders the COMPLETE session video from it — no manual per-turn stitching, and the
    footage index survives driver crashes/restarts (2026-07-24)."""
    if not frames_parts:
        return
    sid = getattr(api, "_persist_session_id", "") or ""
    root = getattr(api, "_persist_hdfs_root", "") or ""
    if not sid or not root:
        return
    try:
        local = f"/tmp/frames_manifest_{sid}.jsonl"
        remote = f"{root}/{sid}/frames_manifest.jsonl"
        # A new render-server process has an empty /tmp. Seed it from HDFS before
        # appending, otherwise `put -f` silently erases all earlier video lineage.
        if not os.path.exists(local):
            subprocess.run(["hdfs", "dfs", "-get", "-f", remote, local],
                           capture_output=True, check=False)
        with open(local, "a") as f:
            f.write(json.dumps({"turn": getattr(api, "_turn_no", None),
                                "n_frames": int(n_frames), "parts": frames_parts,
                                "epoch": getattr(api, "_rec_epoch", None),
                                "t": time.time()}) + "\n")
        os.system(f"hdfs dfs -mkdir -p {root}/{sid} 2>/dev/null; "
                  f"hdfs dfs -put -f {local} {remote}")
    except Exception:
        traceback.print_exc()  # manifest is best-effort; never fail the turn


def do_run(code: str, max_steps: int, num_frames: int, reset: bool = True,
           meta: dict | None = None) -> dict:
    ctx = STATE["ctx"]
    env, api = ctx["env"], ctx["api"]
    camera = ctx.get("camera")
    record = int(num_frames) > 0 and camera is not None
    t0 = time.time()
    frames_path = ""
    n_frames = 0
    with STATE["lock"]:
        print(f"[do_run] start record={record} reset={reset}", flush=True)
        if reset or STATE["session_ns"] is None:
            # New episode + fresh session namespace (turn 1 of a session).
            L.reset_episode(env, api)
            STATE["session_ns"] = L.make_namespace(env, api)
            # Dense capture (watchable, near-real-time video): every 2 CONTROL steps.
            # Sparse 15-20-step sampling made every motion look like teleportation
            # (user review 2026-07-21). No frame caps: every captured frame is kept.
            every = int(os.environ.get("CAPX_VIDEO_EVERY", "2"))
            api.enable_recording(camera if record else None, every=every)
        # Harness->api side channel (e.g. out-of-trajectory edge annotations), generic like
        # drain_turn_payload so future needs don't require a server change.
        if meta and hasattr(api, "apply_turn_meta"):
            try:
                api.apply_turn_meta(meta)
            except Exception:
                traceback.print_exc()
        tp = time.time()
        rc, stdout, stderr = L.run_policy(env, api, code, ns=STATE["session_ns"])
        n_cap = len(getattr(api, "_rec_frames", []))
        print(f"[do_run] policy done rc={rc} dt={time.time()-tp:.1f}s captured={n_cap}", flush=True)
        success = bool(L.check_success(env, api))
        summary = L.scene_summary(env, api)
        event_log = api.drain_event_log()
        frames_parts: list[str] = []
        manifest_parts: list[dict] = []
        if hasattr(api, "drain_turn_frame_parts"):
            # Zero-truncation recording: this turn's footage is a series of npz part
            # files on local disk (nothing capped, nothing dropped). Upload every part.
            for part in api.drain_turn_frame_parts():
                dst = _upload_frames_file(part["path"])
                if dst:
                    frames_parts.append(dst)
                    manifest_parts.append({"path": dst, "start": int(part["start"]),
                                           "n": int(part["n"])})
                n_frames += int(part["n"])
            frames_path = frames_parts[-1] if frames_parts else ""
            if n_frames:
                print(f"[do_run] frames this turn n_frames={n_frames} "
                      f"parts={len(frames_parts)}", flush=True)
            _record_frames_manifest(api, manifest_parts, n_frames)
        elif record or getattr(api, "_rec_frames", None):
            frames = api.collect_frames()  # legacy api without chunked storage
            n_frames = int(frames.shape[0])
            if n_frames:
                frames_path = _save_frames_npz(frames)
        STATE["n_runs"] += 1
    print(f"[server] run #{STATE['n_runs']}: rc={rc} success={success} "
          f"n_frames={n_frames} frames={frames_path} dt={time.time()-t0:.1f}s", flush=True)
    # Real-time playback rate: control steps happen at 1/(control_period*dt) Hz and we
    # captured every `_rec_every` of them — encoding at this fps plays back at true
    # sim speed (the fixed fps=20 encode of sparse captures played ~26x fast).
    try:
        hz = 1.0 / (env.robot.control_period * env.dt)
        video_fps = max(2, round(hz / max(1, getattr(api, "_rec_every", 2))))
    except Exception:
        video_fps = 8
    result = {
        "rc": rc, "stdout": stdout, "stderr": stderr, "success": bool(success),
        "scene_summary": summary, "event_log": event_log, "n_frames": n_frames,
        "feedback_frames": [], "frames_npz": frames_path,
        "frames_npz_parts": frames_parts,  # ALL of this turn's footage, in order
        "video_fps": video_fps,
        "render_seconds": round(time.time() - t0, 1),
    }
    # Generic api extension hook (e.g. look()'s turn_images) -- future api-side additions
    # flow through without a server change/reboot.
    try:
        if hasattr(api, "drain_turn_payload"):
            result.update(api.drain_turn_payload() or {})
    except Exception:
        traceback.print_exc()
    STATE["last_result"] = {"n_run": STATE["n_runs"], "result": result}
    return result


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/ping", "/", "/meta"):
            self._send(200, _meta_payload())
        elif path == "/last_result":
            lr = STATE.get("last_result")
            if lr:
                self._send(200, {"ok": True, **lr})
            else:
                self._send(404, {"ok": False, "error": "no completed run yet"})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path in ("/acquire", "/release", "/switch"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                session = str(req.get("session") or "")
                assert session, "missing session"
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"ok": False, "error": f"bad request: {exc!r}"})
                return
            if path == "/acquire":
                code, obj = lease_acquire(session, float(req.get("ttl_s", 1800.0)),
                                          wait_s=float(req.get("wait_s", 0.0)))
                self._send(code, obj)
            elif path == "/switch":
                code, obj = do_switch(str(req.get("env") or ""), session)
                self._send(code, obj)
            else:
                self._send(200, lease_release(session))
            return
        if path not in ("/run_policy", "/reload"):
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not STATE["booted"]:
            self._send(503, {"ok": False, "error": "env still booting", "boot_error": STATE["boot_error"]})
            return
        if path == "/reload":
            # Hot-reload the harness on the main thread (mutates ctx the renderer uses).
            job = {"kind": "reload", "done": threading.Event(), "result": None}
            STATE["jobs"].put(job)
            if not job["done"].wait(timeout=120.0):
                self._send(504, {"ok": False, "error": "reload timed out"})
                return
            res = job["result"] or {}
            self._send(200 if res.get("ok") else 500, res)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"ok": False, "error": f"bad request: {exc!r}"})
            return
        code = req.get("code", "")
        if not code.strip():
            self._send(400, {"ok": False, "error": "empty code"})
            return
        # Lease gate: while an RL rollout holds this pod, other sessions bounce with 409
        # (no lease active -> everything passes; eval-driver compat).
        session = req.get("session") or None
        ok, err = lease_check_and_touch(session)
        if not ok:
            self._send(409, err)
            return
        # Hand the work to the MAIN thread (kit rendering requires it) and wait for the result.
        job = {"kind": "run", "code": code, "max_steps": int(req.get("max_steps", 3000)),
               "num_frames": int(req.get("num_frames", 0)),
               "reset": bool(req.get("reset", True)),
               "meta": req.get("meta") or None,
               "session": session,
               "done": threading.Event(), "result": None}
        STATE["jobs"].put(job)
        if not job["done"].wait(timeout=float(os.environ.get("CAPX_RUN_TIMEOUT_S", "1800"))):
            self._send(504, {"ok": False, "error": "run timed out on main-thread queue"})
            return
        res = job["result"] or {}
        if res.get("ok"):
            self._send(200, res)
        else:
            self._send(500, res)


class V6Server(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True
    # Long-poll /acquire holds a connection per queued waiter; the default accept
    # backlog (5) drops bursts exactly like the BEHAVIOR pool's request_queue_size
    # saturation did. Size it for waiters + turns + pings.
    request_queue_size = 128


def register(registry: str, url: str) -> None:
    if not registry:
        return
    # Port-unique temp file: multiple engine processes on one pod register
    # concurrently; a SHARED temp file races (engine 1's upload shipped engine 0's
    # URL on the first 2-engine probe pod, 2026-07-24).
    local = f"/tmp/_cosigen_endpoint_{ARGS.port}.txt"
    with open(local, "w") as f:
        f.write(url + "\n")
    os.system(f"hdfs dfs -mkdir -p {os.path.dirname(registry)} 2>/dev/null; "
              f"hdfs dfs -put -f {local} {registry}")
    print(f"[server] registered endpoint {url} -> {registry}", flush=True)


def _remove_driver_source() -> None:
    """Driver-side modules ship in the shared tarball (the RL rollout workers need them)
    but never execute on a render pod, and a stale copy here is pure misdirection: after
    a driver-side failure, a v12 agent spent ~10 runs studying this file on the pod as if
    it were the code that failed it (2026-07-25). Remove it from the pod at boot."""
    for name in ("cosigen_agentic_loop.py",):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        try:
            os.remove(path)
            print(f"[server] removed driver-side source {path}", flush=True)
        except FileNotFoundError:
            pass
        except Exception:
            traceback.print_exc()


def main() -> None:
    _remove_driver_source()
    host = ARGS.advertise_host or socket.gethostname()
    hostpart = f"[{host}]" if ":" in host else host
    url = f"http://{hostpart}:{ARGS.port}"

    httpd = V6Server(("::", ARGS.port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[server] HTTP up on :: port {ARGS.port}; advertising {url}", flush=True)
    _boot_reservation()  # /switch requester gets the engine before anyone else
    register(ARGS.registry, url)

    try:
        STATE["ctx"] = boot(ARGS.env, ARGS.max_steps)
        STATE["booted"] = True
        register(ARGS.registry, url)
        print("[server] READY_FOR_RUN", flush=True)
    except Exception as exc:  # noqa: BLE001
        STATE["boot_error"] = repr(exc)
        traceback.print_exc()
        print(f"[server] BOOT_FAILED: {exc!r}", flush=True)

    # Main-thread work loop: execute queued /run_policy jobs here so all sim + RENDER calls run
    # on the thread that launched the kit app (rendering from a worker thread hangs).
    while True:
        try:
            job = STATE["jobs"].get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            if job.get("kind") == "reload":
                job["result"] = do_reload()
            else:
                job["result"] = do_run(job["code"], job["max_steps"], job["num_frames"],
                                       reset=job.get("reset", True), meta=job.get("meta"))
                job["result"]["ok"] = True
        except Exception as exc:  # noqa: BLE001
            job["result"] = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
        finally:
            if job.get("kind") != "reload":
                lease_turn_done(job.get("session"))
            job["done"].set()


if __name__ == "__main__":
    main()
