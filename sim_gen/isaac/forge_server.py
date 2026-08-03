"""Forge server: a warm L20 pod that runs robobench-format task smokes on demand.

One construction agent <-> one forge. The agent uploads its task package (scene.py /
configs / smoke.py, robobench conventions) and triggers smoke runs; each run is a fresh
subprocess of the pod's Isaac python (app boots per run — ~3-4 min warm — giving process
isolation: no registry collisions, no Kit teardown leaks). The pod itself never imports
Isaac.

Endpoints:
  GET  /ping                     {ok, busy, n_runs, workdir}
  POST /submit {task, files}     write files under simgen_tasks/<task>/ (whole package)
  POST /run    {task, module="smoke", args=[], timeout_s=900}
                                 run `python -u -m simgen_tasks.<task>.<module> ...`
                                 -> {rc, seconds, stdout_tail, all_pass}
  GET  /fetch?path=<rel>         file from the workdir, base64 (videos/npz)

Deployed by scripts (launch_forge_pool.py) which stages the Isaac env + CoSiGen repo
and registers this server's URL to HDFS.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6  # Arnold pods are IPv6; default AF_INET can't bind '::'
from pathlib import Path

ISAAC_PY = "/home/tiger/isaaclab_build/env_isaaclab/bin/python"
COSIGEN = "/home/tiger/CoSiGen"

STATE = {"n_runs": 0, "busy": False}
RUN_LOCK = threading.Lock()


def _tail(s: str, n: int = 12000) -> str:
    return s[-n:] if len(s) > n else s


class Handler(BaseHTTPRequestHandler):
    workdir: Path  # set at startup

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/ping":
            self._send(200, {"ok": True, "busy": STATE["busy"], "n_runs": STATE["n_runs"],
                             "workdir": str(self.workdir)})
            return
        if path == "/fetch":
            from urllib.parse import parse_qs, urlparse
            rel = parse_qs(urlparse(self.path).query).get("path", [""])[0]
            target = (self.workdir / rel).resolve()
            if not str(target).startswith(str(self.workdir.resolve())) or not target.is_file():
                self._send(404, {"ok": False, "error": f"no such workdir file: {rel}"})
                return
            data = target.read_bytes()
            self._send(200, {"ok": True, "path": rel, "size": len(data),
                             "b64": base64.b64encode(data).decode()})
            return
        self._send(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"ok": False, "error": f"bad json: {exc!r}"})
            return
        if self.path == "/submit":
            task = str(req.get("task", "")).strip()
            files = req.get("files") or {}
            if not task.isidentifier() or not files:
                self._send(400, {"ok": False, "error": "need identifier task + files"})
                return
            pkg = self.workdir / "simgen_tasks" / task
            pkg.mkdir(parents=True, exist_ok=True)
            (self.workdir / "simgen_tasks" / "__init__.py").touch()
            written = []
            for rel, content in files.items():
                dest = (pkg / rel).resolve()
                if not str(dest).startswith(str(pkg.resolve())):
                    self._send(400, {"ok": False, "error": f"path escapes package: {rel}"})
                    return
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
                written.append(rel)
            (pkg / "__init__.py").touch()
            self._send(200, {"ok": True, "task": task, "written": written})
            return
        if self.path == "/run":
            task = str(req.get("task", "")).strip()
            module = str(req.get("module", "smoke"))
            args = [str(a) for a in (req.get("args") or [])]
            timeout_s = min(float(req.get("timeout_s", 900)), 3600.0)
            if not (self.workdir / "simgen_tasks" / task).is_dir():
                self._send(404, {"ok": False, "error": f"task {task!r} not submitted"})
                return
            if not RUN_LOCK.acquire(blocking=False):
                self._send(409, {"ok": False, "error": "forge busy with another run"})
                return
            try:
                STATE["busy"] = True
                res = self._run(task, module, args, timeout_s)
                STATE["n_runs"] += 1
                self._send(200, res)
            finally:
                STATE["busy"] = False
                RUN_LOCK.release()
            return
        self._send(404, {"ok": False, "error": "unknown endpoint"})

    def _run(self, task: str, module: str, args: list[str], timeout_s: float) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{self.workdir}:{COSIGEN}:" + env.get("PYTHONPATH", "")
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        cmd = [ISAAC_PY, "-u", "-m", f"simgen_tasks.{task}.{module}", *args]
        print(f"[run] {' '.join(cmd)} (timeout {timeout_s:.0f}s)", flush=True)
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(self.workdir), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            # Kit shutdown hangs are routine — kill the whole process group.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            out, _ = proc.communicate()
            rc, out = -9, (out or "") + f"\n[forge] TIMEOUT after {timeout_s:.0f}s — killed"
        seconds = round(time.time() - t0, 1)
        print(f"[run] done rc={rc} in {seconds}s", flush=True)
        return {"ok": True, "rc": rc, "seconds": seconds,
                "all_pass": "SIM_GEN_SMOKE: ALL PASS" in (out or "")
                            or "ALL PASS" in (out or "").splitlines()[-1:][0] if out else False,
                "stdout_tail": _tail(out or "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workdir", default="/home/tiger/forge_work")
    ap.add_argument("--registry", default="", help="HDFS file to register this URL to")
    args = ap.parse_args()

    Handler.workdir = Path(args.workdir)
    Handler.workdir.mkdir(parents=True, exist_ok=True)
    (Handler.workdir / "simgen_tasks").mkdir(exist_ok=True)
    (Handler.workdir / "simgen_tasks" / "__init__.py").touch()

    if args.registry:
        host = os.environ.get("MY_HOST_IPV6") or socket.getfqdn()
        url = f"http://[{host}]:{args.port}" if ":" in host else f"http://{host}:{args.port}"
        local = "/tmp/forge_url.txt"
        Path(local).write_text(url + "\n")
        subprocess.run(["hdfs", "dfs", "-put", "-f", local, args.registry], check=False)
        print(f"[forge] registered {url} -> {args.registry}", flush=True)

    print(f"[forge] serving on :{args.port}, workdir {Handler.workdir}", flush=True)
    V6Server(("::", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
