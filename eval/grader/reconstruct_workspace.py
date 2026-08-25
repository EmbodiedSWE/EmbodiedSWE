#!/usr/bin/env python3
"""Reconstruct the agent's /workspace over time from its FULL trajectory, and enumerate
every distinct version of the delivered program (solve.py + its import closure).

Mutation sources, applied IN ORDER within each turn (persistent cwd tracked across turns
for Claude's stateful shell; codex supplies an explicit workdir per call):

  Claude: Write / Edit tool calls (structured, exact)
          Bash: cat >/>> heredocs, rm/mv/cp (incl. from /submissions), cd tracking,
                `python - <<EOF` string-surgery scripts EXECUTED in a sandbox built from
                the current VFS ( /workspace and /submissions rewritten to sandbox paths),
                so the file state after each edit script is exact by construction.
  Codex:  apply_patch (*** Add/Update/Delete File) with @@ anchors + forward cursor,
          the same shell parsing for heredocs/rm/mv/cp/python-heredocs.

Ground-truth resync: at every recorded submission time (submissions/NN/submitted.json,
wall_s), the VFS solution/ is compared to the actual snapshot and overwritten by it —
divergences are counted in the fidelity report, and drift between submissions can never
survive past the next boundary. The final state is additionally compared byte-for-byte
against the run's mirrored final workspace by check_final().

Output per run (results/<label>/versions/):
    v0001/ ...      materialized full workspaces
    versions.json   [{version, ts_ms, wall_min, closure_hash}]
    fidelity.json   counters + resync divergences + unparsed-opaque commands
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WS = "/workspace"


# ------------------------------------------------------------------ path normalization
def norm(path: str, cwd: str) -> str | None:
    """workspace-relative path, or None if outside /workspace."""
    path = path.strip().strip('"\'')
    if not path.startswith("/"):
        path = (Path(cwd) / path).as_posix()
    path = re.sub(r"/{2,}", "/", path)
    parts = []
    for seg in path.split("/"):
        if seg == "..":
            if parts:
                parts.pop()
        elif seg not in ("", "."):
            parts.append(seg)
    path = "/" + "/".join(parts)
    if path == WS:
        return None
    if path.startswith(WS + "/"):
        return path[len(WS) + 1:]
    return None


# ------------------------------------------------------------------ virtual filesystem
class VFS:
    def __init__(self, submissions_dir: Path | None):
        self.files: dict[str, str] = {}
        self.cwd = WS
        self.subs = submissions_dir  # actual snapshots pulled from the volume
        self.history: dict[str, set] = {}  # every content hash a file has EVER had
        self.fidelity = {"writes": 0, "edits": 0, "edit_mismatches": [], "heredocs": 0,
                         "patches": 0, "patch_mismatches": [], "py_exec_ok": 0,
                         "py_exec_fail": 0, "opaque_cmds": 0, "resync_divergences": [],
                         "resync_timing_skips": 0, "unquoted_heredoc_dollar": 0}

    # NOTE: all content comparisons are insensitive to TRAILING newlines: codex
    # apply_patch's end-of-file newline handling is not reproducible bit-exactly (measured
    # both with and without a final blank line on disk), and trailing newlines cannot
    # change Python semantics. Nothing else is normalized.
    @staticmethod
    def _key(content: str) -> str:
        return hashlib.sha256(content.rstrip("\n").encode()).hexdigest()

    def _remember(self, path: str, content: str):
        self.history.setdefault(path, set()).add(self._key(content))

    def seen_before(self, path: str, content: str) -> bool:
        return self._key(content) in self.history.get(path, set())

    def _set(self, p: str, content: str):
        """Set a file, keeping the tree consistent: a path cannot be both a file and a
        directory (agents create both over time, e.g. `v3` the file then `v3/solve.py`)."""
        for k in [k for k in self.files if k.startswith(p + "/")]:
            del self.files[k]          # p was a directory before; it is a file now
        parts = p.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            if parent in self.files:
                del self.files[parent]  # a parent segment was a file before
        self.files[p] = content

    # -- primitive ops -------------------------------------------------------------
    def write(self, path: str, content: str, cwd: str | None = None):
        p = norm(path, cwd or self.cwd)
        if p is not None:
            self._set(p, content)
            self.fidelity["writes"] += 1

    def append(self, path: str, content: str, cwd: str | None = None):
        p = norm(path, cwd or self.cwd)
        if p is not None:
            self._set(p, self.files.get(p, "") + content)

    def edit(self, path: str, old: str, new: str, replace_all: bool):
        p = norm(path, self.cwd)
        if p is None:
            return
        cur = self.files.get(p)
        if cur is None or old not in cur:
            self.fidelity["edit_mismatches"].append(p)
            return
        self.files[p] = cur.replace(old, new) if replace_all else cur.replace(old, new, 1)
        self.fidelity["edits"] += 1

    def rm(self, path: str, cwd: str):
        p = norm(path, cwd)
        if p is None:
            return
        if p in self.files:
            del self.files[p]
        for k in [k for k in self.files if k.startswith(p.rstrip("/") + "/")]:
            del self.files[k]

    # -- sources that may live outside /workspace (submissions snapshots) ----------
    def _read_source(self, path: str, cwd: str) -> dict[str, str] | None:
        """Resolve a cp/mv source (may contain a *.py glob) to {relative_name: content}."""
        path = path.strip().strip('"\'')
        if not path.startswith("/"):
            path = (Path(cwd) / path).as_posix()
        if path.startswith("/submissions/") and self.subs is not None:
            rel = path[len("/submissions/"):]
            if "*" in rel:
                d, pat = rel.rsplit("/", 1)
                base = self.subs / d
                if base.is_dir():
                    return {f.name: f.read_text(errors="replace")
                            for f in sorted(base.glob(pat)) if f.is_file()}
                return None
            f = self.subs / rel
            if f.is_file():
                return {f.name: f.read_text(errors="replace")}
            if f.is_dir():
                return {x.name: x.read_text(errors="replace")
                        for x in sorted(f.iterdir()) if x.is_file()}
            return None
        p = norm(path, cwd)
        if p is None:
            return None
        if "*" in p:
            d, pat = p.rsplit("/", 1) if "/" in p else ("", p)
            rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
            out = {}
            for k, v in self.files.items():
                kd, kn = k.rsplit("/", 1) if "/" in k else ("", k)
                if kd == d and rx.match(kn):
                    out[kn] = v
            return out or None
        if p in self.files:
            return {p.rsplit("/", 1)[-1]: self.files[p]}
        if any(k.startswith(p + "/") for k in self.files):  # directory
            return {k[len(p) + 1:]: v for k, v in self.files.items() if k.startswith(p + "/")}
        return None

    def cp(self, srcs: list[str], dst: str, cwd: str):
        gathered: dict[str, str] = {}
        for s in srcs:
            got = self._read_source(s, cwd)
            if got:
                gathered.update(got)
        if not gathered:
            return
        dst = dst.strip().strip('"\'')
        if not dst.startswith("/"):
            dst = (Path(cwd) / dst).as_posix()
        d = norm(dst, cwd)
        if d is None:
            return
        many = len(gathered) > 1
        dst_is_dir = many or d in ("solution",) or any(
            k.startswith(d + "/") for k in self.files) or dst.endswith("/") or d == ""
        for name, content in gathered.items():
            self._set((d + "/" + name) if (dst_is_dir or many) else d, content)

    # -- executing the agent's own python edit scripts ------------------------------
    def run_python_heredoc(self, script: str, cwd: str):
        """Materialize the VFS in a temp sandbox, run the agent's edit script with
        /workspace and /submissions rewritten to sandbox paths, absorb file changes."""
        try:
            with tempfile.TemporaryDirectory(prefix="vfsx_") as td:
                root = Path(td)
                ws = root / "workspace"
                for rel, content in self.files.items():
                    f = ws / rel
                    try:
                        f.parent.mkdir(parents=True, exist_ok=True)
                        f.write_text(content)
                    except (FileExistsError, NotADirectoryError, OSError):
                        # a path segment collides with an existing file (e.g. agent files
                        # named with literal shell variables like '$SUB/$name')
                        continue
                (ws / "solution").mkdir(parents=True, exist_ok=True)
                if self.subs is not None and self.subs.is_dir():
                    shutil.copytree(self.subs, root / "submissions", dirs_exist_ok=True)
                else:
                    (root / "submissions").mkdir(exist_ok=True)
                body = script.replace(WS, str(ws)).replace(
                    "/submissions", str(root / "submissions"))
                run_cwd = cwd.replace(WS, str(ws)).replace(
                    "/submissions", str(root / "submissions"))
                if not run_cwd.startswith(str(root)):
                    run_cwd = str(ws)  # script cwd escaped the sandbox (e.g. /bench)
                Path(run_cwd).mkdir(parents=True, exist_ok=True)
                try:
                    r = subprocess.run([sys.executable, "-"], input=body, cwd=run_cwd,
                                       capture_output=True, text=True, timeout=20)
                    ok = r.returncode == 0
                except Exception:  # noqa: BLE001
                    ok = False
                self.fidelity["py_exec_ok" if ok else "py_exec_fail"] += 1
                # absorb every text file now under the sandbox workspace
                new_files = {}
                for f in ws.rglob("*"):
                    if not f.is_file():
                        continue
                    try:
                        new_files[f.relative_to(ws).as_posix()] = f.read_text()
                    except (UnicodeDecodeError, OSError):
                        pass  # binary artifact from a failed local run; not a source file
                self.files = new_files
        except Exception as exc:  # noqa: BLE001 -- a weird script must not kill the run
            self.fidelity["py_exec_fail"] += 1
            print(f"[vfs] pyexec sandbox error: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    # -- one shell command, operations applied in order ------------------------------
    def shell(self, cmd: str, workdir: str | None = None):
        cwd = workdir or self.cwd
        ops: list[tuple[int, tuple]] = []
        for m in re.finditer(r"(?:^|&&|;|\|\||\n)\s*cd\s+(\S+)", cmd):
            ops.append((m.start(), ("cd", m.group(1))))
        for m in re.finditer(
                r"cat\s*(>>?)\s*(?:\"([^\"]+)\"|'([^']+)'|(\S+))\s*<<-?\s*(?:'(\w+)'|\"(\w+)\"|(\w+))\n(.*?)\n(?:\5|\6|\7)(?=\n|$)",
                cmd, re.DOTALL):
            body = m.group(8)
            if m.group(7) and "$" in body:  # unquoted delimiter: shell would expand $
                self.fidelity["unquoted_heredoc_dollar"] += 1
            ops.append((m.start(), ("heredoc", m.group(1),
                                    m.group(2) or m.group(3) or m.group(4), body)))
        for m in re.finditer(
                r"python[3]?\s+-\s*<<-?\s*(?:'(\w+)'|\"(\w+)\"|(\w+))\n(.*?)\n(?:\1|\2|\3)(?=\n|$)",
                cmd, re.DOTALL):
            body = m.group(4)
            if m.group(3) and "$" in body:
                self.fidelity["unquoted_heredoc_dollar"] += 1
            ops.append((m.start(), ("pyexec", body)))
        for m in re.finditer(r"""python[3]?\s+-c\s+(?:"((?:[^"\\]|\\.)*)"|'([^']*)')""", cmd):
            body = m.group(1) or m.group(2) or ""
            if re.search(r"open\([^)]*['\"](w|a)['\"]", body) or ".write_text(" in body:
                ops.append((m.start(), ("pyexec", body)))
        for m in re.finditer(r"(?:^|&&|;|\n)\s*rm\s+((?:-\w+\s+)*)([^;&|\n]+)", cmd):
            for tok in m.group(2).split():
                if not tok.startswith("-"):
                    ops.append((m.start(), ("rm", tok)))
        for m in re.finditer(r"(?:^|&&|;|\n)\s*(cp|mv)\s+((?:-\w+\s+)*)([^;&|\n]+)", cmd):
            toks = [t for t in m.group(3).split() if not t.startswith("-")]
            if len(toks) >= 2:
                ops.append((m.start(), (m.group(1), toks[:-1], toks[-1])))
        if not ops and re.search(r"sed\s+-i|tee\s|>\s*\S+\.(py|json|txt|yaml|cfg)", cmd):
            self.fidelity["opaque_cmds"] += 1
        for _pos, op in sorted(ops, key=lambda x: x[0]):
            kind = op[0]
            if kind == "cd":
                target = op[1].strip().strip('"\'')
                cwd = target if target.startswith("/") else (Path(cwd) / target).as_posix()
            elif kind == "heredoc":
                (self.append if op[1] == ">>" else self.write)(op[2], op[3] + "\n", cwd)
                self.fidelity["heredocs"] += 1
            elif kind == "pyexec":
                self.run_python_heredoc(op[1], cwd)
            elif kind == "rm":
                self.rm(op[1], cwd)
            elif kind == "cp":
                self.cp(op[1], op[2], cwd)
            elif kind == "mv":
                self.cp(op[1], op[2], cwd)
                for s in op[1]:
                    if not s.startswith("/submissions"):
                        self.rm(s, cwd)
        if workdir is None:
            self.cwd = cwd  # Claude's shell is stateful

    # -- codex apply_patch -----------------------------------------------------------
    def apply_patch(self, patch: str):
        self.fidelity["patches"] += 1
        lines = patch.splitlines()
        i, cur_path, mode, add_buf = 0, None, None, []

        def flush_add():
            # real apply_patch normalizes to a single trailing newline (measured against
            # on-pod files: a trailing lone '+' never yields a trailing blank line)
            if cur_path is not None and mode == "add":
                self.write(cur_path,
                           ("\n".join(add_buf)).rstrip("\n") + "\n" if add_buf else "")

        while i < len(lines):
            ln = lines[i]
            if ln.startswith("*** Add File: "):
                flush_add()
                cur_path, mode, add_buf = ln[len("*** Add File: "):].strip(), "add", []
            elif ln.startswith("*** Update File: "):
                flush_add()
                cur_path, mode = ln[len("*** Update File: "):].strip(), "update"
                i = self._apply_update(cur_path, lines, i + 1) - 1
            elif ln.startswith("*** Delete File: "):
                flush_add()
                self.rm(ln[len("*** Delete File: "):].strip(), self.cwd)
                cur_path, mode = None, None
            elif ln.startswith(("*** End Patch", "*** Begin Patch")):
                pass
            elif mode == "add" and ln.startswith("+"):
                add_buf.append(ln[1:])
            i += 1
        flush_add()

    def _apply_update(self, path: str, lines: list[str], i: int) -> int:
        """Update File hunks with real apply_patch semantics: a forward cursor through the
        file plus optional `@@ <anchor>` context lines to disambiguate repeated blocks."""
        p = norm(path, self.cwd)
        content = self.files.get(p) if p is not None else None
        if content is None:
            self.fidelity["patch_mismatches"].append(path)
            while i < len(lines) and not lines[i].startswith("*** "):
                i += 1
            return i
        src = content
        cursor = 0
        ok = True
        while i < len(lines) and not lines[i].startswith("*** "):
            if lines[i].startswith("@@"):
                anchor = lines[i][2:].strip()
                if anchor:
                    at = src.find(anchor, cursor)
                    if at >= 0:
                        cursor = at
                i += 1
                continue
            hunk = []
            while i < len(lines) and not lines[i].startswith(("*** ", "@@")) \
                    and (lines[i][:1] in (" ", "-", "+") or lines[i] == ""):
                hunk.append(lines[i])
                i += 1
            if not hunk:
                i += 1
                continue
            old = "\n".join(h[1:] for h in hunk if h[:1] in (" ", "-"))
            new = "\n".join(h[1:] for h in hunk if h[:1] in (" ", "+"))
            at = src.find(old, cursor) if old else -1
            if old and at >= 0:
                src = src[:at] + new + src[at + len(old):]
                cursor = at + len(new)
            elif old:
                self.fidelity["patch_mismatches"].append(f"{p}: hunk not found")
                ok = False
        if p is not None:
            # keep best effort even on a mismatch (already recorded); normalize the
            # trailing newline exactly like real apply_patch does
            self.files[p] = src.rstrip("\n") + "\n" if src else src
        return i


# ------------------------------------------------------------------ trajectory events
JS_CMD_RE = re.compile(r'cmd\s*:\s*"((?:[^"\\]|\\.)*)"')
JS_WORKDIR_RE = re.compile(r'workdir\s*:\s*"((?:[^"\\]|\\.)*)"')
JS_PATCH_RE = re.compile(r'(?:patch|input)\s*[=:]\s*"((?:[^"\\]|\\.)*)"')


def js_unescape(s: str) -> str:
    try:
        return json.loads('"' + s + '"')
    except Exception:  # noqa: BLE001
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def collect_executions(traj: Path) -> dict:
    """{tool_call_id: executed_ok} from the tool RESULTS logged in later requests.
    A tool_use/custom_tool_call whose result never appears was NEVER executed (stream
    abort, retry, run end) — the relay logs the model's intent, not the execution. A
    Claude tool_result with is_error means the tool ran but FAILED (e.g. Edit's
    old_string not found): it mutated nothing and must not be replayed either."""
    ex: dict[str, bool] = {}
    with traj.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            req = rec.get("request") or {}
            for m in req.get("messages") or []:      # Claude
                c = m.get("content")
                if not isinstance(c, list):
                    continue
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = b.get("tool_use_id")
                        if tid:
                            ex[tid] = not bool(b.get("is_error"))
            for it in req.get("input") or []:        # Codex
                if isinstance(it, dict) and it.get("type") in (
                        "custom_tool_call_output", "function_call_output"):
                    cid = it.get("call_id")
                    if cid:
                        ex[cid] = True
    return ex


def turn_events(rec: dict, executed: dict | None = None):
    resp = rec.get("response") or {}
    for b in resp.get("content") or []:      # Claude
        if not isinstance(b, dict) or b.get("type") != "tool_use":
            continue
        if executed is not None and not executed.get(b.get("id"), False):
            continue  # never executed (phantom/aborted) or errored on the pod
        nm, inp = b.get("name"), b.get("input") or {}
        if nm == "Write":
            yield ("write", inp.get("file_path", ""), inp.get("content", ""))
        elif nm == "Edit":
            yield ("edit", inp.get("file_path", ""), inp.get("old_string", ""),
                   inp.get("new_string", ""), bool(inp.get("replace_all")))
        elif nm in ("Bash", "Monitor"):
            yield ("shell", inp.get("command", ""), None)
    for it in resp.get("output") or []:      # Codex
        if not isinstance(it, dict) or it.get("type") != "custom_tool_call":
            continue
        if executed is not None and not executed.get(it.get("call_id"), False):
            continue
        js = str(it.get("input", ""))
        wd = None
        wm = JS_WORKDIR_RE.search(js)
        if wm:
            wd = js_unescape(wm.group(1))
        if "apply_patch" in js:
            pm = JS_PATCH_RE.search(js)
            if pm and "*** Begin Patch" in js_unescape(pm.group(1)):
                yield ("patch", js_unescape(pm.group(1)))
                continue
        for m in JS_CMD_RE.finditer(js):
            cmd = js_unescape(m.group(1))
            if "apply_patch" in cmd:
                hm = re.search(r"apply_patch\s*<<-?\s*'?\"?(\w+)'?\"?\n(.*?)\n\1",
                               cmd, re.DOTALL)
                if hm:
                    yield ("patch", hm.group(2))
                    continue
            yield ("shell", cmd, wd or WS)


def import_closure(files: dict[str, str]) -> str:
    import ast
    if files.get("solution/solve.py") is None:
        return ""
    seen, todo = set(), ["solution/solve.py"]
    while todo:
        path = todo.pop()
        if path in seen or path not in files:
            continue
        seen.add(path)
        try:
            tree = ast.parse(files[path])
        except Exception:  # noqa: BLE001
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        for n in sorted(names):
            for cand in (f"solution/{n}.py", f"{n}.py"):
                if cand in files:
                    todo.append(cand)
    h = hashlib.sha256()
    for p in sorted(seen):
        h.update(p.encode()); h.update(files[p].encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ main reconstruction
def load_submission_resyncs(run_dir: Path):
    """[(ts_ms, sub_name, {file: content})] from the actual snapshots, sorted by time.
    submitted.json carries an ABSOLUTE timestamp ("at"), directly comparable to the
    trajectory's ts_ms — no wall-time anchoring guesswork."""
    from datetime import datetime
    out = []
    sdir = run_dir / "submissions"
    if not sdir.is_dir():
        return out
    for d in sorted(sdir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "__pycache__":
            continue
        ts_ms = None
        sj = d / "submitted.json"
        if sj.exists():
            try:
                meta = json.loads(sj.read_text())
                if meta.get("at"):
                    ts_ms = datetime.fromisoformat(meta["at"]).timestamp() * 1000.0
            except Exception:  # noqa: BLE001
                pass
        files = {f.relative_to(d).as_posix(): f.read_text(errors="replace")
                 for f in d.rglob("*")
                 if f.is_file() and f.name != "submitted.json"
                 and "__pycache__" not in f.parts}
        if ts_ms is not None and files:
            out.append((ts_ms, d.name, files))
    out.sort(key=lambda x: x[0])
    return out


def _replay_events(traj: Path, subs_dir: Path | None, executed: dict | None = None,
                   on_turn=None) -> VFS:
    """Pure event replay of the whole trajectory. on_turn(vfs, ts) after each record."""
    vfs = VFS(subs_dir)
    with traj.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for ev in turn_events(rec, executed):
                if ev[0] == "write":
                    vfs.write(ev[1], ev[2])
                elif ev[0] == "edit":
                    vfs.edit(ev[1], ev[2], ev[3], ev[4])
                elif ev[0] == "shell":
                    vfs.shell(ev[1], ev[2])
                elif ev[0] == "patch":
                    vfs.apply_patch(ev[1])
                for p, c in vfs.files.items():
                    vfs._remember(p, c)
            if on_turn is not None:
                on_turn(vfs, rec.get("ts_ms"))
    return vfs


def reconstruct(label: str, results: Path) -> dict:
    run_dir = results / label
    traj = run_dir / "raw_requests.trimmed.jsonl"
    if not traj.exists():
        traj = run_dir / "raw_requests.jsonl"
    if not traj.exists():
        traj = run_dir / "raw_requests.full.jsonl"
    if not traj.exists():
        return reconstruct_from_submissions_only(label, results)

    subs_dir = run_dir / "submissions"
    if not subs_dir.is_dir():
        subs_dir = None

    # Which tool calls actually EXECUTED (and did not error): gate every replayed event.
    executed = collect_executions(traj)

    # PASS 1: pure replay -> the complete content history each file ever has. A snapshot
    # state found anywhere in it is consistent with the replay (only its timing is skewed);
    # only content the replay NEVER produces is a genuinely missed edit.
    full_history = _replay_events(traj, subs_dir, executed).history

    vfs = VFS(subs_dir)
    resyncs = load_submission_resyncs(run_dir)
    ri = 0

    out_dir = run_dir / "versions"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    versions = []
    t0 = None
    last_hash = ""
    n_v = 0

    def snapshot(ts):
        nonlocal n_v, last_hash
        h = import_closure(vfs.files)
        if not h or h == last_hash:
            return
        n_v += 1
        vdir = out_dir / f"v{n_v:04d}"
        for p, content in vfs.files.items():
            dst = vdir / p
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(content)
            except (FileExistsError, NotADirectoryError, OSError) as exc:
                print(f"[vfs] materialize skip {p}: {exc}", file=sys.stderr)
        versions.append({"version": f"v{n_v:04d}", "ts_ms": ts,
                         "wall_min": round((ts - t0) / 60000.0, 2) if ts and t0 else None,
                         "closure_hash": h})
        last_hash = h

    def apply_resyncs_until(ts):
        # A snapshot equal to ANY state of the pure replay (past or future) is consistent
        # with the event stream — its timing is just skewed relative to request receipt
        # timestamps; leave the replay alone. Only content the replay never produces is a
        # genuinely missed edit: force it in and count the divergence.
        nonlocal ri
        while ri < len(resyncs) and ts is not None and resyncs[ri][0] <= ts:
            sub_ts, name, files = resyncs[ri]
            forced = False
            for rel, content in files.items():
                key = f"solution/{rel}"
                cur = vfs.files.get(key)
                if cur is not None and cur.rstrip("\n") == content.rstrip("\n"):
                    continue
                if VFS._key(content) in full_history.get(key, set()):
                    vfs.fidelity["resync_timing_skips"] += 1
                    continue
                if cur is not None and key.endswith(".py"):
                    vfs.fidelity["resync_divergences"].append(f"{name}:{rel}")
                vfs.files[key] = content
                vfs._remember(key, content)
                forced = True
            if forced:
                snapshot(int(sub_ts))
            ri += 1

    with traj.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            ts = rec.get("ts_ms")
            if ts and t0 is None:
                t0 = ts
            if ts is not None:
                apply_resyncs_until(ts)
            for ev in turn_events(rec, executed):
                if ev[0] == "write":
                    vfs.write(ev[1], ev[2])
                elif ev[0] == "edit":
                    vfs.edit(ev[1], ev[2], ev[3], ev[4])
                elif ev[0] == "shell":
                    vfs.shell(ev[1], ev[2])
                elif ev[0] == "patch":
                    vfs.apply_patch(ev[1])
                for p, c in vfs.files.items():   # content-history for resync skew detection
                    vfs._remember(p, c)
            snapshot(ts)
    apply_resyncs_until(float("inf") if t0 is not None else None)

    # FINAL GT ANCHOR: the mirrored final workspace is recorded ground truth for the end
    # of the run. Force the final state to it (python sources), so the last graded
    # version is exactly the code the run really ended with; any correction is reported.
    gt_ws = run_dir / "gt" / "workspace"
    if gt_ws.is_dir():
        forced = False
        for f in sorted(gt_ws.rglob("*.py")):
            if "__pycache__" in f.parts or ".agent" in f.parts:
                continue
            rel = f.relative_to(gt_ws).as_posix()
            content = f.read_text(errors="replace")
            cur = vfs.files.get(rel)
            if cur is not None and cur.rstrip("\n") == content.rstrip("\n"):
                continue
            vfs.fidelity.setdefault("final_gt_corrections", []).append(rel)
            vfs._set(rel, content)
            forced = True
        # files the replay has but the real final workspace does not (agent deleted them
        # via mechanisms we do not parse): drop, the disk is ground truth
        gt_rel = {f.relative_to(gt_ws).as_posix() for f in gt_ws.rglob("*.py")}
        for rel in [r for r in vfs.files if r.endswith(".py") and r not in gt_rel]:
            del vfs.files[rel]
            vfs.fidelity.setdefault("final_gt_deletions", []).append(rel)
            forced = True
        if forced:
            snapshot(versions[-1]["ts_ms"] + 1 if versions and versions[-1]["ts_ms"]
                     else None)

    (out_dir / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")
    fid = {k: (v if not isinstance(v, list) else sorted(set(v))[:80])
           for k, v in vfs.fidelity.items()}
    (out_dir / "fidelity.json").write_text(json.dumps(fid, indent=1) + "\n")
    # keep the final VFS for check_final
    fin = out_dir / "final_state"
    for p, content in vfs.files.items():
        dst = fin / p
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
        except (FileExistsError, NotADirectoryError, OSError) as exc:
            print(f"[vfs] materialize skip {p}: {exc}", file=sys.stderr)
    return {"label": label, "versions": len(versions), "files_final": len(vfs.files),
            "fidelity": {k: (len(v) if isinstance(v, list) else v)
                         for k, v in vfs.fidelity.items()}}


def reconstruct_from_submissions_only(label: str, results: Path) -> dict:
    """Fallback for the few runs whose trajectory was lost (launcher crash before the
    volume commit): every submission snapshot is a version (its absolute timestamp is
    recorded), plus the final workspace as the final version. Explicitly flagged."""
    run_dir = results / label
    resyncs = load_submission_resyncs(run_dir)
    gt_ws = run_dir / "gt" / "workspace"
    if not resyncs and not gt_ws.is_dir():
        return {"label": label, "error": "no trajectory and no submissions/workspace"}
    out_dir = run_dir / "versions"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    versions = []
    n_v = 0
    last_hash = ""
    for sub_ts, name, files in resyncs:
        state = {f"solution/{rel}": content for rel, content in files.items()}
        h = import_closure(state)
        if not h or h == last_hash:
            continue
        n_v += 1
        vdir = out_dir / f"v{n_v:04d}"
        for p, content in state.items():
            dst = vdir / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
        versions.append({"version": f"v{n_v:04d}", "ts_ms": int(sub_ts), "wall_min": None,
                         "closure_hash": h, "from_submission": name})
        last_hash = h
    fin_state = {}
    if gt_ws.is_dir():
        for f in gt_ws.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and ".agent" not in f.parts:
                try:
                    fin_state[f.relative_to(gt_ws).as_posix()] = f.read_text()
                except (UnicodeDecodeError, OSError):
                    pass
        h = import_closure(fin_state)
        if h and h != last_hash:
            n_v += 1
            vdir = out_dir / f"v{n_v:04d}"
            for p, content in fin_state.items():
                dst = vdir / p
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(content)
            versions.append({"version": f"v{n_v:04d}", "ts_ms": None, "wall_min": None,
                             "closure_hash": h, "from_submission": "final_workspace"})
    (out_dir / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")
    (out_dir / "fidelity.json").write_text(json.dumps(
        {"mode": "submissions_only_no_trajectory"}, indent=1) + "\n")
    fin = out_dir / "final_state"
    for p, content in (fin_state or {}).items():
        dst = fin / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
    return {"label": label, "versions": len(versions),
            "fidelity": {"mode": "submissions_only_no_trajectory"}}


def check_final(label: str, results: Path, gt_workspace: Path) -> dict:
    """Byte-compare the reconstructed final state against the mirrored final workspace."""
    fin = results / label / "versions" / "final_state"
    gt_pys = {p.relative_to(gt_workspace).as_posix(): p for p in gt_workspace.rglob("*.py")
              if ".agent" not in p.parts and "__pycache__" not in p.parts}
    exact, diff, missing = [], [], []
    for rel, p in sorted(gt_pys.items()):
        rp = fin / rel
        if not rp.exists():
            missing.append(rel)
        elif rp.read_text().rstrip("\n") == p.read_text(errors="replace").rstrip("\n"):
            exact.append(rel)
        else:
            diff.append(rel)
    return {"label": label, "exact": len(exact), "diff": diff, "missing": missing,
            "gt_total": len(gt_pys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results"))
    ap.add_argument("--check-against", default=None,
                    help="path to the run's ground-truth final workspace")
    args = ap.parse_args()
    results = Path(args.results)
    print(json.dumps(reconstruct(args.label, results), indent=1))
    if args.check_against:
        print(json.dumps(check_final(args.label, results, Path(args.check_against)), indent=1))


if __name__ == "__main__":
    main()
