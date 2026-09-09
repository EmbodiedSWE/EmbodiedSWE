"""orchestrate — the data_gen conductor: one agent-written solve in, a ladder of exact,
replay-verified demonstration sets out, unattended.

    python data_engine/scripts/orchestrate.py <run_dir> [--name gen_x] [--num-envs 512]
        [--stage-hours 12] [--agent-cmd "python .../claude_agent_loop.py --model M"]
        [--isaac-py /path/venv/bin/python]

`run_dir` is an eval-run-shaped folder: run.json (the preset) + workspace/solution/ (the
solve and its sibling modules) [+ workspace/candidates/<k>/ alternates the repair agent may
consult].

THE LADDER — exact, nested trajectory sets, each containing the previous one whole:

    1 nominal -> 5 (scene) -> 20 (strategy) -> 50 (phase) -> 200 (physics) -> 600 rendered

SIX GATES, every one HARD, every one on its own persistent clock of `stage_hours`; a gate
that is not met when its clock runs out STOPS the campaign and says which rung failed
(PRECHECK_FAILED / STAGE_FAILED). Nothing "moves on" past a missed rung: the ladder is
nested, so everything after it would be dead work.

    PRE-CHECKS  the delivered solve must succeed nominally (1 env) AND replay — its recorded
                actions, fed back open-loop, must reproduce the success — then complete a
                num_envs-wide batch with at least one success THAT REPLAYS AT THAT WIDTH.
                Failures -> repair / vectorize agent sessions, repeated until it passes.
    SCENE       an agent session authors variant cells (new scenes / strategies / phase-entry
    STRATEGY    resets) under scenes/. Gate: the pool holds >= <level>_target episodes that are
    PHASE       graded successful, replay-verified, non-vacuous, produced by the CURRENT code
                of their cell (batches carry a code fingerprint), and EVERY cell this level kept
                contributes at least one of them. On pass the stage delivers <level>_set.json:
                exactly <level>_target episodes = the previous rung whole + a round-robin fill
                across the level's cells, byte-identical trajectories never counted twice.
    DYNAMICS    a noise session authors executed-action noise into the solves through
                env.step(action, noise=...) (labels stay clean) and declares PHYSICAL_PARAMS on
                the base scene; accepted by ONE certification of the base cell (clean + noisy
                num_envs batch pair). Then scripted harvest batches round-robin over the
                phase set's cells at noise_scale, each batch's successes replay-verified at
                once, a cell that yields nothing dropped; physics_set.json = exactly
                physics_target = the 50 + survivors round-robin across cells.
    VISUAL      scene contracts (VISUAL_PARAMS + CAMERAS) for every scene in the physics set,
                then the render queue: exactly physics_target x visual_draws (episode, look)
                items, pass-major (every episode's declared cameras before any re-look).

REPLAY VERDICTS are a property of a whole batch: replay_check always replays every recorded
episode of a batch together, each in the slot it was recorded in (states are world-frame;
on GPU PhysX each env's contacts shape every other env's), and a replay cut by a clock
records its pending episodes as unverifiable — never re-run.

Sessions get the clock that is left (authoring) or intervention_session_min, whichever is
smaller; nothing launches with under LAUNCH_FLOOR_S of clock. status.json is the live stage,
orchestration.json the resume ledger (clock starts, sessions, code hash), manifest.json the
dataset index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # data_engine/
sys.path.insert(0, str(ROOT))

from engine import contract  # noqa: E402  (pure logic — no Isaac import)


# --------------------------------------------------------------------------- config

class Stage(str, Enum):
    """Live pipeline stage, written verbatim into status.json."""

    INIT = "init"
    NOMINAL = "nominal_probe"
    REPAIR = "repair"
    WIDE = "wide_probe"
    VECTORIZE = "session_vectorize"
    SESSION = "session"
    NOISE_PLAN = "session_noise"
    HARVEST = "dynamics"
    SESSION_VISUAL = "session_visual"
    RENDER = "visual"
    DONE = "DONE"
    FAILED = "FAILED"                     # an exception
    PRECHECK_FAILED = "PRECHECK_FAILED"   # no working solve within the pre-check clock
    STAGE_FAILED = "STAGE_FAILED"         # a rung not met within its clock
    MODEL_DOWN = "MODEL_DOWN"             # the model route stayed dead (exit 75)


class ModelDown(Exception):
    """The agent's model route stayed dead past the pause budget (exit 75)."""


LEVELS = ("scene", "strategy", "phase")


@dataclass(frozen=True)
class Config:
    """Every tunable, with its reason. CLI flags map 1:1 (see build_config)."""

    run_dir: Path
    name: str = "gen_auto"
    sessions: tuple[str, ...] = LEVELS
    agent_cmd: str = ""               # "" = no agent runtime: gates that need one fail
    isaac_py: str = sys.executable

    # ONE clock per gate (hours). Every gate is hard: sessions/batches repeat until the rung
    # is met; when the clock expires the campaign stops and reports the rung.
    stage_hours: float = 12.0
    # Repair / vectorize / noise / visual sessions are capped at this (minutes) — the eval
    # harness's own per-attempt budget; authoring sessions get the whole remaining clock.
    intervention_session_min: float = 120.0

    # THE LADDER rungs (exact set sizes). Spread rule: every cell a stage kept must
    # contribute >= 1 verified episode, and >= 1 new cell must exist.
    scene_target: int = 5
    strategy_target: int = 20
    phase_target: int = 50
    physics_target: int = 200
    visual_draws: int = 3             # looks per physics-set episode (200 x 3 = 600)

    num_envs: int = 4                 # scripted batch width; launchers set the production width
    # One nominal seed: the eval harness's convention (agents developed on seed 0), and
    # without PHYSICAL_PARAMS every seed is the same world.
    nominal_seeds: tuple[int, ...] = (0,)
    # Seed spaces are kept disjoint so no scripted batch ever repeats another's world:
    # nominal probes 0.., the wide probe/certification pair at 50, harvest batches from
    # 5000 upward, render pose jitter from 1000 upward.
    wide_probe_seed: int = 50
    dynamics_seed_base: int = 5000
    visual_seed_base: int = 1000
    # A launch that dies before running the agent (zero turns, nonzero exit) is a broken
    # command or environment, not a flaky one: a few retries a minute apart catch a
    # transient (gateway hiccup at CLI start); more cannot help.
    session_attempts_per_round: int = 5
    session_retry_sleep_s: float = 60.0

    noise_scale: float = 1.0          # the harvest's executed-noise scale (probes run at 0)
    render_envs: int = 16             # episodes replayed in parallel per render pass
    render_shard_frames: int = 400_000  # frames per render invocation (sized to the 3 h batch timeout)
    render_kit_args: str = ""
    job_concurrency: int = 1          # harvest batches / render shards in flight (one 512-env batch saturates a 4090)
    trim_margin: int = 45             # render: rows kept past sustained success (3 s at 15 Hz)
    cam_eye_jitter: tuple[float, float, float] = (0.30, 0.30, 0.20)
    cam_target_jitter: tuple[float, float, float] = (0.08, 0.08, 0.06)

    batch_timeout_s: float = 3 * 3600   # one batch (a 512-env, 10k-step batch runs 25-70 min)
    # a 1-env nominal run: the longest successful eval solve took 38 min on this GPU class;
    # 2.4x that is the cut — beyond it the solve is stuck and repair is the answer
    nominal_timeout_s: float = 90 * 60
    # SAFETY only, never part of a session's cap: how long past its own cap the agent
    # process may live before the orchestrator kills it. The loop kills at the cap (SIGTERM,
    # 30 s, SIGKILL) and reaps its children (2 x 5 s); 15 min covers an Isaac process that
    # ignores SIGTERM on its way out.
    session_grace_s: float = 15 * 60
    # "K/N": this process is render worker K of N — it renders only its deterministic share
    # of the manifest (items with crc32(episode:pass) % N == K) and nothing else. "" = the
    # campaign's own orchestrator, which renders everything not already rendered (pulling
    # workers' output from the mirror at DGEN_MIRROR_TASK_DIR before each sweep).
    render_worker: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.sessions) - set(LEVELS)
        if unknown:
            raise ValueError(f"unknown authoring sessions: {sorted(unknown)}")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        rungs = [self.target(lvl) for lvl in self.sessions] + [self.physics_target]
        if any(r < 0 for r in rungs) or any(a > b for a, b in zip(rungs, rungs[1:])):
            raise ValueError(f"ladder targets must be non-negative and non-decreasing: {rungs}")
        if self.stage_hours <= 0:
            raise ValueError("stage_hours must be > 0")
        if self.physics_target > self.base_target and self.noise_scale <= 0:
            raise ValueError("noise_scale must be > 0 when harvesting (noise is mandatory)")
        for name in ("visual_draws", "render_envs", "render_shard_frames", "job_concurrency"):
            if getattr(self, name) < (0 if name == "visual_draws" else 1):
                raise ValueError(f"{name} out of range")
        if self.render_worker:
            k, n = self.worker_share
            if not (0 <= k < n):
                raise ValueError(f"render_worker must be K/N with 0 <= K < N, got {self.render_worker!r}")

    @property
    def worker_share(self) -> tuple[int, int]:
        k, _, n = self.render_worker.partition("/")
        return int(k), int(n)

    def target(self, level: str) -> int:
        return int(getattr(self, f"{level}_target"))

    @property
    def base_target(self) -> int:
        """Size of the dynamics base = the last authoring rung (0 = no authoring)."""
        return self.target(self.sessions[-1]) if self.sessions else 0


# --------------------------------------------------------------------- subprocess

def sh(cmd: list, *, timeout: float, log: Path | None = None,
       check: bool = True) -> subprocess.CompletedProcess:
    """Run a stage subprocess. A timeout is survivable: the process is reaped, the log
    annotated and a synthetic rc=-9 returned with `.timed_out=True`. A SIGKILL we did
    not send (rc=-9, `.timed_out=False`) is the kernel OOM killer and is said so."""
    print(f"[orchestrate] $ {' '.join(map(str, cmd))}", flush=True)
    try:
        p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes)
               else exc.stdout or "")
        err = ((exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes)
               else exc.stderr or "")
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(out + err + f"\n[orchestrate] TIMED OUT after {timeout:.0f}s\n")
        print(f"[orchestrate] command TIMED OUT after {timeout:.0f}s", flush=True)
        p = subprocess.CompletedProcess(cmd, -9, out, err)
        p.timed_out = True
        return p
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(p.stdout + p.stderr)
    p.timed_out = False
    if p.returncode == -9:
        print("[orchestrate] command was KILLED (SIGKILL we did not send — almost certainly "
              "the OOM killer)", flush=True)
        if log:
            with log.open("a") as fh:
                fh.write("\n[orchestrate] KILLED by SIGKILL (not a timeout; likely OOM)\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd[:3]}\n"
                           f"full stdout/stderr:\n{p.stdout}{p.stderr}")
    return p


# ----------------------------------------------------------------------- campaign

@dataclass(frozen=True)
class Cell:
    """One runnable (scene, strategy[, phase]) combination."""

    scene: str
    strategy: str
    phase: str | None = None

    @property
    def key(self) -> str:
        return f"{self.scene}/{self.strategy}" + (f"/{self.phase}" if self.phase else "")

    @staticmethod
    def parse(key: str) -> "Cell":
        parts = key.split("/")
        return Cell(parts[0], parts[1], parts[2] if len(parts) > 2 else None)


BASE_CELL = Cell("scene_0", "strategy_0")


def read_json(path: Path, default: dict | None = None) -> dict:
    return json.loads(path.read_text()) if path.is_file() else (default or {})


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class Campaign:
    """The on-disk campaign: paths, ledger, clocks, cells, metas, sets, status, manifest.
    Everything re-derives from disk so a resumed run never trusts stale state."""

    def __init__(self, run_dir: Path, name: str, final_level: str) -> None:
        self.run_dir, self.name, self.final_level = run_dir, name, final_level

    # ---- paths
    @property
    def gen(self) -> Path:
        return self.run_dir / "data_gen" / self.name

    @property
    def logs(self) -> Path:
        return self.gen / "logs"

    def scene_py(self, scene: str) -> Path:
        return self.gen / "scenes" / scene / "scene" / "scene.py"

    def grader_py(self, scene: str) -> Path:
        return self.gen / "scenes" / scene / "grader" / "grader.py"

    def solve_py(self, cell: Cell) -> Path:
        return self.gen / "scenes" / cell.scene / "strategies" / cell.strategy / "solve.py"

    def phase_dir(self, cell: Cell) -> Path:
        return self.solve_py(cell).parent / "phases" / (cell.phase or "")

    def log_path(self, batch: str) -> Path:
        """Evidence goes into briefs BY PATH: the complete log stays on disk for the
        agent's shell and can never blow the model's context (a 512-env probe log once
        weighed over a million tokens)."""
        return self.logs / f"{batch}.log"

    def log_size(self, batch: str) -> str:
        p = self.log_path(batch)
        if not p.is_file():
            return "missing — the batch died before Isaac wrote anything"
        n = p.stat().st_size
        return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"

    # ---- ledger + clocks
    @property
    def ledger_path(self) -> Path:
        return self.gen / "orchestration.json"

    def ledger(self) -> dict:
        return read_json(self.ledger_path, {"schema_version": 1})

    def update_ledger(self, **fields) -> dict:
        cur = self.ledger()
        cur.update(fields)
        write_json_atomic(self.ledger_path, cur)
        return cur

    def deadline(self, budget: str, hours: float) -> float:
        """Persistent clock start: a restart consumes the original budget."""
        cur = self.ledger()
        starts = dict(cur.get("budget_started_at", {}))
        if budget not in starts:
            starts[budget] = time.time()
            cur["budget_started_at"] = starts
            write_json_atomic(self.ledger_path, cur)
        return float(starts[budget]) + hours * 3600

    def credit(self, budget: str, seconds: float) -> None:
        """Move a clock's start forward: time spent PAUSED (model route down) is not the
        stage's to pay for."""
        cur = self.ledger()
        starts = dict(cur.get("budget_started_at", {}))
        if budget in starts:
            starts[budget] = float(starts[budget]) + seconds
            cur["budget_started_at"] = starts
            write_json_atomic(self.ledger_path, cur)

    def session_attempts(self, prefix: str) -> list[tuple[str, dict]]:
        return sorted((tag, dict(o)) for tag, o in self.ledger().get("sessions", {}).items()
                      if tag.startswith(prefix))

    def record_session(self, tag: str, outcome: dict) -> None:
        cur = self.ledger()
        sessions = dict(cur.get("sessions", {}))
        sessions[tag] = outcome
        cur["sessions"] = sessions
        write_json_atomic(self.ledger_path, cur)

    # ---- cells
    def cells(self) -> list[Cell]:
        out = []
        for sc in sorted(self.gen.glob("scenes/scene_*")):
            if not (sc / "scene" / "scene.py").is_file():
                continue
            for st in sorted(sc.glob("strategies/strategy_*")):
                if not (st / "solve.py").is_file():
                    continue
                out.append(Cell(sc.name, st.name))
                for ph in sorted(st.glob("phases/*")):
                    if ph.is_dir() and (ph / "reset").is_dir() and any((ph / "reset").glob("*.py")):
                        out.append(Cell(sc.name, st.name, ph.name))
        return out

    # ---- metas, sets
    def batch_meta(self, batch: str) -> dict:
        return read_json(self.gen / "data" / batch / "meta.json")

    def batch_metas(self, pattern: str = "batch_*") -> list[dict]:
        return [json.loads(m.read_text()) for m in sorted(self.gen.glob(f"data/{pattern}/meta.json"))]

    def stage_set(self, level: str) -> list[str]:
        p = self.gen / f"{level}_set.json"
        return json.loads(p.read_text()).get("episodes", []) if p.is_file() else []

    def base_set(self) -> list[str]:
        """The dynamics base = the last authoring stage's set."""
        return self.stage_set(self.final_level) if self.final_level else []

    def physics_set(self) -> list[str]:
        p = self.gen / "physics_set.json"
        return json.loads(p.read_text()).get("episodes", []) if p.is_file() else []

    @property
    def render_manifest_path(self) -> Path:
        return self.gen / "render_manifest.json"

    # ---- status + manifest
    def write_status(self, stage: Stage | str, **fields) -> None:
        write_json_atomic(self.gen / "status.json", {
            "schema_version": 1, "stage": str(getattr(stage, "value", stage)), **fields,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    def write_manifest(self) -> dict:
        """Index the dataset: once physics_set.json exists THE dataset is exactly that set;
        before it, the delivered ladder sets. Everything else on disk is diagnostic."""
        members = set(self.physics_set()) or set(self.base_set())
        eps, total = [], 0
        for em in sorted(self.gen.glob("data/*/ep_*/meta.json")):
            rel = str(em.parent.relative_to(self.gen))
            total += 1
            if rel in members:
                meta = json.loads(em.read_text())
                eps.append({"path": rel, "cell": meta.get("cell"), "phase_entry": meta.get("entry"),
                            "reset": meta.get("reset"), "steps": meta.get("steps"),
                            "score": meta.get("score"), "parameters": meta.get("parameters")})
        manifest = {"episodes": eps, "dataset_size": len(eps), "episodes_on_disk": total,
                    "refreshed": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_json_atomic(self.gen / "manifest.json", manifest)
        return manifest


# ------------------------------------------------------------------- agent runner

@dataclass(frozen=True)
class SessionResult:
    tag: str
    returncode: int
    session_dir: Path
    turns: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def worked(self) -> bool:
        """The session ran the agent (exit 0, or died after >= 1 real turn). Only a
        zero-turn nonzero exit is a LAUNCH failure."""
        return self.ok or self.turns > 0

    def as_dict(self) -> dict:
        return {"returncode": self.returncode, "ok": self.ok, "turns": self.turns,
                "worked": self.worked, "session_dir": str(self.session_dir),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class AgentRunner:
    """One autonomous agent session over an orchestrate-authored brief, with the campaign
    CLIs (create_cell / generate) shimmed onto PATH. The agent command is bring-your-own:
    any executable accepting --prompt-file/--workdir/--cap-min/--transcript/--env."""

    TOOLS = ("create_cell", "generate")

    def __init__(self, cfg: Config, camp: Campaign) -> None:
        self.cfg, self.camp = cfg, camp

    @property
    def available(self) -> bool:
        return bool(self.cfg.agent_cmd)

    def _shims(self) -> Path:
        shims = self.camp.gen / ".shims"
        shims.mkdir(exist_ok=True)
        base_pp = os.environ.get("PYTHONPATH")
        pp = "PYTHONPATH=" + shlex.quote(str(ROOT.parent) + (f":{base_pp}" if base_pp else ""))
        for tool in self.TOOLS:
            script = shims / tool
            script.write_text(f"#!/bin/bash\nexec env {pp} {shlex.quote(self.cfg.isaac_py)} "
                              f"{shlex.quote(str(ROOT / 'agent' / 'cli' / (tool + '.py')))} \"$@\"\n")
            script.chmod(0o755)
        return shims

    def brief(self, name: str, **fields) -> str:
        return (ROOT / "agent" / "prompts" / f"{name}.md").read_text().format(**fields)

    def start(self, tag: str, instructions_text: str, cap_min: float,
              level: str | None = None) -> tuple[subprocess.Popen, Path]:
        """Launch one session and return (process, session dir) — the caller watches it
        (gate polls) and calls finish()."""
        session = self.camp.gen / ".agent" / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
        session.mkdir(parents=True, exist_ok=True)
        kill_at = datetime.fromtimestamp(time.time() + cap_min * 60, timezone.utc)
        (session / "instructions.md").write_text(
            f"# Session budget: {cap_min:.0f} minutes — hard kill at {kill_at:%H:%M} UTC\n\n"
            "Every process you started is killed with the session. An edit you have not "
            "re-verified by then ships unverified; schedule the last verification run to finish "
            "before the kill. The session is also closed early, by the orchestrator, the moment "
            "the stage's goal below is met.\n\n" + instructions_text)
        cmd = [*shlex.split(self.cfg.agent_cmd),
               "--prompt-file", session / "instructions.md", "--workdir", self.camp.gen,
               "--cap-min", str(cap_min), "--transcript", session / "transcript.jsonl",
               "--env", f"PATH={self._shims()}:{os.environ.get('PATH', '')}",
               "--env", f"DGEN_ROOT={self.camp.gen}",
               "--env", "DGEN_SCENE=scene_0", "--env", "DGEN_STRATEGY=strategy_0",
               "--env", f"DGEN_NUM_ENVS={self.cfg.num_envs}",
               "--env", f"ISAAC_PY={self.cfg.isaac_py}"]
        if level:
            cmd += ["--env", f"DGEN_LEVEL={level}"]   # create_cell's level default
        print(f"[orchestrate] $ {' '.join(map(str, cmd))}", flush=True)
        log = (session / "agent.log").open("w")
        proc = subprocess.Popen(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT)
        proc._log_handle = log  # closed in finish()
        return proc, session

    def finish(self, tag: str, proc: subprocess.Popen, session: Path) -> SessionResult:
        rc = proc.wait()
        proc._log_handle.close()
        transcript = session / "transcript.jsonl"
        turns = 0
        if transcript.is_file():
            with transcript.open(errors="replace") as fh:
                turns = sum(1 for line in fh
                            if '"type":"assistant"' in line and '"model":"<synthetic>"' not in line)
        result = SessionResult(tag, rc, session, turns)
        self.camp.record_session(tag, result.as_dict())
        return result


# ------------------------------------------------------------------- orchestrator

class StageFailed(Exception):
    def __init__(self, stage: Stage, **fields):
        super().__init__(fields.get("error", stage.value))
        self.stage, self.fields = stage, fields


class Orchestrator:
    """The gate machine. Each gate reads its preconditions from disk (idempotent resume),
    runs sessions/batches until met or its clock ends, and records outcomes."""

    # Nothing (batch, replay, session) launches with less than this much clock left: Isaac
    # boots in ~3 min and no probe, replay or session did useful work in under ~30 min in
    # any observed run; below the floor only boot cost is paid.
    LAUNCH_FLOOR_S = 30 * 60
    # The agent loop exits 75 after MODEL_DOWN_S (10 min) of API failures with no model
    # turn; the orchestrator then gives the route six more 5-min probes (30 min) with the
    # stage clock credited before releasing the machine (exit 75) — a gateway outage
    # once consumed every stage of a 12-task wave while pods billed.
    MODEL_PROBE_S = 300.0
    MODEL_PAUSE_MAX_S = 1800.0

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.camp = Campaign(cfg.run_dir, cfg.name, cfg.sessions[-1] if cfg.sessions else "")
        self.agent = AgentRunner(cfg, self.camp)
        self.preset: str = json.loads((cfg.run_dir / "run.json").read_text())["preset"]
        self.budget = "checks"
        self._cells_at_stage_start: set[str] = set()

    # ---- clocks -------------------------------------------------------------------
    def _deadline(self, budget: str) -> float:
        return self.camp.deadline(f"stage_{budget}", self.cfg.stage_hours)

    def _enter(self, budget: str) -> None:
        """The clock starts the moment the gate begins: probes and replays count too."""
        self.budget = budget
        self._deadline(budget)

    def _left_s(self, budget: str | None = None) -> float:
        return self._deadline(budget or self.budget) - time.time()

    def _batch_timeout(self, cap: float | None = None) -> float:
        """A batch never outlives its clock. Callers launch only above LAUNCH_FLOOR_S."""
        return min(cap or self.cfg.batch_timeout_s, self._left_s())

    # ---- model route --------------------------------------------------------------
    def _model_reachable(self) -> bool:
        import urllib.error
        import urllib.request
        import claude_agent_loop as loop
        argv = shlex.split(self.cfg.agent_cmd)
        model = (argv[argv.index("--model") + 1] if "--model" in argv
                 else os.environ.get("DGEN_AGENT_MODEL", loop.DEFAULT_MODEL))
        body = json.dumps({"model": model, "max_tokens": 4096,
                           "messages": [{"role": "user", "content": "Reply OK"}]}).encode()
        req = urllib.request.Request(
            f"{loop.SEEDCODE_BASE}/v1/messages", data=body, method="POST",
            headers={"x-api-key": loop.SEEDCODE_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        try:
            # a healthy route answers a one-token request in ~1 s; a minute is dead
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError) as exc:
            print(f"[orchestrate] model probe failed: {exc}", flush=True)
            return False

    def _wait_for_model(self, budget: str) -> None:
        """The agent loop reported a dead route (exit 75): pause, crediting the clock, and
        re-probe; still dead after MODEL_PAUSE_MAX_S -> ModelDown (exit 75)."""
        self.camp.write_status(Stage.MODEL_DOWN,
                               paused_since=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        t0 = time.time()
        while time.time() - t0 < self.MODEL_PAUSE_MAX_S:
            print(f"[orchestrate] model route down — pausing {self.MODEL_PROBE_S / 60:.0f} min "
                  "(clock credited)", flush=True)
            time.sleep(self.MODEL_PROBE_S)
            self.camp.credit(f"stage_{budget}", self.MODEL_PROBE_S)
            if self._model_reachable():
                print("[orchestrate] model route is back — resuming", flush=True)
                return
        raise ModelDown(f"model route dead for {self.MODEL_PAUSE_MAX_S / 60:.0f} min")

    # ---- sessions -----------------------------------------------------------------
    # While a session runs, the gate's cheap check (metadata only, no Isaac) is polled at
    # this interval and the session is CLOSED the moment the goal is met: an agent that
    # keeps working past the goal costs the clock nothing more. An Isaac probe takes >= 3
    # min, so nothing the poll reads can change faster than this.
    POLL_S = 300.0

    def _sessions(self, kind: str, stage: Stage, evaluate, make_brief,
                  whole_clock: bool = False, level: str | None = None, quick=None) -> bool:
        """THE evidence-fed agent loop every gate uses. Sessions that DID WORK repeat until
        `evaluate() -> (ok, evidence)` passes or the clock has under LAUNCH_FLOOR_S left.
        A rejected deliverable re-enters the session WITH its rejection as evidence. A
        session's cap is the clock left (whole_clock: authoring) or at most
        intervention_session_min. `quick() -> bool` is the gate's metadata-only check,
        polled every POLL_S while the agent works; when it turns true the session is closed
        (SIGTERM to the loop, which reaps the agent's processes) and `evaluate` runs. Only
        zero-turn nonzero exits are launch failures."""
        ok, evidence = evaluate()
        if ok:
            return True
        if not self.agent.available:
            print(f"[orchestrate] {kind}: no agent runtime configured — cannot pass", flush=True)
            return False
        budget = self.budget

        def worked(o: dict) -> bool:
            return bool(o.get("worked", o.get("ok")))

        rnd = max((int(m.group(1)) for tag, _ in self.camp.session_attempts(f"{kind}_")
                   if (m := re.match(rf"{kind}_(\d{{3}})_attempt_", tag))), default=0)
        if rnd and not any(worked(o) for _, o in
                           self.camp.session_attempts(f"{kind}_{rnd:03d}_attempt_")):
            rnd -= 1  # an interrupted round resumes; launch failures never consume it
        while not ok and self._left_s(budget) >= self.LAUNCH_FLOOR_S:
            rnd += 1
            prefix = f"{kind}_{rnd:03d}_attempt_"
            completed = [o for _, o in self.camp.session_attempts(prefix) if worked(o)]
            while (not completed and self._left_s(budget) >= self.LAUNCH_FLOOR_S
                   and len(self.camp.session_attempts(prefix)) < self.cfg.session_attempts_per_round):
                attempt = len(self.camp.session_attempts(prefix)) + 1
                tag = f"{prefix}{attempt:03d}"
                self.camp.write_status(stage, round=rnd, attempt=attempt)
                print(f"[orchestrate] {kind} round {rnd}, attempt {attempt}", flush=True)
                left_min = self._left_s(budget) / 60
                cap = left_min if whole_clock else min(self.cfg.intervention_session_min, left_min)
                result = self._watch(tag, make_brief(evidence), cap, level, quick)
                if result.returncode == 75:
                    self._wait_for_model(budget)
                    continue
                if result.worked:
                    if not result.ok:
                        print(f"[orchestrate] {kind} round {rnd} exited {result.returncode} after "
                              f"{result.turns} turns — evaluating its deliverable anyway", flush=True)
                    completed = [result.as_dict()]
                else:
                    print(f"[orchestrate] {kind} round {rnd} attempt {attempt} exited "
                          f"{result.returncode} without running the agent; retrying after "
                          f"{self.cfg.session_retry_sleep_s:.0f}s", flush=True)
                    time.sleep(self.cfg.session_retry_sleep_s)
            if not completed:
                break
            ok, evidence = evaluate()
        if not ok:
            print(f"[orchestrate] {kind}: {budget} clock has under "
                  f"{self.LAUNCH_FLOOR_S // 60:.0f} min left — no further sessions", flush=True)
        return ok

    def _watch(self, tag: str, brief: str, cap_min: float, level: str | None, quick) -> SessionResult:
        """Run one session to its end: the agent stops, the cap kills it (the loop does that;
        the orchestrator's own kill is a safety net cap + grace later), or the gate's quick
        check turns true and the session is closed."""
        import signal
        proc, session = self.agent.start(tag, brief, cap_min, level)
        safety = time.time() + cap_min * 60 + self.cfg.session_grace_s
        next_poll = time.time() + self.POLL_S
        while proc.poll() is None:
            time.sleep(5)
            if time.time() > safety:
                print(f"[orchestrate] {tag}: agent process outlived its cap + grace — killing", flush=True)
                proc.kill()
            elif quick is not None and time.time() >= next_poll:
                next_poll = time.time() + self.POLL_S
                if quick():
                    print(f"[orchestrate] {tag}: goal met — closing the session", flush=True)
                    proc.send_signal(signal.SIGTERM)   # the loop reaps the agent and exits 0
                    try:
                        proc.wait(timeout=self.cfg.session_grace_s)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        return self.agent.finish(tag, proc, session)

    def _hours_left(self) -> str:
        return f"{max(0.0, self._left_s()) / 3600:.1f}"

    # ---- batches and the verified pool -------------------------------------------
    def _fingerprint(self, cell: Cell) -> str:
        return contract.cell_fingerprint(self.camp.gen, cell.key)

    def run_batch(self, cell: Cell, batch: str, extra: list[str], cap: float | None = None) -> dict:
        """One generate invocation (physics only). A cut batch records an empty verdict
        saying WHY (timed out vs killed) so it is never re-run identically."""
        if self._left_s() < self.LAUNCH_FLOOR_S:
            print(f"[orchestrate] {batch}: under {self.LAUNCH_FLOOR_S // 60:.0f} min left on the "
                  f"{self.budget} clock — not launched", flush=True)
            return {}
        cmd = [self.cfg.isaac_py, ROOT / "scripts" / "generate.py", self.camp.gen, "--headless",
               "--batch", batch, "--scene", cell.scene, "--strategy", cell.strategy, *extra]
        if cell.phase:
            cmd += ["--phase", cell.phase]
        p = sh(cmd, log=self.camp.log_path(batch), timeout=self._batch_timeout(cap), check=False)
        meta = self.camp.batch_meta(batch)
        if p.returncode == -9 and not meta:
            meta = {"batch": batch, "cell": cell.key, "episodes": 0, "successes": 0,
                    "success_rate": 0.0, "verdicts": [],
                    "timed_out": bool(p.timed_out), "killed": not p.timed_out}
            write_json_atomic(self.camp.gen / "data" / batch / "meta.json", meta)
        return meta

    def _run_or_read_batch(self, cell: Cell, base_name: str, extra: list[str],
                           cap: float | None = None) -> tuple[str, dict]:
        """A completed batch is immutable and reused on resume; a directory without
        meta.json is an interrupted attempt, so a suffixed retry is launched."""
        data = self.camp.gen / "data"
        matches = ([data / base_name] if (data / base_name).is_dir() else [])
        matches += sorted(data.glob(f"{base_name}_retry*"))
        for path in reversed(matches):
            meta = self.camp.batch_meta(path.name)
            if meta:
                return path.name, meta
        name = base_name if not matches else f"{base_name}_retry{len(matches):02d}"
        return name, self.run_batch(cell, name, extra, cap=cap)

    def _replay_verify(self, episodes: list[str]) -> dict[str, bool]:
        """{episode: reproduces its success under open-loop replay}. Pending episodes are
        checked per recording batch, cheapest batch first — replay_check replays the WHOLE
        batch, every recorded episode in its own slot, so the verdict is the full-batch
        verdict. A check cut by any clock records its pending episodes as unverifiable
        (never re-run)."""
        pending: dict[Path, list[str]] = {}
        for e in episodes:
            if "replay" not in read_json(self.camp.gen / e / "meta.json"):
                pending.setdefault((self.camp.gen / e).parent, []).append(e)
        for batch_dir, eps in sorted(pending.items(), key=lambda kv: (len(kv[1]), kv[0])):
            if self._left_s() < self.LAUNCH_FLOOR_S:
                print(f"[orchestrate] replay of {batch_dir.name}: under "
                      f"{self.LAUNCH_FLOOR_S // 60:.0f} min left — not launched", flush=True)
                continue
            p = sh([self.cfg.isaac_py, ROOT / "scripts" / "replay_check.py", self.camp.gen,
                    "--headless", "--episodes", *eps],
                   log=self.camp.logs / f"replay_{batch_dir.name}.log",
                   timeout=self._batch_timeout(), check=False)
            if p.returncode != 0:
                why = "timed_out" if p.timed_out else ("killed" if p.returncode == -9 else "crashed")
                n = 0
                for e in eps:
                    mp = self.camp.gen / e / "meta.json"
                    meta = read_json(mp)
                    if "replay" not in meta:
                        meta["replay"] = {"success": False, why: True, "rc": p.returncode}
                        write_json_atomic(mp, meta)
                        n += 1
                print(f"[orchestrate] replay of {batch_dir.name} {why} — {n} episodes recorded "
                      "unverifiable", flush=True)
        return {e: bool((read_json(self.camp.gen / e / "meta.json").get("replay") or {}).get("success"))
                for e in episodes}

    def _pool(self) -> dict[str, dict]:
        """Candidate episodes: graded successful, not vacuous, from a COMPLETE batch whose
        recorded code fingerprint equals the cell's current code (an episode never vouches
        for code that was edited after it was recorded). -> {episode: {cell, batch, replay}}"""
        fps: dict[str, str] = {}
        out: dict[str, dict] = {}
        for bmeta_p in sorted(self.camp.gen.glob("data/*/meta.json")):
            bmeta = read_json(bmeta_p)
            cell = bmeta.get("cell")
            if not cell or not bmeta.get("episodes"):
                continue
            if cell not in fps:
                fps[cell] = contract.cell_fingerprint(self.camp.gen, cell)  # a pruned cell hashes to <missing>
            if bmeta.get("cell_fingerprint") != fps[cell]:
                continue
            for ep_meta in sorted(bmeta_p.parent.glob("ep_*/meta.json")):
                meta = read_json(ep_meta)
                if meta.get("success") and not meta.get("vacuous"):
                    ep = str(ep_meta.parent.relative_to(self.camp.gen))
                    out[ep] = {"cell": cell, "batch": bmeta_p.parent.name,
                               "replay": (meta.get("replay") or {}).get("success")}
        return out

    def _traj_identity(self, ep: str) -> str:
        p = self.camp.gen / ep / "traj.npz"
        return hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else ep

    # ---- pre-checks ---------------------------------------------------------------
    # `robot.reset(` added 2026-09-07: a mid-solve joint reset (spatula's `dewind()` helper)
    # teleports the arm — open-loop replay cannot reproduce it, and it was not being named.
    _PRIVILEGED_WRITES = re.compile(
        r"scene\.[a-z_]*drive\b|set_external_force|write_root_(state|pose|velocity)|"
        r"write_joint_state|\bset_states\(|\.set_state\(|controller\.reset\(|robot\.reset\(")
    # Controller/articulation RETUNES are a class of their own: legal under the eval contract
    # ("gains/targets only — all free"), and the idiom of every reference solve, but the replay
    # rebuilds the world with the PRESET parameters and never re-applies them — so a success
    # recorded under a retuned law diverges from step 1 when its actions run under the preset
    # law. Before 2026-09-06 only `.cfg.x =` was matched: bulb's brief named ONE line
    # (`rot_scale`) while `_kp/_kd`, `_control_period = 1` and the finger
    # `write_joint_stiffness_to_sim` went unnamed, and four tasks spent 3-4 blind repair
    # rounds re-deriving what this regex now states.
    _CONTROLLER_RETUNES = re.compile(
        r"\._k[pd]\s*=|\._q_default\s*=|_control_period\s*=|\.cfg\.\w+\s*=|"
        r"write_joint_(stiffness|damping)|task_(prop|deriv)_gains\s*=|\bk[pd]_null\s*=")

    def _matching_lines(self, cell: Cell, pattern: re.Pattern) -> list[str]:
        hits = []
        for f in sorted(self.camp.solve_py(cell).parent.glob("*.py")):
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                code = line.split("#", 1)[0]
                if pattern.search(code):
                    hits.append(f"{f.name}:{n}: {code.strip()}")
        return hits

    def _privileged_writes(self, cell: Cell) -> list[str]:
        """'file:line: code' wherever a cell acts on the world other than through the
        robot's actions (drive inputs, forces, state writes, controller resets)."""
        return self._matching_lines(cell, self._PRIVILEGED_WRITES)

    def _controller_retunes(self, cell: Cell) -> list[str]:
        """'file:line: code' wherever a cell changes controller or articulation parameters."""
        return self._matching_lines(cell, self._CONTROLLER_RETUNES)

    def _recorded_law(self, ep: str) -> str:
        """One-paragraph summary of the controller the episode was RECORDED under (the
        recorder samples it into the episode meta), so the brief can show the agent the law
        its actions were computed for — the law the replay will not have."""
        c = read_json(self.camp.gen / ep / "meta.json").get("controller") or {}
        if not c:
            return ""
        parts = []
        for leaf in c.get("leaves", []):
            cfg = leaf.get("cfg") or {}
            desc = [f"control_period={leaf.get('control_period')}"]
            for k in ("kp", "kd"):
                if leaf.get(k) is not None:
                    desc.append(f"{k}={leaf[k]}")
            for k in ("rot_scale", "pos_scale", "kp_null", "kd_null", "ema_factor"):
                if k in cfg:
                    desc.append(f"cfg.{k}={cfg[k]}")
            parts.append(f"{leaf.get('class')}: " + ", ".join(desc))
        names, ks, ds = c.get("joint_names") or [], c.get("joint_stiffness") or [], c.get("joint_damping") or []
        pd = [f"{n} k={k:g}/d={d:g}" for n, k, d in zip(names, ks, ds) if k or d]
        if pd:
            parts.append("articulation joint PD (non-zero): " + ", ".join(pd))
        return "\n    ".join(parts)

    def _probe_nominal(self) -> tuple[bool, dict]:
        """A 1-env nominal run must succeed AND replay. Probes are named by the code
        fingerprint (a repair gets a fresh probe by construction). A probe cut at
        nominal_timeout_s is a stuck solve: no further seeds, straight to repair."""
        fp = self._fingerprint(BASE_CELL)
        last, note = "", ""
        for seed in self.cfg.nominal_seeds:
            name, meta = self._run_or_read_batch(
                BASE_CELL, f"batch_nominal_{fp}_s{seed}",
                ["--nominal", "--num_envs", "1", "--seed", str(seed)],
                cap=self.cfg.nominal_timeout_s)
            last = name
            if meta.get("timed_out") or meta.get("killed"):
                why = "was CUT" if meta.get("timed_out") else "was KILLED (out of memory)"
                note = (f"The nominal run (seed {seed}) {why} after "
                        f"{self.cfg.nominal_timeout_s / 60:.0f} min without finishing (the log ends "
                        "where it stopped). A 1-env nominal run must finish well inside that: the "
                        "solve is stuck in a retry loop or far too slow. Find where it stops making "
                        "progress and make it either succeed or give up quickly.")
                print(f"[orchestrate] nominal probe seed={seed} {why}", flush=True)
                break
            if not meta.get("successes"):
                print(f"[orchestrate] nominal probe seed={seed} failed "
                      f"({meta.get('successes', 'no meta')} successes)", flush=True)
                continue
            ep = f"data/{name}/ep_0000"
            if self._replay_verify([ep])[ep]:
                return True, {"batch": name, "replay_note": ""}
            writes = self._privileged_writes(BASE_CELL)
            retunes = self._controller_retunes(BASE_CELL)
            law = self._recorded_law(ep)
            note = (f"The nominal run (seed {seed}, `{ep}`) SUCCEEDED but its recorded robot actions "
                    "do NOT reproduce the success when replayed. The replay rebuilds the batch's "
                    "world, restores the recorded initial state, puts the controller under the law "
                    "the run was recorded with (gains, control period, nullspace posture, gripper PD "
                    "— including every change made mid-solve, at its recorded step) and feeds the "
                    "recorded actions back open-loop. Controller parameters your solve sets are "
                    "therefore NOT the cause. What does break a replay: acting on the world outside "
                    "env.step (state writes, external forces, drive inputs), reading anything the "
                    "recording does not carry (wall clock, unseeded randomness), or a motion whose "
                    "outcome hinges on contact details that do not reproduce (widen margins, settle "
                    "before releasing, verify before moving on)."
                    + ("\nThe solve acts on the world outside env.step here — never allowed:\n    "
                       + "\n    ".join(writes) if writes else "")
                    + (f"\nFor reference, the law the successful run was recorded under (re-applied "
                       f"on replay):\n    {law}" if law else "")
                    + (f"\nController parameters the solve sets (fine; recorded and re-applied):\n    "
                       + "\n    ".join(retunes) if retunes else ""))
            print(f"[orchestrate] nominal probe seed={seed}: success does not REPLAY"
                  + (f" ({len(writes)} privileged writes)" if writes else "")
                  + (f" ({len(retunes)} controller parameter lines, re-applied on replay)" if retunes else ""),
                  flush=True)
        return False, {"batch": last, "replay_note": note}

    def ensure_nominal(self) -> bool:
        self._enter("checks")
        self.camp.write_status(Stage.NOMINAL)
        candidates = sorted(str(p) for p in (self.cfg.run_dir / "workspace" / "candidates").glob("*/"))
        source = read_json(self.cfg.run_dir / "run.json")

        def quick() -> bool:
            # the agent replay-checked a 1-env success of the base cell under its current code
            return any(rec["cell"] == BASE_CELL.key and rec["replay"]
                       and self.camp.batch_meta(rec["batch"]).get("num_envs") == 1
                       for rec in self._pool().values())

        ok = self._sessions(
            "repair", Stage.REPAIR, evaluate=self._probe_nominal, quick=quick,
            make_brief=lambda ev: self.agent.brief(
                "repair", gen=self.camp.gen, solve=self.camp.solve_py(BASE_CELL),
                grader=self.camp.grader_py("scene_0"),
                fail_log_path=self.camp.log_path(ev["batch"]), fail_log_size=self.camp.log_size(ev["batch"]),
                replay_note=ev.get("replay_note", ""),
                nominal_seeds=", ".join(map(str, self.cfg.nominal_seeds)), num_envs=self.cfg.num_envs,
                hours_left=self._hours_left(),
                source_status=source.get("source_status", "unknown"),
                candidates=("\n".join(f"  - {c}" for c in candidates) or "  (none)")))
        self.camp.write_status(Stage.NOMINAL, nominal_ok=ok)
        print(f"[orchestrate] nominal {'PASSES' if ok else 'still failing'}", flush=True)
        return ok

    def ensure_wide(self) -> bool:
        """The num_envs-wide batch must complete env-batched with >= 1 success that
        REPLAYS at that width — the ladder is built from wide batches."""
        if self.cfg.num_envs <= 1:
            return True

        def wide_ok() -> tuple[bool, dict]:
            name, meta = self._run_or_read_batch(
                BASE_CELL, f"batch_wide_probe_{self._fingerprint(BASE_CELL)}",
                ["--num_envs", str(self.cfg.num_envs), "--seed", str(self.cfg.wide_probe_seed)])
            got = meta.get("successes") or 0
            replayed = 0
            if got:
                eps = [str(p.relative_to(self.camp.gen)) for p in
                       sorted((self.camp.gen / "data" / name).glob("ep_*"))
                       if read_json(p / "meta.json").get("success")]
                replayed = sum(self._replay_verify(eps).values())
            why = ("no meta (crashed/killed/cut)" if not meta else
                   f"{got}/{self.cfg.num_envs} succeeded, {replayed} of them replay")
            print(f"[orchestrate] wide probe {name}: {why}", flush=True)
            return bool(meta) and replayed >= 1, {"batch": name, "successes": got, "replayed": replayed}

        def quick() -> bool:
            # a replayed success of the base cell in a batch of the campaign's width
            return any(rec["cell"] == BASE_CELL.key and rec["replay"]
                       and self.camp.batch_meta(rec["batch"]).get("num_envs") == self.cfg.num_envs
                       for rec in self._pool().values())

        self.camp.write_status(Stage.WIDE)
        ok = self._sessions(
            "vectorize", Stage.VECTORIZE, evaluate=wide_ok, quick=quick,
            make_brief=lambda ev: self.agent.brief(
                "vectorize", gen=self.camp.gen, num_envs=self.cfg.num_envs,
                solve=self.camp.solve_py(BASE_CELL),
                fail_log_path=self.camp.log_path(ev["batch"]), fail_log_size=self.camp.log_size(ev["batch"]),
                probe_seed=self.cfg.wide_probe_seed, hours_left=self._hours_left(),
                successes=ev.get("successes", 0), replayed=ev.get("replayed", 0)))
        self.camp.write_status(Stage.WIDE, wide_ok=ok)
        print(f"[orchestrate] wide probe {'OK' if ok else 'still failing'}", flush=True)
        return ok

    # ---- authoring stages ---------------------------------------------------------
    @staticmethod
    def _authored_at(level: str, cell_key: str) -> bool:
        parts = cell_key.split("/")
        if level == "scene":
            return parts[0] != BASE_CELL.scene
        if level == "strategy":
            return parts[0] == BASE_CELL.scene and parts[1] != BASE_CELL.strategy and len(parts) == 2
        return len(parts) == 3

    def _replayed_nominal(self) -> str | None:
        for ep, rec in sorted(self._pool().items()):
            if rec["batch"].startswith("batch_nominal_") and rec["replay"]:
                return ep
        return None

    def _prev_set(self, level: str) -> list[str]:
        """The rung below: the previous stage's set, or for the first stage the replay-
        verified nominal episode (the 1 the ladder starts from). That episode is read from
        the pre-check fact: the scene brief has the agent add the PHYSICAL_PARAMS/VISUAL_
        PARAMS/CAMERAS declarations to scene_0, which changes the base fingerprint and would
        otherwise make rung 1 vanish from the pool (scene_0 is re-verified under its final
        code by certify_base and the harvest batches)."""
        i = self.cfg.sessions.index(level)
        if i > 0:
            return self.camp.stage_set(self.cfg.sessions[i - 1])
        nominal = (self.camp.ledger().get("prechecks_passed") or {}).get("nominal") or self._replayed_nominal()
        return [nominal] if nominal else []

    def _gate(self, level: str, dry: bool = False) -> tuple[bool, dict]:
        """The rung: >= target verified episodes, every cell this level kept proven by >= 1,
        >= 1 new cell. Replays are spent only where they can change the verdict: first to
        prove each unproven new cell (cheapest batch first), then to fill the count (new
        cells first, then older cells); nothing is replayed while a new cell has no
        successful episode at all (the gate cannot pass — the agent must fix or delete it).
        `dry`: the mid-session poll — counts only verdicts already on disk, replays nothing,
        writes nothing."""
        need = self.cfg.target(level)
        if not dry:
            self._prune_clone_cells(self._cells_at_stage_start)   # BEFORE counting: a pruned
        prev = self._prev_set(level)                               # cell must never enter a set
        pool = self._pool()
        new_cells = sorted(c.key for c in self.camp.cells() if self._authored_at(level, c.key))
        by_cell: dict[str, dict[str, list[str]]] = {}
        for ep, rec in pool.items():
            by_cell.setdefault(rec["cell"], {}).setdefault(rec["batch"], []).append(ep)

        def verified_of(cell: str) -> int:
            return sum(1 for ep, rec in pool.items() if rec["cell"] == cell and rec["replay"])

        def replay_cell(cell: str, until_total: int | None, until_proven: bool) -> None:
            for batch, eps in sorted(by_cell.get(cell, {}).items(), key=lambda kv: (len(kv[1]), kv[0])):
                if until_proven and verified_of(cell) >= 1:
                    return
                if until_total is not None and self._verified_total(pool, prev) >= until_total:
                    return
                pend = [e for e in eps if pool[e]["replay"] is None]
                if pend:
                    res = self._replay_verify(pend)
                    for e in pend:
                        pool[e]["replay"] = res[e]

        dead = [c for c in new_cells if not by_cell.get(c)]
        if new_cells and not dead and not dry:
            for c in new_cells:
                replay_cell(c, None, until_proven=True)
            for c in new_cells:
                replay_cell(c, need, until_proven=False)
            for c in sorted(set(by_cell) - set(new_cells)):
                replay_cell(c, need, until_proven=False)
        unproven = [c for c in new_cells if verified_of(c) == 0]
        verified = {ep: rec["cell"] for ep, rec in pool.items() if rec["replay"]}
        for ep in prev:                        # rung below: kept as delivered, whatever its cell's code is now
            verified.setdefault(ep, pool.get(ep, {}).get("cell") or self.camp.batch_meta(ep.split("/")[1]).get("cell", ""))
        total = len(verified)
        ok = bool(new_cells) and not unproven and total >= need
        if ok:
            picked, short = contract.select_stage_set(prev, verified, set(new_cells), need,
                                                      identity=self._traj_identity)
            if short:
                ok = False
                if not dry:
                    print(f"[orchestrate] {level} gate: {total} verified but only {len(picked)} "
                          "DISTINCT trajectories — duplicates do not count", flush=True)
            elif not dry:
                write_json_atomic(self.camp.gen / f"{level}_set.json", {
                    "episodes": picked, "target": need, "level": level, "previous_set": len(prev),
                    "cells": sorted({verified.get(e, "") for e in picked}),
                    "code_hash": contract.code_hash(),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        evidence = {"total": total, "need": need, "new_cells": new_cells, "dead_cells": dead,
                    "unproven_cells": unproven}
        if not dry:
            print(f"[orchestrate] {level} gate: {total}/{need} verified+replayed, new cells "
                  f"{len(new_cells)}, no episode yet {dead}, none replayed yet {unproven} -> "
                  f"{'PASSED — ' + level + '_set.json delivered' if ok else 'not yet'}", flush=True)
        return ok, evidence

    @staticmethod
    def _verified_total(pool: dict, prev: list[str]) -> int:
        return len({ep for ep, rec in pool.items() if rec["replay"]} | set(prev))

    def author(self, level: str) -> bool:
        """One authoring gate: the framework's condition brief plus this rung's evidence;
        sessions get the whole remaining clock; accepted mechanically by _gate."""
        self._enter(level)
        if self.camp.stage_set(level):
            return True
        self.camp.write_status(Stage.SESSION, level=level)
        dry = sh([sys.executable, ROOT / "scripts" / "diversify.py", self.camp.gen,
                  ROOT / "configs" / f"{level}_default.yaml", "--host", "--dry"],
                 timeout=self.cfg.batch_timeout_s, check=True)
        match = re.search(r"^instructions: (.+)$", dry.stdout, re.M)
        if match is None:
            raise RuntimeError(f"diversify --dry reported no instructions path:\n{dry.stdout}{dry.stderr}")
        instructions = Path(match.group(1)).read_text()

        def make_brief(ev: dict) -> str:
            lst = lambda xs: ("\n".join(f"  - {c}" for c in xs) or "  (none)")
            prev = self._prev_set(level)
            return instructions + f"""

## Your goal for this stage (mechanical, checked by the orchestrator)

This stage delivers EXACTLY {ev['need']} trajectories: the {len(prev)} already delivered by the
previous stage, plus {ev['need'] - len(prev)} new ones. A trajectory counts only if it is a graded
success, its recorded actions REPLAY (reproduce the success at the batch's width), and the code
of its cell has not changed since it was recorded — re-test a cell after your last edit to it.

How the {ev['need'] - len(prev)} are chosen — this decides what is worth your time:
- ROUND-ROBIN ACROSS YOUR CELLS, one trajectory per cell per round, your new cells first.
  Diversity comes from the number of distinct WORKING cells, not from episode count: with
  3 cells and {ev['need'] - len(prev)} slots each cell contributes ~{max(1, (ev['need'] - len(prev)) // 3)}; a 4th
  working cell is worth more than 500 extra episodes in an existing one. Episodes beyond a
  cell's share are never used — do not pad, do not overshoot.
- Byte-identical trajectories (the same seed re-run in the same world) count once.
- EVERY cell you keep must contribute at least one counted trajectory, and at least one new
  cell must exist. A cell that never yields blocks the stage: fix it or delete it.

The orchestrator checks this every few minutes while you work and CLOSES YOUR SESSION as
soon as the goal is met — you do not need to stop yourself, and working past the goal
changes nothing. Replay-check your own batches (`replay_check`) so the check can see them.
State now: {ev['total']}/{ev['need']} counted.
Cells with no successful episode at all:
{lst(ev['dead_cells'])}
Cells with successes but none that replays yet:
{lst(ev['unproven_cells'])}
Probe cells at a small width (`--num_envs 8` to `32`): the gate replays whole batches, and a
512-env batch costs an hour of clock to verify however few episodes it needs from it."""

        self._cells_at_stage_start = {c.key for c in self.camp.cells()}
        ok = self._sessions(f"author_{level}", Stage.SESSION, evaluate=lambda: self._gate(level),
                            make_brief=make_brief, whole_clock=True, level=level,
                            quick=lambda: self._gate(level, dry=True)[0])
        print(f"[orchestrate] {level} stage {'PASSED' if ok else 'NOT MET'} "
              f"(cells now: {len(self.camp.cells())})", flush=True)
        return ok

    def _prune_clone_cells(self, before: set[str]) -> None:
        """Delete new cells byte-identical to an existing cell (unedited templates)."""
        seen = {self._fingerprint(c): c.key for c in self.camp.cells() if c.key in before}
        for cell in self.camp.cells():
            if cell.key in before or not self.camp.solve_py(cell).is_file():
                continue
            fp = self._fingerprint(cell)
            if fp not in seen:
                seen[fp] = cell.key
                continue
            root = self.camp.gen / "scenes" / cell.scene
            if cell.scene in {k.split("/")[0] for k in before}:
                root = root / "strategies" / cell.strategy
                if cell.phase and f"{cell.scene}/{cell.strategy}" in {"/".join(k.split("/")[:2]) for k in before}:
                    root = self.camp.phase_dir(cell)
            print(f"[orchestrate] pruning {cell.key}: byte-identical to {seen[fp]}", flush=True)
            shutil.rmtree(root, ignore_errors=True)

    # ---- dynamics -----------------------------------------------------------------
    def _has_noise_channel(self, cell: Cell) -> bool:
        files = list(self.camp.solve_py(cell).parent.glob("*.py"))
        if cell.phase:
            files += self.camp.phase_dir(cell).glob("*.py")
        return any(re.search(r"\bnoise\s*=", p.read_text(errors="replace")) for p in files if p.is_file())

    def _declares(self, scene: str, name: str) -> bool:
        return re.search(rf"^\s*{name}\s*=\s*\{{[^}}]", self.camp.scene_py(scene).read_text(), re.M) is not None

    def certify_base(self) -> tuple[bool, dict]:
        """Noise acceptance on the base cell: it passes noise= to env.step, the base scene
        declares PHYSICAL_PARAMS (otherwise every env is the same world and the "physics"
        set has no physics variation), and an identical-seed clean+noisy num_envs pair
        shows the clean solve still succeeding, the noise executing (coverage > 0) and at
        least one PERTURBED episode succeeding AND replaying."""
        fp = self._fingerprint(BASE_CELL)
        recorded = self.camp.ledger().get("noise_cert", {}).get(fp)
        if recorded is not None:
            return bool(recorded.get("ok")), dict(recorded)
        ev = {"fingerprint": fp, "noise_channel": self._has_noise_channel(BASE_CELL),
              "physical_params": self._declares("scene_0", "PHYSICAL_PARAMS"),
              "clean_successes": None, "coverage": 0.0, "perturbed_successes": 0,
              "perturbed_replayed": 0, "noisy_batch": ""}
        if ev["noise_channel"] and ev["physical_params"]:
            args = ["--num_envs", str(self.cfg.num_envs), "--seed", str(self.cfg.wide_probe_seed)]
            _, clean = self._run_or_read_batch(BASE_CELL, f"batch_cert_{fp}_clean", args)
            noisy_name, noisy = self._run_or_read_batch(
                BASE_CELL, f"batch_cert_{fp}_noisy", args + ["--noise_scale", str(self.cfg.noise_scale)])
            perturbed = [str(p.parent.relative_to(self.camp.gen)) for p in
                         sorted((self.camp.gen / "data" / noisy_name).glob("ep_*/meta.json"))
                         if (m := read_json(p)).get("success") and (m.get("noise") or {}).get("perturbed")]
            ev.update(clean_successes=clean.get("successes"),
                      coverage=float((noisy.get("noise") or {}).get("perturbed_row_frac") or 0.0),
                      perturbed_successes=len(perturbed), noisy_batch=noisy_name,
                      perturbed_replayed=sum(self._replay_verify(perturbed).values()) if perturbed else 0)
            if clean.get("timed_out") or clean.get("killed") or noisy.get("timed_out") or noisy.get("killed"):
                print("[orchestrate] certification batch cut — not recorded, re-run next time", flush=True)
                return False, ev
        ev["ok"] = bool(ev["noise_channel"] and ev["physical_params"] and ev["clean_successes"]
                        and ev["coverage"] > 0 and ev["perturbed_replayed"] >= 1)
        certs = dict(self.camp.ledger().get("noise_cert", {}))
        certs[fp] = ev
        self.camp.update_ledger(noise_cert=certs)
        print(f"[orchestrate] noise certification: channel={ev['noise_channel']} physical_params="
              f"{ev['physical_params']} clean={ev['clean_successes']} coverage={ev['coverage']:.4f} "
              f"perturbed ok/replayed={ev['perturbed_successes']}/{ev['perturbed_replayed']} -> "
              f"{'CERTIFIED' if ev['ok'] else 'REJECTED'}", flush=True)
        return ev["ok"], ev

    def ensure_noise_plan(self) -> bool:
        self._enter("dynamics")
        base_eps = [self.camp.gen / e for e in self.camp.base_set()]
        if not base_eps:
            print("[orchestrate] no base set — nothing to author noise against", flush=True)
            return False

        def sample_video() -> Path | None:
            ep = base_eps[0]
            vids = sorted(ep.glob("imgs/*.mp4"))
            if not vids:
                cmd = [self.cfg.isaac_py, ROOT / "scripts" / "render.py", self.camp.gen,
                       "--episodes", ep, "--num_envs", "1", "--no-sheet", "--headless"]
                if self.cfg.render_kit_args:
                    cmd += [f"--kit_args={self.cfg.render_kit_args}"]
                sh(cmd, log=self.camp.logs / "noise_sample_render.log",
                   timeout=self._batch_timeout(), check=False)
                vids = sorted(ep.glob("imgs/*.mp4"))
            return vids[0] if vids else None

        cells = sorted({read_json(e / "meta.json").get("cell", "") for e in base_eps})

        def make_brief(ev: dict) -> str:
            video = sample_video()
            return self.agent.brief(
                "noise", gen=self.camp.gen, solve=self.camp.solve_py(BASE_CELL),
                scene=self.camp.scene_py("scene_0"),
                cell_solves="\n".join(f"  - {c}: `{self.camp.solve_py(Cell.parse(c))}`" for c in cells),
                video=(video or "NOT RENDERED — render it yourself first"), sample_ep=base_eps[0],
                num_envs=self.cfg.num_envs, noise_scale=self.cfg.noise_scale,
                probe_seed=self.cfg.wide_probe_seed,
                certification=json.dumps({k: ev.get(k) for k in (
                    "noise_channel", "physical_params", "clean_successes", "coverage",
                    "perturbed_successes", "perturbed_replayed")}),
                fail_log_path=self.camp.log_path(ev.get("noisy_batch", "")),
                fail_log_size=self.camp.log_size(ev.get("noisy_batch", "")),
                hours_left=self._hours_left())

        def quick() -> bool:
            # The agent's own noisy batch of the base cell has a perturbed success that replays
            # — at the CAMPAIGN WIDTH, and only for code the certification has not already
            # judged. certify_base caches its verdict per code fingerprint, so once it has
            # rejected the current code nothing but a code change can pass; a poll that said
            # "goal met" on an 8-env test of that same code closed every session after its
            # first 5 minutes and re-read the cached rejection — 80 rounds in 6.5 h with the
            # agent never given time to change anything (pc_gpu_ram, 2026-09-07).
            if not (self._has_noise_channel(BASE_CELL) and self._declares("scene_0", "PHYSICAL_PARAMS")):
                return False
            fp = self._fingerprint(BASE_CELL)
            judged = self.camp.ledger().get("noise_cert", {}).get(fp)
            if judged is not None and not judged.get("ok"):
                return False
            for ep, rec in self._pool().items():
                if rec["cell"] == BASE_CELL.key and rec["replay"] \
                        and self.camp.batch_meta(rec["batch"]).get("num_envs") == self.cfg.num_envs \
                        and (read_json(self.camp.gen / ep / "meta.json").get("noise") or {}).get("perturbed"):
                    return True
            return False

        return self._sessions("noise", Stage.NOISE_PLAN, evaluate=self.certify_base,
                              make_brief=make_brief, quick=quick)

    def _survivors(self) -> tuple[list[str], dict[str, str]]:
        """Replay-verified harvest successes (current code only), with their cells."""
        pool = self._pool()
        eps = sorted(ep for ep, rec in pool.items() if rec["batch"].startswith("batch_dynamics_") and rec["replay"])
        return eps, {ep: pool[ep]["cell"] for ep in eps}

    def _write_physics_set(self) -> tuple[bool, int, int]:
        base = self.camp.base_set()
        eps, cell_of = self._survivors()
        picked, short = contract.select_physics_set(base, eps, self.cfg.physics_target,
                                                    cell_of=cell_of.get, identity=self._traj_identity)
        if short == 0:
            write_json_atomic(self.camp.gen / "physics_set.json", {
                "episodes": picked, "target": self.cfg.physics_target, "base_set": len(base),
                "survivors_total": len(eps), "code_hash": contract.code_hash(),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return short == 0, len(picked), short

    def harvest(self) -> bool:
        """Scripted physics diversification: round-robin over the base set's cells, each
        batch num_envs wide with its own physics/solve draws and the certified noise at
        noise_scale; each batch's successes are replay-verified at once; a cell whose first
        batch yields nothing verified is dropped. Ends when physics_set.json is exact."""
        self._enter("dynamics")
        if self.cfg.physics_target <= self.cfg.base_target:
            return self._write_physics_set()[0]
        existing = {c.key for c in self.camp.cells()}
        base_cells = sorted({read_json(self.camp.gen / e / "meta.json").get("cell", "")
                             for e in self.camp.base_set()} & existing)
        live = [Cell.parse(c) for c in base_cells if c not in set(self.camp.ledger().get("harvest_dead", []))]
        idx = len(list(self.camp.gen.glob("data/batch_dynamics_*")))
        while live and self._left_s() >= self.LAUNCH_FLOOR_S:
            delivered, picked, short = self._write_physics_set()
            self.camp.write_status(Stage.HARVEST, physics_set=picked, physics_target=self.cfg.physics_target,
                                   batches=idx, live_cells=[c.key for c in live])
            if delivered:
                break
            _, cell_of = self._survivors()
            got = {c.key: 0 for c in live}
            for cell in cell_of.values():
                if cell in got:
                    got[cell] += 1
            cell = min(live, key=lambda c: (got[c.key], c.key))
            idx += 1
            name = f"batch_dynamics_{idx:04d}"
            # every batch owns a disjoint slice of the physics-draw index space (slot 0 of
            # each batch is the nominal world, the other num_envs-1 slots draw consecutively)
            meta = self.run_batch(cell, name, ["--num_envs", str(self.cfg.num_envs),
                                               "--seed", str(self.cfg.dynamics_seed_base + idx),
                                               "--env_draw", str(idx * max(1, self.cfg.num_envs - 1)),
                                               "--solve_draw", str(idx),
                                               "--noise_scale", str(self.cfg.noise_scale)])
            if not meta:
                break
            eps = [str(p.parent.relative_to(self.camp.gen)) for p in
                   sorted((self.camp.gen / "data" / name).glob("ep_*/meta.json"))
                   if read_json(p).get("success")]
            verified = sum(self._replay_verify(eps).values()) if eps else 0
            print(f"[orchestrate] harvest {name} ({cell.key}): {len(eps)} successes, "
                  f"{verified} replay", flush=True)
            if verified == 0 and got[cell.key] == 0:
                live = [c for c in live if c.key != cell.key]
                self.camp.update_ledger(harvest_dead=sorted(set(self.camp.ledger().get("harvest_dead", []))
                                                            | {cell.key}))
                print(f"[orchestrate] harvest: {cell.key} yields nothing verified — dropped", flush=True)
        delivered, picked, short = self._write_physics_set()
        self.camp.write_status(Stage.HARVEST, physics_set=picked, physics_target=self.cfg.physics_target,
                               physics_shortfall=short)
        print(f"[orchestrate] dynamics: physics set {picked}/{self.cfg.physics_target}"
              + ("" if delivered else f" — short by {short}"), flush=True)
        return delivered

    # ---- visual -------------------------------------------------------------------
    def _render_probe_ok(self, scene: str, scene_hash: str, ep: Path) -> bool:
        """Render ONE verified episode of the scene under the `_probe` suffix: proves the
        edited scene builds on the replay path too."""
        key = f"{scene}:{scene_hash}"
        recorded = self.camp.ledger().get("render_probes", {}).get(key)
        if recorded is not None:
            return bool(recorded)
        if self._left_s() < self.LAUNCH_FLOOR_S:
            return False                      # not recorded: re-probed when there is clock
        for stale in (ep / "imgs").glob("*_probe*") if (ep / "imgs").is_dir() else []:
            stale.unlink()
        # a handful of frames is enough to prove the edited scene builds and renders
        cmd = [self.cfg.isaac_py, ROOT / "scripts" / "render.py", self.camp.gen, "--episodes", ep,
               "--num_envs", "1", "--no-sheet", "--headless", "--max-frames", "8", "--view-suffix", "_probe"]
        if self._declares(scene, "VISUAL_PARAMS"):
            cmd += ["--visual_draw", "1"]
        if self.cfg.render_kit_args:
            cmd += [f"--kit_args={self.cfg.render_kit_args}"]
        p = sh(cmd, log=self.camp.logs / f"render_probe_{scene}_{scene_hash}.log",
               timeout=self._batch_timeout(), check=False)
        ok = p.returncode == 0 and any((ep / "imgs").glob("render_*_probe.json"))
        probes = dict(self.camp.ledger().get("render_probes", {}))
        probes[key] = ok
        self.camp.update_ledger(render_probes=probes)
        print(f"[orchestrate] render probe {scene}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    def ensure_visual_params(self, physics: list[str]) -> bool:
        """VISUAL_PARAMS and CAMERAS are required on every scene in the physics set; a
        scene edit is accepted only if one of its physics-set episodes still renders."""
        scenes = sorted({read_json(self.camp.gen / e / "meta.json").get("cell", "").split("/")[0]
                         for e in physics} - {""})
        first_ep = {}
        for e in physics:
            first_ep.setdefault(read_json(self.camp.gen / e / "meta.json").get("cell", "").split("/")[0],
                                self.camp.gen / e)

        def evaluate() -> tuple[bool, dict]:
            missing, broken = [], []
            for s in scenes:
                if not (self._declares(s, "VISUAL_PARAMS") and self._declares(s, "CAMERAS")):
                    missing.append(s)
                    continue
                h = hashlib.sha256(self.camp.scene_py(s).read_bytes()).hexdigest()[:12]
                if not self._render_probe_ok(s, h, first_ep[s]):
                    broken.append(s)
            return not missing and not broken, {"missing": missing, "broken": broken}

        return self._sessions(
            "visual", Stage.SESSION_VISUAL, evaluate=evaluate,
            quick=lambda: all(self._declares(s, "VISUAL_PARAMS") and self._declares(s, "CAMERAS")
                              for s in scenes),
            make_brief=lambda ev: self.agent.brief(
                "visual_params", gen=self.camp.gen, scenes=", ".join(ev["missing"] + ev["broken"]),
                scene_paths="\n".join(str(self.camp.scene_py(s)) for s in ev["missing"] + ev["broken"]),
                num_envs=self.cfg.num_envs, hours_left=self._hours_left()))

    def _render_manifest(self, physics: list[str]) -> dict:
        manifest = read_json(self.camp.render_manifest_path)
        if (manifest.get("draws") != self.cfg.visual_draws
                or {it["episode"] for it in manifest.get("items", [])} != set(physics)):
            manifest = contract.build_manifest(self.camp.gen, physics, self.cfg.visual_draws)
        for j in range(self.cfg.visual_draws):
            contract.mark_items_complete(manifest, self.camp.gen, physics, j)
        write_json_atomic(self.camp.render_manifest_path, manifest)
        return manifest

    def _pull_renders(self, physics: list[str]) -> int:
        """Copy render output that workers pushed to the mirror (DGEN_MIRROR_TASK_DIR) into
        this campaign's episode dirs; returns files copied."""
        mirror = os.environ.get("DGEN_MIRROR_TASK_DIR")
        if not mirror:
            return 0
        n = 0
        for e in physics:
            src = Path(mirror) / "data_gen" / self.camp.name / e / "imgs"
            if not src.is_dir():
                continue
            dst = self.camp.gen / e / "imgs"
            dst.mkdir(exist_ok=True)
            for f in src.iterdir():
                if f.is_file() and not (dst / f.name).exists():
                    shutil.copyfile(f, dst / f.name)
                    n += 1
        if n:
            print(f"[orchestrate] pulled {n} render files from workers", flush=True)
        return n

    def _my_items(self, items: list[dict]) -> list[dict]:
        """A render worker's deterministic share of the queue; the campaign's own
        orchestrator takes everything (workers' finished items are pulled before each sweep)."""
        if not self.cfg.render_worker:
            return items
        import zlib
        k, n = self.cfg.worker_share
        return [it for it in items if zlib.crc32(f"{it['episode']}:{it['pass']}".encode()) % n == k]

    def _remark(self, manifest: dict, physics: list[str]) -> None:
        for j in range(self.cfg.visual_draws):
            contract.mark_items_complete(manifest, self.camp.gen, physics, j)
        write_json_atomic(self.camp.render_manifest_path, manifest)

    def visual(self) -> bool:
        """The render queue: exactly len(physics) x visual_draws items, pass-major; each
        completed item recorded atomically; resume renders only what is missing. Render
        workers (cfg.render_worker) render their share only and skip the contract gate."""
        self._enter("visual")
        physics = self.camp.physics_set()
        if not physics:
            raise StageFailed(Stage.STAGE_FAILED, stage_name="visual", error="no physics set to render")
        if self.cfg.visual_draws <= 0:
            return True
        if self.cfg.render_worker:
            scenes = {read_json(self.camp.gen / e / "meta.json").get("cell", "").split("/")[0] for e in physics}
            lacking = sorted(s for s in scenes
                             if not (self._declares(s, "VISUAL_PARAMS") and self._declares(s, "CAMERAS")))
            if lacking:
                raise StageFailed(Stage.STAGE_FAILED, stage_name="visual",
                                  error=f"render worker: scenes without VISUAL_PARAMS/CAMERAS {lacking} — "
                                        "the campaign's own visual gate has not passed yet")
        elif not self.ensure_visual_params(physics):
            return False
        manifest = self._render_manifest(physics)
        remaining = lambda: [it for it in manifest["items"] if not it["complete"]]
        target = len(manifest["items"])
        while self._my_items(remaining()) and self._left_s() >= self.LAUNCH_FLOOR_S:
            if self._pull_renders(physics):
                self._remark(manifest, physics)
            progressed = 0
            for shard in contract.plan_shards(self._my_items(remaining()), self.cfg.render_shard_frames):
                if self._left_s() < self.LAUNCH_FLOOR_S:
                    break
                scene, j = shard["scene"], shard["pass"]
                eps = [self.camp.gen / e for e in shard["episodes"]]
                for ep in eps:
                    for view in contract.pass_views(ep, j):
                        (ep / "imgs" / f"render_{view}.json").unlink(missing_ok=True)
                        (ep / "imgs" / f"{view}.mp4").unlink(missing_ok=True)
                cmd = [self.cfg.isaac_py, ROOT / "scripts" / "render.py", self.camp.gen, "--episodes", *eps,
                       "--num_envs", str(self.cfg.render_envs), "--no-sheet",
                       "--trim-margin", str(self.cfg.trim_margin), "--headless"]
                if j:
                    cmd += ["--view-suffix", f"_draw{j}", "--pose-jitter",
                            *map(str, (*self.cfg.cam_eye_jitter, *self.cfg.cam_target_jitter)),
                            "--pose-jitter-seed", str(self.cfg.visual_seed_base + j)]
                    if self._declares(scene, "VISUAL_PARAMS"):
                        cmd += ["--visual_draw", str(j)]
                if self.cfg.render_kit_args:
                    cmd += [f"--kit_args={self.cfg.render_kit_args}"]
                stamp = hashlib.sha256("".join(shard["episodes"]).encode()).hexdigest()[:8]
                sh(cmd, log=self.camp.logs / f"render_{scene}_d{j}_{stamp}.log",
                   timeout=self._batch_timeout(), check=False)
                flipped = contract.mark_items_complete(manifest, self.camp.gen, shard["episodes"], j)
                progressed += flipped
                write_json_atomic(self.camp.render_manifest_path, manifest)
                print(f"[orchestrate] visual {scene} pass={j}: {flipped}/{len(shard['episodes'])} items"
                      + ("" if flipped else " — RENDER PRODUCED NOTHING, check the log"), flush=True)
                self.camp.write_status(Stage.RENDER, samples_target=target,
                                       samples_done=target - len(remaining()))
            if not progressed and not self._pull_renders(physics):
                print("[orchestrate] visual made no progress over a full sweep — stopping", flush=True)
                break
        if self._pull_renders(physics):
            self._remark(manifest, physics)
        left = remaining()
        self.camp.write_status(Stage.RENDER, visual_ok=not left, samples_target=target,
                               samples_done=target - len(left))
        print(f"[orchestrate] visual: {target - len(left)}/{target} samples"
              + ("" if not left else f" — short by {len(left)}"), flush=True)
        return not left

    # ---- the pipeline -------------------------------------------------------------
    def ensure_init(self) -> None:
        if not (self.camp.gen / "gen.yaml").is_file():
            from engine.initialization import init
            init(self.cfg.run_dir, name=self.cfg.name)
        now_hash = contract.code_hash()
        ledger = self.camp.ledger()
        if ledger.get("code_hash") != now_hash:
            if ledger.get("code_hash") is not None:
                print(f"[orchestrate] CODE HASH CHANGED {ledger.get('code_hash')} -> {now_hash}: "
                      "dropping noise certifications", flush=True)
            history = list(ledger.get("code_hash_history", []))
            history.append({"hash": now_hash, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            self.camp.update_ledger(code_hash=now_hash, code_hash_history=history, noise_cert={})
        self.camp.write_status(Stage.INIT, preset=self.preset)

    def run(self) -> None:
        try:
            if self.cfg.render_worker:
                # a render worker: the campaign (physics set + scene contracts) was copied
                # from the mirror; render this worker's share; push nothing but frames
                k, n = self.cfg.worker_share
                print(f"[orchestrate] render worker {k}/{n}", flush=True)
                done = self.visual()
                print(f"[orchestrate] render worker {k}/{n} "
                      f"{'finished its share' if done else 'stopped short'}", flush=True)
                return
            self.ensure_init()
            if not self.camp.physics_set():
                if not self.camp.base_set():
                    # The pass is a recorded FACT, not re-derived from the base cell's current
                    # fingerprint: later stages may legitimately edit scene_0 (declarations the
                    # dynamics stage requires), and a resume that re-probed under the new
                    # fingerprint once failed pre-checks on their already-spent clock.
                    checks = self.camp.ledger().get("prechecks_passed")
                    if checks:
                        fp = self._fingerprint(BASE_CELL)
                        print(f"[orchestrate] pre-checks passed earlier ({checks['at']}, code {checks['fingerprint']})"
                              + (f"; base cell code is now {fp}" if fp != checks["fingerprint"] else ""), flush=True)
                    elif self.ensure_nominal() and self.ensure_wide():
                        self.camp.update_ledger(prechecks_passed={
                            "fingerprint": self._fingerprint(BASE_CELL),
                            "nominal": self._replayed_nominal(),
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    else:
                        raise StageFailed(
                            Stage.PRECHECK_FAILED,
                            error=f"no working solve within {self.cfg.stage_hours:g} h: the nominal run "
                                  f"must succeed and replay, and a {self.cfg.num_envs}-env batch must "
                                  "have a success that replays")
                    for level in self.cfg.sessions:
                        if not self.author(level):
                            raise StageFailed(Stage.STAGE_FAILED, stage_name=level,
                                              error=f"{level} rung ({self.cfg.target(level)}) not met "
                                                    f"within {self.cfg.stage_hours:g} h")
                if not self.ensure_noise_plan():
                    raise StageFailed(Stage.STAGE_FAILED, stage_name="dynamics",
                                      error=f"no certified noise plan within {self.cfg.stage_hours:g} h")
                if not self.harvest():
                    raise StageFailed(Stage.STAGE_FAILED, stage_name="dynamics",
                                      error=f"physics set ({self.cfg.physics_target}) not filled within "
                                            f"{self.cfg.stage_hours:g} h")
            if not self.visual():
                raise StageFailed(Stage.STAGE_FAILED, stage_name="visual",
                                  error=f"render queue not completed within {self.cfg.stage_hours:g} h")
            manifest = self.camp.write_manifest()
            self.camp.write_status(Stage.DONE, ladder={lvl: len(self.camp.stage_set(lvl)) for lvl in self.cfg.sessions},
                                   physics_set=len(self.camp.physics_set()), dataset_size=manifest["dataset_size"])
            print("[orchestrate] DONE", flush=True)
        except StageFailed as exc:
            self.camp.write_manifest()
            self.camp.write_status(exc.stage, ladder={lvl: len(self.camp.stage_set(lvl)) for lvl in self.cfg.sessions},
                                   physics_set=len(self.camp.physics_set()), **exc.fields)
            print(f"[orchestrate] {exc.stage.value}: {exc} — stopping.", flush=True)
            raise SystemExit(3)               # 3 = a gate was not met (0 done, 75 model route dead)
        except ModelDown as exc:
            self.camp.write_status(Stage.MODEL_DOWN, error=str(exc))
            print(f"[orchestrate] {exc} — exiting 75 (resume when the route is back)", flush=True)
            raise SystemExit(75)
        except SystemExit:
            raise
        except BaseException as exc:
            if self.camp.gen.is_dir():
                self.camp.write_status(Stage.FAILED, error_type=type(exc).__name__, error=str(exc))
            raise


# ----------------------------------------------------------------------------- cli

def parse_int_tuple(value: str) -> tuple[int, ...]:
    out = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_vec3(value: str) -> tuple[float, float, float]:
    xyz = tuple(float(v.strip()) for v in value.split(","))
    if len(xyz) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z")
    return xyz


def build_config(argv: list[str] | None = None) -> Config:
    d = Config(run_dir=Path("."))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--name", default=d.name)
    ap.add_argument("--sessions", default=",".join(d.sessions), help="authoring levels, comma-separated")
    ap.add_argument("--agent-cmd", default=d.agent_cmd)
    ap.add_argument("--isaac-py", default=d.isaac_py)
    ap.add_argument("--stage-hours", type=float, default=d.stage_hours, help="hard cap of EVERY gate")
    ap.add_argument("--intervention-session-min", type=float, default=d.intervention_session_min)
    for lvl in LEVELS:
        ap.add_argument(f"--{lvl}-target", type=int, default=d.target(lvl))
    ap.add_argument("--physics-target", type=int, default=d.physics_target)
    ap.add_argument("--visual-draws", type=int, default=d.visual_draws)
    ap.add_argument("--num-envs", type=int, default=d.num_envs)
    ap.add_argument("--wide-probe-seed", type=int, default=d.wide_probe_seed)
    ap.add_argument("--nominal-seeds", type=parse_int_tuple, default=d.nominal_seeds)
    ap.add_argument("--session-attempts-per-round", type=int, default=d.session_attempts_per_round)
    ap.add_argument("--session-retry-sleep-s", type=float, default=d.session_retry_sleep_s)
    ap.add_argument("--noise-scale", type=float, default=d.noise_scale)
    ap.add_argument("--render-envs", type=int, default=d.render_envs)
    ap.add_argument("--render-shard-frames", type=int, default=d.render_shard_frames)
    ap.add_argument("--render-kit-args", default=d.render_kit_args)
    ap.add_argument("--job-concurrency", type=int, default=d.job_concurrency)
    ap.add_argument("--trim-margin", type=int, default=d.trim_margin)
    ap.add_argument("--camera-eye-jitter", type=parse_vec3, default=d.cam_eye_jitter)
    ap.add_argument("--camera-target-jitter", type=parse_vec3, default=d.cam_target_jitter)
    ap.add_argument("--batch-timeout-s", type=float, default=d.batch_timeout_s)
    ap.add_argument("--nominal-timeout-s", type=float, default=d.nominal_timeout_s)
    ap.add_argument("--session-grace-s", type=float, default=d.session_grace_s)
    ap.add_argument("--dynamics-seed-base", type=int, default=d.dynamics_seed_base)
    ap.add_argument("--visual-seed-base", type=int, default=d.visual_seed_base)
    ap.add_argument("--render-worker", default=d.render_worker, help="K/N: render-worker mode")
    a = ap.parse_args(argv)
    return Config(
        run_dir=Path(a.run_dir).resolve(), name=a.name,
        sessions=tuple(s for s in a.sessions.split(",") if s),
        agent_cmd=a.agent_cmd, isaac_py=a.isaac_py, stage_hours=a.stage_hours,
        intervention_session_min=a.intervention_session_min,
        scene_target=a.scene_target, strategy_target=a.strategy_target, phase_target=a.phase_target,
        physics_target=a.physics_target, visual_draws=a.visual_draws, num_envs=a.num_envs,
        wide_probe_seed=a.wide_probe_seed, nominal_seeds=tuple(a.nominal_seeds),
        session_attempts_per_round=a.session_attempts_per_round,
        session_retry_sleep_s=a.session_retry_sleep_s, noise_scale=a.noise_scale,
        render_envs=a.render_envs, render_shard_frames=a.render_shard_frames,
        render_kit_args=a.render_kit_args, job_concurrency=a.job_concurrency,
        trim_margin=a.trim_margin, cam_eye_jitter=a.camera_eye_jitter,
        cam_target_jitter=a.camera_target_jitter, batch_timeout_s=a.batch_timeout_s,
        nominal_timeout_s=a.nominal_timeout_s,
        session_grace_s=a.session_grace_s, dynamics_seed_base=a.dynamics_seed_base,
        visual_seed_base=a.visual_seed_base, render_worker=a.render_worker)


def main(argv: list[str] | None = None) -> None:
    Orchestrator(build_config(argv)).run()


if __name__ == "__main__":
    main()
