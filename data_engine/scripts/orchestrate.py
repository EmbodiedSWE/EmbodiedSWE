"""orchestrate — the data_gen conductor: one verified solve in, a diversified,
per-episode-verified demonstration dataset out, unattended.

    python data_engine/scripts/orchestrate.py <run_dir> \\
        [--name gen_auto] [--sessions scene,strategy,phase] \\
        [--target-eps 50] [--num-envs 4] [--wall-hours 20] \\
        [--agent-cmd "python /path/to/agent_loop.py"] [--isaac-py /path/venv/bin/python]

`run_dir` is an eval-run-shaped folder: run.json (names the preset) + workspace/solution/
(the oracle solve) [+ workspace/candidates/<k>/ alternates].

Stage pipeline (each stage idempotent — a killed run resumes under the same --name):

    INIT       bake the campaign (local scene + suite grader, or a campaign-local
               success-only grader when the suite has none, + oracle strategy)
    NOMINAL    1-env oracle probe; failure escalates: alternate candidates -> agent
               repair sessions -> proceed regardless (never a blocking gate)
    WIDE       num_envs-wide MECHANICAL probe (the batch must complete env-batched
               with at least one success); failure loops evidence-fed vectorize
               sessions, bounded by vectorize_rounds — never a pipeline-killing
               exit: width stays enforced by construction (every scripted batch
               launches at num_envs; a scalar solve crashes instantly and its
               cell quarantines), and yield is an economics signal, not a gate
    SESSIONS   one authoring session per level (scene/strategy/phase): agent-created
               cells, each proven by graded test batches; scene sessions must ship
               VISUAL_PARAMS (verified; one targeted follow-up session if missing)
    FARM       coverage-balanced batches across cells (fewest successes first,
               least-recently-farmed tie-break, 0-yield cells quarantined) until
               EVERY live cell holds per_cell_target successes — completion is
               per-cell coverage, never a global count one wide batch can satisfy
    NOISE      one agent session: WATCH a rendered verified episode (view tool),
               then author per-phase executed-action noise INTO the solve through
               the env.step(action, noise=...) channel (labels stay clean by
               construction); accepted only when a num_envs probe run at
               compound_noise_scale clears the yield floor AND shows real
               measured noise coverage
    COMPOUND   scripted physics diversification: fresh num_envs rollouts of fertile
               cells with new draws + the accepted agent-authored noise enabled,
               graded per sim pass, until compound_eps verified survivors
    MULTIPLY   visual diversification of every verified episode: pass 0 renders the
               scene's declared cameras (nominal), passes 1..multiply_draws-1 add a
               jittered camera named draw<j> + the scene's VISUAL_PARAMS look j —
               all replay-rendered (video-native), success inherited, post-success
               tails trimmed at meta.success_step + trim_margin

Failure semantics: generate/render outcomes are recorded as batch/pass results;
required control-plane commands raise with their complete output. status.json
tracks the live stage, orchestration.json is the durable resume ledger, and
manifest.json indexes verified episodes.
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
from random import Random

import yaml

ROOT = Path(__file__).resolve().parents[1]      # data_engine/
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- config

class Stage(str, Enum):
    """Live pipeline stage, written verbatim into status.json (values are the
    monitoring contract — keep them stable)."""

    INIT = "init"
    NOMINAL = "nominal_probe"
    REPAIR = "repair"
    POST_PROBE = "post_probe"
    WIDE = "wide_probe"
    VECTORIZE = "session_vectorize"
    SESSION = "session"
    SESSION_VISUAL = "session_visual"
    FARM = "farm"
    NOISE_PLAN = "session_noise"
    COMPOUND = "compound"
    COMPOUND_DONE = "compound_done"
    MULTIPLY = "multiply"
    DONE = "DONE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    WALL_BUDGET = "WALL_BUDGET"


@dataclass(frozen=True)
class Config:
    """Every tunable, with its reason. CLI flags map 1:1 (see build_config)."""

    run_dir: Path
    name: str = "gen_auto"
    sessions: tuple[str, ...] = ("scene", "strategy", "phase")
    # None = use the authoring condition's budget_min. Repair/vectorize/noise
    # sessions have no condition file and use intervention_session_min.
    session_min: float | None = None
    intervention_session_min: float = 120.0
    agent_cmd: str = ""               # "" = no agent runtime: skip agent stages
    isaac_py: str = sys.executable
    wall_hours: float = 20.0

    # farm completion is PER CELL: every live (non-quarantined) cell must hold this
    # many verified successes. A global count is meaningless when one num_envs-wide
    # batch of the base cell can satisfy it while every authored cell stays unfarmed.
    per_cell_target: int = 50
    # General default; deployment launchers choose the production width explicitly.
    num_envs: int = 4
    # a cell that has attempted this many batches with ZERO successes leaves the
    # pool: min-successes scheduling alone pins the farm on a never-succeeding cell
    # (it always has the fewest successes, so it wins every pick)
    quarantine_zero_yield_batches: int = 3

    # The wide probe is a MECHANICAL gate: the batch must complete env-batched
    # (fail-fast catches scalar solves in seconds) and produce at least one
    # success (a wide-running solve whose envs interfere yields zero). Yield
    # beyond that is economics — low-yield cells just cost more farm batches —
    # so vectorize sessions are BOUNDED, not a fight to a quality bar.
    wide_probe_seed: int = 50
    nominal_seeds: tuple[int, ...] = (0, 1, 2)
    repair_rounds: int = 2
    vectorize_rounds: int = 4
    # A round retries a failed (non-ok) session launch at most this many times,
    # sleeping between tries. Unbounded instant retries once produced 29k
    # attempts in hours when a launcher bug made every session exit at import.
    session_attempts_per_round: int = 5
    session_retry_sleep_s: float = 60.0

    # compound: scripted physics diversification — fresh num_envs rollouts with new
    # draws + the agent-authored solve noise executed at compound_noise_scale.
    # The noise plan is accepted only if a probe clears noise_floor_yield AND shows
    # at least noise_min_coverage perturbed (step, env) rows: mandatory, measured.
    compound_eps: int = 450
    compound_hours: float = 10.0
    compound_noise_scale: float = 1.0
    noise_rounds: int = 3
    noise_floor_yield: float = 0.05
    noise_min_coverage: float = 0.02

    # multiply: visual passes per verified episode. Pass 0 = the scene's declared
    # cameras, nominal look; pass j >= 1 = one jittered camera named draw<j> + the
    # scene's VISUAL_PARAMS look j. Video-native (imgs/<view>.mp4 + render json).
    multiply_draws: int = 5
    multiply_hours: float = 18.0
    render_envs: int = 16             # episodes replayed in parallel per render pass
    render_chunk: int = 192           # ep dirs per render.py invocation (argv sanity)
    render_kit_args: str = ""         # deployment-specific Kit settings, if any
    # rows kept past each episode's sustained-success step in every render
    # (~3 s at 15 Hz): the recovery is data, the parked tail is not. -1 = no trim.
    trim_margin: int = 45
    cam_eye: tuple[float, float, float] = (1.05, 1.05, 0.85)
    cam_target: tuple[float, float, float] = (0.42, 0.0, 0.15)
    cam_eye_jitter: tuple[float, float, float] = (0.30, 0.30, 0.20)
    cam_target_jitter: tuple[float, float, float] = (0.08, 0.08, 0.06)

    # timeouts: a batch is one episode set; a session's last agent turn can overrun
    # its budget by a full tool-call timeout, hence the grace
    batch_timeout_s: float = 3 * 3600
    command_timeout_s: float = 4 * 3600
    session_grace_s: float = 3300
    farm_seed_base: int = 100
    compound_seed_base: int = 5000
    visual_seed_base: int = 1000

    def __post_init__(self) -> None:
        allowed = {"scene", "strategy", "phase"}
        unknown = set(self.sessions) - allowed
        if unknown:
            raise ValueError(f"unknown authoring sessions: {sorted(unknown)}")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        for name in ("per_cell_target", "compound_eps", "multiply_draws"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.quarantine_zero_yield_batches < 1:
            raise ValueError("quarantine_zero_yield_batches must be >= 1")
        if not 0.0 <= self.noise_floor_yield <= 1.0:
            raise ValueError("noise_floor_yield must be in [0, 1]")
        if not 0.0 <= self.noise_min_coverage <= 1.0:
            raise ValueError("noise_min_coverage must be in [0, 1]")
        if self.compound_eps and self.compound_noise_scale <= 0:
            raise ValueError("compound_noise_scale must be > 0 when compounding is "
                             "enabled (noise is mandatory in compound)")
        for name in ("repair_rounds", "vectorize_rounds", "noise_rounds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.multiply_draws and self.render_envs < 1:
            raise ValueError("render_envs must be >= 1 when multiply is enabled")


# --------------------------------------------------------------------- subprocess

def sh(cmd: list, *, timeout: float, log: Path | None = None,
       check: bool = True) -> subprocess.CompletedProcess:
    """Run a stage subprocess. Timeouts are survivable by design: the process is
    reaped, the log annotated, and a synthetic rc=-9 returned — a timed-out stage
    must never kill the campaign."""
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
        print(f"[orchestrate] stage TIMED OUT after {timeout:.0f}s — continuing", flush=True)
        return subprocess.CompletedProcess(cmd, -9, out, err)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(p.stdout + p.stderr)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed rc={p.returncode}: {cmd[:3]}\n"
            f"full stdout/stderr:\n{p.stdout}{p.stderr}"
        )
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


def read_json(path: Path, default: dict | None = None) -> dict:
    return json.loads(path.read_text()) if path.is_file() else (default or {})


def write_json_atomic(path: Path, value: dict) -> None:
    """Never expose a half-written status/ledger to monitors or resumed runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class Campaign:
    """The on-disk campaign: paths, cells, metas, status, manifest. All counts
    re-derive from the data pool so a resumed run never trusts stale state."""

    def __init__(self, run_dir: Path, name: str) -> None:
        self.run_dir = run_dir
        self.name = name

    @property
    def gen(self) -> Path:
        return self.run_dir / "data_gen" / self.name

    @property
    def logs(self) -> Path:
        return self.gen / "logs"

    def scene_py(self, scene: str) -> Path:
        return self.gen / "scenes" / scene / "scene" / "scene.py"

    def solve_py(self, cell: Cell) -> Path:
        return self.gen / "scenes" / cell.scene / "strategies" / cell.strategy / "solve.py"

    def grader_py(self, scene: str) -> Path:
        return self.gen / "scenes" / scene / "grader" / "grader.py"

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
        """Persistent budget start: restarts consume the original budget rather
        than silently receiving a fresh allowance."""
        cur = self.ledger()
        starts = dict(cur.get("budget_started_at", {}))
        if budget not in starts:
            starts[budget] = time.time()
            cur["budget_started_at"] = starts
            write_json_atomic(self.ledger_path, cur)
        return float(starts[budget]) + hours * 3600

    def session_outcome(self, tag: str) -> dict:
        return dict(self.ledger().get("sessions", {}).get(tag, {}))

    def session_attempts(self, prefix: str) -> list[tuple[str, dict]]:
        return sorted(
            (tag, dict(outcome))
            for tag, outcome in self.ledger().get("sessions", {}).items()
            if tag.startswith(prefix)
        )

    def record_session(self, tag: str, outcome: dict) -> None:
        cur = self.ledger()
        sessions = dict(cur.get("sessions", {}))
        sessions[tag] = outcome
        cur["sessions"] = sessions
        write_json_atomic(self.ledger_path, cur)

    def active_candidate(self) -> int:
        return int(self.ledger().get("active_candidate", 0))

    def activate_candidate(self, index: int, source: Path) -> None:
        """Replace only the campaign-local base strategy. The source eval run and
        shared repository remain immutable, and resume stays under one campaign
        name."""
        dst = self.gen / "scenes" / "scene_0" / "strategies" / "strategy_0"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(source, dst, ignore=shutil.ignore_patterns("__pycache__"))
        if not (dst / "meta.json").is_file():
            write_json_atomic(
                dst / "meta.json",
                {"episodes": 0, "successes": 0, "success_rate": None,
                 "batches": [], "refreshed": None},
            )
        self.update_ledger(active_candidate=index, active_candidate_name=source.name)

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
                    # a runnable phase cell has entry builders; ledgers are not cells
                    if ph.is_dir() and (ph / "reset").is_dir() and any((ph / "reset").glob("*.py")):
                        out.append(Cell(sc.name, st.name, ph.name))
        return out

    # ---- metas and counts

    def batch_meta(self, batch: str) -> dict:
        p = self.gen / "data" / batch / "meta.json"
        return read_json(p)

    def batch_metas(self, pattern: str = "batch_*") -> list[dict]:
        return [json.loads(m.read_text())
                for m in sorted(self.gen.glob(f"data/{pattern}/meta.json"))]

    def successes(self, pattern: str) -> int:
        return sum(m.get("successes", 0) for m in self.batch_metas(pattern))

    def farm_successes(self) -> int:
        """FARM batches only: session test batches must not satisfy the farm target."""
        return self.successes("batch_farm_*")

    def log_path(self, batch: str) -> Path:
        """Evidence goes into briefs BY PATH, never inline: the complete log stays
        on disk for the agent's shell (nothing truncated), and a brief can never
        blow the model's context window — a 512-env probe log once weighed in at
        over a million tokens and killed 433 consecutive pc_ram sessions."""
        return self.logs / f"{batch}.log"

    def log_size(self, batch: str) -> str:
        p = self.log_path(batch)
        if not p.is_file():
            return "missing — the batch died before Isaac wrote anything"
        n = p.stat().st_size
        return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"

    # ---- status + manifest

    def write_status(self, stage: Stage | str, **fields) -> None:
        p = self.gen / "status.json"
        ledger = self.ledger()
        status = {
            "schema_version": 1,
            "stage": str(getattr(stage, "value", stage)),
            **fields,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        started = ledger.get("budget_started_at", {}).get("pipeline")
        if started is not None:
            status["pipeline_started_at"] = datetime.fromtimestamp(
                float(started), tz=timezone.utc
            ).isoformat(timespec="seconds")
        write_json_atomic(p, status)

    def write_manifest(self) -> dict:
        """Index final dataset membership only.

        Nominal/wide probes and agent-session test batches remain on disk as
        diagnostic evidence but never count toward delivery.
        """
        eps, ok, total = [], 0, 0
        diagnostic_total = diagnostic_ok = 0
        for em in sorted(self.gen.glob("data/*/ep_*/meta.json")):
            meta = json.loads(em.read_text())
            dataset_member = em.parent.parent.name.startswith(
                ("batch_farm_", "batch_compound_")
            )
            if not dataset_member:
                diagnostic_total += 1
                diagnostic_ok += int(bool(meta.get("success")))
                continue
            total += 1
            if meta.get("success"):
                ok += 1
                eps.append({"path": str(em.parent.relative_to(self.gen)),
                            "cell": meta.get("cell"), "phase_entry": meta.get("entry"),
                            "reset": meta.get("reset"), "steps": meta.get("steps"),
                            "score": meta.get("score"), "parameters": meta.get("parameters")})
        manifest = {"successful_episodes": ok,
                    "total_episodes": total,
                    "episodes": eps,
                    "diagnostic_episodes": diagnostic_total,
                    "diagnostic_successful_episodes": diagnostic_ok,
                    "refreshed": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_json_atomic(self.gen / "manifest.json", manifest)
        return manifest

    def successful_episode_dirs(
        self, patterns: tuple[str, ...] = ("batch_farm_*", "batch_compound_*")
    ) -> dict[str, list[Path]]:
        """Verified dataset episodes grouped by scene.

        Probe, repair, vectorize, and authoring-session test batches are diagnostic
        evidence, not final dataset membership.
        """
        by_scene: dict[str, list[Path]] = {}
        for pattern in patterns:
            for m in sorted(self.gen.glob(f"data/{pattern}/meta.json")):
                bmeta = json.loads(m.read_text())
                if not bmeta.get("successes"):
                    continue
                scene = str(bmeta.get("cell", "scene_0/strategy_0")).split("/")[0]
                for em in sorted(m.parent.glob("ep_*/meta.json")):
                    if json.loads(em.read_text()).get("success"):
                        by_scene.setdefault(scene, []).append(em.parent)
        return by_scene


# ------------------------------------------------------------------- agent runner

@dataclass(frozen=True)
class SessionResult:
    tag: str
    returncode: int
    session_dir: Path
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def as_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "ok": self.ok,
            "session_dir": str(self.session_dir),
            "log_path": str(self.log_path),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


class AgentRunner:
    """Runs one autonomous agent session over an orchestrate-authored brief, with
    the campaign CLIs (create_cell/generate/catalog_assets) shimmed onto PATH.
    The agent command is bring-your-own: any executable accepting
    --prompt-file/--workdir/--cap-min/--transcript/--env."""

    TOOLS = ("create_cell", "generate", "catalog_assets")

    def __init__(self, cfg: Config, camp: Campaign) -> None:
        self.cfg, self.camp = cfg, camp

    @property
    def available(self) -> bool:
        return bool(self.cfg.agent_cmd)

    def _shims(self) -> Path:
        shims = self.camp.gen / ".shims"
        shims.mkdir(exist_ok=True)
        # Isaac subprocesses run with the ORCHESTRATOR's PYTHONPATH, never the
        # agent runtime's: an agent-side overlay once shadowed the venv's pinned
        # numpy with 2.x and crashed every render. The orchestrator's own value
        # is baked in verbatim (deployments may use it for a venv-safe extra
        # like a video backend); absent = explicitly unset.
        base_pp = os.environ.get("PYTHONPATH")
        pp = (f"PYTHONPATH={shlex.quote(base_pp)}" if base_pp else "-u PYTHONPATH")
        for tool in self.TOOLS:
            script = shims / tool
            script.write_text(f"#!/bin/bash\nexec env {pp} "
                              f"{shlex.quote(self.cfg.isaac_py)} "
                              f"{shlex.quote(str(ROOT / 'agent' / 'cli' / (tool + '.py')))} "
                              '"$@"\n')
            script.chmod(0o755)
        return shims

    def brief(self, name: str, **fields) -> str:
        return (ROOT / "agent" / "prompts" / f"{name}.md").read_text().format(**fields)

    def run(self, tag: str, instructions_text: str | Path,
            cap_min: float | None = None) -> SessionResult:
        if isinstance(instructions_text, Path):
            instructions = instructions_text
            session = instructions.parent
        else:
            session = self.camp.gen / ".agent" / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
            session.mkdir(parents=True, exist_ok=True)
            instructions = session / "instructions.md"
            instructions.write_text(instructions_text)
        cap = (cap_min if cap_min is not None else
               (self.cfg.session_min or self.cfg.intervention_session_min))
        log_path = session / "agent.log"
        p = sh([*shlex.split(self.cfg.agent_cmd),
            "--prompt-file", instructions, "--workdir", self.camp.gen,
            "--cap-min", str(cap),
            "--transcript", session / "transcript.jsonl",
            "--env", f"PATH={self._shims()}:{os.environ.get('PATH', '')}",
            "--env", f"DGEN_ROOT={self.camp.gen}", "--env", f"DGEN_LEVEL={tag}",
            "--env", "DGEN_SCENE=scene_0", "--env", "DGEN_STRATEGY=strategy_0",
            "--env", f"DGEN_NUM_ENVS={self.cfg.num_envs}",
            "--env", f"ISAAC_PY={self.cfg.isaac_py}"],
           log=log_path,
           timeout=cap * 60 + self.cfg.session_grace_s, check=False)
        result = SessionResult(tag, p.returncode, session, log_path)
        self.camp.record_session(tag, result.as_dict())
        return result


# ------------------------------------------------------------------- orchestrator

BASE_CELL = Cell("scene_0", "strategy_0")


class Orchestrator:
    """The stage machine. Stages run in order; each is a method that reads its
    preconditions from disk (idempotent resume) and records outcomes through the
    campaign's status/manifest."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.camp = Campaign(cfg.run_dir, cfg.name)
        self.agent = AgentRunner(cfg, self.camp)
        self.deadline: float | None = None
        self.preset: str = json.loads((cfg.run_dir / "run.json").read_text())["preset"]
        self.farm_envs = max(1, cfg.num_envs)

    # ---- shared plumbing

    def budget_left(self) -> bool:
        return self.deadline is None or time.time() < self.deadline

    def _probe_fingerprint(self) -> str:
        """Identity of the code a base-cell probe grades: solve + scene + grader
        bytes. Probe results are only reusable while all three are unchanged."""
        h = hashlib.sha256()
        for p in (self.camp.solve_py(BASE_CELL), self.camp.scene_py("scene_0"),
                  self.camp.grader_py("scene_0")):
            h.update(p.read_bytes() if p.is_file() else b"<missing>")
        return h.hexdigest()[:12]

    def run_batch(self, cell: Cell, batch: str, extra: list[str]) -> dict:
        """One generate invocation. NO rendering here (generation is physics-only;
        verified episodes are rendered by MULTIPLY or offline render waves).
        Failures are data: rc is ignored, the batch meta (or its absence) speaks."""
        cmd = [self.cfg.isaac_py, ROOT / "scripts" / "generate.py", self.camp.gen,
               "--headless", "--batch", batch,
               "--scene", cell.scene, "--strategy", cell.strategy, *extra]
        if cell.phase:
            cmd += ["--phase", cell.phase]
        sh(cmd, log=self.camp.logs / f"{batch}.log",
           timeout=self.cfg.batch_timeout_s, check=False)
        return self.camp.batch_meta(batch)

    # ---- stages

    def ensure_init(self) -> None:
        if not (self.camp.gen / "gen.yaml").is_file():
            from engine.initialization import init
            init(self.cfg.run_dir, name=self.cfg.name)
        self.deadline = self.camp.deadline("pipeline", self.cfg.wall_hours)
        self.camp.write_status(Stage.INIT, preset=self.preset)

    def _run_or_read_batch(self, cell: Cell, base_name: str, extra: list[str]) -> tuple[str, dict]:
        """Append-only batch execution with crash-safe retries.

        A completed batch is immutable and reused on resume. A directory without
        meta.json is an interrupted attempt, so a suffixed retry is launched instead
        of colliding with generation.py's append-only guard.
        """
        data = self.camp.gen / "data"
        matches = ([data / base_name] if (data / base_name).is_dir() else [])
        matches += sorted(data.glob(f"{base_name}_retry*"))
        for path in reversed(matches):
            meta = self.camp.batch_meta(path.name)
            if meta:
                return path.name, meta
        name = base_name if not matches else f"{base_name}_retry{len(matches):02d}"
        return name, self.run_batch(cell, name, extra)

    def _probe_nominal(self, tag: str) -> tuple[bool, str]:
        last_batch = ""
        for seed in self.cfg.nominal_seeds:
            base = (f"batch_nominal_{tag}_s{seed}" if tag else
                    ("batch_nominal" if seed == 0 else f"batch_nominal_s{seed}"))
            name, meta = self._run_or_read_batch(
                BASE_CELL, base,
                ["--nominal", "--num_envs", "1", "--seed", str(seed)],
            )
            last_batch = name
            if meta.get("successes"):
                return True, name
            print(f"[orchestrate] nominal probe ({tag or 'primary'}) seed={seed} failed "
                  f"({meta.get('successes', 'no meta')} successes)", flush=True)
        return False, last_batch

    def ensure_nominal(self) -> None:
        """NOT a gate: a failing nominal escalates — candidates, then repair sessions —
        and the campaign proceeds regardless (level sessions can still author working
        cells where the delivered solve does not)."""
        self.camp.write_status(Stage.NOMINAL)
        active = self.camp.active_candidate()
        tag = "" if active == 0 else f"candidate_{active:03d}"
        nominal_ok, last_batch = self._probe_nominal(tag)

        candidates = sorted((self.cfg.run_dir / "workspace" / "candidates").glob("*/solve.py"))
        tried = active
        while not nominal_ok and tried < len(candidates) and self.budget_left():
            cand = candidates[tried].parent
            tried += 1
            print(f"[orchestrate] nominal failing — rotating to candidate {cand.name} "
                  f"({tried}/{len(candidates)})", flush=True)
            self.camp.activate_candidate(tried, cand)
            self.camp.write_status(
                Stage.NOMINAL, candidate_index=tried, candidate_name=cand.name
            )
            nominal_ok, last_batch = self._probe_nominal(f"candidate_{tried:03d}")

        for rnd in range(1, self.cfg.repair_rounds + 1):
            if nominal_ok or not self.agent.available or not self.budget_left():
                break
            prefix = f"repair_{rnd:03d}_attempt_"
            successful = [
                outcome for _, outcome in self.camp.session_attempts(prefix)
                if outcome.get("ok")
            ]
            while (not successful and self.budget_left()
                   and len(self.camp.session_attempts(prefix))
                   < self.cfg.session_attempts_per_round):
                attempt = len(self.camp.session_attempts(prefix)) + 1
                tag = f"{prefix}{attempt:03d}"
                self.camp.write_status(Stage.REPAIR, round=rnd, attempt=attempt)
                print(f"[orchestrate] nominal failing — repair round {rnd}/"
                      f"{self.cfg.repair_rounds}, attempt {attempt}", flush=True)
                result = self.agent.run(tag, self.agent.brief(
                    "repair", gen=self.camp.gen,
                    solve=self.camp.solve_py(BASE_CELL),
                    grader=self.camp.grader_py("scene_0"),
                    fail_log_path=self.camp.log_path(last_batch),
                    fail_log_size=self.camp.log_size(last_batch),
                    nominal_seeds=", ".join(map(str, self.cfg.nominal_seeds)),
                    num_envs=self.farm_envs))
                if result.ok:
                    successful = [result.as_dict()]
                else:
                    print(f"[orchestrate] repair round {rnd} attempt {attempt} "
                          f"exited {result.returncode}; retrying after "
                          f"{self.cfg.session_retry_sleep_s:.0f}s", flush=True)
                    time.sleep(self.cfg.session_retry_sleep_s)
            if not successful:
                break
            session_name = Path(successful[-1]["session_dir"]).name
            probe_tag = f"repair_{rnd:03d}_{session_name}"
            nominal_ok, last_batch = self._probe_nominal(probe_tag)

        self.camp.write_status(Stage.POST_PROBE, nominal_ok=nominal_ok)
        verdict = ("PASSES" if nominal_ok else
                   "still failing — proceeding anyway (level sessions may author "
                   "working cells)")
        print(f"[orchestrate] nominal {verdict}", flush=True)

    def ensure_wide(self) -> None:
        """The scripted stages (FARM/COMPOUND) always LAUNCH num_envs-wide — width
        is enforced by construction, and silently narrowing it is banned (user
        directive). This stage checks the MECHANICAL property those launches need:
        the solve completes an env-batched wide batch with at least one success.
        Yield beyond that is economics (a low-yield cell just costs more farm
        batches), so vectorize sessions are BOUNDED by vectorize_rounds and the
        pipeline proceeds afterward either way: a still-scalar solve crashes its
        wide batches in seconds (fail-fast) and quarantines, while authoring
        sessions can still ship vectorized cells."""
        if self.farm_envs <= 1:
            return

        def wide_ok() -> tuple[bool, str]:
            """Probes are named by the fingerprint of the code they grade, so
            reuse-on-resume is exact by construction: byte-identical code reuses
            its recorded result; any edit (repair, vectorize, a session touching
            the base cell) gets a fresh probe. A stale pass can no longer vouch
            for a solve it never ran."""
            name, meta = self._run_or_read_batch(
                BASE_CELL, f"batch_wide_probe_{self._probe_fingerprint()}",
                ["--num_envs", str(self.farm_envs),
                 "--seed", str(self.cfg.wide_probe_seed)],
            )
            got = meta.get("successes") or 0
            print(f"[orchestrate] wide probe {name}: "
                  f"{'no meta (crashed)' if not meta else got}/{self.farm_envs} "
                  "(mechanical gate: completes env-batched with >= 1 success)",
                  flush=True)
            return bool(meta) and got >= 1, name

        self.camp.write_status(Stage.WIDE)
        wide, last_failed = wide_ok()
        rounds = [
            int(m.group(1))
            for tag, _ in self.camp.session_attempts("vectorize_")
            if (m := re.match(r"vectorize_(\d{3})_attempt_", tag))
        ]
        rnd = max(rounds, default=0)
        if rnd and not any(
            outcome.get("ok")
            for _, outcome in self.camp.session_attempts(
                f"vectorize_{rnd:03d}_attempt_"
            )
        ):
            rnd -= 1  # retry the incomplete round; transient exits do not consume it
        while (not wide and self.agent.available and self.budget_left()
               and rnd < self.cfg.vectorize_rounds):
            rnd += 1
            prefix = f"vectorize_{rnd:03d}_attempt_"
            successful = [
                outcome for _, outcome in self.camp.session_attempts(prefix)
                if outcome.get("ok")
            ]
            while (not successful and self.budget_left()
                   and len(self.camp.session_attempts(prefix))
                   < self.cfg.session_attempts_per_round):
                attempt = len(self.camp.session_attempts(prefix)) + 1
                tag = f"{prefix}{attempt:03d}"
                self.camp.write_status(Stage.VECTORIZE, round=rnd, attempt=attempt)
                print(f"[orchestrate] wide probe failing — vectorize round {rnd}/"
                      f"{self.cfg.vectorize_rounds}, attempt {attempt}", flush=True)
                result = self.agent.run(tag, self.agent.brief(
                    "vectorize", gen=self.camp.gen, num_envs=self.farm_envs,
                    solve=self.camp.solve_py(BASE_CELL),
                    fail_log_path=self.camp.log_path(last_failed),
                    fail_log_size=self.camp.log_size(last_failed),
                    probe_seed=self.cfg.wide_probe_seed))
                if result.ok:
                    successful = [result.as_dict()]
                else:
                    print(f"[orchestrate] vectorize round {rnd} attempt {attempt} "
                          f"exited {result.returncode}; retrying after "
                          f"{self.cfg.session_retry_sleep_s:.0f}s", flush=True)
                    time.sleep(self.cfg.session_retry_sleep_s)
            if not successful:
                break
            wide, last_failed = wide_ok()
        self.camp.write_status(Stage.WIDE, wide_ok=wide, vectorize_rounds=rnd)
        if wide:
            print(f"[orchestrate] wide probe OK — {self.farm_envs}-env batches enabled",
                  flush=True)
        else:
            print(f"[orchestrate] wide probe still failing after {rnd} vectorize "
                  "round(s) — PROCEEDING: batches keep launching wide (never "
                  "narrowed); a scalar base cell will crash-quarantine while "
                  "authored cells can still farm", flush=True)

    def run_sessions(self) -> None:
        """One authoring session per configured level, via the framework's official
        condition briefs (diversify.py --host --dry)."""
        if not self.agent.available:
            return
        for level in self.cfg.sessions:
            marker = self.camp.gen / f".session_{level}.json"
            if read_json(marker).get("ok") or not self.budget_left():
                continue
            self.camp.write_status(Stage.SESSION, level=level)
            dry = sh([sys.executable, ROOT / "scripts" / "diversify.py", self.camp.gen,
                      ROOT / "configs" / f"{level}_default.yaml", "--host", "--dry"],
                     timeout=self.cfg.command_timeout_s, check=True)
            match = re.search(r"^instructions: (.+)$", dry.stdout, re.M)
            if match is None:
                raise RuntimeError(
                    "diversify --dry did not report an instructions path; "
                    f"full stdout/stderr:\n{dry.stdout}{dry.stderr}"
                )
            instructions = Path(match.group(1))
            condition = yaml.safe_load((instructions.parent / "condition.yaml").read_text())
            cap = self.cfg.session_min
            if cap is None:
                cap = float(condition.get("budget_min", self.cfg.intervention_session_min))
            before = {c.key for c in self.camp.cells()}
            result = self.agent.run(f"author_{level}", instructions, cap_min=cap)
            after = {c.key for c in self.camp.cells()}
            outcome = {
                **result.as_dict(),
                "level": level,
                "cells_before": len(before),
                "cells_after": len(after),
                "new_cells": sorted(after - before),
            }
            if result.ok:
                write_json_atomic(marker, outcome)
                print(f"[orchestrate] session {level} finished "
                      f"(cells now: {len(after)})", flush=True)
            else:
                print(f"[orchestrate] session {level} exited {result.returncode}; "
                      "completion marker not written", flush=True)

    def ensure_visual_params(self) -> None:
        """VISUAL_PARAMS is a REQUIRED scene deliverable — it is multiply's whole
        look axis (the v18 wave shipped 13 tasks where every draw pass was camera
        jitter only, because no scene declared any). The scene brief demands it;
        scenes still lacking it afterward get one targeted follow-up session,
        verified by re-reading the scene files, not by the session's say-so."""
        if not self.agent.available:
            return
        marker = self.camp.gen / ".session_visual.json"
        prev = read_json(marker)
        if prev.get("ok") or not self.budget_left():
            return
        scenes = sorted({c.scene for c in self.camp.cells()})
        # targets: scenes without the declaration, PLUS scenes a previous attempt
        # edited but broke (declaration present, build dead) — a rejected attempt
        # must re-enter the session, never silently pass the presence check
        missing = [s for s in scenes
                   if "VISUAL_PARAMS" not in self.camp.scene_py(s).read_text()]
        targets = sorted(set(missing) | set(prev.get("broken_scenes", [])))
        if not targets:
            write_json_atomic(marker, {"ok": True, "missing_before": []})
            return
        self.camp.write_status(Stage.SESSION_VISUAL, scenes_missing=targets)
        print(f"[orchestrate] VISUAL_PARAMS session needed for {targets} — targeted "
              "session", flush=True)
        result = self.agent.run("visual_params", self.agent.brief(
            "visual_params", gen=self.camp.gen,
            scenes=", ".join(targets),
            scene_paths="\n".join(str(self.camp.scene_py(s)) for s in targets),
            num_envs=self.farm_envs))
        still = [s for s in scenes
                 if "VISUAL_PARAMS" not in self.camp.scene_py(s).read_text()]
        # BEHAVIORAL verification, not text presence: an edited scene must still
        # BUILD AND RUN. The pen_holder v21 visual session declared the band but
        # referenced an undeclared cfg attribute in assets() — text check passed,
        # every scene_0 batch after it crashed at build. A 1-env probe per edited
        # scene must produce a batch meta (task success NOT required: nominal-
        # failing tasks still build); no meta = the edit broke the scene.
        broken = []
        for s in targets:
            if s in still:
                continue
            cell = next((c for c in self.camp.cells() if c.scene == s), None)
            if cell is None:
                continue
            h = hashlib.sha256(self.camp.scene_py(s).read_bytes()).hexdigest()[:12]
            _, meta = self._run_or_read_batch(
                cell, f"batch_visual_check_{s}_{h}",
                ["--nominal", "--num_envs", "1", "--seed", "0"])
            if not meta:
                broken.append(s)
                print(f"[orchestrate] visual edit BROKE {s}: the post-edit build "
                      "probe produced no meta — session rejected", flush=True)
        outcome = {**result.as_dict(),
                   "ok": result.ok and not still and not broken,
                   "missing_before": targets, "missing_after": still,
                   "broken_scenes": broken}
        write_json_atomic(marker, outcome)
        if outcome["ok"]:
            print("[orchestrate] VISUAL_PARAMS present and verified in every scene",
                  flush=True)
        else:
            print(f"[orchestrate] visual session rejected (missing: {still}, "
                  f"broken: {broken}) — will retry on resume; their multiply draw "
                  "passes vary camera pose only until fixed", flush=True)

    def farm(self) -> bool:
        """Coverage-balanced batches until every live cell holds per_cell_target.

        Quarantine keeps a never-succeeding cell from capturing the
        min-successes scheduler; least-recently-farmed breaks ties.
        """
        # key -> [completed_batches, successes, last_farmed_index]. The third
        # field makes ties genuinely round-robin instead of lexicographically
        # favoring the first cell forever.
        stats: dict[str, list[int]] = {}
        for idx, m in enumerate(self.camp.batch_metas("batch_farm_*"), start=1):
            s = stats.setdefault(str(m.get("cell")), [0, 0, -1])
            s[0] += 1
            s[1] += m.get("successes", 0)
            s[2] = idx

        def stat(cell: Cell) -> list[int]:
            return stats.setdefault(cell.key, [0, 0, -1])

        def live_cells(pool: list[Cell]) -> list[Cell]:
            return [c for c in pool
                    if not (stat(c)[0] >= self.cfg.quarantine_zero_yield_batches
                            and stat(c)[1] == 0)]

        def unmet(pool: list[Cell]) -> list[Cell]:
            """Completion is PER CELL: every live cell must reach per_cell_target.
            A global count is one wide base-cell batch away from 'done', which
            would ship none of the authored diversity (the v18 wave did exactly
            that: 13 tasks, one farm batch each, zero authored-cell episodes)."""
            return [c for c in live_cells(pool)
                    if stat(c)[1] < self.cfg.per_cell_target]

        self.camp.write_status(Stage.FARM, cells=len(self.camp.cells()),
                               farm_envs=self.farm_envs)
        idx = len(list(self.camp.gen.glob("data/batch_farm_*")))
        while self.budget_left():
            manifest = self.camp.write_manifest()
            pool = self.camp.cells()
            todo = unmet(pool)
            self.camp.write_status(
                Stage.FARM, successful=manifest["successful_episodes"],
                farm_successful=self.camp.farm_successes(),
                total=manifest["total_episodes"], batches=idx,
                farm_envs=self.farm_envs,
                per_cell_target=self.cfg.per_cell_target,
                cells_done=len(live_cells(pool)) - len(todo),
                cells_live=len(live_cells(pool)),
                quarantined=sorted(c.key for c in pool
                                   if not any(c is l for l in live_cells(pool))),
                per_cell={c.key: stat(c)[1] for c in pool},
            )
            if not live_cells(pool):
                print("[orchestrate] all cells quarantined (0 successes after "
                      f">={self.cfg.quarantine_zero_yield_batches} batches each) "
                      "— closing the farm", flush=True)
                break
            if not todo:
                break
            cell = min(todo, key=lambda c: (stat(c)[1], stat(c)[2], c.key))
            idx += 1
            meta = self.run_batch(
                cell, f"batch_farm_{idx:04d}",
                ["--num_envs", str(self.farm_envs),
                 "--seed", str(self.cfg.farm_seed_base + idx),
                 "--env_draw", str(idx * max(1, self.farm_envs - 1)),
                 "--solve_draw", str(idx)])
            s = stat(cell)
            s[0] += 1  # a crashed batch still counts toward zero-yield quarantine
            s[1] += meta.get("successes", 0)
            s[2] = idx

        manifest = self.camp.write_manifest()
        pool = self.camp.cells()
        done = bool(live_cells(pool)) and not unmet(pool)
        self.camp.write_status(Stage.FARM,
                               farm_complete=done,
                               successful=manifest["successful_episodes"],
                               farm_successful=self.camp.farm_successes(),
                               total=manifest["total_episodes"],
                               per_cell={c.key: stat(c)[1] for c in pool})
        print(f"[orchestrate] farm {'complete' if done else 'incomplete'}: "
              f"{self.camp.farm_successes()} farmed across {len(pool)} cells "
              f"(target {self.cfg.per_cell_target}/cell) -> "
              f"{self.camp.gen / 'manifest.json'}", flush=True)
        return done

    def ensure_noise_plan(self) -> bool:
        """The compound stage's mandatory noise, authored by an agent that has
        WATCHED the task: the session receives a rendered verified episode
        (the view tool attaches its frames), reads the solve's phase structure,
        and writes per-phase perturbations into the solve through the
        env.step(action, noise=...) channel — labels stay clean structurally
        (see engine.generation.Recorder). Acceptance is measured, never
        claimed: a num_envs probe at compound_noise_scale must clear
        noise_floor_yield AND noise_min_coverage. Bounded by noise_rounds."""
        if self.cfg.compound_eps <= 0:
            return False
        if self.camp.ledger().get("noise_plan", {}).get("ok"):
            return True
        if not self.agent.available:
            print("[orchestrate] no agent runtime — noise plan (mandatory for "
                  "compound) cannot be authored", flush=True)
            return False
        by_scene = self.camp.successful_episode_dirs(("batch_farm_*",))
        base_eps = by_scene.get("scene_0") or next(iter(by_scene.values()), [])
        if not base_eps:
            print("[orchestrate] no verified farm episodes — nothing to author "
                  "noise against", flush=True)
            return False
        need = max(1, round(self.cfg.noise_floor_yield * self.farm_envs))

        def probe() -> tuple[bool, str, dict]:
            name, meta = self._run_or_read_batch(
                BASE_CELL, f"batch_noise_probe_{self._probe_fingerprint()}",
                ["--num_envs", str(self.farm_envs),
                 "--seed", str(self.cfg.wide_probe_seed),
                 "--noise_scale", str(self.cfg.compound_noise_scale)],
            )
            got = meta.get("successes") or 0
            cov = float((meta.get("noise") or {}).get("perturbed_row_frac") or 0.0)
            ok = bool(meta) and got >= need and cov >= self.cfg.noise_min_coverage
            print(f"[orchestrate] noise probe {name}: "
                  f"{'CRASHED (no meta)' if not meta else f'yield {got}/{self.farm_envs}'} "
                  f"(need >= {need}), measured coverage {cov:.3f} (need >= "
                  f"{self.cfg.noise_min_coverage}) -> "
                  f"{'ACCEPTED' if ok else 'rejected'}", flush=True)
            return ok, name, meta

        def sample_video() -> Path | None:
            """The episode the agent watches. Rendered once, reused on retries."""
            ep = base_eps[0]
            vids = sorted(ep.glob("imgs/*.mp4"))
            if vids:
                return vids[0]
            cmd = [self.cfg.isaac_py, ROOT / "scripts" / "render.py", self.camp.gen,
                   "--episodes", ep, "--num_envs", "4", "--no-sheet", "--headless"]
            if "CAMERAS" not in self.camp.scene_py(BASE_CELL.scene).read_text():
                cmd += ["--eye", *map(str, self.cfg.cam_eye),
                        "--target", *map(str, self.cfg.cam_target)]
            if self.cfg.render_kit_args:
                cmd += [f"--kit_args={self.cfg.render_kit_args}"]
            sh(cmd, log=self.camp.logs / "noise_sample_render.log",
               timeout=self.cfg.batch_timeout_s, check=False)
            vids = sorted(ep.glob("imgs/*.mp4"))
            return vids[0] if vids else None

        # a passing probe may already exist for the current code (resume path)
        accepted, batch_name, meta = probe()
        rnd = max((int(m.group(1)) for tag, _ in self.camp.session_attempts("noise_")
                   if (m := re.match(r"noise_(\d{3})_attempt_", tag))), default=0)
        while (not accepted and self.budget_left()
               and rnd < self.cfg.noise_rounds):
            rnd += 1
            prefix = f"noise_{rnd:03d}_attempt_"
            successful = [o for _, o in self.camp.session_attempts(prefix)
                          if o.get("ok")]
            while (not successful and self.budget_left()
                   and len(self.camp.session_attempts(prefix))
                   < self.cfg.session_attempts_per_round):
                attempt = len(self.camp.session_attempts(prefix)) + 1
                tag = f"{prefix}{attempt:03d}"
                self.camp.write_status(Stage.NOISE_PLAN, round=rnd, attempt=attempt)
                video = sample_video()
                print(f"[orchestrate] noise plan round {rnd}/{self.cfg.noise_rounds}, "
                      f"attempt {attempt} (video: {video})", flush=True)
                result = self.agent.run(tag, self.agent.brief(
                    "noise", gen=self.camp.gen,
                    solve=self.camp.solve_py(BASE_CELL),
                    video=(video or "NOT RENDERED — render it yourself first (see "
                                    "the render command below)"),
                    sample_ep=base_eps[0],
                    num_envs=self.farm_envs,
                    noise_scale=self.cfg.compound_noise_scale,
                    need_successes=need,
                    min_coverage=self.cfg.noise_min_coverage,
                    probe_seed=self.cfg.wide_probe_seed,
                    fail_log_path=self.camp.log_path(batch_name),
                    fail_log_size=self.camp.log_size(batch_name)))
                if result.ok:
                    successful = [result.as_dict()]
                else:
                    print(f"[orchestrate] noise round {rnd} attempt {attempt} exited "
                          f"{result.returncode}; retrying after "
                          f"{self.cfg.session_retry_sleep_s:.0f}s", flush=True)
                    time.sleep(self.cfg.session_retry_sleep_s)
            if not successful:
                break
            accepted, batch_name, meta = probe()
        self.camp.update_ledger(noise_plan={
            "ok": accepted, "batch": batch_name, "rounds": rnd,
            "yield": meta.get("successes"), "noise": meta.get("noise"),
        })
        if not accepted:
            print(f"[orchestrate] NO accepted noise plan after {rnd} round(s) — "
                  "compound (mandatory noise) will be skipped; report to the user",
                  flush=True)
        return accepted

    def compound(self, noise_ok: bool) -> bool:
        """Scripted physics diversification: per batch, re-run a fertile cell's
        solve across num_envs envs — each env its own physics draw, the accepted
        agent-authored noise executing at compound_noise_scale — and keep what the
        grader passes. Coverage stays even by always extending the cell with the
        fewest compound successes."""
        if self.cfg.compound_eps <= 0:
            return True
        if not noise_ok:
            self.camp.write_status(
                Stage.COMPOUND_DONE,
                compound_successful=self.camp.successes("batch_compound_*"),
                compound_target=self.cfg.compound_eps,
                compound_skipped_no_noise_plan=True)
            print("[orchestrate] compound SKIPPED: noise is mandatory here and no "
                  "noise plan was accepted", flush=True)
            return False
        cmp_deadline = self.camp.deadline("compound", self.cfg.compound_hours)

        def compound_done() -> int:
            return self.camp.successes("batch_compound_*")

        fertile = [c for c in self.camp.cells()
                   if any(m.get("cell") == c.key and m.get("successes")
                          for m in self.camp.batch_metas("batch_farm_*"))]
        idx = len(list(self.camp.gen.glob("data/batch_compound_*")))
        assert self.deadline is not None
        while (fertile and time.time() < min(cmp_deadline, self.deadline)
               and compound_done() < self.cfg.compound_eps):
            got = {c.key: 0 for c in fertile}
            for m in self.camp.batch_metas("batch_compound_*"):
                if str(m.get("cell")) in got:
                    got[str(m.get("cell"))] += m.get("successes", 0)
            cell = min(fertile, key=lambda c: got[c.key])
            idx += 1
            self.camp.write_status(Stage.COMPOUND, compound_successful=compound_done(),
                                   compound_target=self.cfg.compound_eps, batches=idx)
            self.run_batch(cell, f"batch_compound_{idx:04d}",
                           ["--num_envs", str(self.farm_envs),
                            "--seed", str(self.cfg.compound_seed_base + idx),
                            "--env_draw", str(idx * max(1, self.farm_envs - 1)),
                            "--solve_draw", str(idx),
                            "--noise_scale", str(self.cfg.compound_noise_scale)])
        n = compound_done()
        self.camp.write_status(Stage.COMPOUND_DONE, compound_successful=n,
                               compound_target=self.cfg.compound_eps)
        print(f"[orchestrate] compound stage: {n}/{self.cfg.compound_eps} verified "
              "noise/DR variants", flush=True)
        return n >= self.cfg.compound_eps

    def multiply(self) -> bool:
        """Visual diversification (replay-rendered, PARALLEL: render_envs episodes per
        tiled pass). SUCCESSFUL episodes only — whole-batch rendering once burned days
        filming failures. Pass j=0 is the nominal view and runs last (it owns imgs/);
        j>=1 stash to imgs_draw<j>/. No re-testing: the replayed states ARE the
        verified ones."""
        if self.cfg.multiply_draws <= 0:
            return True
        by_scene = self.camp.successful_episode_dirs()
        n_eps = sum(map(len, by_scene.values()))
        done = read_json(self.camp.gen / ".multiply_done")
        if done.get("ok") and done.get("episodes") == n_eps:
            return True
        mul_deadline = self.camp.deadline("multiply", self.cfg.multiply_hours)
        self.camp.write_status(Stage.MULTIPLY, scenes=len(by_scene),
                               episodes=n_eps)
        marker_dir = self.camp.gen / ".multiply"
        marker_dir.mkdir(exist_ok=True)

        def pass_views(ep: Path, j: int) -> list[str]:
            """Views this pass owns in the render contract (imgs/render_<view>.json):
            pass 0 = every declared/nominal view (anything not named draw<k>);
            pass j = its own draw<j> camera."""
            views = [p.stem.removeprefix("render_")
                     for p in (ep / "imgs").glob("render_*.json")]
            if j:
                return [v for v in views if v == f"draw{j}"]
            return [v for v in views if not re.fullmatch(r"draw\d+", v)]

        missing_visual: list[str] = []
        for scene, eps in by_scene.items():
            if time.time() > mul_deadline:
                print("[orchestrate] multiply budget exhausted", flush=True)
                return False
            has_visual = "VISUAL_PARAMS" in self.camp.scene_py(scene).read_text()
            if not has_visual:
                missing_visual.append(scene)
            # Markers bind to the exact episode set they rendered: if a resumed
            # run grew the verified pool (farm/compound continued after a crash),
            # a stale "ok" must not leave the new episodes without this pass.
            eps_rel = [str(e.relative_to(self.camp.gen)) for e in eps]
            for j in range(self.cfg.multiply_draws):
                pass_marker = marker_dir / f"{scene}_pass_{j}.json"
                marker = read_json(pass_marker)
                if marker.get("ok") and marker.get("episode_set") == eps_rel:
                    continue
                if time.time() > mul_deadline:
                    print("[orchestrate] multiply budget exhausted", flush=True)
                    return False
                # A retried pass starts from a clean slate for ITS OWN views only
                # (other passes' videos and contracts are immutable): stale
                # render_<view>.json must not vouch for a re-render that died.
                for ep in eps:
                    for view in pass_views(ep, j):
                        (ep / "imgs" / f"render_{view}.json").unlink(missing_ok=True)
                        (ep / "imgs" / f"{view}.mp4").unlink(missing_ok=True)
                rng = Random(self.cfg.visual_seed_base + j)
                jit = (lambda base, band: [b + rng.uniform(-w, w) for b, w in zip(base, band)])
                pass_ok = True
                for lo in range(0, len(eps), self.cfg.render_chunk):
                    remaining = mul_deadline - time.time()
                    if remaining <= 0:
                        print("[orchestrate] multiply budget exhausted", flush=True)
                        return False
                    chunk = eps[lo:lo + self.cfg.render_chunk]
                    cmd = [self.cfg.isaac_py, ROOT / "scripts" / "render.py", self.camp.gen,
                           "--episodes", *chunk,
                           "--num_envs", str(self.cfg.render_envs), "--no-sheet",
                           "--trim-margin", str(self.cfg.trim_margin)]
                    if j:
                        # one jittered camera per draw, named draw<j> — its videos
                        # and contracts coexist with the nominal views, and the
                        # LeRobot bake selects it with --cams draw<j>
                        cmd += ["--eye", *map(str, jit(self.cfg.cam_eye,
                                                       self.cfg.cam_eye_jitter)),
                                "--target", *map(str, jit(self.cfg.cam_target,
                                                          self.cfg.cam_target_jitter)),
                                "--cam", f"draw{j}",
                                "--cams", f"draw{j}"]  # this pass renders ONLY its own view
                        if has_visual:
                            cmd += ["--visual_draw", str(j)]
                    elif "CAMERAS" not in self.camp.scene_py(scene).read_text():
                        # nominal pass with no declared cameras: the configured
                        # default view, under its standard name
                        cmd += ["--eye", *map(str, self.cfg.cam_eye),
                                "--target", *map(str, self.cfg.cam_target)]
                    if self.cfg.render_kit_args:
                        cmd += [f"--kit_args={self.cfg.render_kit_args}"]
                    p = sh(cmd, log=self.camp.logs / f"multiply_{scene}_d{j}_{lo}.log",
                           timeout=remaining, check=False)
                    pass_ok &= p.returncode == 0
                n_videos = sum(len(pass_views(e, j)) for e in eps)
                missing = [str(e.relative_to(self.camp.gen))
                           for e in eps if not pass_views(e, j)]
                # a render that produced nothing must be LOUD (a whole multiply pass
                # once silently no-opped on render-incapable GPUs)
                print(f"[orchestrate] multiply {scene} pass={j} "
                      f"(visual_draw={'yes' if has_visual and j else 'no'}): "
                      f"{n_videos} videos "
                      f"{'' if n_videos else '— RENDER PRODUCED NOTHING, check the log'}",
                      flush=True)
                if missing:
                    print("[orchestrate] multiply missing videos for episodes:\n"
                          + "\n".join(missing), flush=True)
                    pass_ok = False
                if not pass_ok:
                    self.camp.write_status(
                        Stage.MULTIPLY, multiply_ok=False, scene=scene, visual_pass=j,
                        missing_episodes=missing, scenes_without_visual_params=missing_visual,
                    )
                    return False
                write_json_atomic(
                    pass_marker,
                    {"ok": True, "scene": scene, "visual_pass": j,
                     "episodes": len(eps), "episode_set": eps_rel,
                     "videos": n_videos,
                     "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                )
        if missing_visual:
            print("[orchestrate] WARNING: no VISUAL_PARAMS in: "
                  + ", ".join(missing_visual)
                  + " — their draw passes vary camera pose only", flush=True)
        write_json_atomic(
            self.camp.gen / ".multiply_done",
            {"ok": True, "episodes": n_eps,
             "scenes_without_visual_params": missing_visual,
             "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        )
        print("[orchestrate] multiply pass complete", flush=True)
        return True

    # ---- the pipeline

    def run(self) -> None:
        try:
            self.ensure_init()
            self.ensure_nominal()
            self.ensure_wide()
            self.run_sessions()
            self.ensure_visual_params()
            farm_done = self.farm()
            noise_ok = self.ensure_noise_plan()
            compound_done = self.compound(noise_ok)
            multiply_done = self.multiply()
            complete = farm_done and compound_done and multiply_done
            manifest = self.camp.write_manifest()
            terminal = Stage.DONE if complete else (
                Stage.WALL_BUDGET if not self.budget_left() else Stage.INCOMPLETE
            )
            self.camp.write_status(
                terminal,
                farm_complete=farm_done,
                compound_complete=compound_done,
                multiply_complete=multiply_done,
                successful=manifest["successful_episodes"],
                total=manifest["total_episodes"],
            )
        except BaseException as exc:
            if self.camp.gen.is_dir():
                self.camp.write_status(
                    Stage.FAILED, error_type=type(exc).__name__, error=str(exc)
                )
            raise


# ----------------------------------------------------------------------------- cli

def parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_vec3(value: str) -> tuple[float, float, float]:
    try:
        xyz = tuple(float(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected X,Y,Z") from exc
    if len(xyz) != 3:
        raise argparse.ArgumentTypeError("expected exactly three comma-separated values")
    return xyz


def build_config(argv: list[str] | None = None) -> Config:
    d = Config(run_dir=Path("."))  # defaults source
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--name", default=d.name)
    ap.add_argument("--sessions", default=",".join(d.sessions),
                    help="authoring levels to run, comma-separated ('' = none)")
    ap.add_argument("--session-min", type=float, default=d.session_min,
                    help="override every agent-session budget in minutes; omitted = "
                         "use each authoring condition's budget")
    ap.add_argument("--intervention-session-min", type=float,
                    default=d.intervention_session_min,
                    help="repair/vectorize session budget (these have no condition YAML)")
    ap.add_argument("--per-cell-target", type=int, default=d.per_cell_target,
                    help="verified successes required of EVERY live cell before the "
                         "farm closes (per-cell coverage, never a global count)")
    ap.add_argument("--num-envs", type=int, default=d.num_envs,
                    help="scripted-stage batch width (farm/probe/compound)")
    ap.add_argument("--wall-hours", type=float, default=d.wall_hours)
    ap.add_argument("--agent-cmd", default=d.agent_cmd,
                    help="agent loop launcher (gets --prompt-file/--workdir/--cap-min/"
                         "--transcript/--env); '' skips all agent stages")
    ap.add_argument("--isaac-py", default=d.isaac_py)
    ap.add_argument("--wide-probe-seed", type=int, default=d.wide_probe_seed)
    ap.add_argument("--nominal-seeds", type=parse_int_tuple,
                    default=d.nominal_seeds,
                    help="comma-separated seeds for the nominal probe")
    ap.add_argument("--repair-rounds", type=int, default=d.repair_rounds)
    ap.add_argument("--vectorize-rounds", type=int, default=d.vectorize_rounds,
                    help="bounded vectorize sessions; afterward the pipeline proceeds "
                         "(batches always launch wide; scalar cells crash-quarantine)")
    ap.add_argument("--quarantine-zero-yield-batches", type=int,
                    default=d.quarantine_zero_yield_batches,
                    help="completed zero-success batches before a cell leaves the farm pool")
    ap.add_argument("--compound-eps", type=int, default=d.compound_eps,
                    help="verified noise/DR variants to collect (0 disables)")
    ap.add_argument("--compound-hours", type=float, default=d.compound_hours)
    ap.add_argument("--compound-noise-scale", type=float,
                    default=d.compound_noise_scale,
                    help="noise_scale compound batches run at (solve-authored noise)")
    ap.add_argument("--noise-rounds", type=int, default=d.noise_rounds,
                    help="bounded noise-plan authoring sessions")
    ap.add_argument("--noise-floor-yield", type=float, default=d.noise_floor_yield,
                    help="share of envs a noised probe must keep succeeding")
    ap.add_argument("--noise-min-coverage", type=float, default=d.noise_min_coverage,
                    help="minimum measured share of perturbed (step, env) rows — "
                         "noise is mandatory and gamed-zero noise must not pass")
    ap.add_argument("--multiply-draws", type=int, default=d.multiply_draws,
                    help="visual replay passes per verified episode (0 disables)")
    ap.add_argument("--trim-margin", type=int, default=d.trim_margin,
                    help="rows rendered past each episode's sustained-success step "
                         "(-1 = never trim)")
    ap.add_argument("--multiply-hours", type=float, default=d.multiply_hours)
    ap.add_argument("--render-envs", type=int, default=d.render_envs,
                    help="episodes replayed in parallel per multiply render pass")
    ap.add_argument("--render-chunk", type=int, default=d.render_chunk)
    ap.add_argument("--render-kit-args", default=d.render_kit_args,
                    help="deployment-specific Kit arguments forwarded to render.py")
    ap.add_argument("--camera-eye", type=parse_vec3, default=d.cam_eye,
                    help="fallback ad-hoc camera eye as X,Y,Z")
    ap.add_argument("--camera-target", type=parse_vec3, default=d.cam_target,
                    help="fallback ad-hoc camera target as X,Y,Z")
    ap.add_argument("--camera-eye-jitter", type=parse_vec3, default=d.cam_eye_jitter)
    ap.add_argument("--camera-target-jitter", type=parse_vec3,
                    default=d.cam_target_jitter)
    ap.add_argument("--batch-timeout-s", type=float, default=d.batch_timeout_s)
    ap.add_argument("--command-timeout-s", type=float, default=d.command_timeout_s)
    ap.add_argument("--session-grace-s", type=float, default=d.session_grace_s)
    ap.add_argument("--farm-seed-base", type=int, default=d.farm_seed_base)
    ap.add_argument("--compound-seed-base", type=int, default=d.compound_seed_base)
    ap.add_argument("--visual-seed-base", type=int, default=d.visual_seed_base)
    a = ap.parse_args(argv)
    return Config(
        run_dir=Path(a.run_dir).resolve(), name=a.name,
        sessions=tuple(s for s in a.sessions.split(",") if s),
        session_min=a.session_min,
        intervention_session_min=a.intervention_session_min,
        agent_cmd=a.agent_cmd, isaac_py=a.isaac_py,
        wall_hours=a.wall_hours, per_cell_target=a.per_cell_target,
        num_envs=a.num_envs,
        quarantine_zero_yield_batches=a.quarantine_zero_yield_batches,
        wide_probe_seed=a.wide_probe_seed, nominal_seeds=a.nominal_seeds,
        repair_rounds=a.repair_rounds, vectorize_rounds=a.vectorize_rounds,
        compound_eps=a.compound_eps, compound_hours=a.compound_hours,
        compound_noise_scale=a.compound_noise_scale,
        noise_rounds=a.noise_rounds,
        noise_floor_yield=a.noise_floor_yield,
        noise_min_coverage=a.noise_min_coverage,
        multiply_draws=a.multiply_draws, multiply_hours=a.multiply_hours,
        trim_margin=a.trim_margin,
        render_envs=a.render_envs, render_chunk=a.render_chunk,
        render_kit_args=a.render_kit_args,
        cam_eye=a.camera_eye, cam_target=a.camera_target,
        cam_eye_jitter=a.camera_eye_jitter,
        cam_target_jitter=a.camera_target_jitter,
        batch_timeout_s=a.batch_timeout_s,
        command_timeout_s=a.command_timeout_s,
        session_grace_s=a.session_grace_s,
        farm_seed_base=a.farm_seed_base,
        compound_seed_base=a.compound_seed_base,
        visual_seed_base=a.visual_seed_base)


def main() -> None:
    Orchestrator(build_config()).run()


if __name__ == "__main__":
    main()
