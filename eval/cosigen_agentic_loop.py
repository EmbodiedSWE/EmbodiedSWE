"""Native swalm coding-agent loop for a persistent CoSiGen simulator.

Responsibilities are deliberately separated:

* ``cosigen_loop.py``: simulator-side control/read/RL/checkpoint API;
* ``cosigen_render_server.py``: persistent Isaac process + HTTP transport;
* this file: native swalm ``ClaudeCodeAgent`` conversation and tools.

Unlike the former hand-written LLMCaller loop, swalm owns tool calling, memory
condensation, the Portal coding sandbox, multimodal observations, and SkillTool.
"""
from __future__ import annotations

import asyncio
import base64
import traceback
import json
import re
import subprocess
import time
import urllib.request

from pathlib import Path
from typing import Any

# The multimodal ClaudeCode agent exists only in NEWER swalm-core layouts (the
# devbox/eval stack). The RL training job pins an older swalm-core (0506 era,
# module layout swalm.core.agent.claude_code) that lacks it -- and the RL rollout
# module (cosigen_rl_rollout.py) only imports the STATE/TOOL classes from this
# file, so the eval-only agent import must not break the module for that stack.
try:
    from swalm.agent.claude_code_with_multimodal_tool.agent.agent import (
        ClaudeCodeWithMultimodalToolAgent,
    )
except ModuleNotFoundError:
    ClaudeCodeWithMultimodalToolAgent = None
from swalm.core.client.env_manager import EnvManagerClient
from swalm.core.client.portal import PortalConfig
from swalm.core.tool.base import ToolBase, ToolResponse
from swalm.core.utils.bytedance.env import is_cn_region


# A turn's printed output goes into the model's context, so its size cannot be left to
# whatever the program printed (see CoSiGenToolState.stdout_for_agent). Outputs up to the
# inline budget pass through untouched -- normal turns run 1k-50k chars -- and larger ones
# are stashed whole in the agent's workspace with the head/tail shown.
STDOUT_INLINE_CHARS = 60_000
STDOUT_HEAD_CHARS = 30_000
STDOUT_TAIL_CHARS = 15_000


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(opener.open(req, timeout=timeout).read())


def _get_json(url: str, timeout: float = 20.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(opener.open(url, timeout=timeout).read())


def _read_hdfs_json(path: str) -> dict | None:
    """Read a small JSON file from HDFS (the search-progress mirror). None when it is
    absent or unreadable — callers treat that as 'no progress yet'."""
    try:
        out = subprocess.run(["hdfs", "dfs", "-cat", path],
                             capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout) if out.returncode == 0 and out.stdout else None
    except Exception:  # noqa: BLE001 -- progress reads must never break a session
        return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


ANNOTATOR_PROMPT = """\
OUT-OF-BAND annotation request (you are a scene annotator, not the task solver; this
exchange is not part of any conversation). A robot-control turn just moved the world
from a PARENT state to a CURRENT state. Using the oracle data below (object/end-effector
poses, scene summaries{and_images}), answer with ONLY a JSON object:

{{"state_diff": "what objectively changed between parent and current (object
  positions/orientations, gripper/contact state, progress or regressions)",
 "scene_diff": "what visibly changed between the two images{scene_note}"}}

Be objective; no assumptions, no advice, no code.

PARENT scene: {parent_scene}
PARENT poses: {parent_obs}
CURRENT scene: {current_scene}
CURRENT poses: {current_obs}
"""

# =====================================================================================
# DRIVER-SIDE PROMPT PARTS — the framing the driver appends around the pod's prompt,
# as editable triple-quoted blocks (same convention as cosigen_loop.py: <ANGLE_CAPS>
# tokens are substituted with plain .replace by _fill, so text may contain literal
# braces). The pod prompt itself (API doc, checkpoint/tuning sections) is composed
# server-side in cosigen_loop.py.
# =====================================================================================

def _fill(template: str, **tokens: str) -> str:
    for key, value in tokens.items():
        template = template.replace(f"<{key}>", value)
    return template


# Appended to the pod prompt at every launch. <TOOLKIT_BITS>/<USES> mirror the pod's
# actual facilities; <CHECKPOINT_HINT>/<SKILL_HINT> are empty on pods without a tree.
DRIVER_APPENDIX = """

You are running as a native swalm coding agent with your own workspace. First invoke `cosigen-staged-planning`. Keep your control programs as files in the workspace (write and edit them with your normal file tools), run one with execute(path=...), read its output and the image of the state it reached, and revise the file. Write them in the style of a standalone Isaac Lab script where useful — `env`, `api`, `torch` and isaaclab imports are live in the program's namespace (see RAW SIMULATOR ACCESS in the API doc) — while using the toolkit (<TOOLKIT_BITS>) for <USES>. Any run that moves the world is followed by assess, where you say what the image and the log show, what failure modes you can see, and whether the state is worth keeping. <CHECKPOINT_HINT><SKILL_HINT>Do not stop until execute reports success=true.
Note that your shell and file tools see only your own workspace; the simulator runs on a different host with its own filesystem, which your executed programs can read (open(), glob) but your shell cannot."""

CHECKPOINT_HINT = ("When a program does what you want, save it with "
                   "checkpoint(label=..., path=...). ")
def skill_hint(has_ckpt: bool, has_opt: bool) -> str:
    """Name only the skills this arm actually ships. The fixed "checkpoint / optimize
    skills" wording told the no-optimize arm that an optimize skill exists (its own
    system prompt, on the wire, v21) — pointing it at a tool that is not in its tool
    list and dissolving the difference between the arms."""
    names = (["checkpoint"] if has_ckpt else []) + (["optimize"] if has_opt else [])
    if not names:
        return ""
    which = " / ".join(names)
    return (f"Invoke the {which} skill{'s' if len(names) > 1 else ''} when "
            f"appropriate. ")

RESUME_TREE_NOTE = """
You are resuming durable session `<SESSION_ID>` after a driver or render-server restart. Your first execute call will restore the persisted world/tree. Then inspect list_checkpoints() and redefine any Python functions needed by the continuing trajectory."""

RESUME_LIVE_NOTE = """
You are resuming session `<SESSION_ID>` after a driver restart. The simulator kept the live world state, so your work so far is physically intact. Begin by inspecting the scene (object poses, describe_scene) to re-establish where things stand, and redefine any Python helpers you need."""

RESUME_WORKSPACE_NOTE = """
Programs you have already written in this session (they are kept for you; re-send one with execute to run it again):
<SAVED>"""

# The opening user message (the task + harness instructions live in the SYSTEM
# message — see NativeCoSiGenAgent._init_system_messages) and the between-chunks
# continuation message.
KICKOFF_MESSAGE = ("Begin. Your task, the simulator API and the required workflow are in "
                   "your system instructions. Make your staged plan and start working.")

CONTINUATION_MESSAGE = ("continue — simulator success is still false. Review the staged "
                        "plan<CKPT_BIT> and outcomes. "
                        "Change strategy substantially where it keeps failing.")


class CoSiGenToolState:
    """State shared by native swalm simulator tools."""

    def __init__(
        self,
        server: str,
        artifact_dir: str,
        *,
        max_steps: int = 100_000,
        num_frames: int = 150,
        resume: bool = False,
        resume_tree: bool | None = None,
        reset_every_turn: bool = False,
        trial_mode: bool = False,
        session_id: str | None = None,
        annotator=None,
        annotator_vision: bool = True,
        opt_server: str | None = None,
    ) -> None:
        self.annotator = annotator            # swalm LLM caller for node annotations
        self.annotator_vision = annotator_vision
        self.server = server.rstrip("/")
        # Dedicated SEARCH pod (user decision 2026-07-26). Parameter searches take
        # minutes to an hour; running them on the turn pod blocks every ordinary turn
        # behind them (one main thread per pod) and leaves the agent's world disturbed.
        # A second pod makes optimize genuinely asynchronous: searches never block
        # turns, and the agent's live world/lineage stays untouched.
        self.opt_server = opt_server.rstrip("/") if opt_server else None
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "turn_images").mkdir(exist_ok=True)
        self.max_steps = int(max_steps)
        self.num_frames = int(num_frames)
        self.first_execute = not resume
        self.resume = bool(resume)
        # Restoring the persisted tree jumps the world back to the last PERSISTED node —
        # correct after a render-server restart, but it rewinds un-checkpointed progress
        # when the server session survived (and a no-checkpoint baseline only ever
        # persists its root, so restore would wipe everything). resume_tree=False keeps
        # the live server-side world as-is.
        self.resume_tree = self.resume if resume_tree is None else bool(resume_tree)
        # No-checkpoint baseline protocol (user decision 2026-07-25): without a tree the
        # honest paradigm is EPISODIC — every execution resets the world and the agent
        # iterates on one self-contained solver script (like eval/examples solvers),
        # rather than accumulating un-rewindable state across turns.
        self.reset_every_turn = bool(reset_every_turn)
        # Trial/checkpoint protocol for the checkpointing arms: each execution starts from
        # the current node, and a node is created only when the agent checkpoints a program
        # that exits cleanly. Off for the episodic baseline (it resets to the initial state).
        self.trial_mode = bool(trial_mode) and not self.reset_every_turn
        self.code_dir = self.artifact_dir / "code"
        self.code_dir.mkdir(exist_ok=True)
        self.session_id = _safe_name(
            session_id or f"{self.artifact_dir.name}-{int(time.time())}"
        )
        self.last_result: dict[str, Any] = {}
        self.pending_annotation: tuple[str, dict] | None = None
        # Turn artifacts must APPEND across driver restarts: numbering restarting at 1
        # on --resume overwrote earlier turn jsons/images (destroyed a recorded eval
        # success, 2026-07-24). Continue from the highest existing turn file.
        existing = sorted(self.artifact_dir.glob("turn_*.json"))
        self.executions = int(existing[-1].stem.split("_")[1]) if existing else 0
        self.task_completed = False
        self.agent = None  # backlinked by NativeCoSiGenAgent for token accounting
        self.portal = None  # PortalClient, backlinked so tools can read workspace files
        # Harness watchdog (set by run_native_cosigen_agent for --watchdog on): consulted
        # after every executed turn; its soft reminders ride the tool feedback.
        self.watchdog = None
        self._runs_since_checkpoint = 0
        self._runs_since_optimize = 0
        # ---- async parameter search (optimize tool) ----
        # opt_job holds the in-flight search: its asyncio task, the program it is
        # tuning (path + saved version), and the latest progress the pod published.
        self.opt_job: dict[str, Any] | None = None
        self.opt_notes: list[str] = []   # progress lines to ride the next tool result
        self.opt_status_file = "/workspace/optimize_status.json"
        # ---- inspection gate ----
        # A run that moved the world has to be looked at before the next one: v21 agents
        # spent whole stretches re-tuning a grasp while the leg lay flat on the table,
        # which the end-state image shows at a glance. Set when a run steps the world,
        # cleared by the assess tool, and checked by execute.
        self.pending_review: dict[str, Any] | None = None
        self.tunable_flagged = ""     # judge's note when the run's constants are tunable
        self.reviews = 0
        self.inspected_node = ""      # node whose image/log the agent has just been shown

    async def read_workspace_file(self, path: str) -> str:
        """Read one of the agent's own files out of its sandbox workspace.

        The agent writes its programs there with the ordinary Write/Edit/Bash tools, but the
        simulator is a separate process on another host: it never sees that filesystem. The
        driver holds the portal connection, so it is the piece that can fetch a program by
        path and hand its text to the simulator.

        Uses download_files, the same API the agent's own tools go through: the streaming
        endpoint does not exist on this portal version (its 404 was a missing ROUTE, which
        looked exactly like a missing file — v12 checkpoint failure, 2026-07-25)."""
        if self.portal is None:
            raise RuntimeError("no portal connection: pass code inline instead of a path")
        resp = await self.portal.download_files([path], plain_text=True)
        content = (resp.files or {}).get(path)
        if content is None:
            raise FileNotFoundError(
                f"{path}: {(resp.errors or {}).get(path, 'not found in the workspace')}")
        return content

    # ----------------- async parameter search (optimize tool) -----------------
    async def start_optimize(self, request: dict, *, program_path: str,
                             program_version: str, node: str | None) -> dict:
        """Launch a search on the SEARCH pod and return immediately.

        The search pod is put at the agent's current checkpoint node first (it loads the
        session's persisted tree, then goes to that node), so the search explores from
        exactly the state the agent is standing on. Everything is tagged with the saved
        program version, because the agent keeps editing while the search runs and a
        result must say which file it belongs to."""
        if self.opt_job and not self.opt_job["task"].done():
            return {"started": False,
                    "reason": (f"a search is already running on "
                               f"{self.opt_job['program_path']} "
                               f"({self.opt_job['version']}); it will report when it "
                               f"finishes — read {self.opt_status_file} for progress")}
        server = self.opt_server or self.server
        progress_hdfs = (f"hdfs://haruna/tmp/zeyu.shen/cosigen_opt_progress/"
                         f"{self.session_id}.json")
        request.setdefault("options", {})["progress_hdfs"] = progress_hdfs
        job: dict[str, Any] = {
            "server": server, "program_path": program_path,
            "version": program_version, "node": node, "started": time.time(),
            "progress": {"state": "starting"}, "result": None, "task": None,
            "dedicated": bool(self.opt_server), "progress_hdfs": progress_hdfs,
        }
        job["task"] = asyncio.create_task(self._run_optimize(job, request, node))
        self.opt_job = job
        await self._write_opt_status()
        return {"started": True, "server": server, "dedicated": bool(self.opt_server)}

    async def _run_optimize(self, job: dict, request: dict, node: str | None) -> None:
        """Background task: sync the search pod to `node`, run the search, keep
        job['progress'] fresh from the pod's /ping while it runs."""
        server = job["server"]
        try:
            if self.opt_server:  # dedicated pod: bring it to the agent's node
                prep_meta: dict[str, Any] = {
                    "trial_mode": self.trial_mode,
                    "max_steps": self.max_steps,
                    "session_id": self.session_id,
                    "resume_tree": True,
                }
                code = (f"print(goto({node!r}))" if node
                        else "print('search pod at episode start')")
                await asyncio.to_thread(
                    _post_json, server + "/run_policy",
                    {"code": code, "reset": node is None, "max_steps": self.max_steps,
                     "num_frames": 0, "meta": prep_meta, "session": self.session_id},
                    1800.0)
            poller = asyncio.create_task(self._poll_optimize(job))
            try:
                result = await asyncio.to_thread(
                    _post_json, server + "/run_policy",
                    {"code": "pass  # optimize turn (program ships via meta)",
                     "reset": False, "max_steps": self.max_steps, "num_frames": 0,
                     "meta": {"trial_mode": self.trial_mode, "optimize": request},
                     "session": self.session_id},
                    40000.0)
            finally:
                poller.cancel()
            job["result"] = result
            opt = result.get("optimize_result") or {}
            job["progress"] = {"state": "done" if opt.get("best") is not None
                               else "failed", **opt}
            self._runs_since_optimize = 0
            self.opt_notes.append(self._opt_summary(job))
        except Exception as exc:  # noqa: BLE001 -- a failed search must not kill the run
            job["progress"] = {"state": "failed", "error": repr(exc)}
            job["result"] = {"rc": -1, "stderr": repr(exc)}
            self.opt_notes.append(
                f"optimize on {job['program_path']} ({job['version']}) FAILED: {exc!r}")
            traceback.print_exc()
        finally:
            await self._write_opt_status()

    async def _poll_optimize(self, job: dict, every: float = 20.0) -> None:
        """Copy the pod's per-generation progress into the job + workspace file.

        Two channels, in order: the pod's /ping payload (served from its HTTP thread
        while the search holds the main thread), and an HDFS mirror the search writes
        itself. The mirror matters because /ping lives in the render server file, which
        only updates when a pod is rebooted — on a hot-reloaded pod it is the only
        channel that carries progress."""
        last_done = -1
        while True:
            await asyncio.sleep(every)
            prog = None
            try:
                prog = (await asyncio.to_thread(
                    _get_json, job["server"] + "/ping")).get("optimize_progress")
            except Exception:  # noqa: BLE001 -- transient ping failures are fine
                pass
            if not isinstance(prog, dict) and job.get("progress_hdfs"):
                prog = await asyncio.to_thread(_read_hdfs_json, job["progress_hdfs"])
            if not isinstance(prog, dict):
                continue
            job["progress"] = prog
            done = int(prog.get("generations_done") or 0)
            if done > last_done:
                last_done = done
                await self._write_opt_status()
                if done:
                    self.opt_notes.append(
                        f"optimize progress on {job['program_path']} "
                        f"({job['version']}): generation {done}/"
                        f"{prog.get('generations_planned')} done, best so far "
                        f"{prog.get('best')} score {prog.get('best_score')}")

    def _opt_summary(self, job: dict) -> str:
        opt = (job.get("result") or {}).get("optimize_result") or {}
        if not opt.get("best"):
            return (f"optimize on {job['program_path']} ({job['version']}) returned no "
                    f"result — see {self.opt_status_file}")
        extra = ""
        if opt.get("at_bounds"):
            extra += (f" NOTE: best at the edge of the range for {opt['at_bounds']} — "
                      f"widening may find better.")
        if opt.get("budget_truncated"):
            extra += (f" NOTE: stopped by budget after {opt.get('generations_run')}/"
                      f"{opt.get('generations_planned')} generations.")
        return (f"optimize FINISHED for {job['program_path']} (version "
                f"{job['version']}): best {opt.get('best')} score "
                f"{opt.get('best_score')} from {opt.get('n_evals')} evaluations in "
                f"{opt.get('seconds')}s.{extra} Put these values into "
                f"{job['program_path']}, verify with execute, then checkpoint.")

    async def _write_opt_status(self) -> None:
        """Publish the search's state into the agent's workspace: a file it can read at
        any moment with its own tools, so progress never waits on a tool round-trip."""
        job = self.opt_job
        if job is None:
            return
        payload = {
            "program": job["program_path"],
            "program_version": job["version"],
            "anchor_node": job["node"],
            "search_pod": job["server"],
            "dedicated_search_pod": job["dedicated"],
            "started_ago_s": round(time.time() - job["started"], 1),
            **(job.get("progress") or {}),
        }
        text = json.dumps(payload, indent=2, default=str)
        try:
            (self.artifact_dir / "optimize_status.json").write_text(text)
        except OSError:
            pass
        if self.portal is not None:
            try:
                await self.portal.upload_files({self.opt_status_file: text},
                                               plain_text=True)
            except Exception:  # noqa: BLE001 -- the note channel still works
                traceback.print_exc()

    def drain_opt_notes(self) -> list[str]:
        notes, self.opt_notes = self.opt_notes, []
        return notes

    def review_block(self) -> dict | None:
        """The refusal payload while a run that moved the world is unassessed.

        Every tool that moves the world asks this, not just execute: a v22 agent called
        checkpoint with run 13 still unassessed, which ran a program, replaced the pending
        review and left that run's image unread.
        """
        pending = self.pending_review
        if not pending:
            return None
        return {
            "error": "assess the last run first",
            "run": pending["run"],
            "what_to_do": (
                f"Run {pending['run']} moved the world and its end-state image and log "
                f"came back with it. Call assess with what the image shows, what the "
                f"numbers show, the failure modes you can see, and whether this state is "
                f"worth keeping — assess saves it for you when you keep it."),
        }

    def record_review(self, params: dict, pending: dict) -> None:
        """Keep the agent's reading of a run next to the run itself, and open the gate."""
        self.reviews += 1
        row = {"run": pending.get("run"), "name": pending.get("name"),
               "node": pending.get("node"), "keep": bool(params.get("keep")),
               "scene": params.get("scene"), "log": params.get("log"),
               "failure_modes": params.get("failure_modes"),
               "optimize": params.get("optimize"), "t": time.time()}
        try:
            with open(self.artifact_dir / "assessments.jsonl", "a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            traceback.print_exc()
        self.pending_review = None
        self.tunable_flagged = ""

    async def stdout_for_agent(self, run: int, text: str) -> str:
        """Keep a program's printed output whole while keeping the turn's tool result
        inside the model's context window. Output size is entirely the agent's choice
        and one exploratory stage walk over 512 envs printed 25k lines / 1.9M chars
        (~1.1M tokens) -- past the context limit, so the turn's result could not be
        delivered at all. Anything over the window is written to the agent's workspace
        in full and the result carries the head, the tail and that path, so the agent
        reads or greps the rest with its own file tools. Nothing is discarded: the
        durable full copy is in the turn artifact either way."""
        if len(text) <= STDOUT_INLINE_CHARS:
            return text
        path = f"/workspace/logs/turn_{run:04d}_stdout.txt"
        stashed = False
        if self.portal is not None:
            try:
                await self.portal.upload_files({path: text}, plain_text=True)
                stashed = True
            except Exception:  # noqa: BLE001 -- fall back to the inline window
                traceback.print_exc()
        head, tail = text[:STDOUT_HEAD_CHARS], text[-STDOUT_TAIL_CHARS:]
        where = (f"read the whole thing at {path}" if stashed else
                 "the full text is in this turn's artifact on the driver host")
        return (f"{head}\n\n[... this program printed {len(text)} characters; the "
                f"{STDOUT_HEAD_CHARS} above and the {STDOUT_TAIL_CHARS} below are shown "
                f"inline -- {where}. Printing per-object lines for every environment is "
                f"rarely what you want: print env 0, or aggregate across envs. ...]"
                f"\n\n{tail}")

    async def execute(self, code: str, stage_summary: str = "", *, name: str = "",
                      checkpoint: bool = False, label: str = "",
                      optimize: dict | None = None, internal: bool = False,
                      goto_allowed: bool = False) -> dict:
        first_execute_of_driver = not getattr(self, "_sent_session_meta", False)
        self._sent_session_meta = True
        self.executions += 1
        meta: dict[str, Any] = {}
        if goto_allowed:
            meta["goto_allowed"] = True
        if self.trial_mode:
            meta["trial_mode"] = True
            if checkpoint:
                meta["checkpoint"] = {"label": label or stage_summary, "name": name}
        if optimize:
            # Parameter-search turn: the pod runs the shipped program/objective sources
            # instead of `code`. The placeholder passes the server's empty-code check;
            # run_policy never executes it on optimize turns.
            meta["optimize"] = optimize
            code = code or "pass  # optimize turn (program ships via meta)"
        else:
            self._save_code_version(code, name, checkpoint)
        if first_execute_of_driver:
            meta["max_steps"] = self.max_steps
            # RL training rollouts (cosigen_rl_rollout.py) set persist_session=False:
            # episodes are disposable, so the durable HDFS checkpoint-tree persistence
            # that session_id enables server-side is pure wasted I/O there.
            if getattr(self, "persist_session", True):
                meta["session_id"] = self.session_id
                meta["resume_tree"] = self.resume_tree
        # Re-arm the episode recorder EVERY turn: enable_recording() resets the frame
        # buffer, so each turn records fresh footage and each turn's frames_npz is that
        # turn's video. Without this the boot-time buffer (max_frames) silently fills a
        # few thousand steps into a long session and every later turn returns the same
        # frozen early-session frames (v6 bug, 2026-07-22).
        # RL rollouts set record_video=False: no footage wanted, so never (re)arm the
        # recorder (arming it would pay the RTX capture cost every control step).
        if getattr(self, "record_video", True):
            meta["arm_recorder"] = {"every": 2}  # no frame caps: every frame is kept
        if self.pending_annotation:
            cid, caption = self.pending_annotation
            meta["annotations"] = {cid: caption}
            self.pending_annotation = None

        payload = {
            "code": code,
            "reset": self.first_execute or self.reset_every_turn,
            "max_steps": self.max_steps,
            "num_frames": self.num_frames,
            "meta": meta,
            # Episode-lease identity (RL training pools): pods leased via /acquire only
            # execute turns carrying the owning session. Servers ignore this when no
            # lease is active, so eval drivers are unaffected.
            "session": self.session_id,
        }
        self.first_execute = False

        # The server executes each turn on its single main thread and replies within
        # CAPX_RUN_TIMEOUT_S (default 1800s); a LONGER turn gets an HTTP 504 while the
        # turn KEEPS EXECUTING server-side. Re-POSTing after a 504 queues a duplicate
        # execution behind the running turn and eventually kills the driver (this took
        # the v7 opus driver down twice on 2026-07-23). So: on 504, never re-POST —
        # wait for the in-flight turn to finish (n_runs advances) and recover its
        # result via /last_result (older servers without that endpoint: return an
        # honest lost-output result). Connection-level errors (no response at all)
        # keep the plain retry.
        import urllib.error

        n_runs_before: int | None = None
        try:
            n_runs_before = int(_get_json(self.server + "/ping").get("n_runs"))
        except Exception as exc:  # noqa: BLE001
            print(f"[transport] pre-submit ping failed (non-fatal): {exc!r}", flush=True)

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                result = await asyncio.to_thread(
                    _post_json, self.server + "/run_policy", payload, 3600.0
                )
                break
            except Exception as exc:
                # A turn can outlive the reply window in TWO shapes: the server's own
                # 504 (CAPX_RUN_TIMEOUT_S), or our socket read timeout firing first
                # (newer pods reply at 7200s > the 3600s socket timeout, so the client
                # times out with a raw TimeoutError before any 504 arrives — this
                # killed the v8 and v8base drivers on 2026-07-24). Both mean the same
                # thing: the turn is still executing server-side; never re-POST.
                timed_out = (
                    (isinstance(exc, urllib.error.HTTPError) and exc.code == 504)
                    or isinstance(exc, TimeoutError)
                    or (isinstance(exc, urllib.error.URLError)
                        and isinstance(getattr(exc, "reason", None), TimeoutError))
                )
                if timed_out:
                    result = await self._recover_after_504(n_runs_before)
                    break
                last_error = exc
                if attempt == 3:
                    raise
                await asyncio.sleep(30.0 * attempt)
        else:  # pragma: no cover
            raise last_error or RuntimeError("CoSiGen request failed")

        self.last_result = result
        self.task_completed = bool(result.get("success"))
        tree = result.get("tree") or {}
        if tree.get("new_node"):
            ann: dict[str, str] = {}
            if stage_summary:
                # Stored in full — no word cap (user directive 2026-07-26: never
                # truncate the agent's own descriptions).
                ann["action"] = stage_summary
            diffs = await self._annotate_diffs(result.get("diff_context") or {})
            ann.update(diffs)
            if ann:
                self.pending_annotation = (tree["new_node"], ann)

        new_node = (result.get("tree") or {}).get("new_node")
        if new_node:
            # record what actually happened (a node was created), not what was requested
            try:
                with open(self.code_dir / "history.jsonl", "a") as f:
                    f.write(json.dumps({"run": self.executions, "node": new_node,
                                        "checkpointed": True, "t": time.time()}) + "\n")
            except Exception:
                traceback.print_exc()
        optimized = bool(result.get("optimize_result"))
        self._runs_since_checkpoint = 0 if new_node else self._runs_since_checkpoint + 1
        self._runs_since_optimize = 0 if optimized else self._runs_since_optimize + 1
        # Frames are recorded exactly when the world moves (checked across every turn of
        # the v21 sessions), so this is the marker for "this run has to be looked at".
        result["internal_turn"] = bool(internal)  # audits separate agent runs from plumbing
        if not internal and int(result.get("n_frames") or 0) > 0:
            self.pending_review = {"run": self.executions, "name": name,
                                   "node": tree.get("current"),
                                   "checkpointed": bool(new_node)}
        if self.watchdog is not None and not internal:
            try:
                self.watchdog.observe({
                    "run": self.executions, "name": name, "code": code,
                    "stdout": str(result.get("stdout", "")), "rc": result.get("rc"),
                    "checkpointed": bool(new_node), "optimized": optimized,
                    "runs_since_checkpoint": self._runs_since_checkpoint,
                    "runs_since_optimize": self._runs_since_optimize,
                    "node": (result.get("tree") or {}).get("current"),
                })
                advice = await self.watchdog.maybe_advise()
                if advice:
                    result["watchdog_advice"] = advice
                # On the arms that have optimize, the judge's reading of the program is
                # what the assessment has to answer to: hand-picked constants mean the
                # agent either searches them or says why not.
                if self.pending_review and self.watchdog.has_opt:
                    self.tunable_flagged = self.watchdog.last_tunable
            except Exception:  # a watchdog bug must never break the session
                traceback.print_exc()
        self._save_turn_artifacts(result)
        return result

    async def _recover_after_504(self, n_runs_before: int | None) -> dict:
        """The turn overran the server's reply window (HTTP 504) but is still executing
        on the simulator. Wait for it to finish (n_runs advances past the pre-submit
        count), then fetch the completed result if the server caches it (/last_result);
        otherwise return an honest lost-output result so the session continues."""
        print(f"[transport] 504: turn still executing server-side; waiting for it to "
              f"finish (n_runs_before={n_runs_before})", flush=True)
        deadline = time.time() + 6 * 3600.0
        while time.time() < deadline:
            try:
                meta = await asyncio.to_thread(_get_json, self.server + "/ping", 20.0)
                if n_runs_before is None or int(meta.get("n_runs", -1)) > n_runs_before:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"[transport] ping during 504 recovery failed: {exc!r}", flush=True)
            await asyncio.sleep(60.0)
        try:
            got = await asyncio.to_thread(_get_json, self.server + "/last_result", 60.0)
            if got.get("ok") and isinstance(got.get("result"), dict):
                print("[transport] recovered the finished turn's full result via "
                      "/last_result", flush=True)
                return got["result"]
        except Exception as exc:  # noqa: BLE001
            print(f"[transport] /last_result unavailable ({exc!r}); synthesizing "
                  "lost-output feedback", flush=True)
        return {
            "rc": -1, "stdout": "", "success": False, "scene_summary": "",
            "event_log": [], "n_frames": 0, "frames_npz": "", "frames_npz_parts": [],
            "tree": {}, "render_seconds": None,
            "stderr": (
                "[transport] this turn ran longer than the server's reply window; it "
                "FINISHED executing on the simulator, but its printed output/result "
                "could not be returned. The world state includes everything the turn "
                "did. Do NOT resubmit the same code (it already ran). Re-read the "
                "state (scene queries, list_checkpoints()) to see the "
                "outcome, then continue; checkpoint real progress. Prefer breaking "
                "very long stages into shorter turns so their output is not lost."
            ),
        }

    async def _annotate_diffs(self, ctx: dict) -> dict:
        """Out-of-trajectory side-call producing state_diff (+ scene_diff for VLMs) for
        the node just created. Best-effort: never blocks or fails the turn."""
        cur, par = ctx.get("current") or {}, ctx.get("parent") or {}
        if self.annotator is None or not cur or not par:
            return {}
        try:
            have_imgs = bool(self.annotator_vision and cur.get("thumb_b64")
                             and par.get("thumb_b64"))
            text = ANNOTATOR_PROMPT.format(
                and_images=" and the two frames" if have_imgs else "",
                scene_note="" if have_imgs else " (omit if no images provided)",
                parent_scene=par.get("scene", ""), parent_obs=json.dumps(par.get("obs", {})),
                current_scene=cur.get("scene", ""), current_obs=json.dumps(cur.get("obs", {})),
            )
            if have_imgs:
                content = [
                    {"type": "text", "text": text},
                    {"type": "text", "text": "PARENT frame:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{par['thumb_b64']}"}},
                    {"type": "text", "text": "CURRENT frame:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{cur['thumb_b64']}"}},
                ]
            else:
                content = text
            resp = await self.annotator._call_llm([{"role": "user", "content": content}])
            m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
            if not m:
                return {}
            diffs = json.loads(m.group(0))
            return {k: str(v) for k, v in diffs.items()
                    if k in ("state_diff", "scene_diff") and v}
        except Exception as exc:
            print(f"[annotate] diff side-call failed (non-fatal): {exc}", flush=True)
            return {}

    def save_program_version(self, code: str, stem: str) -> str:
        """Snapshot the exact source a background search is tuning, and return its
        version name. Searches outlive the agent's next edits, so a result has to say
        which version of the program it optimized."""
        name = f"{_safe_name(stem) or 'program'}__opt{self.executions:04d}"
        try:
            out_dir = self.code_dir / "optimize"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.py").write_text(
                f"# searched at {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + code)
        except OSError:
            traceback.print_exc()  # bookkeeping must never fail the tool call
        return name

    def _save_code_version(self, code: str, name: str, checkpoint: bool) -> None:
        """Keep every program the agent runs in the artifact dir. The agent's own sandbox
        is deleted when the driver exits, so this is the durable copy of its workspace.
        Named programs (run by path) live at code/<name>.py; anonymous inline programs are
        filed under code/runs/, named from their '# plan:' comment when they have one, so
        the top level only shows the agent's deliberate versions."""
        stem = _safe_name(name)
        sub = ""
        if not stem:
            sub = "runs"
            for line in code.splitlines():
                if line.strip().startswith("# plan:"):
                    slug = _safe_name(line.strip()[7:].strip().replace(" ", "_")[:48])
                    if slug:
                        stem = f"run{self.executions:04d}_{slug}"
                    break
            stem = stem or f"run{self.executions:04d}"
        try:
            out_dir = self.code_dir / sub if sub else self.code_dir
            out_dir.mkdir(exist_ok=True)
            header = (f"# run {self.executions} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                      f"{' (checkpointed)' if checkpoint else ''}\n")
            (out_dir / f"{stem}.py").write_text(header + code)
            with open(self.code_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"run": self.executions, "name": stem,
                                    "checkpoint": bool(checkpoint),
                                    "t": time.time()}) + "\n")
        except Exception:
            traceback.print_exc()  # never fail a turn over bookkeeping

    def workspace_summary(self) -> str:
        """One line per stored program version, newest last, for the resume note."""
        lines = []
        for path in sorted(self.code_dir.glob("*.py")):
            plan = ""
            for line in path.read_text().splitlines():
                if line.strip().startswith("# plan:"):
                    plan = line.strip()[7:].strip()
                    break
            lines.append(f"  {path.stem}{': ' + plan if plan else ''}")
        return "\n".join(lines)

    def _save_turn_artifacts(self, result: dict) -> None:
        # Save each tool result immediately. A driver/server failure must not erase
        # a long trajectory.
        safe = dict(result)
        images = safe.pop("turn_images", []) or []
        safe.pop("diff_context", None)  # carries two thumbnails; not worth persisting per turn
        # Token accounting per turn (token-vs-performance plots): conversation size
        # is the tokens of the LAST LLM call (prompt+completion, i.e. what the 128k
        # budget is measured against); cumulative is the sum over every call so far.
        if self.agent is not None:
            safe["tokens"] = {
                "conversation": int(getattr(self.agent, "_conv_tokens", 0)),
                "cumulative": int(getattr(self.agent, "_cum_tokens", 0)),
            }
            # Midpoint-restart durability (2026-07-25): the chunk-boundary snapshot was
            # too coarse — a driver crash mid-chunk lost the whole conversation. Persist
            # it after EVERY turn, atomically (tmp+rename: a crash mid-write must never
            # truncate the only copy).
            try:
                conv = getattr(self.agent, "conversations", None)
                if conv:
                    tmp = self.artifact_dir / "native_swalm_conversation.json.tmp"
                    with open(tmp, "w") as cf:
                        json.dump(conv, cf, ensure_ascii=False, default=str)
                    tmp.rename(self.artifact_dir / "native_swalm_conversation.json")
            except Exception:
                traceback.print_exc()  # durability is best-effort; never kill a turn
            try:
                with open(self.artifact_dir / "tokens.jsonl", "a") as tf:
                    tf.write(json.dumps({
                        "turn": self.executions,
                        "conversation_tokens": safe["tokens"]["conversation"],
                        "cumulative_tokens": safe["tokens"]["cumulative"],
                        "score": result.get("score"),
                        "success": bool(result.get("success")),
                        "t": time.time(),
                    }) + "\n")
            except Exception:
                traceback.print_exc()  # accounting must never kill a turn
        with open(self.artifact_dir / f"turn_{self.executions:04d}.json", "w") as f:
            json.dump(safe, f, indent=1, ensure_ascii=False, default=str)
        (self.artifact_dir / "session_id.txt").write_text(self.session_id + "\n")
        for i, image in enumerate(images):
            name = _safe_name(image.get("view", "view"))
            path = self.artifact_dir / "turn_images" / (
                f"turn_{self.executions:04d}_{i}_{name}.jpg"
            )
            path.write_bytes(base64.b64decode(image["jpeg_b64"]))


class CoSiGenExecuteTool(ToolBase):
    """Native simulator execution tool."""

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "execute",
                "description": (
                    "Run one Python program in the CoSiGen simulator. Your Python "
                    "variables persist across calls."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Path in your workspace to the program to run. Keep "
                                "your programs as files and edit them between runs; "
                                "pass this instead of code."
                            ),
                        },
                        "code": {
                            "type": "string",
                            "description": (
                                "Inline source, for quick one-off probes and reads. For a "
                                "solver-stage program, write it to a workspace file, run "
                                "it with path, and refine it with your file-editing tools "
                                "between runs instead of re-sending the whole program. "
                                "Start with '# plan: ...' and print the measurements that "
                                "tell you whether it worked."
                            ),
                        },
                        "stage_summary": {
                            "type": "string",
                            "description": (
                                "Intent and observed outcome in <= 20 words."
                            ),
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        path = str(params.get("path", "") or "")
        code = str(params.get("code", "") or "")
        if not (path or code):
            return ToolResponse(output=json.dumps(
                {"error": "give either path (a program in your workspace) or code"}))
        try:
            if path:
                code = await self.state.read_workspace_file(path)
        except Exception as exc:  # noqa: BLE001 -- report, do not kill the session
            return ToolResponse(output=json.dumps(
                {"error": f"could not read {path}: {exc!r}"}))
        blocked = self.state.review_block()
        if blocked:
            return ToolResponse(output=json.dumps(blocked))
        result = await self.state.execute(
            code, str(params.get("stage_summary", "")),
            name=(Path(path).stem if path else ""),
        )
        images = result.get("turn_images") or []
        feedback = {
            "rc": result.get("rc"),
            "success": bool(result.get("success")),
            "stdout": await self.state.stdout_for_agent(
                self.state.executions, str(result.get("stdout", ""))),
            "stderr": result.get("stderr", ""),
            "scene": result.get("scene_summary", ""),
            # rl_log = the same event log from pre-rename server files (see the
            # drain_rl_log alias in cosigen_apis): read both until pods are rebooted.
            "events": result.get("event_log") or result.get("rl_log") or [],
            "tree": result.get("tree") or {},
            "images": (f"{len(images)} image(s) of the end state are attached below"
                       if images else "none (the program did not move anything)"),
            "frames_npz": result.get("frames_npz", ""),
            "render_seconds": result.get("render_seconds"),
        }
        if result.get("watchdog_advice"):
            feedback["harness_advice"] = result["watchdog_advice"]
        notes = self.state.drain_opt_notes()
        if notes:
            feedback["optimize_updates"] = notes
        if self.state.trial_mode:
            # The anchor semantics have to be unmissable: a v11 agent ran 16 programs
            # without ever checkpointing, so nothing it achieved was saved (2026-07-25).
            node = (result.get("tree") or {}).get("current")
            feedback["anchor"] = (
                f"the world was restored to node {node} before this program ran and is "
                f"back there now. Checkpoint a working program when you want to save its "
                f"end state and build your future work off from there."
                if not (result.get("tree") or {}).get("new_node") else
                f"checkpointed: node {node} now holds this program's end state, and your "
                f"next program starts from there.")
        if not result.get("success"):
            feedback["status"] = "task not complete — continue working"
        return _result_with_images(feedback, images)


def _result_with_images(feedback: dict, images: list[dict]) -> ToolResponse:
    """One tool result carrying the turn's numbers and the image of its end state.

    NativeCoSiGenAgent._convert_multimodal_tool_output keeps both, so the agent sees
    where the program left the world without having to ask for it.
    """
    text = json.dumps(feedback, ensure_ascii=False, default=str)
    if not images:
        return ToolResponse(output=text)
    content = [{"type": "text", "text": text}]
    for im in images:
        if im.get("jpeg_b64"):
            content.append({"type": "image", "mimeType": "image/jpeg",
                            "data": im["jpeg_b64"]})
    return ToolResponse(output=json.dumps({"content": content}))


class CoSiGenGotoTool(ToolBase):
    """Move the world to a saved node, showing that node before it happens.

    The first call renders the node and returns its image, its log and the program that
    reached it; the world only moves when the agent calls again with confirm=true. v21
    agents backtracked nine times without once reading the target's log or looking at it,
    so the pieces they needed were there and unused.
    """

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "goto",
                "description": (
                    "Move the world back to a saved node. Called without confirm it shows "
                    "you that node's image, log and program; call again with confirm=true "
                    "to actually go there."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node": {"type": "string",
                                 "description": "Node id, e.g. n3."},
                        "confirm": {
                            "type": "boolean",
                            "description": ("True to move the world there, after you have "
                                            "seen what the node holds."),
                        },
                    },
                    "required": ["node"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        blocked = self.state.review_block()
        if blocked:
            return ToolResponse(output=json.dumps(blocked))
        node = str(params.get("node") or "").strip()
        if not node:
            return ToolResponse(output=json.dumps({"error": "which node?"}))
        if not params.get("confirm"):
            probe = (
                f"print('--- log at {node} ---')\n"
                f"print(get_checkpoint_log({node!r}))\n"
                f"print('--- scene at {node} ---')\n"
                f"print(get_checkpoint_scene({node!r}))\n"
                f"print('--- program that reached it ---')\n"
                f"print(get_checkpoint_code({node!r}))\n"
                f"print(render_checkpoint({node!r}))\n")
            res = await self.state.execute(probe, "", name=f"inspect_{node}",
                                           internal=True)
            if res.get("rc"):
                return ToolResponse(output=json.dumps(
                    {"error": f"could not read {node}", "stderr": res.get("stderr", "")}))
            self.state.inspected_node = node
            return _result_with_images(
                {"node": node,
                 "state_and_log": str(res.get("stdout", "")),
                 "what_to_do": ("If this is the state you want to work from, call goto "
                                "again with confirm=true; otherwise inspect another "
                                "node or keep working where you are.")},
                res.get("turn_images") or [])
        if getattr(self.state, "inspected_node", "") != node:
            return ToolResponse(output=json.dumps({
                "error": f"look at {node} first",
                "what_to_do": f"Call goto with node={node} and no confirm to see its "
                              f"image and log, then confirm.",
            }))
        res = await self.state.execute(f"print(goto({node!r}))", "",
                                      name=f"goto_{node}", internal=True,
                                      goto_allowed=True)
        self.state.inspected_node = ""
        if res.get("rc"):
            return ToolResponse(output=json.dumps(
                {"error": f"goto {node} failed", "stderr": res.get("stderr", "")}))
        return ToolResponse(output=json.dumps(
            {"node": node, "result": str(res.get("stdout", "")),
             "next": "your next program runs from this node"}))


class CoSiGenAssessTool(ToolBase):
    """The look-at-what-happened step, required after any run that moved the world.

    The end-state image and the printed numbers arrive with the run itself; this is where
    the agent has to say what it sees in them, name the failure modes it can identify, and
    decide whether the state is worth keeping. On the arms that have optimize, a run whose
    outcome rests on hand-picked constants must also say either that it is searching them
    or why it is not.
    """

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "assess",
                "description": (
                    "Required after a run that moved the world, before the next run: "
                    "report what the end-state image and the log show, the failure modes "
                    "you can see, and whether to keep this state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene": {
                            "type": "string",
                            "description": ("What the end-state image shows — where the "
                                            "objects and the hands actually ended up."),
                        },
                        "log": {
                            "type": "string",
                            "description": ("What the printed numbers say, including the "
                                            "measurements you care about."),
                        },
                        "failure_modes": {
                            "type": "string",
                            "description": ("What went wrong or is blocking progress, "
                                            "reasoned from what you just saw and read. "
                                            "Say so plainly if the run went as intended."),
                        },
                        "keep": {
                            "type": "boolean",
                            "description": ("True to checkpoint this program so its end "
                                            "state becomes a node you can build on and "
                                            "return to."),
                        },
                        "label": {
                            "type": "string",
                            "description": "Short label for the node, when keep is true.",
                        },
                        "path": {
                            "type": "string",
                            "description": ("Workspace path of the program that produced "
                                            "this state, when keep is true."),
                        },
                        "optimize": {
                            "type": "string",
                            "description": ("Only when the run's outcome rests on numbers "
                                            "you picked by hand: 'searching' if you are "
                                            "handing them to optimize, otherwise why you "
                                            "are not."),
                        },
                    },
                    "required": ["scene", "log", "failure_modes", "keep"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        pending = self.state.pending_review
        if not pending:
            return ToolResponse(output=json.dumps(
                {"error": "nothing to assess: the last run did not move the world"}))
        missing = [k for k in ("scene", "log", "failure_modes")
                   if not str(params.get(k) or "").strip()]
        if missing:
            return ToolResponse(output=json.dumps(
                {"error": f"say something for: {', '.join(missing)}"}))
        if self.state.tunable_flagged and not str(params.get("optimize") or "").strip():
            return ToolResponse(output=json.dumps({
                "error": "this run's outcome rests on numbers you picked by hand",
                "constants": self.state.tunable_flagged,
                "what_to_do": ("Set optimize to 'searching' if you are handing them to "
                               "the optimize tool, or say why searching them is not the "
                               "right move right now."),
            }))
        self.state.record_review(params, pending)
        out: dict[str, Any] = {"recorded": True, "run": pending["run"]}
        if params.get("keep"):
            path = str(params.get("path") or "")
            if not path:
                return ToolResponse(output=json.dumps(
                    {"error": "to keep this state, give the path of the program that "
                              "produced it"}))
            try:
                code = await self.state.read_workspace_file(path)
            except Exception as exc:  # noqa: BLE001
                return ToolResponse(output=json.dumps(
                    {"error": f"could not read {path}: {exc!r}"}))
            # internal: this is the run the agent has just assessed, replayed to turn its
            # end state into a node. Asking for a second assessment of the same program
            # would loop the gate.
            res = await self.state.execute(code, str(params.get("label") or ""),
                                           name=Path(path).stem, checkpoint=True,
                                           label=str(params.get("label") or ""),
                                           internal=True)
            node = (res.get("tree") or {}).get("new_node")
            out["checkpoint"] = (f"saved as {node}: your next program starts there"
                                 if node else
                                 "not saved — the program did not exit cleanly this time")
            out["rc"] = res.get("rc")
        else:
            out["checkpoint"] = "not kept; your next program runs from the current node"
        return ToolResponse(output=json.dumps(out, ensure_ascii=False, default=str))


class CoSiGenCheckpointTool(ToolBase):
    """Checkpoint one of the agent's programs as the edge to a new tree node."""

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "checkpoint",
                "description": (
                    "Save progress: run a program from your workspace against the node you "
                    "are on and make its end state a new node, with that program as the "
                    "node's code. Call it whenever you decide a version is good — after "
                    "running and refining it as many times as you like."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": (
                                "Any short free-text description of the state this "
                                "program reaches."
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Path in your workspace to the program to checkpoint."
                            ),
                        },
                    },
                    "required": ["label", "path"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        blocked = self.state.review_block()
        if blocked:
            return ToolResponse(output=json.dumps(blocked))
        path, label = str(params["path"]), str(params["label"])
        try:
            code = await self.state.read_workspace_file(path)
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(output=json.dumps(
                {"error": f"could not read {path}: {exc!r}", "checkpointed": False}))
        result = await self.state.execute(code, label, name=Path(path).stem,
                                          checkpoint=True, label=label)
        tree = result.get("tree") or {}
        node = tree.get("new_node")
        feedback = {
            "checkpointed": bool(node),
            "node": node or tree.get("current"),
            "rc": result.get("rc"),
            "stdout": await self.state.stdout_for_agent(
                self.state.executions, str(result.get("stdout", ""))),
            "stderr": result.get("stderr", ""),
            "scene": result.get("scene_summary", ""),
            "success": bool(result.get("success")),
        }
        if result.get("watchdog_advice"):
            feedback["harness_advice"] = result["watchdog_advice"]
        notes = self.state.drain_opt_notes()
        if notes:
            feedback["optimize_updates"] = notes
        feedback["result"] = (
            f"checkpointed as {node}: your programs now start from there"
            if node else
            f"not checkpointed — the program did not exit cleanly, so you are still on "
            f"{tree.get('current')}. Fix it and checkpoint again.")
        return ToolResponse(output=json.dumps(feedback, ensure_ascii=False, default=str))


class CoSiGenOptimizeTool(ToolBase):
    """Parameter search over the agent's own program file (user-mandated interface,
    2026-07-25): the agent provides paths + the constants to search; the pod runs one
    copy of the program per env in lockstep and returns the best setting found.

    ASYNCHRONOUS since 2026-07-26: the call returns as soon as the search is launched on
    the search pod, and the agent keeps working. Results stream back per generation —
    into /workspace/optimize_status.json (readable any time) and appended to subsequent
    tool results — each tagged with the program version being tuned."""

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "optimize",
                "description": (
                    "Optimize constants in one of your programs: runs many copies of "
                    "the program in parallel on a second simulator, each with a "
                    "different setting of the named constants, scores every end state "
                    "with your objective file, and searches for the best setting. Call "
                    "it whenever a program needs a parameter setting tuned. This "
                    "returns immediately and the search runs in the background: keep "
                    "working, and results arrive as each round finishes (also readable "
                    "any time in /workspace/optimize_status.json)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program": {
                            "type": "string",
                            "description": ("Workspace path of the program to tune "
                                            "(the file you are iterating on, unchanged)."),
                        },
                        "space": {
                            "type": "object",
                            "description": (
                                "Constants to search: {'<CONST_NAME>': [low, high]} — "
                                "names of top-level numeric constants in the program; "
                                "append 'int' or 'log' as a third element for integer "
                                "or log-scaled search."),
                        },
                        "objective": {
                            "type": "string",
                            "description": ("Workspace path of a file defining "
                                            "def objective(v) -> float (lower is "
                                            "better), scored on each copy's end state."),
                        },
                        "setup": {
                            "type": "string",
                            "description": (
                                "Optional workspace path of setup code to run once on "
                                "the search simulator before the copies, with the raw "
                                "env/api handles the copies do not get. Use it for "
                                "simulator configuration you applied on your own "
                                "simulator and that a checkpoint does not carry — "
                                "controller gains, for instance — so the search tunes "
                                "against the same robot you are working with."),
                        },
                        "generations": {"type": "integer"},
                        "budget_s": {"type": "number"},
                        "repeats": {"type": "integer"},
                        "randomize": {"type": "object"},
                        "max_steps_per_eval": {"type": "integer"},
                    },
                    "required": ["program", "space", "objective"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        prog_path, obj_path = str(params["program"]), str(params["objective"])
        try:
            program_src = await self.state.read_workspace_file(prog_path)
            objective_src = await self.state.read_workspace_file(obj_path)
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(output=json.dumps(
                {"error": f"could not read the files: {exc!r}"}))
        options = {k: params[k] for k in
                   ("generations", "budget_s", "repeats", "randomize",
                    "max_steps_per_eval") if params.get(k) is not None}
        if params.get("setup"):
            try:
                options["setup_src"] = await self.state.read_workspace_file(
                    str(params["setup"]))
            except Exception as exc:  # noqa: BLE001
                return ToolResponse(output=json.dumps(
                    {"error": f"could not read the setup file: {exc!r}"}))
        # The searched file is snapshotted under a version name, so a result that
        # arrives later says which version of the program it belongs to (the agent
        # keeps editing while the search runs).
        version = self.state.save_program_version(program_src, Path(prog_path).stem)
        request = {"program": program_src, "objective": objective_src,
                   "space": params.get("space") or {}, "options": options,
                   "progress_meta": {"program": prog_path, "program_version": version}}
        node = (self.state.last_result.get("tree") or {}).get("current")
        started = await self.state.start_optimize(
            request, program_path=prog_path, program_version=version, node=node)
        if not started.get("started"):
            return ToolResponse(output=json.dumps(
                {"started": False, "reason": started.get("reason")}))
        feedback = {
            "started": True,
            "program": prog_path,
            "program_version": version,
            "searching_from_node": node,
            "space": params.get("space"),
            "status_file": self.state.opt_status_file,
            "result": (
                "the search is running in the background"
                + (" on the search simulator" if started.get("dedicated")
                   else " (no separate search simulator configured, so it shares this one)")
                + f"; it is tuning {prog_path} as saved in version {version}. Carry on "
                f"with your next piece of work — each round's best setting is appended "
                f"to your following tool results and kept in "
                f"{self.state.opt_status_file}. When a result arrives, apply the values, "
                "verify with execute, then checkpoint."),
        }
        return ToolResponse(output=json.dumps(feedback, ensure_ascii=False, default=str))


class CoSiGenViewTool(ToolBase):
    """Native multimodal observation tool."""

    def __init__(self, state: CoSiGenToolState):
        super().__init__()
        self.state = state
        self.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "view",
                "description": (
                    "Show more of the last run than the end-state image you already "
                    "received: frames=k samples k frames evenly across the run, so you "
                    "can see how it got there."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frames": {
                            "type": "integer",
                            "description": ("How many frames to sample across the last "
                                            "run (default 4)."),
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, params: dict) -> ToolResponse:
        k = int(params.get("frames") or 4)
        # Sampling frames means asking the simulator to pull them out of the run's
        # recorded footage, which is a turn of its own; it does not step the world.
        result = await self.state.execute(
            f"print(review_last({max(1, min(k, 12))}))", "", name="view_frames")
        images = result.get("turn_images") or []
        if not images:
            return ToolResponse(output=json.dumps(
                {"images": "none — the last run recorded no footage",
                 "stdout": str(result.get("stdout", ""))[:400]}))
        return _result_with_images(
            {"frames": len(images), "of": "the last run, earliest first"}, images)


class NativeCoSiGenAgent(ClaudeCodeWithMultimodalToolAgent or object):
    """ClaudeCodeAgent with native simulator tools plus normal swalm tools.

    (On the pinned RL training stack the multimodal base is unavailable and this
    class is inert -- the RL loop builds its own agent in cosigen_rl_rollout.py.)"""

    def __init__(self, *args, cosigen_state: CoSiGenToolState,
                 harness_instructions: str = "", **kwargs):
        self._cosigen_state = cosigen_state
        self._harness_instructions = harness_instructions
        # Token accounting for the token-vs-performance analysis (2026-07-24): the
        # base agent OVERWRITES total_tokens_used with each call's total, so nothing
        # durable exists. Track both the live conversation size and the cumulative
        # tokens consumed across every LLM call, and expose them to the tool state
        # so each turn artifact records them.
        self._conv_tokens = 0
        self._cum_tokens = 0
        cosigen_state.agent = self
        super().__init__(*args, **kwargs)

    async def _init_system_messages(self):
        """Append the task + harness instructions as a SYSTEM message.

        They used to be the first user message. That worked (every call resends the
        whole conversation, and the condenser's keep_first=1 preserved it), but only
        by configuration: system messages are preserved structurally — the condenser
        keeps the entire leading system block, whatever its settings. User directive
        2026-07-25: harness instructions live in the system prompt."""
        await super()._init_system_messages()
        if self._harness_instructions:
            self.conversations.append({
                "role": "system",
                "content": self._harness_instructions,
            })

    @property
    def total_tokens_used(self) -> int:
        return self._conv_tokens

    @total_tokens_used.setter
    def total_tokens_used(self, value: int) -> None:
        self._conv_tokens = int(value or 0)
        self._cum_tokens += int(value or 0)

    def _convert_multimodal_tool_output(self, result):
        """Let one tool result carry text AND images.

        swalm's version keeps only the image items and drops the rest, so a tool that
        returned both lost its text — which is why looking at the scene used to need a
        second call after execute. Text items are preserved here, in order, so a turn's
        numbers and the picture of where the program left the world arrive together.
        """
        r = result
        if isinstance(r, str):
            # Most tool results are plain text (Bash, Read, Write): not our envelope and
            # not an error either, so hand them to swalm's text path without noise.
            if not r.lstrip().startswith("{"):
                return None
            try:
                r = json.loads(r)
            except json.JSONDecodeError:
                return None
        content = r.get("content") if isinstance(r, dict) else None
        if not isinstance(content, list):
            return None
        try:
            out, has_image = [], False
            for item in content:
                if not isinstance(item, dict):
                    continue
                mime = item.get("mimeType") or item.get("mime_type")
                if item.get("type") == "image" and mime and item.get("data"):
                    out.append({"type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{item['data']}"}})
                    has_image = True
                elif item.get("type") == "text" and item.get("text"):
                    out.append({"type": "text", "text": str(item["text"])})
            return out if has_image else None
        except Exception:  # noqa: BLE001 -- fall back to swalm's text-only path
            traceback.print_exc()
            return None

    def _init_tools(self, llm_config, portal_config, finish_tool):
        super()._init_tools(llm_config, portal_config, finish_tool)
        self._cosigen_state.portal = self.portal_client
        self.tools.extend([
            CoSiGenExecuteTool(self._cosigen_state),
            CoSiGenViewTool(self._cosigen_state),
        ])
        # the look-at-what-happened step gates every run that moved the world
        self.tools.append(CoSiGenAssessTool(self._cosigen_state))
        # checkpoint and goto only exist where a checkpoint tree does: the episodic
        # baseline must not be offered tools that silently do nothing there
        if self._cosigen_state.trial_mode:
            self.tools.append(CoSiGenCheckpointTool(self._cosigen_state))
            self.tools.append(CoSiGenGotoTool(self._cosigen_state))
        # optimize only exists on pods whose prompt carries the tuning section
        if getattr(self._cosigen_state, "has_opt", False):
            self.tools.append(CoSiGenOptimizeTool(self._cosigen_state))


def _apply_skill_features(docs: str, features: dict[str, bool]) -> str:
    """Resolve `<!-- if:NAME --> ... [<!-- else -->] ... <!-- endif -->` blocks in a
    skill body against the pod's facilities, then renumber `N.` list items so a
    stripped step leaves no gap. No nesting."""
    def _resolve(m: "re.Match[str]") -> str:
        name, body = m.group(1), m.group(2)
        halves = re.split(r"<!--\s*else\s*-->", body, maxsplit=1)
        keep = halves[0] if features.get(name, True) else \
            (halves[1] if len(halves) > 1 else "")
        return keep.strip("\n")
    out = re.sub(r"<!--\s*if:(\w+)\s*-->\n?(.*?)<!--\s*endif\s*-->\n?",
                 lambda m: (_resolve(m) + "\n") if _resolve(m) else "",
                 docs, flags=re.S)
    n = 0
    lines = []
    for line in out.splitlines():
        m = re.match(r"^(\d+)\.(\s)", line)
        if m:
            n += 1
            line = f"{n}.{m.group(2)}" + line[m.end():]
        lines.append(line)
    return "\n".join(lines).strip()


def load_cosigen_skills(eval_dir: str | Path, *, has_ckpt: bool = True,
                        has_opt: bool = True) -> list[dict]:
    """Load the skills matching THIS pod's facilities: whole skills for missing tools
    are dropped (checkpoint tree / optimize), and skill bodies with conditional
    blocks are resolved so no arm reads about calls it does not have."""
    features = {"checkpoint": has_ckpt, "opt": has_opt}
    root = Path(eval_dir) / "skills"
    skills = []
    for directory in sorted(root.iterdir()):
        path = directory / "SKILL.md"
        if not path.exists():
            continue
        raw = path.read_text()
        parts = raw.split("---", 2)
        front, docs = parts[1], parts[2].strip()
        meta = {}
        for line in front.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        name = meta["name"].lower()
        if "rl" in name.split("-"):  # retired RL-subpolicy skills (legacy/)
            continue
        if not has_ckpt and "checkpoint" in name:
            continue
        if not has_opt and "optimize" in name:
            continue
        skills.append({
            "name": meta["name"],
            "description": meta["description"],
            "location": str(directory),
            "base_dir": str(directory),
            "docs": _apply_skill_features(docs, features),
        })
    return skills


def _close_pending_tool_calls(conversations: list[dict]) -> None:
    """The snapshot is written while a tool call is in flight, so a restored conversation
    can end with an assistant tool_use that has no tool result. Left dangling, the agent
    silently REPEATS that call — which for a persistent world may double-apply an action
    it cannot see. Append an honest result instead, so it re-checks state first."""
    answered = {m.get("tool_call_id") for m in conversations if m.get("role") == "tool"}
    for msg in reversed(conversations):
        if msg.get("role") != "assistant":
            continue
        pending = [c for c in (msg.get("tool_calls") or [])
                   if c.get("id") not in answered]
        for call in pending:
            conversations.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": (call.get("function") or {}).get("name"),
                "content": json.dumps({"content": [{"type": "text", "text": (
                    "DRIVER RESTARTED while this call was executing: its result was not "
                    "recorded. The simulator may or may not have applied it. Do NOT assume "
                    "it succeeded or failed — inspect the current state first (poses, "
                    "scene queries, and the checkpoint tree if you have one), then continue."
                )}]}),
            })
        if pending:
            print(f"[native] closed {len(pending)} in-flight tool call(s) from the "
                  "snapshot with an explicit 'result lost' message", flush=True)
        break


async def run_native_cosigen_agent(
    *,
    server: str,
    llm_config,
    artifact_dir: str,
    max_steps: int = 100_000,
    resume: bool = False,
    resume_tree: bool | None = None,
    reset_every_turn: bool = False,
    trial_mode: bool = False,
    resume_conversation: bool = False,
    session_id: str | None = None,
    portal_image: str = "hub.byted.org/base/python:3.11",
    portal_version: str = "2.7.2",
    chunk_iterations: int | None = None,  # None = swalm's own run() default
    max_chunks: int | None = None,
    extra_prompt: str = "",
    token_budget: int | None = None,
    watchdog: str = "",
    opt_server: str | None = None,
) -> list[dict]:
    """Run native swalm chunks until simulator success.

    Production default has no global cap. ``max_chunks`` exists only for bounded
    smoke tests of the native integration. ``token_budget`` (e.g. 128_000 for eval
    sessions) ends the session once the conversation reaches that many tokens: the
    agent's context is capped there (no condensation rescue) and the chunk loop
    stops. Chunk size defaults to swalm's own run() turn limit."""
    from swalm.core.llm.llm_caller_factory import LLMCallerFactory

    state = CoSiGenToolState(
        server,
        artifact_dir,
        max_steps=max_steps,
        num_frames=150,
        resume=resume,
        resume_tree=resume_tree,
        reset_every_turn=reset_every_turn,
        trial_mode=trial_mode,
        session_id=session_id,
        opt_server=opt_server,
        # Node annotator: same model as the agent, out-of-trajectory side-calls that
        # produce the per-node state_diff/scene_diff annotations.
        annotator=LLMCallerFactory.create_llm_caller(llm_config),
        annotator_vision=True,
    )
    prompt = _get_json(server.rstrip("/") + "/ping").get("prompt", "")
    # The server prompt is authoritative about which harness facilities exist on this
    # pod (ablation baselines strip checkpointing/RL): mirror it in the driver-side
    # framing and skill list rather than assuming the full toolkit.
    has_ckpt = "list_checkpoints()" in prompt  # tree section present on this pod
    state.has_opt = "== Tuning the numbers" in prompt  # optimize tool offered iff present
    if watchdog and (state.has_opt or has_ckpt):
        # One judge covers every facility this pod offers, so the reminder can only ever
        # point at a tool the agent actually has. Runs on every checkpoint-capable arm,
        # not just the optimize arm, so the arms differ only by the facilities themselves.
        from cosigen_watchdog import HarnessWatchdog
        state.watchdog = HarnessWatchdog(state.annotator, artifact_dir,
                                         has_opt=state.has_opt, has_ckpt=has_ckpt)
        facilities = ", ".join(
            n for n, on in (("optimize", state.has_opt), ("checkpoint", has_ckpt)) if on)
        print(f"[native] harness watchdog armed for: {facilities}", flush=True)
    prompt += _fill(
        DRIVER_APPENDIX,
        TOOLKIT_BITS=("move_to, look" + (", checkpoint/goto" if has_ckpt else "")),
        USES=("control, vision" + (", backtracking" if has_ckpt else "")),
        CHECKPOINT_HINT=CHECKPOINT_HINT if has_ckpt else "",
        SKILL_HINT=skill_hint(has_ckpt, state.has_opt),
    )
    if extra_prompt:
        prompt += "\n\n" + extra_prompt
    if resume and state.resume_tree:
        prompt += _fill(RESUME_TREE_NOTE, SESSION_ID=state.session_id)
    elif resume:
        prompt += _fill(RESUME_LIVE_NOTE, SESSION_ID=state.session_id)
    if resume:
        saved = state.workspace_summary()
        if saved:
            prompt += _fill(RESUME_WORKSPACE_NOTE, SAVED=saved)

    em_client = EnvManagerClient()
    async with em_client.env_session(portal_image, portal_version) as env_session:
        direct = is_cn_region()
        portal = PortalConfig(
            endpoint=env_session["base_url"] if direct else env_session["proxy_url"],
            direct_mode=direct,
            keepalive_endpoint=env_session.get("keepalive_url"),
        )
        agent = NativeCoSiGenAgent(
            llm_config,
            portal,
            cosigen_state=state,
            harness_instructions=prompt,
            # Offer only the skills whose facilities exist on this pod: a skill for a
            # missing tool (checkpoint on the episodic baseline, optimize on stripped
            # pods) would teach calls that do not exist.
            skills=load_cosigen_skills(Path(__file__).resolve().parent,
                                       has_ckpt=has_ckpt, has_opt=state.has_opt),
            multi_agent_mode=False,
            use_web_tools=False,
            observation_truncate_name="no_truncate",
            # With a token budget the conversation IS the budget: cap the context there
            # and never condense (condensing would let the session outlive its budget).
            max_context_length=(token_budget if token_budget else 900_000),
            condense_threshold=(1.01 if token_budget else 0.80),
        )
        if chunk_iterations is None:
            # swalm's native per-run() turn limit (core.agent.max_iterations, 20):
            # no driver-invented chunk size (user decision 2026-07-26). The chunk
            # boundary is where the driver checks token budget/success and injects
            # the continuation message, so a budgeted session can overshoot its
            # budget by at most one swalm-default chunk of turns.
            from swalm.agent.claude_code.agent.agent import default_run_kwargs
            chunk_iterations = int(default_run_kwargs["max_iterations"])

        # Workspace round-trip self-test BEFORE any turn runs. On v12 the portal's
        # missing streaming route returned 404s that looked exactly like missing
        # files: every agent concluded "workspace paths don't work", stopped using
        # files, and (for trial-mode agents) never checkpointed again. A broken read
        # path must kill the launch loudly, not poison a whole session silently.
        probe = "/workspace/.capx_workspace_selftest"
        await state.portal.upload_files({probe: "ok"}, plain_text=True)
        got = await state.read_workspace_file(probe)
        if got.strip() != "ok":
            raise RuntimeError(f"workspace read self-test failed: got {got!r}")
        print("[native] workspace read self-test passed", flush=True)

        conversations = None
        if resume_conversation:
            snap = Path(artifact_dir) / "native_swalm_conversation.json"
            if snap.exists():
                conversations = json.load(open(snap))
                _close_pending_tool_calls(conversations)
                print(f"[native] resumed conversation from {snap} "
                      f"({len(conversations)} messages)", flush=True)
            else:
                print(f"[native] --resume-conversation: no snapshot at {snap}; "
                      "starting a fresh conversation", flush=True)
        chunks = 0
        while not state.task_completed:
            if max_chunks is not None and chunks >= max_chunks:
                break
            if token_budget and getattr(agent, "total_tokens_used", 0) >= token_budget:
                print(f"[native] token budget reached "
                      f"({agent.total_tokens_used} >= {token_budget}); ending session",
                      flush=True)
                break
            chunks += 1
            # The task + harness instructions are a SYSTEM message (see
            # NativeCoSiGenAgent._init_system_messages); the opening user message is
            # just the kickoff, and later chunks get a short continuation.
            continuation = KICKOFF_MESSAGE if conversations is None else _fill(
                CONTINUATION_MESSAGE,
                CKPT_BIT=", checkpoint tree" if has_ckpt else "",
            )
            conversations = await agent.run(
                continuation,
                max_iterations=int(chunk_iterations),
                conversations=conversations,
            )
            with open(Path(artifact_dir) / "native_swalm_conversation.json", "w") as f:
                json.dump(conversations, f, indent=1, ensure_ascii=False, default=str)
        return conversations
