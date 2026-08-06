"""Text-only RL rollout loop for sim_gen tasks (SimGen-RL training).

The training twin of eval's NATIVE-CC arm -- the `claude` adapter in
eval/docker/entrypoints/agent-entry.sh, which runs the CLI with
`--allowedTools Bash Edit Write Read Glob Grep LS` and none of the CoSiGen
toolkit. The agent here gets the stock ClaudeCodeAgent tools plus `execute`, the
one affordance eval's arm gets for free (its container has Isaac on PATH; our
portal sandbox does not, so reaching the scene has to be a tool call). It is
driven by the TRAINING LLM (swalm AlphaSeedStreaming caller, whose chat tracker
records the token ids/logprobs that alpha-seed packs into training data) rather
than an external API model.

Not present, and why:

  * checkpoint / goto / optimize / assess -- the CoSiGen toolkit, which is the
    `cosigen` adapter's arm ("precisely the difference under test"). The pool
    pool must run a world built under a tool-free CONDITION (eval/configs/no_tools.yaml)
    so the pod's prompt and program namespace drop them too;
  * view, and images in the execute feedback -- multimodal, and M13-12B is not;
  * annotator side-calls -- out-of-trajectory LLM calls producing an image-based
    scene_diff;
  * skills -- they teach the toolkit above; eval's claude arm ships none;
  * num_frames=0 and the per-turn recorder is never armed (no video capture
    cost during training rollouts).

This file lives in CoSiGen/eval ON PURPOSE: the harness (cosigen_loop.py /
cosigen_agentic_loop.py) is actively iterated, and the RL job imports this
module from a synced copy with an mtime-based reload (simgen_rl.task), so a
harness edit reaches new training episodes without a job restart. Server-side
harness edits reach live pods through the existing /reload hot-reload.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid

# ClaudeCodeAgent moved between swalm-core layouts: newer stacks (devbox/eval)
# package it under swalm.agent.claude_code.*; the RL job's PINNED 0506 swalm-core
# has it at swalm.core.agent.claude_code (the same path the alpha-seed whitelist
# names). API-audited against the pinned version: identical __init__/_init_tools/
# run/doubao_fc surface, EXCEPT `skills` (newer-only -- passed conditionally).
try:
    from swalm.agent.claude_code.agent.agent import ClaudeCodeAgent
except ModuleNotFoundError:
    from swalm.core.agent.claude_code import ClaudeCodeAgent
from swalm.core.client.env_manager import EnvManagerClient
from swalm.core.client.portal import PortalConfig
from swalm.core.tool.base import ToolResponse
from swalm.core.utils.bytedance.env import is_cn_region

from cosigen_agentic_loop import (
    CoSiGenExecuteTool,
    CoSiGenToolState,
    _get_json,
)

# NATIVE-CC ARM (user decision 2026-07-30). The agent gets the stock ClaudeCodeAgent
# toolset -- Bash, Read, Write, Edit, MultiEdit, Glob, Grep, TodoWrite, BashOutput,
# KillBash -- and NOTHING from the CoSiGen toolkit except `execute`, which is not a
# feature but the only transport to the simulator (eval's claude arm reaches Isaac by
# running python in its own container; we have no Isaac in the portal sandbox, so the
# same affordance has to arrive as a tool call).
#
# Deliberately NOT registered here: assess, checkpoint, goto, optimize (the harness
# facilities under test in eval/docker/entrypoints/agent-entry.sh, where the `claude`
# adapter runs with --allowedTools Bash Edit Write Read Glob Grep LS and the `cosigen`
# adapter adds the toolkit -- "precisely the difference under test"), and view (the
# multimodal observation tool; this model is text-only).
#
# The POD must agree: build/launch the pool under eval/configs/no_tools.yaml so make_prompt()
# never appends those sections (they are per-tool files, selected by the condition) and
# make_namespace() drops their functions. Otherwise the prompt advertises facilities the driver
# does not expose.


def _tool_name(tool) -> str:
    return tool.__tool_schema__["function"]["name"]


class TextOnlyExecuteTool(CoSiGenExecuteTool):
    """execute for a text-only model: same transport/state (inherited), but the
    feedback drops the image affordances (there is no View tool in this session) and the
    per-turn graded score is recorded for the reward. Built ON the base tool's feedback
    so harness-side feedback evolution flows through unchanged."""

    async def execute(self, params: dict) -> ToolResponse:
        t0 = time.time()
        resp = await super().execute(params)
        # Wall time of the whole tool call vs the sim's own render_seconds: the
        # difference is HTTP + queueing + harness overhead (bottleneck accounting).
        prof = getattr(self.state, "profile", None)
        if prof is not None:
            rs = (self.state.last_result or {}).get("render_seconds")
            prof["tool_wall_s"] += time.time() - t0
            prof["sim_s"] += float(rs) if isinstance(rs, (int, float)) else 0.0
            prof["tool_calls"] += 1
        # Reward bookkeeping: the raw server response carries the scene's graded
        # score() (drain_turn_payload); the model-facing feedback does not.
        score = (self.state.last_result or {}).get("score")
        if isinstance(score, (int, float)):
            self.state.score_history.append(float(score))
        try:
            fb = json.loads(resp.output)
        except (json.JSONDecodeError, TypeError):
            return resp
        fb.pop("images_available", None)
        fb.pop("next_action", None)  # base tool points at view, absent here
        return ToolResponse(output=json.dumps(fb, ensure_ascii=False, default=str))


class SimGenRolloutAgent(ClaudeCodeAgent):
    """ClaudeCodeAgent + the text-only simulator execute tool (training rollouts).

    Mirrors NativeCoSiGenAgent minus the multimodal pieces (View tool, annotator).
    Token accounting matches it: the base agent overwrites total_tokens_used per
    call, so track the live conversation size AND the cumulative consumption --
    the episode loop budgets on the conversation size.
    """

    def __init__(self, *args, cosigen_state: CoSiGenToolState, **kwargs):
        self._cosigen_state = cosigen_state
        self._conv_tokens = 0
        self._cum_tokens = 0
        cosigen_state.agent = self
        super().__init__(*args, **kwargs)

    async def _call_llm(self, *args, **kwargs):
        """Time every LLM call so an episode's wall clock can be split into
        LLM / tool(sim) / other. Measured 2026-07-25: turns cost 189-574s wall
        while the sim itself is only 17-67s, so the remainder had to be located."""
        t0 = time.time()
        try:
            return await super()._call_llm(*args, **kwargs)
        finally:
            prof = getattr(self._cosigen_state, "profile", None)
            if prof is not None:
                prof["llm_s"] += time.time() - t0
                prof["llm_calls"] += 1

    @property
    def total_tokens_used(self) -> int:
        return self._conv_tokens

    @total_tokens_used.setter
    def total_tokens_used(self, value: int) -> None:
        self._conv_tokens = int(value or 0)
        self._cum_tokens += int(value or 0)

    def _init_tools(self, llm_config, portal_config, finish_tool):
        super()._init_tools(llm_config, portal_config, finish_tool)
        # execute reads workspace files by path, so it needs the portal backlink.
        self._cosigen_state.portal = self.portal_client
        self.tools.append(TextOnlyExecuteTool(self._cosigen_state))

    def tool_names(self) -> dict:
        """Live tool names, read from the schemas (never hardcoded): the harness
        renames tools, and a stale literal in the prompt would name a tool that does
        not exist."""
        names = {}
        for t in self.tools:
            if getattr(t, "__tool_schema__", None) and isinstance(t, TextOnlyExecuteTool):
                names["execute"] = _tool_name(t)
        return names


# The behavioural contract is NOT paraphrased here -- it is read from the same file
# the eval harness ships to its agents (eval/prompts/rules/autonomous_operation.md,
# selected by eval/configs/default.yaml), so an edit there reaches training
# rollouts unchanged. Two of its bullets presume affordances that exist only in the
# container arm (`submit`, the `solution/` deliverable). eval's own rule is that a
# prompt may WITHHOLD but never LIE about the world (envbuild.condition.check_facts),
# so those two are dropped rather than reworded into instructions for tools we do not
# have -- naming a non-existent `submit` is exactly how an agent burns turns on calls
# that cannot resolve.
_RULE_DROP_MARKERS = ("`submit`", "solution/")


def _autonomous_operation_rule(eval_dir: str) -> str:
    """eval's autonomous-operation rule, minus the bullets about tools this arm has
    no counterpart for. Returns '' if the file is gone (the framing below still
    states the no-tool-call contract itself)."""
    path = os.path.join(eval_dir, "prompts", "rules", "autonomous_operation.md")
    try:
        with open(path) as f:
            raw = f.read()
    except OSError as exc:
        print(f"[simgen_rl] autonomous_operation rule unreadable ({exc!r}); "
              f"using the inline contract only", flush=True)
        return ""
    kept, drop = [], False
    for line in raw.splitlines():
        if line.startswith("- "):          # a new bullet ends any drop in progress
            drop = any(m in line for m in _RULE_DROP_MARKERS)
        elif drop and not line.startswith(" "):
            drop = False                    # continuation lines are indented
        if not drop:
            kept.append(line)
    return "\n".join(kept).strip()


def _framing(prompt: str, names: dict, eval_dir: str) -> str:
    """Driver-side framing appended to the server's task prompt.

    NATIVE-CC ARM: the agent has the stock ClaudeCodeAgent toolset plus `execute`.
    The server prompt (make_prompt: head + TASK + api_doc + CODE_DIRECTIVE) is
    authoritative about the scene and the callable surface -- with
    a tool-free condition on the pod it already describes exactly the
    facilities this arm has. What is added here is what the pod cannot know: that
    this session is text-only, and the autonomous-operation contract eval states in
    its rules folder. `names` carries the LIVE tool names, read from the schemas."""
    ex = names.get("execute", "execute")
    rule = _autonomous_operation_rule(eval_dir)
    return prompt + (
        f"\n\nYou are running as a native swalm coding agent. Use the `{ex}` tool to "
        "interact with the simulator: it runs one Python program against the live "
        "scene, and your variables persist between programs. Write that code in the "
        "style of a standalone Isaac Lab script -- `env`, `api`, `torch` and isaaclab "
        "imports are live in your namespace (see RAW SIMULATOR ACCESS in the API doc). "
        "The other tools (Bash, Read, Write, Edit, Glob, Grep, TodoWrite) act on your "
        "workspace, NOT on the simulator; there is no Isaac install there, so a "
        "program only reaches the scene through "
        f"`{ex}`.\n\n"
        "This session is TEXT-ONLY: no images can be shown to you (look()/render "
        "captures are unavailable), so verify every stage through printed "
        "measurements, object poses and the scene summary instead.\n\n"
        + (rule + "\n\n" if rule else "")
        + "ALWAYS act by calling a tool: a reply that contains no tool call ends the "
        f"session, exactly as the rule above says. Do not stop until `{ex}` reports "
        "success=true."
    )


async def run_rl_episode(
    *,
    server: str,
    llm_config,
    agent_init_params: dict | None = None,
    artifact_dir: str | None = None,
    session_id: str | None = None,
    sim_max_steps: int = 100_000,
    max_iterations: int = 120,
    env_manager_url: str | None = None,
    env_manager_token: str | None = None,
    portal_version: str = "default",
    portal_image: str = "hub.byted.org/base/python:3.11",
    extra_prompt: str = "",
) -> dict:
    """One training episode on a (leased) render pod. Returns
    {success, final_score, best_score, turns, llm_calls, trajectories, conversations}.

    `max_iterations` is the agent's turn limit for the episode -- the same knob the
    reference coding-RL job sets as trainer.agent_max_iterations and hands to a single
    agent.run(). The episode ends when the agent stops, solves the task, or exhausts
    it; nothing re-prompts a stopped agent.
    """
    server = server.rstrip("/")
    session_id = session_id or f"simgenrl-{uuid.uuid4().hex[:12]}"
    artifact_dir = artifact_dir or os.path.join(
        tempfile.gettempdir(), "simgen_rollouts", session_id)

    state = CoSiGenToolState(
        server,
        artifact_dir,
        max_steps=int(sim_max_steps),
        num_frames=0,          # text-only: nothing rendered for the model
        resume=False,
        session_id=session_id,
        annotator=None,        # no out-of-trajectory LLM side-calls in training
        annotator_vision=False,
    )
    state.persist_session = False  # disposable episode: no HDFS tree persistence
    state.record_video = False     # never arm the recorder (RTX capture cost)
    state.score_history = []       # graded score() per turn, for the reward
    state.profile = {"llm_s": 0.0, "llm_calls": 0, "tool_wall_s": 0.0,
                     "sim_s": 0.0, "tool_calls": 0}
    episode_t0 = time.time()

    server_prompt = _get_json(server + "/ping").get("prompt", "")
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    # NATIVE-CC ARM: this arm registers `execute` and nothing else, so a pod advertising the
    # toolkit sections was built under the wrong CONDITION. Say so loudly once per episode --
    # a silently mismatched prompt and toolset is how a model spends its turns calling tools
    # that do not exist.
    for section, tool in (("== Checkpoint tree", "checkpoint_tree"),
                          ("== Tuning the numbers", "evolutionary_parameter_search"),
                          ("== Staged planning", "staged_planning")):
        if section in server_prompt:
            print(f"[simgen_rl] WARNING pod {server} advertises '{section}' but this "
                  f"arm registers no such tool; rebuild/relaunch the world under a condition "
                  f"whose tools: list omits '{tool}' (eval/configs/no_tools.yaml)", flush=True)

    init_kwargs: dict = {
        "multi_agent_mode": False,
        "use_web_tools": False,
        "observation_truncate_name": "no_truncate",
        # CONDENSATION MUST BE FULLY OFF (root-caused 2026-07-25 on run v4).
        # ClaudeCodeAgent._should_condense returns False iff max_context_length <= 0;
        # a high condense_threshold is NOT enough (threshold * max_context_length =
        # 114800 tokens was still reachable inside our 121856-token context, so
        # condensation fired). Condensation is fundamentally incompatible with
        # SWALM_MESSAGES_PATTERN=continuous_tokens, which this job runs:
        #   * _rebuild_conversation_history REPLACES the history with
        #     system + 1 summary message, so the conversation SHRINKS;
        #   * AlphaSeedStreamingClient.encode_messages_to_ids then sends
        #     messages[history_message_length:], which is now EMPTY ->
        #     tokenizer.apply_chat_template([]) -> IndexError: list index out of
        #     range (152 episodes lost; 380 "chat is []" warnings);
        #   * and the summary call itself appends to an already-at-budget
        #     conversation, so it returns finish_reason='length' with empty
        #     content -> ValueError('Failed to generate summary') (330 more).
        # The context ceiling is enforced server-side instead
        # (request_args.overlong_early_stop, which the swalm handler defaults to True),
        # and overlong trajectories are still trained via the trainer's
        # swalm_*_overlong_train_stratagy=learn_all.
        "max_context_length": 0,
    }
    # NO SKILLS in the native-cc arm. eval's `claude` adapter runs the CLI with
    # --allowedTools Bash Edit Write Read Glob Grep LS and no skill files; the skills
    # under eval/skills teach the CoSiGen toolkit (checkpoint tree, parameter search,
    # staged planning around them), which is the `cosigen` adapter's arm.
    # Row-level agent_init_params (use_doubao_fc / doubao_fc_parser / think_tag ...)
    # win over the defaults above -- they are our own config surface.
    init_kwargs.update(agent_init_params or {})

    em_client = EnvManagerClient(token=env_manager_token, base_url=env_manager_url)
    async with em_client.env_session(portal_image, portal_version) as env_session:
        direct = is_cn_region()
        portal = PortalConfig(
            endpoint=env_session["base_url"] if direct else env_session["proxy_url"],
            direct_mode=direct,
            keepalive_endpoint=env_session.get("keepalive_url"),
        )
        agent = SimGenRolloutAgent(llm_config, portal, cosigen_state=state, **init_kwargs)
        # Prompt is built AFTER the agent so the tool names come from the live
        # schemas (see _framing) rather than a hardcoded string.
        prompt = _framing(server_prompt, agent.tool_names(), eval_dir)
        if extra_prompt:
            prompt += "\n\n" + extra_prompt
        # ONE run, no continuation prompt. This mirrors the reference 12B coding-RL
        # job (seed.merlin_job.c0a2b5ed145a3b7d) and eval's default condition alike:
        #   * alpha-seed's swalm handler passes agent_run_params={'max_iterations':
        #     agent_max_iterations} and calls run() exactly once -- nothing re-prompts
        #     an agent that stopped;
        #   * eval's run_agent.py defaults --keep-going to False and configs/
        #     default.yaml does not set it, so the container's KEEP_GOING nudge loop
        #     is OFF in the default arm.
        # Stopping therefore ends the episode, which is exactly what the
        # autonomous-operation rule in the prompt tells the model. The context ceiling
        # is the server's job (request_args.overlong_early_stop, on by default in the
        # handler), not a driver-side chunk boundary.
        before_execs = state.executions
        conversations = await agent.run(prompt, max_iterations=int(max_iterations))
        llm_calls = max(1, sum(1 for m in conversations
                               if isinstance(m, dict) and m.get("role") == "assistant"))
        if state.executions == before_execs:
            # Zero simulator turns: record WHAT the model emitted so the cause (prose
            # vs unparseable tool syntax vs empty/length-capped output) is visible in
            # the ray log. 61% of run-v4 episodes ended with zero executions and the
            # raw output was never captured.
            last = next((m for m in reversed(conversations)
                         if isinstance(m, dict) and m.get("role") == "assistant"), None)
            body = agent._extract_text_from_message_content(
                (last or {}).get("content", "")) if last else ""
            print(f"[SIMGEN_NO_TOOLCALL] session={session_id} llm_calls={llm_calls} "
                  f"content_len={len(body)} head={body[:400]!r} "
                  f"tail={body[-200:]!r}", flush=True)

        trajectories = agent._chat_tracker.dump_trajectories_for_alphaseed()

    p = state.profile
    wall = time.time() - episode_t0
    other = wall - p["llm_s"] - p["tool_wall_s"]
    print(f"[SIMGEN_PROFILE] session={session_id} wall_s={wall:.0f} "
          f"llm_s={p['llm_s']:.0f}({100*p['llm_s']/max(wall,1):.0f}%) "
          f"llm_calls={p['llm_calls']} llm_per_call_s={p['llm_s']/max(p['llm_calls'],1):.1f} "
          f"tool_wall_s={p['tool_wall_s']:.0f}({100*p['tool_wall_s']/max(wall,1):.0f}%) "
          f"sim_s={p['sim_s']:.0f} tool_calls={p['tool_calls']} "
          f"other_s={other:.0f}({100*other/max(wall,1):.0f}%)", flush=True)

    scores = list(state.score_history)
    return {
        "success": bool(state.task_completed),
        "final_score": scores[-1] if scores else 0.0,
        "best_score": max(scores) if scores else 0.0,
        "turns": int(state.executions),
        "llm_calls": llm_calls,
        "score_history": scores,
        "trajectories": trajectories,
        "conversations": conversations,
    }
