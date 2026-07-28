"""Text-only RL rollout loop for sim_gen tasks (SimGen-RL training).

The RL twin of ``run_native_cosigen_agent`` (cosigen_agentic_loop.py): the SAME
persistent-session protocol against a (leased) render pod -- execute tool,
Jupyter-cell namespace, checkpoint tree, staged solving -- but driven by the
TRAINING LLM (swalm AlphaSeedStreaming caller, whose chat tracker records the
token ids/logprobs that alpha-seed packs into training data) instead of an
external API model, and with every MULTIMODAL component removed (M13-12B is
text-only):

  * no view tool; image fields are stripped from the execute feedback;
  * no annotator side-calls (out-of-trajectory LLM calls whose main product is
    the image-based scene_diff) -- checkpoint nodes keep the agent's own
    stage_summary text annotation;
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
    load_cosigen_skills,
)

# The harness is iterated continuously (tools get renamed / added: `execute`,
# `checkpoint`, `view` ...). Track it by CLASS, never by tool-name string, and read
# every user-facing name out of the live __tool_schema__ -- a stale hardcoded
# name in the prompt would tell the model to call a tool that does not exist.
try:
    from cosigen_agentic_loop import CoSiGenCheckpointTool
except ImportError:  # harness without the checkpoint tool
    CoSiGenCheckpointTool = None
try:
    from cosigen_agentic_loop import CoSiGenOptimizeTool
except ImportError:  # harness without the optimize tool
    CoSiGenOptimizeTool = None
# NOTE CoSiGenViewTool is deliberately NOT imported: it is the multimodal
# observation tool (it returns image content), and this model is text-only.


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
        # The tools read workspace files (optimize's program/objective paths), so the
        # state needs the portal backlink the eval agent also sets.
        self._cosigen_state.portal = self.portal_client
        self.tools.append(TextOnlyExecuteTool(self._cosigen_state))
        # Every NON-multimodal harness tool the eval agent gets, the RL agent gets
        # too (checkpoint = the save-a-program flow, optimize = parallel constant
        # search). Only the image tool (view) is withheld: this model is text-only.
        if CoSiGenCheckpointTool is not None:
            self.tools.append(CoSiGenCheckpointTool(self._cosigen_state))
        # Gated exactly as the eval agent gates it, on the pod advertising the tuning
        # section. Without this the cosigen-parameter-optimize SKILL (loaded whenever
        # the pod has that section) would teach a tool that does not exist.
        if CoSiGenOptimizeTool is not None and getattr(self._cosigen_state, "has_opt", False):
            self.tools.append(CoSiGenOptimizeTool(self._cosigen_state))

    def tool_names(self) -> dict:
        """Live tool names, read from the schemas (never hardcoded)."""
        names = {}
        for t in self.tools:
            schema = getattr(t, "__tool_schema__", None)
            if not schema:
                continue
            if isinstance(t, TextOnlyExecuteTool):
                names["execute"] = _tool_name(t)
            elif CoSiGenCheckpointTool is not None and isinstance(t, CoSiGenCheckpointTool):
                names["checkpoint"] = _tool_name(t)
            elif CoSiGenOptimizeTool is not None and isinstance(t, CoSiGenOptimizeTool):
                names["optimize"] = _tool_name(t)
        return names


def _framing(prompt: str, names: dict) -> str:
    """Driver-side framing appended to the server's task prompt. Mirrors the eval
    framing (run_native_cosigen_agent) with the vision affordances replaced by an
    explicit text-only contract; the server prompt stays authoritative about which
    harness facilities (checkpointing) exist on the pod. `names` carries the LIVE
    tool names (harness renames must never desync the prompt from the schema)."""
    has_ckpt = "checkpoint(" in prompt
    ex = names.get("execute", "execute")
    ckpt_bit = (f" Use `{names['checkpoint']}` to save a working program as a checkpoint."
                if names.get("checkpoint") else "")
    opt_bit = (f" Use `{names['optimize']}` to tune a program's numeric constants "
               "instead of re-running it with hand-nudged values."
               if names.get("optimize") else "")
    toolkit_bits = "move_to" + (", checkpoint/goto" if has_ckpt else "")
    uses = "control" + (", backtracking" if has_ckpt else "")
    skill_hint = ("Invoke the checkpoint skill when appropriate. " if has_ckpt else "")
    return prompt + (
        f"\n\nYou are running as a native swalm coding agent. Use the `{ex}` tool to "
        "interact with the simulator. Write your control code in the style of a "
        "standalone Isaac Lab script where useful -- `env`, `api`, `torch` and isaaclab "
        "imports are live in your namespace (see RAW SIMULATOR ACCESS in the API doc) -- "
        f"while using the toolkit ({toolkit_bits}) for {uses}.{ckpt_bit}{opt_bit} {skill_hint}"
        "This session is TEXT-ONLY: no images can be shown to you (look()/render "
        "captures are unavailable), so verify every stage through printed measurements, "
        "object poses and the scene summary instead. "
        f"ALWAYS act by calling the `{ex}` tool -- a reply that contains no tool call "
        "ends your turn and wastes the step. "
        f"Do not stop until `{ex}` reports success=true."
    )


async def run_rl_episode(
    *,
    server: str,
    llm_config,
    agent_init_params: dict | None = None,
    artifact_dir: str | None = None,
    session_id: str | None = None,
    sim_max_steps: int = 100_000,
    token_budget: int = 110_000,
    max_iterations: int = 64,
    chunk_iterations: int = 8,
    env_manager_url: str | None = None,
    env_manager_token: str | None = None,
    portal_version: str = "default",
    portal_image: str = "hub.byted.org/base/python:3.11",
    extra_prompt: str = "",
) -> dict:
    """One training episode on a (leased) render pod. Returns
    {success, final_score, best_score, turns, llm_calls, trajectories, conversations}.

    `max_iterations` is the TOTAL LLM-call budget for the episode, consumed in
    chunks of `chunk_iterations` with the eval loop's continuation prompt between
    chunks; `token_budget` caps the conversation size (the context IS the budget:
    condensation is disabled so the trained context never silently outlives it).
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
    # The server prompt is authoritative about which facilities this pod has; mirror
    # the eval loop's tests so the tool list, the skills and the framing agree.
    state.has_opt = "== Tuning the numbers" in server_prompt

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
        # The episode's own token budget (checked between chunks below) is the
        # only context guard we need; overlong trajectories are handled by the
        # trainer's swalm_*_overlong_train_stratagy=learn_all.
        "max_context_length": 0,
    }
    # `skills` exists only on newer swalm-core ClaudeCodeAgents; the pinned RL
    # stack's agent has no skill system -- pass it only where supported.
    import inspect
    if "skills" in inspect.signature(ClaudeCodeAgent.__init__).parameters:
        try:
            init_kwargs["skills"] = load_cosigen_skills(
                eval_dir,
                has_ckpt="list_checkpoints()" in server_prompt,
                has_opt="== Tuning the numbers" in server_prompt)
        except (OSError, KeyError) as exc:
            print(f"[simgen_rl] skills unavailable ({exc!r}); continuing without",
                  flush=True)
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
        prompt = _framing(server_prompt, agent.tool_names())
        if extra_prompt:
            prompt += "\n\n" + extra_prompt

        conversations = None
        llm_calls = 0
        while not state.task_completed and llm_calls < int(max_iterations):
            if agent.total_tokens_used >= int(token_budget):
                print(f"[simgen_rl] token budget reached ({agent.total_tokens_used} >= "
                      f"{token_budget}); ending episode", flush=True)
                break
            step = min(int(chunk_iterations), int(max_iterations) - llm_calls)
            has_ckpt = "checkpoint(" in prompt
            continuation = prompt if conversations is None else (
                "continue -- simulator success is still false. Review the staged plan"
                + (", checkpoint tree" if has_ckpt else "")
                + " and outcomes. Change strategy substantially where it keeps failing."
            )
            before_msgs = len(conversations or [])
            before_execs = state.executions
            conversations = await agent.run(
                continuation, max_iterations=step, conversations=conversations)
            # ClaudeCodeAgent ENDS a chunk as soon as a response carries no parseable
            # tool call ("no tool calls, agent finished"), so a chunk often uses far
            # fewer than `step` LLM calls. Charging the full `step` burned the whole
            # episode budget in 15 chunks (run v4). Count the assistant messages the
            # chunk actually appended instead.
            used = max(1, sum(1 for m in conversations[before_msgs:]
                              if isinstance(m, dict) and m.get("role") == "assistant"))
            llm_calls += used
            if state.executions == before_execs:
                # No simulator turn this chunk: record WHAT the model emitted so the
                # cause (prose vs unparseable tool syntax vs empty/length-capped
                # output) is visible in the ray log. 61% of run-v4 episodes ended
                # with zero executions and the raw output was never captured.
                last = next((m for m in reversed(conversations)
                             if isinstance(m, dict) and m.get("role") == "assistant"), None)
                body = agent._extract_text_from_message_content(
                    (last or {}).get("content", "")) if last else ""
                print(f"[SIMGEN_NO_TOOLCALL] session={session_id} chunk_llm_calls={used} "
                      f"total={llm_calls} content_len={len(body)} "
                      f"head={body[:400]!r} tail={body[-200:]!r}", flush=True)

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
