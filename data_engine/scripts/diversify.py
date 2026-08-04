"""Launch a diversify authoring session on a data_gen campaign.

    python data_engine/scripts/diversify.py <gen_root> <condition.yaml> \\
        [key=value ...] [--dry] [--host]

One yaml = one session condition (see data_engine/configs/): level, prompt
modules, offered cli commands, scene/strategy context, model/image/gpu. Trailing
key=value args override any field, hydra-style (values yaml-parsed):

    diversify.py <gen> configs/scene_default.yaml model=claude-sonnet-5 \\
        scene=scene_1 prompts='[scene, failure_mining]'

The launcher creates NO cell — it assembles the instructions (contract + the
condition's prompt modules + campaign facts) into the session ledger
`<gen_root>/.agent/<ts>_<level>/` (with the RESOLVED condition.yaml beside it),
then launches the agent fenced in docker. The agent's world has three handles:
/workspace — its writable home, the campaign (mounted rw at its true depth
under /repo, so relative symlinks keep resolving); /reference — the eval run
this campaign multiplies (read-only): the task and the agent trace of how the
solve was built; /repo — the whole repo, read-only, optional background. GPU
passthrough; one code path for supervised testbeds and future headless
batches. The agent's first authoring act is creating its first cell with
create_cell.py. --host is the UNFENCED escape hatch for engine development.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CRED_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
DEFAULTS = {"scene": "scene_0", "strategy": "strategy_0", "prompts": [], "cli": [],
            "agent": "claude", "model": "claude-opus-5",
            "image": "rb-l1-agent:2.1.216", "gpu": "0", "budget_min": 240}


def load_condition(path: Path, overrides: list[str]) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(path.read_text())}
    for tok in overrides:
        key, eq, val = tok.partition("=")
        if not eq:
            raise SystemExit(f"override '{tok}' is not key=value")
        if key not in cfg:
            raise SystemExit(f"unknown config key '{key}' (have: {sorted(cfg)})")
        cfg[key] = yaml.safe_load(val)
    if "level" not in cfg or cfg["level"] not in ("scene", "strategy", "phase"):
        raise SystemExit("condition needs level: scene|strategy|phase")
    return cfg


def assemble(session: Path, gen_root: Path, cfg: dict, repo_as: Path, gen_as: Path) -> Path:
    """Write instructions.md + the resolved condition.yaml into the session ledger.
    Paths render as the SESSION sees them: `repo_as` = the repo root (/repo in
    a container), `gen_as` = the campaign (/workspace). One global namespace
    everywhere: the agent sees the campaign's real cell names."""
    run_as = (Path("/reference") if repo_as != REPO_ROOT
              else gen_root.parents[1])  # the eval run this campaign multiplies
    base = cfg["strategy"] if cfg["level"] == "phase" else cfg["scene"]

    parts = [(ROOT / "agent" / "prompts" / "_contract.md").read_text()]
    for name in cfg["prompts"]:
        parts.append("\n\n" + (ROOT / "agent" / "prompts" / f"{name}.md")
                     .read_text().replace("{base}", base).strip())
    parts += [
        "\n\n## This session\n",
        f"- your workspace: {gen_as}  (the campaign; your cwd; work only in here)",
        f"- your reference: {run_as}  (read-only — the eval run this campaign multiplies)",
        f"- the repo: {repo_as}  (read-only — optional background)",
        f"- your start point is READ-ONLY — copy it with create_cell, never edit it in place",
        f"- level: {cfg['level']} — start from: {cfg['scene']}"
        + (f"/{cfg['strategy']}" if cfg["level"] in ("strategy", "phase") else ""),
        "- your cli, on PATH (create each cell yourself, one per proposed variant):",
        *[f"  - `create_cell [--count N]` — new cell(s) from your start point" if t == "create_cell"
          else f"  - `generate --headless {gen_as} --scene … [--strategy …] --num_envs 4` — test-launch a cell" if t == "generate"
          else f"  - `{t}`" for t in cfg["cli"]],
        "\n### gen.yaml\n```yaml",
        (gen_root / "gen.yaml").read_text().rstrip(),
        "```",
    ]
    metas = [f"- {m.parent.relative_to(gen_root)}: {m.read_text().strip()}"
             for m in sorted(gen_root.glob("scenes/*/meta.json"))
             + sorted(gen_root.glob("scenes/*/strategies/*/meta.json"))]
    if metas:
        parts += ["\n### current yields (baselines)\n"] + metas

    (session / "condition.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = session / "instructions.md"
    out.write_text(parts[0] + "\n".join(parts[1:]) + "\n")
    return out


def start_point_ro(gen_root: Path, cfg: dict) -> list[Path]:
    """The base material a session starts from, mounted read-only INSIDE the rw
    campaign — create_cell is the only path from reference to workable copy.
    Granularity follows the level because new cells nest inside bases: a new
    strategy is created inside the base scene, a new phase inside the strategy."""
    scene = gen_root / "scenes" / cfg["scene"]
    strategy = scene / "strategies" / cfg["strategy"]
    if cfg["level"] == "scene":
        return [scene]
    if cfg["level"] == "strategy":
        return [scene / "scene", scene / "grader", strategy]
    return [scene / "scene", scene / "grader", strategy / "solve.py"]


def container_cmd(gen_root: Path, instructions: Path, cfg: dict) -> list[str]:
    """docker run: repo read-only at /repo, the campaign rw over it at its true
    depth (relative symlinks keep resolving); /workspace links to the campaign,
    /reference to the eval run it multiplies."""
    rel = gen_root.relative_to(REPO_ROOT)
    run_rel = gen_root.parents[1].relative_to(REPO_ROOT)
    instr_in = Path("/workspace") / instructions.relative_to(gen_root)
    cmd = [
        "docker", "run", "-it", "--rm",
        "--name", f"dgen_{gen_root.name}_{instructions.parent.name}",
        "--device", f"nvidia.com/gpu={cfg['gpu']}", "--shm-size", "2g",
        "-v", f"{REPO_ROOT}:/repo:ro",
        "-v", f"{gen_root}:/repo/{rel}",          # the deeper rw bind wins over the ro repo
        *[a for pth in start_point_ro(gen_root, cfg) if pth.exists()
          for a in ("-v", f"{pth}:/repo/{pth.relative_to(REPO_ROOT)}:ro")],
        "-v", "rb-ovcache:/ovcache",
        "-e", "PYTHONPATH=/repo",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "DGEN_ROOT=/workspace",
        "-e", f"DGEN_LEVEL={cfg['level']}",
        "-e", f"DGEN_SCENE={cfg['scene']}",
        "-e", f"DGEN_STRATEGY={cfg['strategy']}",
    ]
    for var in CRED_VARS:
        if os.environ.get(var):
            cmd += ["-e", var]
    agent_cmd = " ".join([cfg["agent"], "--model", cfg["model"],
                          f"'Read {instr_in} and begin.'"])
    # entry: replace the image's stock empty /workspace with the campaign link,
    # install each DECLARED tool as a PATH command, then hand over to the agent
    shims = " && ".join(
        f"printf '#!/bin/bash\\nexec python /repo/data_engine/agent/cli/{t}.py \"$@\"\\n'"
        f" > /usr/local/bin/{t} && chmod +x /usr/local/bin/{t}" for t in cfg["cli"])
    setup = (f"rmdir /workspace 2>/dev/null; ln -sT /repo/{rel} /workspace"
             f" && ln -sT /repo/{run_rel} /reference")
    # setup runs as root (shims in /usr/local/bin, the / link); the agent itself
    # drops to the image's non-root `agent` user (uid 1000 = the host user, so
    # everything the session writes is host-owned)
    budget_s = int(float(cfg["budget_min"]) * 60)  # wall-clock cap: a dead session frees the GPU
    entry = " && ".join(x for x in (setup, shims,
                                    f"cd /workspace && exec timeout {budget_s} "
                                    f"runuser -u agent -- {agent_cmd}") if x)
    cmd += [cfg["image"], "bash", "-c", entry]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="launch a diversify authoring session",
                                 usage="diversify.py gen_root condition.yaml [key=value ...] [--dry] [--host]")
    ap.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
    ap.add_argument("config", help="condition yaml (see data_engine/configs/)")
    ap.add_argument("overrides", nargs="*", help="hydra-style key=value overrides")
    ap.add_argument("--host", action="store_true",
                    help="UNFENCED host session — engine-development escape hatch only")
    ap.add_argument("--dry", action="store_true", help="assemble only; print what would launch")
    args = ap.parse_args()

    gen_root = Path(args.gen_root).resolve()
    if not (gen_root / "gen.yaml").is_file():
        raise SystemExit(f"not a campaign (no gen.yaml): {gen_root}")
    if not args.host and REPO_ROOT not in gen_root.parents:
        raise SystemExit(f"container sessions need the campaign under the repo tree ({REPO_ROOT})")

    cfg = load_condition(Path(args.config), args.overrides)
    session = gen_root / ".agent" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg['level']}"
    session.mkdir(parents=True)
    repo_as = REPO_ROOT if args.host else Path("/repo")
    gen_as = gen_root if args.host else Path("/workspace")
    instructions = assemble(session, gen_root, cfg, repo_as, gen_as)
    print(f"session: {session}\ninstructions: {instructions}")

    if not args.host:
        cmd = container_cmd(gen_root, instructions, cfg)
        if args.dry:
            import shlex
            print("would launch:\n  " + shlex.join(cmd))
            return
        if not any(os.environ.get(v) for v in CRED_VARS):
            raise SystemExit(f"no credential in env — set one of {CRED_VARS}")
        os.execvp("docker", cmd)
    if args.dry:
        return
    os.chdir(gen_root)
    os.execvp(cfg["agent"], [cfg["agent"], "--model", cfg["model"],
                             f"Read {instructions} and begin."])


if __name__ == "__main__":
    main()
