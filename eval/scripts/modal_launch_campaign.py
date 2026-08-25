#!/usr/bin/env python3
"""The 13-task x 5-model campaign: 65 launchers in parallel on Modal, one RunPod 4090 each.

Modal runs the LAUNCHER/babysitter containers (run_agent_runpod.py — the docker layer of the
harness); each launcher creates its own RunPod pod where agent + relay + Isaac run, enforces
the budget, mirrors artifacts + the trajectory back, and terminates the pod. Artifacts land
in the `cosigen-runs` Modal volume under /runs/<task>_<model_short>/.

    modal run eval/scripts/modal_launch_campaign.py                        # all 65
    modal run eval/scripts/modal_launch_campaign.py --only coffee:claude-opus-5
    modal run eval/scripts/modal_launch_campaign.py --budget-min 8 --only coffee:claude-opus-5

Artifacts back to this machine any time (works while runs are live):

    modal volume get cosigen-runs / eval_result/campaign/
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = "/repo"                     # the CoSiGen tree baked into the image
RUNS = "/runs"                     # durable artifacts (modal volume)
BUDGET_MIN_DEFAULT = 240.0         # user directive 2026-08-14: 4 hours per agent

TASKS = {  # experiment name -> built experiment dir (relative to REPO)
    "allen_bolt": "experiments/allen_bolt_notools",
    "bulb": "experiments/bulb_notools",
    "ikea_table": "experiments/ikea_table_notools",
    "nut_thread": "experiments/nut_thread_notools",
    "pc_gpu": "experiments/pc_gpu_notools",
    "pc_gpu_ram": "experiments/pc_gpu_ram_notools",
    "pc_motherboard": "experiments/pc_motherboard_notools",
    "pc_ram": "experiments/pc_ram_notools",
    "pen_holder": "experiments/pen_holder_notools",
    "tool_packing": "experiments/tool_packing_notools",
    "spatula": "experiments/spatula_notools",
    "syringe": "experiments/syringe_notools",
    "coffee": "experiments/coffee_notools",
    # Cross-embodiment worlds (6 tasks x {gen3n7_panda, xarm7+panda-hand}, osc; built by
    # eval/scripts/build_all_emb.sh).
    **{f"{t}_{e}": f"experiments/{t}_{e}_notools"
       for t in ("allen_bolt", "bulb", "pc_motherboard", "pc_ram", "tool_packing", "spatula")
       for e in ("gen3n7", "xarm7")},
}

MODELS = {  # model id -> agent CLI
    "claude-opus-5": "claude",
    "claude-opus-4-8": "claude",
    "claude-fable-5": "claude",
    "gpt-5.6-sol": "codex",
    "gpt-5.6-terra": "codex",
}


def _short(model: str) -> str:
    return model.replace("claude-", "").replace(".", "_").replace("-", "_")


app = modal.App("cosigen-campaign")

runs_volume = modal.Volume.from_name("cosigen-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-client", "rsync", "tar")
    .pip_install("pyyaml")
    # The repo pieces the launcher needs: the harness (eval/), the relay (sim_gen/) and the
    # built experiment worlds. .venv/.git/env_newton are excluded by .containerignore-style
    # ignore patterns below.
    .add_local_dir("eval", remote_path=f"{REPO}/eval")
    .add_local_dir("sim_gen/super_relay", remote_path=f"{REPO}/sim_gen/super_relay",
                   ignore=["logs/**"])
    # runs/ excluded: local sanity-run artifacts are not world content and would ship
    # into every launcher container otherwise.
    .add_local_dir("experiments", remote_path=f"{REPO}/experiments",
                   ignore=["**/runs/**"])
    # The campaign SSH keypair: the private key stays inside this private image; the public
    # key is what every pod is created with.
    .add_local_file("/Users/bytedance/.ssh/cosigen_campaign",
                    remote_path="/root/.ssh/cosigen_campaign")
    .add_local_file("/Users/bytedance/.ssh/cosigen_campaign.pub",
                    remote_path="/root/.ssh/cosigen_campaign.pub")
    # Oracle Franka solutions (flattened), the hint material for the generalization arms.
    .add_local_dir("CoSiGen_Solutions_flat", remote_path=f"{REPO}/hints")
)


@app.function(
    image=image,
    volumes={RUNS: runs_volume},
    timeout=6 * 60 * 60,        # 4 h budget + pod setup + mirrors + teardown headroom
    retries=0,                  # a failed run is a result, not something to silently redo
    max_containers=70,
)
def run_one(task: str, model: str, budget_min: float = BUDGET_MIN_DEFAULT,
            attempt: str = "", config: str = "no_tools", gpu_count: int = 1,
            gpt_via_tunnel: bool = False, hint_files: str = "",
            hint_note: str = "") -> str:
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    agent = MODELS[model]
    exp = Path(REPO) / TASKS[task]
    suffix = f"_r{attempt}" if attempt else ""
    run_name = f"{_short(model)}{suffix}"
    label = f"{task}_{_short(model)}{suffix}"

    subprocess.run(["chmod", "600", "/root/.ssh/cosigen_campaign"], check=True)
    cmd = [
        sys.executable, f"{REPO}/eval/scripts/run_agent_runpod.py", str(exp),
        # The CONDITION of the run; the built _notools worlds host any config whose features
        # match (default and no_tools block the identical set — only control_mode), which the
        # runner itself re-checks.
        "--config", config,
        "--agent", agent,
        "--model", model,
        # Pinned at the relay, not passed through: Claude Code restarts a compacted
        # conversation on its DEFAULT model id, silently mixing models within one arm
        # (measured 2026-08-15: a fable-5 run spent 44/71 requests on claude-opus-4-8).
        # Rewriting at the relay is what makes the whole run one model (PR #34's lesson).
        "--force-model", model,
        "--budget-min", str(budget_min),
        "--keep-going",
        "--auto-submit-min", "30",
        "--run", run_name,
        "--gpu-count", str(gpu_count),
        "--ssh-key", "/root/.ssh/cosigen_campaign",
    ]
    if gpt_via_tunnel and MODELS[model] == "codex":
        # GPT via AIDP through the laptop bridge (see run_agent_runpod.py TUNNEL_AIDP_URL);
        # requires the forwarder + tunnel daemon running on the laptop for the whole campaign.
        cmd.append("--gpt-via-tunnel")
    for hf in (hint_files.split(",") if hint_files else []):
        cmd += ["--hint-file", hf]
    if hint_note:
        cmd += ["--hint-note", hint_note]
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True)

    # Durable copy: the whole run dir (task/, workspace/, submissions/, trajectory mirror,
    # container.log, run.json, agent_home.tgz) into the volume, live even if this container
    # is later lost.
    run_dir = exp / "runs" / run_name
    dest = Path(RUNS) / label
    if run_dir.is_dir():
        shutil.copytree(run_dir, dest, dirs_exist_ok=True)
        runs_volume.commit()
        print(f"[{label}] artifacts committed to volume at {dest}", flush=True)
    else:
        print(f"[{label}] NO RUN DIR ({run_dir}) — launcher exited rc={proc.returncode}",
              flush=True)
    return f"{label}: launcher rc={proc.returncode}"


# ----- cross-task generalization campaign (2026-08-17) ---------------------------------------
# 12 (target, source) pairs x {claude-opus-5, gpt-5.6-sol} x 3 seeds, plus 2 extra no-hint
# baseline seeds per target x model (seed 1 = the original campaign run). The hint is the
# SOURCE task's oracle Franka solution file(s) only — no conversation carryover.
GEN_PAIRS = [  # (target, source, near/far)
    ("bulb", "nut_thread", "near"),
    ("bulb", "tool_packing", "far"),
    ("allen_bolt", "nut_thread", "near"),
    ("allen_bolt", "pen_holder", "far"),
    ("pc_motherboard", "allen_bolt", "near"),
    ("pc_motherboard", "spatula", "far"),
    ("pc_ram", "pc_gpu", "near"),
    ("pc_ram", "bulb", "far"),
    ("tool_packing", "pen_holder", "near"),
    ("tool_packing", "nut_thread", "far"),
    ("spatula", "allen_bolt", "near"),
    ("spatula", "pc_gpu", "far"),
]
GEN_MODELS = ("claude-opus-5", "gpt-5.6-sol")
# One-line task descriptions, quoted in the hint note so the agent knows what the reference
# solves without having to reverse-engineer it.
TASK_DESC = {
    "nut_thread": "pick up an M16 nut and thread it fully down a fixed bolt",
    "tool_packing": "open a chest's drawers and stow three desk tools into their trays",
    "pen_holder": "pick up lying pencils and insert them tip-up into a cup holder",
    "allen_bolt": "erect an Allen key, seat it in a hex bolt's socket and ratchet the bolt down",
    "spatula": "flip a bread slice in a pan with a spatula and serve it onto a plate",
    "pc_gpu": "pick a GPU card from its holder and seat it in a PC case's PCIe slot",
    "bulb": "pick up a light bulb and screw it into a threaded socket",
    # same-task references for the cross-embodiment campaign (not cross-task sources)
    "pc_motherboard": "pick up an Allen key and ratchet the motherboard's bolts down",
    "pc_ram": "pick RAM sticks from their holders and seat them in a PC's DIMM slots",
}
HINTS_DIR = f"{REPO}/hints"
SOURCE_FILES = {  # source task -> hint file(s) in the flattened oracle-solutions dir
    task: [f"{HINTS_DIR}/{task}__franka__osc__solve.py"] for task in TASK_DESC
}
SOURCE_FILES["spatula"].append(f"{HINTS_DIR}/spatula__franka__osc__franka_session.py")


def _hint_note(source: str) -> str:
    files = SOURCE_FILES[source]
    note = (f"A complete working solution to a different task is provided as a reference: "
            f"/task/hints/{Path(files[0]).name} solves the '{source}' task "
            f"({TASK_DESC[source]}) with the same robot and control interface. It does not "
            f"solve the current task, but you may consult it for control patterns, API "
            f"usage, and conventions.")
    if len(files) > 1:
        note += (f" Its helper module {', '.join(Path(f).name for f in files[1:])} is "
                 f"provided alongside it.")
    return note


@app.local_entrypoint()
def gen(seeds: str = "1,2,3", baseline_seeds: str = "2,3", only_model: str = "",
        budget_min: float = BUDGET_MIN_DEFAULT, only_labels: str = ""):
    """The cross-task generalization campaign: 72 hint runs + 24 no-hint baseline runs.

    modal run --detach eval/scripts/modal_launch_campaign.py::gen
    `--only-labels a,b,...` restricts to those run labels (wave-2 relaunches of jobs whose
    first launcher failed; the label set comes from the manifest-vs-live-pods diff).
    """
    jobs = []   # (task, model, attempt, hint_files, hint_note, meta)
    for target, source, kind in GEN_PAIRS:
        for model in GEN_MODELS:
            if only_model and model != only_model:
                continue
            for s in [int(x) for x in seeds.split(",") if x]:
                attempt = f"gen_{source}_s{s}"
                jobs.append((target, model, attempt, ",".join(SOURCE_FILES[source]),
                             _hint_note(source),
                             {"arm": "gen", "target": target, "source": source,
                              "type": kind, "seed": s}))
    for target in sorted({t for t, _, _ in GEN_PAIRS}):
        for model in GEN_MODELS:
            if only_model and model != only_model:
                continue
            for s in [int(x) for x in baseline_seeds.split(",") if x]:
                attempt = f"base_s{s}"
                jobs.append((target, model, attempt, "", "",
                             {"arm": "baseline", "target": target, "source": None,
                              "type": None, "seed": s}))

    if only_labels:
        wanted = {x.strip() for x in only_labels.split(",") if x.strip()}
        jobs = [j for j in jobs
                if f"{j[0]}_{_short(j[1])}_r{j[2]}" in wanted]
        unknown = wanted - {f"{j[0]}_{_short(j[1])}_r{j[2]}" for j in jobs}
        if unknown:
            raise SystemExit(f"--only-labels entries not in this campaign: {sorted(unknown)}")

    manifest = [{"label": f"{task}_{_short(model)}_r{attempt}", "task": task,
                 "model": model, **meta}
                for task, model, attempt, _, _, meta in jobs]
    out = Path("results/gen_campaign_manifest.json")
    out.parent.mkdir(exist_ok=True)
    if not only_labels:  # a wave-2 relaunch must not clobber the full manifest
        out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"launching {len(jobs)} generalization jobs (budget {budget_min} min each); "
          f"manifest -> {out}", flush=True)

    calls = [run_one.spawn(task, model, budget_min, attempt, "no_tools", 1,
                           False,  # codex via OpenAI direct (tunneling disallowed,
                                   # user directive 2026-08-17)
                           hint_files, hint_note)
             for task, model, attempt, hint_files, hint_note, _ in jobs]
    for (task, model, attempt, *_), call in zip(jobs, calls):
        try:
            print(call.get(), flush=True)
        except Exception as exc:  # noqa: BLE001 -- one dead launcher must not kill the rest
            print(f"{task}:{model}:{attempt} launcher FAILED: {exc!r}", flush=True)


# ----- cross-embodiment campaign (2026-08-18) ------------------------------------------------
# 6 target tasks x {gen3n7_panda, xarm7 (panda hand)} x 2 models x 3 seeds = 72 jobs. The hint
# is the SAME task's oracle Franka solution — the transfer axis is the embodiment.
EMB_TASKS = ("allen_bolt", "bulb", "pc_motherboard", "pc_ram", "tool_packing", "spatula")
EMB_ARMS = {"gen3n7": "a Kinova Gen3 7-DOF arm carrying the Franka panda hand",
            "xarm7": "a UFACTORY xArm7 carrying the Franka panda hand"}


def _emb_hint_note(task: str, emb: str) -> str:
    files = SOURCE_FILES[task]
    note = (f"A complete working solution to this same task is provided as a reference: "
            f"/task/hints/{Path(files[0]).name} solves the '{task}' task with a Franka "
            f"Panda arm, while this run's robot is {EMB_ARMS[emb]}. The embodiment differs "
            f"(kinematics, joint count, reach, control gains), so the reference will not "
            f"work as-is, but its task strategy, phase structure, and scene reasoning may "
            f"be useful.")
    if len(files) > 1:
        note += (f" Its helper module {', '.join(Path(f).name for f in files[1:])} is "
                 f"provided alongside it.")
    return note


@app.local_entrypoint()
def emb(seeds: str = "1,2,3", only_model: str = "", only_labels: str = "",
        budget_min: float = BUDGET_MIN_DEFAULT):
    """The cross-embodiment campaign: 72 hint runs (same-task Franka solution as reference).

    modal run --detach eval/scripts/modal_launch_campaign.py::emb
    """
    jobs = []
    for task in EMB_TASKS:
        for emb_name in EMB_ARMS:
            for model in GEN_MODELS:
                if only_model and model != only_model:
                    continue
                for s in [int(x) for x in seeds.split(",") if x]:
                    jobs.append((f"{task}_{emb_name}", model, f"emb_s{s}",
                                 ",".join(SOURCE_FILES[task]),
                                 _emb_hint_note(task, emb_name),
                                 {"arm": "emb", "target": task, "embodiment": emb_name,
                                  "seed": s}))
    if only_labels:
        wanted = {x.strip() for x in only_labels.split(",") if x.strip()}
        jobs = [j for j in jobs if f"{j[0]}_{_short(j[1])}_r{j[2]}" in wanted]
        unknown = wanted - {f"{j[0]}_{_short(j[1])}_r{j[2]}" for j in jobs}
        if unknown:
            raise SystemExit(f"--only-labels entries not in this campaign: {sorted(unknown)}")

    manifest = [{"label": f"{task}_{_short(model)}_r{attempt}", "task": task,
                 "model": model, **meta}
                for task, model, attempt, _, _, meta in jobs]
    out = Path("results/emb_campaign_manifest.json")
    out.parent.mkdir(exist_ok=True)
    if not only_labels:
        out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"launching {len(jobs)} cross-embodiment jobs (budget {budget_min} min each); "
          f"manifest -> {out}", flush=True)

    calls = [run_one.spawn(task, model, budget_min, attempt, "no_tools", 1,
                           False,  # codex via OpenAI direct
                           hint_files, hint_note)
             for task, model, attempt, hint_files, hint_note, _ in jobs]
    for (task, model, attempt, *_), call in zip(jobs, calls):
        try:
            print(call.get(), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{task}:{model}:{attempt} launcher FAILED: {exc!r}", flush=True)


@app.local_entrypoint()
def main(only: str = "", budget_min: float = BUDGET_MIN_DEFAULT, attempt: str = "",
         config: str = "no_tools", gpu_count: int = 1, gpt_via_tunnel: bool = False,
         hint_source: str = ""):
    """`--only task:model[,task:model...]` restricts the grid; default is all 65.
    `--attempt N` suffixes run names/labels with _rN (relaunches never collide with the
    artifacts of the attempt they replace)."""
    grid: list[tuple[str, str]] = []
    if only:
        for pair in only.split(","):
            task, _, model = pair.partition(":")
            if task not in TASKS or model not in MODELS:
                raise SystemExit(f"unknown pair {pair!r} (task in {list(TASKS)}, "
                                 f"model in {list(MODELS)})")
            grid.append((task, model))
    else:
        grid = [(t, m) for t in TASKS for m in MODELS]

    print(f"launching {len(grid)} runs, budget {budget_min} min each"
          f"{' (attempt ' + attempt + ')' if attempt else ''}", flush=True)
    hint_files = ",".join(SOURCE_FILES[hint_source]) if hint_source else ""
    hint_note = _hint_note(hint_source) if hint_source else ""
    calls = [run_one.spawn(task, model, budget_min, attempt, config, gpu_count,
                           gpt_via_tunnel, hint_files, hint_note)
             for task, model in grid]
    for (task, model), call in zip(grid, calls):
        # A failed launcher must not kill this entrypoint: with an ephemeral app, the local
        # process dying takes every OTHER launcher down with it, orphaning their pods
        # (measured 2026-08-16: one capacity failure -> all babysitters dead, pods overran
        # their budgets until manually stopped).
        try:
            print(call.get(), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{task}:{model} launcher FAILED: {exc!r}", flush=True)
