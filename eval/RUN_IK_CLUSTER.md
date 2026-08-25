# Running the diff_ik eval campaign on a cluster (13 tasks × gpt-5.6-sol × tools)

What the RunPod launcher (`eval/scripts/run_agent_runpod.py`) does per run, so the same
flow can be reproduced on any GPU host. One run = one GPU (RTX 4090-class; rendering GPU
required — Isaac Sim/PhysX), 240 min budget.

## 0. Environment (once per cluster image)

Python 3.11 venv with: torch 2.7 (cu128), `isaacsim==5.1` (pip, EULA:
`OMNI_KIT_ACCEPT_EULA=YES`), `isaaclab==2.3.2`, this repo on `PYTHONPATH`.
`eval/scripts/modal_grade_env.py` has a working pip recipe to copy. Agent CLI: `codex`
(npm), plus node. The relay venv needs `aiohttp`.

## 1. Build the 13 experiment worlds (once, needs a GPU; ~4 min each)

The stage specs live in `eval/scripts/ik_preflight_build_runpod.py` (`TASKS`). For each:

    OMNI_KIT_ACCEPT_EULA=YES python eval/scripts/build_env.py \
        --name <task>_ik --stage <spec> --config default
    # e.g. --name bulb_ik --stage assembly.bulb:franka:diff_ik
    #      ikea_table + syringe use :bimanual_franka:diff_ik

Run the preflight first (same file, `PREFLIGHT` string): diff_ik must track to <2 cm and
COSIGEN_STATELOG must produce loadable snapshots.

## 2. One agent run (what the launcher does, host-native version)

1. Stage the built world read-only for the agent: `experiments/<task>_ik/stages/*/bench`
   -> `/bench`, the task folder -> `/task` (root-owned, read-only). Create empty
   `/workspace`, `/submissions` (agent-writable).
2. Start the relay (per-run, own port):
   `sim_gen/super_relay/server.py --port 8118 --log-dir <run>/trajlog
    --upstream-base https://aidp.bytedance.net/... `
   ON A CORP-NETWORK CLUSTER THE LAPTOP BRIDGE/SPOOL IS NOT NEEDED — point the relay's
   upstream at AIDP directly (`chat_bridge.py` already speaks the chat shape; strip
   `reasoning_effort` on 400 as in `laptop_aidp_bridge.py`).
3. Write the codex config exactly as `run_agent_runpod.py::write_codex_config` does
   (model provider `relay`, wire_api `chat`, base_url `http://127.0.0.1:8118/v1`,
   model `gpt-5.6-sol`; the relay force-rewrites the model id on every request).
4. Run `eval/docker/entrypoints/agent-entry.sh` as the container/host entry with env:
   `MODEL=gpt-5.6-sol`, `KEEP_GOING=1`, `BUDGET_MIN=240`, `AUTO_SUBMIT_MIN=30`,
   `COSIGEN_STATELOG=/workspace/.statelog`,
   `SUCCESS_CHECK="timeout -k 30 3900 <venv>/bin/python /opt/verify_solution.py
     --preset <preset> --solution /workspace/solution"`
   (preset from the world's MANIFEST, e.g. `assembly.bulb.franka.diff_ik`).
   The entry loops agent legs until the budget ends or SUCCESS_CHECK returns rc 0.
5. Artifacts to keep per run: `/workspace` (incl. `.statelog/` — reached-state grading
   source of record), `/submissions`, relay `trajlog/` (raw_requests.jsonl), the entry's
   stdout (container.log), and a `run.json` (copy the fields written at the end of
   `run_agent_runpod.py`).

## 3. Already done (do not redo)

syringe_ik was SOLVED by gpt-5.6-sol on RunPod before the capacity outage
(31 min, doses 0.33/0.33/0.33, parked, official verify passed). Artifacts:
`experiments/syringe_ik/runs/gpt_5_6_sol_ik1` on the laptop. The other 12 are unrun.
