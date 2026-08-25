#!/usr/bin/env python3
"""IK campaign preflight + experiment build, on ONE throwaway pod.

1. provision a pod from the Isaac snapshot
2. overlay the CURRENT repo code (robobench *.py + new URDF + eval/) onto its repo copy
3. PREFLIGHT (all must pass, else abort before any campaign spend):
   a. assembly.pc_gpu.franka.diff_ik boots; a scripted EE-pose action sequence moves the
      hand toward a commanded target (IK actually tracks; final error asserted < 2 cm)
   b. COSIGEN_STATELOG snapshots appear during (a), load, and carry states+score
   c. puzzle.syringe.bimanual_franka.diff_ik builds (our one-line preset addition)
4. build the 13 tools experiments with diff_ik presets (build_env.py per task)
5. tar experiments back to the laptop; terminate the pod
"""
import subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/bytedance/Desktop/CoSiGen/eval/scripts")
import run_agent_runpod as R

REPO = Path("/Users/bytedance/Desktop/CoSiGen")
TASKS = [  # (stage spec, exp name)
    ("assembly.allen_bolt:franka:diff_ik", "allen_bolt_ik"),
    ("assembly.bulb:franka:diff_ik", "bulb_ik"),
    ("assembly.ikea_table:bimanual_franka:diff_ik", "ikea_table_ik"),
    ("assembly.nut_thread:franka:diff_ik", "nut_thread_ik"),
    ("assembly.pc_gpu:franka:diff_ik", "pc_gpu_ik"),
    ("assembly.pc_gpu_ram:franka:diff_ik", "pc_gpu_ram_ik"),
    ("assembly.pc_motherboard:franka:diff_ik", "pc_motherboard_ik"),
    ("assembly.pc_ram:franka:diff_ik", "pc_ram_ik"),
    ("packing.pen_holder:franka:diff_ik", "pen_holder_ik"),
    ("packing.tool_packing:franka:diff_ik", "tool_packing_ik"),
    ("puzzle.coffee:franka:diff_ik", "coffee_ik"),
    ("puzzle.spatula:franka:diff_ik", "spatula_ik"),
    ("puzzle.syringe:bimanual_franka:diff_ik", "syringe_ik"),
]

PREFLIGHT = r'''
import os, glob, torch
os.environ["COSIGEN_STATELOG"] = "/tmp/slog"
os.environ["COSIGEN_STATELOG_EVERY"] = "20"
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import robobench
from robobench.core.registries import ENVS
robobench.discover()

# (a) diff_ik tracking
env = ENVS.get("assembly.pc_gpu.franka.diff_ik")().build(num_envs=1)
env.reset()
art = env.robot.articulation
ee_i = art.body_names.index(env.robot.EE_BODY)
p0 = art.data.body_pos_w[0, ee_i].clone()
target = p0 + torch.tensor([0.10, 0.05, -0.08], device=p0.device)
# diff_ik action = same 8-D DELTAS as osc: [dpos(3), drot(3), grip(2)]
for _ in range(240):
    p = art.data.body_pos_w[0, ee_i]
    d = (target - p) * 8.0
    d = torch.clamp(d, -1.0, 1.0)
    act = torch.cat([d, torch.zeros(3, device=d.device),
                     torch.tensor([1.0, 1.0], device=d.device)]).unsqueeze(0)
    env.step(act)
p1 = art.data.body_pos_w[0, ee_i]
err = float((p1 - target).norm())
print(f"PREFLIGHT_IK err={err:.4f} m", flush=True)
assert err < 0.02, f"diff_ik did not track: {err:.3f} m"

# (b) statelog end-to-end
snaps = sorted(glob.glob("/tmp/slog/*/*.pt"))
print(f"PREFLIGHT_STATELOG {len(snaps)} snapshots", flush=True)
assert len(snaps) >= 3, "statelog produced too few snapshots"
d = torch.load(snaps[-1], map_location="cpu", weights_only=False)
assert "states" in d and "score" in d and "sim_step" in d
print("PREFLIGHT_STATELOG_LOAD ok tag=%s score=%s" % (d.get("tag"), d.get("score")), flush=True)

print("PREFLIGHT_MAIN_PASS", flush=True)
os._exit(0)
'''


def main():
    ssh_pub = (Path.home() / ".ssh" / "cosigen_campaign.pub").read_text().strip()
    pod = R.create_pod("rb-ik-preflight", ssh_pub, str(Path.home() / ".ssh" / "cosigen_campaign"),
                       gpu_count=1)
    try:
        R.provision_pod(pod)
        # overlay current code onto the snapshot's repo
        stage = Path("/tmp/_ik_code")
        subprocess.run(["rm", "-rf", str(stage)], check=False)
        for sub in ("robobench", "eval"):
            for f in (REPO / sub).rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(REPO)
                if any(p in rel.parts for p in ("__pycache__", "assets")) and f.suffix != ".urdf":
                    continue  # assets unchanged upstream except the new URDF
                if f.suffix in (".py", ".yaml", ".md", ".json", ".urdf", ".toml", ".sh"):
                    dst = stage / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(f.read_bytes())
        n = R.upload_dir(pod, stage, "/opt/cosigen/CoSiGen.overlay")
        R.sh(pod, "cp -R /opt/cosigen/CoSiGen.overlay/. /opt/cosigen/CoSiGen/ && "
                  "echo OVERLAID", quiet=False)
        print(f"code overlay: {n} files", flush=True)

        # preflight
        R.sh(pod, "mkdir -p /tmp/slog && cat > /tmp/preflight.py <<'PYEOF'\n"
                  + PREFLIGHT + "\nPYEOF", quiet=True)
        rc, out = R.sh(pod, "cd /opt/cosigen/CoSiGen && "
                            "PYTHONPATH=/opt/cosigen/CoSiGen "
                            "OMNI_KIT_ACCEPT_EULA=YES timeout 1800 "
                            "/opt/cosigen/.venv/bin/python /tmp/preflight.py 2>&1 | "
                            "grep -E 'PREFLIGHT|Error|error|assert' | tail -20", quiet=False)
        print(out, flush=True)
        if "PREFLIGHT_MAIN_PASS" not in out:
            raise SystemExit("PREFLIGHT FAILED — aborting before any campaign spend")

        # (c) ikea bimanual diff_ik builds (newly registered mode) — fresh process (one env per process,
        # matching how agents actually run; two builds in one process collide on /World)
        syr = """
import os
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import robobench
from robobench.core.registries import ENVS
robobench.discover()
e = ENVS.get("assembly.ikea_table.bimanual_franka.diff_ik")().build(num_envs=1)
e.reset()
print("PREFLIGHT_BIMANUAL_BUILD ok action_dim=%s" % (e.robot.action_dim,), flush=True)
os._exit(0)
"""
        R.sh(pod, "cat > /tmp/syr_check.py <<'PYEOF'\n" + syr + "\nPYEOF", quiet=True)
        rc, out = R.sh(pod, "cd /opt/cosigen/CoSiGen && "
                            "PYTHONPATH=/opt/cosigen/CoSiGen OMNI_KIT_ACCEPT_EULA=YES "
                            "timeout 1800 /opt/cosigen/.venv/bin/python /tmp/syr_check.py 2>&1 | "
                            "grep -E 'PREFLIGHT|Error|error' | tail -8", quiet=False)
        print(out, flush=True)
        if "PREFLIGHT_BIMANUAL_BUILD ok" not in out:
            raise SystemExit("PREFLIGHT FAILED (bimanual ikea) — aborting before any campaign spend")

        # build the 13 experiments
        for spec, name in TASKS:
            rc, out = R.sh(pod, f"cd /opt/cosigen/CoSiGen && "
                                f"PYTHONPATH=/opt/cosigen/CoSiGen OMNI_KIT_ACCEPT_EULA=YES "
                                f"timeout 1800 /opt/cosigen/.venv/bin/python "
                                f"eval/scripts/build_env.py --name {name} "
                                f"--stage {spec} --config default 2>&1 | tail -5; "
                                f"test -f /opt/cosigen/CoSiGen/experiments/{name}/MANIFEST "
                                f"&& echo BUILD_DONE_{name}",
                           quiet=False, timeout=2000)
            print(f"[{name}] {out.strip()[-300:]}", flush=True)
            if f"BUILD_DONE_{name}" not in out:
                raise SystemExit(f"BUILD FAILED for {name} — aborting")
        # fetch experiments
        data = R.download_tar(pod, "tar czf - -C /opt/cosigen/CoSiGen/experiments "
                                   + " ".join(n for _, n in TASKS), timeout=1800)
        outp = REPO / "experiments" / "_ik_batch.tgz"
        outp.write_bytes(data)
        subprocess.run(["tar", "xzf", str(outp), "-C", str(REPO / "experiments")], check=True)
        outp.unlink()
        missing = [n for _, n in TASKS
                   if not (REPO / "experiments" / n / "MANIFEST").is_file()]
        if missing:
            raise SystemExit(f"FETCH INCOMPLETE, missing locally: {missing}")
        print("EXPERIMENTS FETCHED AND VERIFIED (13/13)", flush=True)
    finally:
        R.terminate_pod(pod)
        print("pod terminated", flush=True)


if __name__ == "__main__":
    main()
