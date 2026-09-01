"""Re-download the vendored AGILE lower-body locomotion policy (provenance + reproducibility).

`g1/policies/agile_locomotion.pt` is the FROZEN locomotion policy the G1's `loco_*` control modes
run on their 12 leg DOFs (see `robobench/controllers/loco_policy.py`). It is vendored into the repo
for the same reason the G1 USD is (`fetch_g1.py`): robobench stays relocatable and consumes no live
Nucleus at runtime. This script regenerates the file, so the copy is reproducible and auditable.

Source: the `policy_path` of Isaac Lab's own G1 loco-manipulation task —
`isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py`, which
reads `f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"`. On Isaac Sim 5.1 that resolves
to the S3 cloud root below. Single file, no dependencies to follow.

WHAT IT IS (measured off the checkpoint, not guessed — `main()` re-asserts all of it):

  - TorchScript module, 500,080 bytes, sha256 f04a58b8...b775dde.
  - `forward(x: Tensor) -> Tensor`, **83 in -> 12 out**, batched on dim 0.
  - Architecture: a `normalizer` (an IDENTITY pass-through in this export — it holds no running
    statistics, so there is nothing to freeze) into an MLP actor 83 -> 256 -> 256 -> 128 -> 12.
  - No buffers, no hidden state: two calls on the same input return the same output. The only
    state in the control loop is the `last_action` term the CALLER feeds back through the
    observation (`loco_policy.py` owns that).

  The 83 inputs are `[command(4) | observation(79)]`, and the 12 outputs are leg joint position
  residuals, scaled by 0.25 and added to the legs' default joint positions. The exact term order of
  the 79 is Isaac Lab's `AgileTeacherPolicyObservationsCfg` — `loco_policy.py` is where that lives
  and is the file to read; it is not re-stated here.

Run with a plain torch Python (no AppLauncher, no Isaac needed):
    python robobench/robots/assets/fetch_g1_locomotion.py
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

# Isaac Sim 5.1 cloud asset root (= ISAACLAB_NUCLEUS_DIR), pinned for reproducibility.
URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Policies/Agile/agile_locomotion.pt"
)
OUT = os.path.join(os.path.dirname(__file__), "g1", "policies", "agile_locomotion.pt")

#: What the file must be. A checkpoint that silently changed shape would not raise anywhere in the
#: control loop — it would just make the robot fall over — so it is pinned here and checked on fetch.
SHA256 = "f04a58b834057eb1c9f38350dc12feaf929ff2cc7d5b75d2871e23811b775dde"
SIZE = 500080
OBS_DIM = 79  # the observation half of the input
CMD_DIM = 4  # [vx, vy, wz, hip_height]
ACT_DIM = 12  # hips (6) + knees (2) + ankles (4)


def verify(path: str) -> None:
    """Assert the checkpoint is the one this repo is written against: size, digest, and — by an
    actual forward pass — its input/output widths and its statelessness."""
    import torch

    blob = open(path, "rb").read()
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != SIZE or digest != SHA256:
        raise ValueError(f"{path}: expected {SIZE} B / sha256 {SHA256}, got {len(blob)} B / {digest}")

    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    x = torch.zeros(2, CMD_DIM + OBS_DIM)
    with torch.no_grad():
        y, again = policy(x), policy(x)
    if tuple(y.shape) != (2, ACT_DIM):
        raise ValueError(f"{path}: expected ({CMD_DIM}+{OBS_DIM}) -> {ACT_DIM}, got output {tuple(y.shape)}")
    if not torch.equal(y, again):
        raise ValueError(f"{path}: policy is not deterministic — it carries state this repo does not reset")
    print(f"verified: {CMD_DIM}+{OBS_DIM} -> {ACT_DIM}, stateless, sha256 {digest[:12]}...")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"GET {URL}")
    with urllib.request.urlopen(URL) as r:
        blob = r.read()
    with open(OUT, "wb") as f:
        f.write(blob)
    print(f"  -> {OUT} ({len(blob)} B)")
    verify(OUT)


if __name__ == "__main__":
    main()
