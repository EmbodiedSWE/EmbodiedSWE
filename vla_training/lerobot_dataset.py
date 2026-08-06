"""LeRobot dataset construction for sim_gen VLA training (pi05 BC).

The schema mirrors openpi's own conversion template
(openpi/examples/libero/convert_libero_data_to_lerobot.py) because the
pi05_libero recipe -- the repo's standard, well-adopted pi0.5 BC recipe --
consumes exactly these keys through its LiberoInputs transforms:

    image        (256,256,3) uint8   base camera
    wrist_image  (256,256,3) uint8   wrist camera
    state        (8,)  float32       proprio
    actions      (7,)  float32       action target
    task         str                 language instruction -> prompt
                                     (DataConfig.prompt_from_task=True)

Real trajectory converters and the mock generator both go through
`create_dataset()` / `add_step()` so the schema lives in ONE place: when real
sim_gen trajectories arrive, write a converter that maps them onto add_step()
-- nothing on the training side changes.

Requires openpi's pinned lerobot (run under `uv run` from the openpi repo, or
with its .venv active), plus HF_LEROBOT_HOME pointing at the dataset root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CAPX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAPX_ROOT))

FPS = 10
FEATURES = {
    "image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "actions": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}


def ledger_tasks() -> list[dict]:
    """Accepted sim_gen campaign tasks ({task, tier, env}) -- the same 40-odd
    task set the RL run trains on, via simgen_rl's ledger enumeration."""
    from simgen_rl.prepare_dataset import load_envs

    return load_envs()


def task_prompt(task_dir: str) -> str:
    """Language instruction from a task dir name, e.g. 'open_oven_i7' ->
    'open oven'. Placeholder quality: real converters should carry the scene's
    own TASK text instead when available."""
    return task_dir.rsplit("_i", 1)[0].replace("_", " ")


def create_dataset(repo_id: str, *, overwrite: bool = True):
    """A LeRobotDataset in the pi05_libero schema, rooted at HF_LEROBOT_HOME."""
    import shutil

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    out = HF_LEROBOT_HOME / repo_id
    if out.exists():
        if not overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def add_step(dataset, *, image: np.ndarray, wrist_image: np.ndarray,
             state: np.ndarray, actions: np.ndarray, task: str) -> None:
    """One frame in the recipe schema (shapes/dtypes validated by lerobot)."""
    dataset.add_frame({
        "image": image,
        "wrist_image": wrist_image,
        "state": state,
        "actions": actions,
        "task": task,
    })
