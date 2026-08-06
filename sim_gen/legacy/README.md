# legacy_mujoco — the MuJoCo reference implementation of the sim_gen pipeline

The pipeline's active backend is Isaac Lab (robobench format; see ../PIPELINE.md).
This directory preserves the fully-working MuJoCo variant as a reference:

- `core/`          minimal MuJoCo task framework (BaseScene/SceneCfg/Recorder/Checks)
- `push_cube_ref/` hand-written reference task (the format exemplar)
- `push_cube_d1/`  the first agent-generated task (validated 23/23)
- `validate.py`    the MuJoCo validation gauntlet (generic gates + smoke re-run)
- `spawn_agent.py` the single-task construction driver (OAuth billing path)

It is the simplest complete embodiment of the task contract (scene/oracle/rubric/
checks) and runs on any CPU box in seconds — useful for demonstrating the pipeline
without GPUs. Not maintained as the production path.
