"""The 28 published tasks, with explicit evaluation bindings (not all ENVS aliases).

Source: https://arxiv.org/html/2609.27308v1#A2.T3
Bindings use the implementations in this checkout; notably knot uses ALOHA.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    preset: str

    @property
    def id(self) -> str:
        return ".".join(self.preset.split(".")[:2])

    @property
    def suite(self) -> str:
        return self.preset.split(".")[0]

    @property
    def scene(self) -> str:
        return self.preset.split(".")[1]

    @property
    def backend(self) -> str:
        return "newton" if self.suite == "deformable" else "physx"

    @property
    def stage_arg(self) -> str:
        suite, scene, robot, mode = self.preset.split(".")
        return f"{suite}.{scene}:{robot}:{mode}"


TASKS = tuple(TaskSpec(preset) for preset in (
    "assembly.allen_bolt.franka.osc",
    "assembly.bulb.franka.osc",
    "assembly.ikea_table.bimanual_franka.osc",
    "assembly.so101.bimanual_franka.osc",
    "assembly.nut_thread.franka.osc",
    "assembly.pc_gpu.franka.osc",
    "assembly.pc_gpu_ram.franka.osc",
    "assembly.pc_motherboard.franka.osc",
    "assembly.pc_ram.franka.osc",
    "packing.pen_holder.franka.osc",
    "packing.tool_packing.franka.osc",
    "packing.egg_carton.g1.pink_ik",
    "packing.clear_organic_objects.g1.pink_ik",
    "puzzle.coffee.franka.osc",
    "puzzle.spatula.franka.osc",
    "puzzle.syringe.bimanual_franka.osc",
    "puzzle.push_shapes.g1.pink_ik",
    "puzzle.classify_objects.g1.pink_ik",
    "puzzle.stack_blocks.g1.pink_ik",
    "deformable.tshirt.franka.joint",
    "deformable.latte.bimanual_franka.joint",
    "deformable.dumpling.franka.joint",
    "deformable.knot.aloha.joint",
    "cutting.slice.franka.osc",
    "cutting.dice.franka.osc",
    "locomanip.fruit_delivery.g1.loco_pink_ik",
    "locomanip.box_to_bin.g1.loco_pink_ik",
    "locomanip.wheel_carry.g1.loco_pink_ik",
))


def select_tasks(names: list[str] | None = None) -> tuple[TaskSpec, ...]:
    if not names:
        return TASKS
    unknown = set(names) - {task.id for task in TASKS}
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown))}; use list to see the roster")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate task selections")
    return tuple(task for task in TASKS if task.id in names)


def validate_registry(tasks=TASKS) -> None:
    """Exercise the actual registries; requires torch/numpy, but never boots Isaac."""
    import importlib
    import robobench
    from robobench.core import ENVS

    robobench.discover()
    for task in tasks:
        cfg = ENVS.get(task.preset)()
        graders = importlib.import_module(f"robobench.suites.{task.suite}.grader").GRADERS
        if cfg.scene != task.scene or cfg.scene not in graders or cfg.robot == "null":
            raise ValueError(f"Invalid task binding or missing grader: {task.preset}")
