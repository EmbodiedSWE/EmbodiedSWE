"""Checkpoint tree — save a reached world state to disk, restore any saved state in a later
process, and organise the saved states around the agent's own stage plan.

    from checkpoint_tree import CheckpointTree
    tree = CheckpointTree(env)                       # right after env.reset(); reconciles n0
    tree.plan(["leg 1 seated", "legs 1-2 seated", "legs 1-3 seated", "all 4 legs seated"])
    result = tree.run_stage(1)                       # goto(previous boundary) -> stage_1.run(env)
                                                     #   -> stage_1.check(env) -> save boundary
    tree.goto_stage(2)                               # restore the latest stage-2 boundary
    print(tree.show())                               # plan progress + every node

The tree is a plain tree of saved world states (`env.get_states()` tensors, one `.pt` per node)
plus non-restorable attempt records. A stage boundary is an ordinary node tagged with its stage
index; a stage may have several boundary nodes (different ways of reaching the same subgoal) and
the agent chooses which one to continue from. Every script starts from the task's fresh reset
(node `n0`) unless it explicitly calls `goto()` / `goto_stage()` / `run_stage()`.

Stage modules live at `/workspace/solution/stages/stage_<k>.py` and expose
    def run(env) -> None      # drives the world from the previous boundary to this stage's goal
    def check(env) -> bool    # the agent's OWN verification of that goal (reported, never fatal)
so `solve.py` can be their composition; `run_stage(k)` imports that module when no callables are
passed. Nothing here grades anything: the rubric is applied elsewhere to the delivered program.

Files under /workspace/.checkpoints: tree.json (manifest), <cid>.pt (state), <cid>.code.py
(program that reached it), <cid>.log.txt (its full printed output), plus the viewer's snapshots.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

TOOL = {
    "name": "checkpoint_tree",
    "description": (
        "Save world states as an annotated tree of nodes on disk and restore any of them in a "
        "later run (save / goto); plan the task in stages and save one boundary node per "
        "completed stage (plan / run_stage / goto_stage)."
    ),
    "exports": ("CheckpointTree",),
    "skill": "cosigen-checkpoint-tree",
    "prompt_doc": "tools/checkpoint_tree.md",
    "requires_features": ("set_states",),
}

MANIFEST = "tree.json"
DEFAULT_ROOT = "/workspace/.checkpoints"
STAGES_DIR = "/workspace/solution/stages"
OUTCOMES = ("ok", "partial", "failed", "crashed")
POS_TOL = 2e-3          # m: a state entry that moved less than this is "unchanged" in a diff
MOTION_LIN = 0.02       # m/s: above this the world is still moving (checkpoint not settled)
MOTION_ANG = 0.2        # rad/s
MOTION_JOINT = 0.05     # rad/s
LIMIT_MARGIN = 0.05     # rad: a robot joint closer than this to a limit is flagged


# ----- records -----------------------------------------------------------------------------------
@dataclass
class Record:
    """A checkpoint (restorable, has a .pt) or an attempt (provenance only, no state)."""
    cid: str
    kind: str                       # "checkpoint" | "attempt"
    parent: str | None
    label: str                      # the STATE reached, in task terms (checkpoint) / action (attempt)
    action: str = ""
    note: str = ""
    created: float = 0.0
    stage: int | None = None        # stage index when the node is a stage boundary
    origin: str = "fresh"           # what the live world was when this was made
    outcome: str = "ok"
    reusable: bool = True
    code_name: str = ""
    snapshot: str = ""
    success: bool | None = None     # the scene's own success predicate at save time, if any
    state_diff: str = ""            # computed: what moved versus the parent
    scene_diff: str = ""            # the agent's, after looking at parent/node snapshots
    joints: str = ""
    health: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Record":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}   # tolerate old manifests
        known.setdefault("kind", "checkpoint")
        known.setdefault("label", "")
        return cls(**known)


def _jsonable(v: Any) -> Any:
    if hasattr(v, "detach"):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _leaves(tree, prefix=""):
    """(dotted path, tensor) for every tensor in a get_states() tree."""
    import torch
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif torch.is_tensor(tree):
        yield prefix, tree


def state_diff_text(old, new, tol: float = POS_TOL) -> str:
    """What changed between two saved states, by max absolute difference per entry."""
    a = dict(_leaves(old))
    moved = []
    for path, t in _leaves(new):
        u = a.get(path)
        if u is None or u.shape != t.shape or not t.is_floating_point():
            continue
        d = float((t.detach().cpu().float() - u.detach().cpu().float()).abs().max())
        if d > tol:
            moved.append((d, f"{path.replace('scene.', '')} max|Δ|={d:.4f}"))
    if not moved:
        return f"nothing changed beyond tolerance ({tol} m/rad)"
    moved.sort(reverse=True)
    return "; ".join(s for _, s in moved[:10]) + (f"; +{len(moved) - 10} more" if len(moved) > 10 else "")


# ----- health: is the live world a settled, restorable state? ------------------------------------
def _articulations(env):
    arts = getattr(getattr(env, "iscene", None), "articulations", None) or {}
    return dict(arts)


def _rigid_objects(env):
    return dict(getattr(getattr(env, "iscene", None), "rigid_objects", None) or {})


def _arm_joint_limits(env):
    """(arm joint indices, limits [J,2], joint_pos [J]) of the robot articulation, or
    ([], None, None) when the env has no robot articulation. Arm joints = the robot's
    `ARM_JOINTS` patterns when it declares them, else every joint not matching its
    `GRIPPER_JOINTS` patterns."""
    robot = getattr(env, "robot", None)
    art = getattr(robot, "articulation", None)
    if art is None:
        return [], None, None
    d = art.data
    lims = getattr(d, "soft_joint_pos_limits", None)
    lims = d.joint_pos_limits if lims is None else lims
    if lims is None:
        return [], None, None
    names = list(getattr(art, "joint_names", []) or [])
    arm_pats = list(getattr(robot, "ARM_JOINTS", ()) or ())
    grip_pats = list(getattr(robot, "GRIPPER_JOINTS", ()) or ())
    if arm_pats:
        ids = [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in arm_pats)]
    else:
        ids = [i for i, n in enumerate(names) if not any(re.fullmatch(p, n) for p in grip_pats)]
    return ids, lims[0], d.joint_pos[0]


def checkpoint_health(env) -> dict[str, Any]:
    """Read-only: motion (world still moving?), robot ARM joint-limit margin, scene success."""
    h: dict[str, Any] = {"max_lin_vel": 0.0, "max_ang_vel": 0.0, "max_joint_vel": 0.0,
                         "min_joint_margin_rad": None, "scene_success": None, "unsafe_reasons": []}
    try:
        for name, obj in _rigid_objects(env).items():
            d = obj.data
            h["max_lin_vel"] = max(h["max_lin_vel"], float(d.root_lin_vel_w.norm(dim=-1).max()))
            h["max_ang_vel"] = max(h["max_ang_vel"], float(d.root_ang_vel_w.norm(dim=-1).max()))
        for name, art in _articulations(env).items():
            h["max_joint_vel"] = max(h["max_joint_vel"], float(art.data.joint_vel.abs().max()))
        # Limit margin: the robot's ARM joints only. Gripper fingers sit at a limit whenever
        # they are fully open or closed (Franka open = +0.04 of [0, 0.04]), and articulated
        # scene objects (lids, drawers) reach their limits as GOALS — neither is unsafe.
        ids, lims, q = _arm_joint_limits(env)
        if lims is not None and len(ids):
            margin = float(min((q[ids] - lims[ids, 0]).min(), (lims[ids, 1] - q[ids]).min()))
            h["min_joint_margin_rad"] = margin
        scene = getattr(env, "scene", None)
        if scene is not None and callable(getattr(scene, "success", None)):
            s = scene.success()
            h["scene_success"] = bool(s.any()) if hasattr(s, "any") else bool(s)
    except Exception as exc:  # noqa: BLE001 -- a readout bug must be visible, not hidden
        print(f"[checkpoint_tree] health readout failed: {exc!r}", flush=True)
        traceback.print_exc()
    if h["max_lin_vel"] > MOTION_LIN or h["max_ang_vel"] > MOTION_ANG or h["max_joint_vel"] > MOTION_JOINT:
        h["unsafe_reasons"].append("world still moving")
    if h["min_joint_margin_rad"] is not None and h["min_joint_margin_rad"] < LIMIT_MARGIN:
        h["unsafe_reasons"].append(f"robot joint within {h['min_joint_margin_rad']:.3f} rad of a limit")
    h["unsafe"] = bool(h["unsafe_reasons"])
    return h


def health_summary(h: dict[str, Any] | None) -> str:
    if not h:
        return "(no health record)"
    m = h.get("min_joint_margin_rad")
    return (f"lin {h.get('max_lin_vel', 0):.3f} m/s, ang {h.get('max_ang_vel', 0):.2f} rad/s, "
            f"joints {h.get('max_joint_vel', 0):.3f} rad/s, limit margin "
            f"{'n/a' if m is None else f'{m:.3f} rad'}, scene success={h.get('scene_success')}"
            + (f" — UNSAFE: {'; '.join(h['unsafe_reasons'])}" if h.get("unsafe") else ""))


def _joints_text(env) -> str:
    try:
        parts = []
        for name, art in _articulations(env).items():
            d = art.data
            lims = getattr(d, "soft_joint_pos_limits", None)
            lims = d.joint_pos_limits if lims is None else lims
            names = list(getattr(art, "joint_names", []) or [])
            q = d.joint_pos[0]
            if len(names) != q.shape[0]:
                names = [f"j{i}" for i in range(q.shape[0])]
            parts.append(name + ": " + ", ".join(
                f"{n} {float(p):+.2f}" + (f" [{float(l[0]):+.2f},{float(l[1]):+.2f}]" if lims is not None else "")
                for n, p, l in zip(names, q, lims[0] if lims is not None else [None] * len(names))))
        return "; ".join(parts)
    except Exception as exc:  # noqa: BLE001
        return f"(joints readout failed: {exc!r})"


# ----- the tree ----------------------------------------------------------------------------------
class CheckpointTree:
    def __init__(self, env, root: str | Path | None = None, *, verbose: bool = True,
                 assume_reset: bool = True):
        self.env = env
        self.root = Path(DEFAULT_ROOT if root is None else root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.records: dict[str, Record] = {}
        self.stages: list[str] = []
        self._seq = 0
        self._aseq = 0
        self._viewer = None
        self.current: str | None = None      # process-local active origin
        self._origin = "fresh"
        self._load()
        self._reconcile_origin(assume_reset)

    # ----- persistence -----
    def _load(self) -> None:
        p = self.root / MANIFEST
        if not p.is_file():
            return
        d = json.loads(p.read_text())
        for cid, n in (d.get("nodes") or {}).items():
            n.setdefault("kind", "checkpoint")
            self.records[cid] = Record.from_json(n)
        for aid, a in (d.get("attempts") or {}).items():
            a.setdefault("kind", "attempt")
            a.setdefault("cid", aid)
            self.records[aid] = Record.from_json(a)
        self.stages = list(d.get("plan") or [])
        self._seq = max([int(c[1:]) for c in self.records if c.startswith("n") and c[1:].isdigit()], default=0)
        self._aseq = max([int(c[1:]) for c in self.records if c.startswith("a") and c[1:].isdigit()], default=0)

    def _save_manifest(self) -> None:
        nodes = {c: r.as_json() for c, r in self.records.items() if r.kind == "checkpoint"}
        for c, n in nodes.items():   # `children` kept for readers of the old schema
            n["children"] = [k for k, r in self.records.items() if r.kind == "checkpoint" and r.parent == c]
        payload = {"nodes": nodes,
                   "attempts": {c: r.as_json() for c, r in self.records.items() if r.kind == "attempt"},
                   "plan": self.stages, "seq": self._seq, "attempt_seq": self._aseq,
                   "current": None, "saved_at": time.time()}
        (self.root / MANIFEST).write_text(json.dumps(payload, indent=1) + "\n")

    def _state_path(self, cid: str) -> Path:
        return self.root / f"{cid}.pt"

    def _reconcile_origin(self, assume_reset: bool) -> None:
        """n0 is the task's fresh reset. A new process starts there iff its live world matches."""
        import torch
        live = self.env.get_states()
        if "n0" not in self.records:
            if not assume_reset:
                self._say("no n0 exists and the live world was not asserted to be reset; call goto() before saving")
                return
            torch.save(live, self._state_path("n0"))
            self.records["n0"] = Record(cid="n0", kind="checkpoint", parent=None, label="initial state",
                                        action="environment reset", created=time.time(),
                                        health=checkpoint_health(self.env), joints=_joints_text(self.env))
            self.records["n0"].success = self.records["n0"].health.get("scene_success")
            self.current, self._origin = "n0", "fresh"
            self._save_manifest()
            self._say(f"created fresh origin n0 at {self.root}")
            return
        diff = state_diff_text(torch.load(self._state_path("n0"), weights_only=False), live)
        if diff.startswith("nothing changed"):
            self.current, self._origin = "n0", "fresh"
            self._say(f"reopened {len(self)} checkpoint(s), {sum(r.kind == 'attempt' for r in self.records.values())} "
                      f"attempt(s); {len(self.stages)} planned stage(s); active origin n0 (fresh)")
        else:
            self.current = None
            print(f"[checkpoint_tree] live world differs from n0 ({diff}); no active origin — call goto() "
                  "explicitly before saving", flush=True)

    # ----- the plan -----
    def plan(self, stages: list[str]) -> str:
        """Declare the stage plan: one entry per stage, in order, each naming the STATE that
        completes it ('one leg seated', 'card seated in its slot'). A single entry is a valid
        plan. Replaces any earlier plan; existing boundary nodes keep their stage tags."""
        stages = [str(s).strip() for s in stages if str(s).strip()]
        if not stages:
            raise ValueError("a plan needs at least one stage")
        self.stages = stages
        self._save_manifest()
        self._say(f"plan recorded: {len(stages)} stage(s)")
        return self.plan_text()

    replan = plan

    def boundaries(self, k: int) -> list[str]:
        """Restorable boundary nodes of stage k, oldest first."""
        return sorted((c for c, r in self.records.items()
                       if r.kind == "checkpoint" and r.stage == k and r.reusable),
                      key=lambda c: self.records[c].created)

    def plan_text(self) -> str:
        if not self.stages:
            return "plan: (none recorded — call tree.plan([...]); a single stage is fine)"
        lines = [f"plan ({len(self.stages)} stage{'s' if len(self.stages) > 1 else ''}):"]
        for i, s in enumerate(self.stages, 1):
            b = self.boundaries(i)
            lines.append(f"  {'✓' if b else '○'} stage {i}: {s}" + (f"  — boundary nodes: {', '.join(b)}" if b else ""))
        return "\n".join(lines)

    # ----- saving -----
    def save(self, label: str, note: str = "", *, action: str = "", program: str | Path | None = None,
             log: str = "", outcome: str = "ok", reusable: bool | None = None,
             metrics: dict | None = None, stage: int | None = None, as_root: bool = False,
             require: str = "warn") -> str:
        """Save the live world as a child of this process's active origin.

        label   the STATE reached, in the task's own terms
        stage   mark the node as a boundary of that stage (run_stage does this for you)
        Health is read-only and reported; `require="reject"` refuses an UNSAFE (still moving /
        joint at a limit) state, "warn" (default) saves it with a warning, "off" says nothing.
        The saved node becomes this process's active origin; later processes start at n0.
        """
        import torch
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        parent = None if as_root else self.current
        if not as_root and parent not in self.records:
            raise RuntimeError("no active origin in this process (live world did not match n0); call goto() first "
                               "or pass as_root=True")
        if parent is not None and not self.records[parent].reusable:
            raise RuntimeError(f"active origin {parent} is not reusable")
        state = self.env.get_states()
        health = checkpoint_health(self.env)
        if health["unsafe"] and require != "off":
            msg = f"UNSAFE candidate ({'; '.join(health['unsafe_reasons'])})"
            if require == "reject":
                raise RuntimeError(f"[checkpoint_tree] REJECTED {msg}")
            self._say(f"WARNING {msg}: saved anyway; hold longer before saving a state you want to continue from")
        if reusable is None:
            reusable = outcome in ("ok", "partial")
        self._seq += 1
        cid = f"n{self._seq}"
        torch.save(state, self._state_path(cid))
        diff = ""
        if parent is not None:
            diff = state_diff_text(torch.load(self._state_path(parent), weights_only=False), state)
        snapshot = ""
        if self._viewer is not None:
            try:
                snapshot = str(self._viewer.snapshot(f"ckpt_{cid}_{label}") or "")
            except Exception as exc:  # noqa: BLE001 -- a snapshot bug must not lose the save
                print(f"[checkpoint_tree] snapshot failed for {cid}: {exc!r}", flush=True)
        self.records[cid] = Record(
            cid=cid, kind="checkpoint", parent=parent, label=label, action=action or label, note=note,
            created=time.time(), stage=stage, origin="independent" if as_root else self._origin,
            outcome=outcome, reusable=bool(reusable), code_name=self._store_program(cid, program),
            snapshot=snapshot, success=health.get("scene_success"), state_diff=diff,
            joints=_joints_text(self.env), health=health, metrics=dict(_jsonable(metrics or {})))
        self._store_log(cid, log)
        self.current, self._origin = cid, f"continued:{cid}"
        self._save_manifest()
        self._say(f"saved {cid} '{label}'" + (f" [stage {stage} boundary]" if stage else "") +
                  f" (parent {parent}, outcome {outcome}); health: {health_summary(health)}")
        if diff:
            self._say(f"changed vs {parent}: {diff}")
        return cid

    def record_attempt(self, action: str, outcome: str, note: str = "", log: str = "",
                       program: str | Path | None = None, *, parent: str | None = None,
                       metrics: dict | None = None, stage: int | None = None) -> str:
        """Remember a tried edge WITHOUT a restorable state (failures stay non-reusable)."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        parent = parent or self.current
        self._aseq += 1
        aid = f"a{self._aseq}"
        self.records[aid] = Record(
            cid=aid, kind="attempt", parent=parent, label=action, action=action, note=note,
            created=time.time(), stage=stage, origin=self._origin, outcome=outcome, reusable=False,
            code_name=self._store_program(aid, program), health=checkpoint_health(self.env),
            metrics=dict(_jsonable(metrics or {})))
        self._store_log(aid, log)
        self._save_manifest()
        self._say(f"recorded attempt {aid} from {parent}: {action} -> {outcome}")
        return aid

    @contextmanager
    def attempt(self, action: str, *, program: str | Path | None = None, stage: int | None = None) -> Iterator[dict]:
        """`with tree.attempt('press deeper') as a:` — records the block as an attempt from the
        current origin: `crashed` if it raises (re-raised), else a['outcome'] (default 'failed')."""
        a = {"outcome": "failed", "note": "", "log": "", "metrics": None}
        try:
            yield a
        except BaseException:
            self.record_attempt(action, "crashed", note=traceback.format_exc()[-2000:], log=a["log"],
                                program=program, metrics=a["metrics"], stage=stage)
            raise
        self.record_attempt(action, a["outcome"], note=a["note"], log=a["log"], program=program,
                            metrics=a["metrics"], stage=stage)

    def _store_program(self, rid: str, program: str | Path | None) -> str:
        if not program:
            return ""
        src = Path(program)
        if src.is_file():
            shutil.copy(src, self.root / f"{rid}.code.py")
            return src.name
        (self.root / f"{rid}.code.py").write_text(str(program))
        return "(inline)"

    def _store_log(self, rid: str, log: str) -> None:
        (self.root / f"{rid}.log.txt").write_text("" if log is None else str(log))

    # ----- restoring -----
    def goto(self, cid: str, q_ref: str = "preserve") -> str:
        """Restore the world to a saved node and make it the active origin. Returns
        tried_from(cid). q_ref="reseed" additionally resets controller integrators."""
        import torch
        r = self.records.get(cid)
        if r is None or r.kind != "checkpoint":
            raise KeyError(f"no checkpoint '{cid}'; checkpoints: {sorted(c for c, x in self.records.items() if x.kind == 'checkpoint')}")
        if not r.reusable:
            raise ValueError(f"{cid} is not reusable (outcome={r.outcome})")
        self.env.set_states(torch.load(self._state_path(cid), weights_only=False))
        if q_ref == "reseed":
            n = self._reset_controllers()
            self._say(f"reseeded {n} controller(s)")
        elif q_ref != "preserve":
            raise ValueError("q_ref must be 'preserve' or 'reseed'")
        self.current, self._origin = cid, f"goto:{cid}"
        self._say(f"world restored to {cid} '{r.label}'" + (f" [stage {r.stage} boundary]" if r.stage else "")
                  + f"; health: {health_summary(checkpoint_health(self.env))}")
        self._say("the restore kept gripper/actuator setpoints: send an intentional first command before stepping")
        return self.tried_from(cid)

    def goto_stage(self, k: int, cid: str | None = None, q_ref: str = "preserve") -> str:
        """Restore a boundary of stage k: the latest one, or the given node (must be tagged k)."""
        if cid is None:
            b = self.boundaries(k)
            if not b:
                raise KeyError(f"stage {k} has no boundary node yet; run_stage({k}) first")
            cid = b[-1]
        elif self.records.get(cid) is None or self.records[cid].stage != k:
            raise KeyError(f"{cid} is not a boundary of stage {k}")
        return self.goto(cid, q_ref=q_ref)

    def _reset_controllers(self) -> int:
        n = 0
        stack = [getattr(self.env, "robot", None)]
        seen = set()
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            ctrl = getattr(obj, "controller", None)
            if ctrl is not None and callable(getattr(ctrl, "reset", None)):
                ctrl.reset()
                n += 1
            kids = getattr(obj, "robots", None) or getattr(ctrl, "controllers", None) or {}
            stack.extend(kids.values() if isinstance(kids, dict) else list(kids))
        return n

    # ----- the stage protocol -----
    def run_stage(self, k: int, run: Callable | None = None, check: Callable | None = None, *,
                  from_node: str | None = None, program: str | Path | None = None,
                  label: str | None = None) -> dict:
        """Develop stage k from the previous boundary and save its boundary when its check passes.

        1. origin: `from_node` if given, else the latest boundary of stage k-1 (n0 for k=1) —
           restored with goto() unless it is already the active origin;
        2. `run(env)` drives the world (stdout is captured as the node's log and also shown);
        3. `check(env)` -> bool is the agent's own verification, REPORTED, never fatal;
        4. check passed (or no check given): the state is saved as a stage-k boundary node with
           `program`; check failed: an attempt is recorded and NOTHING is saved.
        `run`/`check` default to /workspace/solution/stages/stage_<k>.py's functions.
        Returns {"stage", "passed", "node" | "attempt", "health", "origin"}.
        """
        if run is None or check is None:
            mod = self._load_stage_module(k)
            run = run or getattr(mod, "run", None)
            check = check if check is not None else getattr(mod, "check", None)
            program = program or mod.__file__
            if run is None:
                raise AttributeError(f"{mod.__file__} defines no run(env)")
        if from_node is None:
            if k <= 1:
                from_node = "n0"
            else:
                b = self.boundaries(k - 1)
                if not b:
                    raise RuntimeError(f"stage {k - 1} has no boundary node; run_stage({k - 1}) first, "
                                       f"or pass from_node=<cid> explicitly")
                from_node = b[-1]
        if self.current != from_node:
            self.goto(from_node)
        name = label or (self.stages[k - 1] if 0 < k <= len(self.stages) else f"stage {k} reached")
        self._say(f"stage {k} '{name}': running from {from_node}")
        buf = io.StringIO()
        with _Tee(buf):
            try:
                run(self.env)
            except BaseException as exc:
                log = buf.getvalue()
                aid = self.record_attempt(f"stage {k}: {name}", "crashed", note=repr(exc), log=log,
                                          program=program, parent=from_node, stage=k)
                print(f"[checkpoint_tree] stage {k} CRASHED ({exc!r}); recorded as {aid}", flush=True)
                raise
            passed = None
            if check is not None:
                try:
                    passed = bool(check(self.env))
                except BaseException as exc:  # noqa: BLE001 -- a broken check is the agent's bug to see
                    print(f"[checkpoint_tree] stage {k} check raised {exc!r}; treating as FAILED", flush=True)
                    traceback.print_exc()
                    passed = False
        log = buf.getvalue()
        health = checkpoint_health(self.env)
        if passed is False:
            aid = self.record_attempt(f"stage {k}: {name}", "failed", note="stage check returned False",
                                      log=log, program=program, parent=from_node, stage=k)
            print(f"[checkpoint_tree] stage {k} check FAILED — not saved (attempt {aid}); health: "
                  f"{health_summary(health)}. Fix the stage and run_stage({k}) again, or goto_stage({k - 1}) "
                  "and change approach.", flush=True)
            return {"stage": k, "passed": False, "attempt": aid, "health": health, "origin": from_node}
        cid = self.save(name, note="" if passed is None else "stage check passed", action=f"stage {k}: {name}",
                        program=program, log=log, stage=k)
        print(f"[checkpoint_tree] stage {k} {'check passed' if passed else 'saved (no check given)'} -> boundary "
              f"{cid}. {self.plan_text()}", flush=True)
        return {"stage": k, "passed": passed, "node": cid, "health": health, "origin": from_node}

    @staticmethod
    def _load_stage_module(k: int):
        path = Path(STAGES_DIR) / f"stage_{k}.py"
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist: write it with run(env) and check(env), "
                                    "or pass run=/check= callables")
        spec = importlib.util.spec_from_file_location(f"stage_{k}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"stage_{k}"] = mod
        spec.loader.exec_module(mod)
        return mod

    # ----- reading -----
    def annotate(self, cid: str, *, action: str | None = None, scene_diff: str | None = None,
                 note: str | None = None) -> None:
        r = self.records[cid]
        if action is not None:
            r.action = action
        if scene_diff is not None:
            r.scene_diff = scene_diff
        if note is not None:
            r.note = note
        self._save_manifest()

    def tried_from(self, cid: str | None = None, *, verbose: bool = False) -> str:
        """Everything already tried from a node — checkpoints and attempts — with outcomes."""
        cid = cid or self.current
        kids = sorted((r for r in self.records.values() if r.parent == cid), key=lambda r: r.created)
        if not kids:
            return f"nothing tried from {cid} yet"
        out = [f"already tried from {cid}:"]
        for r in kids:
            out.append(f"  [{r.cid}] {r.kind} outcome={r.outcome} reusable={r.reusable}"
                       + (f" stage={r.stage}" if r.stage else "") + f": {r.action or r.label}")
            if r.note:
                out.append(f"     note: {r.note}")
            if r.state_diff:
                out.append(f"     changed: {r.state_diff}")
            if r.code_name:
                out.append(f"     code: {r.code_name} ({self.root / (r.cid + '.code.py')})")
            if verbose:
                out.append(f"     log:\n" + self.get_log(r.cid))
        return "\n".join(out)

    def show(self) -> str:
        """The plan with its boundaries, then the whole tree (current node marked)."""
        out = [self.plan_text(), ""]
        roots = sorted((c for c, r in self.records.items() if r.kind == "checkpoint" and r.parent is None),
                       key=lambda c: self.records[c].created)

        def walk(cid: str, depth: int) -> None:
            r = self.records[cid]
            pad = "  " * depth
            out.append(f"{pad}[{cid}] {r.label}" + (f"  <stage {r.stage}>" if r.stage else "")
                       + ("  [SUCCESS]" if r.success else "") + ("  <- current" if cid == self.current else ""))
            out.append(f"{pad}      health: {health_summary(r.health)}")
            if r.action and r.action != r.label:
                out.append(f"{pad}      action: {r.action}")
            if r.state_diff:
                out.append(f"{pad}      changed: {r.state_diff}")
            if r.scene_diff:
                out.append(f"{pad}      visual: {r.scene_diff}")
            if r.note:
                out.append(f"{pad}      note: {r.note}")
            if r.snapshot:
                out.append(f"{pad}      snapshot: {r.snapshot}")
            atts = [a for a in self.records.values() if a.kind == "attempt" and a.parent == cid]
            if atts:
                out.append(f"{pad}      failed attempts from here: " + "; ".join(f"[{a.cid}] {a.action} ({a.outcome})" for a in atts))
            for ch in sorted((c for c, x in self.records.items() if x.kind == "checkpoint" and x.parent == cid),
                             key=lambda c: self.records[c].created):
                walk(ch, depth + 1)

        for c in roots:
            walk(c, 0)
        return "\n".join(out)

    def get_log(self, cid: str) -> str:
        p = self.root / f"{cid}.log.txt"
        return p.read_text() if p.is_file() else ""

    def attach_viewer(self, viewer) -> None:
        """Every save also captures a snapshot PNG through this scene_view Viewer."""
        self._viewer = viewer

    def health(self) -> dict[str, Any]:
        h = checkpoint_health(self.env)
        self._say(f"health live: {health_summary(h)}")
        return h

    def __len__(self) -> int:
        return sum(r.kind == "checkpoint" for r in self.records.values())

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[checkpoint_tree] {msg}", flush=True)


class _Tee:
    """Capture stdout into a buffer while still printing it."""
    def __init__(self, buf: io.StringIO):
        self.buf = buf

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._orig
        return False

    def write(self, s: str) -> int:
        self.buf.write(s)
        return self._orig.write(s)

    def flush(self) -> None:
        self._orig.flush()
