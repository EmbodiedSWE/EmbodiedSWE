"""Checkpoint tree — save a reached world state to disk and start a later run from it.

    from checkpoint_tree import CheckpointTree

    tree = CheckpointTree(env)                     # after env.reset()
    tree.save("grasp holds",                       # -> 'n1'
              action="approached from +x and closed at 0.004",
              program=__file__, log=captured_stdout)
    ...
    tree.goto("n1")                                # in a LATER script, in a fresh process

Why disk and not memory: every script you run is a new process that boots its own simulator,
so a state kept in RAM dies with it. Nodes live under `<root>/` (default
`/workspace/.checkpoints`): `tree.json` (the manifest), `<cid>.pt` (the state),
`<cid>.code.py` (the program that produced it, when given), `<cid>.png` (a snapshot, when a
viewer is attached).

The tree is a tree: `save()` hangs the new node off whichever node is current, so a risky
variation costs nothing already earned — `goto()` the parent and branch again. Every node
carries the original three-field annotation surface:

  action      what was attempted and how it ended (yours; defaults to the label)
  state_diff  what actually changed vs the parent — COMPUTED automatically from the two
              saved states (object movements, joint deltas), so it is never missing
  scene_diff  what changed visually — fill it after LOOKING at the parent/node snapshots
              (`tree.annotate(cid, scene_diff=...)`); with a viewer attached, save()
              captures the snapshot for you

`goto()` and `tried_from()` replay those annotations (plus each branch's code and a
[SUCCESS]/[CRASHED-style] flag), which is what stops a second identical attempt.

State is whatever `env.get_states()` returns, restored with `env.set_states()`. Both calls
are the raw mechanism; a condition that blocks `set_states` cannot grant this tool.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Self-declaration, read by eval/tools/__init__.py::discover(). A plain dict on purpose: this
# file is also installed standalone on the agent's PYTHONPATH, where a relative import of the
# tools package would fail.
TOOL = {
    "name": "checkpoint_tree",
    "description": (
        "Save world states as an annotated tree of nodes on disk and restore any of them in "
        "a later run (save / goto): each node carries action, an automatically computed "
        "state diff vs its parent, an optional snapshot and the program that reached it."
    ),
    "exports": ("CheckpointTree",),
    "skill": "cosigen-checkpoint-tree",
    "prompt_doc": "tools/checkpoint_tree.md",
    "requires_features": ("set_states",),
}

MANIFEST = "tree.json"


@dataclass
class Node:
    cid: str
    parent: str | None
    depth: int
    label: str
    note: str
    created: float
    children: list[str]
    # the original three-field annotation surface
    action: str = ""
    state_diff: str = ""
    scene_diff: str = ""
    # provenance: the program that produced this state (file name; text at <cid>.code.py),
    # its printed log (stored IN FULL — zero-truncation), snapshot path, success at save time
    code_name: str = ""
    log: str = ""
    snapshot: str = ""
    success: bool | None = None

    def as_json(self) -> dict:
        return {
            "cid": self.cid, "parent": self.parent, "depth": self.depth,
            "label": self.label, "note": self.note, "created": self.created,
            "children": list(self.children),
            "action": self.action, "state_diff": self.state_diff,
            "scene_diff": self.scene_diff, "code_name": self.code_name,
            "log": self.log, "snapshot": self.snapshot, "success": self.success,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Node":
        return cls(cid=d["cid"], parent=d.get("parent"), depth=int(d.get("depth", 0)),
                   label=d.get("label", ""), note=d.get("note", ""),
                   created=float(d.get("created", 0.0)), children=list(d.get("children") or []),
                   action=d.get("action", ""), state_diff=d.get("state_diff", ""),
                   scene_diff=d.get("scene_diff", ""), code_name=d.get("code_name", ""),
                   log=d.get("log", ""), snapshot=d.get("snapshot", ""),
                   success=d.get("success"))


def _walk_state(tree, prefix=""):
    """Yield (dotted_path, leaf) for every tensor and list in a get_states() tree."""
    import torch

    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk_state(v, f"{prefix}.{k}" if prefix else str(k))
    elif torch.is_tensor(tree) or isinstance(tree, list):
        yield prefix, tree


def _short(name: str) -> str:
    return name.replace("scene.", "").replace("robot.", "robot ").replace(".root_state", "")


def state_diff_text(parent_state, new_state, pos_tol: float = 2e-3,
                    val_tol: float = 5e-3, rot_tol_deg: float = 2.0) -> str:
    """Deterministic parent->node diff over env 0, spelled out in full: every changed
    quantity as from -> to with its signed delta, plus an explicit list of what did NOT
    change.

    GENERAL BY CONVENTION, not by embodiment: the only structure this assumes is the state
    dict's own naming — a tensor stored under a `root_state` key is a body pose in Isaac
    Lab's layout (pos xyz + quat wxyz + velocities), the same name-based convention the
    scene state readers/writers use, and is reported as position-per-axis + rotation angle.
    Every OTHER leaf — joint vectors of any width, cloth nodal arrays, particle sets, bool
    flags, int counters, per-env lists — is compared generically element-by-element, so a
    new embodiment or scene never breaks the diff; at worst an exotic quantity is reported
    as plain element changes instead of task words. Prose judgment stays with the agent;
    the numbers never go missing or stay vague."""
    import math

    import torch

    parent = dict(_walk_state(parent_state))
    lines: list[str] = []
    unchanged: list[str] = []

    def pose_line(name: str, a, b) -> tuple[str | None, str]:
        """a, b: (..., >=7) rows for one env — report each body row that moved. When nothing
        clears tolerance, the second value says how much motion the tolerance is hiding, so
        "unchanged" is readable as a display grouping, never as a claim of exact stillness."""
        rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        body_lines = []
        max_dist = max_ang = 0.0
        for r in range(rows_a.shape[0]):
            p0, p1 = rows_a[r, 0:3], rows_b[r, 0:3]
            dp = p1 - p0
            dist = float(torch.linalg.norm(dp))
            dot = min(1.0, abs(float((rows_a[r, 3:7] * rows_b[r, 3:7]).sum())))
            ang = math.degrees(2.0 * math.acos(dot))
            max_dist, max_ang = max(max_dist, dist), max(max_ang, ang)
            parts = []
            if dist > pos_tol:
                axes = ", ".join(f"{ax} {p0[i]:+.3f}->{p1[i]:+.3f} ({dp[i]:+.3f})"
                                 for i, ax in enumerate("xyz")
                                 if abs(float(dp[i])) > pos_tol / 2)
                parts.append(f"pos ({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})->"
                             f"({p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f}), moved {dist:.3f}m"
                             + (f" [{axes}]" if axes else ""))
            if ang > rot_tol_deg:
                parts.append(f"rotated {ang:.1f} deg")
            if parts:
                tag = f" body[{r}]" if rows_a.shape[0] > 1 else ""
                body_lines.append(f"{name}{tag}: " + "; ".join(parts))
        if body_lines:
            return "; ".join(body_lines), ""
        hidden = []
        if max_dist > 0:
            hidden.append(f"moved <={max_dist * 1000:.2g}mm")
        if max_ang > 0:
            hidden.append(f"rotated <={max_ang:.2g}deg")
        return None, ", ".join(hidden)

    def element_line(name: str, a, b) -> tuple[str | None, str]:
        """Generic per-element diff for numeric/bool tensors of any shape."""
        fa, fb = a.reshape(-1), b.reshape(-1)
        if fa.dtype.is_floating_point:
            delta = fb.float() - fa.float()
            idx = (delta.abs() > val_tol).nonzero().reshape(-1).tolist()
            fmt = lambda i: f"[{i}] {fa[i]:+.3f}->{fb[i]:+.3f} ({delta[i]:+.3f})"  # noqa: E731
        else:  # bool / int state: any difference counts, shown verbatim
            idx = (fa != fb).nonzero().reshape(-1).tolist()
            fmt = lambda i: f"[{i}] {fa[i].item()}->{fb[i].item()}"  # noqa: E731
        if not idx:
            hidden = ""
            if fa.dtype.is_floating_point:
                m = float((fb.float() - fa.float()).abs().max()) if fa.numel() else 0.0
                if m > 0:
                    hidden = f"max delta {m:.2g}"
            return None, hidden
        # EVERY changed element, no cap: an 8-element display limit (removed 2026-08-01, user
        # decision) was an unapproved truncation of the record.
        shown = ", ".join(fmt(i) for i in idx)
        return (f"{name}: {shown}; "
                f"{fa.numel() - len(idx)} of {fa.numel()} elements unchanged"), ""

    for path, leaf_new in _walk_state(new_state):
        leaf_old = parent.get(path)
        name = _short(path)
        if leaf_old is None:
            lines.append(f"{name}: appeared in this state (not in parent)")
            continue
        if isinstance(leaf_new, list):
            old0 = leaf_old[0] if isinstance(leaf_old, list) and leaf_old else None
            new0 = leaf_new[0] if leaf_new else None
            if old0 == new0:
                unchanged.append(name)
            else:
                lines.append(f"{name}: {old0!r} -> {new0!r}")
            continue
        if not (hasattr(leaf_old, "shape") and leaf_old.shape == leaf_new.shape):
            lines.append(f"{name}: shape changed "
                         f"{getattr(leaf_old, 'shape', '?')} -> {tuple(leaf_new.shape)}")
            continue
        a, b = leaf_old[0], leaf_new[0]
        # Pose semantics BY NAME (the state dict's own convention), with a layout guard so a
        # misnamed or restructured asset degrades to the generic path instead of misparsing.
        if (path.endswith("root_state") and leaf_new.dtype.is_floating_point
                and leaf_new.shape[-1] >= 7):
            line, hidden = pose_line(name, a.float(), b.float())
        else:
            line, hidden = element_line(name, a, b)
        if line:
            lines.append(line)
        else:
            unchanged.append(f"{name} ({hidden})" if hidden else name)
    if not lines:
        return ("nothing changed beyond tolerance; unchanged: "
                + (", ".join(unchanged) if unchanged else "(no comparable state)"))
    if unchanged:
        lines.append("unchanged: " + ", ".join(unchanged))
    return "; ".join(lines)


class CheckpointTree:
    """A disk-backed, annotated tree of saved simulator states for one task.

    Construct it with a live env after `env.reset()`. Reopening it in a later process picks
    up the same tree from disk, including which node was current. `attach_viewer(viewer)`
    (a scene_view.Viewer) makes every save also capture a per-node snapshot PNG.
    """

    def __init__(self, env, root: str | Path = "/workspace/.checkpoints", *, verbose: bool = True):
        self.env = env
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.nodes: dict[str, Node] = {}
        self.current: str | None = None
        self._seq = 0
        self._viewer = None
        self._load_manifest()

    def attach_viewer(self, viewer) -> None:
        """Capture a snapshot PNG for every node saved from now on (parent/node snapshots
        are the raw material for `scene_diff` — look at both, then `annotate`)."""
        self._viewer = viewer

    # ----- persistence ---------------------------------------------------------------------
    def _load_manifest(self) -> None:
        path = self.root / MANIFEST
        if not path.is_file():
            return
        try:
            d = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 -- a corrupt tree must be loud, not silent
            print(f"[checkpoint_tree] {path} is unreadable ({exc!r}); starting a new tree")
            return
        self.nodes = {cid: Node.from_json(n) for cid, n in (d.get("nodes") or {}).items()}
        self.current = d.get("current")
        self._seq = int(d.get("seq", len(self.nodes)))
        self._say(f"reopened {len(self.nodes)} node(s) from {self.root}; current={self.current}")

    def _save_manifest(self) -> None:
        payload = {
            "nodes": {cid: n.as_json() for cid, n in self.nodes.items()},
            "current": self.current,
            "seq": self._seq,
            "saved_at": time.time(),
        }
        (self.root / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")

    def _state_path(self, cid: str) -> Path:
        return self.root / f"{cid}.pt"

    # ----- the operations that matter ------------------------------------------------------
    def save(self, label: str, note: str = "", *, action: str = "",
             program: str | Path | None = None, log: str = "") -> str:
        """Save the CURRENT world state as a child of the current node. Returns its id.

        label    the STATE reached, in the task's own terms ("part_0 secured in its mount",
                 "cloth folded over the crease"), not the action attempted
        note     measurements worth keeping next to the state
        action   what was attempted and how it ended (defaults to the label)
        program  path of the script that produced this state; its text is copied to
                 <cid>.code.py so the node's edge is exactly that program
        log      the printed output of that run, stored in full
        """
        import torch

        self._seq += 1
        cid = f"n{self._seq}"
        parent = self.current
        state = self.env.get_states()
        torch.save(state, self._state_path(cid))

        # annotation: automatic numeric state diff vs the parent
        diff = ""
        if parent and self._state_path(parent).is_file():
            try:
                parent_state = torch.load(self._state_path(parent), weights_only=False)
                diff = state_diff_text(parent_state, state)
            except Exception as exc:  # noqa: BLE001 -- a diff bug must not lose the save
                diff = f"(state diff failed: {exc!r})"

        code_name = ""
        if program:
            src = Path(program)
            if src.is_file():
                shutil.copyfile(src, self.root / f"{cid}.code.py")
                code_name = src.name
            else:
                print(f"[checkpoint_tree] program {src} not found; node saved without code",
                      flush=True)

        snapshot = ""
        if self._viewer is not None:
            try:
                snapshot = self._viewer.snapshot(f"ckpt_{cid}_{label}")
            except Exception as exc:  # noqa: BLE001 -- a snapshot bug must not lose the save
                print(f"[checkpoint_tree] snapshot failed: {exc!r}", flush=True)

        success = None
        try:
            scene = getattr(self.env, "scene", None)
            if scene is not None and hasattr(scene, "success"):
                success = bool(scene.success()[0].item())
        except Exception:  # noqa: BLE001 -- optional flag only
            success = None

        self.nodes[cid] = Node(
            cid=cid, parent=parent,
            depth=(self.nodes[parent].depth + 1) if parent and parent in self.nodes else 0,
            label=label, note=note, created=time.time(), children=[],
            action=action or label, state_diff=diff, code_name=code_name,
            log=str(log or ""), snapshot=str(snapshot or ""), success=success,
        )
        if parent and parent in self.nodes:
            self.nodes[parent].children.append(cid)
        self.current = cid
        self._save_manifest()
        self._say(f"saved {cid} '{label}'" + (f" (parent {parent})" if parent else " (root)")
                  + (" [SUCCESS]" if success else ""))
        # scene_diff is the agent's to write, and BOTH images it needs exist right now:
        # this node's snapshot was just captured, the parent's at its own save. Hand over
        # the exact paths and the exact call, so "look, then describe" costs one step.
        parent_snap = self.nodes[parent].snapshot if parent and parent in self.nodes else ""
        if snapshot and parent_snap:
            self._say(f"scene_diff: LOOK at {parent_snap} (parent) vs {snapshot} (this), "
                      f"then tree.annotate('{cid}', scene_diff='what visibly changed')")
        return cid

    def goto(self, cid: str) -> str:
        """Restore the world to a saved node and make it current.

        Returns `tried_from(cid)` — every branch already attempted from there, with its
        action, what it changed, its visual diff and its code. Read it before re-attempting
        anything: repeating a branch that already failed the same way is the most common way
        to waste a run."""
        import torch

        if cid not in self.nodes:
            raise KeyError(f"no checkpoint '{cid}'; have {sorted(self.nodes)}")
        path = self._state_path(cid)
        if not path.is_file():
            raise FileNotFoundError(f"{cid} has no saved state at {path}")
        self.env.set_states(torch.load(path, weights_only=False))
        self.current = cid
        self._save_manifest()
        self._say(f"world restored to {cid} '{self.nodes[cid].label}'")
        return self.tried_from(cid)

    def annotate(self, cid: str, *, action: str | None = None,
                 state_diff: str | None = None, scene_diff: str | None = None,
                 note: str | None = None) -> None:
        """Fill or refine a node's annotations — in particular `scene_diff` after you have
        looked at the parent's and the node's snapshots."""
        node = self.nodes.get(cid)
        if node is None:
            raise KeyError(f"no checkpoint '{cid}'")
        if action is not None:
            node.action = action
        if state_diff is not None:
            node.state_diff = state_diff
        if scene_diff is not None:
            node.scene_diff = scene_diff
        if note is not None:
            node.note = note
        self._save_manifest()

    # ----- reading the tree ----------------------------------------------------------------
    def tried_from(self, cid: str | None = None) -> str:
        """Every branch already attempted from a node — action, state diff, visual diff and
        code — one block per child, in the original digest layout."""
        cid = cid or self.current
        if cid not in self.nodes:
            return ""
        kids = self.nodes[cid].children
        if not kids:
            return f"nothing tried from {cid} yet"
        blocks = [f"already tried from {cid}:"]
        for ch in kids:
            n = self.nodes[ch]
            flags = " [SUCCESS]" if n.success else ""
            head = f"  [{ch}] {n.action or n.label}{flags}"
            detail = ""
            if n.state_diff:
                detail += f"\n     changed: {n.state_diff}"
            if n.scene_diff:
                detail += f"\n     visual: {n.scene_diff}"
            if n.note:
                detail += f"\n     note: {n.note}"
            code_path = self.root / f"{ch}.code.py"
            if code_path.is_file():
                code = code_path.read_text().strip()
                indented = "\n".join("       " + ln for ln in code.splitlines())
                blocks.append(f"{head}{detail}\n     code ({n.code_name}):\n{indented}")
            else:
                blocks.append(f"{head}{detail}")
        return "\n".join(blocks)

    def show(self) -> str:
        """The whole tree with each node's annotations, current node marked — the same
        surface the original live tree view printed."""
        if not self.nodes:
            return "(no checkpoints yet)"
        roots = [c for c, n in self.nodes.items() if not n.parent]
        out: list[str] = []

        def walk(cid: str) -> None:
            n = self.nodes[cid]
            mark = " <- current" if cid == self.current else ""
            flag = " [SUCCESS]" if n.success else ""
            pad = "  " * n.depth
            out.append(f"{pad}[{cid}] {n.label}{flag}{mark}")
            if n.action and n.action != n.label:
                out.append(f"{pad}      action: {n.action}")
            if n.state_diff:
                out.append(f"{pad}      state diff: {n.state_diff}")
            if n.scene_diff:
                out.append(f"{pad}      scene diff: {n.scene_diff}")
            if n.note:
                out.append(f"{pad}      note: {n.note}")
            if n.snapshot:
                out.append(f"{pad}      snapshot: {n.snapshot}")
            for ch in n.children:
                walk(ch)

        for r in sorted(roots, key=lambda c: int(c[1:]) if c[1:].isdigit() else 0):
            walk(r)
        return "\n".join(out)

    def get_log(self, cid: str) -> str:
        """The full printed log stored with a node (never truncated)."""
        node = self.nodes.get(cid)
        return node.log if node else ""

    def __len__(self) -> int:
        return len(self.nodes)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[checkpoint_tree] {msg}", flush=True)
