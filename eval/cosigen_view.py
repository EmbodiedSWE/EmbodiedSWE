"""Per-env oracle view + restorable-state tensor utilities (shared by the API
classes in cosigen_apis and the parameter-search engine in cosigen_opt)."""
from __future__ import annotations

import copy

import numpy as np
import torch


# --------------------------- assembly control API ---------------------------
def _q_wxyz(t):
    return np.asarray(t, dtype=np.float64).reshape(-1)[:4]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(float).copy()



def zero_root_velocities(tree) -> None:
    """Zero the velocity columns (7:13) of every root-state tensor (last dim 13 =
    pos3+quat4+vel6) in a state tree, in place. Restore/reset hygiene (2026-07-26):
    a state captured near motion carries residual velocities, and writing it into
    many envs at once seeds simultaneous jitter — the measured trigger of the
    pathological PhysX slow state (search generations at 1.6-2.5 s/step; the leaked
    residue then ran v19_full's ordinary turns 56-59 at 0.85-3.0 s/step). Protocol
    anchors are end-of-program states, so zeroing preserves their semantics."""
    if isinstance(tree, dict):
        for v in tree.values():
            zero_root_velocities(v)
    elif (torch.is_tensor(tree) and tree.dim() >= 2 and tree.shape[-1] == 13
          and tree.dtype.is_floating_point):
        tree[..., 7:13] = 0.0


def _clip_words(text: str, max_words: int = 40) -> str:
    """NO-OP by user directive (2026-07-22): zero truncation, ever. Length discipline
    comes from the annotator guidance (<= 20 words), not from cutting stored text.
    Kept as a seam so every caption/label writer funnels through one place."""
    del max_words
    return str(text)

def expand_state_tree(tree, n: int):
    """Deep-copy a get_states() tree captured for ONE env, repeated to n envs (dim0 1->n).
    Used by snapshot restore + the RL trainer's start-state broadcast."""
    if isinstance(tree, dict):
        return {k: expand_state_tree(v, n) for k, v in tree.items()}
    if torch.is_tensor(tree):
        assert tree.shape[0] == 1, f"snapshot tensor dim0 != 1 ({tuple(tree.shape)})"
        return tree.repeat(n, *([1] * (tree.dim() - 1))).clone()
    if isinstance(tree, list):
        # per-env list state (e.g. whiteboard's words): repeat the single-env row so
        # scene.set_state can index one entry per env (was an IndexError with n>1).
        assert len(tree) == 1, f"snapshot list len != 1 ({len(tree)})"
        return [copy.deepcopy(tree[0]) for _ in range(n)]
    return tree


def _overwrite_replica_rows(dst, src) -> None:
    """Copy rows 1.. of every per-env leaf in `src` over `dst` in place (row 0 keeps
    dst's env-0 state). Used by replica parking: on goto/restore, env 0 gets the
    checkpoint state while replicas keep their quiescent parked state instead of a
    broadcast copy that would leave 511 arms in live (never-sleeping) grasp contact."""
    if isinstance(dst, dict) and isinstance(src, dict):
        for k in dst:
            if k in src:
                _overwrite_replica_rows(dst[k], src[k])
    elif torch.is_tensor(dst) and torch.is_tensor(src) and dst.shape == src.shape:
        dst[1:] = src[1:]
    elif isinstance(dst, list) and isinstance(src, list) and len(dst) == len(src):
        dst[1:] = copy.deepcopy(src[1:])


class AssemblyView:
    """Per-env READ-ONLY oracle view (the optimize tool's objective(v) receives one of
    these). Everything returns numpy copies. `i` is the env index in the batch."""

    def __init__(self, api: "AssemblyApi", i: int):
        self._api = api
        self._i = int(i)
        self.last_action: np.ndarray | None = None  # previous policy action, [-1,1]^D

    # -- objects --
    def object_pose(self, name: str):
        """(pos_xyz(3,), quat_wxyz(4,)) of 'leg_i' / 'stud_i' / 'table', WORLD frame."""
        s = self._api._object_root(name)  # (N,13) or synthetic
        return _np(s[self._i, :3]), _np(s[self._i, 3:7])

    def object_vel(self, name: str):
        """(6,) linear+angular world velocity of the object."""
        return _np(self._api._object_root(name)[self._i, 7:13])

    # -- robot --
    def _wrist_state(self, arm: int):
        """Wrist body state row for `arm`, MultiRobot-aware. Bimanual apis keep a LIST
        of articulations (_arts) with a per-articulation wrist index (_wrist); reading
        api.art (the single primary articulation) with per-arm indices silently returns
        the WRONG body there — that made eef-based measurements static garbage on
        bimanual scenes (found 2026-07-24)."""
        api = self._api
        arts = getattr(api, "_arts", None)
        if arts is not None:
            return arts[arm].data.body_link_state_w[self._i, api._wrist[arm]]
        return api.art.data.body_link_state_w[self._i, api._wrist_idx[arm]]

    def eef_pose(self, arm: int = 0):
        s = self._wrist_state(arm)
        return _np(s[:3]), _np(s[3:7])

    def eef_vel(self, arm: int = 0):
        return _np(self._wrist_state(arm)[7:13])

    def joint_pos(self, names=None):
        ids = self._api._resolve_joint_ids(names)
        return _np(self._api.art.data.joint_pos[self._i, ids])

    def joint_vel(self, names=None):
        ids = self._api._resolve_joint_ids(names)
        return _np(self._api.art.data.joint_vel[self._i, ids])

    def hand_pos(self, arm: int | None = None):
        """Positions of the hand joints (the gripper DOFs). On multi-robot apis pass
        `arm` for THAT arm's gripper; the argless form keeps the legacy behavior
        (primary articulation's hand — which on bimanual apis is arm 0 only)."""
        api = self._api
        arts = getattr(api, "_arts", None)
        if arm is not None and arts is not None and hasattr(api, "_hids"):
            return _np(arts[arm].data.joint_pos[self._i, api._hids[arm]])
        return _np(api.art.data.joint_pos[self._i, api._hand_ids])

    # -- control targets / task --
    def targets(self, arm: int = 0):
        """The CURRENT wrist target pose (7,) the controller is tracking for `arm`."""
        tgt = self._api._tgt[arm][self._i].clone()
        if getattr(self._api, "_targets_env0_world", False):
            tgt[:3] += self._api.origin[self._i] - self._api.origin[0]
        return _np(tgt)

    def hand_frac(self, arm: int | None = None) -> float:
        """Current commanded hand closure in [0,1] (pass `arm` on multi-robot apis)."""
        api = self._api
        if arm is not None and hasattr(api, "_hfrac"):
            return float(api._hfrac[arm][self._i])
        return float(api._hand_frac[self._i])

    def seated(self):
        return [bool(v) for v in self._api.scene.seated()[self._i].tolist()]

    def rich_obs(self, objects=("leg_0",), arms=(0, 1)) -> np.ndarray:
        """Full state summary vector for the named objects and arms. Per object:
        position relative to each arm's wrist (3 x arms), quat (4), lin+ang velocity
        (6). Per arm: wrist linear velocity (3) + height (1). Plus all hand joint
        positions and the commanded closure."""
        parts = []
        eefs = {a: self.eef_pose(a)[0] for a in arms}
        for name in objects:
            p, q = self.object_pose(name)
            for a in arms:
                parts.append(np.asarray(p) - np.asarray(eefs[a]))
            parts.append(np.asarray(q))
            parts.append(np.ravel(self.object_vel(name))[:6])
        for a in arms:
            parts.append(np.ravel(self.eef_vel(a))[:3])
            parts.append([float(eefs[a][2])])
            # commanded-target minus actual wrist position: with a compliant controller this
            # tracking error is a CONTACT-FORCE proxy (how hard the arm is pressing) — key
            # state for contact-rich skills like squeezing/pressing.
            parts.append(np.asarray(self.targets(a))[:3] - np.asarray(eefs[a]))
        # Grippers: PER-ARM finger joints + commanded closure on multi-robot apis
        # (the legacy single hand block silently reported only arm 0's gripper —
        # a right-arm screwing policy could not see its own grip/slip, 2026-07-24).
        if getattr(self._api, "_arts", None) is not None and hasattr(self._api, "_hids"):
            for a in arms:
                parts.append(self.hand_pos(a))
                parts.append([self.hand_frac(a)])
        else:
            parts.append(self.hand_pos())
            parts.append([self.hand_frac()])
        return np.concatenate([np.ravel(x) for x in parts]).astype(np.float32)
