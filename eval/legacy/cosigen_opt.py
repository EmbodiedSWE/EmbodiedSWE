"""Evolutionary parameter search over an agent-written maneuver.

The agent supplies the STRUCTURE (its ordinary single-env program file, plus which of
its constants are free and an objective file). This module supplies the search: it
replicates the current world state across every env, substitutes a different candidate
value for the named constants in each copy, runs all the copies of the agent's program
in LOCKSTEP against the shared physics, scores each end state, and adapts the sampling
distribution. There are no analytic gradients through the physics, so the search is
derivative-free (evolutionary / coordinate).

A solver of this kind is a phase machine carrying 15-55 tuned constants, numbers that
have historically come out of multi-day human tuning campaigns. Finding them is
exactly what this module automates.

THE interface (the agent's mental model — nothing batched to write; 2026-07-25):
  optimize_program(env, api, program_src=..., objective_src=..., space=...)
    program_src   the agent's OWN single-env program, unchanged; space names its
                  top-level constants. Every env runs a copy via a per-env toolkit
                  facade; a step-barrier scheduler advances the one global physics
                  clock only when every live copy has declared its step request, so
                  copies may branch/retry/finish independently (see _Lockstep).
    objective_src a small module defining  objective(v) -> float  (lower is better),
                  evaluated on each env's END state via its oracle view.
The agent reaches this through ONE tool named `optimize` (driver-side: it ships the
two workspace files here). The old batched program(P, views) interface and the
sequential single-env search were removed 2026-07-26 (user decision): parallel
search over the agent's own file is strictly better than both.
"""
from __future__ import annotations

import ast
import threading
import time
import traceback as _tb

import numpy as np
import torch


# --------------------------- start-state replication ---------------------------
def shift_root_states(tree, delta: torch.Tensor) -> None:
    """Add per-env origin deltas (n,3) to the POSITION columns of every root-state tensor
    (last dim 13 = pos3+quat4+vel6) in an expanded state tree, in place.

    Scene/robot get_state trees hold WORLD-frame root states and the env replicas live on
    an origin grid, so broadcasting env 0's state verbatim parks every replica's objects in
    env 0's cell — each replica's arm is then tens of meters from its objects and every
    candidate but one is born broken. Re-offsetting positions gives each replica a faithful
    copy of env 0's state in its own cell."""
    if isinstance(tree, dict):
        for v in tree.values():
            shift_root_states(v, delta)
    elif (torch.is_tensor(tree) and tree.dim() >= 2 and tree.shape[-1] == 13
          and tree.dtype.is_floating_point):
        d = delta.view(delta.shape[0], *([1] * (tree.dim() - 2)), 3).to(tree.device)
        tree[..., 0:3] += d


def jitter_scene(state_n: dict, randomize: dict | None, gen: torch.Generator) -> None:
    """Declarative start-state jitter, in place: {"leg_2": {"pos": 0.01, "yaw": 0.3}} —
    sigma of xy noise (m) and yaw noise (rad) per object. Use it to tune parameters that
    must hold up under start-state variation rather than for one exact pose."""
    if not randomize:
        return
    scene = state_n["scene"]

    def _first_tensor(tree):
        if torch.is_tensor(tree):
            return tree
        if isinstance(tree, dict):
            for v in tree.values():
                t = _first_tensor(v)
                if t is not None:
                    return t
        return None

    n = _first_tensor(scene).shape[0]

    def _apply(root: torch.Tensor, cfg: dict) -> None:
        s_pos, s_yaw = float(cfg.get("pos", 0.0)), float(cfg.get("yaw", 0.0))
        s_z = float(cfg.get("z", 0.0))
        if s_pos > 0:
            root[:, :2] += torch.randn(n, 2, generator=gen, device=root.device) * s_pos
        if s_z > 0:
            root[:, 2] += torch.rand(n, generator=gen, device=root.device) * s_z
        if s_yaw > 0:
            ang = torch.randn(n, generator=gen, device=root.device) * s_yaw
            half = ang * 0.5
            qz = torch.zeros(n, 4, device=root.device)
            qz[:, 0], qz[:, 3] = torch.cos(half), torch.sin(half)
            w1, x1, y1, z1 = qz[:, 0], qz[:, 1], qz[:, 2], qz[:, 3]
            w2, x2, y2, z2 = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
            root[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            root[:, 4] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            root[:, 5] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            root[:, 6] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    for name, cfg in randomize.items():
        for key, sub in scene.items():
            if key != name:
                continue
            root = sub.get("root_state") if isinstance(sub, dict) else None
            if torch.is_tensor(root):
                _apply(root, cfg)


# --------------------------- search space ---------------------------
class Space:
    """Named scalar parameters with bounds, mapped to/from a normalized [0,1] cube.

    space = {"press_dz": (0.001, 0.01),               # float
             "settle_steps": (5, 60, "int"),           # rounded to int
             "kp": (10.0, 1000.0, "log"),              # searched in log space
             "approach_deg": ("choices", (45, 90, 135))}  # a fixed set of values
    """

    def __init__(self, space: dict, fix: dict | None = None):
        if not space:
            raise ValueError("space is empty: give at least one parameter with bounds")
        self.fix = dict(fix or {})
        self.names, self.lo, self.hi, self.kind = [], [], [], []
        self.choices: dict[str, list] = {}
        for name, spec in space.items():
            if name in self.fix:
                continue
            if str(spec[0]) == "choices":
                values = list(spec[1])
                if len(values) < 2:
                    raise ValueError(f"{name}: give at least two choices")
                self.names.append(name)
                self.choices[name] = values
                # searched as an index on the normalized axis
                self.lo.append(0.0)
                self.hi.append(float(len(values) - 1))
                self.kind.append("choices")
                continue
            lo, hi = float(spec[0]), float(spec[1])
            kind = str(spec[2]) if len(spec) > 2 else "float"
            if not hi > lo:
                raise ValueError(f"{name}: upper bound must exceed lower ({lo}, {hi})")
            if kind == "log" and lo <= 0:
                raise ValueError(f"{name}: log-scale bounds must be positive")
            self.names.append(name)
            self.lo.append(np.log(lo) if kind == "log" else lo)
            self.hi.append(np.log(hi) if kind == "log" else hi)
            self.kind.append(kind)
        self.lo = np.asarray(self.lo, dtype=np.float64)
        self.hi = np.asarray(self.hi, dtype=np.float64)
        self.dim = len(self.names)

    def denorm(self, u: np.ndarray) -> np.ndarray:
        """(m,dim) in [0,1] -> real parameter values (m,dim). 'choices' columns carry
        the rounded INDEX (as_dict maps it to the actual value)."""
        x = self.lo + np.clip(u, 0.0, 1.0) * (self.hi - self.lo)
        out = x.copy()
        for j, kind in enumerate(self.kind):
            if kind == "log":
                out[:, j] = np.exp(x[:, j])
            elif kind in ("int", "choices"):
                out[:, j] = np.round(x[:, j])
        return out

    def norm_values(self, values: dict) -> np.ndarray | None:
        """Real parameter values -> a (dim,) point in the normalized cube (the seed
        for the incumbent candidate). None when any searched name is missing."""
        u = np.zeros(self.dim)
        for j, (name, kind) in enumerate(zip(self.names, self.kind)):
            if name not in values:
                return None
            v = values[name]
            if kind == "choices":
                vals = self.choices[name]
                idx = vals.index(v) if v in vals else int(
                    np.argmin([abs(float(c) - float(v)) for c in vals]))
                x = float(idx)
            elif kind == "log":
                if float(v) <= 0:
                    return None
                x = np.log(float(v))
            else:
                x = float(v)
            span = self.hi[j] - self.lo[j]
            u[j] = float(np.clip((x - self.lo[j]) / span, 0.0, 1.0)) if span > 0 else 0.5
        return u

    def as_dict(self, row: np.ndarray) -> dict:
        d = {}
        for n, v, k in zip(self.names, row, self.kind):
            if k == "choices":
                vals = self.choices[n]
                d[n] = vals[int(np.clip(round(v), 0, len(vals) - 1))]
            elif k == "int":
                d[n] = int(v)
            else:
                d[n] = float(v)
        d.update(self.fix)
        return d

# --------------------------- proposal distribution (CMA-ES) ---------------------------
class _CMAES:
    """CMA-ES over the normalized [0,1] cube — the standard derivative-free optimizer
    for continuous problems of this size (Hansen's formulation: rank-weighted
    recombination, cumulative step-size adaptation, rank-mu + rank-one covariance
    update). It replaced a plain cross-entropy sampler (2026-07-26) because CEM models
    no parameter correlations and has no principled step-size control: our objectives
    are exactly the correlated kind (a grasp offset trades off against approach angle).

    Bounds: samples are clipped into the cube, and `tell` learns from the CLIPPED
    points, so the distribution never drifts outside the box it can actually sample.
    Integer / 'choices' axes are rounded by Space.denorm; the sigma floor keeps their
    grid reachable instead of collapsing between two levels."""

    def __init__(self, dim: int, rng: np.random.Generator, popsize: int,
                 sigma0: float = 0.3, sigma_floor: float = 0.02):
        self.dim = int(dim)
        self.rng = rng
        self.lam = max(4, int(popsize))
        self.sigma = float(sigma0)
        self.sigma_floor = float(sigma_floor)
        self.mean = np.full(self.dim, 0.5)

        # rank-based recombination weights (mu best of lam)
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)

        n = self.dim
        # standard adaptation constants
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self._eigen_gen = 0

    # ---- sampling ----
    def _update_eigen(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # enforce symmetry
        vals, vecs = np.linalg.eigh(self.C)
        vals = np.maximum(vals, 1e-20)
        self.D = np.sqrt(vals)
        self.B = vecs
        self._eigen_gen = self.gen

    def ask(self, n: int) -> np.ndarray:
        if self.gen - self._eigen_gen > max(1, self.lam // (10 * max(self.dim, 1))):
            self._update_eigen()
        z = self.rng.standard_normal((n, self.dim))
        y = z @ (self.B * self.D).T            # correlated step, unit sigma
        u = self.mean[None, :] + self.sigma * y
        return np.clip(u, 0.0, 1.0)

    # ---- adaptation ----
    def tell(self, u: np.ndarray, scores: np.ndarray) -> None:
        finite = np.isfinite(scores)
        if finite.sum() < 2:
            return
        u, scores = np.clip(u[finite], 0.0, 1.0), scores[finite]
        order = np.argsort(scores)               # minimization
        mu = min(self.mu, len(order))
        w = self.w[:mu] / self.w[:mu].sum()
        elite = u[order[:mu]]

        old_mean = self.mean.copy()
        self.mean = w @ elite
        if self.sigma <= 0:
            return
        y_w = (self.mean - old_mean) / self.sigma

        # cumulative step-size adaptation
        C_invsqrt_yw = self.B @ ((self.B.T @ y_w) / self.D)
        mueff = 1.0 / np.sum(w ** 2)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * mueff) * C_invsqrt_yw
        self.gen += 1
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                / self.chiN) < (1.4 + 2 / (self.dim + 1))
        self.pc = (1 - self.cc) * self.pc + (
            np.sqrt(self.cc * (2 - self.cc) * mueff) * y_w if hsig else 0.0)

        # covariance: rank-one (evolution path) + rank-mu (elite steps)
        ys = (elite - old_mean[None, :]) / self.sigma
        rank_mu = (ys * w[:, None]).T @ ys
        c1a = self.c1 * (0.0 if hsig else self.cc * (2 - self.cc))
        self.C = ((1 - self.c1 - self.cmu + c1a) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * rank_mu)

        self.sigma = float(max(
            self.sigma_floor,
            self.sigma * np.exp((self.cs / self.damps)
                                * (np.linalg.norm(self.ps) / self.chiN - 1))))


# --------------------------- program-path (lockstep) search ---------------------------
class _EvalAborted(RuntimeError):
    """Raised inside a candidate thread when the per-eval step budget is exhausted:
    the candidate stops where it is and its end state is scored as-is."""


class _CandidateEvicted(_EvalAborted):
    """Raised inside a candidate thread whose env was evicted as a physics-work
    outlier (see _Lockstep.work_outliers). Subclasses _EvalAborted so the existing
    'stop and score where you are' path applies unchanged."""


class _Lockstep:
    """Step-barrier scheduler: K candidate threads run the agent's single-env program
    against per-env facades; a facade call that needs physics BLOCKS in request(), and
    the one global physics step advances only when every live thread is either blocked
    in a request or finished. Divergent control flow across candidates is therefore
    fine — each copy consumes the shared clock at its own pace, and copies that finish
    early simply hold their targets while the rest continue.

    ALL GPU traffic is batched here (2026-07-26 perf fix): candidate threads compute
    their control math in numpy against a per-step CPU snapshot and QUEUE writes; the
    scheduler applies every queued target/hand write in a handful of tensor ops, steps
    physics once, then bulk-fetches the state everyone reads. Measured on the 512-env
    L20 grasp-contact benchmark: engine overhead is ~66 ms/step (gate 61, apply 2,
    snapshot 2, wake 1) against 1870 ms/step of raw PhysX contact solving — i.e. after
    this fix the engine is ~3% of the bill and the physics itself dominates. Free-space
    searches run at ~240 ms/step total."""

    def __init__(self, api, max_total_steps: int):
        self.api = api
        self.max_total = int(max_total_steps)
        self.cond = threading.Condition()
        self.active: set[int] = set()
        self.finished: set[int] = set()
        self.pending: dict[int, int] = {}  # tid -> steps still owed
        self.total = 0
        self.abort = False
        # ---- batched-write queues (guarded by self.qlock) ----
        self.qlock = threading.Lock()
        self.q_pos: dict[tuple[int, int], np.ndarray] = {}   # (arm, i) -> (3,) env0-frame
        self.q_quat: dict[tuple[int, int], np.ndarray] = {}  # (arm, i) -> (4,) wxyz
        self.q_hand: dict[tuple[int, int], float] = {}       # (arm, i) -> frac
        # ---- geometry / api shape ----
        d = api.origin - api.origin[0:1]
        self.delta_t = d.detach().to(api.device)             # (N,3) torch
        self.delta = self.delta_t.cpu().numpy()              # (N,3) numpy
        self.env0_world = bool(getattr(api, "_targets_env0_world", False))
        self.arms = list(range(len(api._tgt)))
        self.multi_arm = getattr(api, "_arts", None) is not None
        # CPU mirror of commanded hand closure per arm (only rows we touch are written)
        self.hand_mirror: dict[int, np.ndarray] = {}
        if hasattr(api, "_hfrac"):
            self.hand_mirror = {a: api._hfrac[a].detach().cpu().numpy().copy()
                                for a in self.arms}
        elif hasattr(api, "_hand_frac"):
            self.hand_mirror = {0: api._hand_frac.detach().cpu().numpy().copy()}
        # ---- per-step CPU snapshot ----
        self.obj_names: set[str] = set()
        self.want_seated = False
        self.snap: dict = {}
        # per-phase timing accumulators (seconds), reported per generation
        self.t_apply = self.t_step = self.t_snap = self.t_wake = self.t_gate = 0.0
        # per-step physics wall times (seconds): the SHAPE of a slow generation tells
        # warmup (decaying curve = one-off PhysX buffer growth) from genuinely
        # expensive contact (flat curve). Reported as buckets by optimize_program.
        self.step_times: list[float] = []
        self.refresh_snapshot()

    # ---- thread-side write/read API (numpy in, numpy out) ----
    def queue_target(self, arm: int, i: int, pos=None, quat=None) -> None:
        with self.qlock:
            if pos is not None:
                self.q_pos[(arm, i)] = np.asarray(pos, dtype=np.float64).reshape(3)
            if quat is not None:
                self.q_quat[(arm, i)] = np.asarray(quat, dtype=np.float64).reshape(4)

    def queue_hand(self, arm: int, i: int, frac: float) -> None:
        with self.qlock:
            self.q_hand[(arm, i)] = float(np.clip(frac, 0.0, 1.0))

    def pending_target_pos(self, arm: int, i: int):
        """The row's queued (not yet applied) position, if any — a thread that wrote
        and reads back before stepping must see its own write."""
        with self.qlock:
            p = self.q_pos.get((arm, i))
        return None if p is None else p.copy()

    def obj_row(self, name: str, i: int) -> np.ndarray:
        """(13,) root state of `name` in env i, WORLD frame, from the snapshot. First
        access of a new name fetches it directly (once) and registers it for every
        later per-step snapshot."""
        rows = self.snap.get(("obj", name))
        if rows is None:
            rows = self.api._object_root(name).detach().cpu().numpy()
            self.snap[("obj", name)] = rows
            self.obj_names.add(name)
        return rows[i]

    def tgt_row(self, arm: int, i: int) -> np.ndarray:
        return self.snap[("tgt", arm)][i]

    def eef_row(self, arm: int, i: int) -> np.ndarray:
        return self.snap[("eef", arm)][i]

    def seated_row(self, i: int) -> list[bool]:
        if not self.want_seated:
            self.want_seated = True
            self.snap["seated"] = self.api.scene.seated().detach().cpu().numpy()
        return [bool(v) for v in self.snap["seated"][i]]

    # ---- scheduler-side batched GPU work ----
    def _apply_writes(self) -> None:
        with self.qlock:
            q_pos, self.q_pos = self.q_pos, {}
            q_quat, self.q_quat = self.q_quat, {}
            q_hand, self.q_hand = self.q_hand, {}
        api = self.api
        for arm in self.arms:
            rows = [(i, p) for (a, i), p in q_pos.items() if a == arm]
            if rows:
                idx = torch.tensor([i for i, _ in rows], device=api.device,
                                   dtype=torch.long)
                vals = torch.as_tensor(np.stack([p for _, p in rows]),
                                       dtype=torch.float32, device=api.device)
                if not self.env0_world:
                    vals = vals + self.delta_t[idx]
                api._tgt[arm][idx, :3] = vals
            rows = [(i, q) for (a, i), q in q_quat.items() if a == arm]
            if rows:
                idx = torch.tensor([i for i, _ in rows], device=api.device,
                                   dtype=torch.long)
                vals = torch.as_tensor(np.stack([q for _, q in rows]),
                                       dtype=torch.float32, device=api.device)
                api._tgt[arm][idx, 3:7] = vals
        if q_hand:
            touched_arms = {a for (a, _i) in q_hand}
            for arm in touched_arms:
                mirror = self.hand_mirror.get(arm)
                if mirror is None:
                    continue
                for (a, i), f in q_hand.items():
                    if a == arm:
                        mirror[i] = f
                full = torch.as_tensor(mirror, dtype=torch.float32, device=api.device)
                if hasattr(api, "set_hand_frac_tensor_arm"):
                    api.set_hand_frac_tensor_arm(full, (arm,))
                else:
                    api.set_hand_frac_tensor(full)

    def refresh_snapshot(self) -> None:
        api = self.api
        snap: dict = {}
        for arm in self.arms:
            snap[("tgt", arm)] = api._tgt[arm][:, :7].detach().cpu().numpy()
            if self.multi_arm:
                s = api._arts[arm].data.body_link_state_w[:, api._wrist[arm], :7]
            else:
                s = api.art.data.body_link_state_w[:, api._wrist_idx[arm], :7]
            snap[("eef", arm)] = s.detach().cpu().numpy()
        for name in self.obj_names:
            snap[("obj", name)] = api._object_root(name).detach().cpu().numpy()
        if self.want_seated:
            snap["seated"] = api.scene.seated().detach().cpu().numpy()
        self.snap = snap

    def finish(self, tid: int) -> None:
        with self.cond:
            self.finished.add(tid)
            self.pending.pop(tid, None)
            self.cond.notify_all()

    def request(self, tid: int, n: int) -> None:
        """Candidate thread: 'advance the world n steps with my targets as set'."""
        with self.cond:
            if self.abort:
                raise _EvalAborted("per-eval step budget exhausted")
            self.pending[tid] = int(max(1, n))
            self.cond.notify_all()
            while self.pending.get(tid, 0) > 0 and not self.abort:
                self.cond.wait(timeout=5.0)
            if self.abort:
                raise _EvalAborted("per-eval step budget exhausted")

    def run(self) -> None:
        """Main (physics) thread: step globally whenever every live candidate is ready."""
        while True:
            t0 = time.time()
            with self.cond:
                while True:
                    live = self.active - self.finished
                    if not live:
                        return
                    if all(t in self.pending for t in live):
                        break
                    self.cond.wait(timeout=5.0)
                if self.total >= self.max_total:
                    self.abort = True
                    self.cond.notify_all()
                    return
            t1 = time.time()
            # Physics OUTSIDE the lock: candidate threads are all blocked in request()
            # here, so no facade writes race the step or the snapshot refresh.
            self._apply_writes()
            t2 = time.time()
            self.api.step(1)
            t3 = time.time()
            self.step_times.append(t3 - t2)
            self.refresh_snapshot()
            t4 = time.time()
            self.total += 1
            with self.cond:
                for tid in list(self.pending):
                    self.pending[tid] -= 1
                    if self.pending[tid] <= 0:
                        del self.pending[tid]
                self.cond.notify_all()
            t5 = time.time()
            self.t_gate += t1 - t0
            self.t_apply += t2 - t1
            self.t_step += t3 - t2
            self.t_snap += t4 - t3
            self.t_wake += t5 - t4

    def timing_report(self) -> str:
        n = max(self.total, 1)
        return (f"gate {self.t_gate/n*1000:.0f} apply {self.t_apply/n*1000:.0f} "
                f"physics {self.t_step/n*1000:.0f} snapshot {self.t_snap/n*1000:.0f} "
                f"wake {self.t_wake/n*1000:.0f} ms/step over {self.total} steps")


_SETUP_ADVICE = (
    " Simulator configuration that has to apply to every copy — controller gains are "
    "the usual one — goes in the optimize setup file, which runs once on the search "
    "simulator with the raw handles before the copies start.")


class _Blocked:
    """A name the searched copies may not touch, raising the same explanation however
    it is reached. Attribute access matters as much as the call: raw handles are used
    as `env.robot.robots[...]`, and the placeholder function this replaced produced
    "'function' object has no attribute 'robot'" — which tells the agent nothing about
    what to do instead (measured on the v20 full arm, whose programs all configure OSC
    gains this way, 2026-07-27)."""

    def __init__(self, name: str, advice: str = ""):
        self.__dict__["_name"] = name
        self.__dict__["_advice"] = advice

    def _refuse(self, how: str):
        raise RuntimeError(
            f"{self._name} is not available inside optimize ({how}): every candidate "
            f"is one sandboxed copy sharing the simulator, so a global handle cannot be "
            f"given to it. Use the control toolkit (move_to, step, grippers) and the "
            f"read-only queries in the searched program.{self._advice}")

    def __call__(self, *_a, **_k):
        self._refuse("called it")

    def __getattr__(self, item):
        self._refuse(f"read .{item}")

    def __setattr__(self, item, _value):
        self._refuse(f"assigned .{item}")

    def __getitem__(self, item):
        self._refuse(f"indexed [{item!r}]")


def _blocked(name: str, advice: str = ""):
    return _Blocked(name.split("(")[0], advice)


class _EnvToolkit:
    """Per-env facade of the agent-facing toolkit, used by one candidate thread.

    Geometry contract: the program sees EXACTLY env 0's world. Reads subtract this
    env's origin delta, writes add it back (unless the api already stores env-0-frame
    targets, `_targets_env0_world`), so the same program text is geometrically
    identical in every copy.

    Everything here is plain numpy against the scheduler's per-step snapshot: a
    candidate thread never touches a GPU tensor (see _Lockstep, 2026-07-26 perf fix)."""

    def __init__(self, api, i: int, sched: _Lockstep, tid: int):
        self.api = api
        self.i = int(i)
        self.sched = sched
        self.tid = tid
        self.delta = sched.delta[self.i]  # (3,) numpy world offset of this env's cell

    # ---- physics clock ----
    def step(self, n: int = 1) -> None:
        self.sched.request(self.tid, int(n))

    # ---- control ----
    @staticmethod
    def _np3(value, n: int) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(n)

    def _target_pos(self, arm: int) -> np.ndarray:
        """Commanded target in env-0 frame; a queued (unapplied) write wins."""
        q = self.sched.pending_target_pos(arm, self.i)
        if q is not None:
            return q  # queued values are already env-0 frame
        p = self.sched.tgt_row(arm, self.i)[:3].copy()
        if not self.sched.env0_world:
            p = p - self.delta
        return p

    def _eef_pos(self, arm: int) -> np.ndarray:
        return self.sched.eef_row(arm, self.i)[:3] - self.delta

    def move_to(self, arm: int, position, quaternion_wxyz=None, max_steps: int = 300,
                pos_tol: float = 0.02, max_step: float = 0.01) -> float:
        goal = self._np3(position, 3)
        if quaternion_wxyz is not None:
            self.sched.queue_target(arm, self.i, None, self._np3(quaternion_wxyz, 4))
        for _ in range(int(max_steps)):
            cur = self._target_pos(arm)
            delta = goal - cur
            dist = float(np.linalg.norm(delta))
            if dist <= pos_tol:
                break
            self.sched.queue_target(
                arm, self.i, cur + delta * min(1.0, max_step / max(dist, 1e-6)))
            self.step(1)
        # settle: converge the LIVE wrist, not just the virtual target (OSC lag)
        for _ in range(int(max_steps)):
            if float(np.linalg.norm(self._eef_pos(arm) - goal)) <= pos_tol * 1.2:
                break
            self.step(1)
        return float(np.linalg.norm(self._eef_pos(arm) - goal))

    def _set_hand(self, arm: int, frac: float, steps: int) -> None:
        self.sched.queue_hand(arm, self.i, frac)
        self.step(int(steps))

    def open_gripper(self, arm: int = 0, steps: int = 20) -> None:
        self._set_hand(arm, 0.0, steps)

    def close_gripper(self, arm: int = 0, steps: int = 30) -> None:
        self._set_hand(arm, 1.0, steps)

    # ---- read-only queries (env-0 frame) ----
    def get_object_pose(self, name: str):
        s = self.sched.obj_row(name, self.i)
        return (s[:3] - self.delta, s[3:7].copy())

    def get_state(self, name: str):
        s = self.sched.obj_row(name, self.i).copy()
        s[:3] -= self.delta
        return s

    def get_eef_pose(self, arm: int = 0):
        s = self.sched.eef_row(arm, self.i)
        return (s[:3] - self.delta, s[3:7].copy())

    def get_seated(self):
        return self.sched.seated_row(self.i)

    def get_robot_state(self) -> dict:
        return {"left_ee_pose": np.concatenate(self.get_eef_pose(0)),
                "right_ee_pose": (np.concatenate(self.get_eef_pose(1))
                                  if getattr(self.api, "_arts", None) is not None
                                  or len(getattr(self.api, "_wrist_idx", [0])) > 1
                                  else None),
                "steps_used": self.sched.total}

    def get_link_positions(self, pattern: str = ".*hand.*") -> dict:
        # Rare diagnostic call: a direct (unbatched) read is acceptable here.
        import re as _re2
        out = {}
        arts = getattr(self.api, "_arts", None)
        pairs = (list(zip(("left", "right"), arts)) if arts is not None
                 else [("robot", self.api.art)])
        for label, art in pairs:
            for j, bname in enumerate(art.body_names):
                if _re2.search(pattern, bname):
                    p = art.data.body_pos_w[self.i, j].detach().cpu().numpy()
                    out[f"{label}/{bname}"] = p - self.delta
        return out

    def list_objects(self):
        return self.api.list_objects()

    def describe_scene(self, structured: bool = False):
        return self.api.describe_scene(structured)

    def look(self, *_a, **_k) -> str:
        return "(look is disabled during optimize: recording is off for the search)"

    def review_rollout(self, *_a, **_k) -> str:
        return "(review_rollout is disabled during optimize)"

    def namespace(self, quiet: bool) -> dict:
        ns = {
            "np": np, "torch": torch,
            "move_to": self.move_to, "step": self.step,
            "open_gripper": self.open_gripper, "close_gripper": self.close_gripper,
            "set_hand_joints": _blocked("set_hand_joints()"),
            "get_object_pose": self.get_object_pose, "get_state": self.get_state,
            "get_eef_pose": self.get_eef_pose, "get_seated": self.get_seated,
            "get_robot_state": self.get_robot_state,
            "get_link_positions": self.get_link_positions,
            "list_objects": self.list_objects, "describe_scene": self.describe_scene,
            "look": self.look, "review_rollout": self.review_rollout,
        }
        for name in ("env", "api", "checkpoint", "commit", "goto", "list_checkpoints",
                     "render_checkpoint", "get_checkpoint_scene", "get_checkpoint_log",
                     "get_checkpoint_code", "optimize", "snapshot", "restore"):
            ns[name] = _blocked(name, _SETUP_ADVICE if name in ("env", "api") else "")
        if quiet:
            ns["print"] = lambda *a, **k: None
        return ns


def _t_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(float).copy()


def _inject_constants(program_src: str, names: list[str]) -> tuple["ast.Module", dict]:
    """Compile the agent's program with its top-level constant assignments for `names`
    rewritten to read from the injected __capx_P dict, and return the CURRENT values
    those constants hold in the file (the incumbents — they seed the search, so the
    result can never be worse than what the agent already had, up to physics noise).
    Only assignments to a plain name with a numeric-literal value are rewritten (a
    later computed reassignment is the program's own business). Raises with the
    available constant names if a requested name is not found."""
    tree = ast.parse(program_src)
    found: set[str] = set()
    available: list[str] = []
    incumbents: dict = {}

    def _is_literal(node) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and _is_literal(node.operand))

    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target is None or node.value is None or not _is_literal(node.value):
            continue
        available.append(target)
        if target in names:
            found.add(target)
            try:
                incumbents[target] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
            node.value = ast.Subscript(
                value=ast.Name(id="__capx_P", ctx=ast.Load()),
                slice=ast.Constant(value=target), ctx=ast.Load())
    missing = [n for n in names if n not in found]
    if missing:
        raise ValueError(
            f"space names {missing} not found as top-level numeric constants in the "
            f"program. Top-level constants available: {sorted(set(available))}")
    return ast.fix_missing_locations(tree), incumbents


def _load_objective(objective_src: str):
    ns: dict = {"np": np, "torch": torch}
    exec(compile(objective_src, "<objective>", "exec"), ns)
    fn = ns.get("objective")
    if not callable(fn):
        raise ValueError("the objective file must define  def objective(v) -> float")
    return fn


def _install_step_profiler(api) -> dict:
    """Split every control step of env.step into its six pipeline sub-calls
    (robobench BaseEnv.step: apply_action -> write_data_to_sim -> sim.step ->
    iscene.update -> robot.post_step -> scene.post_step), plus our _action math.
    Diagnostic for the contact-phase slowdown (2026-07-26); profile_step=True only."""
    env = api._env
    targets = [
        ("action", api, "_action"),
        ("apply_action", env.robot, "apply_action"),
        ("write_sim", env.iscene, "write_data_to_sim"),
        ("sim_step", env.sim, "step"),
        ("scene_update", env.iscene, "update"),
        ("robot_post", env.robot, "post_step"),
        ("scene_post", env.scene, "post_step"),
    ]
    prof = {name: 0.0 for name, _o, _a in targets}
    prof["n"] = 0
    # per-CONTROL-STEP series per component (seconds), for bucketed curves that
    # localize WHICH component carries a slow phase, not just the average.
    prof["series"] = {name: [] for name, _o, _a in targets}
    originals = []

    # (A torch.cuda.synchronize-based "CPU vs GPU" split was trialed here and removed
    # 2026-07-26: PhysX submits work in its OWN CUDA context, which torch's
    # synchronize does not cover, so the split's numbers were not interpretable.)
    def _wrap(name, obj, attr):
        orig = getattr(obj, attr)

        def timed(*a, **k):
            t0 = time.time()
            out = orig(*a, **k)
            dt = time.time() - t0
            prof[name] += dt
            if name == "sim_step":
                prof["n"] += 1
            return out

        originals.append((obj, attr, orig))
        setattr(obj, attr, timed)

    for name, obj, attr in targets:
        _wrap(name, obj, attr)

    # wrap api.step LAST: it snapshots the accumulators around one control step and
    # appends the delta to the per-step series. When a step crosses the slow
    # threshold (or recovers), dump the physical state suspects: the fastest-moving
    # body per object across envs — catching jitter/contact storms in the act.
    orig_api_step = api.step
    names = [n for n, _o, _a in targets]
    prof["slow_mode"] = False

    def _state_suspects(tag: str, step_i: int, dt: float) -> None:
        """Per-ENV agitation census at a cost transition. Counting HOW MANY envs are
        agitated (not just the max) is what separates "cost scales with the number of
        disturbed envs" from "one pathological env taxes the whole batch"."""
        try:
            reg = getattr(env.iscene, "rigid_objects", {}) or {}
            spin = None
            speed = None
            for obj in reg.values():
                v = obj.data.root_vel_w  # (N, 6) world lin+ang velocity
                s, w = v[:, :3].norm(dim=1), v[:, 3:].norm(dim=1)
                speed = s if speed is None else torch.maximum(speed, s)
                spin = w if spin is None else torch.maximum(spin, w)
            if spin is None:
                return
            hot = (spin > 5.0) | (speed > 0.5)
            vhot = (spin > 20.0) | (speed > 2.0)
            order = torch.argsort(spin, descending=True)[:5]
            top = ", ".join(f"env{int(i)}:{float(spin[i]):.1f}rad/s"
                            f"/{float(speed[i]):.2f}m/s" for i in order)
            print(f"[optimize] {tag} at step {step_i} ({dt * 1000:.0f} ms): "
                  f"agitated envs {int(hot.sum())}/{spin.numel()} "
                  f"(violent {int(vhot.sum())}); top {top}", flush=True)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must not break the run
            print(f"[optimize] state-suspect dump failed: {exc!r}", flush=True)

    def timed_api_step(n_steps: int = 1):
        before = {p: prof[p] for p in names}
        t0 = time.time()
        out = orig_api_step(n_steps)
        dt = time.time() - t0
        for p in names:
            prof["series"][p].append(prof[p] - before[p])
        step_i = len(prof["series"]["sim_step"])
        if dt > 1.0 and not prof["slow_mode"]:
            prof["slow_mode"] = True
            _state_suspects("SLOW-STATE ENGAGED", step_i, dt)
        elif dt < 0.5 and prof["slow_mode"]:
            prof["slow_mode"] = False
            _state_suspects("slow state cleared", step_i, dt)
        return out

    api.step = timed_api_step
    originals.append((api, "step", orig_api_step))
    prof["_originals"] = originals
    return prof


def _remove_step_profiler(api, prof: dict) -> None:
    for obj, attr, orig in prof["_originals"]:
        setattr(obj, attr, orig)


def optimize_program(env, api, *, program_src: str, objective_src: str, space: dict,
                     generations: int = 8, budget_s: float = 3600.0,
                     popsize: int | None = None, repeats: int = 1,
                     randomize: dict | None = None, fix: dict | None = None,
                     max_steps_per_eval: int = 3000, seed: int = 0,
                     profile_step: bool = False, progress_cb=None,
                     setup_src: str = "", verbose: bool = True) -> dict:
    """Search `space` by running COPIES OF THE AGENT'S OWN PROGRAM, one per env.

    Per generation: replicate the current state into every env, substitute each copy's
    candidate values for the named constants, run all copies in lockstep (each in its
    own thread against a per-env toolkit facade), then score every end state with the
    agent's objective. The caller restores the world afterwards.

    `progress_cb(record)` (optional) is called after every generation so a long search
    can be streamed to the agent while it works on something else."""
    from cosigen_loop import AssemblyView, expand_state_tree  # lazy: import cycle
    from cosigen_view import zero_root_velocities

    sp = Space(space, fix)
    objective = _load_objective(objective_src)
    prog_ast, incumbents = _inject_constants(program_src, sp.names)
    prog_code = compile(prog_ast, "<optimize:program>", "exec")
    u_seed = sp.norm_values(incumbents)
    n_env = int(env.num_envs)
    reps = max(1, int(repeats))
    # Default: the pod's FULL width. Measured 2026-07-26 on the pathological
    # grasp search: 512-wide ran 2048 evaluations in 831 s (0.41 s/eval) versus
    # ~10 s/eval at popsize 64, because the pathological-contact penalty saturates
    # (8 bad envs cost the same as 64) while evaluations scale with width. Later
    # generations clean up as the distribution contracts (gens 2-3: ~320 ms/step).
    pop = int(popsize) if popsize else max(2, n_env // reps)
    if pop < 2:
        raise ValueError(f"population too small: num_envs={n_env}, repeats={repeats}")
    if pop * reps > n_env:
        raise ValueError(f"popsize*repeats ({pop}*{reps}) exceeds num_envs ({n_env})")
    device = env.device
    rng = np.random.default_rng(int(seed))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    if setup_src:
        # The agent's own setup code, run ONCE here with full raw access before any
        # candidate. A dedicated search pod inherits only the WORLD STATE of the
        # checkpoint node, so simulator-level configuration the agent applied on its
        # turn pod (OSC gains, controller wiring, solver knobs -- none of which live in
        # a checkpoint) is absent, and numbers tuned here would be tuned against a
        # differently-behaving robot. This runs on the main thread outside the
        # per-candidate namespace, which is why it may touch env/api that the copies
        # cannot: it applies once, globally, identically for every candidate.
        from cosigen_loop import make_namespace
        if verbose:
            print("[optimize] applying the search-pod setup program", flush=True)
        setup_ns = make_namespace(env, api)
        try:
            exec(compile(setup_src, "<optimize:setup>", "exec"), setup_ns, setup_ns)
        except Exception as exc:
            raise RuntimeError(
                f"the setup program raised on the search pod, so the search would have "
                f"tuned against an unconfigured robot: {exc!r}\n"
                f"{_tb.format_exc(limit=4)}") from exc

    views = [AssemblyView(api, i) for i in range(n_env)]
    start = {"sim": env.get_states(torch.tensor([0], device=device)),
             "api": api._api_state_env0()}
    origin_delta = (env.iscene.env_origins - env.iscene.env_origins[0:1]).to(device)

    def reset_population() -> None:
        state_n = expand_state_tree(start["sim"], n_env)
        shift_root_states(state_n, origin_delta)
        jitter_scene(state_n, randomize, gen)
        # Reset hygiene (2026-07-26): anchors captured near motion carry residual
        # velocities; replicating them into every env seeds simultaneous jitter,
        # the measured trigger of the 1.6-2.5 s/step PhysX slow state.
        zero_root_velocities(state_n)
        env.set_states(state_n)
        api._restore_api_state(start["api"])
        if not getattr(api, "_targets_env0_world", False):
            for tgt in getattr(api, "_tgt", []):
                tgt[:, :3] += origin_delta

    sampler = _CMAES(sp.dim, rng, popsize=pop)
    if u_seed is not None:
        # The agent's current values may already be decent (earlier experiments):
        # center the first generation on them AND evaluate them verbatim as one
        # candidate, so the search cannot return worse than the incumbent.
        sampler.mean = u_seed.copy()
        if verbose:
            print(f"[optimize] seeding the search with the program's current values: "
                  f"{sp.as_dict(sp.denorm(u_seed[None, :])[0])}", flush=True)
    history: list[dict] = []
    best = {"score": float("inf"), "params": None, "gen": -1}
    all_u, all_scores = [], []
    t0 = time.time()
    was_active = getattr(api, "_replicas_active", False)
    api._replicas_active = True  # replicas must act: parking off for the search
    rec_cam = getattr(api, "_rec_cam", None)
    api._rec_cam = None  # recording off: search footage is 512 overlaid candidates
    budget_before = api.max_steps
    old_stack = threading.stack_size(512 * 1024)  # 512 threads: keep stacks small
    prof = _install_step_profiler(api) if profile_step else None
    try:
        for g in range(int(generations)):
            if time.time() - t0 > budget_s:
                break
            u = sampler.ask(pop)
            if g == 0 and u_seed is not None:
                u[0] = u_seed  # the incumbent runs as candidate 0 of generation 0
            values = sp.denorm(u)
            values_env = np.repeat(values, reps, axis=0)
            # Envs beyond pop*reps stay idle in their reset pose (resting contacts
            # only): a smaller population must not pay full-width contact physics.
            n_run = values_env.shape[0]
            reset_population()
            sched = _Lockstep(api, max_steps_per_eval)
            errors: dict[int, str] = {}

            def _run(i: int, sched=sched, errors=errors, values_env=values_env) -> None:
                tk = _EnvToolkit(api, i, sched, tid=i)
                ns = tk.namespace(quiet=(i != 0))
                ns["__capx_P"] = sp.as_dict(values_env[i])
                try:
                    exec(prog_code, ns)
                except _EvalAborted:
                    pass  # budget exhausted: score the state reached so far
                except Exception:
                    errors[i] = _tb.format_exc(limit=3)
                finally:
                    sched.finish(i)

            threads = [threading.Thread(target=_run, args=(i,), daemon=True)
                       for i in range(n_run)]
            sched.active = set(range(n_run))  # tids are env indices
            sched.finished = set()
            for t in threads:
                t.start()
            sched.run()
            for t in threads:
                t.join(timeout=60.0)

            # A program bug (not a bad candidate) fails everywhere identically:
            # surface it to the agent instead of returning a meaningless "search".
            if len(errors) > 0.8 * n_run and g == 0:
                first = next(iter(errors.values()))
                raise RuntimeError(
                    f"the program raised in {len(errors)}/{n_run} copies — this is a "
                    f"program error, not a parameter problem. First traceback:\n{first}")

            raw = np.full(n_run, np.nan)
            for i in range(n_run):
                if i in errors:
                    continue
                try:
                    raw[i] = float(objective(views[i]))
                except Exception as exc:  # noqa: BLE001
                    if verbose and i == 0:
                        print(f"[optimize] objective raised on env {i}: {exc!r}",
                              flush=True)
            scores = np.array([np.nanmean(raw[j * reps:(j + 1) * reps]) if reps > 1
                               else raw[j] for j in range(pop)])
            sampler.tell(u, scores)
            all_u.append(u)
            all_scores.append(scores)
            j = int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0
            if np.isfinite(scores[j]) and scores[j] < best["score"]:
                best = {"score": float(scores[j]), "params": sp.as_dict(values[j]),
                        "gen": g}
            rec = {"gen": g,
                   "best_in_gen": (float(np.nanmin(scores))
                                   if np.isfinite(scores).any() else None),
                   "median": (float(np.nanmedian(scores))
                              if np.isfinite(scores).any() else None),
                   "failed_copies": int(len(errors)),
                   "steps_used": sched.total,
                   "best_so_far": best["score"], "t": round(time.time() - t0, 1)}
            history.append(rec)
            if progress_cb is not None:
                # Stream to the agent between generations: a 3600 s search should not
                # be an opaque wait (async optimize tool, 2026-07-26).
                try:
                    progress_cb({"generations_done": g + 1,
                                 "generations_planned": int(generations),
                                 "best": best["params"], "best_score": best["score"],
                                 "median_this_gen": rec["median"],
                                 "n_evals": int(sum(len(s) for s in all_scores)),
                                 "seconds": rec["t"], "history": list(history)})
                except Exception:  # noqa: BLE001 -- streaming must never break a search
                    print("[optimize] progress callback failed:\n"
                          + _tb.format_exc(limit=3), flush=True)
            if verbose:
                print(f"[optimize] gen {g}: best_in_gen={rec['best_in_gen']} "
                      f"median={rec['median']} best_so_far={best['score']:.4f} "
                      f"({rec['t']}s, {rec['steps_used']} steps)", flush=True)
                print(f"[optimize] engine timing: {sched.timing_report()}", flush=True)
                if prof is not None and prof["n"]:
                    parts = ("action", "apply_action", "write_sim", "sim_step",
                             "scene_update", "robot_post", "scene_post")
                    n = prof["n"]  # = physics substeps this generation
                    split = " ".join(f"{p} {prof[p] / n * 1000:.0f}" for p in parts)
                    print(f"[optimize] substep profile (ms per SUBSTEP, {n} substeps): "
                          f"{split}", flush=True)
                    series = prof["series"]
                    n_steps = len(series["sim_step"])
                    if n_steps:
                        bucket = max(1, n_steps // 8)
                        for p in parts:
                            s = series[p]
                            if sum(s) / max(n_steps, 1) < 0.002:
                                continue  # skip sub-ms components in the curves
                            curve = " ".join(
                                f"{sum(s[i:i + bucket]) / len(s[i:i + bucket]) * 1000:.0f}"
                                for i in range(0, n_steps, bucket))
                            print(f"[optimize]   {p} ms/step curve: {curve}", flush=True)
                    for p in parts:
                        prof[p] = 0.0
                        prof["series"][p] = []
                    prof["n"] = 0
                st = sched.step_times
                if st:
                    bucket = max(1, len(st) // 8)
                    curve = " ".join(
                        f"{sum(st[i:i + bucket]) / len(st[i:i + bucket]) * 1000:.0f}"
                        for i in range(0, len(st), bucket))
                    print(f"[optimize] per-step physics ms, {bucket}-step buckets: "
                          f"{curve}", flush=True)
    finally:
        threading.stack_size(old_stack)
        api.max_steps = budget_before
        api._replicas_active = was_active
        api._rec_cam = rec_cam
        if prof is not None:
            _remove_step_profiler(api, prof)

    sens = {}
    if all_u:
        U, S = np.concatenate(all_u), np.concatenate(all_scores)
        ok = np.isfinite(S)
        for j, name in enumerate(sp.names):
            col = U[ok, j]
            sens[name] = (0.0 if col.std() < 1e-9 or S[ok].std() < 1e-9
                          else abs(float(np.corrcoef(col, S[ok])[0, 1])))
    # Red-team findings from the first agent-driven search (2026-07-26): a best value
    # sitting on a declared bound, and a budget-truncated search, were both silent.
    # Say them out loud so the agent knows to widen the range / raise budget_s.
    at_bounds: dict[str, str] = {}
    if best["params"]:
        u_best = sp.norm_values(best["params"])
        if u_best is not None:
            for j, name in enumerate(sp.names):
                if sp.kind[j] == "choices":
                    continue
                if u_best[j] <= 0.02:
                    at_bounds[name] = "low"
                elif u_best[j] >= 0.98:
                    at_bounds[name] = "high"
    truncated = len(history) < int(generations)
    if verbose and at_bounds:
        print(f"[optimize] note: best value at the {at_bounds} edge of your search "
              "range — the true optimum may lie beyond it; consider widening",
              flush=True)
    if verbose and truncated:
        print(f"[optimize] note: stopped by budget_s after {len(history)}/"
              f"{int(generations)} generations", flush=True)
    return {"best": best["params"], "best_score": best["score"],
            "best_gen": best["gen"], "history": history, "sensitivity": sens,
            "at_bounds": at_bounds, "budget_truncated": truncated,
            "generations_run": len(history), "generations_planned": int(generations),
            "n_evals": int(sum(len(s) for s in all_scores)),
            "seconds": round(time.time() - t0, 1)}


