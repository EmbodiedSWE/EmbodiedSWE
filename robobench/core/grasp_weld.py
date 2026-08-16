"""Weld-on-closure grasp contract — the shared scene mixin.

One pool of pre-authored, disabled FixedJoints per (env, site). PhysX latches a joint's local
frames on FIRST enable and ignores rewrites on a re-enable, so every engage consumes a fresh
pool joint: the live hand->part pose is written while the joint is still disabled, it is
enabled once, and on release it is retired for good.

Engage (reconciled every physics substep, debounced): pinch point within `grasp_weld_dist` of
a site's LIVE grip band + the hand's closure test (below = closed on air, above = nothing
snagged) + fingers STALLED (a closing sweep passes through the window; a real pinch stops in
it). Release: closure past window-top + margin (hysteresis). One hold per env (a hand pinches
one part). Embodiment-agnostic two ways: no hand on the stage (e.g. robot="null") -> no
joints, no-op contract; and a robot class may declare `GRASP_IFACE` to key the contract to
its own hand — absent that, the scene's panda ClassVars apply, bit-identically to the
historical inline copies. Holds ride get_state/set_state.

`GRASP_IFACE` (dict on the robot class; every key optional, defaults = the panda values):
    hand_body       str    weld/pinch body name (default "panda_hand")
    finger_joints   str    regex for the closing joints (default "panda_finger_joint.*")
    approach        (3,)   hand-local approach axis, pinch point = hand + offset*approach
                           (default (0,0,1))
    pinch_offset    float  hand origin -> pinch point along `approach` (default 0.1034)
    stall_vel       float  max |finger vel| sum: fingers stopped ON the part (default 0.01;
                           NOTE units follow the joints — m/s prismatic, rad/s revolute)
    closure         tuple  ("joint_sum",)                the panda path: the site window tests
                                                         the finger-joint SUM (exact legacy)
                           ("aperture",)                 the site window (metres) tests the
                                                         pad-body separation - sep_off: unit-
                                                         correct for revolute/linkage hands
                           ("wrap", prox_lo, prox_hi, squeeze_min)
                                                         curling hands with no aperture window:
                                                         thumb/pair flank the band + proximals
                                                         stalled INSIDE (prox_lo, prox_hi) with
                                                         command - achieved > squeeze_min
    pad_bodies      tuple  pad/tip body names — 2 for "aperture" (the faces), 3 for "wrap"
                           (thumb first, then the finger pair)
    sep_off         float  pad-body-origin separation -> face separation offset (m)
    pinch_axis      (3,)   hand-local pinch axis ("wrap" flank test only)
    wrap_off        (2,)   site window -> wrap proxy band offsets (m): the tip-origin proxy
                           reads wider than the true gap by the fingers' own geometry
                           (default (0.025, 0.050))
    prox_release    float  "wrap" release: proximals retreat below prox_lo - this (rad,
                           default 0.15)

Import-light: torch/pxr/isaaclab imports live inside the methods (bind/step time).
"""

from __future__ import annotations

from typing import Any, ClassVar


class GraspWeldMixin:
    """Scene mixin: weld-on-closure grasping against the scene's `grasp_sites()`."""

    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034  # hand origin -> finger-pad centre, along approach
    GRASP_POOL: ClassVar[int] = 8  # engages per (env, site) per run; exhausted -> warn, no weld
    GRASP_STALL: ClassVar[float] = 0.01  # max |finger vel| sum (m/s): fingers stopped ON the part
    GRASP_DEBOUNCE: ClassVar[int] = 8  # consecutive qualifying substeps before the weld engages
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.008  # release at window-top + this (m), hysteresis

    # ----- per-robot hand keys --------------------------------------------------------------------
    def _gw_iface(self) -> dict:
        """The active hand keys: the robot's `GRASP_IFACE` when declared, else the scene's panda
        ClassVars (exact legacy behavior). Resolved at bind; the robot OBJECT exists by then even
        though it binds after the scene (ClassVar access needs no binding)."""
        base = dict(
            hand_body=self.GRASP_HAND_BODY,
            finger_joints=self.GRASP_FINGER_JOINTS,
            approach=(0.0, 0.0, 1.0),
            pinch_offset=self.GRASP_PINCH_OFFSET,
            stall_vel=self.GRASP_STALL,
            closure=("joint_sum",),
            pad_bodies=None,
            sep_off=0.0,
            pinch_axis=(0.0, 1.0, 0.0),
            wrap_off=(0.025, 0.050),
            prox_release=0.15,
        )
        ifc = getattr(getattr(self.env, "robot", None), "GRASP_IFACE", None)
        if ifc:
            base.update(ifc)
        return base

    # ----- bind-time: discovery + joint pools ------------------------------------------------------
    def _grasp_weld_bind(self) -> None:
        """Discover the hand, author the (disabled) joint pools, allocate the hold state. Called
        from `bind()` — authoring must happen BEFORE the sim starts playing, or PhysX only picks
        the joints up after a full `sim.reset()`."""
        import torch

        env = self.env
        n = env.num_envs
        self._gw_on = bool(getattr(self.cfg, "grasp_weld", False))
        self._gw_art = None  # articulation handle, resolved lazily (the robot binds after us)
        self._gw_sites: list = []
        if not self._gw_on:
            return
        self._gw_if = self._gw_iface()
        hand0 = self._gw_find_hand_prim()
        if hand0 is None:  # no gripper in this embodiment (e.g. robot="null") -> no-op contract
            self._gw_on = False
            print(f"[grasp-weld] no '{self._gw_if['hand_body']}' on the stage — contract disabled", flush=True)
            return
        self._gw_sites = list(self.grasp_sites())
        s = len(self._gw_sites)
        dev = env.device
        self.grasp_held = torch.zeros(n, s, dtype=torch.bool, device=dev)
        self._gw_rel_p = torch.zeros(n, s, 3, device=dev)
        self._gw_rel_q = torch.zeros(n, s, 4, device=dev)
        self._gw_count = torch.zeros(n, s, dtype=torch.int32, device=dev)
        self._gw_pool_i = [[0] * s for _ in range(n)]
        self._gw_pool_warned: set = set()
        self._gw_author_pools(hand0)

    def _gw_find_hand_prim(self) -> str | None:
        """The hand body's prim path under env_0 (clones are identical), or None if absent."""
        from pxr import Usd

        root = self.env.stage.GetPrimAtPath("/World/envs/env_0")
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == self._gw_if["hand_body"]:
                return str(prim.GetPath())
        return None

    def _gw_part_path(self, obj, env_i: int) -> str:
        """The part's RIGID-BODY prim path in env `env_i`. The asset root from the cfg is not
        always the body (some USDs nest it one level down), so walk the subtree for the first
        `RigidBodyAPI` prim — the joint must bind the body, or PhysX ignores it."""
        from pxr import Usd, UsdPhysics

        p = obj.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        root = p.replace("env_.*", f"env_{env_i}")
        prim = self.env.stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise RuntimeError(f"[grasp-weld] part prim missing: {root}")
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(child.GetPath())
        raise RuntimeError(f"[grasp-weld] no RigidBodyAPI prim under {root}")

    def _gw_author_pools(self, hand0: str) -> None:
        """One pool of disabled FixedJoints per (env, site): body0 = the hand, body1 = the part,
        frames identity until an engage writes the live relative pose."""
        from pxr import Gf, UsdPhysics

        stage = self.env.stage
        self._gw_paths: list[list[list[str]]] = []  # [env][site][k]
        for i in range(self.env.num_envs):
            hand = hand0.replace("env_0", f"env_{i}")
            rows = []
            for name, obj, _p0, _p1, _win in self._gw_sites:
                part = self._gw_part_path(obj, i)
                row = []
                for k in range(self.GRASP_POOL):
                    jp = f"/World/envs/env_{i}/gweld_{name}_{k}"
                    j = UsdPhysics.FixedJoint.Define(stage, jp)
                    j.CreateBody0Rel().SetTargets([hand])
                    j.CreateBody1Rel().SetTargets([part])
                    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    j.CreateJointEnabledAttr(False)
                    j.CreateExcludeFromArticulationAttr(True)  # maximal-coordinate, not an arm DOF
                    row.append(jp)
                rows.append(row)
            self._gw_paths.append(rows)

    def _gw_resolve_hand(self) -> bool:
        """Cache the articulation handle + indices on first use (the robot binds after the scene)."""
        if self._gw_art is not None:
            return True
        try:
            art = self.env.robot.articulation
            ifc = self._gw_if
            self._gw_hand_i = art.body_names.index(ifc["hand_body"])
            self._gw_fingers = art.find_joints([ifc["finger_joints"]])[0]
            mode = ifc["closure"][0]
            if mode == "joint_sum":
                assert len(self._gw_fingers) == 2
            else:
                assert len(self._gw_fingers) >= 1
                self._gw_pads = [art.body_names.index(b) for b in ifc["pad_bodies"]]
                if mode == "wrap":
                    assert len(self._gw_pads) == 3, "wrap closure: (thumb, finger, finger)"
                    self._gw_prox = self._gw_fingers[: len(self._gw_fingers) // 2]
        except Exception as e:  # articulated but not a gripper we understand -> disable, loudly
            self._gw_on = False
            print(f"[grasp-weld] DISABLED after error: {e!r}", flush=True)
            return False
        self._gw_art = art
        return True

    # ----- per-substep reconcile -------------------------------------------------------------------
    def _gw_closure(self):
        """(gap, stalled): the closure measure the site windows test, and the stall gate.
        joint_sum: finger-joint sum (exact legacy panda path). aperture: pad-body separation
        minus `sep_off` — the physical jaw gap, unit-correct for revolute/linkage hands."""
        import torch

        art = self._gw_art
        ifc = self._gw_if
        stalled = art.data.joint_vel[:, self._gw_fingers].abs().sum(dim=-1) < ifc["stall_vel"]
        mode = ifc["closure"][0]
        if mode == "joint_sum":
            gap = art.data.joint_pos[:, self._gw_fingers].sum(dim=-1)
        elif mode == "aperture":
            d = art.data.body_pos_w[:, self._gw_pads[0]] - art.data.body_pos_w[:, self._gw_pads[1]]
            gap = (d.norm(dim=-1) - ifc["sep_off"]).clamp_min(0.0)
        else:  # wrap: tip-origin proxy — thumb to finger-pair mid (monotone opening measure)
            fp = art.data.body_pos_w[:, self._gw_pads]
            gap = (fp[:, 0] - 0.5 * (fp[:, 1] + fp[:, 2])).norm(dim=-1)
        return gap, stalled

    def _gw_windows(self):
        """Per-site (lo, hi) the closure gap is tested against. The wrap proxy reads wider than
        the true gap by the fingers' own geometry: offset the site window by `wrap_off`."""
        ifc = self._gw_if
        if ifc["closure"][0] == "wrap":
            o0, o1 = ifc["wrap_off"]
            return [(win[0] + o0, win[1] + o1) for _n, _o, _p0, _p1, win in self._gw_sites]
        return [win for _n, _o, _p0, _p1, win in self._gw_sites]

    def _grasp_weld_step(self) -> None:
        """Reconcile engages + releases against the closure criterion. Called from `post_step()`."""
        if not getattr(self, "_gw_on", False) or not self._gw_sites or not self._gw_resolve_hand():
            return
        import torch

        from isaaclab.utils.math import quat_apply

        art = self._gw_art
        ifc = self._gw_if
        hp = art.data.body_pos_w[:, self._gw_hand_i]
        hq = art.data.body_quat_w[:, self._gw_hand_i]
        gap, stalled = self._gw_closure()
        wins = self._gw_windows()
        approach = torch.tensor(ifc["approach"], dtype=hp.dtype, device=hp.device).expand_as(hp)
        pinch = hp + quat_apply(hq, approach * ifc["pinch_offset"])

        mode = ifc["closure"][0]
        if mode == "wrap":
            _w, prox_lo, prox_hi, squeeze_min = ifc["closure"]
            prox_q = art.data.joint_pos[:, self._gw_prox].amin(dim=-1)
            cmd = art.data.joint_pos_target[:, self._gw_prox].amax(dim=-1)
            # a free-air curl reaches the command and fails the squeeze margin; only pressing
            # the part stalls the proximals early — the wrap's contact detector
            closed_ok = (prox_q > prox_lo) & (prox_q < prox_hi) & ((cmd - prox_q) > squeeze_min)
            released = prox_q < (prox_lo - ifc["prox_release"])
        else:
            closed_ok = stalled  # the window test joins per site below
            released = None

        # Releases first (a re-grasp in the same step then sees a free hand).
        for row, s in self.grasp_held.nonzero(as_tuple=False).tolist():
            if mode == "wrap":
                if bool(released[row]):
                    self._gw_release(row, s)
            elif gap[row] > wins[s][1] + self.GRASP_RELEASE_MARGIN:
                self._gw_release(row, s)

        free = ~self.grasp_held.any(dim=-1)  # (n,)
        dists = self._gw_site_dists(pinch)  # (n, s)
        c = getattr(self.cfg, "grasp_weld_dist", 0.010)
        if mode == "wrap":
            flank = self._gw_wrap_flank(hq)
            ok = torch.stack(
                [
                    (dists[:, s] < c) & (gap > wins[s][0]) & (gap < wins[s][1]) & closed_ok & flank[:, s]
                    for s in range(len(self._gw_sites))
                ],
                dim=-1,
            ) & free.unsqueeze(-1)
        else:
            ok = torch.stack(
                [
                    (dists[:, s] < c) & (gap > wins[s][0]) & (gap < wins[s][1]) & closed_ok & stalled
                    for s in range(len(self._gw_sites))
                ],
                dim=-1,
            ) & free.unsqueeze(-1)
        self._gw_count = torch.where(ok, self._gw_count + 1, torch.zeros_like(self._gw_count))
        ready = (self._gw_count >= self.GRASP_DEBOUNCE).any(dim=-1) & free
        for row in ready.nonzero(as_tuple=False).flatten().tolist():
            masked = torch.where(
                self._gw_count[row] >= self.GRASP_DEBOUNCE, dists[row], torch.full_like(dists[row], torch.inf)
            )
            s = int(masked.argmin())
            self._gw_engage(row, s, hp[row], hq[row], gap[row])

    def _gw_wrap_flank(self, hq) -> "object":
        """(n, s) bool: thumb and finger-pair on OPPOSITE sides of each site's grip band along
        the hand's live pinch axis — the wrap actually encloses the part."""
        import torch

        from isaaclab.utils.math import quat_apply

        art = self._gw_art
        ifc = self._gw_if
        fp = art.data.body_pos_w[:, self._gw_pads]  # (n, 3, 3)
        pair_mid = 0.5 * (fp[:, 1] + fp[:, 2])
        axis = torch.tensor(ifc["pinch_axis"], dtype=fp.dtype, device=fp.device)
        axis_w = quat_apply(hq, axis.expand(fp.shape[0], 3))
        out = []
        for _name, obj, p0, p1, _win in self._gw_sites:
            pp, pq = obj.data.root_pos_w, obj.data.root_quat_w
            n = pp.shape[0]
            a = pp + quat_apply(pq, torch.tensor(p0, device=pp.device).expand(n, 3))
            b = pp + quat_apply(pq, torch.tensor(p1, device=pp.device).expand(n, 3))
            mid = 0.5 * (a + b)
            dl = ((fp[:, 0] - mid) * axis_w).sum(dim=-1)
            dr = ((pair_mid - mid) * axis_w).sum(dim=-1)
            out.append(dl * dr < 0)
        return torch.stack(out, dim=-1)

    def _gw_site_dists(self, pinch):
        """Pinch-point distance to every site's live grip band, shape (num_envs, num_sites)."""
        import torch

        from isaaclab.utils.math import quat_apply

        n = pinch.shape[0]
        out = []
        for _name, obj, p0, p1, _win in self._gw_sites:
            pp, pq = obj.data.root_pos_w, obj.data.root_quat_w
            a = pp + quat_apply(pq, torch.tensor(p0, device=pinch.device).expand(n, 3))
            b = pp + quat_apply(pq, torch.tensor(p1, device=pinch.device).expand(n, 3))
            ab = b - a
            t = ((pinch - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-12)
            closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
            out.append((pinch - closest).norm(dim=-1))
        return torch.stack(out, dim=-1)

    def _gw_engage(self, env_i: int, s: int, hp, hq, gap) -> None:
        """Weld (env_i, site s) to the hand at the live relative pose, on a fresh pool joint."""
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

        name, obj = self._gw_sites[s][0], self._gw_sites[s][1]
        rel_p = quat_apply_inverse(hq.unsqueeze(0), (obj.data.root_pos_w[env_i] - hp).unsqueeze(0))[0]
        rel_q = quat_mul(quat_conjugate(hq.unsqueeze(0)), obj.data.root_quat_w[env_i].unsqueeze(0))[0]
        if not self._gw_set_joint(env_i, s, rel_p, rel_q):
            return
        self._gw_rel_p[env_i, s] = rel_p
        self._gw_rel_q[env_i, s] = rel_q
        self.grasp_held[env_i, s] = True
        self._gw_count[env_i] = 0
        print(f"[grasp-weld] env {env_i}: GRIPPED {name} (aperture {float(gap) * 1000:.1f} mm)", flush=True)

    def _gw_set_joint(self, env_i: int, s: int, rel_p, rel_q) -> bool:
        """Write the hand-frame pose onto the next fresh pool joint and enable it. False = pool dry."""
        from pxr import Gf, UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k >= self.GRASP_POOL:
            if (env_i, s) not in self._gw_pool_warned:
                self._gw_pool_warned.add((env_i, s))
                print(f"[grasp-weld] env {env_i}: pool dry for {self._gw_sites[s][0]} — no weld", flush=True)
            return False
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
        p, q = rel_p.tolist(), rel_q.tolist()
        j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
        j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        j.GetJointEnabledAttr().Set(True)
        return True

    def _gw_release(self, env_i: int, s: int) -> None:
        """Cut (env_i, site s): disable the joint and retire it (frames latched — never reused)."""
        from pxr import UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k < self.GRASP_POOL:
            j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
            j.GetJointEnabledAttr().Set(False)
        self._gw_pool_i[env_i][s] = k + 1
        self.grasp_held[env_i, s] = False
        print(f"[grasp-weld] env {env_i}: RELEASED {self._gw_sites[s][0]}", flush=True)

    # ----- episode + state plumbing ----------------------------------------------------------------
    def _grasp_weld_release_all(self, env_ids) -> None:
        """Cut every hold for `env_ids` (a fresh episode starts empty-handed). Called from `reset()`."""
        if not getattr(self, "_gw_on", False):
            return
        for row, s in self.grasp_held[env_ids].nonzero(as_tuple=False).tolist():
            self._gw_release(int(env_ids[row]), s)
        self._gw_count[env_ids] = 0

    def _grasp_weld_state(self, env_ids) -> dict[str, Any]:
        """The contract's restorable state (empty when the contract is off)."""
        if not getattr(self, "_gw_on", False):
            return {}
        return {
            "grasp_held": self.grasp_held[env_ids].clone(),
            "grasp_rel_p": self._gw_rel_p[env_ids].clone(),
            "grasp_rel_q": self._gw_rel_q[env_ids].clone(),
        }

    def _grasp_weld_restore(self, state: dict[str, Any], env_ids) -> None:
        """Re-arm the holds `get_state` recorded, at their RECORDED hand-frame poses (the bodies
        were just written, so live measurement is redundant), on fresh pool joints. Called from
        `set_state()` after the bodies are restored."""
        if not getattr(self, "_gw_on", False) or "grasp_held" not in state:
            return
        for row in range(len(env_ids)):
            i = int(env_ids[row])
            for s in range(len(self._gw_sites)):
                if self.grasp_held[i, s]:
                    self._gw_release(i, s)
                if bool(state["grasp_held"][row, s]) and self._gw_set_joint(
                    i, s, state["grasp_rel_p"][row, s], state["grasp_rel_q"][row, s]
                ):
                    self._gw_rel_p[i, s] = state["grasp_rel_p"][row, s]
                    self._gw_rel_q[i, s] = state["grasp_rel_q"][row, s]
                    self.grasp_held[i, s] = True
        self._gw_count[env_ids] = 0
