"""Smoke / oracle test for ToolPackingScene — NullRobot, joint + drop staging, RECORDED.

One linear run (pen-holder smoke skeleton):
  1. show      — settle the reset layout (box shut, three tools scattered in front);
  2. oracle    — the scripted solve: swing both doors open (incremental joint writes — the
                 drives are passive, damping only), then per drawer top->mid->bot: pull it
                 out, kinematically carry its item above the tray and DROP it in (a genuine
                 drop, released a few cm up and off-centre), push the drawer shut — score
                 climbing 30 -> 60 -> 90 — and finally shut both doors -> 100 + success();
  3. negative A — an item dropped into the WRONG drawer (knife into the middle tray) must
                 not count: score unchanged, its stowed() stays False;
  4. negative B — everything stowed but ONE drawer left open: no 100, no success; shutting
                 it flips success — proves the closed-gate;
  5. negative C — an item parked on the box ROOF must not count as stowed;
  6. state round-trip — get_state/set_state at full score must preserve score + success.
  --demo runs ONLY show + oracle and saves the deliverable video.

ALWAYS records via the viewport rgb annotator (RTX driver-version override, warmup flush,
npz -> optional HDFS), the pen-holder/crate recipe. Bodies are driven straight through scene
handles; the NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.packing.smokes.tool_packing_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no negative controls) — "
                         "the user-facing deliverable video")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="tool_packing_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (render-server provenance): kit mis-decodes the L20 driver version and silently
# rejects RTX -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("packing.tool_packing")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        wx, wy = c.workbench_pos
        # frame BOTH the box and the scatter row (row centre c.items_center, pushed to
        # -0.52 to clear the door sweep): look at their midpoint, eye pulled back
        mid_x = wx + (c.box_pos[0] + c.items_center[0]) / 2
        look = np.array((mid_x, wy + c.box_pos[1], c.surface_z + 0.10))
        env.sim.set_camera_view(tuple(np.array((wx - 0.95, wy - 0.85, c.surface_z + 0.85)) + o),
                                tuple(look + o), camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    carry: dict[str, torch.Tensor] | None = None  # name -> target root pose while carried

    # --- separation monitor (VISUAL-mesh, ALL part pairs) -----------------------------------
    # MEASURED, per step, in the layer the viewer sees: actual visual-mesh point clouds
    # (read from the live stage) of the doors, drawers AND items, against volume models of
    # every other part (tray boxes, shell slabs, door plates). Hand-picked pair lists are
    # exactly how penetration kept slipping through — a pass on the checked pairs coexisted
    # with overlap in an unchecked one. Negative = a vertex inside another part's volume.
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom

    from isaaclab.utils.math import quat_apply as _qa
    from isaaclab.utils.math import quat_apply_inverse as _qai

    def _stage_points(root_path: str, want: str | None = None, cap: int = 800) -> torch.Tensor:
        root = env.stage.GetPrimAtPath(root_path)
        pts = []
        for p in _Usd.PrimRange(root, _Usd.TraverseInstanceProxies()):
            if not p.IsA(_UsdGeom.Mesh) or "col_" in p.GetName() or "collision" in p.GetName():
                continue
            if want is not None and f"/{want}/" not in str(p.GetPath()) + "/":
                continue
            arr = np.array(_UsdGeom.Mesh(p).GetPointsAttr().Get())
            pts.append(arr[:: max(1, len(arr) // cap)])
        if not pts:
            raise RuntimeError(f"[smoke] no visual meshes under {root_path} ({want}) — "
                               f"separation gate cannot run")
        return torch.tensor(np.concatenate(pts), dtype=torch.float32, device=device)

    _B = scene.box.body_names.index
    # point clouds: {label: (kind, ref, points)} — kind 'link' refs a box body index,
    # kind 'item' refs a RigidObject
    _clouds = {}
    for _nm in ("E_door_1_12", "E_door_2_13", "E_drawer_1_7", "E_drawer_2_4", "E_drawer_3_1"):
        _clouds[_nm] = ("link", _B(_nm), _stage_points("/World/envs/env_0/Toolbox", _nm))
    for _inm, _d, _m, _iscale, _z in c.manifest:
        # apply the spawn scale: the stage mesh points are the UNSCALED asset — without
        # this the gate measures a phantom full-size item at the scaled item's pose (the
        # 0.55 stapler read as -10.5 mm into a floor it was resting exactly on)
        _clouds[_inm] = ("item", scene.items[_inm],
                         _stage_points(f"/World/envs/env_0/Item_{_inm}", None, 400) * _iscale)
    # volume models: {label: (frame body index, [(lo, hi), ...])} in that body's frame.
    # Trays are modelled as their SOLID slabs (floor + 4 walls) — a stowed item lives
    # inside the tray's bbox by design, so a bbox volume would flag every success.
    # per-slab interior-depth ALLOWANCE (m): thin walls saturate the depth metric (an 8 mm
    # wall reads at most -4 mm even for a full crossing — which hid a door-through-drawer
    # crossing under the old global 3 mm rest allowance), so they tolerate only 1.5 mm;
    # thick caps/floors keep 3 mm for collider-vs-visual rest seating.
    THIN, THICK = 0.0015, 0.003
    # The two doors are DESIGNED bypass-lap panels: their meeting stiles overlap in
    # depth (slab models overlap ~1.3 mm even shut — the vendored asset filters this
    # one collision pair for the same reason), and while a swinging lip crosses the
    # other panel's plane the solid-slab model reads up to ~3.4 mm "inside". That is
    # the lap design, not a crossing — the door<->door pair alone gets the lap depth.
    DOOR_LAP = 0.004
    _DOORS = {"E_door_1_12", "E_door_2_13"}

    def _tray_slabs() -> list:
        lo = (-0.211, -0.0055, -0.0234)
        hi = (0.2146, 0.2789, 0.0471)
        w = 0.008
        floor_top = 0.004  # MEASURED inner floor plane (raised above the outer bottom)
        mk = lambda a, b, al: (torch.tensor(a, device=device), torch.tensor(b, device=device), al)
        return [
            mk(lo, (hi[0], hi[1], floor_top), THICK),                # floor (to inner plane)
            mk(lo, (hi[0], lo[1] + w, hi[2]), THIN),                 # front wall
            mk((lo[0], hi[1] - w, lo[2]), hi, THIN),                 # back wall
            mk(lo, (lo[0] + w, hi[1], hi[2]), THIN),                 # left wall
            mk((hi[0] - w, lo[1], lo[2]), hi, THIN),                 # right wall
        ]

    _vols = {}
    for _nm in ("E_drawer_1_7", "E_drawer_2_4", "E_drawer_3_1"):
        _vols[_nm] = (_B(_nm), _tray_slabs())
    _vols["E_door_1_12"] = (_B("E_door_1_12"), [
        (torch.tensor((0.0, 0.0, 0.0), device=device),
         torch.tensor((0.2196, 0.0069, 0.2482), device=device), THIN)])
    _vols["E_door_2_13"] = (_B("E_door_2_13"), [
        (torch.tensor((0.0255, 0.0056, 0.0), device=device),
         torch.tensor((0.2474, 0.0125, 0.2482), device=device), THIN)])
    # Shell slabs from the MEASURED mesh planes of E_bodyM1_10 (unique-coordinate
    # clustering) — the first model guessed wall thicknesses from the bbox and invented a
    # solid back wall 13.5 mm in front of the real back plane, flagging the resting top
    # drawer as -4.6 mm "penetration". The real shell: side rails, thin back plane, top
    # and bottom caps, NO shelf boards between bays.
    _shell_slabs = [  # (lo+hi, allowance)
        (torch.tensor((-0.2340, -0.1250, 0.0000, -0.2209, 0.1931, 0.2700), device=device), THICK),
        (torch.tensor((0.2158, -0.1250, 0.0000, 0.2340, 0.1931, 0.2700), device=device), THICK),
        (torch.tensor((-0.2340, 0.1900, 0.0000, 0.2340, 0.1931, 0.2700), device=device), THIN),
        (torch.tensor((-0.2340, -0.1250, 0.2554, 0.2340, 0.1931, 0.2700), device=device), THICK),
        (torch.tensor((-0.2340, -0.1250, 0.0000, 0.2340, 0.1931, 0.0142), device=device), THICK),
    ]
    min_sep = {"value": float("inf"), "raw": float("inf"), "step": -1, "pair": ""}
    # the deliverable is the RECORDED frames: sub-frame solver transients (an impact
    # penetrating a few mm for one 120 Hz step, then depenetrating) are real physics and
    # invisible in footage — gate on frame-capture steps, report the every-step min too
    min_sep_frames = {"value": float("inf"), "raw": float("inf"), "step": -1, "pair": ""}
    pair_min_frames: dict[str, float] = {}  # per-pair recorded-frame worst margin

    def _box_sdf(loc: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Signed distance of points to an AABB: positive outside, negative inside."""
        d_out = torch.clamp(torch.maximum(lo - loc, loc - hi), min=0.0).norm(dim=1)
        d_in = torch.minimum(loc - lo, hi - loc).amin(dim=1)
        return torch.where(d_in > 0, -d_in, d_out)

    def _world_pts(entry) -> torch.Tensor:
        kind, ref, grid = entry
        if kind == "link":
            p = scene.box.data.body_pos_w[0, ref]
            q = scene.box.data.body_quat_w[0, ref]
        else:
            p = ref.data.root_pos_w[0]
            q = ref.data.root_quat_w[0]
        return p + _qa(q.expand(len(grid), 4), grid)

    def _check_separation(at_frame: bool) -> None:
        """Track the worst MARGIN = sdf + slab allowance (negative = beyond what that
        slab's thickness/rest tolerances explain = a real crossing) plus the raw min."""
        pos = scene.box.data.body_pos_w[0]
        quat = scene.box.data.body_quat_w[0]
        bbody = _B(c.box_body)
        worst_m, worst_v, worst_pair = float("inf"), float("inf"), ""

        def _upd(m: float, v: float, pair: str) -> None:
            nonlocal worst_m, worst_v, worst_pair
            if m < worst_m:
                worst_m, worst_v, worst_pair = m, v, pair
            if at_frame and m < pair_min_frames.get(pair, float("inf")):
                pair_min_frames[pair] = m

        for src, entry in _clouds.items():
            pts_w = _world_pts(entry)
            n = len(pts_w)
            for dst, (bidx, slabs) in _vols.items():
                if dst == src:
                    continue
                if entry[0] == "link" and entry[1] == bidx:
                    continue
                lap = src in _DOORS and dst in _DOORS
                loc = _qai(quat[bidx].expand(n, 4), pts_w - pos[bidx])
                for lo, hi, allow in slabs:
                    v = float(_box_sdf(loc, lo, hi).min())
                    _upd(v + (DOOR_LAP if lap else allow), v, f"{src}->{dst}")
            loc = _qai(quat[bbody].expand(n, 4), pts_w - pos[bbody])
            for k, (slab, allow) in enumerate(_shell_slabs):
                v = float(_box_sdf(loc, slab[0:3], slab[3:6]).min())
                _upd(v + allow, v, f"{src}->shell{k}")
        if worst_m < min_sep["value"]:
            min_sep.update(value=worst_m, raw=worst_v, step=step_i, pair=worst_pair)
        if at_frame and worst_m < min_sep_frames["value"]:
            min_sep_frames.update(value=worst_m, raw=worst_v, step=step_i, pair=worst_pair)

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            if carry is not None:
                for name, pose in carry.items():
                    st = torch.zeros(n, 13, device=device)
                    st[:, 0:7] = pose
                    # cancel gravity integration: a zero-velocity re-pin free-falls g*dt
                    st[:, 9] = 9.81 * env.sim.get_physics_dt()
                    scene.items[name].write_root_state_to_sim(st, all_ids)
            env.step(no_action)
            at_frame = annot is not None and step_i % args.record_every == 0
            _check_separation(at_frame)
            if at_frame:
                # flush the RTX lighting accumulation before grabbing: soft-shadow/AO
                # sampling accumulates across rendered frames, and with one sparse render
                # per capture a teleported item's shadow imprint GHOSTS at its old spot
                # for dozens of captures (scissors-shaped smudge, user-flagged). The
                # ghost decays with RENDER COUNT (1/capture = strong at +18 captures;
                # 3/capture = faint at +54 renders), so feed the accumulator until it
                # converges — 8 flushes measured ghost-free at every capture.
                for _ in range(8):
                    env.sim.render()
                fr = np.asarray(annot.get_data())
                if fr.size:
                    frames.append(fr[..., :3].astype(np.uint8))
            step_i += 1

    def set_joint(j_col: int, target: float, ramp: int = 60) -> None:
        """Sweep ONE articulation joint to `target` with incremental state writes (the
        passive drives then hold it: damping, no stiffness)."""
        start = float(scene.box.data.joint_pos[0, j_col])
        for k in range(ramp):
            a = (k + 1) / ramp
            jp = scene.box.data.joint_pos.clone()
            jv = scene.box.data.joint_vel.clone()
            jp[:, j_col] = start + (target - start) * a
            jv[:, j_col] = 0.0
            scene.box.write_joint_state_to_sim(jp, jv, env_ids=all_ids)
            step()

    # Stapler/knife lie with the LONG axis across the drawer width; the SCISSORS lie along
    # the pull axis: when the closing drawer slides its contents into the front wall (real
    # inertia — the shut decelerates the tray, not the items), a tip-first contact stops
    # them dead, while a broadside contact let the thin blades CLIMB the wall and hang out
    # of the closed drawer. Centre drops keep the 22 cm scissors off the exposed-span rims.
    DROP_YAW = {"stapler": 0.0, "scissors": 0.0, "knife": math.pi / 2}
    # drop point along the tray (drawer-local y) and pull depth per item: the 22 cm
    # along-Y scissors need a deeper pull, and a drop centred so BOTH ends clear — at
    # 0.10 the tail (y -0.01) sat inside the tray's front-wall band during the low
    # carry pin (-2.5 mm visual intrusion, caught by the per-slab gate); at 0.13 the
    # tail clears the wall band (y 0.02 > 0.0025) and the head (y 0.24) hangs inside
    # the open bay below the roof, touching nothing
    DROP_Y = {"stapler": 0.14, "scissors": 0.13, "knife": 0.14}
    PULL = {"stapler": 0.22, "scissors": 0.24, "knife": 0.22}

    def drawer_world_drop_pose(name: str, d: int, z_up: float) -> torch.Tensor:
        """(n,7) pose above the EXPOSED (pulled-out) part of drawer d's tray, item flat,
        long axis across the tray. The tray centre is no good as a target: it sits 0.137
        behind the drawer face, i.e. mostly UNDER the shell roof even at full pull — the
        first smoke dropped the stapler onto the roof, where it slid off to the table."""
        from isaaclab.utils.math import quat_mul

        b = scene._drawer_b[d]
        bp = scene.box.data.body_pos_w[:, b]
        bq = scene.box.data.body_quat_w[:, b]
        target = torch.tensor((0.0, DROP_Y[name], 0.0), device=device).expand(n, 3)
        half = DROP_YAW[name] / 2
        qz = torch.tensor((math.cos(half), 0.0, 0.0, math.sin(half)), device=device).expand(n, 4)
        pose = torch.zeros(n, 7, device=device)
        pose[:, 0:3] = bp + quat_apply(bq, target)
        pose[:, 2] += z_up
        pose[:, 3:7] = quat_mul(bq, qz)
        return pose

    def carry_and_drop(name: str, d: int) -> None:
        """Kinematically carry item `name` above drawer d's (open) tray, then drop it in."""
        nonlocal carry
        lift = drawer_world_drop_pose(name, d, 0.30)
        carry = {name: lift}
        step(1)  # the pin teleports the item on this step
        if annot is not None:
            # the pickup TELEPORT leaves the item's stale soft-shadow imprint at its old
            # spot, strongest in the first captures after the jump — give the lighting
            # accumulator a dedicated settle right where the ghost is born
            for _ in range(20):
                env.sim.render()
        step(29)  # hover above the exposed tray
        low = drawer_world_drop_pose(name, d, 0.02)  # low release: soft, small transients
        low[:, 0] += 0.01  # off-centre on purpose: the drop is genuine, the tray captures it
        carry = {name: low}
        step(20)
        carry = None  # release -> free fall into the tray
        step(60)

    def report(tag: str) -> None:
        s = scene.score()[0].item()
        print(f"[smoke] {tag}: score={s} stowed={scene.stowed()[0].tolist()} "
              f"drawers={[round(v, 3) for v in scene.drawer_pos()[0].tolist()]} "
              f"doors={[round(v, 2) for v in scene.door_pos()[0].tolist()]} "
              f"success={bool(scene.success()[0])}", flush=True)

    def diagnose() -> None:
        """Every item's position in its ASSIGNED drawer's frame vs the tray box."""
        from isaaclab.utils.math import quat_apply_inverse

        pos = scene.box.data.body_pos_w
        quat = scene.box.data.body_quat_w
        print("[smoke]   links: " + " ".join(
            f"{nm}@z={pos[0, k, 2]:.3f}" for k, nm in enumerate(scene.box.body_names)), flush=True)
        print(f"[smoke]   joint order: {scene.box.joint_names} "
              f"drawer_j={scene._drawer_j} door_j={scene._door_j} drawer_b={scene._drawer_b}",
              flush=True)
        for i, (name, drawer, _m, _s, _z) in enumerate(c.manifest):
            b = scene._drawer_b[scene._assigned[i]]
            item = scene.items[name]
            loc = quat_apply_inverse(quat[0, b], item.data.root_pos_w[0] - pos[0, b])
            print(f"[smoke]   {name}: world={[round(v, 3) for v in item.data.root_pos_w[0].tolist()]} "
                  f"{drawer}-frame={[round(v, 3) for v in loc.tolist()]} "
                  f"tray c={c.tray_center} h={c.tray_half} "
                  f"|v|={item.data.root_lin_vel_w[0].norm().item():.3f}", flush=True)

    def save_frames() -> None:
        if frames:
            arr = np.stack(frames)
            np.savez_compressed(args.out, frames=arr, fps=120 // args.record_every // 2)
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            report("FAILED")
            diagnose()
            save_frames()
            raise AssertionError(msg)

    dj, doj = scene._drawer_j, scene._door_j
    items = [name for name, _d, _m, _s, _z in c.manifest]
    assigned = scene._assigned

    # ---- 1. show ----
    env.reset()
    step(60)
    report("show (reset layout)")
    expect(int(scene.score()[0]) == 0, "fresh layout must score 0")

    # ---- 2. oracle ----
    # Doors open to 115 deg: clear of the drawer channel (panel band ~1 cm outside the
    # pulled drawer's plan band) AND ~2 cm clear of the shell corner — at 128 deg the
    # kinematic release happened pressed against the corner and the contact impulse swung
    # the free door back over the drawer path (the crossing the user caught on camera).
    set_joint(doj[0], math.radians(-115.0))  # left door open
    set_joint(doj[1], math.radians(115.0))  # right door open
    want = 0
    for i, name in enumerate(items):
        d = assigned[i]
        set_joint(dj[d], -PULL[name])  # pull the drawer out (exposure for the drop)
        carry_and_drop(name, d)
        expect(bool(scene.stowed()[0, i]), f"{name} must be stowed after the drop")
        set_joint(dj[d], 0.0, ramp=150)  # shut GENTLY: a fast shut slams the contents
        # into the front wall (the tray decelerates, the items keep going)
        step(30)
        want += 30
        report(f"oracle: {name} stowed + drawer shut")
        expect(int(scene.score()[0]) == want, f"score must be {want} here")
    set_joint(doj[0], 0.0)
    set_joint(doj[1], 0.0)
    step(60)
    report("oracle: box shut")
    expect(int(scene.score()[0]) == 100, "full solve must score 100")
    expect(bool(scene.success()[0]), "full solve must be success()")

    if not args.demo:
        # ---- 3. negative A: wrong drawer ----
        env.reset()
        step(30)
        set_joint(doj[0], math.radians(-115.0))
        set_joint(doj[1], math.radians(115.0))
        set_joint(dj[1], -0.22)  # middle drawer
        carry_and_drop("knife", 1)  # knife belongs in the BOTTOM drawer
        expect(not bool(scene.stowed()[0, 2]), "knife in the middle tray must NOT count")
        expect(int(scene.score()[0]) == 0, "wrong-drawer stow must score 0")
        report("negative A (wrong drawer) OK")

        # ---- 4. negative B: the closed-gates staircase ----
        # (The first version closed the DOORS over a still-open drawer: the door sweep hit
        # the protruding tray and batted the stapler onto the table — physically right, so
        # the ordering "drawers before doors" is enforced by geometry; the staircase below
        # proves each score gate separately without fighting it.)
        env.reset()
        step(30)
        set_joint(doj[0], math.radians(-115.0))
        set_joint(doj[1], math.radians(115.0))
        for i, name in enumerate(items):
            set_joint(dj[assigned[i]], -PULL[name])
            carry_and_drop(name, assigned[i])
            if i != 0:
                set_joint(dj[assigned[i]], 0.0, ramp=150)
        step(20)
        expect(int(scene.score()[0]) == 80, "3 stowed + 2 drawers shut, top open = 80")
        expect(not bool(scene.success()[0]), "open drawer must block success")
        set_joint(dj[assigned[0]], 0.0, ramp=150)
        step(30)
        expect(int(scene.score()[0]) == 90, "all drawers shut, doors open = 90")
        expect(not bool(scene.success()[0]), "open doors must block success")
        set_joint(doj[0], 0.0)
        set_joint(doj[1], 0.0)
        step(40)
        expect(int(scene.score()[0]) == 100, "everything shut = 100")
        expect(bool(scene.success()[0]), "everything shut must be success")
        report("negative B (closed-gates staircase) OK")

        # ---- 5. negative C: item on the roof ----
        env.reset()
        step(30)
        roof = torch.zeros(n, 13, device=device)
        wx, wy = c.workbench_pos
        roof[:, 0] = wx + c.box_pos[0]
        roof[:, 1] = wy + c.box_pos[1]
        roof[:, 2] = c.surface_z + 0.32
        roof[:, 3] = 1.0
        roof[:, 0:3] += env.iscene.env_origins
        scene.items["stapler"].write_root_state_to_sim(roof, all_ids)
        step(40)
        expect(not bool(scene.stowed()[0, 0]), "an item on the roof must NOT count")
        report("negative C (roof) OK")

        # ---- 6. state round-trip at full score ----
        env.reset()
        step(30)
        set_joint(doj[0], math.radians(-115.0))
        set_joint(doj[1], math.radians(115.0))
        for i, name in enumerate(items):
            set_joint(dj[assigned[i]], -PULL[name])
            carry_and_drop(name, assigned[i])
            set_joint(dj[assigned[i]], 0.0, ramp=150)
        set_joint(doj[0], 0.0)
        set_joint(doj[1], 0.0)
        step(60)
        expect(int(scene.score()[0]) == 100, "round-trip precondition: full solve")
        saved = scene.get_state(all_ids)
        env.reset()
        step(20)
        expect(int(scene.score()[0]) == 0, "post-reset must score 0")
        scene.set_state(saved, all_ids)
        step(20)
        expect(int(scene.score()[0]) == 100, "set_state must restore the full-score state")
        expect(bool(scene.success()[0]), "set_state must restore success")
        report("state round-trip OK")

    # ---- the deliverable gate: measured VISUAL-mesh separation over the WHOLE run ----
    print(f"[smoke] worst separation MARGIN (raw sdf + per-slab allowance): every-step "
          f"{min_sep['value'] * 1000:.1f} mm (raw {min_sep['raw'] * 1000:.1f}) at step "
          f"{min_sep['step']} ({min_sep['pair']}) | RECORDED FRAMES: "
          f"{min_sep_frames['value'] * 1000:.1f} mm (raw {min_sep_frames['raw'] * 1000:.1f}) "
          f"at step {min_sep_frames['step']} ({min_sep_frames['pair']}; "
          f"negative margin = beyond that slab's rest tolerance)", flush=True)
    for pair, m in sorted(pair_min_frames.items(), key=lambda kv: kv[1])[:5]:
        print(f"[smoke]   recorded-frame margin {m * 1000:7.1f} mm  {pair}", flush=True)
    expect(min_sep_frames["value"] > 0.0,
           "no visual-mesh intrusion beyond per-slab allowance in any RECORDED frame")

    # ---- save the recording ----
    save_frames()
    if frames and args.hdfs_dir:
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} && hdfs dfs -put -f {args.out} {args.hdfs_dir}/")
        print(f"[smoke] uploaded {args.out} -> {args.hdfs_dir}/", flush=True)
    print("[smoke] ALL PHASES PASSED", flush=True)


if __name__ == "__main__":
    main()
    app.close()
