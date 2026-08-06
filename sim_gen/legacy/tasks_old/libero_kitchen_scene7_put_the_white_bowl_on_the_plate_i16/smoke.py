"""Smoke / oracle test for DishRackScene — NullRobot, kinematic-hold reorientation, RECORDED.

One linear run (pen_holder-smoke skeleton):
  1. show      — settle the reset layout; score must be 0;
  2. random    — randomization-is-real by READBACK across 3 seeds (rack xy + yaw move,
                 plate moves) + the Bernoulli layout mirror flips the dish side over 8 resets;
  3. null      — null policy: 240 idle steps must score ~0 and never succeed;
  4. oracle x3 — the loadout: per dish a gravity-compensated kinematic hold — lift, REORIENT
                 IN PLACE (plate 90 deg to vertical, bowl/cup 180 deg to opening-down),
                 translate, lower, release, real settling into slot / over peg; lift latch
                 pays 0.1 after the first lift; success() on seeds 0/1/2;
  5. rubric    — item-by-item instant placements (plate -> bowl -> cup): score strictly
                 increases and ends at 1.0 (order-free, final-state judging);
  6. negative A — the SEED's own goal state: the upright white bowl set ON the flat plate on
                 the counter (sanity-asserted as a genuine seed-success: xy within 6 cm,
                 bowl resting on the plate) must score ~0 here;
  7. negative B — on-top cheat at the rack: the plate laid FLAT across the rail tops (the
                 seed's on-top relation aimed at the target) is not racked;
  8. near-miss — bowl UPRIGHT on the peg (right place, wrong orientation) and bowl INVERTED
                 beside the peg (45 mm off — rim lands on the peg) are not counted;
  9. calibration — inverted-bowl capping sweep: offsets 0/10/20 mm all cap, 45 mm never;
                 plate released at 0/8 deg lean racks and settles within the geometric lean
                 bound (<= 20 deg) — the slot genuinely constrains the plate.

ALWAYS records video via the viewport rgb annotator (RTX driver-version override, 3-render
ghost flush) and saves frames.npz in the CURRENT WORKING DIRECTORY. Bodies are driven
straight through scene handles; the NullRobot applies nothing.

Run (forge): python -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

try:
    from .scene import DishRackSceneCfg  # noqa: E402  (import registers the scene + env)
except ImportError:  # pragma: no cover - bare-module fallback
    from scene import DishRackSceneCfg  # noqa: E402


# ----- small float-quaternion helpers ------------------------------------------------------------
def qmul(a: list, b: list) -> list:
    """Hamilton product a*b, (w,x,y,z) lists."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def qslerp(a: list, b: list, f: float) -> list:
    """Spherical interpolation between unit quats (shortest arc)."""
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = [-v for v in b]
        dot = -dot
    if dot > 0.9995:
        out = [x + (y - x) * f for x, y in zip(a, b)]
    else:
        th = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(th)
        out = [(math.sin((1 - f) * th) * x + math.sin(f * th) * y) / s
               for x, y in zip(a, b)]
    n = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / n for v in out]


Q_X90 = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]  # plate flat -> vertical
Q_FLIP = [0.0, 1.0, 0.0, 0.0]  # 180 deg about x: vessel opening -> straight down


# ----- driver: gravity-compensated kinematic hold + recording -----------------------------------
class Driver:
    """Steps the env; while `hold` is set to (body, state), that body's root state is
    rewritten before every physics step with +g*dt of upward velocity so PhysX's gravity
    integration cancels to zero (the pen_holder lesson: a naive zero-velocity re-pin
    free-falls g*dt per step). After each chunk the body is re-pinned at zero velocity and
    the scene buffers refreshed before judging."""

    def __init__(self, env) -> None:
        self.env = env
        self.scene = env.scene
        self.no_action = torch.empty(0, device=env.device)
        self.all_ids = torch.arange(env.num_envs, device=env.device)
        self.hold: tuple | None = None  # (body, state (N,13))
        self.g_dt = 9.81 * env.dt
        self.step_i = 0
        self.frames: list[np.ndarray] = []
        self.annot = None

    def make_state(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> torch.Tensor:
        st = torch.zeros(self.env.num_envs, 13, device=self.env.device)
        st[:, 0:3] = self.env.iscene.env_origins + torch.tensor(
            [float(v) for v in pos], device=self.env.device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=self.env.device)
        return st

    def step(self, k: int) -> None:
        for _ in range(k):
            if self.hold is not None:
                body, st = self.hold
                pre = st.clone()
                pre[:, 9] += self.g_dt  # cancel the gravity kick -> truly static hold
                body.write_root_state_to_sim(pre, self.all_ids)
            self.env.step(self.no_action, render=True)
            if self.annot is not None and self.step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    self.env.sim.render()
                arr = np.asarray(self.annot.get_data())
                if arr.size:
                    self.frames.append(arr[..., :3].astype(np.uint8).copy())
            self.step_i += 1
        if self.hold is not None:
            body, st = self.hold
            body.write_root_state_to_sim(st, self.all_ids)
            self.env.iscene.update(0.0)

    def settle_until(self, pred, max_steps: int = 300, poll: int = 15) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            self.step(poll)
            waited += poll
            if pred():
                return True
        return False


# ----- rack-frame helpers -------------------------------------------------------------------------
def rack_pose(scene) -> tuple[torch.Tensor, list]:
    """(rack world pos (3,), rack quat as list) for env 0."""
    return scene.rack.data.root_pos_w[0].clone(), [float(v) for v in
                                                   scene.rack.data.root_quat_w[0]]


def rack_to_world(scene, local) -> list:
    """Rack-frame point -> world point (env-origin INCLUDED), env 0."""
    from isaaclab.utils.math import quat_apply

    p, _q = rack_pose(scene)
    w = p + quat_apply(scene.rack.data.root_quat_w[0:1],
                       torch.tensor([[float(v) for v in local]],
                                    device=p.device))[0]
    return [float(v) for v in w]


def final_local_poses(c) -> dict:
    """The three goal poses in the RACK frame: (local pos, world-quat builder key)."""
    return {
        "plate": ([0.0, c.slot_y, c.slab_top_local + c.plate_r + 0.002], "vertical"),
        "bowl": ([c.peg1_xy[0], c.peg1_xy[1], c.slab_top_local + c.bowl_h / 2 + 0.003], "flip"),
        "cup": ([c.peg2_xy[0], c.peg2_xy[1], c.slab_top_local + c.cup_h / 2 + 0.003], "flip"),
    }


def goal_quat(scene, kind: str) -> list:
    """World quat for a goal pose: 'vertical' = rack_q * qx90 (plate edge-on across the
    slot), 'flip' = opening straight down (yaw-free)."""
    _p, rq = rack_pose(scene)
    return qmul(rq, Q_X90) if kind == "vertical" else Q_FLIP


# ----- the teleport-oracle loadout ---------------------------------------------------------------
def oracle_solution(scene_or_env, drv: Driver | None = None) -> dict:
    """Solve the CURRENT episode from its reset state: for each dish (plate -> bowl -> cup)
    kinematically hold it, lift, REORIENT IN PLACE (plate to vertical, vessels to
    opening-down), translate above its rack station, lower, release — real physics does the
    final seating (plate leans onto a rail, vessels drop over their pegs). One instant
    re-place recovery per item is allowed (teleport-oracle); judging stays physical."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    drv = drv or Driver(env)
    c = scene.cfg
    origin = env.iscene.env_origins[0]

    drv.step(50)  # settle the fresh layout
    goals = final_local_poses(c)
    preds = {"plate": lambda: bool(scene.plate_racked()[0]),
             "bowl": lambda: bool(scene.bowl_capped()[0]),
             "cup": lambda: bool(scene.cup_capped()[0])}
    carry_z = {"plate": 0.28, "bowl": 0.25, "cup": 0.25}
    score_after_lift = None
    recoveries = 0

    for name in ("plate", "bowl", "cup"):
        body = scene.dishes[name]
        local, kind = goals[name]
        target_p = rack_to_world(scene, local)
        target_p = [target_p[0] - float(origin[0]), target_p[1] - float(origin[1]),
                    target_p[2] - float(origin[2])]
        q1 = goal_quat(scene, kind)
        p0 = (body.data.root_pos_w[0] - origin).tolist()
        q0 = [float(v) for v in body.data.root_quat_w[0]]
        cz = carry_z[name]

        def hold(pos, quat) -> None:
            drv.hold = (body, drv.make_state(pos, quat))

        # a) lift straight up at the spawn spot
        for t in range(70):
            f = (t + 1) / 70
            hold((p0[0], p0[1], p0[2] + (cz - p0[2]) * f), q0)
            drv.step(1)
        if score_after_lift is None:
            score_after_lift = float(scene.score()[0])
        # b) reorient IN PLACE (nothing below to clip)
        for t in range(80):
            f = (t + 1) / 80
            hold((p0[0], p0[1], cz), qslerp(q0, q1, f))
            drv.step(1)
        # c) translate to above the rack station
        for t in range(90):
            f = (t + 1) / 90
            hold((p0[0] + (target_p[0] - p0[0]) * f,
                  p0[1] + (target_p[1] - p0[1]) * f, cz), q1)
            drv.step(1)
        # d) lower to just above the goal pose, release
        for t in range(90):
            f = (t + 1) / 90
            hold((target_p[0], target_p[1], cz + (target_p[2] - cz) * f), q1)
            drv.step(1)
        drv.hold = None
        drv.step(30)
        ok = drv.settle_until(preds[name], max_steps=240)
        if not ok:  # instant re-place recovery (teleport-oracle), then judge physically
            body.write_root_state_to_sim(drv.make_state(target_p, q1), drv.all_ids)
            recoveries += 1
            drv.step(30)
            ok = drv.settle_until(preds[name], max_steps=240)
        print(f"[smoke]   oracle {name}: seated={ok} score={float(scene.score()[0]):.2f}",
              flush=True)

    ok = drv.settle_until(lambda: bool(scene.success()[0]), max_steps=300)
    return {"success": ok, "score_after_lift": score_after_lift,
            "recoveries": recoveries, "final_score": float(scene.score()[0])}


# ----- main battery -------------------------------------------------------------------------------
def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("simgen.dish_rack")().build(
        num_envs=args.num_envs, device=device, scene_cfg=DishRackSceneCfg())
    scene = env.scene
    c = scene.cfg
    drv = Driver(env)
    origin0 = env.iscene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin0.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -1.10, 0.85)) + o),
                                tuple(np.array((0.05, -0.02, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        drv.annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        drv.annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(drv.annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)
        drv.annot = None

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:12s} | plate={bool(scene.plate_racked()[0])} "
              f"bowl={bool(scene.bowl_capped()[0])} cup={bool(scene.cup_capped()[0])} "
              f"score={float(scene.score()[0]):.3f} lifted={bool(scene.lifted[0])} "
              f"success={bool(scene.success()[0])} frames={len(drv.frames)}", flush=True)

    def diagnose(tag: str) -> None:
        for name, body in scene.dishes.items():
            loc = scene._rack_local(body.data.root_pos_w)[0]
            ax = scene._axis_w(body)[0]
            print(f"[smoke]   {tag} {name}: rack_loc=({loc[0]:.3f},{loc[1]:.3f},{loc[2]:.3f}) "
                  f"axis_z={float(ax[2]):.3f} "
                  f"|v|={float(body.data.root_lin_vel_w[0].norm()):.3f}", flush=True)

    def place(body, local, quat_w, lift_free: bool = False) -> None:
        """Instant kinematic placement at a rack-frame pose (world quat), buffers refreshed.
        Placement heights stay below the lift latch unless `lift_free`."""
        p = rack_to_world(scene, local)
        st = torch.zeros(env.num_envs, 13, device=env.device)
        st[:, 0:3] = torch.tensor(p, device=env.device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat_w], device=env.device)
        body.write_root_state_to_sim(st, drv.all_ids)
        env.iscene.update(0.0)

    goals = final_local_poses(c)

    # =========================== 1. show ====================================================
    torch.manual_seed(0)
    env.reset()
    report("reset")
    drv.step(60)
    report("show")
    finite = all(torch.isfinite(b.data.root_state_w).all()
                 for b in [scene.rack, *scene.dishes.values()])
    check("reset settles finite with score 0",
          finite and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # =========================== 2. randomization by readback ===============================
    obs = []
    for s in (201, 202, 203):
        torch.manual_seed(s)
        env.reset()
        drv.step(5)
        q = scene.rack.data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0] * q[3]), 1 - 2 * float(q[3]) ** 2)
        obs.append((scene.rack.data.root_pos_w[0, :2] - origin0[:2],
                    scene.plate.data.root_pos_w[0, :2] - origin0[:2], yaw))
    rack_moves = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    plate_moves = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    yaw_moves = min(abs((a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi)
                    for i, a in enumerate(obs) for j, b in enumerate(obs) if i < j)
    print(f"[smoke] randomization: rack dxy={rack_moves * 1000:.1f}mm "
          f"plate dxy={plate_moves * 1000:.1f}mm min rack dyaw={yaw_moves:.3f}rad", flush=True)
    check("randomization is real (rack + plate move, rack yaw varies)",
          rack_moves > 0.005 and plate_moves > 0.005 and yaw_moves > 0.03)
    sides = set()
    for s in range(210, 218):
        torch.manual_seed(s)
        env.reset()
        sides.add(1 if float(scene.bowl.data.root_pos_w[0, 1] - origin0[1]) > 0 else -1)
    print(f"[smoke] mirror sides over 8 resets: {sorted(sides)}", flush=True)
    check("mirror randomization flips the dish layout side", len(sides) == 2)

    # =========================== 3. null policy =============================================
    torch.manual_seed(42)
    env.reset()
    drv.step(240)
    report("null")
    check("null policy scores ~0 and no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # =========================== 4. oracle on 3 seeds ========================================
    for s in (0, 1, 2):
        torch.manual_seed(s)
        env.reset()
        m = oracle_solution(env, drv)
        report(f"oracle-s{s}")
        print(f"[smoke]   oracle seed {s}: recoveries={m['recoveries']} "
              f"final_score={m['final_score']:.2f}", flush=True)
        if s == 0:
            check("lift latch pays 0.1 after first lift, before any racking",
                  0.09 <= m["score_after_lift"] <= 0.11)
        check(f"oracle solve reaches success() on seed {s}", m["success"])
        if not m["success"]:
            diagnose(f"oracle-s{s}")

    # =========================== 5. rubric monotonicity (item-by-item) =======================
    torch.manual_seed(7)
    env.reset()
    drv.step(50)
    scores = [float(scene.score()[0])]
    mono_ok = True
    for name, pred in (("plate", lambda: bool(scene.plate_racked()[0])),
                       ("bowl", lambda: bool(scene.bowl_capped()[0])),
                       ("cup", lambda: bool(scene.cup_capped()[0]))):
        local, kind = goals[name]
        place(scene.dishes[name], local, goal_quat(scene, kind))
        mono_ok &= drv.settle_until(pred, max_steps=240)
        scores.append(float(scene.score()[0]))
    report("item-by-item")
    print(f"[smoke] monotonic score sequence: {[f'{v:.3f}' for v in scores]}", flush=True)
    increasing = all(b > a + 1e-6 for a, b in zip(scores, scores[1:]))
    check("rubric strictly increases item-by-item", mono_ok and increasing)
    check("item-by-item loadout reaches 1.0 + success",
          abs(scores[-1] - 1.0) < 1e-6 and bool(scene.success()[0]))
    if not (mono_ok and increasing):
        diagnose("mono")

    # =========================== 6. negative A: the seed's own strategy ======================
    # "Put the white bowl on the plate": leave the plate flat on the counter and set the
    # upright bowl on it. A perfect execution of the SEED task — and must score ~0 here.
    torch.manual_seed(11)
    env.reset()
    drv.step(50)
    plate_p = (scene.plate.data.root_pos_w[0] - origin0).tolist()
    st = drv.make_state((plate_p[0], plate_p[1],
                         plate_p[2] + c.plate_t / 2 + c.bowl_h / 2 + 0.002))
    scene.bowl.write_root_state_to_sim(st, drv.all_ids)
    drv.step(60)
    report("seed-strat")
    bowl_p = scene.bowl.data.root_pos_w[0]
    plate_pw = scene.plate.data.root_pos_w[0]
    xy_d = float((bowl_p[:2] - plate_pw[:2]).norm())
    gap = float(bowl_p[2] - c.bowl_h / 2 - (plate_pw[2] + c.plate_t / 2))
    seed_goal_met = xy_d < 0.06 and abs(gap) < 0.010  # the seed's own On(bowl, plate)
    print(f"[smoke] seed-strat: xy_d={xy_d * 1000:.1f}mm rest_gap={gap * 1000:.1f}mm",
          flush=True)
    check("negative A: seed strategy (upright bowl on the flat plate) scores ~0",
          seed_goal_met and float(scene.score()[0]) <= 0.05
          and int(scene.n_racked()[0]) == 0 and not bool(scene.success()[0]))
    if not seed_goal_met:
        diagnose("seed-strat")

    # =========================== 7. negative B: flat-on-rails cheat ==========================
    torch.manual_seed(12)
    env.reset()
    drv.step(50)
    place(scene.plate, [0.0, c.slot_y, c.slab_top_local + c.rail_h + c.plate_t / 2 + 0.002],
          (1.0, 0.0, 0.0, 0.0))
    drv.step(60)
    loc = scene._rack_local(scene.plate.data.root_pos_w)[0]
    on_rails = abs(float(loc[1]) - c.slot_y) < 0.05 and float(loc[2]) > c.slab_top_local + 0.03
    print(f"[smoke] flat-on-rails: rack_loc=({loc[0]:.3f},{loc[1]:.3f},{loc[2]:.3f})",
          flush=True)
    check("negative B: plate laid flat across the rail tops is not racked",
          on_rails and not bool(scene.plate_racked()[0]) and float(scene.score()[0]) <= 0.05)

    # =========================== 8. near-miss controls =======================================
    torch.manual_seed(13)
    env.reset()
    drv.step(50)
    # bowl UPRIGHT resting on the peg top: right station, wrong orientation
    place(scene.bowl, [c.peg1_xy[0], c.peg1_xy[1],
                       c.slab_top_local + c.peg_h + c.bowl_h / 2 + 0.002],
          (1.0, 0.0, 0.0, 0.0))
    drv.step(60)
    check("near-miss: upright bowl on the peg is not counted",
          not bool(scene.bowl_capped()[0]))
    # bowl INVERTED but 45 mm off the peg: rim lands on the peg -> tilted / out of band
    place(scene.bowl, [c.peg1_xy[0] + 0.045, c.peg1_xy[1],
                       c.slab_top_local + c.bowl_h / 2 + 0.004], Q_FLIP)
    drv.step(60)
    loc = scene._rack_local(scene.bowl.data.root_pos_w)[0]
    print(f"[smoke] beside-peg: rack_loc=({loc[0]:.3f},{loc[1]:.3f},{loc[2]:.3f}) "
          f"axis_z={float(scene._axis_w(scene.bowl)[0, 2]):.3f}", flush=True)
    check("near-miss: inverted bowl beside the peg (45 mm off) is not counted",
          not bool(scene.bowl_capped()[0]))

    # =========================== 9. calibration probes =======================================
    # A) capping tolerance cliff: centred and small offsets cap; 45 mm never does.
    cap_results = {}
    for off in (0.0, 0.010, 0.020, 0.045):
        torch.manual_seed(14)
        env.reset()
        drv.step(30)
        place(scene.bowl, [c.peg1_xy[0] + off, c.peg1_xy[1],
                           c.slab_top_local + c.bowl_h / 2 + 0.004], Q_FLIP)
        drv.step(20)
        cap_results[off] = drv.settle_until(lambda: bool(scene.bowl_capped()[0]),
                                            max_steps=150)
    print(f"[smoke] CALIBRATION capping: "
          f"{ {f'{int(k * 1000)}mm': v for k, v in cap_results.items()} }", flush=True)
    check("calibration: capping offsets <= 20 mm all cap, 45 mm never does",
          cap_results[0.0] and cap_results[0.010] and cap_results[0.020]
          and not cap_results[0.045])
    # B) slot lean capture: plate released at 0 and 5 deg lean racks; the settled lean
    # stays within the geometric bound (rails/gap -> ~13 deg; assert <= 20). A leaned disc
    # pivots about its BOTTOM edge, so the release pose keeps that edge on the slot
    # centreline (centre offset -r*sin, height r*cos) — a centre-pivot lean would clip a
    # rail at placement.
    leans = []
    for lean_deg in (0.0, 5.0):
        torch.manual_seed(15)
        env.reset()
        drv.step(30)
        _p, rq = rack_pose(scene)
        th = math.radians(lean_deg)
        h = th / 2
        q_lean = qmul(qmul(rq, [math.cos(h), math.sin(h), 0.0, 0.0]), Q_X90)
        place(scene.plate,
              [0.0, c.slot_y - c.plate_r * math.sin(th),
               c.slab_top_local + c.plate_r * math.cos(th) + 0.004], q_lean)
        drv.step(20)
        ok = drv.settle_until(lambda: bool(scene.plate_racked()[0]), max_steps=200)
        settled_lean = math.degrees(math.asin(min(1.0,
                                                  abs(float(scene._axis_w(scene.plate)[0, 2])))))
        leans.append((lean_deg, ok, settled_lean))
    print(f"[smoke] CALIBRATION slot: {[(a, b, f'{v:.1f}deg') for a, b, v in leans]}",
          flush=True)
    check("calibration: plate released at 0/5 deg lean racks with settled lean <= 20 deg",
          all(ok and v <= 20.0 for _a, ok, v in leans))

    # =========================== save + verdict =============================================
    arr = (np.stack(drv.frames, axis=0) if drv.frames
           else np.zeros((0, 600, 960, 3), np.uint8))
    np.savez_compressed(args.out, frames=arr, env="simgen.dish_rack")
    print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    n_ok = sum(ok for _n, ok in checks)
    if n_ok == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_ok}/{len(checks)}", flush=True)
    else:
        for name, ok in checks:
            if not ok:
                print(f"[smoke] FAILED CHECK: {name}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_ok}/{len(checks)}", flush=True)

    # Kit teardown hangs are routine — hard-exit behind a watchdog.
    threading.Timer(10.0, lambda: os._exit(0)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
