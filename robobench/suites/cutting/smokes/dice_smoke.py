"""Dice smoke — no robot: the KNIFE is driven kinematically through the dicing choreography.

NullRobot env on `cutting.dice`. The smoke lifts the knife off its rest and writes its pose
every step: settle, then for every scored plane (x-planes, then the y-planes with the blade
turned 90 deg) move above the plane's LIVE cut point, descend gently with the edge level until
the SCENE gate has released every weld pair of that plane (or the edge reaches the board),
hold briefly, lift. Every release is the scene's own per-pair gate; separation is real
physics. Verdict = `DiceFoodGrader.check_success()`.

Physics/gate validation only (a script-moved body renders behind the physics on this stack).

python -m robobench.suites.cutting.smokes.dice_smoke --headless
python -m robobench.suites.cutting.smokes.dice_smoke --food potato --planes 1 --livestream 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="cutting.dice")
parser.add_argument("--food", default="", help="override the scene's food (e.g. potato)")
parser.add_argument("--planes", type=int, default=0, help="stop after this many planes (0 = all)")
parser.add_argument("--gate", nargs="*", default=[], metavar="KEY=VAL",
                    help="scene cfg overrides, e.g. depth_past_center=0.01")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

import math  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.cutting.smokes import close_and_exit  # noqa: E402

# gentle chop choreography (sim seconds)
MOVE_T, DESCENT_T, DWELL_T, HOLD_T, LIFT_T, SETTLE_T = 1.0, 3.0, 1.0, 0.5, 1.0, 1.5


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    from robobench.suites.cutting.grader import DiceFoodGrader
    from robobench.suites.cutting.scenes.dice_food import DiceFoodSceneCfg

    env_cfg = ENVS.get(args.env)()
    if args.food or args.gate:
        over = {kv.split("=", 1)[0]: float(kv.split("=", 1)[1]) for kv in args.gate}
        env_cfg.scene_cfg = DiceFoodSceneCfg(food=args.food or DiceFoodSceneCfg.food, **over)
    env = env_cfg.build(num_envs=args.num_envs, device=device)
    scene = env.scene
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    dt = env.physics_dt if hasattr(env, "physics_dt") else 1.0 / 240.0

    env.reset()
    grader = DiceFoodGrader(env)

    board_top = scene._board_top
    park_z = board_top + scene.food_height() + 0.10
    down_z = board_top + 0.001
    q_chop = tuple(float(v) for v in scene.cfg.knife_rot)  # edge-level chop pose
    el = scene._edge_local
    edge_local = el[(el[:, 0] - 0.07).abs().argmin()]  # mid-blade point of the REAL edge
    origin0 = env.iscene.env_origins[0]

    def _rotmat(q):
        w, x, y, z = [float(v) for v in q]
        return torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], device=device)

    def live_plane(idx):
        """Env-0 xy of the plane's LIVE cut point (mean of its weld pairs) + the blade yaw
        from the live plane normal (a y-plane comes out turned 90 deg)."""
        aim = scene.plane_aim_w(idx)[0] - origin0
        nrm = scene.plane_axis_w(idx)[0]
        return float(aim[0]), float(aim[1]), math.atan2(float(nrm[1]), float(nrm[0]))

    def gate_forensics(idx):
        """The scene gate's terms for env 0, per weld pair of a plane."""
        from isaaclab.utils.math import quat_apply

        c = scene.cfg
        axis, p = scene.planes[idx]
        kp, kq = scene.knife.data.root_pos_w[:1], scene.knife.data.root_quat_w[:1]
        N = scene._edge_local.shape[0]
        pts = kp + quat_apply(kq.expand(N, 4), scene._edge_local)
        bn = quat_apply(kq, torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        aim, ax_w = scene._pair_geometry()
        for k in scene._plane_pairs[idx]:
            a, axv = aim[0, k], ax_w[0, k]
            j = (pts[:, :2] - a[:2]).norm(dim=-1).argmin()
            rel = pts[j] - a
            perp = torch.stack([-axv[1], axv[0], torch.zeros((), device=device)])
            thr = max(float(a[2]) - c.depth_past_center, scene._board_top + 0.0015)
            pa, pb = scene.pairs[k][0], scene.pairs[k][1]
            print(f"SMOKE plane {idx} ({axis}{p:+.3f}) pair {scene.cells[pa]}-{scene.cells[pb]} "
                  f"cut={bool(scene.pair_cut[0, k])} near={float((rel * axv).sum()) * 1000:+.1f}mm(|.|<6) "
                  f"flesh={float((rel * perp).sum()) * 1000:+.1f}mm(|.|<25) "
                  f"align={math.degrees(math.acos(min(1.0, abs(float((bn * axv).sum()))))):.1f}deg(<15) "
                  f"edge_z={float(pts[j][2]):.4f} thr={thr:.4f}", flush=True)

    st = torch.zeros(n, 13, device=device)
    cur = [0.0, 0.0, park_z]
    cur_q = [q_chop]

    def hold(x, y, ez):
        """Place the mid-blade edge point at (x, y, ez) with the current blade yaw."""
        R = _rotmat(cur_q[0])
        root = torch.tensor([x, y, ez], device=device) - R @ edge_local
        st[:, 0:3] = root + env.iscene.env_origins
        st[:, 3:7] = torch.tensor(cur_q[0], device=device)
        st[:, 7:] = 0.0
        scene.knife.write_root_state_to_sim(st)

    def advance(seconds, target=None, stop_on_cut=None):
        a = list(cur)
        n_steps = max(1, int(seconds / dt))
        for k in range(n_steps):
            if stop_on_cut is not None and bool(scene.cut[0, stop_on_cut]):
                t01 = k / n_steps
                cur[0], cur[1], cur[2] = [a[i] + t01 * (target[i] - a[i]) for i in range(3)]
                return
            if target is not None:
                t01 = min(1.0, k / n_steps)
                hold(a[0] + t01 * (target[0] - a[0]), a[1] + t01 * (target[1] - a[1]),
                     a[2] + t01 * (target[2] - a[2]))
            else:
                hold(*cur)
            env.step(no_action, render=render)
        if target is not None:
            cur[0], cur[1], cur[2] = target

    wx, wy, _ = live_plane(0)
    cur[0], cur[1] = wx, wy
    advance(SETTLE_T)
    order = list(range(len(scene.planes)))  # x-planes, then the y-planes
    if args.planes > 0:
        order = order[:args.planes]
    for idx in order:
        wx, wy, yaw = live_plane(idx)
        cur_q[0] = _qmul((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), q_chop)
        advance(MOVE_T, [wx, wy, park_z])
        wx, wy, _ = live_plane(idx)  # re-aim after the move
        advance(DESCENT_T, [wx, wy, down_z], stop_on_cut=idx)
        if bool(scene.cut[0, idx]):
            advance(HOLD_T)  # hold in the split so it reads, then lift
        else:
            advance(DWELL_T)
            gate_forensics(idx)
        advance(LIFT_T, [cur[0], cur[1], park_z])
        print(f"SMOKE plane {idx} cut={bool(scene.cut[0, idx])}", flush=True)
    advance(SETTLE_T)

    print(f"SMOKE [{args.env}] planes cut={scene.cut[0].int().tolist()} pairs cut="
          f"{int(scene.pair_cut[0].sum())}/{scene.pair_cut.shape[1]} "
          f"pieces={scene.pieces_count().tolist()} "
          f"success={grader.check_success().tolist()} progress={grader.progress().tolist()}",
          flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
