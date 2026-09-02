"""Slice smoke — no robot: the KNIFE is driven kinematically through the chop choreography.

NullRobot env. The smoke takes the presented knife off its stand and writes its pose every
step: settle, then for every scored plane (end planes last) move above the plane, descend
gently with the edge level until the SCENE gate releases the plane's weld (or the edge
reaches the board), hold briefly, lift. Every release is the scene's own gate on the live
food frame; separation is real physics. Verdict = `SliceFoodGrader.check_success()`.

Physics/gate validation only. A body moved by script renders behind the physics on this
stack, so a video of this smoke shows the knife lagging the split — record the arm-driven
solution (experiments/) for truthful footage.

python -m robobench.suites.cutting.smokes.slice_smoke --env cutting.slice --headless
python -m robobench.suites.cutting.smokes.slice_smoke --food banana --planes 1 --livestream 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="cutting.slice")
parser.add_argument("--food", default="", help="override the scene's food (e.g. banana)")
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
    from robobench.suites.cutting.grader import SliceFoodGrader
    from robobench.suites.cutting.scenes.slice_food import SliceFoodSceneCfg

    env_cfg = ENVS.get(args.env)()
    if args.food or args.gate:
        over = {kv.split("=", 1)[0]: float(kv.split("=", 1)[1]) for kv in args.gate}
        env_cfg.scene_cfg = SliceFoodSceneCfg(food=args.food or "carrot", **over)
    env = env_cfg.build(num_envs=args.num_envs, device=device)
    scene = env.scene
    render = (not args.headless) or livestream_on
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    dt = env.physics_dt if hasattr(env, "physics_dt") else 1.0 / 240.0
    ids = torch.arange(n, device=device)

    env.reset()
    grader = SliceFoodGrader(env)
    scene.set_knife_staged(ids, False)  # this smoke drives the knife itself

    board_top = scene._board_top
    park_z = board_top + scene.food_height() + 0.10
    down_z = board_top + 0.001
    q_chop = tuple(float(v) for v in scene.cfg.knife_rot)  # edge-level chop pose
    # the mid-blade point of the REAL edge (knife frame) — what we place at the aim
    el = scene._edge_local
    edge_local = el[(el[:, 0] - 0.07).abs().argmin()]
    ref_i = len(scene.pieces) // 2
    ref_cent = torch.tensor(scene._cents[ref_i], device=device)

    def _rotmat(q):
        w, x, y, z = [float(v) for v in q]
        return torch.tensor([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], device=device)

    def live_plane(idx):
        """Env-0 world xy of the plane's flesh centre + blade yaw, in the LIVE food frame."""
        rq = scene.pieces[ref_i].data.root_quat_w[0]
        rp = scene.pieces[ref_i].data.root_pos_w[0]
        R = _rotmat(rq)
        world = rp + R @ (scene._plane_aims[idx] - ref_cent)
        nx = R @ torch.tensor([1.0, 0.0, 0.0], device=device)
        return float(world[0]), float(world[1]), math.atan2(float(nx[1]), float(nx[0]))

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

    # take the knife off its stand and settle the food
    wx, wy, yaw = live_plane(0)
    cur[0], cur[1] = wx, wy
    advance(SETTLE_T)
    order = list(range(len(scene.planes)))
    order = order[1:] + order[:1]  # end planes LAST (a curled end rocks the whole food)
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
        advance(LIFT_T, [cur[0], cur[1], park_z])
        print(f"SMOKE plane {idx} cut={bool(scene.cut[0, idx])}", flush=True)
    advance(SETTLE_T)

    print(f"SMOKE [{args.env}] planes cut={scene.cut[0].int().tolist()} "
          f"pieces={scene.pieces_count().tolist()} "
          f"success={grader.check_success().tolist()} progress={grader.progress().tolist()}",
          flush=True)
    close_and_exit(env, app)


if __name__ == "__main__":
    main()
