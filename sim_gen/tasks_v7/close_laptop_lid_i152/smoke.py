"""Smoke / rubric-REJECTION battery for BallastLidChestScene — NullRobot, teleported probes.

solve.py is the acceptance proof (ballast the tray, gravity closes the lid). This
battery proves the rubric REJECTS wrong outcomes, and that the counterweight — the
task's strategic differentiator from the seed — is physically load-bearing. Every
probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge);
no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; lid resting at its 46 deg
                           open stop, tray empty, all blocks on the ground; score 0,
                           no success;
   2. randomization      — two seeded resets: READBACK chest yaw, chest xy, block0
                           scatter (chest-local xy) and block0 relative yaw all differ;
   3. null-policy        — 240 idle steps: the lid keeps resting at its open stop,
                           score ~0, no success;
   4. SEED STRATEGY      — the seed's plan ("push the lid shut" and nothing else):
                           lid CONSTRUCTED closed (pure hinge-consistent quaternion
                           write, zero velocity), NO ballast -> the counterweight
                           swings it back OPEN past 30 deg; no success, score ~0.
                           This is the physics that forces the ballast strategy;
   5. one-block near-miss— ONE block seated in the tray: the lid must stay at its
                           open stop (counterweight still wins by ~0.36 N.m); no
                           success, score <= 0.30 (ballast1 credit only);
   6. wrong place        — TWO blocks dropped inside the CHEST CAVITY instead of the
                           lid tray: no closing leverage, lid stays open, tray count
                           0, no success, score ~0;
   7. near-miss ajar     — two blocks properly in the tray but a third block wedged
                           on the chest's front rim holds the lid AJAR (~6 deg > the
                           3 deg tolerance): no success, score <= 0.70 (all partial
                           credit, no success bonus);
   8. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_laptop_lid_i152.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qy = task_scene._qmul, task_scene._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    ang = float(scene.open_angle_deg()[0])
    print(f"[smoke] {tag:16s} | angle={ang:+6.2f}deg tray={int(scene.tray_count()[0])} "
          f"b1={bool(scene._ballast1[0])} b2={bool(scene._ballast2[0])} "
          f"cl={bool(scene._closing[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_lid_chest")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.70, 0.62)) + o),
                                tuple(np.array((0.42, 0.00, 0.20)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def ang() -> float:
        _refresh()
        return float(scene.open_angle_deg()[0])

    def hinge_world() -> torch.Tensor:
        _refresh()
        h = torch.tensor([c.hinge_x, 0.0, c.hinge_z], device=device).expand(n, 3)
        return scene.chest.data.root_pos_w + quat_apply(scene.chest.data.root_quat_w, h)

    def set_lid_angle(deg: float) -> None:
        """Teleport the lid to opening angle `deg` — the lid origin sits ON the hinge
        axis, so this is a hinge-consistent pure pose write (zero velocity)."""
        q = _qmul(scene.chest.data.root_quat_w,
                  _qy(torch.full((n,), -math.radians(deg), device=device)))
        _write_body(scene.lid, hinge_world(), q)

    def drop_into_tray(i: int, y_loc: float) -> None:
        """Release block i inside the tray airspace (the solve's transport)."""
        pt = torch.zeros(n, 3, device=device)
        pt[:, 0] = (c.tray_x0 + c.tray_x1) / 2
        pt[:, 1] = y_loc
        pt[:, 2] = 0.035
        _refresh()
        pos = scene.lid.data.root_pos_w + quat_apply(scene.lid.data.root_quat_w, pt)
        _write_body(scene.blocks[i], pos, scene.lid.data.root_quat_w)

    def chest_world(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.chest.data.root_pos_w + quat_apply(scene.chest.data.root_quat_w, loc)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    gz = [float(b.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
          for b in scene.blocks]
    check("settle/no-NaN: layout settles finite; lid resting at its open stop, tray "
          "empty, blocks on the ground; score 0, no success",
          bool(scene._finite()[0]) and abs(ang() - c.open_stop_deg) < 3.0
          and int(scene.tray_count()[0]) == 0 and max(gz) < 0.10
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        cyaw = yaw_of(scene.chest.data.root_quat_w[0])
        cp = scene.chest.data.root_pos_w[0, :2].clone()
        b0 = scene._chest_local(scene.blocks[0].data.root_pos_w)[0, :2].clone()
        byaw = dyaw(yaw_of(scene.blocks[0].data.root_quat_w[0]), cyaw)
        return cyaw, cp, b0, byaw

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_cp, a_b0, a_by = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_cp, b_b0, b_by = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_cp = float((a_cp - b_cp).norm())
    d_b0 = float((a_b0 - b_b0).norm())
    d_by = abs(a_by - b_by)
    print(f"[smoke] randomization deltas: chest_yaw={d_yawv:.1f}deg "
          f"chest_xy={d_cp * 1000:.1f}mm block0_scatter={d_b0 * 1000:.1f}mm "
          f"block0_rel_yaw={d_by:.1f}deg", flush=True)
    check("randomization-is-real: chest yaw, chest xy, block scatter and block yaw "
          "readback all differ across seeds",
          d_yawv > 2.0 and d_cp > 0.003 and d_b0 > 0.003 and d_by > 2.0)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the lid keeps resting at its open "
          "stop, score ~0, no success",
          abs(ang() - c.open_stop_deg) < 3.0 and float(scene.score()[0]) <= 0.05
          and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # "Close the laptop lid" — push the hinged panel shut, release, done. CONSTRUCT
    # its end state: lid at 0 deg (hinge-consistent write, zero velocity), NO
    # ballast anywhere. The counterweight must swing it back OPEN — the physics
    # that voids the seed's plan and forces the ballast strategy.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_lid_angle(0.0)
    _step(480)  # 4 s: swing back up, settle onto the stop
    _report("seed-strategy")
    _REC["on"] = False
    check("negative (SEED strategy): lid pushed fully shut with NO ballast swings "
          "back OPEN past 30 deg on its own — no success, score ~0",
          ang() > 30.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 5. near-miss: one block is not enough ======================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    drop_into_tray(0, -0.050)
    _step(300)
    _report("one-block")
    _REC["on"] = False
    check("near-miss (one block): a single block seated in the tray leaves the lid "
          "at its open stop (counterweight wins) — no success, score <= 0.30",
          int(scene.tray_count()[0]) == 1 and ang() > c.open_stop_deg - 12.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)

    # ================= 6. negative: blocks in the WRONG place (chest cavity) ======================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    for i, y in ((0, -0.06), (1, 0.06)):
        p = chest_world((0.02, y, 0.09))
        _write_body(scene.blocks[i], p, scene.chest.data.root_quat_w)
    _step(300)
    _report("wrong-place")
    check("negative (wrong place): two blocks dropped inside the CHEST CAVITY give "
          "no closing leverage — lid stays open, tray count 0, no success, score ~0",
          int(scene.tray_count()[0]) == 0 and ang() > c.open_stop_deg - 12.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 7. near-miss: lid held AJAR by a wedged block ==============================
    # Two blocks properly in the tray, but the third block sits on the chest's FRONT
    # rim: the closing lid lands on it and rests ~6 deg open — outside the 3 deg
    # tolerance. All three latches fire (that is the point: latched partial credit
    # stays), but success must refuse and the score must stay at the 0.70 cap.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    # block2 on the front wall top (12 mm ledge; the lid pins it before it can tip)
    p = chest_world((0.174, 0.0, c.wall_h + c.block_size[2] / 2 + 0.003))
    _write_body(scene.blocks[2], p, scene.chest.data.root_quat_w)
    # lid to 8 deg (just above the block-contact angle), then ballast the tray
    set_lid_angle(8.0)
    drop_into_tray(0, -0.050)
    drop_into_tray(1, +0.050)
    _step(360)
    _report("ajar")
    _REC["on"] = False
    a = ang()
    print(f"[smoke] ajar rest angle: {a:.2f} deg (tolerance {c.closed_max_deg:.1f})",
          flush=True)
    check("near-miss (ajar): two blocks in the tray but the lid rests on a block "
          "wedged on the chest rim, ~6 deg open — no success, score capped at 0.70",
          c.closed_max_deg + 0.5 < a < 15.0 and int(scene.tray_count()[0]) >= 2
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7000005)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_lid_chest")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SMOKE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
