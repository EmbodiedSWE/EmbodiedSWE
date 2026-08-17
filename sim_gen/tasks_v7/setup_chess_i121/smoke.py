"""Smoke / rubric-REJECTION battery for CastlingGalleryScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct contact-pushed
trajectory with monotone latched credit). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the flanges are
CAPTIVE under the lip rails, and the one-piece-wide gallery makes the order
mandatory — are load-bearing physics, not fiat. Every probe is CONSTRUCTED as a
settled state (teleport / short force, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN     — seeded reset settles finite; both pieces upright and
                          captive in the gallery; score 0, no success;
   2. randomization     — two seeded resets: READBACK plinth yaw, plinth xy and
                          the piece start spots all differ;
   3. target-side       — over 10 resets both pocket sides occur, and the beacon
                          tile readback sits on the drawn target side every time;
   4. null-policy       — 240 idle steps: score ~0, no success;
   5. LIP CAPTIVITY     — physical probe with a proven actuator: a small
                          horizontal shove first MOVES the rook >= 5 mm (the
                          wrench path works), then a 3x-weight straight-UP pull
                          for 1.5 s AT THE JUNCTION (the network's largest lip
                          opening) raises it < 12 mm and it stays captive —
                          "cannot be lifted out" is physics, not fiat;
   6. SEED STRATEGY     — the seed's plan (free pick-and-place): both pieces
                          PLACED from above ON TOP of the lip rails exactly over
                          their goal cells -> not IN the channels, no success,
                          score ~0;
   7. near-miss         — rook correctly seated, king constructed 6 cm short of
                          home -> no success, score capped at 0.75;
   8. wrong pocket      — rook at the DECOY pocket's end, king at home: the
                          full "looks castled" tableau on the wrong side -> no
                          success, score ~0;
   9. swapped pieces    — KING at the target pocket end, ROOK at the gold home
                          cell -> no success, score ~0;
  10. ORDER IS PHYSICS  — fresh reset, king pushed straight for home with the
                          rook still in the gallery: the king provably advances
                          (>= 3 cm) yet shoves the rook train into the gold stop
                          and can never enter the home band -> no success;
  11. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_chess_i121.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

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


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    k, r = scene._piece_loc()
    print(f"[smoke] {tag:16s} | king=({float(k[0, 0]):+.3f},{float(k[0, 1]):+.3f},"
          f"{float(k[0, 2]):+.3f}) rook=({float(r[0, 0]):+.3f},{float(r[0, 1]):+.3f},"
          f"{float(r[0, 2]):+.3f}) l1={bool(scene._l1[0])} l2={bool(scene._l2[0])} "
          f"l3={bool(scene._l3[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.castling_gallery")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.95, -0.80, 0.72)) + o),
                                tuple(np.array((0.0, 0.0, 0.14)) + o),
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

    def plinth_dir(local_dir) -> torch.Tensor:
        """Plinth-local unit direction -> world, (n,3)."""
        _refresh()
        d = torch.tensor(local_dir, device=device, dtype=torch.float).expand(n, 3)
        return quat_apply(scene.plinth.data.root_quat_w, d)

    def place(body, x, y, z) -> None:
        """Teleport `body` to plinth-local (x, y, z) with the plinth's heading."""
        _refresh()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = x
        loc[:, 1] = y
        loc[:, 2] = z
        _write_body(body, scene.local_to_world(loc), scene.plinth.data.root_quat_w)

    z_chan = c.plinth_h + 0.002                      # flange on the channel floor
    z_lips = c.plinth_h + c.wall_h + c.lip_h + 0.002  # standing ON TOP of the lip rails
    xj = c.junction_x

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    k0, r0 = scene._piece_loc()
    check("settle/no-NaN: seeded reset settles finite; both pieces upright and captive "
          "in the gallery; score 0, no success",
          bool(torch.isfinite(scene.king.data.root_state_w).all())
          and bool(torch.isfinite(scene.rook.data.root_state_w).all())
          and bool(scene.in_channel(k0)[0]) and bool(scene.in_channel(r0)[0])
          and bool(scene.upright(scene.king)[0]) and bool(scene.upright(scene.rook)[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        k, r = scene._piece_loc()
        return (yaw_of(scene.plinth.data.root_quat_w[0]),
                scene.plinth.data.root_pos_w[0, :2].clone(),
                float(k[0, 0]), float(r[0, 0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_p, a_kx, a_rx = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_p, b_kx, b_rx = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_p = float((a_p - b_p).norm())
    print(f"[smoke] randomization deltas: plinth_yaw={d_yawv:.1f}deg "
          f"plinth_xy={d_p * 1000:.1f}mm king_x={abs(a_kx - b_kx) * 1000:.1f}mm "
          f"rook_x={abs(a_rx - b_rx) * 1000:.1f}mm", flush=True)
    check("randomization-is-real: plinth yaw, plinth xy and the piece start spots "
          "readback differ across seeds",
          d_yawv > 2.0 and d_p > 0.003
          and (abs(a_kx - b_kx) > 0.003 or abs(a_rx - b_rx) > 0.003))

    # ================= 3. target side: both pockets occur, beacon tracks it =======================
    sides, beacon_ok = set(), True
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        side = float(scene.tgt[0])
        sides.add(side)
        bl = scene._local(scene.beacon.data.root_pos_w)[0]
        beacon_ok = beacon_ok and bool(side * float(bl[1]) > 0.15)
    print(f"[smoke] over 10 resets: target sides {sorted(sides)} beacon_ok={beacon_ok}",
          flush=True)
    check("target-side coverage: both pocket sides drawn over 10 resets, beacon tile "
          "readback on the target side every time",
          sides == {1.0, -1.0} and beacon_ok)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. LIP CAPTIVITY: proven actuator, then a 3x-weight pull ====================
    # (a) a small horizontal shove must MOVE the rook — otherwise the pull below
    # would be vacuous (a dead wrench path rejects everything).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    r_before = scene._local(scene.rook.data.root_pos_w)[0].clone()
    _push(scene.rook, 0.9 * plinth_dir((1.0, 0.0, 0.0)), 36)
    _step(60)
    _refresh()
    r_after = scene._local(scene.rook.data.root_pos_w)[0].clone()
    slid = float((r_after[:2] - r_before[:2]).norm())
    # (b) straight-up pull at 3x the piece's weight for 1.5 s AT THE JUNCTION —
    # the worst case: the lip slots cross there and leave the largest opening in
    # the whole network (its inscribed circle is still smaller than the flange).
    # The flange must jam against the lip / corner undersides and stay captive.
    place(scene.rook, xj, 0.0, z_chan)
    _step(30)
    _refresh()
    z_pre = float(scene._local(scene.rook.data.root_pos_w)[0, 2])
    f_up = torch.zeros(n, 3, device=device)
    f_up[:, 2] = 3.0 * c.piece_mass * 9.81
    z_max = -1.0
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of pulling
        _push(scene.rook, f_up, 20)
        _refresh()
        z_max = max(z_max, float(scene._local(scene.rook.data.root_pos_w)[0, 2]))
    _step(60)
    _refresh()
    r_end = scene._piece_loc()[1]
    rise = z_max - z_pre
    print(f"[smoke] captivity: shove slid the rook {slid * 1000:.1f}mm; 3x-weight pull "
          f"at the junction rose {rise * 1000:.1f}mm (head room "
          f"{1000 * (c.wall_h - c.flange_h):.0f}mm)", flush=True)
    _report("captivity")
    check("LIP CAPTIVITY: the horizontal shove moves the rook >= 5 mm (live wrench "
          "path), yet a 3x-weight upward pull for 1.5 s AT THE JUNCTION opening "
          "raises it < 12 mm and it stays captive in the channel",
          slid >= 0.005 and rise < 0.012 and bool(scene.in_channel(r_end)[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # "Pick each piece up and place it on its square" — CONSTRUCT that end state:
    # both pieces placed from open air ON TOP of the lip rails, standing exactly
    # over their goal cells. Perfect xy, wrong physics: never IN the channels.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place(scene.rook, xj, scene.tgt * c.pocket_y, z_lips)
    place(scene.king, c.home_x, 0.0, z_lips)
    _step(150)
    _report("seed-strategy")
    k1, r1 = scene._piece_loc()
    check("negative (SEED strategy): both pieces placed from above ON TOP of the lip "
          "rails over their goal cells — not in the channels, no success, score ~0",
          not bool(scene.in_channel(k1)[0]) and not bool(scene.in_channel(r1)[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. near-miss: king centimetres short =======================================
    # Rook correctly seated; king constructed IN the gallery but 6 cm short of the
    # gold wall. Partial credit may latch fully (l1+l2+l3) — success must refuse
    # and the score must stay at the 0.75 cap.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    place(scene.rook, xj, scene.tgt * 0.129, z_chan)
    place(scene.king, c.home_x - 0.060, 0.0, z_chan)
    _step(120)
    _report("near-miss")
    check("near-miss: rook seated, king 6 cm short of home — no success, score <= 0.75",
          bool(scene.rook_seated()[0]) and not bool(scene.king_home()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.7501)

    # ================= 8. negative: the DECOY pocket ==============================================
    # The full "castled" tableau mirrored onto the WRONG side: rook at the decoy
    # pocket's end, king at the gold home cell. Everything matches but the beacon.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place(scene.rook, xj, -scene.tgt * 0.129, z_chan)
    place(scene.king, c.home_x, 0.0, z_chan)
    _step(120)
    _report("wrong-pocket")
    check("negative (wrong pocket): rook garaged in the DECOY pocket, king at home — "
          "no success, score ~0",
          bool(scene.king_home()[0]) and not bool(scene.rook_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 9. negative: swapped pieces ================================================
    # KING garaged at the target pocket end, ROOK at the gold home cell: right
    # cells, wrong identities.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    place(scene.king, xj, scene.tgt * 0.129, z_chan)
    place(scene.rook, c.home_x, 0.0, z_chan)
    _step(120)
    _report("swapped")
    check("negative (swapped pieces): king at the pocket end, rook at the home cell — "
          "no success, score ~0",
          not bool(scene.rook_seated()[0]) and not bool(scene.king_home()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.05)

    # ================= 10. ORDER IS PHYSICS: the rook blocks the king =============================
    # Fresh reset (rook still in the gallery). Push the king straight for home at a
    # steady ~1x piece weight. The king must provably ADVANCE (the actuation is
    # live), but it can only shove the rook train into the gold stop: flange radii
    # bound the king to x <= 0.162, short of the 0.181 home band. Order cannot be
    # cheated: no garage, no home.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _refresh()
    kx_start = float(scene._piece_loc()[0][0, 0])
    _push(scene.king, 2.0 * plinth_dir((1.0, 0.0, 0.0)), 600)
    _step(120)
    _report("blocked-order")
    k2, r2 = scene._piece_loc()
    kx_end = float(k2[0, 0])
    print(f"[smoke] blocked push: king x {kx_start:+.3f} -> {kx_end:+.3f} "
          f"(home band starts at {c.home_x - c.cell_tol:+.3f}); rook x "
          f"{float(r2[0, 0]):+.3f}", flush=True)
    check("ORDER IS PHYSICS: with the rook ungaraged, a 5 s straight push moves the "
          "king >= 3 cm yet it can never enter the home band (blocked by the rook "
          "train), no success",
          kx_end >= kx_start + 0.03 and not bool(scene.king_home()[0])
          and float(k2[0, 0]) < c.home_x - c.cell_tol
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.castling_gallery")
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
    main()
