"""Smoke / rubric-REJECTION battery for BendGalleryScene — NullRobot, teleported probes.

solve.py is the acceptance proof (a pure contact-dynamics trajectory threads the dipper
out of the gallery and scores 1.0 on seeds 0/1/2). This battery proves the rubric
REJECTS wrong outcomes and that the physical claims the task rests on are load-bearing:
the roof really blocks the seed's straight lift, the dead-end tunnel really dead-ends,
the far plaza wall really forces the corner rotation, and the full-extraction clause of
success() really refuses a cup-on-pad with the handle still inside. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a solution:
success() is monitored at EVERY step and must never turn True anywhere in the battery
(the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; dipper flat (up_z ~ +1), tail
                           deep in the dead-end tunnel; score 0, no success;
   2. randomization      — three seeded resets: READBACK gallery xy + yaw, handle
                           insertion depth (gallery-frame tail x) and pad local xy all
                           differ pairwise;
   3. null-policy        — 240 idle steps: score ~0, no success;
   4. SEED STRATEGY      — the seed's own move (grasp the cup, lift it straight up):
                           a sustained 1.7x-weight vertical pull at the CoM. The rod
                           RISES (the probe is real) but wedges against the tunnel-1
                           roof: tail stays under the roof, never extracted; on
                           release it falls back — nothing latched, score 0;
   5. MECHANISM: jam     — a gentle velocity-regulated push straight OUT of the
                           dead-end tunnel (+x): the rod slides (probe moved) then
                           JAMS on the far plaza wall with the tail still deep inside
                           — a straight rod can never clear; `cleared` never latches;
   6. MECHANISM: dead end— the same push INTO the tunnel (-x): the rod slides back
                           and JAMS on the cap wall — the tunnel really dead-ends;
   7. NEGATIVE: corridor — cup UPRIGHT and centred ON the pad, but the handle laid
                           back through the exit tunnel with the tail still inside
                           the gallery (the pose the __post_init__ honesty assert
                           guarantees exists) -> extracted() False, NO success: the
                           full-extraction clause is load-bearing;
   8. latched credit     — teleporting the rod from 7 back into the dead-end tunnel
                           drops every live predicate, but the latched cleared+emerged
                           credit survives (score >= 0.44, still no success);
   9. near-miss: pad     — fully extracted, upright, settled, but the cup 85 mm from
                           the pad centre (window 50 mm) -> no success, score <= 0.75;
  10. NEGATIVE: flipped  — fully extracted, cup centred on the pad but UPSIDE-DOWN
                           (body rolled pi) -> not upright, no success;
  11. settle gate        — the exact success pose sliding at ~0.45 m/s -> success
                           refuses while anything moves (probe dismantled before it
                           can decelerate into a real success);
  12. rejection audit    — success() was never True at any step of this battery;
  13. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.pick_up_cup_i76.smoke --headless
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

_qx, _qz = task_scene._qx, task_scene._qz
_qmul, _qinv, _qapply = task_scene._qmul, task_scene._qinv, task_scene._qapply
encode_force = task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUD = {"on": False, "hits": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if _AUD["on"] and bool(env.scene.success()[0]):
            _AUD["hits"] += 1
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    cl = scene.cup_local()[0]
    tl = scene.tail_local()[0]
    print(f"[smoke] {tag:18s} | cup_l=({float(cl[0]):+.3f},{float(cl[1]):+.3f}) "
          f"tail_l=({float(tl[0]):+.3f},{float(tl[1]):+.3f},{float(tl[2]):+.3f}) "
          f"up_z={float(scene.rod_up_z()[0]):+.3f} "
          f"clr={bool(scene._cleared[0])} ent={bool(scene._entered[0])} "
          f"emg={bool(scene._emerged[0])} ext={bool(scene.extracted()[0])} "
          f"pad={bool(scene.cup_on_pad()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bend_gallery")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    rest_z = c.handle_h / 2 + 0.003
    cup_arm = c.rod_len / 2 - c.cup_r  # rod centre -> cup centre along +x (0.155)
    zero = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.25, -0.85, 0.80)) + o),
                                tuple(np.array((0.40, -0.02, 0.05)) + o),
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

    def gal_q() -> torch.Tensor:
        _refresh()
        return scene.gallery.data.root_quat_w

    def gal_dir(local_xyz) -> torch.Tensor:
        """Gallery-frame direction -> world, (N,3)."""
        d = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return _qapply(gal_q(), d)

    def gal_yaw() -> float:
        q = gal_q()[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def pad_local_xy() -> torch.Tensor:
        _refresh()
        return _qapply(_qinv(scene.gallery.data.root_quat_w),
                       scene.pad.data.root_pos_w - scene.gallery.data.root_pos_w)[:, :2]

    def write_rod(center_xy_w: torch.Tensor, yaw_local: float, roll: float = 0.0,
                  z: float = rest_z, lin_vel_w: torch.Tensor | None = None) -> None:
        """Teleport the dipper: world xy centre, gallery-relative yaw, optional body
        roll (about +x) and world velocity. Pure transport — the judge does the rest."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = center_xy_w
        st[:, 2] = env.iscene.env_origins[:, 2] + z
        q = _qmul(gal_q(), _qz(torch.full((n,), yaw_local, device=device)))
        if roll != 0.0:
            q = _qmul(q, _qx(torch.full((n,), roll, device=device)))
        st[:, 3:7] = q
        if lin_vel_w is not None:
            st[:, 7:10] = lin_vel_w
        scene.dipper.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def clear_force() -> None:
        scene.dipper.set_external_force_and_torque(zero, zero, env_ids=_all_ids(),
                                                   is_global=True)

    def rod_state():
        _refresh()
        tl = scene.tail_local()[0]
        z_rel = float(scene.dipper.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        return float(tl[0]), float(tl[2]), z_rel

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    tl = scene.tail_local()[0]
    check("settle/no-NaN: layout settles finite; dipper FLAT (up_z ~ +1) with its tail "
          "deep in the dead-end tunnel; score 0, no success",
          bool(scene._finite()[0]) and float(scene.rod_up_z()[0]) > 0.98
          and float(tl[0]) < -0.14 and abs(float(tl[1]) - 0.17) < 0.03
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        gp = (scene.gallery.data.root_pos_w - scene.env_origins)[0, :2].clone()
        return gp, gal_yaw(), float(scene.tail_local()[0, 0]), pad_local_xy()[0].clone()

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())
    pairs = [(0, 1), (0, 2), (1, 2)]

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    d_g = max(float((obs[i][0] - obs[j][0]).norm()) for i, j in pairs)
    d_y = max(dyaw(obs[i][1], obs[j][1]) for i, j in pairs)
    d_d = max(abs(obs[i][2] - obs[j][2]) for i, j in pairs)
    d_p = max(float((obs[i][3] - obs[j][3]).norm()) for i, j in pairs)
    print(f"[smoke] randomization deltas (max over 3 seed pairs): gallery_xy="
          f"{d_g * 1000:.1f}mm gallery_yaw={d_y:.1f}deg depth={d_d * 1000:.1f}mm "
          f"pad_local_xy={d_p * 1000:.1f}mm", flush=True)
    check("randomization-is-real: gallery xy+yaw, handle insertion depth and pad "
          "local xy readback all differ across seeds",
          d_g > 0.004 and d_y > 2.0 and d_d > 0.004 and d_p > 0.004)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 4. SEED STRATEGY: the straight lift is mechanically blocked ================
    # The seed picks the cup up: one grasp, one vertical lift. Here that exact move — a
    # sustained 1.7x-weight upward pull at the CoM — wedges the rod against the
    # tunnel-1 roof: the rod RISES (the probe is real) but the covered tail can never
    # come up, and the dipper stays inside the gallery. Released, it falls back.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    q_ref = scene.dipper.data.root_quat_w.clone()
    z0 = rod_state()[2]
    rise_max, tailx_max, tailz_max, ext_any = 0.0, -1.0, 0.0, False
    f_up = 1.7 * c.rod_mass * 9.81  # ~5.8 N vs 3.4 N weight
    for _ in range(120):
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = f_up
        f_enc = encode_force(1, q_ref, scene.dipper.data.root_quat_w, f_w)
        scene.dipper.set_external_force_and_torque(f_enc.unsqueeze(1), zero,
                                                   env_ids=_all_ids(), is_global=True)
        _step(1)
        tx, tz, zr = rod_state()
        rise_max = max(rise_max, zr - z0)
        tailx_max, tailz_max = max(tailx_max, tx), max(tailz_max, tz)
        ext_any = ext_any or bool(scene.extracted()[0])
    clear_force()
    _step(150)
    _report("seed-lift")
    print(f"[smoke] lift probe: rise_max={rise_max * 1000:.1f}mm "
          f"tail_x_max={tailx_max * 1000:.0f}mm tail_z_max={tailz_max * 1000:.1f}mm "
          f"(roof underside {c.roof_lo * 1000:.0f}mm)", flush=True)
    check("SEED STRATEGY (lift blocked): a 1.7x-weight vertical pull raises the rod "
          "(probe real) but the roof wedges it — tail never leaves the roofed tunnel, "
          "never extracted; on release it falls back, nothing latched, score 0",
          rise_max >= 0.004 and tailx_max < 0.0 and tailz_max < c.roof_lo + 0.005
          and not ext_any and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))
    _REC["on"] = False

    def slide_probe(sign: float, steps: int = 300) -> float:
        """Velocity-regulated push along gallery +/-x at the CoM; returns the signed
        gallery-frame x displacement of the rod centre. Gentle by construction
        (~0.08 m/s target, 3 N cap) so the jam is a quasi-static wedge, not a slam."""
        q_ref = scene.dipper.data.root_quat_w.clone()
        d_w = gal_dir((sign, 0.0, 0.0))
        x0 = float(_qapply(_qinv(gal_q()), scene.dipper.data.root_pos_w
                           - scene.gallery.data.root_pos_w)[0, 0])
        for _ in range(steps):
            v = float((scene.dipper.data.root_lin_vel_w[0] * d_w[0]).sum())
            fmag = max(0.0, min(3.0, 1.5 + 8.0 * (0.08 - v)))
            f_enc = encode_force(1, q_ref, scene.dipper.data.root_quat_w, fmag * d_w)
            scene.dipper.set_external_force_and_torque(f_enc.unsqueeze(1), zero,
                                                       env_ids=_all_ids(), is_global=True)
            _step(1)
        clear_force()
        _step(90)
        x1 = float(_qapply(_qinv(gal_q()), scene.dipper.data.root_pos_w
                           - scene.gallery.data.root_pos_w)[0, 0])
        return (x1 - x0) * sign

    # ================= 5. MECHANISM: the straight push out jams on the far wall ===================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    moved = slide_probe(+1.0)
    _report("fwd-jam")
    head_x = float(scene._rod_pts_local()[0, 5, 0])
    tail_x = float(scene.tail_local()[0, 0])
    print(f"[smoke] forward push: slid {moved * 1000:.0f}mm, jammed with head_x="
          f"{head_x * 1000:.0f}mm (far wall {c.plaza * 1000:.0f}mm) tail_x="
          f"{tail_x * 1000:.0f}mm (mouth at 0)", flush=True)
    check("MECHANISM (must rotate): pushing the rod straight out of the dead-end "
          "tunnel slides it (probe moved) then JAMS on the far plaza wall with the "
          "tail still deep inside — `cleared` never latches, score 0",
          moved >= 0.008 and head_x > 0.23 and tail_x < -0.10
          and not bool(scene._cleared[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. MECHANISM: the tunnel really dead-ends ==================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    moved = slide_probe(-1.0)
    _report("dead-end")
    tail_x = float(scene.tail_local()[0, 0])
    print(f"[smoke] backward push: slid {moved * 1000:.0f}mm, jammed with tail_x="
          f"{tail_x * 1000:.0f}mm (cap interior at {-c.t1_len * 1000:.0f}mm)", flush=True)
    check("MECHANISM (dead end): pushing the rod deeper slides it (probe moved) then "
          "JAMS on the cap wall — the back way out does not exist",
          moved >= 0.008 and -0.25 < tail_x < -0.19
          and not bool(scene.extracted()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 7. NEGATIVE: cup on the pad, handle still inside ===========================
    # The pose the __post_init__ honesty assert guarantees exists: cup UPRIGHT and
    # centred on the pad, rod laid back through the exit tunnel, tail still inside the
    # gallery. Settled and tidy — and refused: extraction is judged on EVERY point.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    h_w = gal_dir((0.0, -1.0, 0.0))  # rod +x pointing out of the exit tunnel
    cup_xy = scene.pad.data.root_pos_w[:, :2].clone()
    write_rod(cup_xy - cup_arm * h_w[:, :2], -math.pi / 2)
    _step(150)
    _report("corridor")
    tl = scene.tail_local()[0]
    exit_y = -(c.t2_len + c.out_margin)
    print(f"[smoke] corridor pose: tail_local=({float(tl[0]) * 1000:+.0f},"
          f"{float(tl[1]) * 1000:+.0f})mm vs exit line y={exit_y * 1000:.0f}mm — "
          f"tail still inside the gallery", flush=True)
    check("negative (corridor): cup upright and centred ON the pad but the tail "
          "still inside the gallery — extracted() False, no success, score <= 0.75",
          bool(scene.cup_on_pad()[0]) and bool(scene.upright()[0])
          and bool(scene.settled()[0]) and float(tl[1]) > exit_y
          and not bool(scene.extracted()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= 8. latched credit survives =================================================
    # Continue from 7 (cleared+emerged latched there): teleport the rod back into the
    # dead-end tunnel. Every live predicate drops; the latched credit does not.
    s7 = float(scene.score()[0])
    back_xy = (scene.gallery.data.root_pos_w[:, :2]
               + _qapply(gal_q(), torch.tensor([0.03, 0.17, 0.0], device=device)
                         .expand(n, 3))[:, :2])
    write_rod(back_xy, 0.0)
    _step(60)
    _report("latched")
    check("latched credit: rod teleported back inside the gallery — live extracted/"
          "on-pad drop but the latched cleared+emerged credit survives "
          "(score >= 0.44, still no success)",
          s7 >= 0.44 and not bool(scene.extracted()[0])
          and not bool(scene.cup_on_pad()[0]) and float(scene.score()[0]) >= 0.44
          and not bool(scene.success()[0]))

    # ================= 9. near-miss: off the pad ==================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    gx_w = gal_dir((1.0, 0.0, 0.0))
    cup_xy = scene.pad.data.root_pos_w[:, :2] + 0.085 * gx_w[:, :2]
    write_rod(cup_xy - cup_arm * gx_w[:, :2], 0.0)
    _step(150)
    _report("near-miss-pad")
    d = float((scene.cup_pos_w()[:, :2] - scene.pad.data.root_pos_w[:, :2])
              .norm(dim=-1)[0])
    print(f"[smoke] cup {d * 1000:.0f}mm from the pad centre (window "
          f"{c.pad_tol * 1000:.0f}mm)", flush=True)
    check("near-miss (pad): fully extracted, upright, settled — but the cup 85 mm "
          "from the pad centre (window 50 mm): no success, score <= 0.75",
          bool(scene.extracted()[0]) and bool(scene.upright()[0])
          and bool(scene.settled()[0]) and d > c.pad_tol
          and not bool(scene.cup_on_pad()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)

    # ================= 10. NEGATIVE: upside-down cup on the pad ===================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    gx_w = gal_dir((1.0, 0.0, 0.0))
    cup_xy = scene.pad.data.root_pos_w[:, :2].clone()
    # rolled pi, the cup rim (body z = cup_h - handle_h/2 = +0.055) points down:
    # rest root height ~0.055; drop from a few mm above and let it settle.
    write_rod(cup_xy - cup_arm * gx_w[:, :2], 0.0, roll=math.pi,
              z=c.cup_h - c.handle_h / 2 + 0.005)
    _step(180)
    _report("flipped")
    check("negative (flipped): fully extracted, cup centred on the pad but "
          "UPSIDE-DOWN — not upright, no success",
          bool(scene.extracted()[0]) and bool(scene.cup_on_pad()[0])
          and float(scene.rod_up_z()[0]) < -0.90 and not bool(scene.upright()[0])
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. settle gate ============================================================
    # The exact success pose — extracted, cup on the pad, upright — but sliding at
    # ~0.45 m/s. success() must refuse while anything moves. The probe is dismantled
    # (transport) well before friction could decelerate it into a real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    gx_w = gal_dir((1.0, 0.0, 0.0))
    cup_xy = scene.pad.data.root_pos_w[:, :2].clone()
    vel = -0.45 * gx_w
    vel[:, 2] = 0.0
    write_rod(cup_xy - cup_arm * gx_w[:, :2], 0.0, lin_vel_w=vel)
    moving_ok = True
    for _ in range(5):
        _step(1)
        v = float(scene.dipper.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and bool(scene.cup_on_pad()[0]) \
            and not bool(scene.settled()[0]) and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can decelerate into a real success
    parked = scene.pad.data.root_pos_w[:, :2].clone()
    parked[:, 0] += 0.30
    parked[:, 1] -= 0.12
    write_rod(parked, 0.0)
    _step(30)
    _REC["on"] = False
    check("settle gate: the exact success pose still sliding at ~0.45 m/s is refused "
          "while anything moves",
          moving_ok)

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.bend_gallery")
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
